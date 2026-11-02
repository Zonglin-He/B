"""Research-only algorithms for fewer intervention forwards, not smaller K."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from trace_ovss.method.trace.constants import EPS
from trace_ovss.method.trace.indexing import pack_region_indices
from trace_ovss.method.trace.reliability import trace_reliability


ALGORITHMS = {
    "standard": "Full three-view TRACE",
    "staged_full": "Full TRACE, target-first execution control",
    "quality_only": "Two-view shared quality",
    "selective_01": "Target-first selective tracing (1% weight bound)",
    "selective_05": "Target-first selective tracing (5% weight bound)",
}
ALGORITHM_DEFINITIONS = {
    "standard": "Unmodified production algorithm and original view ordering.",
    "staged_full": (
        "Evaluate target views first, then residual/counterfactual pairs in descending "
        "target-support order. Evaluate all candidates. Controls staging and batching."
    ),
    "quality_only": (
        "Evaluate target and counterfactual views only and use min(T,C). "
        "Residual support is unobserved, not zero. No dominance attenuation."
    ),
    "selective_01": (
        "Evaluate all target views, then pairs in descending measured target support. "
        "Stop when unseen target-support mass divided by that mass plus observed "
        "effective reliability is at most 0.01. Omitted weights become zero."
    ),
    "selective_05": (
        "Same target-first algorithm with a fixed 0.05 omitted-weight bound. "
        "The bound controls normalized loss coefficients, not mIoU."
    ),
}


class TargetFirstSelector:
    """Select probes using only observed T and completed full reliabilities.

    For an unseen candidate, 0 <= rho <= T. With observed mass A and unseen
    upper mass U, omitting unseen candidates changes normalized coefficients
    by at most U/(A+U) in total variation. This is conditional on identical
    support measurements; floating-point changes due to batching are separate.
    """

    def __init__(self, targets, tolerance, *, complete=False):
        self.targets = [float(value) for value in targets]
        if any(not math.isfinite(t) or not 0 <= t <= 1 for t in self.targets):
            raise ValueError("Targets must be finite probabilities")
        if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError("Tolerance must be in [0,1)")
        self.tolerance = tolerance
        self.complete = complete
        self.observed = {}
        self.pending = sorted(
            (i for i, t in enumerate(self.targets) if complete or t > EPS),
            key=lambda i: (-self.targets[i], i),
        )

    @property
    def observed_mass(self):
        return math.fsum(r for r in self.observed.values() if r > EPS)

    @property
    def unseen_upper_mass(self):
        return math.fsum(self.targets[i] for i in self.pending
                         if self.targets[i] > EPS)

    @property
    def omitted_mass_bound(self):
        a, u = self.observed_mass, self.unseen_upper_mass
        return u / (a + u) if a + u else 0.0

    def next_indices(self, count):
        if count < 1:
            raise ValueError("A probe round must contain at least one candidate")
        if not self.complete and self.omitted_mass_bound <= self.tolerance:
            return []
        return self.pending[:count]

    def observe(self, index, rho):
        rho = float(rho)
        if index not in self.pending:
            raise ValueError("Candidate was already measured or is not eligible")
        if not math.isfinite(rho) or not 0 <= rho <= self.targets[index]:
            raise ValueError("Measured reliability must be between zero and T")
        self.observed[index] = rho
        self.pending.remove(index)


def make_algorithm_class(base, algorithm):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown efficiency algorithm: {algorithm}")
    if algorithm == "standard":
        return base

    class EfficientTRACE(base):
        @torch.no_grad()
        def evaluate_trace_views(self, images, original_probs, candidates):
            self._algorithm_summary = dict(
                diagnostic_views=0, diagnostic_calls=0, fully_traced_candidates=0,
                omitted_mass_bound=0.0, omitted_upper_mass=0.0,
            )
            self.last_trace_view_batch_size = max(1, int(self.trace_view_batch_size))
            if not candidates:
                return []
            indices = pack_region_indices([r["region"] for r in candidates], images.device)
            weak, shifted, supports = {}, {}, {}

            def build_views(keys):
                ids = tuple(dict.fromkeys(i for i, _ in keys))
                masks = F.interpolate(
                    torch.from_numpy(np.stack([candidates[i]["region"] for i in ids]))
                    .to(device=images.device, dtype=images.dtype).unsqueeze(1),
                    size=images.shape[-2:], mode="nearest",
                )
                masks = {i: masks[j:j + 1] for j, i in enumerate(ids)}
                views = []
                for i, kind in keys:
                    patch = candidates[i]["batch_idx"]
                    image, mask = images[patch:patch + 1], masks[i]
                    if kind in {"target", "residual"} and patch not in weak:
                        weak[patch] = self.make_trace_weak_image(image)
                    if kind == "target":
                        view = image * mask + weak[patch] * (1.0 - mask)
                    elif kind == "residual":
                        view = image * (1.0 - mask) + weak[patch] * mask
                    elif kind == "counterfactual":
                        if patch not in shifted:
                            shifted[patch] = self.make_trace_counterfactual_background(image)
                        view = image * mask + shifted[patch] * (1.0 - mask)
                    else:
                        raise ValueError(kind)
                    views.append(view)
                return torch.cat(views)

            def query(keys):
                values = []
                configured_batch = self.trace_view_batch_size
                self.trace_view_batch_size = min(configured_batch, self.last_trace_view_batch_size)
                try:
                    for batch_keys, logits in self._forward_trace_view_batches(keys, build_views):
                        self._algorithm_summary["diagnostic_views"] += len(batch_keys)
                        self._algorithm_summary["diagnostic_calls"] += 1
                        for (i, _), item in zip(batch_keys, logits):
                            values.append(item.softmax(0)[candidates[i]["class_idx"]]
                                          .flatten().index_select(0, indices[i]).mean())
                        del logits, item
                finally:
                    # Carry OOM reductions across rounds, without editing config.
                    self.last_trace_view_batch_size = min(
                        self.trace_view_batch_size, self.last_trace_view_batch_size)
                    self.trace_view_batch_size = configured_batch
                if values:
                    supports.update(zip(keys, torch.stack(values).cpu().tolist()))

            if algorithm == "quality_only":
                query([(i, kind) for i in range(len(candidates))
                       for kind in ("target", "counterfactual")])
                selector = None
            else:
                query([(i, "target") for i in range(len(candidates))])
                tolerance = {"staged_full": 0.0, "selective_01": .01,
                             "selective_05": .05}[algorithm]
                selector = TargetFirstSelector(
                    [supports[i, "target"] for i in range(len(candidates))],
                    tolerance, complete=algorithm == "staged_full",
                )
                while True:
                    count = max(1, self.last_trace_view_batch_size // 2)
                    selected = selector.next_indices(count)
                    if not selected:
                        break
                    query([(i, kind) for i in selected
                           for kind in ("residual", "counterfactual")])
                    for i in selected:
                        rho = trace_reliability(supports[i, "target"],
                            supports[i, "counterfactual"], supports[i, "residual"]).reliability
                        selector.observe(i, rho)
                self._algorithm_summary.update(
                    fully_traced_candidates=len(selector.observed),
                    omitted_mass_bound=selector.omitted_mass_bound,
                    omitted_upper_mass=selector.unseen_upper_mass,
                )

            stats = []
            for i, row in enumerate(candidates):
                target = supports[i, "target"]
                residual = supports.get((i, "residual"))
                cf = supports.get((i, "counterfactual"))
                if algorithm == "quality_only":
                    quality, dominance, reliability = min(target, cf), None, min(target, cf)
                elif i in selector.observed:
                    score = trace_reliability(target, cf, residual)
                    quality, dominance, reliability = score.quality, score.dominance, score.reliability
                else:
                    quality, dominance, reliability = None, None, 0.0
                original = float(row["confidence"]) if "confidence" in row else float(
                    original_probs[row["batch_idx"], row["class_idx"]].flatten()
                    .index_select(0, indices[i]).mean())
                stats.append(dict(row, original_support=original, target_support=target,
                    residual_support=residual, counterfactual_support=cf, quality=quality,
                    dominance=dominance, reliability=reliability,
                    diagnostic_algorithm=algorithm, fully_traced=residual is not None,
                    omitted=cf is None, reliability_upper_bound=target))
            return stats

        @torch.no_grad()
        def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
            self._algorithm_summary = dict(
                diagnostic_views=0, diagnostic_calls=0, fully_traced_candidates=0,
                omitted_mass_bound=0.0, omitted_upper_mass=0.0,
            )
            self.last_trace_view_batch_size = max(1, int(self.trace_view_batch_size))
            result = super().estimate_trace_reliability(images, valid_mask, image_ids)
            self.last_reliability_summary.update(self._algorithm_summary)
            return result

    return EfficientTRACE
