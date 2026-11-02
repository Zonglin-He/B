"""Research-only two-view quality with explicit coefficient-omission bounds.

Use the unwrapped TRACE class as the factory base. This replaces its view
evaluator; composing it with removal factories would bypass their evaluator.
No bound here controls gradients, Adam trajectories, class IoU, or mIoU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from trace_ovss.method.trace.constants import EPS
from trace_ovss.method.trace.indexing import pack_region_indices

if __package__:
    from .trace_algorithm_variants import TargetFirstSelector
else:
    from trace_algorithm_variants import TargetFirstSelector


TOLERANCE = 0.01
LAZY_ALGORITHMS = {
    "quality_staged": "Two-view quality, target-first complete control",
    "quality_lazy_01": "Two-view lazy CF with a global 1% coefficient bound",
    "quality_lazy_class_01": "Two-view lazy CF with image/query-class 1% bounds",
}
LAZY_DEFINITIONS = {
    "quality_staged": (
        "Evaluate every target view, then every counterfactual in descending "
        "target-support order. Use q=min(T,C), including the existing EPS filter."
    ),
    "quality_lazy_01": (
        "Evaluate every target view. Probe counterfactuals in descending T until "
        "U/(A+U)<=0.01 for the adaptation episode. A sums measured q>EPS and U "
        "sums unmeasured T>EPS. Omitted C/q remain null and their weights are zero."
    ),
    "quality_lazy_class_01": (
        "Require U_g/(A_g+U_g)<=0.01 separately for every (image_idx,class_idx) "
        "query-class group. Rank pending candidates from uncertified groups by "
        "descending T. This implies the global coefficient bound and preserves "
        "nonzero group support conditional on identical support measurements. "
        "It does not guarantee segmentation accuracy or Adam-trajectory fidelity."
    ),
}


class LazyQualitySelector:
    """Coordinate existing selectors without reading any unobserved C or q.

    With the existing EPS filter, the omitted true mass M_g satisfies M_g<=U_g.
    For A_g>0, group-normalized coefficient TV is M_g/(A_g+M_g). If A_g=0
    and U_g>0, that group cannot stop. Summing U_g<=eps*A_g/(1-eps) proves
    the global bound. A group is certified empty only after U_g reaches zero.
    """

    def __init__(self, targets, groups, algorithm):
        if algorithm not in LAZY_ALGORITHMS:
            raise ValueError(f"Unknown lazy-quality algorithm: {algorithm}")
        targets, groups = list(targets), list(groups)
        if len(targets) != len(groups):
            raise ValueError("Every target must have an image/query-class group")
        self.algorithm = algorithm
        complete = algorithm == "quality_staged"
        tolerance = 0.0 if complete else TOLERANCE
        self.global_selector = TargetFirstSelector(targets, tolerance, complete=complete)
        self.targets = self.global_selector.targets
        self.group_indices = {}
        self.locations = {}
        for index, group in enumerate(groups):
            indices = self.group_indices.setdefault(group, [])
            self.locations[index] = (group, len(indices))
            indices.append(index)
        self.group_selectors = {
            group: TargetFirstSelector(
                [self.targets[i] for i in indices], tolerance, complete=complete,
            )
            for group, indices in self.group_indices.items()
        }

    def next_indices(self, count):
        if count < 1:
            raise ValueError("A probe round must contain at least one candidate")
        if self.algorithm != "quality_lazy_class_01":
            return self.global_selector.next_indices(count)
        eligible = [
            self.group_indices[group][local]
            for group, selector in self.group_selectors.items()
            for local in selector.next_indices(count)
        ]
        return sorted(eligible, key=lambda i: (-self.targets[i], i))[:count]

    def observe(self, index, quality):
        self.global_selector.observe(index, quality)
        group, local = self.locations[index]
        self.group_selectors[group].observe(local, quality)

    def group_bound(self, index):
        group, _ = self.locations[index]
        return self.group_selectors[group].omitted_mass_bound

    def summary(self):
        selectors = tuple(self.group_selectors.values())
        measured = len(self.global_selector.observed)
        return dict(
            measured_cf_candidates=measured,
            fully_traced_candidates=0,
            omitted_candidates=len(self.targets) - measured,
            eps_pruned_candidates=sum(
                t <= EPS and i not in self.global_selector.observed
                for i, t in enumerate(self.targets)
            ),
            observed_quality_mass=self.global_selector.observed_mass,
            omitted_upper_mass=self.global_selector.unseen_upper_mass,
            omitted_mass_bound=self.global_selector.omitted_mass_bound,
            max_group_bound=max((s.omitted_mass_bound for s in selectors), default=0.0),
            group_count=len(selectors),
            observed_active_groups=sum(s.observed_mass > 0 for s in selectors),
            groups_above_tolerance=sum(s.omitted_mass_bound > TOLERANCE for s in selectors),
        )


def make_lazy_quality_class(base, algorithm):
    """Return a sidecar subclass; do not wrap an ablation/algorithm subclass."""
    if algorithm not in LAZY_ALGORITHMS:
        raise ValueError(f"Unknown lazy-quality algorithm: {algorithm}")

    class LazyQualityTRACE(base):
        def _reset_lazy_quality_summary(self):
            self._lazy_quality_summary = dict(
                diagnostic_views=0,
                diagnostic_calls=0,
                omission_tolerance=0.0 if algorithm == "quality_staged" else TOLERANCE,
                **LazyQualitySelector([], [], algorithm).summary(),
            )
            self.last_trace_view_batch_size = max(1, int(self.trace_view_batch_size))

        @torch.no_grad()
        def evaluate_trace_views(self, images, original_probs, candidates):
            self._reset_lazy_quality_summary()
            if not candidates:
                return []
            # Require source-image identity: patch indices do not identify groups
            # when several patches of one image share an adaptation episode.
            groups = [(int(row["image_idx"]), int(row["class_idx"])) for row in candidates]
            indices = pack_region_indices([row["region"] for row in candidates], images.device)
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
                    if kind == "target":
                        if patch not in weak:
                            weak[patch] = self.make_trace_weak_image(image)
                        background = weak[patch]
                    elif kind == "counterfactual":
                        if patch not in shifted:
                            shifted[patch] = self.make_trace_counterfactual_background(image)
                        background = shifted[patch]
                    else:
                        raise ValueError(kind)
                    views.append(image * mask + background * (1.0 - mask))
                return torch.cat(views)

            def query(keys):
                values = []
                configured_batch = self.trace_view_batch_size
                self.trace_view_batch_size = min(configured_batch, self.last_trace_view_batch_size)
                try:
                    for batch_keys, logits in self._forward_trace_view_batches(keys, build_views):
                        self._lazy_quality_summary["diagnostic_views"] += len(batch_keys)
                        self._lazy_quality_summary["diagnostic_calls"] += 1
                        for (i, _), item in zip(batch_keys, logits):
                            values.append(
                                item.softmax(0)[candidates[i]["class_idx"]]
                                .flatten().index_select(0, indices[i]).mean()
                            )
                        del logits, item
                finally:
                    self.last_trace_view_batch_size = min(
                        self.trace_view_batch_size, self.last_trace_view_batch_size,
                    )
                    self.trace_view_batch_size = configured_batch
                if values:
                    supports.update(zip(keys, torch.stack(values).cpu().tolist()))

            query([(i, "target") for i in range(len(candidates))])
            selector = LazyQualitySelector(
                [supports[i, "target"] for i in range(len(candidates))], groups, algorithm,
            )
            while selected := selector.next_indices(self.last_trace_view_batch_size):
                query([(i, "counterfactual") for i in selected])
                for i in selected:
                    selector.observe(i, min(supports[i, "target"], supports[i, "counterfactual"]))
            self._lazy_quality_summary.update(selector.summary())

            stats = []
            for i, row in enumerate(candidates):
                target = supports[i, "target"]
                cf = supports.get((i, "counterfactual"))
                quality = min(target, cf) if cf is not None else None
                original = float(row["confidence"]) if "confidence" in row else float(
                    original_probs[row["batch_idx"], row["class_idx"]].flatten()
                    .index_select(0, indices[i]).mean()
                )
                stats.append(dict(
                    row, original_support=original, target_support=target,
                    residual_support=None, counterfactual_support=cf, quality=quality,
                    dominance=None, reliability=quality if quality is not None else 0.0,
                    diagnostic_algorithm=algorithm, fully_traced=False,
                    measured_cf=cf is not None, omitted=cf is None,
                    reliability_upper_bound=target,
                    group_omitted_mass_bound=selector.group_bound(i),
                ))
            return stats

        @torch.no_grad()
        def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
            self._reset_lazy_quality_summary()
            result = super().estimate_trace_reliability(images, valid_mask, image_ids)
            self.last_reliability_summary.update(self._lazy_quality_summary)
            return result

    return LazyQualityTRACE
