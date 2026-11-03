"""Image-balanced TRACE: a new objective, not an equivalent implementation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .trace.constants import EPS
from .trace.indexing import pack_region_indices
from .trace.objective import TraceObjectiveMixin
from .two_view import TwoViewTRACE, shared_support

__all__ = ["ImageBalancedTRACE", "image_balanced_region_entropy",
           "prepare_image_balanced_objective"]


def prepare_image_balanced_objective(entropy_per_pixel, region_terms):
    """Cache the old detached objective separately for each source image.

    Every term must carry its source ``image_idx`` as well as its crop's
    ``batch_idx``. Empty regions and rho <= EPS retain the inherited behavior.
    """
    by_image = {}
    for term in region_terms:
        by_image.setdefault(int(term["image_idx"]), []).append(term)
    prepared = []
    for terms in by_image.values():
        group = TraceObjectiveMixin.prepare_trace_objective(entropy_per_pixel, terms)
        if group[0]:
            prepared.append(group)
    return prepared


def image_balanced_region_entropy(entropy_per_pixel, region_terms, *, prepared=None):
    """Pure tensor objective: mean of within-source weighted region means.

    No model, labels, mutation, or reliability gradient is involved. Entropy
    has shape [prompt, crop, H, W] or [crop, H, W], using the existing TRACE
    region-term schema plus mandatory source ``image_idx``. The optional
    prepared cache is valid only for unchanged terms, device and dtype.
    """
    if prepared is None:
        prepared = prepare_image_balanced_objective(entropy_per_pixel, region_terms)
    total = entropy_per_pixel.sum() * 0.0
    for group in prepared:
        total = total + TraceObjectiveMixin.trace_region_normalized_entropy(
            entropy_per_pixel, (), prepared=group,
        )
    return total / len(prepared) if prepared else total


class ImageBalancedTRACE(TwoViewTRACE):
    """Fixed T/C/min diagnostics with equal normalized source-image budgets.

    Candidate admission, geometry, readouts, optimizer, adapt and reset are
    inherited unchanged. Only diagnostic selection, source identity retention
    and the objective hooks differ from TwoViewTRACE.
    """

    def __init__(self, *, view_mode="min", **kwargs):
        if view_mode not in ("T", "C", "min"):
            raise ValueError("view_mode must be 'T', 'C', or 'min'")
        self.view_mode = view_mode
        super().__init__(**kwargs)

    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        if self.view_mode == "min":
            return super().evaluate_trace_views(images, original_probs, candidates)
        self._two_view_counts = dict(diagnostic_views=0, diagnostic_calls=0)
        if not candidates:
            return []
        kind = "target" if self.view_mode == "T" else "counterfactual"
        indices = pack_region_indices([row["region"] for row in candidates], images.device)
        keys = [(i, kind) for i in range(len(candidates))]
        contexts, supports = {}, {}

        def build_views(batch_keys):
            ids = tuple(i for i, _ in batch_keys)
            masks = F.interpolate(
                torch.from_numpy(np.stack([candidates[i]["region"] for i in ids]))
                .to(device=images.device, dtype=images.dtype).unsqueeze(1),
                size=images.shape[-2:], mode="nearest",
            )
            views = []
            for offset, (i, _) in enumerate(batch_keys):
                patch = candidates[i]["batch_idx"]
                image, mask = images[patch:patch + 1], masks[offset:offset + 1]
                if patch not in contexts:
                    helper = (self.make_trace_weak_image if kind == "target"
                              else self.make_trace_counterfactual_background)
                    contexts[patch] = helper(image)
                views.append(image * mask + contexts[patch] * (1.0 - mask))
            return torch.cat(views)

        for batch_keys, logits in self._forward_trace_view_batches(keys, build_views):
            values = [item.softmax(0)[candidates[i]["class_idx"]].flatten()
                      .index_select(0, indices[i]).mean()
                      for (i, _), item in zip(batch_keys, logits)]
            supports.update(zip(batch_keys, torch.stack(values).cpu().tolist()))
            self._two_view_counts["diagnostic_views"] += len(batch_keys)
            self._two_view_counts["diagnostic_calls"] += 1
            del logits, values

        rows = []
        for i, row in enumerate(candidates):
            support = supports[i, kind]
            quality = shared_support(support, support)
            original = float(row["confidence"]) if "confidence" in row else float(
                original_probs[row["batch_idx"], row["class_idx"]].flatten()
                .index_select(0, indices[i]).mean())
            rows.append(dict(
                row, original_support=original,
                target_support=support if self.view_mode == "T" else None,
                counterfactual_support=support if self.view_mode == "C" else None,
                quality=quality, reliability=quality, residual_support=None,
                dominance=None, diagnostic_algorithm=self.view_mode, omitted=False,
            ))
        return rows

    @torch.no_grad()
    def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
        result = super().estimate_trace_reliability(images, valid_mask, image_ids)
        # The inherited estimator keeps active candidates in exactly this order
        # but drops source identity; preserve it without rebuilding candidates.
        active = (row for row in self.last_candidate_stats
                  if float(row["reliability"]) > EPS)
        for term, row in zip(result["region_terms"], active, strict=True):
            term["image_idx"] = int(row["image_idx"])
        return result

    prepare_trace_objective = staticmethod(prepare_image_balanced_objective)
    trace_region_normalized_entropy = staticmethod(image_balanced_region_entropy)
