"""Role-free two-view TRACE, isolated from historical dominance variants."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .trace import TRACE
from .trace.indexing import pack_region_indices


def shared_support(target: float, perturbed: float) -> float:
    """Largest score bounded above by both measured supports."""
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (target, perturbed)):
        raise ValueError("Supports must be finite probabilities")
    return min(target, perturbed)


class TwoViewTRACE(TRACE):
    """Two measured supports and one detached weight per connected candidate.

    The inherited engine supplies candidate construction, episodic reset,
    LayerNorm adaptation and reconstruction. This entry point never evaluates
    candidate-suppressed views or applies semantic-role or dominance rules.
    """

    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        self._two_view_counts = dict(diagnostic_views=0, diagnostic_calls=0)
        if not candidates:
            return []
        indices = pack_region_indices([row["region"] for row in candidates], images.device)
        weak, shifted, supports = {}, {}, {}
        keys = [(i, kind) for i in range(len(candidates))
                for kind in ("target", "counterfactual")]

        def build_views(batch_keys):
            ids = tuple(dict.fromkeys(i for i, _ in batch_keys))
            masks = F.interpolate(
                torch.from_numpy(np.stack([candidates[i]["region"] for i in ids]))
                .to(device=images.device, dtype=images.dtype).unsqueeze(1),
                size=images.shape[-2:], mode="nearest",
            )
            lookup = {i: masks[j:j + 1] for j, i in enumerate(ids)}
            views = []
            for i, kind in batch_keys:
                patch = candidates[i]["batch_idx"]
                image, mask = images[patch:patch + 1], lookup[i]
                if kind == "target":
                    if patch not in weak:
                        weak[patch] = self.make_trace_weak_image(image)
                    context = weak[patch]
                elif kind == "counterfactual":
                    if patch not in shifted:
                        shifted[patch] = self.make_trace_counterfactual_background(image)
                    context = shifted[patch]
                else:
                    raise ValueError(f"Unrecognized two-view intervention: {kind}")
                views.append(image * mask + context * (1.0 - mask))
            return torch.cat(views)

        for batch_keys, logits in self._forward_trace_view_batches(keys, build_views):
            self._two_view_counts["diagnostic_views"] += len(batch_keys)
            self._two_view_counts["diagnostic_calls"] += 1
            values = [item.softmax(0)[candidates[i]["class_idx"]].flatten()
                      .index_select(0, indices[i]).mean()
                      for (i, _), item in zip(batch_keys, logits)]
            supports.update(zip(batch_keys, torch.stack(values).cpu().tolist()))
            del logits, values

        rows = []
        for i, row in enumerate(candidates):
            target = supports[i, "target"]
            perturbed = supports[i, "counterfactual"]
            quality = shared_support(target, perturbed)
            original = float(row["confidence"]) if "confidence" in row else float(
                original_probs[row["batch_idx"], row["class_idx"]].flatten()
                .index_select(0, indices[i]).mean())
            rows.append(dict(row, original_support=original, target_support=target,
                             counterfactual_support=perturbed, quality=quality,
                             reliability=quality, residual_support=None, dominance=None,
                             diagnostic_algorithm="two_view", omitted=False))
        return rows

    @torch.no_grad()
    def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
        self._two_view_counts = dict(diagnostic_views=0, diagnostic_calls=0)
        result = super().estimate_trace_reliability(images, valid_mask, image_ids)
        self.last_reliability_summary.update(self._two_view_counts)
        return result
