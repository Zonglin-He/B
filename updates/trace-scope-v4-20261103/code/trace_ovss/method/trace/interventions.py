"""Candidate-specific soft interventions and batched model evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    BLUR_KERNEL,
    COUNTERFACTUAL_CONTRAST,
    TRACE_VIEWS,
)
from .reliability import trace_reliability
from .indexing import pack_region_indices


class TraceInterventionMixin:
    """Build per-candidate views and evaluate them in bounded batches."""

    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        if not candidates:
            self.last_trace_view_batch_size = max(1, int(self.trace_view_batch_size))
            return []
        view_index = []
        weak_images = {}
        counterfactual_images = {}
        regions = [candidate["region"] for candidate in candidates]
        region_indices = pack_region_indices(regions, images.device)
        for candidate_idx in range(len(candidates)):
            for intervention in TRACE_VIEWS:
                view_index.append((candidate_idx, intervention))

        def build_views(keys):
            candidate_ids = tuple(dict.fromkeys(index for index, _ in keys))
            masks = F.interpolate(
                torch.from_numpy(np.stack([regions[index] for index in candidate_ids]))
                .to(device=images.device, dtype=images.dtype).unsqueeze(1),
                size=images.shape[-2:], mode="nearest",
            )
            input_masks = {index: masks[offset:offset + 1]
                           for offset, index in enumerate(candidate_ids)}
            views = []
            current_candidate = None
            for candidate_idx, intervention in keys:
                batch_idx = candidates[candidate_idx]["batch_idx"]
                image = images[batch_idx : batch_idx + 1]
                input_mask = input_masks[candidate_idx]
                if candidate_idx != current_candidate:
                    outside_mask = 1.0 - input_mask
                    preserved = image * input_mask
                    current_candidate = candidate_idx
                if batch_idx not in weak_images:
                    weak_images[batch_idx] = self.make_trace_weak_image(image)
                if batch_idx not in counterfactual_images:
                    counterfactual_images[batch_idx] = (
                        self.make_trace_counterfactual_background(image)
                    )
                weak = weak_images[batch_idx]
                counterfactual = counterfactual_images[batch_idx]
                if intervention == "target":
                    view = preserved + weak * outside_mask
                elif intervention == "residual":
                    view = image * outside_mask + weak * input_mask
                else:
                    view = preserved + counterfactual * outside_mask
                views.append(view)
            return torch.cat(views, dim=0)

        # Consume logits immediately. Only scalar supports survive each batch.
        supports = []
        for keys, logits in self._forward_trace_view_batches(view_index, build_views):
            for (candidate_idx, _), logits_item in zip(keys, logits):
                class_idx = candidates[candidate_idx]["class_idx"]
                probabilities = logits_item.softmax(dim=0)
                supports.append(
                    probabilities[class_idx].flatten()
                    .index_select(0, region_indices[candidate_idx]).mean()
                )
            del logits, logits_item, probabilities

        # Normal candidates already contain exactly this mean probability.
        # The fallback supports independently constructed diagnostic probes.
        original_supports = []
        for candidate, indices in zip(candidates, region_indices):
            if "confidence" in candidate:
                original_supports.append(float(candidate["confidence"]))
            else:
                original_supports.append(float(
                    original_probs[candidate["batch_idx"], candidate["class_idx"]]
                    .flatten().index_select(0, indices).mean()
                ))
        values = torch.stack(supports).cpu().tolist()
        support_by_key = dict(zip(view_index, values))

        stats = []
        for candidate_idx, candidate in enumerate(candidates):
            target_support = support_by_key[(candidate_idx, "target")]
            counterfactual_support = support_by_key[(candidate_idx, "counterfactual")]
            residual_support = support_by_key[(candidate_idx, "residual")]

            score = trace_reliability(
                target_support=target_support,
                counterfactual_support=counterfactual_support,
                residual_support=residual_support,
            )
            stats.append(
                {
                    **candidate,
                    "original_support": original_supports[candidate_idx],
                    "target_support": target_support,
                    "residual_support": residual_support,
                    "counterfactual_support": counterfactual_support,
                    "quality": score.quality,
                    "dominance": score.dominance,
                    "reliability": score.reliability,
                }
            )
        return stats

    def _forward_trace_view_batches(self, view_index, build_views):
        effective_batch_size = max(1, int(self.trace_view_batch_size))
        start = 0
        while start < len(view_index):
            keys = view_index[start : start + effective_batch_size]
            chunk = None
            logits = None
            try:
                chunk = build_views(keys)
                logits = self.forward_eval_logits(
                    chunk,
                    interpolate=False,
                    save_weights=False,
                ).detach()
                logits = torch.nan_to_num(
                    logits.float(),
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )
            except torch.cuda.OutOfMemoryError:
                del chunk, logits
                if effective_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                effective_batch_size = max(1, effective_batch_size // 2)
                continue

            yield keys, logits
            del logits, chunk
            start += len(keys)

        self.last_trace_view_batch_size = effective_batch_size

    def make_trace_weak_image(self, image):
        pixels = image * self._std + self._mean
        gray = (
            0.2989 * pixels[:, 0:1]
            + 0.5870 * pixels[:, 1:2]
            + 0.1140 * pixels[:, 2:3]
        )
        weak = F.avg_pool2d(
            gray,
            kernel_size=BLUR_KERNEL,
            stride=1,
            padding=BLUR_KERNEL // 2,
        )
        return (weak - self._mean) / self._std

    def make_trace_counterfactual_background(self, image):
        pixels = image * self._std + self._mean
        spatial_mean = pixels.mean(dim=(2, 3), keepdim=True)
        low_contrast = (
            (pixels - spatial_mean) * COUNTERFACTUAL_CONTRAST + spatial_mean
        )
        shifted = torch.stack(
            (low_contrast[:, 1], low_contrast[:, 2], low_contrast[:, 0]),
            dim=1,
        )
        shifted = shifted.clamp(0.0, 255.0)
        return (shifted - self._mean) / self._std
