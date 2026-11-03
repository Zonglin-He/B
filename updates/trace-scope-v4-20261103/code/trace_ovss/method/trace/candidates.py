"""Deterministic candidate construction from dense OVSS predictions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy import ndimage

from .indexing import pack_region_indices


class TraceCandidateMixin:
    """Extract top-K connected components across patches of each image."""

    def collect_trace_candidates(
        self,
        probabilities,
        predictions,
        valid_mask=None,
        image_ids=None,
    ):
        batch, _, height, width = probabilities.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, height, width),
                device=probabilities.device,
                dtype=torch.bool,
            )

        if image_ids is None:
            image_ids = torch.arange(batch, device=probabilities.device)
        image_ids = torch.as_tensor(
            image_ids,
            device=probabilities.device,
            dtype=torch.long,
        )
        if image_ids.numel() != batch:
            raise ValueError(
                f"image_ids has {image_ids.numel()} entries for {batch} patches"
            )

        image_ids_cpu = image_ids.detach().cpu().tolist()
        predictions_cpu = predictions.detach().cpu().numpy()
        valid_cpu = valid_mask.detach().cpu().numpy().astype(bool)
        candidates_by_image = defaultdict(list)
        pending_candidates = []
        for batch_idx in range(batch):
            image_idx = int(image_ids_cpu[batch_idx])
            pred_np = predictions_cpu[batch_idx]
            valid_np = valid_cpu[batch_idx]

            for class_idx in sorted(int(idx) for idx in np.unique(pred_np[valid_np])):
                class_mask = np.logical_and(pred_np == class_idx, valid_np)
                components, count = ndimage.label(class_mask)
                for component_idx in range(1, count + 1):
                    region = components == component_idx
                    area = int(region.sum())
                    if area == 0:
                        continue
                    pending_candidates.append(
                        {
                            "batch_idx": batch_idx,
                            "image_idx": image_idx,
                            "class_idx": class_idx,
                            "class_name": self.classes[class_idx],
                            "region": region,
                            "area": area,
                            "component_idx": component_idx,
                        }
                    )

        # Transfer all scores together; per-component .item() serializes CUDA.
        if pending_candidates:
            region_indices = pack_region_indices(
                [candidate["region"] for candidate in pending_candidates],
                probabilities.device,
            )
            pending_confidences = [
                probabilities[candidate["batch_idx"], candidate["class_idx"]]
                .flatten().index_select(0, indices).mean()
                for candidate, indices in zip(pending_candidates, region_indices)
            ]
            confidences = torch.stack(pending_confidences).detach().cpu().tolist()
            for candidate, confidence in zip(pending_candidates, confidences):
                candidate["confidence"] = confidence
                candidates_by_image[candidate["image_idx"]].append(candidate)

        selected = []
        all_candidates = []
        for image_idx in sorted(candidates_by_image):
            candidates = candidates_by_image[image_idx]
            candidates.sort(
                key=lambda item: (
                    -item["confidence"],
                    -item["area"],
                    item["class_idx"],
                    item["component_idx"],
                )
            )
            selected.extend(candidates[: self.trace_max_regions])
            all_candidates.extend(candidates)

        self.last_candidate_proposals = (
            all_candidates if self.record_candidate_proposals else []
        )
        return selected
