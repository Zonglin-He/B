"""Orchestration from base predictions to detached candidate reliabilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import EPS, TRACE_VIEWS
from .indexing import pack_region_indices


class TraceEstimatorMixin:
    """Estimate candidate reliability without adding gradient paths."""

    @torch.no_grad()
    def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
        logits = self.forward_eval_logits(
            images,
            interpolate=False,
            save_weights=False,
        )
        logits = torch.nan_to_num(
            logits.float(),
            nan=-1e4,
            posinf=1e4,
            neginf=-1e4,
        )
        probabilities = logits.softmax(dim=1)
        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        probabilities = probabilities / probabilities.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(EPS)
        predictions = probabilities.argmax(dim=1)

        batch, _, height, width = probabilities.shape
        lowres_valid_mask = self.prepare_trace_valid_mask(
            valid_mask,
            batch,
            height,
            width,
            images.device,
        )
        candidates = self.collect_trace_candidates(
            probabilities,
            predictions,
            lowres_valid_mask,
            image_ids=image_ids,
        )
        if not candidates:
            self.last_candidate_stats = []
            self.last_region_terms = []
            self.last_reliability_summary = self._empty_trace_summary()
            return {"region_terms": []}

        stats = self.evaluate_trace_views(images, probabilities, candidates)
        region_terms = []
        reliabilities = []
        region_indices = pack_region_indices([stat["region"] for stat in stats], images.device)

        for stat, indices in zip(stats, region_indices):
            reliability = float(stat["reliability"])
            reliabilities.append(reliability)
            if reliability <= EPS:
                continue
            region_terms.append(
                {
                    "batch_idx": int(stat["batch_idx"]),
                    "indices": indices,
                    "rho": reliability,
                }
            )

        self.last_candidate_stats = stats
        self.last_region_terms = region_terms
        self.last_reliability_summary = {
            "candidates": len(stats),
            "selected_candidates": len(region_terms),
            "diagnostic_views": len(stats) * len(TRACE_VIEWS),
            "reliability_mean": (
                float(np.mean(reliabilities)) if reliabilities else 0.0
            ),
            "reliability_max": (
                float(np.max(reliabilities)) if reliabilities else 0.0
            ),
            "reliability_sum": (
                float(np.sum(reliabilities)) if reliabilities else 0.0
            ),
        }
        return {"region_terms": region_terms}

    @staticmethod
    def prepare_trace_valid_mask(valid_mask, batch, height, width, device):
        if valid_mask is None:
            return torch.ones(
                (batch, height, width),
                device=device,
                dtype=torch.bool,
            )
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(device=device, dtype=torch.float32)
        if valid_mask.shape[-2:] != (height, width):
            valid_mask = F.interpolate(
                valid_mask,
                size=(height, width),
                mode="nearest",
            )
        return valid_mask[:, 0] > 0.5

    @staticmethod
    def _empty_trace_summary():
        return {
            "candidates": 0,
            "selected_candidates": 0,
            "diagnostic_views": 0,
            "reliability_mean": 0.0,
            "reliability_max": 0.0,
            "reliability_sum": 0.0,
        }
