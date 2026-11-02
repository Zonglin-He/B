"""Research-only removal controls; production TRACE has no ablation switches."""

from collections import defaultdict

import numpy as np
import torch


VARIANTS = {
    "full": "Full TRACE",
    "no_connected_regions": "w/o Connected-Region Construction",
    "no_source_tracing": "w/o Confidence-Source Tracing",
    "no_reliability_weighting": "w/o Reliability Weighting",
    "no_region_normalization": "w/o Region Normalization",
}

DEFINITIONS = {
    "full": "Unmodified production TRACE.",
    "no_connected_regions": (
        "Replace connected components by one class-union mask per input patch. "
        "Keep original-confidence ranking and top-K per source image. This removes "
        "spatial connectedness, not the existence of a mask or its selection budget."
    ),
    "no_source_tracing": (
        "Keep the same connected candidates but weight them by original candidate "
        "confidence, without evaluating any intervention view."
    ),
    "no_reliability_weighting": (
        "Keep the same candidates and all three intervention forwards, but replace "
        "their final loss weights by one. This isolates applying the traced weights."
    ),
    "no_region_normalization": (
        "Keep candidates, three views, and reliability. Sum weighted token entropies "
        "and normalize by weighted token count, rather than averaging within regions."
    ),
}


def make_ablation_class(base, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation: {variant}")
    if variant == "full":
        return base

    class AblatedTRACE(base):
        def collect_trace_candidates(self, probabilities, predictions, valid_mask=None, image_ids=None):
            if variant != "no_connected_regions":
                return super().collect_trace_candidates(
                    probabilities, predictions, valid_mask, image_ids
                )
            from trace_ovss.method.trace.indexing import pack_region_indices

            batch, _, height, width = probabilities.shape
            if valid_mask is None:
                valid_mask = torch.ones((batch, height, width), dtype=torch.bool,
                                        device=probabilities.device)
            ids = torch.arange(batch) if image_ids is None else torch.as_tensor(image_ids)
            if ids.numel() != batch:
                raise ValueError("Image ID count does not match the input patches")
            ids = ids.detach().cpu().tolist()
            predictions_np = predictions.detach().cpu().numpy()
            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            pending = []
            for batch_idx in range(batch):
                for class_idx in sorted(int(c) for c in np.unique(predictions_np[batch_idx][valid_np[batch_idx]])):
                    mask = (predictions_np[batch_idx] == class_idx) & valid_np[batch_idx]
                    pending.append(dict(batch_idx=batch_idx, image_idx=int(ids[batch_idx]),
                        class_idx=class_idx, class_name=self.classes[class_idx], region=mask,
                        area=int(mask.sum()), component_idx=1))
            by_image = defaultdict(list)
            if pending:
                indices = pack_region_indices([row["region"] for row in pending], probabilities.device)
                scores = torch.stack([
                    probabilities[row["batch_idx"], row["class_idx"]].flatten()
                    .index_select(0, idx).mean() for row, idx in zip(pending, indices)
                ]).detach().cpu().tolist()
                for row, confidence in zip(pending, scores):
                    row["confidence"] = confidence
                    by_image[row["image_idx"]].append(row)
            selected, proposals = [], []
            for image_idx in sorted(by_image):
                ordered = sorted(by_image[image_idx], key=lambda row: (
                    -row["confidence"], -row["area"], row["class_idx"], row["component_idx"]
                ))
                selected.extend(ordered[:self.trace_max_regions])
                proposals.extend(ordered)
            self.last_candidate_proposals = proposals if self.record_candidate_proposals else []
            return selected

        @torch.no_grad()
        def evaluate_trace_views(self, images, original_probs, candidates):
            if variant == "no_source_tracing":
                self.last_trace_view_batch_size = 0
                return [dict(row, original_support=float(row["confidence"]),
                    target_support=None, residual_support=None, counterfactual_support=None,
                    quality=None, dominance=None, reliability=float(row["confidence"]))
                    for row in candidates]
            stats = super().evaluate_trace_views(images, original_probs, candidates)
            if variant == "no_reliability_weighting":
                stats = [dict(row, reliability=1.0) for row in stats]
            return stats

        @torch.no_grad()
        def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
            result = super().estimate_trace_reliability(images, valid_mask, image_ids)
            if variant == "no_source_tracing":
                self.last_reliability_summary["diagnostic_views"] = 0
                self.last_trace_view_batch_size = 0
            return result

        @staticmethod
        def prepare_trace_objective(entropy_per_pixel, region_terms):
            if variant != "no_region_normalization":
                return base.prepare_trace_objective(entropy_per_pixel, region_terms)
            # The production loss already computes each region's mean; multiplying
            # its coefficient by area changes it to a weighted token mean.
            terms = [dict(term, rho=float(term["rho"]) * term["indices"].numel())
                     for term in region_terms]
            return base.prepare_trace_objective(entropy_per_pixel, terms)

    return AblatedTRACE
