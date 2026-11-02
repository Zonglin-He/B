"""Region-normalized entropy objective for TRACE."""

from __future__ import annotations

import torch

from .constants import EPS


class TraceObjectiveMixin:
    """Apply detached candidate reliability to one spatial entropy loss."""

    @staticmethod
    def prepare_trace_objective(entropy_per_pixel, region_terms):
        """Cache fixed indices, scalar weights and the normalizer per episode."""
        denominator = entropy_per_pixel.new_tensor(0.0)
        prepared = []
        weights = entropy_per_pixel.new_tensor([float(term["rho"]) for term in region_terms])
        for term, rho in zip(region_terms, weights):
            reliability = float(term["rho"])
            if reliability <= EPS:
                continue
            indices = term.get("indices")
            if indices is None:
                indices = term["mask"].to(
                    device=entropy_per_pixel.device, dtype=torch.bool,
                ).flatten().nonzero().flatten()
            if indices.numel() == 0:
                continue
            prepared.append((int(term["batch_idx"]), indices, rho))
            denominator = denominator + rho

        # Preserve the Python-float near-zero comparison without a CPU sync.
        normalizer = torch.where(
            denominator.double() > EPS, denominator, torch.ones_like(denominator)
        )
        return prepared, normalizer

    @staticmethod
    def trace_region_normalized_entropy(entropy_per_pixel, region_terms, *, prepared=None):
        if prepared is None:
            prepared = TraceObjectiveMixin.prepare_trace_objective(
                entropy_per_pixel, region_terms
            )
        terms, normalizer = prepared
        numerator = entropy_per_pixel.sum() * 0.0
        for batch_idx, indices, rho in terms:
            if entropy_per_pixel.ndim == 4:
                region_entropy = (
                    entropy_per_pixel[:, batch_idx].flatten(1)
                    .index_select(1, indices).mean()
                )
            else:
                region_entropy = (
                    entropy_per_pixel[batch_idx].flatten()
                    .index_select(0, indices).mean()
                )
            numerator = numerator + rho * region_entropy
        return numerator / normalizer if terms else numerator

    @staticmethod
    def trace_softmax_entropy(logits, dim=-3):
        return -(logits.softmax(dim) * logits.log_softmax(dim)).sum(dim)
