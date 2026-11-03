"""TRACE-only spatial response controls; production C remains unchanged."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .image_balanced import ImageBalancedTRACE
from .probe_geometry import translated_candidate
from .trace.indexing import pack_region_indices


class SpatialProbeMixin:
    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        mode = self.probe_mode
        if mode not in ("native", "translated", "global", "confidence"):
            raise ValueError("Unknown TRACE spatial probe")
        self._two_view_counts = dict(diagnostic_views=0, diagnostic_calls=0)
        self.last_spatial_probe_receipts = []
        if not candidates:
            return []
        indices = pack_region_indices([r["region"] for r in candidates], images.device)
        original = [float(r["confidence"]) if "confidence" in r else float(
            original_probs[r["batch_idx"], r["class_idx"]].flatten().index_select(0, i).mean())
            for r, i in zip(candidates, indices, strict=True)]
        supports = list(original)
        contexts = {}
        keys = [] if mode == "confidence" else (
            [(p, "global") for p in sorted({r["batch_idx"] for r in candidates})]
            if mode == "global" else [(i, mode) for i in range(len(candidates))])
        # Construct only scalar geometry receipts up front; image tensors remain
        # bounded by the inherited view-batch iterator rather than all candidates.
        for i, row in enumerate(candidates):
            receipt = dict(candidate_index=i, batch_idx=row["batch_idx"], mode=mode,
                           readout_and_loss_remain_original=True)
            if mode == "translated":
                input_mask = F.interpolate(torch.from_numpy(row["region"]).to(
                    device=images.device, dtype=images.dtype)[None, None],
                    size=images.shape[-2:], mode="nearest")[0, 0].bool().cpu().numpy()
                _, geometry = translated_candidate(input_mask)
                receipt.update(geometry)
            self.last_spatial_probe_receipts.append(receipt)

        def build_views(batch_keys):
            views = []
            for index, kind in batch_keys:
                patch = index if kind == "global" else candidates[index]["batch_idx"]
                image = images[patch:patch+1]
                if patch not in contexts:
                    contexts[patch] = self.make_trace_counterfactual_background(image)
                if kind == "global":
                    views.append(contexts[patch])
                    continue
                row = candidates[index]
                mask = F.interpolate(torch.from_numpy(row["region"]).to(
                    device=images.device, dtype=images.dtype)[None, None],
                    size=images.shape[-2:], mode="nearest")
                if kind == "translated":
                    geometry = self.last_spatial_probe_receipts[index]
                    y, x = mask[0, 0].bool().nonzero(as_tuple=True)
                    shifted = torch.zeros_like(mask)
                    shifted[0, 0, y+geometry["dy"], x+geometry["dx"]] = 1
                    if int(shifted.sum()) != int(mask.sum()):
                        raise ValueError("Translated preservation area changed")
                    mask = shifted
                views.append(image*mask + contexts[patch]*(1-mask))
            return torch.cat(views)

        for batch_keys, logits in self._forward_trace_view_batches(keys, build_views):
            values, destinations = [], []
            for (index, kind), item in zip(batch_keys, logits, strict=True):
                probabilities = item.softmax(0)
                targets = [i for i, row in enumerate(candidates) if row["batch_idx"] == index] if kind == "global" else [index]
                for i in targets:
                    # Even sham/global responses are read at the original mask
                    # and original query, never at the preservation mask.
                    values.append(probabilities[candidates[i]["class_idx"]].flatten().index_select(0, indices[i]).mean())
                    destinations.append(i)
            for i, value in zip(destinations, torch.stack(values).cpu().tolist(), strict=True):
                supports[i] = value
            self._two_view_counts["diagnostic_views"] += len(batch_keys)
            self._two_view_counts["diagnostic_calls"] += 1

        return [dict(row, original_support=original[i], target_support=None,
                     counterfactual_support=supports[i] if mode != "confidence" else None,
                     quality=supports[i], reliability=supports[i], residual_support=None,
                     dominance=None, diagnostic_algorithm="C" if mode == "native" else "spatial_"+mode,
                     omitted=False) for i, row in enumerate(candidates)]


class SpatialProbeTRACE(SpatialProbeMixin, ImageBalancedTRACE):
    def __init__(self, *, probe_mode, **kwargs):
        if probe_mode not in ("native", "translated", "global", "confidence"):
            raise ValueError("Unknown TRACE spatial probe")
        self.probe_mode = probe_mode
        super().__init__(view_mode="C", **kwargs)
