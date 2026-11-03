"""Label-free, no-wrap translated candidate masks for TRACE spatial controls."""

from __future__ import annotations

import numpy as np


def translated_candidate(mask):
    """Choose among legal translations by overlap, distance, then lexical dy/dx.

    Translation never clips, wraps, rotates, resizes or changes the shape/area.
    Only diagnostic preservation moves; the caller retains the original readout
    and loss mask. Unshiftable full-frame candidates stay present and reported.
    """
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise ValueError("Nonempty boolean HW mask required")
    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    y0, y1 = -int(yy.min()), height-1-int(yy.max())
    x0, x1 = -int(xx.min()), width-1-int(xx.max())
    offsets = {(dy, dx) for dy in (y0, 0, y1) for dx in (x0, 0, x1)}
    offsets.discard((0, 0))
    if not offsets:
        offsets = {(0, 0)}
    choices = []
    for dy, dx in offsets:
        overlap = int(mask[yy+dy, xx+dx].sum())
        choices.append(((overlap, -(dy*dy+dx*dx), dy, dx), dy, dx))
    _, dy, dx = min(choices)
    shifted = np.zeros_like(mask)
    shifted[yy+dy, xx+dx] = True
    area, overlap = int(mask.sum()), int((mask & shifted).sum())
    return shifted, dict(dy=dy, dx=dx, original_area=area, shifted_area=int(shifted.sum()),
                         intersection=overlap, overlap_of_candidate=overlap/area,
                         iou=overlap/(2*area-overlap), shiftable=(dy, dx) != (0, 0),
                         readout_and_loss_remain_original=True)
