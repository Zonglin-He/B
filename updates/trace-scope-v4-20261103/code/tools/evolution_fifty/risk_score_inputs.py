"""Pure CPU inputs for a later separately admitted M01 scoring stage.

No files are opened here. Real GT decoding is permitted only after the new
global support seal and successful retained replay exit have been verified.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .bootstrap import require
from .pixel_transitions import transitions
from .unit_risk import unit_term


def candidate_image_mask(candidate, image_shape):
    """Project original token mask only for the declared full single-crop case."""
    require(len(image_shape) == 2 and all(type(v) is int and 0 < v <= 4096 for v in image_shape),
            "Invalid offline candidate canvas")
    require(candidate["batch_idx"] == candidate["image_idx"],
            "Single-crop source/crop membership differs")
    term = unit_term(candidate, "cpu")
    shape = candidate["region"]["shape"]
    require(all(s <= t for s, t in zip(shape, image_shape, strict=True)), "Token mask exceeds image canvas")
    mask = torch.zeros(shape[0]*shape[1], dtype=torch.float32)
    mask[term["indices"]] = 1
    projected = F.interpolate(mask.reshape(1, 1, *shape), size=tuple(image_shape), mode="nearest")
    return projected[0, 0].numpy().astype(bool)


def decode_annotation(content, shape, classes, reduce_zero_label):
    """Match historical nearest-neighbor GT handling; never infer a label map."""
    require(isinstance(content, bytes) and 0 < len(content) <= 16*1024**2,
            "Annotation bytes outside frozen CPU bound")
    require(len(shape) == 2 and all(type(v) is int and 0 < v <= 4096 for v in shape)
            and type(classes) is int and 1 <= classes <= 255
            and type(reduce_zero_label) is bool, "Invalid offline label geometry/vocabulary")
    with Image.open(io.BytesIO(content)) as image:
        require(image.width*image.height <= 16*1024**2, "Annotation decoded dimensions exceed bound")
        gt = np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    if gt.ndim == 3:
        gt = gt[..., 0]
    require(gt.ndim == 2 and np.issubdtype(gt.dtype, np.integer), "Invalid indexed annotation")
    gt = gt.astype(np.int64, copy=True)
    if reduce_zero_label:
        background, valid = gt == 0, gt != 255
        gt[valid] -= 1
        gt[background] = 255
    valid = gt != 255
    require(bool(((gt[valid] >= 0) & (gt[valid] < classes)).all()), "GT class outside frozen vocabulary")
    return gt


def initial_region_quality(before, gt, mask, classes):
    require(mask.dtype == np.bool_ and mask.shape == gt.shape, "Invalid quality control region")
    counts = transitions(before, before, np.where(mask, gt, 255), classes)
    n = int(counts[0])
    value = float(counts[1]/n) if n else None
    return dict(valid_pixels=n, correct_pixels=int(counts[1]), fraction_correct=value,
        fixed_bin=None if value is None else int(np.searchsorted([.25, .5, .75], value, side="right")),
        group_use="Offline initial-GT quality control only; never candidate selection or deployment weights")
