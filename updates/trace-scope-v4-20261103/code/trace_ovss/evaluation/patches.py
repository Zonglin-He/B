"""Aggregate overlapping dense-logit patches back into image coordinates."""

from __future__ import annotations

import torch


def aggregate_patch_logits(
    patch_logits: torch.Tensor,
    patch_grid_shapes,
    image_shapes,
    patch_size=(224, 224),
    patch_stride=112,
) -> list[torch.Tensor]:
    """Average logits in overlapping patch regions for each source image."""

    patch_height, patch_width = (int(value) for value in patch_size)
    reconstructed = []
    patch_idx = 0
    for grid_shape, image_shape in zip(patch_grid_shapes, image_shapes):
        grid_h, grid_w = (int(value) for value in grid_shape)
        image_h, image_w = (int(value) for value in image_shape)
        if grid_h == grid_w == 1 and image_h <= patch_height and image_w <= patch_width:
            reconstructed.append(patch_logits[patch_idx, :, :image_h, :image_w].clone())
            patch_idx += 1
            continue
        logits = patch_logits.new_zeros(
            (patch_logits.shape[1], image_h, image_w)
        )
        count = patch_logits.new_zeros((1, image_h, image_w))
        for row in range(grid_h):
            for column in range(grid_w):
                y0 = row * int(patch_stride)
                x0 = column * int(patch_stride)
                y1 = min(y0 + patch_height, image_h)
                x1 = min(x0 + patch_width, image_w)
                logits[:, y0:y1, x0:x1] += patch_logits[
                    patch_idx,
                    :,
                    : y1 - y0,
                    : x1 - x0,
                ]
                count[:, y0:y1, x0:x1] += 1
                patch_idx += 1
        reconstructed.append(logits / count.clamp_min(1))

    if patch_idx != patch_logits.shape[0]:
        raise ValueError(
            f"Consumed {patch_idx} patches but received "
            f"{patch_logits.shape[0]}"
        )
    return reconstructed
