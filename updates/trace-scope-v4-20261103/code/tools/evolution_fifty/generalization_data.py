"""Frozen RGB-only OpenCV preprocessing for the new generalization protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from trace_ovss.data.datasets import CLIP_MEAN, CLIP_STD, SegmentationPatchDataset
from .bootstrap import require
from .generalization import CORRUPTIONS, read_rgb_bytes


class GeneralizationRGBDataset(SegmentationPatchDataset):
    def __init__(self, config, rows):
        require(rows and len({r["official_id"] for r in rows}) == len(rows), "Missing or duplicate RGB sources")
        self.rows = rows
        forbidden = set()
        for row in rows:
            status = Path(row["label"]).stat()
            forbidden.add((status.st_dev, status.st_ino))
        self.forbidden_label_inodes = forbidden
        for row in rows:
            path = Path(row["image"])
            status = path.stat()
            require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
                    and (status.st_dev, status.st_ino) not in forbidden, "RGB aliases annotation or noncanonical input")
            require(isinstance(row["image_sha256"], str) and len(row["image_sha256"]) == 64, "Missing RGB digest")
            kind = row.get("corruption", "original")
            require(kind == "original" or (kind in CORRUPTIONS and row["severity"] in (1, 3, 5)
                    and row["preprocessing"] == "already_corrupted_and_cv2_resized_no_second_resize"
                    and row["image"] != row["clean_image"]), "Corruption input must be independently frozen")
        require(tuple(config["resize"]) == tuple(config["patch_size"]) == (224, 224), "Generalization uses224x224")
        super().__init__(config["dataset"], config["data_dir"], resize=config["resize"],
            patch_size=config["patch_size"], patch_stride=config["patch_stride"], include_auxiliary=False)

    def _discover_samples(self):
        return [(Path(r["image"]), Path(r["label"])) for r in self.rows]

    def __getitem__(self, index):
        row = self.rows[index]
        content = read_rgb_bytes(row["image"], self.forbidden_label_inodes)
        require(hashlib.sha256(content).hexdigest() == row["image_sha256"], "Frozen generalization RGB drift")
        bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        require(bgr is not None and bgr.dtype == np.uint8, "Invalid generalization RGB")
        original_shape = bgr.shape[:2]
        if row.get("corruption", "original") != "original":
            require(original_shape == (224, 224), "Corruption cache must not be resized again")
        else:
            bgr = cv2.resize(bgr, self._target_size(original_shape), interpolation=cv2.INTER_LINEAR)
        pixels = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float()
        pixels = (pixels-CLIP_MEAN)/CLIP_STD
        patches, grid = self._patchify(pixels)
        return dict(img_patches=patches, meta=dict(img_path=row["image"], seg_map_path=row["label"],
            ori_shape=original_shape, img_shape=pixels.shape[-2:], patch_shape=self.patch_size,
            patch_grid_shape=grid))
