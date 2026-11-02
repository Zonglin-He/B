"""Standalone segmentation datasets and overlapping-patch preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .vocabulary import BenchmarkVocabulary, load_vocabulary


CLIP_MEAN = torch.tensor((122.7709, 116.7460, 104.0937)).view(3, 1, 1)
CLIP_STD = torch.tensor((68.5005, 66.6322, 70.3232)).view(3, 1, 1)


@dataclass(frozen=True)
class DatasetSpec:
    image_dir: str
    mask_dir: str
    image_suffix: str
    mask_suffix: str
    split_file: str | None = None
    reduce_zero_label: bool = False


DATASET_SPECS = {
    "PascalVOC20Dataset": DatasetSpec(
        "JPEGImages",
        "SegmentationClass",
        ".jpg",
        ".png",
        "ImageSets/Segmentation/val.txt",
        True,
    ),
    "PascalVOC21Dataset": DatasetSpec(
        "JPEGImages",
        "SegmentationClass",
        ".jpg",
        ".png",
        "ImageSets/Segmentation/val.txt",
    ),
    "PascalContext59Dataset": DatasetSpec(
        "JPEGImages",
        "SegmentationClassContext",
        ".jpg",
        ".png",
        "ImageSets/SegmentationContext/val.txt",
        True,
    ),
    "PascalContext60Dataset": DatasetSpec(
        "JPEGImages",
        "SegmentationClassContext",
        ".jpg",
        ".png",
        "ImageSets/SegmentationContext/val.txt",
    ),
    "COCOObjectDataset": DatasetSpec(
        "images/val2017",
        "annotations/val2017",
        ".jpg",
        "_instanceTrainIds.png",
    ),
    "COCOStuffDataset": DatasetSpec(
        "images/val2017",
        "annotations/val2017",
        ".jpg",
        "_labelTrainIds.png",
    ),
    "CityscapesDataset": DatasetSpec(
        "leftImg8bit/val",
        "gtFine/val",
        "_leftImg8bit.png",
        "_gtFine_labelTrainIds.png",
    ),
    "COCOCDDataset": DatasetSpec(
        "images",
        "masks/test_balanced",
        ".jpg",
        ".png",
        "splits/test_balanced.txt",
    ),
    "WaterbirdsSegDataset": DatasetSpec(
        "images",
        "masks",
        ".jpg",
        ".png",
    ),
}


class SegmentationPatchDataset(Dataset):
    """Load one public benchmark split and return normalized image patches."""

    ignore_index = 255

    def __init__(
        self,
        dataset_name: str,
        data_dir: str | Path,
        *,
        resize: tuple[int, int],
        patch_size: tuple[int, int],
        patch_stride: int,
        include_auxiliary: bool = True,
    ) -> None:
        try:
            self.spec = DATASET_SPECS[dataset_name]
        except KeyError as exc:
            raise KeyError(f"Unsupported TRACE dataset: {dataset_name}") from exc
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.resize = tuple(int(value) for value in resize)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.patch_stride = int(patch_stride)
        self.include_auxiliary = bool(include_auxiliary)
        self.vocabulary: BenchmarkVocabulary = load_vocabulary(dataset_name)
        self.classes = self.vocabulary.classes
        self.class_extensions = self.vocabulary.extensions
        self.extension_to_class = self.vocabulary.extension_to_class
        self.samples = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found for {dataset_name} under "
                f"{self.data_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask_image = source.copy()

        original_shape = (image.height, image.width)
        target_size = self._target_size(original_shape)
        image = image.resize(
            target_size,
            resample=Image.Resampling.BILINEAR,
        )
        mask_image = mask_image.resize(
            target_size,
            resample=Image.Resampling.NEAREST,
        )
        mask = np.asarray(mask_image)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = self._prepare_labels(mask)

        image_rgb = np.asarray(image).copy()
        image = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        image = (image - CLIP_MEAN) / CLIP_STD
        patches, grid_shape = self._patchify(image)
        gt = torch.from_numpy(np.ascontiguousarray(mask)).long().unsqueeze(0)

        item = {
            "gt": gt,
            "img_patches": patches,
            "meta": {
                "img_path": str(image_path),
                "seg_map_path": str(mask_path),
                "ori_shape": original_shape,
                "img_shape": image.shape[-2:],
                "patch_shape": self.patch_size,
                "patch_grid_shape": grid_shape,
            },
        }
        if self.include_auxiliary:
            patch_height, patch_width = self.patch_size
            item["img"] = image
            item["gt_patches"] = (
                gt.unfold(1, patch_height, self.patch_stride)
                .unfold(2, patch_width, self.patch_stride)
                .permute(1, 2, 0, 3, 4)
                .reshape(-1, 1, patch_height, patch_width)
                .contiguous()
            )
        return item

    def _target_size(self, image_shape: tuple[int, int]) -> tuple[int, int]:
        height, width = image_shape
        return self.resize if width > height else self.resize[::-1]

    def _prepare_labels(self, mask: np.ndarray) -> np.ndarray:
        labels = mask.astype(np.int64, copy=True)
        if self.spec.reduce_zero_label:
            background = labels == 0
            valid = labels != self.ignore_index
            labels[valid] -= 1
            labels[background] = self.ignore_index
        return labels

    def _patchify(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        patch_height, patch_width = self.patch_size
        height, width = image.shape[-2:]
        if height < patch_height or width < patch_width:
            raise ValueError(
                f"Resized image {height}x{width} is smaller than patch "
                f"{patch_height}x{patch_width}"
            )
        patches = (
            image.unfold(1, patch_height, self.patch_stride)
            .unfold(2, patch_width, self.patch_stride)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
        )
        grid_shape = (patches.shape[0], patches.shape[1])
        return (
            patches.reshape(-1, 3, patch_height, patch_width),
            grid_shape,
        )

    def _discover_samples(self) -> list[tuple[Path, Path]]:
        image_root = self.data_dir / self.spec.image_dir
        mask_root = self.data_dir / self.spec.mask_dir
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_root}")
        if not mask_root.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {mask_root}")

        if self.spec.split_file is not None:
            split_path = self.data_dir / self.spec.split_file
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing split file: {split_path}")
            relative_stems = [
                line.strip()
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(relative_stems) != len(set(relative_stems)):
                raise ValueError(f"Duplicate image IDs in split file: {split_path}")
            images = [
                image_root / f"{stem}{self.spec.image_suffix}"
                for stem in relative_stems
            ]
        else:
            images = sorted(image_root.rglob(f"*{self.spec.image_suffix}"))

        samples = []
        missing = []
        for image_path in images:
            relative = image_path.relative_to(image_root)
            relative_text = relative.as_posix()
            stem = relative_text[: -len(self.spec.image_suffix)]
            mask_path = mask_root / f"{stem}{self.spec.mask_suffix}"
            if image_path.is_file() and mask_path.is_file():
                samples.append((image_path, mask_path))
            else:
                missing.append(str(image_path if not image_path.is_file() else mask_path))
        if missing:
            raise FileNotFoundError(
                f"Incomplete {self.dataset_name} split: {len(missing)} missing image/mask pairs; "
                f"first missing paths: {missing[:5]}"
            )
        # Batch-level TTA depends on sample grouping, so normalize the order
        # for both split-file and directory-discovered benchmarks.
        samples.sort(key=lambda item: str(item[0]))
        return samples


def trace_collate(batch: list[dict]) -> dict:
    """Keep full-resolution samples separate and concatenate model patches."""

    meta_keys = batch[0]["meta"].keys()
    result = {
        "gt": [item["gt"] for item in batch],
        "img_patches": torch.cat(
            [item["img_patches"] for item in batch],
            dim=0,
        ),
        "meta": {
            key: [item["meta"][key] for item in batch]
            for key in meta_keys
        },
    }
    if "img" in batch[0]:
        result["img"] = [item["img"] for item in batch]
    if "gt_patches" in batch[0]:
        result["gt_patches"] = torch.cat([item["gt_patches"] for item in batch], dim=0)
    return result


def prepare_data(config: dict, *, workers: int = 4, include_auxiliary: bool = True):
    dataset = SegmentationPatchDataset(
        config["dataset"],
        config["data_dir"],
        resize=config["resize"],
        patch_size=config["patch_size"],
        patch_stride=config["patch_stride"],
        include_auxiliary=include_auxiliary,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=workers,
        collate_fn=trace_collate,
        pin_memory=True,
        persistent_workers=workers > 0,
        shuffle=False,
    )
    return loader, dataset.vocabulary
