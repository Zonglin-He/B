"""Inherited benchmark optimization settings for simplified TRACE."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


VISION_OUTPUTS = tuple(range(-1, -19, -1))


@dataclass(frozen=True)
class TraceBenchmarkConfig:
    dataset: str
    data_subdir: str
    resize: tuple[int, int]
    patch_size: tuple[int, int]
    patch_stride: int
    batch_size: int
    lr: float
    steps: int
    trace_max_regions: int
    class_extensions: bool
    trace_view_batch_size: int = 16

    def resolved(self, data_root: str | Path | None = None) -> dict:
        root = Path(
            data_root
            or os.environ.get("TRACE_DATA_ROOT", "data")
        ).expanduser()
        payload = asdict(self)
        payload["data_dir"] = (root / self.data_subdir).resolve()
        return payload


# K and optimization settings are inherited from the previous role-aware
# method. Its historical main-table scores do not validate this new method.
# One view-batch default was selected by training-split runtime profiling;
# it controls memory and can affect half-precision rounding. The formula,
# interventions and region-normalized objective are fixed across datasets.
BEST_CONFIGS: dict[str, TraceBenchmarkConfig] = {
    "voc20": TraceBenchmarkConfig(
        dataset="PascalVOC20Dataset",
        data_subdir="VOCdevkit/VOC2012",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=1,
        lr=1.105e-3,
        steps=9,
        trace_max_regions=8,
        class_extensions=True,
    ),
    "voc21": TraceBenchmarkConfig(
        dataset="PascalVOC21Dataset",
        data_subdir="VOCdevkit/VOC2012",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=16,
        lr=6.3e-4,
        steps=36,
        trace_max_regions=33,
        class_extensions=True,
    ),
    "pc59": TraceBenchmarkConfig(
        dataset="PascalContext59Dataset",
        data_subdir="VOCdevkit/VOC2012",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=7,
        lr=1.5e-3,
        steps=11,
        trace_max_regions=24,
        class_extensions=True,
    ),
    "pc60": TraceBenchmarkConfig(
        dataset="PascalContext60Dataset",
        data_subdir="VOCdevkit/VOC2012",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=9,
        lr=2.25e-3,
        steps=11,
        trace_max_regions=22,
        class_extensions=True,
    ),
    "coco_object": TraceBenchmarkConfig(
        dataset="COCOObjectDataset",
        data_subdir="coco_stuff",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=14,
        lr=3e-3,
        steps=29,
        trace_max_regions=34,
        class_extensions=True,
    ),
    "coco_stuff": TraceBenchmarkConfig(
        dataset="COCOStuffDataset",
        data_subdir="coco_stuff",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=5,
        lr=3.4e-3,
        steps=12,
        trace_max_regions=33,
        class_extensions=True,
    ),
    "cityscapes": TraceBenchmarkConfig(
        dataset="CityscapesDataset",
        data_subdir="cityscapes",
        resize=(1120, 560),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=1,
        lr=6.3e-4,
        steps=5,
        trace_max_regions=32,
        class_extensions=True,
    ),
    "coco_cd": TraceBenchmarkConfig(
        dataset="COCOCDDataset",
        data_subdir="coco_cd_ovss",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=28,
        lr=8.6e-3,
        steps=5,
        trace_max_regions=4,
        class_extensions=False,
    ),
    "waterbirds_seg": TraceBenchmarkConfig(
        dataset="WaterbirdsSegDataset",
        data_subdir="waterbirds_seg/correlation_0.95",
        resize=(224, 224),
        patch_size=(224, 224),
        patch_stride=112,
        batch_size=16,
        lr=3e-3,
        steps=4,
        trace_max_regions=7,
        class_extensions=False,
    ),
}


def best_config(
    dataset: str,
    data_root: str | Path | None = None,
) -> dict:
    if dataset not in BEST_CONFIGS:
        raise KeyError(f"Unknown TRACE dataset: {dataset}")
    return BEST_CONFIGS[dataset].resolved(data_root)
