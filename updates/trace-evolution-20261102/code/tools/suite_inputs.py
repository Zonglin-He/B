"""Freeze input identity without loading model weights or image annotations."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


EVALUATION_COUNTS = {"coco_cd": 500, "pc60": 5105, "coco_stuff": 5000, "voc21": 1449,
                     "voc20": 1449, "pc59": 5105, "coco_object": 5000}


def validate_population(dataset, split, requested, available):
    if requested is not None and available < requested:
        raise RuntimeError(f"{dataset}/{split} requires {requested} images; found {available}")
    if split == "evaluation" and available != EVALUATION_COUNTS[dataset]:
        raise RuntimeError(f"Incomplete {dataset} evaluation split: expected {EVALUATION_COUNTS[dataset]}, found {available}")


def validate_discovery(data):
    """Also check archived loaders that silently filtered absent masks."""
    spec, root = data.spec, data.data_dir
    image_root, mask_root = root / spec.image_dir, root / spec.mask_dir
    if spec.split_file:
        names = [line.strip() for line in (root / spec.split_file).read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate IDs in declared split")
        images = [image_root / f"{name}{spec.image_suffix}" for name in names]
    else:
        images = list(image_root.rglob(f"*{spec.image_suffix}"))
    for image in images:
        stem = image.relative_to(image_root).as_posix()[:-len(spec.image_suffix)]
        mask = mask_root / f"{stem}{spec.mask_suffix}"
        if not image.is_file() or not mask.is_file():
            raise RuntimeError(f"Incomplete declared image/mask pair: {image}, {mask}")
    if len(images) != len(data.samples):
        raise RuntimeError("Dataset discovery differs from the complete declared population")


def freeze_protocol(path, requested):
    if path.exists() and json.loads(path.read_text()) != requested:
        raise RuntimeError("Suite inputs changed; choose a fresh suite directory")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(requested, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def expected_inputs(package_root, data_root, plan, data_dirs=None, *, selection_seed=20261024):
    root = package_root.resolve()
    sys.path.insert(0, str(root))
    import trace_ovss
    if not Path(trace_ovss.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Input preparation imported a different TRACE package")
    from trace_ovss.configs import best_config
    from trace_ovss.data.datasets import DATASET_SPECS, SegmentationPatchDataset

    panels = {}
    for key, specification in plan.items():
        dataset, split, limit = specification
        config = best_config(dataset, data_root)
        if data_dirs and dataset in data_dirs:
            config["data_dir"] = Path(data_dirs[dataset]).resolve()
        original = DATASET_SPECS[config["dataset"]]
        if split == "calibration":
            if dataset.startswith(("voc", "pc")):
                spec = replace(original, split_file=original.split_file.replace("val.txt", "train.txt"))
            elif dataset in {"coco_stuff", "coco_object"}:
                spec = replace(original, image_dir=original.image_dir.replace("val2017", "train2017"),
                               mask_dir=original.mask_dir.replace("val2017", "train2017"))
            else:
                raise ValueError(f"Unaudited calibration split: {dataset}")
        else:
            spec = original
        DATASET_SPECS[config["dataset"]] = spec
        try:
            data = SegmentationPatchDataset(config["dataset"], config["data_dir"],
                resize=config["resize"], patch_size=config["patch_size"],
                patch_stride=config["patch_stride"], include_auxiliary=False)
        finally:
            DATASET_SPECS[config["dataset"]] = original
        validate_discovery(data)
        validate_population(dataset, split, limit, len(data))
        indices = sorted(np.random.default_rng(selection_seed).permutation(len(data))[:limit].tolist()) if limit else list(range(len(data)))
        pairs = [data.samples[index] for index in indices]
        classes = data.vocabulary.extensions if config["class_extensions"] else data.vocabulary.classes
        mapping = data.vocabulary.extension_to_class if config["class_extensions"] else None
        panels[key] = dict(
            dataset=dataset, split=split, seed=0, selection_seed=selection_seed,
            image_ids=[str(image) for image, _ in pairs], label_ids=[str(label) for _, label in pairs],
            file_metadata=[[path.stat().st_size, path.stat().st_mtime_ns] for pair in pairs for path in pair],
            vocabulary_sha256=hashlib.sha256(json.dumps([classes, mapping]).encode()).hexdigest(),
            config=json.loads(json.dumps({**config, "data_dir": str(config["data_dir"])})),
            available_images=len(data))
    return panels


def validate_cell_inputs(saved, expected):
    if saved["images"] != len(expected["image_ids"]) or len(saved["image_ids"]) != saved["images"]:
        raise RuntimeError("Cached cell has an incomplete fixed panel")
    for key in ("dataset", "split", "seed", "selection_seed", "image_ids", "label_ids",
                "vocabulary_sha256", "available_images"):
        if saved[key] != expected[key]:
            raise RuntimeError(f"Cached cell does not match frozen inputs: {key}")
    a, b = dict(saved["config"]), dict(expected["config"])
    a.pop("trace_view_batch_size")
    b.pop("trace_view_batch_size")
    if a != b:
        raise RuntimeError("Cached cell does not match frozen dataset configuration")
