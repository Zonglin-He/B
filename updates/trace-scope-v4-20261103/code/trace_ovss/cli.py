"""Command-line clean-validation evaluation for standalone TRACE."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from trace_ovss.configs import BEST_CONFIGS, VISION_OUTPUTS, best_config
from trace_ovss.data import prepare_data
from trace_ovss.data.vocabulary import load_prompt_templates
from trace_ovss.evaluation import SegmentationMetrics, aggregate_patch_logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TRACE on a public OVSS clean-validation split."
    )
    parser.add_argument("--dataset", choices=sorted(BEST_CONFIGS), required=True)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("TRACE_DATA_ROOT", "data"),
        help="Root containing the public benchmark directories.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/trace"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-type", default="naclip")
    parser.add_argument("--backbone", default="ViT-L/14")
    parser.add_argument(
        "--checkpoint-cache",
        default=os.environ.get("TRACE_CHECKPOINT_CACHE"),
    )
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--class-extensions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--trace-max-regions", type=int, default=None)
    parser.add_argument("--trace-view-batch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the frozen/overridden configuration and exit.",
    )
    return parser


def resolved_config(args) -> dict:
    config = best_config(args.dataset, args.data_root)
    if args.data_dir is not None:
        config["data_dir"] = Path(args.data_dir).expanduser().resolve()
    for key in (
        "trace_max_regions",
        "trace_view_batch_size",
        "batch_size",
        "lr",
        "steps",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.class_extensions is not None:
        config["class_extensions"] = args.class_extensions
    return config


def build_patch_valid_mask(meta, patch_size, patch_stride, device):
    patch_h, patch_w = (int(value) for value in patch_size)
    extents = []
    for image_shape, grid_shape in zip(
        meta["img_shape"],
        meta["patch_grid_shape"],
    ):
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        for row in range(grid_h):
            for column in range(grid_w):
                start_y = row * int(patch_stride)
                start_x = column * int(patch_stride)
                valid_h = max(0, min(patch_h, image_h - start_y))
                valid_w = max(0, min(patch_w, image_w - start_x))
                extents.append((valid_h, valid_w))
    if not extents:
        return None
    if all(h == patch_h and w == patch_w for h, w in extents):
        return torch.ones((len(extents), 1, patch_h, patch_w), dtype=torch.bool, device=device)
    sizes = torch.tensor(extents, device=device)
    rows = torch.arange(patch_h, device=device)[None, :, None]
    columns = torch.arange(patch_w, device=device)[None, None, :]
    return ((rows < sizes[:, 0, None, None]) & (columns < sizes[:, 1, None, None])).unsqueeze(1)


def build_patch_image_ids(meta, device):
    image_ids = []
    for image_idx, grid_shape in enumerate(meta["patch_grid_shape"]):
        image_ids.extend(
            [image_idx] * (int(grid_shape[0]) * int(grid_shape[1]))
        )
    return torch.tensor(image_ids, device=device, dtype=torch.long)


def collapse_class_extensions(probabilities, mapping):
    extension_to_class = torch.as_tensor(
        mapping,
        dtype=torch.long,
        device=probabilities.device,
    )
    num_classes = max(mapping) + 1
    indices = extension_to_class.view(-1, *([1] * (probabilities.ndim - 1)))
    collapsed = probabilities.new_full((num_classes, *probabilities.shape[1:]), -torch.inf)
    return collapsed.scatter_reduce_(
        0, indices.expand_as(probabilities), probabilities, reduce="amax", include_self=False
    )


def load_prompts(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return load_prompt_templates()
    prompts = tuple(
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not prompts:
        raise ValueError(f"No prompt templates found in {path}")
    return prompts


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args) -> dict:
    config = resolved_config(args)
    serializable = {
        **config,
        "data_dir": str(config["data_dir"]),
        "vision_outputs": list(VISION_OUTPUTS),
        "model_type": args.model_type,
        "backbone": args.backbone,
    }
    if args.print_config:
        print(json.dumps(serializable, indent=2))
        return {"config": serializable}

    from trace_ovss.method.trace import TRACE

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Seed before constructing the loader and model to preserve the released
    # initialization order across standalone evaluations.
    set_global_seed(args.seed)
    output_dir = args.output.expanduser().resolve() / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(serializable, indent=2) + "\n",
        encoding="utf-8",
    )

    loader, vocabulary = prepare_data(config, workers=args.workers, include_auxiliary=False)
    original_classes = vocabulary.classes
    if config["class_extensions"]:
        active_classes = vocabulary.extensions
        extension_mapping = vocabulary.extension_to_class
    else:
        active_classes = original_classes
        extension_mapping = None

    method = TRACE(
        classes=active_classes,
        lr=config["lr"],
        steps=config["steps"],
        trace_max_regions=config["trace_max_regions"],
        trace_view_batch_size=config["trace_view_batch_size"],
        vision_outputs=VISION_OUTPUTS,
        model_type=args.model_type,
        backbone=args.backbone,
        prompts=load_prompts(args.prompt_file),
        checkpoint_cache=args.checkpoint_cache,
        device=device,
    )

    trial_metrics = []
    started = time.perf_counter()
    for trial in range(args.trials):
        seed = args.seed + trial
        set_global_seed(seed)
        metrics = SegmentationMetrics(
            len(original_classes),
            loader.dataset.ignore_index,
        )
        progress = tqdm(loader, desc=f"{args.dataset} seed={seed}")
        for batch_idx, data in enumerate(progress):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            inputs = data["img_patches"].to(device, non_blocking=True)
            valid_mask = build_patch_valid_mask(
                data["meta"],
                config["patch_size"],
                config["patch_stride"],
                device,
            )
            image_ids = build_patch_image_ids(data["meta"], device)
            if image_ids.numel() != inputs.shape[0]:
                raise ValueError(
                    "Patch metadata and tensor disagree: "
                    f"{image_ids.numel()} versus {inputs.shape[0]}"
                )

            method.adapt(
                inputs,
                valid_mask=valid_mask,
                image_ids=image_ids,
            )
            patch_logits = method.evaluate(inputs)
            reconstructed = aggregate_patch_logits(
                patch_logits,
                data["meta"]["patch_grid_shape"],
                data["meta"]["img_shape"],
                config["patch_size"],
                config["patch_stride"],
            )
            for logits, ground_truth in zip(reconstructed, data["gt"]):
                probabilities = logits.softmax(dim=0)
                if extension_mapping is not None:
                    probabilities = collapse_class_extensions(
                        probabilities,
                        extension_mapping,
                    )
                metrics.update(
                    probabilities.argmax(dim=0),
                    ground_truth[0],
                )

        values = metrics.compute()
        trial_metrics.append({"seed": seed, **values})

    summary = {
        "dataset": args.dataset,
        "trials": trial_metrics,
        "mIoU_mean": float(np.mean([item["mIoU"] for item in trial_metrics])),
        "mIoU_std": float(np.std([item["mIoU"] for item in trial_metrics])),
        "mDice_mean": float(np.mean([item["mDice"] for item in trial_metrics])),
        "mDice_std": float(np.std([item["mDice"] for item in trial_metrics])),
        "mAcc_mean": float(np.mean([item["mAcc"] for item in trial_metrics])),
        "mAcc_std": float(np.std([item["mAcc"] for item in trial_metrics])),
        "duration_sec": time.perf_counter() - started,
        "config": serializable,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
