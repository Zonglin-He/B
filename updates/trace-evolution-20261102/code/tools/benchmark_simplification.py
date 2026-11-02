"""Reproducible, sequential GPU cells for TRACE implementation comparisons."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from trace_ablation_variants import DEFINITIONS, VARIANTS, make_ablation_class
from trace_execution_audit import ExecutionAudit


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(package):
    return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def execution_environment():
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"
    ], text=True).strip().splitlines()
    return dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=next(csv.reader(inventory))[0].strip(), gpu_inventory=inventory,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32)


def with_batch_size(config, batch_size):
    if batch_size is None:
        return dict(config)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return {**config, "batch_size": int(batch_size)}


def with_budget(config, *, max_regions=None, steps=None, lr=None):
    """Research-run overrides; production defaults and scoring remain intact."""
    updated = dict(config)
    for key, value in (("trace_max_regions", max_regions), ("steps", steps)):
        if value is not None:
            if isinstance(value, bool) or value < 1 or int(value) != value:
                raise ValueError(f"{key} must be a positive integer")
            updated[key] = int(value)
    if lr is not None:
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        updated["lr"] = float(lr)
    return updated


def adaptation_parameter_record(method, algorithm):
    """Audit live tensors, not just the adapter's declared parameter names."""
    actual = {f"visual.{name}": p for name, p in method.model.visual.named_parameters()
              if p.requires_grad}
    declared = list(method.adapted_parameter_names)
    all_trainable = {id(p) for p in method.model.parameters() if p.requires_grad}
    optimizer = [p for group in method.optimizer.param_groups for p in group["params"]]
    optimizer_ids = [id(p) for p in optimizer]
    actual_ids = {id(p) for p in actual.values()}
    if (not actual or len(declared) != len(set(declared)) or set(declared) != set(actual)
            or all_trainable != actual_ids or len(optimizer_ids) != len(set(optimizer_ids))
            or set(optimizer_ids) != actual_ids):
        raise RuntimeError("Declared, live trainable and optimizer parameter sets disagree")
    by_id = {id(p): name for name, p in actual.items()}
    return dict(algorithm=algorithm, names=declared, actual_names=sorted(actual),
                optimizer_names=sorted(by_id[id(p)] for p in optimizer),
                parameter_numel={name: p.numel() for name, p in actual.items()},
                trainable_parameters=sum(p.numel() for p in actual.values()),
                encoder_blocks=len(method.model.visual.transformer.resblocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), default="evaluation")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--view-batch-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-regions", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=20261024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--algorithm", default="standard", choices=(
        "standard", "staged_full", "quality_only", "selective_01", "selective_05",
        "quality_staged", "quality_lazy_01", "quality_lazy_class_01",
        "quality_tail6", "quality_tail12", "quality_tail0", "quality_tail6_uncached"))
    parser.add_argument("--force-uniform-policy", action="store_true",
                        help="Reference-only control: apply its thing policy to every candidate")
    args = parser.parse_args()
    if args.algorithm != "standard" and args.variant != "full":
        parser.error("Efficiency algorithms must not be combined with removal controls")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    root = args.package_root.resolve()
    sys.path.insert(0, str(root))
    from torch.utils.data import DataLoader, Subset
    from trace_ovss.cli import (build_patch_image_ids, build_patch_valid_mask,
                                collapse_class_extensions, set_global_seed)
    from trace_ovss.configs import VISION_OUTPUTS, best_config
    from trace_ovss.data.datasets import DATASET_SPECS, prepare_data, trace_collate
    from trace_ovss.evaluation import SegmentationMetrics, aggregate_patch_logits
    from trace_ovss.method.trace import TRACE

    if Path(inspect.getfile(TRACE)).resolve().is_relative_to(root) is False:
        raise RuntimeError("Imported TRACE does not belong to the requested package")
    config = with_budget(with_batch_size(best_config(args.dataset, args.data_root), args.batch_size),
                         max_regions=args.max_regions, steps=args.steps, lr=args.lr)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir.resolve()
    if args.view_batch_size is not None:
        config["trace_view_batch_size"] = args.view_batch_size
    if args.split == "calibration":
        spec = DATASET_SPECS[config["dataset"]]
        if args.dataset.startswith(("voc", "pc")):
            spec = replace(spec, split_file=spec.split_file.replace("val.txt", "train.txt"))
        elif args.dataset in {"coco_stuff", "coco_object"}:
            spec = replace(spec, image_dir=spec.image_dir.replace("val2017", "train2017"),
                           mask_dir=spec.mask_dir.replace("val2017", "train2017"))
        else:
            raise ValueError("This runner only supports audited VOC/Context/COCO training calibration")
        DATASET_SPECS[config["dataset"]] = spec

    set_global_seed(args.seed)
    loader, vocabulary = prepare_data(config, workers=0, include_auxiliary=False)
    dataset = loader.dataset
    total_images = len(dataset)
    indices = (sorted(np.random.default_rng(args.selection_seed).permutation(total_images)
                      [:min(args.max_images, total_images)].tolist())
               if args.max_images else list(range(total_images)))
    paths = [str(dataset.samples[index][0]) for index in indices]
    labels = [str(dataset.samples[index][1]) for index in indices]
    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                        shuffle=False, num_workers=args.workers, collate_fn=trace_collate,
                        pin_memory=True)
    classes = vocabulary.extensions if config["class_extensions"] else vocabulary.classes
    mapping = vocabulary.extension_to_class if config["class_extensions"] else None
    kwargs = {}
    if "class_roles" in inspect.signature(TRACE).parameters:
        from trace_ovss.data.roles import class_roles_for, extend_class_roles
        roles = class_roles_for(config["dataset"], vocabulary.classes)
        kwargs["class_roles"] = extend_class_roles(roles, mapping) if mapping else roles
        if args.force_uniform_policy:
            kwargs["class_roles"] = tuple("thing" for _ in classes)

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Choose a fresh cell directory; existing evidence cannot be overwritten")
    hashes = source_hashes(root / "trace_ovss")
    experiment_files = [Path(__file__), Path(__file__).with_name("trace_ablation_variants.py"),
                        Path(__file__).with_name("trace_execution_audit.py")]
    algorithm_definition = "Unmodified production execution"
    method_class = make_ablation_class(TRACE, args.variant)
    if args.algorithm.startswith("quality_tail"):
        from trace_cached_prefix import CACHED_PREFIX_DEFINITIONS, make_cached_prefix_class
        experiment_files.extend([Path(__file__).with_name("trace_algorithm_variants.py"),
                                 Path(__file__).with_name("trace_cached_prefix.py")])
        algorithm_definition = CACHED_PREFIX_DEFINITIONS[args.algorithm]
        method_class = make_cached_prefix_class(TRACE, args.algorithm)
    elif args.algorithm in {"quality_staged", "quality_lazy_01", "quality_lazy_class_01"}:
        from trace_lazy_quality import LAZY_DEFINITIONS, make_lazy_quality_class
        experiment_files.extend([Path(__file__).with_name("trace_algorithm_variants.py"),
                                 Path(__file__).with_name("trace_lazy_quality.py")])
        algorithm_definition = LAZY_DEFINITIONS[args.algorithm]
        method_class = make_lazy_quality_class(TRACE, args.algorithm)
    elif args.algorithm != "standard":
        from trace_algorithm_variants import ALGORITHM_DEFINITIONS, make_algorithm_class
        experiment_files.append(Path(__file__).with_name("trace_algorithm_variants.py"))
        algorithm_definition = ALGORITHM_DEFINITIONS[args.algorithm]
        method_class = make_algorithm_class(TRACE, args.algorithm)
    experiment_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in experiment_files
    }
    protocol = dict(name=args.name, dataset=args.dataset, split=args.split,
                    full_split=len(indices) == total_images, images=len(indices), available_images=total_images,
                    seed=args.seed, selection_seed=args.selection_seed,
                    image_ids=paths, label_ids=labels, config={**config, "data_dir": str(config["data_dir"])},
                    file_metadata=[[path.stat().st_size, path.stat().st_mtime_ns]
                                   for index in indices for path in dataset.samples[index]],
                    package_root=str(root), source_hashes=hashes,
                    vocabulary_sha256=hashlib.sha256(json.dumps([classes, mapping]).encode()).hexdigest(),
                    force_uniform_policy=args.force_uniform_policy,
                    variant=args.variant, variant_definition=DEFINITIONS[args.variant],
                    algorithm=args.algorithm, algorithm_definition=algorithm_definition,
                    configuration_overrides={key: value for key, value in (
                        ("batch_size", args.batch_size), ("trace_max_regions", args.max_regions),
                        ("steps", args.steps), ("lr", args.lr)) if value is not None},
                    experiment_hashes=experiment_hashes,
                    environment=execution_environment(),
                    timing_scope="GPU-synchronized adaptation and final inference; excludes loading/scoring/model setup",
                    selection="Image IDs chosen without labels or outcomes; batching recorded in config",
                    command=sys.argv)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "progress.json", dict(status="initializing", images=0))
    method = method_class(classes=classes, lr=config["lr"], steps=config["steps"],
                   trace_max_regions=config["trace_max_regions"],
                   trace_view_batch_size=config["trace_view_batch_size"],
                   vision_outputs=VISION_OUTPUTS, device="cuda", **kwargs)
    blocks = method.model.visual.transformer.resblocks
    parameter_record = adaptation_parameter_record(method, args.algorithm)
    write_json(args.output / "adaptation_parameters.json", parameter_record)
    audit = ExecutionAudit(method)
    metrics = SegmentationMetrics(len(vocabulary.classes))
    episodes, predictions, diagnostic_rows = [], [], []
    started = time.perf_counter()
    try:
        for episode, data in enumerate(loader):
            images = data["img_patches"].cuda(non_blocking=True)
            valid = build_patch_valid_mask(data["meta"], config["patch_size"], config["patch_stride"], "cuda")
            image_ids = build_patch_image_ids(data["meta"], "cuda")
            audit.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            losses = method.adapt(images, valid_mask=valid, image_ids=image_ids)
            torch.cuda.synchronize()
            adapted = time.perf_counter()
            logits = method.evaluate(images)
            torch.cuda.synchronize()
            inferred = time.perf_counter()
            peak = torch.cuda.max_memory_allocated() / 2**20
            reconstructed = aggregate_patch_logits(logits, data["meta"]["patch_grid_shape"],
                data["meta"]["img_shape"], config["patch_size"], config["patch_stride"])
            for output, gt, image_path in zip(reconstructed, data["gt"], data["meta"]["img_path"]):
                probabilities = output.softmax(0)
                if mapping is not None:
                    probabilities = collapse_class_extensions(probabilities, mapping)
                prediction = probabilities.argmax(0).cpu()
                before_i, before_u = metrics.intersection.clone(), metrics.union.clone()
                metrics.update(prediction, gt[0])
                predictions.append(dict(image_id=image_path,
                    prediction_sha256=hashlib.sha256(prediction.numpy().tobytes()).hexdigest(),
                    intersection=(metrics.intersection-before_i).tolist(),
                    union=(metrics.union-before_u).tolist()))
            summary = method.last_reliability_summary
            episodes.append(dict(episode=episode, images=len(data["meta"]["img_path"]),
                adaptation_seconds=adapted-tick, inference_seconds=inferred-adapted,
                total_seconds=inferred-tick, peak_allocated_mib=peak, update_steps=len(losses),
                candidates=summary["candidates"], diagnostic_views=summary["diagnostic_views"],
                fully_traced_candidates=summary.get("fully_traced_candidates", summary["candidates"]),
                omitted_mass_bound=summary.get("omitted_mass_bound", 0.0),
                omitted_upper_mass=summary.get("omitted_upper_mass", 0.0),
                effective_view_batch_size=method.last_trace_view_batch_size,
                patches=len(images), **audit.snapshot()))
            for stat in method.last_candidate_stats:
                diagnostic_rows.append(dict(
                    episode=episode, image_id=data["meta"]["img_path"][stat["image_idx"]],
                    patch_in_batch=stat["batch_idx"], class_idx=stat["class_idx"], area=stat["area"],
                    mask_sha256=hashlib.sha256(stat["region"].tobytes()).hexdigest(),
                    **{key: stat[key] for key in ("target_support", "residual_support",
                        "counterfactual_support", "quality", "dominance", "reliability")},
                    **{key: stat.get(key) for key in ("diagnostic_algorithm", "fully_traced",
                        "omitted", "reliability_upper_bound")}))
            write_json(args.output / "progress.json", dict(status="running", images=len(predictions),
                total_images=len(indices), episode=episode, wall_seconds=time.perf_counter()-started,
                metrics=metrics.compute()))
            if episode % 10 == 0:
                print(f"{args.name}/{args.dataset} {len(predictions)}/{len(indices)}", flush=True)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(error=type(error).__name__, message=str(error),
                                                       completed_images=len(predictions)))
        raise
    finally:
        audit.close()
    assert hashes == source_hashes(root / "trace_ovss"), "Source changed during the cell"
    assert experiment_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in experiment_files
    }, "Experiment implementation changed during the cell"
    if adaptation_parameter_record(method, args.algorithm) != parameter_record:
        raise RuntimeError("Adapted or optimizer parameter set changed during the cell")
    assert [row["image_id"] for row in predictions] == paths
    with (args.output / "cost.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episodes[0].keys())
        writer.writeheader()
        writer.writerows(episodes)
    write_json(args.output / "per_image_counts.json", predictions)
    if diagnostic_rows:
        with (args.output / "candidate_diagnostics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
            writer.writeheader()
            writer.writerows(diagnostic_rows)
    warmup = min(2, len(episodes)-1)
    measured = episodes[warmup:]
    n = sum(row["images"] for row in measured)
    seconds = sum(row["total_seconds"] for row in measured)
    result = dict(status="completed", name=args.name, dataset=args.dataset, variant=args.variant, images=len(indices),
                  algorithm=args.algorithm,
                  split=args.split, full_split=protocol["full_split"], metrics=metrics.compute(),
                  miou_unrounded=float(np.nanmean(np.where(metrics.union.numpy() > 0,
                      metrics.intersection.numpy()/np.maximum(metrics.union.numpy(), 1), np.nan))) * 100,
                  intersection=metrics.intersection.tolist(), union=metrics.union.tolist(),
                  ms_per_image=1000*seconds/n, peak_allocated_mib=max(row["peak_allocated_mib"] for row in measured),
                  diagnostic_views_per_image=sum(row["diagnostic_views"] for row in measured)/n,
                  block_inputs_per_image=sum(row["block_inputs"] for row in measured)/n,
                  block_calls_per_image=sum(row["block_calls"] for row in measured)/n,
                  encoder_blocks=len(blocks),
                  warmup_episodes=warmup, wall_seconds=time.perf_counter()-started,
                  protocol_sha256=hashlib.sha256((args.output / "protocol.json").read_bytes()).hexdigest())
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", dict(status="completed", images=len(indices)))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"intersection", "union"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
