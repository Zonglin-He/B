"""Single-cell baseline evaluation on sealed pilot/full panels; no tuning or queues."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time
import traceback


METHODS = ("noadapt", "tent", "tpt", "mlmp", "native_noadapt", "legacy_trace_noadapt")
POPULATIONS = {"voc21": 1449, "pc60": 5105, "coco_object": 5000, "coco_cd": 500}
CHECKPOINT_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
LEGACY_COMMIT = "ba9f57bf47846ec727c616c5163233a98ca0bf94"
VISION_OUTPUTS = tuple(range(-1, -19, -1))
NO_UPDATE = {"noadapt", "native_noadapt", "legacy_trace_noadapt"}
MULTIPROMPT = {"mlmp", "native_noadapt", "legacy_trace_noadapt"}


def digest(path):
    with Path(path).open("rb") as handle:
        result = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(payload)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalized(value):
    return json.loads(json.dumps(value, default=str))


def tree_hashes(root):
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
            if path.is_file() and not {"__pycache__", ".git"}.intersection(path.parts)
            and path.suffix not in {".pyc", ".pyo"}}


def source_snapshot(package_root, baseline_root, controls, experiment=None):
    experiment = Path(experiment or __file__).resolve()
    return dict(native=tree_hashes(package_root / "trace_ovss"),
                baseline=tree_hashes(baseline_root),
                experiment={str(experiment): digest(experiment)},
                controls={str(path): digest(path) for path in controls})


def check_snapshot(frozen, current):
    require(current == frozen, "Source/control/checkpoint hashes changed during the cell")


def input_snapshot(panel):
    paths = [Path(path) for pair in zip(panel["image_ids"], panel["label_ids"]) for path in pair]
    metadata, hashes = [], []
    for path in paths:
        before = path.stat()
        hashes.append(digest(path))
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Input changed while hashing: {path}")
        metadata.append([after.st_size, after.st_mtime_ns])
    require(metadata == panel["file_metadata"], "Raw input size/mtime differs from sealed panel")
    return dict(file_metadata=metadata, sha256=hashes)


def check_input_hashes(panel, inputs, expected):
    paths = [path for pair in zip(panel["image_ids"], panel["label_ids"]) for path in pair]
    require(isinstance(expected, dict) and all(path in expected for path in paths),
            "Parent input hashes do not cover the sealed panel")
    require(inputs["sha256"] == [expected[path] for path in paths],
            "Input content differs from parent input hashes")


def execution_environment(package_root):
    """Read the same environment fields as the sealed native worker."""
    tools = package_root / "tools"
    spec = importlib.util.spec_from_file_location("matched_native_environment", tools / "benchmark_simplification.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(tools))
        spec.loader.exec_module(module)
        return module.execution_environment()
    finally:
        sys.path[:] = previous


def check_environment(package_root, expected):
    current = execution_environment(package_root)
    require(current == expected, "Execution environment differs from sealed native suite")
    return current


def validate_panel(panel):
    count = len(panel["image_ids"])
    require(count > 0 and len(panel["label_ids"]) == count, "Incomplete sealed image/label IDs")
    require(len(set(panel["image_ids"])) == len(set(panel["label_ids"])) == count,
            "Duplicate image/label IDs")
    require(len(panel["file_metadata"]) == 2 * count, "Incomplete sealed file metadata")
    require(type(panel["available_images"]) is int and count <= panel["available_images"],
            "Invalid sealed evaluation population")
    batch_size = panel["config"]["batch_size"]
    require(type(batch_size) is int and batch_size > 0, "Invalid sealed episode batch size")
    require(count > 2 * batch_size, "Need more than two episodes for warmup2 timing")
    return count


def select_panel(suite, dataset, panel_key=None):
    key = panel_key if panel_key is not None else dataset + "/128"
    require(key in {dataset + "/128", dataset + "/full"},
            "Panel key must match dataset/128 or dataset/full")
    require(key in suite["input_panels"], f"Missing explicitly sealed panel: {key}")
    panel = suite["input_panels"][key]
    require(panel["dataset"] == dataset and panel["split"] == "evaluation"
            and panel["seed"] == 0 and panel["selection_seed"] == suite["selection_seed"]
            and panel["available_images"] == POPULATIONS[dataset], "Panel identity/population mismatch")
    expected = POPULATIONS[dataset] if key.endswith("/full") else 128
    require(validate_panel(panel) == expected, f"Expected exactly {expected} sealed image/label IDs for {key}")
    return key, panel


def full_suite_controls(args, suite, seal, panel):
    """Full panels require the existing fixed-config seal and frozen runtime."""
    runtime = (args.suite_root / "runtime").resolve()
    require(args.package_root.resolve() == runtime, "Full panel requires the sealed suite runtime")
    fixed_path = args.suite_root / "fixed_configs.json"
    require("fixed_configs.json" in seal["artifact_sha256"], "Missing fixed configurations seal")
    require(digest(fixed_path) == seal["artifact_sha256"]["fixed_configs.json"],
            "Fixed configurations seal mismatch")
    require(read_json(fixed_path).get(args.dataset) == panel["config"],
            "Fixed configuration differs from sealed panel")
    require(bool(suite.get("tool_hashes")) and tree_hashes(runtime / "tools") == suite["tool_hashes"],
            "Frozen runtime tools differ from suite protocol")
    input_path = args.suite_root / "input_hashes.json"
    require("input_hashes.json" in seal["artifact_sha256"], "Missing parent input hashes seal")
    require(digest(input_path) == seal["artifact_sha256"]["input_hashes.json"],
            "Parent input hashes seal mismatch")
    return [fixed_path, input_path, *[runtime / "tools" / name for name in sorted(suite["tool_hashes"])]]


def select_inputs(panel, samples, config, classes, mapping):
    """Resolve sealed pairs without replacing their order or silently dropping IDs."""
    require(normalized(config) == panel["config"], "Native configuration differs from sealed panel")
    validate_panel(panel)
    require(len(samples) == panel["available_images"], "Evaluation population changed")
    lookup = {str(image): (index, str(label)) for index, (image, label) in enumerate(samples)}
    require(len(lookup) == len(samples), "Duplicate native image IDs")
    indices = []
    for image, label in zip(panel["image_ids"], panel["label_ids"]):
        require(image in lookup, f"Missing sealed image ID: {image}")
        index, actual_label = lookup[image]
        require(actual_label == label, f"Sealed label ID/order mismatch: {image}")
        indices.append(index)
    require(indices == sorted(indices), "Sealed image order differs from native evaluation order")
    vocab_hash = hashlib.sha256(json.dumps([classes, mapping]).encode()).hexdigest()
    require(vocab_hash == panel["vocabulary_sha256"], "Vocabulary hash mismatch")
    return indices


def check_batch(panel, offset, images, labels):
    batch_size = panel["config"]["batch_size"]
    require(type(offset) is int and 0 <= offset < len(panel["image_ids"]) and offset % batch_size == 0,
            "Runtime episode boundary offset mismatch")
    end = min(offset + batch_size, len(panel["image_ids"]))
    require(images == panel["image_ids"][offset:end] and labels == panel["label_ids"][offset:end],
            "Runtime image/label order or episode boundary mismatch")


def archived_settings(root, dataset, method, panel_config):
    """Verify config seals and their bound manifest, not current checkout defaults."""
    archive = root.parent
    reference = {"native_noadapt": "noadapt", "legacy_trace_noadapt": "mlmp"}.get(method, method)
    config_path = archive / "released_configs" / f"{dataset}_{reference}.json"
    seals_path = archive / "config_seals.json"
    provenance_path = archive / "source_provenance.json"
    seals, provenance = read_json(seals_path), read_json(provenance_path)
    require(provenance["legacy_commit"] == LEGACY_COMMIT, "Unexpected baseline commit")
    for name, expected in provenance["baseline_sources"].items():
        require(digest(root / name) == expected, f"Archived baseline source mismatch: {name}")
    require(seals["configs"].get(str(config_path)) == digest(config_path), "Archived config seal mismatch")
    manifests = [p for p in archive.glob("manifest*.json") if digest(p) == seals["manifest_sha256"]]
    require(len(manifests) == 1, "Missing or ambiguous sealed baseline manifest")
    manifest_path = manifests[0]
    jobs = read_json(manifest_path)["jobs"]
    require(any(j.get("dataset") == dataset and j.get("method") == reference
                and j.get("settings", {}).get("config_file") == str(config_path) for j in jobs),
            "Archived manifest does not bind dataset/method/config")
    record = read_json(config_path)
    require((record["dataset"], record["method"]) == (dataset, reference), "Archived config identity mismatch")
    config = record["config"]
    require(config["lr"] == .001 and config["steps"] == 10, "Baseline LR/steps are not frozen .001/10")
    for key in ("dataset", "resize", "patch_size", "patch_stride", "class_extensions", "data_dir"):
        require(config[key] == panel_config[key], f"Archived data configuration mismatch: {key}")
    if reference == "mlmp":
        require(config.get("alpha_cls") == 1.0 and config.get("prompt_integration", "loss") == "loss",
                "MLMP alpha_cls/prompt_integration mismatch")
    if method == "tpt":
        require(config.get("n_ctx") == 4, "TPT n_ctx mismatch")
    source = Path(record["source"])
    require(digest(source) == record["source_sha256"], "Archived recipe source hash mismatch")
    if reference == "mlmp":
        if source.suffix == ".sh":
            assignments = {}
            for line in source.read_text().splitlines():
                tokens = shlex.split(line, comments=True) if not line.rstrip().endswith("\\") else []
                if len(tokens) == 1 and "=" in tokens[0]:
                    key, value = tokens[0].split("=", 1)
                    assignments[key] = value
            layers = [int(x) for x in assignments["OUT_VISION"].split()]
        else:
            values = dict(line.split(": ", 1) for line in source.read_text().splitlines() if ": " in line)
            layers = json.loads(values["vision_outputs"])
        require(layers == list(VISION_OUTPUTS), "Archived MLMP VISION_OUTPUTS mismatch")
    return record, [seals_path, provenance_path, manifest_path, config_path, source]


def check_module_roots(prefixes, root):
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] not in prefixes or module is None:
            continue
        locations = list(getattr(module, "__path__", ()))
        if getattr(module, "__file__", None):
            locations.append(module.__file__)
        require(all(Path(p).resolve().is_relative_to(root) for p in locations),
                f"Module from wrong checkout: {name}")


def load_baseline(root, method):
    check_module_roots({"ovss", "utils", "mmcv", "mmengine"}, root)
    sys.path.insert(0, str(root))
    name = {"noadapt": "tent", "legacy_trace_noadapt": "mlmp"}.get(method, method)
    spec = importlib.util.spec_from_file_location(f"matched_baseline_{name}", root / "adapt" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_module_roots({"ovss", "utils", "mmcv", "mmengine"}, root)
    return getattr(module, {"tent": "TENT", "tpt": "TPT", "mlmp": "MLMP"}[name])


def preflight(args):
    suite_path = args.suite_root / "suite_protocol.json"
    seal_path = args.suite_root / "SUITE_SEALED.json"
    seal = read_json(seal_path)
    require(seal["artifact_sha256"]["suite_protocol.json"] == digest(suite_path),
            "Suite protocol seal mismatch")
    suite = read_json(suite_path)
    require(suite["config_fixed"] and suite["seed"] == 0 and suite["selection_seed"] == 20261024,
            "Unsupported suite configuration/seed")
    panel_key, panel = select_panel(suite, args.dataset, getattr(args, "panel_key", None))
    full_split = panel_key.endswith("/full")
    panel_controls = full_suite_controls(args, suite, seal, panel) if full_split else []
    require(tree_hashes(args.package_root / "trace_ovss") == suite["source_hashes"],
            "Native source differs from suite protocol")
    if full_split:
        check_environment(args.package_root, suite["environment"])
    checkpoint = Path.home() / ".cache/clip/ViT-L-14.pt"
    require(digest(checkpoint) == CHECKPOINT_SHA256, "Checkpoint SHA256 mismatch; no automatic download")
    record, controls = archived_settings(args.baseline_root, args.dataset, args.method, panel["config"])
    controls += [suite_path, seal_path, checkpoint, *panel_controls]
    frozen = source_snapshot(args.package_root, args.baseline_root, controls)
    check_module_roots({"trace_ovss"}, args.package_root)
    sys.path.insert(0, str(args.package_root))
    from trace_ovss.configs import best_config, VISION_OUTPUTS as native_layers
    from trace_ovss.data.datasets import prepare_data
    from trace_ovss.data.vocabulary import load_prompt_templates
    import yaml

    check_module_roots({"trace_ovss"}, args.package_root)
    require(tuple(native_layers) == VISION_OUTPUTS, "Native VISION_OUTPUTS mismatch")
    config = best_config(args.dataset, suite["data_root"])
    if args.dataset in suite.get("data_dirs", {}):
        config["data_dir"] = Path(suite["data_dirs"][args.dataset])
    require(normalized(config) == panel["config"], "Native configuration differs from sealed panel")
    loader, vocab = prepare_data(config, workers=0, include_auxiliary=False)
    classes = vocab.extensions if config["class_extensions"] else vocab.classes
    mapping = vocab.extension_to_class if config["class_extensions"] else None
    indices = select_inputs(panel, loader.dataset.samples, config, classes, mapping)
    prompts = list(load_prompt_templates())
    legacy_prompts = yaml.safe_load((args.baseline_root / "prompts.yaml").read_text())["prompt_templates"]
    require(prompts == legacy_prompts, "Native/legacy multi-prompt templates or ordering differ")
    inputs = input_snapshot(panel)
    if full_split:
        check_input_hashes(panel, inputs, read_json(args.suite_root / "input_hashes.json"))
        full_suite_controls(args, suite, seal, panel)
    check_snapshot(frozen, source_snapshot(args.package_root, args.baseline_root, controls))
    return dict(suite=suite, panel=panel, panel_key=panel_key, full_split=full_split,
                config=config, dataset=loader.dataset, vocab=vocab,
                classes=classes, mapping=mapping, indices=indices, archived=record,
                controls=controls, sources=frozen, inputs=inputs, checkpoint=checkpoint, prompts=prompts)


def method_settings(method, config, baseline_root):
    native = method == "native_noadapt"
    constructor = dict(lr=config["lr"] if native else .001, steps=config["steps"] if native else 10)
    if native:
        constructor.update(trace_max_regions=config["trace_max_regions"],
                           trace_view_batch_size=config["trace_view_batch_size"],
                           vision_outputs=list(VISION_OUTPUTS))
    else:
        constructor.update(ovss_type="naclip", ovss_backbone="ViT-L/14")
        if method in {"mlmp", "legacy_trace_noadapt"}:
            constructor.update(alpha_cls=1.0, prompt_integration="loss", vision_outputs=list(VISION_OUTPUTS),
                               prompt_dir=str(baseline_root / "prompts.yaml"))
        if method == "tpt":
            constructor["n_ctx"] = 4
    return dict(constructor=constructor, optimizer="Adam", betas=[.9, .999], weight_decay=0.0,
                update_steps=0 if method in NO_UPDATE else 10,
                reset="none (no updates)" if method in NO_UPDATE else "model and optimizer at each episode",
                backend="native DenseCLIP" if native else "archived MLMP NaCLIP",
                prompt_scope=("native TRACE default multi-prompt text and multi-layer inference" if native else
                              "archived multi-prompt/multi-layer MLMP" if method in {"mlmp", "legacy_trace_noadapt"} else
                              "four learned context tokens, initialized from a photo of a" if method == "tpt" else
                              "single reference prompt: a photo of a {}"),
                backend_equivalence_claim=False,
                comparison_scope="Matched checkpoint/queries/data/episodes; backend and method-specific prompts retained. "
                                 "native_noadapt versus noadapt is not a pure backend-equivalence test; "
                                 "legacy_trace_noadapt matches native prompts/layers for an inference-path control.")


def verify_optimizer(method, settings):
    import torch

    require(type(method.optimizer) is torch.optim.Adam, "Unexpected optimizer")
    for group in method.optimizer.param_groups:
        require(group["lr"] == settings["constructor"]["lr"] and list(group["betas"]) == settings["betas"]
                and group["weight_decay"] == 0, "Runtime optimizer differs from frozen settings")
    require(method.steps == settings["constructor"]["steps"], "Runtime step count mismatch")
    names = {id(p): name for name, p in method.model.named_parameters()}
    if hasattr(method, "ctx"):
        names[id(method.ctx)] = "ctx"
    return [names[id(p)] for group in method.optimizer.param_groups for p in group["params"]]


def summarize_cost(rows):
    require(len(rows) > 2, "Need more than two episodes for warmup2 timing")
    measured = rows[2:]
    n = sum(row["images"] for row in measured)
    return dict(warmup_episodes=2, measured_images=n,
                ms_per_image=1000 * sum(r["total_seconds"] for r in measured) / n,
                adaptation_ms_per_image=1000 * sum(r["adaptation_seconds"] for r in measured) / n,
                inference_ms_per_image=1000 * sum(r["inference_seconds"] for r in measured) / n,
                peak_allocated_mib=max(r["peak_allocated_mib"] for r in measured),
                visual_inputs_per_image=sum(r["inputs"] for r in measured) / n,
                visual_calls_per_image=sum(r["calls"] for r in measured) / n,
                all_visual_inputs=sum(r["inputs"] for r in rows),
                all_visual_calls=sum(r["calls"] for r in rows))


def execute(args, prepared, predictions):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trace_ovss.cli import collapse_class_extensions, set_global_seed
    from trace_ovss.data.datasets import trace_collate
    from trace_ovss.evaluation import SegmentationMetrics, aggregate_patch_logits
    from trace_ovss.method.trace import TRACE

    p = prepared
    config, panel = p["config"], p["panel"]
    panel_key = p["panel_key"]
    panel_identity = dict(panel_key=panel_key, images=len(panel["image_ids"]), full_split=p["full_split"])
    environment = p["suite"]["environment"]
    current_environment = check_environment(args.package_root, environment) if p["full_split"] else None
    torch.set_float32_matmul_precision(environment["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = environment["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = environment["cudnn_tf32"]
    set_global_seed(panel["seed"])
    settings = method_settings(args.method, config, args.baseline_root)
    settings["prompt_templates"] = p["prompts"] if args.method in MULTIPROMPT else ["a photo of a {}"]
    protocol = dict(method=args.method, dataset=args.dataset, **{k: panel[k] for k in (
        "split", "seed", "selection_seed", "image_ids", "label_ids", "file_metadata", "vocabulary_sha256",
        "available_images", "config")}, **panel_identity, method_settings=settings,
        archived_settings=p["archived"], batch_scope="Sealed panel batches replace archived baseline batches",
        source_hashes=p["sources"], input_hashes=p["inputs"], checkpoint_sha256=CHECKPOINT_SHA256,
        adapter_sha256=digest(Path(__file__)),
        package_root=str(args.package_root), baseline_root=str(args.baseline_root),
        suite_protocol_sha256=digest(args.suite_root / "suite_protocol.json"),
        environment=current_environment if current_environment is not None else dict(
                         python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(), float32_matmul_precision=torch.get_float32_matmul_precision(),
                         matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
        timing_scope="GPU-synchronized adaptation and final inference; excludes loading/scoring/model setup",
        warmup_episodes=2, labels_used_for_adaptation=False, command=sys.argv)
    write_json(args.output / "protocol.json", protocol)
    cls = TRACE if args.method == "native_noadapt" else load_baseline(args.baseline_root, args.method)
    method = cls(classes=p["classes"], device="cuda", **settings["constructor"])
    optimizer_parameters = verify_optimizer(method, settings)
    actual_prompts = list(method.prompt_templates) if hasattr(method, "prompt_templates") else [method.prompt]
    require(actual_prompts == settings["prompt_templates"], "Runtime prompt mismatch")
    if args.method in MULTIPROMPT:
        require(tuple(method.vision_outputs) == VISION_OUTPUTS, "Runtime vision output mismatch")
    loader = DataLoader(Subset(p["dataset"], p["indices"]), batch_size=config["batch_size"], shuffle=False,
                        num_workers=args.workers, collate_fn=trace_collate, pin_memory=True)
    metrics = SegmentationMetrics(len(p["vocab"].classes))
    counts, episodes = dict(calls=0, inputs=0), []

    def count_conv1(module, inputs):
        counts["calls"] += 1
        counts["inputs"] += len(inputs[0])

    hook = method.model.visual.conv1.register_forward_pre_hook(count_conv1)
    started = time.perf_counter()
    try:
        with (args.output / "cost.csv").open("x", newline="") as handle:
            writer = None
            for episode, data in enumerate(loader):
                check_batch(panel, len(predictions), data["meta"]["img_path"], data["meta"]["seg_map_path"])
                images = data["img_patches"].cuda(non_blocking=True)
                counts.update(calls=0, inputs=0)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                losses = [] if args.method in NO_UPDATE else method.adapt(images)
                torch.cuda.synchronize()
                adapted = time.perf_counter()
                logits = method.evaluate(images)
                torch.cuda.synchronize()
                inferred = time.perf_counter()
                peak = torch.cuda.max_memory_allocated() / 2**20
                require(counts["calls"] > 0 and counts["inputs"] >= len(images), "Missing conv1 measurements")
                require(len(losses) == settings["update_steps"] and all(np.isfinite(x) for x in losses),
                        "Invalid adaptation losses or update count")
                require(bool(torch.isfinite(logits).all()), "Nonfinite prediction logits")
                outputs = aggregate_patch_logits(logits, data["meta"]["patch_grid_shape"], data["meta"]["img_shape"],
                                                 config["patch_size"], config["patch_stride"])
                require(len(outputs) == len(data["gt"]) == len(data["meta"]["img_path"]), "Incomplete outputs")
                for output, gt, image, label in zip(outputs, data["gt"], data["meta"]["img_path"], data["meta"]["seg_map_path"]):
                    probabilities = output.softmax(0)
                    if p["mapping"] is not None:
                        probabilities = collapse_class_extensions(probabilities, p["mapping"])
                    prediction = probabilities.argmax(0).cpu()
                    before_i, before_u = metrics.intersection.clone(), metrics.union.clone()
                    metrics.update(prediction, gt[0])
                    predictions.append(dict(image_id=image, label_id=label, episode=episode,
                        prediction_sha256=hashlib.sha256(prediction.numpy().tobytes()).hexdigest(),
                        intersection=(metrics.intersection-before_i).tolist(), union=(metrics.union-before_u).tolist()))
                row = dict(episode=episode, images=len(outputs), image_ids=json.dumps(data["meta"]["img_path"]),
                           patches=len(images), adaptation_seconds=adapted-tick, inference_seconds=inferred-adapted,
                           total_seconds=inferred-tick, peak_allocated_mib=peak, update_steps=len(losses), **counts)
                episodes.append(row)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(f"{args.dataset}/{args.method}: {len(predictions)}/{len(panel['image_ids'])}", flush=True)
    finally:
        hook.remove()
    require([r["image_id"] for r in predictions] == panel["image_ids"], "Incomplete or reordered predictions")
    require([r["label_id"] for r in predictions] == panel["label_ids"], "Incomplete or reordered prediction labels")
    present = metrics.union > 0
    require(bool(present.any()), "No scorable classes in panel")
    miou = float((metrics.intersection[present].double() / metrics.union[present].double()).mean()) * 100
    return dict(status="completed", method=args.method, dataset=args.dataset, **panel_identity,
                metrics=metrics.compute(), miou_unrounded=miou, method_settings=settings,
                optimizer_parameters=optimizer_parameters, intersection=metrics.intersection.tolist(),
                union=metrics.union.tolist(), wall_seconds=time.perf_counter()-started, **summarize_cost(episodes))


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.log.write(value)
        return self.stream.write(value)

    def flush(self):
        self.log.flush()
        self.stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=tuple(POPULATIONS), required=True)
    parser.add_argument("--panel-key", help="Sealed input_panels key: dataset/128 (default) or dataset/full")
    for name in ("suite-root", "package-root", "baseline-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.panel_key is not None and args.panel_key not in {args.dataset + "/128", args.dataset + "/full"}:
        parser.error("--panel-key must match --dataset and end in /128 or /full")
    for name in ("suite_root", "package_root", "baseline_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # A fresh directory also prevents concurrent writers and retries over partial evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    with (args.output / "stdout.log").open("x") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            prepared, predictions, failure = None, [], None
            try:
                prepared = preflight(args)
                result = execute(args, prepared, predictions)
            except BaseException as error:
                failure = dict(error=type(error).__name__, message=str(error))
                traceback.print_exc()
            if prepared is not None:
                try:
                    if prepared.get("full_split", False):
                        full_suite_controls(args, prepared["suite"], read_json(args.suite_root / "SUITE_SEALED.json"),
                                            prepared["panel"])
                        check_environment(args.package_root, prepared["suite"]["environment"])
                    check_snapshot(prepared["sources"], source_snapshot(args.package_root, args.baseline_root,
                                                                       prepared["controls"]))
                    require(input_snapshot(prepared["panel"]) == prepared["inputs"], "Input content hashes changed")
                except BaseException as error:
                    failure = dict(failure or {}, integrity_error=f"{type(error).__name__}: {error}")
                    traceback.print_exc()
            write_json(args.output / "per_image_counts.json", predictions)
            if failure is not None:
                write_json(args.output / "failure.json", dict(failure, completed_images=len(predictions)))
                raise RuntimeError(f"Cell failed; evidence retained in {args.output}")
            result.update(source_input_checks="pre/post passed",
                          adapter_sha256=digest(Path(__file__)),
                          protocol_sha256=digest(args.output / "protocol.json"),
                          per_image_counts_sha256=digest(args.output / "per_image_counts.json"),
                          cost_sha256=digest(args.output / "cost.csv"))
            write_json(args.output / "result.json", result)
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
