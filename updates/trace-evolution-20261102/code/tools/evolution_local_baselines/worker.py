"""Hash-bound native local baselines; only score opens annotation contents.

Importing this module never imports the archived methods, constructs a model, or
initializes CUDA. Prediction and gates are single attempts, not resumable jobs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import inspect
import io
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_local_baselines import common

require, read_json, write_json = common.require, common.read_json, common.write_json
sha256, canonical_hash = common.sha256, common.canonical_hash
ROOT, DATASETS, METHODS, JOBS = common.ROOT, common.DATASETS, common.METHODS, common.JOBS
ALGORITHM = common.ALGORITHM
NATIVE = dict(mlmp=("mlmp", "MLMP"), noadapt=("tent", "TENT"),
              tent=("tent", "TENT"), tpt=("tpt", "TPT"),
              watt=("watt", "WATT"), clipartt=("clipartt", "CLIPARTT"))
COUNTERS = ("optimizer_steps", "loss_entries", "visual_calls", "visual_inputs")
SCORE_FILES = {"SCORING_STARTED.json", "score_inputs.json", "scores_by_image.jsonl", "result.json"}
TIMING_SCOPE = (
    "Synchronized native model-side wallclock: native adaptation/reset, finite/state checks, "
    "optimizer and visual hooks, final native evaluate, reconstruction and CPU transfer. "
    "Excludes construction, RGB decode/H2D, artifact writes and CPU scoring. Episode 0 "
    "excluded only from timings; all images and all actual operations remain counted."
)


def scope(manifest):
    return {k: manifest[k] for k in ("algorithm", "dataset", "method", "stage", "seed")}


def finite_json(value):
    """Failure contexts must remain serializable even for NaN/Inf failures."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    return repr(value)


def fail(message, **context):
    error = RuntimeError(message)
    error.worker_context = finite_json(context)
    raise error


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def require_finite(value, phase):
    # Group boolean observations by their real device; never stack CPU Adam
    # step counters with CUDA moments and never repair native numeric values.
    groups = {}
    for tensor in tensors(value):
        if tensor.is_floating_point() or tensor.is_complex():
            groups.setdefault(tensor.device, []).append(torch.isfinite(tensor.detach()).all())
    for device, flags in groups.items():
        if not bool(torch.stack(flags).all()):
            fail("Nonfinite native observation", phase=phase, device=str(device))


def guard(manifest, paths):
    common.guard_paths(manifest, [Path(p) for p in paths])


def verify_inputs(manifest):
    """RGB bytes plus vocabulary; annotations are metadata-only here."""
    panel, cfg = manifest["panel"], manifest["config"]
    rows = panel["inputs"]
    require(panel["ordered_ids"] == [r["official_id"] for r in rows], "Input identity/order drift")
    require(len(set(panel["ordered_ids"])) == len(rows), "Duplicate official identity")
    common.annotation_inodes(manifest)  # All eight full panels, not only this job.
    paths = [r["image"] for r in rows]
    paths += [str(p) for p in Path(manifest["baseline_root"]).rglob("*") if p.is_file()]
    paths += [manifest["checkpoint"]["path"]]
    guard(manifest, paths)
    for row in rows:
        confirmation.verify_label_metadata(row)
        require(sha256(row["image"]) == row["image_sha256"], "RGB content drift")
        if "label_metadata_details" in row:
            s, expected = Path(row["label"]).stat(), row["label_metadata_details"]
            actual = dict(size_bytes=s.st_size, mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino)
            require(all(expected[k] == v for k, v in actual.items()), "Annotation inode drift")
    data = confirmation.make_data(cfg, rows)
    vocab = data.vocabulary
    queries = list(vocab.extensions if cfg["class_extensions"] else vocab.classes)
    mapping = list(vocab.extension_to_class) if cfg["class_extensions"] else None
    require(len(data) == len(rows) and list(vocab.classes) == panel["classes"]
            and queries == panel["queries"] and mapping == panel["query_to_class"]
            and data.spec.reduce_zero_label == panel["reduce_zero_label"], "Frozen vocabulary/config drift")
    if "vocabulary_sha256" in panel:
        require(canonical_hash([queries, mapping]) == panel["vocabulary_sha256"], "Vocabulary hash drift")
    require([str(p[0]) for p in data.samples] == paths[:len(rows)], "RGB discovery order drift")
    return data


def started(output, manifest_sha, phase):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = output.parent / f"{output.name}.STARTED.json"
    require(not output.exists() and not attempt.exists(), "No retry/resume/overwrite")
    receipt = dict(phase=phase, manifest_sha256=manifest_sha, output=str(output), retry_allowed=False)
    # Exclusive creation outside the deletable output directory is intentional.
    with attempt.open("x") as stream:
        json.dump(receipt, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
    output.mkdir()
    write_json(output / "STARTED.json", receipt)
    return receipt


@contextmanager
def attempted(output, manifest_sha, phase):
    started(output, manifest_sha, phase)
    try:
        yield
    except BaseException as error:
        destination = Path(output) / ("GATE_FAILED.json" if phase == "gate" else "PREDICTION_FAILED.json")
        if not destination.exists():
            write_json(destination, dict(status="failed", phase=phase, manifest_sha256=manifest_sha,
                error=f"{type(error).__name__}: {error}"[:4000], retry_allowed=False,
                context=finite_json(getattr(error, "worker_context", {}))))
        raise


def constructor_kwargs(manifest):
    """Only official constructor primitives; the backbone is a pinned filepath."""
    method, settings = manifest["method"], manifest["settings"]
    require(method in NATIVE and (manifest["dataset"], method) in JOBS, "Unauthorized baseline pair")
    checkpoint = Path(manifest["checkpoint"]["path"])
    require(checkpoint.is_absolute() and checkpoint.is_file(), "Pinned checkpoint missing")
    require(sha256(checkpoint) == manifest["checkpoint"]["sha256"], "Checkpoint content drift")
    kwargs = dict(ovss_type="naclip", ovss_backbone=str(checkpoint), lr=.001,
                  device="cuda")
    if method != "watt":
        kwargs["steps"] = 10
    if method in ("mlmp", "watt"):
        kwargs["prompt_dir"] = str(Path(manifest["baseline_root"]) / "prompts.yaml")
    if method == "mlmp":
        kwargs.update(vision_outputs=tuple(range(-1, -19, -1)), alpha_cls=1., prompt_integration="loss")
    elif method == "tpt":
        kwargs["n_ctx"] = 4
    elif method == "watt":
        kwargs.update(watt_l=2, watt_m=5)
    elif method == "clipartt":
        kwargs["clipartt_k"] = 3
    primitive = dict(kwargs)
    if method == "mlmp":
        primitive["vision_outputs"] = list(primitive["vision_outputs"])
    require(primitive == settings["constructor"], "Native constructor primitives differ")
    kwargs.update(classes=manifest["panel"]["queries"], runtime_calculation=False)
    return kwargs


def verify_native_imports(root):
    """Authenticate regular modules and every namespace search location."""
    root = Path(root).resolve()
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("adapt", "ovss", "utils", "mmengine"):
            filename = getattr(module, "__file__", None)
            paths = getattr(module, "__path__", None)
            if filename is not None:
                valid = Path(filename).resolve().is_relative_to(root)
            else:
                spec = getattr(module, "__spec__", None)
                valid = (paths is not None and spec is not None and spec.origin is None
                         and spec.submodule_search_locations is not None
                         and all(Path(p).resolve().is_relative_to(root) for p in spec.submodule_search_locations))
            if paths is not None:
                locations = list(paths)
                valid = valid and bool(locations) and all(Path(p).resolve().is_relative_to(root) for p in locations)
            require(valid, f"Contaminated native import: {name}")


def build_method(manifest):
    root = Path(manifest["baseline_root"]).resolve()
    kwargs = constructor_kwargs(manifest)
    # Never delete or reload an existing foreign or mixed native namespace.
    verify_native_imports(root)
    sys.path.insert(0, str(root))
    module_name, class_name = NATIVE[manifest["method"]]
    cls = getattr(importlib.import_module(f"adapt.{module_name}"), class_name)
    require(Path(inspect.getfile(cls)).resolve() == root / "adapt" / f"{module_name}.py", "Wrong native class")
    verify_native_imports(root)
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    method = cls(**kwargs)
    verify_native_imports(root)
    actual_prompts = getattr(method, "prompt_templates", [method.prompt] if hasattr(method, "prompt") else None)
    require(actual_prompts == manifest["settings"]["prompt_templates"], "Native prompt drift")
    require_finite(method.model.state_dict(), "constructed-model")
    if hasattr(method, "ctx"):
        require_finite(method.ctx, "constructed-ctx")
    return method


def native_parameters(method):
    params = dict(method.model.named_parameters())
    if hasattr(method, "ctx"):
        require(id(method.ctx) not in {id(p) for p in params.values()}, "TPT ctx unexpectedly inside backbone")
        params["ctx"] = method.ctx
    return params


def optimized_parameters(method):
    named = native_parameters(method)
    inverse = {id(p): name for name, p in named.items()}
    actual = [p for g in method.optimizer.param_groups for p in g["params"]]
    require(actual and len({id(p) for p in actual}) == len(actual), "Empty/duplicate optimizer membership")
    require(all(id(p) in inverse and p.requires_grad for p in actual), "Unknown/frozen optimized parameter")
    names = [inverse[id(p)] for p in actual]
    if hasattr(method, "ctx"):
        require(names == ["ctx"], "TPT must optimize only native external ctx")
    else:
        selected, selected_names = method.collect_ln_params(method.model.visual)
        require(len(selected) == len(selected_names) and len({id(p) for p in selected}) == len(selected),
                "Duplicate/invalid native parameter selection")
        require([id(p) for p in actual] == [id(p) for p in selected],
                "Ordered optimizer membership differs from native selection")
        require(names == list(selected_names), "Native selected parameter names differ")
    return {name: named[name] for name in names}


def native_written_parameters(method, method_name):
    """Adam parameters plus the exact all-LayerNorm set WATT averages."""
    written = optimized_parameters(method)
    if method_name == "watt":
        named = dict(method.model.named_parameters())
        for name, module in method.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                for key, parameter in module.named_parameters():
                    if key in ("weight", "bias"):
                        full = f"{name}.{key}" if name else key
                        require(full in named and named[full] is parameter, "Unknown native WATT LayerNorm parameter")
                        written.setdefault(full, parameter)
    return written


def exact_state(actual, expected, phase):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                and actual.dtype == expected.dtype and torch.equal(actual.detach(), expected.detach().to(actual.device)),
                f"Native reset tensor differs: {phase}")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Native reset keys differ: {phase}")
        for key in expected:
            exact_state(actual[key], expected[key], f"{phase}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected), f"Native reset list differs: {phase}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            exact_state(a, b, f"{phase}.{i}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Native reset scalar differs: {phase}")


class NativeState:
    """Understand native full snapshots, including TPT's non-model ctx."""

    def __init__(self, method, *, method_name=None):
        self.method = method
        self.method_name = method_name or type(method).__name__.lower()
        require(isinstance(method.optimizer, torch.optim.Adam), "Native optimizer must be Adam")
        for group in method.optimizer.param_groups:
            require(group["lr"] == .001 and group["betas"] == (.9, .999)
                    and group["weight_decay"] == 0., "Native Adam hyperparameter drift")
        self.optimized = optimized_parameters(method)
        self.written = native_written_parameters(method, self.method_name)
        if hasattr(method, "ctx"):
            require(set(method._clean_state) == {"model", "ctx", "optim"}, "Invalid native TPT clean state")
            self.model = method._clean_state["model"]
            self.optimizer = method._clean_state["optim"]
            self.ctx = method._clean_state["ctx"]
        else:
            self.model, self.optimizer, self.ctx = method.model_state, method.optimizer_state, None
        require(set(self.model) == set(method.model.state_dict()), "Native snapshot must cover full backbone")
        self.saved_sha = shared.state_digest(dict(model=self.model, optimizer=self.optimizer, ctx=self.ctx))
        self.initial = self.check()

    def check(self):
        method = self.method
        require(shared.state_digest(dict(model=self.model, optimizer=self.optimizer, ctx=self.ctx)) == self.saved_sha,
                "Native saved snapshot changed")
        require(list(optimized_parameters(method)) == list(self.optimized), "Optimizer membership changed")
        exact_state(method.model.state_dict(), self.model, "model")
        exact_state(method.optimizer.state_dict(), self.optimizer, "optimizer")
        if self.ctx is not None:
            exact_state(method.ctx, self.ctx, "ctx")
        return dict(snapshot_sha256=self.saved_sha,
                    model_sha256=shared.state_digest(method.model.state_dict()),
                    trainable_sha256=shared.state_digest({n: p for n, p in native_parameters(method).items() if p.requires_grad}),
                    optimized_parameter_names=list(self.optimized),
                    native_trainable_parameter_names=[n for n, p in native_parameters(method).items() if p.requires_grad],
                    nonoptimized_parameter_names=[n for n in native_parameters(method) if n not in self.optimized],
                    optimizer_sha256=shared.state_digest(method.optimizer.state_dict()),
                    ctx_sha256=shared.state_digest(method.ctx) if self.ctx is not None else None)

    def check_nonoptimized(self):
        current = native_written_parameters(self.method, self.method_name)
        require(list(current) == list(self.written)
                and all(current[n] is self.written[n] for n in current), "Native-written membership changed")
        require_finite(current, "native-written-parameters")
        # WATT explicitly writes every LN, including frozen text LNs. Their
        # native averaging roundoff is compared by the gate, not suppressed.
        for name, parameter in self.method.model.named_parameters():
            if name not in self.written:
                exact_state(parameter, self.model[name], f"nonoptimized.{name}")


class RawFiniteMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "nan_to_num" in str(func):
            require_finite((args, kwargs or {}), "before-native-nan-to-num")
        return func(*args, **(kwargs or {}))


def capture_parameters(method, parameters=None):
    selected = optimized_parameters(method) if parameters is None else parameters
    return {n: p.detach().cpu().clone() for n, p in selected.items()}


def capture_gradients(method):
    return {n: p.grad.detach().cpu().clone() if p.grad is not None else None
            for n, p in optimized_parameters(method).items()}


class NativeObservation:
    """Read-only hooks on actual native reset, Adam steps and conv1 calls."""

    def __init__(self, method, state, *, audited=True, capture=False):
        self.method, self.state, self.audited, self.capture = method, state, audited, capture
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.resets, self.steps = [], []

    def __enter__(self):
        self.stack = ExitStack()
        method = self.method
        original_reset = method.reset

        def reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            self.resets.append(self.state.check())
            return result

        # Restore instance lookup exactly, including the absence of an override.
        old = method.__dict__.get("reset")
        existed = "reset" in method.__dict__
        method.reset = reset
        self.stack.callback(lambda: setattr(method, "reset", old) if existed else delattr(method, "reset"))

        def pre_step(optimizer, args, kwargs):
            if self.audited:
                require_finite([p for p in native_parameters(method).values()], "pre-step-parameters")
                require_finite([p.grad for p in native_parameters(method).values()], "pre-step-gradients")
                require_finite(optimizer.state_dict(), "pre-step-Adam-state")
                require(any(p.grad is not None for p in self.state.optimized.values()), "Vacuous optimizer update")
            if self.capture:
                self.steps.append(dict(gradients=capture_gradients(method),
                    pre_parameters=capture_parameters(method, self.state.written)))

        def post_step(optimizer, args, kwargs):
            self.counts["optimizer_steps"] += 1
            if self.audited:
                require_finite(native_parameters(method), "post-step-parameters")
                require_finite(optimizer.state_dict(), "post-step-all-Adam-tensors")
            if self.capture:
                self.steps[-1]["parameters"] = capture_parameters(method, self.state.written)
                self.steps[-1]["optimizer_sha256"] = shared.state_digest(optimizer.state_dict())
                self.steps[-1]["adam_tensor_devices"] = sorted({str(t.device) for t in tensors(optimizer.state_dict())})

        def visual(module, args):
            self.counts["visual_calls"] += 1
            self.counts["visual_inputs"] += int(args[0].shape[0])

        try:
            self.stack.callback(method.optimizer.register_step_pre_hook(pre_step).remove)
            self.stack.callback(method.optimizer.register_step_post_hook(post_step).remove)
            self.stack.callback(method.model.visual.conv1.register_forward_pre_hook(visual).remove)
            if self.audited:
                for name, module in method.model.named_modules():
                    def raw(module, args, output, name=name):
                        require_finite(output, f"raw-module:{name}")
                    self.stack.callback(module.register_forward_hook(raw).remove)
                self.stack.enter_context(RawFiniteMode())
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


@contextmanager
def checked_evaluate(method):
    original = method.evaluate
    old, existed = method.__dict__.get("evaluate"), "evaluate" in method.__dict__

    def evaluate(*args, **kwargs):
        logits = original(*args, **kwargs)
        require_finite(logits, "native-evaluate-logits-before-softmax-argmax")
        return logits

    method.evaluate = evaluate
    try:
        yield
    finally:
        if existed:
            method.evaluate = old
        else:
            del method.evaluate


def validate_prediction(prediction, manifest):
    require(prediction.ndim == 2 and prediction.shape == (224, 224)
            and np.issubdtype(prediction.dtype, np.integer)
            and (prediction >= 0).all() and (prediction < len(manifest["panel"]["classes"])).all(),
            "Invalid native prediction labels/geometry")


def validate_cost(row, manifest, episode):
    n, settings = len(manifest["panel"]["inputs"]), manifest["settings"]
    images = min(2, n - 2 * episode)
    require(images > 0 and all(row[k] == v for k, v in scope(manifest).items()), "Cost scope drift")
    require(type(row["episode"]) is int and row["episode"] == episode
            and type(row["images"]) is int and row["images"] == images
            and type(row["patches"]) is int and row["patches"] == images
            and type(row["warmup"]) is bool and row["warmup"] == (episode == 0), "Cost order/batch drift")
    require(all(type(row[k]) is int and row[k] >= 0 for k in COUNTERS), "Noninteger actual operation count")
    expected = dict(optimizer_steps=settings["expected_update_steps"],
                    loss_entries=settings["expected_loss_entries"], visual_calls=settings["expected_visual_calls"])
    require(all(row[k] == v for k, v in expected.items())
            and row["visual_inputs"] == row["visual_calls"] * images,
            "Actual native optimizer/loss/visual budget differs from frozen settings")
    require(finite_number(row["seconds"]) and row["seconds"] >= 0, "Invalid native timing")
    require(row["native_reset_verified"] is True and row["gt_contents_opened"] is False, "Invalid native episode isolation")


def run_episode(method, state, images, meta, manifest, episode, *, audited=True, capture=False):
    observation = NativeObservation(method, state, audited=audited, capture=capture)
    tick = time.perf_counter()
    try:
        return _run_episode(method, state, images, meta, manifest, episode, observation, capture=capture)
    except BaseException as error:
        error.worker_context = finite_json(dict(getattr(error, "worker_context", {}),
            **scope(manifest), episode=episode, actual_partial_counters=observation.counts,
            expected_settings=manifest["settings"], seconds_until_failure=time.perf_counter()-tick))
        raise


def _run_episode(method, state, images, meta, manifest, episode, observed, *, capture=False):
    require(list(meta["img_path"]) == [r["image"] for r in manifest["panel"]["inputs"][2*episode:2*episode+2]],
            "Actual native RGB batch order differs")
    require(int(images.shape[0]) == len(meta["img_path"]), "Native fixed crop/batch differs")
    with observed:
        shared.synchronize(torch.device(method.device))
        tick = time.perf_counter()
        if manifest["method"] == "noadapt":
            method.reset()
            losses = []
        else:
            losses = method.adapt(images)
        require(isinstance(losses, (list, tuple)) and all(math.isfinite(float(x)) for x in losses), "Nonfinite native loss")
        observed.counts["loss_entries"] = len(losses)
        require(len(observed.resets) == 1 and observed.resets[0] == state.initial,
                "Native adapt must invoke its exact reset once")
        with checked_evaluate(method):
            predictions = shared.label_predictions(method, images, meta, manifest["config"], manifest["panel"]["query_to_class"])
        require_finite(native_parameters(method), "post-adapt-parameters")
        require_finite(method.optimizer.state_dict(), "post-adapt-Adam")
        state.check_nonoptimized()
        shared.synchronize(torch.device(method.device))
        seconds = time.perf_counter() - tick
    require(len(predictions) == len(meta["img_path"]), "Missing native prediction")
    for prediction in predictions:
        validate_prediction(prediction, manifest)
    row = dict(**scope(manifest), episode=episode, images=len(predictions), patches=int(images.shape[0]),
               seconds=seconds, warmup=episode == 0, native_reset_verified=True,
               gt_contents_opened=False, **observed.counts)
    validate_cost(row, manifest, episode)
    final = capture_parameters(method, state.written) if capture else None
    method.reset()
    require(state.check() == state.initial, "Post-episode native reset drift")
    return row, [float(x) for x in losses], predictions, observed.steps, final


def artifact_names(manifest):
    n = len(manifest["panel"]["inputs"])
    return ({f"predictions/{i:06d}.npy" for i in range(n)}
            | {f"episodes/{e:06d}.json" for e in range((n + 1) // 2)}
            | {"STARTED.json", "initial_state.json", "prediction_protocol.json", "prediction_result.json", "cost.jsonl"})


def bounded_lines(path, maximum):
    with Path(path).open() as stream:
        i = 0
        while line := stream.readline(2_000_001):
            require(i < maximum and len(line) < 2_000_000, "Oversized/extra stream row")
            value = json.loads(line, parse_constant=lambda x: fail("Nonfinite JSON constant", value=x))
            require(isinstance(value, dict), "Invalid stream record")
            yield value
            i += 1


def summarize(rows):
    costs = dict.fromkeys(COUNTERS, 0)
    seconds, images, episodes = 0., 0, 0
    for row in rows:
        for k in costs:
            costs[k] += row[k]
        if not row["warmup"]:
            seconds += row["seconds"]
            images += row["images"]
            episodes += 1
    return dict(costs=costs, timings=dict(measured_episodes=episodes, measured_images=images,
        seconds=seconds if episodes else None, seconds_per_image=seconds / images if images else None,
        images_per_second=images / seconds if images and seconds > 0 else None,
        warmup_episode=0, accuracy_includes_warmup=True), timing_scope=TIMING_SCOPE)


def validate_cost_stream(output, manifest):
    output = Path(output)
    n = len(manifest["panel"]["inputs"])
    count = 0

    def rows():
        nonlocal count
        for episode, row in enumerate(bounded_lines(output / "cost.jsonl", (n + 1) // 2)):
            validate_cost(row, manifest, episode)
            receipt = read_json(output / f"episodes/{episode:06d}.json")
            require(all(receipt[k] == v for k, v in row.items()), "Episode/cost stream drift")
            require(receipt["ordered_ids"] == manifest["panel"]["ordered_ids"][2*episode:2*episode+2], "Episode identity order drift")
            losses = receipt["losses"]
            require(isinstance(losses, list) and len(losses) == row["loss_entries"]
                    and all(finite_number(v) for v in losses), "Invalid native loss stream")
            require(receipt["reset_state"] == read_json(output / "initial_state.json")["native_state"], "Episode reset state drift")
            count += 1
            yield row

    summary = summarize(rows())
    require(count == (n + 1) // 2, "Missing cost/episode rows")
    result = read_json(output / "prediction_result.json")
    require(all(result.get(k) == v for k, v in scope(manifest).items())
            and result["images"] == n and result["status"] == "predicted"
            and result["config"] == manifest["config"]
            and result["settings"] == manifest["settings"], "Prediction result binding differs")
    require(all(result[k] == v for k, v in summary.items()), "Native aggregate cost/timing drift")
    return summary


def inventory(output):
    return {str(p.relative_to(output)) for p in Path(output).rglob("*") if p.is_file()}


def bound_output(output, manifest, manifest_sha):
    output = Path(output)
    require(str(output) == manifest["output"] and len(manifest_sha) == 64, "Output/manifest binding differs")
    path = common.manifest_path(Path(manifest["suite_root"]), manifest["dataset"], manifest["method"], manifest["stage"])
    guard(manifest, [path])
    require(sha256(path) == manifest_sha and read_json(path) == manifest, "Manifest source binding drift")
    common.verify_prepared(Path(manifest["suite_root"]), manifest["suite_prepared_sha256"], frozen=True)
    checkpoint = Path(manifest["checkpoint"]["path"])
    guard(manifest, [checkpoint])
    require(sha256(checkpoint) == manifest["checkpoint"]["sha256"], "Pinned checkpoint drift")
    return output


def verify_prediction_seal(output, manifest, manifest_sha):
    output = bound_output(output, manifest, manifest_sha)
    guard(manifest, [p for p in output.rglob("*") if p.is_file()])
    sealed = read_json(output / "PREDICTIONS_SEALED.json")
    require(sealed["complete"] is True and sealed["gt_contents_opened"] is False
            and sealed["manifest_sha256"] == manifest_sha and sealed["output"] == str(output)
            and sealed["suite_prepared_sha256"] == manifest["suite_prepared_sha256"]
            and sealed["images"] == len(manifest["panel"]["inputs"])
            and all(sealed.get(k) == v for k, v in scope(manifest).items()), "Prediction seal scope differs")
    expected = artifact_names(manifest)
    require(set(sealed["artifact_sha256"]) == expected, "Prediction artifact set differs")
    allowed = expected | {"PREDICTIONS_SEALED.json"} | SCORE_FILES | {"SCORED_SEALED.json", "SCORING_FAILED.json"}
    require(inventory(output) <= allowed and expected <= inventory(output), "Unexpected prediction payload")
    for name, digest in sealed["artifact_sha256"].items():
        require(sha256(output / name) == digest, f"Prediction artifact drift: {name}")
    protocol = read_json(output / "prediction_protocol.json")
    require(protocol["manifest"] == manifest and protocol["manifest_sha256"] == manifest_sha
            and protocol["output"] == str(output) and protocol["gt_contents_opened"] is False
            and protocol["timing_scope"] == TIMING_SCOPE, "Prediction protocol binding differs")
    require(protocol["gates"] == common.verify_gates(manifest), "Prediction gate provenance differs")
    attempt = read_json(output / "STARTED.json")
    external = output.parent / f"{output.name}.STARTED.json"
    guard(manifest, [external])
    require(attempt == read_json(external) and attempt == dict(phase="predict", manifest_sha256=manifest_sha,
        output=str(output), retry_allowed=False), "Persistent native attempt binding differs")
    initial = read_json(output / "initial_state.json")
    require(initial["checkpoint"] == manifest["checkpoint"]
            and initial["baseline_source_sha256"] == manifest["baseline_source_sha256"]
            and initial["settings"] == manifest["settings"], "Native initial provenance differs")
    validate_cost_stream(output, manifest)
    for i in range(len(manifest["panel"]["inputs"])):
        validate_prediction(np.load(output / f"predictions/{i:06d}.npy", allow_pickle=False), manifest)
    return sealed


def pooled_result(counts):
    counts = np.asarray(counts)
    require(counts.ndim == 2 and counts.shape[0] == 2 and np.issubdtype(counts.dtype, np.integer)
            and (counts >= 0).all() and (counts[0] <= counts[1]).all(), "Invalid pooled intersection/union")
    intersection, union = counts
    present = union > 0
    return dict(intersection=intersection.tolist(), union=union.tolist(),
        miou_unrounded=100 * math.fsum(int(i) / int(u) for i, u in zip(intersection, union, strict=True) if u)
        / int(present.sum()) if present.any() else None)


def verify_score_seal(output, manifest, manifest_sha, barrier=None):
    require(manifest["stage"] == "full", "Only full baseline cells may score")
    verify_prediction_seal(output, manifest, manifest_sha)
    output = Path(output)
    sealed = read_json(output / "SCORED_SEALED.json")
    prediction_sha = sha256(output / "PREDICTIONS_SEALED.json")
    require(sealed["complete"] is True and sealed["manifest_sha256"] == manifest_sha
            and sealed["prediction_seal_sha256"] == prediction_sha
            and set(sealed["artifact_sha256"]) == SCORE_FILES, "Score seal scope differs")
    require(not (output / "SCORING_FAILED.json").exists(), "Failed scoring cannot pass verification")
    for name, digest in sealed["artifact_sha256"].items():
        require(sha256(output / name) == digest, f"Score artifact drift: {name}")
    barrier = Path(barrier) if barrier is not None else Path(manifest["suite_root"]) / "ALL_FULL_PREDICTIONS_SEALED.json"
    require(common.verify_barrier(barrier, manifest) == sealed["barrier_sha256"], "Score barrier drift")
    result, inputs = read_json(output / "result.json"), read_json(output / "score_inputs.json")
    start = read_json(output / "SCORING_STARTED.json")
    for value in (result, inputs, start):
        require(value["manifest_sha256"] == manifest_sha
                and value["prediction_seal_sha256"] == prediction_sha
                and value["barrier_sha256"] == sealed["barrier_sha256"], "Score proof binding differs")
    require(all(result.get(k) == v for k, v in scope(manifest).items())
            and result["status"] == "completed" and result["config"] == manifest["config"]
            and result["settings"] == manifest["settings"]
            and result["label_contents_first_opened_after_all18_full_predictions_sealed"] is True,
            "Score result contract differs")
    totals = np.zeros((2, len(manifest["panel"]["classes"])), dtype=np.int64)
    rows, n = manifest["panel"]["inputs"], len(manifest["panel"]["inputs"])
    count = 0
    require(set(inputs["label_sha256"]) == {r["label"] for r in rows}, "Scored label set differs")
    for i, row in enumerate(bounded_lines(output / "scores_by_image.jsonl", n)):
        require(type(row["index"]) is int and row["index"] == i and row["official_id"] == rows[i]["official_id"],
                "Score stream identity/order differs")
        require(row["prediction_sha256"] == sha256(output / f"predictions/{i:06d}.npy")
                and row["label_sha256"] == inputs["label_sha256"][rows[i]["label"]], "Score stream input binding differs")
        counts = np.asarray([row["intersection"], row["union"]])
        require(counts.shape == totals.shape and row["pooled"] == pooled_result(counts), "Score stream counts differ")
        require(int(counts[1].sum()) <= 2 * 224 * 224, "Impossible score union counts")
        totals += counts
        count += 1
    require(count == n and result["images"] == n and result["pooled"] == pooled_result(totals), "Pooled score/stream mismatch")
    require(all(result[k] == v for k, v in result["pooled"].items()), "Canonical top-level pooled result differs")
    summary = validate_cost_stream(output, manifest)
    require(all(result[k] == v for k, v in summary.items()), "Scored cost/timing differs")
    return result


def prepare_gpu_attempt(args, phase):
    manifest, output = common.admission(args)
    require(manifest["stage"] == ("gate" if phase == "gate" else "full"), "Worker phase/stage mismatch")
    common.require_gpu_envelope(manifest)  # Before attempt markers, model, or CUDA.
    if phase == "predict":
        common.verify_gates(manifest)
    return manifest, output


def predict(args):
    manifest, output = prepare_gpu_attempt(args, "predict")
    with attempted(output, args.manifest_sha256, "predict"):
        data = verify_inputs(manifest)
        with confirmation.prior.gpu_lock() as lock:
            runtime = shared.configure_deterministic_runtime()
            gates = common.verify_gates(manifest)
            write_json(output / "prediction_protocol.json", dict(manifest=manifest, manifest_path=str(args.manifest),
                manifest_sha256=args.manifest_sha256, output=str(output), gt_contents_opened=False,
                gpu_lock=lock, deterministic_runtime=runtime, gates=gates, timing_scope=TIMING_SCOPE))
            method = build_method(manifest)
            state = NativeState(method, method_name=manifest["method"])
            write_json(output / "initial_state.json", dict(native_state=state.initial, checkpoint=manifest["checkpoint"],
                baseline_source_sha256=manifest["baseline_source_sha256"], settings=manifest["settings"]))
            (output / "predictions").mkdir()
            (output / "episodes").mkdir()
            n = len(data)
            with (output / "cost.jsonl").open("x") as stream:
                for episode, start in enumerate(range(0, n, 2)):
                    batch = shared.image_collate([data[i] for i in range(start, min(start+2, n))])
                    images, meta = batch["img_patches"].to(method.device), batch["meta"]
                    row, losses, predictions, _, _ = run_episode(method, state, images, meta, manifest, episode)
                    for offset, prediction in enumerate(predictions):
                        with (output / f"predictions/{start+offset:06d}.npy").open("xb") as destination:
                            np.save(destination, prediction, allow_pickle=False)
                    write_json(output / f"episodes/{episode:06d}.json", dict(row, losses=losses,
                        ordered_ids=manifest["panel"]["ordered_ids"][start:start+len(predictions)], reset_state=state.initial))
                    stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
                    stream.flush()
            summary = summarize(bounded_lines(output / "cost.jsonl", (n+1)//2))
            write_json(output / "prediction_result.json", dict(status="predicted", images=n, config=manifest["config"],
                settings=manifest["settings"], **scope(manifest), **summary))
            require(shared.deterministic_runtime_state() == runtime, "Deterministic runtime drift")
            common.admission(args)
            verify_inputs(manifest)
            common.verify_prepared(Path(manifest["suite_root"]), manifest["suite_prepared_sha256"], frozen=True)
            require(common.verify_gates(manifest) == gates, "Native gate binding drift")
            required = artifact_names(manifest)
            require(inventory(output) == required, "Unexpected unsealed prediction artifact")
            write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
                output=str(output), manifest_sha256=args.manifest_sha256, images=n, **scope(manifest),
                suite_prepared_sha256=manifest["suite_prepared_sha256"],
                artifact_sha256={name: sha256(output / name) for name in sorted(required)}))
            verify_prediction_seal(output, manifest, args.manifest_sha256)
    return summary


def maximum_delta(a, b):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, "Gate tensor structure differs")
        require_finite((a, b), "gate-comparison")
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), "Gate tensor membership differs")
        return max((maximum_delta(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        require(type(a) is type(b) and len(a) == len(b), "Gate trajectory length differs")
        return max((maximum_delta(x, y) for x, y in zip(a, b, strict=True)), default=0.)
    if a is None or b is None:
        require(a is b, "Gate missing gradient differs")
        return 0.
    require(finite_number(a) and finite_number(b), "Invalid gate scalar")
    return abs(a-b)


def gate(args):
    manifest, output = prepare_gpu_attempt(args, "gate")
    require(len(manifest["panel"]["inputs"]) == 4 and manifest["config"]["batch_size"] == 2,
            "Real native gate requires four RGB images/two batch-2 episodes")
    with attempted(output, args.manifest_sha256, "gate"):
        data = verify_inputs(manifest)
        with confirmation.prior.gpu_lock() as lock:
            runtime = shared.configure_deterministic_runtime()
            method, reference = build_method(manifest), build_method(manifest)
            state = NativeState(method, method_name=manifest["method"])
            reference_state = NativeState(reference, method_name=manifest["method"])
            require(state.initial == reference_state.initial, "Native gate initial state differs")
            require(list(state.written) == list(reference_state.written), "Native gate written parameter membership differs")
            comparisons = []
            for episode, start in enumerate((0, 2)):
                batch = shared.image_collate([data[i] for i in (start, start+1)])
                images, meta = batch["img_patches"].to(method.device), batch["meta"]
                rng = shared.rng_state(torch.device(method.device))
                # Alternate execution order; both calls use the native class,
                # with only read-only trajectory capture on the reference.
                results = {}
                order = ("native", "observed") if episode == 0 else ("observed", "native")
                for name in order:
                    shared.restore_rng(rng, torch.device(method.device))
                    target, saved = (reference, reference_state) if name == "native" else (method, state)
                    results[name] = run_episode(target, saved, images, meta, manifest, episode,
                        audited=name == "observed", capture=True)
                a, b = results["native"], results["observed"]
                deltas = dict(loss=maximum_delta(a[1], b[1]),
                    gradient=maximum_delta([r["gradients"] for r in a[3]], [r["gradients"] for r in b[3]]),
                    parameters=maximum_delta([r["parameters"] for r in a[3]], [r["parameters"] for r in b[3]]),
                    final_parameters=maximum_delta(a[4], b[4]))
                changed = sum(int(np.count_nonzero(x != y)) for x, y in zip(a[2], b[2], strict=True))
                if not (all(v <= 1e-5 for v in deltas.values()) and changed == 0):
                    fail("Native instrumentation parity gate failed", deltas=deltas,
                         changed_labels=changed, episode=episode, actual_cost=b[0],
                         native_cost=a[0], native_losses=a[1], observed_losses=b[1])
                per_step = [dict(step=i+1,
                    gradient_max_abs=maximum_delta(x["gradients"], y["gradients"]),
                    parameter_max_abs=maximum_delta(x["parameters"], y["parameters"]),
                    pre_parameter_max_abs=maximum_delta(x["pre_parameters"], y["pre_parameters"]),
                    gradient_sha256=shared.state_digest(y["gradients"]),
                    pre_parameter_sha256=shared.state_digest(y["pre_parameters"]),
                    post_parameter_sha256=shared.state_digest(y["parameters"]),
                    adam_state_sha256=y["optimizer_sha256"], adam_tensor_devices=y["adam_tensor_devices"],
                    all_gradients_parameters_Adam_finite=True)
                    for i, (x, y) in enumerate(zip(a[3], b[3], strict=True))]
                comparisons.append(dict(episode=episode, order=list(order), batch_images=2,
                    parity_parameter_names=list(state.written),
                    ordered_ids=manifest["panel"]["ordered_ids"][start:start+2], max_abs=deltas,
                    changed_labels=changed, actual_cost=b[0], reset_state=state.initial,
                    finite_updates_verified=True, losses=b[1], update_receipts=per_step))
            require(shared.deterministic_runtime_state() == runtime, "Native gate runtime drift")
            common.admission(args)
            verify_inputs(manifest)
            common.verify_prepared(Path(manifest["suite_root"]), manifest["suite_prepared_sha256"], frozen=True)
            report = dict(passed=True, **scope(manifest), manifest_sha256=args.manifest_sha256,
                manifest=manifest, settings_constructor=manifest["settings"]["constructor"],
                source_sha256=manifest["source_sha256"], parent_proof_sha256=manifest["parent_proof_sha256"],
                suite_prepared_sha256=manifest["suite_prepared_sha256"], baseline_source_sha256=manifest["baseline_source_sha256"],
                checkpoint=manifest["checkpoint"], settings=manifest["settings"], gt_contents_opened=False,
                images=4, configured_batch_size=2, batch_images=[2, 2], comparisons=comparisons,
                tolerance=1e-5, changed_labels=0, gpu_lock=lock, deterministic_runtime=runtime)
            write_json(output / "gate.json", report)
            write_json(output / "GATE_PASSED.json", dict(report, gate_sha256=sha256(output / "gate.json")))
    return report


def score(args):
    manifest, output = common.admission(args)
    require(manifest["stage"] == "full", "Only full baseline predictions may score")
    barrier = Path(args.barrier) if args.barrier is not None else Path(manifest["suite_root"]) / "ALL_FULL_PREDICTIONS_SEALED.json"
    require(not any((output / name).exists() for name in SCORE_FILES | {"SCORED_SEALED.json", "SCORING_FAILED.json"}),
            "No scoring retry/resume/overwrite")
    verify_inputs(manifest)
    verify_prediction_seal(output, manifest, args.manifest_sha256)
    barrier_sha = common.verify_barrier(barrier, manifest)
    prediction_sha = sha256(output / "PREDICTIONS_SEALED.json")
    binding = dict(manifest_sha256=args.manifest_sha256, prediction_seal_sha256=prediction_sha, barrier_sha256=barrier_sha)
    write_json(output / "SCORING_STARTED.json", dict(**binding, **scope(manifest)))
    try:
        totals = np.zeros((2, len(manifest["panel"]["classes"])), dtype=np.int64)
        hashes = {}
        with (output / "scores_by_image.jsonl").open("x") as stream:
            for i, row in enumerate(manifest["panel"]["inputs"]):
                confirmation.verify_label_metadata(row)
                # First and only phase permitted to open label contents. The
                # complete global barrier was verified before this loop.
                content = Path(row["label"]).read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                hashes[row["label"]] = digest
                with Image.open(io.BytesIO(content)) as source:
                    gt = np.asarray(source.resize((224, 224), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                require(gt.ndim == 2, "Invalid annotation dimensions")
                gt = gt.astype(np.int64, copy=True)
                if manifest["panel"]["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                require(((gt == 255) | ((gt >= 0) & (gt < totals.shape[1]))).all(), "Annotation outside frozen vocabulary")
                confirmation.verify_label_metadata(row)
                prediction_path = output / f"predictions/{i:06d}.npy"
                prediction = np.load(prediction_path, allow_pickle=False)
                validate_prediction(prediction, manifest)
                counts = np.asarray(shared.streaming_counts(prediction, gt, totals.shape[1]), dtype=np.int64)
                totals += counts
                stream.write(json.dumps(dict(index=i, official_id=row["official_id"], label_sha256=digest,
                    prediction_sha256=sha256(prediction_path), intersection=counts[0].tolist(), union=counts[1].tolist(),
                    pooled=pooled_result(counts)), allow_nan=False, sort_keys=True) + "\n")
        require(common.verify_barrier(barrier, manifest) == barrier_sha, "Barrier changed while scoring")
        common.admission(args)
        verify_inputs(manifest)
        verify_prediction_seal(output, manifest, args.manifest_sha256)
        summary = validate_cost_stream(output, manifest)
        pooled = pooled_result(totals)
        result = dict(status="completed", images=len(manifest["panel"]["inputs"]), **scope(manifest), **binding,
            config=manifest["config"], settings=manifest["settings"], pooled=pooled, **pooled, **summary,
            label_contents_first_opened_after_all18_full_predictions_sealed=True)
        write_json(output / "score_inputs.json", dict(**binding, label_sha256=hashes))
        write_json(output / "result.json", result)
        write_json(output / "SCORED_SEALED.json", dict(complete=True, **binding,
            artifact_sha256={name: sha256(output / name) for name in sorted(SCORE_FILES)}))
        verify_score_seal(output, manifest, args.manifest_sha256, barrier)
        return result
    except BaseException as error:
        if not (output / "SCORING_FAILED.json").exists():
            write_json(output / "SCORING_FAILED.json", dict(status="failed", **binding, retry_allowed=False,
                error=f"{type(error).__name__}: {error}"[:4000], context=finite_json(getattr(error, "worker_context", {}))))
        raise


def verify(args):
    manifest, output = common.admission(args)
    require(manifest["stage"] == "full", "Offline verification expects full prediction output")
    verify_inputs(manifest)
    if (output / "SCORED_SEALED.json").exists():
        return verify_score_seal(output, manifest, args.manifest_sha256, args.barrier)
    require(not any((output / name).exists() for name in SCORE_FILES | {"SCORING_FAILED.json"}), "Incomplete scoring attempt")
    return verify_prediction_seal(output, manifest, args.manifest_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gate", "predict", "score", "verify"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--barrier", type=Path)
    args = parser.parse_args(argv)
    return dict(gate=gate, predict=predict, score=score, verify=verify)[args.phase](args)


if __name__ == "__main__":
    main()
