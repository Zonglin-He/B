"""Synthetic CPU tests. No real model construction, module forwards or CUDA."""

import ast
from collections import OrderedDict
import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_local_baselines import common, worker


@pytest.fixture(autouse=True)
def no_cuda_or_model(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    real_build = worker.build_method
    with mock.patch.object(worker, "build_method", side_effect=AssertionError("CPU tests cannot build native models")), \
            mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU tests cannot initialize CUDA")), \
            mock.patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("CPU tests cannot forward models")), \
            mock.patch.object(worker.confirmation.SegmentationPatchDataset, "__getitem__",
                              side_effect=AssertionError("Original GT-returning loader is forbidden")):
        yield real_build


def synthetic_manifest(root, method="tent", n=3, stage="full", reduce_zero=False):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        image, label = root / f"rgb-{i}.png", root / f"label-{i}.png"
        Image.fromarray(np.full((2, 2, 3), i, np.uint8)).save(image)
        values = [[1, 2], [2, 255]] if reduce_zero else [[0, 1], [1, 255]]
        Image.fromarray(np.asarray(values, np.uint8)).save(label)
        s = label.stat()
        rows.append(dict(official_id=f"official-{i}", image=str(image), label=str(label),
            image_sha256=worker.sha256(image), label_metadata=[s.st_size, s.st_mtime_ns]))
    checkpoint = root / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic checkpoint; must never be loaded")
    baseline = root / "baseline_source"
    baseline.mkdir(exist_ok=True)
    (baseline / "prompts.yaml").write_text("synthetic prompt file\n")
    cfg = dict(dataset="PascalVOCDataset", data_dir=str(root), resize=[224, 224],
               patch_size=[224, 224], patch_stride=112, batch_size=2, lr=.001,
               steps=10, class_extensions=False)
    panel = dict(inputs=rows, ordered_ids=[r["official_id"] for r in rows],
        full_split=stage == "full", classes=["a", "b"], queries=["a", "b"],
        query_to_class=None, reduce_zero_label=reduce_zero,
        vocabulary_sha256=worker.canonical_hash([["a", "b"], None]))
    # Every alias check must include eight panels, even this tiny fixture.
    for dataset in common.DATASETS:
        path = root / "parents" / dataset / "panel.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        worker.write_json(path, panel, replace=path.exists())
    identity = dict(path=str(checkpoint), sha256=worker.sha256(checkpoint))
    return dict(schema_version=1, algorithm=common.ALGORITHM, dataset="coco_cd", method=method,
        stage=stage, seed=0, suite_root=str(root), runtime_root=str(root / "runtime"),
        suite_prepared_sha256="a" * 64, source_sha256={"worker.py": "b" * 64},
        baseline_source_sha256={"prompts.yaml": worker.sha256(baseline / "prompts.yaml")},
        baseline_root=str(baseline), config=cfg, panel=panel, checkpoint=identity,
        settings=common.settings(root, method, identity),
        output=str(root / ("gates" if stage == "gate" else "full") / f"coco_cd.{method}"),
        parent_proof_sha256="c" * 64)


def cost_row(manifest, episode):
    images = min(2, len(manifest["panel"]["inputs"]) - 2 * episode)
    settings = manifest["settings"]
    return dict(**worker.scope(manifest), episode=episode, images=images, patches=images,
        warmup=episode == 0, seconds=.125, optimizer_steps=settings["expected_update_steps"],
        loss_entries=settings["expected_loss_entries"], visual_calls=settings["expected_visual_calls"],
        visual_inputs=settings["expected_visual_calls"] * images,
        native_reset_verified=True, gt_contents_opened=False)


def sealed_output(root, monkeypatch, method="tent", n=3, reduce_zero=False):
    manifest = synthetic_manifest(root, method, n, reduce_zero=reduce_zero)
    path = common.manifest_path(root, "coco_cd", method, "full")
    path.parent.mkdir(exist_ok=True)
    worker.write_json(path, manifest)
    identity = worker.sha256(path)
    output = Path(manifest["output"])
    worker.started(output, identity, "predict")
    gates = {f"{d}.{m}": "d" * 64 for d, m in common.JOBS}
    monkeypatch.setattr(common, "verify_gates", lambda m: gates)
    monkeypatch.setattr(common, "verify_prepared", lambda *args, **kwargs: None)
    worker.write_json(output / "prediction_protocol.json", dict(manifest=manifest,
        manifest_path=str(path), manifest_sha256=identity, output=str(output),
        gt_contents_opened=False, gates=gates, timing_scope=worker.TIMING_SCOPE))
    native_state = dict(snapshot_sha256="e" * 64, model_sha256="f" * 64,
        trainable_sha256="0" * 64, optimized_parameter_names=["weight"], optimizer_sha256="1" * 64,
        ctx_sha256=None)
    worker.write_json(output / "initial_state.json", dict(native_state=native_state,
        checkpoint=manifest["checkpoint"], baseline_source_sha256=manifest["baseline_source_sha256"],
        settings=manifest["settings"]))
    (output / "predictions").mkdir()
    (output / "episodes").mkdir()
    for i in range(n):
        with (output / f"predictions/{i:06d}.npy").open("xb") as stream:
            np.save(stream, np.zeros((224, 224), np.uint16), allow_pickle=False)
    rows = []
    with (output / "cost.jsonl").open("x") as stream:
        for episode in range((n + 1) // 2):
            row = cost_row(manifest, episode)
            rows.append(row)
            worker.write_json(output / f"episodes/{episode:06d}.json", dict(row,
                losses=[.5] * row["loss_entries"], reset_state=native_state,
                ordered_ids=manifest["panel"]["ordered_ids"][2*episode:2*episode+2]))
            stream.write(json.dumps(row) + "\n")
    worker.write_json(output / "prediction_result.json", dict(status="predicted", images=n,
        **worker.scope(manifest), config=manifest["config"], settings=manifest["settings"], **worker.summarize(rows)))
    worker.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
        output=str(output), manifest_sha256=identity, images=n, **worker.scope(manifest),
        suite_prepared_sha256=manifest["suite_prepared_sha256"],
        artifact_sha256={name: worker.sha256(output / name) for name in worker.artifact_names(manifest)}))
    args = SimpleNamespace(manifest=path, manifest_sha256=identity, output=output,
        barrier=root / "ALL_FULL_PREDICTIONS_SEALED.json")
    return manifest, output, args


def reseal_prediction(output):
    seal = worker.read_json(output / "PREDICTIONS_SEALED.json")
    seal["artifact_sha256"] = {name: worker.sha256(output / name) for name in seal["artifact_sha256"]}
    worker.write_json(output / "PREDICTIONS_SEALED.json", seal, replace=True)


def scoring_dependencies(monkeypatch, manifest, output, barrier_sha="2" * 64):
    monkeypatch.setattr(common, "admission", lambda args: (manifest, output))
    monkeypatch.setattr(worker, "verify_inputs", lambda m: None)
    monkeypatch.setattr(common, "verify_barrier", lambda path, m: barrier_sha)


@pytest.mark.parametrize("method,updates,losses,visual", [
    ("noadapt", 0, 0, 1), ("tent", 10, 10, 11), ("tpt", 10, 10, 11),
    ("mlmp", 10, 10, 12), ("watt", 70, 5, 141), ("clipartt", 10, 10, 21)])
def test_constructor_and_sealed_actual_cost(tmp_path, monkeypatch, method, updates, losses, visual):
    manifest, output, args = sealed_output(tmp_path, monkeypatch, method)
    kwargs = worker.constructor_kwargs(manifest)
    assert kwargs["ovss_backbone"] == manifest["checkpoint"]["path"]
    assert kwargs["ovss_type"] == "naclip" and kwargs["lr"] == .001
    assert kwargs["classes"] == ["a", "b"]
    if method == "mlmp":
        assert kwargs["vision_outputs"] == tuple(range(-1, -19, -1))
        assert kwargs["alpha_cls"] == 1. and kwargs["prompt_integration"] == "loss"
    if method == "tpt":
        assert kwargs["n_ctx"] == 4
    summary = worker.validate_cost_stream(output, manifest)
    assert summary["costs"] == dict(optimizer_steps=2*updates, loss_entries=2*losses,
        visual_calls=2*visual, visual_inputs=3*visual)
    assert summary["timings"]["measured_images"] == 1
    assert worker.verify_prediction_seal(output, manifest, args.manifest_sha256)["complete"] is True


def test_constructor_rejects_wrong_checkpoint_settings_and_pair(tmp_path):
    m = synthetic_manifest(tmp_path)
    m["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Checkpoint content"):
        worker.constructor_kwargs(m)
    m["checkpoint"]["sha256"] = worker.sha256(m["checkpoint"]["path"])
    m["settings"]["constructor"]["lr"] = .01
    with pytest.raises(RuntimeError, match="constructor primitives"):
        worker.constructor_kwargs(m)
    m["dataset"] = "voc20"
    with pytest.raises(RuntimeError, match="Unauthorized baseline pair"):
        worker.constructor_kwargs(m)


def test_archived_namespace_import_allows_repeated_build_admission(no_cuda_or_model):
    root = Path("/home/wwww/hzl/results/reviewer_response_20260929/baseline_source")
    tree = ast.parse((root / "adapt/tent.py").read_text())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "utils.misc"]
    assert len(imports) == 1
    native_import = compile(ast.Module(body=imports, type_ignores=[]), str(root / "adapt/tent.py"), "exec")
    with mock.patch.dict(sys.modules), mock.patch.object(sys, "path", [str(root), *sys.path]):
        for name in tuple(sys.modules):
            if name.split(".")[0] in ("adapt", "ovss", "utils", "mmengine"):
                del sys.modules[name]
        exec(native_import, {})
        namespace, misc = sys.modules["utils"], sys.modules["utils.misc"]
        assert namespace.__file__ is None
        assert list(namespace.__path__) and all(Path(p).is_relative_to(root) for p in namespace.__path__)
        assert Path(misc.__file__).resolve() == root / "utils/misc.py"
        worker.verify_native_imports(root)
        manifest = dict(baseline_root=str(root), method="tent",
            baseline_source_sha256={"adapt/tent.py": worker.sha256(root / "adapt/tent.py")})
        with mock.patch.object(worker, "constructor_kwargs", return_value={}), \
                mock.patch.object(worker.importlib, "import_module", side_effect=RuntimeError("CPU stop before native model import")) as stop, \
                mock.patch.object(worker.importlib, "reload", side_effect=AssertionError("Never reload native packages")):
            for _ in range(2):
                with pytest.raises(RuntimeError, match="CPU stop before native model import"):
                    no_cuda_or_model(manifest)
            assert stop.call_count == 2
        assert sys.modules["utils"] is namespace and sys.modules["utils.misc"] is misc


def test_build_method_completes_repeated_cpu_constructor_boundary(tmp_path, no_cuda_or_model):
    manifest = synthetic_manifest(tmp_path)
    fake = SimpleNamespace(prompt_templates=manifest["settings"]["prompt_templates"],
        model=SimpleNamespace(state_dict=lambda: {"finite": torch.ones(1)}))
    with mock.patch.object(worker, "load_native_class", return_value=lambda **kwargs: fake), \
            mock.patch.object(worker, "verify_native_imports") as verify:
        for _ in range(2):
            assert no_cuda_or_model(manifest) is fake
        assert verify.call_args_list == [mock.call(Path(manifest["baseline_root"]).resolve())] * 2


@pytest.mark.parametrize("damage", ["external", "mixed", "empty", "mixed_spec", "ordinary_external", "ordinary_mixed", "fileless"])
def test_native_import_guard_rejects_contamination_without_reload(tmp_path, damage):
    root, external = tmp_path / "frozen", tmp_path / "external"
    module = ModuleType("utils")
    module.__file__ = None
    module.__spec__ = importlib.machinery.ModuleSpec("utils", loader=None, is_package=True)
    locations = [str(root / "utils")]
    if damage == "external":
        locations = [str(external / "utils")]
    elif damage == "mixed":
        locations.append(str(external / "utils"))
    elif damage == "empty":
        locations = []
    elif damage == "ordinary_external":
        module.__file__ = str(external / "utils/__init__.py")
    elif damage == "ordinary_mixed":
        module.__file__ = str(root / "utils/__init__.py")
        locations.append(str(external / "utils"))
    module.__path__ = locations
    module.__spec__.submodule_search_locations = list(locations)
    if damage == "mixed_spec":
        module.__spec__.submodule_search_locations.append(str(external / "utils"))
    elif damage == "fileless":
        del module.__path__
        module.__spec__ = None
    before = dict(module.__dict__)
    with mock.patch.dict(sys.modules, {"utils": module}), \
            mock.patch.object(worker.importlib, "reload", side_effect=AssertionError("Never reload native packages")):
        with pytest.raises(RuntimeError, match="Contaminated native import: utils"):
            worker.verify_native_imports(root)
        assert sys.modules["utils"] is module and module.__dict__ == before


def test_watt_cannot_claim_ten_updates(tmp_path):
    m = synthetic_manifest(tmp_path, "watt")
    row = cost_row(m, 0)
    row["optimizer_steps"] = 10
    with pytest.raises(RuntimeError, match="budget differs"):
        worker.validate_cost(row, m, 0)


@pytest.mark.parametrize("calls", [11, 13])
def test_mlmp_rejects_visual_budgets_other_than_twelve(tmp_path, calls):
    m = synthetic_manifest(tmp_path, "mlmp")
    assert m["settings"]["expected_visual_calls"] == 12
    row = cost_row(m, 0)
    row.update(visual_calls=calls, visual_inputs=2*calls)
    with pytest.raises(RuntimeError, match="budget differs"):
        worker.validate_cost(row, m, 0)


@pytest.mark.parametrize("key,value", [("images", 1), ("patches", 3), ("optimizer_steps", True),
    ("visual_inputs", 999), ("seconds", float("nan")), ("warmup", 1),
    ("native_reset_verified", False), ("gt_contents_opened", True)])
def test_cost_rejects_malformed_observations(tmp_path, key, value):
    m = synthetic_manifest(tmp_path)
    row = cost_row(m, 0)
    row[key] = value
    with pytest.raises(RuntimeError):
        worker.validate_cost(row, m, 0)


@pytest.mark.parametrize("damage", ["missing", "extra", "order", "loss", "reset", "aggregate", "protocol", "initial", "prediction", "inventory", "gate"])
def test_seal_and_semantic_tampering(tmp_path, monkeypatch, damage):
    m, output, args = sealed_output(tmp_path, monkeypatch)
    if damage in ("missing", "extra", "order"):
        rows = list(worker.bounded_lines(output / "cost.jsonl", 2))
        rows = rows[:1] if damage == "missing" else rows + rows[:1] if damage == "extra" else rows[::-1]
        (output / "cost.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif damage in ("loss", "reset"):
        path = output / "episodes/000000.json"
        receipt = worker.read_json(path)
        receipt["losses"] = [] if damage == "loss" else receipt["losses"]
        if damage == "reset":
            receipt["reset_state"]["ctx_sha256"] = "x" * 64
        worker.write_json(path, receipt, replace=True)
    elif damage == "aggregate":
        path = output / "prediction_result.json"
        result = worker.read_json(path)
        result["costs"]["optimizer_steps"] = 1
        worker.write_json(path, result, replace=True)
    elif damage == "protocol":
        path = output / "prediction_protocol.json"
        result = worker.read_json(path)
        result["manifest"]["source_sha256"] = {}
        worker.write_json(path, result, replace=True)
    elif damage == "initial":
        path = output / "initial_state.json"
        value = worker.read_json(path)
        value["checkpoint"]["sha256"] = "a" * 64
        worker.write_json(path, value, replace=True)
    elif damage == "prediction":
        with (output / "predictions/000000.npy").open("wb") as stream:
            np.save(stream, np.full((224, 224), 2, np.uint16), allow_pickle=False)
    elif damage == "inventory":
        (output / "extra.json").write_text("{}")
    elif damage == "gate":
        monkeypatch.setattr(common, "verify_gates", lambda m: {})
    reseal_prediction(output)
    with pytest.raises(RuntimeError):
        worker.verify_prediction_seal(output, m, args.manifest_sha256)


def test_raw_payload_hash_drift(tmp_path, monkeypatch):
    m, output, args = sealed_output(tmp_path, monkeypatch)
    with (output / "cost.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="artifact drift"):
        worker.verify_prediction_seal(output, m, args.manifest_sha256)


def test_persistent_attempt_survives_removed_output(tmp_path):
    output = tmp_path / "job"
    worker.started(output, "a" * 64, "gate")
    (output / "STARTED.json").unlink()
    output.rmdir()
    with pytest.raises(RuntimeError, match="No retry"):
        worker.started(output, "a" * 64, "gate")


@pytest.mark.parametrize("phase", ["gate", "predict"])
def test_gpu_envelope_precedes_attempt_and_model(tmp_path, monkeypatch, phase):
    m = synthetic_manifest(tmp_path, stage="gate" if phase == "gate" else "full", n=4)
    output = Path(m["output"])
    monkeypatch.setattr(common, "admission", lambda args: (m, output))
    def reject(manifest):
        raise RuntimeError("wrong GPU envelope")
    monkeypatch.setattr(common, "require_gpu_envelope", reject)
    args = SimpleNamespace(manifest_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="wrong GPU envelope"):
        getattr(worker, phase)(args)
    assert not output.exists()
    assert not (output.parent / f"{output.name}.STARTED.json").exists()


def test_failure_context_is_finite_json_and_not_retried(tmp_path):
    output = tmp_path / "job"
    with pytest.raises(RuntimeError, match="numeric failure"):
        with worker.attempted(output, "a" * 64, "predict"):
            worker.fail("numeric failure", loss=float("nan"), values=[float("inf"), np.float32(-np.inf)])
    failure = worker.read_json(output / "PREDICTION_FAILED.json")
    assert failure["context"] == dict(loss="nan", values=["inf", "-inf"])
    assert failure["retry_allowed"] is False
    json.dumps(failure, allow_nan=False)
    with pytest.raises(RuntimeError, match="No retry"):
        worker.started(output, "a" * 64, "predict")


def test_raw_finite_rejects_before_nan_to_num():
    value = torch.tensor([float("nan"), 1.])
    with pytest.raises(RuntimeError, match="Nonfinite native observation") as error:
        with worker.RawFiniteMode():
            torch.nan_to_num(value)
    assert error.value.worker_context["phase"] == "before-native-nan-to-num"
    assert torch.isnan(value[0])


def test_checked_native_logits_precedes_readout():
    original = lambda images: torch.tensor([float("inf")])
    method = SimpleNamespace(evaluate=original)
    with pytest.raises(RuntimeError, match="Nonfinite native observation"):
        with worker.checked_evaluate(method):
            method.evaluate(None)
    assert method.evaluate is original


class Handle:
    def __init__(self, hooks, callback):
        self.hooks, self.callback = hooks, callback
        hooks.append(callback)

    def remove(self):
        self.hooks.remove(self.callback)


class TensorBackbone:
    """Plain Python tensor state holder, not a torch model or forward stub."""

    def __init__(self, external_ctx=False):
        self.weight = torch.tensor([1., 2.], requires_grad=not external_ctx)
        self.hooks = []
        self.visual = SimpleNamespace(conv1=SimpleNamespace(register_forward_pre_hook=
            lambda callback: Handle(self.hooks, callback)))

    def named_parameters(self):
        return iter([("weight", self.weight)])

    def state_dict(self):
        return {"weight": self.weight.detach()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.weight.copy_(state["weight"])

    def named_modules(self):
        return []

    def observe_visual(self, images):
        for callback in tuple(self.hooks):
            callback(self.visual.conv1, (images,))


class TensorNative:
    def __init__(self, tpt=False, method="tent"):
        self.model = TensorBackbone(tpt)
        self.method_name = method
        if tpt:
            self.ctx = torch.tensor([.1, .2], requires_grad=True)
        param = self.ctx if tpt else self.model.weight
        self.optimizer = torch.optim.Adam([param], lr=.001, betas=(.9, .999), weight_decay=0.)
        if tpt:
            self._clean_state = dict(model=copy.deepcopy(self.model.state_dict()),
                ctx=self.ctx.detach().clone(), optim=copy.deepcopy(self.optimizer.state_dict()))
        else:
            self.model_state = copy.deepcopy(self.model.state_dict())
            self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def reset(self):
        state = self._clean_state if hasattr(self, "ctx") else dict(model=self.model_state, optim=self.optimizer_state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optim"])
        if hasattr(self, "ctx"):
            with torch.no_grad():
                self.ctx.copy_(state["ctx"])

    def collect_ln_params(self, visual):
        return [self.model.weight], ["weight"]

    def evaluate(self, images):
        for _ in range(2 if self.method_name == "mlmp" else 1):
            self.model.observe_visual(images)
        return torch.ones((len(images), 2, 2, 2))


@pytest.mark.parametrize("name,eval_calls,total_calls", [("mlmp", 2, 12), ("tent", 1, 11)])
def test_synthetic_native_evaluate_observes_actual_visual_budget(tmp_path, name, eval_calls, total_calls):
    manifest = synthetic_manifest(tmp_path, name)
    method = TensorNative(method=name)
    state = worker.NativeState(method)
    images = torch.zeros((2, 3, 2, 2))
    losses = []
    with worker.NativeObservation(method, state) as observed:
        method.reset()
        for _ in range(10):
            method.model.observe_visual(images)
            method.model.weight.grad = torch.ones_like(method.model.weight)
            method.optimizer.step()
            method.optimizer.zero_grad()
            losses.append(.5)
        before = observed.counts["visual_calls"]
        logits = method.evaluate(images)
        observed.counts["loss_entries"] = len(losses)
        assert observed.counts["visual_calls"] - before == eval_calls
    assert logits.shape == (2, 2, 2, 2)
    assert observed.counts == dict(optimizer_steps=10, loss_entries=10,
        visual_calls=total_calls, visual_inputs=2*total_calls)
    worker.validate_cost(dict(cost_row(manifest, 0), **observed.counts), manifest, 0)
    method.reset()
    assert state.check() == state.initial


@pytest.mark.parametrize("tpt", [False, True])
def test_native_full_snapshot_optimizer_and_ctx_reset(tpt):
    method = TensorNative(tpt)
    state = worker.NativeState(method)
    param = method.ctx if tpt else method.model.weight
    param.grad = torch.ones_like(param)
    method.optimizer.step()
    method.optimizer.zero_grad()
    with pytest.raises(RuntimeError, match="Native reset"):
        state.check()
    method.reset()
    assert state.check() == state.initial
    assert state.initial["optimized_parameter_names"] == (["ctx"] if tpt else ["weight"])
    assert (state.initial["ctx_sha256"] is not None) == tpt


def test_native_snapshot_and_membership_drift():
    method = TensorNative()
    state = worker.NativeState(method)
    method.model_state["weight"][0] = 3.
    with pytest.raises(RuntimeError, match="saved snapshot changed"):
        state.check()
    method = TensorNative()
    method.optimizer.add_param_group(dict(params=[torch.tensor([1.], requires_grad=True)]))
    with pytest.raises(RuntimeError, match="Unknown/frozen"):
        worker.NativeState(method)


@pytest.fixture
def archived_tent_with_cpu_backend():
    # Execute the actual archived constructor, never the OVSS/CLIP loader.
    # These are tiny synthetic parameter containers; no forward is permitted.
    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
    model.ln_final = torch.nn.LayerNorm(2)
    model.token_embedding = torch.nn.Embedding(2, 2)
    model.visual = torch.nn.Module()
    model.visual.ln_pre = torch.nn.LayerNorm(2)
    model.visual.ln_post = torch.nn.LayerNorm(2)
    model.positional_embedding = torch.nn.Parameter(torch.ones(2))
    model.text_projection = torch.nn.Parameter(torch.eye(2))
    model.logit_scale = torch.nn.Parameter(torch.ones(()))
    model.encode_text = lambda tokens: torch.ones((tokens.shape[0], 2))
    ovss = ModuleType("ovss")
    ovss.load_ovss = lambda ovss_type, ovss_backbone, device: (model, lambda texts: torch.zeros((len(texts), 2), dtype=torch.long))
    misc = ModuleType("utils.misc")
    misc.load_prompts_from_yaml = lambda path: (_ for _ in ()).throw(AssertionError("No prompt file read in this fixture"))
    misc.print_clip_parameters = lambda m: None
    misc.print_optimizer_parameters = lambda optimizer, m: None
    path = Path("/home/wwww/hzl/results/reviewer_response_20260929/baseline_source/adapt/tent.py")
    spec = importlib.util.spec_from_file_location("archived_tent_cpu_regression", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"ovss": ovss, "utils.misc": misc}):
        spec.loader.exec_module(module)
        native = module.TENT(ovss_type="naclip", ovss_backbone="synthetic-pinned-path", lr=.001,
                             classes=["a", "b"], steps=10, device="cpu")
        yield native


def test_actual_archived_tent_constructor_preserves_unused_trainables(archived_tent_with_cpu_backend):
    native = archived_tent_with_cpu_backend
    state = worker.NativeState(native)
    unused = ("positional_embedding", "text_projection", "logit_scale")
    params = dict(native.model.named_parameters())
    assert all(params[name].requires_grad for name in unused)
    selected, names = native.collect_ln_params(native.model.visual)
    assert state.initial["optimized_parameter_names"] == names
    assert [id(p) for g in native.optimizer.param_groups for p in g["params"]] == [id(p) for p in selected]
    assert all(name in state.initial["nonoptimized_parameter_names"] for name in unused)
    before = {name: params[name].detach().clone() for name in unused}
    for parameter in selected:
        parameter.grad = torch.ones_like(parameter)
    native.optimizer.step()
    native.optimizer.zero_grad()
    state.check_nonoptimized()
    assert all(torch.equal(params[name], before[name]) and params[name].requires_grad for name in unused)
    native.reset()
    assert state.check() == state.initial
    with torch.no_grad():
        native.model.logit_scale.add_(1)
    with pytest.raises(RuntimeError, match="nonoptimized.logit_scale"):
        state.check_nonoptimized()


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "reordered"])
def test_archived_tent_optimizer_ordered_membership_is_exact(archived_tent_with_cpu_backend, damage):
    native = archived_tent_with_cpu_backend
    parameters = native.optimizer.param_groups[0]["params"]
    if damage == "missing":
        parameters.pop()
    elif damage == "extra":
        parameters.append(native.model.logit_scale)
    elif damage == "duplicate":
        parameters.append(parameters[0])
    else:
        parameters.reverse()
    with pytest.raises(RuntimeError, match="membership|duplicate"):
        worker.NativeState(native)


def test_actual_archived_mlmp_evaluate_encodes_images_twice_on_cpu():
    root = Path("/home/wwww/hzl/results/reviewer_response_20260929/baseline_source")
    model_spec = importlib.util.spec_from_file_location("archived_clip_cpu_readout", root / "ovss/clip/model.py")
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)
    ovss, misc = ModuleType("ovss"), ModuleType("utils.misc")
    def forbidden(*args, **kwargs):
        raise AssertionError("Archived MLMP readout test cannot load models or assets")
    ovss.load_ovss = forbidden
    misc.load_prompts_from_yaml = misc.print_clip_parameters = misc.print_optimizer_parameters = forbidden
    spec = importlib.util.spec_from_file_location("archived_mlmp_cpu_readout", root / "adapt/mlmp.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"ovss": ovss, "utils.misc": misc}):
        spec.loader.exec_module(module)

    class SyntheticBackend:
        def __init__(self):
            self.logit_scale = torch.tensor(0.)
            self.visual = SimpleNamespace(patch_size=1)
            self.weights_track, self.calls = [], []
            self.forward = MethodType(model_module.CLIP.forward, self)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def encode_image(self, images, output_layers, return_vanilla_cls, *, out_type, weights=None):
            assert images.device.type == "cpu" and return_vanilla_cls is False
            self.calls.append(dict(images=images, layers=output_layers, kind=out_type, weights=weights))
            if out_type == "all":
                self.features = torch.arange(1, 18*2*5*2+1, dtype=torch.float32).reshape(18, 2, 5, 2)
                return self.features
            assert out_type == "weighted_mean" and weights.shape == (18, 2)
            return (self.features * weights[:, :, None, None]).sum(0)

    # Bind the real archived readout without running any model constructor.
    native = object.__new__(module.MLMP)
    native.model, native.runtime = SyntheticBackend(), False
    native.vision_outputs = tuple(range(-1, -19, -1))
    native.text_x = torch.arange(1, 8*3*2+1, dtype=torch.float32).reshape(8, 3, 2)
    images = torch.zeros((2, 3, 2, 2))
    logits = native.evaluate(images)
    calls = native.model.calls
    assert [row["kind"] for row in calls] == ["all", "weighted_mean"]
    assert all(row["images"] is images and row["layers"] == native.vision_outputs for row in calls)
    assert sum(len(row["images"]) for row in calls) == 4
    assert calls[0]["weights"] is None and torch.isfinite(calls[1]["weights"]).all()
    assert len(native.model.weights_track) == 1
    assert logits.shape == (2, 3, 2, 2) and torch.isfinite(logits).all()


@pytest.fixture
def archived_watt_cpu_factory():
    path = Path("/home/wwww/hzl/results/reviewer_response_20260929/baseline_source/adapt/watt.py")
    tree = ast.parse(path.read_text())
    source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WATT")
    names = {"collect_ln_params", "weight_average", "copy_model_and_optimizer", "load_model_and_optimizer", "reset"}
    body = [copy.deepcopy(n) for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names)
    extracted = ast.ClassDef(name="WATT", bases=[], keywords=[], body=body, decorator_list=[])
    env = dict(torch=torch, nn=torch.nn, copy=copy, OrderedDict=OrderedDict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])), str(path), "exec"), env)
    adaptation = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "perform_adaptation")
    collections = [n for n in ast.walk(adaptation) if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Call) and ast.unparse(n.iter) == "self.model.named_modules()"]
    writes = [n for n in adaptation.body if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "self.model.load_state_dict"]
    assert len(collections) == len(writes) == 1
    collect = compile(ast.Module(body=collections, type_ignores=[]), str(path), "exec")
    write = compile(ast.Module(body=writes, type_ignores=[]), str(path), "exec")

    def factory():
        native = object.__new__(env["WATT"])
        native.model = torch.nn.Module()
        native.model.visual = torch.nn.Module()
        native.model.visual.ln_pre = torch.nn.LayerNorm(3)
        native.model.ln_final = torch.nn.LayerNorm(3)
        native.model.transformer = torch.nn.Module()
        native.model.transformer.ln = torch.nn.LayerNorm(3)
        native.model.logit_scale = torch.nn.Parameter(torch.tensor(.1))
        native.model.requires_grad_(False)
        native.model.visual.requires_grad_(True)
        with torch.no_grad():
            for _, module in native.model.named_modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.weight.copy_(torch.tensor([.1, .2, .3]))
                    module.bias.copy_(torch.tensor([.1, .2, .3]))
        hooks = []
        native.model.visual.conv1 = SimpleNamespace(register_forward_pre_hook=lambda fn: Handle(hooks, fn))
        selected, _ = native.collect_ln_params(native.model.visual)
        native.optimizer = torch.optim.Adam(selected, lr=.001, betas=(.9, .999), weight_decay=0.)
        native.model_state, native.optimizer_state = native.copy_model_and_optimizer(native.model, native.optimizer)

        def average():
            context = dict(env, self=native, weights={})
            exec(collect, context)
            snapshots = [copy.deepcopy(context["weights"]) for _ in range(7)]
            context["avg_state_dict"] = native.weight_average(snapshots)
            exec(write, context)
            return list(context["weights"])

        return native, average
    return factory


def test_archived_watt_averaging_roundoff_allowed_only_for_native_written_lns(archived_watt_cpu_factory):
    native, average = archived_watt_cpu_factory()
    state = worker.NativeState(native)
    before = native.model.ln_final.weight.detach().clone()
    selected = average()
    delta = worker.maximum_delta(before, native.model.ln_final.weight)
    assert 0 < delta < 1e-5
    assert set(state.written) == set(selected)
    assert "ln_final.weight" in state.written and "ln_final.weight" not in state.optimized
    assert "transformer.ln.weight" in state.written and "logit_scale" not in state.written
    state.check_nonoptimized()
    with pytest.raises(RuntimeError, match="Native reset tensor differs"):
        state.check()
    native.reset()
    assert state.check() == state.initial
    assert len(state.initial) == 8
    average()
    with torch.no_grad():
        native.model.logit_scale.add_(1e-4)
    with pytest.raises(RuntimeError, match="nonoptimized.logit_scale"):
        state.check_nonoptimized()
    native.reset()
    strict = worker.NativeState(native, method_name="tent")
    average()
    with pytest.raises(RuntimeError, match="nonoptimized.*weight"):
        strict.check_nonoptimized()


def test_watt_gate_parameter_capture_includes_non_adam_native_writes(archived_watt_cpu_factory):
    native, average = archived_watt_cpu_factory()
    state = worker.NativeState(native)
    with worker.NativeObservation(native, state, capture=True) as observed:
        native.reset()
        for _ in range(2):
            for parameter in state.optimized.values():
                parameter.grad = torch.ones_like(parameter)
            native.optimizer.step()
            native.optimizer.zero_grad()
            average()
    for step in observed.steps:
        assert list(step["gradients"]) == list(state.optimized)
        assert list(step["pre_parameters"]) == list(step["parameters"]) == list(state.written)
        assert "ln_final.weight" in step["parameters"]
    first, second = observed.steps
    assert worker.maximum_delta(first["parameters"]["ln_final.weight"], second["pre_parameters"]["ln_final.weight"]) > 0
    final = worker.capture_parameters(native, state.written)
    assert "ln_final.weight" in final and "transformer.ln.bias" in final
    reference = {name: value.clone() for name, value in final.items()}
    final["ln_final.weight"].add_(1e-3)
    assert worker.maximum_delta(final, reference) > 1e-5
    assert worker.shared.state_digest(final) != worker.shared.state_digest(reference)
    native.reset()
    assert state.check() == state.initial


def test_watt_native_written_lns_must_still_be_finite(archived_watt_cpu_factory):
    native, _ = archived_watt_cpu_factory()
    state = worker.NativeState(native)
    with torch.no_grad():
        native.model.ln_final.weight[0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite native observation"):
        state.check_nonoptimized()


def test_readonly_native_hooks_count_actual_adam_steps_and_cleanup():
    method = TensorNative(tpt=True)
    state = worker.NativeState(method)
    before = method.ctx.detach().clone()
    with worker.NativeObservation(method, state, capture=True) as observed:
        method.reset()
        method.ctx.grad = torch.tensor([1., 2.])
        method.optimizer.step()
        method.optimizer.zero_grad()
        assert observed.counts["optimizer_steps"] == 1
        assert observed.counts["visual_calls"] == 0
        assert observed.steps[0]["gradients"]["ctx"].tolist() == [1., 2.]
        assert observed.resets == [state.initial]
    assert not torch.equal(before, method.ctx)
    assert method.model.hooks == [] and "reset" not in method.__dict__
    assert len(method.optimizer._optimizer_step_pre_hooks) == len(method.optimizer._optimizer_step_post_hooks) == 0
    method.reset()
    assert state.check() == state.initial


@pytest.mark.parametrize("failure", ["gradient", "post-Adam"])
def test_actual_finite_step_hooks_fail_without_numeric_repairs(failure):
    method = TensorNative()
    state = worker.NativeState(method)
    method.model.weight.grad = torch.tensor([float("nan"), 1.]) if failure == "gradient" else torch.tensor([1e30, 1e30])
    with pytest.raises(RuntimeError, match="Nonfinite native observation") as error:
        with worker.NativeObservation(method, state):
            method.optimizer.step()
    assert error.value.worker_context["phase"] == ("pre-step-gradients" if failure == "gradient" else "post-step-all-Adam-tensors")
    if failure == "gradient":
        assert torch.isnan(method.model.weight.grad[0])
    else:
        assert not torch.isfinite(method.optimizer.state[method.model.weight]["exp_avg_sq"]).all()


def test_verify_inputs_rgb_order_vocab_and_all_eight_aliases(tmp_path, monkeypatch):
    m = synthetic_manifest(tmp_path)
    data = SimpleNamespace(vocabulary=SimpleNamespace(classes=["a", "b"]),
        spec=SimpleNamespace(reduce_zero_label=False), samples=[(Path(r["image"]), Path(r["label"])) for r in m["panel"]["inputs"]])
    class RGBData:
        vocabulary, spec, samples = data.vocabulary, data.spec, data.samples
        def __len__(self):
            return len(self.samples)
    monkeypatch.setattr(worker.confirmation, "make_data", lambda cfg, rows: RGBData())
    labels = {Path(r["label"]) for r in m["panel"]["inputs"]}
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path not in labels, "Annotation bytes opened during RGB verification"
        return original_open(path, *args, **kwargs)
    with mock.patch.object(Path, "open", guarded_open):
        assert len(worker.verify_inputs(m)) == 3
    RGBData.samples = RGBData.samples[::-1]
    with pytest.raises(RuntimeError, match="discovery order"):
        worker.verify_inputs(m)
    RGBData.samples = data.samples
    RGBData.vocabulary = SimpleNamespace(classes=["b", "a"])
    with pytest.raises(RuntimeError, match="vocabulary/config"):
        worker.verify_inputs(m)
    # An alias to a label present only in another full panel must fail before hashing.
    peer_label = tmp_path / "peer-label.png"
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(peer_label)
    panel_path = tmp_path / "parents/voc20/panel.json"
    peer = worker.read_json(panel_path)
    peer["inputs"][0]["label"] = str(peer_label)
    worker.write_json(panel_path, peer, replace=True)
    image = Path(m["panel"]["inputs"][0]["image"])
    image.unlink()
    image.hardlink_to(peer_label)
    with pytest.raises(RuntimeError, match="aliases a suite annotation"):
        worker.verify_inputs(m)


@pytest.mark.parametrize("reduce_zero", [False, True])
def test_cpu_streamed_scoring_and_offline_verification(tmp_path, monkeypatch, reduce_zero):
    m, output, args = sealed_output(tmp_path, monkeypatch, reduce_zero=reduce_zero)
    scoring_dependencies(monkeypatch, m, output)
    events = []
    def barrier(path, manifest):
        events.append("barrier")
        return "2" * 64
    monkeypatch.setattr(common, "verify_barrier", barrier)
    labels = {Path(r["label"]) for r in m["panel"]["inputs"]}
    original_read = Path.read_bytes
    def watched(path):
        if path in labels:
            assert events and events[0] == "barrier"
            events.append("label")
        return original_read(path)
    with mock.patch.object(Path, "read_bytes", watched):
        result = worker.score(args)
    assert events.count("label") == 3
    assert result["miou_unrounded"] == pytest.approx(100 / 6)
    assert result["intersection"] == [3 * 112 * 112, 0]
    assert result["union"] == [3 * 3 * 112 * 112, 3 * 2 * 112 * 112]
    assert result["images"] == 3 and "per_image_counts" not in result
    with mock.patch.object(Path, "read_bytes", lambda path: (_ for _ in ()).throw(AssertionError("No GT in offline verify"))
                           if path in labels else original_read(path)):
        assert worker.verify_score_seal(output, m, args.manifest_sha256) == result
        assert worker.verify(args) == result
    assert len(list(worker.bounded_lines(output / "scores_by_image.jsonl", 3))) == 3
    with pytest.raises(RuntimeError, match="No scoring retry"):
        worker.score(args)


def test_incomplete_barrier_never_opens_gt_or_claims_score(tmp_path, monkeypatch):
    m, output, args = sealed_output(tmp_path, monkeypatch)
    scoring_dependencies(monkeypatch, m, output)
    def missing(path, manifest):
        raise RuntimeError("All eighteen predictions required")
    monkeypatch.setattr(common, "verify_barrier", missing)
    original_read = Path.read_bytes
    labels = {Path(r["label"]) for r in m["panel"]["inputs"]}
    def no_gt(path):
        assert path not in labels
        return original_read(path)
    with mock.patch.object(Path, "read_bytes", no_gt), pytest.raises(RuntimeError, match="eighteen"):
        worker.score(args)
    assert not (output / "SCORING_STARTED.json").exists()


def test_failed_score_is_terminal_and_finite(tmp_path, monkeypatch):
    m, output, args = sealed_output(tmp_path, monkeypatch)
    scoring_dependencies(monkeypatch, m, output)
    with mock.patch.object(worker.shared, "streaming_counts", side_effect=RuntimeError("synthetic score failure")):
        with pytest.raises(RuntimeError, match="synthetic score failure"):
            worker.score(args)
    assert (output / "SCORING_STARTED.json").exists()
    assert worker.read_json(output / "SCORING_FAILED.json")["retry_allowed"] is False
    with pytest.raises(RuntimeError, match="No scoring retry"):
        worker.score(args)
    with pytest.raises(RuntimeError, match="Incomplete scoring"):
        worker.verify(args)


@pytest.mark.parametrize("damage", ["row_order", "counts", "top_metric", "score_inputs", "barrier"])
def test_offline_score_stream_binding_tamper(tmp_path, monkeypatch, damage):
    m, output, args = sealed_output(tmp_path, monkeypatch)
    scoring_dependencies(monkeypatch, m, output)
    worker.score(args)
    if damage in ("row_order", "counts"):
        rows = list(worker.bounded_lines(output / "scores_by_image.jsonl", 3))
        if damage == "row_order":
            rows = rows[::-1]
        else:
            rows[0]["intersection"][0] += 1
        (output / "scores_by_image.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif damage == "top_metric":
        result = worker.read_json(output / "result.json")
        result["miou_unrounded"] += 1
        worker.write_json(output / "result.json", result, replace=True)
    elif damage == "score_inputs":
        inputs = worker.read_json(output / "score_inputs.json")
        inputs["label_sha256"] = {}
        worker.write_json(output / "score_inputs.json", inputs, replace=True)
    else:
        monkeypatch.setattr(common, "verify_barrier", lambda path, m: "3" * 64)
    seal = worker.read_json(output / "SCORED_SEALED.json")
    seal["artifact_sha256"] = {name: worker.sha256(output / name) for name in worker.SCORE_FILES}
    worker.write_json(output / "SCORED_SEALED.json", seal, replace=True)
    with pytest.raises(RuntimeError):
        worker.verify_score_seal(output, m, args.manifest_sha256, args.barrier)


def test_pooled_counts_are_not_mean_of_image_ious():
    a, b = np.array([[1, 0], [1, 0]]), np.array([[0, 1], [99, 1]])
    assert worker.pooled_result(a+b)["miou_unrounded"] == pytest.approx(50.5)
    assert worker.pooled_result(np.zeros((2, 2), np.int64))["miou_unrounded"] is None
    with pytest.raises(RuntimeError, match="Invalid pooled"):
        worker.pooled_result([[2, 0], [1, 0]])
    with pytest.raises(RuntimeError, match="Invalid pooled"):
        worker.pooled_result([[.5, 0], [1, 0]])


@pytest.mark.parametrize("content,maximum", [("{}\n{}\n", 1), ('{"loss":NaN}\n', 1), ("[]\n", 1), ("x"*2_000_001, 1)])
def test_stream_parser_is_bounded_and_finite(tmp_path, content, maximum):
    path = tmp_path / "stream.jsonl"
    path.write_text(content)
    with pytest.raises((RuntimeError, ValueError)):
        list(worker.bounded_lines(path, maximum))


def test_gate_comparison_detects_trajectory_and_none_differences():
    assert worker.maximum_delta({"ctx": torch.tensor([1.])}, {"ctx": torch.tensor([1.000001])}) < 1e-5
    assert worker.maximum_delta([1.], [1.1]) > 1e-5
    with pytest.raises(RuntimeError, match="missing gradient"):
        worker.maximum_delta(None, torch.ones(1))
    with pytest.raises(RuntimeError, match="Nonfinite"):
        worker.maximum_delta(torch.tensor([float("inf")]), torch.ones(1))
