"""CPU-only synthetic protocol tests. No model is built or forwarded here."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_allocation import common, worker
from trace_ovss.method import allocation_runtime as runtime
from trace_ovss.method.image_balanced import ImageBalancedTRACE


@pytest.fixture(autouse=True)
def no_model_or_cuda():
    with mock.patch.object(worker, "build_method", side_effect=AssertionError("No model construction in CPU tests")), \
            mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA in CPU tests")), \
            mock.patch.object(ImageBalancedTRACE, "forward_eval_logits", side_effect=AssertionError("No model forward")), \
            mock.patch.object(ImageBalancedTRACE, "forward_adapt_logits", side_effect=AssertionError("No model forward")):
        yield


def synthetic_manifest(root, dataset="pc60", n=3, batch=2):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        image, label = root / f"{dataset}.{i}.rgb.png", root / f"{dataset}.{i}.gt.png"
        Image.fromarray(np.full((2, 2, 3), i, np.uint8)).save(image)
        Image.fromarray(np.array([[0, 1], [1, 255]], np.uint8)).save(label)
        status = label.stat()
        rows.append(dict(image=str(image), label=str(label), official_id=f"official-{i}",
            image_sha256=worker.sha256(image), label_metadata=[status.st_size, status.st_mtime_ns]))
    cfg = dict(dataset="PascalContext60Dataset" if dataset == "pc60" else "COCOObjectDataset",
        data_dir=str(root), data_subdir="synthetic", resize=[224, 224], patch_size=[224, 224],
        patch_stride=112, batch_size=batch, lr=.001, steps=2, trace_max_regions=4,
        trace_view_batch_size=16, class_extensions=True)
    return dict(schema_version=1, algorithm=common.ALGORITHM, dataset=dataset, stage="full", seed=0,
        permutation_seed=common.PERMUTATION_SEED, arms=common.ARMS.copy(), config=cfg,
        suite_root=str(root), runtime_root=str(root / "runtime"), output=str(root / "full" / dataset),
        suite_prepared_sha256="a" * 64, parent_proof_sha256="b" * 64,
        source_sha256={"tools/evolution_allocation/worker.py": "c" * 64},
        checkpoint=dict(path=str(root / "checkpoint.pt"), sha256="d" * 64),
        panel=dict(inputs=rows, ordered_ids=[r["official_id"] for r in rows], full_split=True,
            classes=["a", "b"], queries=["a", "b"], query_to_class=[0, 1], reduce_zero_label=False))


def candidate(source=0, q=.25, query=0, area=1, component=0):
    mask = np.zeros((2, 2), bool)
    mask.flat[:area] = True
    return dict(batch_idx=source, image_idx=source, class_idx=query, confidence=.8,
        area=area / 4, component_idx=component, original_support=.8, target_support=None,
        counterfactual_support=q, quality=q, reliability=q, residual_support=None,
        dominance=None, diagnostic_algorithm="C", omitted=False, region=mask)


def terms_for(stats):
    return [dict(batch_idx=r["batch_idx"], image_idx=r["image_idx"], rho=r["reliability"],
        indices=torch.from_numpy(np.flatnonzero(r["region"].reshape(-1)))) for r in stats if r["reliability"] > worker.EPS]


def allocation_metadata(stats, arm, sources, dtype=torch.float32):
    terms = terms_for(stats)
    if not terms:
        return None
    entropy = torch.ones((len(sources), 2, 2), dtype=dtype)
    if arm == "image_region":
        prepared = ImageBalancedTRACE.prepare_trace_objective(entropy, terms)
        return worker.native_allocation_metadata(entropy, terms, prepared)
    options = dict(seed=worker.PERMUTATION_SEED, source_keys=sources) if arm == "image_shuffled" else {}
    return worker.token_accounting_metadata(entropy, runtime.allocation_control_metadata(
        runtime.prepare_allocation_control(entropy, terms, control=arm, **options)))


def cost_row(manifest, arm, episode, position, stats, identity):
    cfg = manifest["config"]
    images = min(cfg["batch_size"], len(manifest["panel"]["inputs"]) - episode * cfg["batch_size"])
    views, active = len(stats), sum(r["reliability"] > worker.EPS for r in stats)
    steps = cfg["steps"] if active else 0
    calls = (views + cfg["trace_view_batch_size"] - 1) // cfg["trace_view_batch_size"]
    return dict(arm=arm, episode=episode, order_position=position, warmup=episode == 0,
        images=images, patches=images, candidate_count=views, active_candidates=active,
        steps=steps, seconds=.125, candidate_sha256=identity, **worker.scope(manifest),
        eval_forward_calls=2, eval_forward_inputs=2 * images, adapt_forward_calls=steps,
        adapt_forward_inputs=steps * images, diagnostic_forward_calls=calls,
        diagnostic_forward_inputs=views, diagnostic_encoder_calls=calls, diagnostic_encoder_inputs=views,
        encoder_calls=2 + steps + calls, encoder_inputs=(2 + steps) * images + views)


def sealed_output(root, dataset="pc60", active=False):
    manifest = synthetic_manifest(root, dataset)
    path = common.manifest_path(root, dataset, "full")
    path.parent.mkdir(exist_ok=True)
    worker.write_json(path, manifest, replace=path.exists())
    peer_dataset = next(d for d in worker.DATASETS if d != dataset)
    peer_path = common.manifest_path(root, peer_dataset, "full")
    if not peer_path.exists():
        worker.write_json(peer_path, synthetic_manifest(root, peer_dataset))
    output = Path(manifest["output"])
    worker.started(output, worker.sha256(path), "predict")
    worker.write_json(output / "prediction_protocol.json", dict(manifest=manifest, manifest_path=str(path),
        manifest_sha256=worker.sha256(path), output=str(output), gt_contents_opened=False,
        gate_seal_sha256={d: "e" * 64 for d in worker.DATASETS}, timing_scope=worker.TIMING_SCOPE))
    worker.write_json(output / "initial_state.json", {"synthetic": True})
    (output / "episodes").mkdir()
    rows = []
    for arm in worker.ARMS:
        (output / "predictions" / arm).mkdir(parents=True)
        for i in range(len(manifest["panel"]["inputs"])):
            with (output / f"predictions/{arm}/{i:06d}.npy").open("xb") as stream:
                np.save(stream, np.zeros((224, 224), np.uint16), allow_pickle=False)
    cfg = manifest["config"]
    account = worker.CostAccumulator()
    for episode, start in enumerate(range(0, 3, cfg["batch_size"])):
        stop = min(start + cfg["batch_size"], 3)
        sources = {i: f"{dataset}/official-{start + i}" for i in range(stop - start)}
        stats = [candidate(i, .2 + j * .1, j, j + 1, j) for i in sources for j in range(2)] if active else []
        for position, arm in enumerate(worker.arm_order(episode)):
            records, coverage = worker.coefficient_receipts(stats, arm, sources, [0, 1])
            row = cost_row(manifest, arm, episode, position, stats, worker.diagnostic_identity(records))
            worker.write_json(output / f"episodes/{episode:06d}.{arm}.json", dict(row,
                ordered_ids=manifest["panel"]["ordered_ids"][start:stop], losses=[.2] * row["steps"],
                candidates=records, coverage=coverage, allocation_dtype="torch.float32",
                allocation_metadata=allocation_metadata(stats, arm, sources), gt_contents_opened=False))
            rows.append(row)
            account.add(row, coverage)
    with (output / "cost.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    worker.write_json(output / "prediction_result.json", dict(status="predicted", images=3, arms=worker.ARMS,
        config=cfg, timing_scope=worker.TIMING_SCOPE, **worker.scope(manifest),
        shuffle_coverage=account.shuffle_coverage,
        timings=worker.confirmation.prior.timing_summary(rows, worker.ARMS),
        costs={a: {k: sum(r[k] for r in rows if r["arm"] == a) for k in worker.COUNTERS} for a in worker.ARMS}))
    worker.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
        output=str(output), manifest_sha256=worker.sha256(path), images=3, arms=worker.ARMS,
        suite_prepared_sha256=manifest["suite_prepared_sha256"], **worker.scope(manifest),
        artifact_sha256={name: worker.sha256(output / name) for name in sorted(worker.artifact_names(manifest))}))
    return manifest, output, SimpleNamespace(manifest=path, manifest_sha256=worker.sha256(path), output=output,
        barrier=root / "ALL_FULL_PREDICTIONS_SEALED.json")


@pytest.fixture
def synthetic_schema(monkeypatch):
    # Population-size admission is owned/tested by common; use tiny file panels
    # only for CPU streaming protocol tests, without altering common's schema.
    def validate(manifest):
        assert manifest["algorithm"] == worker.ALGORITHM
        assert manifest["stage"] == "full" and manifest["arms"] == worker.ARMS
        assert manifest["dataset"] in worker.DATASETS
    monkeypatch.setattr(worker, "validate_manifest", validate)
    monkeypatch.setattr(worker, "verify_gates", lambda manifest: {d: "e" * 64 for d in worker.DATASETS})


def make_barrier(root, outputs):
    path = root / "ALL_FULL_PREDICTIONS_SEALED.json"
    worker.write_json(path, dict(complete=True, stage="full", gt_contents_opened=False,
        suite_prepared_sha256="a" * 64,
        prediction_seal_sha256={str(output): worker.sha256(output / "PREDICTIONS_SEALED.json") for output in outputs}))
    return path


def rewrite(path, value):
    path.unlink()
    worker.write_json(path, value)


@pytest.mark.parametrize("arm", worker.ARMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_metadata_and_loss_coefficients(arm, dtype):
    stats = [candidate(0, .2, 0, 1), candidate(0, .7, 1, 3, 1), candidate(1, .4, 0, 2)]
    sources = {0: "pc60/image-a", 1: "pc60/image-b"}
    rows, coverage = worker.coefficient_receipts(stats, arm, sources, [0, 1], dtype=dtype)
    metadata = allocation_metadata(stats, arm, sources, dtype)
    worker.validate_allocation_metadata(metadata, rows, arm, dtype)
    assert sum(r["ideal_coefficient"] for r in rows) == pytest.approx(1)
    assert [r["native_C"] for r in rows] == [.2, .7, .4]
    assert coverage["active_candidates"] == 3
    entropy = torch.arange(8, dtype=dtype).reshape(2, 2, 2).requires_grad_()
    terms = terms_for(stats)
    options = dict(seed=worker.PERMUTATION_SEED, source_keys=sources) if arm == "image_shuffled" else {}
    loss = runtime.allocation_control_entropy(entropy, terms, control=arm, **options)
    expected = sum(r["objective_coefficient"] * entropy[r["batch_idx"]][s["region"]].mean()
                   for r, s in zip(rows, stats))
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
    assert worker.diagnostic_identity(rows) == worker.diagnostic_identity(
        worker.coefficient_receipts(stats, "image_region", sources, [0, 1], dtype=dtype)[0])


def test_native_hook_delegates_exactly_without_forward():
    method = object.__new__(worker.ExperimentTRACE)
    method.arm, method.source_keys = "image_region", {0: "pc60/a"}
    entropy = torch.tensor([[[.2, .4], [.6, .8]]], requires_grad=True)
    terms = terms_for([candidate()])
    prepared = method.prepare_trace_objective(entropy, terms)
    native = ImageBalancedTRACE.prepare_trace_objective(entropy, terms)
    actual = method.trace_region_normalized_entropy(entropy, terms, prepared=prepared)
    expected = ImageBalancedTRACE.trace_region_normalized_entropy(entropy, terms, prepared=native)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, entropy, retain_graph=True)[0], torch.autograd.grad(expected, entropy)[0])


@pytest.mark.parametrize("arm", worker.ARMS[1:])
def test_non_native_hooks_use_device_runtime_without_forward(arm):
    method = object.__new__(worker.ExperimentTRACE)
    method.arm = arm
    method.bind_sources({0: "pc60/actual-source"})
    entropy = torch.ones((1, 2, 2), requires_grad=True)
    terms = terms_for([candidate()])
    with mock.patch.object(runtime, "prepare_allocation_control", wraps=runtime.prepare_allocation_control) as prepare:
        cache = method.prepare_trace_objective(entropy, terms)
        loss = method.trace_region_normalized_entropy(entropy, terms, prepared=cache)
    assert prepare.call_count == 1 and loss.requires_grad
    assert method.allocation_metadata["control"] == arm
    method.bind_sources({0: "pc60/new-episode"})
    assert method.allocation_metadata is None


@pytest.mark.parametrize("arm", worker.ARMS)
def test_eps_activity_and_empty_receipts(arm):
    rows, coverage = worker.coefficient_receipts([candidate(q=worker.EPS)], arm, {0: "pc60/empty"}, [0, 1])
    assert not rows[0]["active"] and rows[0]["objective_coefficient"] == 0
    assert coverage["active_candidates"] == 0
    worker.validate_allocation_metadata(None, rows, arm, torch.float32)
    assert worker.coefficient_receipts([], arm, {}, [0, 1])[0] == []


def test_shuffle_stable_source_mapping_and_no_forced_derangement():
    stats = [candidate(0, .1 + i * .1, 0, 1, i) for i in range(6)] + [candidate(1)]
    sources = {0: "pc60/stable-path-id", 1: "pc60/singleton"}
    before = torch.get_rng_state().clone()
    rows, coverage = worker.coefficient_receipts(stats, "image_shuffled", sources, [0, 1])
    assert torch.equal(before, torch.get_rng_state())
    assert sorted(r["assigned_C"] for r in rows[:6]) == sorted(r["native_C"] for r in rows[:6])
    assert rows[-1]["assigned_from_candidate"] == 6
    assert coverage["eligible_candidates"] == 6 and coverage["eligible_sources"] == 1
    assert worker.source_permutation(6, sources[0]) == worker.source_permutation(6, sources[0])
    from trace_ovss.method.allocation_controls import keyed_source_permutation
    assert worker.source_permutation(6, sources[0]) == list(keyed_source_permutation(6,
        seed=worker.PERMUTATION_SEED, source_key=sources[0]))
    assert any(worker.source_permutation(2, f"pc60/{i}") == [0, 1] for i in range(20))


@pytest.mark.parametrize("field", ["native_C", "active", "counterfactual_support", "source_key", "class_idx"])
def test_diagnostic_identity_includes_support_activity_and_source(field):
    rows, _ = worker.coefficient_receipts([candidate()], "image_region", {0: "pc60/a"}, [0, 1])
    changed = copy.deepcopy(rows)
    changed[0][field] = "different" if field == "source_key" else not changed[0][field] if field == "active" else .125
    assert worker.diagnostic_identity(rows) != worker.diagnostic_identity(changed)
    changed = copy.deepcopy(rows)
    changed[0]["assigned_C"] = .9
    changed[0]["objective_coefficient"] = .9
    assert worker.diagnostic_identity(rows) == worker.diagnostic_identity(changed)


def test_real_tensor_rounding_is_not_ideal_C():
    rows, _ = worker.coefficient_receipts([candidate(q=.1), candidate(q=.2, component=1)],
        "image_region", {0: "pc60/a"}, [0, 1])
    assert rows[0]["native_tensor_C"] != rows[0]["native_C"]
    assert rows[0]["objective_coefficient"] != rows[0]["ideal_coefficient"]


@pytest.mark.parametrize("arm", worker.ARMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("prompts", [1, 7])
def test_token_accounting_keeps_device_and_portable_values_without_changing_loss(arm, dtype, prompts):
    stats = [candidate(q=.127, area=3), candidate(q=.731, area=2, component=1)]
    sources = {0: "pc60/token-accounting"}
    terms = terms_for(stats)
    shape = (1, 2, 2) if prompts == 1 else (prompts, 1, 2, 2)
    entropy = torch.arange(np.prod(shape), dtype=dtype).reshape(shape).requires_grad_()
    options = dict(seed=worker.PERMUTATION_SEED, source_keys=sources) if arm == "image_shuffled" else {}
    cache = runtime.prepare_allocation_control(entropy, terms, control=arm, **options)
    before = runtime.allocation_control_entropy(entropy, terms, prepared=cache)
    gradient = torch.autograd.grad(before, entropy, retain_graph=True)[0]
    metadata = worker.token_accounting_metadata(entropy, runtime.allocation_control_metadata(cache))
    rows, _ = worker.coefficient_receipts(stats, arm, sources, [0, 1], dtype=dtype)
    worker.validate_allocation_metadata(metadata, rows, arm, dtype)
    after = runtime.allocation_control_entropy(entropy, terms, prepared=cache)
    assert torch.equal(before, after)
    assert torch.equal(gradient, torch.autograd.grad(after, entropy)[0])
    for row in metadata["candidates"]:
        expected = float(torch.tensor(row["coefficient"], dtype=dtype) / row["area"] / prompts)
        assert row["portable_token_coefficient"] == expected == row["token_coefficient_reference"]
        assert row["token_device_minus_portable"] == 0


@pytest.mark.parametrize("field", ["token_coefficient", "token_coefficient_reference",
                                   "portable_token_coefficient", "token_device_minus_portable",
                                   "coefficient", "coefficient_reference", "coefficient_device_minus_portable"])
def test_token_accounting_rejects_corrupt_fields(field):
    stats, sources = [candidate(area=3)], {0: "pc60/a"}
    metadata = allocation_metadata(stats, "image_region", sources)
    metadata["candidates"][0][field] += .125
    rows, _ = worker.coefficient_receipts(stats, "image_region", sources, [0, 1])
    with pytest.raises(RuntimeError, match="Token accounting differs"):
        worker.validate_allocation_metadata(metadata, rows, "image_region", torch.float32)


def test_token_device_difference_is_observed_not_silently_replaced():
    stats, sources = [candidate(area=3)], {0: "pc60/a"}
    metadata = allocation_metadata(stats, "image_region", sources)
    rows, _ = worker.coefficient_receipts(stats, "image_region", sources, [0, 1])
    # Synthetic device-rounding receipt, NOT a measurement from CUDA.
    record = metadata["candidates"][0]
    shifted = float(torch.nextafter(torch.tensor(record["token_coefficient"]), torch.tensor(float("inf"))))
    record.update(token_coefficient=shifted, token_coefficient_reference=shifted,
                  token_device_minus_portable=shifted - record["portable_token_coefficient"])
    with pytest.raises(RuntimeError, match="Token accounting differs"):
        worker.validate_allocation_metadata(metadata, rows, "image_region", torch.float32)
    metadata["device"] = "cuda:0"
    worker.validate_allocation_metadata(metadata, rows, "image_region", torch.float32)
    assert record["token_coefficient"] != record["portable_token_coefficient"]


def test_candidate_coefficient_device_difference_has_separate_portable_accounting():
    stats, sources = [candidate(area=3)], {0: "pc60/a"}
    metadata = allocation_metadata(stats, "image_region", sources)
    rows, _ = worker.coefficient_receipts(stats, "image_region", sources, [0, 1])
    record = metadata["candidates"][0]
    # Synthetic exported observation with a different rounded division result.
    record["coefficient"] = float(torch.nextafter(torch.tensor(record["coefficient"]), torch.tensor(float("inf"))))
    record["coefficient_reference"] = record["coefficient"]
    record["coefficient_device_minus_portable"] = record["coefficient"] - record["portable_coefficient"]
    record["token_coefficient"] = float(torch.tensor(record["coefficient"]) / record["area"])
    record["token_coefficient_reference"] = record["token_coefficient"]
    record["token_device_minus_portable"] = record["token_coefficient"] - record["portable_token_coefficient"]
    metadata["device"] = "cuda:0"
    worker.validate_allocation_metadata(metadata, rows, "image_region", torch.float32)
    assert record["coefficient"] != rows[0]["objective_coefficient"]
    record["portable_coefficient"] = record["coefficient"]
    with pytest.raises(RuntimeError, match="Token accounting differs"):
        worker.validate_allocation_metadata(metadata, rows, "image_region", torch.float32)


@pytest.mark.parametrize("field,value", [("token_accounting", None), ("device", "cuda:1"), ("device", "unknown")])
def test_token_accounting_rejects_missing_contract_or_device(field, value):
    stats, sources = [candidate()], {0: "pc60/a"}
    metadata = allocation_metadata(stats, "image_region", sources)
    metadata[field] = value
    rows, _ = worker.coefficient_receipts(stats, "image_region", sources, [0, 1])
    with pytest.raises(RuntimeError, match="scope differs"):
        worker.validate_allocation_metadata(metadata, rows, "image_region", torch.float32)


def test_same_device_mismatch_keeps_both_values_and_source_context():
    stats, sources = [candidate(area=3)], {0: "pc60/a"}
    metadata = allocation_metadata(stats, "image_region", sources)
    metadata["candidates"][0]["token_coefficient"] += .125
    with pytest.raises(RuntimeError, match="Same-device token coefficient differs") as caught:
        worker.token_accounting_metadata(torch.ones((1, 2, 2)), metadata)
    detail = caught.value.worker_context["token_accounting"]
    assert detail["observed_token_coefficient"] != detail["same_device_token_coefficient"]
    assert detail["area"] == 3 and detail["prompts"] == 1 and detail["term_index"] == 0
    json.dumps(detail, allow_nan=False)


def test_nonfinite_token_failure_remains_serializable():
    metadata = allocation_metadata([candidate()], "image_region", {0: "pc60/a"})
    metadata["candidates"][0]["token_coefficient"] = float("nan")
    with pytest.raises(RuntimeError, match="Same-device token coefficient differs") as caught:
        worker.token_accounting_metadata(torch.ones((1, 2, 2)), metadata)
    assert caught.value.worker_context["token_accounting"]["observed_token_coefficient"] == "nan"
    json.dumps(worker.tuning.failure_receipt(caught.value, "gate", "a" * 64), allow_nan=False)


def test_run_policy_preserves_numeric_failure_context(monkeypatch):
    failure = RuntimeError("Synthetic accounting failure")
    failure.worker_context = dict(token_accounting=dict(observed=.5, reference=.25))
    @contextmanager
    def nothing(*args):
        yield {}
    monkeypatch.setattr(worker, "finite_observation", nothing)
    monkeypatch.setattr(worker.shared, "synchronize", lambda device: None)
    monkeypatch.setattr(worker.shared, "restored_state", lambda method: {})
    method = SimpleNamespace(device=torch.device("cpu"), reset=lambda: None, count_work=nothing,
                             adapt=mock.Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        worker.run_policy(method, "image_region", torch.ones(1, 3, 2, 2), None, None,
                          dict(img_path=["a"]), {}, None, {}, {0: "pc60/a"})
    assert caught.value.worker_context["token_accounting"] == dict(observed=.5, reference=.25)
    assert caught.value.worker_context["arm"] == "image_region"


@pytest.mark.parametrize("field", ["portable_coefficient", "portable_token_coefficient"])
@pytest.mark.parametrize("corrupt", [.125, float("nan")])
def test_actual_accounting_failures_survive_run_policy_and_strict_json(monkeypatch, field, corrupt):
    stats, sources = [candidate(area=3)], {0: "pc60/a"}
    method = object.__new__(worker.ExperimentTRACE)
    method.device, method.allocation_dtype = torch.device("cpu"), torch.float32
    method.allocation_metadata = allocation_metadata(stats, "image_region", sources)
    original = method.allocation_metadata["candidates"][0][field]
    method.allocation_metadata["candidates"][0][field] = original + corrupt
    method.last_candidate_stats = stats
    method.reset, method.bind_sources = lambda: None, lambda keys: None
    method.adapt = lambda *args, **kwargs: [.2]
    @contextmanager
    def nothing(*args):
        yield {}
    method.count_work = nothing
    monkeypatch.setattr(worker, "finite_observation", nothing)
    monkeypatch.setattr(worker.shared, "synchronize", lambda device: None)
    monkeypatch.setattr(worker.shared, "restored_state", lambda method: {})
    monkeypatch.setattr(worker.shared, "label_predictions", lambda *args: [np.zeros((2, 2), np.uint16)])
    with pytest.raises(RuntimeError, match="Token accounting differs") as caught:
        worker.run_policy(method, "image_region", torch.ones(1, 3, 2, 2), None, None,
                          dict(img_path=["a"]), {}, [0, 1], {}, sources)
    detail = caught.value.worker_context["token_accounting"]
    stored = "stored_" + field
    recomputed = "recomputed_" + field
    assert detail[stored] != detail[recomputed] and detail[recomputed] == original
    assert detail["area"] == 3 and detail["prompts"] == detail["groups"] == 1
    assert detail["dtype"] == "torch.float32" and detail["device"] == "cpu"
    assert caught.value.worker_context["arm"] == "image_region"
    receipt = worker.tuning.failure_receipt(caught.value, "gate", "a" * 64)
    assert receipt["retry_allowed"] is False and receipt["context"]["token_accounting"] == detail
    json.dumps(receipt, allow_nan=False)


def test_source_binding_checks_actual_meta_path_and_order(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    meta = dict(img_path=[r["image"] for r in manifest["panel"]["inputs"][:2]])
    assert worker.stable_sources(manifest, meta, 0, 2) == {0: "pc60/official-0", 1: "pc60/official-1"}
    meta["img_path"].reverse()
    with pytest.raises(RuntimeError, match="source order"):
        worker.stable_sources(manifest, meta, 0, 2)


def test_true_rotating_order():
    assert worker.arm_order(0) == worker.ARMS
    assert [worker.arm_order(i)[0] for i in range(5)] == worker.ARMS
    assert worker.arm_order(5) == worker.ARMS
    with pytest.raises(RuntimeError):
        worker.arm_order(True)


@pytest.mark.parametrize("active", [False, True])
def test_seal_cost_and_cpu_verify(tmp_path, synthetic_schema, monkeypatch, active):
    manifest, output, args = sealed_output(tmp_path, active=active)
    assert worker.verify_prediction_seal(output, manifest, args.manifest_sha256) == manifest
    result = worker.validate_cost_stream(output, manifest)
    assert result["images"] == 3
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, output))
    monkeypatch.setattr(worker, "verify_inputs", lambda manifest: None)
    assert worker.verify(args)["verified"]


@pytest.mark.parametrize("key,value", [("steps", 1), ("patches", 3), ("images", 0),
    ("candidate_count", 9), ("active_candidates", 2), ("encoder_calls", 99),
    ("diagnostic_forward_inputs", 0), ("seconds", float("nan")), ("eval_forward_calls", True)])
def test_cost_rejects_schedule_and_budget_drift(tmp_path, key, value):
    manifest = synthetic_manifest(tmp_path)
    row = cost_row(manifest, "image_token", 0, 0, [candidate()], "a" * 64)
    row[key] = value
    with pytest.raises(RuntimeError):
        worker.validate_cost(row, "image_token", manifest["config"])


@pytest.mark.parametrize("mode", ["payload", "extra", "missing", "symlink", "manifest", "attempt"])
def test_prediction_seal_rejects_drift(tmp_path, synthetic_schema, mode):
    manifest, output, args = sealed_output(tmp_path)
    prediction = output / "predictions/image_region/000000.npy"
    if mode == "payload":
        rewrite(output / "initial_state.json", {"tampered": True})
    elif mode == "extra":
        worker.write_json(output / "extra.json", {})
    elif mode == "missing":
        prediction.unlink()
    elif mode == "symlink":
        prediction.unlink()
        prediction.symlink_to(output / "predictions/image_region/000001.npy")
    elif mode == "manifest":
        manifest["source_sha256"] = {"other.py": "e" * 64}
        rewrite(args.manifest, manifest)
    else:
        output.with_name(output.name + ".STARTED.json").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        worker.verify_prediction_seal(output)


def test_receipt_rejects_forged_actual_coefficient_and_activity(tmp_path, synthetic_schema):
    manifest, output, _ = sealed_output(tmp_path, active=True)
    path = output / "episodes/000000.image_region.json"
    original = worker.read_json(path)
    changed = copy.deepcopy(original)
    changed["allocation_metadata"]["candidates"][0]["coefficient"] += .1
    rewrite(path, changed)
    with pytest.raises(RuntimeError, match="Token accounting differs"):
        worker.validate_cost_stream(output, manifest)
    changed = copy.deepcopy(original)
    changed["candidates"][0]["native_C"] += .1
    rewrite(path, changed)
    with pytest.raises(RuntimeError, match="Coefficient/source/permutation"):
        worker.validate_cost_stream(output, manifest)


def test_barrier_requires_both_full_outputs_before_gt(tmp_path, synthetic_schema):
    manifest, output, args = sealed_output(tmp_path)
    make_barrier(tmp_path, [output])
    with pytest.raises(RuntimeError, match="both exact full populations"):
        common.verify_barrier(args.barrier, manifest, output, args.manifest_sha256)


def test_score_stream_and_full_cpu_reverification(tmp_path, synthetic_schema, monkeypatch):
    manifest, output, args = sealed_output(tmp_path, active=True)
    _, peer, _ = sealed_output(tmp_path, "coco_object")
    make_barrier(tmp_path, [output, peer])
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, output))
    monkeypatch.setattr(worker, "verify_inputs", lambda manifest: None)
    original_read = Path.read_bytes
    label_paths = {r["label"] for r in manifest["panel"]["inputs"]}
    opened = []
    def guarded_read(path):
        if str(path) in label_paths:
            assert (output / "SCORING_STARTED.json").is_file()
            assert common.verify_barrier(args.barrier, manifest, output, args.manifest_sha256)
            opened.append(str(path))
        return original_read(path)
    with mock.patch.object(Path, "read_bytes", guarded_read):
        result = worker.score(args)
    assert len(opened) == 3 and result["images"] == 3 and len(result["ten_arm_scope"]) == 10
    assert result["arm_results"]["image_region"]["miou_unrounded"] == pytest.approx(100 / 6)
    assert len(list(worker.full.bounded_json_lines(output / "scores_by_image.jsonl", 3))) == 3
    # Verification must never reopen GT bytes.
    def no_gt(path):
        assert str(path) not in label_paths
        return original_read(path)
    with mock.patch.object(Path, "read_bytes", no_gt):
        assert worker.verify(args)["scored"] is True
    with pytest.raises(RuntimeError, match="No score retry"):
        worker.score(args)


@pytest.mark.parametrize("attack", ["missing_peer", "peer_payload", "barrier_scope"])
def test_score_failclosed_before_any_gt(tmp_path, synthetic_schema, monkeypatch, attack):
    manifest, output, args = sealed_output(tmp_path)
    _, peer, _ = sealed_output(tmp_path, "coco_object")
    barrier = make_barrier(tmp_path, [output] if attack == "missing_peer" else [output, peer])
    if attack == "peer_payload":
        rewrite(peer / "initial_state.json", {"drift": True})
    if attack == "barrier_scope":
        value = worker.read_json(barrier)
        value["suite_prepared_sha256"] = "f" * 64
        rewrite(barrier, value)
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, output))
    monkeypatch.setattr(worker, "verify_inputs", lambda manifest: None)
    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("GT must remain closed")):
        with pytest.raises(RuntimeError):
            worker.score(args)
    assert not (output / "SCORING_STARTED.json").exists()


def test_incomplete_scoring_is_not_verifiable_or_retryable(tmp_path, synthetic_schema, monkeypatch):
    manifest, output, args = sealed_output(tmp_path)
    worker.write_json(output / "SCORING_STARTED.json", {})
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, output))
    monkeypatch.setattr(worker, "verify_inputs", lambda manifest: None)
    with pytest.raises(FileNotFoundError):
        worker.verify(args)
    with pytest.raises(RuntimeError, match="No score retry"):
        worker.score(args)


def test_started_survives_failed_gate_and_forbids_retry(tmp_path):
    output = tmp_path / "gate"
    claim = worker.started(output, "a" * 64, "gate")
    assert claim.is_file() and worker.read_json(output / "STARTED.json")["retry_allowed"] is False
    with pytest.raises(RuntimeError, match="no retry"):
        worker.started(output, "a" * 64, "gate")
    (output / "STARTED.json").unlink()
    output.rmdir()
    with pytest.raises(RuntimeError, match="no retry"):
        worker.started(output, "a" * 64, "gate")


def test_failure_receipt_no_recovery():
    receipt = worker.tuning.failure_receipt(RuntimeError("Nonfinite raw diagnostic"), "gate", "a" * 64)
    assert receipt["failure_kind"] == "nonfinite" and receipt["retry_allowed"] is False


@pytest.mark.parametrize("phase", ["gate", "predict", "score", "verify"])
def test_cli_dispatch_without_models(tmp_path, monkeypatch, phase):
    stub = mock.Mock(return_value={"phase": phase})
    monkeypatch.setattr(worker, phase, stub)
    argv = ["--phase", phase, "--manifest", str(tmp_path / "manifest.json"),
            "--manifest-sha256", "a" * 64, "--output", str(tmp_path / "output")]
    if phase == "score":
        argv += ["--barrier", str(tmp_path / "barrier.json")]
    assert worker.main(argv) == {"phase": phase}
    assert stub.call_count == 1


@pytest.mark.parametrize("phase,barrier", [("score", False), ("predict", True), ("gate", True), ("verify", True)])
def test_cli_rejects_missing_or_misplaced_barrier(tmp_path, phase, barrier):
    argv = ["--phase", phase, "--manifest", str(tmp_path / "manifest.json"),
            "--manifest-sha256", "a" * 64, "--output", str(tmp_path / "output")]
    if barrier:
        argv += ["--barrier", str(tmp_path / "barrier.json")]
    with pytest.raises(SystemExit):
        worker.main(argv)


def test_common_schema_unchanged_and_exact(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    manifest["stage"] = "gate"
    extra = dict(manifest, unexpected="reject")
    with pytest.raises(RuntimeError, match="schema"):
        worker.validate_manifest(extra)
    assert worker.ARMS == ["image_region", "episode_region", "image_token", "episode_token", "image_shuffled"]
    assert worker.tuning.ARMS == ["no_adapt", "image_uniform", "image_confidence", "image_C"]


def test_shuffle_aggregate_recomputed_not_trusted(tmp_path, synthetic_schema):
    manifest, output, _ = sealed_output(tmp_path, active=True)
    predicted = worker.validate_cost_stream(output, manifest)
    coverage = predicted["shuffle_coverage"]
    assert set(coverage) == set(worker.SHUFFLE_COUNTS)
    assert coverage["active_candidates"] == coverage["eligible_candidates"] == 6
    assert coverage["active_sources"] == coverage["eligible_sources"] == 3
    assert all(type(v) is int for v in coverage.values())
    predicted["shuffle_coverage"]["changed_score_candidates"] += 1
    rewrite(output / "prediction_result.json", predicted)
    with pytest.raises(RuntimeError, match="Aggregate"):
        worker.validate_cost_stream(output, manifest)


def test_hardlinked_output_never_hashes_gt(tmp_path, synthetic_schema):
    manifest, output, args = sealed_output(tmp_path)
    target = output / "predictions/image_region/000000.npy"
    target.unlink()
    target.hardlink_to(manifest["panel"]["inputs"][0]["label"])
    with mock.patch.object(worker, "sha256", side_effect=AssertionError("Must stat before payload hashing")):
        with pytest.raises(RuntimeError, match="aliases annotation"):
            worker.verify_prediction_seal(output, manifest, args.manifest_sha256)


def test_input_guards_read_rgb_and_vocab_but_no_label_bytes(tmp_path, monkeypatch):
    manifest = synthetic_manifest(tmp_path)
    panel = manifest["panel"]
    data = SimpleNamespace(vocabulary=SimpleNamespace(classes=panel["classes"], extensions=panel["queries"],
        extension_to_class=panel["query_to_class"]), spec=SimpleNamespace(reduce_zero_label=False))
    monkeypatch.setattr(worker.tuning, "make_data", lambda config, rows: data)
    # The inherited RGB/vocabulary validator is isolated here. Global peer-
    # annotation admission has separate real-inode tests below.
    monkeypatch.setattr(common, "guard_global_inputs", lambda manifest: None)
    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("No annotation opens")):
        assert worker.verify_inputs(manifest) is data
    wrong = copy.deepcopy(manifest)
    wrong["panel"]["queries"] = ["different"]
    with pytest.raises(RuntimeError, match="vocabulary"):
        worker.verify_inputs(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["panel"]["inputs"][0]["image_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="RGB input drift"):
        worker.verify_inputs(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["panel"]["inputs"][0]["label_metadata"][0] += 1
    with pytest.raises(RuntimeError, match="Label metadata"):
        worker.verify_inputs(wrong)


@pytest.mark.parametrize("dataset", worker.DATASETS)
@pytest.mark.parametrize("relative", ["prediction_protocol.json", "PREDICTIONS_SEALED.json",
    "predictions/image_region/000000.npy", "STARTED.json"])
def test_peer_annotation_alias_rejected_before_any_payload_hash(tmp_path, synthetic_schema, dataset, relative):
    manifest, output, args = sealed_output(tmp_path, dataset)
    peer_dataset = next(d for d in worker.DATASETS if d != dataset)
    peer = common.read_json(common.manifest_path(tmp_path, peer_dataset, "full"))
    target = output / relative
    target.unlink()
    target.hardlink_to(peer["panel"]["inputs"][0]["label"])
    with mock.patch.object(worker, "sha256", side_effect=AssertionError("Must reject peer GT before hashing")):
        with pytest.raises(RuntimeError, match="aliases annotation"):
            worker.verify_prediction_seal(output, manifest, args.manifest_sha256)


def test_predict_requires_gates_and_persists_failure_before_model(tmp_path, monkeypatch):
    manifest = synthetic_manifest(tmp_path)
    args = SimpleNamespace(output=Path(manifest["output"]), manifest=tmp_path / "manifest.json", manifest_sha256="a" * 64)
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, args.output))
    monkeypatch.setattr(common, "require_gpu_envelope", lambda manifest: None)
    with pytest.raises(FileNotFoundError):
        worker.predict(args)
    assert (args.output / "STARTED.json").is_file()
    failure = worker.read_json(args.output / "PREDICTION_FAILED.json")
    assert failure["retry_allowed"] is False
    with pytest.raises(RuntimeError, match="no retry"):
        worker.predict(args)


@pytest.mark.parametrize("phase", ["gate", "predict"])
def test_direct_gpu_worker_requires_controller_before_attempt_or_model(tmp_path, monkeypatch, phase):
    manifest = synthetic_manifest(tmp_path, n=4)
    manifest["stage"] = "gate" if phase == "gate" else "full"
    output = tmp_path / ("gates" if phase == "gate" else "full") / "pc60"
    manifest["output"] = str(output)
    args = SimpleNamespace(output=output, manifest=tmp_path / "manifest.json", manifest_sha256="a" * 64)
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, output))
    with pytest.raises(RuntimeError, match="recorded controller invocation"):
        getattr(worker, phase)(args)
    assert not output.exists()


@pytest.mark.parametrize("fault", [None, "gradient", "parameter", "labels", "support", "no_update"])
def test_mocked_gate_core_preserves_strict_thresholds_without_forward(tmp_path, monkeypatch, fault):
    manifest = synthetic_manifest(tmp_path)
    cfg, sources = manifest["config"], {0: "pc60/a", 1: "pc60/b"}
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    model = SimpleNamespace(named_parameters=lambda: [("ln.weight", parameter)])
    method = SimpleNamespace(reset=mock.Mock(), model=model, device=torch.device("cpu"), allocation_metadata=None)
    reference = SimpleNamespace(reset=mock.Mock(), model=model, device=torch.device("cpu"))
    monkeypatch.setattr(worker.shared, "restored_state", lambda current: {"restored": True})
    calls = []
    @contextmanager
    def gradients(current):
        delta = .0001 if fault == "gradient" and len(calls) == 1 else 0.
        yield {"ln.weight": [torch.tensor([.2 + delta])] * cfg["steps"]}
    monkeypatch.setattr(worker.full, "gradient_receipt", gradients)
    def policy(current, arm, *args):
        direct = current is reference
        stats = [candidate(q=.3 if fault == "support" and not direct else .25)]
        records, coverage = worker.coefficient_receipts(stats, arm, sources, [0, 1])
        row = cost_row(manifest, arm, 0, 0, stats, worker.diagnostic_identity(records))
        prediction = np.zeros((224, 224), np.uint16)
        if fault == "labels" and not direct:
            prediction[0, 0] = 1
        if fault == "parameter" and not direct:
            with torch.no_grad():
                parameter.add_(.0001)
        losses = [] if fault == "no_update" else [.2] * cfg["steps"]
        calls.append(arm)
        return losses, [prediction, prediction.copy()], records, coverage, row
    monkeypatch.setattr(worker, "run_policy", policy)
    invoke = lambda: worker.compare_gate_episode(method, reference, torch.zeros(2, 3, 2, 2), None, None,
        dict(img_path=["a", "b"]), cfg, [0, 1], sources)
    if fault is None:
        report = invoke()
        assert report["changed_label_pixels"] == 0 and len(report["actual_arm_counts"]) == 6
        assert calls == ["image_region", *worker.ARMS]
        assert method.reset.call_count == 11 and reference.reset.call_count == 2
    else:
        with pytest.raises(RuntimeError):
            invoke()


@pytest.mark.parametrize("reduce,all_ignore", [(True, False), (False, True)])
def test_scoring_native_reduce_and_ignore_semantics(tmp_path, synthetic_schema, monkeypatch, reduce, all_ignore):
    manifest, output, args = sealed_output(tmp_path)
    _, peer, _ = sealed_output(tmp_path, "coco_object")
    for row in manifest["panel"]["inputs"]:
        value = np.full((2, 2), 255, np.uint8) if all_ignore else np.array([[0, 1], [2, 255]], np.uint8)
        Image.fromarray(value).save(row["label"])
        status = Path(row["label"]).stat()
        row["label_metadata"] = [status.st_size, status.st_mtime_ns]
    manifest["panel"]["reduce_zero_label"] = reduce
    # Rebind the synthetic manifest and protocol before creating the barrier.
    rewrite(args.manifest, manifest)
    args.manifest_sha256 = worker.sha256(args.manifest)
    protocol = worker.read_json(output / "prediction_protocol.json")
    protocol.update(manifest=manifest, manifest_sha256=args.manifest_sha256)
    rewrite(output / "prediction_protocol.json", protocol)
    claim_path = output.with_name(output.name + ".STARTED.json")
    rewrite(claim_path, dict(status="started", manifest_sha256=args.manifest_sha256))
    begin = worker.read_json(output / "STARTED.json")
    begin.update(manifest_sha256=args.manifest_sha256, attempt_sha256=worker.sha256(claim_path))
    rewrite(output / "STARTED.json", begin)
    seal = worker.read_json(output / "PREDICTIONS_SEALED.json")
    seal.update(manifest_sha256=args.manifest_sha256,
        artifact_sha256={name: worker.sha256(output / name) for name in worker.artifact_names(manifest)})
    rewrite(output / "PREDICTIONS_SEALED.json", seal)
    make_barrier(tmp_path, [output, peer])
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, output))
    monkeypatch.setattr(worker, "verify_inputs", lambda manifest: None)
    result = worker.score(args)
    value = result["arm_results"]["image_region"]
    assert value["miou_unrounded"] is None if all_ignore else value["miou_unrounded"] == pytest.approx(25.)
    assert result["shuffle_coverage"] == dict.fromkeys(worker.SHUFFLE_COUNTS, 0)
    assert worker.verify(args)["scored"]


@pytest.mark.parametrize("arm", worker.ARMS)
def test_native_eps_rounding_guard_preserved_in_receipts(arm):
    q = np.nextafter(float(worker.EPS), float("inf"))
    stats = [candidate(q=float(q))]
    sources = {0: "pc60/near-eps"}
    records, _ = worker.coefficient_receipts(stats, arm, sources, [0, 1])
    assert records[0]["active"]
    worker.validate_allocation_metadata(allocation_metadata(stats, arm, sources), records, arm, torch.float32)


@pytest.mark.parametrize("attack", ["image_counts", "result", "scope", "label_coverage"])
def test_scored_semantic_drift_rejected_even_with_new_payload_hash(tmp_path, synthetic_schema, monkeypatch, attack):
    manifest, output, args = sealed_output(tmp_path)
    _, peer, _ = sealed_output(tmp_path, "coco_object")
    make_barrier(tmp_path, [output, peer])
    monkeypatch.setattr(worker, "admission", lambda args: (manifest, output))
    monkeypatch.setattr(worker, "verify_inputs", lambda manifest: None)
    worker.score(args)
    if attack == "image_counts":
        name = "scores_by_image.jsonl"
        records = list(worker.full.bounded_json_lines(output / name, 3))
        records[0]["arms"]["image_region"]["intersection"][0] += 1
        (output / name).unlink()
        with (output / name).open("x") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
    else:
        name = "score_inputs.json" if attack == "label_coverage" else "result.json"
        value = worker.read_json(output / name)
        if attack == "result":
            value["arm_results"]["image_region"]["miou_unrounded"] += 1
        elif attack == "scope":
            value["ten_arm_scope"].pop()
        else:
            value["label_sha256"].pop(next(iter(value["label_sha256"])))
        rewrite(output / name, value)
    seal = worker.read_json(output / "SCORED_SEALED.json")
    seal["artifact_sha256"][name] = worker.sha256(output / name)
    rewrite(output / "SCORED_SEALED.json", seal)
    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("No GT in offline verifier")):
        with pytest.raises(RuntimeError):
            worker.verify(args)

