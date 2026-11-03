import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from tools.evolution_fifty import generalization_score as metric
from tools.evolution_fifty import generalization_score_worker as worker
from tools.evolution_fifty.generalization import ARMS, CORRUPTIONS


def png(values, palette=False):
    image = Image.fromarray(np.asarray(values, dtype=np.uint8))
    if palette:
        image = image.convert("P")
        image.putpalette([255, 0, 0, 0, 255, 0]+[0]*762)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_palette_indices_and_cv2_nearest_are_not_colors_or_pil_nearest():
    raw = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    actual = metric.decode_annotation(png(raw, True), 2, False)
    expected = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, np.asarray(Image.fromarray(raw).resize((224, 224), Image.Resampling.NEAREST)))


def test_reduce_zero_ignore_and_invalid_class():
    actual = metric.decode_annotation(png([[0, 1, 2, 255]]), 2, True)
    assert set(np.unique(actual)) == {0, 1, 255}
    with pytest.raises(ValueError, match="vocabulary"):
        metric.decode_annotation(png([[0, 3]]), 2, False)


@pytest.mark.parametrize("content", [b"", b"not an image", b"x"*(16*1024**2+1)])
def test_annotation_bad_or_oversized(content):
    with pytest.raises((ValueError, OSError)):
        metric.decode_annotation(content, 2, False)


def test_counts_ignore_and_pooled_native_union_zero():
    labels = np.zeros((224, 224), np.uint16)
    labels[:112] = 255
    prediction = np.zeros_like(labels)
    prediction[:112] = 1
    counted = metric.counts(prediction, labels, 3)
    assert counted.tolist() == [[25088, 0, 0], [25088, 0, 0]]
    assert metric.miou(counted) == 100
    assert np.isnan(metric.miou(np.zeros((2, 3), np.int64)))


def test_corruption_macro_is_not_globally_pooled_and_requires_all15():
    values = np.zeros((15, 3, 2, 1), dtype=np.int64)
    values[:, :, 1, 0] = 100
    values[0, :, 1, 0] = 10000
    values[0, :, 0, 0] = 10000
    points = metric.miou(values)
    result = metric.corruption_summary(list(CORRUPTIONS), points, np.tile(points, (5, 1, 1)))
    assert result["macro15"]["arms"][ARMS[0]]["mIoU"] == pytest.approx(100/15)
    assert metric.miou(values.sum(0))[0] != pytest.approx(100/15)
    assert result["C_versus_Confidence"] == dict(wins=0, ties=15, losses=0, undefined=0)
    with pytest.raises(ValueError, match="Full15"):
        metric.corruption_summary(list(CORRUPTIONS[:-1]), points[:-1], np.tile(points[:-1], (5, 1, 1)))


def test_bootstrap_matches_whole_episode_manual_and_chunk_independent():
    values = np.zeros((3, 2, 3, 2, 2), dtype=np.int64)
    values[..., 1, :] = 10
    values[0, ..., 0, :] = 2
    values[1, ..., 0, :] = 5
    values[2, ..., 0, :] = 9
    values[:, 1, 2, 0, :] += 1
    actual, info = metric.bootstrap(values, "ViT-B/16", "voc20", replicates=29, chunk=7)
    other, _ = metric.bootstrap(values, "ViT-B/16", "voc20", replicates=29, chunk=4)
    assert np.array_equal(actual, other)
    seed, key = metric.stream_seed("ViT-B/16", "voc20")
    rng = np.random.Generator(np.random.PCG64(seed))
    expected = np.stack([metric.miou(values[rng.integers(0, 3, size=3)].sum(0)) for _ in range(29)])
    assert np.array_equal(actual, expected)
    assert np.allclose(actual[:, 1, 2]-actual[:, 0, 2], 10)
    assert info["key"] == key
    assert seed != metric.stream_seed("ViT-L/14", "voc20")[0]


def test_undefined_draws_retained_not_dropped():
    values = np.zeros((2, 1, 3, 2, 1), dtype=np.int64)
    values[0, ..., 1, :] = 10
    samples, _ = metric.bootstrap(values, "ViT-B/16", "voc20", replicates=100)
    result = metric.interval(samples[:, 0, 0])
    assert result["CI95_pp"] is None and 0 < result["undefined_draws"] < 100
    assert result["total_draws"] == 100


def test_bootstrap_count_bound_and_invalid_counts():
    values = np.zeros((2, 1, 3, 2, 1), dtype=np.int64)
    values[..., 1, :] = 2**52
    with pytest.raises(ValueError, match="exact float64"):
        metric.bootstrap(values, "ViT-B/16", "voc20")
    values[..., 0, :] = -1
    with pytest.raises(ValueError, match="population"):
        metric.bootstrap(values, "ViT-B/16", "voc20")


def fixture(tmp_path):
    parent, suite = tmp_path/"parent", tmp_path/"score"
    parent.mkdir()
    suite.mkdir()
    rows, inventory = [], {}
    name = "synthetic"
    for i in range(3):
        label = tmp_path/f"gt{i}.png"
        label.write_bytes(png([[0, 1], [1, 0]], True))
        state = label.stat()
        rows.append(dict(official_id=f"source{i}", label=str(label),
            label_metadata=[state.st_size, state.st_mtime_ns],
            label_metadata_details=dict(size_bytes=state.st_size, mtime_ns=state.st_mtime_ns,
                device=state.st_dev, inode=state.st_ino, contents_read=False)))
    def save(relative, value):
        path = parent/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, np.ndarray):
            np.save(path, value, allow_pickle=False)
        else:
            path.write_text(json.dumps(value))
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for episode, start in enumerate(range(0, 3, 2)):
        for arm in ARMS:
            save(f"conditions/{name}/episodes/{episode:06d}.{arm}.json", dict(episode=episode, arm=arm,
                ordered_ids=[r["official_id"] for r in rows[start:start+2]], GT_opened=False))
    for arm in ARMS:
        for i in range(3):
            target = metric.decode_annotation((tmp_path/f"gt{i}.png").read_bytes(), 2, False)
            if arm == ARMS[1]:
                target = 1-target
            save(f"conditions/{name}/predictions/{arm}/{i:06d}.npy", target.astype(np.uint16))
    spec = dict(id=name, input_key="clean.fake", images=3,
                config=dict(batch_size=2, steps=1, trace_max_regions=2, trace_view_batch_size=4),
                vocabulary=dict(classes=["a", "b"], reduce_zero_label=False),
                backbone="ViT-B/16", dataset="fake", corruption="original", arms=list(ARMS))
    raw_rows = []
    for episode, start in enumerate(range(0, 3, 2)):
        n = min(2, 3-start)
        order = worker.prediction.runtime.prior.arm_order(ARMS, episode, "alternating")
        for position, arm in enumerate(order):
            row = dict(arm=arm, images=n, patches=n, steps=0, seconds=.1+episode*.2+position*.3,
                episode=episode, order_position=position, warmup=episode == 0,
                candidate_count=0, active_candidates=0, candidate_sha256=None if arm == ARMS[0] else "0"*64,
                **{k:0 for k in worker.prediction.runtime.COUNTERS})
            calls = 1 if arm == ARMS[0] else 2
            row.update(eval_forward_calls=calls, eval_forward_inputs=calls*n,
                       encoder_calls=calls, encoder_inputs=calls*n)
            worker.prediction.runtime.validate_cost(row, arm, spec["config"])
            raw_rows.append(row)
    costpath = parent/f"conditions/{name}/cost.jsonl"
    costpath.write_text("".join(json.dumps(r)+"\n" for r in raw_rows))
    inventory[str(costpath.relative_to(parent))] = hashlib.sha256(costpath.read_bytes()).hexdigest()
    legal = dict(parameter_tensors=1, parameter_elements=1, trainable_names=["backbone.visual.ln_pre.weight"],
                 visual_LN_names=["visual.ln_pre.weight"])
    save(f"conditions/{name}/initial_state.json", dict(legal_visual_LN=legal,
        frozen_parameter_sha256="a"*64, reset=dict(trainable_sha256="b"*64, optimizer_sha256="c"*64)))
    save(f"conditions/{name}/prediction_result.json", dict(condition=name, images=3, arms=list(ARMS), GT_opened=False,
        actual_work={a:{k:sum(r[k] for r in raw_rows if r["arm"] == a)
                          for k in worker.prediction.runtime.COUNTERS} for a in ARMS},
        raw_model_seconds={a:sum(r["seconds"] for r in raw_rows if r["arm"] == a) for a in ARMS},
        warmup_adjusted_throughput=worker.prediction.runtime.prior.timing_summary(raw_rows, ARMS),
        legal_visual_LN=legal, frozen_parameter_sha256="a"*64))
    bound = dict(parent=str(parent), specs=[spec], inputs={"clean.fake":rows},
                 seals={name:dict(artifact_sha256=inventory, resource_observation={})})
    return suite, bound, spec


def test_streaming_synthetic_full_controller(tmp_path, monkeypatch):
    suite, bound, spec = fixture(tmp_path)
    original = metric.bootstrap
    monkeypatch.setattr(metric, "bootstrap", lambda v, b, d: original(v, b, d, replicates=20))
    result = worker.execute(suite, bound)
    assert result["complete"] and result["image_method_evaluations"] == 9
    actual = json.loads((suite/"CONDITION_RESULTS.json").read_text())[spec["id"]]
    assert actual["C_minus_Confidence_pp"] == 100
    assert actual["paired_difference"]["CI95_pp"] == [100, 100]
    counts = [json.loads(s) for s in (suite/(spec["id"]+".episode_counts.jsonl")).read_text().splitlines()]
    assert [r["ordered_ids"] for r in counts] == [["source0", "source1"], ["source2"]]
    assert not (suite/"ANALYSIS_SEALED.json").exists()


@pytest.mark.parametrize("fault", ["hash", "dtype", "shape", "episode_order", "GT_alias"])
def test_streaming_fails_closed_on_payload_fault(tmp_path, fault):
    suite, bound, spec = fixture(tmp_path)
    relative = f"conditions/{spec['id']}/predictions/{ARMS[0]}/000000.npy"
    path = Path(bound["parent"])/relative
    inventory = bound["seals"][spec["id"]]["artifact_sha256"]
    if fault in ("dtype", "shape"):
        np.save(path, np.zeros((224, 224) if fault == "dtype" else (3, 3),
                            dtype=np.uint8 if fault == "dtype" else np.uint16))
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "episode_order":
        relative = f"conditions/{spec['id']}/episodes/000000.{ARMS[0]}.json"
        path = Path(bound["parent"])/relative
        value = json.loads(path.read_text())
        value["ordered_ids"].reverse()
        path.write_text(json.dumps(value))
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "GT_alias":
        path.unlink()
        os.link(bound["inputs"]["clean.fake"][0]["label"], path)
    else:
        path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        worker.score_condition(suite, bound, spec, dict(bytes=0), {})
    assert not (suite/"ANALYSIS_SEALED.json").exists()


def test_prepare_rejects_running_parent_before_any_new_files_or_GT(tmp_path, monkeypatch):
    def blocked():
        raise ValueError("still running")
    monkeypatch.setattr(worker, "parent_terminal", blocked)
    monkeypatch.setattr(worker, "bind_parent", lambda: pytest.fail("must not bind while running"))
    with pytest.raises(ValueError, match="still running"):
        worker.prepare(tmp_path/"new")
    assert not (tmp_path/"new").exists()


def test_terminal_requires_exact_successful_invocation(monkeypatch):
    values = dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
        ExecMainPID="2043872", ExecMainCode="1", ExecMainStatus="0", Result="success",
        InvocationID="2f14c1b37a0741f39500d48dae7ff353")
    def fake(*args, **kwargs):
        return SimpleNamespace(stdout="\n".join(f"{k}={v}" for k, v in values.items()))
    monkeypatch.setattr(worker.subprocess, "run", fake)
    assert worker.parent_terminal() == values
    values["SubState"] = "running"
    with pytest.raises(ValueError):
        worker.parent_terminal()
    values["SubState"], values["InvocationID"] = "exited", "wrong"
    with pytest.raises(ValueError):
        worker.parent_terminal()


def test_metadata_hardlinked_to_GT_rejected_without_open(tmp_path, monkeypatch):
    label = tmp_path/"GT.png"
    label.write_bytes(png([[0]]))
    report = tmp_path/"report"
    os.link(label, report)
    state = label.stat()
    monkeypatch.setattr(worker, "stable_bytes", lambda *a, **k: pytest.fail("GT alias opened"))
    with pytest.raises(ValueError, match="aliases"):
        worker.metadata(report, forbidden={(state.st_dev, state.st_ino)})


def admission_fixture(tmp_path, monkeypatch):
    suite, bound, _ = fixture(tmp_path)
    (suite/"runtime").mkdir()
    (suite/"SCORE_INPUTS.json").write_text(json.dumps(bound))
    (suite/"PROTOCOL.json").write_text("{}")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    prepared = dict(root=str(suite), parent_prepared_sha256=worker.PARENT_SHA, runtime_hashes={},
        inputs_sha256=digest(suite/"SCORE_INPUTS.json"), protocol_sha256=digest(suite/"PROTOCOL.json"))
    (suite/"SUITE_PREPARED.json").write_text(json.dumps(prepared))
    report = suite/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic independent CPU audit; not a real artifact admission")
    expected = digest(suite/"SUITE_PREPARED.json")
    audit = dict(decision="ADMITTED", suite_prepared_sha256=expected, independent_reviewer="synthetic",
        CPU_GT_scoring_allowed=True, G_primary_GT_scoring_allowed=True,
        report_path=str(report), report_sha256=digest(report))
    (suite/"ADMITTED.json").write_text(json.dumps(audit))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(worker, "__file__", str(suite/"runtime"/"test.py"))
    monkeypatch.setattr(worker, "parent_terminal", lambda: {})
    monkeypatch.setattr(worker, "bind_parent", lambda: bound)
    monkeypatch.setattr(worker.prediction, "runtime_imports", lambda _: None)
    return suite, bound, expected, audit


def test_separate_synthetic_CPU_admission_binding(tmp_path, monkeypatch):
    suite, _, expected, _ = admission_fixture(tmp_path, monkeypatch)
    admitted, _, _, binding = worker.admission(suite, expected)
    assert admitted == suite
    worker.verify_audit_binding(binding)
    (suite/"INDEPENDENT_REPORT.md").write_text("modified after admission")
    with pytest.raises(ValueError):
        worker.verify_audit_binding(binding)


@pytest.mark.parametrize("fault", ["prediction_permission", "wrong_suite", "report_GT_alias", "receipt_GT_alias"])
def test_separate_admission_rejects_scope_or_GT_alias(tmp_path, monkeypatch, fault):
    suite, bound, expected, audit = admission_fixture(tmp_path, monkeypatch)
    if fault == "prediction_permission":
        audit.pop("G_primary_GT_scoring_allowed")
        audit["G_primary_predictions_allowed"] = True
    elif fault == "wrong_suite":
        audit["suite_prepared_sha256"] = "0"*64
    if fault.endswith("GT_alias"):
        target = suite/("INDEPENDENT_REPORT.md" if fault == "report_GT_alias" else "ADMITTED.json")
        target.unlink()
        os.link(bound["inputs"]["clean.fake"][0]["label"], target)
        real_read = worker.stable_bytes
        def guarded(path, *args, **kwargs):
            if Path(path) == target:
                pytest.fail("GT-alias audit payload opened")
            return real_read(path, *args, **kwargs)
        monkeypatch.setattr(worker, "stable_bytes", guarded)
    else:
        (suite/"ADMITTED.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        worker.admission(suite, expected)


@pytest.mark.parametrize("fault", ["child_hash", "progress", "seal_publish"])
def test_final_failures_do_not_leave_analysis_seal(tmp_path, monkeypatch, fault):
    suite = tmp_path/"score"
    suite.mkdir()
    (suite/"runtime").mkdir()
    child = suite/"CONDITION_RESULTS.json"
    child.write_text("{}")
    digest = hashlib.sha256(child.read_bytes()).hexdigest()
    prepared = dict(runtime_hashes={}, inputs_sha256="i", protocol_sha256="p")
    inputs = dict(barrier_sha256="b", inputs={})
    result = dict(conditions=19, cells=57, image_method_evaluations=104214,
                  artifact_sha256={child.name:"0"*64 if fault == "child_hash" else digest})
    monkeypatch.setattr(worker, "admission", lambda *a: (suite, prepared, inputs, {}))
    monkeypatch.setattr(worker, "verify_audit_binding", lambda _: None)
    monkeypatch.setattr(worker, "execute", lambda *a: result)
    monkeypatch.setattr(worker, "bind_parent", lambda: inputs)
    monkeypatch.setattr(worker, "metadata", lambda *a: ({}, ""))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    def fail(*args, **kwargs):
        raise OSError("synthetic finalization failure")
    if fault == "progress":
        monkeypatch.setattr(worker, "write_json", fail)
    if fault == "seal_publish":
        monkeypatch.setattr(worker, "publish_seal", fail)
    with pytest.raises((ValueError, OSError)):
        worker.run(suite, "expected")
    assert (suite/"FAILED.json").exists()
    assert not (suite/"ANALYSIS_SEALED.json").exists()


def identity(path):
    state = path.stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


@pytest.mark.parametrize("target", ["SUITE_PREPARED.json", "INPUTS.json"])
def test_parent_bootstrap_identity_rejects_GT_before_first_read(tmp_path, monkeypatch, target):
    parent = tmp_path/"parent"
    parent.mkdir()
    inp = parent/"INPUTS.json"
    inp.write_text("{}")
    prepared = parent/"SUITE_PREPARED.json"
    prepared.write_text(json.dumps(dict(root=str(parent), schema_version=1,
        inputs_sha256=hashlib.sha256(inp.read_bytes()).hexdigest())))
    pins = {n:identity(parent/n) for n in ("SUITE_PREPARED.json", "INPUTS.json")}
    monkeypatch.setattr(worker, "PARENT", parent)
    monkeypatch.setattr(worker, "PARENT_SHA", hashlib.sha256(prepared.read_bytes()).hexdigest())
    monkeypatch.setattr(worker, "PARENT_METADATA_IDENTITIES", pins)
    monkeypatch.setattr(worker, "parent_terminal", lambda: {})
    gt = tmp_path/"GT.png"
    gt.write_bytes(png([[0]]))
    path = parent/target
    path.unlink()
    os.link(gt, path)
    original = worker.stable_bytes
    def check_open(path_arg, *args, **kwargs):
        if Path(path_arg) == path:
            pytest.fail("initial GT alias reached byte reader")
        return original(path_arg, *args, **kwargs)
    monkeypatch.setattr(worker, "stable_bytes", check_open)
    with pytest.raises(ValueError, match="Bootstrap metadata identity"):
        worker.bind_parent()


@pytest.mark.parametrize("target", ["SUITE_PREPARED.json", "SCORE_INPUTS.json"])
def test_score_bootstrap_uses_trusted_parent_GT_registry_before_read(tmp_path, monkeypatch, target):
    suite, bound, expected, _ = admission_fixture(tmp_path, monkeypatch)
    path = suite/target
    path.unlink()
    os.link(bound["inputs"]["clean.fake"][0]["label"], path)
    original = worker.stable_bytes
    def check_open(path_arg, *args, **kwargs):
        if Path(path_arg) == path:
            pytest.fail("score bootstrap GT alias reached byte reader")
        return original(path_arg, *args, **kwargs)
    monkeypatch.setattr(worker, "stable_bytes", check_open)
    with pytest.raises(ValueError, match="aliases"):
        worker.admission(suite, expected)


@pytest.mark.parametrize("extra_is_GT", [False, True])
def test_actual_runtime_members_and_inodes_rejected_before_any_hash(tmp_path, monkeypatch, extra_is_GT):
    root = tmp_path/"runtime"
    root.mkdir()
    (root/"safe.py").write_text("safe=1")
    gt = tmp_path/"GT.png"
    gt.write_bytes(png([[0]]))
    if extra_is_GT:
        os.link(gt, root/"extra.py")
    else:
        (root/"extra.py").write_text("unexpected=1")
    state = gt.stat()
    monkeypatch.setattr(worker, "stable_bytes", lambda *a, **k: pytest.fail("hashed before member/inode preflight"))
    with pytest.raises(ValueError):
        worker.guarded_inventory(root, {"safe.py":"unused"}, {(state.st_dev, state.st_ino)})


def test_guarded_inventory_expected_source_swap_cannot_open_GT(tmp_path, monkeypatch):
    root = tmp_path/"runtime"
    root.mkdir()
    source = root/"safe.py"
    source.write_text("safe=1")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    gt = tmp_path/"GT.png"
    gt.write_bytes(png([[0]]))
    state = gt.stat()
    original = worker.stable_bytes
    def swap(path, *args, **kwargs):
        Path(path).unlink()
        os.link(gt, path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(worker, "stable_bytes", swap)
    with pytest.raises(ValueError, match="Frozen opened-file identity"):
        worker.guarded_inventory(root, {"safe.py":digest}, {(state.st_dev, state.st_ino)})


def test_snapshot_preflights_every_source_before_copy_and_supports_empty_files(tmp_path):
    source = tmp_path/"source"
    for tree in ("trace_ovss", "tools", "tests"):
        (source/tree).mkdir(parents=True)
        (source/tree/"__init__.py").write_bytes(b"")
    (source/"tools"/"safe.py").write_text("safe=1")
    target = tmp_path/"snapshot"
    inventory = worker.guarded_snapshot(source, target, set())
    assert len(inventory) == 4
    worker.guarded_inventory(target, inventory, set())
    gt = tmp_path/"GT.png"
    gt.write_bytes(png([[0]]))
    os.link(gt, source/"tests"/"extra.py")
    state = gt.stat()
    target2 = tmp_path/"refused_snapshot"
    with pytest.raises(ValueError, match="aliases"):
        worker.guarded_snapshot(source, target2, {(state.st_dev, state.st_ino)})
    assert not target2.exists()


@pytest.mark.parametrize("artifact", ["initial_state.json", "cost.jsonl"])
@pytest.mark.parametrize("fault", ["missing", "hash", "GT_alias"])
def test_raw_cost_and_initial_state_must_exist_match_and_not_alias(tmp_path, artifact, fault):
    suite, bound, spec = fixture(tmp_path)
    path = Path(bound["parent"])/f"conditions/{spec['id']}/{artifact}"
    if fault == "missing":
        path.unlink()
    elif fault == "hash":
        path.write_text("corrupt")
    else:
        path.unlink()
        os.link(bound["inputs"]["clean.fake"][0]["label"], path)
    with pytest.raises((ValueError, OSError)):
        worker.score_condition(suite, bound, spec, dict(bytes=0), {})
    assert not list(suite.glob("*.episode_counts.jsonl"))


@pytest.mark.parametrize("field", ["raw_model_seconds", "actual_work", "warmup_adjusted_throughput", "legal_visual_LN"])
def test_cost_summary_reconciles_raw_time_work_and_initial_state(tmp_path, field):
    suite, bound, spec = fixture(tmp_path)
    relative = f"conditions/{spec['id']}/prediction_result.json"
    path = Path(bound["parent"])/relative
    value = json.loads(path.read_text())
    value[field] = {}
    path.write_text(json.dumps(value))
    bound["seals"][spec["id"]]["artifact_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        worker.score_condition(suite, bound, spec, dict(bytes=0), {})
    assert not list(suite.glob("*.episode_counts.jsonl"))


def test_colored_RGB_annotations_rejected_not_silently_red_channel():
    with pytest.raises(ValueError, match="Color-coded"):
        metric.decode_annotation(png([[[1, 77, 200], [0, 9, 8]]]), 2, False)
    result = metric.decode_annotation(png([[[1, 1, 1], [0, 0, 0]]]), 2, False)
    assert set(np.unique(result)) == {0, 1}


def test_actual_DenseCLIP_legal_names_have_registration_prefix_and_bind_before_GT(tmp_path):
    from test_generalization_runtime import method
    legal = worker.prediction.runtime.legal_parameters(method())
    assert legal["trainable_names"] != legal["visual_LN_names"]
    worker.validate_visual_LN_names(legal)
    suite,bound,spec = fixture(tmp_path)
    parent = Path(bound["parent"])
    inventory = bound["seals"][spec["id"]]["artifact_sha256"]
    for name in ("initial_state.json","prediction_result.json"):
        relative = f"conditions/{spec['id']}/{name}"
        path = parent/relative
        value = json.loads(path.read_text())
        value["legal_visual_LN"] = legal
        path.write_text(json.dumps(value))
        inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = worker.condition_cost(parent,spec,inventory,worker.label_inodes(bound["inputs"]))
    assert summary["legal_visual_LN"] == legal
    assert not list(suite.glob("*.episode_counts.jsonl"))


@pytest.mark.parametrize("fault",["unprefixed","foreign","missing","duplicate","order","nonvisual","count"])
def test_visual_LN_mapping_is_strict_not_arbitrary_prefix_or_membership_relaxation(fault):
    visual = ["visual.ln_pre.bias","visual.ln_pre.weight"]
    legal = dict(parameter_tensors=2,trainable_names=["backbone."+n for n in visual],visual_LN_names=visual)
    if fault == "unprefixed":
        legal["trainable_names"] = visual.copy()
    elif fault == "foreign":
        legal["trainable_names"][0] = "text_encoder.visual.ln_pre.bias"
    elif fault == "missing":
        legal["trainable_names"].pop()
    elif fault == "duplicate":
        legal["trainable_names"][1] = legal["trainable_names"][0]
        legal["visual_LN_names"][1] = legal["visual_LN_names"][0]
    elif fault == "order":
        legal["trainable_names"].reverse()
    elif fault == "nonvisual":
        legal["visual_LN_names"][0] = "text.ln_pre.bias"
        legal["trainable_names"][0] = "backbone.text.ln_pre.bias"
    else:
        legal["parameter_tensors"] = True
    with pytest.raises(ValueError,match="visual-LN membership"):
        worker.validate_visual_LN_names(legal)


def test_new_prepared_metadata_hashes_known_bytes_without_any_read_reopen(tmp_path, monkeypatch):
    original = Path.open
    def no_read(path, mode="r", *args, **kwargs):
        assert mode == "xb", "Reopened newly written metadata for reading"
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_read)
    path = tmp_path/"prepared.json"
    value = dict(synthetic=True)
    actual = worker.write_prepared_json(path, value)
    expected = hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()).hexdigest()
    assert actual == expected
