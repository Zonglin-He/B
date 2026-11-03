"""Synthetic controller integration, NOT actual artifact/origin/service proof.

The two-row fixture stubs service queries, source reconstruction, S5 population,
original episode grouping, native binding,
the full prediction stat graph, cost semantics, and new prediction/GT decoding.
Actual metadata/audit hashes, protected writers, compact S5 streaming, the real
10000-draw estimator and run finalization remain exercised. No actual results
or payloads are accessed. Separate tests exercise real small stat graphs.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.evolution_fifty import curve_score_worker as worker


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return worker.protected.write_prepared_json(path, value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins(root):
    return dict(root=str(root), prepared_sha256="1"*64, unit="trace-synthetic-g03.service", pid=123,
                invocation="a"*32, global_sha256="2"*64, started_sha256="3"*64,
                gate_sha256="4"*64, admission_sha256="5"*64, report_sha256="6"*64)


def permit(root, prepared, flag, **extra):
    report = root/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic fixture permission only; never actual admission.\n")
    value = dict(decision="ADMITTED", suite_prepared_sha256=prepared, independent_reviewer="SyntheticCPU",
                 report_path=str(report), report_sha256=digest(report), **{flag: True}, **extra)
    admission = save(root/"ADMITTED.json", value)
    ids = [worker.output.file_pin(path, set(), 4*1024**2, single=True)[:4] for path in (root/"ADMITTED.json", report)]
    return dict(admission_path=str(root/"ADMITTED.json"), admission_sha256=admission, admission_identity=ids[0],
                report_path=str(report), report_sha256=digest(report), report_identity=ids[1], independent_reviewer="SyntheticCPU")


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    parent, old, g, root = (tmp_path/name for name in ("new_predictions", "old_score", "G1954", "sources"))
    for directory in (parent, old, g):
        (directory/"runtime").mkdir(parents=True)
        (directory/"runtime"/"fixture.py").write_text("# Source role only, never imported.\n")
    for tree in ("trace_ovss", "tools", "tests"):
        (root/tree).mkdir(parents=True)
        (root/tree/"fixture.py").write_text("# Unexecuted synthetic snapshot member.\n")
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(worker, "S5_ROOT", old)
    monkeypatch.setattr(worker.protected, "PARENT", g)
    monkeypatch.setattr(worker, "role_roots", lambda p: ((root, True), (p/"runtime", False), (old/"runtime", False), (g/"runtime", False)))
    events, clean = [], []
    for i in range(2):
        image, label = tmp_path/f"image{i}.fixture", tmp_path/f"label{i}.fixture"
        image.write_bytes(b"Synthetic non-image source metadata fixture")
        label.write_bytes(b"Synthetic non-GT annotation fixture")
        st = label.stat()
        details = dict(device=st.st_dev, inode=st.st_ino, size_bytes=st.st_size, mtime_ns=st.st_mtime_ns, contents_read=False)
        clean.append(dict(official_id=f"synthetic:{i}", image=str(image), image_sha256=digest(image), label=str(label),
                          label_metadata=[st.st_size, st.st_mtime_ns], label_metadata_details=details))
    anchor = dict(id="naclip.ViT-L-14.voc20.gaussian_noise.s5", dataset="voc20", model_type="naclip", backbone="ViT-L/14",
        images=1449, arms=list(worker.analysis.ARMS), corruption="gaussian_noise", severity=5,
        input_key="corruption.gaussian_noise", config=dict(batch_size=1, resize=[224, 224], readout="same_TRACE_readout"),
        vocabulary=dict(classes=list(range(20)), reduce_zero_label=True), vision_outputs=[-1], checkpoint=dict(sha256="7"*64))
    rows, specs = {}, []
    for condition in worker.prediction.curve.conditions():
        spec = copy.deepcopy(anchor)
        spec.update(id="naclip.ViT-L-14.voc20."+condition["id"], input_key=condition["id"],
                    corruption=condition["corruption"], severity=condition["severity"])
        specs.append(spec)
        rows[spec["input_key"]] = copy.deepcopy(clean)
    native = dict(rows={"clean.voc20": clean}, annotation_metadata={r["label"]: r["label_metadata_details"] for r in clean})
    source = dict(native=native, rows=rows, specs=specs, anchor=anchor, input_seal_sha256="8"*64,
                  parent_native_hashes={f"synthetic_native_{i}": "9"*64 for i in range(68)})
    real_s5_specs = worker.s5_specs
    # This is a two-row orchestration fixture, not the formal 1449-row binding.
    monkeypatch.setattr(worker, "s5_specs", lambda s: [dict(spec, images=len(clean)) for spec in real_s5_specs(s)])
    old_specs = worker.s5_specs(source)
    for spec in old_specs:
        native["rows"][spec["input_key"]] = copy.deepcopy(clean)
    ordered = [[r["official_id"]] for r in clean]
    monkeypatch.setattr(worker, "shared_order", lambda s: copy.deepcopy(ordered))
    monkeypatch.setattr(worker.prediction, "source_inputs", lambda: events.append("source") or copy.deepcopy(source))
    def service(unit, pid, invocation):
        events.append(("service", unit, pid, invocation))
        return dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0", ExecMainPID=str(pid),
                    ExecMainCode="1", ExecMainStatus="0", Result="success", InvocationID=invocation)
    monkeypatch.setattr(worker.prediction, "service", service)
    monkeypatch.setattr(worker.shared, "native_binding", lambda hashes, forbidden: events.append(("native", len(hashes))))
    monkeypatch.setattr(worker.prediction, "prerequisite_bindings", lambda p, f: dict(synthetic_prior=True))
    monkeypatch.setattr(worker.prediction, "verify_gates", lambda *a: events.append("gates"))
    monkeypatch.setattr(worker.prediction.general, "runtime_imports", lambda root: events.append("origin_stub"))
    monkeypatch.setattr(worker.protected.prediction, "validate_inputs", lambda *a: None)
    monkeypatch.setattr(worker.prediction.curve, "validate_S5_reuse", lambda *a: dict(valid=True))
    monkeypatch.setattr(worker.protected, "condition_cost", lambda p, s, i, f: dict(raw_model_seconds={a: 1. for a in worker.analysis.ARMS}, synthetic_cost=True))
    monkeypatch.setattr(worker.payload, "cost_binding", lambda *a: (dict(raw_model_seconds={arm: 2. for arm in worker.analysis.ARMS}), []))
    graph = {}
    for spec in specs:
        relative = f"conditions/{spec['id']}/predictions/image_C/000001.npy"
        path = parent/relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"Unopened synthetic prediction placeholder")
        graph[relative] = worker.output.file_pin(path, set(), 2*1024**2, single=True)[:4]
    def graph_now(p, selected, forbidden):
        assert len(selected) == 10
        events.append("complete_graph_stub")
        return {name: worker.output.file_pin(p/name, forbidden, 2*1024**2, single=True)[:4] for name in graph}
    monkeypatch.setattr(worker, "prediction_graph", graph_now)
    monkeypatch.setattr(worker.payload, "preflight_payloads", lambda p, s, seals, source: graph_now(p, s, worker.payload.data_exclusions(source)))
    all_old_specs = old_specs + [dict(id=f"unselected.synthetic.{i}") for i in range(14)]
    condition_sha = save(g/"CONDITIONS.json", all_old_specs)
    g_sha = save(g/"SUITE_PREPARED.json", dict(schema_version=1, root=str(g), conditions_sha256=condition_sha))
    monkeypatch.setattr(worker.protected, "PARENT_SHA", g_sha)
    st = (g/"SUITE_PREPARED.json").stat()
    monkeypatch.setattr(worker.protected, "PARENT_METADATA_IDENTITIES", {"SUITE_PREPARED.json": (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)})
    old_children, old_child_hashes = {}, {}
    for spec in old_specs:
        directory = g/"conditions"/spec["id"]
        inventory = {}
        for name in ("cost.jsonl", "initial_state.json", "prediction_result.json"):
            path = directory/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Synthetic cost/reset metadata\n")
            inventory[str(path.relative_to(g))] = digest(path)
        child = dict(complete=True, GT_opened=False, condition=spec, suite_prepared_sha256=g_sha,
                     artifact_sha256=inventory, resource_observation=dict(synthetic=True))
        old_children[spec["id"]] = child
        old_child_hashes[spec["id"]] = save(directory/"PREDICTIONS_SEALED.json", child)
    old_child_hashes.update({s["id"]: "a"*64 for s in all_old_specs[5:]})
    g_barrier = save(g/"ALL_G_PREDICTIONS_SEALED.json", dict(complete=True, GT_opened=False, cells=57,
        image_method_evaluations=104214, suite_prepared_sha256=g_sha, condition_seals=old_child_hashes))
    old_inputs = dict(parent=str(g), prepared_sha256=g_sha, inputs=native["rows"], specs=all_old_specs,
                      seals=old_children, barrier_sha256=g_barrier)
    old_inputs_sha = save(old/"SCORE_INPUTS.json", old_inputs)
    old_protocol_sha = save(old/"PROTOCOL.json", dict(synthetic_geometry="G_OpenCV_nearest"))
    old_runtime = {"fixture.py": digest(old/"runtime"/"fixture.py")}
    old_prepared = save(old/"SUITE_PREPARED.json", dict(schema_version=1, root=str(old), inputs_sha256=old_inputs_sha,
        protocol_sha256=old_protocol_sha, parent_prepared_sha256=g_sha, runtime_hashes=old_runtime))
    old_audit = permit(old, old_prepared, "CPU_GT_scoring_allowed", G_primary_GT_scoring_allowed=True)
    old_start = save(old/"STARTED.json", dict(pid=worker.S5_PID, suite_prepared_sha256=old_prepared,
        audit=old_audit, parent_barrier_sha256=g_barrier, GT_opened=False))
    counts_artifacts = {}
    for spec in old_specs:
        path = old/(spec["id"]+".episode_counts.jsonl")
        with path.open("xb") as stream:
            for e, row in enumerate(clean):
                counts = np.ones((3, 2, 20), dtype=np.int64)
                counts[:, 1] = 2
                value = dict(episode=e, ordered_ids=[row["official_id"]],
                    labels=[dict(source_id=row["official_id"], label=row["label"], sha256=digest(Path(row["label"])))],
                    I_U_by_arm=counts.tolist())
                stream.write((json.dumps(value)+"\n").encode())
        counts_artifacts[path.name] = digest(path)
    old_seal = save(old/"ANALYSIS_SEALED.json", dict(complete=True, GT_opened=True, new_GPU=False,
        suite_prepared_sha256=old_prepared, conditions=19, cells=57, image_method_evaluations=104214,
        parent_barrier_sha256=g_barrier, prospective_analysis_sha256=worker.protected.ANALYSIS_SHA,
        audit_binding=old_audit, started_sha256=old_start, artifact_sha256=counts_artifacts))
    for key, value in dict(S5_PREPARED=old_prepared, S5_INPUTS=old_inputs_sha, S5_PROTOCOL=old_protocol_sha,
                           S5_SEAL=old_seal, S5_ADMISSION=old_audit["admission_sha256"], S5_REPORT=old_audit["report_sha256"]).items():
        monkeypatch.setattr(worker, key, value)
    prospective = tmp_path/"analysis.json"
    monkeypatch.setattr(worker, "ANALYSIS", prospective)
    monkeypatch.setattr(worker, "ANALYSIS_SHA", save(prospective, worker.analysis.protocol()))
    parent_inputs = save(parent/"INPUTS.json", source)
    prior_pins = dict(pid=456, invocation="b"*32, global_sha256="c"*64)
    parent_protocol = save(parent/"PROTOCOL.json", worker.prediction.protocol(source, prior_pins))
    parent_prepared = save(parent/"SUITE_PREPARED.json", dict(schema_version=1, root=str(parent), inputs_sha256=parent_inputs,
        protocol_sha256=parent_protocol, prior_V01=prior_pins, prior_binding=dict(synthetic_prior=True),
        runtime_hashes={"fixture.py": digest(parent/"runtime"/"fixture.py")}))
    parent_audit = permit(parent, parent_prepared, "G03_predictions_allowed", GT_scoring_allowed=False)
    parent_start = save(parent/"STARTED.json", dict(suite_prepared_sha256=parent_prepared, pid=123, GT_opened=False,
        audit=parent_audit, prior=dict(synthetic_prior=True)))
    children = {}
    for spec in specs:
        inventory = {name: digest(parent/name) for name in graph if name.startswith("conditions/"+spec["id"]+"/")}
        child = dict(complete=True, GT_opened=False, condition=spec, suite_prepared_sha256=parent_prepared,
            images=1449, image_method_evaluations=4347, artifact_sha256=inventory, resource_observation=dict(synthetic=True))
        children[spec["id"]] = save(parent/"conditions"/spec["id"]/"PREDICTIONS_SEALED.json", child)
    gate_sha = "d"*64
    barrier = save(parent/"ALL_G_PREDICTIONS_SEALED.json", dict(complete=True, GT_opened=False,
        suite_prepared_sha256=parent_prepared, cells=30, image_method_evaluations=43470, started_sha256=parent_start,
        gate_binding=dict(barrier_sha256=gate_sha), condition_seals=children, input_seal_sha256=source["input_seal_sha256"],
        prior_binding=dict(synthetic_prior=True), audit_binding=parent_audit))
    parent_pins = dict(pins(parent), prepared_sha256=parent_prepared, global_sha256=barrier,
        started_sha256=parent_start, gate_sha256=gate_sha, admission_sha256=parent_audit["admission_sha256"],
        report_sha256=parent_audit["report_sha256"])
    def score(suite, bound, spec, budget, labels, progress):
        events.append(("new_GT_decode_stub", spec["id"]))
        assert len(labels) == 2
        path = suite/(spec["id"]+".episode_counts.jsonl")
        sha, _ = worker.write_output(suite, path.name, b'{"synthetic_new_counts":true}\n', budget)
        progress(dict(phase="synthetic_score", GT_opened=True))
        values = np.ones((2, 3, 2, 20), dtype=np.int64)
        values[:, :, 1] = 3
        return values, copy.deepcopy(ordered), dict(raw_model_seconds={a: 2. for a in worker.analysis.ARMS}), path, sha
    monkeypatch.setattr(worker.payload, "score_condition", score)
    return dict(parent=parent, old=old, g=g, source=source, ordered=ordered, pins=parent_pins, events=events, root=root)


def admitted(stage, tmp_path, monkeypatch):
    suite = tmp_path/"new_score"
    sha = worker.prepare(suite, stage["pins"])
    permit(suite, sha, "CPU_GT_scoring_allowed", G03_GT_scoring_allowed=True, GPU_allowed=False)
    # Explicit fixture-origin stub; no genuine-runtime proof is claimed.
    monkeypatch.setattr(worker, "__file__", str(suite/"runtime"/"tools"/"fixture.py"))
    return suite, sha


def test_canonical_S5_ID_contract_has_naclip_prefix():
    source = dict(anchor=dict(id="unused", input_key="unused", corruption="unused", severity=1))
    specs = worker.s5_specs(source)
    assert [s["id"] for s in specs] == [f"naclip.ViT-L-14.voc20.{k}.s5" for k in worker.analysis.CURVE_TYPES]
    assert [s["input_key"] for s in specs] == ["corruption."+k for k in worker.analysis.CURVE_TYPES]


def test_parent_success_precedes_source_and_no_actual_hook(stage, monkeypatch):
    monkeypatch.setattr(worker.prediction, "forbid_annotations", lambda *a: pytest.fail("Scoring installed prediction GT hook"))
    bound = worker.bind_parent(stage["pins"])
    assert stage["events"][0][0] == "service" and stage["events"][1][0] == "service" and stage["events"][2] == "source"
    assert len(bound["specs"]) == 10 and len(bound["reused"]) == 5
    assert all(v["cost"]["synthetic_cost"] for v in bound["reused"].values())
    assert not any(isinstance(e, tuple) and e[0] == "new_GT_decode_stub" for e in stage["events"])


@pytest.mark.parametrize("fault", ["root", "prepared_sha256", "unit", "pid", "invocation", "global_sha256", "started_sha256", "gate_sha256", "admission_sha256", "report_sha256", "missing"])
def test_explicit_operator_pin_shapes_refuse_before_source(tmp_path, monkeypatch, fault):
    value = pins(tmp_path)
    if fault == "missing":value.pop("unit")
    else:value[fault] = {"root": "relative", "unit": "--invalid", "pid": True}.get(fault, "bad")
    monkeypatch.setattr(worker.prediction, "source_inputs", lambda: pytest.fail("Bad pins reached source metadata"))
    with pytest.raises(ValueError):worker.bind_parent(value)


@pytest.mark.parametrize("which", ["new", "old"])
def test_terminal_failure_precedes_source(stage, monkeypatch, which):
    def terminal(*a):raise ValueError("Synthetic retained terminal mismatch")
    monkeypatch.setattr(worker, "parent_terminal" if which == "new" else "reused_terminal", terminal)
    monkeypatch.setattr(worker.prediction, "source_inputs", lambda: pytest.fail("Terminal failure reached source"))
    with pytest.raises(ValueError, match="terminal"):worker.bind_parent(stage["pins"])


def test_synthetic_prepare_admit_full_bootstrap_and_global_last(stage, tmp_path, monkeypatch):
    suite, sha = admitted(stage, tmp_path, monkeypatch)
    old_hashes = {p.name: digest(p) for p in stage["old"].glob("*.jsonl")}
    original = worker.publish_seal
    def publish(path, value):
        assert path.name == "ANALYSIS_SEALED.json" and (suite/"queue_state.json").exists()
        assert value["new_prediction_cells"] == 30 and value["cells"] == 45
        original(path, value)
    monkeypatch.setattr(worker, "publish_seal", publish)
    worker.run(suite, sha, stage["pins"])
    result = json.loads((suite/"CURVE_RESULTS.json").read_bytes())
    assert result["bootstrap"]["replicates"] == 10000 and len(result["curves"]) == 5 and len(result["costs"]) == 15
    assert "no equal-budget claim" in result["cost_scope"]
    assert sum(v["provenance"].startswith("Reused") for v in result["costs"].values()) == 5
    seal = json.loads((suite/"ANALYSIS_SEALED.json").read_bytes())
    assert len(seal["artifact_sha256"]) == 13 and not (suite/"FAILED.json").exists()
    for name, expected in seal["artifact_sha256"].items():assert digest(suite/name) == expected
    assert old_hashes == {p.name: digest(p) for p in stage["old"].glob("*.jsonl")}
    with pytest.raises(ValueError, match="resume/retry"):worker.run(suite, sha, stage["pins"])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("field,wrong", [("CPU_GT_scoring_allowed", False), ("G03_GT_scoring_allowed", False), ("GPU_allowed", True)])
def test_new_both_GT_permissions_and_no_GPU_required(stage, tmp_path, monkeypatch, field, wrong):
    suite, sha = admitted(stage, tmp_path, monkeypatch)
    path = suite/"ADMITTED.json"
    value = json.loads(path.read_bytes());value[field] = wrong
    path.unlink();save(path, value)
    with pytest.raises(ValueError, match="CPU_GT"):worker.admission(suite, sha, stage["pins"])
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("target", ["new_global", "S5_global", "new_start", "S5_start", "native68", "analysis", "last_payload", "last_reused"])
def test_upstream_drift_before_any_new_GT(stage, monkeypatch, target):
    if target == "native68":
        source = copy.deepcopy(stage["source"]);source["parent_native_hashes"].pop(next(iter(source["parent_native_hashes"])))
        monkeypatch.setattr(worker.prediction, "source_inputs", lambda: source)
    elif target == "last_payload":next(reversed(list(stage["parent"].rglob("*.npy")))).unlink()
    elif target == "last_reused":sorted(stage["old"].glob("*.episode_counts.jsonl"))[-1].unlink()
    else:
        path = {"new_global": stage["parent"]/"ALL_G_PREDICTIONS_SEALED.json", "S5_global": stage["old"]/"ANALYSIS_SEALED.json",
                "new_start": stage["parent"]/"STARTED.json", "S5_start": stage["old"]/"STARTED.json", "analysis": worker.ANALYSIS}[target]
        with path.open("ab") as stream:stream.write(b"drift")
    with pytest.raises((ValueError, FileNotFoundError)):worker.bind_parent(stage["pins"])
    assert not any(isinstance(e, tuple) and e[0] == "new_GT_decode_stub" for e in stage["events"])


@pytest.mark.parametrize("fault", ["score", "last_reused", "final_parent", "progress", "publish", "audit", "output"])
def test_owned_failures_no_global_no_resume(stage, tmp_path, monkeypatch, fault):
    suite, sha = admitted(stage, tmp_path, monkeypatch)
    def fail(*a, **k):raise OSError("Synthetic ordinary IO error")
    if fault == "score":monkeypatch.setattr(worker.payload, "score_condition", fail)
    elif fault == "progress":monkeypatch.setattr(worker.shared, "write_json", fail)
    elif fault == "publish":monkeypatch.setattr(worker, "publish_seal", fail)
    else:
        execute = worker.execute
        def changed(*a):
            if fault == "last_reused":sorted(stage["old"].glob("*.episode_counts.jsonl"))[-1].unlink()
            result = execute(*a)
            if fault == "final_parent":
                with (stage["parent"]/"INPUTS.json").open("ab") as stream:stream.write(b"drift")
            elif fault == "audit":
                with (suite/"INDEPENDENT_REPORT.md").open("ab") as stream:stream.write(b"drift")
            elif fault == "output":
                with (suite/"CURVE_RESULTS.json").open("ab") as stream:stream.write(b"drift")
            return result
        monkeypatch.setattr(worker, "execute", changed)
    with pytest.raises((ValueError, FileNotFoundError, OSError)):worker.run(suite, sha, stage["pins"])
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
    # Even when an upstream failure wins first, no continuation is attempted.
    assert (suite/"STARTED.json").exists()


def test_foreign_STARTED_never_claimed(stage, tmp_path, monkeypatch):
    suite, sha = admitted(stage, tmp_path, monkeypatch)
    original = worker.output.os.open
    def opened(path, flags, *a, **k):
        if path == "STARTED.json" and flags & worker.output.os.O_CREAT:
            (suite/"STARTED.json").write_bytes(b"Foreign creator")
        return original(path, flags, *a, **k)
    monkeypatch.setattr(worker.output.os, "open", opened)
    with pytest.raises(FileExistsError):worker.run(suite, sha, stage["pins"])
    assert (suite/"STARTED.json").read_bytes() == b"Foreign creator" and not (suite/"FAILED.json").exists()


def test_STARTED_close_failure_is_owned(stage, tmp_path, monkeypatch):
    suite, sha = admitted(stage, tmp_path, monkeypatch)
    original = worker.output.os.fdopen
    class CloseError:
        def __init__(self, stream):self.stream = stream
        def __getattr__(self, name):return getattr(self.stream, name)
        def __enter__(self):return self
        def __exit__(self, *args):
            self.stream.__exit__(*args)
            raise OSError("Synthetic STARTED close")
    def fdopen(fd, *a, **k):
        stream = original(fd, *a, **k)
        return CloseError(stream) if worker.os.readlink(f"/proc/self/fd/{fd}") == str(suite/"STARTED.json") else stream
    monkeypatch.setattr(worker.output.os, "fdopen", fdopen)
    with pytest.raises(OSError, match="close"):worker.run(suite, sha, stage["pins"])
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
    assert not any(isinstance(e, tuple) and e[0] == "new_GT_decode_stub" for e in stage["events"])


def compact(tmp_path, labels=None):
    path = tmp_path/"compact.episode_counts.jsonl"
    clean = [dict(official_id="synthetic:0", label="/synthetic/label")]
    row = dict(episode=0, ordered_ids=["synthetic:0"], labels=labels or [dict(source_id="synthetic:0", label="/synthetic/label", sha256="a"*64)],
               I_U_by_arm=np.zeros((3, 2, 20), dtype=np.int64).tolist())
    path.write_bytes((json.dumps(row)+"\n").encode())
    binding = dict(path=str(path), identity=worker.output.file_pin(path, set(), worker.REUSED_CAP, single=True)[:4], sha256=digest(path))
    return path, binding, clean


def test_reused_label_receipts_bind_path_hash_and_cross_stream_agreement(tmp_path):
    path, binding, clean = compact(tmp_path)
    hashes = {}
    worker.reused_labels(binding, [["synthetic:0"]], clean, hashes)
    assert hashes == {"/synthetic/label": "a"*64}
    with pytest.raises(ValueError, match="disagree"):
        worker.reused_labels(binding, [["synthetic:0"]], clean, {"/synthetic/label": "b"*64})
    clean[0]["label"] = "/synthetic/wrong"
    with pytest.raises(ValueError, match="label path"):worker.reused_labels(binding, [["synthetic:0"]], clean, {})


def test_reused_second_stat_binds_supplied_identity_before_open(tmp_path, monkeypatch):
    path, binding, clean = compact(tmp_path)
    replacement = tmp_path/"identical.fixture"
    replacement.write_bytes(path.read_bytes())
    original_pin, original_open = worker.output.file_pin, Path.open
    def pin(p, *a, **k):
        result = original_pin(p, *a, **k)
        if p == path:replacement.replace(path)
        return result
    def opened(p, *a, **k):
        assert p != path, "Rebound S5 count inode was opened"
        return original_open(p, *a, **k)
    monkeypatch.setattr(worker.output, "file_pin", pin)
    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(ValueError, match="second stat"):worker.reused_labels(binding, [["synthetic:0"]], clean, {})


@pytest.mark.parametrize("name", ["cost.jsonl", "initial_state.json", "prediction_result.json"])
def test_old_cost_rebinds_all_three_metadata_files(stage, monkeypatch, name):
    spec = worker.s5_specs(stage["source"])[0]
    path = stage["g"]/"conditions"/spec["id"]/name
    def cost(*a):
        other = path.with_suffix(".replacement")
        other.write_bytes(path.read_bytes());other.replace(path)
        return dict(synthetic=True)
    monkeypatch.setattr(worker.protected, "condition_cost", cost)
    with pytest.raises(ValueError, match="cost/reset"):worker.reused_cost(spec, {}, set())


@pytest.mark.parametrize("fault", ["missing", "GT_alias", "runtime_alias"])
def test_real_small_complete_prediction_graph_before_payload_open(tmp_path, monkeypatch, fault):
    parent = tmp_path/"parent";parent.mkdir()
    spec = dict(id="synthetic.condition", images=2, config=dict(batch_size=1))
    prefix = parent/"conditions"/spec["id"]
    for arm in worker.analysis.ARMS:
        for i in range(2):
            for path in (prefix/"predictions"/arm/f"{i:06d}.npy", prefix/"episodes"/f"{i:06d}.{arm}.json"):
                path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(b"Unopened fixture")
    for name in ("cost.jsonl", "initial_state.json", "prediction_result.json"):(prefix/name).write_bytes(b"Unopened metadata")
    label, source = tmp_path/"GT.fixture", tmp_path/"source.fixture"
    label.write_bytes(b"Synthetic GT");source.write_bytes(b"Synthetic source")
    forbidden = {(p.stat().st_dev, p.stat().st_ino) for p in (label, source)}
    assert len(worker.prediction_graph(parent, [spec], forbidden)) == 15
    path = prefix/"predictions"/worker.analysis.ARMS[-1]/"000001.npy"
    path.unlink()
    if fault != "missing":path.hardlink_to(label if fault == "GT_alias" else source)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Stat graph opened payload"))
    with pytest.raises((ValueError, FileNotFoundError)):worker.prediction_graph(parent, [spec], forbidden)


def test_summary_writer_hard_256MiB_and_shared_reserve(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="cap"):
        worker.write_output(tmp_path, "never", b"x", dict(bytes=worker.OUTPUT_CAP))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda p: type("Disk", (), dict(free=worker.RESERVE))())
    with pytest.raises(ValueError, match="cap"):
        worker.write_output(tmp_path, "never", b"x", dict(bytes=0))
    assert not list(tmp_path.iterdir())


def test_final_output_samebytes_inode_swap_cannot_publish_stale_identity(stage,tmp_path,monkeypatch):
    suite,sha = admitted(stage,tmp_path,monkeypatch)
    target = suite/"CURVE_RESULTS.json"
    original_execute,original_pin = worker.execute,worker.output.file_pin
    final,swaps = [],[]
    def execute(*args,**kwargs):
        result = original_execute(*args,**kwargs)
        final.append(True)
        return result
    def swap_after_pin(path,*args,**kwargs):
        pin = original_pin(path,*args,**kwargs)
        if final and Path(path) == target and not swaps:
            replacement = suite/"samebytes.json"
            replacement.write_bytes(target.read_bytes())
            replacement.replace(target)
            swaps.append(True)
        return pin
    monkeypatch.setattr(worker,"execute",execute)
    monkeypatch.setattr(worker.output,"file_pin",swap_after_pin)
    with pytest.raises(ValueError,match="identity"):
        worker.run(suite,sha,stage["pins"])
    assert swaps and (suite/"FAILED.json").exists()
    assert not (suite/"ANALYSIS_SEALED.json").exists()
    with pytest.raises(ValueError,match="resume/retry"):
        worker.run(suite,sha,stage["pins"])
