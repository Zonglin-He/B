"""M12 synthetic implementation tests, not actual provenance/admission proof.

Services, formal source populations, native/origin checks, cost semantics and
(except explicitly real-validator regressions) episode semantics are stubs. Real selected-file
pins, hash-bound metadata, renderer, protected I/O and lifecycle run in /tmp.
No actual result tree, dataset, service, model, GPU or execution lock is used.
"""

import copy
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_fifty import case_worker as worker

REAL_EPISODE_METADATA = worker.parent.episode_metadata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return worker.protected.write_prepared_json(path, value)


def permission(root, prepared, **flags):
    report = root/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic implementation fixture, never actual admission.\n")
    digest = save(root/"ADMITTED.json", dict(decision="ADMITTED", suite_prepared_sha256=prepared,
        independent_reviewer="SyntheticCPU", report_path=str(report), report_sha256=sha(report), **flags))
    ids = [worker.output.file_pin(p, set(), 4*1024**2, single=True)[:4] for p in (root/"ADMITTED.json", report)]
    return dict(admission_path=str(root/"ADMITTED.json"), admission_sha256=digest, admission_identity=ids[0],
        report_path=str(report), report_sha256=sha(report), report_identity=ids[1], independent_reviewer="SyntheticCPU")


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root, parent, g = (tmp_path/name for name in ("sources", "M02", "Gbootstrap"))
    for path in (parent/"runtime", g/"runtime"):
        path.mkdir(parents=True)
        (path/"fixture.py").write_text("# Source role only, never imported.\n")
    for tree in ("trace_ovss", "tools", "tests"):
        (root/tree).mkdir(parents=True)
        (root/tree/"fixture.py").write_text("# Unexecuted synthetic runtime member.\n")
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(worker.parent, "PARENT", parent)
    monkeypatch.setattr(worker.protected, "PARENT", g)
    source = dict(rows={}, specs=[], annotation_metadata={}, GT_opened=False)
    events, payloads = [], set()
    raw_costs, selected_receipts, seals, seal_hashes = {}, {}, {}, {}
    for dataset in worker.DATASETS:
        rows = []
        for index in (0, 1):
            rgb, gt = tmp_path/f"{dataset}.{index}.png", tmp_path/f"{dataset}.{index}.GT.png"
            pixels = np.full((17, 23, 3), 60+index*30, np.uint8)
            Image.fromarray(pixels).save(rgb)
            Image.fromarray(np.full((17, 23), index, np.uint8)).save(gt)
            state = gt.stat()
            details = dict(device=state.st_dev, inode=state.st_ino, size_bytes=state.st_size,
                           mtime_ns=state.st_mtime_ns, contents_read=False)
            row = dict(official_id=f"synthetic:{dataset}:{index}", image=str(rgb), image_sha256=sha(rgb), label=str(gt),
                label_metadata=[state.st_size, state.st_mtime_ns], label_metadata_details=details)
            rows.append(row)
            source["annotation_metadata"][str(gt)] = details
            payloads.update((str(rgb), str(gt)))
        source["rows"][dataset] = rows
        spec = dict(id="M02."+dataset, dataset=dataset, images=2, modes=list(worker.MODES), preprocessing="native_PIL_bilinear",
            backbone="ViT-L/14", model_type="naclip", corruption="original", checkpoint=dict(synthetic=True),
            vision_outputs=worker.parent.prediction.visual_layers("ViT-L/14"),
            config=dict(batch_size=2, resize=[224, 224], patch_size=[224, 224], trace_max_regions=4,
                        steps=0, trace_view_batch_size=2),
            vocabulary=dict(classes=["zero", "one"], queries=["zero", "one"], reduce_zero_label=False))
        source["specs"].append(spec)
        inventory, cost_rows = {}, []
        for position, mode in enumerate(worker.MODES):
            for index in (0, 1):
                path = parent/dataset/"predictions"/mode/f"{index:06d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = io.BytesIO();np.save(stream, np.full((224, 224), index, np.uint16), allow_pickle=False)
                path.write_bytes(stream.getvalue())
                inventory[str(path.relative_to(parent))] = sha(path)
                payloads.add(str(path))
            cost = dict(episode=0, mode=mode, arm=mode, order_position=position, images=2,
                        candidate_count=0, active_candidates=0, steps=0, optimizer_steps=0, warmup=True)
            cost_rows.append(cost)
            record = dict(cost, ordered_ids=[r["official_id"] for r in rows], GT_opened=False, losses=[], candidates=[],
                          preservation_geometry=[], scientific_identity="M02_spatial_"+mode,
                          candidate_sha256=worker.parent.runtime.image_worker.prior.candidate_identity([]))
            path = parent/dataset/"episodes"/f"000000.{mode}.json"
            inventory[str(path.relative_to(parent))] = save(path, record)
            payloads.add(str(path))
            selected_receipts[(dataset, mode)] = record
        raw_costs[dataset] = cost_rows
        for name in ("cost.jsonl", "initial_state.json", "prediction_result.json"):
            path = parent/dataset/name
            path.write_bytes(b"Synthetic closed-validator cost/reset fixture\n")
            inventory[str(path.relative_to(parent))] = sha(path)
        seal = dict(complete=True, dataset=dataset, images=2, modes=list(worker.MODES), GT_opened=False,
                    image_method_evaluations=8, artifact_sha256=inventory, resource_observation=dict(synthetic=True))
        seals[dataset] = seal
        seal_hashes[dataset] = save(parent/dataset/"PREDICTIONS_SEALED.json", seal)
    g_inputs = {"clean."+d: copy.deepcopy(source["rows"][d]) for d in worker.DATASETS}
    for key in ("clean.voc20", "corruption.frost"):
        row = copy.deepcopy(source["rows"]["coco_object"][0])
        image = tmp_path/(key+".png")
        image.write_bytes(Path(row["image"]).read_bytes())
        row["image"], row["image_sha256"] = str(image), sha(image)
        g_inputs[key] = [row]
        payloads.add(str(image))
    g_specs = [dict(copy.deepcopy(s), id="G."+s["dataset"], arms=["synthetic"]) for s in source["specs"]]
    g_inputs_sha = save(g/"INPUTS.json", g_inputs)
    g_conditions_sha = save(g/"CONDITIONS.json", g_specs)
    g_prepared_sha = save(g/"SUITE_PREPARED.json", dict(schema_version=1, root=str(g), inputs_sha256=g_inputs_sha,
        conditions_sha256=g_conditions_sha, runtime_hashes={"fixture.py": sha(g/"runtime"/"fixture.py")}))
    monkeypatch.setattr(worker.protected, "PARENT_SHA", g_prepared_sha)
    monkeypatch.setattr(worker.protected, "PARENT_METADATA_IDENTITIES", {
        name: tuple(worker.output.file_pin(g/name, set(), 128*1024**2, single=True)[:4])
        for name in ("SUITE_PREPARED.json", "INPUTS.json")})
    source.update(parent_input_suite=str(g), parent_prepared_sha256=g_prepared_sha, parent_inputs_sha256=g_inputs_sha,
                  parent_conditions_sha256=g_conditions_sha, G_condition_ids=[s["id"] for s in g_specs], old_predictions_reused=False)
    inputs_sha = save(parent/"INPUTS.json", source)
    proto_sha = save(parent/"PROTOCOL.json", dict(synthetic_M02=True))
    prepared_sha = save(parent/"SUITE_PREPARED.json", dict(schema_version=1, root=str(parent), inputs_sha256=inputs_sha,
        protocol_sha256=proto_sha, parent_native_hashes={f"synthetic_native_{i}": "b"*64 for i in range(68)},
        runtime_hashes={"fixture.py": sha(parent/"runtime"/"fixture.py")}))
    old_audit = permission(parent, prepared_sha, M02_predictions_allowed=True, GT_scoring_allowed=False)
    for dataset, seal in seals.items():
        seal["suite_prepared_sha256"] = prepared_sha
        path = parent/dataset/"PREDICTIONS_SEALED.json"
        path.unlink()
        seal_hashes[dataset] = save(path, seal)
    start_sha = save(parent/"STARTED.json", dict(suite_prepared_sha256=prepared_sha, audit=old_audit, pid=123, GT_opened=False))
    barrier_sha = save(parent/"ALL_PREDICTIONS_SEALED.json", dict(complete=True, GT_opened=False,
        suite_prepared_sha256=prepared_sha, cells=8, image_method_evaluations=16, dataset_seals=seal_hashes, started_sha256=start_sha,
        audit_binding=old_audit, gate_binding=dict(synthetic=True)))
    for name, value in dict(PARENT_SHA=prepared_sha, PARENT_BARRIER_SHA=barrier_sha, PARENT_START_SHA=start_sha,
                             PARENT_ADMISSION_SHA=old_audit["admission_sha256"], PARENT_REPORT_SHA=old_audit["report_sha256"]).items():
        monkeypatch.setattr(worker.parent, name, value)
    terminal = dict(ExecMainPID="123", InvocationID="c"*32, Result="success", SubState="exited")
    analysis = tmp_path/"M02_ANALYSIS_PROTOCOL_v1.json"
    analysis_sha = save(analysis, dict(prediction_prepared_sha256=prepared_sha, arm_order=list(worker.MODES),
        fixed_contrasts={k: list(v) for k, v in worker.parent.analysis.CONTRASTS.items()},
        uncertainty=dict(replicates=10000, root_seed=20261102)))
    monkeypatch.setattr(worker.parent, "ANALYSIS", analysis)
    monkeypatch.setattr(worker.parent, "ANALYSIS_SHA", analysis_sha)
    monkeypatch.setattr(worker.parent, "POPULATIONS", {d: 2 for d in worker.DATASETS})
    binding = dict(parent=str(parent), prepared_sha256=prepared_sha, barrier_sha256=barrier_sha,
                   terminal=terminal, source=source, seals=seals)
    monkeypatch.setattr(worker.parent, "bind_parent", lambda: pytest.fail("Unprotected old parent bootstrap called"))
    monkeypatch.setattr(worker.parent.prediction, "source_inputs", lambda: pytest.fail("Unprotected old source bootstrap called"))
    monkeypatch.setattr(worker.parent.prediction, "validate_inputs", lambda *a: events.append("M02_population_stub"))
    monkeypatch.setattr(worker.parent.prediction.general, "validate_inputs", lambda *a: events.append("G_population_stub"))
    monkeypatch.setattr(worker.parent, "parent_terminal", lambda: copy.deepcopy(terminal))
    monkeypatch.setattr(worker.parent, "native_binding", lambda value, forbidden: events.append(("native_stub", len(value))))
    monkeypatch.setattr(worker.parent.prediction, "verify_gates", lambda *a: events.append("gate_stub"))
    monkeypatch.setattr(worker.parent.prediction.general, "runtime_imports", lambda *a: events.append("origin_stub"))
    def cost(root, spec, inventory, forbidden):
        events.append(("cost_stub", spec["dataset"]))
        return dict(synthetic_cost=True), copy.deepcopy(raw_costs[spec["dataset"]])
    monkeypatch.setattr(worker.parent, "cost_binding", cost)
    def episode(root, spec, inventory, forbidden, number, raw, coverage):
        assert number == 0 and len(raw) == 4
        events.append(("episode_semantics_stub", spec["dataset"]))
        return {mode: [r["official_id"] for r in source["rows"][spec["dataset"]]] for mode in worker.MODES}
    monkeypatch.setattr(worker.parent, "episode_metadata", episode)
    scope = tmp_path/"SCOPE_v4.json"
    monkeypatch.setattr(worker, "SCOPE", scope)
    scope_sha = save(scope, dict(version="SCOPE_v4", M12="Four fixed descriptive cases"))
    prospective = tmp_path/"M12_CASE_PROTOCOL_v1.json"
    monkeypatch.setattr(worker, "CASE_PROTOCOL", prospective)
    monkeypatch.setattr(worker, "CASE_PROTOCOL_SHA", save(prospective, worker.core.protocol()))
    return dict(binding=binding, source=source, root=root, parent=parent, scope_sha=scope_sha,
                payloads=payloads, events=events, receipts=selected_receipts, g=g, g_inputs=g_inputs, raw_costs=raw_costs)


def admitted(stage, tmp_path, monkeypatch):
    suite = tmp_path/"M12_cases"
    expected = worker.prepare(suite, stage["scope_sha"])
    permission(suite, expected, CPU_GT_scoring_allowed=True, M12_qualitative_GT_allowed=True,
               RGB_reading_allowed=True, GPU_allowed=False, MODEL_allowed=False)
    # Explicit fixture-origin stub; this is not genuine-runtime admission proof.
    monkeypatch.setattr(worker, "__file__", str(suite/"runtime"/"tools"/"fixture.py"))
    return suite, expected


def test_fixed_four_cases_protocol_and_complete_selected_stat_graph(stage):
    selected = worker.fixed_cases(stage["source"])
    assert [(c["dataset"], c["index"], c["episode"], c["batch_idx"]) for c in selected] == [
        ("coco_object", 0, 0, 0), ("coco_object", 1, 0, 1), ("pc60", 0, 0, 0), ("pc60", 1, 0, 1)]
    graph = worker.preflight_selected(stage["binding"], selected)
    assert len(graph) == 38 and {v["role"] for v in graph.values()} == {"rgb", "GT", "prediction", "episode", "cost_reset"}
    assert worker.protocol(stage["scope_sha"])["execution_resources"]["MemoryMax"] == 4*1024**3
    assert "Known aggregate M02" in worker.protocol(stage["scope_sha"])["exposure"]


def test_prepare_and_admission_do_not_open_selected_case_content(stage, tmp_path, monkeypatch):
    original = Path.open
    def guard(path, *a, **k):
        assert str(path) not in stage["payloads"], "Metadata phase opened selected case content"
        return original(path, *a, **k)
    monkeypatch.setattr(Path, "open", guard)
    suite, expected = admitted(stage, tmp_path, monkeypatch)
    worker.admission(suite, expected, stage["scope_sha"])
    assert not (suite/"STARTED.json").exists()


def test_real_renderer_synthetic_full_lifecycle_global_last_and_no_parent_mutation(stage, tmp_path, monkeypatch):
    suite, expected = admitted(stage, tmp_path, monkeypatch)
    source_hashes = {path: sha(Path(path)) for path in stage["payloads"]}
    publish = worker.publish_seal
    def final(path, value):
        assert json.loads((suite/"queue_state.json").read_bytes())["phase"] == "M12_cases_complete_awaiting_seal_and_exit"
        assert value["cases"] == 4 and len(value["artifact_sha256"]) == 9
        publish(path, value)
    monkeypatch.setattr(worker, "publish_seal", final)
    fds = len(os.listdir('/proc/self/fd'))
    worker.run(suite, expected, stage["scope_sha"])
    assert len(os.listdir('/proc/self/fd')) == fds
    result = json.loads((suite/"CASES_SEALED.json").read_bytes())
    assert not result["GPU_used"] and not result["MODEL_used"] and not result["uploaded"] and not result["B_published"]
    for name, digest in result["artifact_sha256"].items():assert sha(suite/name) == digest
    for path in (suite/"cases").glob("*.png"):
        rendering = json.loads(path.with_suffix(".json").read_bytes())["rendering"]
        with Image.open(path) as image:
            assert list(image.size) == rendering["PNG_size"]
            assert image.width == 960 and image.height >= 608
            assert 0 < rendering["tile_top"] < image.height
    assert source_hashes == {path: sha(Path(path)) for path in stage["payloads"]}
    assert sum(isinstance(e, tuple) and e[0] == "episode_semantics_stub" for e in stage["events"]) == 2
    with pytest.raises(ValueError, match="resume/retry"):worker.run(suite, expected, stage["scope_sha"])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("flag", ["CPU_GT_scoring_allowed", "M12_qualitative_GT_allowed", "RGB_reading_allowed", "GPU_allowed", "MODEL_allowed"])
def test_all_new_permissions_required_before_ANY_case_content(stage, tmp_path, monkeypatch, flag):
    suite, expected = admitted(stage, tmp_path, monkeypatch)
    path = suite/"ADMITTED.json"
    value = json.loads(path.read_bytes());value[flag] = flag in ("GPU_allowed", "MODEL_allowed")
    path.unlink();save(path, value)
    original = Path.open
    def guard(path, *a, **k):
        assert str(path) not in stage["payloads"]
        return original(path, *a, **k)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError, match="admission"):worker.run(suite, expected, stage["scope_sha"])
    assert not (suite/"STARTED.json").exists()


@pytest.mark.parametrize("role", ["rgb", "GT", "prediction", "episode", "cost_reset"])
def test_late_selected_missing_file_refuses_before_any_content(stage, monkeypatch, role):
    selected = worker.fixed_cases(stage["source"])
    graph = worker.preflight_selected(stage["binding"], selected)
    path = Path([name for name, record in graph.items() if record["role"] == role][-1])
    path.unlink()
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Incomplete graph opened any content"))
    with pytest.raises(FileNotFoundError):worker.preflight_selected(stage["binding"], selected)


@pytest.mark.parametrize("victim", ["GT", "rgb", "source", "another_prediction"])
def test_selected_prediction_alias_roles_refuse_stat_only(stage, monkeypatch, victim):
    graph = worker.preflight_selected(stage["binding"], worker.fixed_cases(stage["source"]))
    path = Path(next(name for name, r in graph.items() if r["role"] == "prediction"))
    if victim == "source":target = stage["root"]/"tools"/"fixture.py"
    else:target = Path([name for name, r in graph.items() if r["role"] == ("prediction" if victim == "another_prediction" else victim)][-1])
    path.unlink();path.hardlink_to(target)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Role preflight opened content"))
    with pytest.raises(ValueError):worker.preflight_selected(stage["binding"], worker.fixed_cases(stage["source"]))


@pytest.mark.parametrize("which", ["SCOPE", "CASE_PROTOCOL"])
def test_scope_and_prospective_protocol_drift_precede_payloads(stage, monkeypatch, which):
    with getattr(worker, which).open("ab") as stream:stream.write(b"drift")
    original = Path.open
    def guard(path, *a, **k):
        assert str(path) not in stage["payloads"]
        return original(path, *a, **k)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError):worker.bind_inputs(stage["scope_sha"])


def test_outcome_selection_or_bool_index_rejected(stage, monkeypatch):
    selected = worker.core.select_cases(stage["source"])
    selected[0]["index"] = False
    monkeypatch.setattr(worker.core, "select_cases", lambda s: selected)
    with pytest.raises(ValueError, match="selection"):worker.fixed_cases(stage["source"])


@pytest.mark.parametrize("kind", ["cost.jsonl", "initial_state.json", "prediction_result.json", "episode"])
def test_closed_validator_samebytes_replacement_cannot_be_adopted_before_GT(stage, monkeypatch, kind):
    bound = worker.bind_inputs(stage["scope_sha"])
    if kind == "episode":
        path = stage["parent"]/"pc60"/"episodes"/"000000.confidence.json"
        original = worker.parent.episode_metadata
        def replaced(*a):
            value = original(*a)
            if a[1]["dataset"] == "pc60":
                new = path.with_suffix(".replacement");new.write_bytes(path.read_bytes());new.replace(path)
            return value
        monkeypatch.setattr(worker.parent, "episode_metadata", replaced)
    else:
        path = stage["parent"]/"pc60"/kind
        original = worker.parent.cost_binding
        def replaced(*a):
            value = original(*a)
            if a[1]["dataset"] == "pc60":
                new = path.with_suffix(".replacement");new.write_bytes(path.read_bytes());new.replace(path)
            return value
        monkeypatch.setattr(worker.parent, "cost_binding", replaced)
    original_open = Path.open
    forbidden = {str(Path(row[k])) for rows in stage["source"]["rows"].values() for row in rows for k in ("image", "label")}
    def guard(path, *a, **k):
        assert str(path) not in forbidden, "Unbound metadata reached RGB/GT"
        return original_open(path, *a, **k)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError, match="identity drift"):worker.validate_selected_metadata(bound)


@pytest.mark.parametrize("field,value", [("episode", False), ("episode", 0.), ("order_position", False), ("images", 2.)])
def test_receipt_cost_pairing_is_type_exact_before_GT(stage, monkeypatch, field, value):
    path = stage["parent"]/"pc60"/"episodes"/"000000.native.json"
    record = json.loads(path.read_bytes());record[field] = value
    path.unlink();save(path, record)
    stage["binding"]["seals"]["pc60"]["artifact_sha256"][str(path.relative_to(stage["parent"]))] = sha(path)
    cases = worker.fixed_cases(stage["source"])
    # Direct isolated metadata-phase fixture; no forged real seal is admitted.
    bound = dict(parent_binding=copy.deepcopy(stage["binding"]), cases=cases,
                 selected_payloads=worker.preflight_selected(stage["binding"], cases))
    with pytest.raises(ValueError, match="typed episode"):worker.validate_selected_metadata(bound)


@pytest.mark.parametrize("fault", ["render", "final_parent", "scope", "prospective", "progress", "publish", "report", "output"])
def test_owned_errors_failed_no_global_and_no_retry(stage, tmp_path, monkeypatch, fault):
    suite, expected = admitted(stage, tmp_path, monkeypatch)
    def fail(*a, **k):raise OSError("Synthetic ordinary IO failure")
    if fault == "render":monkeypatch.setattr(worker.core, "render_case", fail)
    elif fault == "progress":monkeypatch.setattr(worker.parent, "write_json", fail)
    elif fault == "publish":monkeypatch.setattr(worker, "publish_seal", fail)
    else:
        original = worker.execute
        def changed(*a):
            result = original(*a)
            path = {"final_parent": stage["parent"]/"INPUTS.json", "scope": worker.SCOPE,
                    "prospective": worker.CASE_PROTOCOL, "report": suite/"INDEPENDENT_REPORT.md",
                    "output": suite/"CASES.json"}[fault]
            with path.open("ab") as stream:stream.write(b"drift")
            return result
        monkeypatch.setattr(worker, "execute", changed)
    with pytest.raises((ValueError, OSError)):worker.run(suite, expected, stage["scope_sha"])
    assert (suite/"STARTED.json").exists() and (suite/"FAILED.json").exists() and not (suite/"CASES_SEALED.json").exists()


def test_final_JSON_samebytes_replacement_between_pin_and_read_fails_no_resume(stage, tmp_path, monkeypatch):
    suite, expected = admitted(stage, tmp_path, monkeypatch)
    original = worker.stable_bytes
    target = suite/"CASES.json"
    replaced = []
    def read(path, cap, digest=None, expected_identity=None):
        if Path(path) == target:
            assert expected_identity is not None and not replaced
            replacement = suite/"samebytes.tmp"
            replacement.write_bytes(target.read_bytes())
            assert worker.output.file_pin(replacement, set(), cap, single=True)[:2] != tuple(expected_identity[:2])
            replacement.replace(target)
            replaced.append(True)
        return original(path, cap, digest, expected_identity=expected_identity)
    monkeypatch.setattr(worker, "stable_bytes", read)
    with pytest.raises(ValueError, match="Frozen opened-file identity differs"):
        worker.run(suite, expected, stage["scope_sha"])
    assert replaced and (suite/"STARTED.json").exists() and (suite/"FAILED.json").exists()
    assert len(list((suite/"cases").glob("*.png"))) == 4
    assert not (suite/"CASES_SEALED.json").exists()
    with pytest.raises(ValueError, match="resume/retry"):
        worker.run(suite, expected, stage["scope_sha"])


def test_STARTED_close_error_is_owned_without_selected_read(stage, tmp_path, monkeypatch):
    suite, expected = admitted(stage, tmp_path, monkeypatch)
    original = worker.output.os.fdopen
    class CloseError:
        def __init__(self, stream):self.stream = stream
        def __getattr__(self, name):return getattr(self.stream, name)
        def __enter__(self):return self
        def __exit__(self, *a):
            self.stream.__exit__(*a)
            raise OSError("Synthetic STARTED close failure")
    def fdopen(fd, *a, **k):
        stream = original(fd, *a, **k)
        return CloseError(stream) if os.readlink(f"/proc/self/fd/{fd}") == str(suite/"STARTED.json") else stream
    monkeypatch.setattr(worker.output.os, "fdopen", fdopen)
    monkeypatch.setattr(worker, "execute", lambda *a: pytest.fail("STARTED close error reached payloads"))
    with pytest.raises(OSError, match="STARTED close"):worker.run(suite, expected, stage["scope_sha"])
    assert (suite/"FAILED.json").exists() and not (suite/"CASES_SEALED.json").exists()


def test_foreign_STARTED_is_not_owned_or_modified(stage, tmp_path, monkeypatch):
    suite, expected = admitted(stage, tmp_path, monkeypatch)
    original = worker.output.os.open
    def opened(path, flags, *a, **k):
        if path == "STARTED.json" and flags & os.O_CREAT:(suite/"STARTED.json").write_bytes(b"Foreign creator")
        return original(path, flags, *a, **k)
    monkeypatch.setattr(worker.output.os, "open", opened)
    with pytest.raises(FileExistsError):worker.run(suite, expected, stage["scope_sha"])
    assert (suite/"STARTED.json").read_bytes() == b"Foreign creator" and not (suite/"FAILED.json").exists()


def test_output_cap_and_reserve_checked_before_creation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="budget"):worker.write_output(tmp_path, "never", b"x", dict(bytes=worker.OUTPUT_CAP))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda p: type("Disk", (), dict(free=worker.RESERVE))())
    with pytest.raises(ValueError, match="budget"):worker.write_output(tmp_path, "never", b"x", dict(bytes=0))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("input_key", ["clean.coco_object", "clean.voc20", "corruption.frost"])
@pytest.mark.parametrize("role", ["M02_runtime", "G_runtime", "current_source", "reference", "G_conditions", "M02_inputs"])
def test_bootstrap_all_G_RGB_alias_roles_refuse_without_any_data_open(stage, monkeypatch, input_key, role):
    image = Path(stage["g_inputs"][input_key][0]["image"])
    reference = stage["g"]/"reference"/"fixture.txt"
    reference.parent.mkdir()
    reference.write_text("Synthetic reference role, no execution.\n")
    target = {"M02_runtime": stage["parent"]/"runtime"/"fixture.py",
              "G_runtime": stage["g"]/"runtime"/"fixture.py",
              "current_source": stage["root"]/"tools"/"fixture.py", "reference": reference,
              "G_conditions": stage["g"]/"CONDITIONS.json", "M02_inputs": stage["parent"]/"INPUTS.json"}[role]
    target.unlink()
    target.hardlink_to(image)
    data_inodes = {(Path(row[k]).stat().st_dev, Path(row[k]).stat().st_ino)
                   for rows in stage["g_inputs"].values() for row in rows for k in ("image", "label")}
    original, opened_data = Path.open, []
    def guard(path, *a, **k):
        state = path.stat()
        if (state.st_dev, state.st_ino) in data_inodes:
            opened_data.append(str(path))
            pytest.fail("Bootstrap opened a data-role inode through a metadata/source alias")
        return original(path, *a, **k)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError, match="aliases"):
        worker.bind_inputs(stage["scope_sha"])
    assert not opened_data and not any(isinstance(e, tuple) and e[0] == "native_stub" for e in stage["events"])


def test_exact_terminal_precedes_even_trusted_bootstrap_reads(stage, monkeypatch):
    def terminal():raise ValueError("Synthetic retained terminal differs")
    monkeypatch.setattr(worker.parent, "parent_terminal", terminal)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Unsuccessful terminal reached bootstrap bytes"))
    with pytest.raises(ValueError, match="terminal"):
        worker.bind_inputs(stage["scope_sha"])


@pytest.mark.parametrize("name", ["SUITE_PREPARED.json", "INPUTS.json"])
def test_trusted_G_both_operator_stat_pins_checked_before_initial_open(stage, monkeypatch, name):
    identities = dict(worker.protected.PARENT_METADATA_IDENTITIES)
    value = list(identities[name]);value[1] += 1
    identities[name] = tuple(value)
    monkeypatch.setattr(worker.protected, "PARENT_METADATA_IDENTITIES", identities)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Unbound bootstrap metadata was opened"))
    with pytest.raises(ValueError, match="bootstrap stat pin"):
        worker.bind_inputs(stage["scope_sha"])


def test_all_G_roles_separate_from_exact_M02_source(stage):
    bound = worker.bind_inputs(stage["scope_sha"])
    binding = bound["parent_binding"]
    assert worker.same_provenance(binding["source"], stage["source"])
    assert set(binding["source"]["rows"]) == set(worker.DATASETS)
    assert set(binding["data_roles"]["rows"]) == set(stage["g_inputs"])
    assert "clean.voc20" in binding["data_roles"]["rows"] and "corruption.frost" in binding["data_roles"]["rows"]
    assert len(binding["data_roles"]["image_pins"]) == 6


def candidate_bound(stage, *, component=1, distinct_C=False):
    """Generated receipts accepted by the REAL old episode/cost validators."""
    mask = np.zeros((16, 16), dtype=bool);mask[0, 0] = True
    candidate = dict(batch_idx=0, image_idx=0, class_idx=1, component_idx=component, area=1,
        confidence=.8, reliability=.4, region=dict(shape=[16, 16], encoding="packbits-little-hex",
        true_pixels=1, data=np.packbits(mask.ravel(), bitorder="little").tobytes().hex()))
    _, geometry = worker.core.translated_candidate(np.repeat(np.repeat(mask, 14, axis=0), 14, axis=1))
    for dataset in worker.DATASETS:
        for position, mode in enumerate(worker.MODES):
            item = copy.deepcopy(candidate)
            if distinct_C:item["reliability"] = [.4, .123, .7, .8][position]
            raw = stage["raw_costs"][dataset][position]
            raw.update({k: 0 for k in worker.parent.runtime.image_worker.COUNTERS})
            views = int(mode != "confidence")
            raw.update(candidate_count=1, active_candidates=1, patches=2, seconds=.01,
                eval_forward_calls=2, eval_forward_inputs=4, diagnostic_forward_calls=views,
                diagnostic_forward_inputs=views, diagnostic_encoder_calls=views, diagnostic_encoder_inputs=views,
                encoder_calls=2+views, encoder_inputs=4+views)
            record = dict(raw, ordered_ids=[r["official_id"] for r in stage["source"]["rows"][dataset]], GT_opened=False,
                scientific_identity="M02_spatial_"+mode, losses=[], candidates=[item],
                preservation_geometry=[dict(geometry if mode == "translated" else {}, candidate_index=0,
                    batch_idx=0, mode=mode, readout_and_loss_remain_original=True)],
                candidate_sha256=worker.core.candidate_identity([dict(item, region=mask.copy())]))
            path = stage["parent"]/dataset/"episodes"/f"000000.{mode}.json"
            path.unlink()
            stage["binding"]["seals"][dataset]["artifact_sha256"][str(path.relative_to(stage["parent"]))] = save(path, record)
    cases = worker.fixed_cases(stage["source"])
    return dict(parent_binding=copy.deepcopy(stage["binding"]), cases=cases,
                selected_payloads=worker.preflight_selected(stage["binding"], cases), scope_sha256=stage["scope_sha"],
                case_protocol=worker.core.protocol())


def episode_coverage():
    coverage = {mode: dict(candidates=0, EPS_active_candidates=0, sources=0) for mode in worker.MODES}
    coverage.update(translated_shiftable=0, translated_unshiftable=0, translated_overlap_sum=0.)
    return coverage


def test_real_old_episode_valid_component_zero_rejected_before_any_selected_content(stage, tmp_path, monkeypatch):
    bound = candidate_bound(stage, component=0)
    # Prove the precise gap with unchanged real metadata/cost math, not a no-op validator.
    for spec in stage["source"]["specs"]:
        dataset = spec["dataset"]
        actual = REAL_EPISODE_METADATA(stage["parent"], spec, bound["parent_binding"]["seals"][dataset]["artifact_sha256"],
            worker.source_exclusions(stage["source"]), 0, stage["raw_costs"][dataset], episode_coverage())
        assert all(ids == [r["official_id"] for r in stage["source"]["rows"][dataset]] for ids in actual.values())
    monkeypatch.setattr(worker.parent, "episode_metadata", REAL_EPISODE_METADATA)
    forbidden = {name for name, record in bound["selected_payloads"].items() if record["role"] in ("rgb", "GT", "prediction")}
    original, opened = Path.open, []
    def guard(path, *a, **k):
        if str(path) in forbidden:
            opened.append(str(path));pytest.fail("Invalid original receipt reached selected RGB/GT/NPY bytes")
        return original(path, *a, **k)
    monkeypatch.setattr(Path, "open", guard)
    suite = tmp_path/"isolated_execute";suite.mkdir()
    with pytest.raises(ValueError, match="candidate source/query/score"):
        worker.execute(suite, bound)
    assert not opened and not list(suite.iterdir())


def test_real_old_episode_distinct_legal_mode_C_survives_strict_metadata_and_render(stage, tmp_path, monkeypatch):
    bound = candidate_bound(stage, distinct_C=True)
    monkeypatch.setattr(worker.parent, "episode_metadata", REAL_EPISODE_METADATA)
    suite = tmp_path/"isolated_execute";suite.mkdir()
    result = worker.execute(suite, bound)
    assert result["complete"] and len(result["artifact_sha256"]) == 9
    record = json.loads((suite/"cases"/"coco_object.000000.json").read_bytes())
    assert record["rendering"]["candidate"]["C"] == .4
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["bool_image", "bool_geometry_index", "bool_geometry_batch", "ordered_ids",
                                  "digest", "maximum", "query_bound", "geometry_value_type"])
def test_strict_metadata_type_identity_and_resource_boundaries_open_no_pixels(stage, tmp_path, monkeypatch, fault):
    bound = candidate_bound(stage)
    spec = bound["parent_binding"]["source"]["specs"][0]
    path = stage["parent"]/"coco_object"/"episodes"/"000000.translated.json"
    record = json.loads(path.read_bytes())
    if fault == "bool_image":
        record["candidates"][0]["image_idx"] = False
        item = record["candidates"][0]
        record["candidate_sha256"] = worker.core.candidate_identity([
            dict(item, region=worker.core.candidate_mask(item)[::14, ::14].copy())])
    elif fault == "bool_geometry_index":record["preservation_geometry"][0]["candidate_index"] = False
    elif fault == "bool_geometry_batch":record["preservation_geometry"][0]["batch_idx"] = False
    elif fault == "ordered_ids":record["ordered_ids"][1] = "other-source"
    elif fault == "digest":record["candidate_sha256"] = "a"*64
    elif fault == "maximum":spec["config"]["trace_max_regions"] = True
    elif fault == "query_bound":spec["vocabulary"]["queries"][1] = "q"*1025
    else:
        value = record["preservation_geometry"][0]["dy"]
        record["preservation_geometry"][0]["dy"] = float(value)
    path.unlink()
    bound["parent_binding"]["seals"]["coco_object"]["artifact_sha256"][str(path.relative_to(stage["parent"]))] = save(path, record)
    bound["selected_payloads"] = worker.preflight_selected(bound["parent_binding"], bound["cases"])
    forbidden = {name for name, r in bound["selected_payloads"].items() if r["role"] in ("rgb", "GT", "prediction")}
    original = Path.open
    def guard(path, *a, **k):
        assert str(path) not in forbidden, "Invalid metadata/resource bound reached selected pixel bytes"
        return original(path, *a, **k)
    monkeypatch.setattr(Path, "open", guard)
    suite = tmp_path/"isolated_execute";suite.mkdir()
    with pytest.raises(ValueError):worker.execute(suite, bound)
    assert not list(suite.iterdir())
