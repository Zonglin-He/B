"""CPU-only full-suite admission, historical-proof and bounded queue tests."""

import copy
import csv
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools.evolution_image_full import queue
from tools.evolution_image_full import run as worker
from test_image_full_worker import config

SOURCE = queue.ROOT
SOURCE_CONTINUITY = queue.validate_source_continuity


def replace_json(path, value):
    path.unlink()
    queue.write_json(path, value)


def test_locked_plan_and_exact_six_inventory():
    plan = SOURCE / "docs/image-full-plan.json"
    assert queue.sha256(plan) == queue.PLAN_SHA256
    value = queue.read_json(plan)
    queue.validate_plan(value)
    assert [e["dataset"] for e in queue.cells()] == list(queue.DATASETS)
    assert [e["images"] for e in queue.cells()] == [5105, 5000, 1449, 1449, 500, 5000]
    assert all(e["arms"] == ["image_uniform", "image_confidence", "image_C"] for e in queue.cells())
    value["selected_global_mode"] = "T"
    with pytest.raises(RuntimeError):
        queue.validate_plan(value)


@pytest.mark.skipif(not queue.SELECTION_ROOT.exists(), reason="Pinned local evidence unavailable")
def test_actual_pinned_calibration_selection_proof_and_transitive_mutation(tmp_path, monkeypatch):
    original_open = Path.open

    def guard(path, *args, **kwargs):
        assert "/cells/confirmation/" not in str(path), "Selection must not read confirmation artifacts"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    files = queue.selection_proof_files(queue.SELECTION_ROOT)
    assert len(files) == 15
    destination = tmp_path / "selection"
    for relative in sorted(files):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(queue.SELECTION_ROOT / relative, target)
    queue.selection_proof_files(destination)
    transitive = destination / "cells/calibration/pc60/sorted/scores_by_episode.jsonl"
    transitive.write_text(transitive.read_text() + "\n")
    with pytest.raises(RuntimeError, match="scored artifact drift"):
        queue.selection_proof_files(destination)


@pytest.mark.skipif(not queue.PARENT_ROOT.exists(), reason="Pinned local evidence unavailable")
@pytest.mark.parametrize("dataset", queue.DATASETS)
def test_actual_six_pinned_parents_keep_symlink_paths_and_exact_scope(dataset):
    protocol, receipt = queue.read_parent(dataset)
    assert protocol["images"] == queue.POPULATIONS[dataset]
    assert receipt["protocol_sha256"] == queue.PARENT_HASHES[dataset][0]
    if dataset.startswith("coco"):
        assert any(str(Path(r["image"]).resolve()) != r["image"] for r in protocol["inputs"])
    if dataset == "voc20":
        assert "cli_recovery" in receipt["protocol_path"] and "/mechanism/" in receipt["protocol_path"]
        assert protocol["config"]["batch_size"] == 1


@pytest.mark.skipif(not queue.PARENT_ROOT.exists(), reason="Pinned local evidence unavailable")
def test_real_parent_duplicate_and_changed_pin_rejected(tmp_path, monkeypatch):
    protocol, _ = queue.read_parent("pc60")
    duplicate = copy.deepcopy(protocol)
    duplicate["inputs"][1] = duplicate["inputs"][0]
    with pytest.raises(RuntimeError, match="Duplicate"):
        queue.validate_parent(duplicate, "pc60")
    protocol["inputs"].reverse()
    with pytest.raises(RuntimeError, match="official order"):
        queue.validate_parent(protocol, "pc60")
    monkeypatch.setitem(queue.PARENT_HASHES, "pc60", ("0" * 64, "0" * 64))
    with pytest.raises(RuntimeError, match="Pinned parent drift"):
        queue.read_parent("pc60")


@pytest.mark.skipif(not queue.SELECTION_ROOT.exists(), reason="Pinned local evidence unavailable")
@pytest.mark.parametrize("name", ["trace_ovss/method/image_balanced.py", "trace_ovss/method/trace/candidates.py",
    "trace_ovss/data/datasets.py", "trace_ovss/evaluation/patches.py", "tools/evolution_confirmation/run.py",
    "tools/evolution/run_cell.py", "tools/evolution_followup/standalone.py"])
def test_selected_source_continuity_rejects_fresh_self_consistent_mutation(name):
    inventory = queue.source_inventory(SOURCE)
    inherited = queue.validate_source_continuity(inventory, queue.SELECTION_ROOT)
    assert len(inherited) == 61
    inventory[name] = "0" * 64  # Even if a new snapshot self-consistently records this hash.
    with pytest.raises(RuntimeError, match="Selected-C inherited source continuity"):
        queue.validate_source_continuity(inventory, queue.SELECTION_ROOT)


@pytest.mark.skipif(not queue.SELECTION_ROOT.exists(), reason="Pinned local evidence unavailable")
def test_no_new_file_can_be_hidden_inside_inherited_math_tree():
    inventory = queue.source_inventory(SOURCE)
    inventory["trace_ovss/method/unreviewed.py"] = "f" * 64
    with pytest.raises(RuntimeError, match="continuity"):
        queue.validate_source_continuity(inventory, queue.SELECTION_ROOT)


def counter_row(arm, episode, images, cfg, position=0):
    views = images if arm == "image_C" else 0
    calls = (views + cfg["trace_view_batch_size"] - 1) // cfg["trace_view_batch_size"]
    return dict(arm=arm, episode=episode, images=images, patches=images, seconds=.01,
        order_position=position, warmup=episode == 0, steps=cfg["steps"], candidate_count=images,
        active_candidates=images, candidate_sha256="a" * 64,
        eval_forward_calls=2, eval_forward_inputs=2 * images,
        adapt_forward_calls=cfg["steps"], adapt_forward_inputs=cfg["steps"] * images,
        diagnostic_forward_calls=calls, diagnostic_forward_inputs=views,
        diagnostic_encoder_calls=calls, diagnostic_encoder_inputs=views,
        encoder_calls=2 + cfg["steps"] + calls, encoder_inputs=(2 + cfg["steps"]) * images + views)


@pytest.fixture
def miniature_suite(tmp_path, monkeypatch):
    """Synthetic preparation artifacts only; never call real prepare or GPU APIs."""
    suite = tmp_path / "suite"
    runtime = suite / "runtime"
    for tree in queue.audited.TREES:
        (runtime / tree).mkdir(parents=True)
        (runtime / tree / "synthetic.py").write_text("# synthetic source\n")
    for folder in ("configs", "parents", "selection", "manifests", "logs"):
        (suite / folder).mkdir()
    monkeypatch.setattr(queue, "ROOT", runtime)
    monkeypatch.setattr(queue, "worker", lambda: worker)
    monkeypatch.setattr(queue, "selection_proof_files", lambda root: set())
    # Historical continuity itself is tested above against the actual sealed proof.
    monkeypatch.setattr(queue, "validate_source_continuity", lambda inventory, root: {})
    monkeypatch.chdir(runtime)
    for dataset in queue.DATASETS:
        monkeypatch.setitem(queue.POPULATIONS, dataset, 5)
    plan = queue.read_json(SOURCE / "docs/image-full-plan.json")
    plan["populations"] = dict(queue.POPULATIONS)
    queue.write_json(suite / "frozen_plan.json", plan)
    monkeypatch.setattr(queue, "PLAN_SHA256", queue.sha256(suite / "frozen_plan.json"))
    hashes = queue.source_inventory(runtime)
    entries = queue.cells()
    checkpoint = dict(path="/synthetic/checkpoint", sha256="c" * 64, bytes=1, mtime_ns=1)
    for entry in entries:
        dataset = entry["dataset"]
        folder = suite / "parents" / dataset
        folder.mkdir()
        inputs = tmp_path / "inputs" / dataset
        inputs.mkdir(parents=True)
        rows = []
        for i in range(5):
            image, label = inputs / f"{i:06d}.jpg", inputs / f"{i:06d}.png"
            image.write_bytes(b"synthetic RGB identity")
            Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).save(label)
            rows.append(dict(image=str(image), label=str(label), image_sha256=queue.sha256(image),
                label_metadata=[label.stat().st_size, label.stat().st_mtime_ns]))
        cfg = config(2)
        cfg.update(resize=[224, 224], patch_size=[224, 224])
        queue.write_json(suite / entry["config"], cfg)
        vocabulary = dict(classes=["a", "b"], queries=["a", "b"], query_to_class=None, reduce_zero_label=False)
        parent = dict(dataset=dataset, seed=0, images=5, available_images=5, full_split=True,
                      inputs=rows, config=cfg, checkpoint=checkpoint, **vocabulary)
        queue.write_json(folder / "prediction_protocol.json", parent)
        parent_sha = queue.sha256(folder / "prediction_protocol.json")
        monkeypatch.setitem(queue.PARENT_HASHES, dataset, (parent_sha, "s" * 64))
        queue.write_json(folder / "parent_receipt.json", dict(protocol_path=str(queue.parent_directory(dataset) / "prediction_protocol.json"),
            protocol_sha256=parent_sha, seal_path=str(queue.parent_directory(dataset) / "PREDICTIONS_SEALED.json"), seal_sha256="s" * 64,
            complete=True, gt_contents_opened=False, images=5, sealed_protocol_sha256=parent_sha))
        manifest = dict(format_version=1, worker="image_full", stage="full", dataset=dataset, seed=0, order_seed=None,
            max_images=5, arms=queue.ARMS.copy(), arm_order="alternating", suite_root=str(suite), config=cfg,
            config_path=str(suite / entry["config"]), config_sha256=queue.sha256(suite / entry["config"]),
            panel=dict(inputs=rows, ordered_ids=[r["image"] for r in rows], full_split=True, **vocabulary),
            checkpoint=checkpoint, source_sha256={str(runtime / k): v for k, v in hashes.items()},
            selection_sha256=queue.SELECTION_SHA256, parent_protocol_sha256=parent_sha, parent_seal_sha256="s" * 64,
            plan_sha256=queue.PLAN_SHA256, timing_scope=worker.TIMING_SCOPE, preupdate_prediction_status=worker.PREUPDATE_STATUS)
        queue.write_json(suite / entry["manifest"], manifest)
        entry["manifest_sha256"] = queue.sha256(suite / entry["manifest"])
    prepared = dict(format_version=1, worker="image_full_queue", suite_root=str(suite), runtime_root=str(runtime),
        runtime_hashes=hashes, plan_sha256=queue.PLAN_SHA256, python=sys.executable, python_sha256=queue.sha256(sys.executable),
        cells=entries, arms=queue.ARMS.copy(), selected_mode="C", selection_sha256=queue.SELECTION_SHA256,
        exposure_note=queue.EXPOSURE, child_receipts=[])
    prepared["artifact_sha256"] = queue.inventory(suite, ("configs", "parents", "selection", "manifests", "logs"))
    prepared["artifact_sha256"]["frozen_plan.json"] = queue.sha256(suite / "frozen_plan.json")
    queue.write_json(suite / "SUITE_PREPARED.json", prepared)
    digest = queue.sha256(suite / "SUITE_PREPARED.json")
    args = SimpleNamespace(suite=suite, suite_sha256=digest)
    events, hooks = [], {}

    def admission(arguments):
        entry = next(e for e in prepared["cells"] if e["dataset"] == arguments.dataset)
        return suite, entry, queue.read_json(suite / entry["manifest"])

    monkeypatch.setattr(worker, "admission", admission)
    monkeypatch.setattr(worker, "verify_inputs", lambda manifest: None)

    def child(root, runtime_hashes, python, arguments, label, *, cpu=False):
        phase = arguments[1]
        dataset = arguments[arguments.index("--dataset") + 1]
        entry = next(e for e in entries if e["dataset"] == dataset)
        manifest = queue.read_json(suite / entry["manifest"])
        cfg = manifest["config"]
        events.append((phase, dataset, cpu))
        if hooks.get("before"):
            hooks["before"](phase, dataset)
        if phase == "gate":
            path = suite / f"gates/{dataset}.json"
            claim = worker.confirmation.claim_gate_attempt(path, entry["manifest_sha256"])
            comparisons = [dict(candidate_masks_equal=True, reset_equal=True, candidates=2, changed_label_pixels=0,
                max_support_abs_difference=0., max_coefficient_abs_difference=0., max_loss_abs_difference=0., max_gradient_abs_difference=0.,
                actual_arm_counts={a: counter_row(a, ep, 2, cfg) for a in queue.ARMS}) for ep in range(2)]
            queue.write_json(path, dict(passed=True, dataset=dataset, images=4, gt_contents_opened=False,
                manifest_sha256=entry["manifest_sha256"], attempt_sha256=queue.sha256(claim), configured_batch_size=2,
                batch_images=[2, 2], comparisons=comparisons, suite_prepared_sha256=digest,
                source_sha256=manifest["source_sha256"], checkpoint=checkpoint, ordered_ids=manifest["panel"]["ordered_ids"][:4]))
        elif phase == "predict":
            assert not cpu and not (suite / queue.BARRIER).exists()
            output = suite / entry["output"]
            output.mkdir(parents=True)
            queue.write_json(output / "prediction_protocol.json", dict(manifest=manifest, manifest_path=str(suite / entry["manifest"]),
                manifest_sha256=entry["manifest_sha256"], suite_prepared_sha256=digest, selection_sha256=queue.SELECTION_SHA256))
            queue.write_json(output / "initial_state.json", {})
            rows = []
            for arm in queue.ARMS:
                (output / "predictions" / arm).mkdir(parents=True)
                for i in range(5):
                    np.save(output / f"predictions/{arm}/{i:06d}.npy", np.array([[0, 1], [1, 0]], dtype=np.uint16), allow_pickle=False)
            (output / "episodes").mkdir()
            for episode, start in enumerate(range(0, 5, 2)):
                for position, arm in enumerate(worker.prior.arm_order(queue.ARMS, episode, "alternating")):
                    row = counter_row(arm, episode, min(2, 5 - start), cfg, position)
                    rows.append(row)
                    queue.write_json(output / f"episodes/{episode:06d}.{arm}.json", dict(row, candidates=[]))
            (output / "cost.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            queue.write_json(output / "prediction_result.json", dict(status="predicted", images=5, full_split=True,
                costs={a: {k: sum(r[k] for r in rows if r["arm"] == a) for k in worker.COUNTERS} for a in queue.ARMS},
                timings=worker.prior.timing_summary(rows, queue.ARMS), timing_scope=worker.TIMING_SCOPE))
            queue.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False, images=5,
                arms=queue.ARMS.copy(), suite_prepared_sha256=digest,
                artifact_sha256={n: queue.sha256(output / n) for n in sorted(worker.artifacts(manifest))}))
        elif phase == "score":
            assert cpu and (suite / queue.BARRIER).exists()
            worker.score(SimpleNamespace(suite=suite, suite_sha256=digest, dataset=dataset))
        else:
            raise AssertionError("No preparation, selection, search or retry phase admitted")
        return dict(phase=phase, dataset=dataset, cpu_only=cpu, exit_code=0)

    monkeypatch.setattr(queue, "child", child)
    return SimpleNamespace(suite=suite, args=args, prepared=prepared, events=events, hooks=hooks)


def test_full_bounded_queue_six_gates_six_predict_then_six_cpu_scores_and_18_rows(miniature_suite, monkeypatch):
    fixture = miniature_suite
    original = Path.read_bytes

    def guard(path):
        if "/inputs/" in str(path) and path.suffix == ".png":
            assert (fixture.suite / queue.BARRIER).exists(), "Evaluation GT opened before global barrier"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    queue.run(fixture.args)
    assert [phase for phase, _, _ in fixture.events] == ["gate"] * 6 + ["predict"] * 6 + ["score"] * 6
    assert [dataset for _, dataset, _ in fixture.events[:6]] == list(queue.DATASETS)
    seal = queue.read_json(fixture.suite / "SUITE_SEALED.json")
    assert seal["complete"] and seal["formal_cells"] == 6 and seal["arm_results"] == 18 and seal["passed_gt_free_gates"] == 6
    assert seal["service_success_verified"] is False
    queue.check_hashes(fixture.suite, seal["artifact_sha256"])
    with (fixture.suite / "rows.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18 and {r["arm"] for r in rows} == set(queue.ARMS)
    assert all(float(r["miou_unrounded"]) == 100. for r in rows)
    assert all(r["measured_images"] == "3" for r in rows)
    with pytest.raises(RuntimeError, match="no retry/resume"):
        queue.run(fixture.args)


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "reorder", "arms", "config", "checkpoint", "order", "parentpin", "source"])
def test_prepared_exact_admission_rejects_self_resealed_changes(miniature_suite, mutation):
    fixture = miniature_suite
    prepared = copy.deepcopy(fixture.prepared)
    entry = prepared["cells"][0]
    if mutation in ("duplicate", "extra", "reorder", "arms"):
        if mutation == "duplicate":
            prepared["cells"][1] = copy.deepcopy(entry)
        elif mutation == "extra":
            prepared["cells"].append(copy.deepcopy(entry))
        elif mutation == "reorder":
            prepared["cells"].reverse()
        else:
            entry["arms"] = ["image_T"]
    else:
        path = fixture.suite / entry["manifest"]
        manifest = queue.read_json(path)
        if mutation == "config":
            manifest["config"]["lr"] *= 2
        elif mutation == "checkpoint":
            manifest["checkpoint"]["sha256"] = "0" * 64
        elif mutation == "order":
            manifest["panel"]["inputs"].reverse()
            manifest["panel"]["ordered_ids"].reverse()
        elif mutation == "parentpin":
            manifest["parent_seal_sha256"] = "0" * 64
        else:
            manifest["source_sha256"] = {}
        replace_json(path, manifest)
        entry["manifest_sha256"] = queue.sha256(path)
        prepared["artifact_sha256"][entry["manifest"]] = queue.sha256(path)
    with pytest.raises(RuntimeError):
        queue.validate_panels(fixture.suite, prepared)


def test_prepared_pin_and_relocation_rejected(miniature_suite):
    fixture = miniature_suite
    with pytest.raises(RuntimeError, match="prepared-suite SHA"):
        queue.verify_prepared(fixture.suite, "0" * 64)
    relocated = fixture.suite.with_name("relocated")
    shutil.copytree(fixture.suite, relocated)
    with pytest.raises(RuntimeError, match="root"):
        queue.verify_prepared(relocated, fixture.args.suite_sha256)


@pytest.mark.skipif(not queue.SELECTION_ROOT.exists(), reason="Pinned local evidence unavailable")
def test_admission_rejects_selected_math_drift_even_with_matching_new_self_inventory(miniature_suite, monkeypatch):
    fixture = miniature_suite
    inventory = queue.audited.source_inventory(SOURCE)
    inventory["trace_ovss/method/image_balanced.py"] = "0" * 64
    prepared = copy.deepcopy(fixture.prepared)
    prepared["runtime_hashes"] = inventory
    replace_json(fixture.suite / "SUITE_PREPARED.json", prepared)
    pinned = queue.sha256(fixture.suite / "SUITE_PREPARED.json")
    # Simulate a freshly frozen runtime whose own file hashes all agree.
    monkeypatch.setattr(queue, "source_inventory", lambda root: inventory)
    monkeypatch.setattr(queue, "validate_source_continuity",
                        lambda hashes, root: SOURCE_CONTINUITY(hashes, queue.SELECTION_ROOT))
    with pytest.raises(RuntimeError, match="Selected-C inherited source continuity"):
        queue.verify_prepared(fixture.suite, pinned)


@pytest.mark.parametrize("failure_phase", ["gate", "predict", "score"])
def test_child_failure_is_terminal_no_resume_or_suite_seal(miniature_suite, failure_phase):
    fixture = miniature_suite

    def fail(phase, dataset):
        if phase == failure_phase:
            raise RuntimeError("synthetic child failure")

    fixture.hooks["before"] = fail
    with pytest.raises(RuntimeError, match="synthetic child failure"):
        queue.run(fixture.args)
    assert queue.read_json(fixture.suite / "queue_state.json")["phase"] == "failed"
    assert not (fixture.suite / "SUITE_SEALED.json").exists()
    assert (fixture.suite / "EXECUTION_STARTED.json").exists()
    with pytest.raises(RuntimeError, match="no retry/resume"):
        queue.run(fixture.args)


def test_early_score_artifact_blocks_prediction_without_gt_read(miniature_suite):
    fixture = miniature_suite

    def poison(phase, dataset):
        if phase == "predict" and dataset == queue.DATASETS[0]:
            path = fixture.suite / queue.cells()[-1]["output"]
            path.mkdir(parents=True)
            queue.write_json(path / "SCORING_STARTED.json", {})

    fixture.hooks["before"] = poison
    with pytest.raises(RuntimeError, match="before stage prediction barrier"):
        queue.run(fixture.args)
    assert not (fixture.suite / queue.BARRIER).exists()
    assert not any(phase == "score" for phase, _, _ in fixture.events)


def test_prediction_drift_at_score_barrier_prevents_first_gt_open(miniature_suite, monkeypatch):
    fixture = miniature_suite

    def corrupt(phase, dataset):
        if phase == "score":
            path = fixture.suite / fixture.prepared["cells"][-1]["output"] / "predictions/image_C/000000.npy"
            path.write_bytes(b"changed")

    fixture.hooks["before"] = corrupt
    original = Path.read_bytes

    def guard(path):
        assert not ("/inputs/" in str(path) and path.suffix == ".png"), "GT must remain unopened"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    with pytest.raises(RuntimeError, match="Frozen artifact changed"):
        queue.run(fixture.args)
    assert not any((fixture.suite / e["output"] / "SCORING_STARTED.json").exists() for e in fixture.prepared["cells"])


def test_gate_started_marker_tampering_fails(miniature_suite):
    fixture = miniature_suite

    def mutate(phase, dataset):
        if phase == "predict":
            path = fixture.suite / f"gates/{queue.DATASETS[0]}.json.STARTED.json"
            path.write_text("{}")

    fixture.hooks["before"] = mutate
    with pytest.raises(RuntimeError, match="Frozen artifact changed"):
        queue.run(fixture.args)
    assert not (fixture.suite / queue.BARRIER).exists()
