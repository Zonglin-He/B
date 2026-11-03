"""Synthetic CPU-only orchestration tests; no weights, datasets, GPU, or services."""

import copy
import csv
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from tools.evolution_confirmation import queue


SOURCE = queue.ROOT


def replace_json(path, value):
    path.unlink()
    queue.write_json(path, value)


def metric(value):
    intersection, union = int(round(value * 10000)), 1000000
    return dict(miou_unrounded=intersection / union * 100, intersection=[intersection], union=[union])


def selection_results(margins=None, costs=None):
    margins = margins or {"T": [1., 1.], "C": [.5, .5], "min": [0., 0.]}
    costs = costs or {"T": 10, "C": 10, "min": 20}
    return {d: dict(stage="calibration", images=128, order_seed=None, arm_results={"image_confidence": metric(50),
        **{"image_" + m: metric(50 + margins[m][i]) for m in queue.MODES}},
        costs={"image_" + m: dict(diagnostic_encoder_inputs=costs[m]) for m in queue.MODES})
        for i, d in enumerate(queue.DATASETS)}


@pytest.fixture
def prepared_suite(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for tree in queue.audited.TREES:
        (source / tree).mkdir(parents=True)
    required = ("tools/evolution_confirmation/queue.py", "tools/evolution_confirmation/run.py",
                "tools/evolution_followup/queue.py", "tests/test_confirmation_queue.py", "trace_ovss/configs.py")
    for relative in required:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic frozen source only\n")
    monkeypatch.setattr(queue, "ROOT", source)
    monkeypatch.setattr(queue.audited, "ROOT", source)
    monkeypatch.setattr(queue.audited, "POPULATIONS", {d: 2 for d in queue.DATASETS})
    datasets, exclusions = {}, {}
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for dataset in queue.DATASETS:
        ids = [f"{dataset}:{i:06d}" for i in range(384)]
        ids.sort(key=lambda identity: (queue.hashlib.sha256(
            f"trace-confirmation-v1\n20261029\n{dataset}\n{identity}\n".encode()).hexdigest(), identity))
        rows = []
        for i, identity in enumerate(ids):
            image, label = inputs / f"{dataset}-{i}.rgb", inputs / f"{dataset}-{i}.gt"
            image.write_bytes(identity.encode())
            label.write_bytes(b"ANNOTATIONS MUST NOT BE READ BY QUEUE")
            stat = label.stat()
            rows.append(dict(official_id=identity, image=str(image), label=str(label), image_sha256=queue.sha256(image),
                label_metadata=dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                    device=stat.st_dev, inode=stat.st_ino, contents_read=False)))
        datasets[dataset] = dict(calibration=rows[:128], confirmation=rows[128:], eligible_ids=ids,
            selected_ids_sha256={s: queue.canonical_sha([r["official_id"] for r in rows[a:b]])
                                for s, a, b in (("calibration", 0, 128), ("confirmation", 128, 384))})
        exclusions[dataset] = dict(union_ids=[])
    bundle = dict(schema_version=1, kind="prospective_known_history_training_source_cohorts", datasets=datasets,
        exclusions=exclusions, selection=dict(seed=20261029, calibration_size=128, confirmation_size=256,
            annotation_pixels_read=False, class_coverage_selection=False),
        audit=dict(global_never_seen_certified=False, historical_index_exhaustive=False))
    cohort_path = tmp_path / "cohorts.json"
    queue.write_json(cohort_path, bundle)
    parents = tmp_path / "parent-source"
    checkpoint = dict(path="/synthetic/checkpoint.pt", sha256="a" * 64, bytes=123, mtime_ns=1)
    for dataset in queue.DATASETS:
        directory = parents / "cells/mechanism" / dataset
        directory.mkdir(parents=True)
        protocol = dict(dataset=dataset, seed=0, full_split=True, images=2, inputs=[{}, {}], checkpoint=checkpoint,
            config=dict(resize=[224, 224], patch_size=[224, 224], batch_size=2, lr=.009, steps=10))
        queue.write_json(directory / "prediction_protocol.json", protocol)
        queue.write_json(directory / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
            images=2, artifact_sha256={"prediction_protocol.json": queue.sha256(directory / "prediction_protocol.json")}))

    def verify_manifest(path, expected_hash):
        queue.require(queue.sha256(path) == expected_hash, "Manifest SHA mismatch")
        return queue.read_json(path)

    def verify_prediction(output):
        seal = queue.read_json(output / "PREDICTIONS_SEALED.json")
        queue.check_hashes(output, seal["artifact_sha256"])
        return queue.read_json(output / "prediction_protocol.json")

    fake_worker = SimpleNamespace(checked_sources=lambda: {str(source / p): queue.sha256(source / p) for p in required},
        verify_manifest=verify_manifest, validate_manifest=lambda m: None, verify_prediction_seal=verify_prediction)
    monkeypatch.setattr(queue, "worker", lambda: fake_worker)
    events, hooks = [], {}

    def child(suite, hashes, python, arguments, label, *, cpu=False):
        assert queue.source_inventory(suite / "runtime") == hashes
        kind = arguments[1]
        def arg(name):
            return arguments[arguments.index(name) + 1]
        events.append((kind, label, cpu))
        if hooks.get("before"):
            hooks["before"](kind, arguments, suite)
        if kind == "prepare":
            assert cpu
            path = Path(arg("--manifest"))
            dataset, stage = arg("--dataset"), arg("--stage")
            order = int(arg("--order-seed")) if "--order-seed" in arguments else None
            rows = copy.deepcopy(bundle["datasets"][dataset][stage])
            if order is not None:
                rows = rows[1:] + rows[:1] if order == 20261030 else list(reversed(rows))
            config_path = Path(arg("--config"))
            manifest = dict(dataset=dataset, stage=stage, seed=0, max_images=len(rows), arms=queue.ARMS,
                order_seed=order, suite_root=str(suite), cohorts_path=str(suite / "frozen_cohorts.json"),
                cohort_sha256=queue.sha256(suite / "frozen_cohorts.json"), checkpoint=checkpoint,
                config=queue.read_json(config_path), config_path=str(config_path), config_sha256=queue.sha256(config_path),
                source_sha256={str(suite / "runtime" / p): hashes[p] for p in required},
                panel=dict(inputs=rows, ordered_ids=[r["official_id"] for r in rows], full_split=False),
                timing_scope="synthetic_model_only", exposure_scope=queue.EXPOSURE)
            queue.write_json(path, manifest)
        elif kind == "gate":
            assert not cpu
            manifest = queue.read_json(arg("--manifest"))
            output = Path(arg("--output"))
            claim = output.with_name(output.name + ".STARTED.json")
            queue.write_json(claim, dict(status="started", manifest_sha256=arg("--manifest-sha256")))
            queue.write_json(output, dict(passed=True, gt_contents_opened=False, images=4,
                dataset=manifest["dataset"], manifest_sha256=arg("--manifest-sha256"), attempt_sha256=queue.sha256(claim)))
        elif kind == "predict":
            assert not cpu
            manifest_path = Path(arg("--manifest"))
            manifest = queue.read_json(manifest_path)
            output = Path(arg("--output"))
            if manifest["stage"] == "confirmation":
                assert (suite / "SELECTION_SEALED.json").is_file()
            output.mkdir(parents=True)
            rows = []
            for arm in queue.ARMS:
                factor = 0 if arm in ("image_uniform", "image_confidence") else 1 if arm in ("image_T", "image_C") else 2
                rows += [dict(arm=arm, episode=i // 2, images=2, diagnostic_encoder_inputs=2 * factor, seconds=1.)
                         for i in range(0, manifest["max_images"], 2)]
            with (output / "cost.jsonl").open("x") as stream:
                stream.writelines(json.dumps(r) + "\n" for r in rows)
            costs = {arm: dict(diagnostic_encoder_inputs=sum(r["diagnostic_encoder_inputs"] for r in rows if r["arm"] == arm))
                     for arm in queue.ARMS}
            timing = dict(measured_images=manifest["max_images"] - 2, seconds=manifest["max_images"] / 2 - 1,
                measured_episodes=manifest["max_images"] // 2 - 1, seconds_per_image=.5, images_per_second=2.,
                warmup_episode=0, accuracy_includes_warmup=True)
            queue.write_json(output / "prediction_result.json", dict(costs=costs,
                timings={arm: timing for arm in queue.ARMS}, timing_scope="synthetic_model_only"))
            queue.write_json(output / "prediction_protocol.json", dict(manifest=manifest, manifest_path=str(manifest_path),
                manifest_sha256=arg("--manifest-sha256"), selection=(None if manifest["stage"] == "calibration" else
                    dict(path=str(suite / "SELECTION_SEALED.json"), sha256=queue.sha256(suite / "SELECTION_SEALED.json"),
                         selected_mode=queue.read_json(suite / "SELECTION_SEALED.json")["selected_mode"]))))
            queue.write_json(output / "PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
                images=manifest["max_images"], arms=queue.ARMS, artifact_sha256={n: queue.sha256(output / n)
                    for n in ("prediction_protocol.json", "prediction_result.json", "cost.jsonl")}))
        elif kind == "score":
            assert cpu
            output = Path(arg("--output"))
            manifest = queue.read_json(output / "prediction_protocol.json")["manifest"]
            barrier = suite / f"ALL_{manifest['stage'].upper()}_PREDICTIONS_SEALED.json"
            assert barrier.is_file()
            assert len(queue.read_json(barrier)["prediction_seal_sha256"]) == (2 if manifest["stage"] == "calibration" else 6)
            pred = queue.read_json(output / "prediction_result.json")
            pred_hash = queue.sha256(output / "PREDICTIONS_SEALED.json")
            values = dict(episode_min=48, image_uniform=47, image_confidence=50, image_T=49, image_C=50.01, image_min=50.03)
            queue.write_json(output / "result.json", dict(status="completed", dataset=manifest["dataset"],
                stage=manifest["stage"], order_seed=manifest["order_seed"], images=manifest["max_images"], full_split=False,
                cohort_sha256=manifest["cohort_sha256"], arm_results={a: metric(v) for a, v in values.items()},
                costs=pred["costs"], timings=pred["timings"], prediction_seal_sha256=pred_hash,
                label_contents_first_opened_after_all_stage_predictions_sealed=True))
            queue.write_json(output / "SCORING_STARTED.json", dict(prediction_seal_sha256=pred_hash))
            queue.write_json(output / "score_inputs.json", dict(label_sha256={}))
            (output / "scores_by_episode.jsonl").write_text("Synthetic CPU scores, no GT opened\n")
            queue.write_json(output / "SCORED_SEALED.json", dict(prediction_seal_sha256=pred_hash,
                artifact_sha256={n: queue.sha256(output / n) for n in queue.SCORING_FILES[:-1]}))
        else:
            pytest.fail(f"Unexpected worker call {kind}")
        if hooks.get("after"):
            hooks["after"](kind, arguments, suite)
        log = suite / f"logs/{label}.log"
        log.write_text("Synthetic child only\n")
        receipt = dict(log=str(log.relative_to(suite)), log_sha256=queue.sha256(log), cpu_only=cpu, exit_code=0)
        path = suite / f"logs/{label}.receipt.json"
        queue.write_json(path, receipt)
        return dict(receipt, receipt=str(path.relative_to(suite)), receipt_sha256=queue.sha256(path))

    monkeypatch.setattr(queue, "child", child)
    suite = tmp_path / "suite"
    args = SimpleNamespace(source_root=source, plan=SOURCE / "docs/confirmation-image-budget-plan.json",
        cohorts=cohort_path, parent_root=parents, python=Path(sys.executable), output=suite)
    queue.prepare(args)
    prepared = queue.read_json(suite / "SUITE_PREPARED.json")
    verify = queue.verify_prepared
    monkeypatch.setattr(queue, "verify_prepared", lambda suite, digest, **kw: verify(suite, digest))
    events.clear()
    return SimpleNamespace(suite=suite, prepared=prepared, args=args, events=events, hooks=hooks,
                           run_args=SimpleNamespace(suite=suite, suite_sha256=queue.sha256(suite / "SUITE_PREPARED.json")))


def test_prepare_eight_manifests_two_configs_and_no_live_parent_dependency(prepared_suite):
    f = prepared_suite
    assert len(f.prepared["cells"]) == 8
    assert len(list((f.suite / "configs").iterdir())) == 2
    assert all(queue.read_json(f.suite / e["manifest"])["config"]["lr"] == .009 for e in f.prepared["cells"])
    shutil.rmtree(f.args.parent_root)
    queue.verify_prepared(f.suite, f.run_args.suite_sha256)
    with pytest.raises(RuntimeError, match="no overwrite"):
        queue.prepare(f.args)


def test_complete_order_barriers_selection_and_all_48_rows(prepared_suite):
    f = prepared_suite
    queue.run(f.run_args)
    assert [e[0] for e in f.events] == ["gate"] * 2 + ["predict"] * 2 + ["score"] * 2 + ["predict"] * 6 + ["score"] * 6
    selection = queue.read_json(f.suite / "SELECTION_SEALED.json")
    assert selection["selected_mode"] == "C"
    assert selection["rule"] == queue.read_json(f.suite / "frozen_plan.json")["selection"]
    assert set(selection["calibration_score_sha256"]) == {e["output"] for e in f.prepared["cells"][:2]}
    for stage, count in (("calibration", 2), ("confirmation", 6)):
        barrier = queue.read_json(f.suite / f"ALL_{stage.upper()}_PREDICTIONS_SEALED.json")
        assert barrier["complete"] is True and barrier["stage"] == stage
        assert len(barrier["prediction_seal_sha256"]) == count
    with (f.suite / "rows.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48 and {r["arm"] for r in rows} == set(queue.ARMS)
    assert any(r["arm"] == "image_T" and float(r["miou_unrounded"]) < 50 for r in rows)
    assert sum(r["is_selected_probe"] == "True" for r in rows) == 8
    seal = queue.read_json(f.suite / "SUITE_SEALED.json")
    assert seal["arm_results"] == 48 and seal["passed_gt_free_gates"] == 2
    queue.check_hashes(f.suite, seal["artifact_sha256"])
    assert queue.read_json(f.suite / "queue_state.json")["phase"] == "completed"
    with pytest.raises(RuntimeError, match="retry"):
        queue.run(f.run_args)


@pytest.mark.parametrize("target", ["runtime/tools/evolution_confirmation/run.py", "frozen_cohorts.json", "configs/pc60.json"])
def test_changed_source_cohort_config_rejected_before_any_child(prepared_suite, target):
    f = prepared_suite
    path = f.suite / target
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed"):
        queue.run(f.run_args)
    assert not f.events


def test_reviewed_hash_and_frozen_cwd_required(prepared_suite):
    f = prepared_suite
    with pytest.raises(RuntimeError, match="hash mismatch"):
        queue.verify_prepared(f.suite, "0" * 64)
    with pytest.raises(RuntimeError, match="frozen snapshot cwd"):
        # Access the real verifier retained in the wrapper closure.
        f_verify = queue.verify_prepared.__closure__[0].cell_contents
        f_verify(f.suite, f.run_args.suite_sha256, frozen=True)


@pytest.mark.parametrize("kind,number", [("gate", 1), ("predict", 1), ("predict", 2),
    ("score", 1), ("score", 2), ("predict", 3), ("predict", 8), ("score", 3), ("score", 8)])
def test_child_failure_stops_no_retry_or_false_completion(prepared_suite, kind, number):
    f = prepared_suite
    counts = {}
    def fail(current, arguments, suite):
        counts[current] = counts.get(current, 0) + 1
        if current == kind and counts[current] == number:
            raise RuntimeError("Synthetic child failure")
    f.hooks["before"] = fail
    with pytest.raises(RuntimeError, match="Synthetic child failure"):
        queue.run(f.run_args)
    assert queue.read_json(f.suite / "queue_state.json")["phase"] == "failed"
    assert not (f.suite / "SUITE_SEALED.json").exists()
    before = len(f.events)
    with pytest.raises(RuntimeError, match="retry"):
        queue.run(f.run_args)
    assert len(f.events) == before


@pytest.mark.parametrize("stage", ["calibration", "confirmation"])
def test_gt_early_read_marker_fails_before_scoring(prepared_suite, stage):
    f = prepared_suite
    def inject(kind, arguments, suite):
        if kind == "predict":
            output = Path(arguments[arguments.index("--output") + 1])
            if f"/{stage}/" in str(output):
                queue.write_json(output / "score_inputs.json", {})
    f.hooks["after"] = inject
    with pytest.raises(RuntimeError, match="GT/scoring artifacts"):
        queue.run(f.run_args)
    assert sum(e[0] == "score" for e in f.events) == (0 if stage == "calibration" else 2)


def test_confirmation_before_selection_is_rejected(prepared_suite):
    f = prepared_suite
    def inject(kind, arguments, suite):
        if kind == "score":
            (suite / f.prepared["cells"][2]["output"]).mkdir(parents=True)
    f.hooks["after"] = inject
    with pytest.raises(RuntimeError, match="before calibration selection"):
        queue.run(f.run_args)
    assert not (f.suite / "SELECTION_SEALED.json").exists()


def test_selection_tamper_after_confirmation_stops(prepared_suite):
    f = prepared_suite
    def inject(kind, arguments, suite):
        if kind == "predict" and (suite / "SELECTION_SEALED.json").exists():
            path = suite / "SELECTION_SEALED.json"
            selection = queue.read_json(path)
            replace_json(path, dict(selection, selected_mode="T"))
    f.hooks["after"] = inject
    with pytest.raises(RuntimeError, match="Selection changed|selection linkage changed"):
        queue.run(f.run_args)
    assert sum(e[0] == "score" for e in f.events) == 2


@pytest.mark.parametrize("target", ["image", "label"])
def test_input_identity_drift_rejected_before_child(prepared_suite, target):
    f = prepared_suite
    bundle = queue.read_json(f.suite / "frozen_cohorts.json")
    path = Path(bundle["datasets"]["pc60"]["calibration"][0][target])
    path.write_bytes(b"changed payload")
    with pytest.raises(RuntimeError, match="image changed|GT metadata changed"):
        queue.run(f.run_args)
    assert not f.events


def test_queue_never_opens_annotation_contents(prepared_suite, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.suffix != ".gt", "Queue opened ground-truth contents"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    queue.run(prepared_suite.run_args)


@pytest.mark.parametrize("change", ["count", "order", "arm", "config_override", "escape", "missing_source", "missing_small"])
def test_rehashed_admission_changes_rejected(prepared_suite, change):
    f = prepared_suite
    prepared = copy.deepcopy(f.prepared)
    if change == "count":
        prepared["cells"].pop()
    elif change == "order":
        prepared["cells"][-1]["order_seed"] = 4
    elif change == "arm":
        prepared["cells"][0]["arms"] = ["image_min"]
    elif change == "escape":
        prepared["cells"][0]["output"] = "../escape"
    elif change == "missing_source":
        prepared["runtime_hashes"].pop(next(iter(prepared["runtime_hashes"])))
    elif change == "missing_small":
        prepared["artifact_sha256"].pop("configs/pc60.json")
    else:
        entry = prepared["cells"][0]
        path = f.suite / entry["manifest"]
        manifest = queue.read_json(path)
        manifest["config"]["lr"] = .1
        replace_json(path, manifest)
        entry["manifest_sha256"] = queue.sha256(path)
        prepared["artifact_sha256"][entry["manifest"]] = queue.sha256(path)
    replace_json(f.suite / "SUITE_PREPARED.json", prepared)
    with pytest.raises(RuntimeError):
        queue.verify_prepared(f.suite, queue.sha256(f.suite / "SUITE_PREPARED.json"))


@pytest.mark.parametrize("change", ["count", "overlap", "hash", "gt", "rank", "global", "selection", "exclude"])
def test_cohort_admission_bounds_and_exposure_claims(prepared_suite, change):
    bundle = queue.read_json(prepared_suite.suite / "frozen_cohorts.json")
    pc = bundle["datasets"]["pc60"]
    if change == "count":
        pc["calibration"].pop()
    elif change == "overlap":
        pc["confirmation"][0] = pc["calibration"][0]
    elif change == "hash":
        pc["calibration"][0].pop("image_sha256")
    elif change == "gt":
        bundle["selection"]["annotation_pixels_read"] = True
    elif change == "rank":
        pc["eligible_ids"].reverse()
    elif change == "global":
        bundle["audit"]["global_never_seen_certified"] = True
    elif change == "exclude":
        bundle["exclusions"]["pc60"]["union_ids"] = [pc["calibration"][0]["official_id"]]
    else:
        bundle["selection"]["seed"] = 0
    with pytest.raises((RuntimeError, KeyError)):
        queue.validate_cohorts(bundle)


@pytest.mark.parametrize("margins,costs,expected", [
    ({"T": [1, 1], "C": [10, .9], "min": [0, 0]}, {"T": 10, "C": 1, "min": 0}, "T"),
    ({"T": [1, 1], "C": [.981, .981], "min": [.95, .95]}, {"T": 20, "C": 10, "min": 0}, "C"),
    ({"T": [1, 1], "C": [.98, 2], "min": [.9, .9]}, {"T": 10, "C": 10, "min": 0}, "C"),
    ({"T": [1, 1], "C": [1, 1], "min": [1, 1]}, {"T": 10, "C": 10, "min": 10}, "T"),
    ({"T": [-1, -1], "C": [-2, 4], "min": [-3, -3]}, {"T": 10, "C": 0, "min": 0}, "T"),
    ({"T": [1, 1], "C": [.985, .985], "min": [.97, .97]}, {"T": 20, "C": 10, "min": 0}, "C"),
    ({"T": [1, 1], "C": [1, 1], "min": [1.1, 1.1]}, {"T": 0, "C": 0, "min": 20}, "min"),
])
def test_selection_worst_dataset_tie_band_actual_cost_macro_fixed_order(margins, costs, expected):
    rule = queue.read_json(SOURCE / "docs/confirmation-image-budget-plan.json")["selection"]
    mode, candidates = queue.select_mode(selection_results(margins, costs), rule)
    assert mode == expected
    assert candidates[mode]["diagnostic_encoder_inputs"] == 2 * costs[mode]


def test_selection_requires_unrounded_pooled_calibration_only():
    rule = queue.read_json(SOURCE / "docs/confirmation-image-budget-plan.json")["selection"]
    results = selection_results()
    results["pc60"]["arm_results"]["image_T"]["miou_unrounded"] = 51.01
    with pytest.raises(RuntimeError, match="Rounded/nonpooled"):
        queue.select_mode(results, rule)
    with pytest.raises(RuntimeError, match="both calibration"):
        queue.select_mode({"confirmation": {}}, rule)
    results = selection_results()
    results["pc60"]["stage"] = "confirmation"
    with pytest.raises(RuntimeError, match="rejects confirmation"):
        queue.select_mode(results, rule)


@pytest.mark.parametrize("field,value", [("passed", False), ("gt_contents_opened", True), ("images", 128),
    ("dataset", "wrong"), ("manifest_sha256", "0" * 64)])
def test_gate_false_success_scope_and_gt_rejected(prepared_suite, field, value):
    f = prepared_suite
    def corrupt(kind, arguments, suite):
        if kind == "gate":
            path = Path(arguments[arguments.index("--output") + 1])
            replace_json(path, dict(queue.read_json(path), **{field: value}))
    f.hooks["after"] = corrupt
    with pytest.raises(RuntimeError, match="gate failed"):
        queue.run(f.run_args)
    assert [e[0] for e in f.events] == ["gate"]


@pytest.mark.parametrize("change,expected", [("rounded", "Rounded/nonpooled"), ("gt", "Incomplete scored"),
    ("cost", "Actual diagnostic totals"), ("timing", "Timing scope"), ("scope", "Incomplete scored")])
def test_score_rejects_resealed_wrong_metrics_scope_costs_and_timing(prepared_suite, change, expected):
    f = prepared_suite
    def corrupt(kind, arguments, suite):
        if kind != "score":
            return
        output = Path(arguments[arguments.index("--output") + 1])
        result = queue.read_json(output / "result.json")
        predicted = queue.read_json(output / "prediction_result.json")
        if change == "rounded":
            result["arm_results"]["image_C"]["miou_unrounded"] = 50.
        elif change == "gt":
            result["label_contents_first_opened_after_all_stage_predictions_sealed"] = False
        elif change == "scope":
            result["images"] = 256
        elif change == "cost":
            result["costs"]["image_T"]["diagnostic_encoder_inputs"] -= 2
            predicted["costs"] = result["costs"]
        else:
            result["timings"]["image_T"]["measured_images"] += 2
            predicted["timings"] = result["timings"]
        replace_json(output / "prediction_result.json", predicted)
        prediction_seal = queue.read_json(output / "PREDICTIONS_SEALED.json")
        prediction_seal["artifact_sha256"]["prediction_result.json"] = queue.sha256(output / "prediction_result.json")
        replace_json(output / "PREDICTIONS_SEALED.json", prediction_seal)
        pred_hash = queue.sha256(output / "PREDICTIONS_SEALED.json")
        result["prediction_seal_sha256"] = pred_hash
        replace_json(output / "result.json", result)
        replace_json(output / "SCORING_STARTED.json", dict(prediction_seal_sha256=pred_hash))
        replace_json(output / "SCORED_SEALED.json", dict(prediction_seal_sha256=pred_hash,
            artifact_sha256={n: queue.sha256(output / n) for n in queue.SCORING_FILES[:-1]}))
    f.hooks["after"] = corrupt
    with pytest.raises(RuntimeError, match=expected):
        queue.run(f.run_args)
    assert sum(e[0] == "score" for e in f.events) == 1


def test_calibration_payload_tamper_before_selection_rejected(prepared_suite):
    f = prepared_suite
    def corrupt(kind, arguments, suite):
        if kind == "score" and sum(e[0] == "score" for e in f.events) == 2:
            first = suite / f.prepared["cells"][0]["output"] / "cost.jsonl"
            first.write_text(first.read_text() + "\n")
    f.hooks["after"] = corrupt
    with pytest.raises(RuntimeError, match="Frozen artifact changed"):
        queue.run(f.run_args)
    assert not (f.suite / "SELECTION_SEALED.json").exists()
    assert sum(e[0] == "predict" for e in f.events) == 2


def test_changed_barrier_rejected_before_next_score(prepared_suite):
    f = prepared_suite
    def corrupt(kind, arguments, suite):
        if kind == "score":
            path = suite / "ALL_CALIBRATION_PREDICTIONS_SEALED.json"
            replace_json(path, dict(queue.read_json(path), complete=False))
    f.hooks["after"] = corrupt
    with pytest.raises(RuntimeError, match="Frozen artifact changed"):
        queue.run(f.run_args)
    assert sum(e[0] == "score" for e in f.events) == 1


def test_child_spawn_error_preserves_failed_receipt(tmp_path, monkeypatch):
    for tree in queue.audited.TREES:
        (tmp_path / "runtime" / tree).mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    def fail(*args, **kwargs):
        raise OSError("Synthetic spawn failure")
    monkeypatch.setattr(queue.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failure"):
        queue.child(tmp_path, queue.source_inventory(tmp_path / "runtime"), Path("/fake/python"), ["module"], "spawn", cpu=True)
    receipt = queue.read_json(tmp_path / "logs/spawn.receipt.json")
    assert receipt["exit_code"] is None and "spawn failure" in receipt["error"]
    assert (tmp_path / "logs/spawn.log").stat().st_size == 0


def test_local_queue_lock_is_nonblocking_and_does_not_take_gpu_lock(tmp_path):
    with queue.queue_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with queue.queue_lock(tmp_path):
                pytest.fail("Concurrent queue admitted")
    with queue.queue_lock(tmp_path):
        pass


def test_source_snapshot_symlink_rejected(tmp_path):
    for tree in queue.audited.TREES:
        (tmp_path / tree).mkdir()
    (tmp_path / "tools/escape.py").symlink_to("/dev/null")
    with pytest.raises(RuntimeError, match="symlink"):
        queue.source_inventory(tmp_path)


def test_plan_immutable_no_scope_expansion():
    plan = queue.read_json(SOURCE / "docs/confirmation-image-budget-plan.json")
    queue.validate_plan(plan)
    for key in plan:
        altered = copy.deepcopy(plan)
        altered.pop(key)
        with pytest.raises(RuntimeError, match="Immutable"):
            queue.validate_plan(altered)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_child_primitive_cpu_log_bound_and_failed_exit_receipt(tmp_path, monkeypatch, exit_code):
    for tree in queue.audited.TREES:
        (tmp_path / "runtime" / tree).mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    hashes = queue.source_inventory(tmp_path / "runtime")
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            self.stdout = io.BytesIO(b"x" * (queue.audited.LOG_BYTES + 123))
        def wait(self, **kwargs):
            return exit_code
    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "main-owned")
    if exit_code:
        with pytest.raises(RuntimeError, match="Child failed"):
            queue.child(tmp_path, hashes, Path("/fake/python"), ["module", "prepare"], "child", cpu=True)
    else:
        queue.child(tmp_path, hashes, Path("/fake/python"), ["module", "prepare"], "child", cpu=True)
    receipt = queue.read_json(tmp_path / "logs/child.receipt.json")
    assert receipt["exit_code"] == exit_code and receipt["retained_bytes"] == queue.audited.LOG_BYTES
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(RuntimeError, match="no retry"):
        queue.child(tmp_path, hashes, Path("/fake/python"), ["module"], "child")


def test_wrong_import_root_and_parser_reject_overrides(monkeypatch):
    monkeypatch.setattr(queue.audited, "ROOT", Path("/wrong/snapshot"))
    with pytest.raises(RuntimeError, match="outside new frozen snapshot"):
        queue.primitives()
    for arguments in (["prepare", "--cohorts", "c", "--output", "o", "--lr", ".01"],
                      ["run", "--suite", "s"], ["run", "--suite", "s", "--suite-sha256", "a", "--resume"]):
        with pytest.raises(SystemExit):
            queue.parser().parse_args(arguments)
