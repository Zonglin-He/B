"""Bounded two-stage confirmation controller; prepare is CPU-only, run is explicit.

Systemd, its retained successful service, and GPU serialization belong to main.
There is no retry, resume, configuration search, or historical service dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.evolution_followup import queue as audited


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("pc60", "coco_object")
ARMS = ["episode_min", "image_uniform", "image_confidence", "image_T", "image_C", "image_min"]
MODES = ("T", "C", "min")
ORDER_SEEDS = (None, 20261030, 20261031)
COUNTS = {"calibration": 128, "confirmation": 256}
WORKER = "tools.evolution_confirmation.run"
PLAN_SHA256 = "8569e1f9a51ee0647fe365eef3be9902b46697b017db2c4990acae1e0388ca03"
PLAN_CANONICAL_SHA256 = "4ea3d345806ac748df50f1d0cc3f7ec86056834eb9d1e57871147c64e45f3ed4"
EXPOSURE = ("Prospective confirmation relative only to documented exclusion inventory coverage; "
            "missing historical records are not certified absent. Not globally unseen raw/source cohorts.")
SCORING_FILES = ("SCORING_STARTED.json", "scores_by_episode.jsonl", "score_inputs.json", "result.json", "SCORED_SEALED.json")
require, sha256, read_json, write_json = audited.require, audited.sha256, audited.read_json, audited.write_json
check_hashes, queue_lock = audited.check_hashes, audited.queue_lock


def primitives():
    require(audited.ROOT == ROOT, "Audited primitives imported outside new frozen snapshot")
    return audited


def worker():
    from tools.evolution_confirmation import run
    require(run.ROOT == ROOT, "Worker imported outside new frozen snapshot")
    return run


def source_inventory(root):
    return primitives().source_inventory(root)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_plan(plan):
    require(canonical_sha(plan) == PLAN_CANONICAL_SHA256, "Immutable confirmation design changed")


def cells():
    result = []
    for stage, count in COUNTS.items():
        for dataset in DATASETS:
            for order in ((None,) if stage == "calibration" else ORDER_SEEDS):
                suffix = "sorted" if order is None else str(order)
                stem = f"{dataset}.{stage}.{suffix}"
                result.append(dict(key=f"{stage}/{dataset}/{suffix}", stage=stage, dataset=dataset,
                    images=count, arms=ARMS.copy(), order_seed=order, config=f"configs/{dataset}.json",
                    manifest=f"manifests/{stem}.json", output=f"cells/{stage}/{dataset}/{suffix}"))
    return result


def state(suite, phase, active=None, *, gates=0, predictions=0, scores=0, error=None):
    temporary = suite / "queue_state.tmp"
    write_json(temporary, dict(phase=phase, active_cell=active, passed_gates=gates,
        sealed_predictions=predictions, scored_cells=scores, calibration_cells=2,
        confirmation_cells=6, calibration_predictions=min(predictions, 2),
        confirmation_predictions=max(predictions - 2, 0), calibration_scores=min(scores, 2),
        confirmation_scores=max(scores - 2, 0), total_cells=8, updated_unix=time.time(), error=error))
    temporary.replace(suite / "queue_state.json")


def child(suite, runtime_hashes, python, arguments, label, *, cpu=False):
    """Audited foreground/log-tail pattern, with an exit receipt even on failure."""
    require(source_inventory(suite / "runtime") == runtime_hashes, "Frozen runtime changed before child")
    log, receipt_path = suite / f"logs/{label}.log", suite / f"logs/{label}.receipt.json"
    require(not log.exists() and not receipt_path.exists(), "Child already attempted; no retry")
    env = dict(os.environ, PYTHONPATH=str(suite / "runtime"), PYTHONDONTWRITEBYTECODE="1")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    argv = [str(python), "-B", "-m", *arguments]
    tail, total, process, code, error = bytearray(), 0, None, None, None
    try:
        process = subprocess.Popen(argv, cwd=suite / "runtime", env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            block = process.stdout.read(64 * 1024)
            if not block:
                break
            total += len(block)
            tail.extend(block)
            del tail[:-audited.LOG_BYTES]
        code = process.wait()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:2000]
        if process is not None:
            process.terminate()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        with log.open("xb") as stream:
            stream.write(tail)
        receipt = dict(argv=argv, log=str(log.relative_to(suite)), log_sha256=sha256(log),
            output_bytes=total, retained_bytes=len(tail), exit_code=code, cpu_only=cpu, error=error)
        write_json(receipt_path, receipt)
    require(code == 0, f"Child failed ({code}): {label}; no retry")
    return dict(receipt, receipt=str(receipt_path.relative_to(suite)), receipt_sha256=sha256(receipt_path))


def read_parent(root, dataset):
    directory = Path(root) / "cells/mechanism" / dataset
    protocol_path, seal_path = directory / "prediction_protocol.json", directory / "PREDICTIONS_SEALED.json"
    protocol, seal = read_json(protocol_path), read_json(seal_path)
    require(seal.get("complete") is True and seal.get("gt_contents_opened") is False
            and seal["artifact_sha256"]["prediction_protocol.json"] == sha256(protocol_path),
            "Parent protocol not linked to complete GT-free prediction seal")
    require(protocol["dataset"] == dataset and protocol["seed"] == 0 and protocol["full_split"] is True
            and protocol["images"] == seal["images"] == audited.POPULATIONS[dataset]
            and len(protocol["inputs"]) == protocol["images"], "Wrong inherited parent scope")
    require(protocol["config"]["resize"] == protocol["config"]["patch_size"] == [224, 224],
            "Inherited preprocessing changed")
    checkpoint = protocol["checkpoint"]
    require(set(("path", "sha256", "bytes", "mtime_ns")) <= checkpoint.keys()
            and Path(checkpoint["path"]).is_absolute() and len(checkpoint["sha256"]) == 64,
            "Incomplete inherited checkpoint identity")
    return protocol, dict(protocol_path=str(protocol_path.resolve()), protocol_sha256=sha256(protocol_path),
                         seal_path=str(seal_path.resolve()), seal_sha256=sha256(seal_path))


def validate_cohorts(bundle):
    require(bundle.get("kind") == "prospective_known_history_training_source_cohorts"
            and bundle.get("schema_version") == 1, "Missing training-source cohort provenance")
    selection = bundle["selection"]
    require(selection["seed"] == 20261029 and selection["calibration_size"] == 128
            and selection["confirmation_size"] == 256 and selection["annotation_pixels_read"] is False
            and selection["class_coverage_selection"] is False, "Cohort selection protocol changed")
    require(bundle["audit"]["global_never_seen_certified"] is False
            and bundle["audit"]["historical_index_exhaustive"] is False, "Unsupported global exposure claim")
    require(set(bundle["datasets"]) == set(DATASETS), "Cohort dataset admission changed")
    require(bundle.get("gt_contents_opened", False) is False, "Cohort opened GT early")
    for dataset in DATASETS:
        value = bundle["datasets"][dataset]
        eligible = value["eligible_ids"]
        require(len(eligible) == len(set(eligible)) and len(eligible) >= 384, "Incomplete eligible cohort identities")
        ranked = sorted(eligible, key=lambda identity: (hashlib.sha256(
            f"trace-confirmation-v1\n20261029\n{dataset}\n{identity}\n".encode("ascii")).hexdigest(), identity))
        require(eligible == ranked, "Cohort hash-ranked order changed")
        excluded = set(bundle["exclusions"][dataset]["union_ids"])
        identities = {"official_id": set(), "image": set(), "image_sha256": set(), "label": set()}
        for stage, count in COUNTS.items():
            rows = bundle["datasets"][dataset][stage]
            require(isinstance(rows, list) and len(rows) == count, "Cohort image admission bound changed")
            ids = [r["official_id"] for r in rows]
            start = 0 if stage == "calibration" else 128
            require(ids == eligible[start:start + count] and not excluded.intersection(ids)
                    and value["selected_ids_sha256"][stage] == canonical_sha(ids), "Cohort selection/exclusion identities changed")
            for row in rows:
                require(set(("official_id", "image", "label", "image_sha256", "label_metadata")) <= row.keys(),
                        "Incomplete cohort image identity")
                require(Path(row["image"]).is_absolute() and Path(row["label"]).is_absolute()
                        and row["image"] != row["label"] and isinstance(row["official_id"], str)
                        and row["official_id"] and len(row["image_sha256"]) == 64
                        and all(c in "0123456789abcdef" for c in row["image_sha256"])
                        and valid_label_metadata(row["label_metadata"]),
                        "Incomplete cohort image/GT metadata identity")
                for key, seen in identities.items():
                    require(row[key] not in seen, "Duplicate/overlapping calibration-confirmation cohort identity")
                    seen.add(row[key])


def valid_label_metadata(value):
    if isinstance(value, dict):
        return (set(value) == {"size_bytes", "mtime_ns", "device", "inode", "contents_read"}
                and value["contents_read"] is False
                and all(type(value[k]) is int and value[k] >= 0 for k in value if k != "contents_read"))
    return (isinstance(value, list) and len(value) == 2
            and all(type(v) is int and v >= 0 for v in value))


def verify_inputs(bundle):
    # RGB hashes are allowed; annotation contents are never opened by this controller.
    for dataset in DATASETS:
        for stage in COUNTS:
            for row in bundle["datasets"][dataset][stage]:
                require(sha256(row["image"]) == row["image_sha256"], "Frozen cohort RGB image changed")
                stat, meta = Path(row["label"]).stat(), row["label_metadata"]
                expected = (dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                 device=stat.st_dev, inode=stat.st_ino, contents_read=False)
                            if isinstance(meta, dict) else [stat.st_size, stat.st_mtime_ns])
                require(meta == expected, "Frozen cohort GT metadata changed")


def validate_panels(suite, prepared):
    bundle = read_json(suite / "frozen_cohorts.json")
    validate_cohorts(bundle)
    verify_inputs(bundle)
    module = worker()
    source_keys = [str(Path(p).resolve().relative_to(ROOT)) for p in module.checked_sources()]
    require(source_keys and all(k in prepared["runtime_hashes"] for k in source_keys), "Incomplete worker source inventory")
    expected_sources = {str(suite / "runtime" / k): prepared["runtime_hashes"][k] for k in source_keys}
    manifests = {}
    for entry in prepared["cells"]:
        path = suite / entry["manifest"]
        manifest = module.verify_manifest(path, entry["manifest_sha256"])
        module.validate_manifest(manifest)
        dataset, stage = entry["dataset"], entry["stage"]
        parent = read_json(suite / f"parents/{dataset}/prediction_protocol.json")
        require(manifest["dataset"] == dataset and manifest["stage"] == stage
                and manifest["seed"] == 0 and manifest["max_images"] == entry["images"]
                and manifest["arms"] == ARMS and manifest["order_seed"] == entry["order_seed"],
                "Manifest differs from locked cell scope")
        require(manifest["suite_root"] == str(suite)
                and manifest["cohorts_path"] == str(suite / "frozen_cohorts.json")
                and manifest["cohort_sha256"] == prepared["cohort_sha256"], "Manifest cohort/suite binding changed")
        require(manifest["config"] == parent["config"] == read_json(suite / entry["config"])
                and manifest["config_path"] == str(suite / entry["config"])
                and manifest["config_sha256"] == sha256(suite / entry["config"]), "Inherited config override")
        require(manifest["checkpoint"] == parent["checkpoint"]
                and manifest["source_sha256"] == expected_sources, "Checkpoint/source binding changed")
        panel = manifest["panel"]
        require(panel["full_split"] is False and len(panel["inputs"]) == entry["images"], "Panel scope changed")
        expected_rows = {r["official_id"]: r for r in bundle["datasets"][dataset][stage]}
        require(len({r["official_id"] for r in panel["inputs"]}) == len(expected_rows)
                and panel["ordered_ids"] == [r["official_id"] for r in panel["inputs"]], "Panel identities incomplete")
        for row in panel["inputs"]:
            expected = expected_rows.get(row["official_id"])
            require(expected is not None and all(row.get(k) == v for k, v in expected.items()),
                    "Panel not bound to exact frozen cohort image identities")
        manifests[entry["key"]] = manifest
    return manifests


def inventory(suite, folders):
    return {str(p.relative_to(suite)): sha256(p) for folder in folders
            for p in sorted((suite / folder).rglob("*")) if p.is_file()}


def prepare(args):
    require(not args.output.exists() and not args.output.is_symlink(), "Fresh output required; no overwrite/resume")
    source, suite = args.source_root.resolve(), args.output.resolve()
    require(not any(suite.is_relative_to(source / tree) for tree in audited.TREES), "Output inside source tree")
    require(sha256(args.plan) == PLAN_SHA256, "Immutable frozen plan bytes changed")
    validate_plan(read_json(args.plan))
    validate_cohorts(read_json(args.cohorts))
    parents = {d: read_parent(args.parent_root, d) for d in DATASETS}
    require(parents[DATASETS[0]][0]["checkpoint"] == parents[DATASETS[1]][0]["checkpoint"], "Parent checkpoints differ")
    require(args.python.is_file(), "Existing Python required")
    required = ("tools/evolution_confirmation/queue.py", "tools/evolution_confirmation/run.py",
                "tools/evolution_followup/queue.py", "tests/test_confirmation_queue.py")
    require(all((source / p).is_file() for p in required), "Incomplete confirmation source snapshot")
    suite.mkdir(parents=True, exist_ok=False)
    for folder in ("logs", "configs", "parents", "manifests"):
        (suite / folder).mkdir()
    state(suite, "snapshot")
    for original, name in ((args.plan, "frozen_plan.json"), (args.cohorts, "frozen_cohorts.json")):
        before = sha256(original)
        shutil.copyfile(original, suite / name)
        require(sha256(original) == sha256(suite / name) == before, "Plan/cohort changed while copying")
    runtime_hashes = primitives().snapshot(source, suite / "runtime")
    prepared = dict(format_version=1, worker="confirmation_queue", runtime_root=str(suite / "runtime"),
        runtime_hashes=runtime_hashes, plan_sha256=PLAN_SHA256, cohort_sha256=sha256(suite / "frozen_cohorts.json"),
        python=str(args.python.absolute()), python_sha256=sha256(args.python), cells=cells(), parents={},
        arms=ARMS, exposure_note=EXPOSURE, child_receipts=[])
    for dataset, (parent, binding) in parents.items():
        destination = suite / "parents" / dataset
        destination.mkdir()
        for key, name in (("protocol", "prediction_protocol.json"), ("seal", "PREDICTIONS_SEALED.json")):
            shutil.copyfile(binding[f"{key}_path"], destination / name)
            require(sha256(destination / name) == binding[f"{key}_sha256"], "Parent changed during copy")
        write_json(suite / f"configs/{dataset}.json", parent["config"])
        write_json(destination / "checkpoint_identity.json", parent["checkpoint"])
        prepared["parents"][dataset] = binding
    base = inventory(suite, ("configs", "parents"))
    base.update({name: sha256(suite / name) for name in ("frozen_plan.json", "frozen_cohorts.json")})
    for entry in prepared["cells"]:
        state(suite, "prepare", entry["key"])
        check_hashes(suite, base)
        require(sha256(args.python) == prepared["python_sha256"], "Python binary changed")
        arguments = [WORKER, "prepare", "--cohorts", str(suite / "frozen_cohorts.json"),
            "--dataset", entry["dataset"], "--stage", entry["stage"], "--config", str(suite / entry["config"]),
            "--checkpoint", parents[entry["dataset"]][0]["checkpoint"]["path"],
            "--manifest", str(suite / entry["manifest"])]
        if entry["order_seed"] is not None:
            arguments += ["--order-seed", str(entry["order_seed"])]
        prepared["child_receipts"].append(child(suite, runtime_hashes, args.python, arguments,
                                               "prepare-" + entry["key"].replace("/", "-"), cpu=True))
        entry["manifest_sha256"] = sha256(suite / entry["manifest"])
    validate_panels(suite, prepared)
    prepared["artifact_sha256"] = {**base, **inventory(suite, ("manifests", "logs"))}
    check_hashes(suite, prepared["artifact_sha256"])
    require(source_inventory(suite / "runtime") == runtime_hashes, "Snapshot changed during CPU prepare")
    write_json(suite / "SUITE_PREPARED.json", prepared)
    state(suite, "prepared")
    print(json.dumps(dict(suite=str(suite), suite_sha256=sha256(suite / "SUITE_PREPARED.json"))))


def verify_prepared(suite, expected_hash, *, frozen=False):
    require(sha256(suite / "SUITE_PREPARED.json") == expected_hash, "Reviewed prepared-suite hash mismatch")
    prepared = read_json(suite / "SUITE_PREPARED.json")
    require(prepared["format_version"] == 1 and prepared["worker"] == "confirmation_queue", "Wrong prepared suite")
    require(prepared["runtime_root"] == str(suite / "runtime"), "Snapshot relocated")
    if frozen:
        require(ROOT == suite / "runtime" and Path.cwd().resolve() == ROOT, "Run from new frozen snapshot cwd")
    require(source_inventory(suite / "runtime") == prepared["runtime_hashes"], "Frozen runtime changed")
    check_hashes(suite, prepared["artifact_sha256"])
    require(sha256(prepared["python"]) == prepared["python_sha256"], "Frozen Python binary changed")
    require(sha256(suite / "frozen_plan.json") == prepared["plan_sha256"] == PLAN_SHA256, "Frozen plan changed")
    validate_plan(read_json(suite / "frozen_plan.json"))
    require(sha256(suite / "frozen_cohorts.json") == prepared["cohort_sha256"], "Frozen cohort changed")
    expected = cells()
    require(len(prepared["cells"]) == len(expected), "Eight-cell admission bound changed")
    for entry, locked in zip(prepared["cells"], expected):
        require(set(entry) == set(locked) | {"manifest_sha256"}
                and all(entry[k] == v for k, v in locked.items()), "Cell admission scope/order/paths changed")
    require(prepared["arms"] == ARMS, "Six-arm admission changed")
    required_small = set(inventory(suite, ("configs", "parents", "manifests"))) | {"frozen_plan.json", "frozen_cohorts.json"}
    require(required_small <= prepared["artifact_sha256"].keys(), "Incomplete small-artifact freeze")
    for dataset in DATASETS:
        parent_dir = suite / "parents" / dataset
        protocol = read_json(parent_dir / "prediction_protocol.json")
        seal = read_json(parent_dir / "PREDICTIONS_SEALED.json")
        require(seal["complete"] is True and seal["gt_contents_opened"] is False
                and seal["artifact_sha256"]["prediction_protocol.json"] == sha256(parent_dir / "prediction_protocol.json")
                and protocol["checkpoint"] == read_json(parent_dir / "checkpoint_identity.json"), "Frozen parent binding changed")
    validate_panels(suite, prepared)
    return prepared


def validate_gate(suite, entry):
    path = suite / f"gates/{entry['dataset']}.json"
    report = read_json(path)
    require(report["passed"] is True and report["gt_contents_opened"] is False and report["images"] == 4
            and report["dataset"] == entry["dataset"] and report["manifest_sha256"] == entry["manifest_sha256"],
            "Four-image GT-free gate failed or scope changed")
    claim = path.with_name(path.name + ".STARTED.json")
    require(report["attempt_sha256"] == sha256(claim)
            and read_json(claim) == dict(status="started", manifest_sha256=entry["manifest_sha256"]),
            "Gate attempt claim missing or changed")
    return sha256(path)


def validate_prediction(suite, entry):
    output = suite / entry["output"]
    protocol = worker().verify_prediction_seal(output)
    require(protocol["manifest_sha256"] == entry["manifest_sha256"]
            and protocol["manifest"] == read_json(suite / entry["manifest"])
            and protocol["manifest_path"] == str(suite / entry["manifest"]), "Prediction manifest binding changed")
    if entry["stage"] == "confirmation":
        selection_path = suite / "SELECTION_SEALED.json"
        require(protocol["selection"] == dict(path=str(selection_path), sha256=sha256(selection_path),
                selected_mode=read_json(selection_path)["selected_mode"]), "Confirmation prediction selection linkage changed")
    else:
        require(protocol["selection"] is None, "Calibration incorrectly bound to selection")
    seal = read_json(output / "PREDICTIONS_SEALED.json")
    require(seal["complete"] is True and seal["gt_contents_opened"] is False
            and seal["images"] == entry["images"] and seal["arms"] == ARMS, "Prediction admission scope changed")
    return sha256(output / "PREDICTIONS_SEALED.json")


def pooled_miou(result):
    intersection, union = result["intersection"], result["union"]
    require(isinstance(intersection, list) and len(intersection) == len(union) > 0, "Missing pooled class counts")
    require(all(type(i) is int and type(u) is int and 0 <= i <= u for i, u in zip(intersection, union)),
            "Invalid pooled class counts")
    ratios = [i / u for i, u in zip(intersection, union) if u]
    require(bool(ratios), "No valid pooled classes")
    value = 100 * math.fsum(ratios) / len(ratios)
    supplied = result["miou_unrounded"]
    require(type(supplied) in (float, int) and math.isfinite(supplied)
            and math.isclose(supplied, value, rel_tol=0, abs_tol=1e-10), "Rounded/nonpooled mIoU supplied")
    return value


def validate_score(suite, entry):
    output = suite / entry["output"]
    prediction_hash = validate_prediction(suite, entry)
    seal = read_json(output / "SCORED_SEALED.json")
    require(seal["prediction_seal_sha256"] == prediction_hash
            and set(seal["artifact_sha256"]) == set(SCORING_FILES[:-1]), "Score seal linkage/coverage changed")
    check_hashes(output, seal["artifact_sha256"])
    require(read_json(output / "SCORING_STARTED.json")["prediction_seal_sha256"] == prediction_hash,
            "Scoring start linkage changed")
    result = read_json(output / "result.json")
    require(result["status"] == "completed" and result["dataset"] == entry["dataset"]
            and result["stage"] == entry["stage"] and result["order_seed"] == entry["order_seed"]
            and result["images"] == entry["images"] and result["full_split"] is False
            and result["prediction_seal_sha256"] == prediction_hash
            and result["cohort_sha256"] == read_json(suite / entry["manifest"])["cohort_sha256"]
            and result["label_contents_first_opened_after_all_stage_predictions_sealed"] is True
            and set(result["arm_results"]) == set(result["costs"]) == set(result["timings"]) == set(ARMS),
            "Incomplete scored arm/image scope")
    predicted = read_json(output / "prediction_result.json")
    require(result["costs"] == predicted["costs"] and result["timings"] == predicted["timings"],
            "Scoring changed measured costs/timing")
    with (output / "cost.jsonl").open() as stream:
        costs = []
        for line in stream:
            require(len(costs) < entry["images"] * len(ARMS) and len(line) <= 64 * 1024, "Unbounded cost stream")
            costs.append(json.loads(line))
    require(all(type(r["diagnostic_encoder_inputs"]) is int and r["diagnostic_encoder_inputs"] >= 0
                and type(r["seconds"]) in (float, int) and math.isfinite(r["seconds"]) and r["seconds"] >= 0
                for r in costs), "Invalid actual cost counters/timing")
    batch = read_json(suite / entry["config"])["batch_size"]
    for arm in ARMS:
        rows = [r for r in costs if r["arm"] == arm]
        require([r["episode"] for r in rows] == list(range((entry["images"] + batch - 1) // batch))
                and [r["images"] for r in rows] == [min(batch, entry["images"] - i) for i in range(0, entry["images"], batch)],
                "Actual cost scope omits/duplicates images or warmup")
        require(sum(r["diagnostic_encoder_inputs"] for r in rows)
                == result["costs"][arm]["diagnostic_encoder_inputs"], "Actual diagnostic totals differ from all-image stream")
    require(all(r["arm"] in ARMS for r in costs), "Unexpected cost arm")
    for arm in ARMS:
        measured = [r for r in costs if r["arm"] == arm and r["episode"] != 0]
        seconds = sum(r["seconds"] for r in measured)
        images = sum(r["images"] for r in measured)
        expected = dict(measured_episodes=len(measured), measured_images=images,
            seconds=seconds if measured else None, seconds_per_image=seconds / images if images else None,
            images_per_second=images / seconds if images and seconds > 0 else None,
            warmup_episode=0, accuracy_includes_warmup=True)
        require(result["timings"][arm] == expected, "Timing scope must omit only first episode, not accuracy/costs")
    for arm in ARMS:
        pooled_miou(result["arm_results"][arm])
        cost = result["costs"][arm]["diagnostic_encoder_inputs"]
        require(type(cost) is int and cost >= 0, "Missing/invalid actual diagnostic encoder inputs")
    return result


def select_mode(calibration_results, rule):
    """No confirmation argument is accepted; best-anchored tie band is not transitive."""
    require(set(calibration_results) == set(DATASETS), "Selection requires both calibration datasets only")
    require(all(r["stage"] == "calibration" and r["images"] == 128 and r["order_seed"] is None
                for r in calibration_results.values()), "Selection rejects confirmation/noncalibration scope")
    require(rule["tie_band_pp"] == .02, "Locked tie band changed")
    candidates = {}
    for mode in MODES:
        arm = "image_" + mode
        margins = {d: pooled_miou(calibration_results[d]["arm_results"][arm])
                      - pooled_miou(calibration_results[d]["arm_results"]["image_confidence"]) for d in DATASETS}
        costs = [calibration_results[d]["costs"][arm]["diagnostic_encoder_inputs"] for d in DATASETS]
        require(all(type(v) is int and v >= 0 for v in costs), "Invalid actual diagnostic input totals")
        candidates[mode] = dict(margin_pp=margins, worst_dataset_margin_pp=min(margins.values()),
            macro_margin_pp=math.fsum(margins.values()) / 2, diagnostic_encoder_inputs=sum(costs))
    best = max(v["worst_dataset_margin_pp"] for v in candidates.values())
    tied = [m for m in MODES if best - candidates[m]["worst_dataset_margin_pp"] <= .02 + 1e-12]
    selected = min(tied, key=lambda m: (candidates[m]["diagnostic_encoder_inputs"],
                                      -candidates[m]["macro_margin_pp"], MODES.index(m)))
    return selected, candidates


def no_scoring(suite, entries):
    require(not any((suite / e["output"] / name).exists() for e in entries for name in SCORING_FILES),
            "GT/scoring artifacts appeared before stage prediction barrier")


def run(args):
    suite = args.suite.resolve()
    prepared = verify_prepared(suite, args.suite_sha256, frozen=True)
    gates = predictions = scores = 0
    with queue_lock(suite):
        require(not any((suite / p).exists() for p in ("EXECUTION_STARTED.json", "SUITE_SEALED.json", "cells", "gates",
            "SELECTION_SEALED.json", "ALL_CALIBRATION_PREDICTIONS_SEALED.json", "ALL_CONFIRMATION_PREDICTIONS_SEALED.json")),
            "No automatic retry/resume/overwrite")
        write_json(suite / "EXECUTION_STARTED.json", dict(suite_prepared_sha256=args.suite_sha256, pid=os.getpid()))
        receipts, results, gate_hashes, prediction_hashes, score_hashes = [], [], {}, {}, {}
        selection_hash = None
        calibration = [e for e in prepared["cells"] if e["stage"] == "calibration"]
        confirmation = [e for e in prepared["cells"] if e["stage"] == "confirmation"]

        def progress(phase, active=None, error=None):
            state(suite, phase, active, gates=gates, predictions=predictions, scores=scores, error=error)

        def launch(arguments, label, cpu=False):
            verify_prepared(suite, args.suite_sha256, frozen=True)
            check_hashes(suite, {**gate_hashes, **score_hashes})
            if selection_hash is not None:
                require(sha256(suite / "SELECTION_SEALED.json") == selection_hash, "Selection changed after sealing")
            else:
                require(not any((suite / e["output"]).exists() for e in confirmation),
                        "Confirmation appeared before calibration selection")
            receipts.append(child(suite, prepared["runtime_hashes"], Path(prepared["python"]), arguments, label, cpu=cpu))

        try:
            (suite / "gates").mkdir()
            for entry in calibration:
                progress("gate", entry["key"])
                launch([WORKER, "gate", "--manifest", str(suite / entry["manifest"]),
                    "--manifest-sha256", entry["manifest_sha256"], "--output", str(suite / f"gates/{entry['dataset']}.json")],
                    "gate-" + entry["dataset"])
                gate_hashes[f"gates/{entry['dataset']}.json"] = validate_gate(suite, entry)
                no_scoring(suite, prepared["cells"])
                gates += 1
            for stage, entries in (("calibration", calibration), ("confirmation", confirmation)):
                for entry in entries:
                    progress(f"{stage}_predict", entry["key"])
                    if stage == "confirmation":
                        require(selection_hash is not None, "Selection must be sealed before confirmation")
                    else:
                        require(not any((suite / e["output"]).exists() for e in confirmation),
                                "Confirmation appeared before calibration selection")
                    no_scoring(suite, entries)
                    launch([WORKER, "predict", "--manifest", str(suite / entry["manifest"]),
                        "--manifest-sha256", entry["manifest_sha256"], "--output", str(suite / entry["output"])],
                        "predict-" + entry["key"].replace("/", "-"))
                    prediction_hashes[entry["output"]] = validate_prediction(suite, entry)
                    no_scoring(suite, entries)
                    predictions += 1
                current = {e["output"]: validate_prediction(suite, e) for e in entries}
                require(current == {e["output"]: prediction_hashes[e["output"]] for e in entries}, "Prediction barrier changed")
                no_scoring(suite, entries)
                barrier = f"ALL_{stage.upper()}_PREDICTIONS_SEALED.json"
                write_json(suite / barrier, dict(complete=True, stage=stage,
                    suite_prepared_sha256=args.suite_sha256, prediction_seal_sha256=current, gt_contents_opened=False))
                score_hashes[barrier] = sha256(suite / barrier)
                for entry in entries:
                    progress(f"{stage}_score", entry["key"])
                    require({e["output"]: validate_prediction(suite, e) for e in entries} == current,
                            "Prediction changed before score")
                    launch([WORKER, "score", "--output", str(suite / entry["output"])],
                           "score-" + entry["key"].replace("/", "-"), cpu=True)
                    result = validate_score(suite, entry)
                    results.append((entry, result))
                    for name in ("SCORED_SEALED.json", "result.json"):
                        relative = entry["output"] + "/" + name
                        score_hashes[relative] = sha256(suite / relative)
                    scores += 1
                if stage == "calibration":
                    progress("selection")
                    require(not any((suite / e["output"]).exists() for e in confirmation),
                            "Selection attempted after confirmation started")
                    rule = read_json(suite / "frozen_plan.json")["selection"]
                    check_hashes(suite, score_hashes)
                    verified_calibration = {e["dataset"]: validate_score(suite, e) for e in calibration}
                    selected, candidates = select_mode(verified_calibration, rule)
                    write_json(suite / "SELECTION_SEALED.json", dict(complete=True, selected_mode=selected,
                        cohort_sha256=prepared["cohort_sha256"], rule=rule, candidates=candidates,
                        suite_prepared_sha256=args.suite_sha256, calibration_barrier_sha256=sha256(suite / barrier),
                        calibration_score_sha256={e["output"]: sha256(suite / e["output"] / "SCORED_SEALED.json")
                                                  for e in calibration}))
                    selection_hash = sha256(suite / "SELECTION_SEALED.json")
            verify_prepared(suite, args.suite_sha256, frozen=True)
            check_hashes(suite, {**gate_hashes, **score_hashes})
            require(sha256(suite / "SELECTION_SEALED.json") == selection_hash, "Selection changed after confirmation")
            for entry in prepared["cells"]:
                require(validate_prediction(suite, entry) == prediction_hashes[entry["output"]], "Final prediction changed")
                validate_score(suite, entry)
            with (suite / "rows.csv").open("x", newline="") as stream:
                columns = ["dataset", "stage", "order_seed", "arm", "images", "full_split", "miou_unrounded", "margin_vs_image_confidence_pp",
                    "diagnostic_encoder_inputs", "selected_global_mode", "is_selected_probe", "manifest_sha256",
                    "config_sha256", "selection_sha256", "prediction_seal_sha256", "score_seal_sha256",
                    "measured_images", "seconds", "seconds_per_image", "images_per_second", "exposure_scope"]
                table = csv.DictWriter(stream, fieldnames=columns)
                table.writeheader()
                for entry, result in results:
                    for arm in ARMS:
                        table.writerow(dict(dataset=entry["dataset"], stage=entry["stage"], order_seed=entry["order_seed"],
                            arm=arm, images=entry["images"], full_split=False,
                            miou_unrounded=pooled_miou(result["arm_results"][arm]),
                            margin_vs_image_confidence_pp=pooled_miou(result["arm_results"][arm])
                                - pooled_miou(result["arm_results"]["image_confidence"]),
                            diagnostic_encoder_inputs=result["costs"][arm]["diagnostic_encoder_inputs"],
                            selected_global_mode=selected, is_selected_probe=arm == "image_" + selected,
                            manifest_sha256=entry["manifest_sha256"], config_sha256=sha256(suite / entry["config"]),
                            selection_sha256=selection_hash, prediction_seal_sha256=prediction_hashes[entry["output"]],
                            score_seal_sha256=sha256(suite / entry["output"] / "SCORED_SEALED.json"),
                            **{k: result["timings"][arm][k] for k in ("measured_images", "seconds", "seconds_per_image", "images_per_second")},
                            exposure_scope=EXPOSURE))
            write_json(suite / "execution_children.json", receipts)
            progress("completed")
            artifacts = {**prepared["artifact_sha256"], **gate_hashes, **score_hashes, **inventory(suite, ("logs",))}
            for name in ("SUITE_PREPARED.json", "EXECUTION_STARTED.json", "SELECTION_SEALED.json",
                         "execution_children.json", "queue_state.json", "rows.csv"):
                artifacts[name] = sha256(suite / name)
            artifacts.update({p + "/PREDICTIONS_SEALED.json": h for p, h in prediction_hashes.items()})
            check_hashes(suite, artifacts)
            write_json(suite / "SUITE_SEALED.json", dict(complete=True, calibration_cells=2, confirmation_cells=6,
                formal_cells=8, arm_results=48, passed_gt_free_gates=2, full_split=False, selected_mode=selected,
                selection_sha256=selection_hash, cohort_sha256=prepared["cohort_sha256"],
                suite_prepared_sha256=args.suite_sha256, runtime_hashes=prepared["runtime_hashes"],
                artifact_sha256=artifacts, exposure_note=EXPOSURE))
        except BaseException as exc:
            progress("failed", error=f"{type(exc).__name__}: {exc}"[:2000])
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare", help="CPU-only frozen eight-manifest preparation")
    prep.add_argument("--cohorts", type=Path, required=True)
    prep.add_argument("--plan", type=Path, default=ROOT / "docs/confirmation-image-budget-plan.json")
    prep.add_argument("--source-root", type=Path, default=ROOT)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--python", type=Path, default=audited.PYTHON)
    prep.add_argument("--parent-root", type=Path, default=audited.PARENT)
    execute = sub.add_parser("run", help="Requires explicitly reviewed SUITE_PREPARED hash and frozen cwd")
    execute.add_argument("--suite", type=Path, required=True)
    execute.add_argument("--suite-sha256", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    fresh = args.phase == "prepare" and not args.output.exists()
    try:
        prepare(args) if args.phase == "prepare" else run(args)
    except BaseException as exc:
        if fresh and (args.output / "queue_state.json").exists():
            state(args.output, "failed", error=f"{type(exc).__name__}: {exc}"[:2000])
        raise


if __name__ == "__main__":
    main()
