"""Post-replay CPU scoring. Fresh independent admission is mandatory.

prepare() cannot run before the exact support replay has sealed and exited.
Neither preparation nor admission decodes annotations or prediction payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from tools.evolution_followup.queue import snapshot
from tools.evolution_tuning.common import write_json
from .bootstrap import require
from .intake import sha256, write_new
from .risk_matching import membership, summarize as matched_summary
from .risk_outcomes import measure
from .risk_replay import offline_true_support
from .risk_replay_manifest import bind_parent, bound_json
from .risk_score_inputs import candidate_image_mask, decode_annotation, initial_region_quality
from .risk_score_payload import annotation_bytes, hard_array, stable_bytes, support_array
from .risk_summary import score_groups, summary
from .risk_worker import strict_runtime_inventory

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/home/wwww/hzl/results/trace_mechanism_20261102")
REPLAY = BASE/"M01_support_replay_20261102_1738"
REPLAY_SHA = "01c04f034d05452a9adc536a5f7dd0946e90fbfa82440e93600590b44c0c98f5"
RISK_PROTOCOL = BASE/"M01_RISK_ANALYSIS_PROTOCOL_v1.json"
RISK_PROTOCOL_SHA = "5e028bc62c1da112c9f33f1e52522a47c9b5b15cffa623542f613d8d45234d63"
UNIT = "trace-risk-support-20261102-1738.service"
INVOCATION = "e3dc7670213c4c6c897b0c037a28515a"
CANDIDATES, UNIQUE_READOUTS = 2042, 4340


def replay_terminal():
    keys = ("LoadState", "ActiveState", "SubState", "MainPID", "ExecMainPID", "ExecMainCode", "ExecMainStatus", "Result", "InvocationID")
    output = subprocess.run(["systemctl", "--user", "show", UNIT, "--property="+",".join(keys)],
        check=True, text=True, capture_output=True, timeout=30)
    actual = dict(line.split("=", 1) for line in output.stdout.splitlines() if "=" in line)
    expected = dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
        ExecMainPID="2012362", ExecMainCode="1", ExecMainStatus="0", Result="success", InvocationID=INVOCATION)
    require(all(actual.get(k) == v for k, v in expected.items()), "Exact support replay has not successfully exited")
    return actual


def canonical_child(root, relative):
    require(isinstance(relative, str) and relative == str(Path(relative)) and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Invalid scored relative path")
    path = root/relative
    require(path.resolve(strict=True) == path and path.is_file(), "Aliased scored input")
    return path


def json_digest(path):
    digest = sha256(path)
    return bound_json(path, digest), digest


def publish_seal(path, value):
    """Successful staging close precedes the final non-overwriting publication.

    Keep the staging inode as evidence. No fallible cleanup follows os.link().
    """
    temporary = path.with_name("."+path.name+".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)


def metadata_bind(replay, expected_sha, risk_path):
    """Strict reference graph and population checks, with no array/GT opens."""
    replay = Path(replay)
    require(replay.is_absolute() and replay.resolve(strict=True) == replay, "Canonical support suite required")
    require(not any((replay/n).exists() for n in ("FAILED.json", "CANCELLED.json")), "Failed replay cannot be scored")
    prepared = bound_json(replay/"SUITE_PREPARED.json", expected_sha)
    require(prepared["root"] == str(replay) and prepared["schema_version"] == 1, "Wrong support suite identity")
    require(strict_runtime_inventory(replay/"runtime") == prepared["runtime_hashes"], "Support runtime source changed")
    bound_json(replay/"PROTOCOL.json", prepared["protocol_sha256"])
    inputs = bound_json(replay/"REPLAY_INPUTS.json", prepared["inputs_sha256"])
    original = bind_parent(Path(inputs["parent"]), inputs["prepared_sha256"], inputs["barrier_sha256"])
    for source in original["datasets"].values():
        for case in source["cases"]:
            for point in case["checkpoints"]:
                point["snapshot_sha256"] = source["artifact_sha256"][point["snapshot"]]
                point["labels_sha256"] = source["artifact_sha256"][point["labels"]]
    require(original == inputs, "Original candidate prediction references changed")
    barrier, barrier_sha = json_digest(replay/"ALL_SUPPORTS_SEALED.json")
    require(barrier["complete"] is True and barrier["GT_opened"] is False and barrier["no_adaptation"] is True
            and barrier["suite_prepared_sha256"] == expected_sha
            and set(barrier["dataset_seals"]) == set(inputs["datasets"]) == {"coco_object", "pc60"},
            "Incomplete all-support barrier")
    risk, risk_sha = json_digest(risk_path)
    require(risk_sha == RISK_PROTOCOL_SHA and risk["parent_prepared_sha256"] == inputs["prepared_sha256"]
            and risk["parent_barrier_sha256"] == inputs["barrier_sha256"]
            and set(risk["datasets"]) == set(inputs["datasets"])
            and risk["regions"] == ["inside", "outside", "whole"], "Risk protocol parent/regions differ")
    for path, digest in risk["source_code_sha256"].items():
        relative = Path(path).relative_to("/home/wwww/.codex/worktrees/trace-simplification/hzl")
        require(sha256(ROOT/relative) == digest, "Prospective risk estimator source changed")
    all_refs, count, unique, stored = {}, 0, 0, 0
    for dataset, source in inputs["datasets"].items():
        spec = source["spec"]
        require(spec["native_config"]["dataset"] == {"coco_object":"COCOObjectDataset", "pc60":"PascalContext60Dataset"}[dataset]
                and spec["native_config"]["resize"] == spec["native_config"]["patch_size"] == [224, 224],
                "Offline label/readout protocol differs")
        seal = bound_json(replay/dataset/"SUPPORTS_SEALED.json", barrier["dataset_seals"][dataset])
        require(seal["complete"] is True and seal["GT_opened"] is False
                and seal["suite_prepared_sha256"] == expected_sha and seal["candidates"] == len(source["cases"]),
                "Dataset support seal differs")
        references = bound_json(replay/dataset/"REFERENCES.json", seal["references_sha256"])
        require(len(references) == len(source["cases"]) == len(risk["datasets"][dataset]["rows"]), "Scoring population differs")
        expected_inventory = {f"{dataset}/REFERENCES.json":seal["references_sha256"]}
        rows, seen = [], {}
        for case, ref in zip(source["cases"], references, strict=True):
            require(all(case[k] == ref[k] for k in ("case", "episode", "source_id", "candidate_id"))
                    and len(ref["checkpoints"]) == len(case["checkpoints"]), "Support candidate identity differs")
            candidate = case["example"]["candidate"]
            query = candidate["class_idx"]
            require(type(query) is int and 0 <= query < len(spec["queries"]), "Invalid candidate query index")
            category = query if spec["query_to_class"] is None else spec["query_to_class"][query]
            require(category == candidate["preupdate_candidate_category_idx"], "Query-to-benchmark class differs")
            rows.append(dict(case=case["case"], source_id=case["source_id"], candidate_id=case["candidate_id"],
                episode=case["episode"], C=candidate["counterfactual_support"], confidence=candidate["confidence"],
                benchmark_class=category, area=candidate["area"], shape=candidate["region"]["shape"]))
            for point, actual in zip(case["checkpoints"], ref["checkpoints"], strict=True):
                require(all(actual[k] == v for k, v in point.items()) and actual["exact_changed_labels"] == 0,
                        "Support checkpoint identity differs")
                support = actual["reference"]
                key = hashlib.sha256(json.dumps(dict(dataset=dataset, original_episode=case["episode"],
                    source_id=case["source_id"], snapshot_sha256=point["snapshot_sha256"]), sort_keys=True).encode()).hexdigest()
                name = f"{dataset}/supports/{key}.npz"
                require(support["support"] == name and seal["artifact_sha256"].get(name) == support["sha256"],
                        "Support payload binding differs")
                if name in seen:
                    require(seen[name] == support, "Deduplicated support metadata differs")
                else:
                    seen[name] = support
                    stored += canonical_child(replay, name).stat().st_size
                expected_inventory[name] = support["sha256"]
        require(rows == risk["datasets"][dataset]["rows"] and score_groups(rows) == risk["datasets"][dataset]["score_groups"],
                "Prospective initial scores/memberships differ")
        require(seal["artifact_sha256"] == expected_inventory and seal["unique_readouts"] == len(seen),
                "Support inventory incomplete or unexpected")
        require(risk["checkpoints"][dataset] == sorted({1, spec["native_config"]["steps"]}), "Risk checkpoints differ")
        all_refs[dataset] = references
        count += len(rows)
        unique += len(seen)
    require(barrier["candidates"] == count == CANDIDATES and unique == UNIQUE_READOUTS and barrier["stored_bytes"] == stored,
            "Exact all-support totals differ")
    return dict(replay=str(replay), replay_sha256=expected_sha, barrier_sha256=barrier_sha,
        risk_protocol_sha256=risk_sha, original=inputs, references=all_refs, risk_protocol=risk,
        terminal=replay_terminal(), candidates=count, unique_readouts=unique, GT_opened=False)


def prepare(output):
    replay_terminal()
    bound = metadata_bind(REPLAY, REPLAY_SHA, RISK_PROTOCOL)
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh canonical scoring suite required")
    output.mkdir()
    write_new(output/"SCORE_INPUTS.json", bound)
    write_new(output/"PROTOCOL.json", dict(experiment="M01_offline_risk", new_GPU=False, no_new_adaptation=True,
        support_replay_sha256=REPLAY_SHA, risk_protocol_sha256=bound["risk_protocol_sha256"],
        matching="Initial confidence width.05, BENCHMARK class mapped from query index AND original query index (extension queries not merged), token-area bins; optional initial GT-correct fraction quartiles; median ties/unmatched/undefined quality retained",
        matched_estimator="Equal-weight mean of within frozen stratum high-C minus low-C metrics; original-episode shared draws10000; a missing side in any fixed stratum makes the draw undefined, never drop/merge",
        arrays="Only one source support per checkpoint, baseline cached only within current original episode; no full probability accumulation",
        labels="Historical PIL-nearest224, reduce_zero_label=False for both exact parent datasets, ignore255",
        GT_boundary="Only after all supports sealed AND exact retained successful exit AND independent exact CPU scoring admission",
        disclosure="ConfidenceTopK-selected/historically exposed128-source panels per dataset; duplicated source pixels are candidate occurrences, not full-split mIoU; not a causal C or independent-pool claim",
        primary="Mean true-class support inside decline>.001; fixed sensitivity0/.005,1/final; negative/null retained",
        failure="Failclosed; preserve partial output; no automatic retry; global analysis seal is last operation"))
    runtime = snapshot(ROOT, output/"runtime")
    write_new(output/"SUITE_PREPARED.json", dict(schema_version=1, root=str(output), runtime_hashes=runtime,
        inputs_sha256=sha256(output/"SCORE_INPUTS.json"), protocol_sha256=sha256(output/"PROTOCOL.json"),
        GT_opened=False, CPU_scoring_started=False))
    return sha256(output/"SUITE_PREPARED.json")


def admission(suite, expected_sha):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only scoring environment required")
    suite = Path(suite).resolve(strict=True)
    prepared = bound_json(suite/"SUITE_PREPARED.json", expected_sha)
    require(prepared["root"] == str(suite) and Path(__file__).resolve().is_relative_to(suite/"runtime")
            and strict_runtime_inventory(suite/"runtime") == prepared["runtime_hashes"], "Wrong or modified scoring runtime")
    bound_json(suite/"PROTOCOL.json", prepared["protocol_sha256"])
    inputs = bound_json(suite/"SCORE_INPUTS.json", prepared["inputs_sha256"])
    forbidden = {(row["label_metadata_details"]["device"], row["label_metadata_details"]["inode"])
        for source in inputs["original"]["datasets"].values() for case in source["cases"]
        for row in (case["example"]["source_input"],)}
    status = (suite/"ADMITTED.json").stat()
    require((status.st_dev, status.st_ino) not in forbidden, "Admission receipt aliases annotation")
    audit_identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
    raw, audit_sha = stable_bytes(suite/"ADMITTED.json", 4*1024**2, expected_identity=audit_identity)
    audit = json.loads(raw)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("CPU_GT_scoring_allowed") is True,
            "New independent exact CPU scoring admission required, not label-free replay admission")
    replay_terminal()
    require(metadata_bind(REPLAY, REPLAY_SHA, RISK_PROTOCOL) == inputs, "Scoring parent references changed")
    report = Path(audit["report_path"])
    require(report == suite/"INDEPENDENT_REPORT.md" and report.resolve(strict=True) == report,
            "Independent report must have the exact local audit path")
    status = report.stat()
    require((status.st_dev, status.st_ino) not in forbidden, "Audit report aliases annotation")
    report_identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
    stable_bytes(report, 4*1024**2, audit["report_sha256"], expected_identity=report_identity)
    binding = dict(admission_path=str(suite/"ADMITTED.json"), admission_sha256=audit_sha,
        admission_identity=list(audit_identity), report_identity=list(report_identity),
        report_path=str(report), report_sha256=audit["report_sha256"], independent_reviewer=audit["independent_reviewer"])
    return suite, inputs, binding


def verify_audit_binding(binding):
    stable_bytes(Path(binding["admission_path"]), 4*1024**2, binding["admission_sha256"],
        expected_identity=tuple(binding["admission_identity"]))
    stable_bytes(Path(binding["report_path"]), 4*1024**2, binding["report_sha256"],
        expected_identity=tuple(binding["report_identity"]))


def checkpoint_data(replay, parent, dataset, point, shape, classes):
    ref = point["reference"]
    content, _ = stable_bytes(canonical_child(replay, ref["support"]), 32*1024**2, ref["sha256"])
    support = support_array(content, ref, classes, shape)
    content, _ = stable_bytes(canonical_child(parent/dataset, point["labels"]), 2*1024**2, point["labels_sha256"])
    hard = hard_array(content, classes, shape)
    require(np.array_equal(support.argmax(0), hard)
            and hashlib.sha256(hard.tobytes()).hexdigest() == ref["hard_labels_sha256"],
            "Saved hard predictions and support replay differ")
    return support, hard


def execute(suite, inputs):
    replay, parent = Path(inputs["replay"]), Path(inputs["original"]["parent"])
    risk = inputs["risk_protocol"]
    total, seals = 0, {}
    for dataset, source in inputs["original"]["datasets"].items():
        output = suite/dataset
        output.mkdir()
        spec, rows = source["spec"], risk["datasets"][dataset]["rows"]
        shape = spec["native_config"]["resize"]
        classes = len(spec["queries"]) if spec["query_to_class"] is None else max(spec["query_to_class"])+1
        checkpoints = risk["checkpoints"][dataset]
        outcomes = {step:{region:[] for region in risk["regions"]} for step in checkpoints}
        qualities, annotations, baseline, active_episode = [], {}, {}, None
        with (output/"candidate_outcomes.jsonl").open("x") as stream:
            for case, ref in zip(source["cases"], inputs["references"][dataset], strict=True):
                if active_episode != case["episode"]:
                    baseline.clear()
                    active_episode = case["episode"]
                initial = ref["checkpoints"][0]
                cache_key = (case["source_id"], initial["snapshot_sha256"])
                source_row = case["example"]["source_input"]
                if cache_key not in baseline:
                    content, digest = annotation_bytes(source_row)
                    label = decode_annotation(content, shape, classes, False)
                    before_support, before = checkpoint_data(replay, parent, dataset, initial, shape, classes)
                    baseline[cache_key] = (before, offline_true_support(before_support, label), label)
                    del before_support, content
                    annotations[case["source_id"]] = dict(path=source_row["label"], sha256=digest,
                        frozen_metadata=source_row["label_metadata_details"])
                else:
                    # Every candidate's duplicated theta0 hard receipt is checked,
                    # even when the large source support was already decoded.
                    content, _ = stable_bytes(canonical_child(parent/dataset, initial["labels"]),
                        2*1024**2, initial["labels_sha256"])
                    duplicate = hard_array(content, classes, shape)
                    require(np.array_equal(duplicate, baseline[cache_key][0])
                            and hashlib.sha256(duplicate.tobytes()).hexdigest() == initial["reference"]["hard_labels_sha256"],
                            "Repeated initial hard prediction differs")
                before, before_true, label = baseline[cache_key]
                mask = candidate_image_mask(case["example"]["candidate"], shape)
                quality = initial_region_quality(before, label, mask, classes)
                qualities.append(quality)
                record = dict(case=case["case"], episode=case["episode"], source_id=case["source_id"],
                    candidate_id=case["candidate_id"], initial_GT_quality=quality, checkpoints={})
                require([p["step"] for p in ref["checkpoints"][1:]] == checkpoints, "Checkpoint order differs at score time")
                for point in ref["checkpoints"][1:]:
                    after_support, after = checkpoint_data(replay, parent, dataset, point, shape, classes)
                    after_true = offline_true_support(after_support, label)
                    measurement = measure(before, after, label, before_true, after_true, mask, classes)
                    record["checkpoints"][point["step"]] = measurement
                    for region in risk["regions"]:
                        outcomes[point["step"]][region].append(measurement[region])
                    del after_support, after_true
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                total += 1
                write_json(suite/"queue_state.json", dict(phase="offline_GT_scoring", dataset=dataset,
                    candidates=total, GT_opened=True, pid=os.getpid()), replace=True)
        query_indices = [c["example"]["candidate"]["class_idx"] for c in source["cases"]]
        grouping = dict(initial_scores=membership(rows, query_indices=query_indices),
            initial_scores_and_GT_quality=membership(rows, qualities, query_indices))
        write_new(output/"MATCHING_MEMBERSHIP.json", grouping)
        result = dict(dataset=dataset, candidates=len(rows), original_episodes=len({r["episode"] for r in rows}),
            unique_sources=len(annotations), completed=True, checkpoints={})
        for step in checkpoints:
            result["checkpoints"][step] = {}
            for region in risk["regions"]:
                result["checkpoints"][step][region] = dict(
                    score_quintiles=summary(rows, outcomes[step][region], risk["datasets"][dataset]["score_groups"]["groups"], dataset),
                    matched={name:matched_summary(rows, outcomes[step][region], group, dataset) for name, group in grouping.items()})
        write_new(output/"ANNOTATION_READ_RECEIPTS.json", annotations)
        write_new(output/"result.json", result)
        names = ("candidate_outcomes.jsonl", "MATCHING_MEMBERSHIP.json", "ANNOTATION_READ_RECEIPTS.json", "result.json")
        publish_seal(output/"ANALYSIS_SEALED.json", dict(complete=True, candidates=len(rows),
            support_barrier_sha256=inputs["barrier_sha256"], risk_protocol_sha256=inputs["risk_protocol_sha256"],
            artifact_sha256={name:sha256(output/name) for name in names}))
        seals[dataset] = sha256(output/"ANALYSIS_SEALED.json")
    require(total == inputs["candidates"] and set(seals) == set(inputs["original"]["datasets"]), "Incomplete risk analysis")
    return dict(complete=True, candidates=total, dataset_seals=seals,
        support_barrier_sha256=inputs["barrier_sha256"], support_retained_terminal=inputs["terminal"],
        risk_protocol_sha256=inputs["risk_protocol_sha256"], GT_opened=True, new_GPU=False,
        scope="M01 fixed candidate-risk panel only, not full-split mIoU or all TRACE experiments")


def run(suite, expected_sha):
    suite, inputs, audit_binding = admission(suite, expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json", "FAILED.json", "ANALYSIS_SEALED.json")),
            "No scoring resume/retry")
    require(shutil.disk_usage(suite).free >= 16*1024**3, "Shared disk reserve required")
    verify_audit_binding(audit_binding)
    write_new(suite/"STARTED.json", dict(suite_prepared_sha256=expected_sha,
        support_barrier_sha256=inputs["barrier_sha256"], retained_terminal=inputs["terminal"],
        audit_binding=audit_binding, admitted_GT_boundary=True, pid=os.getpid(), automatic_retry=False))
    started_sha = sha256(suite/"STARTED.json")
    start = time.perf_counter()
    try:
        result = execute(suite, inputs)
        require(not torch.cuda.is_initialized(), "CPU scoring unexpectedly initialized CUDA")
        verify_audit_binding(audit_binding)
        require(sha256(suite/"STARTED.json") == started_sha, "Scoring startup binding changed")
        result.update(suite_prepared_sha256=expected_sha, elapsed_seconds=time.perf_counter()-start,
            audit_binding=audit_binding, started_sha256=started_sha)
        write_json(suite/"queue_state.json", dict(phase="analysis_complete_awaiting_seal_and_exit",
            candidates=inputs["candidates"], GT_opened=True, pid=os.getpid()), replace=True)
        publish_seal(suite/"ANALYSIS_SEALED.json", result)
    except BaseException as error:
        write_new(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite), GT_opened=False))
    else:
        require(args.suite_sha256, "Frozen scoring suite SHA required")
        run(args.suite, args.suite_sha256)
