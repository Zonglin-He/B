"""Separately admitted CPU scoring after the exact G1954 stage has exited.

No preparation while predictions run. Metadata admission never reads labels,
RGBs, checkpoints, or prediction arrays. This is not a prediction admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from tools.evolution_tuning.common import write_json
from . import generalization_score as metric
from . import generalization_worker as prediction
from .bootstrap import require
from .generalization import ARMS
from .intake import sha256, write_new
from .risk_score_payload import annotation_bytes, hard_array, stable_bytes
from .risk_score_worker import canonical_child, publish_seal, verify_audit_binding

ROOT = Path(__file__).resolve().parents[2]
BASE = prediction.BASE
PARENT = BASE/"G_primary_20261102_1954"
PARENT_SHA = "95affb0d3320cd9f4d81ab727eebaaf3c7c6a157cbb3b67ac0f5da356640a496"
ANALYSIS = BASE/"G_ANALYSIS_PROTOCOL_v1.json"
ANALYSIS_SHA = "c9bc701da574d453e06d25b431a8bb2d65ae6e8e823053113913208dc5a12e6b"
RESERVE, OUTPUT_CAP = 16*1024**3, 256*1024**2
PARENT_ADMISSION_SHA = "1f984fed2ae97efdcad73be2f57cba477e341951049bae191ec7387513dca551"
PARENT_REPORT_SHA = "96a151c75140d393ba343eb5b83c0194d6b06c92656060555d126fd5937a219e"
# Operator-observed identities of the already admitted immutable parent metadata.
# These anchors precede any JSON parse or annotation-inode registry construction.
PARENT_METADATA_IDENTITIES = {
    "SUITE_PREPARED.json": (66306, 58205232, 1044064, 1793620502967854056),
    "INPUTS.json": (66306, 58204962, 28009597, 1793620502916854055),
}


def parent_terminal():
    keys = ("LoadState", "ActiveState", "SubState", "MainPID", "ExecMainPID", "ExecMainCode",
            "ExecMainStatus", "Result", "InvocationID")
    output = subprocess.run(["systemctl", "--user", "show", "trace-generalization-primary-20261102-1954.service",
        "--property="+",".join(keys)], check=True, text=True, capture_output=True, timeout=30)
    actual = dict(line.split("=", 1) for line in output.stdout.splitlines() if "=" in line)
    expected = dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
        ExecMainPID="2043872", ExecMainCode="1", ExecMainStatus="0", Result="success",
        InvocationID="2f14c1b37a0741f39500d48dae7ff353")
    require(all(actual.get(k) == v for k, v in expected.items()), "Exact G prediction service has not successfully exited")
    return actual


def metadata(path, digest=None, forbidden=frozenset(), limit=128*1024**2, expected_identity=None):
    path = Path(path)
    state = path.stat()
    require((state.st_dev, state.st_ino) not in forbidden, "G metadata aliases an annotation")
    identity = (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
    require(expected_identity is None or identity == tuple(expected_identity), "Bootstrap metadata identity differs")
    raw, actual = stable_bytes(path, limit, digest, expected_identity=identity)
    return json.loads(raw), actual


def label_inodes(inputs):
    return {(r["label_metadata_details"]["device"], r["label_metadata_details"]["inode"])
            for rows in inputs.values() for r in rows}


def write_prepared_json(path, value):
    """Hash the known new bytes, never reopen a just-written metadata pathname."""
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    require(len(content) <= 128*1024**2, "Prepared G scoring metadata exceeds bound")
    with path.open("xb") as stream:
        require(stream.write(content) == len(content), "Short G prepared metadata write")
        stream.flush()
        os.fsync(stream.fileno())
        state, actual = os.fstat(stream.fileno()), path.stat()
        require(path.resolve(strict=True) == path and
                (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
                == (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns),
                "New G prepared metadata pathname changed")
    return hashlib.sha256(content).hexdigest()


def source_members(root, forbidden, *, snapshot_source=False):
    """Enumerate and stat ALL eligible names before opening any source file."""
    root = Path(root)
    require(root.resolve(strict=True) == root and root.is_dir(), "Noncanonical scoring source root")
    result, size = {}, 0
    trees = [root/n for n in ("trace_ovss", "tools", "tests")] if snapshot_source else [root]
    for tree in trees:
        require(tree.is_dir() and not tree.is_symlink(), "Missing or aliased source tree")
        for path in sorted(tree.rglob("*")):
            relative = path.relative_to(root)
            if snapshot_source and (any(p in ("__pycache__", ".pytest_cache", ".git") for p in relative.parts)
                                    or path.suffix in (".pyc", ".pyo")):
                continue
            require(not path.is_symlink() and path.resolve(strict=True) == path,
                    "Source symlink is not admitted")
            require(not any(p in ("__pycache__", ".pytest_cache", ".git") for p in relative.parts)
                    and path.suffix not in (".pyc", ".pyo"), "Unexpected executable cache")
            if path.is_dir():
                continue
            require(path.is_file(), "Unexpected nonregular source member")
            state = path.stat()
            require((state.st_dev, state.st_ino) not in forbidden, "Source member aliases annotation")
            size += state.st_size
            require(size <= 64*1024**2, "Scoring source inventory exceeds bound")
            result[str(relative)] = (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
    return result


def guarded_inventory(root, expected, forbidden):
    members = source_members(root, forbidden)
    require(set(members) == set(expected), "Scoring source membership differs before hashing")
    for relative, identity in members.items():
        source_content(canonical_child(root, relative), identity, expected[relative])
    require(source_members(root, forbidden) == members, "Source identities changed while validating")
    return expected


def source_content(path, identity, digest=None):
    if identity[2]:
        return stable_bytes(path, 64*1024**2, digest, expected_identity=identity)
    # stable_bytes deliberately rejects empty payloads, but empty __init__.py is valid source.
    with path.open("rb") as stream:
        state = os.fstat(stream.fileno())
        current = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        require(current(state) == identity and path.resolve(strict=True) == path,
                "Empty source fd identity differs")
        require(stream.read(1) == b"" and current(path.stat()) == current(os.fstat(stream.fileno())) == identity
                and path.stat().st_ctime_ns == state.st_ctime_ns, "Empty source changed")
    actual = hashlib.sha256(b"").hexdigest()
    require(digest is None or digest == actual, "Empty source digest differs")
    return b"", actual


def guarded_snapshot(source, destination, forbidden):
    members = source_members(source, forbidden, snapshot_source=True)
    destination.mkdir()
    inventory = {}
    for relative, identity in members.items():
        path = canonical_child(source, relative)
        content, inventory[relative] = source_content(path, identity)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            require(stream.write(content) == len(content), "Short source snapshot write")
    require(source_members(source, forbidden, snapshot_source=True) == members, "Source changed during guarded snapshot")
    guarded_inventory(destination, inventory, forbidden)
    return inventory


def validate_visual_LN_names(legal):
    """DenseCLIP registers visual tensors under its backbone, not a second copy."""
    actual,visual = legal.get("trainable_names"),legal.get("visual_LN_names")
    count = legal.get("parameter_tensors")
    require(type(count) is int and count > 0 and type(actual) is list and type(visual) is list
            and len(actual) == len(visual) == count
            and all(isinstance(n,str) and n.startswith("backbone.visual.") for n in actual)
            and all(isinstance(n,str) and n.startswith("visual.") for n in visual)
            and len(set(actual)) == len(set(visual)) == count
            and actual == sorted(actual) and visual == sorted(visual)
            and actual == ["backbone."+name for name in visual],
            "G registered/canonical visual-LN membership differs")


def condition_cost(parent, spec, inventory, forbidden):
    """Close raw cost/state provenance without opening GT or predictions."""
    prefix = f"conditions/{spec['id']}/"
    summary, _ = metadata(canonical_child(parent, prefix+"prediction_result.json"),
                          inventory[prefix+"prediction_result.json"], forbidden, limit=2*1024**2)
    initial, _ = metadata(canonical_child(parent, prefix+"initial_state.json"),
                          inventory[prefix+"initial_state.json"], forbidden, limit=2*1024**2)
    path = canonical_child(parent, prefix+"cost.jsonl")
    state = path.stat()
    require((state.st_dev, state.st_ino) not in forbidden, "Raw cost aliases annotation")
    raw, _ = stable_bytes(path, 16*1024**2, inventory[prefix+"cost.jsonl"],
        expected_identity=(state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns))
    rows = []
    for line in raw.splitlines():
        require(0 < len(line) <= 64*1024, "G raw cost line exceeds bound")
        rows.append(json.loads(line))
    cfg, images = spec["config"], spec["images"]
    expected_episodes = (images+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(rows) == expected_episodes*len(ARMS), "G raw cost population differs")
    for episode in range(expected_episodes):
        order = prediction.runtime.prior.arm_order(ARMS, episode, "alternating")
        expected_images = min(cfg["batch_size"], images-episode*cfg["batch_size"])
        for position, row in enumerate(rows[episode*len(ARMS):(episode+1)*len(ARMS)]):
            require(row["episode"] == episode and row["arm"] == order[position]
                    and row["order_position"] == position and row["warmup"] is (episode == 0)
                    and row["images"] == expected_images, "G raw cost episode/arm order differs")
            prediction.runtime.validate_cost(row, row["arm"], cfg)
            require(type(row["seconds"]) in (float, int) and math.isfinite(row["seconds"])
                    and row["seconds"] >= 0, "Nonfinite G raw time")
    require(summary["condition"] == spec["id"] and summary["images"] == images
            and summary["arms"] == list(ARMS) and summary["GT_opened"] is False
            and summary["actual_work"] == {a:{k:sum(r[k] for r in rows if r["arm"] == a)
                for k in prediction.runtime.COUNTERS} for a in ARMS}
            and summary["raw_model_seconds"] == {a:sum(r["seconds"] for r in rows if r["arm"] == a) for a in ARMS}
            and summary["warmup_adjusted_throughput"] == prediction.runtime.prior.timing_summary(rows, ARMS),
            "G raw costs and sealed timing/work summary differ")
    legal = initial["legal_visual_LN"]
    require(summary["legal_visual_LN"] == legal
            and summary["frozen_parameter_sha256"] == initial["frozen_parameter_sha256"]
            and set(initial["reset"]) == {"trainable_sha256", "optimizer_sha256"},
            "G initial-state summary binding differs")
    validate_visual_LN_names(legal)
    for digest in (*initial["reset"].values(), initial["frozen_parameter_sha256"]):
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "Invalid G initial-state digest")
    return summary


def bind_parent():
    terminal = parent_terminal()
    require(PARENT.is_absolute() and PARENT.resolve(strict=True) == PARENT
            and not any((PARENT/n).exists() for n in ("FAILED.json", "CANCELLED.json")), "Unsuccessful G parent")
    prepared, _ = metadata(PARENT/"SUITE_PREPARED.json", PARENT_SHA,
        expected_identity=PARENT_METADATA_IDENTITIES["SUITE_PREPARED.json"])
    require(prepared["root"] == str(PARENT) and prepared["schema_version"] == 1, "Wrong G parent identity")
    inputs, _ = metadata(PARENT/"INPUTS.json", prepared["inputs_sha256"],
        expected_identity=PARENT_METADATA_IDENTITIES["INPUTS.json"])
    forbidden = label_inodes(inputs)
    specs, _ = metadata(PARENT/"CONDITIONS.json", prepared["conditions_sha256"], forbidden)
    prediction.validate_inputs(inputs, specs)
    metadata(PARENT/"PROTOCOL.json", prepared["protocol_sha256"], forbidden)
    guarded_inventory(PARENT/"runtime", prepared["runtime_hashes"], forbidden)
    barrier, barrier_sha = metadata(PARENT/"ALL_G_PREDICTIONS_SEALED.json", forbidden=forbidden)
    require(barrier["complete"] is True and barrier["GT_opened"] is False
            and barrier["suite_prepared_sha256"] == PARENT_SHA and barrier["cells"] == 57
            and barrier["image_method_evaluations"] == 104214
            and set(barrier["condition_seals"]) == {s["id"] for s in specs}, "Incomplete G prediction barrier")
    audit = barrier["audit_binding"]
    require(audit["admission_path"] == str(PARENT/"ADMITTED.json")
            and audit["report_path"] == str(PARENT/"INDEPENDENT_REPORT.md")
            and audit["admission_sha256"] == PARENT_ADMISSION_SHA
            and audit["report_sha256"] == PARENT_REPORT_SHA, "Wrong G parent audit binding")
    admission, _ = metadata(PARENT/"ADMITTED.json", PARENT_ADMISSION_SHA, forbidden)
    require(admission["decision"] == "ADMITTED" and admission["suite_prepared_sha256"] == PARENT_SHA
            and admission["G_primary_predictions_allowed"] is True
            and admission["GT_scoring_allowed"] is False, "Wrong G prediction permission")
    report_state = (PARENT/"INDEPENDENT_REPORT.md").stat()
    require((report_state.st_dev, report_state.st_ino) not in forbidden, "Parent report aliases annotation")
    verify_audit_binding(audit)
    started, _ = metadata(PARENT/"STARTED.json", barrier["started_sha256"], forbidden)
    require(started["suite_prepared_sha256"] == PARENT_SHA and started["audit"] == audit
            and started["pid"] == 2043872 and started["GT_opened"] is False, "Wrong G startup receipt")
    gate, _ = metadata(PARENT/"ALL_GATES_PASSED.json", barrier["gate_binding"]["barrier_sha256"], forbidden)
    require(gate["complete"] is True and gate["suite_prepared_sha256"] == PARENT_SHA
            and gate["GT_opened"] is False and gate["conditions"] == barrier["gate_binding"]["conditions"]
            and set(gate["conditions"]) == {s["id"] for s in specs}, "Missing G gates")
    for name, digest in gate["conditions"].items():
        child, _ = metadata(PARENT/"gates"/(name+".json"), digest, forbidden)
        require(child["passed"] is True and child["condition"] == name
                and child["actual_RGB_sources"] == 4 and child["GT_opened"] is False, "Invalid G child gate")
    seals = {}
    for spec in specs:
        name, images, batch = spec["id"], spec["images"], spec["config"]["batch_size"]
        seal, _ = metadata(PARENT/"conditions"/name/"PREDICTIONS_SEALED.json",
                           barrier["condition_seals"][name], forbidden)
        require(seal["complete"] is True and seal["GT_opened"] is False
                and seal["suite_prepared_sha256"] == PARENT_SHA and seal["condition"] == spec
                and seal["images"] == images and seal["image_method_evaluations"] == images*len(ARMS),
                "G condition seal/population differs")
        prefix = f"conditions/{name}/"
        expected = {prefix+f"predictions/{a}/{i:06d}.npy" for a in ARMS for i in range(images)}
        expected |= {prefix+f"episodes/{e:06d}.{a}.json" for a in ARMS for e in range((images+batch-1)//batch)}
        expected |= {prefix+n for n in ("initial_state.json", "cost.jsonl", "prediction_result.json")}
        require(set(seal["artifact_sha256"]) == expected, "G sealed prediction inventory incomplete")
        condition_cost(PARENT, spec, seal["artifact_sha256"], forbidden)
        seals[name] = seal
    analysis, _ = metadata(ANALYSIS, ANALYSIS_SHA, forbidden)
    require(analysis["bootstrap"]["replicates"] == 10000
            and analysis["bootstrap"]["root_seed"] == 20261102
            and analysis["groups"] == {k:list(v) for k,v in metric.CORRUPTION_GROUPS.items()}, "Prospective G analysis differs")
    return dict(parent=str(PARENT), prepared_sha256=PARENT_SHA, barrier_sha256=barrier_sha,
                terminal=terminal, inputs=inputs, specs=specs, seals=seals, analysis_sha256=ANALYSIS_SHA)


def prepare(output):
    parent_terminal()
    bound = bind_parent()
    output = Path(output).absolute()
    require(output.resolve() == output and not output.exists(), "Fresh canonical G score suite required")
    output.mkdir()
    inputs_sha = write_prepared_json(output/"SCORE_INPUTS.json", bound)
    protocol_sha = write_prepared_json(output/"PROTOCOL.json", dict(experiment="G_primary_offline_CPU_score", new_GPU=False,
        parent_prepared_sha256=PARENT_SHA, prospective_analysis_sha256=ANALYSIS_SHA,
        GT_boundary="All57fullprediction seals AND exact successful exit AND new exact independent CPU_GT admission",
        label_decode="PIL preserves palette integer indices, then OpenCV INTER_NEAREST224; VOC20 background0 to ignore255 and valid class minus1; original GT unchanged under image-only corruption",
        estimator="Native class I/U streamed one source/arm; pooled per condition then arithmetic mean across15 types/groups",
        bootstrap=dict(replicates=10000, root_seed=20261102, chunk=32, generator="NumPy PCG64",
            key="ASCII compact JSON [20261102,backbone,dataset,primary], SHA256 first16bytes little-endian uint128",
            pairing="Same original source-ID episode draws for all arms and conditions within backbone/dataset; SCLIP/NACLIP B16VOC share draws",
            undefined="Retain all draws; any nonfinite draw yields null interval plus undefined count; no dropping or bin merging"),
        timing="Inherit sealed raw model time/work and descriptive warmup-adjusted throughput, offline score overhead separate; no equal-budget claim",
        disclosures="Historical clean-setting exposure; NoAdapt same TRACE readout, literature R separate from local L; pointwise fixed-configuration uncertainty only",
        failures="Failclosed, no resume/automatic retry, atomic global seal last",
        limits=dict(output_cap_bytes=OUTPUT_CAP, shared_disk_reserve_bytes=RESERVE, GPU=False)))
    inventory = guarded_snapshot(ROOT, output/"runtime", label_inodes(bound["inputs"]))
    return write_prepared_json(output/"SUITE_PREPARED.json", dict(schema_version=1, root=str(output), runtime_hashes=inventory,
        inputs_sha256=inputs_sha, protocol_sha256=protocol_sha,
        parent_prepared_sha256=PARENT_SHA, GT_opened=False, CPU_scoring_started=False))


def admission(suite, expected_sha):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only G scoring required")
    parent_terminal()
    parent = bind_parent()
    forbidden = label_inodes(parent["inputs"])
    suite = Path(suite).resolve(strict=True)
    prepared, _ = metadata(suite/"SUITE_PREPARED.json", expected_sha, forbidden)
    require(prepared["root"] == str(suite) and prepared["parent_prepared_sha256"] == PARENT_SHA
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong G scoring runtime")
    inputs, _ = metadata(suite/"SCORE_INPUTS.json", prepared["inputs_sha256"], forbidden)
    guarded_inventory(suite/"runtime", prepared["runtime_hashes"], forbidden)
    prediction.runtime_imports(suite/"runtime")
    metadata(suite/"PROTOCOL.json", prepared["protocol_sha256"], forbidden)
    require(parent == inputs, "G scoring parent metadata changed")
    audit, audit_sha = metadata(suite/"ADMITTED.json", forbidden=forbidden, limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("CPU_GT_scoring_allowed") is True
            and audit.get("G_primary_GT_scoring_allowed") is True, "New exact G CPU GT admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong G score audit report")
    identities = []
    for path in (suite/"ADMITTED.json", report):
        state = path.stat()
        require((state.st_dev, state.st_ino) not in forbidden, "G audit aliases annotation")
        identities.append([state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns])
    binding = dict(admission_path=str(suite/"ADMITTED.json"), admission_sha256=audit_sha,
        admission_identity=identities[0], report_path=str(report), report_sha256=audit["report_sha256"],
        report_identity=identities[1], independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite, prepared, inputs, binding


def score_condition(suite, bound, spec, budget, label_hashes):
    parent, name = Path(bound["parent"]), spec["id"]
    inventory = bound["seals"][name]["artifact_sha256"]
    rows, cfg = bound["inputs"][spec["input_key"]], spec["config"]
    classes, batch = len(spec["vocabulary"]["classes"]), cfg["batch_size"]
    forbidden = label_inodes(bound["inputs"])
    episodes = np.zeros(((len(rows)+batch-1)//batch, len(ARMS), 2, classes), dtype=np.int64)
    ordered = []
    cost = condition_cost(parent, spec, inventory, forbidden)
    counts_path = suite/(name+".episode_counts.jsonl")
    with counts_path.open("x", encoding="utf-8") as stream:
        for episode, start in enumerate(range(0, len(rows), batch)):
            current = rows[start:start+batch]
            ids = [r["official_id"] for r in current]
            ordered.append(ids)
            for arm in ARMS:
                relative = f"conditions/{name}/episodes/{episode:06d}.{arm}.json"
                saved, _ = metadata(canonical_child(parent, relative), inventory[relative], forbidden)
                require(saved["episode"] == episode and saved["arm"] == arm
                        and saved["ordered_ids"] == ids and saved["GT_opened"] is False,
                        "G actual episode arm/order differs")
            gt_receipts = []
            for index, row in enumerate(current, start):
                content, digest = annotation_bytes(row)
                labels = metric.decode_annotation(content, classes, spec["vocabulary"]["reduce_zero_label"])
                key = row["label"]
                require(key not in label_hashes or label_hashes[key] == digest, "Repeated G source GT bytes changed")
                label_hashes[key] = digest
                gt_receipts.append(dict(source_id=row["official_id"], label=key, sha256=digest))
                for a, arm in enumerate(ARMS):
                    relative = f"conditions/{name}/predictions/{arm}/{index:06d}.npy"
                    path = canonical_child(parent, relative)
                    state = path.stat()
                    require((state.st_dev, state.st_ino) not in forbidden, "Prediction aliases GT")
                    identity = (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
                    raw, _ = stable_bytes(path, 2*1024**2, inventory[relative], expected_identity=identity)
                    prediction_labels = hard_array(raw, classes, (224, 224))
                    episodes[episode, a] += metric.counts(prediction_labels, labels, classes)
            line = json.dumps(dict(episode=episode, ordered_ids=ids, labels=gt_receipts,
                                   I_U_by_arm=episodes[episode].tolist()), allow_nan=False)+"\n"
            size = len(line.encode())
            require(shutil.disk_usage(suite).free >= RESERVE+size and budget["bytes"]+size <= OUTPUT_CAP,
                    "G scoring storage budget exceeded")
            require(stream.write(line) == len(line), "Short G episode-count write")
            budget["bytes"] += size
            stream.flush()
            write_json(suite/"queue_state.json", dict(phase="CPU_G_score", condition=name,
                completed_sources=min(start+batch, len(rows)), total_sources=len(rows), GT_opened=True,
                pid=os.getpid()), replace=True)
        stream.flush()
        os.fsync(stream.fileno())
    return episodes, ordered, cost, counts_path


def execute(suite, bound):
    budget, labels, grouped, results, artifacts = dict(bytes=0), {}, {}, {}, {}
    for spec in bound["specs"]:
        values, ordered, cost, path = score_condition(suite, bound, spec, budget, labels)
        key = (spec["backbone"], spec["dataset"])
        group = grouped.setdefault(key, [])
        require(not group or group[0][2] == ordered, "G shared bootstrap original episodes differ across conditions")
        group.append((spec, values, ordered))
        point = metric.miou(values.sum(0))
        results[spec["id"]] = dict(condition=spec, episodes=len(values), images=spec["images"],
            source="L_local", cost=cost, resources=bound["seals"][spec["id"]]["resource_observation"],
            point_mIoU=[metric.finite_or_none(v) for v in point])
        artifacts[path.name] = sha256(path)
    summaries = {}
    for (backbone, dataset), group in grouped.items():
        data = np.stack([row[1] for row in group], axis=1)
        samples, stream = metric.bootstrap(data, backbone, dataset)
        points = metric.miou(data.sum(0))
        for i, (spec, _, _) in enumerate(group):
            results[spec["id"]].update(metric.describe(points[i], samples[:, i]))
        summary = dict(stream=stream, ordered_source_episodes=group[0][2], conditions=[row[0]["id"] for row in group])
        corrupted = [i for i, row in enumerate(group) if row[0]["corruption"] != "original"]
        if corrupted:
            summary["corruption"] = metric.corruption_summary([group[i][0]["corruption"] for i in corrupted],
                                                             points[corrupted], samples[:, corrupted])
        summaries[backbone+"."+dataset] = summary
    for name, value in (("CONDITION_RESULTS.json", results), ("PAIRED_SUMMARIES.json", summaries),
                        ("ANNOTATIONS_READ.json", labels)):
        content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
        require(shutil.disk_usage(suite).free >= RESERVE+len(content)
                and budget["bytes"]+len(content) <= OUTPUT_CAP, "G scoring summary exceeds pre-write budget")
        with (suite/name).open("xb") as stream:
            require(stream.write(content) == len(content), "Short G score summary write")
            stream.flush()
            os.fsync(stream.fileno())
        budget["bytes"] += len(content)
        artifacts[name] = sha256(suite/name)
    return dict(complete=True, conditions=len(results), cells=sum(len(v["condition"]["arms"]) for v in results.values()),
        image_method_evaluations=sum(v["images"]*len(ARMS) for v in results.values()), artifact_sha256=artifacts,
        GT_opened=True, new_GPU=False, output_bytes=budget["bytes"])


def run(suite, expected_sha):
    suite, prepared, inputs, audit = admission(suite, expected_sha)
    require(not any((suite/n).exists() for n in ("STARTED.json", "FAILED.json", "ANALYSIS_SEALED.json")), "No G score resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP, "Insufficient G scoring headroom")
    verify_audit_binding(audit)
    write_new(suite/"STARTED.json", dict(suite_prepared_sha256=expected_sha, audit=audit, pid=os.getpid(),
        GT_opened=False, automatic_retry=False, parent_barrier_sha256=inputs["barrier_sha256"]))
    started_sha, begin = sha256(suite/"STARTED.json"), time.perf_counter()
    try:
        result = execute(suite, inputs)
        require(result["conditions"] == 19 and result["cells"] == 57
                and result["image_method_evaluations"] == 104214, "Incomplete G score population")
        require(bind_parent() == inputs, "G parent changed before final seal")
        guarded_inventory(suite/"runtime", prepared["runtime_hashes"], label_inodes(inputs["inputs"]))
        metadata(suite/"SCORE_INPUTS.json", prepared["inputs_sha256"])
        metadata(suite/"PROTOCOL.json", prepared["protocol_sha256"])
        metadata(suite/"SUITE_PREPARED.json", expected_sha)
        verify_audit_binding(audit)
        require(sha256(suite/"STARTED.json") == started_sha and not torch.cuda.is_initialized(), "G score startup/CUDA changed")
        for relative, digest in result["artifact_sha256"].items():
            require(sha256(canonical_child(suite, relative)) == digest, "G score child artifact changed")
        write_json(suite/"queue_state.json", dict(phase="G_analysis_payloads_complete_awaiting_seal_and_exit",
            GT_opened=True, pid=os.getpid()), replace=True)
        publish_seal(suite/"ANALYSIS_SEALED.json", dict(**result, suite_prepared_sha256=expected_sha,
            parent_barrier_sha256=inputs["barrier_sha256"], prospective_analysis_sha256=ANALYSIS_SHA,
            audit_binding=audit, started_sha256=started_sha, elapsed_seconds=time.perf_counter()-begin))
    except BaseException as error:
        write_new(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite), GT_opened=False))
    else:
        require(args.suite_sha256, "Exact G scoring prepared SHA required")
        run(args.suite, args.suite_sha256)
