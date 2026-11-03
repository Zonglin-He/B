"""Bounded label-free fixed-state M01 replay; GT remains a separate CPU phase.

No adaptation, resume or automatic retry. Preparation does not authorize GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup.queue import check_hashes, snapshot
from tools.evolution_followup.standalone import gpu_lock
from tools.evolution_tuning.common import write_json
from .bootstrap import read_json, require
from .intake import sha256, write_new
from .risk_replay import decode_ln_snapshot, fixed_ln_state, replay_one, source_support
from .risk_replay_manifest import bind_parent
from .risk_worker import build_method, checked_checkpoint, deterministic_runtime, forbid_annotations, strict_runtime_inventory

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path("/home/wwww/hzl/results/trace_mechanism_20261102/M01_unit_risk_20261102_1540")
PARENT_PREPARED = "625c870d17e41be1ef6516612190662ded8dc47ccef35d8c2c7c0e2cc64b8cac"
PARENT_BARRIER = "e21a5846160e1f65b95def70a19e43b3a18209489dd459b307666024da457528"
PARENT_UNIT = "trace-unit-risk-20261102-1540.service"
PARENT_INVOCATION = "b8ed603e881b43ce8310f1f2ec10daf5"
DISK_LIMIT = 96*1024**3
FREE_REQUIRED = 128*1024**3


def parent_terminal():
    keys = ("LoadState", "ActiveState", "SubState", "MainPID", "ExecMainPID", "ExecMainStatus", "Result", "InvocationID")
    result = subprocess.run(["systemctl", "--user", "show", PARENT_UNIT,
        "--property="+",".join(keys)], check=True, text=True, capture_output=True, timeout=30)
    actual = dict(row.split("=", 1) for row in result.stdout.splitlines() if "=" in row)
    expected = dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
        ExecMainPID="1826607", ExecMainStatus="0", Result="success", InvocationID=PARENT_INVOCATION)
    require(all(actual.get(k) == v for k, v in expected.items()), "Parent has no exact retained successful terminal identity")
    return actual


def payload(path, digest, max_bytes):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and not path.is_symlink() and path.is_file(), "Aliased replay payload")
    before = path.stat()
    require(0 < before.st_size <= max_bytes, "Replay payload exceeds bound")
    content = path.read_bytes()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(path.stat()) == identity(before)
            and hashlib.sha256(content).hexdigest() == digest, "Replay payload changed")
    return content


def reference_key(dataset, case, checkpoint):
    value = dict(dataset=dataset, original_episode=case["episode"], source_id=case["source_id"],
                 snapshot_sha256=checkpoint["snapshot_sha256"])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def native_identity():
    parent = read_json(PARENT/"SUITE_PREPARED.json")
    require(sha256(PARENT/"SUITE_PREPARED.json") == PARENT_PREPARED, "Parent preparation changed")
    prefixes = ("trace_ovss/", "tools/evolution/", "tools/evolution_confirmation/",
                "tools/evolution_followup/", "tools/evolution_image_full/")
    critical = ("tools/evolution_fifty/risk_worker.py", "tools/evolution_fifty/unit_risk.py")
    names = {n: h for n, h in parent["runtime_hashes"].items() if n.startswith(prefixes) or n in critical}
    require(all(n in names for n in critical) and any(n.startswith("trace_ovss/") for n in names),
            "Parent native runtime inventory incomplete")
    require(all(sha256(ROOT/n) == h for n, h in names.items()), "Replay native implementation differs from parent")
    return names


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh replay suite required")
    terminal = parent_terminal()
    bound = bind_parent(PARENT, PARENT_PREPARED, PARENT_BARRIER)
    native_hashes = native_identity()
    require({k: len(v["cases"]) for k, v in bound["datasets"].items()}
            == dict(coco_object=1024, pc60=1018), "Parent candidate population changed")
    for dataset in bound["datasets"].values():
        checked_checkpoint(dataset["spec"]["checkpoint"])
        require(tuple(dataset["spec"]["native_config"]["resize"])
                == tuple(dataset["spec"]["native_config"]["patch_size"]) == (224, 224),
                "Replay is bound to the existing one-crop224 protocol")
        for case in dataset["cases"]:
            for point in case["checkpoints"]:
                point["snapshot_sha256"] = dataset["artifact_sha256"][point["snapshot"]]
                point["labels_sha256"] = dataset["artifact_sha256"][point["labels"]]
    output.mkdir()
    write_new(output/"REPLAY_INPUTS.json", bound)
    write_new(output/"PROTOCOL.json", dict(experiment="M01_fixed_state_label_free_support_replay",
        parent=str(PARENT), parent_prepared_sha256=PARENT_PREPARED, parent_barrier_sha256=PARENT_BARRIER,
        parent_retained_terminal=terminal, no_adaptation=True, no_optimizer_step_or_reset=True,
        saved_steps=dict(coco_object=[0, 1, 30], pc60=[0, 1, 7]),
        probability_readout="Production patch aggregation, original-dtype softmax, MAX extension mapping, no renormalization",
        exact_gate="EVERY saved hard prediction must match replay exactly BEFORE support storage/reuse",
        real_gate="Both datasets, first original episode >=4 actual RGB sources, saved0/1state: independent direct one-crop softmax/MAX support <=1e-5 and0changedlabels before any full support outputs",
        reuse="Only same dataset/spec plus original episode/source and exact saved snapshot digest; same cached support must match every sealed hard label",
        integrity="Full frozen-tensor byte digest and all registered paths/live optimizer binding; verification costs included in elapsed/counters, no accuracy-budget claim",
        annotation_boundary="No GT read until ALL replay supports sealed and replay service genuinely exited successfully; separate CPU scoring artifact",
        disk_payload_limit_bytes=DISK_LIMIT, free_disk_required_bytes=FREE_REQUIRED,
        minimum_free_bytes_before_each_support_write=16*1024**3,
        arrays="Lossless native-float per-source class support, never full-episode probability accumulation; one case at a time",
        failure="Failclosed, retire contaminated model, preserve evidence; no retry/resume or support seal on failure",
        GT_opened=False, scientific_completion=False))
    runtime = snapshot(ROOT, output/"runtime")
    write_new(output/"SUITE_PREPARED.json", dict(schema_version=1, root=str(output), runtime_hashes=runtime,
        protocol_sha256=sha256(output/"PROTOCOL.json"), inputs_sha256=sha256(output/"REPLAY_INPUTS.json"),
        parent_native_hashes=native_hashes, parent_prepared_sha256=PARENT_PREPARED,
        parent_barrier_sha256=PARENT_BARRIER, GPU_started=False))
    return sha256(output/"SUITE_PREPARED.json")


def admission(suite, expected_sha):
    suite = Path(suite).resolve(strict=True)
    require(sha256(suite/"SUITE_PREPARED.json") == expected_sha, "Wrong frozen replay suite")
    prepared = read_json(suite/"SUITE_PREPARED.json")
    require(prepared["root"] == str(suite) and Path(__file__).resolve().is_relative_to(suite/"runtime"),
            "Replay must execute from its exact frozen runtime")
    check_hashes(suite/"runtime", prepared["runtime_hashes"])
    require(strict_runtime_inventory(suite/"runtime") == prepared["runtime_hashes"], "Replay runtime membership differs")
    require(sha256(suite/"PROTOCOL.json") == prepared["protocol_sha256"]
            and sha256(suite/"REPLAY_INPUTS.json") == prepared["inputs_sha256"], "Replay protocol/input drift")
    audit = read_json(suite/"ADMITTED.json")
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("label_free_support_replay_allowed") is True,
            "New independent exact-artifact support-replay admission required")
    parent_terminal()
    require(native_identity() == prepared["parent_native_hashes"], "Replay native source identity changed")
    bound = bind_parent(PARENT, prepared["parent_prepared_sha256"], prepared["parent_barrier_sha256"])
    frozen = read_json(suite/"REPLAY_INPUTS.json")
    # Snapshot/label digests are copied into the prospective per-checkpoint rows.
    for dataset in bound["datasets"].values():
        for case in dataset["cases"]:
            for point in case["checkpoints"]:
                point["snapshot_sha256"] = dataset["artifact_sha256"][point["snapshot"]]
                point["labels_sha256"] = dataset["artifact_sha256"][point["labels"]]
    require(bound == frozen, "Parent replay references changed after freeze")
    return suite, prepared, frozen


def decode_labels(content, shape, classes):
    require(len(content) <= 2*1024**2, "Hard-prediction payload exceeds bound")
    stream = io.BytesIO(content)
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), "Invalid hard-prediction NPY version")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    dimensions, fortran, dtype = reader(stream, max_header_size=4096)
    require(dimensions == tuple(shape) and not fortran and dtype == np.dtype("uint16")
            and len(content) == stream.tell()+np.prod(shape)*dtype.itemsize, "Hard-prediction header differs")
    labels = np.load(io.BytesIO(content), allow_pickle=False)
    require(bool((labels < classes).all()), "Hard prediction outside mapped vocabulary")
    return labels


def run(suite, expected_sha):
    suite, prepared, bound = admission(suite, expected_sha)
    require(not any((suite/name).exists() for name in ("STARTED.json", "FAILED.json")), "No replay resume/retry")
    require(shutil.disk_usage(suite).free >= FREE_REQUIRED, "Insufficient bounded support storage headroom")
    labels = {r["label"] for d in bound["datasets"].values() for rows in d["episodes"].values() for r in rows}
    forbid_annotations(sorted(labels))
    write_new(suite/"STARTED.json", dict(pid=os.getpid(), suite_prepared_sha256=expected_sha,
        GT_opened=False, no_adaptation=True, automatic_retry=False))
    try:
        gates(suite, expected_sha, bound)
        execute(suite, expected_sha, bound)
    except BaseException as error:
        write_new(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}",
            GT_opened=False, automatic_retry=False))
        raise


def load_episode(method, cfg, rows):
    for row in rows:
        confirmation.verify_label_metadata(row)
        require(sha256(row["image"]) == row["image_sha256"], "Replay RGB drift")
    data = confirmation.make_data(cfg, rows)
    batch = shared.image_collate([data[i] for i in range(len(data))])
    images, meta = batch["img_patches"].to(method.device), batch["meta"]
    require(list(meta["img_path"]) == [r["image"] for r in rows], "Replay RGB order differs")
    return images, meta


def gates(suite, expected_sha, bound):
    reports = {}
    with gpu_lock() as lock:
        deterministic_runtime()
        for dataset, source in bound["datasets"].items():
            spec, case = source["spec"], source["cases"][0]
            cfg, mapping = spec["native_config"], spec["query_to_class"]
            classes = len(spec["queries"]) if mapping is None else max(mapping)+1
            rows = source["episodes"][str(case["episode"])]
            require(len(rows) >= 4, "Actual replay gate requires at least four RGB sources")
            method = build_method(spec)
            try:
                images, meta = load_episode(method, cfg, rows)
                require(len(images) == len(rows) and all(tuple(g) == (1, 1) for g in meta["patch_grid_shape"])
                        and all(tuple(s) == tuple(cfg["patch_size"]) for s in meta["img_shape"]),
                        "Independent direct gate requires exact full single crops")
                parameters = {n: p for n, p in method.model.named_parameters() if p.requires_grad}
                actual = []
                for point in case["checkpoints"][:2]:
                    parent = Path(bound["parent"])/dataset
                    expected = decode_labels(payload(parent/point["labels"], point["labels_sha256"], 2*1024**2),
                                             cfg["resize"], classes)
                    state = decode_ln_snapshot(payload(parent/point["snapshot"], point["snapshot_sha256"], 64*1024**2),
                                               point["snapshot_sha256"], parameters)
                    local = case["example"]["candidate"]["image_idx"]
                    with method.count_work() as work, fixed_ln_state(method, state):
                        reference = method.evaluate(images)[local].softmax(0)
                        if mapping is not None:
                            reference = torch.stack([reference[[i for i, c in enumerate(mapping) if c == cls]].amax(0)
                                                     for cls in range(classes)])
                        reference = reference.detach().cpu().numpy()
                        actual_support = source_support(method, images, meta, cfg, mapping, local)
                        require(reference.dtype == actual_support.dtype and reference.shape == actual_support.shape,
                                "Independent gate support type/geometry differs")
                        maximum = float(np.abs(reference.astype(np.float64)-actual_support.astype(np.float64)).max())
                        changed = int((actual_support.argmax(0) != expected).sum())
                        require(np.isfinite(maximum) and maximum <= 1e-5 and changed == 0
                                and np.array_equal(reference.argmax(0), expected), "Independent actual replay gate failed")
                    actual.append(dict(step=point["step"], support_max_abs=maximum, changed_labels=changed,
                                       actual_work=dict(work)))
                write_new(suite/f"{dataset}.GATE_PASSED.json", dict(passed=True, states=actual,
                    actual_RGB_sources=len(rows), suite_prepared_sha256=expected_sha, gpu_lock=lock, GT_opened=False))
                reports[dataset] = sha256(suite/f"{dataset}.GATE_PASSED.json")
            finally:
                del method
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        write_new(suite/"ALL_GATES_PASSED.json", dict(suite_prepared_sha256=expected_sha, gates=reports, GT_opened=False))


def execute(suite, expected_sha, bound):
    """The orchestration body is separately CPU-tested; production uses run()."""
    stored_bytes, completed = 0, 0
    inventory, dataset_seals = {}, {}
    gate_boundary = read_json(suite/"ALL_GATES_PASSED.json")
    require(gate_boundary["suite_prepared_sha256"] == expected_sha
            and set(gate_boundary["gates"]) == set(bound["datasets"]), "Both replay gates required first")
    require(all(sha256(suite/f"{d}.GATE_PASSED.json") == h for d, h in gate_boundary["gates"].items()),
            "Replay gate receipts changed")
    start = time.perf_counter()
    with gpu_lock() as lock:
        deterministic_runtime()
        write_new(suite/"GPU_LOCK_ACQUIRED.json", lock)
        for dataset, source in bound["datasets"].items():
            spec, parent = source["spec"], Path(bound["parent"])/dataset
            cfg, mapping = spec["native_config"], spec["query_to_class"]
            classes = len(spec["queries"]) if mapping is None else max(mapping)+1
            method = build_method(spec)
            parameters = {n: p for n, p in method.model.named_parameters() if p.requires_grad}
            output = suite/dataset
            output.mkdir()
            (output/"supports").mkdir()
            references, cache = [], {}
            try:
                cases = {}
                for case in source["cases"]:
                    cases.setdefault(case["episode"], []).append(case)
                for episode, episode_cases in cases.items():
                    rows = source["episodes"][str(episode)]
                    images, meta = load_episode(method, cfg, rows)
                    for case in episode_cases:
                        example = case["example"]
                        local = example["candidate"]["image_idx"]
                        trajectory = json.loads(payload(parent/case["trajectory"],
                            source["artifact_sha256"][case["trajectory"]], 1024**2))
                        require(trajectory["source_id"] == case["source_id"]
                                and trajectory["candidate_id"] == case["candidate_id"]
                                and trajectory["original_episode"] == case["episode"]
                                and trajectory["actual_steps"] == cfg["steps"]
                                and trajectory["gt_contents_opened"] is False,
                                "Sealed trajectory identity differs from replay candidate")
                        points = []
                        for point in case["checkpoints"]:
                            expected = decode_labels(payload(parent/point["labels"], point["labels_sha256"], 2*1024**2),
                                                     cfg["resize"], classes)
                            key = reference_key(dataset, case, point)
                            content = payload(parent/point["snapshot"], point["snapshot_sha256"], 64*1024**2)
                            if key not in cache:
                                state = decode_ln_snapshot(content, point["snapshot_sha256"], parameters)
                                shared.synchronize(method.device)
                                tick = time.perf_counter()
                                with method.count_work() as work:
                                    probabilities = replay_one(method, images, meta, cfg, mapping, local, state, expected)
                                shared.synchronize(method.device)
                                elapsed = time.perf_counter()-tick
                                require(shutil.disk_usage(suite).free >= 16*1024**3+probabilities.nbytes+4096,
                                        "Support write would consume reserved shared-disk headroom")
                                target = output/"supports"/f"{key}.npz"
                                with target.open("xb") as stream:
                                    np.savez_compressed(stream, probabilities=probabilities)
                                stored_bytes += target.stat().st_size
                                require(stored_bytes <= DISK_LIMIT, "Support payload disk bound exceeded")
                                relative = str(target.relative_to(suite))
                                inventory[relative] = sha256(target)
                                cache[key] = dict(support=relative, sha256=inventory[relative],
                                    probabilities_dtype=str(probabilities.dtype), shape=list(probabilities.shape),
                                    hard_labels_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                                    actual_work=dict(work), replay_verification_seconds=elapsed)
                                del probabilities, state
                            else:
                                require(hashlib.sha256(expected.tobytes()).hexdigest()
                                        == cache[key]["hard_labels_sha256"], "Shared snapshot/source has inconsistent sealed hard labels")
                            del content
                            points.append(dict(point, reference=cache[key], exact_changed_labels=0))
                        references.append(dict(case=case["case"], episode=episode, source_id=case["source_id"],
                            candidate_id=case["candidate_id"], checkpoints=points))
                        completed += 1
                        write_json(suite/"queue_state.json", dict(phase="label_free_fixed_state_replay", dataset=dataset,
                            completed_candidates=completed, stored_bytes=stored_bytes, pid=os.getpid(), GT_opened=False), replace=True)
                    del images, meta
                require(len(references) == len(source["cases"]) == spec["candidate_count"], "Incomplete replay dataset")
                write_new(output/"REFERENCES.json", references)
                reference_path = str((output/"REFERENCES.json").relative_to(suite))
                inventory[reference_path] = sha256(output/"REFERENCES.json")
                write_new(output/"SUPPORTS_SEALED.json", dict(complete=True, candidates=len(references),
                    unique_readouts=len(cache), references_sha256=inventory[reference_path], GT_opened=False,
                    suite_prepared_sha256=expected_sha,
                    artifact_sha256={k: v for k, v in inventory.items() if k.startswith(dataset+"/")}))
                dataset_seals[dataset] = sha256(output/"SUPPORTS_SEALED.json")
            finally:
                del method
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        require(set(dataset_seals) == set(bound["datasets"])
                and completed == sum(v["spec"]["candidate_count"] for v in bound["datasets"].values()),
                "Incomplete all-dataset support barrier")
    # Last bookkeeping and lock teardown precede publication of the global seal.
    # The state is advisory: the seal AND retained successful exit remain required.
    write_json(suite/"queue_state.json", dict(phase="payloads_complete_awaiting_support_seal_and_exit",
        candidates=completed, GT_opened=False, scientific_completion=False, pid=os.getpid()), replace=True)
    write_new(suite/"ALL_SUPPORTS_SEALED.json", dict(complete=True, dataset_seals=dataset_seals,
        suite_prepared_sha256=expected_sha, candidates=completed, stored_bytes=stored_bytes,
        elapsed_seconds=time.perf_counter()-start, GT_opened=False, no_adaptation=True, scientific_completion=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite), GPU_started=False))
    else:
        require(args.suite_sha256, "Expected frozen suite digest required")
        run(args.suite, args.suite_sha256)
