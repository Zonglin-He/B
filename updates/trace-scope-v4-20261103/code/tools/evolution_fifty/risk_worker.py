"""Freeze and execute TRACE-M01 label-free candidate trajectories once admitted.

No scoring entry point: GT support replay is a separately sealed offline phase.
This worker never reads annotations or resumes a failed/cancelled attempt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup.queue import snapshot, check_hashes
from tools.evolution_image_full.run import finite_observation
from tools.evolution_followup.standalone import gpu_lock
from tools.evolution_tuning.common import write_json
from .bootstrap import TUNED, read_json, require
from .intake import sha256, write_new
from .risk_panel import DATASETS, candidate_identity
from .unit_risk import run_candidate, parameter_snapshot, source_prediction, unit_term

ROOT = Path(__file__).resolve().parents[2]
PANEL = Path("/home/wwww/hzl/results/trace_mechanism_20261102/M01_risk_panel_v1")


def groups(panel):
    result = {}
    seen = set()
    for example in panel["examples"]:
        key = (example["source_id"], example["candidate_id"])
        require(key not in seen, "Duplicate risk candidate")
        seen.add(key)
        episode = example["episode"]
        require(type(episode) is int and episode >= 0, "Invalid episode")
        result.setdefault(episode, []).append(example)
    require(len(seen) == panel["candidate_count"] > 0, "Risk candidate population mismatch")
    return result


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output == output.resolve(), "Fresh candidate-risk suite required")
    parent = read_json(PANEL / "PREPARED.json")
    for name, expected in parent["artifact_sha256"].items():
        require(sha256(PANEL/name) == expected, "Frozen risk panel changed")
    output.mkdir(parents=True)
    (output / "panels").mkdir()
    datasets, all_labels = {}, set()
    for dataset in DATASETS:
        source = PANEL/f"{dataset}.panel.json"
        panel = read_json(source)
        episodes = groups(panel)
        original = TUNED / "full" / dataset
        for name, expected in panel["source_files_sha256"].items():
            require(sha256(original/name) == expected, "Historical candidate source changed")
        manifest = read_json(original / "prediction_protocol.json")["manifest"]
        require(manifest["config"] == panel["config"] and manifest["checkpoint"] == panel["original_checkpoint"],
                "Historical risk settings/checkpoint differ")
        rows, batch = manifest["panel"]["inputs"], panel["config"]["batch_size"]
        episode_rows = {str(e): rows[e*batch:(e+1)*batch] for e in episodes}
        for episode, examples in episodes.items():
            actual = episode_rows[str(episode)]
            ids = [r["official_id"] for r in actual]
            for example in examples:
                local = example["candidate"]["image_idx"]
                require(example["episode_ids"] == ids and 0 <= local < len(actual)
                        and actual[local] == example["source_input"]
                        and ids[local] == example["source_id"]
                        and example["original_source_index"] == episode*batch+local,
                        "Candidate/source/episode identity mismatch")
            for row in actual:
                confirmation.verify_label_metadata(row)
                require(sha256(row["image"]) == row["image_sha256"], "Risk RGB content changed")
                all_labels.add(str(Path(row["label"]).resolve()))
        checkpoint = panel["original_checkpoint"]
        require(sha256(checkpoint["path"]) == checkpoint["sha256"], "Checkpoint content changed")
        shutil.copyfile(source, output/"panels"/source.name)
        write_new(output/"panels"/f"{dataset}.episodes.json", episode_rows)
        datasets[dataset] = dict(candidate_count=panel["candidate_count"], episode_count=len(episodes),
            panels_sha256=sha256(output/"panels"/source.name),
            episodes_sha256=sha256(output/"panels"/f"{dataset}.episodes.json"),
            query_to_class=manifest["panel"]["query_to_class"], queries=manifest["panel"]["queries"],
            selected_sources=panel["selected_source_ids"], native_config=panel["config"], checkpoint=checkpoint,
            parent_prediction_seal_sha256=sha256(original/"PREDICTIONS_SEALED.json"))
    write_new(output/"PROTOCOL.json", dict(experiment="TRACE-M01", algorithm="unit-candidate-risk-v1",
        datasets=list(DATASETS), confidence_C_only_offline=True, unit_weight=1., steps="1 and native steps, one trajectory",
        annotation_phase="Only after all candidate predictions/state snapshots sealed; separate read-only replay for GT support",
        state_snapshots="Visual-LN parameters only; used solely for frozen readout replay, not further adaptation",
        analyses=["C quintiles", "original confidence quintiles", "within confidence/class/area matched C groups"],
        primary_support_decline_threshold=.001, sensitivity_support_decline_thresholds=[0., .005],
        continuous_support_delta_always_retained=True,
        gate="First selected actual candidate on each dataset: independent direct-mean loss/gradient/parameter/Adam and label parity",
        atol=1e-5, changed_labels=0, actual_C_cache_gate="All selected candidates: same native geometry/query/C/confidence at theta0",
        max_estimated_snapshot_and_label_bytes=8*1024**3, free_disk_required_bytes=16*1024**3,
        M12="All selected source/candidate IDs, mask,confidence,C and checkpoint labels retained; later cases use prospective rules",
        model_side_time_scope="Candidate optimization plus checkpoint inference; elapsed includes state/receipt writes",
        retries=False, labels_read=False))
    inventory = snapshot(ROOT, output/"runtime")
    write_new(output/"SUITE_PREPARED.json", dict(schema_version=1, root=str(output), runtime_hashes=inventory,
        protocol_sha256=sha256(output/"PROTOCOL.json"), risk_input_prepared_sha256=sha256(PANEL/"PREPARED.json"),
        datasets=datasets, forbidden_labels=sorted(all_labels), new_gpu_started=False))
    print(dict(suite=str(output), prepared_sha256=sha256(output/"SUITE_PREPARED.json"),
               candidates={d: v["candidate_count"] for d,v in datasets.items()}, gpu_started=False))


def admission(suite, expected_sha):
    suite = Path(suite).resolve(strict=True)
    require(sha256(suite/"SUITE_PREPARED.json") == expected_sha, "Wrong frozen risk suite")
    prepared = read_json(suite/"SUITE_PREPARED.json")
    require(prepared["root"] == str(suite) and Path(__file__).resolve().is_relative_to(suite/"runtime"),
            "Worker must execute from the exact frozen runtime")
    check_hashes(suite/"runtime", prepared["runtime_hashes"])
    require(strict_runtime_inventory(suite/"runtime") == prepared["runtime_hashes"],
            "Exact runtime inventory membership changed")
    require(sha256(suite/"PROTOCOL.json") == prepared["protocol_sha256"], "Risk protocol changed")
    audit = read_json(suite/"ADMITTED.json")
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit["new_gpu_allowed"] is True,
            "Independent exact-artifact admission required")
    for dataset, spec in prepared["datasets"].items():
        require(sha256(suite/"panels"/f"{dataset}.panel.json") == spec["panels_sha256"]
                and sha256(suite/"panels"/f"{dataset}.episodes.json") == spec["episodes_sha256"], "Frozen risk inputs changed")
        checked_checkpoint(spec["checkpoint"])
    return suite, prepared


def strict_runtime_inventory(root):
    """Runtime admission has no snapshot-time cache/bytecode exclusions."""
    root = Path(root).resolve(strict=True)
    result, size = {}, 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Runtime symlink is not admitted")
        require(path.name != "__pycache__" and path.suffix not in (".pyc", ".pyo"),
                "Untrusted executable bytecode/cache in frozen runtime")
        if path.is_file():
            size += path.stat().st_size
            require(size <= 64*1024**2, "Frozen runtime exceeds source bound")
            result[str(path.relative_to(root))] = sha256(path)
    return result


def forbid_annotations(paths):
    forbidden = set(paths)
    identities = {(Path(p).stat().st_dev, Path(p).stat().st_ino) for p in paths}
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if str(path) in forbidden:
            raise PermissionError("GT content read attempted in risk prediction phase")
        try:
            status = path.stat()
        except OSError:
            return
        if (status.st_dev, status.st_ino) in identities:
            raise PermissionError("GT alias content read attempted in risk prediction phase")
    sys.addaudithook(audit)


def tensor_delta(a, b):
    require(a.keys() == b.keys(), "Gate tensor membership differs")
    return max((float((a[k].detach().cpu()-b[k].detach().cpu()).abs().max()) for k in a), default=0.)


def adam_tensors(method):
    return {f"{index}:{key}": value.detach().cpu().clone()
            for index, state in method.optimizer.state_dict()["state"].items()
            for key, value in state.items() if torch.is_tensor(value)}


def real_gate(method, images, example, meta, cfg, mapping, rng):
    """Independent direct mean reference; no C score in either update path."""
    candidate = example["candidate"]
    actual, reference, gradients = {}, {}, []
    handles = []
    try:
        def observe_grad(optimizer, args, kwargs):
            gradients.append({n: p.grad.detach().cpu().clone() for n,p in method.model.named_parameters()
                              if p.requires_grad and p.grad is not None})
        handles.append(method.optimizer.register_step_pre_hook(observe_grad))
        actual_losses = run_candidate(method, images, candidate,
            lambda step, current: actual.update({step: dict(parameters=parameter_snapshot(current),
                adam=adam_tensors(current),
                labels=source_prediction(current, images, meta, cfg, mapping, candidate["image_idx"]))}), initial_rng=rng)["losses"]
        actual_gradients = gradients[:]
        gradients.clear()
        method.reset()
        shared.restored_state(method)
        shared.restore_rng(rng, method.device)
        term = unit_term(candidate, images.device)
        losses = []
        with finite_observation(method):
            reference[0] = dict(parameters=parameter_snapshot(method),
                adam=adam_tensors(method),
                labels=source_prediction(method, images, meta, cfg, mapping, candidate["image_idx"]))
            for step in range(1, method.steps+1):
                entropy = method.trace_softmax_entropy(method.forward_adapt_logits(images).float())
                loss = (entropy[:,term["batch_idx"]].flatten(1).index_select(1, term["indices"]).mean()
                        if entropy.ndim == 4 else entropy[term["batch_idx"]].flatten().index_select(0, term["indices"]).mean())
                loss.backward()
                method.optimizer.step()
                method.optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                if step in (1, method.steps):
                    reference[step] = dict(parameters=parameter_snapshot(method),
                        adam=adam_tensors(method),
                        labels=source_prediction(method, images, meta, cfg, mapping, candidate["image_idx"]))
        require(len(actual_gradients) == len(gradients) == method.steps, "Gate update/gradient count mismatch")
        gradient_delta = max(tensor_delta(a,b) for a,b in zip(actual_gradients, gradients, strict=True))
        parameter_delta = max(tensor_delta(actual[k]["parameters"], reference[k]["parameters"]) for k in actual)
        adam_delta = max(tensor_delta(actual[k]["adam"], reference[k]["adam"]) for k in actual)
        changed = sum(int((actual[k]["labels"] != reference[k]["labels"]).sum()) for k in actual)
        loss_delta = max(abs(a-b) for a,b in zip(actual_losses, losses, strict=True))
        require(max(gradient_delta, parameter_delta, adam_delta, loss_delta) <= 1e-5 and changed == 0,
                f"Risk gate mismatch: loss={loss_delta},gradient={gradient_delta},parameter={parameter_delta},Adam={adam_delta},labels={changed}")
        return dict(passed=True, loss_max_abs=loss_delta, gradient_max_abs=gradient_delta,
                    parameter_max_abs=parameter_delta, adam_max_abs=adam_delta, changed_labels=changed, steps=method.steps,
                    gt_contents_opened=False, source_id=example["source_id"], candidate_id=example["candidate_id"])
    finally:
        for handle in handles:
            handle.remove()
        method.reset()
        shared.restored_state(method)


def verify_cached_candidates(method, images, valid, ids, examples):
    method.reset()
    shared.restored_state(method)
    method.estimate_trace_reliability(images, valid_mask=valid, image_ids=ids)
    actual = {candidate_identity(r): r for r in method.last_candidate_stats}
    for example in examples:
        old = example["candidate"]
        require(example["candidate_id"] in actual, "Frozen candidate no longer admitted")
        row = actual[example["candidate_id"]]
        packed = np.packbits(row["region"].reshape(-1), bitorder="little").tobytes().hex()
        require(packed == old["region"]["data"] and list(row["region"].shape) == old["region"]["shape"]
                and row["area"] == old["area"] and row["image_idx"] == old["image_idx"]
                and max(abs(float(row[k])-float(old[k])) for k in ("confidence", "reliability")) <= 1e-5,
                "Native theta0 candidate/support replay mismatch")
    method.reset()


def checked_checkpoint(checkpoint):
    path = Path(checkpoint["path"])
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            "Aliased or missing checkpoint")
    before = path.stat()
    require(path.is_file() and before.st_size == checkpoint["bytes"]
            and before.st_mtime_ns == checkpoint["mtime_ns"], "Checkpoint metadata drift")
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    digest = sha256(path)
    require(digest == checkpoint["sha256"] and identity(path.stat()) == identity(before),
            "Checkpoint content/identity changed while validating")
    return identity(before), digest


def build_method(spec):
    before = checked_checkpoint(spec["checkpoint"])
    cfg = spec["native_config"]
    method = confirmation.ExperimentTRACE(classes=spec["queries"], lr=cfg["lr"], steps=cfg["steps"],
        trace_max_regions=cfg["trace_max_regions"], trace_view_batch_size=cfg["trace_view_batch_size"],
        vision_outputs=confirmation.VISION_OUTPUTS, device="cuda", backbone=spec["checkpoint"]["path"], view_mode="C")
    require(checked_checkpoint(spec["checkpoint"]) == before, "Checkpoint changed during model construction")
    method.arm = "image_C"
    # DenseCLIP wraps an eval-mode CLIP but its wrapper defaults to training=True.
    # All inference/readout stays deterministic; actual C replay is gated below.
    method.model.eval()
    return method


def deterministic_runtime():
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Missing deterministic CUBLAS environment")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gate(suite, expected_sha):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    suite, prepared = admission(suite, expected_sha)
    require(not (suite/"STARTED.json").exists() and not (suite/"FAILED.json").exists(), "No gate after prediction/failure")
    forbid_annotations(prepared["forbidden_labels"])
    write_new(suite/"GATE_PHASE_STARTED.json", dict(suite_prepared_sha256=expected_sha, retry_allowed=False))
    try:
        with gpu_lock() as lock:
            deterministic_runtime()
            reports = {}
            for dataset, spec in prepared["datasets"].items():
                panel = read_json(suite/"panels"/f"{dataset}.panel.json")
                episode_rows = read_json(suite/"panels"/f"{dataset}.episodes.json")
                episode, examples = next(iter(groups(panel).items()))
                cfg, rows = spec["native_config"], episode_rows[str(episode)]
                require(len(rows) >= 4, "Actual gate requires at least four RGB sources")
                method = build_method(spec)
                data = confirmation.make_data(cfg, rows)
                batch = shared.image_collate([data[i] for i in range(len(data))])
                images, meta = batch["img_patches"].to(method.device), batch["meta"]
                valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
                ids = build_patch_image_ids(meta, method.device)
                rng = shared.rng_state(method.device)
                with finite_observation(method):
                    verify_cached_candidates(method, images, valid, ids, examples)
                report = real_gate(method, images, examples[0], meta, cfg, spec["query_to_class"], rng)
                report.update(actual_RGB_sources=len(rows), gpu_lock=lock, suite_prepared_sha256=expected_sha)
                write_new(suite/f"{dataset}.GATE_PASSED.json", report)
                reports[dataset] = sha256(suite/f"{dataset}.GATE_PASSED.json")
                del method, images, batch, data
                torch.cuda.empty_cache()
            write_new(suite/"ALL_GATES_PASSED.json", dict(suite_prepared_sha256=expected_sha, gates=reports))
    except BaseException as error:
        write_new(suite/"FAILED.json", dict(phase="gate", error=f"{type(error).__name__}: {error}", retry_allowed=False))
        raise


def predict(suite, expected_sha):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    suite, prepared = admission(suite, expected_sha)
    require(not (suite/"STARTED.json").exists() and not (suite/"FAILED.json").exists(), "No risk resume/retry")
    boundary = read_json(suite/"ALL_GATES_PASSED.json")
    require(boundary["suite_prepared_sha256"] == expected_sha and set(boundary["gates"]) == set(prepared["datasets"]),
            "Both actual dataset gates required before any prediction")
    for dataset, digest in boundary["gates"].items():
        require(sha256(suite/f"{dataset}.GATE_PASSED.json") == digest, "Gate receipt changed")
    require(shutil.disk_usage(suite).free >= 16*1024**3, "Insufficient risk checkpoint disk budget")
    forbid_annotations(prepared["forbidden_labels"])
    write_new(suite/"STARTED.json", dict(suite_prepared_sha256=expected_sha, pid=os.getpid(), labels_read=False))
    try:
        with gpu_lock() as lock:
            write_new(suite/"GPU_LOCK_ACQUIRED.json", lock)
            deterministic_runtime()
            for dataset, spec in prepared["datasets"].items():
                panel = read_json(suite/"panels"/f"{dataset}.panel.json")
                episode_rows = read_json(suite/"panels"/f"{dataset}.episodes.json")
                cfg = spec["native_config"]
                method = build_method(spec)
                output = suite/dataset
                output.mkdir()
                completed, stored_bytes = 0, 0
                inventory = {}
                for episode, examples in groups(panel).items():
                    rows = episode_rows[str(episode)]
                    data = confirmation.make_data(cfg, rows)
                    batch = shared.image_collate([data[i] for i in range(len(data))])
                    images, meta = batch["img_patches"].to(method.device), batch["meta"]
                    require(list(meta["img_path"]) == [r["image"] for r in rows], "Actual RGB episode order changed")
                    valid = build_patch_valid_mask(meta, cfg["patch_size"], cfg["patch_stride"], method.device)
                    ids = build_patch_image_ids(meta, method.device)
                    rng = shared.rng_state(method.device)
                    with finite_observation(method):
                        verify_cached_candidates(method, images, valid, ids, examples)
                    for example in examples:
                        case = output/f"case_{completed:06d}"
                        case.mkdir()
                        tick = time.perf_counter()
                        def checkpoint(step, current):
                            nonlocal stored_bytes
                            prediction = source_prediction(current, images, meta, cfg, spec["query_to_class"],
                                                           example["candidate"]["image_idx"])
                            label_path = case/f"step_{step:02d}.labels.npy"
                            with label_path.open("xb") as stream:
                                np.save(stream, prediction, allow_pickle=False)
                            snapshot_path = case/f"step_{step:02d}.visual_ln.npz"
                            with snapshot_path.open("xb") as stream:
                                np.savez_compressed(stream, **{n:v.numpy() for n,v in parameter_snapshot(current).items()})
                            for path in (label_path, snapshot_path):
                                inventory[str(path.relative_to(output))] = sha256(path)
                                stored_bytes += path.stat().st_size
                            require(stored_bytes <= 8*1024**3, "Risk checkpoint disk bound exceeded")
                        torch.cuda.reset_peak_memory_stats(method.device)
                        with method.count_work() as counts:
                            report = run_candidate(method, images, example["candidate"], checkpoint, initial_rng=rng)
                        report.update(actual_work=dict(counts), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(method.device),
                                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(method.device))
                        write_new(case/"trajectory.json", dict(report, source_id=example["source_id"],
                            candidate_id=example["candidate_id"], original_episode=episode,
                            elapsed_seconds=time.perf_counter()-tick, gt_contents_opened=False))
                        inventory[str((case/"trajectory.json").relative_to(output))] = sha256(case/"trajectory.json")
                        completed += 1
                        write_json(suite/"queue_state.json", dict(phase="predict", dataset=dataset,
                            completed_candidates=completed, total_candidates=spec["candidate_count"],
                            stored_bytes=stored_bytes, labels_read=False, pid=os.getpid()), replace=True)
                    del images, batch, data
                require(completed == spec["candidate_count"], "Incomplete risk candidate population")
                write_new(output/"PREDICTIONS_SEALED.json", dict(complete=True, candidates=completed,
                    gt_contents_opened=False, artifact_sha256=inventory,
                    dataset_spec=spec, suite_prepared_sha256=expected_sha))
                del method
                torch.cuda.empty_cache()
            write_new(suite/"ALL_RISK_PREDICTIONS_SEALED.json", dict(complete=True, gt_contents_opened=False,
                dataset_seals={d:sha256(suite/d/"PREDICTIONS_SEALED.json") for d in prepared["datasets"]}))
            write_json(suite/"queue_state.json", dict(phase="predictions_sealed_awaiting_offline_support_replay",
                labels_read=False, scientific_completion=False, pid=os.getpid()), replace=True)
    except BaseException as error:
        write_new(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", retry_allowed=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "gate", "predict", "run"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.suite)
    else:
        require(args.suite_sha256, "Frozen suite SHA required")
        if args.phase == "run":
            gate(args.suite, args.suite_sha256)
            predict(args.suite, args.suite_sha256)
        else:
            {"gate": gate, "predict": predict}[args.phase](args.suite, args.suite_sha256)
