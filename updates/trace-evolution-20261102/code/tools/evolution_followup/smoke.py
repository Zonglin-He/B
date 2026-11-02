"""Four-image GT-free native TwoViewTRACE / standalone-min equivalence gate.

Only an explicitly launched frozen queue uses this GPU gate. No scoring, label
decode, dispatcher override, background process or retry is implemented here.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import random

import numpy as np
import torch

from tools.evolution_followup import queue
from tools.evolution_followup import standalone as worker
from tools.trace_execution_audit import ExecutionAudit
from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask, collapse_class_extensions
from trace_ovss.configs import VISION_OUTPUTS
from trace_ovss.evaluation import aggregate_patch_logits
from trace_ovss.method.two_view import TwoViewTRACE


ATOL = 1e-5


@torch.no_grad()
def final_predictions(method, images, meta, config, mapping):
    logits = method.evaluate(images)
    queue.require(bool(torch.isfinite(logits).all()), "Nonfinite final logits")
    outputs = aggregate_patch_logits(logits, meta["patch_grid_shape"], meta["img_shape"],
                                     config["patch_size"], config["patch_stride"])
    queue.require(len(outputs) == len(meta["img_shape"]), "Reconstruction count differs")
    predictions = []
    classes = max(mapping) + 1 if mapping is not None else len(method.classes)
    for output, expected_shape in zip(outputs, meta["img_shape"]):
        queue.require(bool(torch.isfinite(output).all()), "Nonfinite reconstructed logits")
        probabilities = output.softmax(0)
        if mapping is not None:
            probabilities = collapse_class_extensions(probabilities, mapping)
        queue.require(bool(torch.isfinite(probabilities).all()), "Nonfinite final probabilities")
        prediction = probabilities.argmax(0).cpu().numpy().astype(np.uint16)
        queue.require(prediction.shape == tuple(expected_shape) and prediction.ndim == 2
                      and np.isfinite(prediction).all() and np.all(prediction < classes), "Invalid final label shape/range")
        predictions.append(prediction)
    return predictions


def compare_episode(method, images, valid, image_ids, meta, config, mapping):
    """One object, exact reset/RNG state, switch classes only outside live audits."""
    original_class = type(method)
    rng = worker.shared.rng_state(method.device)
    method.reset()
    initial = worker.shared.restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    outcomes = []
    try:
        for cls in (TwoViewTRACE, worker.StandaloneTRACE):
            method.__class__ = cls
            method.arm = "min"
            method.reset()
            queue.require(worker.shared.restored_state(method) == initial, "Smoke initial state differs")
            worker.shared.restore_rng(rng, method.device)
            with ExecutionAudit(method) as audit:
                losses = method.adapt(images, valid_mask=valid, image_ids=image_ids)
                predictions = final_predictions(method, images, meta, config, mapping)
                worker.shared.synchronize(method.device)
                counts = audit.snapshot()
            stats = copy.deepcopy(method.last_candidate_stats)
            queue.require(stats and len(losses) == config["steps"], "Smoke did not perform complete adaptation")
            queue.require(all(math.isfinite(v) for v in losses), "Nonfinite adaptation loss")
            for row in stats:
                queue.require(all(row[key] is not None and math.isfinite(row[key]) and 0 <= row[key] <= 1
                                  for key in ("target_support", "counterfactual_support", "reliability")),
                              "Invalid native/min support")
            views = 2 * len(stats)
            calls = (views + config["trace_view_batch_size"] - 1) // config["trace_view_batch_size"]
            queue.require(counts["failed_forward_attempts"] == 0 and counts["diagnostic_calls"] == calls
                          and counts["calls"] == 2 + config["steps"] + calls
                          and counts["inputs"] == (2 + config["steps"]) * len(images) + views,
                          "Smoke actual encoder schedule/counts differ or native OOM fallback occurred")
            queue.require(method.last_trace_view_batch_size == config["trace_view_batch_size"], "View batching changed")
            queue.require(not any(p._version != version for p, version in frozen), "Frozen weights changed")
            outcomes.append(dict(losses=losses, stats=stats, predictions=predictions, counts=counts))
    finally:
        method.__class__ = original_class
        method.reset()
        worker.shared.restore_rng(rng, method.device)
    a, b = outcomes
    masks_equal = worker.candidate_identity(a["stats"]) == worker.candidate_identity(b["stats"])
    same_steps = len(a["losses"]) == len(b["losses"])
    support_diff = max((abs(x[key] - y[key]) for x, y in zip(a["stats"], b["stats"])
                        for key in ("target_support", "counterfactual_support", "reliability")), default=0.)
    loss_diff = max((abs(x - y) for x, y in zip(a["losses"], b["losses"])), default=0.)
    same_shapes = (len(a["predictions"]) == len(b["predictions"])
                   and all(x.shape == y.shape for x, y in zip(a["predictions"], b["predictions"])))
    queue.require(same_shapes, "Native/standalone final label shapes differ")
    changed = int(sum(np.count_nonzero(x != y) for x, y in zip(a["predictions"], b["predictions"])))
    total = sum(x.size for x in a["predictions"])
    report = dict(candidate_masks_equal=masks_equal, same_steps=same_steps,
        max_support_abs_difference=support_diff, max_loss_abs_difference=loss_diff,
        changed_label_pixels=changed, total_label_pixels=total, atol=ATOL,
        finite=True, count_checks_passed=True,
        prediction_shapes=[list(x.shape) for x in a["predictions"]],
        candidate_sha256=worker.candidate_identity(a["stats"]), candidates=len(a["stats"]),
        runs={name: dict(steps=len(row["losses"]), encoder_counts=row["counts"])
              for name, row in zip(("native", "standalone_min"), outcomes)})
    report["passed"] = bool(masks_equal and same_steps and support_diff <= ATOL and loss_diff <= ATOL and changed == 0)
    return report


def run(args):
    suite = args.suite.resolve()
    prepared = queue.verify_prepared(suite, args.suite_sha256, frozen=True)
    entry = next(e for e in prepared["smokes"] if e["dataset"] == args.dataset)
    output = suite / entry["output"]
    queue.require(not output.exists(), "Smoke output exists; no retry/overwrite")
    manifest = worker.verify_manifest_file(suite / entry["manifest"], entry["manifest_sha256"])
    queue.require(worker.checked_sources() == manifest["source_sha256"], "Smoke source changed")
    data, indices, panel = worker.build_panel(args.dataset, manifest["config"], 20261029, 4)
    queue.require(panel == manifest["panel"], "Smoke RGB-only panel changed")
    checkpoint = worker.shared.checkpoint_identity(manifest["checkpoint"]["path"])
    queue.require(checkpoint == manifest["checkpoint"], "Smoke official checkpoint changed")
    report = None
    try:
        with worker.gpu_lock() as lock:
            runtime = worker.shared.configure_deterministic_runtime()
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            config = manifest["config"]
            method = worker.StandaloneTRACE(classes=panel["queries"], lr=config["lr"], steps=config["steps"],
                trace_max_regions=config["trace_max_regions"], trace_view_batch_size=config["trace_view_batch_size"],
                vision_outputs=VISION_OUTPUTS, device="cuda", backbone=checkpoint["path"])
            queue.require(worker.shared.checkpoint_identity(checkpoint["path"]) == checkpoint, "Smoke checkpoint changed on load")
            batch = worker.shared.image_collate([data[i] for i in indices])
            images, meta = batch["img_patches"].to(method.device), batch["meta"]
            valid = build_patch_valid_mask(meta, config["patch_size"], config["patch_stride"], method.device)
            image_ids = build_patch_image_ids(meta, method.device)
            report = compare_episode(method, images, valid, image_ids, meta, config, panel["query_to_class"])
            queue.require(worker.shared.deterministic_runtime_state() == runtime, "Smoke deterministic runtime changed")
            queue.require(worker.checked_sources() == manifest["source_sha256"], "Smoke source changed during gate")
            queue.require(all(queue.sha256(row["image"]) == row["image_sha256"] for row in panel["inputs"]), "Smoke RGB changed")
            queue.verify_prepared(suite, args.suite_sha256, frozen=True)
            report.update(dataset=args.dataset, images=4, gt_contents_opened=False,
                manifest_sha256=entry["manifest_sha256"], suite_prepared_sha256=args.suite_sha256,
                ordered_ids=panel["ordered_ids"], input_image_sha256={r["image"]: r["image_sha256"] for r in panel["inputs"]},
                deterministic_runtime=runtime, checkpoint=checkpoint, gpu_lock=lock,
                initial_model_sha256=worker.state_digest(method.model.state_dict()),
                device=str(method.device), source_sha256=manifest["source_sha256"])
        queue.write_json(output, report)
        queue.require(report["passed"], "Native/standalone complete-episode equivalence gate failed")
    except BaseException as error:
        if not output.exists():
            queue.write_json(output, dict(passed=False, dataset=args.dataset, images=4, gt_contents_opened=False,
                manifest_sha256=entry["manifest_sha256"], error=f"{type(error).__name__}: {error}"[:2000]))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--dataset", choices=queue.DATASETS, required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
