"""Run the fixed TRACE simplification panel sequentially on one GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from benchmark_simplification import source_hashes, write_json
from suite_inputs import expected_inputs, freeze_protocol, validate_cell_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("benchmark_simplification.py")
    manifest_path = args.output / "suite_protocol.json"
    resolved_data_root = str(args.data_root.resolve())
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("data_root") != resolved_data_root:
        raise RuntimeError("Suite data root changed or was not sealed; choose a fresh suite directory")
    plan = {f"calibration/{dataset}/40": (dataset, "calibration", 40)
            for dataset in ("pc60", "coco_stuff")}
    plan.update({f"evaluation/{dataset}/{limit}": (dataset, "evaluation", limit)
                 for dataset, limit in (("coco_cd", 56), ("coco_cd", None),
                                       ("pc60", 512), ("coco_stuff", 512), ("voc21", 512))})
    sealed_inputs = expected_inputs(args.candidate_root, args.data_root.resolve(), plan)
    protocol = dict(
        schema_version=2, data_root=resolved_data_root, input_panels=sealed_inputs,
        reference_root=str(args.reference_root.resolve()), candidate_root=str(args.candidate_root.resolve()),
        reference_sha256=source_hashes(args.reference_root / "trace_ovss"),
        candidate_sha256=source_hashes(args.candidate_root / "trace_ovss"),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_preparation_sha256=hashlib.sha256(Path(__file__).with_name("suite_inputs.py").read_bytes()).hexdigest(),
        datasets=["coco_cd", "pc60", "coco_stuff", "voc21"],
        evaluation_images=dict(coco_cd=None, pc60=512, coco_stuff=512, voc21=512),
        seed=0, selection_seed=20261024, calibration_images=40,
        view_batch_options=[8, 16, 32, 64],
        batch_selection="minimize mean relative synchronized runtime on PC60/COCO-Stuff train images; within 3% prefer smaller view batch; no score-based selection",
        acceptance="fixed-config accuracy/cost comparison; not a new full multi-seed main table",
    )
    freeze_protocol(manifest_path, protocol)
    rows = []

    def cell(name, dataset, package, *, split="evaluation", limit=None, view_batch=None, uniform=False):
        output = args.output / split / dataset / name
        command = [sys.executable, str(runner), "--package-root", str(package),
                   "--data-root", str(args.data_root), "--dataset", dataset,
                   "--split", split, "--name", name, "--output", str(output)]
        if limit:
            command += ["--max-images", str(limit)]
        if view_batch:
            command += ["--view-batch-size", str(view_batch)]
        if uniform:
            command += ["--force-uniform-policy"]
        path = output / "result.json"
        output.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            write_json(args.output / "queue_state.json", dict(status="running", name=name,
                       dataset=dataset, split=split, command=command))
            with (output / "stdout.log").open("a") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                write_json(args.output / "queue_state.json", dict(status="failed", name=name,
                           dataset=dataset, split=split, exit_code=completed.returncode))
                raise RuntimeError(f"Cell failed; inspect {output / 'stdout.log'}")
        result = json.loads(path.read_text())
        saved = json.loads((output / "protocol.json").read_text())
        assert result["status"] == "completed"
        assert saved["source_hashes"] == source_hashes(package / "trace_ovss")
        assert saved["package_root"] == str(package.resolve())
        assert saved["name"] == name and saved["dataset"] == dataset and saved["split"] == split
        validate_cell_inputs(saved, sealed_inputs[f"{split}/{dataset}/{limit}"])
        assert saved["force_uniform_policy"] == uniform
        assert result["protocol_sha256"] == hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()
        if view_batch:
            assert saved["config"]["trace_view_batch_size"] == view_batch
        expected_count = min(limit, saved["available_images"]) if limit else saved["available_images"]
        assert result["images"] == expected_count == saved["images"] == len(saved["image_ids"])
        row = dict(name=name, dataset=dataset, split=split, images=result["images"],
                   full_split=result["full_split"], mIoU=result["metrics"]["mIoU"],
                   miou_unrounded=result["miou_unrounded"], ms_per_image=result["ms_per_image"],
                   peak_allocated_mib=result["peak_allocated_mib"],
                   diagnostic_views_per_image=result["diagnostic_views_per_image"], result_path=str(path))
        rows.append(row)
        with (args.output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        return result

    calibration = {}
    for dataset in ("pc60", "coco_stuff"):
        for size in protocol["view_batch_options"]:
            result = cell(f"unified{size}", dataset, args.candidate_root,
                          split="calibration", limit=40, view_batch=size)
            calibration[dataset, size] = result["ms_per_image"]
    relative_cost = {size: sum(calibration[d, size] / calibration[d, 8]
                              for d in ("pc60", "coco_stuff")) / 2
                     for size in protocol["view_batch_options"]}
    fastest = min(relative_cost.values())
    selected = min(size for size, ratio in relative_cost.items() if ratio <= fastest * 1.03)
    write_json(args.output / "batch_selection.json", dict(selected_view_batch=selected,
               relative_cost=relative_cost, selection_uses_miou=False))

    # Same three-view rule on both sides isolates engineering changes.
    cell("reference_uniform8", "coco_cd", args.reference_root, limit=56, view_batch=8, uniform=True)
    cell("unified8_equivalence", "coco_cd", args.candidate_root, limit=56, view_batch=8)
    for dataset, limit in protocol["evaluation_images"].items():
        cell("reference", dataset, args.reference_root, limit=limit)
        cell("unified8", dataset, args.candidate_root, limit=limit, view_batch=8)
        if selected != 8:
            cell(f"unified{selected}", dataset, args.candidate_root, limit=limit, view_batch=selected)
    if expected_inputs(args.candidate_root, args.data_root.resolve(), plan) != sealed_inputs:
        raise RuntimeError("Dataset inputs changed while the suite was running")
    write_json(args.output / "queue_state.json", dict(status="completed", cells=len(rows),
                                                      selected_view_batch=selected))


if __name__ == "__main__":
    main()
