"""Compare fused attention to the completed fixed manual-attention panel."""

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
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    if json.loads((args.reference_panel / "queue_state.json").read_text())["status"] != "completed":
        raise RuntimeError("Finish the existing GPU panel first")
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("benchmark_simplification.py")
    cases = [("pc60", "calibration", 40), ("coco_stuff", "calibration", 40),
             ("coco_cd", "evaluation", None), ("pc60", "evaluation", 512),
             ("coco_stuff", "evaluation", 512), ("voc21", "evaluation", 512)]
    plan = {f"{split}/{dataset}/{limit}": (dataset, split, limit) for dataset, split, limit in cases}
    inputs = expected_inputs(args.candidate_root, args.data_root.resolve(), plan)
    manifest = dict(data_root=str(args.data_root.resolve()), input_panels=inputs,
                    reference_panel=str(args.reference_panel.resolve()),
                    source_hashes=source_hashes(args.candidate_root / "trace_ovss"),
                    runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                    coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    view_batch_size=16, selection="Fixed before fused evaluation; no formula or optimization retuning",
                    calibration_gate=dict(minimum_delta_mIoU=-.25, macro_delta_mIoU=-.10))
    freeze_protocol(args.output / "suite_protocol.json", manifest)
    rows = []
    for dataset, split, limit in cases:
        key = f"{split}/{dataset}/{limit}"
        reference_dir = args.reference_panel / split / dataset / "unified16"
        reference = json.loads((reference_dir / "result.json").read_text())
        reference_protocol = json.loads((reference_dir / "protocol.json").read_text())
        validate_cell_inputs(reference_protocol, inputs[key])
        assert reference["protocol_sha256"] == hashlib.sha256((reference_dir / "protocol.json").read_bytes()).hexdigest()
        assert reference_protocol["config"]["trace_view_batch_size"] == 16
        output = args.output / split / dataset
        output.mkdir(parents=True, exist_ok=True)
        path = output / "result.json"
        if not path.exists():
            command = [sys.executable, str(runner), "--package-root", str(args.candidate_root),
                       "--data-root", str(args.data_root), "--dataset", dataset, "--split", split,
                       "--name", "fused16", "--view-batch-size", "16", "--output", str(output)]
            if limit:
                command += ["--max-images", str(limit)]
            write_json(args.output / "queue_state.json", dict(status="running", dataset=dataset, split=split, command=command))
            with (output / "stdout.log").open("a") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(args.output / "queue_state.json", dict(status="failed", dataset=dataset, split=split))
                raise RuntimeError(f"Failed cell: {output}")
        result = json.loads(path.read_text())
        saved = json.loads((output / "protocol.json").read_text())
        validate_cell_inputs(saved, inputs[key])
        assert result["status"] == reference["status"] == "completed"
        assert saved["source_hashes"] == manifest["source_hashes"]
        assert saved["config"]["trace_view_batch_size"] == 16
        assert result["protocol_sha256"] == hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()
        predictions = json.loads((output / "per_image_counts.json").read_text())
        assert [row["image_id"] for row in predictions] == inputs[key]["image_ids"]
        row = dict(dataset=dataset, split=split, images=result["images"], full_split=result["full_split"],
                   manual_mIoU=reference["metrics"]["mIoU"], fused_mIoU=result["metrics"]["mIoU"],
                   delta_mIoU=result["miou_unrounded"]-reference["miou_unrounded"],
                   manual_ms_per_image=reference["ms_per_image"], fused_ms_per_image=result["ms_per_image"],
                   runtime_change_pct=100*(result["ms_per_image"]/reference["ms_per_image"]-1),
                   manual_peak_mib=reference["peak_allocated_mib"], fused_peak_mib=result["peak_allocated_mib"],
                   reference_path=str(reference_dir / "result.json"), result_path=str(path))
        rows.append(row)
        with (args.output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        if len(rows) == 2:
            passed = min(r["delta_mIoU"] for r in rows) >= -.25 and sum(r["delta_mIoU"] for r in rows)/2 >= -.10
            write_json(args.output / "calibration_gate.json", dict(passed=passed,
                       criterion=manifest["calibration_gate"], rows=rows))
            if not passed:
                write_json(args.output / "queue_state.json", dict(status="rejected_calibration", cells=len(rows)))
                return
    if expected_inputs(args.candidate_root, args.data_root.resolve(), plan) != inputs:
        raise RuntimeError("Dataset changed during fused panel")
    write_json(args.output / "queue_state.json", dict(status="completed", cells=len(rows)))


if __name__ == "__main__":
    main()
