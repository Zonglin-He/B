"""Combine the frozen role-policy and fused-kernel measurements without selecting rows."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark_simplification import source_hashes, write_json
from suite_inputs import expected_inputs, validate_cell_inputs, validate_population


def read(path):
    return json.loads(path.read_text())


def checked(directory, expected):
    result, protocol = read(directory / "result.json"), read(directory / "protocol.json")
    assert result["status"] == "completed"
    assert result["protocol_sha256"] == hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
    validate_cell_inputs(protocol, expected)
    predictions = read(directory / "per_image_counts.json")
    assert result["images"] == len(predictions) == len(expected["image_ids"])
    assert [row["image_id"] for row in predictions] == expected["image_ids"]
    return result, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    panel, fused = args.root / "panel_v2", args.root / "fused_panel"
    assert read(panel / "queue_state.json")["status"] == read(fused / "queue_state.json")["status"] == "completed"
    plan = {dataset: (dataset, "evaluation", limit) for dataset, limit in
            (("coco_cd", None), ("pc60", 512), ("coco_stuff", 512), ("voc21", 512))}
    frozen = args.root / "frozen_candidate_sdpa_final"
    inputs = expected_inputs(frozen, args.data_root.resolve(), plan)
    rows = []
    for dataset, expected in inputs.items():
        native, np = checked(panel / "evaluation" / dataset / "reference", expected)
        manual, mp = checked(panel / "evaluation" / dataset / "unified16", expected)
        current_cd = args.root / "final_source_coco_cd_oomfix"
        if not (current_cd / "result.json").exists():
            current_cd = args.root / "final_source_coco_cd"
        final_dir = current_cd if dataset == "coco_cd" and (current_cd / "result.json").exists() else fused / "evaluation" / dataset
        final, fp = checked(final_dir, expected)
        validate_population(dataset, "evaluation", plan[dataset][2], fp["available_images"])
        assert mp["config"] == fp["config"]
        row = dict(dataset=dataset, images=final["images"], full_split=final["full_split"],
                   reference_mIoU=native["metrics"]["mIoU"], unified_manual_mIoU=manual["metrics"]["mIoU"],
                   final_mIoU=final["metrics"]["mIoU"], delta_vs_reference=final["miou_unrounded"]-native["miou_unrounded"],
                   reference_ms_per_image=native["ms_per_image"], final_ms_per_image=final["ms_per_image"],
                   runtime_change_vs_reference_pct=100*(final["ms_per_image"]/native["ms_per_image"]-1),
                   runtime_change_vs_unified_manual_pct=100*(final["ms_per_image"]/manual["ms_per_image"]-1),
                   reference_peak_mib=native["peak_allocated_mib"], final_peak_mib=final["peak_allocated_mib"],
                   memory_change_vs_reference_pct=100*(final["peak_allocated_mib"]/native["peak_allocated_mib"]-1),
                   final_result=str(final_dir / "result.json"))
        rows.append(row)
    with (args.root / "final_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.root / "final_evidence.json", dict(
        rows=rows, macro_delta_vs_reference=sum(row["delta_vs_reference"] for row in rows)/4,
        scope="seed 0; COCO-CD full 500, PC60/COCO-Stuff/VOC21 fixed 512-image validation panels",
        timing="GPU synchronized adaptation plus inference, loading/scoring/setup excluded; single RTX 5090 host",
        current_input_audit="retrospective ordered-ID/config/vocabulary and complete-population checks",
        runtime_resource_hashes={name: source_hashes(args.root / name / "trace_ovss")
                                 for name in ("baseline", "frozen_candidate_v2", "frozen_candidate_sdpa_final")}))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
