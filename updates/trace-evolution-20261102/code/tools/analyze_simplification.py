"""Verify matched cells and report every measured accuracy/cost comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark_simplification import write_json
from suite_inputs import expected_inputs, validate_cell_inputs, validate_population


def read_json(path):
    return json.loads(path.read_text())


def load_cell(panel, dataset, name):
    directory = panel / "evaluation" / dataset / name
    result = read_json(directory / "result.json")
    protocol = read_json(directory / "protocol.json")
    assert result["status"] == "completed"
    assert result["protocol_sha256"] == hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
    predictions = read_json(directory / "per_image_counts.json")
    assert [r["image_id"] for r in predictions] == protocol["image_ids"]
    assert result["images"] == len(predictions)
    expected = 56 if name in {"reference_uniform8", "unified8_equivalence"} else (None if dataset == "coco_cd" else 512)
    validate_population(dataset, "evaluation", expected, protocol["available_images"])
    requested = protocol["available_images"] if expected is None else expected
    if result["images"] != requested or result["full_split"] != (requested == protocol["available_images"]):
        raise RuntimeError("Cell does not cover the planned fixed cohort")
    return result, protocol, predictions


def verify_match(first, other):
    for key in ("image_ids", "label_ids", "vocabulary_sha256", "seed", "selection_seed", "split"):
        assert first[key] == other[key], f"Mismatched {key}"
    a, b = dict(first["config"]), dict(other["config"])
    a.pop("trace_view_batch_size")
    b.pop("trace_view_batch_size")
    assert a == b, "Only view-batch configuration may differ in this panel"


def analyze(panel):
    state = read_json(panel / "queue_state.json")
    if state["status"] != "completed":
        raise RuntimeError("Wait for the complete fixed panel before reporting its results")
    selection = read_json(panel / "batch_selection.json")
    selected = selection["selected_view_batch"]
    suite = read_json(panel / "suite_protocol.json")
    cd = read_json(panel / "evaluation/coco_cd/reference/protocol.json")
    data_root = Path(suite.get("data_root", str(Path(cd["config"]["data_dir"]).parent)))
    plan = {f"{dataset}/{limit}": (dataset, "evaluation", limit)
            for dataset, limit in (("coco_cd", None), ("coco_cd", 56),
                                  ("pc60", 512), ("coco_stuff", 512), ("voc21", 512))}
    inputs = expected_inputs(Path(suite["candidate_root"]), data_root, plan)
    write_json(panel / "input_identity_audit.json", dict(
        kind="Retrospective input audit; not a pre-run content seal for the legacy coordinator",
        data_root=str(data_root), current_input_panels=inputs))
    rows = []
    for dataset in ("coco_cd", "pc60", "coco_stuff", "voc21"):
        baseline, base_protocol, _ = load_cell(panel, dataset, "reference")
        limit = None if dataset == "coco_cd" else 512
        validate_cell_inputs(base_protocol, inputs[f"{dataset}/{limit}"])
        names = ["unified8"] + ([f"unified{selected}"] if selected != 8 else [])
        for name in names:
            result, protocol, _ = load_cell(panel, dataset, name)
            verify_match(base_protocol, protocol)
            validate_cell_inputs(protocol, inputs[f"{dataset}/{limit}"])
            rows.append(dict(dataset=dataset, images=result["images"], full_split=result["full_split"],
                variant=name, reference_mIoU=baseline["metrics"]["mIoU"], mIoU=result["metrics"]["mIoU"],
                delta_mIoU=result["miou_unrounded"]-baseline["miou_unrounded"],
                reference_ms_per_image=baseline["ms_per_image"], ms_per_image=result["ms_per_image"],
                runtime_change_pct=100*(result["ms_per_image"]/baseline["ms_per_image"]-1),
                peak_allocated_mib=result["peak_allocated_mib"],
                reference_views_per_image=baseline["diagnostic_views_per_image"],
                diagnostic_views_per_image=result["diagnostic_views_per_image"],
                result_path=str(panel / "evaluation" / dataset / name / "result.json")))
    with (panel / "matched_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    left, lp, lpred = load_cell(panel, "coco_cd", "reference_uniform8")
    right, rp, rpred = load_cell(panel, "coco_cd", "unified8_equivalence")
    verify_match(lp, rp)
    validate_cell_inputs(lp, inputs["coco_cd/56"])
    validate_cell_inputs(rp, inputs["coco_cd/56"])
    equal_predictions = sum(a["prediction_sha256"] == b["prediction_sha256"] for a, b in zip(lpred, rpred))
    numeric = ("target_support", "residual_support", "counterfactual_support", "quality", "dominance", "reliability")
    key = lambda row: tuple(row[k] for k in ("episode", "image_id", "patch_in_batch", "class_idx", "area", "mask_sha256"))
    def diagnostics(name):
        with (panel / "evaluation/coco_cd" / name / "candidate_diagnostics.csv").open() as handle:
            data = list(csv.DictReader(handle))
        index = {key(row): row for row in data}
        assert len(index) == len(data), "Duplicate candidate keys"
        return index
    a, b = diagnostics("reference_uniform8"), diagnostics("unified8_equivalence")
    assert a.keys() == b.keys(), "Candidates differ in the engineering control"
    maximum_errors = {field: max(abs(float(a[k][field])-float(b[k][field])) for k in a) for field in numeric}
    exact_scores = all(error == 0 for error in maximum_errors.values())
    write_json(panel / "engineering_equivalence.json", dict(
        images=len(lpred), identical_prediction_images=equal_predictions,
        candidates=len(a), max_absolute_support_errors=maximum_errors,
        exact_candidate_scores=exact_scores, exact_predictions=equal_predictions == len(lpred),
        scope="Same three-view formula, same view batch, same image IDs, pretrained GPU checkpoint; bounded control only"))
    assert exact_scores and equal_predictions == len(lpred)
    chosen = [row for row in rows if row["variant"] == f"unified{selected}"]
    write_json(panel / "decision_evidence.json", dict(
        selected_view_batch=selected, selection_uses_evaluation_scores=False,
        datasets=len(chosen), minimum_delta_mIoU=min(row["delta_mIoU"] for row in chosen),
        macro_delta_mIoU=sum(row["delta_mIoU"] for row in chosen)/len(chosen),
        scope="One full COCO-CD split and fixed 512-image subsets elsewhere, seed 0",
        rows=rows))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    analyze(parser.parse_args().panel)
