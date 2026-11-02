"""Report score and prediction equality separately for the final mask change."""

import argparse
import csv
import json
from pathlib import Path

from benchmark_simplification import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    def load(name):
        root = args.root / name
        result = json.loads((root / "result.json").read_text())
        protocol = json.loads((root / "protocol.json").read_text())
        predictions = json.loads((root / "per_image_counts.json").read_text())
        with (root / "candidate_diagnostics.csv").open() as handle:
            records = list(csv.DictReader(handle))
        key = lambda row: tuple(row[field] for field in ("episode", "image_id", "patch_in_batch", "class_idx", "area", "mask_sha256"))
        indexed = {key(row): row for row in records}
        assert len(indexed) == len(records)
        return result, protocol, predictions, indexed

    a, ap, predictions_a, candidates_a = load("reference")
    b, bp, predictions_b, candidates_b = load("current")
    for key in ("image_ids", "label_ids", "vocabulary_sha256", "config", "seed", "selection_seed"):
        assert ap[key] == bp[key], key
    assert candidates_a.keys() == candidates_b.keys()
    fields = ("target_support", "residual_support", "counterfactual_support", "quality", "dominance", "reliability")
    errors = {field: max(abs(float(candidates_a[key][field])-float(candidates_b[key][field]))
                         for key in candidates_a) for field in fields}
    result = dict(images=a["images"], candidates=len(candidates_a), maximum_score_errors=errors,
                  identical_prediction_images=sum(x["prediction_sha256"] == y["prediction_sha256"]
                                                    for x, y in zip(predictions_a, predictions_b)),
                  reference_mIoU=a["miou_unrounded"], current_mIoU=b["miou_unrounded"],
                  delta_mIoU=b["miou_unrounded"]-a["miou_unrounded"],
                  bitwise_update_trajectory_claim=False,
                  scope="Same images/formula/kernel/view batch; input masks generated per view batch in current source")
    write_json(args.root / "comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
