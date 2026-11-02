"""Offline allocation and label-transition audit of sealed two-view episodes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def allocation_shares(weights, epsilon=1e-8):
    effective = np.where(np.asarray(weights) > epsilon, weights, 0.0)
    totals = effective.sum(axis=1, keepdims=True)
    return np.divide(effective, totals, out=np.zeros_like(effective), where=totals > 0)


def analyze(cell, output):
    from tools.evolution.run_cell import sha256, verify_prediction_seal

    protocol = verify_prediction_seal(cell)
    score_seal = json.loads((cell / "SCORED_SEALED.json").read_text())
    if score_seal["prediction_seal_sha256"] != sha256(cell / "PREDICTIONS_SEALED.json"):
        raise RuntimeError("Scoring references another prediction seal")
    for name, expected in score_seal["artifact_sha256"].items():
        if sha256(cell / name) != expected:
            raise RuntimeError("Scored artifact changed")
    if output.exists():
        raise FileExistsError("Use a fresh analysis directory")
    output.mkdir(parents=True)
    labels = [entry[0] for entry in protocol["variants"]]
    aggregates = {}
    row_count = 0
    with (output / "candidate_transitions.csv").open("w", newline="") as handle:
        fields = ["episode", "candidate", "image_index", "class_query", "policy", "confidence",
                  "T", "C", "q", "rho", "episode_allocation_share", "group", "valid_pixels",
                  "pre_correct", "pre_wrong", "correct_to_wrong", "wrong_to_correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for meta_path in sorted((cell / "episodes").glob("*.json")):
            if ".reference." in meta_path.name:
                continue
            meta = json.loads(meta_path.read_text())
            with np.load(meta_path.with_suffix(".npz"), allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            if meta["policy_labels"] != labels:
                raise RuntimeError("Episode policy order changed")
            # This first study fixes 224x224 source images, one crop per image.
            # Do not silently duplicate spatial locations on overlapping crops.
            if any(tuple(grid) != (1, 1) for grid in meta["meta"]["patch_grid_shape"]):
                raise RuntimeError("Overlapping-crop audit needs an explicit coordinate map")
            if len(data["masks"]) != meta["candidate_count"]:
                raise RuntimeError("Candidate count differs")
            shares = allocation_shares(data["rho"])
            for image_index in meta["image_indices"]:
                pre = np.load(cell / "predictions/pre" / f"{image_index:06d}.npy", allow_pickle=False)
                source = protocol["inputs"][image_index]
                with Image.open(source["label"]) as image:
                    gt = np.asarray(image.resize((pre.shape[1], pre.shape[0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if protocol["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                for policy_index, label in enumerate(labels):
                    post = np.load(cell / "predictions" / label / f"{image_index:06d}.npy", allow_pickle=False)
                    for candidate in np.flatnonzero(data["image_index"] == image_index):
                        mask = np.asarray(Image.fromarray(data["masks"][candidate].astype(np.uint8))
                            .resize((pre.shape[1], pre.shape[0]), Image.Resampling.NEAREST), dtype=bool)
                        valid = mask & (gt != 255)
                        correct = valid & (pre == gt)
                        wrong = valid & (pre != gt)
                        cw = int(np.count_nonzero(correct & (post != gt)))
                        wc = int(np.count_nonzero(wrong & (post == gt)))
                        confidence, q = float(data["confidence"][candidate]), float(data["q"][candidate])
                        group = "high_conf_low_quality" if confidence >= .9 and q < .25 else "other"
                        share = float(shares[policy_index, candidate])
                        row = dict(episode=meta["episode"], candidate=int(candidate), image_index=image_index,
                            class_query=int(data["query_index"][candidate]), policy=label,
                            confidence=confidence, T=float(data["T"][candidate]), C=float(data["C"][candidate]),
                            q=q, rho=float(data["rho"][policy_index, candidate]), episode_allocation_share=share,
                            group=group, valid_pixels=int(valid.sum()), pre_correct=int(correct.sum()),
                            pre_wrong=int(wrong.sum()), correct_to_wrong=cw, wrong_to_correct=wc)
                        writer.writerow(row)
                        row_count += 1
                        key = (label, group)
                        total = aggregates.setdefault(key, dict(candidates=0, pre_correct=0, pre_wrong=0,
                            correct_to_wrong=0, wrong_to_correct=0, allocation_share_sum=0.0))
                        total["candidates"] += 1
                        for name in ("pre_correct", "pre_wrong", "correct_to_wrong", "wrong_to_correct"):
                            total[name] += row[name]
                        total["allocation_share_sum"] += share
    summary = []
    for (policy, group), total in aggregates.items():
        summary.append(dict(policy=policy, group=group, **total,
            correct_to_wrong_pct=(100 * total["correct_to_wrong"] / total["pre_correct"] if total["pre_correct"] else None),
            wrong_to_correct_pct=(100 * total["wrong_to_correct"] / total["pre_wrong"] if total["pre_wrong"] else None)))
    report = dict(cell=str(cell), dataset=protocol["dataset"], images=len(protocol["inputs"]),
        full_split=protocol["full_split"], candidate_policy_rows=row_count, results=summary,
        groups_fixed_before_outcomes=dict(confidence_min=.9, quality_max_exclusive=.25),
        allocation_admission="Weights <=1e-8 receive zero loss share, matching the evaluated objective.",
        interpretation="Spatial outcomes after joint episode adaptation, not isolated candidate causal effects.",
        dependence="Candidates/pixels within an episode are not independent replicates.",
        scoring_seal_sha256=sha256(cell / "SCORED_SEALED.json"))
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "ANALYSIS_SEALED.json").write_text(json.dumps(dict(
        artifact_sha256={name: sha256(output / name) for name in ("summary.json", "candidate_transitions.csv")}), indent=2) + "\n")
    return report


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
    args = parser.parse_args()
    report = analyze(args.cell, args.output)
    print(json.dumps({k: report[k] for k in ("dataset", "images", "candidate_policy_rows")}), flush=True)


if __name__ == "__main__":
    main()
