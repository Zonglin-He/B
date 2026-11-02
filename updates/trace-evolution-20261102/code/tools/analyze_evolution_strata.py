"""Descriptive class/confidence/area-adjusted risk analysis, not causal attribution."""

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CONFIDENCE_EDGES = (0, .7, .9, .95, .99, 1.000001)
AREA_EDGES = (1, 4, 16, 64, 257)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def analyze(analysis, output):
    if output.exists():
        raise FileExistsError("Choose a fresh analysis directory")
    seal = json.loads((analysis / "ANALYSIS_SEALED.json").read_text())
    for name, expected in seal["artifact_sha256"].items():
        if digest(analysis / name) != expected:
            raise RuntimeError("Initial transition analysis seal changed")
    summary = json.loads((analysis / "summary.json").read_text())
    cell = Path(summary["cell"])
    if digest(cell / "SCORED_SEALED.json") != summary["scoring_seal_sha256"]:
        raise RuntimeError("Transition analysis references another score seal")
    prediction = json.loads((cell / "PREDICTIONS_SEALED.json").read_text())
    protocol = json.loads((cell / "prediction_protocol.json").read_text())
    mapping = protocol["query_to_class"]
    output.mkdir(parents=True)
    specification = dict(confidence_edges=CONFIDENCE_EDGES, area_edges=AREA_EDGES,
        min_candidates_per_stratum=8, groups="Lowest/highest rank quartile of initial q within each stratum; skip overlapping q tails.",
        covariates="Predicted semantic class, initial confidence bin, original candidate token area bin.",
        uncertainty="Descriptive finite-panel association; no candidate-independent significance test.",
        outcomes="Pixel C-to-W and W-to-C after joint episode adaptation.",
        exposure="Fixed bins before this stratified computation, not a prospectively untouched test: earlier overall diagnostics were visible.")
    (output / "analysis_protocol.json").write_text(json.dumps(specification, indent=2) + "\n")
    area = {}
    for path in sorted((cell / "episodes").glob("*.npz")):
        relative = str(path.relative_to(cell))
        if prediction["artifact_sha256"].get(relative) != digest(path):
            raise RuntimeError("Candidate mask cache changed")
        with np.load(path, allow_pickle=False) as data:
            for candidate, mask in enumerate(data["masks"]):
                area[int(path.stem), candidate] = int(mask.sum())
    candidates, strata = {}, {}
    source = analysis / "candidate_transitions.csv"
    with source.open() as handle:
        for row in csv.DictReader(handle):
            key = int(row["episode"]), int(row["candidate"])
            confidence, q = float(row["confidence"]), float(row["q"])
            query = int(row["class_query"])
            cls = mapping[query] if mapping is not None else query
            covariate = (confidence, q, cls, area[key])
            if key in candidates:
                if candidates[key] != covariate:
                    raise RuntimeError("Policy comparisons changed initial covariates")
                continue
            candidates[key] = covariate
            stratum = (cls, bisect_right(CONFIDENCE_EDGES, confidence) - 1,
                       bisect_right(AREA_EDGES, area[key]) - 1)
            strata.setdefault(stratum, []).append(key)
    membership, eligible = {}, {}
    for stratum, keys in strata.items():
        if len(keys) < 8:
            continue
        order = sorted(keys, key=lambda key: (candidates[key][1], key))
        n = len(keys) // 4
        low, high = order[:n], order[-n:]
        if candidates[low[-1]][1] >= candidates[high[0]][1]:
            continue
        eligible[stratum] = n
        for group, members in (("low", low), ("high", high)):
            for key in members:
                membership[key] = stratum, group
    counts = {}
    with source.open() as handle:
        for row in csv.DictReader(handle):
            key = int(row["episode"]), int(row["candidate"])
            if key not in membership:
                continue
            stratum, group = membership[key]
            total = counts.setdefault((row["policy"], stratum, group), [0, 0, 0, 0])
            for index, name in enumerate(("pre_correct", "correct_to_wrong", "pre_wrong", "wrong_to_correct")):
                total[index] += int(row[name])
    results = []
    policy_names = sorted({key[0] for key in counts})
    for policy in policy_names:
        rates = {"cw": [], "wc": []}
        for stratum, weight in eligible.items():
            low, high = (counts[policy, stratum, group] for group in ("low", "high"))
            for name, den, num in (("cw", 0, 1), ("wc", 2, 3)):
                if low[den] and high[den]:
                    rates[name].append((weight, 100 * low[num] / low[den], 100 * high[num] / high[den]))
        record = dict(policy=policy)
        for name, values in rates.items():
            total_weight = math.fsum(v[0] for v in values)
            low = math.fsum(w * a for w, a, b in values) / total_weight if total_weight else None
            high = math.fsum(w * b for w, a, b in values) / total_weight if total_weight else None
            record[name] = dict(low_quality_pct=low, high_quality_pct=high,
                high_minus_low_pp=high-low if high is not None else None, strata=len(values))
        results.append(record)
    report = dict(dataset=summary["dataset"], images=summary["images"], full_split=summary["full_split"],
        candidates=len(candidates), eligible_strata=len(eligible), selected_candidates=len(membership),
        results=results, protocol=specification, source_analysis_sha256=digest(analysis / "ANALYSIS_SEALED.json"))
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "STRATIFIED_ANALYSIS_SEALED.json").write_text(json.dumps(dict(
        artifact_sha256={name: digest(output / name) for name in ("analysis_protocol.json", "summary.json")}), indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("dataset", "candidates", "eligible_strata", "selected_candidates")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
    args = parser.parse_args()
    analyze(args.analysis, args.output)


if __name__ == "__main__":
    main()
