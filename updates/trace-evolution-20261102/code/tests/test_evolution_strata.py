"""Hand-check matched-stratum arithmetic and preserve unfavorable signs."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_evolution_strata import analyze, digest


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def source(tmp_path, *, reverse=False, ties=False):
    cell, analysis = tmp_path / "cell", tmp_path / "analysis"
    (cell / "episodes").mkdir(parents=True)
    analysis.mkdir()
    np.savez_compressed(cell / "episodes/000000.npz", masks=np.ones((8, 1, 1), dtype=bool))
    write(cell / "PREDICTIONS_SEALED.json", dict(artifact_sha256={
        "episodes/000000.npz": digest(cell / "episodes/000000.npz")}))
    write(cell / "SCORED_SEALED.json", dict(fixture=True))
    write(cell / "prediction_protocol.json", dict(query_to_class=None))
    with (analysis / "candidate_transitions.csv").open("w", newline="") as handle:
        fields = ["episode", "candidate", "confidence", "q", "class_query", "policy",
                  "pre_correct", "correct_to_wrong", "pre_wrong", "wrong_to_correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in range(8):
            cw = (2 if candidate < 2 else 10) if reverse else (10 if candidate < 2 else 2)
            writer.writerow(dict(episode=0, candidate=candidate, confidence=.95,
                q=.5 if ties else (candidate + 1)/10, class_query=0, policy="confidence",
                pre_correct=100, correct_to_wrong=cw, pre_wrong=100, wrong_to_correct=20))
    write(analysis / "summary.json", dict(cell=str(cell), dataset="synthetic", images=8,
        full_split=False, scoring_seal_sha256=digest(cell / "SCORED_SEALED.json")))
    write(analysis / "ANALYSIS_SEALED.json", dict(artifact_sha256={
        name: digest(analysis / name) for name in ("summary.json", "candidate_transitions.csv")}))
    return analysis


@pytest.mark.parametrize("reverse,expected", [(False, -8), (True, 8)])
def test_stratified_rates_match_hand_counts(tmp_path, reverse, expected):
    upstream = source(tmp_path, reverse=reverse)
    output = tmp_path / "strata"
    analyze(upstream, output)
    report = json.loads((output / "summary.json").read_text())
    assert report["candidates"] == 8
    assert report["eligible_strata"] == 1 and report["selected_candidates"] == 4
    assert report["results"][0]["cw"]["high_minus_low_pp"] == expected
    assert report["results"][0]["wc"]["high_minus_low_pp"] == 0


def test_tied_quality_is_not_invented_risk_stratification(tmp_path):
    upstream = source(tmp_path, ties=True)
    output = tmp_path / "strata"
    analyze(upstream, output)
    report = json.loads((output / "summary.json").read_text())
    assert report["eligible_strata"] == 0 and report["results"] == []


def test_changed_upstream_counts_fail_the_seal(tmp_path):
    upstream = source(tmp_path)
    with (upstream / "candidate_transitions.csv").open("a") as handle:
        handle.write("unsealed\n")
    with pytest.raises(RuntimeError, match="seal changed"):
        analyze(upstream, tmp_path / "strata")
