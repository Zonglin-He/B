"""Read a completed fixed-control suite; never choose a winner or run models."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.evolution_allocation.common import (
    ARMS, DATASETS, POPULATIONS, ROOT, read_json, require, sha256, verify_prepared,
)
from tools.evolution_allocation.queue import bound_service_terminal, validate_results


def contrasts(scores):
    require(set(scores) == set(ARMS), "All five signed results are required")
    ir, er, it, et, shuffled = (scores[a] for a in ARMS)
    return dict(image_minus_episode_region=ir - er, image_minus_episode_token=it - et,
        region_minus_token_image=ir - it, region_minus_token_episode=er - et,
        normalization_by_pooling_interaction=(ir - er) - (it - et),
        native_minus_shuffled=ir - shuffled)


def coverage_ratios(counts):
    keys = ("active_candidates", "active_sources", "eligible_candidates", "eligible_sources",
        "moved_candidates", "moved_sources", "changed_score_candidates", "changed_score_sources")
    require(set(counts) == set(keys) and all(type(counts[k]) is int and counts[k] >= 0 for k in keys),
            "Missing/malformed actual shuffled-allocation coverage")
    for suffix in ("candidates", "sources"):
        active = counts[f"active_{suffix}"]
        eligible = counts[f"eligible_{suffix}"]
        moved = counts[f"moved_{suffix}"]
        changed = counts[f"changed_score_{suffix}"]
        require(changed <= moved <= eligible <= active, "Impossible permutation coverage")
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None
    return dict(counts=counts, eligible_source_fraction=ratio("eligible_sources", "active_sources"),
        eligible_candidate_fraction=ratio("eligible_candidates", "active_candidates"),
        moved_fraction_of_eligible=ratio("moved_candidates", "eligible_candidates"),
        changed_score_fraction_of_eligible=ratio("changed_score_candidates", "eligible_candidates"))


def analyze(suite, expected, service):
    prepared = verify_prepared(suite, expected)
    execution = read_json(suite / "EXECUTION_STARTED.json")
    terminal = bound_service_terminal(suite, prepared, execution, service)
    seal = read_json(suite / "SUITE_SEALED.json")
    require(seal["complete"] is True and seal["suite_prepared_sha256"] == expected
            and seal["full_outcomes"] == 10 and seal["datasets"] == list(DATASETS)
            and seal["arms"] == ARMS and seal["full_metrics_sha256"] == sha256(suite / "full_metrics.csv"),
            "Incomplete final suite")
    require(seal["execution_started_sha256"] == sha256(suite / "EXECUTION_STARTED.json"), "Execution receipt changed after final seal")
    state = read_json(suite / "queue_state.json")
    require(state["phase"] == "completed" and state["pid"] == execution["pid"], "Nonterminal/unrelated controller")
    rows, score_seals = validate_results(suite, prepared)
    require(score_seals == seal["score_seals"], "Final canonical score bindings differ")
    datasets = {}
    for dataset in DATASETS:
        output = suite / "full" / dataset
        result = read_json(output / "result.json")
        predicted = read_json(output / "prediction_result.json")
        require(result["shuffle_coverage"] == predicted["shuffle_coverage"], "Scored/predicted coverage differs")
        scores = {a: result["arm_results"][a]["miou_unrounded"] for a in ARMS}
        current = read_json(output / "PREDICTIONS_SEALED.json")["artifact_sha256"]
        original = read_json(suite / f"parents/{dataset}/PREDICTIONS_SEALED.json")["artifact_sha256"]
        equal = sum(current[f"predictions/image_region/{i:06d}.npy"] == original[f"predictions/image_C/{i:06d}.npy"]
                    for i in range(POPULATIONS[dataset]))
        original_score = read_json(suite / f"parents/{dataset}/result.json")["arm_results"]["image_C"]["miou_unrounded"]
        datasets[dataset] = dict(images=POPULATIONS[dataset], pooled_mIoU=scores, contrasts_pp=contrasts(scores),
            shuffle_coverage=coverage_ratios(result["shuffle_coverage"]),
            native_replay=dict(equal_prediction_file_hashes=equal, total_images=POPULATIONS[dataset],
                all_prediction_hashes_equal=equal == POPULATIONS[dataset],
                parent_mIoU=original_score, current_minus_parent_pp=scores["image_region"] - original_score,
                scope="Comparison of sealed hash inventories; no original payload decoded or rehashed"))
    require(bound_service_terminal(suite, prepared, execution, service) == terminal, "Service terminal changed during interpretation")
    return dict(suite=str(suite), suite_prepared_sha256=expected, service_terminal=terminal,
        full_outcomes=10, rows=rows, datasets=datasets,
        inference="Development-exposed fixed-setting mechanism contrasts; no independent uncertainty, winner selection or method replacement")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args(argv)
    require(ROOT == args.suite / "runtime", "Analyze with the completed frozen runtime")
    import json
    print(json.dumps(analyze(args.suite, args.suite_sha256, args.service), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
