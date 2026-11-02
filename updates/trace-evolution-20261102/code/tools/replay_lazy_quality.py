"""Replay query decisions on fixed measured supports; never predicts new mIoU."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

from run_algorithm_efficiency_suite import validate_algorithm_artifact
from run_matched_baseline_suite import sha, verify_suite, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot, output = args.pilot.resolve(), args.output.resolve()
    verify_suite(pilot)
    manifest = json.loads((pilot / "suite_protocol.json").read_text())
    # Only the selector is new; all supports and their provenance stay frozen.
    sys.path.insert(0, str(pilot / "runtime"))
    from trace_lazy_quality import LazyQualitySelector, LAZY_ALGORITHMS
    eps = manifest["objective_epsilon"]
    output.mkdir(parents=True, exist_ok=False)
    summaries, details, sources = [], [], {}
    for dataset in ("pc60", "voc21", "coco_cd", "coco_object"):
        directory = pilot / "evaluation" / dataset / "quality_only/attempt_001"
        validate_algorithm_artifact(directory, manifest["input_panels"][dataset + "/128"], manifest, "quality_only")
        path = directory / "candidate_diagnostics.csv"
        sources[str(path)] = sha(path)
        groups = defaultdict(list)
        for row in csv.DictReader(path.open()):
            groups[int(row["episode"])].append(row)
        for algorithm in LAZY_ALGORITHMS:
            selected_count = count = lost_groups = active_groups = 0
            maximum = max_group = 0.
            for episode, rows in groups.items():
                targets = [float(r["target_support"]) for r in rows]
                quality = [float(r["quality"]) for r in rows]
                keys = [(r["image_id"], r["class_idx"]) for r in rows]
                selector = LazyQualitySelector(targets, keys, algorithm)
                while selected := selector.next_indices(16):
                    for i in selected:
                        selector.observe(i, quality[i])
                measured = selector.global_selector.observed
                before = {k for k, q in zip(keys, quality) if q > eps}
                after = {keys[i] for i, q in measured.items() if q > eps}
                summary = selector.summary()
                selected_count += len(measured)
                count += len(rows)
                lost_groups += len(before - after)
                active_groups += len(before)
                maximum = max(maximum, summary["omitted_mass_bound"])
                max_group = max(max_group, summary["max_group_bound"])
                details.append(dict(dataset=dataset, algorithm=algorithm, episode=episode,
                    candidates=len(rows), active_groups_lost=len(before - after), **summary))
            summaries.append(dict(dataset=dataset, algorithm=algorithm, candidates=count,
                cf_evaluated=selected_count, modeled_diagnostic_inputs=count + selected_count,
                diagnostic_inputs_saved_fraction=(count - selected_count) / (2 * count),
                active_groups=active_groups, active_groups_lost=lost_groups,
                max_omitted_mass_bound=maximum, max_group_bound=max_group,
                new_model_forward=False, new_mIoU_available=False, measured_speedup_available=False))
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "episodes.csv", details)
    write_json(output / "provenance.json", dict(sources=sources, pilot_seal=sha(pilot / "SUITE_SEALED.json"),
        selector_sha256=sha(Path(__file__).with_name("trace_lazy_quality.py")),
        fixed_support_replay=True, timing_or_accuracy_claim=False,
        meaning="Query decisions use saved complete supports. New batching can change floating-point supports; "
                "these are conditional counts, not GPU timings or adaptation outcomes."))


if __name__ == "__main__":
    main()
