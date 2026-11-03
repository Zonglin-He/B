"""E02: one-pass, label-blind allocation audit of sealed native C diagnostics.

No new adaptation, inference or mIoU is performed. The crossed arms remain
scientifically pending even after this cache-admission analysis completes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from trace_ovss.method.score_factorial import ARMS, assign_score_factorial
from .bootstrap import TUNED, read_json, require
from .intake import sha256, write_new


def run(output):
    output = Path(output).absolute()
    require(not output.exists() and output == output.resolve(), "New E02 cache-audit directory required")
    roots = {d: TUNED / "full" / d for d in ("coco_object", "pc60")}
    source_files = [Path(__file__), Path(__file__).parents[2] / "trace_ovss/method/score_factorial.py"]
    source_files += [root / "PREDICTIONS_SEALED.json" for root in roots.values()]
    output.mkdir(parents=True)
    write_new(output / "PROTOCOL.json", dict(experiment="E02", stage="cached_assignment_audit_only",
        source_sha256={str(p): sha256(p) for p in source_files},
        arms=list(ARMS), tie_rule="lexical zero-padded crop:query:component candidate ID within source",
        admission_studies=["natural_per_arm_admission", "common_active_conditioned"],
        model_or_label_reads=False, new_predictions=False,
        next_required="Actual frozen four-arm adaptation/full prediction/scoring/paired E08 intervals"))
    datasets = {}
    for dataset, root in roots.items():
        seal = read_json(root / "PREDICTIONS_SEALED.json")
        require(seal["complete"] is True and seal["gt_contents_opened"] is False, "Unsealed C predictions")
        inventory = seal["artifact_sha256"]
        require(sha256(root / "prediction_protocol.json") == inventory["prediction_protocol.json"],
                "Protocol artifact drift")
        protocol = read_json(root / "prediction_protocol.json")
        manifest = protocol["manifest"]
        ordered = manifest["panel"]["ordered_ids"]
        batch = manifest["config"]["batch_size"]
        expected_episodes = (len(ordered) + batch-1) // batch
        totals = {name: {arm: Counter() for arm in ARMS} for name in ("natural", "common_active")}
        seen_ids, original_candidates, mismatched_membership_sources = [], 0, 0
        for episode in range(expected_episodes):
            name = f"episodes/{episode:06d}.image_C.json"
            path = root / name
            require(sha256(path) == inventory[name], "Cached diagnostic hash differs")
            receipt = read_json(path)
            ids = receipt["ordered_ids"]
            require(receipt["arm"] == "image_C" and receipt["episode"] == episode
                    and ids == ordered[episode*batch:(episode+1)*batch], "Cached source order differs")
            seen_ids.extend(ids)
            candidates = []
            for row in receipt["candidates"]:
                require(row["diagnostic_algorithm"] == "C"
                        and row["reliability"] == row["counterfactual_support"] == row["quality"],
                        "Not current native C diagnostics")
                source = row["image_idx"]
                require(type(source) is int and 0 <= source < len(ids), "Wrong source index")
                identity = f"{row['batch_idx']:04d}:{row['class_idx']:04d}:{row['component_idx']:06d}"
                candidates.append(dict(source_id=ids[source], candidate_id=identity,
                    area=row["area"], kappa=row["confidence"], C=row["reliability"]))
            original_candidates += len(candidates)
            natural = assign_score_factorial(candidates)
            common = assign_score_factorial(candidates, common_active=True)
            for study, result in (("natural", natural), ("common_active", common)):
                for arm, records in result["arms"].items():
                    active = len(records["active_source_ids"])
                    coefficients = records["ideal_region_coefficients"]
                    require(not active or abs(sum(coefficients)-1.) <= 1e-10, "Ideal total source budget differs")
                    totals[study][arm].update(dict(candidates=len(candidates),
                        retained_candidates=result["retained_candidates"],
                        active_candidates=len(records["active_candidate_ids"]), active_sources=active,
                        non_singleton_sources=records["non_singleton_sources"],
                        changed_assigned_scores=records["changed_assigned_scores"]))
            k_ids = set(natural["arms"]["Dk_Ok"]["active_candidate_ids"])
            c_ids = set(natural["arms"]["Dc_Oc"]["active_candidate_ids"])
            mismatched_membership_sources += len({source for source, _ in k_ids ^ c_ids})
        require(seen_ids == ordered, "Cached full population incomplete")
        datasets[dataset] = dict(images=len(seen_ids), episodes=expected_episodes,
            original_candidates=original_candidates, admission_studies=totals,
            sources_with_different_native_active_membership=mismatched_membership_sources)
    protocol = read_json(output / "PROTOCOL.json")
    for path, expected in protocol["source_sha256"].items():
        require(sha256(path) == expected, "Frozen audit source changed")
    write_new(output / "result.json", dict(experiment="E02", status="partial_cached_assignment_only",
        datasets=datasets, real_adaptations=0, new_miou_results=0,
        runtime_coefficients_measured=False, labels_read=False,
        remaining=protocol["next_required"]))
    write_new(output / "CACHE_AUDIT_SEALED.json", dict(complete=True, scope="cached assignment audit; no four-arm efficacy result",
        artifact_sha256={n: sha256(output / n) for n in ("PROTOCOL.json", "result.json")}))
    print(dict(output=str(output), datasets=list(datasets), inference=False, adaptations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
