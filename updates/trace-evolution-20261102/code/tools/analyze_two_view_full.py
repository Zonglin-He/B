"""CPU-only class and allocation diagnosis after matched full evaluation."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from run_matched_baseline_suite import sha, verify_suite, write_csv, write_json
from benchmark_matched_baselines import tree_hashes


def require_owned_artifact(path, owner):
    try:
        relative = path.resolve().relative_to(owner.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Comparison artifact is outside its owning suite") from error
    seal = json.loads((owner / "SUITE_SEALED.json").read_text())["artifact_sha256"]
    if seal.get(relative) != sha(path):
        raise ValueError("Comparison artifact does not match its owning seal")


def validate_vocabulary(vocabulary, panel):
    classes = vocabulary.extensions if panel["config"]["class_extensions"] else vocabulary.classes
    mapping = vocabulary.extension_to_class if panel["config"]["class_extensions"] else None
    digest = hashlib.sha256(json.dumps([classes, mapping]).encode()).hexdigest()
    if digest != panel["vocabulary_sha256"]:
        raise ValueError("Analysis vocabulary differs from frozen evaluation vocabulary")
    return mapping


def allocation_by_class(rows, mapping, classes, epsilon):
    episodes = defaultdict(list)
    for row in rows:
        episodes[int(row["episode"])].append(row)
    output = {i: dict(candidates=0, target_sum=0., cf_sum=0., cf_limiting=0,
                     quality_coefficient_sum=0., target_coefficient_sum=0.) for i in range(classes)}
    for group in episodes.values():
        values = [(r, float(r["target_support"]), float(r["counterfactual_support"]),
                   float(r["reliability"])) for r in group]
        if any(not all(math.isfinite(x) and 0 <= x <= 1 for x in (t, c, q))
               or abs(q - min(t, c)) > 1e-12 for _, t, c, q in values):
            raise ValueError("Expected finite, exact two-view quality weights")
        qsum = math.fsum(q for _, _, _, q in values if q > epsilon)
        tsum = math.fsum(t for _, t, _, _ in values if t > epsilon)
        for row, target, cf, quality in values:
            index = int(row["class_idx"])
            cls = mapping[index] if mapping is not None else index
            value = output[cls]
            value["candidates"] += 1
            value["target_sum"] += target
            value["cf_sum"] += cf
            value["cf_limiting"] += cf < target
            value["quality_coefficient_sum"] += quality / qsum if quality > epsilon else 0.
            value["target_coefficient_sum"] += target / tsum if target > epsilon else 0.
    n = len(episodes)
    for value in output.values():
        count = value["candidates"]
        value.update(mean_target=value.pop("target_sum") / count if count else None,
                     mean_cf=value.pop("cf_sum") / count if count else None,
                     cf_limiting_fraction=value.pop("cf_limiting") / count if count else None,
                     mean_episode_quality_share=value.pop("quality_coefficient_sum") / n if n else None,
                     mean_episode_target_share=value.pop("target_coefficient_sum") / n if n else None)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace, baselines, output = (p.resolve() for p in (args.trace_root, args.baseline_root, args.output))
    verify_suite(trace)
    verify_suite(baselines)
    output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((trace / "suite_protocol.json").read_text())
    if tree_hashes(trace / "runtime/trace_ovss") != parent["source_hashes"]:
        raise ValueError("Runtime source/resources changed before class analysis")
    sys.path.insert(0, str(trace / "runtime"))
    import trace_ovss
    if not Path(trace_ovss.__file__).resolve().is_relative_to(trace / "runtime"):
        raise ValueError("Analysis loaded another TRACE package")
    from trace_ovss.data.vocabulary import load_vocabulary
    comparisons = list(csv.DictReader((baselines / "comparison.csv").open()))
    rows, sources = [], {}
    for comparison in comparisons:
        if comparison["algorithm"] != "quality_only" or comparison["comparison_complete"] != "True":
            continue
        dataset = comparison["dataset"]
        path, baseline_path = Path(comparison["result_path"]), Path(comparison["baseline_result_path"])
        require_owned_artifact(path, trace)
        require_owned_artifact(baseline_path, baselines)
        result, baseline = json.loads(path.read_text()), json.loads(baseline_path.read_text())
        panel = parent["input_panels"][dataset + "/full"]
        if result["images"] != panel["available_images"] or baseline["images"] != result["images"]:
            raise ValueError("Class comparison requires matching full populations")
        vocabulary = load_vocabulary(panel["config"]["dataset"])
        mapping = validate_vocabulary(vocabulary, panel)
        diagnostics = path.parent / "candidate_diagnostics.csv"
        require_owned_artifact(diagnostics, trace)
        allocation = allocation_by_class(list(csv.DictReader(diagnostics.open())), mapping,
                                         len(vocabulary.classes), parent["objective_epsilon"])
        for i, name in enumerate(vocabulary.classes):
            ti, tu = result["intersection"][i], result["union"][i]
            bi, bu = baseline["intersection"][i], baseline["union"][i]
            t, b = (100 * ti / tu if tu else None), (100 * bi / bu if bu else None)
            rows.append(dict(dataset=dataset, class_index=i, class_name=name, TRACE_IoU=t,
                best_baseline=comparison["best_available_baseline"], baseline_IoU=b,
                delta_IoU=t - b if t is not None and b is not None else None,
                trace_union=tu, baseline_union=bu, **allocation[i]))
        for artifact in (path, baseline_path, diagnostics):
            sources[str(artifact)] = sha(artifact)
    write_csv(output / "class_diagnosis.csv", rows)
    write_json(output / "provenance.json", dict(sources=sources,
        vocabulary_sha256={c["dataset"]: parent["input_panels"][c["dataset"] + "/full"]["vocabulary_sha256"] for c in comparisons},
        trace_seal_sha256=sha(trace / "SUITE_SEALED.json"), baseline_seal_sha256=sha(baselines / "SUITE_SEALED.json"),
        meaning="IoU is pooled over the full split. Allocation columns average normalized "
                "coefficients over episodes containing candidates, not gradient attribution. "
                "Target-only shares are offline counterfactual coefficients, not adaptation results.",
        parameter_selection=False, calibration_required_before_tuning=True))


if __name__ == "__main__":
    main()
