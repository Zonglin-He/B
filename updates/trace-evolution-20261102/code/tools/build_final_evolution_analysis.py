"""Run the same sealed, descriptive transition audit on all five control splits."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_evolution import analyze
from tools.analyze_evolution_strata import analyze as stratify
from tools.evolution_completion import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite an existing analysis packet")
    args.output.mkdir(parents=True)
    reports = {}
    for dataset in ("coco_cd", "pc60", "voc21", "coco_object", "coco_stuff"):
        destination = args.output / dataset
        reports[dataset] = analyze(args.suite / "cells/mechanism" / dataset, destination)
        stratify(destination, args.output / ("strata_" + dataset))
        print(json.dumps(dict(dataset=dataset, status="completed")), flush=True)
    summary = dict(suite=str(args.suite.resolve()), datasets=reports,
        analysis_source_sha256={name: digest(Path(__file__).parent / name) for name in (
            "analyze_evolution.py", "analyze_evolution_strata.py", "build_final_evolution_analysis.py")},
        interpretation="Descriptive outcomes after joint adaptation, not independent candidate interventions.",
        runtime=dict(gpu_work=False, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    seal_paths = [path]
    seal_paths += [args.output / d / "ANALYSIS_SEALED.json" for d in reports]
    seal_paths += [args.output / ("strata_" + d) / "STRATIFIED_ANALYSIS_SEALED.json" for d in reports]
    (args.output / "FINAL_ANALYSIS_SEALED.json").write_text(json.dumps(dict(
        artifact_sha256={str(p.relative_to(args.output)): digest(p) for p in seal_paths}), indent=2) + "\n")


if __name__ == "__main__":
    main()
