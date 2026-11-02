"""Freeze the current single-probe paper without historical paper assets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    suite = Path("/home/wwww/hzl/results/trace_image_full_20261029_0333")
    prepared_sha = "e0f7374042eadb65fbc8146de6dfcf35a71151dc1c95c115b0d4cf9c8444edaa"
    if digest(suite / "SUITE_PREPARED.json") != prepared_sha:
        raise RuntimeError("Prepared suite identity differs")
    for name in ("main.log", "supplement.log"):
        log = (source / "build" / name).read_text(errors="replace")
        if any(term in log for term in ("Overfull", "Float too large", "undefined references",
                                        "undefined citations", "multiply defined")):
            raise RuntimeError(f"Unresolved publication diagnostic: {name}")
    paths = [Path(name) for name in (
        "main.tex", "supplement.tex", "appendix.tex", "preamble.tex",
        "references.bib", "aaai2027.sty", "aaai2027.bst", "reported_baselines.json",
        "build/main.pdf", "build/supplement.pdf", "build/main.aux", "build/main.bbl",
        "build/supplement.bbl", "build/main.log", "build/supplement.log",
        "generated/configurations.tex", "generated/image_budget_comparison.tex",
        "generated/image_budget_cost.tex", "generated/image_budget_provenance.json",
        "generated/image_full_comparison.tex", "generated/image_full_reference.tex",
        "generated/image_full_cost.tex", "generated/image_full_provenance.json",
        "tools/build.py", "tools/export_image_full.py", "tools/export_image_budget.py",
        "tools/export_evolution.py", "tools/freeze_image_full_review.py",
    )]
    paths += [path.relative_to(source) for path in sorted((source / "figure").glob("image_full_*"))
              if path.is_file() and "integration_check" not in path.name]
    before = {str(path): digest(source / path) for path in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    paper = args.output / "paper"
    for path in paths:
        destination = paper / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / path, destination)
        if digest(destination) != before[str(path)] or digest(source / path) != before[str(path)]:
            raise RuntimeError(f"Source changed while freezing: {path}")
    hashes = {str(Path("paper") / path): value for path, value in before.items()}
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(), files=hashes,
        method="ImageBalancedTRACE global C; one context probe; image-normalized objective",
        suite_prepared_sha256=prepared_sha,
        evidence_roots=[str(suite), "/home/wwww/hzl/results/trace_image_budget_20261029_0233",
                        "/home/wwww/hzl/results/trace_parameter_provenance_20261029_0030",
                        "/home/wwww/hzl/results/trace_two_view_full_baselines_20261025/coco_cd"],
        scope="Frozen manuscript and publication provenance; no author reviews or target score. "
              "Original evidence roots remain read-only. No new inference or scoring.",
    )
    (args.output / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(hashes)} current single-probe files at {args.output}")


if __name__ == "__main__":
    main()
