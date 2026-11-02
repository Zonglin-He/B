"""Freeze a self-contained manuscript packet without internal author reviews."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    paper = args.output / "paper"
    paper.mkdir()
    files = ["main.tex", "supplement.tex", "appendix.tex", "preamble.tex",
             "references.bib", "aaai2027.sty", "aaai2027.bst", "reported_baselines.json"]
    for name in files:
        shutil.copy2(source / name, paper / name)
    historical = shutil.ignore_patterns("native_comparison.tex", "cost_snapshot.tex",
                                        "provenance.json", "native_operating_points.pdf")
    for name in ("generated", "figure"):
        shutil.copytree(source / name, paper / name, ignore=historical)
    (paper / "build").mkdir()
    for name in ("main.pdf", "supplement.pdf", "main.aux", "main.log", "supplement.log"):
        shutil.copy2(source / "build" / name, paper / "build" / name)
    tools = paper / "tools"
    tools.mkdir()
    for name in ("build.py", "export_evolution.py", "export_update_risk.py",
                 "export_probe_audit.py", "export_standalone.py"):
        shutil.copy2(source / "tools" / name, tools / name)
    hashes = {}
    for path in sorted(paper.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    manuscript_root=str(source), files=hashes,
                    evidence_roots=[
                        "/home/wwww/hzl/results/trace_self_evolution_20261028_1338",
                        "/home/wwww/hzl/results/trace_evolution_product_pixel_recovery_20261028_2215",
                        "/home/wwww/hzl/results/trace_evolution_voc20_cli_recovery_20261028_2312",
                        "/home/wwww/hzl/results/trace_evolution_final_analysis_20261028_2342",
                        "/home/wwww/hzl/results/trace_probe_complementarity_20261029_0030",
                        "/home/wwww/hzl/results/trace_parameter_provenance_20261029_0030",
                        "/home/wwww/hzl/results/trace_standalone_followup_20261029_0105"],
                    scope="Frozen paper and small provenance artifacts; no author review or target score.")
    (args.output / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(hashes)} manuscript files at {args.output}")


if __name__ == "__main__":
    main()
