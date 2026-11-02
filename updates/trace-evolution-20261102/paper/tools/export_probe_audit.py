"""Render a sealed probe-complementarity audit, not a performance claim."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads((args.audit / "AUDIT_SEALED.json").read_text())
    for name, expected in seal["artifact_sha256"].items():
        path = args.audit / name
        if not path.resolve().is_relative_to(args.audit.resolve()) or digest(path) != expected:
            raise RuntimeError("Probe audit artifact does not match its seal")
    data = json.loads((args.audit / "summary.json").read_text())
    if data["status"] != "completed" or not data["execution"]["cpu_only"]:
        raise RuntimeError("Expected the completed CPU-only audit")
    names = {"voc21": "VOC21", "pc60": "PC60", "coco_object": "COCO Object",
             "coco_cd": "COCO-CD", "coco_stuff": "COCO-Stuff"}
    if set(data["datasets"]) != set(names):
        raise RuntimeError("Report all five audit datasets, not selected signs")
    rows = []
    for dataset, name in names.items():
        row = data["datasets"][dataset]
        fractions = row["limit_all"]["fractions"]
        tv = [row["tv"][key]["all_episodes"] for key in ("min_vs_T_only", "min_vs_C_only")]
        if any(s["eligible_episodes"] != row["episodes"] or s["undefined_episodes"] for s in tv):
            raise RuntimeError("Undefined allocation comparison needs an explicit denominator")
        cells = [name, f"{row['candidates']:,}",
                 *[f"{100 * fractions[key]:.2f}" for key in ("T_lt_C", "C_lt_T", "abs_gap_ge_005")],
                 *[f"{s['mean']:.3f}" for s in tv]]
        rows.append(" & ".join(cells) + r" \\")
    output = args.paper / "generated/probe_complementarity.tex"
    output.write_text("\n".join((r"\begin{table*}[t]",
        r"\caption{Descriptive two-probe complementarity on the five full-split candidate caches. Percentages pool candidates; $T<C$ and $C<T$ use absolute tolerance $10^{-6}$, with remaining cases tied. TV is the episode-mean total variation between normalized minimum weights and the indicated single-view allocation, after each arm's $10^{-8}$ admission rule. Both views change allocation, but this alone does not establish extra segmentation gain or justify their cost.}",
        r"\label{tab:probe-complementarity}", r"\centering\small",
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"Dataset & Candidates & $T<C$ (\%) & $C<T$ (\%) & $|T-C|\geq.05$ (\%) & TV vs. $T$ & TV vs. $C$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", "")))
    provenance = dict(audit=str(args.audit.resolve()), audit_seal_sha256=digest(args.audit / "AUDIT_SEALED.json"),
                      script_sha256=digest(Path(__file__)), output_sha256=digest(output))
    (args.paper / "generated/probe_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
