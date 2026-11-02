"""Export all completed fixed-group transitions without selecting favorable signs."""

import argparse
import hashlib
import json
from pathlib import Path


NAMES = {"voc21": "VOC21", "pc60": "PC60", "coco_object": "COCO Object",
         "coco_cd": "COCO-CD", "coco_stuff": "COCO-Stuff"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root, seal_name):
    seal = read(root / seal_name)
    for name, expected in seal["artifact_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise RuntimeError("Transition/stratification evidence changed")


def value(number, best=False):
    if number is None:
        return "--"
    rendered = f"{number:.2f}"
    return "$" + (r"\mathbf{" + rendered + "}" if best else rendered) + "$"


def render(root):
    verify(root, "FINAL_ANALYSIS_SEALED.json")
    summary = read(root / "summary.json")
    rows, strata_rows, count_rows = [], [], []
    for dataset, name in NAMES.items():
        report = summary["datasets"][dataset]
        verify(root / dataset, "ANALYSIS_SEALED.json")
        source = read(root / dataset / "summary.json")
        if source != report or not report["full_split"]:
            raise RuntimeError("Incomplete or inconsistent transition report")
        group = {row["policy"]: row for row in report["results"]
                 if row["group"] == "high_conf_low_quality" and row["policy"] in ("trace", "confidence")}
        confidence, trace = group["confidence"], group["trace"]
        for key in ("candidates", "pre_correct", "pre_wrong"):
            if confidence[key] != trace[key]:
                raise RuntimeError("Policies do not share initially defined candidate footprints")
        cw = [row["correct_to_wrong_pct"] for row in (confidence, trace)]
        wc = [row["wrong_to_correct_pct"] for row in (confidence, trace)]
        cells = [name, value(cw[0], cw[0] <= cw[1]), value(wc[0], wc[0] >= wc[1]),
                 value(cw[1], cw[1] <= cw[0]), value(wc[1], wc[1] >= wc[0])]
        rows.append(" & ".join(cells) + r" \\")
        count_rows.append(" & ".join([name, *[f"{trace[k]:,}" for k in
                          ("candidates", "pre_correct", "pre_wrong")]]) + r" \\")
        stratified = root / ("strata_" + dataset)
        verify(stratified, "STRATIFIED_ANALYSIS_SEALED.json")
        strata = read(stratified / "summary.json")
        if strata["source_analysis_sha256"] != digest(root / dataset / "ANALYSIS_SEALED.json"):
            raise RuntimeError("Stratification references another initial transition audit")
        stats = next(r["cw"] for r in strata["results"] if r["policy"] == "confidence")
        gap = stats["high_minus_low_pp"]
        strata_rows.append(" & ".join((name, str(stats["strata"]), value(stats["low_quality_pct"]),
            value(stats["high_quality_pct"]), f"${gap:+.2f}$" if gap is not None else "--")) + r" \\")
    main = "\n".join((r"\begin{table}[!t]",
        r"\caption{Label transitions (\%) on high-confidence, low-support footprints ($\kappa_R\geq0.9$, $\rho_R<0.25$), full seed-0 splits. C$\to$W conditions on initially correct pixels; W$\to$C on initially wrong pixels. Group sizes and exact denominators appear in Appendix Table~\ref{tab:risk-group-counts}. Policies share initial candidates; outcomes follow joint adaptation. Bold marks the better rate in each pair.}",
        r"\label{tab:update-risk}", r"\centering\small", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r" & \multicolumn{2}{c}{Confidence} & \multicolumn{2}{c}{TRACE} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Dataset & C$\to$W $\downarrow$ & W$\to$C $\uparrow$ & C$\to$W $\downarrow$ & W$\to$C $\uparrow$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    appendix = "\n".join((r"\begin{table*}[t]",
        r"\caption{Descriptive risk stratification under the fixed confidence-weighted update policy. Candidates are stratified by predicted semantic category, initial confidence, and token area; within each stratum, the lowest and highest common-support quartiles are compared. Values are candidate-count-weighted means of stratum-level C$\to$W rates, using strata with initially correct pixels in both groups. The difference is high minus low support (percentage points); it is not a causal effect or a candidate-independent significance test.}",
        r"\label{tab:risk-strata}", r"\centering\small", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Dataset & Valid strata & Low support & High support & Difference \\", r"\midrule",
        *strata_rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    counts = "\n".join((r"\begin{table}[t]",
        r"\caption{Fixed high-confidence/low-support group sizes. $n^{\mathrm C}$ and $n^{\mathrm W}$ are the initially correct and wrong valid footprint pixels, respectively. Both policies use identical denominators; candidate/pixel counts are not independent sample sizes.}",
        r"\label{tab:risk-group-counts}", r"\centering\small", r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule", r"Dataset & Regions & $n^{\mathrm C}$ & $n^{\mathrm W}$ \\",
        r"\midrule", *count_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    return main, appendix, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    main_table, appendix_table, counts_table = render(args.analysis)
    output = args.paper / "generated"
    (output / "update_risk.tex").write_text(main_table)
    (output / "risk_strata.tex").write_text(appendix_table)
    (output / "risk_group_counts.tex").write_text(counts_table)
    provenance = dict(analysis=str(args.analysis.resolve()), final_analysis_seal_sha256=digest(args.analysis / "FINAL_ANALYSIS_SEALED.json"),
        script_sha256=digest(Path(__file__)), output_sha256={name: digest(output / name) for name in
        ("update_risk.tex", "risk_strata.tex", "risk_group_counts.tex")})
    (output / "risk_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
