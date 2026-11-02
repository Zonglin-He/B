"""Read sealed historical artifacts and write manuscript-local tables only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated"
TRACE = Path("/home/wwww/hzl/results/trace_two_view_full_20261025")
NATIVE = Path("/home/wwww/hzl/results/trace_two_view_full_baselines_20261025")
DATASETS = ("pc60", "voc21", "coco_cd", "coco_object")
LABELS = dict(pc60="PC60", voc21="VOC21", coco_cd="COCO-CD", coco_object="COCO Object")
COUNTS = dict(pc60=5105, voc21=1449, coco_cd=500, coco_object=5000)
EXPECTED = dict(pc60=30.5843252287, voc21=52.3964894036,
                coco_cd=75.1895608125, coco_object=32.6782695892)
HASHES = {}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    old = HASHES.setdefault(str(path), digest)
    need(old == digest, f"Source changed while exporting: {path}")
    return raw


def data(path):
    return json.loads(read(path))


def rows(path):
    return list(csv.DictReader(read(path).decode().splitlines()))


def owned(root, relative, seal):
    path = root / relative
    raw = read(path)
    need(seal["artifact_sha256"].get(str(relative)) == HASHES[str(path)],
         f"Artifact is not owned by supplied seal: {path}")
    return raw


def checked_cell(root, relative, seal, dataset, algorithm=None, reference=None):
    result = json.loads(owned(root, relative / "result.json", seal))
    protocol = json.loads(owned(root, relative / "protocol.json", seal))
    need(result["status"] == "completed", f"Incomplete result: {relative}")
    need(result["full_split"] and protocol["full_split"], "Not a full split")
    need(result["images"] == protocol["images"] == COUNTS[dataset], "Image count mismatch")
    need(protocol["seed"] == 0 and protocol["dataset"] == dataset, "Protocol identity mismatch")
    need(result["protocol_sha256"] == HASHES[str(root / relative / "protocol.json")],
         "Result/protocol hash mismatch")
    if algorithm is not None:
        need(result["algorithm"] == protocol["algorithm"] == algorithm, "Algorithm mismatch")
    if reference is not None:
        for key in ("image_ids", "label_ids", "vocabulary_sha256", "config"):
            need(protocol[key] == reference[key], f"Native inputs differ: {dataset}/{key}")
        need(protocol["labels_used_for_adaptation"] is False, "Label-use declaration mismatch")
    intersections, unions = result["intersection"], result["union"]
    need(len(intersections) == len(unions), "Metric shape mismatch")
    need(all(0 <= i <= u for i, u in zip(intersections, unions)), "Invalid metric counts")
    ratios = [i / u for i, u in zip(intersections, unions) if u > 0]
    score = 100 * math.fsum(ratios) / len(ratios)
    need(math.isclose(score, result["miou_unrounded"], abs_tol=1e-9), "Pooled mIoU mismatch")
    return result, protocol, score


def write(name, content):
    target = OUT / name
    need(target.parent.resolve() == OUT.resolve(), "Output escaped generated/")
    target.write_text(content.rstrip() + "\n", encoding="utf-8")


def table(caption, label, spec, header, body, wide=False, placement="t"):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[{placement}]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\centering\\small", f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
             "\\toprule", header + r" \\", "\\midrule"]
    lines.extend(body)
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"])
    return "\n".join(lines)


def main():
    need(OUT.is_dir(), "Create generated/ before export")
    trace_seal = data(TRACE / "SUITE_SEALED.json")
    native_seal = data(NATIVE / "SUITE_SEALED.json")
    configs = json.loads(owned(TRACE, Path("fixed_configs.json"), trace_seal))
    suite = json.loads(owned(TRACE, Path("suite_protocol.json"), trace_seal))
    need(suite["seed"] == 0 and suite["config_fixed"] and not suite["parameter_search"],
         "Unexpected suite selection identity")
    reported_path = ROOT / "reported_baselines.json"
    reported = data(reported_path)
    source_pdf = Path(reported["source_pdf"])
    read(source_pdf)
    summaries, protocols, native, costs = {}, {}, {}, {}
    for dataset in DATASETS:
        relative = Path("full_validation") / dataset / "quality_only" / "attempt_001"
        result, protocol, score = checked_cell(TRACE, relative, trace_seal, dataset, "quality_only")
        need(math.isclose(score, EXPECTED[dataset], abs_tol=1e-9), "User-verified score mismatch")
        need(protocol["config"] == configs[dataset], "Fixed configuration mismatch")
        summaries[dataset], protocols[dataset] = score, protocol
        costs[dataset] = list(csv.DictReader(owned(TRACE, relative / "cost.csv", trace_seal)
                                          .decode().splitlines()))
        need(sum(int(r["images"]) for r in costs[dataset]) == COUNTS[dataset], "Cost count mismatch")
        for row in costs[dataset]:
            need(int(row["diagnostic_views"]) == 2 * int(row["candidates"]), "Not two-view execution")
        for method in ("noadapt", "tent", "tpt", "mlmp"):
            base_relative = Path(dataset) / method
            base, bp, bs = checked_cell(NATIVE, base_relative, native_seal, dataset,
                                        reference=protocol)
            need(base["method"] == bp["method"] == method, "Native method mismatch")
            if method != "noadapt":
                constructor = bp["method_settings"]["constructor"]
                need(constructor["lr"] == .001 and constructor["steps"] == 10,
                     "Native optimization setting mismatch")
            native[dataset, method] = bs

    published_ds = ("voc21", "pc60", "coco_object")
    body = []
    for method in reported["methods"]:
        cells = []
        for dataset in published_ds:
            mean, sd = reported["datasets"][dataset][method]
            cells.append(f"${mean:.2f}$" if sd is None else f"${mean:.2f}\\pm{sd:.2f}$")
        body.append(method + " (R) & " + " & ".join(cells) + r" \\")
    body.append("\\midrule")
    body.append(r"\method{} (L, seed 0) & " + " & ".join(
        f"$\\mathbf{{{summaries[d]:.2f}}}$" for d in published_ds) + r" \\")
    deltas = {d: summaries[d] - max(v[0] for v in reported["datasets"][d].values())
              for d in published_ds}
    body.append(r"$\Delta$ vs. best reported (pp) & " + " & ".join(
        f"$+{deltas[d]:.2f}$" for d in published_ds) + r" \\")
    write("reported_comparison.tex", table(
        r"Clean-split mIoU ($\uparrow$) with NACLIP ViT-L/14. R: reported values, including "
        r"standard deviations, from the main Table~5 of \citet{noori2025mlmp}; not rerun locally. "
        r"L: local full-split seed-0 two-view results. Reported baselines use batch 2, "
        r"learning rate 0.001, and 10 steps; our per-dataset settings differ "
        r"(Appendix~\ref{app:protocol}). Deltas compare operating points, not equal tuning or "
        r"optimization budgets. Exact checkpoint and query-extension equivalence is not asserted.",
        "tab:reported", "lrrr", "Method / source & VOC21 & PC60 & COCO Object", body, wide=True))
    write("comparison_summary.tex",
          "\\method{} reaches " + ", ".join(f"{summaries[d]:.2f}" for d in published_ds) +
          " mIoU on VOC21, PC60, and COCO Object, respectively. Relative to the strongest "
          "reported baseline, MLMP, the differences are " +
          ", ".join(f"+{deltas[d]:.2f}" for d in published_ds) + " percentage points, respectively.")

    cd_rows = [f"{label} (L) & ${native['coco_cd', m]:.2f}$" + r" \\" for m, label in
               (("noadapt", "No Adapt"), ("tent", "TENT"), ("tpt", "TPT"), ("mlmp", "MLMP"))]
    cd_rows += ["\\midrule", r"\method{} (L) & " + f"$\\mathbf{{{summaries['coco_cd']:.2f}}}$" + r" \\"]
    cd_delta = summaries["coco_cd"] - max(native["coco_cd", m] for m in ("noadapt", "tent", "tpt", "mlmp"))
    cd_rows.append(r"$\Delta$ vs. best local (pp) & " + f"$+{cd_delta:.2f}$" + r" \\")
    write("cd_local.tex", table(
        r"COCO-CD mIoU ($\uparrow$), complete 500-image panel, seed 0. L denotes already "
        r"completed local native-backend evaluations, not numbers reported in the MLMP paper. "
        r"The input panel and episodes match; backends and settings remain method-specific. "
        r"Baselines use 0.001/10 steps; \method{} uses 0.0086/5 steps. Not equal-budget or equal-tuning.",
        "tab:cd-local", "lr", "Method / source & COCO-CD", cd_rows))

    native_body = [(label + " & " + " & ".join(f"${native[d,m]:.2f}$" for d in DATASETS) + r" \\")
                   for m, label in (("noadapt", "No Adapt"), ("tent", "TENT"),
                                    ("tpt", "TPT"), ("mlmp", "MLMP"))]
    native_body += ["\\midrule", r"\method{} & " + " & ".join(
        f"$\\mathbf{{{summaries[d]:.2f}}}$" for d in DATASETS) + r" \\"]
    native_deltas = {d: summaries[d] - max(native[d,m] for m in ("noadapt", "tent", "tpt", "mlmp"))
                     for d in DATASETS}
    native_body.append(r"TRACE gain & " + " & ".join(
        f"$+{native_deltas[d]:.2f}$" for d in DATASETS) + r" \\")
    write("native_comparison.tex", table(
        r"Historical local native-backend comparison, full splits, seed 0, mIoU ($\uparrow$). "
        r"All rows use the same input panels and episodes, not identical backends, prompts, "
        r"tuning histories, or optimization budgets. Native baselines use learning rate 0.001 "
        r"and 10 steps; \method{} uses dataset-specific settings. These native deltas are separate "
        r"from the published-baseline deltas in main-paper Table~\mainref{tab:reported}.", "tab:native", "lrrrr",
        "Method & PC60 & VOC21 & COCO-CD & Object", native_body, placement="ht"))

    config_body = []
    cost_body = []
    cost_records = {}
    for dataset in DATASETS:
        config = configs[dataset]
        config_body.append(" & ".join((LABELS[dataset], str(config["batch_size"]),
                           f"{config['lr']:.5f}", str(config["steps"]), str(config["trace_max_regions"]),
                           str(config["trace_view_batch_size"]), "Yes" if config["class_extensions"] else "No")) + r" \\")
        measured = costs[dataset][2:]
        n = sum(int(r["images"]) for r in measured)
        ms = 1000 * math.fsum(float(r["total_seconds"]) for r in measured) / n
        views = sum(int(r["diagnostic_views"]) for r in measured) / n
        memory = max(float(r["peak_allocated_mib"]) for r in measured)
        cost_records[dataset] = dict(measured_images=n, warmup_episodes=2, ms_per_image=ms,
                                     diagnostic_views_per_image=views, peak_allocated_mib=memory)
        cost_body.append(f"{LABELS[dataset]} & {n:,} & {views:.2f} & {ms:.2f} & {memory:.1f}" + r" \\")
    write("configurations.tex", table(
        r"Sealed two-view per-dataset settings. Batch denotes source images per episode, "
        r"$K$ the candidate limit per source image across crops, $B_v$ the maximum diagnostic "
        r"view-batch size, and Ext. query extensions. Input crop size is 224 and ViT patch size is 14.",
        "tab:configurations", "lrrrrrc", r"Dataset & Batch & LR & Steps & $K$ & $B_v$ & Ext.", config_body))
    write("cost_snapshot.tex", table(
        r"Recorded two-view research-run cost, not dedicated-entry-point validation or a speedup "
        r"comparison. The first two episodes are excluded from timing, not from mIoU. Inputs and "
        r"latency are per measured image; memory is peak allocated MiB. Synchronized adaptation "
        r"and final inference exclude loading, scoring, and model setup. Source environment: "
        r"RTX 5090, PyTorch 2.7.0+cu128. Settings differ across datasets.", "tab:cost", "lrrrr",
        r"Dataset & Images & \shortstack{Diag. inputs/\\image} & ms/img & MiB", cost_body))

    # A publication record is separate from historical result provenance.
    snapshot = dict(scope="CPU-only manuscript export; no inference or experiment execution",
                    algorithm="quality_only", dedicated_entrypoint_gpu_validated=False,
                    trace_scores=summaries, reported_deltas=deltas, native_deltas=native_deltas,
                    published_reference=reported, source_sha256=HASHES,
                    cost_snapshot=cost_records,
                    validation=["Referenced artifacts match owning seals", "Protocol hash links match",
                                "Pooled mIoU reconstructed from class counts", "Native input panels and vocabulary match",
                                "Two diagnostic inputs per candidate in all cost rows"],
                    excluded_claims=["GPU equivalence of dedicated entry point", "equal tuning budgets",
                                     "calibrated reliability", "causal identification", "new mechanism results"])
    for path, digest in list(HASHES.items()):
        need(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f"Source changed: {path}")
    write("provenance.json", json.dumps(snapshot, indent=2, sort_keys=True))
    print(json.dumps(dict(trace_scores=summaries, reported_deltas=deltas,
                          native_deltas=native_deltas, source_files=len(HASHES)), indent=2))


if __name__ == "__main__":
    main()
