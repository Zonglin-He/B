"""Import a complete deterministic evolution panel, without loading a model.

Exit codes: 0 ready/imported, 3 pending (old tables untouched), 2 invalid.
--check-only still performs deep verification; it never writes output files.
Final import supplies both --product-suite-root (B) and --voc20-suite-root (C)
alongside --suite-root (A), and requires all 24 declared cells to be sealed.
The active suite is never modified. Historical failed suites are never resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import types


PAPER_ROOT = Path(__file__).resolve().parents[1]
CORE = ("coco_cd", "pc60", "voc21", "coco_object")
POLICIES = ("trace", "uniform", "confidence", "target", "counterfactual", "mean", "permuted")
PERMUTATION_SEEDS = (0, 1, 2)
APPEND_POLICIES = ("product", "pixel_normalized")
NATIVE_ROOT = Path("/home/wwww/hzl/results/trace_two_view_full_baselines_20261025")
LABELS = {"coco_cd": "COCO-CD", "pc60": "PC60", "voc21": "VOC21",
          "coco_object": "COCO Object", "coco_stuff": "COCO-Stuff", "voc20": "VOC20"}


class Invalid(ValueError):
    pass


def require(value, message):
    if not value:
        raise Invalid(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_path(root, relative):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes root: {relative}")
    return path


def runtime_hashes(root):
    return {str(p.relative_to(root)): sha(checked_path(root, p.relative_to(root)))
            for p in sorted(root.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}}


def verify_manifest(root, voc20=False):
    manifest_path = root / "manifest.json"
    seal = read_json(root / "MANIFEST_SEALED.json")
    require(sha(manifest_path) == seal.get("manifest_sha256"), "Manifest seal mismatch")
    manifest = read_json(manifest_path)
    require(runtime_hashes(root / "runtime") == manifest.get("runtime_hashes"),
            "Frozen runtime hash mismatch")
    require(manifest.get("method") == "trace_ovss.method.two_view.TwoViewTRACE"
            and manifest.get("rho") == "min(T,C)", "Wrong method identity")
    require(manifest.get("permutation_seeds") == list(PERMUTATION_SEEDS),
            "Expected exactly the three fixed permutation seeds 0,1,2")
    cells = manifest.get("cells", [])
    count = 4 if voc20 else 18
    require(len(cells) == count and len({c["key"] for c in cells}) == count,
            f"Expected {count} uniquely identified manifest cells")
    expected = {f"{phase}/{d}" for phase in ("smoke", "mechanism")
                for d in (*CORE, "coco_stuff")}
    expected |= {f"repetition/{d}/seed{s}" for d in CORE for s in (1, 2)}
    if voc20:
        expected = {"smoke/voc20", "mechanism/voc20", "repetition/voc20/seed1",
                    "repetition/voc20/seed2"}
    require({c["key"] for c in cells} == expected, "Manifest cell population differs")
    for cell in cells:
        parts = cell["key"].split("/")
        require(cell["dataset"] == parts[1], "Cell dataset/key mismatch")
        repetition = parts[0] == "repetition"
        policies = ["trace"] if repetition or voc20 else list(POLICIES)
        require(len(cell.get("policies", [])) == len(policies)
                and set(cell["policies"]) == set(policies),
                f"Incomplete or duplicated policy set: {cell['key']}")
        require(cell["seed"] == (int(parts[2][4:]) if repetition else 0), "Cell seed mismatch")
        require(cell["limit"] == (4 if parts[0] == "smoke" else None), "Cell split limit mismatch")
        require(cell["key"] in manifest.get("input_panels", {}), "Missing frozen panel")
        checked_path(root / "cells", cell["key"])
    return manifest, seal


def load_controller(root, manifest):
    path = root / "runtime/tools/self_evolution_controller.py"
    require(manifest["runtime_hashes"].get("tools/self_evolution_controller.py") == sha(path),
            "Controller is not bound to the runtime seal")
    # Execute the verified stdlib module under a non-main name, without .pyc writes.
    module = types.ModuleType("_frozen_evolution_verifier")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    module.verify_runtime(root, manifest)
    return module


def expected_panel(manifest, cell):
    panel = dict(manifest["input_panels"][cell["key"]], seed=cell["seed"])
    panel["image_sha256"] = {p: manifest["image_sha256"][p] for p in panel["image_ids"]}
    return panel


def policy_scores(result, policies):
    require(result.get("status") == "completed", "Result is not completed")
    expected = {(p, s) for p in policies
                for s in (PERMUTATION_SEEDS if p == "permuted" else (None,))}
    rows = result.get("policy_results", [])
    keys = [(r["policy"], r.get("permutation_seed")) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            "Policy results are missing, duplicated, or unrequested")
    scores = {}
    for row, key in zip(rows, keys):
        value = row["miou_unrounded"]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100,
                "Invalid mIoU")
        scores[key] = float(value)
    return scores


def same_panel(old, new):
    for key in ("image_ids", "label_ids", "vocabulary_sha256", "file_metadata", "available_images"):
        require(old.get(key) == new.get(key) and key in old and key in new,
                f"Native baseline panel mismatch: {key}")
    for key in ("dataset", "resize", "patch_size", "patch_stride", "batch_size", "class_extensions", "data_dir"):
        require(key in old["config"] and old["config"][key] == new["config"].get(key),
                f"Native baseline preprocessing/episode mismatch: {key}")


def native_cd(panel, source_hashes):
    root = NATIVE_ROOT
    seal_path = root / "SUITE_SEALED.json"
    seal = read_json(seal_path)
    source_hashes[str(seal_path)] = sha(seal_path)
    scores = {}
    for method in ("noadapt", "tent", "tpt", "mlmp"):
        records = {}
        for name in ("result.json", "protocol.json"):
            relative = f"coco_cd/{method}/{name}"
            path = checked_path(root, relative)
            digest = sha(path)
            require(seal["artifact_sha256"].get(relative) == digest, "Native artifact seal mismatch")
            source_hashes[str(path)] = digest
            records[name] = read_json(path)
        result, protocol = records["result.json"], records["protocol.json"]
        same_panel(protocol, panel)
        require(result.get("status") == "completed" and result.get("full_split") is True
                and protocol.get("full_split") is True and protocol.get("seed") == 0
                and result["images"] == len(panel["image_ids"]), "Incomplete native result")
        require(result["protocol_sha256"] == source_hashes[str(root / f"coco_cd/{method}/protocol.json")],
                "Native result/protocol hash mismatch")
        require(protocol.get("method") == result.get("method") == method
                and protocol.get("labels_used_for_adaptation") is False, "Native method identity mismatch")
        i, u = result["intersection"], result["union"]
        # Historical native metrics serialize integer-valued counts as float64.
        require(len(i) == len(u) and all(whole_count(a) and whole_count(b) and 0 <= a <= b
                                       for a, b in zip(i, u)), "Invalid native class counts")
        ratios = [a / b for a, b in zip(i, u) if b]
        require(bool(ratios), "No valid native classes")
        value = 100 * math.fsum(ratios) / len(ratios)
        require(math.isclose(value, result["miou_unrounded"], abs_tol=1e-10, rel_tol=0),
                "Native pooled mIoU mismatch")
        scores[method] = value
    return scores


def whole_count(value):
    return type(value) in (int, float) and math.isfinite(value) and value == int(value) and value >= 0


def verify_index(root, count, fingerprints):
    seal_path = root / "SUITE_SEALED.json"
    seal = read_json(seal_path)
    entries = seal.get("artifact_sha256", {})
    require({"manifest.json", "queue_state.json"} <= set(entries), "Incomplete suite seal")
    for relative, digest in entries.items():
        path = checked_path(root, relative)
        require(sha(path) == digest, f"Suite artifact seal mismatch: {path}")
        fingerprints[str(path)] = digest
    fingerprints[str(seal_path)] = sha(seal_path)
    state = read_json(root / "queue_state.json")
    require(state.get("status") == "completed" and state.get("completed") == count
            and state.get("total") == count, "Suite has not completed its entire plan")
    return set(entries)


def verify_reported(reported, fingerprints):
    """Re-extract main Table 5, including null SDs, rather than appendix variants."""
    path = Path(reported["source_pdf"])
    require(sha(path) == reported.get("source_pdf_sha256"), "Reported baseline PDF hash mismatch")
    fingerprints[str(path)] = sha(path)
    text = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                          capture_output=True, text=True).stdout
    require("Table 5:" in text, "Missing main Table 5")
    table = text.split("Table 5:", 1)[1].split("shown in Table 4", 1)[0]
    require(reported["methods"] == ["No Adapt", "TENT", "TPT", "WATT", "CLIPArTT", "MLMP"],
            "Unexpected published method order")
    for dataset, label in (("voc20", "V20"), ("voc21", "V21"), ("pc60", "P60"),
                           ("coco_object", "COCOObject"), ("coco_stuff", "COCOStuff")):
        matches = re.findall(r"^\s*" + label + r" \(Original\)\s+([^\n]+)", table, re.M)
        require(len(matches) == 1, f"Ambiguous Table 5 row: {dataset}")
        # pdftotext emits the literal plus/minus glyph, not a TeX command.
        tokens = re.findall(r"(\d+\.\d+)(?:\s*\u00b1\s*(\d+\.\d+))?", matches[0])
        values = [[float(mean), float(sd) if sd else None] for mean, sd in tokens]
        require(len(values) == 6 and values == [reported["datasets"][dataset][m]
                                               for m in reported["methods"]],
                f"Published values differ from PDF main Table 5: {dataset}")


def verify_append_cell(directory, dataset, panel, parent, manifest, controller, fingerprints):
    """Verify cached controls without importing their GPU worker or scoring labels."""
    prediction = read_json(directory / "PREDICTIONS_SEALED.json")
    scored = read_json(directory / "SCORED_SEALED.json")
    covered = controller.verify_artifacts(directory, prediction)
    scored_covered = controller.verify_artifacts(directory, scored)
    required = {"prediction_protocol.json", "expected_panel.json", "initial_state.json",
                "prediction_result.json", "replay_gate.json", "checkpoint_identity.json"}
    require(required <= covered and {"result.json", "score_inputs.json", "scores_by_episode.jsonl"}
            <= scored_covered, "Append seal coverage is incomplete")
    prediction_sha = sha(directory / "PREDICTIONS_SEALED.json")
    require(prediction.get("complete") is True and prediction.get("gt_contents_opened") is False
            and prediction.get("images") == len(panel["image_ids"])
            and scored.get("prediction_seal_sha256") == prediction_sha, "Append seal chain differs")
    protocol = read_json(directory / "prediction_protocol.json")
    parent_protocol = read_json(parent / "prediction_protocol.json")
    require(read_json(directory / "expected_panel.json") == panel
            and protocol.get("expected_panel_sha256") == sha(directory / "expected_panel.json"),
            "Append expected panel differs")
    for key in ("panel", "inputs", "config", "checkpoint", "classes", "queries", "query_to_class",
                "reduce_zero_label", "deterministic_runtime", "dataset", "seed", "full_split"):
        require(key in parent_protocol and protocol.get(key) == parent_protocol[key],
                f"Append parent protocol mismatch: {key}")
    binding = dict(parent_cell=str(parent), parent_suite=str(parent.parents[2]),
                   parent_prediction_seal_sha256=sha(parent / "PREDICTIONS_SEALED.json"),
                   parent_suite_seal_sha256=sha(parent.parents[2] / "SUITE_SEALED.json"),
                   parent_manifest_sha256=sha(parent.parents[2] / "manifest.json"))
    require(protocol.get("parent_replay") == binding, "Append parent identity differs")
    sources = protocol.get("source_sha256", {})
    require(bool(sources), "Append source identity missing")
    allowed = {**{str(parent.parents[2] / "runtime" / p): h
                  for p, h in manifest["runtime_hashes"].items()},
               **{str(directory.parents[1] / "runtime" / p): h
                  for p, h in runtime_hashes(directory.parents[1] / "runtime").items()}}
    for path, digest in sources.items():
        require(allowed.get(path) == digest and sha(path) == digest, "Append worker source differs")
        fingerprints[path] = digest
    variants = [[p, p, None] for p in APPEND_POLICIES]
    require(protocol.get("variants") == variants, "Append variants differ")
    gate = read_json(directory / "replay_gate.json")
    require(gate.get("passed") is True and gate.get("changed_label_pixels") == 0
            and gate.get("tolerance") == 0 and gate.get("gt_contents_opened") is False
            and gate.get("images") == panel["config"]["batch_size"], "Append exact replay gate failed")
    initial = read_json(directory / "initial_state.json")
    old_initial = read_json(parent / "initial_state.json")
    require(initial.get("initial_identity_matches_parent") is True
            and initial.get("parent_initial_state_sha256") == sha(parent / "initial_state.json")
            and all(initial.get(k) == old_initial[k] for k in ("model_sha256", "optimizer_sha256")),
            "Append initial state differs")
    parent_seal = read_json(parent / "PREDICTIONS_SEALED.json")["artifact_sha256"]
    episodes = (len(panel["image_ids"]) + panel["config"]["batch_size"] - 1) // panel["config"]["batch_size"]
    for episode in range(episodes):
        prefix = f"episodes/{episode:06d}"
        require({prefix + suffix for suffix in (".cachelink.json", ".parent.npz", ".json", ".npz")}
                <= covered, "Append cache link coverage differs")
        link = read_json(directory / (prefix + ".cachelink.json"))
        parent_receipt = read_json(parent / (prefix + ".json"))
        require(link.get("parent_npz") == str(parent / (prefix + ".npz"))
                and link.get("parent_npz_sha256") == parent_seal.get(prefix + ".npz")
                and link.get("parent_npz_sha256") == prediction["artifact_sha256"][prefix + ".parent.npz"]
                and link.get("parent_json_sha256") == parent_seal.get(prefix + ".json")
                and link.get("parent_candidate_sha256") == parent_receipt.get("candidate_sha256"),
                "Append candidate cache identity differs")
    names = ["pre", *APPEND_POLICIES]
    payload = {f"predictions/{name}/{i:06d}.npy" for name in names
               for i in range(len(panel["image_ids"]))}
    require(payload <= covered and payload == {str(p.relative_to(directory))
            for p in (directory / "predictions").rglob("*.npy")}, "Append prediction population differs")
    result = read_json(directory / "result.json")
    require(result.get("dataset") == dataset and result.get("seed") == 0
            and result.get("full_split") is True and result.get("images") == len(panel["image_ids"])
            and result.get("prediction_seal_sha256") == prediction_sha
            and result.get("label_contents_first_opened_after_prediction_seal") is True,
            "Append result identity differs")
    policy_scores(result, APPEND_POLICIES)
    require(set(read_json(directory / "score_inputs.json").get("label_sha256", {}))
            == set(panel["label_ids"]), "Append offline label population differs")
    totals = {name: [[0] * len(protocol["classes"]) for _ in range(2)] for name in names}
    batch, images = panel["config"]["batch_size"], len(panel["image_ids"])
    records = 0
    with (directory / "scores_by_episode.jsonl").open() as handle:
        for episode, line in enumerate(handle):
            row = json.loads(line)
            require(row["episode"] == episode and row["image_indices"] == list(range(
                episode * batch, min((episode + 1) * batch, images)))
                and set(row["policies"]) == set(names), "Append scored episode coverage differs")
            for name, counts in row["policies"].items():
                i, u = counts["intersection"], counts["union"]
                require(len(i) == len(u) == len(protocol["classes"])
                        and all(type(a) is int and type(b) is int and 0 <= a <= b
                                for a, b in zip(i, u)), "Invalid append class counts")
                totals[name] = [[a + b for a, b in zip(x, y)]
                                for x, y in zip(totals[name], (i, u))]
            records += 1
    require(records == (images + batch - 1) // batch, "Missing append episodes")
    rows = [result["pre_result"], *result["policy_results"]]
    require(result["pre_result"] == read_json(parent / "result.json")["pre_result"],
            "Append pre-adaptation reference differs from parent")
    require([r["policy_label"] for r in rows] == names
            and all(r["policy"] == r["policy_label"] and r.get("permutation_seed") is None
                    for r in rows), "Append metric identity differs")
    for row in rows:
        i, u = totals[row["policy_label"]]
        ratios = [a / b for a, b in zip(i, u) if b]
        require(row["intersection"] == i and row["union"] == u and ratios
                and math.isclose(100 * math.fsum(ratios) / len(ratios), row["miou_unrounded"],
                                 abs_tol=1e-10, rel_tol=0), "Append pooled mIoU mismatch")
    for name in (*required, "PREDICTIONS_SEALED.json", "SCORED_SEALED.json", "result.json",
                 "score_inputs.json", "scores_by_episode.jsonl"):
        fingerprints[str(directory / name)] = sha(directory / name)
    return result


def signed(value):
    return f"{value:+.2f}"


def metric(value, best, sd=None):
    rendered = f"{value:.2f}" + (f"\\pm{sd:.2f}" if sd is not None else "")
    if math.isclose(value, best, abs_tol=1e-12, rel_tol=0):
        rendered = r"\mathbf{" + rendered + "}"
    return "$" + rendered + "$"


def tex_table(caption, label, datasets, body, wide=True, row_header="Method"):
    env = "table*" if wide else "table"
    return "\n".join((f"\\begin{{{env}}}[t]", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", r"\centering\small",
                      r"\begin{tabular}{@{}l" + "r" * len(datasets) + "@{}}",
                      r"\toprule", row_header + " & " + " & ".join(datasets) + r" \\",
                      r"\midrule", *body, r"\bottomrule", r"\end{tabular}", f"\\end{{{env}}}", ""))


def render(results, reported, local, controls=None):
    datasets = ["voc21", "pc60", "coco_object"]
    if "voc20" in results:
        datasets.insert(0, "voc20")
    if "coco_stuff" in results:
        datasets.append("coco_stuff")
    scores = {d: policy_scores(r, ["trace"] if d == "voc20" else POLICIES)
              for d, r in results.items()}
    full = {d: s["trace", None] for d, s in scores.items()}
    pre = {d: r.get("pre_result", {}).get("miou_unrounded") for d, r in results.items()}
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 100
                for v in pre.values()), "Missing or invalid current-readout No Adapt mIoU")
    best_reported = {d: max(v[0] for v in reported["datasets"][d].values()) for d in datasets}
    deltas = {d: full[d] - best_reported[d] for d in datasets}
    body = []
    best_column = {d: max(best_reported[d], full[d], pre[d]) for d in datasets}
    for method in reported["methods"]:
        body.append(method + " (R) & " + " & ".join(metric(
            reported["datasets"][d][method][0], best_column[d],
            reported["datasets"][d][method][1]) for d in datasets) + r" \\")
    body += [r"\midrule", r"No Adapt, current readout (L) & " + " & ".join(
        metric(pre[d], best_column[d]) for d in datasets) + r" \\",
        r"\method{} (L, seed 0) & " + " & ".join(
        metric(full[d], best_column[d]) for d in datasets) + r" \\",
        r"$\Delta$ vs. current No Adapt (pp) & " + " & ".join(
        "$" + signed(full[d] - pre[d]) + "$" for d in datasets) + r" \\",
        r"$\Delta$ vs. best reported (pp) & " + " & ".join("$" + signed(deltas[d]) + "$"
                                                                    for d in datasets) + r" \\"]
    outputs = {"reported_comparison.tex": tex_table(
        r"Clean full-split mIoU ($\uparrow$), NACLIP ViT-L/14. R: means and standard deviations "
        r"reported in main Table~5 of \citet{noori2025mlmp}, not rerun locally. L: deterministic "
        r"full seed-0 TRACE result and its own reset-model No Adapt readout. "
        r"The local adaptation delta is separate from the published-reference delta. "
        r"A missing reported SD is omitted, "
        r"not replaced with zero. Published baselines use batch 2, lr 0.001, "
        r"10 steps; TRACE retains dataset settings. Not equal optimization/tuning budgets or an "
        r"exact checkpoint/query-extension equivalence claim. Bold marks the highest unrounded "
        r"value in each column.", "tab:reported", [LABELS[d] for d in datasets], body)}
    best_cd = max(*local.values(), full["coco_cd"], pre["coco_cd"])
    cd_rows = [name + " (L) & " + metric(local[key], best_cd) + r" \\" for key, name in
               (("noadapt", "No Adapt"), ("tent", "TENT"), ("tpt", "TPT"), ("mlmp", "MLMP"))]
    cd_delta = full["coco_cd"] - max(local.values())
    cd_rows += [r"\midrule", r"No Adapt, current readout (L) & " + metric(pre["coco_cd"], best_cd) + r" \\",
                r"\method{} (L, seed 0) & " + metric(full["coco_cd"], best_cd) + r" \\",
                r"$\Delta$ vs. current No Adapt (pp) & $" + signed(full["coco_cd"] - pre["coco_cd"]) + r"$ \\",
                r"$\Delta$ vs. best local (pp) & $" + signed(cd_delta) + r"$ \\"]
    outputs["cd_local.tex"] = tex_table(
        r"COCO-CD full-split mIoU ($\uparrow$). TRACE uses the deterministic mechanism suite; "
        r"local native baselines are the previously completed sealed runs. Ordered panels, "
        r"vocabulary and episode boundaries are checked to match. Backends and settings remain "
        r"method-specific, not equal-budget or equal-tuning; no additional MLMP run is used.",
        "tab:cd-local", ["COCO-CD"], cd_rows, wide=False)
    # Five published-comparison datasets; keep the native CD controls separate.
    mechanisms = datasets if "voc20" in results else list(CORE) + (
        ["coco_stuff"] if "coco_stuff" in results else [])
    aggregates = {}
    for d in dict.fromkeys([*mechanisms, "coco_cd"]):
        s = scores[d]
        aggregates[d] = {p: (s[p, None], None) for p in POLICIES if (p, None) in s}
        if ("permuted", 0) in s:
            values = [s["permuted", seed] for seed in PERMUTATION_SEEDS]
            aggregates[d]["permuted"] = (statistics.mean(values), statistics.pstdev(values))
        if controls and d in controls:
            aggregates[d].update({p: (v, None) for (p, _), v in
                                 policy_scores(controls[d], APPEND_POLICIES).items()})
    rows = []
    policy_names = [("trace", "Full (min)"), ("uniform", "Uniform"), ("confidence", "Confidence"),
                    ("target", "T-only"), ("counterfactual", "C-only"), ("mean", "Mean"),
                    ("permuted", "Shuffled")]
    if controls:
        policy_names += [("product", "Product"), ("pixel_normalized", "Pixel-normalized")]
    def entry(d, p):
        if p not in aggregates[d]:
            return "--"
        value, sd = aggregates[d][p]
        return metric(value, max(v[0] for v in aggregates[d].values()), sd)
    for p, name in policy_names:
        rows.append(name + " & " + " & ".join(entry(d, p)
                    for d in mechanisms) + r" \\")
    outputs["mechanism_status.tex"] = tex_table(
        r"Deterministic full-split fixed-candidate controls, seed 0, mIoU ($\uparrow$). "
        r"Full uses the minimum; Uniform and Confidence use region-level weights. Shuffled is "
        r"mean $\pm$ population SD across fixed permutation seeds 0, 1, 2 (not independent "
        r"training seeds or a confidence interval). Bold marks the highest actual column value; "
        r"diagnostics are shared, so this is not standalone latency evidence. Product and "
        r"pixel-normalized controls reuse sealed initial diagnostics on PC60/Object only. "
        r"VOC20 has only Full evaluated; -- means genuinely unrun, not zero or a tied result.",
        "tab:mechanism", [LABELS[d] for d in mechanisms], rows)
    if "voc20" in results:
        outputs["cd_mechanism.tex"] = tex_table(
            r"COCO-CD deterministic full-split fixed-candidate controls. "
            r"Shuffled is mean $\pm$ population SD over permutation seeds, not independent "
            r"sampling uncertainty. Bold marks the actual best value.", "tab:cd-mechanism", ["COCO-CD"],
            [name + " & " + entry("coco_cd", p) + r" \\" for p, name in policy_names
             if p in aggregates["coco_cd"]], wide=False)
    outputs["comparison_summary.tex"] = (
        r"\method{} obtains " + ", ".join(f"{full[d]:.2f}" for d in datasets)
        + " mIoU on " + ", ".join(LABELS[d] for d in datasets)
        + ", respectively. Differences from the strongest published baseline are "
        + ", ".join(signed(deltas[d]) for d in datasets)
        + " percentage points, respectively; these compare method-specific operating points. "
        + "On COCO-CD, the difference from the best verified local native baseline is "
        + signed(cd_delta) + " percentage points. "
        + "The separate cost snapshot describes the historical research implementation, not "
        + "standalone timing of this deterministic shared-diagnostic suite.\n")
    return outputs, dict(full_scores=full, current_noadapt=pre,
                         adaptation_deltas={d: full[d] - pre[d] for d in full}, published_best=best_reported,
                         published_deltas=deltas, cd_local_delta=cd_delta,
                         mechanism_scores=aggregates,
                         min_minus_controls={d: {p: full[d] - v[0] for p, v in a.items()
                                                if p != "trace"} for d, a in aggregates.items()})


def render_replays(completed, replays):
    rows = []
    for d, result in completed.items():
        base = next(r["miou_unrounded"] for r in result["policy_results"] if r["policy"] == "trace")
        checks = [replays.get(f"repetition/{d}/seed{s}") for s in (1, 2)]
        values = [f"${base:.8f}$"] + [f"${r['miou']:.8f}$" if r else "--" for r in checks]
        values += ["$" + f"{r['miou'] - base:+.8f}" + "$" if r else "--" for r in checks]
        rows.append(LABELS[d] + " & " + " & ".join(values) + r" \\")
    return tex_table(
        r"Deterministic seed replay checks, full-split mIoU. Seeds 1/2 replay the fixed "
        r"protocol and are not independent uncertainty samples; no seed SD or confidence "
        r"interval is estimated. Differences are replay minus seed 0 in percentage points. "
        r"-- means no replay was run. Eight decimals are shown to expose numerical agreement.",
        "tab:seed-replay", ["Seed 0", "Seed 1", "Seed 2", r"$\Delta_1$", r"$\Delta_2$"], rows,
        row_header="Dataset")


def render_configs(completed, panels):
    rows = []
    for d in completed:
        c = panels[f"mechanism/{d}"]["config"]
        values = [str(c[k]) for k in ("batch_size", "lr", "steps", "trace_max_regions", "trace_view_batch_size")]
        rows.append(LABELS[d] + " & " + " & ".join(values)
                    + (" & Yes" if c["class_extensions"] else " & No") + r" \\")
    return tex_table(
        r"Frozen per-dataset settings; all allocation and spatial-normalization controls retain "
        r"their dataset configuration. Batch denotes source images per episode, $K$ candidate limit, "
        r"$B_v$ diagnostic view-batch size, and Ext. query extensions. Complete preprocessing "
        r"and source identities are retained in the generated provenance.", "tab:configurations",
        ["Batch", "LR", "Steps", "$K$", "$B_v$", "Ext."], rows, row_header="Dataset")


def publish(paper, outputs):
    directory = paper / "generated"
    require(directory.is_dir(), "generated/ does not exist")
    require(directory.resolve().is_relative_to(paper.resolve()), "generated/ escapes paper root")
    require(all(Path(n).name == n for n in outputs), "Invalid output name")
    for name in outputs:
        checked_path(directory, name)
    old = {name: (directory / name).read_bytes() if (directory / name).exists() else None
           for name in outputs}
    replaced = []
    # Stage the complete bundle first; rollback synchronous publication failures.
    with tempfile.TemporaryDirectory(prefix=".evolution-import-", dir=directory) as temporary:
        stage = Path(temporary)
        for name, content in outputs.items():
            (stage / name).write_text(content, encoding="utf-8")
        try:
            for name in outputs:
                os.replace(stage / name, directory / name)
                replaced.append(name)
        except BaseException:
            for name in reversed(replaced):
                if old[name] is None:
                    (directory / name).unlink(missing_ok=True)
                else:
                    (stage / name).write_bytes(old[name])
                    os.replace(stage / name, directory / name)
            raise


def verify_suite(root, fingerprints, voc20=False, strict=False):
    manifest, manifest_seal = verify_manifest(root, voc20=voc20)
    controller = load_controller(root, manifest)
    verified, panels = {}, {}
    fingerprints.update({
        str(root / "manifest.json"): manifest_seal["manifest_sha256"],
        str(root / "MANIFEST_SEALED.json"): sha(root / "MANIFEST_SEALED.json")})
    entries = verify_index(root, len(manifest["cells"]), fingerprints) if strict else None
    for cell in manifest["cells"]:
        directory = root / "cells" / cell["key"]
        if not (directory / "SCORED_SEALED.json").exists():
            continue
        if entries is not None:
            require({f"cells/{cell['key']}/{n}" for n in (
                "result.json", "PREDICTIONS_SEALED.json", "SCORED_SEALED.json")} <= entries,
                "Cell is outside suite seal coverage")
        panel = expected_panel(manifest, cell)
        try:
            result = controller.verify_cell(directory, cell, panel, manifest)
        except Exception as error:
            raise Invalid(f"Deep cell verification failed: {cell['key']}: {error}") from error
        require(result.get("dataset") == cell["dataset"] and result.get("seed") == cell["seed"]
                and result.get("images") == len(panel["image_ids"])
                and result.get("full_split") == (cell["limit"] is None), "Result cell identity differs")
        policy_scores(result, cell["policies"])
        verified[cell["key"]], panels[cell["key"]] = result, panel
        for name in ("PREDICTIONS_SEALED.json", "SCORED_SEALED.json", "result.json",
                     "prediction_protocol.json", "initial_state.json", "expected_panel.json"):
            fingerprints[str(directory / name)] = sha(directory / name)
    if strict:
        require(len(verified) == len(manifest["cells"]), "Incomplete final suite cells")
    return manifest, controller, verified, panels


def run(suite_root, paper_root=PAPER_ROOT, check_only=False, *, product_suite_root=None,
        voc20_suite_root=None):
    root, paper = Path(suite_root).resolve(), Path(paper_root).resolve()
    final = product_suite_root is not None or voc20_suite_root is not None
    require(not final or (product_suite_root is not None and voc20_suite_root is not None),
            "Final import requires both explicit product and VOC20 suite paths")
    fingerprints = {str(Path(__file__).resolve()): sha(Path(__file__))}
    manifest, controller, verified, panels = verify_suite(root, fingerprints, strict=final)
    missing = [f"mechanism/{d}" for d in CORE if f"mechanism/{d}" not in verified]
    if missing:
        return dict(status="pending", missing=missing, verified_cells=sorted(verified),
                    tables_unchanged=True, check_only=check_only), 3
    completed = {d: verified[f"mechanism/{d}"] for d in (*CORE, "coco_stuff")
                 if f"mechanism/{d}" in verified}
    suites = {"A": dict(root=str(root), manifest=manifest, verified_cells=sorted(verified))}
    controllers = [(root, manifest, controller)]
    directories = {key: root / "cells" / key for key in verified}
    controls = {}
    if final:
        b, c = Path(product_suite_root).resolve(), Path(voc20_suite_root).resolve()
        require(len({root, b, c}) == 3, "Suite paths must be distinct")
        b_manifest = read_json(b / "manifest.json")
        require(sha(b / "manifest.json") == read_json(b / "MANIFEST_SEALED.json").get("manifest_sha256"),
                "Append manifest seal mismatch")
        require(runtime_hashes(b / "runtime") == b_manifest.get("source_sha256"),
                "Append frozen source hash mismatch")
        require(Path(b_manifest["parent"]).resolve() == root
                and b_manifest["parent_manifest_sha256"] == sha(root / "manifest.json"),
                "Append parent manifest identity differs")
        require(b_manifest.get("datasets") == {"pc60": 5105, "coco_object": 5000}
                and b_manifest.get("controls") == list(APPEND_POLICIES)
                and b_manifest.get("candidate_cache_reuse") is True
                and b_manifest.get("parameter_search") is False, "Append control population differs")
        fingerprints[str(b / "MANIFEST_SEALED.json")] = sha(b / "MANIFEST_SEALED.json")
        entries = verify_index(b, 2, fingerprints)
        receipt_spec = b_manifest.get("parent_completion_receipt")
        if receipt_spec:
            receipt_path = Path(receipt_spec["path"])
            require(sha(receipt_path) == receipt_spec["sha256"], "Parent completion receipt changed")
            receipt = read_json(receipt_path)
            require(receipt.get("root") == str(root) and receipt.get("all_cells_verified") == 18
                    and receipt.get("manifest_sha256") == sha(root / "manifest.json")
                    and receipt.get("suite_seal_sha256") == sha(root / "SUITE_SEALED.json")
                    and receipt.get("queue_state_sha256") == sha(root / "queue_state.json")
                    and receipt.get("process_exit_status_certified") is False,
                    "Parent completion receipt binding differs")
            fingerprints[str(receipt_path)] = receipt_spec["sha256"]
        for d, images in b_manifest["datasets"].items():
            require({f"cells/{d}/{n}" for n in ("result.json", "replay_gate.json",
                    "PREDICTIONS_SEALED.json", "SCORED_SEALED.json")} <= entries,
                    "Append cell is outside suite seal coverage")
            key = f"controls/{d}"
            parent = root / "cells/mechanism" / d
            require(len(panels[f"mechanism/{d}"]["image_ids"]) == images, "Append full panel size differs")
            controls[d] = verify_append_cell(b / "cells" / d, d, panels[f"mechanism/{d}"],
                                              parent, manifest, controller, fingerprints)
            panels[key] = panels[f"mechanism/{d}"]
            directories[key] = b / "cells" / d
        suites["B"] = dict(root=str(b), manifest=b_manifest, verified_cells=sorted(controls))
        c_manifest, c_controller, c_verified, c_panels = verify_suite(c, fingerprints, voc20=True, strict=True)
        dependencies = c_manifest.get("dependencies", [])
        require(len(dependencies) == 2 and {Path(d["path"]).resolve() for d in dependencies} == {root, b},
                "VOC20 dependencies do not identify A/B")
        for dependency in dependencies:
            dep_root = Path(dependency["path"]).resolve()
            require(dependency["manifest_sha256"] == sha(dep_root / "manifest.json")
                    and dependency["total"] == (18 if dep_root == root else 2),
                    "VOC20 dependency identity differs")
            if dependency.get("completion_receipt"):
                spec = dependency["completion_receipt"]
                require(sha(spec["path"]) == spec["sha256"], "VOC20 dependency receipt changed")
                fingerprints[spec["path"]] = spec["sha256"]
        suites["C"] = dict(root=str(c), manifest=c_manifest, verified_cells=sorted(c_verified))
        controllers.append((c, c_manifest, c_controller))
        verified.update(c_verified)
        panels.update(c_panels)
        directories.update({key: c / "cells" / key for key in c_verified})
        completed["voc20"] = c_verified["mechanism/voc20"]
    report_path = paper / "reported_baselines.json"
    reported = read_json(report_path)
    fingerprints[str(report_path)] = sha(report_path)
    if final or "source_pdf_sha256" in reported:
        verify_reported(reported, fingerprints)
    local = native_cd(panels["mechanism/coco_cd"], fingerprints)
    outputs, comparisons = render(completed, reported, local, controls)
    replays = {}
    for key in sorted(verified):
        if key.startswith("repetition/"):
            d = verified[key]["dataset"]
            same_panel(panels[key], panels[f"mechanism/{d}"])
            require(panels[key]["config"] == panels[f"mechanism/{d}"]["config"],
                    "Replay fixed configuration differs")
            score = policy_scores(verified[key], ["trace"])["trace", None]
            payloads = []
            for directory in (directories[key], directories[f"mechanism/{d}"]):
                payloads.append({p: h for p, h in read_json(directory / "PREDICTIONS_SEALED.json")
                                 ["artifact_sha256"].items() if p.startswith("predictions/trace/")})
            replays[key] = dict(seed=verified[key]["seed"], miou=score,
                                delta_from_seed0=score - comparisons["full_scores"][d],
                                identical_trace_predictions=bool(payloads[0]) and payloads[0] == payloads[1],
                                interpretation="deterministic replay, not an independent variance sample")
    outputs["seed_replay.tex"] = render_replays(completed, replays)
    if final:
        outputs["configurations.tex"] = render_configs(completed, panels)
    protocols = {key: read_json(directory / "prediction_protocol.json") for key, directory in directories.items()}
    provenance = dict(schema=2, suite_root=str(root), method=manifest["method"], rho=manifest["rho"],
        manifest_sha256=sha(root / "manifest.json"), runtime_hashes=manifest["runtime_hashes"],
        verified_cells=sorted(directories), verified_cell_count=len(directories), final_import=final,
        source_sha256=fingerprints, supersedes=manifest.get("supersedes"),
        suites={name: dict(root=s["root"], manifest_sha256=sha(Path(s["root"]) / "manifest.json"),
                           runtime_hashes=s["manifest"].get("runtime_hashes", s["manifest"].get("source_sha256")),
                           verified_cells=s["verified_cells"], supersedes=s["manifest"].get("supersedes"))
                for name, s in suites.items()},
        seed=0, deterministic_runtime={k: p.get("deterministic_runtime") for k, p in protocols.items()},
        protocols={k: dict(protocol_sha256=fingerprints[str(directories[k] / "prediction_protocol.json")],
                           panel_sha256=sha(directories[k] / "expected_panel.json"),
                           config=panels[k]["config"], vocabulary_sha256=panels[k].get("vocabulary_sha256"),
                           source_sha256=p.get("source_sha256"), seed=p.get("seed", verified.get(k, {}).get("seed")),
                           checkpoint=p.get("checkpoint"), parent_replay=p.get("parent_replay"),
                           control_definitions=p.get("control_definitions"))
                   for k, p in protocols.items()},
        comparisons=comparisons, reported_baselines=reported, replay_checks=replays,
        replay_policy="Seeds 1/2 are deterministic replay checks, not independent samples; no seed SD is computed.",
        shuffle_policy="Fixed permutation seeds 0/1/2, population SD (ddof=0); not sampling uncertainty.",
        cost_policy="Historical cost_snapshot.tex unchanged; shared mechanism timing is not standalone cost.",
        verification_limits="Sealed payload hashes and pooled episode counts are verified; no model or label pixels are loaded. Raw input image identities are bound to sealed protocols, not rehashed by this importer. Completion receipt does not certify reclaimed process exit status. Historical quality_only scores are not imported.",
        output_sha256={n: hashlib.sha256(t.encode()).hexdigest() for n, t in outputs.items()})
    for source_root, source_manifest, verifier in controllers:
        verifier.verify_runtime(source_root, source_manifest)
    if final:
        require(runtime_hashes(b / "runtime") == b_manifest["source_sha256"], "Append sources changed during import")
    for path, digest in fingerprints.items():
        require(sha(path) == digest, f"Source changed during import: {path}")
    outputs["evolution_provenance.json"] = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not check_only:
        publish(paper, outputs)
    return dict(status="ready" if check_only else "imported", check_only=check_only,
                tables_unchanged=check_only, outputs=sorted(outputs), **comparisons), 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--product-suite-root", type=Path)
    parser.add_argument("--voc20-suite-root", type=Path)
    args = parser.parse_args(argv)
    try:
        status, code = run(args.suite_root, args.paper_root, args.check_only,
                           product_suite_root=args.product_suite_root, voc20_suite_root=args.voc20_suite_root)
    except (Invalid, OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        status, code = dict(status="invalid", error=str(error), tables_unchanged=True), 2
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
