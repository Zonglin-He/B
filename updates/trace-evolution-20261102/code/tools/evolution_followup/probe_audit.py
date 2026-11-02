"""Bounded, GT-free descriptive audit of the five frozen A1338 probe caches.

No scientific runner is imported. Only NumPy and the standard library are used.
Cache payload integrity is inherited from linked sealed inventories, not rehashed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import zipfile

import numpy as np


SOURCE = Path("/home/wwww/hzl/results/trace_self_evolution_20261028_1338")
OUTPUT = Path("/home/wwww/hzl/results/trace_probe_complementarity_20261029_0030")
DATASETS = ("coco_cd", "pc60", "voc21", "coco_object", "coco_stuff")
IMAGES = (500, 5105, 1449, 5000, 5000)
EPS = 1e-8
TOL = 1e-6
GAP = 0.05
TV_THRESHOLD = 0.05
MAX_EPISODES = 10000
MAX_CANDIDATES = 100000
MAX_ARRAY_BYTES = 64 * 1024 * 1024
ARRAY_KEYS = {"masks", "image_index", "patch_index", "query_index", "confidence",
              "T", "C", "q", "rho", "donor_candidate_index"}
LIMIT_KEYS = ("T_lt_C", "C_lt_T", "tie", "abs_gap_ge_005",
              "T_lt_C_gap_ge_005", "C_lt_T_gap_ge_005")
CLAIM_BOUNDARY = (
    "Descriptive mechanism characterization of existing development-exposed frozen arms, "
    "not a new independent validation. No causality, independent significance, necessity, "
    "or speed claim. Candidates within episodes are not independent replicates. "
    "All five datasets and both comparison directions are retained; no outcome selection."
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked(root, relative, inventory, bindings):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Seal path escapes root")
    actual = sha256(path)
    require(inventory.get(relative) == actual, f"Seal mismatch: {path}")
    bindings[str(path)] = actual
    return actual


def verify_sources(source):
    """Verify small seals/protocols/source; inspect, but do not hash, cache payloads."""
    source = Path(source).resolve()
    bindings = {}
    suite = read_json(source / "SUITE_SEALED.json")
    bindings[str(source / "SUITE_SEALED.json")] = sha256(source / "SUITE_SEALED.json")
    inventory = suite["artifact_sha256"]
    manifest_hash = checked(source, "manifest.json", inventory, bindings)
    manifest_seal = read_json(source / "MANIFEST_SEALED.json")
    require(manifest_seal["manifest_sha256"] == manifest_hash, "Manifest seal mismatch")
    bindings[str(source / "MANIFEST_SEALED.json")] = sha256(source / "MANIFEST_SEALED.json")
    manifest = read_json(source / "manifest.json")
    checked(source, "queue_state.json", inventory, bindings)
    state = read_json(source / "queue_state.json")
    require(state["status"] == "completed" and state["completed"] == state["total"]
            == suite["total_cells"] == len(manifest["cells"]), "Source suite incomplete")
    runtime = source / "runtime"
    for relative, expected in manifest["runtime_hashes"].items():
        checked(runtime, relative, {relative: expected}, bindings)
    require((runtime / "trace_ovss/method/trace/constants.py").read_text().splitlines()[-1]
            == "EPS = 1e-8", "Frozen EPS differs")
    cells = {}
    for dataset, expected_images in zip(DATASETS, IMAGES):
        prefix = f"cells/mechanism/{dataset}"
        cell = source / prefix
        for name in ("PREDICTIONS_SEALED.json", "SCORED_SEALED.json", "result.json"):
            checked(source, f"{prefix}/{name}", inventory, bindings)
        prediction_hash = bindings[str(cell / "PREDICTIONS_SEALED.json")]
        pred = read_json(cell / "PREDICTIONS_SEALED.json")
        scored = read_json(cell / "SCORED_SEALED.json")
        require(pred.get("complete") is True and pred["images"] == expected_images
                and pred["gt_contents_opened"] is False, "Incomplete prediction seal")
        require(scored["prediction_seal_sha256"] == prediction_hash,
                "Scoring does not link to prediction seal")
        checked(cell, "result.json", scored["artifact_sha256"], bindings)
        artifacts = pred["artifact_sha256"]
        for name in ("prediction_protocol.json", "prediction_result.json", "expected_panel.json",
                     "initial_state.json", "checkpoint_identity.json"):
            checked(cell, name, artifacts, bindings)
        protocol = read_json(cell / "prediction_protocol.json")
        result = read_json(cell / "result.json")
        require(protocol["dataset"] == result["dataset"] == dataset
                and protocol["seed"] == result["seed"] == 0
                and protocol["full_split"] is True and result["full_split"] is True
                and protocol["images"] == len(protocol["inputs"]) == result["images"]
                == expected_images and result["status"] == "completed"
                and result["prediction_seal_sha256"] == prediction_hash,
                "Cell does not match fixed full seed-0 mechanism scope")
        require(protocol["expected_panel_sha256"] == bindings[str(cell / "expected_panel.json")],
                "Expected panel is not bound to protocol")
        panel = read_json(cell / "expected_panel.json")
        require(all(panel[k] == v for k, v in
                    manifest["input_panels"][f"mechanism/{dataset}"].items()),
                "Manifest/expected panel disagreement")
        require(protocol["config"] == panel["config"], "Configuration differs from sealed panel")
        declared = [c for c in manifest["cells"] if c["key"] == f"mechanism/{dataset}"]
        require(len(declared) == 1 and declared[0]["limit"] is None
                and declared[0]["seed"] == 0, "Cell not predeclared in manifest")
        require(Path(protocol["package_root"]).resolve() == runtime, "Unexpected frozen runtime")
        for absolute, expected in protocol["source_sha256"].items():
            path = Path(absolute)
            require(path.is_relative_to(runtime), "Source hash outside frozen runtime")
            relative = str(path.relative_to(runtime))
            require(manifest["runtime_hashes"].get(relative) == expected
                    == bindings.get(str(path)), "Protocol/runtime source disagreement")
        labels = [entry[0] for entry in protocol["variants"]]
        for arm, policy in (("trace", "trace"), ("target", "target"),
                            ("counterfactual", "counterfactual")):
            require([arm, policy, None] in protocol["variants"] and labels.count(arm) == 1,
                    "Missing or ambiguous frozen allocation arm")
        batch = protocol["config"]["batch_size"]
        count = (expected_images + batch - 1) // batch
        require(0 < count <= MAX_EPISODES, "Episode bound exceeded")
        episodes = {f"episodes/{i:06d}.{suffix}": artifacts.get(f"episodes/{i:06d}.{suffix}")
                    for i in range(count) for suffix in ("npz", "json")}
        require(all(isinstance(v, str) and len(v) == 64 for v in episodes.values()),
                "Candidate caches missing from prediction seal")
        actual_npz = {str(p.relative_to(cell)) for p in (cell / "episodes").glob("*.npz")}
        require(actual_npz == {k for k in episodes if k.endswith(".npz")},
                "Episode NPZ population differs from sealed inventory")
        require(all((cell / name).is_file() for name in episodes), "Missing episode receipt")
        cells[dataset] = dict(cell=str(cell), images=expected_images, batch_size=batch,
            episodes=count, budget=protocol["config"]["trace_max_regions"], labels=labels,
            source_seals={name: bindings[str(cell / name)] for name in
                          ("PREDICTIONS_SEALED.json", "SCORED_SEALED.json")},
            sealed_cache_inventory=episodes,
            sealed_cache_inventory_sha256=canonical_hash(episodes))
        print(f"verified linkage: {dataset}: {count} episodes, {expected_images} images", flush=True)
    return dict(source=str(source), cells=cells, verified_file_sha256=bindings,
        verification_scope=("Rehashed suite-linked manifest, queue state, five cell prediction/scoring "
            "seals and results, prediction protocols/panels/state/checkpoint identity metadata, "
            "and all 84 frozen-runtime source files. Checked complete episode NPZ population "
            "and NPZ/receipt coverage in linked sealed cache inventories. Episode receipt hashes "
            "are checked during streaming. No NPZ or prediction payload rehash; no prediction "
            "pixels, checkpoint payload, score-count streams, input images or annotations opened. "
            "Existing inventory hashes bind cache provenance, not a fresh cache-byte integrity proof."))


def probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Expected a finite one-dimensional probability vector")
    effective = np.where(values > EPS, values, 0.0)
    total = float(effective.sum())
    return (effective / total if total > 0 else None), int(np.count_nonzero(effective))


def limiting_counts(target, counterfactual):
    delta = target - counterfactual
    large = np.abs(delta) >= GAP
    t_lower, c_lower = delta < -TOL, delta > TOL
    return dict(T_lt_C=int(t_lower.sum()), C_lt_T=int(c_lower.sum()),
        tie=int((np.abs(delta) <= TOL).sum()), abs_gap_ge_005=int(large.sum()),
        T_lt_C_gap_ge_005=int((t_lower & large).sum()),
        C_lt_T_gap_ge_005=int((c_lower & large).sum()))


def compare_allocations(a, b):
    if a is None or b is None:
        return dict(tv=None, same_rank=None, same_multiset=None, same_assignment=None)
    order = np.argsort(a, kind="stable")
    da, db = np.diff(a[order]), np.diff(b[order])
    # Exact weak ordering includes exact ties, without inventing a rank from near ties.
    same_rank = bool(np.all(db >= 0) and np.array_equal(da == 0, db == 0))
    return dict(tv=float(0.5 * np.abs(a - b).sum()), same_rank=same_rank,
        same_multiset=bool(np.allclose(np.sort(a), np.sort(b), atol=TOL, rtol=0)),
        same_assignment=bool(np.allclose(a, b, atol=TOL, rtol=0)))


def audit_episode(target, counterfactual):
    target = np.asarray(target, dtype=np.float64)
    counterfactual = np.asarray(counterfactual, dtype=np.float64)
    pt, nt = probabilities(target)
    pc, nc = probabilities(counterfactual)
    if target.shape != counterfactual.shape:
        raise ValueError("Probe shapes differ")
    q = np.minimum(target, counterfactual)
    pq, nq = probabilities(q)
    active = q > EPS
    counts = limiting_counts(target, counterfactual)
    active_counts = limiting_counts(target[active], counterfactual[active])
    return dict(candidates=len(q), active=dict(min=nq, T_only=nt, C_only=nc),
        limit_all=counts, limit_min_active=active_counts,
        both_limiters_present_all=counts["T_lt_C"] > 0 and counts["C_lt_T"] > 0,
        both_limiters_present_min_active=(active_counts["T_lt_C"] > 0
                                         and active_counts["C_lt_T"] > 0),
        min_vs_T_only=compare_allocations(pq, pt), min_vs_C_only=compare_allocations(pq, pc))


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def tv_summary(records):
    valid = [r for r in records if r["tv"] is not None]
    values = [r["tv"] for r in valid]
    n = len(valid)
    return dict(eligible_episodes=n, undefined_episodes=len(records) - n,
        mean=float(np.mean(values)) if n else None, median=float(np.median(values)) if n else None,
        count_gt_005=sum(v > TV_THRESHOLD for v in values),
        fraction_gt_005=fraction(sum(v > TV_THRESHOLD for v in values), n),
        same_rank_episodes=sum(r["same_rank"] for r in valid),
        same_multiset_episodes=sum(r["same_multiset"] for r in valid),
        same_assignment_episodes=sum(r["same_assignment"] for r in valid),
        same_rank_but_tv_gt_005=sum(r["same_rank"] and r["tv"] > TV_THRESHOLD for r in valid),
        same_multiset_but_different_assignment=sum(r["same_multiset"] and not r["same_assignment"]
                                                  for r in valid))


def summarize(rows):
    candidates = sum(r["candidates"] for r in rows)
    active = sum(r["active"]["min"] for r in rows)
    counts = {}
    for name, denom in (("limit_all", candidates), ("limit_min_active", active)):
        totals = {k: sum(r[name][k] for r in rows) for k in LIMIT_KEYS}
        counts[name] = dict(denominator=denom, counts=totals,
                            fractions={k: fraction(v, denom) for k, v in totals.items()})
    return dict(episodes=len(rows), candidates=candidates, min_active_candidates=active, **counts,
        active_episode_counts={arm: dict(zero=sum(r["active"][arm] == 0 for r in rows),
            one=sum(r["active"][arm] == 1 for r in rows),
            at_least_two=sum(r["active"][arm] >= 2 for r in rows))
            for arm in ("min", "T_only", "C_only")},
        both_limiters_present_all_episodes=sum(r["both_limiters_present_all"] for r in rows),
        both_limiters_present_min_active_episodes=sum(r["both_limiters_present_min_active"] for r in rows),
        tv={direction: dict(
            all_episodes=tv_summary([r[direction] for r in rows]),
            at_least_two_min_active=tv_summary([r[direction] for r in rows if r["active"]["min"] >= 2]))
            for direction in ("min_vs_T_only", "min_vs_C_only")})


def stream_cell(info, output_handle):
    cell = Path(info["cell"])
    rows = []
    schema = Counter()
    consumed = hashlib.sha256()
    for episode in range(info["episodes"]):
        relative = f"episodes/{episode:06d}.json"
        require(sha256(cell / relative) == info["sealed_cache_inventory"][relative],
                "Episode receipt hash mismatch")
        receipt = read_json(cell / relative)
        start = episode * info["batch_size"]
        expected = list(range(start, min(start + info["batch_size"], info["images"])))
        require(receipt["episode"] == episode and receipt["image_indices"] == expected
                and receipt["dataset_indices"] == expected
                and receipt["policy_labels"] == info["labels"]
                and receipt["candidate_budget"] == info["budget"], "Episode metadata mismatch")
        path = cell / f"episodes/{episode:06d}.npz"
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            require(len(members) == len(ARRAY_KEYS)
                    and {m.filename for m in members} == {k + ".npy" for k in ARRAY_KEYS}
                    and all(m.file_size <= MAX_ARRAY_BYTES for m in members),
                    "NPZ schema or decompressed array size outside bound")
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == ARRAY_KEYS, "NPZ keys changed")
            arrays = {k: archive[k] for k in ("T", "C", "q", "rho")}
        t, c, q, rho = (arrays[k] for k in ("T", "C", "q", "rho"))
        require(q.ndim == 1 and len(q) == receipt["candidate_count"] <= MAX_CANDIDATES
                and len(q) <= len(expected) * info["budget"]
                and t.shape == c.shape == q.shape
                and rho.shape == (len(info["labels"]), len(q)), "Candidate array shape mismatch")
        require(np.array_equal(q, np.minimum(t, c)), "q is not exactly min(T,C)")
        for label, values in (("trace", q), ("target", t), ("counterfactual", c)):
            require(np.array_equal(rho[info["labels"].index(label)], values),
                    f"Saved {label} allocation differs from probe")
        for name, values in arrays.items():
            schema[f"{name}:{values.dtype}"] += 1
            consumed.update(json.dumps([episode, name, str(values.dtype), values.shape]).encode())
            consumed.update(values.tobytes(order="C"))
        row = dict(dataset=cell.name, episode=episode, **audit_episode(t, c))
        output_handle.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
        if (episode + 1) % 250 == 0:
            print(f"streamed {cell.name}: {episode + 1}/{info['episodes']}", flush=True)
    return rows, dict(observed_array_schema=dict(schema),
                      consumed_T_C_q_rho_sha256=consumed.hexdigest())


def render_report(summary):
    lines = ["# Probe Complementarity Audit", "", CLAIM_BOUNDARY, "", "## Scope And Definitions", "",
        "All five A1338 full-split seed-0 mechanism cells; frozen candidates and arms. "
        "No GT or saved score streams used. Optional GT association is not performed.",
        "Candidate comparisons: T<C iff T-C < -1e-6; C<T iff T-C > 1e-6; "
        "otherwise tie. Large gap: |T-C| >= 0.05 (no threshold tuning).",
        "Active means weight > 1e-8. For each episode and each arm independently, "
        "zero inactive weights, divide by the sum of active weights, then compute "
        "TV = 0.5 * sum(abs(p_min-p_arm)) over aligned candidate indices. "
        "Empty distributions have undefined TV (null), never silently zero. "
        "Means/medians are episode-weighted, not candidate-weighted. Primary TV rows "
        "retain every episode with both distributions defined; the >=2 min-active subset is separate.",
        "", "## Candidate Limiting Counts", "",
        "Fractions below pool candidates within each dataset for description only. "
        "The JSON also retains the min-active-only denominator and directional large gaps.", "",
        "| Dataset | Episodes | Candidates | T<C | C<T | Ties | Gap >=.05 | >=2 min-active episodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, stats in summary["datasets"].items():
        s = stats["limit_all"]
        def show(key):
            pct = s["fractions"][key]
            return f"{s['counts'][key]} ({100*pct:.3f}%)" if pct is not None else "0 (null)"
        lines.append(f"| {dataset} | {stats['episodes']} | {stats['candidates']} | "
            f"{show('T_lt_C')} | {show('C_lt_T')} | {show('tie')} | {show('abs_gap_ge_005')} | "
            f"{stats['active_episode_counts']['min']['at_least_two']} |")
    lines += ["", "## Normalized Allocation TV", "",
        "Both directions retained. The last column gives mean/median/fraction>.05 "
        "for the >=2 min-active subset (its eligible n is in summary.json).", "",
        "| Dataset | Comparison | Eligible n | Mean | Median | TV>.05 n (%) | >=2 subset: mean / median / fraction |",
        "|---|---|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "null" if value is None else f"{value:.6f}"
    for dataset, stats in summary["datasets"].items():
        for direction, result in stats["tv"].items():
            a, b = result["all_episodes"], result["at_least_two_min_active"]
            pct = a["fraction_gt_005"]
            pct_text = "null" if pct is None else f"{100*pct:.3f}%"
            lines.append(f"| {dataset} | {direction} | {a['eligible_episodes']} | {fmt(a['mean'])} | "
                f"{fmt(a['median'])} | {a['count_gt_005']} ({pct_text}) | "
                f"{fmt(b['mean'])} / {fmt(b['median'])} / {fmt(b['fraction_gt_005'])} |")
    lines += ["", "## Rank And Multiset Are Not Allocation", "",
        "Same rank means the exact weak ordering of normalized shares, including exact ties, "
        "is unchanged. It does not mean the relative magnitudes are unchanged. Same multiset "
        "compares sorted normalized shares at absolute tolerance 1e-6 (rtol=0); "
        "same assignment compares shares at their original candidate indices with that tolerance. "
        "A permutation can preserve the multiset while changing assignments. These diagnostics "
        "compare min against each single-view arm, not a new permuted-arm outcome evaluation.", "",
        "| Dataset | Comparison | Same rank | Same rank and TV>.05 | Same multiset | Same assignment | Multiset same, assignment different |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for dataset, stats in summary["datasets"].items():
        for direction, result in stats["tv"].items():
            a = result["all_episodes"]
            lines.append(f"| {dataset} | {direction} | {a['same_rank_episodes']} | "
                f"{a['same_rank_but_tv_gt_005']} | {a['same_multiset_episodes']} | "
                f"{a['same_assignment_episodes']} | {a['same_multiset_but_different_assignment']} |")
    lines += ["", "## Empty And Limiter Nulls", ""]
    for dataset, stats in summary["datasets"].items():
        lines.append(f"- {dataset}: active episode bins (zero/one/>=2), "
            f"min/T/C = {json.dumps(stats['active_episode_counts'], sort_keys=True)}; "
            f"both limiting directions present: {stats['both_limiters_present_all_episodes']} all-candidate "
            f"episodes, {stats['both_limiters_present_min_active_episodes']} min-active episodes.")
    lines += ["", "## Provenance And Verification", "", summary["verification_scope"], "",
        "The source inventory lists expected episode NPZ/receipt SHA-256 hashes from linked "
        "prediction seals. Consumed T/C/q/rho bytes are additionally fingerprinted in summary.json; "
        "that new fingerprint is not compared with an old independent fingerprint. "
        "Masks and identity arrays are not decompressed. No all-payload integrity claim is made.", "",
        f"Source: `{summary['source']}`.",
        f"Audit code SHA-256: `{summary['audit_source_sha256']}`.",
        f"Test source SHA-256: `{summary['test_source_sha256']}`.",
        "`source_inventory.json` binds source seals and verified frozen-runtime source hashes. "
        "`episodes.jsonl` retains one descriptive record per episode. "
        "`AUDIT_SEALED.json` hashes all audit outputs and binds the two new source files.", "",
        "No datasets, parameters, metrics or arms were selected using these results. "
        "No model import, inference, GPU work, annotation access or manuscript edit occurred.", ""]
    return "\n".join(lines)


def run(source=SOURCE, output=OUTPUT):
    output = Path(output)
    require(not output.exists(), "Output already exists; never overwrite or resume")
    started = time.monotonic()
    provenance = verify_sources(source)
    script = Path(__file__).resolve()
    test = script.parents[2] / "tests/test_probe_complementarity.py"
    code_hashes = {str(path): sha256(path) for path in (script, test)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source_inventory.json", provenance)
    summary = dict(format_version=1, status="completed", created_at=datetime.now(timezone.utc).isoformat(),
        source=str(Path(source).resolve()), claim_boundary=CLAIM_BOUNDARY,
        definitions=dict(eps_admission_strict_gt=EPS, comparison_absolute_tolerance=TOL,
            abs_gap_threshold_inclusive=GAP, tv_threshold_strict_gt=TV_THRESHOLD,
            undefined_tv="null if either distribution has no admitted weight",
            rank="exact weak ordering including exact ties",
            allocation_equal="absolute tolerance 1e-6, rtol=0",
            aggregation="episode-weighted TV; candidate-pooled limiting counts per dataset"),
        audit_source_sha256=code_hashes[str(script)], test_source_sha256=code_hashes[str(test)],
        verification_scope=provenance["verification_scope"],
        execution=dict(python=platform.python_version(), numpy=np.__version__, cpu_only=True,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gt_opened=False, score_streams_opened=False, selected_datasets=list(DATASETS)),
        datasets={})
    with (output / "episodes.jsonl").open("x") as handle:
        for dataset in DATASETS:
            rows, observed = stream_cell(provenance["cells"][dataset], handle)
            summary["datasets"][dataset] = dict(**summarize(rows), **observed,
                images=provenance["cells"][dataset]["images"])
            print(f"completed {dataset}: {len(rows)} episodes", flush=True)
    require("torch" not in sys.modules and "PIL.Image" not in sys.modules,
            "Unexpected model/image dependency imported")
    require(all(sha256(path) == expected for path, expected in code_hashes.items()),
            "Audit/test source changed during execution")
    require(all(sha256(path) == expected for path, expected in
                provenance["verified_file_sha256"].items()), "Verified source changed during audit")
    summary["execution"].update(elapsed_seconds=time.monotonic() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_modules_imported=False, post_run_verified_sources_unchanged=True)
    write_json(output / "summary.json", summary)
    with (output / "RESULTS.md").open("x") as handle:
        handle.write(render_report(summary))
    write_json(output / "AUDIT_SEALED.json", dict(format_version=1, status="completed",
        created_at=datetime.now(timezone.utc).isoformat(), source_sha256=code_hashes,
        source_suite_seal_sha256=provenance["verified_file_sha256"][str(Path(source).resolve() / "SUITE_SEALED.json")],
        artifact_sha256={name: sha256(output / name) for name in
                        ("source_inventory.json", "episodes.jsonl", "summary.json", "RESULTS.md")},
        verification_scope=provenance["verification_scope"], claim_boundary=CLAIM_BOUNDARY))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Read-only seal linkage check; no outputs")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_sources(SOURCE)
        print(json.dumps(dict(status="verified", verification_scope=result["verification_scope"],
            episodes={d: c["episodes"] for d, c in result["cells"].items()}), indent=2))
    else:
        result = run()
        print(json.dumps(dict(status=result["status"], output=str(OUTPUT),
                              execution=result["execution"]), indent=2))


if __name__ == "__main__":
    main()
