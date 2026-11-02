"""CPU-only, record-backed TRACE response analysis; never imports the runtime.

Run with --help. Outputs are CSV and JSON, in a fresh directory only. Undefined
ratios are null, not zero. All deltas are Full minus the matched removal control.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys

from analyze_voc21_tuning import (EPS, Evidence, count_vector,
    digest, distribution, normalized_weights, require, spearman, total_variation)


DATASETS = {
    "voc21": ("PascalVOC21Dataset", "cls_voc21.txt", 21),
    "pc60": ("PascalContext60Dataset", "cls_context60.txt", 60),
    "coco_object": ("COCOObjectDataset", "cls_coco_object.txt", 81),
    "coco_cd": ("COCOCDDataset", "coco_cd.txt", 3),
}
VARIANTS = ("full", "no_connected_regions", "no_source_tracing",
            "no_reliability_weighting", "no_region_normalization")
IDENTITY = ("dataset", "split", "image_ids", "label_ids", "vocabulary_sha256",
            "seed", "selection_seed", "available_images", "config")
KEY = ("episode", "image_id", "patch_in_batch", "class_idx", "mask_sha256")
SUPPORT = ("target_support", "residual_support", "counterfactual_support", "quality", "dominance")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
DEFAULT_SUITE = Path("/home/wwww/hzl/results/trace_removal_suite_20261025_0730")
DEFAULT_VOC = Path("/home/wwww/hzl/results/voc21_batch_refine_20261025_0820/analysis")
DEFAULT_OUTPUT = Path("/home/wwww/hzl/results/trace_efficiency_diagnosis_20261025/dataset_analysis")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate_counts(records, classes):
    intersection, union = [0] * classes, [0] * classes
    for row in records:
        a = count_vector(row["intersection"], classes, "intersection")
        b = count_vector(row["union"], classes, "union")
        require(all(i <= u for i, u in zip(a, b)), "Intersection exceeds union")
        intersection = [x + y for x, y in zip(intersection, a)]
        union = [x + y for x, y in zip(union, b)]
    iou = [ratio(100 * i, u) for i, u in zip(intersection, union)]
    defined = [value for value in iou if value is not None]
    return intersection, union, iou, ratio(math.fsum(defined), len(defined))


def close(a, b, message, tolerance=1e-9):
    require(a is not None and b is not None and math.isfinite(float(a))
            and math.isfinite(float(b)) and math.isclose(float(a), float(b),
            rel_tol=0, abs_tol=tolerance), message)


def read_csv(evidence, path):
    return list(csv.DictReader(io.StringIO(evidence.read(path).decode("utf-8"))))


def integer(value, name, minimum=0):
    number = float(value)
    require(math.isfinite(number) and number.is_integer() and number >= minimum,
            f"Invalid {name}: {value}")
    return int(number)


def boolean(value):
    require(value in ("True", "False"), f"Invalid Boolean: {value}")
    return value == "True"


def parse_vocabulary(text, extensions, expected_classes):
    groups = [[part.strip() for part in line.split(",")]
              for line in text.splitlines() if line.strip()]
    require(len(groups) == expected_classes and all(all(group) for group in groups),
            "Invalid vocabulary class axis")
    classes = [group[0] for group in groups]
    classes[0] = "background"
    owners = {}
    for index, group in enumerate(groups):
        for query in group:
            require(owners.get(query.casefold(), index) == index, "Ambiguous query owner")
            owners[query.casefold()] = index
    queries = [query for group in groups for query in group] if extensions else classes
    mapping = [i for i, group in enumerate(groups) for _ in group] if extensions else None
    hashed = hashlib.sha256(json.dumps([queries, mapping]).encode()).hexdigest()
    return dict(classes=classes, queries=queries, query_to_class=(mapping if mapping is not None
                else list(range(expected_classes))), vocabulary_sha256=hashed)


def verify_identity(first, other):
    for field in IDENTITY:
        require(first[field] == other[field], f"Matched identity mismatch: {field}")
    for field in ("images", "full_split", "force_uniform_policy"):
        if field in first and field in other:
            require(first[field] == other[field], f"Matched identity mismatch: {field}")
    if "file_metadata" in first and "file_metadata" in other:
        require(first["file_metadata"] == other["file_metadata"], "Matched file_metadata mismatch")


def worker_semantics(source):
    """Allow only the audited worker's environment/file-metadata instrumentation."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == "execution_environment")]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            node.names = [name for name in node.names if name.name not in {"os", "subprocess"}]
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "protocol"
                                                for t in node.targets)):
            require(isinstance(node.value, ast.Call), "Unexpected protocol constructor")
            node.value.keywords = [k for k in node.value.keywords
                                   if k.arg not in {"environment", "file_metadata"}]
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import) or node.names]
    return ast.dump(tree, include_attributes=False)


def verify_implementation(first, other, evidence):
    require(first["source_hashes"] == other["source_hashes"], "Production source mismatch")
    a, b = first["experiment_hashes"], other["experiment_hashes"]
    require(a.keys() == b.keys(), "Experiment source set mismatch")
    changed = [name for name in a if a[name] != b[name]]
    require(set(changed) <= {"benchmark_simplification.py"}, "Ablation implementation mismatch")
    if changed:
        sources = [evidence.read(Path(p["package_root"]) / "tools/benchmark_simplification.py")
                   for p in (first, other)]
        require(worker_semantics(sources[0]) == worker_semantics(sources[1]),
                "Worker differences exceed environment/file-metadata instrumentation")
    for field in ("python", "torch", "cuda", "gpu"):
        require(first["environment"][field] == other["environment"][field],
                f"Minimal execution environment mismatch: {field}")
    return "instrumentation_only" if changed else "identical"


@dataclass
class Cell:
    path: Path
    protocol: dict
    result: dict
    records: list
    vocabulary: dict
    candidates: dict
    costs: list
    miou: float
    iou: list
    stage: str

    @property
    def context(self):
        p = self.protocol
        return dict(cell_id=f'{self.stage}/{p["dataset"]}/{p["variant"]}', dataset=p["dataset"],
                    stage=self.stage, variant=p["variant"], images=p["images"],
                    full_split=p["full_split"], vocabulary_sha256=p["vocabulary_sha256"],
                    result_path=str(self.path / "result.json"))


def validate_candidates(rows, protocol, vocabulary):
    indexed, patch_owners, by_image = {}, {}, defaultdict(int)
    batch = protocol["config"]["batch_size"]
    image_episodes = {image: index // batch for index, image in enumerate(protocol["image_ids"])}
    image_slots = {image: index % batch for index, image in enumerate(protocol["image_ids"])}
    config = protocol["config"]
    one_patch = (config.get("resize") == config.get("patch_size")
                 and config.get("resize") is not None and len(set(config["resize"])) == 1)
    for source_row, original in enumerate(rows, 2):
        row = dict(original)
        for field in ("episode", "patch_in_batch", "class_idx", "area"):
            row[field] = integer(row[field], field, 1 if field == "area" else 0)
        require(image_episodes.get(row["image_id"]) == row["episode"], "Candidate image/episode mismatch")
        if one_patch:
            require(row["patch_in_batch"] == image_slots[row["image_id"]], "One-patch image slot mismatch")
        require(row["class_idx"] < len(vocabulary["queries"]), "Candidate query index outside vocabulary")
        row["evaluation_class_idx"] = vocabulary["query_to_class"][row["class_idx"]]
        patch = row["episode"], row["patch_in_batch"]
        require(patch_owners.get(patch, row["image_id"]) == row["image_id"], "Patch belongs to two images")
        patch_owners[patch] = row["image_id"]
        require(len(row["mask_sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["mask_sha256"]),
                "Invalid candidate mask hash")
        for field in (*SUPPORT, "reliability"):
            row[field] = float(row[field]) if row[field] != "" else None
            require(row[field] is None or math.isfinite(row[field]) and 0 <= row[field] <= 1,
                    f"Invalid candidate {field}")
        require(row["reliability"] is not None, "Missing candidate reliability")
        if protocol["variant"] == "no_source_tracing":
            require(all(row[k] is None for k in SUPPORT), "Raw control unexpectedly contains traced scores")
        else:
            require(all(row[k] is not None for k in SUPPORT), "Missing traced support")
            q = min(row["target_support"], row["counterfactual_support"])
            d = max(row["residual_support"] - row["target_support"], 0)
            rho = 1.0 if protocol["variant"] == "no_reliability_weighting" else q * (1 - d)
            for field, expected in (("quality", q), ("dominance", d), ("reliability", rho)):
                close(row[field], expected, f"Candidate formula mismatch: {field}", 1e-12)
        key = tuple(row[k] for k in KEY)
        require(key not in indexed, "Duplicate candidate key")
        row["source_csv_row"] = source_row
        indexed[key] = row
        by_image[row["image_id"]] += 1
    require(all(n <= protocol["config"]["trace_max_regions"] for n in by_image.values()),
            "Per-image candidate budget exceeded")
    return indexed


def validate_costs(rows, candidates, protocol, result):
    costs = []
    batch = protocol["config"]["batch_size"]
    expected_episodes = math.ceil(protocol["images"] / batch)
    require(len(rows) == expected_episodes, "Missing cost episode")
    grouped = defaultdict(list)
    for row in candidates.values():
        grouped[row["episode"]].append(row)
    for episode, source in enumerate(rows):
        row = dict(source)
        for key in ("episode", "images", "update_steps", "candidates", "diagnostic_views",
                    "effective_view_batch_size", "calls", "inputs"):
            row[key] = integer(row[key], key)
        for key in ("adaptation_seconds", "inference_seconds", "total_seconds", "peak_allocated_mib"):
            row[key] = float(row[key])
            require(math.isfinite(row[key]) and row[key] >= 0, f"Invalid cost {key}")
        require(row["episode"] == episode, "Cost episode order mismatch")
        require(row["images"] == min(batch, protocol["images"] - episode * batch), "Cost image count mismatch")
        require(row["candidates"] == len(grouped[episode]), "Cost candidate count mismatch")
        multiplier = 0 if protocol["variant"] == "no_source_tracing" else 3
        require(row["diagnostic_views"] == multiplier * row["candidates"], "Three-view count mismatch")
        close(row["total_seconds"], row["adaptation_seconds"] + row["inference_seconds"], "Cost seconds mismatch")
        costs.append(row)
    warmup = integer(result["warmup_episodes"], "warmup_episodes")
    require(warmup == min(2, expected_episodes - 1), "Unexpected warmup count")
    measured = costs[warmup:]
    n = sum(r["images"] for r in measured)
    close(result["ms_per_image"], 1000 * math.fsum(r["total_seconds"] for r in measured) / n,
          "Measured ms/image mismatch")
    close(result["diagnostic_views_per_image"], sum(r["diagnostic_views"] for r in measured) / n,
          "Measured views/image mismatch")
    close(result["peak_allocated_mib"], max(r["peak_allocated_mib"] for r in measured),
          "Measured peak allocation mismatch")
    return costs


def load_cell(table_row, manifest, evidence):
    path = Path(table_row["result_path"]).resolve().parent
    p = evidence.json(path / "protocol.json")
    r = evidence.json(path / "result.json")
    require(r["status"] == "completed", "Incomplete cell")
    require(r["protocol_sha256"] == evidence.hashes[str(path / "protocol.json")], "Protocol hash mismatch")
    ds, variant = table_row["dataset"], table_row["variant"]
    require(ds in DATASETS and variant in VARIANTS, "Unexpected dataset/variant")
    class_type, filename, classes = DATASETS[ds]
    require(p["config"]["dataset"] == class_type, "Wrong evaluation class space")
    for field, value in (("dataset", ds), ("variant", variant), ("split", "evaluation")):
        require(p[field] == r[field] == value, f"Result/protocol/table mismatch: {field}")
    require(p["force_uniform_policy"] is False, "Unexpected candidate policy override")
    records = evidence.json(path / "per_image_counts.json")
    n = integer(table_row["images"], "images", 1)
    require(p["images"] == r["images"] == len(records) == n, "Cohort count mismatch")
    require(p["full_split"] is r["full_split"] is boolean(table_row["full_split"]), "Full-split identity mismatch")
    require(p["full_split"] == (n == p["available_images"]), "Incorrect full-split flag")
    require(p["full_split"] or n == 512, "Not a full or 512-image cohort")
    for field in ("image_ids", "label_ids"):
        require(len(p[field]) == len(set(p[field])) == n, f"Duplicate or missing {field}")
    require([row["image_id"] for row in records] == p["image_ids"], "Per-image order mismatch")
    panel = f'{ds}/{None if p["full_split"] else n}'
    verify_identity(p, manifest["input_panels"][panel])
    require(p["source_hashes"] == manifest["source_hashes"], "Suite source mismatch")
    root = Path(p["package_root"]).resolve()
    actual_inventory = {f.relative_to(root / "trace_ovss").as_posix()
                        for f in (root / "trace_ovss").rglob("*")
                        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"}
    require(actual_inventory == p["source_hashes"].keys(), "Frozen source inventory mismatch")
    require(set(p["experiment_hashes"]) == {"benchmark_simplification.py", "trace_ablation_variants.py"},
            "Unexpected experiment source inventory")
    for subdir, hashes in (("trace_ovss", p["source_hashes"]), ("tools", p["experiment_hashes"])):
        require(bool(hashes), "Missing frozen source hashes")
        for relative, hashed in hashes.items():
            source = (root / subdir / relative).resolve()
            require(source.is_relative_to(root / subdir), "Unsafe source path")
            if str(source) not in evidence.hashes:
                evidence.read(source)
            require(evidence.hashes[str(source)] == hashed, f"Frozen source changed: {source}")
    resource = root / "trace_ovss/resources/class_names" / filename
    vocabulary = parse_vocabulary(evidence.read(resource).decode(), p["config"]["class_extensions"], classes)
    require(vocabulary["vocabulary_sha256"] == p["vocabulary_sha256"], "Vocabulary hash mismatch")
    intersection, union, iou, miou = aggregate_counts(records, classes)
    require(intersection == count_vector(r["intersection"], classes, "result intersection")
            and union == count_vector(r["union"], classes, "result union"), "Aggregate counts mismatch")
    close(miou, r["miou_unrounded"], "Result mIoU mismatch")
    close(miou, table_row["mIoU"], "Summary mIoU mismatch")
    close(round(miou, 2), r["metrics"]["mIoU"], "Rounded mIoU mismatch", .011)
    for field in ("ms_per_image", "diagnostic_views_per_image", "peak_allocated_mib"):
        close(r[field], table_row[field], f"Summary {field} mismatch")
    cost_rows = read_csv(evidence, path / "cost.csv")
    candidate_path = path / "candidate_diagnostics.csv"
    if candidate_path.exists():
        candidate_rows = read_csv(evidence, candidate_path)
    else:
        require(all(integer(row["candidates"], "candidates") == 0 for row in cost_rows),
                "Missing candidate diagnostics")
        candidate_rows = []
    candidates = validate_candidates(candidate_rows, p, vocabulary)
    costs = validate_costs(cost_rows, candidates, p, r)
    return Cell(path, p, r, records, vocabulary, candidates, costs, miou, iou, table_row["stage"])


def score_stats(rows):
    """Candidate-weighted summaries, explicitly distinct from image/episode means."""
    rows = list(rows)
    result = dict(candidates=len(rows), area_tokens_sum=sum(r["area"] for r in rows))
    for field in ("area", *SUPPORT, "reliability"):
        result.update({f"{field}_{key}": value for key, value in distribution([r[field] for r in rows]).items()})
    traced = [r for r in rows if r["quality"] is not None]
    n = len(traced)
    for field, predicate in (
        ("dominance_positive", lambda r: r["dominance"] > 0),
        ("dominance_at_least_half", lambda r: r["dominance"] >= .5),
        ("dominance_changes_nominal_score", lambda r: r["quality"] * r["dominance"] > 0),
        ("target_is_strict_min", lambda r: r["target_support"] < r["counterfactual_support"]),
        ("counterfactual_is_strict_min", lambda r: r["counterfactual_support"] < r["target_support"]),
        ("min_tie", lambda r: r["target_support"] == r["counterfactual_support"])):
        count = sum(predicate(r) for r in traced)
        result[f"{field}_count"] = count
        result[f"{field}_fraction"] = ratio(count, n)
    result["traced_candidates"] = n
    result["q_weighted_dominance_penalty_fraction"] = ratio(
        math.fsum(r["quality"] * r["dominance"] for r in traced), math.fsum(r["quality"] for r in traced))
    attenuation = [ratio(r["quality"], max(r["target_support"], r["counterfactual_support"])) for r in traced]
    result.update({f"min_over_max_support_{k}": v for k, v in distribution(attenuation).items()})
    for threshold, label in ((EPS, "eps"), (.01, "0_01"), (.05, "0_05"), (.1, "0_1"), (.5, "0_5")):
        count = sum(r["reliability"] <= threshold for r in rows)
        result[f"reliability_le_{label}_count"] = count
        result[f"reliability_le_{label}_fraction"] = ratio(count, len(rows))
    return result


def grouped_candidates(cell, field):
    groups = defaultdict(list)
    for row in cell.candidates.values():
        groups[row[field]].append(row)
    return groups


def candidate_tables(cell):
    p, ctx = cell.protocol, cell.context
    by_episode = grouped_candidates(cell, "episode")
    by_image = grouped_candidates(cell, "image_id")
    measured = cell.costs[cell.result["warmup_episodes"]:]
    summary = dict(**ctx, **score_stats(cell.candidates.values()), evaluation_classes=len(cell.iou),
        active_queries=len(cell.vocabulary["queries"]), episodes=len(cell.costs),
        image_batch_size=p["config"]["batch_size"], view_batch_size=p["config"]["trace_view_batch_size"],
        candidate_budget_per_image=p["config"]["trace_max_regions"], lr=p["config"]["lr"], steps=p["config"]["steps"],
        candidates_per_image=len(cell.candidates) / p["images"],
        candidates_per_episode=len(cell.candidates) / len(cell.costs),
        images_with_candidates=len(by_image), image_candidate_coverage=len(by_image) / p["images"],
        episodes_with_candidates=len(by_episode), episode_candidate_coverage=len(by_episode) / len(cell.costs),
        evaluation_classes_with_candidates=len({r["evaluation_class_idx"] for r in cell.candidates.values()}),
        selected_area_tokens_per_image=sum(r["area"] for r in cell.candidates.values()) / p["images"],
        images_reaching_candidate_budget=sum(len(by_image[i]) == p["config"]["trace_max_regions"] for i in p["image_ids"]),
        all_episode_diagnostic_views=sum(r["diagnostic_views"] for r in cell.costs),
        all_episode_views_per_image=sum(r["diagnostic_views"] for r in cell.costs) / p["images"],
        measured_images=sum(r["images"] for r in measured), warmup_episodes=cell.result["warmup_episodes"],
        measured_views_per_image=cell.result["diagnostic_views_per_image"],
        measured_ms_per_image=cell.result["ms_per_image"],
        measured_adaptation_seconds=math.fsum(r["adaptation_seconds"] for r in measured),
        measured_inference_seconds=math.fsum(r["inference_seconds"] for r in measured),
        true_pixel_coverage=None, gt_candidate_precision=None,
        cost_scope="warmup-excluded observational profile; no controlled speedup claim")
    episodes = [dict(**ctx, **{k: v for k, v in cost.items() if k not in {"images", "candidates"}},
                     episode_images=cost["images"], **score_stats(by_episode[cost["episode"]]),
                     warmup=cost["episode"] < cell.result["warmup_episodes"])
                for cost in cell.costs]
    images = [dict(**ctx, image_id=identifier, image_index=i, episode=i // p["config"]["batch_size"],
                   **score_stats(by_image[identifier])) for i, identifier in enumerate(p["image_ids"])]
    classes = []
    by_class = grouped_candidates(cell, "evaluation_class_idx")
    for i, name in enumerate(cell.vocabulary["classes"]):
        rows = by_class[i]
        classes.append(dict(**ctx, class_idx=i, class_name=name, **score_stats(rows),
            images_with_candidates=len({r["image_id"] for r in rows}),
            episodes_with_candidates=len({r["episode"] for r in rows}),
            candidate_fraction=len(rows) / len(cell.candidates) if cell.candidates else None))
    return summary, episodes, images, classes


def paired_candidates(full, control, *, same_support):
    require(full.candidates.keys() == control.candidates.keys(), "Candidate identities differ")
    for key, row in full.candidates.items():
        other = control.candidates[key]
        require(row["area"] == other["area"], "Candidate area mismatch")
        if same_support:
            for field in SUPPORT:
                close(row[field], other[field], f"Paired candidate support mismatch: {field}", 1e-12)


def nominal_allocations(rows, raw):
    """Real-valued, detached coefficients on region means, not GPU gradients."""
    rho = [r["reliability"] for r in rows]
    return {
        "full": normalized_weights(rho),
        "raw_confidence": normalized_weights(raw),
        # The estimator removes rho <= EPS before the objective multiplies by area.
        "no_region_normalization": normalized_weights([
            r["reliability"] * r["area"] if r["reliability"] > EPS else 0.0 for r in rows]),
        "quality_only": normalized_weights([r["quality"] for r in rows]),
        "target_only": normalized_weights([r["target_support"] for r in rows]),
        "counterfactual_only": normalized_weights([r["counterfactual_support"] for r in rows]),
    }


def allocation_tables(full, raw):
    paired_candidates(full, raw, same_support=False)
    groups = grouped_candidates(full, "episode")
    episode_rows, episode_classes = [], []
    class_mass, image_mass = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
    for cost in full.costs:
        ep = cost["episode"]
        rows = groups[ep]
        confidence = [raw.candidates[tuple(r[k] for k in KEY)]["reliability"] for r in rows]
        weights = nominal_allocations(rows, confidence)
        episode_class_mass = defaultdict(lambda: defaultdict(float))
        record = dict(**full.context, episode=ep, episode_images=cost["images"], candidates=len(rows),
                      rho_conf_spearman=spearman([r["reliability"] for r in rows], confidence))
        for mode, values in weights.items():
            record[f"{mode}_coefficient_sum"] = math.fsum(values)
            record[f"{mode}_effective_candidates"] = ratio(1, math.fsum(v * v for v in values))
            record[f"{mode}_max_coefficient"] = max(values) if values else None
            for row, value in zip(rows, values):
                class_mass[row["evaluation_class_idx"]][mode] += value
                episode_class_mass[row["evaluation_class_idx"]][mode] += value
                image_mass[row["image_id"]][mode] += value
        for name, first, second in (
            ("full_vs_raw", "full", "raw_confidence"),
            ("dominance_only", "full", "quality_only"),
            ("region_normalization", "full", "no_region_normalization"),
            ("min_vs_target", "quality_only", "target_only"),
            ("min_vs_counterfactual", "quality_only", "counterfactual_only")):
            record[f"{name}_tv"] = total_variation(weights[first], weights[second])
        episode_rows.append(record)
        for index, name in enumerate(full.vocabulary["classes"]):
            episode_classes.append(dict(**full.context, episode=ep, episode_images=cost["images"],
                class_idx=index, class_name=name,
                **{f"{mode}_episode_coefficient_mass": episode_class_mass[index][mode] for mode in weights}))
    modes = tuple(nominal_allocations([], []))
    class_rows = [dict(**full.context, class_idx=i, class_name=name,
        **{f"{mode}_mean_episode_coefficient_mass": class_mass[i][mode] / len(full.costs) for mode in modes})
        for i, name in enumerate(full.vocabulary["classes"])]
    image_rows = [dict(**full.context, image_id=identifier, image_index=i,
        episode=i // full.protocol["config"]["batch_size"],
        **{f"{mode}_episode_coefficient_mass": image_mass[identifier][mode] for mode in modes})
        for i, identifier in enumerate(full.protocol["image_ids"])]
    return episode_rows, class_rows, image_rows, episode_classes


def delta(a, b):
    return a - b if a is not None and b is not None else None


def class_comparison(full, control, records_a=None, records_b=None):
    if records_a is None:
        records_a, records_b = full.records, control.records
    a = aggregate_counts(records_a, len(full.iou))
    b = aggregate_counts(records_b, len(full.iou))
    return [dict(**full.context, comparator=control.protocol["variant"],
        control_result_path=str(control.path / "result.json"), class_idx=i, class_name=name,
        full_intersection=a[0][i], full_union=a[1][i], control_intersection=b[0][i], control_union=b[1][i],
        full_images_with_nonzero_union=sum(r["union"][i] > 0 for r in records_a),
        control_images_with_nonzero_union=sum(r["union"][i] > 0 for r in records_b),
        full_iou_pct=a[2][i], control_iou_pct=b[2][i], delta_full_minus_control_pp=delta(a[2][i], b[2][i]))
        for i, name in enumerate(full.vocabulary["classes"])]


def image_comparison(full, control):
    rows = []
    for index, (a, b) in enumerate(zip(full.records, control.records)):
        require(a["image_id"] == b["image_id"], "Image pairing mismatch")
        ia = ratio(100 * sum(a["intersection"]), sum(a["union"]))
        ib = ratio(100 * sum(b["intersection"]), sum(b["union"]))
        rows.append(dict(**full.context, comparator=control.protocol["variant"],
            image_id=a["image_id"], image_index=index, episode=index // full.protocol["config"]["batch_size"],
            full_micro_iou_pct=ia, control_micro_iou_pct=ib, delta_micro_iou_pp=delta(ia, ib),
            identical_prediction=a["prediction_sha256"] == b["prediction_sha256"],
            full_defined_classes=sum(u > 0 for u in a["union"]),
            control_defined_classes=sum(u > 0 for u in b["union"]),
            control_result_path=str(control.path / "result.json")))
    return rows


def class_responses(deltas, allocations, candidates):
    allocation_index = {(r["dataset"], r["class_idx"]): r for r in allocations}
    candidate_index = {(r["dataset"], r["class_idx"]): r for r in candidates
                       if r["stage"] == "ablation" and r["variant"] == "full"}
    rows = []
    for row in deltas:
        comparator = row["comparator"]
        if comparator not in ("no_source_tracing", "no_region_normalization"):
            continue
        key = row["dataset"], row["class_idx"]
        allocation, candidate = allocation_index[key], candidate_index[key]
        mode = "raw_confidence" if comparator == "no_source_tracing" else comparator
        full_mass = allocation["full_mean_episode_coefficient_mass"]
        control_mass = allocation[f"{mode}_mean_episode_coefficient_mass"]
        rows.append(dict(**row, full_mean_episode_coefficient_mass=full_mass,
            control_mean_episode_coefficient_mass=control_mass,
            delta_full_minus_control_coefficient_mass=full_mass - control_mass,
            coefficient_mass_delta_pp=100 * (full_mass - control_mass),
            **{field: candidate[field] for field in ("candidates", "candidate_fraction", "area_tokens_sum",
                "images_with_candidates", "episodes_with_candidates", "quality_mean", "reliability_mean",
                "dominance_positive_fraction", "target_is_strict_min_fraction", "counterfactual_is_strict_min_fraction")},
            interpretation="side-by-side different estimands: pooled class IoU and equal-episode coefficient mass; "
                           "not an estimated association, not gradient or causal attribution"))
    return rows


def verify_prefix_source(protocol, evidence):
    source = Path(protocol["package_root"]) / "trace_ovss/method/trace/candidates.py"
    tree = ast.parse(evidence.read(source))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "collect_trace_candidates")
    sorts = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "sort" and isinstance(n.func.value, ast.Name) and n.func.value.id == "candidates"]
    expected = ast.parse('lambda item: (-item["confidence"], -item["area"], item["class_idx"], item["component_idx"])', mode="eval").body
    require(len(sorts) == 1 and len(sorts[0].keywords) == 1 and sorts[0].keywords[0].arg == "key"
            and ast.dump(sorts[0].keywords[0].value) == ast.dump(expected), "Unsupported production candidate ranking")
    prefix = ast.parse("selected.extend(candidates[:self.trace_max_regions])", mode="eval").body
    require(any(isinstance(n, ast.Call) and ast.dump(n) == ast.dump(prefix) for n in ast.walk(method)),
            "Production selector does not take a per-image prefix")
    return str(source)


def k_prefix_budget(full, raw, source_path):
    """Cached prefix arithmetic; never rerank by rho or claim new-run equivalence."""
    paired_candidates(full, raw, same_support=False)
    require(list(full.candidates) == list(raw.candidates), "Prefix requires identical saved candidate order")
    images = grouped_candidates(full, "image_id")
    all_rows = list(full.candidates.values())
    for rows in images.values():
        sort_keys = [(-raw.candidates[tuple(r[k] for k in KEY)]["reliability"], -r["area"], r["class_idx"])
                     for r in rows]
        require(sort_keys == sorted(sort_keys), "Cached per-image raw confidence/ranking is not nonincreasing")
    rho_total = math.fsum(r["reliability"] for r in all_rows)
    active_total = math.fsum(r["reliability"] for r in all_rows if r["reliability"] > EPS)
    class_set = {r["evaluation_class_idx"] for r in all_rows}
    image_classes = {(r["image_id"], r["evaluation_class_idx"]) for r in all_rows}
    episode_classes = {(r["episode"], r["evaluation_class_idx"]) for r in all_rows}
    episode_mass = defaultdict(float)
    for r in all_rows:
        if r["reliability"] > EPS:
            episode_mass[r["episode"]] += r["reliability"]
    current = full.protocol["config"]["trace_max_regions"]
    results = []
    for label, requested in (("K4", 4), ("K8", 8), ("K16", 16), ("current", current)):
        effective = min(requested, current)
        kept = [r for image in full.protocol["image_ids"] for r in images[image][:effective]]
        retained = math.fsum(r["reliability"] for r in kept)
        retained_active = math.fsum(r["reliability"] for r in kept if r["reliability"] > EPS)
        kept_classes = {r["evaluation_class_idx"] for r in kept}
        kept_episode_mass = defaultdict(float)
        for r in kept:
            if r["reliability"] > EPS:
                kept_episode_mass[r["episode"]] += r["reliability"]
        episode_ratios = [ratio(kept_episode_mass[c["episode"]], episode_mass[c["episode"]]) for c in full.costs]
        results.append(dict(**full.context, prefix=label, requested_k=requested, cached_current_k=current,
            effective_cached_k=effective, requested_exceeds_cache=requested > current,
            cached_candidates=len(all_rows), retained_candidates=len(kept),
            retained_candidate_fraction=ratio(len(kept), len(all_rows)),
            retained_candidates_per_image=len(kept) / full.protocol["images"],
            cached_rho_mass=rho_total, retained_rho_mass=retained, retained_rho_fraction=ratio(retained, rho_total),
            retained_active_rho_fraction=ratio(retained_active, active_total),
            mean_episode_retained_full_coefficient_mass=distribution(episode_ratios)["mean"],
            episodes_with_defined_rho_fraction=distribution(episode_ratios)["n"],
            cached_classes=len(class_set), retained_classes=len(kept_classes),
            retained_class_fraction=ratio(len(kept_classes), len(class_set)), evaluation_classes=len(full.iou),
            retained_evaluation_class_indices=json.dumps(sorted(kept_classes)),
            lost_cached_class_names=json.dumps([full.vocabulary["classes"][i] for i in sorted(class_set - kept_classes)]),
            cached_image_class_pairs=len(image_classes),
            retained_image_class_pairs=len({(r["image_id"], r["evaluation_class_idx"]) for r in kept}),
            cached_episode_class_pairs=len(episode_classes),
            retained_episode_class_pairs=len({(r["episode"], r["evaluation_class_idx"]) for r in kept}),
            cached_diagnostic_views=3 * len(all_rows), retained_diagnostic_views=3 * len(kept),
            saved_diagnostic_views=3 * (len(all_rows) - len(kept)),
            diagnostic_view_reduction_fraction=ratio(len(all_rows) - len(kept), len(all_rows)),
            source_order_verified=True, raw_confidence_nonincreasing=True, order_source_path=source_path,
            raw_confidence_source_path=str(raw.path / "candidate_diagnostics.csv"),
            scope="cached approximation; original CSV tie order preserved, component_idx not recorded; "
                  "K above cache is capped, not simulated; no FP16 re-encoding, batched-forward or adaptation-trajectory equivalence; no GT/mIoU inference"))
    return results


def write_csv(path, rows):
    require(bool(rows), f"Empty output table: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_supplied_tables(audit, rows, cells, candidate_rows, evidence):
    checks = []
    selected = {(r["dataset"], r["variant"]): r for r in rows if r["stage"] == "ablation"}
    ablations = read_csv(evidence, audit / "ablation_summary.csv")
    require(len(ablations) == len(selected), "Ablation table length mismatch")
    seen = set()
    for row in ablations:
        key = row["dataset"], row["variant"]
        require(key not in seen and key in selected, "Duplicate/unknown ablation table row")
        seen.add(key)
        for field in selected[key]:
            require(row[field] == selected[key][field], f"Ablation table field mismatch: {field}")
        full = cells[("ablation", key[0], "full")]
        cell = cells[("ablation", *key)]
        close(row["delta_vs_full"], cell.miou - full.miou, "Ablation table delta mismatch")
    policies = read_csv(evidence, audit / "candidate_policy_summary.csv")
    expected = {(r["dataset"], r["variant"]): r for r in candidate_rows if r["stage"] == "ablation"}
    require(len(policies) == len(expected), "Policy table length mismatch")
    seen = set()
    for row in policies:
        key = row["dataset"], row["variant"]
        require(key in expected and key not in seen, "Duplicate/unknown policy row")
        seen.add(key)
        for field, computed in (("candidates_per_image", "candidates_per_image"),
                                ("selected_area_tokens_per_image", "selected_area_tokens_per_image"),
                                ("median_candidate_area_tokens", "area_median")):
            close(row[field], expected[key][computed], f"Policy table mismatch: {field}")
    checks.append(dict(check="supplied_summary_ablation_and_policy_tables", status="passed",
                       rows=len(ablations), source_path=str(audit)))
    return checks


def crosscheck_voc(path, cells, class_deltas, allocations, evidence):
    summary = evidence.json(path / "summary.json")
    require(summary["status"] == "completed", "Incomplete VOC analysis")
    checks = []
    for name, hashed in summary["output_sha256"].items():
        if name.endswith(".csv"):
            require(hashlib.sha256(evidence.read(path / name)).hexdigest() == hashed, "VOC output seal mismatch")
    for stage, audits in (("ablation", summary["matched512"]), ("full_validation", {"full": summary["full_validation"]})):
        for variant, audit in audits.items():
            cell = cells[(stage, "voc21", variant)]
            require(Path(audit["path"]).resolve() == cell.path, "VOC crosscheck uses a different cell")
            close(audit["miou"], cell.miou, "VOC crosscheck mIoU mismatch")
    old_rows = read_csv(evidence, path / "matched512_class_deltas.csv")
    expected = {(r["comparator"], r["class_idx"]): r for r in class_deltas if r["dataset"] == "voc21"}
    checked = 0
    for row in old_rows:
        if row["comparator"] == "legacy_reference_panel":
            continue
        key = row["comparator"], int(row["class_idx"])
        require(key in expected and row["class_name"] == expected[key]["class_name"], "VOC class alignment mismatch")
        close(row["delta_full_minus_control_pp"], expected[key]["delta_full_minus_control_pp"], "VOC class delta mismatch")
        checked += 1
    require(checked == 63, "Incomplete VOC class crosscheck")
    previous = read_csv(evidence, path / "episode_weight_comparison.csv")
    current = {r["episode"]: r for r in allocations if r["dataset"] == "voc21"}
    require(len(previous) == len(current), "VOC episode count mismatch")
    for row in previous:
        now = current[int(row["episode"])]
        for old, new in (("normalized_weight_tv", "full_vs_raw_tv"), ("dominance_only_tv", "dominance_only_tv"),
                         ("rho_conf_spearman", "rho_conf_spearman")):
            close(row[old], now[new], f"VOC allocation crosscheck: {old}")
    checks.append(dict(check="voc21_prior_analysis_counts_class_deltas_and_episode_weights", status="passed",
                       rows=checked, source_path=str(path)))
    return checks


def ensure_output(output, inputs):
    output = output.resolve()
    require(not output.exists(), f"Output must be a fresh directory: {output}")
    for source in inputs:
        source = source.resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                "Output overlaps an input/source tree")
    return output


def analyze(suite, voc_analysis, output):
    suite, voc_analysis = suite.resolve(), voc_analysis.resolve()
    output = ensure_output(output, (suite, voc_analysis, Path(__file__).resolve().parents[1]))
    evidence = Evidence()
    audit = suite / "final_audit"
    require(evidence.json(audit / "queue_state.json")["status"] == "completed", "Final audit not completed")
    require(evidence.json(suite / "queue_state.json")["status"] == "completed", "Suite not completed")
    manifest = evidence.json(suite / "suite_protocol.json")
    rows = read_csv(evidence, audit / "summary.csv")
    cells, excluded = {}, []
    for row in rows:
        if row["dataset"] not in DATASETS:
            excluded.append(dict(**row, exclusion_reason="not a requested evaluation class space"))
            continue
        key = row["stage"], row["dataset"], row["variant"]
        require(key not in cells, "Duplicate summary cell")
        cells[key] = load_cell(row, manifest, evidence)
        cell = cells[key]
        require(cell.path.is_relative_to(suite), "Cell outside supplied suite")
        if cell.path.is_relative_to(audit / "fresh_full_coco_cd"):
            require(row["dataset"] == "coco_cd" and row["variant"] == "full",
                    "Unexpected refreshed control identity")
            request = evidence.json(audit / "requested_protocol.json")
            require(request["parent_protocol_sha256"] == evidence.hashes[str(suite / "suite_protocol.json")],
                    "Final audit parent seal mismatch")
            for name, hashed in cell.protocol["experiment_hashes"].items():
                require(request["tools"].get(name) == hashed, "Refreshed worker seal mismatch")
        else:
            selected = evidence.json(suite.joinpath(*key) / "selected_result.json")
            require(Path(selected["result_path"]).resolve() == cell.path / "result.json"
                    and selected["result_sha256"] == evidence.hashes[str(cell.path / "result.json")],
                    "Selected result seal mismatch")
            for name, hashed in cell.protocol["experiment_hashes"].items():
                require(manifest["tool_hashes"].get(name) == hashed, "Suite worker seal mismatch")
    for ds in DATASETS:
        require(all(("ablation", ds, v) in cells for v in VARIANTS), f"Incomplete ablation panel: {ds}")
    tables = {name: [] for name in ("cells", "ablation_deltas", "class_iou", "class_deltas", "image_deltas",
        "episode_class_deltas", "candidate_summary", "episode_diagnostics", "image_diagnostics",
        "class_candidate_summary", "episode_allocations", "class_allocations", "image_allocations",
        "episode_class_allocations", "k_prefix_budget")}
    vocabularies = {}
    for key, cell in cells.items():
        ctx = cell.context
        vocabularies[cell.protocol["vocabulary_sha256"]] = cell.vocabulary
        tables["cells"].append(dict(**ctx, miou_pct=cell.miou, classes=len(cell.iou),
            protocol_path=str(cell.path / "protocol.json"), protocol_sha256=cell.result["protocol_sha256"],
            counts_path=str(cell.path / "per_image_counts.json"), candidates_path=str(cell.path / "candidate_diagnostics.csv"),
            cost_path=str(cell.path / "cost.csv"),
            input_identity_basis="pre-run manifest ordered image/label paths and metadata; raw input contents not rehashed"))
        for i, name in enumerate(cell.vocabulary["classes"]):
            tables["class_iou"].append(dict(**ctx, class_idx=i, class_name=name,
                intersection=cell.result["intersection"][i], union=cell.result["union"][i], iou_pct=cell.iou[i]))
        # The CD full-validation entry aliases exactly the same ablation artifact.
        if key[0] == "full_validation" and any(c.path == cell.path for k, c in cells.items() if k[0] == "ablation"):
            continue
        candidate, episodes, images, classes = candidate_tables(cell)
        tables["candidate_summary"].append(candidate)
        tables["episode_diagnostics"].extend(episodes)
        tables["image_diagnostics"].extend(images)
        tables["class_candidate_summary"].extend(classes)
    for ds in DATASETS:
        full = cells[("ablation", ds, "full")]
        for variant in VARIANTS[1:]:
            control = cells[("ablation", ds, variant)]
            verify_identity(full.protocol, control.protocol)
            implementation = verify_implementation(full.protocol, control.protocol, evidence)
            if variant != "no_connected_regions":
                paired_candidates(full, control, same_support=variant != "no_source_tracing")
            tables["ablation_deltas"].append(dict(**full.context, comparator=variant,
                full_miou_pct=full.miou, control_miou_pct=control.miou,
                delta_full_minus_control_pp=full.miou - control.miou,
                control_result_path=str(control.path / "result.json"), implementation_match=implementation,
                exact_extended_environment_match=full.protocol["environment"] == control.protocol["environment"],
                cost_controlled_comparison=False,
                defined_class_mask_equal=[u is not None for u in full.iou] == [u is not None for u in control.iou]))
            tables["class_deltas"].extend(class_comparison(full, control))
            if variant in ("no_source_tracing", "no_region_normalization"):
                tables["image_deltas"].extend(image_comparison(full, control))
                batch = full.protocol["config"]["batch_size"]
                for ep, start in enumerate(range(0, len(full.records), batch)):
                    for row in class_comparison(full, control, full.records[start:start + batch], control.records[start:start + batch]):
                        row.update(episode=ep, episode_images=len(full.records[start:start + batch]))
                        tables["episode_class_deltas"].append(row)
        episodes, classes, images, episode_classes = allocation_tables(full, cells[("ablation", ds, "no_source_tracing")])
        tables["episode_allocations"].extend(episodes)
        tables["class_allocations"].extend(classes)
        tables["image_allocations"].extend(images)
        tables["episode_class_allocations"].extend(episode_classes)
        tables["k_prefix_budget"].extend(k_prefix_budget(full, cells[("ablation", ds, "no_source_tracing")],
                                                        verify_prefix_source(full.protocol, evidence)))
    tables["class_responses"] = class_responses(tables["class_deltas"], tables["class_allocations"],
                                               tables["class_candidate_summary"])
    checks = validate_supplied_tables(audit, rows, cells, tables["candidate_summary"], evidence)
    checks += crosscheck_voc(voc_analysis, cells, tables["class_deltas"], tables["episode_allocations"], evidence)
    tables["crosschecks"] = checks
    sources = [Path(__file__).resolve(), Path(__file__).with_name("analyze_voc21_tuning.py").resolve(),
               Path(__file__).resolve().parents[1] / "tests/test_dataset_response.py"]
    for path in sources:
        evidence.read(path)
    evidence.recheck()
    boundaries = [
        "Post-hoc descriptive analysis of one saved seed; no causal mediation or significance claim.",
        "All mIoU and class IoU use summed intersection/union. Image micro-IoU and episode/class IoU are descriptive, not additive contributions to dataset mIoU.",
        "No subtraction across full/512 scopes, image order, image-batch size, vocabulary, data config, seed, or production source identity.",
        "VOC21/PC60 full-validation scores are absolute only; COCOObject has no full-validation cell in the supplied audit. CD full-validation aliases its full ablation and is not an independent replication.",
        "Legacy VOC panel and older-vocabulary baselines are excluded from matched comparisons; supplied VOC analysis is used only to crosscheck matching cells.",
        "The refreshed CD worker differs only in environment/file-metadata instrumentation under an AST check. Minimal environment agrees; extended historical environment is missing for the other cells.",
        "Timing is saved observational GPU cost, not newly measured CPU or GPU speed. Excludes warmup for measured ms/image and measured views/image; all-candidate summaries include warmup. No controlled speedup ratio is reported.",
        "Candidate quality means the recorded min(T,CF) proxy, not GT precision. Coverage means images/episodes/classes with candidates. Area sums count patch-token slots; overlapping patches and absent masks prevent true pixel-union or GT coverage recovery.",
        "Query counts differ across VOC21/PC60/COCOObject/CD (56/60/112/3). Absolute rho medians and threshold fractions such as rho<=0.01 are score-distribution descriptions, not cross-dataset GT evidence quality or proof of useless candidates.",
        "Within-dataset query aliases are also unequal across evaluation classes; collapsed class coefficient masses describe actual nominal allocation, not equal query opportunity or class quality.",
        "K-prefix estimates use cached per-image raw-confidence order, not rho rank. Production sort/prefix source and matching raw-control nonincreasing confidence are checked. Saved tie order is retained; component_idx is not recorded. These estimates do not reproduce FP16 batched re-encoding or adaptation trajectories and predict neither GT coverage nor mIoU.",
        "Intersection/union alone cannot identify GT support, FP/FN, per-class accuracy, candidate correctness, or entropy losses.",
        "Coefficients are detached real-valued nominal allocations on region-mean entropy, not measured FP16 losses or gradients. The estimator first removes rho<=EPS; no-normalization then multiplies surviving rho by area and applies the objective gate.",
        "Dominance-positive fraction is d>0; effective allocation change is measured separately by per-episode TV against q-only. A uniform score multiplier cancels upon normalization.",
        "Min bottleneck uses strict T<CF, CF<T, exact ties, and q/max(T,CF); target-only/CF-only/q-only allocations are offline arithmetic probes, not executed ablation accuracy.",
        "Class allocation means use every episode, including zero mass for absent classes and empty episodes; last partial episodes have equal weight. Candidate summaries are candidate-weighted, not class-weighted.",
        "Class-response rows juxtapose pooled class IoU with equal-episode coefficient mass: different estimands, not a fitted association or per-episode mechanism test. Representative drops are selected only by pooled IoU delta, not by coefficient movement.",
        "Episode update_steps, effective_view_batch_size, calls and inputs are copied recorded instrumentation counters, not independently reconstructed compute counts.",
        "Frozen source and observed artifact bytes are hashed and rechecked. Raw image/mask content and checkpoint bytes are not rehashed; pre-run path/metadata identity is not a raw-data content seal.",
    ]
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--suite", str(suite),
               "--voc-analysis", str(voc_analysis), "--output", str(output)]
    summary = dict(schema_version=1, status="completed", cpu_only=True,
        runtime_imported=False, cpu_thread_limit=1, input_suite=str(suite),
        input_voc_analysis=str(voc_analysis), output_directory=str(output),
        evidence_sha256=dict(sorted(evidence.hashes.items())), vocabularies=vocabularies,
        ablation_deltas=tables["ablation_deltas"], candidate_summary=tables["candidate_summary"],
        k_prefix_budget=tables["k_prefix_budget"],
        excluded_rows=excluded, boundaries=boundaries, limitations=boundaries, crosschecks=checks,
        representative_dropping_classes=[dict(dataset=ds, comparator=comparator,
            selection="up to three most negative defined class IoU deltas; descriptive, not selected for significance",
            rows=sorted((r for r in tables["class_responses"] if r["dataset"] == ds
                and r["comparator"] == comparator and r["delta_full_minus_control_pp"] is not None
                and r["delta_full_minus_control_pp"] < 0), key=lambda r: r["delta_full_minus_control_pp"])[:3])
            for ds in DATASETS for comparator in ("no_source_tracing", "no_region_normalization")],
        allocation_summary=[dict(dataset=ds, **{field: distribution([r[field] for r in tables["episode_allocations"]
            if r["dataset"] == ds]) for field in ("full_vs_raw_tv", "dominance_only_tv", "region_normalization_tv",
                "min_vs_target_tv", "min_vs_counterfactual_tv", "rho_conf_spearman")}) for ds in DATASETS],
        reproduce_command="env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "
            + " ".join(f"{key}=1" for key in THREAD_VARS) + " " + shlex.join(command),
        reproduce_note="Use a fresh --output directory; existing artifacts are never overwritten.")
    output.mkdir(parents=True)
    for name, table in tables.items():
        write_csv(output / f"{name}.csv", table)
    summary["output_sha256"] = {p.name: digest(p) for p in sorted(output.glob("*.csv"))}
    summary["output_rows"] = {name: len(table) for name, table in tables.items()}
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--voc-analysis", type=Path, default=DEFAULT_VOC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in THREAD_VARS:
        os.environ[name] = "1"
    summary = analyze(args.suite, args.voc_analysis, args.output)
    print(json.dumps(dict(status=summary["status"], output=summary["output_directory"],
                         tables=summary["output_rows"]), indent=2))


if __name__ == "__main__":
    main()
