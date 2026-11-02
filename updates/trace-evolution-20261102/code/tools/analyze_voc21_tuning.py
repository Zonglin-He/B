"""CPU-only, read-only analysis of frozen VOC21 counts and candidate weights."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys


VARIANTS = ("full", "no_source_tracing", "no_reliability_weighting", "no_region_normalization")
IDENTITY_FIELDS = ("dataset", "split", "image_ids", "label_ids", "vocabulary_sha256",
                   "seed", "selection_seed", "available_images")
CANDIDATE_KEY = ("episode", "image_id", "patch_in_batch", "class_idx", "mask_sha256")
SUPPORT_FIELDS = ("target_support", "residual_support", "counterfactual_support", "quality", "dominance")
EPS = 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Evidence:
    """Hash inspected files and detect changes before publishing the analysis."""

    def __init__(self):
        self.hashes = {}
        self.metadata = {}

    def read(self, path):
        path = path.resolve()
        data = path.read_bytes()
        hashed = hashlib.sha256(data).hexdigest()
        require(str(path) not in self.hashes or self.hashes[str(path)] == hashed,
                f"Evidence changed during analysis: {path}")
        self.hashes[str(path)] = hashed
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def check_metadata(self, path, expected):
        path = path.resolve()
        stat = path.stat()
        actual = [stat.st_size, stat.st_mtime_ns]
        require(actual == expected, f"Input file metadata mismatch: {path}")
        self.metadata[str(path)] = actual

    def recheck(self):
        for path, hashed in self.hashes.items():
            require(digest(Path(path)) == hashed, f"Evidence changed during analysis: {path}")
        for path, expected in self.metadata.items():
            self.check_metadata(Path(path), expected)


def package_hashes(root, evidence):
    package = root / "trace_ovss"
    require(package.is_dir(), f"Missing package: {package}")
    return {path.relative_to(package).as_posix(): hashlib.sha256(evidence.read(path)).hexdigest()
            for path in sorted(package.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def load_vocabulary(root, evidence):
    path = root / "trace_ovss/resources/class_names/cls_voc21.txt"
    groups = [[part.strip() for part in line.split(",")]
              for line in evidence.read(path).decode("utf-8").splitlines() if line.strip()]
    require(len(groups) == 21 and all(all(group) for group in groups), "Expected 21 VOC21 resource rows")
    classes = [group[0] for group in groups]
    classes[0] = "background"
    queries = [query for group in groups for query in group]
    mapping = [index for index, group in enumerate(groups) for _ in group]
    return dict(classes=classes, queries=queries, query_to_class=mapping,
                resource_path=str(path), resource_sha256=digest(path))


def vocabulary_hash(vocabulary, extensions):
    active = ([vocabulary["queries"], vocabulary["query_to_class"]] if extensions
              else [vocabulary["classes"], None])
    return hashlib.sha256(json.dumps(active).encode()).hexdigest()


def count_vector(value, classes, name):
    require(isinstance(value, list) and len(value) == classes, f"Invalid {name} class axis")
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and v >= 0 and int(v) == v for v in value),
            f"Invalid {name} counts")
    return [int(v) for v in value]


def aggregate_counts(records, classes=21):
    intersection, union = [0] * classes, [0] * classes
    for record in records:
        left = count_vector(record["intersection"], classes, "intersection")
        right = count_vector(record["union"], classes, "union")
        require(all(i <= u for i, u in zip(left, right)), "Intersection exceeds union")
        intersection = [a + b for a, b in zip(intersection, left)]
        union = [a + b for a, b in zip(union, right)]
    iou = [100 * i / u if u else None for i, u in zip(intersection, union)]
    present = [value for value in iou if value is not None]
    require(bool(present), "No defined class IoU")
    return intersection, union, iou, math.fsum(present) / len(present)


@dataclass
class Cell:
    directory: Path
    protocol: dict
    result: dict
    intersection: list
    union: list
    iou: list
    audit: dict


def load_cell(directory, images, full_split, variant, evidence, classes=21):
    protocol_path = directory / "protocol.json"
    protocol = evidence.json(protocol_path)
    result = evidence.json(directory / "result.json")
    records = evidence.json(directory / "per_image_counts.json")
    require(result["status"] == "completed", f"Incomplete cell: {directory}")
    require(result["protocol_sha256"] == digest(protocol_path), f"Protocol hash mismatch: {directory}")
    require(protocol["dataset"] == "voc21" and protocol["split"] == "evaluation",
            f"Not VOC21 evaluation: {directory}")
    require(protocol["config"]["dataset"] == "PascalVOC21Dataset", "Wrong evaluation class space")
    require(protocol["available_images"] == 1449, f"Incomplete official VOC21 pool: {directory}")
    require(protocol["images"] == result["images"] == len(records) == images,
            f"Unexpected cohort size: {directory}")
    require(protocol["full_split"] is full_split and result["full_split"] is full_split
            and full_split == (images == protocol["available_images"]), f"Wrong full/subset label: {directory}")
    for key in ("dataset", "split"):
        require(result[key] == protocol[key], f"Result identity mismatch: {key}")
    require(protocol.get("variant", "full") == variant and result.get("variant", "full") == variant,
            f"Variant identity mismatch: {directory}")
    require(protocol.get("force_uniform_policy", False) is False, "Unexpected reference policy override")
    for key in ("image_ids", "label_ids"):
        values = protocol[key]
        require(len(values) == images and len(set(values)) == images, f"Duplicate or missing {key}")
    require([r["image_id"] for r in records] == protocol["image_ids"], "Per-image identity/order mismatch")
    intersection, union, iou, miou = aggregate_counts(records, classes)
    require(intersection == count_vector(result["intersection"], classes, "result intersection")
            and union == count_vector(result["union"], classes, "result union"), "Aggregate count mismatch")
    require(math.isclose(miou, result["miou_unrounded"], rel_tol=0, abs_tol=1e-9), "mIoU count mismatch")
    require(math.isclose(round(miou, 2), result["metrics"]["mIoU"], rel_tol=0, abs_tol=0.011),
            "Rounded mIoU mismatch")
    root = Path(protocol["package_root"]).resolve()
    recorded = protocol["source_hashes"]
    require(bool(recorded), "Missing source hashes")
    for relative, hashed in recorded.items():
        path = (root / "trace_ovss" / relative).resolve()
        require(path.is_relative_to(root / "trace_ovss"), "Unsafe recorded source path")
        require(hashlib.sha256(evidence.read(path)).hexdigest() == hashed,
                f"Source hash mismatch: {path}")
    for relative, hashed in protocol.get("experiment_hashes", {}).items():
        path = (root / "tools" / relative).resolve()
        require(path.is_relative_to(root / "tools"), "Unsafe recorded tool path")
        require(hashlib.sha256(evidence.read(path)).hexdigest() == hashed, f"Tool hash mismatch: {path}")
    audit = dict(path=str(directory), images=images, full_split=full_split, miou=miou,
                 protocol_sha256=result["protocol_sha256"], source_files_verified=len(recorded),
                 experiment_files_verified=len(protocol.get("experiment_hashes", {})),
                 environment=protocol.get("environment"), file_metadata_in_cell="file_metadata" in protocol,
                 image_batch_size=protocol["config"]["batch_size"],
                 view_forward_batch_size=protocol["config"]["trace_view_batch_size"])
    return Cell(directory, protocol, result, intersection, union, iou, audit)


def verify_match(first, other, *, allow_view_batch_difference=False):
    for key in IDENTITY_FIELDS:
        require(first[key] == other[key], f"Matched identity mismatch: {key}")
    a, b = dict(first["config"]), dict(other["config"])
    if allow_view_batch_difference:
        a.pop("trace_view_batch_size")
        b.pop("trace_view_batch_size")
    require(a == b, "Matched configuration mismatch")
    if "file_metadata" in first and "file_metadata" in other:
        require(first["file_metadata"] == other["file_metadata"], "Matched file metadata mismatch")


def verify_frozen_inputs(cell, expected, manifest_path, evidence):
    verify_match(cell.protocol, expected)
    require(manifest_path.stat().st_mtime_ns <= (cell.directory / "protocol.json").stat().st_mtime_ns,
            "Suite input identity was not recorded before the cell")
    metadata = expected["file_metadata"]
    require(len(metadata) == 2 * len(expected["image_ids"]), "Incomplete frozen file metadata")
    for index, pair in enumerate(zip(expected["image_ids"], expected["label_ids"])):
        for offset, path in enumerate(pair):
            evidence.check_metadata(Path(path), metadata[2 * index + offset])
    cell.audit["input_metadata_basis"] = (
        "Cell metadata and pre-run suite manifest" if "file_metadata" in cell.protocol
        else "Pre-run suite manifest; original worker did not embed file metadata")


def class_deltas(full, control, classes, comparator):
    return [dict(comparator=comparator, images=full.result["images"], full_split=False,
                 class_idx=index, class_name=name, full_intersection=full.intersection[index],
                 full_union=full.union[index], control_intersection=control.intersection[index],
                 control_union=control.union[index], full_iou_pct=full.iou[index],
                 control_iou_pct=control.iou[index],
                 delta_full_minus_control_pp=(full.iou[index] - control.iou[index]
                     if full.iou[index] is not None and control.iou[index] is not None else None))
            for index, name in enumerate(classes)]


def rankdata(values):
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            ranks[index] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(first, second):
    require(len(first) == len(second), "Spearman length mismatch")
    if len(first) < 2:
        return None
    a, b = rankdata(first), rankdata(second)
    mean = (len(a) + 1) / 2
    numerator = math.fsum((x - mean) * (y - mean) for x, y in zip(a, b))
    denominator = math.sqrt(math.fsum((x - mean) ** 2 for x in a)
                            * math.fsum((y - mean) ** 2 for y in b))
    return numerator / denominator if denominator else None


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return dict(n=0, mean=None, min=None, median=None, p95=None, max=None)

    def quantile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return dict(n=len(values), mean=math.fsum(values) / len(values), min=values[0],
                median=quantile(0.5), p95=quantile(0.95), max=values[-1])


def normalized_weights(values):
    active = [value if value > EPS else 0.0 for value in values]
    total = math.fsum(active)
    return [value / total for value in active] if total > EPS else [0.0] * len(active)


def total_variation(first, second):
    if not math.fsum(first) or not math.fsum(second):
        return None
    return math.fsum(abs(a - b) for a, b in zip(first, second)) / 2


def read_candidates(cell, evidence, vocabulary):
    rows = list(csv.DictReader(evidence.read(cell.directory / "candidate_diagnostics.csv")
                               .decode("utf-8").splitlines()))
    indexed = {}
    protocol = cell.protocol
    batch_size = protocol["config"]["batch_size"]
    mapping = (vocabulary["query_to_class"] if protocol["config"]["class_extensions"]
               else list(range(21)))
    for row in rows:
        for field in ("episode", "patch_in_batch", "class_idx", "area"):
            row[field] = int(row[field])
        episode = row["episode"]
        require(0 <= episode < math.ceil(protocol["images"] / batch_size), "Invalid candidate episode")
        require(row["image_id"] in protocol["image_ids"][episode * batch_size:(episode + 1) * batch_size],
                "Candidate image/episode identity mismatch")
        require(row["patch_in_batch"] >= 0 and row["area"] > 0, "Invalid candidate patch/area")
        require(0 <= row["class_idx"] < len(mapping), "Candidate query index is not in vocabulary")
        require(len(row["mask_sha256"]) == 64
                and all(char in "0123456789abcdef" for char in row["mask_sha256"]), "Invalid mask hash")
        row["evaluation_class_idx"] = mapping[row["class_idx"]]
        for field in (*SUPPORT_FIELDS, "reliability"):
            row[field] = float(row[field]) if row[field] != "" else None
            require(row[field] is None or math.isfinite(row[field]) and 0 <= row[field] <= 1,
                    f"Invalid candidate {field}")
        require(row["reliability"] is not None, "Missing candidate weight")
        if protocol["variant"] == "no_source_tracing":
            require(all(row[field] is None for field in SUPPORT_FIELDS), "Raw-confidence control has traced scores")
        else:
            require(all(row[field] is not None for field in SUPPORT_FIELDS), "Missing traced scores")
            quality = min(row["target_support"], row["counterfactual_support"])
            dominance = max(row["residual_support"] - row["target_support"], 0)
            expected = 1.0 if protocol["variant"] == "no_reliability_weighting" else quality * (1 - dominance)
            for field, value in (("quality", quality), ("dominance", dominance), ("reliability", expected)):
                require(math.isclose(row[field], value, rel_tol=0, abs_tol=1e-12), f"Candidate formula mismatch: {field}")
        key = tuple(row[field] for field in CANDIDATE_KEY)
        require(key not in indexed, "Duplicate candidate key")
        indexed[key] = row
    cost_rows = list(csv.DictReader(evidence.read(cell.directory / "cost.csv").decode("utf-8").splitlines()))
    require([int(row["episode"]) for row in cost_rows] == list(range(math.ceil(protocol["images"] / batch_size))),
            "Incomplete episode diagnostics")
    for episode, cost in enumerate(cost_rows):
        episode_rows = [row for row in rows if row["episode"] == episode]
        require(len(episode_rows) == int(cost["candidates"]), "Candidate diagnostic count mismatch")
        require(int(cost["images"]) == len(protocol["image_ids"][episode * batch_size:(episode + 1) * batch_size]),
                "Episode image count mismatch")
        expected_views = 0 if protocol["variant"] == "no_source_tracing" else 3 * len(episode_rows)
        require(int(cost["diagnostic_views"]) == expected_views, "Intervention-view count mismatch")
    return indexed


def compare_weights(candidates, vocabulary):
    full, raw = candidates["full"], candidates["no_source_tracing"]
    require(bool(full), "No candidate diagnostics to compare")
    for name, control in candidates.items():
        require(full.keys() == control.keys(), f"Candidate identity mismatch: {name}")
        require(all(row["area"] == control[key]["area"] for key, row in full.items()),
                f"Candidate area mismatch: {name}")
    episodes = defaultdict(list)
    for key, row in full.items():
        episodes[row["episode"]].append(key)
    normalized, episode_rows = {}, []
    for episode, keys in sorted(episodes.items()):
        rho = [full[key]["reliability"] for key in keys]
        confidence = [raw[key]["reliability"] for key in keys]
        a, b = normalized_weights(rho), normalized_weights(confidence)
        q = normalized_weights([full[key]["quality"] for key in keys])
        for key, x, y in zip(keys, a, b):
            normalized[key] = (x, y)
        episode_rows.append(dict(episode=episode, candidates=len(keys), rho_conf_spearman=spearman(rho, confidence),
            normalized_weight_tv=total_variation(a, b), dominance_only_tv=total_variation(a, q),
            full_weight_sum=math.fsum(rho), confidence_weight_sum=math.fsum(confidence),
            full_active_candidates=sum(value > EPS for value in rho),
            raw_active_candidates=sum(value > EPS for value in confidence),
            full_max_normalized_weight=max(a), raw_max_normalized_weight=max(b),
            full_effective_candidates=1 / math.fsum(x * x for x in a) if any(a) else None,
            raw_effective_candidates=1 / math.fsum(x * x for x in b) if any(b) else None))
    class_rows = []
    for index, name in enumerate(vocabulary["classes"]):
        keys = [key for key, row in full.items() if row["evaluation_class_idx"] == index]
        dominance = [full[key]["dominance"] for key in keys]
        class_rows.append(dict(class_idx=index, class_name=name, candidates=len(keys),
            candidate_area_sum=sum(full[key]["area"] for key in keys),
            episodes_with_candidates=len({full[key]["episode"] for key in keys}),
            full_mean_episode_loss_weight=math.fsum(normalized[key][0] for key in keys) / len(episodes),
            raw_mean_episode_loss_weight=math.fsum(normalized[key][1] for key in keys) / len(episodes),
            delta_full_minus_raw_loss_weight=math.fsum(normalized[key][0] - normalized[key][1] for key in keys) / len(episodes),
            rho_conf_spearman=spearman([full[key]["reliability"] for key in keys],
                                       [raw[key]["reliability"] for key in keys]),
            dominance_positive_fraction=sum(value > 0 for value in dominance) / len(keys) if keys else None,
            dominance_mean=distribution(dominance)["mean"], dominance_p95=distribution(dominance)["p95"]))
    dominance = [row["dominance"] for row in full.values()]
    summary = dict(candidates=len(full), episodes=len(episodes), candidate_key_fields=list(CANDIDATE_KEY),
        all_candidate_keys_and_areas_match=True,
        candidate_row_order_matches={name: list(control) == list(full) for name, control in candidates.items()},
        rho_conf_spearman=spearman([row["reliability"] for row in full.values()],
                                   [raw[key]["reliability"] for key in full]),
        per_episode_spearman=distribution([row["rho_conf_spearman"] for row in episode_rows]),
        normalized_weight_tv=distribution([row["normalized_weight_tv"] for row in episode_rows]),
        dominance_only_tv=distribution([row["dominance_only_tv"] for row in episode_rows]),
        dominance=dict(**distribution(dominance), positive_count=sum(value > 0 for value in dominance),
            positive_fraction=sum(value > 0 for value in dominance) / len(dominance),
            at_least_half_count=sum(value >= 0.5 for value in dominance)),
        normalization="Per episode, rho/sum(rho) versus confidence/sum(confidence); weights <= EPS excluded",
        weight_scope="Nominal coefficients from saved detached scores; not FP16 gradients or entropy/loss values",
        loss_weight_class_scope="Equal-episode mean, including zero mass for classes absent from an episode",
        actual_per_class_objective_loss_available=False)
    return summary, episode_rows, class_rows


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_summary(rows):
    defined = [row for row in rows if row["delta_full_minus_control_pp"] is not None]
    return dict(losing_classes=[row for row in sorted(defined, key=lambda r: r["delta_full_minus_control_pp"])
                               if row["delta_full_minus_control_pp"] < 0],
                gaining_classes=[row for row in sorted(defined, key=lambda r: -r["delta_full_minus_control_pp"])
                                 if row["delta_full_minus_control_pp"] > 0])


def render_report(summary):
    weights = summary["weights"]

    def number(value):
        return f"{value:.6f}" if value is not None else "undefined"

    lines = ["# VOC21 Analysis", "", "## Cohorts", "",
        f'- Full official validation: {summary["full_validation"]["images"]} images, '
        f'mIoU {summary["full_validation"]["miou"]:.8f}. Absolute class IoUs: `full_per_class_iou.csv`.',
        "- Matched controls: 512 images in identical order and identical image-batch episodes; not full validation.",
        "- IoU is recomputed after summing per-image intersections/unions, never by averaging image IoUs.", "",
        "| 512-image comparison | Full mIoU | Control mIoU | Full - control (pp) |",
        "| --- | ---: | ---: | ---: |"]
    for name, comparison in summary["comparisons"].items():
        lines.append(f'| {name} | {comparison["full_miou"]:.6f} | {comparison["control_miou"]:.6f} '
                     f'| {comparison["delta_full_minus_control_pp"]:+.6f} |')
    lines += ["", "## Losing Classes First", "",
              "Class deltas below are Full minus the stated control on matched 512 images."]
    for name, comparison in summary["comparisons"].items():
        losses = comparison["losing_classes"]
        text = "; ".join(f'{row["class_name"]} {row["delta_full_minus_control_pp"]:+.3f} pp'
                         for row in losses) or "none"
        lines.append(f"- {name}: {text}.")
    lines += ["", "## Weight Diagnostics", "",
        f'- {weights["candidates"]:,} candidate identities and areas match across all four controls; '
        f'{weights["episodes"]} episodes. Query aliases map to their VOC21 evaluation row, including background at row 0.',
        f'- Pooled rho/confidence Spearman: {number(weights["rho_conf_spearman"])}; '
        f'mean episode normalized-weight TV: {number(weights["normalized_weight_tv"]["mean"])}.',
        f'- Dominance D > T: {weights["dominance"]["positive_count"]:,} candidates '
        f'({100 * weights["dominance"]["positive_fraction"]:.2f}%); '
        f'mean d {weights["dominance"]["mean"]:.6f}, p95 {weights["dominance"]["p95"]:.6f}.',
        "- `matched512_loss_weight_changes.csv` reports per-class normalized loss-weight allocation, not measured entropy loss. "
        "Actual per-class objective losses are unavailable in these artifacts.",
        "- `episode_weight_comparison.csv` includes TV, rank correlation, largest weight and effective candidate count. "
        "A common positive scale on all episode weights cancels.", "", "## Identity And Boundaries", ""]
    lines += [f"- {note}" for note in summary["boundaries"]]
    lines += ["- Protocol checksums, recorded snapshot/tool hashes, current production hashes, ordered image/label IDs, "
              "vocabulary, seed, configuration, official pool size and count aggregation passed. "
              "All inspected evidence hashes are recorded in `summary.json` and rechecked before writing.",
              "- Input size/mtime values match the pre-run suite manifest. They are metadata identities, not image/mask content hashes. "
              "Legacy reference has no embedded environment or file metadata; its metadata agreement is retrospective only.",
              "- Analysis is read-only CPU work. It does not run or select batch_size 1..16 trials, modify the algorithm, "
              "or inspect/alter other running queue cells.", ""]
    return "\n".join(lines)


def analyze(suite, reference_panel, package_root, output):
    suite, reference_panel, package_root, output = (path.resolve() for path in
                                                   (suite, reference_panel, package_root, output))
    require(not output.exists(), f"Choose a fresh analysis output directory: {output}")
    for source in (suite, reference_panel, package_root):
        require(not output.is_relative_to(source), "Analysis output must be outside input evidence/package roots")
    evidence = Evidence()
    manifest_path = suite / "suite_protocol.json"
    manifest = evidence.json(manifest_path)
    vocabulary = load_vocabulary(package_root, evidence)
    current_hashes = package_hashes(package_root, evidence)
    require(current_hashes == manifest["source_hashes"], "Current production source differs from the frozen suite")
    full = load_cell(suite / "full_validation/voc21/full/attempt_001", 1449, True, "full", evidence)
    cells = {name: load_cell(suite / f"ablation/voc21/{name}/attempt_001", 512, False, name, evidence)
             for name in VARIANTS}
    reference = load_cell(reference_panel, 512, False, "full", evidence)
    for name, cell in [("full_validation", full), *cells.items()]:
        panel = "voc21/None" if name == "full_validation" else "voc21/512"
        verify_frozen_inputs(cell, manifest["input_panels"][panel], manifest_path, evidence)
        require(cell.protocol["source_hashes"] == current_hashes, f"Suite source identity mismatch: {name}")
        require(cell.protocol["experiment_hashes"] == cells["full"].protocol["experiment_hashes"],
                f"Suite experiment identity mismatch: {name}")
        require(all(manifest["tool_hashes"].get(tool) == hashed
                    for tool, hashed in cell.protocol["experiment_hashes"].items()),
                f"Suite manifest tool hash mismatch: {name}")
        require(cell.protocol["environment"] == cells["full"].protocol["environment"],
                f"Suite environment identity mismatch: {name}")
        require(cell.protocol["vocabulary_sha256"] == vocabulary_hash(vocabulary, cell.protocol["config"]["class_extensions"]),
                f"Vocabulary hash mismatch: {name}")
    for cell in cells.values():
        verify_match(cells["full"].protocol, cell.protocol)
    verify_match(cells["full"].protocol, reference.protocol, allow_view_batch_difference=True)
    full_pairs = dict(zip(full.protocol["image_ids"], full.protocol["label_ids"]))
    require(all(full_pairs.get(image) == label for image, label in
                zip(cells["full"].protocol["image_ids"], cells["full"].protocol["label_ids"])),
            "Panel image/label pairs are not in full validation")
    candidates = {name: read_candidates(cell, evidence, vocabulary) for name, cell in cells.items()}
    weight_summary, episode_rows, weight_rows = compare_weights(candidates, vocabulary)
    comparisons, delta_rows = {}, []
    for name, cell in [*[(name, cells[name]) for name in VARIANTS[1:]], ("legacy_reference_panel", reference)]:
        rows = class_deltas(cells["full"], cell, vocabulary["classes"], name)
        comparisons[name] = dict(images=512, full_split=False, full_miou=cells["full"].audit["miou"],
            control_miou=cell.audit["miou"],
            delta_full_minus_control_pp=cells["full"].audit["miou"] - cell.audit["miou"],
            image_batch_size=cells["full"].protocol["config"]["batch_size"],
            full_view_forward_batch_size=cells["full"].protocol["config"]["trace_view_batch_size"],
            control_view_forward_batch_size=cell.protocol["config"]["trace_view_batch_size"],
            scope="Implementation comparison, not formula attribution" if name == "legacy_reference_panel"
                  else "Matched research-only removal control", **comparison_summary(rows))
        delta_rows.extend(rows)
    raw_deltas = {row["class_idx"]: row for row in delta_rows if row["comparator"] == "no_source_tracing"}
    for row in weight_rows:
        row["delta_full_minus_raw_iou_pp"] = raw_deltas[row["class_idx"]]["delta_full_minus_control_pp"]
        row["actual_objective_loss_change"] = None
    absolute_rows = [dict(class_idx=index, class_name=name, images=1449, full_split=True,
        intersection=full.intersection[index], union=full.union[index], iou_pct=full.iou[index])
        for index, name in enumerate(vocabulary["classes"])]
    reference.audit["input_metadata_basis"] = "Retrospective agreement with matching suite panel; not a pre-run legacy seal"
    reference.audit["unrecorded_source_resources"] = sorted(
        path.relative_to(Path(reference.protocol["package_root"]) / "trace_ovss").as_posix()
        for path in (Path(reference.protocol["package_root"]) / "trace_ovss").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        and path.relative_to(Path(reference.protocol["package_root"]) / "trace_ovss").as_posix()
        not in reference.protocol["source_hashes"])
    analysis_sources = {}
    for path in (Path(__file__).resolve(), package_root / "tests/test_voc21_analysis.py"):
        if path.is_file():
            analysis_sources[str(path)] = hashlib.sha256(evidence.read(path)).hexdigest()
    boundaries = [
        f'The {reference.audit["miou"]:.2f} legacy and {cells["full"].audit["miou"]:.2f} new panel scores '
        "use the same 512-image cohort, not full VOC21. "
        f'View-forward batching differs ({reference.protocol["config"]["trace_view_batch_size"]} versus '
        f'{cells["full"].protocol["config"]["trace_view_batch_size"]}); kernel and reliability implementation '
        f'also changed; image batch size stays {cells["full"].protocol["config"]["batch_size"]}. '
        "Their delta is not attributable to the reliability formula alone.",
        "Legacy full-VOC21 scalar 52.16 has no supplied per-class counts. No full-cohort class delta "
        "or subtraction against this unmatched 512 panel is performed.",
        "Intersections/unions do not identify ground-truth class pixel counts, false positives and false negatives "
        "separately, per-class accuracy, or measured per-class entropy loss; none are inferred.",
        "Full-validation records are not used as the matched Full512 control: changing episode membership "
        "changes episodic adaptation. Weight diagnostics describe the fixed 512 panel only.",
        "This is post-hoc evaluation-label analysis for future tuning, not an untouched final test or a causal "
        "explanation of class changes. No batch-size trial or selection is made here.",
    ]
    summary = dict(schema_version=1, status="completed", dataset="voc21", class_order=vocabulary["classes"],
        vocabulary=vocabulary, full_validation=full.audit,
        matched512={name: cell.audit for name, cell in cells.items()}, reference_panel=reference.audit,
        comparisons=comparisons, weights=weight_summary, boundaries=boundaries,
        reference_source_differences=sorted(key for key in
            set(current_hashes) | set(reference.protocol["source_hashes"])
            if current_hashes.get(key) != reference.protocol["source_hashes"].get(key)),
        identity_checks=dict(ordered_image_label_ids=True, protocol_hashes=True, recorded_source_tool_hashes=True,
            current_package_matches_suite=True, pre_run_suite_input_metadata_matches=True,
            aggregate_counts_and_miou=True, matched_candidate_keys_and_areas=True,
            reference_environment_comparability="Unavailable in legacy protocol; not asserted",
            dataset_content_hashes="Not recorded; file sizes/mtimes only"),
        analysis_source_sha256=analysis_sources, evidence_sha256=dict(sorted(evidence.hashes.items())),
        input_file_metadata_checked=len(evidence.metadata),
        reproduce_command=shlex.join([sys.executable, str(Path(__file__).resolve()), "--suite", str(suite),
            "--reference-panel", str(reference_panel), "--package-root", str(package_root), "--output", "NEW_OUTPUT/analysis"]))
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "full_per_class_iou.csv", absolute_rows)
    write_csv(output / "matched512_class_deltas.csv", delta_rows)
    write_csv(output / "matched512_loss_weight_changes.csv", weight_rows)
    write_csv(output / "episode_weight_comparison.csv", episode_rows)
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    summary["output_sha256"] = {path.name: digest(path) for path in sorted(output.iterdir())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.suite, args.reference_panel, args.package_root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Analysis rejected: {error}\n")
    print(json.dumps(dict(output=str(args.output.resolve()), full_validation_miou=summary["full_validation"]["miou"],
                          matched_candidates=summary["weights"]["candidates"]), indent=2))


if __name__ == "__main__":
    main()
