"""Offline source-group transitions and class-subset IoU from sealed caches.

No prediction arrays, annotations, model or new adaptation are opened. Source
groups are not candidate-local update effects or subgroup pooled mIoU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .bootstrap import POPULATIONS, TUNED, read_json, require
from .intake import sha256, write_new
from .legacy_rho import ARCHIVE_SHA, load_components
from .pixel_transitions import ARMS, FIELDS, rates, source_spec

M05 = Path("/home/wwww/hzl/results/trace_mechanism_20261102/M05_cached_transitions_v1")
DATASETS = ("coco_object", "pc60")
EPS, REPLICATES, SEED = 1e-8, 10000, 20261102


def source_features(candidates):
    n = len(candidates)
    if not n:
        return dict(candidate_count=0, count_group="none", ordering_group="none", mean_confidence=None,
            median_area_fraction=None, weight_L1=None, comparable_pairs=0, discordant_pairs=0,
            confidence_tied_pairs=0, C_tied_pairs=0, active_C=0, active_confidence=0)
    confidence = np.asarray([r["confidence"] for r in candidates], np.float64)
    C = np.asarray([r["reliability"] for r in candidates], np.float64)
    require(bool(np.isfinite(confidence).all()) and bool(np.isfinite(C).all())
            and bool(((confidence >= 0) & (confidence <= 1)).all())
            and bool(((C >= 0) & (C <= 1)).all()), "Invalid cached candidate probabilities")
    areas = []
    for row in candidates:
        region = row["region"]
        shape = region["shape"]
        require(len(shape) == 2 and all(type(v) is int and v > 0 for v in shape)
                and row["area"] == region["true_pixels"] > 0, "Invalid cached candidate geometry")
        fraction = row["area"]/(shape[0]*shape[1])
        require(0 < fraction <= 1, "Candidate area outside token grid")
        areas.append(fraction)
    i, j = np.triu_indices(n, 1)
    a, b = np.sign(confidence[i]-confidence[j]), np.sign(C[i]-C[j])
    comparable = (a != 0) & (b != 0)
    discordant = comparable & (a != b)
    active_C, active_conf = C > EPS, confidence > EPS
    wC = np.where(active_C, C, 0.)
    wConf = np.where(active_conf, confidence, 0.)
    if wC.sum() > 0:
        wC /= wC.sum()
    if wConf.sum() > 0:
        wConf /= wConf.sum()
    return dict(candidate_count=n, count_group="single" if n == 1 else "multiple",
        ordering_group="single" if n == 1 else "no_comparable" if not comparable.any()
                       else "reordered" if discordant.any() else "same_order",
        mean_confidence=float(confidence.mean()), median_area_fraction=float(np.median(areas)),
        weight_L1=float(np.abs(wC-wConf).sum()), comparable_pairs=int(comparable.sum()),
        discordant_pairs=int(discordant.sum()), confidence_tied_pairs=int((a == 0).sum()),
        C_tied_pairs=int((b == 0).sum()), active_C=int(active_C.sum()), active_confidence=int(active_conf.sum()))


def memberships(features):
    result = ["all", "candidate_count:"+features["count_group"], "ordering:"+features["ordering_group"]]
    if features["candidate_count"]:
        conf, area = features["mean_confidence"], features["median_area_fraction"]
        result += ["source_mean_confidence:"+("lt0.25" if conf < .25 else "0.25to0.5" if conf < .5
                    else "0.5to0.75" if conf < .75 else "ge0.75"),
                   "source_median_candidate_area:"+("le0.01" if area <= .01 else "0.01to0.10" if area <= .1 else "gt0.10")]
    return result


def all_groups():
    return (["all"]+["candidate_count:"+v for v in ("none", "single", "multiple")]
        +["ordering:"+v for v in ("none", "single", "no_comparable", "same_order", "reordered")]
        +["source_mean_confidence:"+v for v in ("lt0.25", "0.25to0.5", "0.5to0.75", "ge0.75")]
        +["source_median_candidate_area:"+v for v in ("le0.01", "0.01to0.10", "gt0.10")])


def grouped_intervals(episodes, dataset):
    seed = int.from_bytes(hashlib.sha256(f"A01:{SEED}:{dataset}".encode()).digest()[:8], "big")
    rng, n = np.random.Generator(np.random.PCG64(seed)), len(episodes)
    output = {name:[] for name in rates(episodes[0, 0])}
    for start in range(0, REPLICATES, 32):
        indices = rng.integers(n, size=(min(32, REPLICATES-start), n))
        values = rates(episodes[indices].sum(1, dtype=np.int64))
        for name, values in values.items():
            output[name].append(values[:, 2]-values[:, 1])
    result = {}
    for name, chunks in output.items():
        values = np.concatenate(chunks)
        undefined = int((~np.isfinite(values)).sum())
        bounds = np.quantile(values, [.025, .975], method="linear") if undefined == 0 else (None, None)
        result[name] = dict(lower95_pp=None if bounds[0] is None else float(bounds[0]),
            upper95_pp=None if bounds[1] is None else float(bounds[1]), undefined_replicates=undefined,
            total_replicates=REPLICATES, no_replicate_dropped=True)
    return result


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "New cache analysis directory required")
    sources = [source_spec(d) for d in DATASETS]
    seal = read_json(M05/"ANALYSIS_SEALED.json")
    names = ["PROTOCOL.json", "result.json", *[f"{d}.transitions.jsonl" for d in DATASETS]]
    m05_hashes = {name:sha256(M05/name) for name in names}
    require(seal["complete"] and all(seal["artifact_sha256"][n] == h for n,h in m05_hashes.items()), "M05 stream not sealed")
    output.mkdir(parents=True)
    write_new(output/"PROTOCOL.json", dict(experiment="A01", stage="cache_first_partial_group_evidence",
        sources=sources, M05_root=str(M05), M05_artifact_sha256=m05_hashes,
        role_archive_sha256=ARCHIVE_SHA, roles="Offline pinned class taxonomy only; no scorer or loss role branch",
        groups=all_groups(), definitions="Source mean confidence bins[0,.25,.5,.75,1];source median candidate token-area bins<=.01/<=.1/>.1; exact pair order excluding ties",
        ordering="All admitted candidates, not only active ones; report ties/no-comparable and active counts separately",
        same_order_caveat="Same ranking does not imply same numeric normalized weights, EPS support, source budget or joint episode updates",
        metrics="Source-group pre-correct retention/damage, pre-wrong correction and net valid-pixel change; NOT source-group pooledmIoU or candidate-local effects",
        class_metrics="Per-class IoU and unweighted macro IoU within predeclared foreground/background/thing/stuff class subsets; no binary-role-collapse claim",
        unavailable="Mask-local candidate correctness/region outcomes need a distinct sealed analysis; do not infer them from source totals",
        uncertainty="10000 original full-episode paired draws, PCG64 rootseed20261102 keyed bydataset, same draws across groups;95pointwise percentile/no multiplicity adjustment",
        zero_denominators="Null point or interval with exact coverage/undefined replicate counts; do not skip cases, merge bins or discard replicates",
        historical_exposure="Fixed TRACE-selected settings and historically development-exposed full data; observational grouping, not candidate causality",
        source_code_sha256={str(Path(__file__).resolve()):sha256(__file__),
            str(Path(__file__).with_name("pixel_transitions.py").resolve()):sha256(Path(__file__).with_name("pixel_transitions.py")),
            str(Path(__file__).with_name("legacy_rho.py").resolve()):sha256(Path(__file__).with_name("legacy_rho.py"))},
        prediction_or_GT_payloads_opened=False, new_GPU_work=False, new_adaptation=False))
    write_new(output/"PREPARED.json", dict(protocol_sha256=sha256(output/"PROTOCOL.json"), executable="CPU cache analysis only", scientific_completion=False))


def run(output):
    output = Path(output).resolve(strict=True)
    p = read_json(output/"PROTOCOL.json")
    require(sha256(output/"PROTOCOL.json") == read_json(output/"PREPARED.json")["protocol_sha256"], "Group protocol drift")
    for path, digest in p["source_code_sha256"].items():
        require(sha256(path) == digest, "Group analysis source drift")
    for name, digest in p["M05_artifact_sha256"].items():
        require(sha256(M05/name) == digest, "Group transition input drift")
    components = load_components()
    write_new(output/"STARTED.json", dict(phase="CPU_cache_only", new_GPU_work=False, automatic_retry=False))
    results = {}
    try:
        for spec in p["sources"]:
            dataset, root = spec["dataset"], Path(spec["root"])
            for name, digest in spec["files_sha256"].items():
                require(sha256(root/name) == digest, "Canonical source drift")
            manifest = read_json(root/"prediction_protocol.json")["manifest"]
            panel, cfg = manifest["panel"], manifest["config"]
            inventory = read_json(root/"PREDICTIONS_SEALED.json")["artifact_sha256"]
            canonical = read_json(root/"result.json")
            inputs, batch = panel["inputs"], cfg["batch_size"]
            n = (len(inputs)+batch-1)//batch
            grouped = {g:np.zeros((n, len(ARMS), len(FIELDS)), np.int64) for g in all_groups()}
            coverage = {g:dict(sources=0, candidates=0, active_C=0, active_confidence=0) for g in all_groups()}
            cache_rows, seen = {}, 0
            stream_path = output/f"{dataset}.source_features.jsonl"
            with (M05/f"{dataset}.transitions.jsonl").open() as transitions, stream_path.open("x") as details:
                for episode in range(n):
                    name = f"episodes/{episode:06d}.image_C.json"
                    content = (root/name).read_bytes()
                    require(hashlib.sha256(content).hexdigest() == inventory[name], "Cached candidate receipt drift")
                    receipt = json.loads(content)
                    rows = inputs[episode*batch:(episode+1)*batch]
                    ids = [r["official_id"] for r in rows]
                    require(receipt["episode"] == episode and receipt["ordered_ids"] == ids, "Candidate episode/order drift")
                    stats = receipt["candidates"]
                    require(all(type(r["image_idx"]) is int and 0 <= r["image_idx"] < len(rows) for r in stats), "Candidate source index invalid")
                    for local, row in enumerate(rows):
                        transition = json.loads(next(transitions))
                        require(transition["index"] == seen and transition["source_id"] == row["official_id"]
                                and transition["episode"] == episode, "Source-transition join drift")
                        candidate_rows = [r for r in stats if r["image_idx"] == local]
                        features = source_features(candidate_rows)
                        counts = np.asarray([[transition["arms"][arm][f] for f in FIELDS] for arm in ARMS], np.int64)
                        rates(counts)
                        for group in memberships(features):
                            grouped[group][episode] += counts
                            coverage[group]["sources"] += 1
                            coverage[group]["candidates"] += features["candidate_count"]
                            coverage[group]["active_C"] += features["active_C"]
                            coverage[group]["active_confidence"] += features["active_confidence"]
                        details.write(json.dumps(dict(source_id=row["official_id"], original_index=seen, episode=episode,
                            **features, groups=memberships(features)))+"\n")
                        seen += 1
                    cache_rows[name] = inventory[name]
                require(transitions.readline() == "" and seen == len(inputs) == POPULATIONS[dataset], "Incomplete source transition population")
            roles = components["modules"]["roles"].class_roles_for(cfg["dataset"], panel["classes"])
            class_rows = []
            for group in ("foreground", "background", "thing", "stuff"):
                indices = [i for i, role in enumerate(roles) if (role != "background" if group == "foreground" else role == group)]
                arm_values = {}
                for arm in ("no_adapt", *ARMS):
                    result = canonical["arm_results"][arm]
                    selected = [i for i in indices if result["union"][i] > 0]
                    arm_values[arm] = dict(nonzero_union_classes=len(selected),
                        macro_class_IoU=None if not selected else float(np.mean([100*result["intersection"][i]/result["union"][i] for i in selected])))
                class_rows.append(dict(group=group, classes=len(indices), class_indices=indices,
                    dataset_classes=len(roles), arms=arm_values))
            group_rows = []
            for group, episodes in grouped.items():
                values = rates(episodes.sum(0, dtype=np.int64))
                intervals = grouped_intervals(episodes, dataset) if coverage[group]["sources"] else {}
                point = {key:None if not np.isfinite(v[2]-v[1]) else float(v[2]-v[1]) for key,v in values.items()}
                group_rows.append(dict(group=group, coverage=coverage[group], source_fraction=coverage[group]["sources"]/seen,
                    C_minus_Confidence={key:dict(delta_pp=value, **intervals.get(key, {})) for key,value in point.items()}))
            results[dataset] = dict(sources=seen, source_groups=group_rows, class_subset_IoU=class_rows,
                class_role_mapping=dict(zip(panel["classes"], roles)), candidate_receipt_sha256=cache_rows,
                not_computed=["candidate-local correctness/update effects", "mask-local foreground/background transitions", "source-group pooledmIoU"])
            write_new(output/f"{dataset}.completed.json", results[dataset])
        write_new(output/"RESULT.json", dict(status="completed_cache_first_group_component_only", datasets=results,
            new_GPU_work=False, prediction_or_GT_payloads_opened=False, all_requested_group_estimands_complete=False))
        names = ["PROTOCOL.json", "PREPARED.json", "STARTED.json", "RESULT.json"]
        names += [f"{d}.{suffix}" for d in DATASETS for suffix in ("source_features.jsonl", "completed.json")]
        write_new(output/"ANALYSIS_SEALED.json", dict(complete=True, scope="cache-first partial grouped evidence only",
            artifact_sha256={name:sha256(output/name) for name in names}, new_GPU_work=False, all_requested_group_estimands_complete=False))
    except BaseException as error:
        write_new(output/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare":prepare, "run":run}[args.phase](args.output)
