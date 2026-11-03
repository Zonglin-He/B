"""E08: paired episode resampling of already sealed class I/U, CPU only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .intake import sha256, write_new

TUNED = Path("/home/wwww/hzl/results/trace_single_probe_waterbirds_full_20261029_0817")
ALLOCATION = Path("/home/wwww/hzl/results/trace_allocation_controls_20261029_2055")
POPULATIONS = dict(voc20=1449, voc21=1449, pc59=5105, pc60=5105,
                   coco_object=5000, coco_stuff=5000, coco_cd=500, waterbirds_seg=5322)
REPLICATES = 10000
SEED = 20261102


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_array(record, classes=None):
    require(set(record) == {"intersection", "union"}, "Wrong class-count keys")
    i, u = record["intersection"], record["union"]
    require(isinstance(i, list) and isinstance(u, list) and len(i) == len(u) > 0,
            "Class counts require nonempty equal-length lists")
    require(classes is None or len(i) == classes, "Class-count dimension drift")
    require(all(type(x) is int and 0 <= x < 2**63 for x in i + u), "Invalid integer count")
    require(all(a <= b for a, b in zip(i, u, strict=True)), "Intersection exceeds union")
    return np.asarray([i, u], dtype=np.int64)


def pooled(counts):
    i, u = np.moveaxis(np.asarray(counts), -2, 0)
    valid = u > 0
    require(bool(valid.any(axis=-1).all()), "No union in a bootstrap replicate")
    ratios = np.divide(i, u, out=np.zeros_like(u, dtype=np.float64), where=valid)
    return 100 * ratios.sum(axis=-1) / valid.sum(axis=-1)


def paired_bootstrap(left, right, *, replicates, seed, chunk=32):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.ndim == 3 and left.shape[1] == 2
            and left.shape[0] >= 2, "Paired episode arrays required")
    require(type(replicates) is int and replicates >= 2 and type(chunk) is int and chunk > 0,
            "Invalid resampling count/chunk")
    require(left.dtype == right.dtype == np.int64 and bool((left >= 0).all())
            and bool((right >= 0).all()), "Nonnegative integer counts required")
    n = len(left)
    require(max(int(left.max()), int(right.max())) <= (2**63 - 1) // n,
            "Pooled/resampled int64 count overflow")
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    # Each replicate samples whole episodes once, shared by both arms/classes.
    # Chunks bound indexed temporary arrays, not the statistical sampling unit.
    for start in range(0, replicates, chunk):
        indices = rng.integers(0, n, size=(min(chunk, replicates-start), n))
        a = left[indices].sum(axis=1, dtype=np.int64)
        b = right[indices].sum(axis=1, dtype=np.int64)
        values[start:start+len(indices)] = pooled(a) - pooled(b)
    return values


def source_spec(root, dataset, left, right):
    output = root / "full" / dataset
    files = ("SCORED_SEALED.json", "PREDICTIONS_SEALED.json", "result.json", "scores_by_episode.jsonl")
    bindings = {name: sha256(output / name) for name in files}
    seal = read_json(output / "SCORED_SEALED.json")
    require(seal["complete"] is True and seal["prediction_seal_sha256"] == bindings["PREDICTIONS_SEALED.json"],
            "Score/prediction seal mismatch")
    for name in ("result.json", "scores_by_episode.jsonl"):
        require(seal["artifact_sha256"][name] == bindings[name], "Unsealed score input: " + name)
    barrier = root / "ALL_FULL_PREDICTIONS_SEALED.json"
    require(sha256(barrier) == seal["barrier_sha256"], "Prediction-before-score barrier differs")
    return dict(dataset=dataset, root=str(root), left=left, right=right,
        images=POPULATIONS[dataset], files_sha256=bindings,
        barrier_sha256=sha256(barrier), suite_seal_sha256=sha256(root / "SUITE_SEALED.json"),
        prepared_sha256=sha256(root / "SUITE_PREPARED.json"),
        scope="historically development-exposed; fixed observed episodes, no re-inference")


def load_counts(spec):
    output = Path(spec["root"]) / "full" / spec["dataset"]
    for name in ("result.json", "scores_by_episode.jsonl"):
        require(sha256(output / name) == spec["files_sha256"][name], "Frozen analysis input changed")
    result = read_json(output / "result.json")
    names = (spec["left"], spec["right"])
    canonical = [count_array({key: result["arm_results"][name][key]
                             for key in ("intersection", "union")}) for name in names]
    classes = canonical[0].shape[-1]
    rows = [[], []]
    images = 0
    with (output / "scores_by_episode.jsonl").open() as stream:
        for episode, line in enumerate(stream):
            require(len(line) < 1024 * 1024, "Oversized score record")
            row = json.loads(line)
            require(type(row["episode"]) is int and row["episode"] == episode,
                    "Duplicate, skipped or unordered paired episode")
            require(type(row["images"]) is int and row["images"] > 0, "Invalid episode image count")
            images += row["images"]
            for values, name in zip(rows, names, strict=True):
                values.append(count_array(row["arms"][name], classes))
    require(images == spec["images"] == result["images"], "Wrong full population")
    arrays = [np.stack(values) for values in rows]
    for array, expected, name in zip(arrays, canonical, names, strict=True):
        require(int(array.max()) <= (2**63-1)//len(array), "Pooled int64 count overflow")
        require(np.array_equal(array.sum(axis=0, dtype=np.int64), expected),
                "Episode counts do not reconstruct canonical pooled counts")
        require(abs(float(pooled(expected)) - result["arm_results"][name]["miou_unrounded"]) < 1e-10,
                "Canonical metric formula differs")
    return arrays, images


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output == output.resolve(), "New canonical E08 directory required")
    specs = [source_spec(TUNED, d, "image_C", "image_confidence") for d in POPULATIONS]
    specs += [source_spec(ALLOCATION, d, "image_region", "image_shuffled") for d in ("coco_object", "pc60")]
    output.mkdir(parents=True)
    protocol = dict(experiment="E08", created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(Path(__file__).resolve()): sha256(__file__),
            str(Path(__file__).with_name("intake.py").resolve()): sha256(Path(__file__).with_name("intake.py"))},
        replicates=REPLICATES, seed=SEED, generator="PCG64",
        interval="pointwise 95% percentile; numpy linear quantile; no multiplicity adjustment",
        resampling_unit="whole observed episode, shared indices for paired arms and all classes",
        class_policy="recompute native pooled mIoU, omitting union-zero classes per replicate/arm",
        uncertainty_scope="conditional on frozen settings, predictions and observed episode composition; not training/order uncertainty",
        future_required="Apply the same estimator to E02 contrasts when new predictions are sealed.",
        sources=specs, labels_or_predictions_decoded=False, new_gpu_work=False)
    write_new(output / "PROTOCOL.json", protocol)
    write_new(output / "PREPARED.json", dict(protocol_sha256=sha256(output / "PROTOCOL.json"),
        stage="prepared_not_analyzed", source_comparisons=len(specs)))
    return protocol


def run(output):
    output = Path(output).resolve(strict=True)
    prepared = read_json(output / "PREPARED.json")
    require(prepared["protocol_sha256"] == sha256(output / "PROTOCOL.json"), "Protocol drift")
    protocol = read_json(output / "PROTOCOL.json")
    for path, expected in protocol["code_sha256"].items():
        require(sha256(path) == expected, "Analysis code changed after freeze")
    write_new(output / "STARTED.json", dict(protocol_sha256=prepared["protocol_sha256"], retry_allowed=False))
    rows = []
    for spec in protocol["sources"]:
        (left, right), images = load_counts(spec)
        key = spec["dataset"] + ":" + spec["left"] + ":" + spec["right"]
        stream_seed = int.from_bytes(hashlib.sha256(f"{protocol['seed']}:{key}".encode()).digest()[:8], "big")
        values = paired_bootstrap(left, right, replicates=protocol["replicates"], seed=stream_seed)
        name = key.replace(":", "__") + ".npy"
        with (output / name).open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
        point_left, point_right = float(pooled(left.sum(axis=0))), float(pooled(right.sum(axis=0)))
        low, high = np.quantile(values, [.025, .975], method="linear")
        rows.append(dict(dataset=spec["dataset"], left=spec["left"], right=spec["right"],
            images=images, episodes=len(left), left_miou=point_left, right_miou=point_right,
            delta_pp=point_left-point_right, lower95_pp=float(low), upper95_pp=float(high),
            positive_replicate_fraction=float((values > 0).mean()),
            replicate_distribution=name, stream_seed=stream_seed))
    with (output / "summary.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_new(output / "result.json", dict(experiment="E08", existing_prediction_contrasts_complete=True,
        all_fifty_complete=False, new_inference=False, rows=rows,
        uncertainty_scope=protocol["uncertainty_scope"], future_required=protocol["future_required"]))
    names = ["PROTOCOL.json", "PREPARED.json", "STARTED.json", "summary.csv", "result.json"]
    names += [r["replicate_distribution"] for r in rows]
    write_new(output / "ANALYSIS_SEALED.json", dict(complete=True, experiment="E08",
        scope="ten existing-prediction contrasts; future E02 contrasts pending",
        artifact_sha256={name: sha256(output / name) for name in names}))
    print(json.dumps(dict(output=str(output), comparisons=len(rows), complete=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
        print(json.dumps(dict(output=str(args.output), stage="prepared_not_analyzed")))
    else:
        run(args.output)


if __name__ == "__main__":
    main()
