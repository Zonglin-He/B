"""TRACE mechanism experiment 5: offline transitions from sealed full predictions.

No inference, adaptation, candidate selection or soft-support reconstruction.
Only the two declared full datasets are streamed once, after protocol freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .bootstrap import TUNED, POPULATIONS, read_json, require
from .intake import sha256, write_new

DATASETS = ("coco_object", "pc60")
ARMS = ("image_uniform", "image_confidence", "image_C")
FIELDS = ("valid", "pre_correct", "pre_wrong", "correct_to_wrong", "wrong_to_correct")
REPLICATES, SEED = 10000, 20261102


def transitions(before, after, gt, classes):
    require(before.shape == after.shape == gt.shape and gt.ndim == 2,
            "Prediction/GT geometry differs")
    require(type(classes) is int and classes > 0, "Invalid class count")
    for array in (before, after, gt):
        require(np.issubdtype(array.dtype, np.integer), "Integer labels required")
    valid = gt != 255
    require(bool(((gt[valid] >= 0) & (gt[valid] < classes)).all()), "Invalid GT labels")
    for array in (before, after):
        require(bool(((array >= 0) & (array < classes)).all()), "Invalid prediction labels")
    a, b = before[valid] == gt[valid], after[valid] == gt[valid]
    return np.asarray([valid.sum(), a.sum(), (~a).sum(), (a & ~b).sum(), (~a & b).sum()],
                      dtype=np.int64)


def rates(counts):
    x = np.asarray(counts)
    require(x.shape[-1] == len(FIELDS) and np.issubdtype(x.dtype, np.integer)
            and bool((x >= 0).all()), "Invalid transition counts")
    n, correct, wrong, damaged, fixed = np.moveaxis(x, -1, 0)
    require(bool((correct + wrong == n).all()) and bool((damaged <= correct).all())
            and bool((fixed <= wrong).all()), "Impossible transition counts")

    def divide(a, b):
        return np.divide(100. * a, b, out=np.full(np.shape(a), np.nan), where=b > 0)

    return dict(correct_retention_pct=divide(correct-damaged, correct),
                wrong_correction_pct=divide(fixed, wrong),
                correct_to_wrong_of_pre_correct_pct=divide(damaged, correct),
                correct_to_wrong_of_valid_pct=divide(damaged, n),
                net_correct_change_pp=divide(fixed-damaged, n))


def confusion(prediction, gt, classes):
    valid = gt != 255
    matrix = np.bincount(gt[valid] * classes + prediction[valid], minlength=classes*classes)
    matrix = matrix.reshape(classes, classes)
    i = np.diag(matrix)
    return np.asarray([i, matrix.sum(0) + matrix.sum(1) - i], dtype=np.int64)


def source_spec(dataset):
    root = TUNED / "full" / dataset
    names = ("prediction_protocol.json", "PREDICTIONS_SEALED.json", "SCORED_SEALED.json",
             "score_inputs.json", "result.json")
    bindings = {name: sha256(root / name) for name in names}
    pred, score = read_json(root / names[1]), read_json(root / names[2])
    require(pred["complete"] is True and pred["gt_contents_opened"] is False
            and score["complete"] is True
            and score["prediction_seal_sha256"] == bindings[names[1]], "Prediction/score seal differs")
    require(pred["artifact_sha256"][names[0]] == bindings[names[0]], "Unsealed prediction protocol")
    for name in ("score_inputs.json", "result.json"):
        require(score["artifact_sha256"][name] == bindings[name], "Unsealed score artifact")
    barrier = TUNED / "ALL_FULL_PREDICTIONS_SEALED.json"
    require(sha256(barrier) == score["barrier_sha256"], "Original prediction-before-GT barrier differs")
    protocol = read_json(root / names[0])
    manifest = protocol["manifest"]
    require(manifest["dataset"] == dataset and manifest["stage"] == "full"
            and manifest["arms"] == ["no_adapt", *ARMS]
            and len(manifest["panel"]["inputs"]) == POPULATIONS[dataset], "Wrong cached full scope")
    return dict(dataset=dataset, root=str(root), files_sha256=bindings,
                images=POPULATIONS[dataset], barrier_sha256=sha256(barrier),
                historical_exposure="full development-exposed, fixed TRACE-selected configuration")


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output == output.resolve(), "New analysis directory required")
    specs = [source_spec(d) for d in DATASETS]
    output.mkdir(parents=True)
    protocol = dict(experiment="TRACE-M05", sources=specs, arms=list(ARMS),
        code_sha256={str(Path(__file__).resolve()): sha256(__file__),
            str(Path(__file__).with_name("intake.py").resolve()): sha256(Path(__file__).with_name("intake.py"))},
        base_prediction="no_adapt: same frozen model reset/geometry/query-to-class mapping",
        population="All5000 Object and5105 PC60 images; original order and episodes",
        gt_policy="Original score label hashes, nearest-neighbor resize and reduce_zero_label; ignore255",
        counts=list(FIELDS), metric_denominators="Retention/damage: pre-correct; correction: pre-wrong; net: valid",
        cluster="Original full evaluation episode; paired identical resampling for every arm",
        replicates=REPLICATES, seed=SEED, intervals="Pointwise95percent percentile, no multiplicity correction",
        uncertainty="Conditional on fixed settings and original episodes; not order/training uncertainty",
        soft_support_available=False, candidate_local_risk_measured=False,
        new_adaptations=0, new_gpu_work=False, no_model_import=True,
        reuse="Only this required pixel-level analysis decodes each relevant prediction once; no broad recertification")
    write_new(output / "PROTOCOL.json", protocol)
    write_new(output / "PREPARED.json", dict(protocol_sha256=sha256(output / "PROTOCOL.json"),
                                           status="prepared_not_analyzed"))


def load_prediction(root, name, inventory, classes):
    path = root / name
    require(path.resolve() == path and path.is_file(), "Aliased/missing prediction")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == inventory[name], "Prediction content differs")
    value = np.load(io.BytesIO(content), allow_pickle=False)
    require(value.ndim == 2 and np.issubdtype(value.dtype, np.integer)
            and bool(((value >= 0) & (value < classes)).all()), "Invalid prediction")
    return value


def intervals(episodes, dataset):
    rng_seed = int.from_bytes(hashlib.sha256(f"M05:{SEED}:{dataset}".encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    samples = {key: [] for key in rates(episodes[0, 0])}
    n = len(episodes)
    require(n >= 2 and int(episodes.max()) <= (2**63-1)//n, "Invalid/overflowing cluster counts")
    for start in range(0, REPLICATES, 32):
        indices = rng.integers(n, size=(min(32, REPLICATES-start), n))
        values = rates(episodes[indices].sum(1, dtype=np.int64))
        for key, value in values.items():
            samples[key].append(value[:, 2]-value[:, 1])
    output = {}
    for key, chunks in samples.items():
        values = np.concatenate(chunks)
        require(bool(np.isfinite(values).all()), "Undefined bootstrap denominator; no replicate skipping")
        lo, hi = np.quantile(values, [.025, .975], method="linear")
        output[key] = dict(lower95_pp=float(lo), upper95_pp=float(hi))
    return output


def run(output):
    output = Path(output).resolve(strict=True)
    protocol = read_json(output / "PROTOCOL.json")
    require(sha256(output / "PROTOCOL.json") == read_json(output / "PREPARED.json")["protocol_sha256"],
            "Analysis protocol drift")
    for name, expected in protocol["code_sha256"].items():
        require(sha256(name) == expected, "Frozen analysis code drift")
    for spec in protocol["sources"]:
        for name, expected in spec["files_sha256"].items():
            require(sha256(Path(spec["root"]) / name) == expected, "Analysis input drift")
        require(sha256(TUNED / "ALL_FULL_PREDICTIONS_SEALED.json") == spec["barrier_sha256"], "Barrier drift")
    write_new(output / "STARTED.json", dict(protocol_sha256=sha256(output / "PROTOCOL.json"), retries=False))
    summary, details = [], {}
    for spec in protocol["sources"]:
        dataset, root = spec["dataset"], Path(spec["root"])
        manifest = read_json(root / "prediction_protocol.json")["manifest"]
        panel, cfg = manifest["panel"], manifest["config"]
        inputs, classes = panel["inputs"], len(panel["classes"])
        inventory = read_json(root / "PREDICTIONS_SEALED.json")["artifact_sha256"]
        labels = read_json(root / "score_inputs.json")["label_sha256"]
        canonical = read_json(root / "result.json")
        require(panel["ordered_ids"] == [r["official_id"] for r in inputs]
                and len(set(panel["ordered_ids"])) == len(inputs) == spec["images"], "Source identity drift")
        totals = {a: np.zeros((2, classes), dtype=np.int64) for a in ("no_adapt", *ARMS)}
        episodes = np.zeros(((len(inputs)+cfg["batch_size"]-1)//cfg["batch_size"], len(ARMS), len(FIELDS)), dtype=np.int64)
        stream_path = output / f"{dataset}.transitions.jsonl"
        with stream_path.open("x") as stream:
            for index, row in enumerate(inputs):
                before = load_prediction(root, f"predictions/no_adapt/{index:06d}.npy", inventory, classes)
                require(before.shape == tuple(cfg["resize"]), "Prediction resolution drift")
                content = Path(row["label"]).read_bytes()
                require(hashlib.sha256(content).hexdigest() == labels[row["label"]], "Original GT hash differs")
                with Image.open(io.BytesIO(content)) as image:
                    gt = np.asarray(image.resize((before.shape[1], before.shape[0]), Image.Resampling.NEAREST))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.int64, copy=True)
                if panel["reduce_zero_label"]:
                    background, valid = gt == 0, gt != 255
                    gt[valid] -= 1
                    gt[background] = 255
                transitions(before, before, gt, classes)
                totals["no_adapt"] += confusion(before, gt, classes)
                counts = {}
                for arm_index, arm in enumerate(ARMS):
                    after = load_prediction(root, f"predictions/{arm}/{index:06d}.npy", inventory, classes)
                    value = transitions(before, after, gt, classes)
                    episodes[index//cfg["batch_size"], arm_index] += value
                    totals[arm] += confusion(after, gt, classes)
                    counts[arm] = dict(zip(FIELDS, map(int, value), strict=True))
                stream.write(json.dumps(dict(index=index, source_id=row["official_id"],
                    episode=index//cfg["batch_size"], arms=counts)) + "\n")
        for arm, counts in totals.items():
            expected = canonical["arm_results"][arm]
            require(np.array_equal(counts, np.asarray([expected["intersection"], expected["union"]])),
                    "Decoded prediction/GT does not reconstruct original canonical I/U: " + arm)
        ci = intervals(episodes, dataset)
        aggregate = episodes.sum(0, dtype=np.int64)
        metric_values = rates(aggregate)
        for arm_index, arm in enumerate(ARMS):
            summary.append(dict(dataset=dataset, arm=arm, images=len(inputs),
                **dict(zip(FIELDS, map(int, aggregate[arm_index]), strict=True)),
                **{k: None if not np.isfinite(v[arm_index]) else float(v[arm_index]) for k, v in metric_values.items()},
                miou=canonical["arm_results"][arm]["miou_unrounded"]))
        details[dataset] = dict(images=len(inputs), episodes=len(episodes), all_original_pooled_counts_match=True,
            C_minus_Confidence={key: dict(delta_pp=float(value[2]-value[1]), **ci[key])
                                for key, value in metric_values.items()})
        write_new(output / f"{dataset}.completed.json", details[dataset])
        print(json.dumps(dict(dataset=dataset, images=len(inputs), completed=True)), flush=True)
    with (output / "summary.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_new(output / "result.json", dict(experiment="TRACE-M05", status="completed", datasets=details,
        rows=summary, scope="Full Object/PC60 fixed TRACE-selected settings; no independent calibration or new inference",
        true_class_soft_support_not_measured=True, candidate_local_effects_not_measured=True))
    names = ["PROTOCOL.json", "PREPARED.json", "STARTED.json", "summary.csv", "result.json"]
    names += [f"{d}.{suffix}" for d in DATASETS for suffix in ("transitions.jsonl", "completed.json")]
    write_new(output / "ANALYSIS_SEALED.json", dict(complete=True, new_gpu_work=False,
        protocol_sha256=sha256(output / "PROTOCOL.json"), artifact_sha256={n: sha256(output/n) for n in names}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.phase](args.output)
