"""V01 full-population fixed-grid prediction core, no GT/admission/launch/seal.

The caller must provide frozen input/configuration, actual numerical gates,
shared GPU ownership, capped resources and failure-time method disposal.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from tools.evolution import run_cell as shared
from . import generalization_worker as prediction
from . import value_runtime as runtime
from .bootstrap import POPULATIONS, require
from .intake import sha256


def execute_dataset(method, data, rows, spec, suite, budget, progress):
    dataset, cfg = spec["dataset"], spec["config"]
    require(dataset in runtime.NATIVE_STEPS and len(data) == len(rows) == POPULATIONS[dataset]
            and len({r["official_id"] for r in rows}) == len(rows), "V01 requires the whole fixed population")
    taxonomy = dict(coco_object="COCOObjectDataset", pc60="PascalContext60Dataset")
    points = runtime.points(dataset)
    require(type(spec["points"]) is list, "V01 serialized grid must be an explicit list")
    for point in spec["points"]: runtime.validate_point(dataset, point)
    require(cfg["dataset"] == taxonomy[dataset] and type(cfg["steps"]) is int
            and cfg["steps"] == runtime.NATIVE_STEPS[dataset]
            and spec["points"] == list(points) and spec["backbone"] == "ViT-L/14"
            and spec["model_type"] == "naclip" and spec["preprocessing"] == "native_PIL_bilinear"
            and tuple(spec["vocabulary"]["queries"]) == tuple(method.classes)
            and type(method) is runtime.native.ExperimentTRACE,
            "V01 fixed grid/model/vocabulary/input differs")
    output, inventory = suite/dataset, {}
    output.mkdir()
    for point in points:
        (output/"predictions"/point["id"]).mkdir(parents=True)
    (output/"episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    frozen_digest = shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad})
    legal = runtime.native.legal_parameters(method)
    prediction.write_payload(output/"initial_state.json", dict(reset=initial, legal_visual_LN=legal,
        frozen_parameter_sha256=frozen_digest), inventory, suite, budget)
    costs, names = [], tuple(p["id"] for p in points)
    by_name = {point["id"]:point for point in points}
    with (output/"cost.jsonl").open("x", encoding="utf-8") as stream:
        for episode, start in enumerate(range(0, len(rows), cfg["batch_size"])):
            stop = min(start+cfg["batch_size"], len(rows))
            images, meta, valid, ids = prediction.episode_batch(data, start, stop, method, cfg)
            require(list(meta["img_path"]) == [r["image"] for r in rows[start:stop]], "V01 observed source order differs")
            runtime.validate_episode(dataset,images,valid,ids,meta,cfg)
            rng, identity = shared.rng_state(method.device), None
            for position, name in enumerate(prediction.runtime.prior.arm_order(names, episode, "alternating")):
                point = by_name[name]
                labels, stats, losses, row = runtime.run_point(method, dataset, point, images, valid, ids, meta,
                    cfg, spec["vocabulary"]["query_to_class"], initial_rng=rng)
                require(method.steps == cfg["steps"] and shared.restored_state(method) == initial,
                        "V01 point left budget/Adam/theta0 mutated")
                row.update(weighting_arm=point["arm"], arm=name, episode=episode,
                           order_position=position, warmup=episode == 0)
                identity = row["candidate_sha256"] if identity is None else identity
                require(row["candidate_sha256"] == identity, "V01 candidate identity changed across fixed budget points")
                require(all(p._version == version for p,version in frozen), "V01 frozen weights changed")
                require(len(labels) == stop-start, "Missing V01 source predictions")
                for index, label in enumerate(labels, start):
                    require(label.dtype == np.uint16 and label.shape == tuple(cfg["resize"])
                            and bool((label < len(spec["vocabulary"]["classes"])).all()), "Invalid mapped V01 labels")
                    prediction.write_payload(output/"predictions"/name/f"{index:06d}.npy", label, inventory, suite, budget)
                candidates = runtime.native.image_worker.candidate_receipts(stats, point["arm"],
                                                                            spec["vocabulary"]["query_to_class"])
                prediction.write_payload(output/"episodes"/f"{episode:06d}.{name}.json", dict(**row,
                    losses=losses, candidates=candidates, fixed_point=point,
                    receipt_float64_objective_coefficients_are_ideal_not_runtime=True,
                    ordered_ids=[r["official_id"] for r in rows[start:stop]]), inventory, suite, budget)
                line = json.dumps(row, allow_nan=False)+"\n"
                size = len(line.encode())
                require(shutil.disk_usage(suite).free >= prediction.RESERVE+size
                        and budget["bytes"]+size <= prediction.MAX_PAYLOAD_BYTES, "V01 cost storage budget exceeded")
                require(stream.write(line) == len(line), "Short V01 cost write")
                stream.flush()
                budget["bytes"] += size
                costs.append(row)
            progress(dict(dataset=dataset, completed_sources=stop, total_sources=len(rows),
                          completed_source_episodes=episode+1, stored_bytes=budget["bytes"], GT_opened=False))
            del images, meta, valid, ids
        stream.flush()
        os.fsync(stream.fileno())
    require(shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad}) == frozen_digest,
            "V01 frozen parameter content changed")
    inventory[str((output/"cost.jsonl").relative_to(suite))] = sha256(output/"cost.jsonl")
    prediction.write_payload(output/"prediction_result.json", dict(dataset=dataset, images=len(rows), points=list(points),
        actual_work={name:{k:sum(r[k] for r in costs if r["arm"] == name)
                          for k in (*runtime.native.COUNTERS, "optimizer_steps")} for name in names},
        raw_model_seconds={name:sum(r["seconds"] for r in costs if r["arm"] == name) for name in names},
        warmup_adjusted_throughput=prediction.runtime.prior.timing_summary(costs, names),
        legal_visual_LN=legal, frozen_parameter_sha256=frozen_digest, GT_opened=False,
        time_matching_done=False, accuracy_or_score_selected_points=False,
        metric_status="Complete fixed grid, no time selection or GT/accuracy; episode0 retained"), inventory, suite, budget)
    episodes = (len(rows)+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(costs) == episodes*len(points)
            and all(sum(r["images"] for r in costs if r["arm"] == name) == len(rows) for name in names),
            "Incomplete V01 fixed grid population")
    expected = {f"{dataset}/predictions/{name}/{i:06d}.npy" for name in names for i in range(len(rows))}
    expected |= {f"{dataset}/episodes/{e:06d}.{name}.json" for name in names for e in range(episodes)}
    expected |= {f"{dataset}/{n}" for n in ("initial_state.json", "cost.jsonl", "prediction_result.json")}
    require(set(inventory) == expected, "V01 full population inventory differs")
    return dict(payloads_complete=True, scientific_completion=False, GT_opened=False, dataset=dataset,
        images=len(rows), points=list(points), image_method_evaluations=len(rows)*len(points), artifact_sha256=inventory)
