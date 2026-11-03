"""M03 full-population prediction core, not an admitted launcher.

The caller supplies independently admitted frozen inputs, actual RGB gates,
the shared GPU lock and failure-time method disposal. No GT or seal here.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from tools.evolution import run_cell as shared
from . import generalization_worker as prediction
from . import legacy_runtime as runtime
from .bootstrap import POPULATIONS, require
from .intake import sha256


def execute_dataset(method, data, rows, spec, suite, budget, progress):
    dataset, cfg = spec["dataset"], spec["config"]
    require(isinstance(method, runtime.LegacyExperimentTRACE)
            and dataset in ("coco_object", "pc60")
            and len(data) == len(rows) == POPULATIONS[dataset]
            and len({r["official_id"] for r in rows}) == len(rows),
            "M03 requires the entire fixed dataset and legacy-capable method")
    require(spec["modes"] == list(runtime.MODES)
            and spec["preprocessing"] == "native_PIL_bilinear"
            and spec["backbone"] == "ViT-L/14" and spec["model_type"] == "naclip",
            "M03 model/conditions/preprocessing differs")
    vocabulary = spec["vocabulary"]
    taxonomy = {"coco_object":"COCOObjectDataset", "pc60":"PascalContext60Dataset"}[dataset]
    require(cfg["dataset"] == method.legacy_dataset_name == taxonomy
            and tuple(vocabulary["classes"]) == method.legacy_benchmark_classes
            and tuple(vocabulary["queries"]) == tuple(method.classes)
            and runtime.effective_mapping(vocabulary["query_to_class"], len(method.classes),
                                          len(method.legacy_benchmark_classes))
                == method.legacy_effective_mapping,
            "M03 full vocabulary differs from bound score roles")
    output, inventory = suite/dataset, {}
    output.mkdir()
    for mode in runtime.MODES:
        (output/"predictions"/mode).mkdir(parents=True)
    (output/"episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    frozen_digest = shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad})
    legal = runtime.native.legal_parameters(method)
    prediction.write_payload(output/"initial_state.json", dict(reset=initial, legal_visual_LN=legal,
        frozen_parameter_sha256=frozen_digest, legacy_archive_sha256=method.legacy["archive_sha256"],
        legacy_source_members=method.legacy["member_sha256"], episode_objective_sha256=runtime.OBJECTIVE_SHA),
        inventory, suite, budget)
    costs = []
    with (output/"cost.jsonl").open("x", encoding="utf-8") as stream:
        for episode, start in enumerate(range(0, len(rows), cfg["batch_size"])):
            stop = min(start+cfg["batch_size"], len(rows))
            images, meta, valid, ids = prediction.episode_batch(data, start, stop, method, cfg)
            require(list(meta["img_path"]) == [r["image"] for r in rows[start:stop]],
                    "M03 observed full-source order differs")
            rng, identity = shared.rng_state(method.device), None
            for position, mode in enumerate(prediction.runtime.prior.arm_order(runtime.MODES, episode, "alternating")):
                shared.restore_rng(rng, method.device)
                labels, stats, losses, row = runtime.run_arm(method, mode, images, valid, ids, meta, cfg,
                                                            vocabulary["query_to_class"])
                row.update(episode=episode, order_position=position, warmup=episode == 0)
                identity = row["candidate_sha256"] if identity is None else identity
                require(row["candidate_sha256"] == identity, "M03 changed original candidates across conditions")
                require(all(p._version == version for p,version in frozen), "M03 frozen weights changed")
                require(len(labels) == stop-start, "M03 missing full-source prediction")
                for index, label in enumerate(labels, start):
                    require(label.dtype == np.uint16 and label.shape == tuple(cfg["resize"])
                            and bool((label < len(vocabulary["classes"])).all()), "Invalid mapped M03 labels")
                    prediction.write_payload(output/"predictions"/mode/f"{index:06d}.npy", label,
                                             inventory, suite, budget)
                # These aliases select normalization only, never the diagnostic identity.
                coefficient_arm = "episode_min" if mode.endswith(".episode") else "image_C"
                candidates = runtime.native.image_worker.candidate_receipts(stats, coefficient_arm,
                                                                            vocabulary["query_to_class"])
                prediction.write_payload(output/"episodes"/f"{episode:06d}.{mode}.json", dict(**row,
                    losses=losses, candidates=candidates,
                    receipt_float64_objective_coefficients_are_ideal_not_runtime=True,
                    ordered_ids=[r["official_id"] for r in rows[start:stop]], scientific_identity="M03_"+mode),
                    inventory, suite, budget)
                line = json.dumps(row, allow_nan=False)+"\n"
                size = len(line.encode())
                require(shutil.disk_usage(suite).free >= prediction.RESERVE+size
                        and budget["bytes"]+size <= prediction.MAX_PAYLOAD_BYTES,
                        "M03 raw cost exceeds disk budget")
                require(stream.write(line) == len(line), "Short M03 cost write")
                stream.flush()
                budget["bytes"] += size
                costs.append(row)
                method.reset()
                require(shared.restored_state(method) == initial, "M03 exact episode/condition reset differs")
            progress(dict(dataset=dataset, completed_sources=stop, total_sources=len(rows),
                          completed_source_episodes=episode+1, stored_bytes=budget["bytes"], GT_opened=False))
            del images, meta, valid, ids
        stream.flush()
        os.fsync(stream.fileno())
    require(shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad}) == frozen_digest,
            "M03 full-dataset frozen parameter bytes changed")
    inventory[str((output/"cost.jsonl").relative_to(suite))] = sha256(output/"cost.jsonl")
    prediction.write_payload(output/"prediction_result.json", dict(dataset=dataset, images=len(rows),
        modes=list(runtime.MODES), legal_visual_LN=legal, frozen_parameter_sha256=frozen_digest,
        legacy_archive_sha256=method.legacy["archive_sha256"], legacy_source_members=method.legacy["member_sha256"],
        episode_objective_sha256=runtime.OBJECTIVE_SHA,
        actual_work={mode:{k:sum(r[k] for r in costs if r["arm"] == mode)
                          for k in (*runtime.native.COUNTERS, "optimizer_steps")} for mode in runtime.MODES},
        raw_model_seconds={mode:sum(r["seconds"] for r in costs if r["arm"] == mode) for mode in runtime.MODES},
        warmup_adjusted_throughput=prediction.runtime.prior.timing_summary(costs, runtime.MODES),
        GT_opened=False, metric_status="No GT/score; current candidate pool, not whole historical reproduction"),
        inventory, suite, budget)
    episodes = (len(rows)+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(costs) == episodes*len(runtime.MODES)
            and all(sum(r["images"] for r in costs if r["arm"] == mode) == len(rows) for mode in runtime.MODES),
            "Incomplete M03 full population")
    expected = {f"{dataset}/predictions/{mode}/{i:06d}.npy" for mode in runtime.MODES for i in range(len(rows))}
    expected |= {f"{dataset}/episodes/{e:06d}.{mode}.json" for mode in runtime.MODES for e in range(episodes)}
    expected |= {f"{dataset}/{n}" for n in ("initial_state.json", "cost.jsonl", "prediction_result.json")}
    require(set(inventory) == expected, "M03 full dataset artifact inventory differs")
    return dict(payloads_complete=True, scientific_completion=False, GT_opened=False, dataset=dataset,
                images=len(rows), modes=list(runtime.MODES), image_method_evaluations=len(rows)*len(runtime.MODES),
                artifact_sha256=inventory)
