"""Full M02 dataset execution core, not an independently admitted launcher.

The controller must supply a frozen manifest, real RGB gates, the shared GPU
lock and independent artifact admission. This core never reads annotations or
publishes a completion seal; its caller must release the lock before sealing.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from tools.evolution import run_cell as shared
from . import generalization_worker as prediction
from . import spatial_runtime as runtime
from .bootstrap import POPULATIONS, require
from .intake import sha256


def execute_dataset(method, data, rows, spec, suite, budget, progress):
    dataset, cfg = spec["dataset"], spec["config"]
    require(dataset in ("coco_object", "pc60") and len(data) == len(rows) == POPULATIONS[dataset]
            and len({r["official_id"] for r in rows}) == len(rows), "M02 requires the entire fixed dataset")
    require(spec["modes"] == list(runtime.MODES) and spec["preprocessing"] == "native_PIL_bilinear"
            and spec["backbone"] == "ViT-L/14" and spec["model_type"] == "naclip",
            "M02 model/arms/preprocessing differs")
    output, inventory = suite/dataset, {}
    output.mkdir()
    for mode in runtime.MODES:
        (output/"predictions"/mode).mkdir(parents=True)
    (output/"episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    frozen_digest = shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad})
    legal = runtime.legal_parameters(method)
    prediction.write_payload(output/"initial_state.json", dict(reset=initial,legal_visual_LN=legal,
        frozen_parameter_sha256=frozen_digest), inventory, suite, budget)
    all_costs = []
    with (output/"cost.jsonl").open("x", encoding="utf-8") as stream:
        for episode, start in enumerate(range(0, len(rows), cfg["batch_size"])):
            stop = min(start+cfg["batch_size"], len(rows))
            images, meta, valid, ids = prediction.episode_batch(data,start,stop,method,cfg)
            require(list(meta["img_path"]) == [r["image"] for r in rows[start:stop]], "M02 observed source order differs")
            rng, identity = shared.rng_state(method.device), None
            for position, mode in enumerate(prediction.runtime.prior.arm_order(runtime.MODES,episode,"alternating")):
                shared.restore_rng(rng,method.device)
                labels, stats, losses, row, geometry = runtime.run_arm(method,mode,images,valid,ids,meta,cfg,
                    spec["vocabulary"]["query_to_class"])
                row.update(arm=mode,episode=episode,order_position=position,warmup=episode == 0)
                identity = row["candidate_sha256"] if identity is None else identity
                require(row["candidate_sha256"] == identity, "M02 changed original candidates across spatial modes")
                require(all(p._version == version for p,version in frozen), "M02 frozen weights changed")
                require(len(labels) == stop-start, "M02 missing full-source prediction")
                for index, label in enumerate(labels,start):
                    require(label.dtype == np.uint16 and label.shape == tuple(cfg["resize"])
                            and bool((label < len(spec["vocabulary"]["classes"])).all()), "Invalid mapped M02 labels")
                    prediction.write_payload(output/"predictions"/mode/f"{index:06d}.npy",label,inventory,suite,budget)
                # This alias selects the unchanged image-balanced coefficient formula,
                # not the identity/name of the diagnostic spatial experiment.
                coefficient_arm = "image_confidence" if mode == "confidence" else "image_C"
                candidates = runtime.image_worker.candidate_receipts(stats,coefficient_arm,
                    spec["vocabulary"]["query_to_class"])
                prediction.write_payload(output/"episodes"/f"{episode:06d}.{mode}.json",dict(**row,
                    losses=losses,candidates=candidates,preservation_geometry=geometry,
                    ordered_ids=[r["official_id"] for r in rows[start:stop]],GT_opened=False,
                    scientific_identity="M02_spatial_"+mode),inventory,suite,budget)
                line = json.dumps(row,allow_nan=False)+"\n"
                size = len(line.encode())
                require(shutil.disk_usage(suite).free >= prediction.RESERVE+size
                        and budget["bytes"]+size <= prediction.MAX_PAYLOAD_BYTES, "M02 raw cost exceeds disk budget")
                require(stream.write(line) == len(line), "Short M02 cost write")
                stream.flush()
                budget["bytes"] += size
                all_costs.append(row)
                method.reset()
                require(shared.restored_state(method) == initial, "M02 episode/mode reset differs")
            progress(dict(dataset=dataset,completed_sources=stop,total_sources=len(rows),
                completed_source_episodes=episode+1,stored_bytes=budget["bytes"],GT_opened=False))
            del images,meta,valid,ids
        stream.flush()
        os.fsync(stream.fileno())
    require(shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad}) == frozen_digest,
            "M02 full-dataset frozen parameter bytes changed")
    inventory[str((output/"cost.jsonl").relative_to(suite))] = sha256(output/"cost.jsonl")
    prediction.write_payload(output/"prediction_result.json",dict(dataset=dataset,images=len(rows),modes=list(runtime.MODES),
        actual_work={mode:{k:sum(r[k] for r in all_costs if r["mode"] == mode)
                          for k in (*runtime.image_worker.COUNTERS,"optimizer_steps")} for mode in runtime.MODES},
        raw_model_seconds={mode:sum(r["seconds"] for r in all_costs if r["mode"] == mode) for mode in runtime.MODES},
        warmup_adjusted_throughput=prediction.runtime.prior.timing_summary(all_costs,runtime.MODES),
        legal_visual_LN=legal,frozen_parameter_sha256=frozen_digest,GT_opened=False,
        metric_status="No GT or score; all source predictions including episode0 retained"),inventory,suite,budget)
    episodes = (len(rows)+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(all_costs) == episodes*len(runtime.MODES)
            and all(sum(r["images"] for r in all_costs if r["mode"] == mode) == len(rows) for mode in runtime.MODES),
            "Incomplete M02 full population")
    expected = {f"{dataset}/predictions/{mode}/{i:06d}.npy" for mode in runtime.MODES for i in range(len(rows))}
    expected |= {f"{dataset}/episodes/{e:06d}.{mode}.json" for mode in runtime.MODES for e in range(episodes)}
    expected |= {f"{dataset}/{n}" for n in ("initial_state.json","cost.jsonl","prediction_result.json")}
    require(set(inventory) == expected, "M02 full dataset artifact inventory differs")
    return dict(payloads_complete=True,scientific_completion=False,GT_opened=False,dataset=dataset,
        images=len(rows),modes=list(runtime.MODES),image_method_evaluations=len(rows)*len(runtime.MODES),
        artifact_sha256=inventory)
