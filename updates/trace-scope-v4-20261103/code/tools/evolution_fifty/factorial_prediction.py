"""M04 full prediction stream, requiring a separately admitted controller.

Natural and common-active predictions are distinct conditions. This core owns
no GPU lock, checkpoint loader, admission, annotation reader or seal publisher.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from tools.evolution import run_cell as shared
from . import factorial_runtime as runtime
from . import generalization_worker as prediction
from .bootstrap import POPULATIONS,require
from .intake import sha256

STUDIES = ("natural","common_active")
CONDITIONS = tuple((study,arm) for study in STUDIES for arm in runtime.ARMS)
MODES = tuple(study+"."+arm for study,arm in CONDITIONS)


def execute_dataset(method,data,rows,spec,suite,budget,progress):
    dataset,cfg = spec["dataset"],spec["config"]
    require(dataset in ("coco_object","pc60") and len(data) == len(rows) == POPULATIONS[dataset]
            and len({r["official_id"] for r in rows}) == len(rows)
            and all(isinstance(r["official_id"],str) and r["official_id"] for r in rows),
            "M04 requires every fixed dataset source and original string IDs")
    require(spec["studies"] == list(STUDIES) and spec["factorial_arms"] == list(runtime.ARMS)
            and spec["preprocessing"] == "native_PIL_bilinear" and spec["backbone"] == "ViT-L/14"
            and spec["model_type"] == "naclip", "M04 declared model/studies/arms/preprocessing differs")
    output,inventory = suite/dataset,{}
    output.mkdir()
    for mode in MODES:
        (output/"predictions"/mode).mkdir(parents=True)
    (output/"episodes").mkdir()
    method.reset()
    initial = shared.restored_state(method)
    frozen = [(p,p._version) for p in method.model.parameters() if not p.requires_grad]
    frozen_digest = shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad})
    legal = runtime.native.legal_parameters(method)
    prediction.write_payload(output/"initial_state.json",dict(reset=initial,legal_visual_LN=legal,
        frozen_parameter_sha256=frozen_digest),inventory,suite,budget)
    costs = []
    with (output/"cost.jsonl").open("x",encoding="utf-8") as stream:
        for episode,start in enumerate(range(0,len(rows),cfg["batch_size"])):
            stop = min(start+cfg["batch_size"],len(rows))
            images,meta,valid,ids = prediction.episode_batch(data,start,stop,method,cfg)
            require(list(meta["img_path"]) == [r["image"] for r in rows[start:stop]], "M04 observed source order differs")
            source_ids = [r["official_id"] for r in rows[start:stop]]
            rng,identity = shared.rng_state(method.device),None
            for position,mode in enumerate(prediction.runtime.prior.arm_order(MODES,episode,"alternating")):
                study,arm = mode.split(".")
                shared.restore_rng(rng,method.device)
                labels,stats,losses,row,assignment,coefficients = runtime.run_arm(method,arm,
                    study == "common_active",images,valid,ids,meta,cfg,spec["vocabulary"]["query_to_class"],source_ids)
                row.update(arm=mode,study=study,episode=episode,order_position=position,warmup=episode == 0)
                identity = row["candidate_sha256"] if identity is None else identity
                require(row["candidate_sha256"] == identity, "M04 original candidates changed across arms/studies")
                require(all(p._version == version for p,version in frozen), "M04 frozen weights changed")
                require(len(labels) == stop-start, "M04 missing source prediction")
                for index,label in enumerate(labels,start):
                    require(label.dtype == np.uint16 and label.shape == tuple(cfg["resize"])
                            and bool((label < len(spec["vocabulary"]["classes"])).all()), "Invalid mapped M04 labels")
                    prediction.write_payload(output/"predictions"/mode/f"{index:06d}.npy",label,inventory,suite,budget)
                candidates = runtime.native.image_worker.candidate_receipts(stats,"image_C",spec["vocabulary"]["query_to_class"])
                prediction.write_payload(output/"episodes"/f"{episode:06d}.{mode}.json",dict(**row,losses=losses,
                    candidates=candidates,assignment=assignment,native_dtype_budget=coefficients,
                    receipt_float64_objective_coefficients_are_ideal_not_runtime=True,
                    ordered_ids=source_ids,scientific_identity="M04_"+mode,GT_opened=False),inventory,suite,budget)
                line = json.dumps(row,allow_nan=False)+"\n"
                size = len(line.encode())
                require(shutil.disk_usage(suite).free >= prediction.RESERVE+size
                        and budget["bytes"]+size <= prediction.MAX_PAYLOAD_BYTES, "M04 raw cost exceeds disk budget")
                require(stream.write(line) == len(line), "Short M04 raw cost write")
                stream.flush()
                budget["bytes"] += size
                costs.append(row)
                method.reset()
                require(shared.restored_state(method) == initial, "M04 episode/condition exact reset differs")
            progress(dict(dataset=dataset,completed_sources=stop,total_sources=len(rows),
                completed_source_episodes=episode+1,stored_bytes=budget["bytes"],GT_opened=False))
            del images,meta,valid,ids
        stream.flush()
        os.fsync(stream.fileno())
    require(shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad}) == frozen_digest,
            "M04 frozen parameter bytes changed")
    inventory[str((output/"cost.jsonl").relative_to(suite))] = sha256(output/"cost.jsonl")
    prediction.write_payload(output/"prediction_result.json",dict(dataset=dataset,images=len(rows),
        modes=list(MODES),studies=list(STUDIES),factorial_arms=list(runtime.ARMS),
        raw_model_seconds={mode:sum(r["seconds"] for r in costs if r["arm"] == mode) for mode in MODES},
        actual_work={mode:{k:sum(r[k] for r in costs if r["arm"] == mode)
                          for k in (*runtime.native.COUNTERS,"optimizer_steps")} for mode in MODES},
        warmup_adjusted_throughput=prediction.runtime.prior.timing_summary(costs,MODES),
        legal_visual_LN=legal,frozen_parameter_sha256=frozen_digest,GT_opened=False,
        metric_status="All predictions including episode0 retained; no GT or score; all arms measure C"),inventory,suite,budget)
    episodes = (len(rows)+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(costs) == episodes*len(MODES)
            and all(sum(r["images"] for r in costs if r["arm"] == mode) == len(rows) for mode in MODES),
            "Incomplete M04 full population")
    expected = {f"{dataset}/predictions/{mode}/{i:06d}.npy" for mode in MODES for i in range(len(rows))}
    expected |= {f"{dataset}/episodes/{e:06d}.{mode}.json" for mode in MODES for e in range(episodes)}
    expected |= {f"{dataset}/{name}" for name in ("initial_state.json","cost.jsonl","prediction_result.json")}
    require(set(inventory) == expected, "M04 exact prediction inventory differs")
    return dict(payloads_complete=True,scientific_completion=False,GT_opened=False,dataset=dataset,
        images=len(rows),studies=list(STUDIES),factorial_arms=list(runtime.ARMS),
        image_method_evaluations=len(rows)*len(MODES),artifact_sha256=inventory)
