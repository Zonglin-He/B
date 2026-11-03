"""G03 full prediction IO core, with an explicit safe progress callback.

Model updates/readout remain the unchanged generalization runtime; this local
copy separates IO from the old core's truncating advisory writer.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from . import generalization_worker as general
from .bootstrap import require


def execute_condition(method,data,spec,output,suite,cost,progress,write_progress):
    cfg,rows,inventory=spec["config"],data.rows,{}
    output.mkdir()
    for arm in general.ARMS:(output/"predictions"/arm).mkdir(parents=True)
    (output/"episodes").mkdir()
    method.reset()
    initial=general.shared.restored_state(method)
    frozen=[(p,p._version) for p in method.model.parameters() if not p.requires_grad]
    frozen_digest=general.shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad})
    legal=general.runtime.legal_parameters(method)
    general.write_payload(output/"initial_state.json",dict(reset=initial,legal_visual_LN=legal,
        frozen_parameter_sha256=frozen_digest),inventory,suite,cost)
    all_costs=[]
    with (output/"cost.jsonl").open("x") as stream:
        for episode,start in enumerate(range(0,len(data),cfg["batch_size"])):
            stop=min(start+cfg["batch_size"],len(data))
            images,meta,valid,ids=general.episode_batch(data,start,stop,method,cfg)
            require(list(meta["img_path"]) == [r["image"] for r in rows[start:stop]],"G03 observed RGB order differs")
            rng,identity=general.shared.rng_state(method.device),None
            for position,arm in enumerate(general.runtime.prior.arm_order(general.ARMS,episode,"alternating")):
                general.shared.restore_rng(rng,method.device)
                predictions,stats,losses,row=general.runtime.run_arm(method,arm,images,valid,ids,meta,cfg,spec["vocabulary"]["query_to_class"])
                row.update(episode=episode,order_position=position,warmup=episode == 0)
                if arm != "no_adapt":
                    identity=row["candidate_sha256"] if identity is None else identity
                    require(row["candidate_sha256"] == identity,"G03 C/Confidence candidate population differs")
                require(all(p._version == version for p,version in frozen),"Frozen G03 weights changed")
                require(len(predictions) == stop-start,"Missing G03 source prediction")
                classes=len(spec["vocabulary"]["classes"])
                for i,prediction in enumerate(predictions,start):
                    require(prediction.shape == tuple(cfg["resize"]) and prediction.dtype == np.uint16
                            and bool((prediction < classes).all()),"Invalid mapped G03 hard prediction")
                    general.write_payload(output/"predictions"/arm/f"{i:06d}.npy",prediction,inventory,suite,cost)
                candidates=[] if arm == "no_adapt" else general.runtime.image_worker.candidate_receipts(stats,arm,spec["vocabulary"]["query_to_class"])
                general.write_payload(output/"episodes"/f"{episode:06d}.{arm}.json",dict(**row,losses=losses,
                    ordered_ids=[r["official_id"] for r in rows[start:stop]],candidates=candidates,
                    no_adapt_candidate_status="not computed, not shared from adapted arms" if arm == "no_adapt" else None,
                    GT_opened=False),inventory,suite,cost)
                line=json.dumps(row,allow_nan=False)+"\n";size=len(line.encode())
                require(shutil.disk_usage(suite).free >= general.RESERVE+size and cost["bytes"]+size <= general.MAX_PAYLOAD_BYTES,
                        "G03 cost stream exceeds shared storage budget")
                require(stream.write(line) == len(line),"Short G03 cost write")
                stream.flush();cost["bytes"]+=size;all_costs.append(row)
                method.reset()
                require(general.shared.restored_state(method) == initial,"G03 episode/arm reset differs")
            progress["completed_source_episodes"]+=1
            write_progress(suite/"queue_state.json",dict(phase="G03_paired_full_prediction",condition=spec["id"],
                sources_in_condition=stop,total_sources_in_condition=len(data),stored_bytes=cost["bytes"],
                **progress,pid=os.getpid(),GT_opened=False),replace=True)
            del images,meta,valid,ids
    require(general.shared.state_digest({n:p for n,p in method.model.named_parameters() if not p.requires_grad}) == frozen_digest,
            "G03 frozen parameter bytes changed")
    inventory[str((output/"cost.jsonl").relative_to(suite))]=general.sha256(output/"cost.jsonl")
    general.write_payload(output/"prediction_result.json",dict(condition=spec["id"],images=len(data),arms=list(general.ARMS),
        actual_work={a:{k:sum(r[k] for r in all_costs if r["arm"] == a) for k in general.runtime.COUNTERS} for a in general.ARMS},
        raw_model_seconds={a:sum(r["seconds"] for r in all_costs if r["arm"] == a) for a in general.ARMS},
        warmup_adjusted_throughput=general.runtime.prior.timing_summary(all_costs,general.ARMS),legal_visual_LN=legal,
        frozen_parameter_sha256=frozen_digest,GT_opened=False,
        metric_status="No GT or score; every original source including episode0 retained"),inventory,suite,cost)
    episodes=(len(data)+cfg["batch_size"]-1)//cfg["batch_size"]
    require(len(all_costs) == episodes*len(general.ARMS)
            and all(sum(r["images"] for r in all_costs if r["arm"] == a) == spec["images"] for a in general.ARMS),
            "Incomplete G03 per-arm full population")
    prefix=str(output.relative_to(suite))+"/"
    expected={prefix+f"predictions/{a}/{i:06d}.npy" for a in general.ARMS for i in range(len(data))}
    expected|={prefix+f"episodes/{e:06d}.{a}.json" for a in general.ARMS for e in range(episodes)}
    expected|={prefix+n for n in ("initial_state.json","cost.jsonl","prediction_result.json")}
    require(set(inventory) == expected,"G03 complete prediction inventory differs")
    return dict(complete=True,condition=spec,images=len(data),image_method_evaluations=len(data)*len(general.ARMS),
                artifact_sha256=inventory,GT_opened=False)
