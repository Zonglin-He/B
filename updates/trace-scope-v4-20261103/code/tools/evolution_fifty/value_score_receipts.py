"""Pure V01 fixed-point receipt checks, not a scorer or execution permission."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from trace_ovss.method.trace.constants import EPS
from . import value_runtime as runtime
from .bootstrap import require
from .legacy_runtime import effective_mapping

TIME_SCOPE = "Synchronized native reset/candidates/probe/adaptation/finalreadout plus common finite/Adam observation; excludes decode/H2D/receipts/GT"


def validate_episode(saved,spec,source_ids,expected_cost):
    dataset,cfg,vocab = spec["dataset"],spec["config"],spec["vocabulary"]
    runtime.native.image_worker.validate_config(cfg)
    require(type(cfg["data_dir"]) is str and type(cfg["data_subdir"]) is str
            and all(type(v) is int for key in ("resize","patch_size") for v in cfg[key]),
            "V01 inherited config path/geometry types differ")
    point = saved["fixed_point"]
    runtime.validate_point(dataset,point)
    require(type(spec["points"]) is list, "V01 serialized point grid differs")
    for declared in spec["points"]:runtime.validate_point(dataset,declared)
    require(cfg["dataset"] == {"coco_object":"COCOObjectDataset","pc60":"PascalContext60Dataset"}[dataset]
            and type(cfg["steps"]) is int and cfg["steps"] == runtime.NATIVE_STEPS[dataset]
            and spec["points"] == list(runtime.points(dataset)), "V01 receipt native config/grid differs")
    require(type(source_ids) is list and 0 < len(source_ids) <= cfg["batch_size"]
            and all(type(s) is str and s for s in source_ids) and len(set(source_ids)) == len(source_ids),
            "V01 receipt positional source IDs differ")
    require(saved["GT_opened"] is False and saved["ordered_ids"] == source_ids
            and saved["images"] == saved["patches"] == len(source_ids) and saved["dataset"] == dataset
            and saved["arm"] == saved["point_id"] == point["id"] and saved["weighting_arm"] == point["arm"]
            and type(saved["native_steps"]) is int and saved["native_steps"] == cfg["steps"]
            and type(saved["requested_steps"]) is int and saved["requested_steps"] == point["steps"]
            and saved["actual_step_observation_in_model_time"] is True
            and type(saved["auxiliary_checkpoint_readouts"]) is int and saved["auxiliary_checkpoint_readouts"] == 0
            and saved["time_scope"] == TIME_SCOPE
            and saved["receipt_float64_objective_coefficients_are_ideal_not_runtime"] is True
            and all(saved[k] == v for k,v in expected_cost.items()), "V01 receipt cost/point/timing identity differs")
    mapping = effective_mapping(vocab["query_to_class"],len(vocab["queries"]),len(vocab["classes"]))
    candidates = saved["candidates"]
    require(type(candidates) is list and type(saved["candidate_count"]) is int and len(candidates) == saved["candidate_count"]
            <= len(source_ids)*cfg["trace_max_regions"], "V01 original candidate population differs")
    decoded,keys,order,diagnostic,query_masks = [],set(),[],[],{}
    per_source,occupied = [0]*len(source_ids),np.zeros((len(source_ids),16,16),dtype=bool)
    def probability(value):
        return type(value) in (float,int) and math.isfinite(value) and 0 <= value <= 1
    for row in candidates:
        require(all(type(row[k]) is int for k in ("image_idx","batch_idx","class_idx","component_idx","area"))
                and 0 <= row["image_idx"] == row["batch_idx"] < len(source_ids)
                and 0 <= row["class_idx"] < len(mapping) and 0 < row["component_idx"] <= 256 and 0 < row["area"] <= 256,
                "V01 original candidate source/query/component differs")
        source = row["image_idx"]
        key = (source,row["class_idx"],row["component_idx"])
        per_source[source] += 1
        require(key not in keys and per_source[source] <= cfg["trace_max_regions"], "V01 duplicate component or per-source Top-K differs")
        keys.add(key)
        region = row["region"]
        require(region["shape"] == [16,16] and region["encoding"] == "packbits-little-hex"
                and type(region["data"]) is str and len(region["data"]) == 64, "V01 native candidate mask differs")
        packed = bytes.fromhex(region["data"])
        require(len(packed) == 32, "V01 packed-mask byte extent differs")
        mask = np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder="little").astype(bool).reshape(16,16)
        require(type(region["true_pixels"]) is int and int(mask.sum()) == region["true_pixels"] == row["area"]
                and ndimage.label(mask)[1] == 1 and not bool((occupied[source]&mask).any()),
                "V01 native mask area/connectivity/source-disjointness differs")
        occupied[source] |= mask
        query = (source,row["class_idx"])
        if query in query_masks:
            require(not bool((ndimage.binary_dilation(mask)&query_masks[query]).any()),
                    "V01 distinct original same-query components are adjacent")
            query_masks[query] |= mask
        else:query_masks[query] = mask.copy()
        require(all(probability(row[k]) for k in ("confidence","original_support","quality","reliability"))
                and row["confidence"] == row["original_support"] and row["quality"] == row["reliability"]
                and type(row["preupdate_candidate_category_idx"]) is int
                and row["preupdate_candidate_category_idx"] == mapping[row["class_idx"]]
                and row["target_support"] is None and row["residual_support"] is None and row["dominance"] is None
                and row["omitted"] is False, "V01 original support/mapping or unsupported diagnostic differs")
        if point["arm"] == "image_C":
            require(row["diagnostic_algorithm"] == "C" and probability(row["counterfactual_support"])
                    and row["reliability"] == row["counterfactual_support"], "V01 single-C score differs")
        else:
            require(row["diagnostic_algorithm"] == "Confidence" and row["counterfactual_support"] is None
                    and row["reliability"] == row["confidence"], "V01 zero-view Confidence score differs")
        order.append((source,-row["confidence"],-row["area"],row["class_idx"],row["component_idx"]))
        diagnostic.append([row[k] for k in ("original_support","counterfactual_support","quality","reliability")])
        decoded.append(dict(row,region=mask))
    require(order == sorted(order), "V01 native original candidate order differs")
    ideal = runtime.native.image_worker.objective_coefficients(decoded,point["arm"])
    require(all(type(r["objective_coefficient"]) in (float,int) and math.isfinite(r["objective_coefficient"])
                and r["objective_coefficient"] == v for r,v in zip(decoded,ideal,strict=True)),
            "V01 ideal float64 coefficient receipt differs")
    active = sum(r["reliability"] > EPS for r in decoded)
    counters = (*runtime.native.COUNTERS,"steps","optimizer_steps","candidate_count","active_candidates")
    require(all(type(saved[k]) is int and saved[k] >= 0 for k in counters)
            and saved["active_candidates"] == active and type(saved["seconds"]) in (float,int)
            and math.isfinite(saved["seconds"]) and saved["seconds"] >= 0, "V01 time/update population differs")
    steps = point["steps"] if active else 0
    views = len(candidates) if point["arm"] == "image_C" else 0
    calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
    require(saved["steps"] == saved["optimizer_steps"] == len(saved["losses"]) == steps
            and all(type(v) in (float,int) and math.isfinite(v) for v in saved["losses"])
            and saved["eval_forward_calls"] == 2 and saved["eval_forward_inputs"] == 2*len(source_ids)
            and saved["adapt_forward_calls"] == steps and saved["adapt_forward_inputs"] == steps*len(source_ids)
            and saved["diagnostic_forward_calls"] == saved["diagnostic_encoder_calls"] == calls
            and saved["diagnostic_forward_inputs"] == saved["diagnostic_encoder_inputs"] == views
            and saved["encoder_calls"] == 2+steps+calls and saved["encoder_inputs"] == (2+steps)*len(source_ids)+views,
            "V01 actual finite losses/Adam/forward schedule differs")
    identity = runtime.native.prior.candidate_identity(decoded)
    require(saved["candidate_sha256"] == identity, "V01 original candidate hash differs")
    return dict(candidate_sha256=identity,diagnostic=diagnostic,arm=point["arm"],candidates=len(decoded),
        EPS_active_candidates=active,active_sources=len({r["image_idx"] for r in decoded if r["reliability"] > EPS}),diagnostic_views=views)


def validate_shared(dataset,checked):
    points = runtime.points(dataset)
    require(set(checked) == {p["id"] for p in points} and len({v["candidate_sha256"] for v in checked.values()}) == 1,
            "V01 incomplete grid or changed original candidates")
    for arm in ("image_C","image_confidence"):
        group = [checked[p["id"]] for p in points if p["arm"] == arm]
        require(all(v["arm"] == arm and v["diagnostic"] == group[0]["diagnostic"] for v in group),
                "V01 initial diagnostic changed with update budget")
