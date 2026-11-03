"""Pure M03 receipt validation using caller-verified pinned component objects.

No file/archive reads, models, scoring, launch, prepare or admission here.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from trace_ovss.method.trace.constants import EPS
from . import legacy_analysis as analysis
from . import legacy_rho
from . import legacy_runtime as runtime
from .bootstrap import require

MODES = analysis.MODES


def validate_episode(saved,spec,source_ids,expected_cost,components):
    cfg,vocabulary = spec["config"],spec["vocabulary"]
    mode = saved["arm"]
    require(mode in MODES and type(source_ids) is list and 0 < len(source_ids) <= cfg["batch_size"]
            and all(type(s) is str and s for s in source_ids) and len(set(source_ids)) == len(source_ids),
            "M03 scoring source/mode differs")
    taxonomy = {"coco_object":"COCOObjectDataset","pc60":"PascalContext60Dataset"}[spec["dataset"]]
    require(cfg["dataset"] == taxonomy and components["archive_sha256"] == legacy_rho.ARCHIVE_SHA
            and set(components["member_sha256"]) == set(legacy_rho.MEMBERS)
            and all(components[k] is False for k in ("old_engine_loaded","old_optimizer_loaded","old_OOM_iterator_selected")),
            "M03 scoring taxonomy/component identity differs")
    mapping = runtime.effective_mapping(vocabulary["query_to_class"],len(vocabulary["queries"]),len(vocabulary["classes"]))
    roles = legacy_rho.query_roles(components,taxonomy,vocabulary["classes"],vocabulary["query_to_class"])
    require(len(roles) == len(vocabulary["queries"]) and all(r in ("thing","stuff","background") for r in roles),
            "M03 scoring query role mapping differs")
    require(saved["GT_opened"] is False and saved["scientific_identity"] == "M03_"+mode
            and saved["ordered_ids"] == source_ids and saved["images"] == saved["patches"] == len(source_ids)
            and saved["legacy_archive_sha256"] == components["archive_sha256"]
            and saved["legacy_source_members"] == components["member_sha256"]
            and saved["episode_objective_sha256"] == runtime.OBJECTIVE_SHA
            and saved["receipt_float64_objective_coefficients_are_ideal_not_runtime"] is True
            and all(saved[k] is False for k in ("old_engine_loaded","old_optimizer_loaded","old_OOM_iterator_selected"))
            and all(saved[k] == v for k,v in expected_cost.items()), "M03 scoring cost/archive/episode identity differs")
    candidates = saved["candidates"]
    require(type(candidates) is list and type(saved["candidate_count"]) is int and len(candidates) == saved["candidate_count"]
            <= len(source_ids)*cfg["trace_max_regions"], "M03 scoring candidate population differs")
    decoded,diagnostic,seen,order = [],[],set(),[]
    per_source,occupied = [0]*len(source_ids),np.zeros((len(source_ids),16,16),dtype=bool)
    views = 0
    def probability(value):
        return type(value) in (float,int) and math.isfinite(value) and 0 <= value <= 1
    for row in candidates:
        require(all(type(row[k]) is int for k in ("image_idx","batch_idx","class_idx","component_idx","area"))
                and 0 <= row["image_idx"] == row["batch_idx"] < len(source_ids)
                and 0 <= row["class_idx"] < len(roles) and row["component_idx"] > 0 and 0 < row["area"] <= 256,
                "M03 scoring original candidate identity differs")
        source = row["image_idx"]
        per_source[source] += 1
        require(per_source[source] <= cfg["trace_max_regions"], "M03 original per-source Top-K exceeded")
        region = row["region"]
        require(region["shape"] == [16,16] and region["encoding"] == "packbits-little-hex"
                and type(region["data"]) is str and len(region["data"]) == 64, "M03 original native mask differs")
        packed = bytes.fromhex(region["data"])
        require(len(packed) == 32, "M03 original mask byte extent differs")
        mask = np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder="little").astype(bool).reshape(16,16)
        require(type(region["true_pixels"]) is int and int(mask.sum()) == region["true_pixels"] == row["area"]
                and ndimage.label(mask)[1] == 1 and not bool((occupied[source]&mask).any()),
                "M03 original component area/connectivity/overlap differs")
        occupied[source] |= mask
        require(all(probability(row[k]) for k in ("confidence","original_support","counterfactual_support","quality","reliability"))
                and row["confidence"] == row["original_support"]
                and row["preupdate_candidate_category_idx"] == mapping[row["class_idx"]],
                "M03 original support or benchmark class mapping differs")
        key = (source,row["class_idx"],row["component_idx"])
        require(key not in seen, "M03 duplicate original component identity")
        seen.add(key)
        order.append((source,-row["confidence"],-row["area"],row["class_idx"],row["component_idx"]))
        keys = ("original_support","target_support","counterfactual_support","residual_support","quality","dominance","reliability")
        if mode.startswith("C."):
            require(row["diagnostic_algorithm"] == "C" and row["target_support"] is None
                    and row["residual_support"] is None and row["dominance"] is None
                    and row["quality"] == row["reliability"] == row["counterfactual_support"],
                    "M03 single-C diagnostic formula differs")
            views += 1
        else:
            role = roles[row["class_idx"]]
            require(row["role"] == role and row["diagnostic_algorithm"] == "pinned_legacy_rho"
                    and probability(row["target_support"]) and probability(row["dominance"])
                    and (probability(row["residual_support"]) if role == "thing" else row["residual_support"] is None),
                    "M03 legacy role/view/support differs")
            score = components["modules"]["reliability"].trace_reliability(
                row["target_support"],row["counterfactual_support"],row["residual_support"],role)
            require(all(row[k] == getattr(score,k) for k in ("quality","dominance","reliability")),
                    "M03 pinned legacy score formula differs")
            views += len(components["modules"]["interventions"].trace_view_types(role))
        diagnostic.append({k:row[k] for k in keys})
        decoded.append(dict(row,region=mask))
    require(order == sorted(order), "M03 original native candidate order differs")
    ideal_arm = "episode_min" if mode.endswith(".episode") else "image_C"
    ideal = runtime.native.image_worker.objective_coefficients(decoded,ideal_arm)
    require(all(type(r["objective_coefficient"]) in (float,int) and math.isfinite(r["objective_coefficient"])
                and r["objective_coefficient"] == v for r,v in zip(decoded,ideal,strict=True)),
            "M03 ideal normalization receipt differs")
    active = sum(r["reliability"] > EPS for r in decoded)
    counters = (*runtime.native.COUNTERS,"steps","optimizer_steps","candidate_count","active_candidates")
    require(all(type(saved[k]) is int and saved[k] >= 0 for k in counters)
            and saved["active_candidates"] == active and type(saved["seconds"]) in (int,float)
            and math.isfinite(saved["seconds"]) and saved["seconds"] >= 0, "M03 actual time/update population differs")
    steps = cfg["steps"] if active else 0
    calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
    require(saved["steps"] == saved["optimizer_steps"] == len(saved["losses"]) == steps
            and all(type(v) in (int,float) and math.isfinite(v) for v in saved["losses"])
            and saved["eval_forward_calls"] == 2 and saved["eval_forward_inputs"] == 2*len(source_ids)
            and saved["adapt_forward_calls"] == steps and saved["adapt_forward_inputs"] == steps*len(source_ids)
            and saved["diagnostic_forward_calls"] == saved["diagnostic_encoder_calls"] == calls
            and saved["diagnostic_forward_inputs"] == saved["diagnostic_encoder_inputs"] == views
            and saved["encoder_calls"] == 2+steps+calls and saved["encoder_inputs"] == (2+steps)*len(source_ids)+views,
            "M03 actual finite forward/Adam schedule differs")
    identity = runtime.native.prior.candidate_identity(decoded)
    require(saved["candidate_sha256"] == identity, "M03 original candidate hash differs")
    return dict(candidate_sha256=identity,diagnostic=diagnostic,candidates=len(decoded),
        EPS_active_candidates=active,active_sources=len({r["image_idx"] for r in decoded if r["reliability"] > EPS}),
        diagnostic_views=views)


def validate_shared(checked):
    require(set(checked) == set(MODES) and len({r["candidate_sha256"] for r in checked.values()}) == 1,
            "M03 original candidates changed across diagnostic/objective conditions")
    for score in ("C","legacy"):
        require(checked[score+".image"]["diagnostic"] == checked[score+".episode"]["diagnostic"],
                "M03 diagnostic responses changed across normalization-only conditions")
