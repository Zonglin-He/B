"""Pure M04 receipt checks for a future separately admitted CPU scorer.

No file opens, annotations, models, launches or scoring permission live here.
Reconstruct the assignment and observed native-dtype budget, not GT outcomes.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from scipy import ndimage

from trace_ovss.method.image_balanced import prepare_image_balanced_objective
from trace_ovss.method.score_factorial import assign_score_factorial
from trace_ovss.method.trace.constants import EPS
from .bootstrap import require
from .factorial_prediction import MODES
from .factorial_runtime import native
from .risk_panel import candidate_identity

DTYPES = {"torch.float16":torch.float16,"torch.float32":torch.float32,
          "torch.float64":torch.float64,"torch.bfloat16":torch.bfloat16}
CUDA_DIVISION_GIT = "134179474539648ba7dee1317959529fbd0e7f89"
CUDA_DIVISION_VERSION = "2.7.0+cu128"
CUDA_DIVISION_SOURCE = "https://github.com/pytorch/pytorch/blob/"+CUDA_DIVISION_GIT+"/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu"


def cuda_receipt_coefficient(rho,normalizer,sources):
    """Rebuild the pinned CUDA tensor-divide then CPU-scalar-divide result."""
    require(torch.version.git_version == CUDA_DIVISION_GIT and torch.__version__ == CUDA_DIVISION_VERSION,
            "M04 CUDA receipt arithmetic requires its pinned PyTorch build")
    require(type(sources) is int and 0 < sources <= 512 and isinstance(rho,torch.Tensor)
            and isinstance(normalizer,torch.Tensor) and rho.shape == normalizer.shape == torch.Size([])
            and rho.device.type == normalizer.device.type == "cpu" and rho.dtype == normalizer.dtype
            and rho.dtype in DTYPES.values() and bool(torch.isfinite(rho)) and bool(torch.isfinite(normalizer))
            and float(rho) >= 0 and float(normalizer) > 0,
            "Invalid M04 native coefficient reconstruction inputs")
    # ATen's CUDA CPU-scalar denominator branch multiplies in opmath precision
    # by a precomputed reciprocal, rather than performing a second division.
    quotient=rho/normalizer
    opmath=torch.float64 if rho.dtype == torch.float64 else torch.float32
    reciprocal=torch.tensor(1.,dtype=opmath)/torch.tensor(sources,dtype=opmath)
    return float((quotient.to(opmath)*reciprocal).to(rho.dtype))


def json_value(value):
    return json.loads(json.dumps(value,allow_nan=False))


def validate_episode(saved,spec,source_ids,expected_cost):
    """Validate one bounded native episode, returning original rows/coverage.

    The caller must verify immutable file hashes and permission before reading
    receipts. This pure check does not replace those controller boundaries.
    """
    require(torch.version.git_version == CUDA_DIVISION_GIT and torch.__version__ == CUDA_DIVISION_VERSION,
            "M04 CUDA receipt arithmetic requires its pinned PyTorch build")
    cfg = spec["config"]
    mode = saved["arm"]
    require(mode in MODES and type(source_ids) is list and 0 < len(source_ids) <= cfg["batch_size"]
            and all(type(s) is str and s for s in source_ids) and len(set(source_ids)) == len(source_ids),
            "M04 episode source/mode scope differs")
    study,arm = mode.split(".")
    require(saved["GT_opened"] is False and saved["scientific_identity"] == "M04_"+mode
            and saved["ordered_ids"] == source_ids and saved["images"] == saved["patches"] == len(source_ids)
            and saved["study"] == study and saved["factorial_arm"] == arm
            and saved["common_active"] is (study == "common_active")
            and saved["all_arms_measure_C"] is True and saved["production_Confidence_timing"] is False
            and saved["coefficient_observation_in_model_time"] is True
            and saved["receipt_float64_objective_coefficients_are_ideal_not_runtime"] is True
            and all(saved[k] == v for k,v in expected_cost.items()), "M04 episode cost/identity differs")
    candidates = saved["candidates"]
    require(type(candidates) is list and type(saved["candidate_count"]) is int
            and len(candidates) == saved["candidate_count"] <= len(source_ids)*cfg["trace_max_regions"],
            "M04 candidate population differs")
    original,decoded = [],[]
    per_source = [0]*len(source_ids)
    occupied = np.zeros((len(source_ids),16,16),dtype=bool)
    for row in candidates:
        require(all(type(row[k]) is int for k in ("image_idx","batch_idx","class_idx","component_idx","area"))
                and 0 <= row["image_idx"] == row["batch_idx"] < len(source_ids)
                and 0 <= row["class_idx"] < len(spec["vocabulary"]["queries"])
                and row["component_idx"] > 0 and 0 < row["area"] <= 256,
                "M04 candidate class/source/area differs")
        source = row["image_idx"]
        per_source[source] += 1
        require(per_source[source] <= cfg["trace_max_regions"], "M04 original per-source Top-K exceeded")
        region = row["region"]
        require(region["shape"] == [16,16] and region["encoding"] == "packbits-little-hex"
                and type(region["data"]) is str and len(region["data"]) == 64,
                "M04 original native mask differs")
        packed = bytes.fromhex(region["data"])
        require(len(packed) == 32, "M04 original mask byte extent differs")
        mask = np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder="little").astype(bool).reshape(16,16)
        require(type(region["true_pixels"]) is int and int(mask.sum()) == region["true_pixels"] == row["area"],
                "M04 original mask area differs")
        require(ndimage.label(mask)[1] == 1 and not bool((occupied[source]&mask).any()),
                "M04 native component must be four-connected and source-disjoint")
        occupied[source] |= mask
        require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and 0 <= row[k] <= 1
                    for k in ("confidence","original_C","reliability","quality","original_support","counterfactual_support"))
                and row["confidence"] == row["original_support"]
                and row["original_C"] == row["counterfactual_support"]
                and row["quality"] == row["reliability"]
                and row["diagnostic_algorithm"] == "M04."+arm,
                "M04 detached support/algorithm differs")
        mapping = spec["vocabulary"]["query_to_class"]
        category = row["class_idx"] if mapping is None else mapping[row["class_idx"]]
        require(row["preupdate_candidate_category_idx"] == category, "M04 benchmark class mapping differs")
        original.append(dict(source_id=source_ids[row["image_idx"]],candidate_id=candidate_identity(row),
            area=row["area"],kappa=row["confidence"],C=row["original_C"]))
        decoded.append(dict(row,region=mask))
    expected = dict(assign_score_factorial(original,common_active=study == "common_active"),
        selected_arm=arm,candidates=original,all_arms_measure_C=True,GT_opened=False)
    require(saved["assignment"] == json_value(expected), "M04 score/order/active assignment differs")
    assigned = expected["arms"][arm]["assigned_scores"]
    active_terms,active_rows = [],[]
    for i,(row,value) in enumerate(zip(decoded,assigned,strict=True)):
        require(type(row["factorial_excluded"]) is bool and row["factorial_excluded"] is (value is None)
                and row["reliability"] == (0. if value is None else value),
                "M04 assigned reliability/exclusion differs")
        if row["reliability"] > EPS:
            active_terms.append(dict(image_idx=row["image_idx"],batch_idx=row["batch_idx"],rho=row["reliability"],
                indices=torch.from_numpy(np.flatnonzero(row["region"].reshape(-1)))))
            active_rows.append(original[i])
    budgets = saved["native_dtype_budget"]
    require(type(budgets) is list and len(budgets) == len(active_terms) == saved["active_candidates"],
            "M04 native active budget population differs")
    if budgets:
        dtype_name = budgets[0]["dtype"]
        require(dtype_name in DTYPES and all(r["dtype"] == dtype_name for r in budgets), "M04 native entropy dtype differs")
        entropy = torch.zeros((len(source_ids),16,16),dtype=DTYPES[dtype_name])
        prepared = prepare_image_balanced_objective(entropy,active_terms)
        expected_budgets = []
        # The objective groups by first source occurrence; reconstruct that order.
        groups = {}
        for term,row in zip(active_terms,active_rows,strict=True):
            groups.setdefault(term["image_idx"],[]).append((term,row))
        for group,(entries,normalizer) in zip(groups.values(),prepared,strict=True):
            for (term,row),(batch,indices,rho) in zip(group,entries,strict=True):
                expected_budgets.append(dict(image_idx=term["image_idx"],batch_idx=batch,
                    source_id=row["source_id"],candidate_id=row["candidate_id"],rho=float(rho),
                    normalizer=float(normalizer),dtype=dtype_name,active_sources=len(prepared),
                    coefficient=cuda_receipt_coefficient(rho,normalizer,len(prepared)),
                    coefficient_arithmetic="native_dtype_rho_div_normalizer_div_active_sources"))
        require(budgets == expected_budgets, "M04 observed native-dtype budget differs")
    ideal = native.image_worker.objective_coefficients(decoded,"image_C")
    require(all(type(r["objective_coefficient"]) in (float,int) and math.isfinite(r["objective_coefficient"])
                and r["objective_coefficient"] == v for r,v in zip(decoded,ideal,strict=True)),
            "M04 ideal receipt coefficient differs")
    counters = (*native.COUNTERS,"optimizer_steps","steps","active_candidates","candidate_count")
    require(all(type(saved[k]) is int and saved[k] >= 0 for k in counters)
            and type(saved["seconds"]) in (float,int) and math.isfinite(saved["seconds"]) and saved["seconds"] >= 0,
            "M04 nonfinite time or invalid counter")
    steps = cfg["steps"] if active_terms else 0
    calls = (len(candidates)+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
    require(saved["steps"] == saved["optimizer_steps"] == steps and len(saved["losses"]) == steps
            and all(type(v) in (float,int) and math.isfinite(v) for v in saved["losses"])
            and saved["eval_forward_calls"] == 2 and saved["eval_forward_inputs"] == 2*len(source_ids)
            and saved["adapt_forward_calls"] == steps and saved["adapt_forward_inputs"] == steps*len(source_ids)
            and saved["diagnostic_forward_calls"] == saved["diagnostic_encoder_calls"] == calls
            and saved["diagnostic_forward_inputs"] == saved["diagnostic_encoder_inputs"] == len(candidates)
            and saved["encoder_calls"] == 2+steps+calls
            and saved["encoder_inputs"] == (2+steps)*len(source_ids)+len(candidates),
            "M04 measured work/loss schedule differs")
    require(saved["candidate_sha256"] == native.image_worker.prior.candidate_identity(decoded),
            "M04 original candidate hash differs")
    return dict(original_candidates=original,candidate_sha256=saved["candidate_sha256"],
        retained_candidates=expected["retained_candidates"],EPS_active_candidates=len(active_terms),
        active_sources=len({r["source_id"] for r in active_rows}))


def validate_shared(checked):
    """All eight conditions must share original confidence, C and identity."""
    require(set(checked) == set(MODES), "M04 missing study/arm receipt")
    anchor = checked[MODES[0]]
    require(all(r["candidate_sha256"] == anchor["candidate_sha256"]
                and r["original_candidates"] == anchor["original_candidates"] for r in checked.values()),
            "M04 original candidate/C/confidence changed across conditions")
