import copy
import math

import numpy as np
import pytest
import torch

from tools.evolution_fifty import factorial_score_receipts as receipts


def fixture(mode="natural.Dc_Oc",dtype=torch.float32,empty=False,*,items=None,mask_locations=None,component_zero=False,source_count=2):
    study,arm = mode.split(".")
    ids = [f"synthetic{i}" for i in range(source_count)]
    spec = dict(config=dict(batch_size=source_count,trace_max_regions=35,steps=7,trace_view_batch_size=2),
        vocabulary=dict(queries=["a","b"],classes=["a","b"],query_to_class=None))
    candidates,decoded,original = [],[],[]
    items = [(0,.9,.2),(0,.1,.8),(1,.7,0.)] if items is None else items
    for i,(source,kappa,C) in enumerate([] if empty else items):
        mask = np.zeros((16,16),dtype=bool)
        mask.flat[[i] if mask_locations is None else mask_locations[i]] = True
        row = dict(image_idx=source,batch_idx=source,class_idx=i%2,component_idx=0 if component_zero and i == 0 else i+1,area=int(mask.sum()),
            confidence=kappa,original_support=kappa,original_C=C,counterfactual_support=C,
            region=dict(shape=[16,16],encoding="packbits-little-hex",
                data=np.packbits(mask.reshape(-1),bitorder="little").tobytes().hex(),true_pixels=int(mask.sum())),
            diagnostic_algorithm="M04."+arm,preupdate_candidate_category_idx=i%2)
        original.append(dict(source_id=ids[source],candidate_id=receipts.candidate_identity(row),area=int(mask.sum()),kappa=kappa,C=C))
        candidates.append(row)
        decoded.append(dict(row,region=mask))
    assignment = dict(receipts.assign_score_factorial(original,common_active=study == "common_active"),
        selected_arm=arm,candidates=original,all_arms_measure_C=True,GT_opened=False)
    scores = assignment["arms"][arm]["assigned_scores"]
    for row,decoded_row,score in zip(candidates,decoded,scores,strict=True):
        for r in (row,decoded_row):
            r.update(quality=0. if score is None else score,reliability=0. if score is None else score,
                factorial_excluded=score is None)
    ideals = receipts.native.image_worker.objective_coefficients(decoded,"image_C")
    for row,decoded_row,coefficient in zip(candidates,decoded,ideals,strict=True):
        row["objective_coefficient"] = decoded_row["objective_coefficient"] = coefficient
    groups = {}
    for row,source in zip(decoded,original,strict=True):
        if row["reliability"] > receipts.EPS:
            groups.setdefault(row["image_idx"],[]).append((row,source))
    budgets = []
    for group in groups.values():
        denominator = torch.tensor(0.,dtype=dtype)
        for row,_ in group:
            denominator = denominator+torch.tensor(row["reliability"],dtype=dtype)
        normalizer = denominator if float(denominator.double()) > receipts.EPS else torch.tensor(1.,dtype=dtype)
        for row,source in group:
            rho = torch.tensor(row["reliability"],dtype=dtype)
            budgets.append(dict(image_idx=row["image_idx"],batch_idx=row["batch_idx"],source_id=source["source_id"],
                candidate_id=source["candidate_id"],rho=float(rho),normalizer=float(normalizer),dtype=str(dtype),
                active_sources=len(groups),coefficient=float(rho/normalizer/len(groups)),
                coefficient_arithmetic="native_dtype_rho_div_normalizer_div_active_sources"))
    steps = 7 if budgets else 0
    calls = (len(candidates)+1)//2
    cost = dict(arm=mode,study=study,factorial_arm=arm,common_active=study == "common_active",images=source_count,patches=source_count,
        all_arms_measure_C=True,production_Confidence_timing=False,coefficient_observation_in_model_time=True,
        seconds=.1,steps=steps,optimizer_steps=steps,candidate_count=len(candidates),active_candidates=len(budgets),
        candidate_sha256=receipts.native.image_worker.prior.candidate_identity(decoded),
        eval_forward_calls=2,eval_forward_inputs=2*source_count,adapt_forward_calls=steps,adapt_forward_inputs=source_count*steps,
        diagnostic_forward_calls=calls,diagnostic_encoder_calls=calls,
        diagnostic_forward_inputs=len(candidates),diagnostic_encoder_inputs=len(candidates),
        encoder_calls=2+steps+calls,encoder_inputs=(2+steps)*source_count+len(candidates))
    saved = dict(**cost,GT_opened=False,scientific_identity="M04_"+mode,ordered_ids=ids,
        receipt_float64_objective_coefficients_are_ideal_not_runtime=True,
        candidates=candidates,assignment=receipts.json_value(assignment),native_dtype_budget=budgets,losses=[.4]*steps)
    return saved,spec,ids,cost


@pytest.mark.parametrize("mode",receipts.MODES)
@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.float64,torch.bfloat16])
def test_all_studies_arms_and_native_dtypes(mode,dtype):
    saved,spec,ids,cost = fixture(mode,dtype)
    result = receipts.validate_episode(saved,spec,ids,cost)
    assert result["EPS_active_candidates"] == saved["active_candidates"]
    assert result["retained_candidates"] == (2 if mode.startswith("common_active.") else 3)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.float64,torch.bfloat16])
@pytest.mark.parametrize("sources",[1,2,3,7,13,26,511])
def test_pinned_cuda_scalar_division_is_exact_reciprocal_multiply(dtype,sources):
    rho,normalizer=torch.tensor(.7391,dtype=dtype),torch.tensor(7.386,dtype=dtype)
    q=(rho/normalizer).item()
    opmath=np.float64 if dtype == torch.float64 else np.float32
    inverse=opmath(opmath(1.)/opmath(sources))
    expected=torch.tensor(float(opmath(opmath(q)*inverse)),dtype=dtype).item()
    assert receipts.cuda_receipt_coefficient(rho,normalizer,sources) == expected
    assert not torch.cuda.is_initialized()


def test_non_power_of_two_source_count_does_not_accept_cpu_divide_as_cuda():
    rho,normalizer=torch.tensor(.1,dtype=torch.float32),torch.tensor(1.,dtype=torch.float32)
    expected=receipts.cuda_receipt_coefficient(rho,normalizer,13)
    assert expected != float(rho/normalizer/13)
    assert expected == float((rho/normalizer)*torch.tensor(np.float32(1./13)))


def test_different_torch_build_is_not_silently_given_same_arithmetic(monkeypatch):
    monkeypatch.setattr(torch.version,"git_version","different")
    with pytest.raises(ValueError,match="pinned PyTorch"):
        receipts.cuda_receipt_coefficient(torch.tensor(.5),torch.tensor(1.),13)


def test_empty_receipts_still_require_the_pinned_build(monkeypatch):
    values=fixture(empty=True)
    monkeypatch.setattr(torch.version,"git_version","different")
    with pytest.raises(ValueError,match="pinned PyTorch"):
        receipts.validate_episode(*values)


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.float64,torch.bfloat16])
def test_originally_active_near_EPS_term_keeps_valid_native_rounded_rho(dtype):
    value=float(np.nextafter(np.float64(receipts.EPS),np.float64(1.)))
    saved,spec,ids,cost=fixture("natural.Dk_Ok",dtype,items=[(0,value,value)],source_count=1)
    assert receipts.validate_episode(saved,spec,ids,cost)["EPS_active_candidates"] == 1
    assert saved["native_dtype_budget"][0]["rho"] == float(torch.tensor(value,dtype=dtype))
    assert math.isfinite(saved["native_dtype_budget"][0]["coefficient"])


def test_negative_native_rho_remains_invalid():
    with pytest.raises(ValueError,match="reconstruction inputs"):
        receipts.cuda_receipt_coefficient(torch.tensor(-.1),torch.tensor(1.),13)


def test_thirteen_source_receipt_accepts_exact_cuda_and_rejects_adjacent_coefficients():
    items=[(0,.1,.1),(0,.9,.9)]+[(i,.7,.7) for i in range(1,13)]
    saved,spec,ids,cost=fixture("natural.Dk_Ok",items=items,source_count=13)
    cpu_value=saved["native_dtype_budget"][0]["coefficient"]
    for row in saved["native_dtype_budget"]:
        q=np.float32(np.float32(row["rho"])/np.float32(row["normalizer"]))
        row["coefficient"]=float(np.float32(q*np.float32(1./13)))
    assert saved["native_dtype_budget"][0]["coefficient"] != cpu_value
    assert receipts.validate_episode(saved,spec,ids,cost)["active_sources"] == 13
    for changed in (cpu_value,float(np.nextafter(np.float32(saved["native_dtype_budget"][0]["coefficient"]),np.float32(1.)))):
        wrong=copy.deepcopy(saved)
        wrong["native_dtype_budget"][0]["coefficient"]=changed
        with pytest.raises(ValueError,match="native-dtype budget"):
            receipts.validate_episode(wrong,spec,ids,cost)


def test_empty_pool_and_all_eight_shared_identity():
    checked = {}
    for mode in receipts.MODES:
        saved,spec,ids,cost = fixture(mode,empty=True)
        checked[mode] = receipts.validate_episode(saved,spec,ids,cost)
        assert checked[mode]["EPS_active_candidates"] == 0
    receipts.validate_shared(checked)


def test_same_original_identity_not_same_assigned_scores():
    checked = {m:receipts.validate_episode(*fixture(m)) for m in receipts.MODES}
    receipts.validate_shared(checked)
    assert checked["natural.Dk_Ok"]["EPS_active_candidates"] == 3
    assert checked["natural.Dc_Oc"]["EPS_active_candidates"] == 2
    for fault in ("missing","changed_C"):
        wrong = copy.deepcopy(checked)
        if fault == "missing": del wrong[receipts.MODES[-1]]
        else: wrong[receipts.MODES[-1]]["original_candidates"][0]["C"] = .3
        with pytest.raises(ValueError): receipts.validate_shared(wrong)


@pytest.mark.parametrize("fault",[
    "assignment","excluded","rho","normalizer","coefficient","dtype","active_sources","budget_identity",
    "budget_order","area","hex","class","source","duplicate","original_support","original_C",
    "mask","candidate_hash","ideal","GT","loss","counter","time","order","mapping","native_flag",
])
def test_invalid_receipts_reject(fault):
    saved,spec,ids,cost = fixture()
    row = saved["candidates"][0]
    budget = saved["native_dtype_budget"][0]
    if fault == "assignment": saved["assignment"]["arms"]["Dc_Ok"]["assigned_scores"][0] = .4
    elif fault == "excluded": row["factorial_excluded"] = True
    elif fault == "rho": budget["rho"] = .3
    elif fault == "normalizer": budget["normalizer"] = math.nan
    elif fault == "coefficient": budget["coefficient"] = math.inf
    elif fault == "dtype": budget["dtype"] = "torch.int64"
    elif fault == "active_sources": budget["active_sources"] += 1
    elif fault == "budget_identity": budget["candidate_id"] = "wrong"
    elif fault == "budget_order": saved["native_dtype_budget"].reverse()
    elif fault == "area": row["area"] = True
    elif fault == "hex": row["region"]["data"] = " "*64
    elif fault == "class": row["class_idx"] = 2
    elif fault == "source": row["batch_idx"] = 1
    elif fault == "duplicate": saved["candidates"][1] = copy.deepcopy(row)
    elif fault == "original_support": row["original_support"] = .5
    elif fault == "original_C": row["original_C"] = math.nan
    elif fault == "mask": row["region"]["data"] = "0"*64
    elif fault == "candidate_hash": saved["candidate_sha256"] = cost["candidate_sha256"] = "0"*64
    elif fault == "ideal": row["objective_coefficient"] = .6
    elif fault == "GT": saved["GT_opened"] = True
    elif fault == "loss": saved["losses"][0] = math.nan
    elif fault == "counter": saved["encoder_inputs"] = cost["encoder_inputs"] = 100
    elif fault == "time": saved["seconds"] = cost["seconds"] = math.inf
    elif fault == "order": saved["ordered_ids"] = ids[::-1]
    elif fault == "mapping": row["preupdate_candidate_category_idx"] = 1
    elif fault == "native_flag": saved["production_Confidence_timing"] = True
    with pytest.raises(ValueError): receipts.validate_episode(saved,spec,ids,cost)


@pytest.mark.parametrize("mode",receipts.MODES)
def test_original_per_source_topk_precedes_common_active(mode):
    saved,spec,ids,cost = fixture(mode,items=[(0,.9,.2),(0,.1,.8),(0,.7,0.)])
    spec["config"]["trace_max_regions"] = 2
    assert saved["candidate_count"] == 3 <= len(ids)*2
    if mode.startswith("common_active."):
        assert saved["assignment"]["retained_candidates"] == 2
    with pytest.raises(ValueError,match="per-source Top-K"):
        receipts.validate_episode(saved,spec,ids,cost)


@pytest.mark.parametrize("mode",receipts.MODES)
@pytest.mark.parametrize("kind",["component_zero","disconnected","overlap","diagonal_only"])
def test_native_components_reject_self_consistent_impossible_receipts(mode,kind):
    kwargs = dict(component_zero=True) if kind == "component_zero" else dict(mask_locations={
        "disconnected":[[0,255],[1],[2]],"overlap":[[0],[0],[2]],"diagonal_only":[[0,17],[1],[2]],
    }[kind])
    saved,spec,ids,cost = fixture(mode,**kwargs)
    with pytest.raises(ValueError,match="class/source/area|four-connected and source-disjoint"):
        receipts.validate_episode(saved,spec,ids,cost)


@pytest.mark.parametrize("mode",receipts.MODES)
def test_connected_masks_separate_source_overlap_and_exact_topk_are_valid(mode):
    saved,spec,ids,cost = fixture(mode,mask_locations=[[0,1,16],[2],[0,1,16]])
    spec["config"]["trace_max_regions"] = 2
    result = receipts.validate_episode(saved,spec,ids,cost)
    assert len(result["original_candidates"]) == 3
