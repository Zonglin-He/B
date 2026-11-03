import copy

import numpy as np
import pytest

from tools.evolution_fifty import value_score_receipts as receipts


def rebind(saved):
    decoded = [dict(r,region=np.unpackbits(np.frombuffer(bytes.fromhex(r["region"]["data"]),dtype=np.uint8),
        bitorder="little").astype(bool).reshape(16,16)) for r in saved["candidates"]]
    for row,value in zip(saved["candidates"],receipts.runtime.native.image_worker.objective_coefficients(decoded,saved["weighting_arm"]),strict=True):
        row["objective_coefficient"] = value
    saved["candidate_sha256"] = receipts.runtime.native.prior.candidate_identity(decoded)


def fixture(dataset,point,*,empty=False,inactive=False):
    cfg = dict(dataset={"coco_object":"COCOObjectDataset","pc60":"PascalContext60Dataset"}[dataset],
        batch_size=2,steps=receipts.runtime.NATIVE_STEPS[dataset],trace_max_regions=35,trace_view_batch_size=2,
        lr=.001,data_dir="/synthetic/not-opened",data_subdir="synthetic",resize=[224,224],patch_size=[224,224],
        patch_stride=224,class_extensions=False)
    spec = dict(dataset=dataset,config=cfg,points=list(receipts.runtime.points(dataset)),
        vocabulary=dict(classes=["background","person","road"],queries=["background","person","road"],query_to_class=None))
    rows = []
    for i,source in enumerate([] if empty else (0,0,1)):
        mask = np.zeros((16,16),dtype=bool)
        mask.flat[i] = True
        confidence = 0. if inactive else (.9,.7,.4)[i]
        C = 0. if inactive else (.6,.8,.5)[i]
        rho = C if point["arm"] == "image_C" else confidence
        rows.append(dict(image_idx=source,batch_idx=source,class_idx=(1,2,0)[i],component_idx=i+1,area=1,
            confidence=confidence,original_support=confidence,target_support=None,residual_support=None,dominance=None,
            counterfactual_support=C if point["arm"] == "image_C" else None,quality=rho,reliability=rho,
            diagnostic_algorithm="C" if point["arm"] == "image_C" else "Confidence",omitted=False,
            preupdate_candidate_category_idx=(1,2,0)[i],region=dict(shape=[16,16],encoding="packbits-little-hex",
                data=np.packbits(mask.reshape(-1),bitorder="little").tobytes().hex(),true_pixels=1)))
    active = sum(r["reliability"] > receipts.EPS for r in rows)
    steps = point["steps"] if active else 0
    views = len(rows) if point["arm"] == "image_C" else 0
    calls = (views+1)//2
    cost = dict(arm=point["id"],point_id=point["id"],weighting_arm=point["arm"],dataset=dataset,
        images=2,patches=2,candidate_count=len(rows),active_candidates=active,steps=steps,optimizer_steps=steps,seconds=.2,
        native_steps=cfg["steps"],requested_steps=point["steps"],actual_step_observation_in_model_time=True,
        time_scope=receipts.TIME_SCOPE,auxiliary_checkpoint_readouts=0,GT_opened=False,
        eval_forward_calls=2,eval_forward_inputs=4,adapt_forward_calls=steps,adapt_forward_inputs=steps*2,
        diagnostic_forward_calls=calls,diagnostic_encoder_calls=calls,diagnostic_forward_inputs=views,diagnostic_encoder_inputs=views,
        encoder_calls=2+steps+calls,encoder_inputs=(2+steps)*2+views)
    saved = dict(**cost,fixed_point=copy.deepcopy(point),ordered_ids=["s0","s1"],candidates=rows,losses=[.4]*steps,
        receipt_float64_objective_coefficients_are_ideal_not_runtime=True)
    rebind(saved)
    cost["candidate_sha256"] = saved["candidate_sha256"]
    return saved,spec,["s0","s1"],cost


@pytest.mark.parametrize("dataset",receipts.runtime.NATIVE_STEPS)
@pytest.mark.parametrize("state",["normal","empty","inactive"])
def test_all_ten_formal_points_and_zero_active_cases_are_exact_and_shared(dataset,state):
    checked = {}
    for point in receipts.runtime.points(dataset):
        args = fixture(dataset,point,empty=state == "empty",inactive=state == "inactive")
        result = receipts.validate_episode(*args)
        assert result["candidates"] == (0 if state == "empty" else 3)
        assert result["EPS_active_candidates"] == (3 if state == "normal" else 0)
        assert result["diagnostic_views"] == (3 if state != "empty" and point["arm"] == "image_C" else 0)
        checked[point["id"]] = result
    receipts.validate_shared(dataset,checked)


@pytest.mark.parametrize("fault",["pointfloat","pointbool","extra_point_key","native_steps","cfg","gridfloat","time","aux","observe","GT","counter","steps","omitted","querymap","coefficient","formula","views"])
@pytest.mark.parametrize("arm",["image_C","image_confidence"])
def test_bad_grid_diagnostic_cost_and_normalization_receipts_fail(fault,arm):
    point = next(p for p in receipts.runtime.points("pc60") if p["arm"] == arm)
    saved,spec,ids,cost = fixture("pc60",point)
    row = saved["candidates"][0]
    if fault == "pointfloat":saved["fixed_point"]["steps"] = float(point["steps"])
    elif fault == "pointbool":saved["fixed_point"]["steps"] = True
    elif fault == "extra_point_key":saved["fixed_point"]["extra"] = 1
    elif fault == "native_steps":saved["native_steps"] = cost["native_steps"] = 6
    elif fault == "cfg":spec["config"]["steps"] = 6
    elif fault == "gridfloat":spec["points"][0]["steps"] = float(spec["points"][0]["steps"])
    elif fault == "time":saved["seconds"] = cost["seconds"] = float("nan")
    elif fault == "aux":saved["auxiliary_checkpoint_readouts"] = cost["auxiliary_checkpoint_readouts"] = 1
    elif fault == "observe":saved["actual_step_observation_in_model_time"] = cost["actual_step_observation_in_model_time"] = False
    elif fault == "GT":saved["GT_opened"] = cost["GT_opened"] = True
    elif fault == "counter":saved["encoder_inputs"] = cost["encoder_inputs"] = 100
    elif fault == "steps":saved["steps"] = cost["steps"] = 0
    elif fault == "omitted":row["omitted"] = True
    elif fault == "querymap":row["preupdate_candidate_category_idx"] = 2
    elif fault == "coefficient":row["objective_coefficient"] += .01
    elif fault == "formula":row["reliability"] += .01
    elif fault == "views":saved["diagnostic_forward_inputs"] = cost["diagnostic_forward_inputs"] = 999
    with pytest.raises(ValueError):receipts.validate_episode(saved,spec,ids,cost)


@pytest.mark.parametrize("fault",["duplicate","reversed","K","disconnected","overlap","component0"])
def test_hash_consistent_impossible_native_pool_fails(fault):
    point = receipts.runtime.points("pc60")[0]
    saved,spec,ids,cost = fixture("pc60",point)
    rows = saved["candidates"]
    if fault == "duplicate":rows[1].update(class_idx=rows[0]["class_idx"],component_idx=rows[0]["component_idx"],preupdate_candidate_category_idx=rows[0]["preupdate_candidate_category_idx"])
    elif fault == "reversed":rows.reverse()
    elif fault == "K":spec["config"]["trace_max_regions"] = 1
    elif fault == "component0":rows[0]["component_idx"] = 0
    elif fault == "overlap":rows[1]["region"] = copy.deepcopy(rows[0]["region"])
    else:
        raw = np.zeros((16,16),dtype=bool)
        raw.flat[[0,255]] = True
        rows[0]["area"] = rows[0]["region"]["true_pixels"] = 2
        rows[0]["region"]["data"] = np.packbits(raw.reshape(-1),bitorder="little").tobytes().hex()
    rebind(saved)
    cost["candidate_sha256"] = saved["candidate_sha256"]
    with pytest.raises(ValueError):receipts.validate_episode(saved,spec,ids,cost)


@pytest.mark.parametrize("fault",["missing","extra","pool","C_budget_drift","Conf_budget_drift"])
def test_shared_grid_preserves_original_pool_and_budget_independent_diagnostics(fault):
    points = receipts.runtime.points("pc60")
    checked = {p["id"]:receipts.validate_episode(*fixture("pc60",p)) for p in points}
    if fault == "missing":del checked[points[0]["id"]]
    elif fault == "extra":checked["foreign"] = checked[points[0]["id"]]
    elif fault == "pool":checked[points[0]["id"]]["candidate_sha256"] = "0"*64
    else:
        arm = "image_C" if fault.startswith("C_") else "image_confidence"
        key = next(p["id"] for p in points if p["arm"] == arm)
        checked[key]["diagnostic"][0][-1] += .1
    with pytest.raises(ValueError):receipts.validate_shared("pc60",checked)


@pytest.mark.parametrize("fault",["batch_float","K_float","view_float","lr_nan","lr_zero","missing_lr","extra","resize_float","patch_float","relative_dir","subdir_type","stride_bool"])
def test_complete_inherited_schema_rejects_invalid_native_config_before_receipts(fault):
    point = receipts.runtime.points("pc60")[0]
    saved,spec,ids,cost = fixture("pc60",point)
    cfg = spec["config"]
    if fault == "batch_float":cfg["batch_size"] = 2.
    elif fault == "K_float":cfg["trace_max_regions"] = 35.
    elif fault == "view_float":cfg["trace_view_batch_size"] = 2.
    elif fault == "lr_nan":cfg["lr"] = float("nan")
    elif fault == "lr_zero":cfg["lr"] = 0.
    elif fault == "missing_lr":del cfg["lr"]
    elif fault == "extra":cfg["unknown"] = 1
    elif fault == "resize_float":cfg["resize"] = [224.,224.]
    elif fault == "patch_float":cfg["patch_size"] = [224.,224.]
    elif fault == "relative_dir":cfg["data_dir"] = "relative"
    elif fault == "subdir_type":cfg["data_subdir"] = None
    else:cfg["patch_stride"] = True
    with pytest.raises((ValueError,RuntimeError)):receipts.validate_episode(saved,spec,ids,cost)


@pytest.mark.parametrize("fault",["same_query_adjacent","component_over_extent"])
def test_all_five_hash_consistent_component_impossibilities_reject(fault):
    for point in receipts.runtime.points("pc60"):
        saved,spec,ids,cost = fixture("pc60",point)
        if fault == "same_query_adjacent":
            saved["candidates"][1].update(class_idx=1,preupdate_candidate_category_idx=1)
        else:saved["candidates"][1]["component_idx"] = 1000000
        rebind(saved)
        cost["candidate_sha256"] = saved["candidate_sha256"]
        with pytest.raises(ValueError,match="adjacent|component"):
            receipts.validate_episode(saved,spec,ids,cost)


def test_distinct_queries_mapping_to_one_class_may_border_and_selected_component_gaps_are_legal():
    for point in receipts.runtime.points("pc60"):
        saved,spec,ids,cost = fixture("pc60",point)
        spec["vocabulary"] = dict(classes=["background","person"],queries=["background","person one","person two"],query_to_class=[0,1,1])
        spec["config"]["class_extensions"] = True
        saved["candidates"][1]["preupdate_candidate_category_idx"] = 1
        saved["candidates"][0]["component_idx"] = 4
        saved["candidates"][1]["component_idx"] = 8
        rebind(saved)
        cost["candidate_sha256"] = saved["candidate_sha256"]
        assert receipts.validate_episode(saved,spec,ids,cost)["candidates"] == 3
