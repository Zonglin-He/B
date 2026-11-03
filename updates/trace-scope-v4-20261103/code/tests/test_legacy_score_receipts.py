import copy

import numpy as np
import pytest

from test_legacy_score_components import fixture as components_fixture
from tools.evolution_fifty import legacy_score_components as loader
from tools.evolution_fifty import legacy_score_receipts as receipts


def fixture(monkeypatch,mode="C.image",*,empty=False,inactive=False,sources=(0,0,1),locations=None):
    content,archive = components_fixture(monkeypatch)
    components = loader.load_components_bytes(content,archive)
    monkeypatch.setattr(receipts.legacy_rho,"ARCHIVE_SHA",archive["sha256"])
    classes = ["background","person","road"]
    spec = dict(dataset="pc60",config=dict(dataset="PascalContext60Dataset",batch_size=2,steps=7,
        trace_max_regions=35,trace_view_batch_size=2),
        vocabulary=dict(classes=classes,queries=classes,query_to_class=None))
    ids,candidates,decoded,views = ["s0","s1"],[],[],0
    for i,source in enumerate([] if empty else sources):
        cls = (1,2,0)[i]
        mask = np.zeros((16,16),dtype=bool)
        mask.flat[[i] if locations is None else locations[i]] = True
        role = ("thing","stuff","background")[i]
        target,context,residual = (.8,.6,.9 if role == "thing" else None)
        if inactive:target,context,residual = 0.,0.,0. if role == "thing" else None
        if mode.startswith("C."):
            target,residual,dominance = None,None,None
            quality = reliability = context
            algorithm = "C"
            views += 1
        else:
            score = components["modules"]["reliability"].trace_reliability(target,context,residual,role)
            quality,dominance,reliability = score.quality,score.dominance,score.reliability
            algorithm = "pinned_legacy_rho"
            views += len(components["modules"]["interventions"].trace_view_types(role))
        row = dict(image_idx=source,batch_idx=source,class_idx=cls,component_idx=i+1,area=int(mask.sum()),
            confidence=(.9,.7,.4)[i],original_support=(.9,.7,.4)[i],target_support=target,
            counterfactual_support=context,residual_support=residual,dominance=dominance,quality=quality,
            reliability=reliability,diagnostic_algorithm=algorithm,preupdate_candidate_category_idx=cls,
            region=dict(shape=[16,16],encoding="packbits-little-hex",true_pixels=int(mask.sum()),
                data=np.packbits(mask.reshape(-1),bitorder="little").tobytes().hex()))
        if mode.startswith("legacy."):row["role"] = role
        candidates.append(row)
        decoded.append(dict(row,region=mask))
    arm = "episode_min" if mode.endswith(".episode") else "image_C"
    for row,value in zip(candidates,receipts.runtime.native.image_worker.objective_coefficients(decoded,arm),strict=True):
        row["objective_coefficient"] = value
    active = sum(r["reliability"] > receipts.EPS for r in candidates)
    steps = 7 if active else 0
    calls = (views+1)//2
    cost = dict(arm=mode,images=2,patches=2,candidate_count=len(candidates),active_candidates=active,
        seconds=.2,steps=steps,optimizer_steps=steps,GT_opened=False,
        candidate_sha256=receipts.runtime.native.prior.candidate_identity(decoded),
        legacy_archive_sha256=components["archive_sha256"],legacy_source_members=components["member_sha256"],
        episode_objective_sha256=receipts.runtime.OBJECTIVE_SHA,
        old_engine_loaded=False,old_optimizer_loaded=False,old_OOM_iterator_selected=False,
        eval_forward_calls=2,eval_forward_inputs=4,adapt_forward_calls=steps,adapt_forward_inputs=steps*2,
        diagnostic_forward_calls=calls,diagnostic_encoder_calls=calls,
        diagnostic_forward_inputs=views,diagnostic_encoder_inputs=views,
        encoder_calls=2+steps+calls,encoder_inputs=(2+steps)*2+views)
    saved = dict(**cost,scientific_identity="M03_"+mode,ordered_ids=ids,candidates=candidates,losses=[.4]*steps,
        receipt_float64_objective_coefficients_are_ideal_not_runtime=True)
    return saved,spec,ids,cost,components


@pytest.mark.parametrize("mode",receipts.MODES)
@pytest.mark.parametrize("state",["normal","empty","inactive"])
def test_all_conditions_native_original_pool_and_role_dependent_view_counts(monkeypatch,mode,state):
    args = fixture(monkeypatch,mode,empty=state == "empty",inactive=state == "inactive")
    result = receipts.validate_episode(*args)
    assert result["candidates"] == (0 if state == "empty" else 3)
    assert result["EPS_active_candidates"] == (3 if state == "normal" else 0)
    assert result["diagnostic_views"] == (0 if state == "empty" else 3 if mode.startswith("C.") else 7)


def test_shared_identity_and_normalization_only_diagnostic_equality(monkeypatch):
    checked = {mode:receipts.validate_episode(*fixture(monkeypatch,mode)) for mode in receipts.MODES}
    receipts.validate_shared(checked)
    wrong = copy.deepcopy(checked)
    wrong["C.episode"]["diagnostic"][0]["counterfactual_support"] += .1
    with pytest.raises(ValueError,match="normalization-only"):receipts.validate_shared(wrong)
    del checked["legacy.episode"]
    with pytest.raises(ValueError,match="original candidates"):receipts.validate_shared(checked)


@pytest.mark.parametrize("fault",["role","formula","views","GT","archive","objective","coeff","target","residual","component0","mask","ids","K","legacy_engine"])
def test_legacy_metadata_drift_rejects(monkeypatch,fault):
    saved,spec,ids,cost,components = fixture(monkeypatch,"legacy.image")
    row = saved["candidates"][0]
    if fault == "role":row["role"] = "stuff"
    elif fault == "formula":row["reliability"] = .1
    elif fault == "views":saved["diagnostic_forward_inputs"] = cost["diagnostic_forward_inputs"] = 9
    elif fault == "GT":saved["GT_opened"] = True
    elif fault == "archive":saved["legacy_archive_sha256"] = "0"*64
    elif fault == "objective":saved["episode_objective_sha256"] = "0"*64
    elif fault == "coeff":row["objective_coefficient"] += .1
    elif fault == "target":row["target_support"] = float("nan")
    elif fault == "residual":row["residual_support"] = None
    elif fault == "component0":row["component_idx"] = 0
    elif fault == "mask":row["region"]["data"] = "0"*64
    elif fault == "ids":saved["ordered_ids"] = ids[::-1]
    elif fault == "K":spec["config"]["trace_max_regions"] = 1
    elif fault == "legacy_engine":saved["old_engine_loaded"] = True
    with pytest.raises(ValueError):receipts.validate_episode(saved,spec,ids,cost,components)


@pytest.mark.parametrize("mode",receipts.MODES)
@pytest.mark.parametrize("fault",["disconnected","overlap","K"])
def test_impossible_original_pool_rejects_before_normalization(monkeypatch,mode,fault):
    options = dict(locations=[[0,255],[1],[2]]) if fault == "disconnected" else dict(locations=[[0],[0],[2]]) if fault == "overlap" else dict(sources=(0,0,0))
    saved,spec,ids,cost,components = fixture(monkeypatch,mode,**options)
    if fault == "K":spec["config"]["trace_max_regions"] = 2
    with pytest.raises(ValueError,match="component area|Top-K"):
        receipts.validate_episode(saved,spec,ids,cost,components)


def rebind(saved,cost):
    decoded = [dict(r,region=np.unpackbits(np.frombuffer(bytes.fromhex(r["region"]["data"]),dtype=np.uint8),
        bitorder="little").astype(bool).reshape(16,16)) for r in saved["candidates"]]
    arm = "episode_min" if saved["arm"].endswith(".episode") else "image_C"
    for row,value in zip(saved["candidates"],receipts.runtime.native.image_worker.objective_coefficients(decoded,arm),strict=True):
        row["objective_coefficient"] = value
    saved["candidate_sha256"] = cost["candidate_sha256"] = receipts.runtime.native.prior.candidate_identity(decoded)


@pytest.mark.parametrize("fault",["duplicate_ID","reversed_order"])
def test_all_four_hash_consistent_impossible_native_sequences_reject(monkeypatch,fault):
    for mode in receipts.MODES:
        saved,spec,ids,cost,components = fixture(monkeypatch,mode,locations=[[0],[2],[4]])
        if fault == "duplicate_ID":
            # Same role/class and view count, different mask: hash alone cannot reject.
            row = copy.deepcopy(saved["candidates"][0])
            row["region"] = saved["candidates"][1]["region"]
            row["confidence"] = row["original_support"] = .7
            saved["candidates"][1] = row
            views = 3 if mode.startswith("C.") else 8
            calls = (views+1)//2
            update = dict(diagnostic_forward_calls=calls,diagnostic_encoder_calls=calls,
                diagnostic_forward_inputs=views,diagnostic_encoder_inputs=views,
                encoder_calls=2+saved["steps"]+calls,encoder_inputs=(2+saved["steps"])*2+views)
            saved.update(update)
            cost.update(update)
        else:saved["candidates"].reverse()
        rebind(saved,cost)
        with pytest.raises(ValueError,match="duplicate original|native candidate order"):
            receipts.validate_episode(saved,spec,ids,cost,components)


@pytest.mark.parametrize("mode",receipts.MODES)
def test_legal_ties_and_local_component_reuse_remain_valid(monkeypatch,mode):
    saved,spec,ids,cost,components = fixture(monkeypatch,mode)
    for row in saved["candidates"]:
        row["component_idx"] = 1
        row["confidence"] = row["original_support"] = .8
    rebind(saved,cost)
    assert receipts.validate_episode(saved,spec,ids,cost,components)["candidates"] == 3
