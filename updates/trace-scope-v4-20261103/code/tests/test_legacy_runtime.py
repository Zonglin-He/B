import copy
import tarfile

import numpy as np
import pytest
import torch

from tools.evolution_fifty import legacy_runtime as runtime
from test_generalization_runtime import method as production_method
from test_image_full_worker import config as native_config,batch_meta


def config(batch=2):
    result = native_config(batch)
    result["dataset"] = "COCOObjectDataset"
    return result


def method():
    base = production_method()
    current = runtime.LegacyExperimentTRACE(dataset_name="COCOObjectDataset",benchmark_classes=base.classes,
        query_to_class=None,classes=base.classes,lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device="cpu",view_mode="C")
    current.model.eval()
    return current


@pytest.mark.parametrize("mode",runtime.MODES)
def test_pinned_legacy_or_C_both_objectives_really_adapt_and_reset(mode):
    torch.manual_seed(23)
    current = method()
    before = runtime.native.shared.restored_state(current)
    images,cfg,meta = torch.randn(3,3,32,32),config(3),batch_meta(3)
    labels,stats,losses,cost = runtime.run_arm(current,mode,images,None,torch.arange(3),meta,cfg,None)
    assert len(labels) == 3 and stats and len(losses) == cost["optimizer_steps"] == cfg["steps"]
    assert cost["diagnostic_encoder_inputs"] == len(stats)*(3 if mode.startswith("legacy") else 1)
    assert not cost["old_engine_loaded"] and not cost["old_OOM_iterator_selected"]
    if mode.startswith("legacy"):
        oracle = current.legacy["modules"]["reliability"].trace_reliability
        for row in stats:
            assert row["reliability"] == oracle(row["target_support"],row["counterfactual_support"],row["residual_support"],row["role"]).reliability
    current.reset()
    assert runtime.native.shared.restored_state(current) == before
    assert not torch.cuda.is_initialized()


def test_singleC_image_update_matches_unchanged_production_exactly():
    current = method()
    base = runtime.ExperimentTRACE(classes=current.classes,lr=current.lr,steps=current.steps,
        trace_max_regions=current.trace_max_regions,trace_view_batch_size=current.trace_view_batch_size,
        vision_outputs=current.vision_outputs,prompts=current.prompt_templates,model=copy.deepcopy(current.model),
        tokenize_fn=current.tokenize,device="cpu",view_mode="C")
    base.model.eval()
    images,cfg,meta = torch.randn(3,3,32,32),config(3),batch_meta(3)
    rng = runtime.native.shared.rng_state(current.device)
    ours,stats,losses,cost = runtime.run_arm(current,"C.image",images,None,torch.arange(3),meta,cfg,None)
    runtime.native.shared.restore_rng(rng,current.device)
    expected,other,reference,ref_cost = runtime.native.run_arm(base,"image_C",images,None,torch.arange(3),meta,cfg,None)
    assert losses == reference and cost["candidate_sha256"] == ref_cost["candidate_sha256"]
    assert [r["reliability"] for r in stats] == [r["reliability"] for r in other]
    for a,b in zip(ours,expected,strict=True):
        np.testing.assert_array_equal(a,b)
    for name,p in current.model.named_parameters():
        torch.testing.assert_close(p,dict(base.model.named_parameters())[name],rtol=0,atol=0)


def test_old_episode_objective_source_is_byte_identical_to_pinned_archive_not_a_substitute():
    with tarfile.open(runtime.legacy_rho.ARCHIVE,"r:gz") as archive:
        member = archive.extractfile("trace_ovss/method/trace/objective.py")
        assert member is not None
        import hashlib
        assert hashlib.sha256(member.read()).hexdigest() == runtime.OBJECTIVE_SHA


def test_old_OOM_retry_iterator_never_selected_and_original_candidates_unchanged(monkeypatch):
    current = method()
    component = current.legacy["modules"]["interventions"].TraceInterventionMixin
    monkeypatch.setattr(component,"_forward_trace_view_batches",lambda *_:pytest.fail("Old retry iterator selected"))
    images,cfg,meta = torch.randn(3,3,32,32),config(3),batch_meta(3)
    rng = runtime.native.shared.rng_state(current.device)
    identities = []
    for mode in runtime.MODES:
        current.reset()
        runtime.native.shared.restore_rng(rng,current.device)
        *_,cost = runtime.run_arm(current,mode,images,None,torch.arange(3),meta,cfg,None)
        identities.append(cost["candidate_sha256"])
    assert len(set(identities)) == 1


def test_empty_candidates_do_not_invent_legacy_views_or_updates():
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    _,stats,losses,cost = runtime.run_arm(current,"legacy.image",images,torch.zeros(2,1,32,32,dtype=torch.bool),
        torch.arange(2),meta,cfg,None)
    assert stats == losses == [] and cost["optimizer_steps"] == cost["diagnostic_forward_calls"] == 0


def test_actual_mixed_thing_stuff_legacy_view_count_is_three_plus_two_not_forced_three():
    base = production_method()
    current = runtime.LegacyExperimentTRACE(dataset_name="PascalContext60Dataset",benchmark_classes=["person","road"],
        query_to_class=None,classes=["person","road"],lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device="cpu",view_mode="C")
    current.model.eval()
    current.score_mode = "legacy"
    assert current.legacy_query_roles == ("thing","stuff")
    images = torch.randn(1,3,32,32)
    masks = [np.array([[1,0],[1,0]],dtype=bool),np.array([[0,1],[0,1]],dtype=bool)]
    candidates = [dict(image_idx=0,batch_idx=0,class_idx=i,component_idx=0,area=2,confidence=.5,region=mask)
                  for i,mask in enumerate(masks)]
    with current.count_work() as work:
        stats = current.evaluate_trace_views(images,torch.full((1,2,2,2),.5),candidates)
    assert work["diagnostic_forward_inputs"] == 5
    assert work["diagnostic_forward_calls"] == (5+current.trace_view_batch_size-1)//current.trace_view_batch_size
    assert stats[0]["role"] == "thing" and stats[0]["residual_support"] is not None
    assert stats[1]["role"] == "stuff" and stats[1]["residual_support"] is None
    assert stats[1]["reliability"] == min(stats[1]["target_support"],stats[1]["counterfactual_support"])
    assert not current.optimizer.state


@pytest.mark.parametrize("fault",["ids","noop"])
def test_legacy_rejects_wrong_identity_or_fake_steps(fault):
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    ids = torch.arange(2)
    if fault == "ids":
        ids = torch.tensor([0,0])
    else:
        current.optimizer.step = lambda *_,**__:None
    with pytest.raises(ValueError):
        runtime.run_arm(current,"legacy.image",images,None,ids,meta,cfg,None)
    assert not current.optimizer._optimizer_step_post_hooks


@pytest.mark.parametrize("mapping",[[.0,1.0],[False,True],["0","1"],[0],[0,2],[-1,0]])
def test_constructor_bad_mapping_rejected_before_native_initialization(monkeypatch,mapping):
    monkeypatch.setattr(runtime.ExperimentTRACE,"__init__",lambda *_a,**_k:pytest.fail("Invalid mapping reached native initialization"))
    with pytest.raises(ValueError,match="mapping"):
        runtime.LegacyExperimentTRACE(dataset_name="COCOObjectDataset",benchmark_classes=["a","b"],
            query_to_class=mapping,classes=["a","b"])


@pytest.mark.parametrize("fault",["taxonomy","reversed","short","float","boolean","none_nonidentity"])
def test_bound_roles_and_runtime_readout_must_agree_before_any_forward(fault):
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    mapping = None
    if fault == "taxonomy":
        cfg["dataset"] = "PascalContext60Dataset"
    elif fault == "reversed":
        mapping = [1,0]
    elif fault == "short":
        mapping = [0]
    elif fault == "float":
        mapping = [0.,1.]
    elif fault == "boolean":
        mapping = [False,True]
    else:
        current.legacy_effective_mapping = (1,0)
    calls = []
    h = current.model.register_forward_pre_hook(lambda *_:calls.append(1))
    try:
        with pytest.raises(ValueError,match="mapping"):
            runtime.run_arm(current,"legacy.image",images,None,torch.arange(2),meta,cfg,mapping)
        assert not calls and not current.optimizer.state
    finally:
        h.remove()


def test_identity_readout_alias_and_repeated_query_extensions_are_legal():
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    runtime.run_arm(current,"legacy.image",images,None,torch.arange(2),meta,cfg,[0,1])
    base = production_method()
    expanded = runtime.LegacyExperimentTRACE(dataset_name="COCOObjectDataset",benchmark_classes=["object","surface"],
        query_to_class=[0,0,1],classes=["object","object alias","surface"],lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device="cpu",view_mode="C")
    expanded.model.eval()
    labels,*_ = runtime.run_arm(expanded,"legacy.image",images,None,torch.arange(2),meta,cfg,[0,0,1])
    assert all(p.max() < 2 for p in labels) and len(expanded.legacy_query_roles) == 3
