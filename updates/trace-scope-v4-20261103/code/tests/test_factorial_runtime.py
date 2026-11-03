import copy

import numpy as np
import pytest
import torch

from tools.evolution_fifty import factorial_runtime as runtime
from test_generalization_runtime import method as production_method
from test_image_full_worker import config,batch_meta


def method():
    base = production_method()
    result = runtime.FactorialExperimentTRACE(classes=base.classes,lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device="cpu",view_mode="C")
    result.model.eval()
    return result


def run(current,arm,common=False,valid=None):
    images = torch.randn(3,3,32,32)
    return runtime.run_arm(current,arm,common,images,valid,torch.arange(3),batch_meta(3),config(3),None,["a","b","c"])


@pytest.mark.parametrize("common",[False,True])
@pytest.mark.parametrize("arm",runtime.ARMS)
def test_all_crossed_arms_really_update_and_record_native_dtype_coefficients(arm,common):
    torch.manual_seed(1)
    current = method()
    initial = runtime.native.shared.restored_state(current)
    output,stats,losses,cost,assignment,coefficients = run(current,arm,common)
    assert len(output) == 3 and stats and losses and coefficients
    assert cost["optimizer_steps"] == len(losses) == config(3)["steps"]
    assert cost["diagnostic_encoder_inputs"] == len(stats)
    assert not cost["production_Confidence_timing"]
    assert assignment["scope"] == ("common_active_conditioned" if common else "natural_per_arm_admission")
    assert all(c["dtype"] == "torch.float32" and c["coefficient"] >= 0 for c in coefficients)
    assert abs(sum(c["coefficient"] for c in coefficients)-1) < 1e-6
    current.reset()
    assert runtime.native.shared.restored_state(current) == initial
    assert not current.optimizer._optimizer_step_post_hooks
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("arm,native_arm",[("Dc_Oc","image_C"),("Dk_Ok","image_confidence")])
def test_uncrossed_native_update_loss_parameter_and_label_parity(arm,native_arm):
    torch.manual_seed(3)
    current = method()
    # Construct the reference's immutable reset snapshots from the same model.
    base = runtime.ExperimentTRACE(classes=current.classes,lr=current.lr,steps=current.steps,
        trace_max_regions=current.trace_max_regions,trace_view_batch_size=current.trace_view_batch_size,
        vision_outputs=current.vision_outputs,prompts=current.prompt_templates,model=copy.deepcopy(current.model),
        tokenize_fn=current.tokenize,device="cpu",view_mode="C")
    base.model.eval()
    images,meta,cfg = torch.randn(3,3,32,32),batch_meta(3),config(3)
    rng = runtime.native.shared.rng_state(current.device)
    ours,stats,losses,cost,_,_ = runtime.run_arm(current,arm,False,images,None,torch.arange(3),meta,cfg,None,["a","b","c"])
    runtime.native.shared.restore_rng(rng,current.device)
    expected,other,ref_losses,ref_cost = runtime.native.run_arm(base,native_arm,images,None,torch.arange(3),meta,cfg,None)
    assert losses == ref_losses and cost["candidate_sha256"] == ref_cost["candidate_sha256"]
    assert [s["reliability"] for s in stats] == [s["reliability"] for s in other]
    for a,b in zip(ours,expected,strict=True):
        np.testing.assert_array_equal(a,b)
    other_parameters = dict(base.model.named_parameters())
    for name,value in current.model.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value,other_parameters[name],rtol=0,atol=0)


def test_all_arms_keep_original_candidate_identity_and_source_multisets():
    current = method()
    images,meta,cfg = torch.randn(3,3,32,32),batch_meta(3),config(3)
    rng = runtime.native.shared.rng_state(current.device)
    identities = []
    for arm in runtime.ARMS:
        current.reset()
        runtime.native.shared.restore_rng(rng,current.device)
        _,stats,_,row,assignment,_ = runtime.run_arm(current,arm,False,images,None,torch.arange(3),meta,cfg,None,["a","b","c"])
        identities.append(row["candidate_sha256"])
        for source in ("a","b","c"):
            indices = [i for i,r in enumerate(assignment["candidates"]) if r["source_id"] == source]
            native_key = "C" if arm.startswith("Dc") else "kappa"
            assert sorted(stats[i]["reliability"] for i in indices) == sorted(assignment["candidates"][i][native_key] for i in indices)
    assert len(set(identities)) == 1


def test_empty_admission_resets_receipts_and_performs_no_update():
    current = method()
    run(current,"Dc_Ok")
    current.reset()
    _,stats,losses,row,assignment,coefficients = run(current,"Dc_Ok",valid=torch.zeros(3,1,32,32,dtype=torch.bool))
    assert stats == losses == coefficients == []
    assert assignment["candidate_count"] == row["optimizer_steps"] == 0


def test_noop_Adam_rejected_and_invalid_source_ids_fail_before_adaptation():
    current = method()
    images,meta,cfg = torch.randn(3,3,32,32),batch_meta(3),config(3)
    with pytest.raises(ValueError,match="source IDs"):
        runtime.run_arm(current,"Dc_Oc",False,images,None,torch.arange(3),meta,cfg,None,["a","a","c"])
    current.optimizer.step = lambda *args,**kwargs: None
    with pytest.raises(ValueError,match="receipts differ"):
        run(current,"Dc_Oc")
    assert not current.optimizer._optimizer_step_post_hooks


def test_natural_and_common_admission_do_not_silently_equate_EPS_pools(monkeypatch):
    current = method()
    current.source_ids = ("source",)
    current.factorial_arm = "Dc_Ok"
    rows = [dict(image_idx=0,batch_idx=0,class_idx=i,component_idx=0,area=1,
                 confidence=kappa,reliability=C,quality=C,counterfactual_support=C)
            for i,(kappa,C) in enumerate(((.9,0.),(.2,.8),(.5,.4)))]
    monkeypatch.setattr(runtime.ExperimentTRACE,"evaluate_trace_views",lambda *_: copy.deepcopy(rows))
    natural = current.evaluate_trace_views(None,None,None)
    assert [r["reliability"] for r in natural] == [.8,0.,.4]
    assert not any(r["factorial_excluded"] for r in natural)
    current.common_active = True
    common = current.evaluate_trace_views(None,None,None)
    assert [r["reliability"] for r in common] == [0.,.4,.8]
    assert [r["factorial_excluded"] for r in common] == [True,False,False]
    assert current.last_factorial_assignment["retained_candidates"] == 2


@pytest.mark.parametrize("ids",[None,[0,1,2],torch.tensor([0,0,2]),torch.tensor([2,0,1]),
    torch.tensor([.9,1.9,2.9]),torch.tensor([0,1,3]),torch.tensor([False,True,True]),
    torch.tensor([[0,1,2]]),torch.tensor([0,1]),torch.tensor(0)])
def test_invalid_positional_image_ids_fail_before_any_forward_or_update(ids):
    current = method()
    images,meta,cfg = torch.randn(3,3,32,32),batch_meta(3),config(3)
    forwards,updates = [],[]
    a = current.model.register_forward_pre_hook(lambda *_: forwards.append(1))
    b = current.optimizer.register_step_post_hook(lambda *_: updates.append(1))
    try:
        with pytest.raises(ValueError,match="explicit integer positional"):
            runtime.run_arm(current,"Dc_Ok",False,images,None,ids,meta,cfg,None,["a","b","c"])
        assert forwards == updates == [] and not current.optimizer.state
        assert current.last_factorial_assignment is None
    finally:
        a.remove()
        b.remove()


@pytest.mark.parametrize("dtype",[torch.uint8,torch.int16,torch.int32,torch.int64])
def test_legal_integer_positional_ids_keep_exact_source_binding(dtype):
    current = method()
    images,meta,cfg = torch.randn(3,3,32,32),batch_meta(3),config(3)
    _,stats,_,_,assignment,_ = runtime.run_arm(current,"Dc_Ok",False,images,None,
        torch.arange(3).to(dtype),meta,cfg,None,["a","b","c"])
    for row,candidate in zip(stats,assignment["candidates"],strict=True):
        assert candidate["source_id"] == ["a","b","c"][row["batch_idx"]]


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.float64])
def test_receipt_uses_actual_native_dtype_denominator_not_float64_ideal(dtype):
    current = method()
    entropy = torch.ones(2,2,2,dtype=dtype,requires_grad=True)
    terms = [dict(image_idx=i//2,batch_idx=i//2,rho=rho,indices=torch.tensor([i%2]))
             for i,rho in enumerate((.9,.3,.1,.2))]
    expected = runtime.ExperimentTRACE.prepare_trace_objective(current,entropy,terms)
    observed = current.prepare_trace_objective(entropy,terms)
    for (a,an),(b,bn) in zip(expected,observed,strict=True):
        torch.testing.assert_close(an,bn,rtol=0,atol=0)
        for (_,_,aw),(_,_,bw) in zip(a,b,strict=True):
            torch.testing.assert_close(aw,bw,rtol=0,atol=0)
    assert all(r["dtype"] == str(dtype) for r in current.last_factorial_coefficients)
    objective = current.trace_region_normalized_entropy(entropy,terms,prepared=observed)
    reference = runtime.native.direct_objective(entropy,terms)
    torch.testing.assert_close(objective,reference,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(objective,entropy)[0],
                               torch.autograd.grad(reference,entropy)[0],rtol=0,atol=0)
