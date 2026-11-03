import numpy as np
import pytest
import torch

from tools.evolution_fifty import spatial_runtime as runtime
from tools.evolution_fifty import generalization_runtime as native
from test_generalization_runtime import method as production_method
from test_image_full_worker import config,batch_meta


def method():
    base = production_method()
    result = runtime.SpatialExperimentTRACE(classes=base.classes,lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device="cpu",view_mode="C")
    result.model.eval()
    return result


@pytest.mark.parametrize("mode",runtime.MODES)
def test_spatial_modes_really_adapt_with_original_candidates_and_exact_costs(mode):
    torch.manual_seed(23)
    current = method()
    images,meta,cfg = torch.randn(3,3,32,32),batch_meta(3),config(3)
    before = runtime.shared.restored_state(current)
    outputs,stats,losses,row,geometry = runtime.run_arm(current,mode,images,None,torch.arange(3),meta,cfg,None)
    assert len(outputs) == 3 and stats and len(losses) == cfg["steps"]
    assert row["active_candidates"] > 0 and geometry
    if mode == "confidence":
        assert row["diagnostic_forward_inputs"] == 0
    elif mode == "global":
        assert row["diagnostic_forward_inputs"] == len({r["batch_idx"] for r in stats})
    else:
        assert row["diagnostic_forward_inputs"] == len(stats)
    current.reset()
    assert runtime.shared.restored_state(current) == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mode,arm",[("native","image_C"),("confidence","image_confidence")])
def test_native_and_confidence_updates_match_production_weights_losses_and_labels(mode,arm):
    torch.manual_seed(3)
    current = method()
    base = native.ExperimentTRACE(classes=current.classes,lr=current.lr,steps=current.steps,
        trace_max_regions=current.trace_max_regions,trace_view_batch_size=current.trace_view_batch_size,
        vision_outputs=current.vision_outputs,prompts=current.prompt_templates,model=current.model,
        tokenize_fn=current.tokenize,device="cpu",view_mode="C")
    base.model.eval()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    rng = runtime.shared.rng_state(current.device)
    ours,stats,losses,row,_ = runtime.run_arm(current,mode,images,None,torch.arange(2),meta,cfg,None)
    weights = {n:p.detach().clone() for n,p in current.model.named_parameters() if p.requires_grad}
    current.reset()
    runtime.shared.restore_rng(rng,current.device)
    expected,other,other_losses,cost = native.run_arm(base,arm,images,None,torch.arange(2),meta,cfg,None)
    assert row["candidate_sha256"] == cost["candidate_sha256"]
    assert losses == other_losses
    for a,b in zip(ours,expected,strict=True):
        np.testing.assert_array_equal(a,b)
    for name,p in base.model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p,weights[name],rtol=0,atol=0)
    assert [s["reliability"] for s in stats] == [s["reliability"] for s in other]


def test_mode_or_config_change_rejected_before_adaptation():
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    with pytest.raises(ValueError,match="method/mode"):
        runtime.run_arm(current,"not_an_arm",images,None,torch.arange(2),meta,cfg,None)
    cfg["steps"] += 1
    with pytest.raises(ValueError,match="configuration differs"):
        runtime.run_arm(current,"native",images,None,torch.arange(2),meta,cfg,None)


def test_all_spatial_modes_keep_same_original_candidates_under_exact_reset():
    current = method()
    images,cfg,meta = torch.randn(4,3,32,32),config(4),batch_meta(4)
    rng = runtime.shared.rng_state(current.device)
    identities = []
    for mode in runtime.MODES:
        current.reset()
        runtime.shared.restore_rng(rng,current.device)
        _,_,_,row,_ = runtime.run_arm(current,mode,images,None,torch.arange(4),meta,cfg,None)
        identities.append(row["candidate_sha256"])
        assert not current.model._forward_hooks
        assert not current.optimizer._optimizer_step_pre_hooks
        assert not current.optimizer._optimizer_step_post_hooks
    assert len(set(identities)) == 1


def test_empty_active_episode_does_not_invent_updates_or_drop_geometry():
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    runtime.run_arm(current,"translated",images,None,torch.arange(2),meta,cfg,None)
    assert current.last_spatial_probe_receipts
    current.reset()
    _,stats,losses,row,geometry = runtime.run_arm(current,"translated",images,
        torch.zeros(2,1,32,32,dtype=torch.bool),torch.arange(2),meta,cfg,None)
    assert stats == losses == geometry == [] and row["steps"] == row["candidate_count"] == 0
    assert row["optimizer_steps"] == 0


def test_noop_optimizer_is_rejected_and_hook_is_removed():
    current = method()
    current.optimizer.step = lambda *args, **kwargs: None
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    with pytest.raises(ValueError,match="schedule"):
        runtime.run_arm(current,"native",images,None,torch.arange(2),meta,cfg,None)
    assert not current.optimizer.state
    assert not current.optimizer._optimizer_step_post_hooks


def test_zero_gradients_still_count_real_finite_Adam_updates():
    current = method()
    initial = {n:p.detach().clone() for n,p in current.model.named_parameters() if p.requires_grad}
    handles = [p.register_hook(lambda gradient: torch.zeros_like(gradient))
               for p in current.model.parameters() if p.requires_grad]
    try:
        images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
        _,_,losses,row,_ = runtime.run_arm(current,"native",images,None,torch.arange(2),meta,cfg,None)
        assert row["optimizer_steps"] == row["steps"] == len(losses) == cfg["steps"]
        assert current.optimizer.state
        for name,p in current.model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(initial[name],p,rtol=0,atol=0)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("fault",["step_jump","nonfinite_moment"])
def test_observed_Adam_rejects_bad_progression_or_nonfinite_state(fault):
    current = method()
    def corrupt(optimizer,*_):
        state = next(iter(optimizer.state.values()))
        if fault == "step_jump":
            state["step"].add_(1)
        else:
            state["exp_avg"].fill_(float("nan"))
    corruption = current.optimizer.register_step_post_hook(corrupt)
    try:
        images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
        with pytest.raises((ValueError,RuntimeError)):
            runtime.run_arm(current,"native",images,None,torch.arange(2),meta,cfg,None)
        assert len(current.optimizer._optimizer_step_post_hooks) == 1
    finally:
        corruption.remove()
