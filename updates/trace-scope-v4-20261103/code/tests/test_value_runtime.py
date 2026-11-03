import copy

import pytest
import torch

from tools.evolution_fifty import value_runtime as runtime
from test_generalization_runtime import method
from test_image_full_worker import config,batch_meta


@pytest.mark.parametrize("dataset,expected",[("coco_object",[15,30,30,60,120]),("pc60",[4,7,7,14,28])])
def test_fixed_grid_points_really_update_count_and_reset(dataset,expected):
    current,cfg = method(),config(2)
    cfg["dataset"] = "COCOObjectDataset" if dataset == "coco_object" else "PascalContext60Dataset"
    current.steps = cfg["steps"] = runtime.NATIVE_STEPS[dataset]
    current.arm = "image_C"
    images,meta = torch.randn(2,3,32,32),batch_meta(2)
    before = runtime.native.shared.restored_state(current)
    rng = runtime.native.shared.rng_state(current.device)
    identities,steps = [],[]
    for point in runtime.points(dataset):
        predictions,stats,losses,cost = runtime.run_point(current,dataset,point,images,None,torch.arange(2),meta,cfg,None,
                                                        initial_rng=rng)
        assert len(predictions) == 2 and stats and len(losses) == cost["optimizer_steps"] == point["steps"]
        identities.append(cost["candidate_sha256"])
        steps.append(cost["optimizer_steps"])
        assert current.steps == cfg["steps"] == runtime.NATIVE_STEPS[dataset]
        assert current.arm == "image_C" and runtime.native.shared.restored_state(current) == before
        assert not current.optimizer._optimizer_step_post_hooks and cost["GT_opened"] is False
    assert steps == expected and len(set(identities)) == 1
    assert not torch.cuda.is_initialized()


def test_point_outside_grid_cannot_change_steps_or_call_adaptation():
    current,cfg = method(),config(2)
    current.steps = cfg["steps"] = 30
    current.arm = "image_C"
    images,meta = torch.randn(2,3,32,32),batch_meta(2)
    bad = dict(arm="image_C",steps=31,id="arbitrary")
    with pytest.raises(ValueError,match="predeclared"):
        runtime.run_point(current,"coco_object",bad,images,None,torch.arange(2),meta,cfg,None,
                          initial_rng=runtime.native.shared.rng_state(current.device))
    assert current.steps == 30 and not current.optimizer.state


def test_noop_optimizer_rejected_and_native_budget_restored():
    current,cfg = method(),config(2)
    current.steps = cfg["steps"] = 7
    current.arm = "image_C"
    current.optimizer.step = lambda *args,**kwargs:None
    images,meta = torch.randn(2,3,32,32),batch_meta(2)
    with pytest.raises(ValueError,match="actual Adam"):
        runtime.run_point(current,"pc60",runtime.points("pc60")[0],images,None,torch.arange(2),meta,cfg,None,
                          initial_rng=runtime.native.shared.rng_state(current.device))
    assert current.steps == 7 and current.arm == "image_C" and not current.optimizer.state
    assert not current.optimizer._optimizer_step_post_hooks


def test_empty_active_point_has_zero_updates_without_dropping_outputs():
    current,cfg = method(),config(2)
    current.steps = cfg["steps"] = 7
    current.arm = "image_C"
    images,meta = torch.randn(2,3,32,32),batch_meta(2)
    predictions,stats,losses,cost = runtime.run_point(current,"pc60",runtime.points("pc60")[0],images,
        torch.zeros(2,1,32,32,dtype=torch.bool),torch.arange(2),meta,cfg,None,
        initial_rng=runtime.native.shared.rng_state(current.device))
    assert len(predictions) == 2 and not stats and not losses and cost["optimizer_steps"] == 0
    assert cost["requested_steps"] == 4 and current.steps == 7


@pytest.mark.parametrize("bad", [None,torch.tensor([0,0]),torch.tensor([7,7]),torch.tensor([0.,1.]),
                                torch.tensor([False,True]),torch.tensor([[0,1]]),torch.tensor([1,0])])
def test_bad_source_IDs_reject_before_reset_or_forward(monkeypatch,bad):
    current,cfg = method(),config(2)
    current.steps = cfg["steps"] = 7
    def forbidden(*args,**kwargs): pytest.fail("Invalid V01 IDs reached model/state work")
    rng = runtime.native.shared.rng_state(current.device)
    monkeypatch.setattr(current,"adapt",forbidden)
    monkeypatch.setattr(current,"reset",forbidden)
    monkeypatch.setattr(runtime.native.shared,"restored_state",forbidden)
    with pytest.raises(ValueError,match="positional"):
        runtime.run_point(current,"pc60",runtime.points("pc60")[0],torch.randn(2,3,32,32),None,bad,
                          batch_meta(2),cfg,None,initial_rng=rng)


@pytest.mark.parametrize("fault", ["float","bool","extra","missing","wrongid"])
def test_ill_typed_point_rejects_before_any_model_work(monkeypatch,fault):
    current,cfg = method(),config(2)
    current.steps = cfg["steps"] = 7
    point = dict(runtime.points("pc60")[0])
    if fault == "float": point["steps"] = float(point["steps"])
    elif fault == "bool": point["steps"] = True
    elif fault == "extra": point["extra"] = 0
    elif fault == "missing": point.pop("id")
    else: point["id"] = "wrong"
    def forbidden(*args,**kwargs): pytest.fail("Malformed point reached model/state")
    rng = runtime.native.shared.rng_state(current.device)
    monkeypatch.setattr(current,"adapt",forbidden)
    monkeypatch.setattr(current,"reset",forbidden)
    monkeypatch.setattr(runtime.native.shared,"restored_state",forbidden)
    with pytest.raises(ValueError,match="grid"):
        runtime.run_point(current,"pc60",point,torch.randn(2,3,32,32),None,torch.arange(2),batch_meta(2),cfg,None,
                          initial_rng=rng)


def test_public_point_rejects_expanded_trainable_set_even_with_consistent_reset(monkeypatch):
    current,cfg = method(),config(2)
    current.steps = cfg["steps"] = 7
    current.model.visual.conv1.weight.requires_grad_(True)
    current.optimizer = torch.optim.Adam([p for p in current.model.parameters() if p.requires_grad],lr=current.lr)
    current.model_state,current.optimizer_state = current.copy_model_and_optimizer(current.model,current.optimizer)
    rng = runtime.native.shared.rng_state(current.device)
    def forbidden(*args,**kwargs): pytest.fail("Illegal visual membership reached adaptation")
    monkeypatch.setattr(current,"adapt",forbidden)
    with pytest.raises(ValueError,match="visual-LN"):
        runtime.run_point(current,"pc60",runtime.points("pc60")[0],torch.randn(2,3,32,32),None,
                          torch.arange(2),batch_meta(2),cfg,None,initial_rng=rng)
