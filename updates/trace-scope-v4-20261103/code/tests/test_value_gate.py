import pytest
import torch

from tools.evolution_fifty import value_gate as gate
from test_generalization_runtime import method
from test_image_full_worker import config, batch_meta


def fixture(dataset="pc60"):
    current,cfg = method(),config(2)
    current.steps = cfg["steps"] = gate.runtime.NATIVE_STEPS[dataset]
    cfg["dataset"] = "COCOObjectDataset" if dataset == "coco_object" else "PascalContext60Dataset"
    return current,cfg,torch.randn(2,3,32,32),batch_meta(2)


@pytest.mark.parametrize("dataset", ["coco_object", "pc60"])
def test_all_five_formal_budgets_every_step_and_final_state_match_direct_reference(dataset):
    current,cfg,images,meta = fixture(dataset)
    before = gate.runtime.native.shared.restored_state(current)
    reset = current.reset
    result = gate.compare_gate(current,dataset,images,None,torch.arange(2),meta,cfg,None)
    assert result["passed"] and len(result["points"]) == 5
    for point in gate.runtime.points(dataset):
        row = result["points"][point["id"]]
        assert row["actual_optimizer_steps"] == row["actual_counts"]["optimizer_steps"] == point["steps"]
        assert row["changed_labels"] == 0 and all(v <= 1e-5 for v in row["max_abs"].values())
        assert row["gate_only_reset_snapshots"] == 2 and row["gate_cost_not_primary_value_time"]
    assert current.reset == reset and "reset" not in current.__dict__
    assert current.steps == cfg["steps"] and gate.runtime.native.shared.restored_state(current) == before
    assert not current.optimizer._optimizer_step_post_hooks and not torch.cuda.is_initialized()


def test_changed_final_labels_fail_gate_and_restore_instrumentation(monkeypatch):
    current,cfg,images,meta = fixture()
    reset = current.reset
    original = gate.runtime.run_point
    def corrupt(*args,**kwargs):
        labels,stats,losses,cost = original(*args,**kwargs)
        labels[0] = (labels[0]+1)%2
        return labels,stats,losses,cost
    monkeypatch.setattr(gate.runtime,"run_point",corrupt)
    with pytest.raises(ValueError,match="numerical/label"):
        gate.compare_gate(current,"pc60",images,None,torch.arange(2),meta,cfg,None)
    assert current.reset == reset and "reset" not in current.__dict__
    assert not current.optimizer._optimizer_step_post_hooks


def test_noop_optimizer_cannot_pass_actual_update_gate():
    current,cfg,images,meta = fixture()
    current.optimizer.step = lambda *args,**kwargs:None
    with pytest.raises(ValueError,match="actual Adam"):
        gate.compare_gate(current,"pc60",images,None,torch.arange(2),meta,cfg,None)
    assert "reset" not in current.__dict__ and not current.optimizer._optimizer_step_post_hooks


def test_empty_candidates_are_not_a_numerical_gate():
    current,cfg,images,meta = fixture()
    with pytest.raises(ValueError,match="Vacuous"):
        gate.compare_gate(current,"pc60",images,torch.zeros(2,1,32,32,dtype=torch.bool),torch.arange(2),meta,cfg,None)
    assert "reset" not in current.__dict__ and not current.optimizer._optimizer_step_post_hooks


def test_bad_IDs_reject_before_gate_reset_reference_or_model(monkeypatch):
    current,cfg,images,meta = fixture()
    def forbidden(*args,**kwargs): pytest.fail("Bad IDs reached gate work")
    monkeypatch.setattr(current,"reset",forbidden)
    monkeypatch.setattr(gate.runtime.native,"make_reference",forbidden)
    with pytest.raises(ValueError,match="positional"):
        gate.compare_gate(current,"pc60",images,None,torch.tensor([0,0]),meta,cfg,None)


@pytest.mark.parametrize("field", ["steps","lr","trace_max_regions","trace_view_batch_size"])
@pytest.mark.parametrize("side", ["cfg","method"])
def test_contradictory_nonbudget_fields_reject_before_reference_or_model(monkeypatch,field,side):
    current,cfg,images,meta = fixture()
    if side == "cfg": cfg[field] = cfg[field]*2 if field == "lr" else cfg[field]+1
    else: setattr(current,field,getattr(current,field)*2 if field == "lr" else getattr(current,field)+1)
    def forbidden(*args,**kwargs): pytest.fail("Contradictory gate config reached model/reset/reference")
    monkeypatch.setattr(current,"reset",forbidden)
    monkeypatch.setattr(current.model,"forward",forbidden)
    monkeypatch.setattr(gate.runtime.native,"make_reference",forbidden)
    monkeypatch.setattr(gate.runtime.native,"legal_parameters",forbidden)
    with pytest.raises(ValueError,match="before reference"):
        gate.compare_gate(current,"pc60",images,None,torch.arange(2),meta,cfg,None)
