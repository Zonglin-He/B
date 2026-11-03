import copy

import pytest
import torch

from tools.evolution_fifty import spatial_gate as gate
from test_spatial_runtime import method
from test_image_full_worker import config, batch_meta


def test_all_spatial_modes_have_actual_loss_gradient_Adam_and_production_parity():
    torch.manual_seed(19)
    current = method()
    images, cfg, meta = torch.randn(2,3,32,32), config(2), batch_meta(2)
    initial = gate.shared.restored_state(current)
    result = gate.compare_gate(current,images,None,torch.arange(2),meta,cfg,None)
    assert result["passed"] and result["GT_opened"] is False
    assert set(result["modes"]) == set(gate.spatial.MODES)
    for mode, row in result["modes"].items():
        assert row["actual_optimizer_steps"] == cfg["steps"] and row["changed_labels"] == 0
        assert max(row["max_abs"].values()) <= 1e-5
        if mode in ("native", "confidence"):
            assert row["unchanged_production_parity"]["passed"]
    assert gate.shared.restored_state(current) == initial
    assert not current.model._forward_hooks and not current.optimizer._optimizer_step_post_hooks
    assert not torch.cuda.is_initialized()


def test_spatial_gate_rejects_empty_panel_instead_of_vacuous_pass():
    current = method()
    images, cfg, meta = torch.randn(2,3,32,32), config(2), batch_meta(2)
    with pytest.raises(ValueError, match="Vacuous"):
        gate.compare_gate(current,images,torch.zeros(2,1,32,32,dtype=torch.bool),torch.arange(2),meta,cfg,None)
    assert not current.model._forward_hooks and not current.optimizer._optimizer_step_post_hooks


@pytest.mark.parametrize("fault", ["gradient", "label", "candidate"])
def test_spatial_gate_detects_actual_receipt_mismatches(fault):
    current = method()
    images, cfg, meta = torch.randn(2,3,32,32), config(2), batch_meta(2)
    rng = gate.shared.rng_state(current.device)
    first = gate.capture(current,"native",images,None,torch.arange(2),meta,cfg,None,rng)
    other = copy.deepcopy(first)
    if fault == "gradient":
        next(iter(other["steps"][0]["gradients"].values())).add_(.1)
    elif fault == "label":
        other["predictions"][0][0,0] ^= 1
    else:
        other["counts"]["candidate_sha256"] = "different"
    with pytest.raises(ValueError):
        gate.compare(first,other,cfg)


@pytest.mark.parametrize("field",["gradients","parameters","adam","loss","final_first","final_last"])
@pytest.mark.parametrize("nonfinite",[float("nan"),float("inf")])
def test_gate_rejects_each_nonfinite_receipt_before_max_aggregation(field,nonfinite):
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    first = gate.capture(current,"native",images,None,torch.arange(2),meta,cfg,None,
                         gate.shared.rng_state(current.device))
    other = copy.deepcopy(first)
    if field == "loss":
        other["losses"][-1] = nonfinite
    elif field.startswith("final"):
        list(other["final"].values())[0 if field == "final_first" else -1].fill_(nonfinite)
    else:
        list(other["steps"][-1][field].values())[-1].fill_(nonfinite)
    with pytest.raises(ValueError,match="nonfinite"):
        gate.compare(first,other,cfg)


def test_gate_rejects_finite_inputs_with_overflowed_derived_difference():
    maximum = torch.finfo(torch.float32).max
    with pytest.raises(ValueError,match="derived"):
        gate.finite_tensor_delta(dict(x=torch.tensor([maximum])),dict(x=torch.tensor([-maximum])))
