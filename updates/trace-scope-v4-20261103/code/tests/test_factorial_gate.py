import copy

import pytest
import torch

from tools.evolution_fifty import factorial_gate as gate
from test_factorial_runtime import method
from test_image_full_worker import config,batch_meta


def test_all_eight_factorial_conditions_have_real_step_tensor_and_native_parity_gates():
    torch.manual_seed(19)
    current = method()
    before = gate.shared.restored_state(current)
    result = gate.compare_gate(current,torch.randn(2,3,32,32),None,torch.arange(2),
        batch_meta(2),config(2),None,["a","b"])
    assert result["passed"] and not result["GT_opened"]
    for study,records in result["studies"].items():
        assert set(records) == set(gate.runtime.ARMS)
        for arm,report in records.items():
            assert report["changed_labels"] == 0 and report["actual_optimizer_steps"] == 2
            assert all(x <= 1e-5 for x in report["max_abs"].values())
            if study == "natural" and arm in ("Dk_Ok","Dc_Oc"):
                assert report["unchanged_production_parity"]["passed"]
    assert gate.shared.restored_state(current) == before
    assert not current.optimizer._optimizer_step_post_hooks and not current.model._forward_hooks
    assert not torch.cuda.is_initialized()


def test_empty_factorial_gate_does_not_invent_passing_updates():
    current = method()
    with pytest.raises(ValueError,match="Vacuous"):
        gate.compare_gate(current,torch.randn(2,3,32,32),torch.zeros(2,1,32,32,dtype=torch.bool),
            torch.arange(2),batch_meta(2),config(2),None,["a","b"])
    assert not current.optimizer._optimizer_step_post_hooks


@pytest.mark.parametrize("fault",["gradient","adam","loss","label"])
def test_factorial_gate_rejects_real_receipt_corruption(fault):
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    actual = gate.capture(current,"Dc_Ok",False,images,None,torch.arange(2),meta,cfg,None,
        ["a","b"],gate.shared.rng_state(current.device))
    other = copy.deepcopy(actual)
    if fault in ("gradient","adam"):
        key = "gradients" if fault == "gradient" else "adam"
        list(other["steps"][-1][key].values())[-1].fill_(float("nan"))
    elif fault == "loss":
        other["losses"][-1] = float("nan")
    else:
        other["predictions"][0][0,0] ^= 1
    with pytest.raises(ValueError):
        gate.compare(actual,other,cfg)
