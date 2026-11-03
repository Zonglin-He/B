import copy

import pytest
import torch

from tools.evolution_fifty import legacy_gate as gate
from test_legacy_runtime import method,config
from test_image_full_worker import batch_meta


def test_four_score_objective_conditions_have_real_every_step_parity_and_native_anchor():
    torch.manual_seed(11)
    current = method()
    before = gate.shared.restored_state(current)
    result = gate.compare_gate(current,torch.randn(2,3,32,32),None,torch.arange(2),batch_meta(2),config(2),None)
    assert result["passed"] and set(result["modes"]) == set(gate.runtime.MODES)
    for mode,record in result["modes"].items():
        assert record["actual_optimizer_steps"] == 2 and record["changed_labels"] == 0
        assert max(record["max_abs"].values()) <= 1e-5
    assert result["modes"]["C.image"]["unchanged_production_C_image_parity"]["passed"]
    assert gate.shared.restored_state(current) == before
    assert not current.optimizer._optimizer_step_post_hooks and not current.model._forward_hooks
    assert not torch.cuda.is_initialized()


def test_empty_legacy_gate_cannot_claim_vacuous_success():
    current = method()
    with pytest.raises(ValueError,match="Vacuous"):
        gate.compare_gate(current,torch.randn(2,3,32,32),torch.zeros(2,1,32,32,dtype=torch.bool),
            torch.arange(2),batch_meta(2),config(2),None)


@pytest.mark.parametrize("fault",["taxonomy","mapping","short"])
def test_gate_rejects_taxonomy_or_readout_mismatch_before_reference_construction(monkeypatch,fault):
    current = method()
    cfg,mapping = config(2),None
    if fault == "taxonomy":
        cfg["dataset"] = "PascalContext60Dataset"
    else:
        mapping = [1,0] if fault == "mapping" else [0]
    monkeypatch.setattr(gate,"make_reference",lambda *_a,**_k:pytest.fail("Mismatch reached gate reference/model construction"))
    with pytest.raises(ValueError,match="mapping"):
        gate.compare_gate(current,torch.randn(2,3,32,32),None,torch.arange(2),batch_meta(2),cfg,mapping)


@pytest.mark.parametrize("field",["gradients","parameters","adam","loss","labels"])
def test_legacy_gate_rejects_tampered_step_or_prediction_receipts(field):
    current = method()
    images,cfg,meta = torch.randn(2,3,32,32),config(2),batch_meta(2)
    actual = gate.capture(current,"legacy.image",images,None,torch.arange(2),meta,cfg,None,
        gate.shared.rng_state(current.device))
    other = copy.deepcopy(actual)
    if field == "loss":
        other["losses"][-1] = float("nan")
    elif field == "labels":
        other["predictions"][0][0,0] ^= 1
    else:
        list(other["steps"][-1][field].values())[-1].fill_(float("nan"))
    with pytest.raises(ValueError):
        gate.compare(actual,other,cfg)
