import copy

import numpy as np
import pytest
import torch

from tools.evolution import run_cell as shared
from tools.evolution_fifty.unit_risk import run_candidate, parameter_snapshot, unit_term
from trace_ovss.method.adapter import ModelAdapterMixin
from trace_ovss.method.image_balanced import image_balanced_region_entropy, prepare_image_balanced_objective
from trace_ovss.method.trace.objective import TraceObjectiveMixin


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(3)

    def forward(self, x):
        logits = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).unsqueeze(0)
        return logits, None, None


class AuditMethod(ModelAdapterMixin):
    prepare_trace_objective = staticmethod(prepare_image_balanced_objective)
    trace_region_normalized_entropy = staticmethod(image_balanced_region_entropy)
    trace_softmax_entropy = staticmethod(TraceObjectiveMixin.trace_softmax_entropy)

    def __init__(self, steps=3):
        self.model = Tiny().eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)
        self.steps = steps
        self.device = torch.device("cpu")

    def reset(self):
        self.load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def forward_adapt_logits(self, images):
        return self.model(images)[0]


def candidate():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    return dict(batch_idx=1, image_idx=1, area=4, C=.2, confidence=.9,
                region=dict(shape=[4, 4], encoding="packbits-little-hex", true_pixels=4,
                            data=np.packbits(mask.reshape(-1), bitorder="little").tobytes().hex()))


def test_real_adam_trajectory_resets_and_cannot_use_C_as_update_weight():
    torch.manual_seed(123)
    method, images = AuditMethod(), torch.randn(2, 3, 4, 4)
    initial = shared.restored_state(method)
    snapshots = []
    report = run_candidate(method, images, candidate(), lambda step, m: snapshots.append((step, parameter_snapshot(m))))
    assert report["actual_steps"] == 3 and report["update_weight"] == 1
    assert [step for step, _ in snapshots] == [0, 1, 3]
    assert report["reset"] == initial == shared.restored_state(method)
    assert not torch.equal(snapshots[0][1]["ln.weight"], snapshots[-1][1]["ln.weight"])
    other = copy.deepcopy(candidate())
    other.update(C=float("nan"), confidence=-100., reliability=0.)
    again = []
    second = run_candidate(method, images, other, lambda step, m: again.append((step, parameter_snapshot(m))))
    assert report["losses"] == second["losses"]
    for (a_step, a), (b_step, b) in zip(snapshots, again, strict=True):
        assert a_step == b_step
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


def test_unit_objective_matches_direct_region_mean_not_episode_mean():
    entropy = torch.arange(32, dtype=torch.float64).reshape(2, 4, 4).requires_grad_()
    term = unit_term(candidate(), "cpu")
    got = image_balanced_region_entropy(entropy, [term])
    wanted = entropy[1].flatten()[term["indices"]].mean()
    torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(got, entropy)[0], torch.autograd.grad(wanted, entropy)[0], rtol=0, atol=0)


def test_checkpoint_failure_preserves_exception_and_restores_theta0_and_Adam():
    method = AuditMethod()
    initial = shared.restored_state(method)
    def callback(step, current):
        if step == 1:
            raise RuntimeError("simulated checkpoint failure")
    with pytest.raises(RuntimeError, match="simulated"):
        run_candidate(method, torch.randn(2, 3, 4, 4), candidate(), callback)
    assert shared.restored_state(method) == initial


def test_single_step_has_one_postupdate_checkpoint_not_duplicate():
    checkpoints = []
    report = run_candidate(AuditMethod(1), torch.randn(2, 3, 4, 4), candidate(), lambda step, _: checkpoints.append(step))
    assert checkpoints == [0, 1] and len(report["checkpoints"]) == 1
