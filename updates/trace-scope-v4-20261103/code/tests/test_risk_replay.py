import hashlib
import io

import numpy as np
import pytest
import torch

from tools.evolution import run_cell as shared
from tools.evolution_fifty.risk_replay import (
    decode_ln_snapshot, fixed_ln_state, offline_true_support, replay_one, source_support,
)
from tools.evolution_fifty.unit_risk import source_prediction


class ReplayMethod:
    def __init__(self):
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.GroupNorm(1, 4)).eval()
        for p in self.model[0].parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model[1].parameters(), lr=.01)
        self.model(torch.ones(2, 3, 3, 3)).sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def evaluate(self, images):
        return self.model(images)

    def reset(self):
        raise AssertionError("Replay must not reset model/Adam")


def fixture():
    torch.manual_seed(19)
    method = ReplayMethod()
    images = torch.randn(2, 3, 3, 3)
    meta = dict(patch_grid_shape=[(1, 1), (1, 1)], img_shape=[(3, 3), (3, 3)])
    cfg = dict(patch_size=(3, 3), patch_stride=2)
    state = {n: p.detach().clone()+.1 for n, p in method.model.named_parameters() if p.requires_grad}
    return method, images, meta, cfg, state


def snapshot_bytes(state):
    stream = io.BytesIO()
    np.savez_compressed(stream, **{n: v.numpy() for n, v in state.items()})
    content = stream.getvalue()
    return content, hashlib.sha256(content).hexdigest()


def test_snapshot_hash_exact_membership_shape_dtype_and_finiteness():
    method, _, _, _, state = fixture()
    expected = {n: p for n, p in method.model.named_parameters() if p.requires_grad}
    raw, digest = snapshot_bytes(state)
    got = decode_ln_snapshot(raw, digest, expected)
    for name in state:
        torch.testing.assert_close(got[name], state[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="digest"):
        decode_ln_snapshot(raw, "0"*64, expected)
    for altered in ({}, {**state, "extra": torch.zeros(1)},
                    {n: v.double() for n, v in state.items()},
                    {n: v[:1] for n, v in state.items()},
                    {n: torch.full_like(v, float("nan")) for n, v in state.items()}):
        data, checksum = snapshot_bytes(altered)
        with pytest.raises(ValueError):
            decode_ln_snapshot(data, checksum, expected)


@pytest.mark.parametrize("mapping", [None, [0, 1, 1, 2]])
def test_fixed_readout_matches_production_hard_labels_and_restores_Adam(mapping):
    method, images, meta, cfg, state = fixture()
    before = {n: p.detach().clone() for n, p in method.model.named_parameters()}
    adam = shared.state_digest(method.optimizer.state_dict())
    with fixed_ln_state(method, state):
        expected = source_prediction(method, images, meta, cfg, mapping, 1)
    probabilities = replay_one(method, images, meta, cfg, mapping, 1, state, expected)
    np.testing.assert_array_equal(probabilities.argmax(0), expected)
    assert shared.state_digest(method.optimizer.state_dict()) == adam
    assert all(torch.equal(p, before[n]) for n, p in method.model.named_parameters())
    assert all(p.grad is None for p in method.model.parameters())
    if mapping is not None:
        assert (probabilities.sum(0) < 1).all()


def test_extension_support_is_max_not_sum_or_renormalized():
    method, images, meta, cfg, state = fixture()
    with fixed_ln_state(method, state):
        raw = source_support(method, images, meta, cfg, None, 0)
        mapped = source_support(method, images, meta, cfg, [0, 1, 1, 2], 0)
    np.testing.assert_array_equal(mapped[1], np.maximum(raw[1], raw[2]))
    np.testing.assert_array_equal(mapped[0], raw[0])


def test_label_gate_failure_restores_original_state_and_does_not_update():
    method, images, meta, cfg, state = fixture()
    original = {n: p.detach().clone() for n, p in method.model.named_parameters()}
    with pytest.raises(ValueError, match="changed sealed"):
        replay_one(method, images, meta, cfg, None, 0, state, np.full((3, 3), 99))
    assert all(torch.equal(p, original[n]) for n, p in method.model.named_parameters())


def test_readout_error_restores_original_LN_without_calling_reset():
    method, images, meta, cfg, state = fixture()
    original = {n: p.detach().clone() for n, p in method.model.named_parameters()}
    def failed(images):
        raise RuntimeError("fixed-state readout failed")
    method.evaluate = failed
    with pytest.raises(RuntimeError, match="readout failed"):
        replay_one(method, images, meta, cfg, None, 0, state, np.zeros((3, 3), np.uint16))
    assert all(torch.equal(p, original[n]) for n, p in method.model.named_parameters())


def test_mutating_evaluator_is_rejected_and_trainables_restored():
    method, images, meta, cfg, state = fixture()
    def mutate(images):
        method.model[1].weight.add_(1)
        return method.model(images)
    method.evaluate = mutate
    with pytest.raises(ValueError, match="mutated model"):
        with fixed_ln_state(method, state):
            source_support(method, images, meta, cfg, None, 0)


@pytest.mark.parametrize("mapping", [[0, 1], [0, 1, 1, 3], [0, 1, -1, 2]])
def test_invalid_query_mapping_rejected(mapping):
    method, images, meta, cfg, state = fixture()
    with pytest.raises(ValueError, match="mapping"):
        with fixed_ln_state(method, state):
            source_support(method, images, meta, cfg, mapping, 0)


def test_offline_support_keeps_ignore255_undefined_and_has_no_softmax_renormalization():
    values = np.array([[[.3, .4]], [[.1, .2]], [[.2, .1]]], np.float32)
    got = offline_true_support(values, np.array([[2, 255]], np.uint8))
    assert got[0, 0] == values[2, 0, 0] and np.isnan(got[0, 1])
    with pytest.raises(ValueError, match="outside mapping"):
        offline_true_support(values, np.array([[3, 0]], np.uint8))
    with pytest.raises(ValueError, match="dtype"):
        offline_true_support(values, np.zeros((1, 2), np.float32))


def test_training_mode_and_stale_gradient_rejected_before_snapshot_install():
    method, _, _, _, state = fixture()
    method.model.train()
    with pytest.raises(ValueError, match="eval-only"):
        with fixed_ln_state(method, state):
            pass
    method.model.eval()
    method.model[1].weight.grad = torch.ones_like(method.model[1].weight)
    with pytest.raises(ValueError, match="existing gradients"):
        with fixed_ln_state(method, state):
            pass


@pytest.mark.parametrize("mutation", ["frozen_data", "training", "requires_grad", "remove_buffer", "Adam", "gradient"])
def test_integrity_failure_retires_method_and_blocks_later_readout(mutation):
    method, images, meta, cfg, state = fixture()
    method.model.register_buffer("guard", torch.ones(1))
    def mutate(images):
        if mutation == "frozen_data":
            method.model[0].weight.data.add_(1)
        elif mutation == "training":
            method.model.train()
        elif mutation == "requires_grad":
            method.model[0].weight.requires_grad_(True)
        elif mutation == "remove_buffer":
            del method.model.guard
        elif mutation == "Adam":
            next(iter(method.optimizer.state.values()))["exp_avg"].add_(1)
        else:
            method.model[1].weight.grad = torch.ones_like(method.model[1].weight)
        return torch.zeros(2, 4, 3, 3)
    method.evaluate = mutate
    adam = shared.state_digest(method.optimizer.state_dict())
    with pytest.raises(ValueError):
        replay_one(method, images, meta, cfg, None, 0, state, np.zeros((3, 3), np.uint16))
    assert method._trace_replay_invalid
    assert shared.state_digest(method.optimizer.state_dict()) == adam
    assert all(p.grad is None for p in method.model.parameters())
    with pytest.raises(ValueError, match="retired"):
        source_support(method, images, meta, cfg, None, 0)
    with pytest.raises(ValueError, match="retired"):
        with fixed_ln_state(method, state):
            pass


def test_exception_preserved_Adam_restored_and_potentially_corrupted_model_retired():
    method, images, meta, cfg, state = fixture()
    adam = shared.state_digest(method.optimizer.state_dict())
    def broken(images):
        method.model[0].weight.data.add_(1)
        next(iter(method.optimizer.state.values()))["exp_avg"].add_(1)
        method.model[1].weight.grad = torch.ones_like(method.model[1].weight)
        raise RuntimeError("original evaluator exception")
    method.evaluate = broken
    with pytest.raises(RuntimeError, match="original evaluator exception"):
        replay_one(method, images, meta, cfg, None, 0, state, np.zeros((3, 3), np.uint16))
    assert method._trace_replay_invalid and shared.state_digest(method.optimizer.state_dict()) == adam
    assert all(p.grad is None for p in method.model.parameters())


def test_half_precision_replay_preserves_production_softmax_rounding():
    method, images, meta, cfg, state = fixture()
    logits = torch.zeros(2, 2, 3, 3, dtype=torch.float16)
    logits[:, 1] = 1e-4
    method.evaluate = lambda _: logits
    expected = source_prediction(method, images, meta, cfg, None, 0)
    assert not expected.any()
    probabilities = replay_one(method, images, meta, cfg, None, 0, state, expected)
    assert probabilities.dtype == np.float16 and not probabilities.argmax(0).any()


@pytest.mark.parametrize("fault", ["metadata_length", "uncovered", "stride_gap", "patch_count", "empty_crop"])
def test_malformed_geometry_rejected_even_when_sealed_labels_would_agree(fault):
    method, images, meta, cfg, state = fixture()
    if fault == "metadata_length":
        meta["img_shape"].append((3, 3))
    elif fault == "uncovered":
        meta["img_shape"][0] = (3, 4)
    elif fault == "stride_gap":
        cfg["patch_stride"] = 4
    elif fault == "patch_count":
        meta["patch_grid_shape"][0] = (1, 2)
        meta["img_shape"][0] = (3, 5)
    else:
        meta["patch_grid_shape"][0] = (1, 3)
    with pytest.raises(ValueError):
        replay_one(method, images, meta, cfg, None, 0, state, np.zeros((3, 3), np.uint16))


def test_snapshot_decompression_bound_checked_before_np_load(monkeypatch):
    method, _, _, _, state = fixture()
    raw, digest = snapshot_bytes({n: torch.zeros(262144) for n in state})
    expected = {n: p for n, p in method.model.named_parameters() if p.requires_grad}
    def forbidden_load(*args, **kwargs):
        raise AssertionError("np.load reached before size preflight")
    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ValueError, match="uncompressed size"):
        decode_ln_snapshot(raw, digest, expected)


def test_integer_support_rejected_before_undefined_ignore_allocation():
    with pytest.raises(ValueError, match="dtype"):
        offline_true_support(np.zeros((2, 1, 2), np.int64), np.array([[0, 255]], np.uint8))


def test_unguarded_readout_rejected_without_calling_evaluator():
    method, images, meta, cfg, _ = fixture()
    def forbidden(_):
        raise AssertionError("Unprotected evaluator was called")
    method.evaluate = forbidden
    with pytest.raises(ValueError, match="active fixed-state guard"):
        source_support(method, images, meta, cfg, None, 0)


@pytest.mark.parametrize("alias", ["buffer", "parameter", "module", "optimizer_binding"])
def test_all_registered_alias_paths_and_live_optimizer_binding_are_checked(alias):
    method, images, meta, cfg, state = fixture()
    method.model.register_buffer("guard", torch.ones(1))
    method.model.register_buffer("guard_alias", method.model.guard)
    method.model.register_parameter("weight_alias", method.model[0].weight)
    method.model.add_module("shared_module_alias", method.model[0])
    def mutate(_):
        if alias == "buffer":
            del method.model.guard_alias
        elif alias == "parameter":
            del method.model.weight_alias
        elif alias == "module":
            del method.model.shared_module_alias
        else:
            old = method.optimizer.param_groups[0]["params"][0]
            replacement = torch.nn.Parameter(old.detach().clone())
            method.optimizer.param_groups[0]["params"][0] = replacement
            method.optimizer.state[replacement] = method.optimizer.state.pop(old)
        return torch.zeros(2, 4, 3, 3)
    method.evaluate = mutate
    with pytest.raises(ValueError, match="structure"):
        replay_one(method, images, meta, cfg, None, 0, state, np.zeros((3, 3), np.uint16))
    assert method._trace_replay_invalid and not method._trace_replay_guard_active


def test_huge_mapping_id_rejected_before_range_allocation(monkeypatch):
    import tools.evolution_fifty.risk_replay as replay_module
    method, images, meta, cfg, state = fixture()
    def forbidden_range(*_):
        raise AssertionError("Huge ID reached a range allocation")
    monkeypatch.setattr(replay_module, "range", forbidden_range, raising=False)
    with pytest.raises(ValueError, match="mapping"):
        replay_one(method, images, meta, cfg, [0, 1, 1, 10**12], 0, state, np.zeros((3, 3), np.uint16))
