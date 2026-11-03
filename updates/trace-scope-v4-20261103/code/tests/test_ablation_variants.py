"""Ablations remove a stated component without changing unrelated operations."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from trace_ablation_variants import VARIANTS, make_ablation_class
from trace_ovss.method.trace import TRACE
from test_simplified_trace import tiny_trace


def test_full_is_exactly_the_production_class():
    assert make_ablation_class(TRACE, "full") is TRACE
    with pytest.raises(ValueError):
        make_ablation_class(TRACE, "not_a_variant")


def test_class_unions_keep_disconnected_pixels_without_using_gt():
    method = tiny_trace()
    method.__class__ = make_ablation_class(TRACE, "no_connected_regions")
    pred = torch.tensor([[[0, 1, 0], [1, 1, 1]], [[0, 1, 0], [1, 1, 1]]])
    probs = torch.stack((torch.where(pred == 0, .8, .2),
                         torch.where(pred == 1, .8, .2)), dim=1)
    valid = torch.ones_like(pred, dtype=torch.bool)
    valid[1, 0, 0] = False
    method.trace_max_regions = 3
    rows = method.collect_trace_candidates(probs, pred, valid, torch.tensor([0, 0]))
    assert len(rows) == 3
    assert any(row["class_idx"] == 0 and row["area"] == 2 for row in rows)
    assert all(not row["region"][0, 0] for row in rows if row["batch_idx"] == 1)
    repeated = method.collect_trace_candidates(probs, pred, valid, torch.tensor([0, 0]))
    assert [(row["batch_idx"], row["class_idx"]) for row in rows] == [
        (row["batch_idx"], row["class_idx"]) for row in repeated]


def test_source_blind_does_not_run_views_or_invent_missing_supports(monkeypatch):
    method = tiny_trace()
    method.__class__ = make_ablation_class(TRACE, "no_source_tracing")
    def forbidden(*args, **kwargs):
        pytest.fail("No confidence-source tracing must not forward intervention views")
    monkeypatch.setattr(method, "_forward_trace_view_batches", forbidden)
    method.estimate_trace_reliability(torch.randn(2, 3, 32, 32))
    assert method.last_reliability_summary["diagnostic_views"] == 0
    for stat in method.last_candidate_stats:
        assert stat["reliability"] == stat["original_support"] == stat["confidence"]
        assert stat["target_support"] is None
        assert stat["residual_support"] is None
        assert stat["counterfactual_support"] is None


def test_uniform_removes_weights_but_retains_the_tracing_component():
    method = tiny_trace()
    method.__class__ = make_ablation_class(TRACE, "no_reliability_weighting")
    terms = method.estimate_trace_reliability(torch.randn(2, 3, 32, 32))["region_terms"]
    assert terms and all(term["rho"] == 1 for term in terms)
    stats = method.last_candidate_stats
    assert all(row["target_support"] is not None for row in stats)
    assert method.last_reliability_summary["diagnostic_views"] == len(stats) * 3


def test_pixel_objective_matches_direct_weight_map_and_its_gradient():
    cls = make_ablation_class(TRACE, "no_region_normalization")
    entropy = torch.tensor([[[[1., 3., 4.]]], [[[2., 5., 6.]]]], requires_grad=True)
    terms = [dict(batch_idx=0, indices=torch.tensor([0]), rho=.7),
             dict(batch_idx=0, indices=torch.tensor([1, 2]), rho=.2)]
    prepared = cls.prepare_trace_objective(entropy, terms)
    actual = cls.trace_region_normalized_entropy(entropy, terms, prepared=prepared)
    weights = torch.tensor([.7, .2, .2])
    expected = (entropy * weights).sum() / (weights.sum() * 2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, entropy)[0],
                               torch.autograd.grad(expected, entropy)[0])
    assert actual != TRACE.trace_region_normalized_entropy(entropy, terms)


@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_removal_control_restores_and_adapts_only_enabled_parameters(variant):
    method = tiny_trace()
    method.__class__ = make_ablation_class(TRACE, variant)
    images = torch.randn(2, 3, 32, 32)
    before = {name: p.detach().clone() for name, p in method.model.named_parameters()}
    losses = method.adapt(images)
    assert len(losses) == method.steps and np.isfinite(losses).all()
    changed = {name for name, p in method.model.named_parameters() if not torch.equal(before[name], p)}
    assert changed and changed <= {name for name, p in method.model.named_parameters() if p.requires_grad}
    method.reset()
    assert all(torch.equal(before[name], p) for name, p in method.model.named_parameters())
