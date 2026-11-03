"""Target-first probing is measured, bounded and distinct from smaller top-K."""

from pathlib import Path
import sys
import weakref

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from trace_algorithm_variants import ALGORITHMS, TargetFirstSelector, make_algorithm_class
from trace_ovss.method.trace import TRACE, trace_reliability
from trace_ovss.method.trace.constants import EPS
from test_simplified_trace import tiny_trace


def test_standard_is_production_not_an_approximation():
    assert make_algorithm_class(TRACE, "standard") is TRACE
    with pytest.raises(ValueError):
        make_algorithm_class(TRACE, "invented")


@pytest.mark.parametrize("tolerance", [.01, .05, .1])
def test_bound_holds_for_random_complete_episodes(tolerance):
    rng = np.random.default_rng(52)
    for _ in range(100):
        n = int(rng.integers(1, 120))
        t, c, d = rng.random((3, n)) ** 3
        truth = np.minimum(t, c) * (1 - np.maximum(d - t, 0))
        truth[truth <= EPS] = 0
        selector = TargetFirstSelector(t, tolerance)
        selected = np.zeros(n)
        while ids := selector.next_indices(8):
            for i in ids:
                selector.observe(i, truth[i])
                selected[i] = truth[i]
        if truth.sum() > 0:
            assert selected.sum() > 0
            tv = .5 * np.abs(truth / truth.sum() - selected / selected.sum()).sum()
            assert tv <= selector.omitted_mass_bound + 1e-12 <= tolerance + 1e-12


def test_target_rank_not_original_confidence_or_unobserved_reliability():
    selector = TargetFirstSelector([.1, .9, .9, .001], .05)
    assert selector.next_indices(2) == [1, 2]
    selector.observe(1, .8)
    selector.observe(2, .7)
    assert selector.next_indices(2) == [0, 3]
    selector.observe(0, .1)
    assert selector.next_indices(2) == []
    assert selector.omitted_mass_bound == pytest.approx(.001 / 1.601)


def test_zero_or_eps_targets_are_provably_irrelevant_to_production_loss():
    selector = TargetFirstSelector([0, EPS, EPS / 2], .01)
    assert selector.next_indices(8) == []
    assert selector.omitted_mass_bound == 0


def test_all_observed_weights_zero_does_not_allow_premature_stop():
    selector = TargetFirstSelector([.9, .5, .1], .05)
    selector.observe(0, 0)
    selector.observe(1, 0)
    assert selector.omitted_mass_bound == 1
    assert selector.next_indices(8) == [2]
    selector.observe(2, 0)
    assert selector.next_indices(8) == []
    assert selector.omitted_mass_bound == 0


def test_staged_full_control_never_uses_pruning():
    selector = TargetFirstSelector([.8, 0, EPS / 2], 0, complete=True)
    selector.observe(0, .8)
    assert selector.next_indices(8) == [2, 1]


@pytest.mark.parametrize("targets,tolerance", [([float("nan")], .01), ([1.1], .1),
    ([-.1], .01), ([.5], -1), ([.5], 1), ([.5], float("nan"))])
def test_invalid_bounds_fail_explicitly(targets, tolerance):
    with pytest.raises(ValueError):
        TargetFirstSelector(targets, tolerance)


def test_bad_or_duplicate_probe_cannot_change_certificate():
    selector = TargetFirstSelector([.8], .01)
    with pytest.raises(ValueError):
        selector.observe(0, .81)
    selector.observe(0, .7)
    with pytest.raises(ValueError):
        selector.observe(0, .7)
    with pytest.raises(ValueError):
        selector.next_indices(0)


@pytest.mark.parametrize("algorithm", tuple(ALGORITHMS))
def test_algorithm_actual_forward_counts_missing_signals_and_reset(algorithm):
    method = tiny_trace()
    method.__class__ = make_algorithm_class(TRACE, algorithm)
    images = torch.randn(2, 3, 32, 32)
    before = {name: p.detach().clone() for name, p in method.model.named_parameters()}
    inputs = []
    hook = method.model.visual.conv1.register_forward_pre_hook(
        lambda model, args: inputs.append(len(args[0])))
    method.estimate_trace_reliability(images)
    hook.remove()
    summary, stats = method.last_reliability_summary, method.last_candidate_stats
    n = len(stats)
    assert summary["diagnostic_views"] == sum(inputs) - len(images)
    if algorithm in {"standard", "staged_full"}:
        assert summary["diagnostic_views"] == 3 * n
    elif algorithm == "quality_only":
        assert summary["diagnostic_views"] == 2 * n
        assert all(row["residual_support"] is None and row["dominance"] is None for row in stats)
        assert all(row["reliability"] == min(row["target_support"], row["counterfactual_support"])
                   for row in stats)
    else:
        assert summary["diagnostic_views"] == n + 2 * summary["fully_traced_candidates"]
        assert summary["omitted_mass_bound"] <= (.01 if algorithm == "selective_01" else .05)
        for row in stats:
            if row["omitted"]:
                assert row["residual_support"] is row["counterfactual_support"] is None
                assert row["quality"] is row["dominance"] is None
                assert row["reliability"] == 0
    assert all(torch.equal(before[name], p) for name, p in method.model.named_parameters())
    losses = method.adapt(images)
    assert len(losses) == method.steps and np.isfinite(losses).all()
    changed = {name for name, p in method.model.named_parameters() if not torch.equal(before[name], p)}
    assert changed <= {name for name, p in method.model.named_parameters() if p.requires_grad}
    method.reset()
    assert all(torch.equal(before[name], p) for name, p in method.model.named_parameters())
    assert method.adapt(images, valid_mask=torch.zeros(2, 1, 32, 32)) == []
    assert method.last_reliability_summary["diagnostic_views"] == 0


def test_staged_control_agrees_with_production_supports_in_cpu_model():
    method = tiny_trace()
    images = torch.randn(2, 3, 32, 32)
    method.estimate_trace_reliability(images)
    reference = method.last_candidate_stats
    method.__class__ = make_algorithm_class(TRACE, "staged_full")
    method.estimate_trace_reliability(images)
    for expected, actual in zip(reference, method.last_candidate_stats):
        for key in ("target_support", "residual_support", "counterfactual_support", "reliability"):
            assert actual[key] == pytest.approx(expected[key], abs=2e-6)


def test_selective_execution_really_omits_views_and_keeps_missing_values(monkeypatch):
    method = tiny_trace()
    method.__class__ = make_algorithm_class(TRACE, "selective_05")
    method.trace_view_batch_size = 16
    images = torch.zeros(1, 3, 32, 32)
    candidates = []
    for i in range(32):
        mask = np.zeros((4, 8), dtype=bool)
        mask.flat[i] = True
        candidates.append(dict(batch_idx=0, image_idx=0, class_idx=0,
            region=mask, confidence=.99, area=1, component_idx=i))
    seen = []

    def fake_batches(keys, builder):
        for start in range(0, len(keys), 16):
            chunk = keys[start:start + 16]
            assert builder(chunk).shape == (len(chunk), 3, 32, 32)
            seen.extend(chunk)
            values = []
            for i, kind in chunk:
                probability = (.9 if i == 0 else .001) if kind != "residual" else .0001
                values.append(torch.log(torch.tensor([probability, 1 - probability]))
                              .view(2, 1, 1).expand(2, 4, 8))
            yield chunk, torch.stack(values)
        method.last_trace_view_batch_size = 16

    monkeypatch.setattr(method, "_forward_trace_view_batches", fake_batches)
    rows = method.evaluate_trace_views(images, torch.full((1, 2, 4, 8), .5), candidates)
    assert len(seen) == 48 < 3 * len(candidates)
    assert seen[:32] == [(i, "target") for i in range(32)]
    assert all(row["target_support"] is not None for row in rows)
    assert sum(row["omitted"] for row in rows) == 24
    for i, row in enumerate(rows):
        if row["omitted"]:
            assert (i, "residual") not in seen and (i, "counterfactual") not in seen
            assert row["counterfactual_support"] is row["residual_support"] is None
            assert row["quality"] is row["dominance"] is None
            assert row["reliability"] == 0
    assert method._algorithm_summary["omitted_mass_bound"] < .05


@pytest.mark.parametrize("algorithm", ["staged_full", "quality_only", "selective_01"])
def test_previous_logits_are_released_before_next_forward(monkeypatch, algorithm):
    method = tiny_trace()
    method.__class__ = make_algorithm_class(TRACE, algorithm)
    method.trace_view_batch_size = 2
    refs = []
    original_batches = method._forward_trace_view_batches
    original_forward = method.forward_eval_logits

    def batches(keys, builder):
        for batch_keys, logits in original_batches(keys, builder):
            refs.append(weakref.ref(logits))
            yield batch_keys, logits
            del logits

    def forward(*args, **kwargs):
        assert not any(ref() is not None for ref in refs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(method, "_forward_trace_view_batches", batches)
    monkeypatch.setattr(method, "forward_eval_logits", forward)
    method.estimate_trace_reliability(torch.randn(2, 3, 32, 32))
    assert len(refs) > 1
    assert not any(ref() is not None for ref in refs)


def test_oom_cap_is_monotonic_across_target_and_pair_queries(monkeypatch):
    method = tiny_trace()
    method.__class__ = make_algorithm_class(TRACE, "staged_full")
    method.trace_view_batch_size = 16
    attempted = []

    def capacity_four(images, **kwargs):
        attempted.append(len(images))
        if len(images) > 4:
            raise torch.cuda.OutOfMemoryError("synthetic capacity limit")
        return torch.zeros(len(images), 2, 2, 2)

    monkeypatch.setattr(method, "forward_eval_logits", capacity_four)
    mask = np.ones((2, 2), dtype=bool)
    candidates = [dict(batch_idx=0, image_idx=0, class_idx=0, region=mask,
                       confidence=.5, area=4, component_idx=i) for i in range(12)]
    stats = method.evaluate_trace_views(torch.zeros(1, 3, 32, 32),
                                       torch.full((1, 2, 2, 2), .5), candidates)
    assert len(stats) == 12
    assert attempted[:3] == [12, 8, 4]
    assert all(count <= 4 for count in attempted[2:])
    assert method.last_trace_view_batch_size == 4
    assert method.trace_view_batch_size == 16
    assert method._algorithm_summary["diagnostic_views"] == 36
