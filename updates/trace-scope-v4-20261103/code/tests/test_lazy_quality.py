"""CPU-only checks for detached, bounded two-view counterfactual probing."""

import math
import weakref

import numpy as np
import pytest
import torch

from tools.trace_algorithm_variants import make_algorithm_class
from tools.trace_lazy_quality import (
    LAZY_ALGORITHMS, LAZY_DEFINITIONS, LazyQualitySelector, make_lazy_quality_class,
)
from trace_ovss.method.trace import TRACE
from trace_ovss.method.trace.constants import EPS
from test_simplified_trace import tiny_trace


def run_selector(targets, counterfactuals, groups, algorithm, batch=16):
    selector = LazyQualitySelector(targets, groups, algorithm)
    selected = np.zeros(len(targets))
    seen = []
    while indices := selector.next_indices(batch):
        for i in indices:
            q = min(targets[i], counterfactuals[i])
            selector.observe(i, q)
            selected[i] = q if q > EPS else 0.0
            seen.append(i)
    return selector, selected, seen


def test_runner_exports_and_script_style_import(monkeypatch):
    import importlib
    from pathlib import Path

    expected = {"quality_staged", "quality_lazy_01", "quality_lazy_class_01"}
    assert set(LAZY_ALGORITHMS) == set(LAZY_DEFINITIONS) == expected
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("trace_lazy_quality")
    assert module.LAZY_ALGORITHMS == LAZY_ALGORITHMS
    assert module.LAZY_DEFINITIONS == LAZY_DEFINITIONS
    assert callable(module.make_lazy_quality_class)


@pytest.mark.parametrize("algorithm", tuple(LAZY_ALGORITHMS))
def test_randomized_global_and_group_certificates(algorithm):
    rng = np.random.default_rng(20261025)
    for _ in range(100):
        count = int(rng.integers(1, 90))
        targets, cfs = rng.random((2, count)) ** 5
        groups = [(int(rng.integers(4)), int(rng.integers(5))) for _ in range(count)]
        selector, selected, _ = run_selector(targets, cfs, groups, algorithm)
        full = np.minimum(targets, cfs)
        full[full <= EPS] = 0.0
        summary = selector.summary()
        assert full.sum() > 0 and selected.sum() > 0
        tv = .5 * np.abs(full / full.sum() - selected / selected.sum()).sum()
        assert tv <= summary["omitted_mass_bound"] + 1e-12 <= .01 + 1e-12
        entropy = rng.random(count) * math.log(60)
        error = abs(np.dot(full / full.sum() - selected / selected.sum(), entropy))
        assert error <= tv * math.log(60) + 1e-12
        if algorithm == "quality_staged":
            np.testing.assert_array_equal(full, selected)
            assert summary["measured_cf_candidates"] == count
        if algorithm == "quality_lazy_class_01":
            assert summary["max_group_bound"] <= .01
            for group in set(groups):
                ids = [i for i, value in enumerate(groups) if value == group]
                a, b = full[ids], selected[ids]
                if a.sum() > 0:
                    assert b.sum() > 0
                    local_tv = .5 * np.abs(a / a.sum() - b / b.sum()).sum()
                    assert local_tv <= selector.group_bound(ids[0]) + 1e-12


def test_group_bound_protects_low_mass_groups_that_global_bound_can_drop():
    targets, cfs = [.9, .001, .000001], [.8, .001, .000001]
    groups = [(0, 0), (1, 0), (1, 1)]
    global_selector, _, global_seen = run_selector(targets, cfs, groups, "quality_lazy_01", 1)
    grouped, _, grouped_seen = run_selector(targets, cfs, groups, "quality_lazy_class_01", 1)
    assert global_seen == [0]
    assert global_selector.summary()["max_group_bound"] == 1
    assert grouped_seen == [0, 1, 2]
    assert grouped.summary()["observed_active_groups"] == 3


@pytest.mark.parametrize("algorithm", ["quality_lazy_01", "quality_lazy_class_01"])
def test_one_percent_boundary_and_adversarial_bound_saturation(algorithm):
    selector, selected, seen = run_selector([.99, .01], [.99, .01],
                                          [(0, 0), (0, 0)], algorithm, 1)
    assert seen == [0]
    assert selector.summary()["omitted_mass_bound"] == .01
    assert .5 * np.abs(np.array([.99, .01]) - selected / selected.sum()).sum() == pytest.approx(.01)
    selector, _, seen = run_selector([.989, .011], [.989, .011],
                                    [(0, 0), (0, 0)], algorithm, 1)
    assert seen == [0, 1]
    assert selector.summary()["omitted_mass_bound"] == 0


@pytest.mark.parametrize("algorithm", ["quality_lazy_01", "quality_lazy_class_01"])
def test_eps_boundary_empty_groups_and_zero_observed_mass(algorithm):
    above = np.nextafter(EPS, np.inf)
    t = [0, EPS / 2, EPS, above, .8, .2]
    c = [1, 1, 1, 1, 0, .1]
    selector, selected, seen = run_selector(t, c, [(0, 0)] * 6, algorithm, 1)
    assert not set(seen) & {0, 1, 2}
    assert seen[:2] == [4, 5]
    assert selected[5] == .1
    empty, weights, _ = run_selector(t, [0] * 6, [(0, 0)] * 6, algorithm, 1)
    assert len(empty.global_selector.observed) == 3
    assert weights.sum() == empty.summary()["omitted_mass_bound"] == 0
    assert selector.summary()["eps_pruned_candidates"] == 3


def test_group_with_zero_measurements_must_continue_independently():
    selector = LazyQualitySelector([.9, .5, .1], [(0, 0), (1, 0), (1, 0)],
                                   "quality_lazy_class_01")
    selector.observe(0, .9)
    selector.observe(1, 0)
    assert selector.group_bound(2) == 1
    assert selector.next_indices(16) == [2]
    selector.observe(2, .05)
    assert selector.next_indices(16) == []


def test_staged_keeps_every_cf_and_uses_stable_target_order():
    selector, _, seen = run_selector([0, EPS, .8, .8], [0, 0, .7, .6],
                                     [(0, 0)] * 4, "quality_staged", 2)
    assert seen == [2, 3, 1, 0]
    assert selector.summary()["omitted_candidates"] == 0
    assert selector.summary()["max_group_bound"] == 0


def test_groups_share_source_image_across_patches_but_not_query_aliases():
    # Equal query indices in one image share a budget, even across patches.
    groups = [(7, 3), (7, 3), (7, 4), (8, 3)]
    selector, _, seen = run_selector([.9, .0001, .0001, .0001],
                                     [.9, .0001, .0001, .0001], groups,
                                     "quality_lazy_class_01", 1)
    assert seen == [0, 2, 3]
    assert selector.summary()["group_count"] == 3


@pytest.mark.parametrize("algorithm", tuple(LAZY_ALGORITHMS))
def test_empty_selector(algorithm):
    selector = LazyQualitySelector([], [], algorithm)
    assert selector.next_indices(16) == []
    assert selector.summary()["max_group_bound"] == 0


def test_invalid_inputs_and_duplicate_observations_fail():
    with pytest.raises(ValueError):
        make_lazy_quality_class(TRACE, "invented")
    with pytest.raises(ValueError):
        LazyQualitySelector([], [], "invented")
    with pytest.raises(ValueError):
        LazyQualitySelector([.5], [], "quality_lazy_01")
    for target in (-.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            LazyQualitySelector([target], [(0, 0)], "quality_lazy_01")
    selector = LazyQualitySelector([.5], [(0, 0)], "quality_lazy_class_01")
    with pytest.raises(ValueError):
        selector.next_indices(0)
    with pytest.raises(ValueError):
        selector.observe(0, .6)
    selector.observe(0, .4)
    with pytest.raises(ValueError):
        selector.observe(0, .4)


@pytest.mark.parametrize("algorithm", tuple(LAZY_ALGORITHMS))
def test_actual_forward_counts_detachment_engine_and_reset(algorithm):
    method = tiny_trace()
    method.__class__ = make_lazy_quality_class(TRACE, algorithm)
    images = torch.randn(2, 3, 32, 32)
    before = {name: p.detach().clone() for name, p in method.model.named_parameters()}
    fixed = (method.lr, method.steps, method.trace_max_regions, method.trace_view_batch_size)
    inputs = []
    hook = method.model.visual.conv1.register_forward_pre_hook(
        lambda model, args: inputs.append(len(args[0])))
    method.estimate_trace_reliability(images, image_ids=[7, 7])
    hook.remove()
    stats, summary = method.last_candidate_stats, method.last_reliability_summary
    assert summary["diagnostic_views"] == len(stats) + summary["measured_cf_candidates"]
    assert summary["diagnostic_views"] == sum(inputs) - len(images)
    assert summary["diagnostic_calls"] == len(inputs) - 1
    assert summary["fully_traced_candidates"] == 0
    assert all(r["image_idx"] == 7 for r in stats)
    assert all(r["residual_support"] is r["dominance"] is None for r in stats)
    assert all(not r["fully_traced"] for r in stats)
    assert all(torch.equal(before[name], p) for name, p in method.model.named_parameters())
    assert all(not term["indices"].requires_grad for term in method.last_region_terms)
    assert len(method.adapt(images, image_ids=[7, 7])) == method.steps
    changed = {name for name, p in method.model.named_parameters() if not torch.equal(before[name], p)}
    assert changed <= {name for name, p in method.model.named_parameters() if p.requires_grad}
    method.reset()
    assert all(torch.equal(before[name], p) for name, p in method.model.named_parameters())
    assert fixed == (method.lr, method.steps, method.trace_max_regions, method.trace_view_batch_size)
    assert method.adapt(images, valid_mask=torch.zeros(2, 1, 32, 32)) == []
    assert method.last_reliability_summary["diagnostic_views"] == 0
    assert method.last_reliability_summary["measured_cf_candidates"] == 0
    assert method.last_reliability_summary["max_group_bound"] == 0


def test_staged_matches_complete_interleaved_quality_on_cpu():
    method = tiny_trace()
    images = torch.randn(2, 3, 32, 32)
    method.__class__ = make_algorithm_class(TRACE, "quality_only")
    method.estimate_trace_reliability(images)
    expected = method.last_candidate_stats
    method.__class__ = make_lazy_quality_class(TRACE, "quality_staged")
    method.estimate_trace_reliability(images)
    actual = method.last_candidate_stats
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a["region"], b["region"])
        for key in ("target_support", "counterfactual_support", "quality", "reliability"):
            assert a[key] == pytest.approx(b[key], abs=2e-6)
    assert method.last_reliability_summary["diagnostic_views"] == 2 * len(actual)
    assert method.last_reliability_summary["measured_cf_candidates"] == len(actual)


@pytest.mark.parametrize("algorithm", ["quality_lazy_01", "quality_lazy_class_01"])
def test_unpruned_lazy_episode_matches_staged_supports_losses_and_parameters(algorithm):
    method = tiny_trace()
    method.trace_max_regions = 1
    images = torch.randn(2, 3, 32, 32)
    method.__class__ = make_lazy_quality_class(TRACE, "quality_staged")
    expected_losses = method.adapt(images)
    expected_rows = method.last_candidate_stats
    expected_params = {name: p.detach().clone() for name, p in method.model.named_parameters()}
    expected_prediction = method.evaluate(images)
    method.__class__ = make_lazy_quality_class(TRACE, algorithm)
    actual_losses = method.adapt(images)
    assert method.last_reliability_summary["omitted_candidates"] == 0
    assert actual_losses == expected_losses
    for a, b in zip(method.last_candidate_stats, expected_rows):
        for key in ("target_support", "counterfactual_support", "quality", "reliability"):
            assert a[key] == b[key]
    assert all(torch.equal(expected_params[name], p) for name, p in method.model.named_parameters())
    assert torch.equal(method.evaluate(images), expected_prediction)


def synthetic_candidates(count):
    candidates = []
    for i in range(count):
        mask = np.zeros((4, 8), dtype=bool)
        mask.flat[i % 32] = True
        candidates.append(dict(batch_idx=0, image_idx=0, class_idx=0, region=mask,
                               confidence=.99, area=1, component_idx=i))
    return candidates


@pytest.mark.parametrize("algorithm", ["quality_lazy_01", "quality_lazy_class_01"])
def test_real_evaluator_skips_cf_preserves_missing_values_and_targets_first(monkeypatch, algorithm):
    method = tiny_trace()
    method.__class__ = make_lazy_quality_class(TRACE, algorithm)
    method.trace_view_batch_size = 16
    candidates = synthetic_candidates(32)
    seen = []

    def fake_batches(keys, builder):
        for start in range(0, len(keys), 16):
            chunk = keys[start:start + 16]
            assert builder(chunk).shape == (len(chunk), 3, 32, 32)
            seen.extend(chunk)
            values = []
            for i, kind in chunk:
                assert kind in {"target", "counterfactual"}
                probability = .9 if i == 0 else .0001
                values.append(torch.log(torch.tensor([probability, 1 - probability]))
                              .view(2, 1, 1).expand(2, 4, 8))
            yield chunk, torch.stack(values)
        method.last_trace_view_batch_size = 16

    monkeypatch.setattr(method, "_forward_trace_view_batches", fake_batches)
    stats = method.evaluate_trace_views(torch.zeros(1, 3, 32, 32),
                                        torch.full((1, 2, 4, 8), .5), candidates)
    assert seen[:32] == [(i, "target") for i in range(32)]
    assert len(seen) == 48
    assert method._lazy_quality_summary["measured_cf_candidates"] == 16
    assert method._lazy_quality_summary["omitted_mass_bound"] <= .01
    for i, row in enumerate(stats):
        assert row["residual_support"] is row["dominance"] is None
        if row["omitted"]:
            assert (i, "counterfactual") not in seen
            assert row["counterfactual_support"] is row["quality"] is None
            assert row["reliability"] == 0
        else:
            assert row["reliability"] == min(row["target_support"], row["counterfactual_support"])


@pytest.mark.parametrize("algorithm", tuple(LAZY_ALGORITHMS))
def test_logits_released_before_every_forward(monkeypatch, algorithm):
    method = tiny_trace()
    method.__class__ = make_lazy_quality_class(TRACE, algorithm)
    method.trace_view_batch_size = 2
    refs = []
    original_batches, original_forward = method._forward_trace_view_batches, method.forward_eval_logits

    def batches(keys, builder):
        for chunk, logits in original_batches(keys, builder):
            refs.append(weakref.ref(logits))
            yield chunk, logits
            del logits

    def forward(*args, **kwargs):
        assert not any(ref() is not None for ref in refs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(method, "_forward_trace_view_batches", batches)
    monkeypatch.setattr(method, "forward_eval_logits", forward)
    method.estimate_trace_reliability(torch.randn(2, 3, 32, 32))
    assert len(refs) > 1
    assert not any(ref() is not None for ref in refs)


@pytest.mark.parametrize("algorithm", tuple(LAZY_ALGORITHMS))
def test_oom_cap_carried_across_target_and_cf_rounds(monkeypatch, algorithm):
    method = tiny_trace()
    method.__class__ = make_lazy_quality_class(TRACE, algorithm)
    method.trace_view_batch_size = 16
    attempted = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def capacity_four(images, **kwargs):
        attempted.append(len(images))
        if len(images) > 4:
            raise torch.cuda.OutOfMemoryError("synthetic CPU-only capacity limit")
        return torch.zeros(len(images), 2, 4, 8)

    monkeypatch.setattr(method, "forward_eval_logits", capacity_four)
    rows = method.evaluate_trace_views(torch.zeros(1, 3, 32, 32),
                                       torch.full((1, 2, 4, 8), .5), synthetic_candidates(20))
    assert len(rows) == 20
    assert attempted[:3] == [16, 8, 4]
    assert all(count <= 4 for count in attempted[2:])
    assert method.last_trace_view_batch_size == 4
    assert method.trace_view_batch_size == 16
    assert method._lazy_quality_summary["diagnostic_views"] == 40
    assert method._lazy_quality_summary["diagnostic_calls"] == 10


def test_new_oom_in_cf_stage_caps_all_later_cf_rounds(monkeypatch):
    method = tiny_trace()
    method.__class__ = make_lazy_quality_class(TRACE, "quality_lazy_class_01")
    method.trace_view_batch_size = 16
    attempted = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def lower_cf_capacity(images, **kwargs):
        attempted.append(len(images))
        if len(attempted) > 2 and len(images) > 4:
            raise torch.cuda.OutOfMemoryError("synthetic CF capacity limit")
        return torch.zeros(len(images), 2, 4, 8)

    monkeypatch.setattr(method, "forward_eval_logits", lower_cf_capacity)
    method.evaluate_trace_views(torch.zeros(1, 3, 32, 32),
                                torch.full((1, 2, 4, 8), .5), synthetic_candidates(32))
    assert attempted[:5] == [16, 16, 16, 8, 4]
    assert all(count <= 4 for count in attempted[4:])
    assert method.last_trace_view_batch_size == 4
    assert method.trace_view_batch_size == 16
    assert method._lazy_quality_summary["diagnostic_views"] == 64
    assert method._lazy_quality_summary["diagnostic_calls"] == 10


def test_failure_restores_configured_batch_and_manual_original_fallback(monkeypatch):
    method = tiny_trace()
    method.__class__ = make_lazy_quality_class(TRACE, "quality_staged")
    configured = method.trace_view_batch_size

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic forward failure")

    with monkeypatch.context() as context:
        context.setattr(method, "forward_eval_logits", fail)
        with pytest.raises(RuntimeError, match="synthetic forward failure"):
            method.evaluate_trace_views(torch.zeros(1, 3, 32, 32),
                                        torch.full((1, 2, 4, 8), .5), synthetic_candidates(1))
    assert method.trace_view_batch_size == configured
    probe = dict(batch_idx=0, image_idx=0, class_idx=0,
                 region=np.array([[True, False], [False, True]]))
    rows = method.evaluate_trace_views(torch.randn(1, 3, 32, 32),
                                       torch.full((1, 2, 2, 2), .5), [probe])
    assert rows[0]["original_support"] == .5
