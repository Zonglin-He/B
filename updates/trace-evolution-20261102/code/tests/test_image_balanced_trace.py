"""CPU-only objective and inherited-engine contracts for ImageBalancedTRACE."""

import copy
import math

import numpy as np
import pytest
import torch

from trace_ovss.method.image_balanced import (
    ImageBalancedTRACE,
    image_balanced_region_entropy,
    prepare_image_balanced_objective,
)
from trace_ovss.method.trace.constants import EPS
from trace_ovss.method.trace.objective import TraceObjectiveMixin
from trace_ovss.method.two_view import TwoViewTRACE
from test_simplified_trace import tiny_trace


def term(batch_idx, image_idx, rho, indices):
    return dict(batch_idx=batch_idx, image_idx=image_idx, rho=rho,
                indices=torch.tensor(indices, dtype=torch.long))


def tiny_balanced(view_mode="min", architecture="reduced"):
    base = tiny_trace(architecture=architecture)
    return ImageBalancedTRACE(
        classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions,
        trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, prompts=base.prompt_templates,
        model=base.model, tokenize_fn=base.tokenize, device="cpu",
        view_mode=view_mode,
    )


@pytest.mark.parametrize("prompts", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_source_is_bitwise_equal_to_old_objective_and_gradient(prompts, dtype):
    torch.manual_seed(8)
    shape = (3, 2, 2, 3) if prompts else (2, 2, 3)
    entropy = torch.rand(shape, dtype=dtype, requires_grad=True)
    terms = [term(0, 41, .3, [0, 2]), term(1, 41, .7, [1, 2, 3, 5]),
             term(1, 41, EPS, [4]), term(0, 41, .5, [])]
    expected = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms)
    actual = image_balanced_region_entropy(entropy, terms)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, entropy)[0],
                       torch.autograd.grad(expected, entropy)[0])
    cached = prepare_image_balanced_objective(entropy, terms)
    assert torch.equal(image_balanced_region_entropy(entropy, terms, prepared=cached),
                       expected)


def test_scaling_only_one_images_positive_weights_preserves_loss_and_both_budgets():
    entropy = torch.tensor([[[1., 3.]], [[5., 9.]]], dtype=torch.float64,
                           requires_grad=True)
    terms = [term(0, -7, .125, [0]), term(0, -7, .375, [1]),
             term(1, 902, .25, [0]), term(1, 902, .25, [1])]
    scaled = [dict(row, rho=row["rho"] * (2 if row["image_idx"] == -7 else 1))
              for row in terms]
    loss = image_balanced_region_entropy(entropy, terms)
    changed = image_balanced_region_entropy(entropy, scaled)
    torch.testing.assert_close(loss, torch.tensor(4.75, dtype=entropy.dtype), rtol=0, atol=0)
    torch.testing.assert_close(changed, loss, rtol=0, atol=0)
    grad = torch.autograd.grad(loss, entropy)[0]
    torch.testing.assert_close(torch.autograd.grad(changed, entropy)[0], grad, rtol=0, atol=0)
    torch.testing.assert_close(grad.sum((1, 2)), torch.tensor([.5, .5], dtype=grad.dtype))
    old = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms)
    old_scaled = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, scaled)
    assert not torch.isclose(old, old_scaled)


def test_candidate_area_and_prompt_count_are_means_not_token_mass():
    entropy = torch.tensor([[[[2., 0., 0., 0.]]], [[[4., 0., 0., 0.]]]],
                           requires_grad=True)
    terms = [term(0, 3, .5, [0]), term(0, 3, .5, [1, 2, 3])]
    loss = image_balanced_region_entropy(entropy, terms)
    assert loss.item() == 1.5
    grad = torch.autograd.grad(loss, entropy)[0]
    torch.testing.assert_close(grad[:, 0, 0, 0].sum(), torch.tensor(.5))
    torch.testing.assert_close(grad[:, 0, 0, 1:].sum(), torch.tensor(.5))


def test_reliability_weights_are_detached_including_cached_objective():
    entropy = torch.tensor([[[2., 8.]], [[1., 5.]]], requires_grad=True)
    rho = torch.tensor([.2, .4, .5], requires_grad=True)
    terms = [term(0, 1, rho[0], [0]), term(0, 1, rho[1], [1]),
             term(1, 8, rho[2], [0, 1])]
    cached = prepare_image_balanced_objective(entropy, terms)
    assert all(not normalizer.requires_grad and all(not q.requires_grad for _, _, q in rows)
               for rows, normalizer in cached)
    image_balanced_region_entropy(entropy, terms, prepared=cached).backward()
    assert rho.grad is None
    assert entropy.grad is not None and torch.isfinite(entropy.grad).all()


def test_empty_and_all_inactive_sources_are_not_in_the_image_denominator():
    entropy = torch.tensor([[[2., 6.]], [[100., 200.]], [[300., 400.]]],
                           requires_grad=True)
    terms = [term(0, 5, .4, [0, 1]), term(1, 77, 0., [0]),
             term(1, 77, EPS, [1]), term(1, 77, -1., [0]),
             term(2, 99, .7, [])]
    assert image_balanced_region_entropy(entropy, terms).item() == 4.
    assert len(prepare_image_balanced_objective(entropy, terms)) == 1
    for empty in ([], terms[1:]):
        loss = image_balanced_region_entropy(entropy, empty)
        assert loss.item() == 0. and loss.requires_grad
        assert torch.count_nonzero(torch.autograd.grad(loss, entropy)[0]) == 0


@pytest.mark.parametrize("q", [EPS / 2, EPS, EPS * (1 + 1e-6), .5])
def test_eps_membership_and_float_normalizer_guard_match_old_method(q):
    entropy = torch.tensor([[[3.]]], requires_grad=True)
    terms = [term(0, 101, q, [0])]
    expected = TraceObjectiveMixin.trace_region_normalized_entropy(entropy, terms)
    assert torch.equal(image_balanced_region_entropy(entropy, terms), expected)
    assert len(prepare_image_balanced_objective(entropy, terms)) == int(q > EPS)


def test_masks_noncontiguous_source_ids_and_crop_ids_have_separate_roles():
    entropy = torch.tensor([[[2., 4.]], [[10., 12.]], [[6., 8.]]], requires_grad=True)
    terms = [term(0, 1007, .25, [0, 1]), term(1, -8, .9, [0, 1]),
             dict(batch_idx=2, image_idx=1007, rho=.75,
                  mask=torch.ones(1, 2, dtype=torch.bool))]
    # Source 1007 spans crop 0 and 2: (.25*3 + .75*7 + 11) / 2.
    assert image_balanced_region_entropy(entropy, terms).item() == 8.5
    with pytest.raises(KeyError, match="image_idx"):
        image_balanced_region_entropy(entropy, [dict(batch_idx=0, rho=.5, indices=torch.tensor([0]))])


def test_one_active_candidate_cancels_only_its_local_weight():
    entropy = torch.tensor([[[2., 6.]], [[10., 20.]]], dtype=torch.float64,
                           requires_grad=True)
    terms = [term(0, 30, .25, [0, 1]), term(1, 80, .25, [0]), term(1, 80, .75, [1])]
    loss = image_balanced_region_entropy(entropy, terms)
    local = [dict(row, rho=.5) if row["image_idx"] == 30 else row for row in terms]
    assert image_balanced_region_entropy(entropy, local).item() == loss.item() == 10.75
    other = [dict(row, rho=.5) if i == 1 else row for i, row in enumerate(terms)]
    assert image_balanced_region_entropy(entropy, other).item() != loss.item()
    assert torch.autograd.grad(loss, entropy)[0][0].sum().item() == .5


@pytest.mark.parametrize("mode,views", [("T", 1), ("C", 1), ("min", 2)])
def test_actual_view_schedule_geometry_and_readouts_without_unused_view(mode, views, monkeypatch):
    method = tiny_balanced(mode)
    method.trace_view_batch_size = 1
    images = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 3, 32, 32) / 10000
    original = torch.full((2, 2, 2, 2), .5)
    candidates = [dict(batch_idx=i, image_idx=71, class_idx=i, confidence=.6,
                       region=np.array([[True, False], [False, True]])) for i in range(2)]
    built, contexts = [], []

    def context(kind, offset, image):
        assert not torch.is_grad_enabled()
        assert mode == "min" or kind == ("target" if mode == "T" else "counterfactual")
        contexts.append(kind)
        return image + offset

    monkeypatch.setattr(method, "make_trace_weak_image", lambda image: context("target", 1., image))
    monkeypatch.setattr(method, "make_trace_counterfactual_background",
                        lambda image: context("counterfactual", 2., image))

    def evaluate(chunk, *, interpolate, save_weights):
        assert not torch.is_grad_enabled() and not interpolate and not save_weights
        built.append(chunk.clone())
        return torch.tensor([[[[2., -3.], [-3., 2.]], [[0., 4.], [4., 0.]]]])

    monkeypatch.setattr(method, "forward_eval_logits", evaluate)
    stats = method.evaluate_trace_views(images, original, candidates)
    assert len(built) == views * len(candidates)
    assert method._two_view_counts == dict(diagnostic_views=views * 2, diagnostic_calls=views * 2)
    assert contexts.count("target") == (2 if mode in ("T", "min") else 0)
    assert contexts.count("counterfactual") == (2 if mode in ("C", "min") else 0)
    kinds = ("target", "counterfactual") if mode == "min" else (
        "target" if mode == "T" else "counterfactual",)
    for i, candidate in enumerate(candidates):
        mask = torch.from_numpy(candidate["region"]).repeat_interleave(16, 0).repeat_interleave(16, 1)
        for k, kind in enumerate(kinds):
            delta = 1. if kind == "target" else 2.
            expected = images[i:i + 1] + (~mask).to(images.dtype) * delta
            torch.testing.assert_close(built[i * views + k], expected)
        row = stats[i]
        q = torch.tensor([2., 0.]).softmax(0)[i].item()
        assert row["quality"] == row["reliability"] == pytest.approx(q)
        assert row["region"] is candidate["region"]
        assert row["original_support"] == .6
        assert row["residual_support"] is None and row["dominance"] is None
        assert row["target_support"] == (None if mode == "C" else pytest.approx(q))
        assert row["counterfactual_support"] == (None if mode == "T" else pytest.approx(q))


def test_estimator_retains_identity_after_inactive_candidates_are_filtered(monkeypatch):
    method = tiny_balanced()
    images = torch.randn(3, 3, 32, 32)
    ids = [71, -9, 71]

    def evaluate(images, probabilities, candidates):
        method._two_view_counts = dict(diagnostic_views=0, diagnostic_calls=0)
        # Eliminate an entire image and interleave inactive/active candidates
        # from the other source, without changing admission or geometry.
        return [dict(row, reliability=(.5 if row["image_idx"] == 71 and i % 2 else EPS))
                for i, row in enumerate(candidates)]

    monkeypatch.setattr(method, "evaluate_trace_views", evaluate)
    result = method.estimate_trace_reliability(images, image_ids=ids)
    active = [row for row in method.last_candidate_stats if row["reliability"] > EPS]
    assert active and len(active) < len(method.last_candidate_stats)
    assert result["region_terms"] is method.last_region_terms
    for row, region in zip(active, method.last_region_terms, strict=True):
        assert region["batch_idx"] == row["batch_idx"]
        assert region["image_idx"] == row["image_idx"] == 71
        assert region["rho"] == row["reliability"]
        assert torch.equal(region["indices"], torch.from_numpy(np.flatnonzero(row["region"])))
    entropy = torch.rand(3, 2, 2, requires_grad=True)
    assert len(method.prepare_trace_objective(entropy, method.last_region_terms)) == 1


@pytest.mark.parametrize("mode,views", [("T", 1), ("C", 1), ("min", 2)])
def test_real_engine_determinism_source_identity_and_diagnostic_counts(mode, views):
    method = tiny_balanced(mode)
    torch.manual_seed(19)
    images = torch.randn(3, 3, 32, 32)
    ids = torch.tensor([17, 902, 17])
    initial = {name: p.detach().clone() for name, p in method.model.named_parameters()}
    method.record_candidate_proposals = True
    method.estimate_trace_reliability(images, image_ids=ids)
    stats = method.last_candidate_stats
    assert stats and method.last_reliability_summary["diagnostic_views"] == views * len(stats)
    assert method.last_reliability_summary["diagnostic_calls"] == math.ceil(views * len(stats) / 3)
    assert method.last_trace_view_batch_size == 3
    assert len([s for s in stats if s["image_idx"] == 17]) <= method.trace_max_regions
    assert len([s for s in stats if s["image_idx"] == 902]) <= method.trace_max_regions
    for row, region in zip(stats, method.last_region_terms, strict=True):
        assert row["image_idx"] == region["image_idx"] == int(ids[row["batch_idx"]])
        assert row["reliability"] == region["rho"]
        assert torch.equal(region["indices"], torch.from_numpy(np.flatnonzero(row["region"])))

    with torch.no_grad():
        entropy = method.trace_softmax_entropy(method.forward_adapt_logits(images).float())
        expected_first_loss = torch.stack([
            TraceObjectiveMixin.trace_region_normalized_entropy(
                entropy, [row for row in method.last_region_terms if row["image_idx"] == source])
            for source in (17, 902)
        ]).mean().item()
    first_losses = method.adapt(images, image_ids=ids)
    assert first_losses[0] == expected_first_loss
    first_prediction = method.evaluate(images).clone()
    first_params = {name: p.detach().clone() for name, p in method.model.named_parameters()}
    method.adapt(images.flip(0), image_ids=ids.flip(0))
    assert method.adapt(images, image_ids=ids) == first_losses
    assert len(first_losses) == method.steps and all(math.isfinite(x) for x in first_losses)
    assert torch.equal(method.evaluate(images), first_prediction)
    assert all(torch.equal(first_params[name], p) for name, p in method.model.named_parameters())
    changed = {name for name in initial if not torch.equal(initial[name], first_params[name])}
    assert changed and changed <= {name for name, p in method.model.named_parameters() if p.requires_grad}
    assert method.model.weights_track == []
    assert method.adapt(images, image_ids=ids, valid_mask=torch.zeros(3, 1, 32, 32)) == []
    assert method.last_region_terms == [] and method.last_candidate_stats == []
    assert method.last_reliability_summary["diagnostic_views"] == 0
    assert method.last_reliability_summary["diagnostic_calls"] == 0
    assert not method.optimizer.state
    assert all(torch.equal(initial[name], p) for name, p in method.model.named_parameters())


@pytest.mark.parametrize("architecture", ["reduced", "vanilla"])
def test_optimizer_and_reset_are_inherited_unchanged(architecture):
    old = tiny_trace(architecture=architecture)
    new = tiny_balanced(architecture=architecture)
    assert new.adapted_parameter_names == old.adapted_parameter_names
    wanted = {id(p) for p in new.model.visual.parameters() if p.requires_grad}
    assert {id(p) for group in new.optimizer.param_groups for p in group["params"]} == wanted
    old_group, new_group = old.optimizer.state_dict()["param_groups"], new.optimizer.state_dict()["param_groups"]
    assert old_group == new_group
    for hook in ("adapt", "reset", "collect_trace_candidates", "forward_eval_logits",
                 "forward_adapt_logits", "trace_softmax_entropy", "_forward_trace_view_batches"):
        assert getattr(ImageBalancedTRACE, hook) is getattr(TwoViewTRACE, hook)


def test_single_source_engine_is_bitwise_equivalent_to_two_view_min():
    method = tiny_balanced()
    torch.manual_seed(43)
    images = torch.randn(2, 3, 32, 32)
    ids = torch.tensor([800, 800])
    losses = method.adapt(images, image_ids=ids)
    prediction = method.evaluate(images).clone()
    stats = copy.deepcopy(method.last_candidate_stats)
    method.__class__ = TwoViewTRACE
    assert method.adapt(images, image_ids=ids) == losses
    assert torch.equal(method.evaluate(images), prediction)
    for row, old in zip(stats, method.last_candidate_stats, strict=True):
        assert all(row[key] == old[key] for key in ("image_idx", "batch_idx", "class_idx",
                   "confidence", "area", "target_support", "counterfactual_support", "reliability"))
        assert np.array_equal(row["region"], old["region"])


@pytest.mark.parametrize("arm", ["Uniform", "Confidence"])
def test_worker_unbound_control_evaluator_keeps_image_balanced_objective(arm, monkeypatch):
    from tools.evolution_followup.standalone import StandaloneTRACE

    method = tiny_balanced()
    method.arm = arm
    monkeypatch.setattr(method, "evaluate_trace_views",
                        StandaloneTRACE.evaluate_trace_views.__get__(method))
    images = torch.randn(3, 3, 32, 32)
    method.estimate_trace_reliability(images, image_ids=[70, -4, 70])
    assert method.last_region_terms
    assert method.last_reliability_summary["diagnostic_views"] == 0
    assert method.last_reliability_summary["diagnostic_calls"] == 0
    entropy = method.trace_softmax_entropy(method.forward_adapt_logits(images))
    terms = method.last_region_terms
    expected = torch.stack([
        TraceObjectiveMixin.trace_region_normalized_entropy(
            entropy, [row for row in terms if row["image_idx"] == image_idx])
        for image_idx in {-4, 70}
    ]).mean()
    torch.testing.assert_close(method.trace_region_normalized_entropy(entropy, terms), expected)
    assert len(method.adapt(images, image_ids=[70, -4, 70])) == method.steps


def test_invalid_mode_fails_before_model_loading():
    with pytest.raises(ValueError, match="view_mode"):
        ImageBalancedTRACE(view_mode="Uniform")
