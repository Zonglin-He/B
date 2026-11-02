"""CPU-only mathematical, malformed-cache and real-cache forward-budget tests."""

import ast
from dataclasses import replace
import itertools
import math
from pathlib import Path
import random
import runpy

import pytest

from tools.analyze_forward_budget import (
    Candidate, DEFAULT_SUMMARY, EPS, Evidence, PANELS, THRESHOLDS, audit_panel,
    coefficient_distance, compact_summary, episode_analysis, float32,
    float32_objective_coefficients, load_panel, normalized_coefficients,
    parse_candidates, policy_skip, reliability, select_panels, sha256,
    singleton_certificate, summarize,
)


ROOT = Path(__file__).resolve().parents[1]


def candidate(target=.8, counterfactual=.6, residual=.9, **kwargs):
    q = min(target, counterfactual)
    item = Candidate(0, "image", 0, 0, 1, sha256(bytes([1, 0, 0, 0])),
                     target, counterfactual, residual, q,
                     reliability(target, counterfactual, residual))
    return replace(item, **kwargs)


def cache_row(item):
    return dict(episode=str(item.episode), image_id=item.image_id, patch_in_batch=str(item.patch),
        class_idx=str(item.class_idx), area=str(item.area), mask_sha256=item.mask_sha256,
        target_support=str(item.target), counterfactual_support=str(item.counterfactual),
        residual_support=str(item.residual), quality=str(item.quality),
        dominance=str(max(item.residual - item.target, 0)), reliability=str(item.rho))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -.01, 1.01])
def test_reject_nonfinite_and_out_of_range_probabilities(bad):
    for location in range(3):
        values = [.3, .5, .7]
        values[location] = bad
        with pytest.raises(ValueError, match="finite probability"):
            reliability(*values)
    with pytest.raises(ValueError):
        normalized_coefficients([bad], [1])


def test_bounds_match_actual_production_on_boundary_grid_and_random_values():
    production = runpy.run_path(str(ROOT / "trace_ovss/method/trace/reliability.py"))["trace_reliability"]
    boundaries = [0, EPS / 2, EPS, math.nextafter(EPS, math.inf), .001, .1, .5, 1]
    rng = random.Random(731)
    triples = list(itertools.product(boundaries, repeat=3))
    triples += [tuple(rng.random() for _ in range(3)) for _ in range(2000)]
    for target, counterfactual, residual in triples:
        value = reliability(target, counterfactual, residual)
        q = min(target, counterfactual)
        assert value == production(target, counterfactual, residual).reliability
        assert q * target <= value + 1e-15 <= q + 2e-15
        assert -1e-15 <= q - value <= q * (1 - target) + 1e-15
        assert q - value == pytest.approx(q * max(residual - target, 0), abs=1e-15)


def test_raw_zero_not_confused_with_epsilon_discard():
    tiny = candidate(target=1e-6, counterfactual=1e-6, residual=1)
    assert tiny.rho > 0 and tiny.rho <= EPS < tiny.quality
    assert not tiny.exact_drop
    assert singleton_certificate([tiny]) is None
    distance = coefficient_distance([tiny.rho], [tiny.quality], [1], 7)
    assert distance["empty_to_active"]
    assert distance["coefficient_l1"] == 1
    assert distance["standard_tv"] is None
    assert distance["coefficient_l1_with_null"] == 2
    assert distance["tv"] == 1
    assert distance["entropy_loss_bound_nats"] == math.log(7)


def test_empty_and_eps_normalization_boundaries():
    assert normalized_coefficients([], []) == ([], 0)
    assert normalized_coefficients([EPS, 0, .5], [1, 1, 0]) == ([0, 0, 0], 0)
    above = math.nextafter(EPS, math.inf)
    assert normalized_coefficients([above], [1]) == ([1], above)
    assert coefficient_distance([], [], [], 1)["tv"] == 0
    assert coefficient_distance([.5], [0], [1], 2)["tv"] == 1
    with pytest.raises(ValueError):
        normalized_coefficients([.5], [])
    with pytest.raises(ValueError):
        normalized_coefficients([.5], [-1])
    with pytest.raises(ValueError):
        coefficient_distance([.5], [.5], [1], 0)


def test_exact_q_eps_and_target_one_drop_gates():
    for residual in [0, .2, 1]:
        for item in (candidate(counterfactual=EPS, residual=residual),
                     candidate(target=1, residual=residual)):
            assert item.exact_drop
            assert coefficient_distance([item.rho], [item.quality], [1], 5)["tv"] == 0
    assert not candidate(target=math.nextafter(1, 0)).exact_drop


def test_singleton_cancellation_requires_episode_survival_certificate():
    good = candidate()
    assert singleton_certificate([good]) == 0
    assert policy_skip([good], "exact_plus_certified_singleton") == [True]
    assert coefficient_distance([good.rho], [good.quality], [1], 60)["tv"] == 0
    zero = candidate(target=0)
    assert singleton_certificate([zero, good]) == 1
    assert singleton_certificate([good, good]) is None
    mixed = coefficient_distance([.1, .8], [.2, .8], [1, 1], 60)
    assert mixed["tv"] > 0  # One candidate per image is not one per episode.


def test_singleton_lower_bound_uses_float32_normalizer_boundary():
    above = math.nextafter(EPS, math.inf)
    item = candidate(target=1, counterfactual=above, residual=0)
    assert item.lower > EPS and float32(item.lower) <= EPS
    assert singleton_certificate([item]) is None
    ideal, _ = normalized_coefficients([above], [1])
    finite, total = float32_objective_coefficients([above], [1])
    assert ideal == [1] and total <= EPS and finite[0] < EPS


def test_actual_production_objective_on_cpu_empty_masks_and_float32_boundary(monkeypatch):
    import torch

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("GPU is forbidden in this audit")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden_cuda)
    path = ROOT / "trace_ovss/method/trace/objective.py"
    tree = ast.parse(path.read_text())
    classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)], type_ignores=[])
    scope = {"torch": torch, "EPS": EPS}
    exec(compile(classes, str(path), "exec"), scope)
    objective = scope["TraceObjectiveMixin"]
    for weights, areas in (([.2, .8], [1, 1]), ([EPS, .8], [1, 0]),
                          ([math.nextafter(EPS, math.inf)], [1]), ([], [])):
        entropy = torch.ones(1, max(1, len(weights)), dtype=torch.float32, device="cpu", requires_grad=True)
        terms = [dict(batch_idx=0, rho=weight, indices=torch.tensor([index] if area else [], dtype=torch.long))
                 for index, (weight, area) in enumerate(zip(weights, areas))]
        coefficients, total = float32_objective_coefficients(weights, areas)
        prepared, normalizer = objective.prepare_trace_objective(entropy, terms)
        assert normalizer.item() == (total if total > EPS else 1)
        loss = objective.trace_region_normalized_entropy(entropy, terms, prepared=(prepared, normalizer))
        assert math.isfinite(loss.item())
        gradient = torch.autograd.grad(loss, entropy)[0].flatten().tolist()
        assert gradient[:len(weights)] == pytest.approx(coefficients, abs=1e-8, rel=1e-6)
        assert not torch.cuda.is_initialized()


def test_observed_zero_penalty_is_not_a_pre_drop_certificate():
    item = candidate(residual=.1)
    assert item.rho == item.quality and not item.exact_drop
    assert not policy_skip([item], "exact_drop")[0]
    assert not policy_skip([item], "bound_skip", 0)[0]


def test_fixed_thresholds_use_bound_not_actual_D():
    low = candidate(target=.9, counterfactual=.01, residual=0)
    high = replace(low, residual=1, rho=reliability(.9, .01, 1))
    for threshold in THRESHOLDS:
        assert policy_skip([low], "bound_skip", threshold) == policy_skip([high], "bound_skip", threshold)
    with pytest.raises(ValueError):
        policy_skip([low], "bound_skip", .004)


def test_small_absolute_error_can_have_large_normalized_error():
    item = candidate(target=1e-4, counterfactual=1e-4, residual=1)
    other = candidate(target=1e-4, counterfactual=1e-4, residual=0)
    assert item.bound < .001
    change = coefficient_distance([item.rho, other.rho], [item.quality, other.rho], [1, 1], 2)
    assert change["tv"] > .49


def test_entropy_bound_exhaustive_extremes_including_empty_objectives():
    for before in ([0, 0], [.1, .9], [.6, .4], [EPS, .2]):
        for after in ([0, 0], [.4, .6], [.9, EPS], [.2, .1]):
            a, _ = normalized_coefficients(before, [1, 1])
            b, _ = normalized_coefficients(after, [1, 1])
            distance = coefficient_distance(before, after, [1, 1], 73)
            for entropies in itertools.product([0, math.log(73)], repeat=2):
                actual = abs(math.fsum((x-y)*h for x, y, h in zip(a, b, entropies)))
                assert actual <= distance["entropy_loss_bound_nats"] + 1e-12


@pytest.mark.parametrize("changes", [
    {"area": "0"}, {"area": "5"}, {"area": "1.5"}, {"class_idx": "2"},
    {"target_support": "nan"}, {"reliability": "inf"}, {"reliability": ".99"},
    {"quality": ".99"}, {"dominance": ".99"}, {"episode": "-1"},
    {"mask_sha256": "not-a-hash"}, {"area": "4"},
])
def test_reject_malformed_candidate_cache(changes):
    row = {**cache_row(candidate()), **changes}
    with pytest.raises(ValueError):
        parse_candidates([row], 2, 4)


def test_duplicate_candidate_and_whole_mask_hash_validation():
    row = cache_row(candidate())
    with pytest.raises(ValueError, match="Duplicate"):
        parse_candidates([row, row], 2, 4)
    whole = candidate(area=4, mask_sha256=sha256(bytes([1, 1, 1, 1])))
    assert parse_candidates([cache_row(whole)], 2, 4)[0].whole_mask(4)
    assert not candidate().whole_mask(4)


def test_whole_mask_input_identity_does_not_remove_residual_penalty():
    image, weak, context = [1, 2, 3], [7, 8, 9], [4, 5, 6]
    target = [x * 1 + w * 0 for x, w in zip(image, weak)]
    counterfactual = [x * 1 + c * 0 for x, c in zip(image, context)]
    residual = [x * 0 + w * 1 for x, w in zip(image, weak)]
    assert target == counterfactual == image and residual == weak
    item = candidate(target=.2, counterfactual=.2, residual=.9)
    assert item.rho < item.quality and not item.exact_drop


def test_two_stage_calls_can_increase_despite_saved_views():
    items = [candidate(target=0), candidate(), candidate()]
    panel = dict(dataset="synthetic", grouped={0: items}, queries=2, warmup=0)
    cost = dict(episode=0, images=1, effective_view_batch_size=16,
                diagnostic_views=9, inputs=16, calls=8)
    row = episode_analysis(panel, cost, "exact_drop")
    assert row["skipped_drop_views"] == 1
    assert row["modeled_diagnostic_calls_saved"] == -1
    assert row["tv"] == 0
    summary = summarize([row], "all_episodes")
    assert summary["diagnostic_views_saved_fraction"] == pytest.approx(1/9)


def test_skip_drop_reactivation_adds_updates_and_can_have_negative_net_savings():
    item = candidate(target=1e-6, counterfactual=1e-6, residual=1)
    panel = dict(dataset="synthetic", grouped={0: [item]}, queries=3, warmup=0, config={"steps": 5})
    cost = dict(episode=0, images=1, effective_view_batch_size=16, diagnostic_views=3,
                inputs=5, calls=3, update_steps=0)
    row = episode_analysis(panel, cost, "bound_skip", .001)
    assert row["empty_to_active"] and row["tv"] == 1
    assert row["skipped_drop_views"] == 1
    assert row["additional_update_forward_inputs"] == 5
    assert row["net_all_forward_inputs_saved"] == -4
    assert row["modeled_all_forward_calls_saved"] == -5
    assert summarize([row], "all_episodes")["all_forward_inputs_saved_fraction"] == -.8


def test_selection_deduplicates_cd_stages_and_rejects_legacy():
    rows = [dict(stage="ablation", dataset=name, variant="full", images=str(images),
                 result_path=f"/x/{'fresh_full_coco_cd' if name == 'coco_cd' else name}/result.json")
            for name, (images, _) in PANELS.items()]
    rows.append({**rows[-1], "stage": "full_validation"})
    assert len(select_panels(rows)) == 4
    with pytest.raises(ValueError, match="exactly one"):
        select_panels(rows + [rows[0]])
    rows[3]["result_path"] = "/old_external/result.json"
    with pytest.raises(ValueError, match="refreshed"):
        select_panels(rows)


@pytest.fixture(scope="module")
def real_panels():
    if not DEFAULT_SUMMARY.exists():
        pytest.skip("Sealed local real-cache panels are not installed")
    evidence = Evidence()
    panels = [load_panel(row, evidence, ROOT) for row in select_panels(evidence.csv(DEFAULT_SUMMARY))]
    evidence.recheck()
    return panels


def test_real_cache_integrity_all_four_panels(real_panels):
    assert [panel["images"] for panel in real_panels] == [512, 512, 512, 500]
    assert sum(len(panel["candidates"]) for panel in real_panels) > 1000
    for panel in real_panels:
        audit = audit_panel(panel)
        assert audit["zero_area_candidates"] == 0
        assert audit["query_classes"] >= 2
        assert audit["candidates"] == sum(row["candidates"] for row in panel["costs"])


def test_real_activations_are_explicitly_nonexact_and_wholemask_savings_zero(real_panels):
    expected = {"voc21": 8, "pc60": 6, "coco_object": 20, "coco_cd": 0}
    for panel in real_panels:
        records = [episode_analysis(panel, cost, "bound_skip", .001) for cost in panel["costs"]]
        summary = summarize(records, "all_episodes")
        compact = compact_summary([summary])[0]
        assert compact["candidate_activations"] == expected[panel["dataset"]]
        assert compact["status"] == "NONEXACT_OFFLINE_NOT_ADOPTED"
        assert audit_panel(panel)["whole_mask_TC_structural_views_potential"] == 0
        assert summary["empty_to_active"] == 0


def test_real_cache_strategies_monotone_and_exact_guards_hold(real_panels):
    for panel in real_panels:
        for cost in panel["costs"]:
            for policy in ("exact_drop", "exact_plus_certified_singleton"):
                exact = episode_analysis(panel, cost, policy)
                assert exact["tv"] < 1e-12 and not exact["empty_to_active"]
            previous = -1
            for threshold in THRESHOLDS:
                row = episode_analysis(panel, cost, "bound_skip", threshold)
                assert row["skipped_drop_views"] >= previous
                assert row["raw_rho_l1"] <= row["raw_rho_bound_sum"] + 1e-12
                assert row["coefficient_l1"] <= 2 + 1e-12
                previous = row["skipped_drop_views"]


def test_real_cached_D_distances_with_independent_plain_arithmetic(real_panels):
    for panel in real_panels:
        for cost in panel["costs"]:
            items = panel["grouped"][cost["episode"]]
            raw = [item.quality * (1 - max(item.residual - item.target, 0)) for item in items]
            approximate = [item.quality if item.quality * (1-item.target) <= .02 or item.exact_drop
                           else weight for item, weight in zip(items, raw)]
            before = [value if value > EPS else 0 for value in raw]
            after = [value if value > EPS else 0 for value in approximate]
            sum_a, sum_b = sum(before), sum(after)
            a = [value / sum_a if sum_a else 0 for value in before]
            b = [value / sum_b if sum_b else 0 for value in after]
            expected = sum(abs(x-y) for x, y in zip(a, b))
            actual = episode_analysis(panel, cost, "bound_skip", .02)
            assert actual["coefficient_l1"] == pytest.approx(expected, abs=2e-12)
