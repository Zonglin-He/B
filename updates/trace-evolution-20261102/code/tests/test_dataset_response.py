"""Focused CPU-only arithmetic and identity boundaries; no runtime or GPU import."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_dataset_response import (EPS, Cell, aggregate_counts, allocation_tables,
    candidate_tables, class_comparison, class_responses, ensure_output, image_comparison,
    k_prefix_budget, nominal_allocations, paired_candidates, parse_vocabulary, score_stats,
    validate_candidates, validate_costs, verify_identity, worker_semantics)


def protocol(variant="full"):
    return dict(dataset="test", split="evaluation", image_ids=["a", "b", "c"],
        label_ids=["a.png", "b.png", "c.png"], vocabulary_sha256="hash", seed=0,
        selection_seed=10, available_images=3, images=3, full_split=True,
        force_uniform_policy=False, variant=variant, config=dict(batch_size=2,
        trace_max_regions=4, trace_view_batch_size=16, lr=.01, steps=1))


def vocabulary():
    return dict(classes=["background", "object"], queries=["bg0", "bg1", "object"],
                query_to_class=[0, 0, 1])


def candidate(image="a", episode=0, class_idx=2, area=1, target=.5, residual=.1, cf=.8, patch=0):
    q = min(target, cf)
    d = max(residual - target, 0)
    return dict(episode=str(episode), image_id=image, patch_in_batch=str(patch),
        class_idx=str(class_idx), area=str(area), mask_sha256="a" * 64,
        target_support=str(target), residual_support=str(residual), counterfactual_support=str(cf),
        quality=str(q), dominance=str(d), reliability=str(q * (1 - d)))


def make_cell(rows=None, variant="full"):
    p = protocol(variant)
    if rows is None:
        rows = [candidate(), candidate(image="b", area=9, class_idx=0, patch=1)]
    if variant == "no_source_tracing":
        rows = [dict(r, target_support="", residual_support="", counterfactual_support="",
                     quality="", dominance="", reliability="0.8") for r in rows]
    candidates = validate_candidates(rows, p, vocabulary())
    records = [dict(image_id=i, prediction_sha256="same", intersection=[1, 2], union=[2, 4])
               for i in p["image_ids"]]
    costs = [dict(episode=0, images=2, candidates=len(candidates), diagnostic_views=3 * len(candidates),
                  adaptation_seconds=.1, inference_seconds=.1, total_seconds=.2),
             dict(episode=1, images=1, candidates=0, diagnostic_views=0,
                  adaptation_seconds=.1, inference_seconds=.1, total_seconds=.2)]
    return Cell(Path("/record") / variant, p, dict(warmup_episodes=1,
        diagnostic_views_per_image=0, ms_per_image=200, peak_allocated_mib=1), records, vocabulary(), candidates, costs,
        50, [50, 50], "ablation")


@pytest.mark.parametrize("field,value", [
    ("dataset", "other"), ("split", "train"), ("image_ids", ["b", "a", "c"]),
    ("label_ids", ["z.png", "b.png", "c.png"]), ("vocabulary_sha256", "old-vocab"),
    ("seed", 1), ("selection_seed", 11), ("available_images", 4), ("images", 2),
    ("full_split", False), ("force_uniform_policy", True),
])
def test_unmatched_identity_rejected(field, value):
    a, b = protocol(), protocol()
    b[field] = value
    with pytest.raises(ValueError, match=field):
        verify_identity(a, b)


def test_episode_batch_and_view_batch_changes_not_silently_allowed():
    for name in ("batch_size", "trace_view_batch_size"):
        a, b = protocol(), protocol()
        b["config"][name] += 1
        with pytest.raises(ValueError, match="config"):
            verify_identity(a, b)


def test_counts_use_ratio_of_sums_with_undefined_class_not_zero():
    records = [dict(intersection=[1, 0], union=[1, 0]), dict(intersection=[1, 0], union=[9, 0])]
    i, u, iou, miou = aggregate_counts(records, 2)
    assert (i, u, iou, miou) == ([2, 0], [10, 0], [20, None], 20)
    assert miou != pytest.approx((100 + 100 / 9) / 2)
    assert aggregate_counts([dict(intersection=[0, 0], union=[0, 0])], 2)[2:] == ([None, None], None)
    assert aggregate_counts([], 2)[2:] == ([None, None], None)


@pytest.mark.parametrize("counts", [[1], [1, -1], [1, .5], [1, float("nan")], [1, True]])
def test_invalid_count_axes_or_values_rejected(counts):
    with pytest.raises(ValueError):
        aggregate_counts([dict(intersection=counts, union=[2, 2])], 2)


def test_intersection_cannot_exceed_union():
    with pytest.raises(ValueError, match="exceeds"):
        aggregate_counts([dict(intersection=[3, 0], union=[2, 0])], 2)


def test_alias_queries_are_mapped_to_evaluation_rows():
    vocab = parse_vocabulary("sky,wall,road\ncat,kitten\n", True, 2)
    assert vocab["classes"] == ["background", "cat"]
    assert vocab["query_to_class"] == [0, 0, 0, 1, 1]
    plain = parse_vocabulary("sky,wall,road\ncat,kitten\n", False, 2)
    assert plain["queries"] == ["background", "cat"]
    assert plain["vocabulary_sha256"] != vocab["vocabulary_sha256"]
    rows = validate_candidates([candidate(class_idx=1)], protocol(), vocabulary())
    assert next(iter(rows.values()))["evaluation_class_idx"] == 0


def test_ambiguous_vocabulary_and_wrong_class_axis_rejected():
    with pytest.raises(ValueError, match="owner"):
        parse_vocabulary("Sky\nsky\n", True, 2)
    with pytest.raises(ValueError, match="class axis"):
        parse_vocabulary("Sky\ncat\n", True, 3)


@pytest.mark.parametrize("field,value,message", [
    ("episode", "1", "episode"), ("image_id", "unknown", "episode"),
    ("area", "0", "area"), ("area", "1.5", "area"), ("class_idx", "3", "query"),
    ("class_idx", "-1", "class_idx"), ("mask_sha256", "bad", "hash"),
    ("reliability", "nan", "reliability"), ("quality", "0.1", "formula"),
    ("target_support", "", "support"),
])
def test_malformed_candidate_rejected(field, value, message):
    row = candidate()
    row[field] = value
    with pytest.raises(ValueError, match=message):
        validate_candidates([row], protocol(), vocabulary())


def test_duplicate_candidate_and_inconsistent_patch_owner_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_candidates([candidate(), candidate()], protocol(), vocabulary())
    with pytest.raises(ValueError, match="two images"):
        validate_candidates([candidate(), candidate(image="b")], protocol(), vocabulary())


def test_single_patch_protocol_rejects_within_episode_swapped_image_ids():
    p = protocol()
    p["config"].update(resize=[224, 224], patch_size=[224, 224])
    with pytest.raises(ValueError, match="image slot"):
        validate_candidates([candidate(image="b", patch=0)], p, vocabulary())
    validate_candidates([candidate(image="b", patch=1)], p, vocabulary())


def test_candidate_alignment_is_keyed_not_csv_position():
    full, raw = make_cell(), make_cell(variant="no_source_tracing")
    raw.candidates = dict(reversed(list(raw.candidates.items())))
    paired_candidates(full, raw, same_support=False)
    rows, classes, images, episode_classes = allocation_tables(full, raw)
    assert len(rows) == 2
    assert rows[0]["full_vs_raw_tv"] == 0
    assert rows[1]["full_vs_raw_tv"] is None
    assert rows[1]["full_effective_candidates"] is None
    assert sum(r["full_mean_episode_coefficient_mass"] for r in classes) == .5
    assert images[-1]["full_episode_coefficient_mass"] == 0
    # Empty last episode remains in the equal-episode denominator.
    assert classes[0]["full_mean_episode_coefficient_mass"] == .25
    assert len(episode_classes) == 4
    assert episode_classes[-1]["full_episode_coefficient_mass"] == 0


def test_nominal_area_weighting_changes_region_mean_allocation():
    rows = list(make_cell().candidates.values())
    weights = nominal_allocations(rows, [.8, .8])
    assert weights["full"] == [.5, .5]
    assert weights["no_region_normalization"] == [.1, .9]
    assert weights["quality_only"] == weights["full"]
    rows[0]["reliability"] = EPS / 2
    rows[0]["area"] = 100
    result = nominal_allocations(rows, [.8, .8])
    assert result["full"][0] == 0
    assert result["no_region_normalization"][0] == 0  # Estimator gate precedes area multiplication.
    rows[0]["reliability"] = EPS
    assert nominal_allocations(rows, [.8, .8])["no_region_normalization"][0] == 0
    rows[0]["reliability"] = 2 * EPS
    assert nominal_allocations(rows, [.8, .8])["no_region_normalization"][0] > 0


def test_uniform_dominance_penalty_cancels_after_episode_normalization():
    rows = list(make_cell([candidate(target=.5, residual=.7),
        candidate(image="b", patch=1, target=.5, residual=.7)]).candidates.values())
    stats = score_stats(rows)
    assert stats["dominance_positive_fraction"] == 1
    assert stats["q_weighted_dominance_penalty_fraction"] == pytest.approx(.2)
    weights = nominal_allocations(rows, [.8, .8])
    assert weights["full"] == weights["quality_only"]


def test_min_bottleneck_ties_zero_support_and_all_empty_rows():
    p = protocol()
    rows = [candidate(target=.2, cf=.8), candidate(image="b", patch=1, target=.8, cf=.2),
            candidate(image="c", episode=1, target=0, cf=0)]
    stats = score_stats(validate_candidates(rows, p, vocabulary()).values())
    assert stats["target_is_strict_min_fraction"] == pytest.approx(1 / 3)
    assert stats["counterfactual_is_strict_min_fraction"] == pytest.approx(1 / 3)
    assert stats["min_tie_fraction"] == pytest.approx(1 / 3)
    assert stats["min_over_max_support_n"] == 2
    assert stats["min_over_max_support_mean"] == .25
    empty = score_stats([])
    assert empty["quality_mean"] is None
    assert empty["dominance_positive_fraction"] is None
    assert empty["reliability_le_eps_fraction"] is None


def test_costs_use_measured_image_count_and_include_partial_episode():
    cell = make_cell()
    costs = [dict(row, peak_allocated_mib=1, update_steps=1, effective_view_batch_size=16,
                  calls=4, inputs=6) for row in cell.costs]
    validated = validate_costs(costs, cell.candidates, cell.protocol, cell.result)
    assert validated[-1]["images"] == 1
    result = dict(cell.result, diagnostic_views_per_image=2)
    with pytest.raises(ValueError, match="views/image"):
        validate_costs(costs, cell.candidates, cell.protocol, result)
    bad = copy.deepcopy(costs)
    bad[0]["diagnostic_views"] -= 1
    with pytest.raises(ValueError, match="Three-view"):
        validate_costs(bad, cell.candidates, cell.protocol, cell.result)


def test_candidate_tables_keep_cohort_and_episode_denominators_separate():
    cell = make_cell()
    summary, episodes, images, classes = candidate_tables(cell)
    assert summary["candidates_per_image"] == pytest.approx(2 / 3)
    assert summary["all_episode_views_per_image"] == 2
    assert summary["measured_views_per_image"] == 0
    assert summary["measured_images"] == 1
    assert episodes[1]["images"] == 3
    assert episodes[1]["episode_images"] == 1
    assert images[2]["candidates"] == 0
    assert len(classes) == 2


def test_class_comparison_keeps_absent_axis_and_correct_sign():
    full, raw = make_cell(), make_cell(variant="no_source_tracing")
    raw.records[0]["intersection"] = [2, 0]
    comparison = class_comparison(full, raw)
    assert comparison[0]["delta_full_minus_control_pp"] == pytest.approx(-100 / 6)
    assert comparison[1]["delta_full_minus_control_pp"] == pytest.approx(100 / 6)
    image_rows = image_comparison(full, raw)
    assert image_rows[0]["delta_micro_iou_pp"] == pytest.approx(100 / 6)
    assert image_rows[2]["episode"] == 1


def test_class_response_joins_on_class_identity_and_documents_association():
    full, raw = make_cell(), make_cell(variant="no_source_tracing")
    _, allocation, _, _ = allocation_tables(full, raw)
    _, _, _, candidates = candidate_tables(full)
    deltas = class_comparison(full, raw)
    rows = class_responses(deltas, list(reversed(allocation)), list(reversed(candidates)))
    assert [r["class_idx"] for r in rows] == [0, 1]
    assert all(r["coefficient_mass_delta_pp"] == 0 for r in rows)
    assert all("not gradient" in r["interpretation"] for r in rows)


def test_worker_match_only_erases_metadata_instrumentation():
    original = 'import json\ndef main():\n protocol = dict(dataset="x", environment=dict(python="v"))\n train(steps=5)\n'
    updated = 'import json\nimport os\nimport subprocess\ndef execution_environment():\n return dict(python="v")\ndef main():\n protocol = dict(dataset="x", file_metadata=[], environment=execution_environment())\n train(steps=5)\n'
    assert worker_semantics(original) == worker_semantics(updated)
    assert worker_semantics(original) != worker_semantics(updated.replace("steps=5", "steps=6"))


def test_output_cannot_overwrite_or_overlap_sources():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="fresh"):
        ensure_output(root, [root])
    with pytest.raises(ValueError, match="overlaps"):
        ensure_output(root / "__not_created_dataset_response_test__", [root])


def test_import_is_stdlib_only_and_does_not_load_gpu_runtime():
    root = str(Path(__file__).resolve().parents[1] / "tools")
    code = (f"import sys,json; sys.path.insert(0,{root!r}); import analyze_dataset_response; "
            "print(json.dumps([n for n in sys.modules if n.split('.')[0] in "
            "('torch','numpy','trace_ovss')]))")
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == []


def prefix_cells():
    rows = [candidate(area=5 - i, class_idx=i % 2, patch=0, target=.1 * (i + 1)) for i in range(5)]
    full, raw = make_cell([]), make_cell([], variant="no_source_tracing")
    full.protocol["config"]["trace_max_regions"] = raw.protocol["config"]["trace_max_regions"] = 5
    for i, row in enumerate(rows):
        row["mask_sha256"] = str(i) * 64
    full.candidates = validate_candidates(rows, full.protocol, vocabulary())
    raw_rows = [dict(r, target_support="", residual_support="", counterfactual_support="",
                    quality="", dominance="", reliability=str(.9 - i * .1)) for i, r in enumerate(rows)]
    raw.candidates = validate_candidates(raw_rows, raw.protocol, vocabulary())
    return full, raw


def test_k_prefix_keeps_raw_order_not_highest_rho_and_caps_at_cache():
    full, raw = prefix_cells()
    rows = k_prefix_budget(full, raw, "/frozen/candidates.py")
    assert rows[0]["retained_candidates"] == 4
    assert rows[0]["retained_rho_fraction"] == pytest.approx(1 / 1.5)
    assert rows[0]["saved_diagnostic_views"] == 3
    assert rows[0]["retained_classes"] == 1
    assert rows[1]["requested_exceeds_cache"] is True
    assert rows[1]["effective_cached_k"] == 5
    assert rows[-1]["retained_rho_fraction"] == 1


def test_k_prefix_rejects_confidence_or_tie_rank_inversion():
    full, raw = prefix_cells()
    list(raw.candidates.values())[1]["reliability"] = .99
    with pytest.raises(ValueError, match="nonincreasing"):
        k_prefix_budget(full, raw, "/frozen/candidates.py")


def test_k_prefix_does_not_repair_mismatched_csv_order():
    full, raw = prefix_cells()
    raw.candidates = dict(reversed(list(raw.candidates.items())))
    with pytest.raises(ValueError, match="saved candidate order"):
        k_prefix_budget(full, raw, "/frozen/candidates.py")


def test_empty_k_prefix_keeps_undefined_mass_and_coverage():
    full, raw = make_cell([]), make_cell([], variant="no_source_tracing")
    row = k_prefix_budget(full, raw, "/frozen/candidates.py")[0]
    assert row["retained_rho_fraction"] is None
    assert row["retained_class_fraction"] is None
    assert row["mean_episode_retained_full_coefficient_mass"] is None
