"""Focused CPU-only tests for target-first scheduling, costs and cache evidence."""

from dataclasses import replace
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

from tools import analyze_forward_budget as readers
from tools.analyze_selective_tracing import (
    DEFAULT_CACHE, EPS, EPSILONS, ROUND_CANDIDATES, counts, episode_analysis,
    load_inputs, q_only, sealed_csv, summarize, target_first, crosscheck_q_only,
)


ROOT = Path(__file__).resolve().parents[1]


def item(t=.8, c=.7, d=.2, **changes):
    return replace(readers.Candidate(0, 'image', 0, 0, 1, readers.sha256(b'mask'),
                                    t, c, d, min(t, c), readers.reliability(t, c, d)), **changes)


def panel_and_cost(items, images=2, steps=3):
    active = any(i.rho > EPS for i in items)
    updates = steps if active else 0
    return (dict(dataset='synthetic', grouped={0: items}, queries=3, class_names=['a', 'b'],
                 query_to_class=[0, 0, 1], warmup=0, config={'steps': steps}),
            dict(episode=0, images=images, effective_view_batch_size=16, update_steps=updates,
                 diagnostic_views=3 * len(items), inputs=3 * len(items) + images * (updates + 2),
                 calls=math.ceil(3 * len(items) / 16) + updates + 2))


def test_stable_t_only_order_bounded_rounds_and_no_unqueried_oracle_reads():
    targets = [.8] * 8 + [.0001] * 9
    calls = []

    def query(indices):
        calls.append(indices)
        assert indices == tuple(range(8))
        return [(.8, 0)] * len(indices)

    result = target_first(targets, query, .01)
    assert calls == [tuple(range(8))]
    assert result['queried'] == list(range(8))
    assert result['weights'][8:] == [0] * 9
    assert result['stopping_bound'] == pytest.approx(.0009 / (6.4 + .0009))
    # Neither lower-ranked C/D nor rho is supplied to the scheduler.
    shuffled = [.1, .9, .9, .2, .5, .3, .7, .6, .4]
    seen = []
    target_first(shuffled, lambda ids: seen.extend(ids) or [(1, 0)] * len(ids), .01)
    assert seen == [1, 2, 6, 7, 4, 8, 5, 3, 0]


def test_eps_gate_exact_zero_and_all_inactive_behavior():
    result = target_first([0, EPS / 2, EPS], lambda ids: pytest.fail('Provably inactive query'), .10)
    assert result['weights'] == [0, 0, 0]
    assert result['exact_t_eps_skips'] == 3 and result['stopping_bound'] == 0
    assert target_first([], lambda ids: [], .01)['rounds'] == []
    calls = []
    result = target_first([1e-6] * 9, lambda ids: calls.append(ids) or [(1e-6, 1)] * len(ids), .10)
    assert len(calls) == 2 and result['rounds'][0]['observed_positive_rho'] == 0
    assert result['rounds'][0]['stopping_bound'] == 1
    assert result['weights'] == [0] * 9 and result['stopping_bound'] == 0


def test_observed_sub_eps_mass_cannot_certify_positive_unseen_tail():
    targets = [1e-4] * 8 + [EPS * 2]
    result = target_first(targets, lambda ids: [(0, 0)] * len(ids), .01)
    assert result['queried'] == list(range(9))
    assert result['rounds'][0]['stopping_bound'] == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError):
        target_first([bad], lambda ids: [(1, 0)], .01)
    with pytest.raises(ValueError):
        target_first([1], lambda ids: [(bad, 0)], .01)
    with pytest.raises(ValueError):
        q_only([1], [bad])


def test_fixed_grid_and_callback_shape():
    assert EPSILONS == (.01, .05, .10) and ROUND_CANDIDATES == 8
    with pytest.raises(ValueError, match='prespecified'):
        target_first([1], lambda ids: [(1, 0)], .02)
    with pytest.raises(ValueError, match='length'):
        target_first([1], lambda ids: [], .01)
    with pytest.raises(ValueError, match='pair'):
        target_first([1], lambda ids: [(1,)], .01)


def test_random_episode_certificate_and_nested_query_prefixes():
    rng = random.Random(20261025)
    boundary = [0, EPS / 2, EPS, math.nextafter(EPS, math.inf), 1]
    for n in [0, 1, 7, 8, 9, 16, 17, 51, 137]:
        for _ in range(12):
            triples = [tuple(rng.choice(boundary) if rng.random() < .2 else rng.random() ** 4
                             for _ in range(3)) for _ in range(n)]
            full = [readers.reliability(*triple) for triple in triples]
            prefixes = []
            for epsilon in EPSILONS:
                result = target_first([triple[0] for triple in triples],
                                      lambda ids: [triples[i][1:] for i in ids], epsilon)
                distance = readers.coefficient_distance(full, result['weights'], [1] * n, 3)
                assert distance['tv'] <= result['stopping_bound'] + 1e-12 <= epsilon + 1e-12
                assert not distance['empty_to_active'] and not distance['active_to_empty']
                assert all(len(row['indices']) <= 8 for row in result['rounds'])
                assert all(row['stopping_bound'] > epsilon for row in result['rounds'][:-1])
                prefixes.append(result['queried'])
            assert prefixes[0][:len(prefixes[1])] == prefixes[1]
            assert prefixes[1][:len(prefixes[2])] == prefixes[2]


def test_episode_normalization_and_class_alias_coverage_loss():
    items = [item(class_idx=i % 2, image_id=str(i)) for i in range(8)]
    items += [item(t=.001, c=.001, class_idx=2, image_id='tail')]
    panel, cost = panel_and_cost(items)
    row, classes, _ = episode_analysis(panel, cost, 'target_first', .01)
    assert row['tv'] == pytest.approx(items[-1].rho / sum(i.rho for i in items))
    assert row['lost_query_classes'] == 1 and row['lost_evaluation_classes'] == 1
    assert row['original_query_classes'] == 3 and row['original_evaluation_classes'] == 2
    assert classes[1]['coverage_lost'] and row['lost_image_class_pairs'] == 1
    assert row['tv'] < .01  # A small TV does not preserve class coverage.
    summary = summarize([row], 'all_episodes')
    assert summary['lost_evaluation_classes_fraction'] == .5


def test_batch_call_overhead_and_partial_rounds():
    panel, cost = panel_and_cost([item(), item(), item()])
    row, _, _ = episode_analysis(panel, cost, 'target_first', .01)
    assert row['view_inputs_saved'] == 0
    assert row['diagnostic_calls_saved'] == -1
    assert row['modeled_diagnostic_calls'] == 2
    panel, cost = panel_and_cost([item(t=0), item(t=0), item()])
    row, _, _ = episode_analysis(panel, cost, 'target_first', .01)
    assert row['view_inputs_saved'] == 4 and row['diagnostic_calls_saved'] == -1
    with pytest.raises(ValueError, match='batch 16'):
        counts({**cost, 'effective_view_batch_size': 8}, 3, 'target_first', 1, 1, True, True, 3)


def test_q_only_can_activate_previously_empty_episode_and_add_update_cost():
    panel, cost = panel_and_cost([item(t=1e-6, c=1e-6, d=1)], images=1, steps=5)
    row, _, _ = episode_analysis(panel, cost, 'q_only')
    assert row['empty_to_active'] and row['tv'] == 1 and row['candidates_activated'] == 1
    assert row['view_inputs_saved'] == 1 and row['all_forward_inputs_saved'] == -4
    assert row['all_forward_calls_saved'] == -5
    assert row['gained_evaluation_classes'] == 1
    selective, _, _ = episode_analysis(panel, cost, 'target_first', .01)
    assert selective['tv'] == 0 and selective['modeled_update_steps'] == 0


def test_float32_denominator_fallback_is_rejected():
    panel, cost = panel_and_cost([item(t=1, c=math.nextafter(EPS, math.inf), d=0)])
    with pytest.raises(ValueError, match='float32 denominator'):
        episode_analysis(panel, cost, 'target_first', .01)


def test_sealed_csv_detects_tampering(tmp_path):
    path = tmp_path / 'rows.csv'
    path.write_text('a\n1\n')
    expected = readers.sha256(path.read_bytes())
    assert sealed_csv(tmp_path, path.name, expected, readers.Evidence()) == [{'a': '1'}]
    path.write_text('a\n2\n')
    with pytest.raises(ValueError, match='seal mismatch'):
        sealed_csv(tmp_path, path.name, expected, readers.Evidence())


def test_imports_no_ml_runtime():
    command = "import sys; import tools.analyze_selective_tracing; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, '-c', command], cwd=ROOT, check=True)


@pytest.mark.skipif(not DEFAULT_CACHE.exists(), reason='Local sealed caches unavailable')
def test_real_cache_provenance_q_only_crosscheck_and_all_episode_certificates():
    evidence = readers.Evidence()
    panels, old_rows, old_classes, matched = load_inputs(DEFAULT_CACHE, ROOT, evidence)
    assert matched['forward'] > 0 and matched['dataset'] > 0
    assert sum(len(panel['candidates']) for panel in panels) == 47113
    for panel in panels:
        rows, classes = [], []
        for cost in panel['costs']:
            row, per_class, _ = episode_analysis(panel, cost, 'q_only')
            rows.append(row)
            classes.extend(per_class)
            for epsilon in EPSILONS:
                row, _, _ = episode_analysis(panel, cost, 'target_first', epsilon)
                assert row['tv'] <= epsilon + 1e-12
                assert row['candidates_activated'] == 0
        crosscheck_q_only(panel, rows, classes, old_rows, old_classes)
    evidence.recheck()
