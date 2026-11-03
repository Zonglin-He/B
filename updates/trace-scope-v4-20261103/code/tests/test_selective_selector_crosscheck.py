"""Post-audit replay of the actual research selector, without importing torch.

Compile only its unmodified class AST. Feed already-cached rho for requested
indices; no model, intervention, support recomputation or production edits.
"""

import ast
import math
from pathlib import Path

import pytest

from tools import analyze_forward_budget as readers
from tools.analyze_selective_tracing import DEFAULT_CACHE, DEFAULT_OUTPUT, EPS, counts, load_inputs


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / 'tools/trace_algorithm_variants.py'


@pytest.fixture(scope='module')
def replay_data():
    if not DEFAULT_OUTPUT.exists() or not SELECTOR.exists():
        pytest.skip('Completed local audit or research selector unavailable')
    evidence = readers.Evidence()
    source = evidence.read(SELECTOR)
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TargetFirstSelector']
    assert len(classes) == 1
    scope = {'math': math, 'EPS': EPS}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SELECTOR), 'exec'), scope)
    provenance = evidence.json(DEFAULT_OUTPUT / 'provenance.json')
    tables = {}
    for name in ('episode_analysis.csv', 'query_rounds.csv'):
        path = DEFAULT_OUTPUT / name
        assert readers.sha256(evidence.read(path)) == provenance['outputs'][name]['sha256']
        tables[name] = evidence.csv(path)
    panels, _, _, _ = load_inputs(DEFAULT_CACHE, ROOT, evidence)
    yield scope['TargetFirstSelector'], panels, tables, readers.sha256(source)
    evidence.recheck()


@pytest.mark.parametrize('epsilon', [.01, .05])
def test_actual_selector_matches_completed_cache_audit(replay_data, epsilon, record_property):
    selector_class, panels, tables, selector_sha = replay_data
    saved = {(row['dataset'], int(row['episode'])): row for row in tables['episode_analysis.csv']
             if row['policy'] == 'target_first' and float(row['epsilon']) == epsilon}
    traces = {}
    for row in tables['query_rounds.csv']:
        if float(row['epsilon']) == epsilon:
            traces.setdefault((row['dataset'], int(row['episode'])), []).append(row)
    checked = 0
    for panel in panels:
        for cost in panel['costs']:
            key = panel['dataset'], cost['episode']
            items = panel['grouped'][cost['episode']]
            selector = selector_class([item.target for item in items], epsilon)
            batch = cost['effective_view_batch_size']
            assert batch == 16
            seen, calls, round_index = [], math.ceil(len(items) / batch), 0
            while indices := selector.next_indices(max(1, batch // 2)):
                trace = traces[key][round_index]
                assert indices == ast.literal_eval(trace['indices'])
                calls += math.ceil(2 * len(indices) / batch)
                for index in indices:
                    selector.observe(index, items[index].rho)
                seen.extend(indices)
                round_index += 1
                assert selector.observed_mass == pytest.approx(float(trace['observed_positive_rho']), abs=1e-13)
                assert selector.unseen_upper_mass == pytest.approx(float(trace['unseen_sum_t']), abs=1e-13)
                assert selector.omitted_mass_bound == pytest.approx(float(trace['stopping_bound']), abs=1e-15)
            row = saved[key]
            assert len(seen) == int(row['cd_queried_candidates'])
            assert round_index == int(row['cd_rounds']) == len(traces.get(key, []))
            assert len(items) + 2 * len(seen) == int(row['modeled_view_inputs'])
            assert calls == int(row['modeled_diagnostic_calls'])
            assert calls + cost['update_steps'] + 2 == int(row['modeled_all_forward_calls'])
            assert len(items) + 2 * len(seen) + cost['images'] * (cost['update_steps'] + 2) == int(row['modeled_all_forward_inputs'])
            assert selector.omitted_mass_bound == pytest.approx(float(row['stopping_bound']), abs=1e-15)
            weights = [selector.observed.get(i, 0) for i in range(len(items))]
            distance = readers.coefficient_distance([item.rho for item in items], weights,
                                                   [item.area for item in items], panel['queries'])
            assert distance['tv'] == pytest.approx(float(row['tv']), abs=1e-15)
            checked += 1
    assert checked == len(saved) == 144
    record_property('selector_sha256', selector_sha)
    record_property('cached_episodes_compared', checked)
    record_property('execution', 'Exact selector class AST; cached rho replay; no model imports or forwards')


@pytest.mark.parametrize('batch', [1, 3, 15])
def test_fixed_16_cache_model_rejects_odd_or_single_view_batches(batch):
    # With B=1 each C,D pair needs two calls; with odd B the last slot is unused.
    # Never substitute ceil(2*M/B) across sequentially gated rounds.
    with pytest.raises(ValueError, match='batch 16'):
        counts({'effective_view_batch_size': batch}, 17, 'target_first', 9, 2, True, True, 3)
