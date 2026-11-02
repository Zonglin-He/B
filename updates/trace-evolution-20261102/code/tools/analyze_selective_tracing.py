"""CPU-only episode-level target-first tracing audit on sealed Full caches.

Run as a script or with python -m tools.analyze_selective_tracing. No runtime,
torch, predictions, labels, optimizer or queue is loaded. Epsilons are fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

if __package__:
    from . import analyze_forward_budget as readers
else:
    import analyze_forward_budget as readers


EPS = readers.EPS
EPSILONS = (.01, .05, .10)
VIEW_BATCH = 16
ROUND_CANDIDATES = VIEW_BATCH // 2
DEFAULT_CACHE = Path('/home/wwww/hzl/results/trace_efficiency_diagnosis_20261025')
DEFAULT_OUTPUT = Path('/home/wwww/hzl/results/trace_algorithm_efficiency_20261025/cached_audit')
require = readers.require


def target_first(targets, query_cd, epsilon):
    """The scheduler can see all T, but C/D only via its bounded query callback.

    Returned weights preserve original index order, not query order. Values
    <=EPS are zero in both the stopping normalizer and the returned objective.
    """
    require(epsilon in EPSILONS, 'Epsilon must be in the prespecified grid')
    targets = tuple(readers.probability(value) for value in targets)
    order = sorted((i for i, target in enumerate(targets) if target > EPS),
                   key=lambda i: (-targets[i], i))
    weights = [0.0] * len(targets)
    queried, rounds = [], []
    cursor = 0

    def state():
        unseen = math.fsum(targets[i] for i in order[cursor:])
        observed = math.fsum(weights)
        bound = unseen / (observed + unseen) if unseen else 0.0
        return unseen, observed, bound

    unseen, observed, bound = state()
    while bound > epsilon:
        indices = tuple(order[cursor:cursor + ROUND_CANDIDATES])
        require(bool(indices), 'Unresolved bound with no unseen candidates')
        responses = list(query_cd(indices))
        require(len(responses) == len(indices), 'C/D query response length mismatch')
        for index, response in zip(indices, responses):
            require(len(response) == 2, 'Expected one C,D pair per queried candidate')
            rho = readers.reliability(targets[index], *response)
            weights[index] = rho if rho > EPS else 0.0
        queried.extend(indices)
        cursor += len(indices)
        unseen, observed, bound = state()
        rounds.append(dict(round=len(rounds) + 1, indices=list(indices),
                           observed_positive_rho=observed, unseen_sum_t=unseen,
                           stopping_bound=bound))
    return dict(weights=weights, queried=queried, rounds=rounds,
                stopping_bound=bound, unseen_sum_t=unseen,
                observed_positive_rho=observed,
                exact_t_eps_skips=sum(t <= EPS for t in targets))


def q_only(targets, counterfactuals):
    """Two views for the same complete selected candidate pool; no D access."""
    require(len(targets) == len(counterfactuals), 'T/C length mismatch')
    values = [min(readers.probability(t), readers.probability(c))
              for t, c in zip(targets, counterfactuals)]
    return [value if value > EPS else 0.0 for value in values]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def counts(cost, candidates, policy, queried, rounds, before_active, after_active, steps):
    require(cost['effective_view_batch_size'] == VIEW_BATCH, 'Expected observed view batch 16')
    require(cost['diagnostic_views'] == 3 * candidates, 'Original view count mismatch')
    original_calls = math.ceil(3 * candidates / VIEW_BATCH)
    require(cost['calls'] == original_calls + cost['update_steps'] + 2, 'Original call mismatch')
    require(cost['inputs'] == 3 * candidates + cost['images'] * (cost['update_steps'] + 2),
            'Original input mismatch')
    require(cost['update_steps'] == (steps if before_active else 0), 'Original step mismatch')
    if policy == 'target_first':
        views = candidates + 2 * queried
        calls = math.ceil(candidates / VIEW_BATCH) + rounds
        require(rounds == math.ceil(queried / ROUND_CANDIDATES), 'Bounded round count mismatch')
    else:
        require(policy == 'q_only', 'Unknown policy')
        views, calls = 2 * candidates, math.ceil(2 * candidates / VIEW_BATCH)
    after_steps = steps if after_active else 0
    inputs = views + cost['images'] * (after_steps + 2)
    all_calls = calls + after_steps + 2
    return dict(original_view_inputs=3 * candidates, modeled_view_inputs=views,
                view_inputs_saved=3 * candidates - views,
                original_diagnostic_calls=original_calls, modeled_diagnostic_calls=calls,
                diagnostic_calls_saved=original_calls - calls,
                original_all_forward_inputs=cost['inputs'], modeled_all_forward_inputs=inputs,
                all_forward_inputs_saved=cost['inputs'] - inputs,
                original_all_forward_calls=cost['calls'], modeled_all_forward_calls=all_calls,
                all_forward_calls_saved=cost['calls'] - all_calls,
                original_update_steps=cost['update_steps'], modeled_update_steps=after_steps)


def episode_analysis(panel, cost, policy, epsilon=None):
    items = panel['grouped'][cost['episode']]
    targets = [item.target for item in items]
    if policy == 'target_first':
        selection = target_first(targets,
            lambda indices: [(items[i].counterfactual, items[i].residual) for i in indices], epsilon)
        after = selection['weights']
    else:
        require(policy == 'q_only' and epsilon is None, 'Invalid comparison')
        after = q_only(targets, [item.counterfactual for item in items])
        selection = dict(queried=[], rounds=[], exact_t_eps_skips=0,
                         stopping_bound=None, unseen_sum_t=None, observed_positive_rho=None)
    before = [item.rho for item in items]
    areas = [item.area for item in items]
    distance = readers.coefficient_distance(before, after, areas, panel['queries'])
    a, total_a = readers.normalized_coefficients(before, areas)
    b, total_b = readers.normalized_coefficients(after, areas)
    cast_a, cast_total_a = readers.float32_objective_coefficients(before, areas)
    cast_b, cast_total_b = readers.float32_objective_coefficients(after, areas)
    require(not total_a or cast_total_a > EPS, 'Reference float32 denominator fallback')
    require(not total_b or cast_total_b > EPS, 'Variant float32 denominator fallback')
    context = dict(dataset=panel['dataset'], episode=cost['episode'], policy=policy,
                   epsilon=epsilon, measured=cost['episode'] >= panel['warmup'])
    classes = []
    for index, name in enumerate(panel['class_names']):
        members = [i for i, item in enumerate(items) if panel['query_to_class'][item.class_idx] == index]
        old, new = math.fsum(a[i] for i in members), math.fsum(b[i] for i in members)
        classes.append(dict(**context, class_idx=index, class_name=name,
                            original_mass=old, variant_mass=new,
                            original_active_candidates=sum(a[i] > 0 for i in members),
                            variant_active_candidates=sum(b[i] > 0 for i in members),
                            coverage_lost=old > 0 and new == 0, coverage_gained=old == 0 and new > 0))
    coverage = {}
    for name, keys in (
        ('query_classes', [item.class_idx for item in items]),
        ('evaluation_classes', [panel['query_to_class'][item.class_idx] for item in items]),
        ('image_class_pairs', [(item.image_id, panel['query_to_class'][item.class_idx]) for item in items]),
    ):
        old = {key for key, weight in zip(keys, a) if weight > 0}
        new = {key for key, weight in zip(keys, b) if weight > 0}
        coverage.update({f'original_{name}': len(old), f'variant_{name}': len(new),
                         f'lost_{name}': len(old - new), f'gained_{name}': len(new - old)})
    queried = set(selection['queried'])
    omitted = math.fsum(value for i, value in enumerate(a) if i not in queried) if policy == 'target_first' else None
    if policy == 'target_first':
        require(not distance['empty_to_active'] and not distance['active_to_empty'],
                'Selective tracing changed objective activity')
        require(distance['tv'] <= selection['stopping_bound'] + 1e-12, 'TV certificate violated')
        require(math.isclose(distance['tv'], omitted, abs_tol=1e-12), 'Omitted mass != TV')
    record = dict(**context, images=cost['images'], candidates=len(items),
        cd_queried_candidates=len(queried), cd_rounds=len(selection['rounds']),
        skipped_candidates=len(items) - len(queried) if policy == 'target_first' else 0,
        exact_t_eps_skips=selection['exact_t_eps_skips'],
        stopping_bound=selection['stopping_bound'], unseen_sum_t=selection['unseen_sum_t'],
        observed_positive_rho=selection['observed_positive_rho'],
        omitted_original_coefficient_mass=omitted,
        float32_reference_coefficient_l1_drift=math.fsum(abs(x-y) for x, y in zip(a, cast_a)),
        float32_variant_coefficient_l1_drift=math.fsum(abs(x-y) for x, y in zip(b, cast_b)),
        **distance, **coverage,
        **counts(cost, len(items), policy, len(queried), len(selection['rounds']),
                 total_a > EPS, total_b > EPS, panel['config']['steps']))
    return record, classes, [dict(**context, **row) for row in selection['rounds']]


def summarize(records, scope):
    rows = [row for row in records if scope == 'all_episodes' or row['measured']]
    require(bool(rows), 'Empty summary scope')
    result = dict(dataset=rows[0]['dataset'], policy=rows[0]['policy'], epsilon=rows[0]['epsilon'],
                  scope=scope, episodes=len(rows))
    additive = ['images', 'candidates', 'cd_queried_candidates', 'cd_rounds', 'skipped_candidates',
                'exact_t_eps_skips', 'candidates_activated', 'candidates_deactivated',
                'empty_to_active', 'active_to_empty']
    additive += [key for key in rows[0] if key.startswith(('original_', 'variant_', 'lost_', 'gained_', 'modeled_'))]
    additive += ['view_inputs_saved', 'diagnostic_calls_saved', 'all_forward_inputs_saved', 'all_forward_calls_saved']
    for key in additive:
        result[key] = sum(row[key] for row in rows)
    for key in ('tv', 'stopping_bound', 'float32_reference_coefficient_l1_drift', 'float32_variant_coefficient_l1_drift'):
        values = [row[key] for row in rows if row[key] is not None]
        for suffix, value in (('mean', math.fsum(values) / len(values) if values else None),
                              ('p95', readers.quantile(values, .95)), ('max', max(values, default=None))):
            result[f'{key}_{suffix}'] = value
    for numerator, denominator in (
        ('view_inputs_saved', 'original_view_inputs'), ('diagnostic_calls_saved', 'original_diagnostic_calls'),
        ('all_forward_inputs_saved', 'original_all_forward_inputs'), ('all_forward_calls_saved', 'original_all_forward_calls'),
        ('lost_query_classes', 'original_query_classes'), ('lost_evaluation_classes', 'original_evaluation_classes'),
        ('lost_image_class_pairs', 'original_image_class_pairs'),
    ):
        result[f'{numerator}_fraction'] = ratio(result[numerator], result[denominator])
    result['episodes_losing_evaluation_class'] = sum(row['lost_evaluation_classes'] > 0 for row in rows)
    return result


def sealed_csv(directory, name, expected, evidence):
    path = directory / name
    require(readers.sha256(evidence.read(path)) == expected, f'Cache seal mismatch: {path}')
    return evidence.csv(path)


def load_inputs(cache_root, package_root, evidence):
    forward = cache_root / 'forward_analysis/final'
    dataset = cache_root / 'dataset_analysis'
    previous = evidence.json(forward / 'provenance.json')
    summary = evidence.json(dataset / 'summary.json')
    verification = evidence.json(dataset / 'verification.json')
    require(verification['status'] == 'passed' and summary['status'] == 'completed', 'Incomplete dataset audit')
    require(readers.sha256(evidence.read(dataset / 'summary.json')) == verification['summary_sha256'],
            'Dataset summary verification seal mismatch')
    selected = sealed_csv(forward, 'selected_source_rows.csv',
                          previous['outputs']['selected_source_rows.csv']['sha256'], evidence)
    require(selected == readers.select_panels(evidence.csv(previous['summary_source'])), 'Selected panel drift')
    panels = [readers.load_panel(row, evidence, package_root) for row in selected]
    cached = sealed_csv(forward, 'candidate_analysis.csv',
                        previous['outputs']['candidate_analysis.csv']['sha256'], evidence)
    original = [(panel['dataset'], item) for panel in panels for item in panel['candidates']]
    require(len(cached) == len(original), 'Forward cache candidate count mismatch')
    for row, (name, item) in zip(cached, original):
        require((row['dataset'], int(row['episode']), row['image_id'], int(row['patch_in_batch']),
                 int(row['class_idx']), row['mask_sha256'], int(row['area'])) ==
                (name, item.episode, item.image_id, item.patch, item.class_idx, item.mask_sha256, item.area),
                'Forward cache original candidate order/identity mismatch')
        require([float(row[key]) for key in ('T', 'C', 'D', 'q', 'rho')] ==
                [item.target, item.counterfactual, item.residual, item.quality, item.rho],
                'Forward cache support mismatch')
    allocations = sealed_csv(dataset, 'episode_allocations.csv', summary['output_sha256']['episode_allocations.csv'], evidence)
    class_allocations = sealed_csv(dataset, 'episode_class_allocations.csv',
                                   summary['output_sha256']['episode_class_allocations.csv'], evidence)
    for panel in panels:
        protocol = evidence.json(Path(panel['directory']) / 'protocol.json')
        vocabulary = summary['vocabularies'][protocol['vocabulary_sha256']]
        groups = [[part.strip() for part in line.split(',')] for line in
                  evidence.read(Path(protocol['package_root']) / 'trace_ovss/resources/class_names' /
                                readers.PANELS[panel['dataset']][1]).decode().splitlines() if line.strip()]
        classes = [group[0] for group in groups]
        classes[0] = 'background'
        mapping = [i for i, group in enumerate(groups) for _ in group] if panel['config']['class_extensions'] else None
        queries = [query for group in groups for query in group] if mapping is not None else classes
        require(readers.sha256(json.dumps([queries, mapping]).encode()) == protocol['vocabulary_sha256'],
                'Vocabulary identity mismatch')
        require(vocabulary['queries'] == queries and vocabulary['classes'] == classes and
                vocabulary['query_to_class'] == (mapping if mapping is not None else list(range(len(classes)))),
                'Dataset cache query-to-evaluation-class mapping mismatch')
        panel.update(class_names=classes, query_to_class=vocabulary['query_to_class'])
    # Recheck the shared input seals, without opening unrelated label/count files.
    matched = {'forward': 0, 'dataset': 0}
    for path, record in list(evidence.files.items()):
        if path in previous['inputs']:
            require(record['sha256'] == previous['inputs'][path]['sha256'], f'Forward provenance drift: {path}')
            matched['forward'] += 1
        if path in summary['evidence_sha256']:
            require(record['sha256'] == summary['evidence_sha256'][path], f'Dataset provenance drift: {path}')
            matched['dataset'] += 1
    require(all(matched.values()), 'No shared provenance inputs')
    return panels, allocations, class_allocations, matched


def crosscheck_q_only(panel, rows, class_rows, old_rows, old_classes):
    path = panel['source_row']['result_path']
    old_rows = [row for row in old_rows if row['result_path'] == path]
    old_classes = [row for row in old_classes if row['result_path'] == path]
    require(len(old_rows) == len(rows), 'Dataset allocation episode count mismatch')
    require(len(old_classes) == len(class_rows), 'Dataset allocation class count mismatch')
    lookup = {int(row['episode']): row for row in old_rows}
    class_lookup = {(int(row['episode']), int(row['class_idx'])): row for row in old_classes}
    require(len(lookup) == len(rows) and len(class_lookup) == len(class_rows), 'Duplicate cached allocation keys')
    for row in rows:
        old = lookup[row['episode']]
        require(int(old['candidates']) == row['candidates'] and int(old['episode_images']) == row['images'],
                'Dataset allocation cohort mismatch')
        require(math.isclose(row['tv'], float(old['dominance_only_tv']), abs_tol=1e-12), 'Cached q-only TV mismatch')
    for row in class_rows:
        old = class_lookup[row['episode'], row['class_idx']]
        require(row['class_name'] == old['class_name'], 'Class name mismatch')
        for key, old_key in (('original_mass', 'full_episode_coefficient_mass'),
                             ('variant_mass', 'quality_only_episode_coefficient_mass')):
            require(math.isclose(row[key], float(old[old_key]), abs_tol=1e-12), 'Cached class allocation mismatch')


def mathematics():
    return dict(
        production_eps=EPS, epsilons=list(EPSILONS), view_batch=VIEW_BATCH, round_candidates=ROUND_CANDIDATES,
        schedule='All T first; decreasing T, stable original episode index ties; each round queries at most 8 C,D pairs.',
        certificate='rho=min(T,C)*(1-max(D-T,0))<=T. Let S=sum observed rho>EPS and U=sum unseen T>EPS. '
                    'The omitted active rho mass R<=U, so TV=R/(S+R)<=U/(S+U). Stop only at round boundaries.',
        eps_gate='T<=EPS is provably inactive for all C,D and omitted from U. Observed rho<=EPS contributes zero to S.',
        zero='If U=0 the bound is zero, including S=0. With S=0,U>0 bound=1, so no positive unseen tail is skipped.',
        normalization='Jointly over the entire episode after EPS pruning, original candidate order. No per-image normalization.',
        q_only='Raw q=min(T,C), all original selected candidates, no D. EPS-prune then episode-normalize. '
               'One interleaved 2N-view stream, ceil(2N/16) calls; no target-first staging or claimed EPS skip savings.',
        empty_objective='Use a unit-mass zero-loss null term for empty objectives; empty-to-active TV=1, not 0.5.',
        cost='Original 3N views and ceil(3N/16) calls checked against recorded counters. '
             'Selective N+2M views and ceil(N/16)+ceil(M/8) calls. Total counts keep source/inference '
             'and configured steps fixed except objective-empty transitions. Negative call savings are retained.',
        coverage='Active candidates (rho>EPS), query-class and alias-collapsed evaluation-class sets, '
                 'plus image-class pairs. Summed episode-class counts are coverage opportunities, not unique classes or GT recall.',
        numeric='Binary64 cached normalized coefficients using fsum; separately emulate float32 casts/sequential denominator. '
                'Reject float32 denominator fallback. Not a bitwise model, loss or gradient simulation.',
        caveats=[
            'Offline cached arithmetic on four original Full panels: 512 VOC21, 512 PC60, 512 COCOObject, fresh 500 COCO-CD.',
            'No mIoU, accuracy preservation/improvement, walltime speedup, FLOPs, gradient or adapted-trajectory evidence.',
            'Fixed candidate identities/masks and cached supports. Different forward batches/precision may change actual supports.',
            'One saved seed; no inference of statistical significance. Epsilons prespecified, no K/lr/steps tuning.',
            'Class loss counts are not bounded by coefficient TV: low-mass classes can disappear.',
            'Input hashes verify the read caches, protocols, frozen source and shared prior audit inputs only. '
            'No raw image/mask/checkpoint content rehash; no labels or predictions opened.',
            'Tables include all-episode and warmup-excluded scopes; no cross-dataset timing comparison.',
        ])


def run(cache_root=DEFAULT_CACHE, output=DEFAULT_OUTPUT, package_root=None):
    package_root = Path(package_root or Path(__file__).resolve().parents[1]).resolve()
    cache_root, output = Path(cache_root).resolve(), Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), 'Output must be fresh or empty; no evidence overwrite')
    evidence = readers.Evidence()
    for path in (Path(__file__), Path(readers.__file__), package_root / 'tests/test_selective_tracing.py'):
        evidence.read(path)
    panels, old_rows, old_classes, matched = load_inputs(cache_root, package_root, evidence)
    episodes, classes, rounds, summaries, crosschecks = [], [], [], [], []
    for panel in panels:
        for policy, epsilon in [('target_first', e) for e in EPSILONS] + [('q_only', None)]:
            records, class_records = [], []
            for cost in panel['costs']:
                row, per_class, trace = episode_analysis(panel, cost, policy, epsilon)
                records.append(row)
                class_records.extend(per_class)
                rounds.extend(trace)
            if policy == 'q_only':
                crosscheck_q_only(panel, records, class_records, old_rows, old_classes)
                crosschecks.append(dict(dataset=panel['dataset'], episodes=len(records),
                                       class_rows=len(class_records), status='passed'))
            episodes.extend(records)
            classes.extend(class_records)
            summaries.extend(summarize(records, scope) for scope in ('all_episodes', 'warmup_excluded'))
    require('torch' not in sys.modules, 'Unexpected runtime import')
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=True)
    tables = {'episode_analysis.csv': episodes, 'episode_class_coverage.csv': classes,
              'query_rounds.csv': rounds, 'strategy_summary.csv': summaries, 'crosschecks.csv': crosschecks}
    for name, rows in tables.items():
        readers.write_csv(output / name, rows)
    provenance = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        output_directory=str(output), cache_root=str(cache_root), execution='CPU-only standard library; no model imports',
        normalization_unit='episode', mathematics=mathematics(), shared_input_seals_verified=matched,
        inputs=evidence.files, outputs={name: dict(sha256=readers.sha256((output / name).read_bytes()), rows=len(rows))
                                      for name, rows in tables.items()},
        panels=[dict(dataset=p['dataset'], images=p['images'], episodes=len(p['costs']),
                     candidates=len(p['candidates']), query_classes=p['queries'], evaluation_classes=len(p['class_names']),
                     result_path=p['source_row']['result_path']) for p in panels],
        production_adoption=False, accuracy_preservation_verified=False, measured_speedup=False,
        independent_agent_review='Not executed by this sidecar; parent owns review and production decisions.',
        reproduction=f'python3 {Path(__file__).resolve()} --cache-root {cache_root} --output NEW_EMPTY_DIRECTORY')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    evidence.recheck()
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run(args.cache_root, args.output, args.package_root)
    print(json.dumps({key: result[key] for key in ('output_directory', 'panels', 'outputs')}, indent=2))


if __name__ == '__main__':
    main()
