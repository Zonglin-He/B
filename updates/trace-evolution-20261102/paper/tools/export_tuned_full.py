"""Bounded stdlib-only metadata import of the pinned eight-population suite.

Data API: import_suite(root=ROOT, paper=PAPER) -> provenance dict;
build_outputs(provenance) -> {owned_filename: UTF-8 text}. CLI --write applies
only those five outputs with apply_patch. Exit 0 verified, 2 invalid, 3 pending.
Never imports the experimental runtime or reads RGB, GT, model, prediction,
episode candidate, or per-image-count payloads. Frozen queue certification
establishes payload integrity at execution time, NOT a current payload rehash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/home/wwww/hzl/results/trace_single_probe_waterbirds_full_20261029_0817')
PAPER = Path(__file__).resolve().parents[1]
PREPARED_SHA = '4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3'
SERVICE = 'trace-waterbirds-full-20261029-0817.service'
DATASETS = ('voc20', 'voc21', 'pc59', 'pc60', 'coco_object', 'coco_stuff', 'coco_cd', 'waterbirds_seg')
POPULATIONS = dict(zip(DATASETS, (1449, 1449, 5105, 5105, 5000, 5000, 500, 5322)))
CLASSES = dict(zip(DATASETS, (20, 21, 59, 60, 81, 171, 3, 3)))
LABELS = dict(zip(DATASETS, ('VOC20', 'VOC21', 'PC59', 'PC60', 'Object', 'Stuff', 'COCO-CD', 'Waterbirds')))
ARMS = ('no_adapt', 'image_uniform', 'image_confidence', 'image_C')
COORDINATES = ('batch_size', 'lr', 'steps', 'trace_max_regions')
COUNTERS = ('diagnostic_forward_calls', 'diagnostic_forward_inputs', 'eval_forward_calls',
            'eval_forward_inputs', 'adapt_forward_calls', 'adapt_forward_inputs', 'encoder_calls',
            'encoder_inputs', 'diagnostic_encoder_calls', 'diagnostic_encoder_inputs')
SCORE_FILES = {'SCORING_STARTED.json', 'result.json', 'score_inputs.json', 'scores_by_episode.jsonl'}
SMALL_PREDICTION_FILES = {'prediction_protocol.json', 'initial_state.json', 'prediction_result.json', 'cost.jsonl'}
OUTPUTS = tuple('tuned_full_' + s for s in ('comparison.tex', 'reference.tex', 'cost.tex',
                                          'configurations.tex', 'provenance.json'))
TIMING_SCOPE = ('Synchronized model-side wallclock: exact reset/state check, candidate admission, '
                'diagnostic views, optimization, final inference/reconstruction/CPU transfer and '
                'observation hooks. NoAdapt is reset plus evaluation only. Excludes setup, RGB '
                'decode/H2D/geometry, disk writes and offline scoring. Episode 0 is warmup, '
                'retained in all counts and scores.')
NATIVE_ROOT = Path('/home/wwww/hzl/results/trace_two_view_full_baselines_20261025')
MIB = 1024 * 1024


class Invalid(ValueError):
    """Evidence does not meet the pinned contract."""


class Pending(Invalid):
    """The retained service is not terminal."""


def require(condition, message):
    if not condition:
        raise Invalid(message)


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA-256')
    return value


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def integer(value, name):
    require(type(value) is int and value >= 0, 'Invalid count: ' + name)
    return value


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid number: ' + name)
    return value


def close(actual, expected, name):
    require(math.isclose(number(actual, name), expected, rel_tol=1e-11, abs_tol=1e-10), name + ' differs')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=unique_object,
                      parse_constant=lambda v: (_ for _ in ()).throw(Invalid('Nonfinite JSON: ' + v)))


class Evidence:
    """Bounded, alias-free metadata reader; all reads are hashed and recorded."""

    def __init__(self, root):
        self.root = Path(root).absolute()
        require(self.root == self.root.resolve(), 'Aliased evidence root')
        self.hashes = {}
        self.bytes_read = 0

    def path(self, relative):
        p = Path(relative)
        require(not p.is_absolute() and '..' not in p.parts and str(p) == relative, 'Unsafe metadata path')
        require(not {'predictions', 'episodes'} & set(p.parts)
                and not any('per_image' in part for part in p.parts)
                and (p.suffix in {'.json', '.jsonl', '.csv', '.md'} or p.parts[0] == 'runtime'),
                'Payload read forbidden: ' + relative)
        target = self.root / p
        require(target.resolve() == target and target.is_file(), 'Missing or aliased metadata: ' + relative)
        return target

    def record(self, relative, digest, expected):
        if expected is not None:
            require(digest == sha(expected), 'Hash differs: ' + relative)
        require(relative not in self.hashes or self.hashes[relative] == digest, 'Metadata changed during import')
        self.hashes[relative] = digest

    def raw(self, relative, expected=None, limit=8 * MIB):
        target = self.path(relative)
        require(target.stat().st_size <= limit, 'Metadata exceeds byte bound: ' + relative)
        with target.open('rb') as stream:
            data = stream.read(limit + 1)
        require(len(data) <= limit, 'Metadata grew beyond bound')
        self.bytes_read += len(data)
        require(self.bytes_read <= 512 * MIB, 'Import exceeds aggregate byte bound')
        self.record(relative, hashlib.sha256(data).hexdigest(), expected)
        return data

    def read(self, relative, expected=None, limit=8 * MIB):
        return decode(self.raw(relative, expected, limit))

    def lines(self, relative, expected, max_rows):
        target = self.path(relative)
        require(target.stat().st_size <= 64 * MIB, 'Stream exceeds byte bound')
        digest, size, rows = hashlib.sha256(), 0, 0
        with target.open('rb') as stream:
            while True:
                line = stream.readline(256 * 1024 + 1)
                if not line:
                    break
                size += len(line)
                rows += 1
                require(len(line) <= 256 * 1024 and size <= 64 * MIB and rows <= max_rows,
                        'Stream row/byte bound exceeded: ' + relative)
                digest.update(line)
                yield decode(line)
        self.bytes_read += size
        require(self.bytes_read <= 512 * MIB, 'Import exceeds aggregate byte bound')
        self.record(relative, digest.hexdigest(), expected)


def service_success(root, prepared, execution):
    fields = ('Id', 'LoadState', 'ActiveState', 'SubState', 'Result', 'MainPID', 'ExecMainPID',
              'ExecMainCode', 'ExecMainStatus', 'ExecStart', 'ExecMainStartTimestamp', 'ExecMainExitTimestamp')
    response = subprocess.run(['systemctl', '--user', 'show', SERVICE, '--property=' + ','.join(fields)],
                              capture_output=True, text=True, check=True, timeout=15)
    state = dict(line.split('=', 1) for line in response.stdout.splitlines() if '=' in line)
    require(set(fields) <= state.keys() and state['Id'] == SERVICE and state['LoadState'] == 'loaded',
            'Retained service identity unavailable')
    command = (f'argv[]={prepared["python"]} -B -m tools.evolution_tuning.continuation run '
               f'--suite {root} --suite-sha256 {PREPARED_SHA} ;')
    require(command in state['ExecStart'] and integer(execution['pid'], 'execution PID') > 0
            and state['ExecMainPID'] == str(execution['pid'])
            and execution['suite_prepared_sha256'] == PREPARED_SHA and execution['retry_allowed'] is False,
            'Service root/hash/command/PID binding differs')
    if state['MainPID'] != '0' or state['SubState'] in {'running', 'start', 'start-post'}:
        raise Pending('Suite service has not reached terminal state')
    require(state['Result'] == 'success' and state['ExecMainCode'] == '1' and state['ExecMainStatus'] == '0'
            and (state['ActiveState'], state['SubState']) in {('active', 'exited'), ('inactive', 'dead')}
            and state['ExecMainStartTimestamp'] and state['ExecMainExitTimestamp'],
            'Retained service did not exit successfully')
    return state


def verify_runtime(ev, prepared):
    inventory = prepared['runtime_hashes']
    require(len(inventory) == 161, 'Frozen source inventory count differs')
    actual = set()
    for index, p in enumerate((ev.root / 'runtime').rglob('*')):
        require(index < 2048 and not p.is_symlink(), 'Unbounded/aliased source inventory')
        if p.is_file():
            actual.add(str(p.relative_to(ev.root / 'runtime')))
    require(actual == set(inventory), 'Frozen source inventory additions/removals')
    for name, expected in inventory.items():
        ev.raw('runtime/' + name, expected)


def verify_panel(dataset, panel, expected_count, stage):
    n = integer(panel['images'], 'panel images')
    ids, rows = panel['ordered_ids'], panel['inputs']
    require(n == expected_count == len(ids) == len(rows) == len(set(ids)), 'Population count differs: ' + dataset)
    require(panel['available'] is True and panel['full_split'] is (stage == 'full') and panel['stage'] == stage
            and len(panel['classes']) == CLASSES[dataset] and panel['queries'], 'Panel scope differs')
    require(ids == [r['official_id'] for r in rows] and len({r['image'] for r in rows}) == n,
            'Panel ordered identities differ')
    for row in rows:
        sha(row['image_sha256'])
        require(Path(row['image']).is_absolute() and Path(row['label']).is_absolute()
                and row['label_metadata_details']['contents_read'] is False, 'Input metadata scope differs')
    return set(ids)


def verify_exposure(dataset, entry):
    calibration, evaluation = entry['calibration'], entry['evaluation']
    c = verify_panel(dataset, calibration, 256, 'calibration')
    e = verify_panel(dataset, evaluation, POPULATIONS[dataset], 'full')
    selection, audit = entry['selection'], entry['input_audit']
    require(selection['historical_development_exposure_retained'] is True
            and selection['global_never_seen_certified'] is False
            and selection['annotation_pixels_read'] is False and selection['size'] == 256
            and selection['seed'] == 2027, 'Historical exposure boundary differs')
    overlap = len(c & e)
    if dataset == 'waterbirds_seg':
        require(overlap == 256, 'Waterbirds calibration/full overlap differs')
        for obj in (calibration, evaluation, selection, audit):
            require(obj['protocol'] == obj['evaluation_scope'] == 'development_overlap'
                    and obj['development_overlap'] is True and obj['allow_waterbirds_development_overlap'] is True
                    and obj['cal_full_source_id_overlap'] == 256 and obj['train_full_source_id_overlap'] == 2458
                    and obj['independent_test_claim'] is False and obj['global_never_seen_certified'] is False,
                    'Waterbirds development-overlap disclosure differs')
        require(audit['historical_selected_full_overlap'] == audit['historical_selected_count'] == 512
                and evaluation['original_full_count'] == 5322, 'Waterbirds historical exposure differs')
    else:
        require(overlap == 0 and audit['own_train_eval_source_id_overlap'] == 0
                and audit['own_train_eval_path_inode_overlap'] == 0
                and audit['own_evaluation_population'] == POPULATIONS[dataset], 'Own calibration/full overlap differs')
    return dict(evaluation_scope='development_overlap' if dataset == 'waterbirds_seg' else 'full_development_exposed',
                calibration_full_source_id_overlap=overlap, independent_test_claim=False,
                selection=selection, input_audit=audit, config_provenance=entry['config_provenance'])


def verify_config(dataset, cfg, initial):
    require(set(cfg) == set(initial) and all(cfg[k] == initial[k] for k in cfg if k not in COORDINATES),
            'Non-search configuration changed: ' + dataset)
    require(1 <= integer(cfg['batch_size'], 'B') <= 32 and 1 <= integer(cfg['steps'], 'steps') <= 64
            and 1 <= integer(cfg['trace_max_regions'], 'K') <= 64 and 1e-6 <= number(cfg['lr'], 'lr') <= .04
            and cfg['trace_view_batch_size'] == 16 and cfg['resize'] == cfg['patch_size'] == [224, 224],
            'Configuration exceeds frozen bounds')


def verify_selections(ev, prepared, terminal, inputs):
    selected = ev.read('ALL_CONFIGS_SELECTED.json', terminal['selected_config_sha256'])
    parent = ev.read('PARENT_IMPORT_SEALED.json', terminal['parent_import_sha256'])
    dependency = prepared['continuation']
    reused = set(DATASETS) - {'waterbirds_seg'}
    require(selected['complete'] is True and selected['full_evaluation_used_for_selection'] is False
            and set(selected['selections']) == set(DATASETS)
            and selected['waterbirds_evaluation_scope'] == 'development_overlap'
            and parent['complete'] is True and set(parent['imported_selections']) == reused
            and parent['parent_prepared_sha256'] == dependency['parent_prepared_sha256'], 'Selection/import scope differs')
    status = parent['parent_service_status']
    require(status['MainPID'] == '0' and status['Result'] == 'success' and status['ExecMainStatus'] == '0'
            and status['LoadState'] == 'loaded' and status['SubState'] == 'exited', 'Parent terminal receipt differs')
    upstream = Evidence(dependency['parent_suite'])
    upstream_prepared = upstream.read('SUITE_PREPARED.json', dependency['parent_prepared_sha256'])
    upstream_terminal = upstream.read('CALIBRATION_SELECTIONS_SEALED.json', parent['parent_selection_seal_sha256'])
    require(upstream_terminal['complete'] is False and upstream_terminal['full_evaluation_ready'] is False
            and set(upstream_terminal['blocked']) == {'waterbirds_seg'}
            and set(upstream_terminal['selections']) == reused
            and upstream_terminal['suite_prepared_sha256'] == dependency['parent_prepared_sha256'], 'Parent selection seal differs')
    require(upstream_prepared['checkpoint'] == prepared['checkpoint'], 'Inherited checkpoint identity differs')
    configs, selections = {}, {}
    for d in DATASETS:
        name = f'selections/{d}/SELECTED_CONFIG_SEALED.json'
        record = ev.read(name, selected['selections'][d])
        require(record['dataset'] == d and record['suite_prepared_sha256'] == PREPARED_SHA
                and record['calibration_only'] is True and record['full_evaluation_used_for_selection'] is False
                and record['calibration_panel_sha256'] == canonical(inputs['datasets'][d]['calibration'])
                and 1 <= integer(record['unique_trials'], 'trials') <= 80
                and len(record['all_trials']) == record['unique_trials']
                and len({r['key'] for r in record['all_trials']}) == record['unique_trials'], 'Selected configuration scope differs')
        cfg, trial = record['config'], record['selected_trial']
        verify_config(d, cfg, prepared['initial_configs'][d])
        require(trial['dataset'] == d and trial['config'] == cfg and trial['status'] == 'completed'
                and trial['images'] == 256 and trial['full_split'] is False
                and any(t['result_path'] == trial['result_path'] and t['status'] == 'completed'
                        for t in record['all_trials']), 'Selected calibration trial binding differs')
        sha(trial['scored_seal_sha256'])
        if d in reused:
            origin = record['origin']
            require(record['selection_reused_without_search'] is True
                    and parent['imported_selections'][d] == selected['selections'][d]
                    and origin['source_suite'] == str(upstream.root)
                    and origin['source_prepared_sha256'] == dependency['parent_prepared_sha256']
                    and origin['selection_path'] == str(upstream.root / name), 'Parent import origin differs')
            original = upstream.read(name, upstream_terminal['selections'][d])
            require(origin['selection_sha256'] == upstream.hashes[name]
                    and origin['original_panel_sha256'] == original['calibration_panel_sha256'], 'Parent config hash differs')
            upstream.raw('trials.csv', origin['trial_csv_sha256'])
            expected = dict(original, suite_prepared_sha256=PREPARED_SHA,
                            calibration_panel_sha256=record['calibration_panel_sha256'],
                            selection_reused_without_search=True, origin=origin)
            require(record == expected, 'Imported selection changed beyond rebinding')
        trial_ev = upstream if d in reused else ev
        trial_path = Path(trial['result_path'])
        require(trial_path.is_relative_to(trial_ev.root / 'trials' / d)
                and trial_path.name == 'result.json', 'Selected trial path escapes calibration tree')
        trial_prefix = str(trial_path.parent.relative_to(trial_ev.root)) + '/'
        trial_seal = trial_ev.read(trial_prefix + 'SCORED_SEALED.json', trial['scored_seal_sha256'])
        trial_result = trial_ev.read(trial_prefix + 'result.json', trial_seal['artifact_sha256']['result.json'])
        require(trial_seal['complete'] is True and trial_result['status'] == 'completed'
                and trial_result['dataset'] == d and trial_result['stage'] == 'calibration'
                and trial_result['calibration'] is True and trial_result['full_split'] is False
                and trial_result['images'] == 256 and trial_result['config'] == cfg
                and set(trial_result['arm_results']) == {'image_C'}
                and trial_result['prediction_seal_sha256'] == trial_seal['prediction_seal_sha256']
                and trial_result['barrier_sha256'] == trial_seal['barrier_sha256'],
                'Selected calibration result/seal binding differs')
        close(trial['mIoU'], pooled(trial_result['arm_results']['image_C'], CLASSES[d]), 'Selected calibration mIoU')
        close(trial['ms_per_image'], 1000 * trial_result['timings']['image_C']['seconds_per_image'], 'Selected calibration timing')
        configs[d], selections[d] = cfg, record
    gate = ev.read('gates/waterbirds_seg/GATE_PASSED.json', terminal['waterbirds_gate_sha256'])
    require(gate['passed'] is True and gate['gt_contents_opened'] is False and gate['images'] == 4
            and gate['suite_prepared_sha256'] == PREPARED_SHA and gate['source_sha256'] == prepared['runtime_hashes']
            and gate['dataset'] == 'waterbirds_seg' and gate['stage'] == 'gate', 'Waterbirds gate binding differs')
    ev.raw('gates/waterbirds_seg.manifest.json', gate['manifest_sha256'])
    return configs, selections, parent, upstream


def counts(value, classes):
    i, u = value['intersection'], value['union']
    require(len(i) == len(u) == classes, 'Pooled class count differs')
    for x, y in zip(i, u):
        require(integer(x, 'intersection') <= integer(y, 'union'), 'Intersection exceeds union')
    return i, u


def pooled(value, classes):
    i, u = counts(value, classes)
    terms = [x / y for x, y in zip(i, u) if y]
    require(terms, 'Empty pooled union')
    score = 100 * math.fsum(terms) / len(terms)
    close(value['miou_unrounded'], score, 'Pooled mIoU')
    return score


def verify_score_stream(rows, dataset, cfg, result):
    n, classes, batch = POPULATIONS[dataset], CLASSES[dataset], cfg['batch_size']
    episodes = (n + batch - 1) // batch
    totals = {a: dict(intersection=[0] * classes, union=[0] * classes) for a in ARMS}
    seen = 0
    for index, row in enumerate(rows):
        require(index < episodes and type(row['episode']) is int and row['episode'] == index
                and type(row['images']) is int and row['images'] == min(batch, n - index * batch)
                and set(row['arms']) == set(ARMS), 'Score episode count/scope differs')
        seen += 1
        for arm in ARMS:
            counts(row['arms'][arm], classes)
            for key in ('intersection', 'union'):
                totals[arm][key] = [x + y for x, y in zip(totals[arm][key], row['arms'][arm][key])]
    require(seen == episodes and set(result['arm_results']) == set(ARMS), 'Incomplete score episodes/arms')
    for arm in ARMS:
        require(all(result['arm_results'][arm][k] == totals[arm][k] for k in totals[arm]), 'Pooled counts differ from episodes')
    return {a: pooled(result['arm_results'][a], classes) for a in ARMS}


def validate_cost(row, arm, cfg):
    n, candidates, active = (integer(row[k], k) for k in ('images', 'candidate_count', 'active_candidates'))
    require(n > 0 and row['arm'] == arm and integer(row['patches'], 'patches') == n
            and 0 <= active <= candidates <= n * cfg['trace_max_regions'], 'Candidate/image cost count differs')
    sha(row['candidate_sha256'])
    require(set(k for k in row if k.endswith(('_calls', '_inputs'))) == set(COUNTERS), 'Counter schema differs')
    expected = dict.fromkeys(COUNTERS, 0)
    if arm == 'no_adapt':
        require(candidates == active == integer(row['steps'], 'steps') == 0, 'NoAdapt is not reset/evaluate only')
        expected.update(eval_forward_calls=1, eval_forward_inputs=n, encoder_calls=1, encoder_inputs=n)
    else:
        steps = cfg['steps'] if active else 0
        require(integer(row['steps'], 'steps') == steps, 'Active-candidate steps differ')
        views = candidates if arm == 'image_C' else 0
        calls = (views + cfg['trace_view_batch_size'] - 1) // cfg['trace_view_batch_size']
        expected.update(diagnostic_forward_inputs=views, diagnostic_encoder_inputs=views,
            diagnostic_forward_calls=calls, diagnostic_encoder_calls=calls, eval_forward_calls=2,
            eval_forward_inputs=2 * n, adapt_forward_calls=steps, adapt_forward_inputs=steps * n,
            encoder_calls=2 + steps + calls, encoder_inputs=(2 + steps) * n + views)
    require(all(integer(row[k], k) == v for k, v in expected.items()), 'Actual cost schedule differs')
    require(number(row['seconds'], 'seconds') > 0, 'Nonpositive measured time')


def verify_cost_stream(rows, dataset, cfg, result):
    n, batch = POPULATIONS[dataset], cfg['batch_size']
    episodes = (n + batch - 1) // batch
    costs = {a: dict.fromkeys(COUNTERS, 0) for a in ARMS}
    timings = {a: dict(seconds=0., measured_images=0, measured_episodes=0) for a in ARMS}
    seen, identities = 0, set()
    for index, row in enumerate(rows):
        episode, position = divmod(index, len(ARMS))
        order = ARMS if episode % 2 == 0 else tuple(reversed(ARMS))
        arm = order[position]
        require(episode < episodes and row['episode'] == episode and row['order_position'] == position
                and row['warmup'] is (episode == 0) and row['images'] == min(batch, n - episode * batch)
                and row['dataset'] == dataset and row['stage'] == 'full' and row['seed'] == 0
                and row['calibration'] is False and row['full_split'] is True, 'Cost episode/arm/warmup scope differs')
        validate_cost(row, arm, cfg)
        if position == 0:
            identities = set()
        if arm != 'no_adapt':
            identities.add((row['candidate_sha256'], row['candidate_count']))
        if position == len(ARMS) - 1:
            require(len(identities) == 1, 'Candidate identities differ across adapted arms')
        for k in COUNTERS:
            costs[arm][k] += row[k]
        if episode:
            t = timings[arm]
            t['seconds'] += row['seconds']
            t['measured_images'] += row['images']
            t['measured_episodes'] += 1
        seen += 1
    require(seen == episodes * len(ARMS) and costs == result['costs'], 'Incomplete/canonical cost totals differ')
    require(result['timing_scope'] == TIMING_SCOPE and set(result['timings']) == set(ARMS), 'Timing scope differs')
    for a, t in timings.items():
        supplied = result['timings'][a]
        require(t['measured_images'] == supplied['measured_images'] == n - batch
                and t['measured_episodes'] == supplied['measured_episodes'] == episodes - 1
                and supplied['warmup_episode'] == 0 and supplied['accuracy_includes_warmup'] is True,
                'Timing denominator differs')
        for k, value in (('seconds', t['seconds']), ('seconds_per_image', t['seconds'] / (n - batch)),
                         ('images_per_second', (n - batch) / t['seconds'])):
            close(supplied[k], value, 'Timing ' + k)
        t['ms_per_image'] = 1000 * t['seconds'] / (n - batch)
    return timings


def verify_cell(ev, dataset, cfg, entry, prepared, barrier, terminal):
    prefix = f'full/{dataset}/'
    n, batch = POPULATIONS[dataset], cfg['batch_size']
    episodes = (n + batch - 1) // batch
    output = str(ev.root / 'full' / dataset)
    seal_sha = barrier['prediction_seal_sha256'][output]
    seal = ev.read(prefix + 'PREDICTIONS_SEALED.json', seal_sha)
    require(seal['complete'] is True and seal['gt_contents_opened'] is False and seal['images'] == n
            and seal['arms'] == list(ARMS) and seal['output'] == output
            and seal['suite_prepared_sha256'] == PREPARED_SHA, 'Prediction seal scope differs')
    artifacts = seal['artifact_sha256']
    expected = ({f'predictions/{a}/{i:06d}.npy' for a in ARMS for i in range(n)}
                | {f'episodes/{e:06d}.{a}.json' for a in ARMS for e in range(episodes)}
                | SMALL_PREDICTION_FILES)
    require(set(artifacts) == expected and all(sha(v) for v in artifacts.values()), 'Certified prediction inventory differs')
    protocol = ev.read(prefix + 'prediction_protocol.json', artifacts['prediction_protocol.json'])
    manifest = ev.read(f'full/{dataset}.manifest.json', seal['manifest_sha256'])
    selected_name = f'selections/{dataset}/SELECTED_CONFIG_SEALED.json'
    require(protocol['manifest'] == manifest and protocol['manifest_sha256'] == seal['manifest_sha256']
            and protocol['manifest_path'] == str(ev.root / f'full/{dataset}.manifest.json')
            and protocol['output'] == output and protocol['gt_contents_opened'] is False
            and protocol['timing_scope'] == TIMING_SCOPE, 'Protocol/manifest binding differs')
    require(manifest['algorithm'] == 'ImageBalancedTRACE:C' and manifest['suite_root'] == str(ev.root)
            and manifest['runtime_root'] == str(ev.root / 'runtime')
            and manifest['suite_prepared_sha256'] == PREPARED_SHA and manifest['config'] == cfg
            and manifest['checkpoint'] == prepared['checkpoint'] and manifest['panel'] == entry['evaluation']
            and manifest['source_sha256'] == prepared['runtime_hashes'] and manifest['arms'] == list(ARMS)
            and manifest['output'] == output and manifest['selected_config_path'] == str(ev.root / selected_name)
            and manifest['selected_config_sha256'] == ev.hashes[selected_name]
            and manifest['all_selections_sha256'] == ev.hashes['ALL_CONFIGS_SELECTED.json'], 'Full manifest binding differs')
    scope = dict(dataset=dataset, stage='full', seed=0, calibration=False, full_split=True)
    for obj in (seal, manifest):
        require(all(obj[k] == v for k, v in scope.items() if k in obj)
                and all(k in obj for k in ('dataset', 'stage', 'seed')), 'Full prediction scope differs')
    scored = ev.read(prefix + 'SCORED_SEALED.json', terminal['scored_seals'][dataset])
    barrier_sha = ev.hashes['ALL_FULL_PREDICTIONS_SEALED.json']
    require(scored['complete'] is True and scored['prediction_seal_sha256'] == seal_sha
            and scored['barrier_sha256'] == barrier_sha and set(scored['artifact_sha256']) == SCORE_FILES,
            'Score seal/barrier binding differs')
    result = ev.read(prefix + 'result.json', scored['artifact_sha256']['result.json'])
    started = ev.read(prefix + 'SCORING_STARTED.json', scored['artifact_sha256']['SCORING_STARTED.json'])
    score_inputs = ev.read(prefix + 'score_inputs.json', scored['artifact_sha256']['score_inputs.json'])
    for obj in (result, started):
        require(all(obj[k] == v for k, v in scope.items()) and obj['manifest_sha256'] == seal['manifest_sha256']
                and obj['prediction_seal_sha256'] == seal_sha and obj['barrier_sha256'] == barrier_sha,
                'Canonical score bindings differ')
    require(result['status'] == 'completed' and result['images'] == n and result['config'] == cfg
            and result['label_contents_first_opened_after_stage_predictions_sealed'] is True,
            'Canonical result count/status differs')
    require(score_inputs['manifest_sha256'] == seal['manifest_sha256'] and score_inputs['barrier_sha256'] == barrier_sha
            and set(score_inputs['label_sha256']) == {r['label'] for r in entry['evaluation']['inputs']}
            and all(sha(v) for v in score_inputs['label_sha256'].values()), 'Scoring input metadata differs')
    scores = verify_score_stream(ev.lines(prefix + 'scores_by_episode.jsonl',
        scored['artifact_sha256']['scores_by_episode.jsonl'], episodes), dataset, cfg, result)
    timings = verify_cost_stream(ev.lines(prefix + 'cost.jsonl', artifacts['cost.jsonl'], episodes * len(ARMS)),
                                dataset, cfg, result)
    initial = ev.read(prefix + 'initial_state.json', artifacts['initial_state.json'])
    require(set(initial) == {'trainable_sha256', 'optimizer_sha256'} and all(sha(v) for v in initial.values()),
            'Initial reset-state receipt differs')
    predicted = ev.read(prefix + 'prediction_result.json', artifacts['prediction_result.json'])
    require(predicted['status'] == 'predicted' and predicted['images'] == n and predicted['arms'] == list(ARMS)
            and predicted['config'] == cfg and all(predicted[k] == result[k] for k in ('timings', 'costs', 'timing_scope')),
            'Prediction/score aggregate metadata differs')
    return dict(dataset=dataset, images=n, config=cfg, scores=scores, timings=timings, costs=result['costs'],
                checkpoint_sha256=prepared['checkpoint']['sha256'],
                pooled_counts={a: {k: result['arm_results'][a][k] for k in ('intersection', 'union')} for a in ARMS},
                exposure=verify_exposure(dataset, entry), ordered_ids_sha256=canonical(entry['evaluation']['ordered_ids']),
                certified_payload_files=len(expected) - len(SMALL_PREDICTION_FILES),
                delta_confidence_pp=scores['image_C'] - scores['image_confidence'],
                extra_time_percent=100 * (timings['image_C']['ms_per_image'] / timings['image_confidence']['ms_per_image'] - 1))


def verify_csv(rows, cells, root):
    require(len(rows) == 32, 'CSV must contain exactly 32 rows')
    by, seen = {c['dataset']: c for c in cells}, set()
    for row in rows:
        key = (row['dataset'], row['arm'])
        require(key not in seen and key[0] in by and key[1] in ARMS, 'Duplicate/unknown CSV arm')
        seen.add(key)
        c, arm = by[key[0]], key[1]
        require(row['images'] == str(c['images']) and row['seed'] == '0' and row['full_split'] == 'True'
                and decode(row['config']) == c['config'] and row['independent_test_claim'] == 'False'
                and row['result_path'] == str(root / 'full' / c['dataset'] / 'result.json')
                and row['evaluation_scope'] == c['exposure']['evaluation_scope']
                and row['calibration_full_source_id_overlap'] == str(c['exposure']['calibration_full_source_id_overlap']),
                'CSV count/config/overlap scope differs')
        close(float(row['mIoU']), c['scores'][arm], 'CSV mIoU')
        close(float(row['ms_per_image']), c['timings'][arm]['ms_per_image'], 'CSV timing')
    require(seen == {(d, a) for d in DATASETS for a in ARMS}, 'CSV population/arm inventory differs')


def verify_references(paper, cells):
    ev = Evidence(paper)
    reported = ev.read('reported_baselines.json')
    require(reported['citation_key'] == 'noori2025mlmp' and len(reported['datasets']) == 5
            and reported['methods'] == ['No Adapt', 'TENT', 'TPT', 'WATT', 'CLIPArTT', 'MLMP'], 'Reported reference schema differs')
    for values in reported['datasets'].values():
        require(set(values) == set(reported['methods']), 'Reported method inventory differs')
        for method, (mean, std) in values.items():
            require(0 <= number(mean, 'reported mean') <= 100, 'Invalid reported mean')
            require(std is None if method == 'No Adapt' else number(std, 'reported std') >= 0,
                    'Reported dispersion differs')
    native = Evidence(NATIVE_ROOT)
    seal = native.read('SUITE_SEALED.json')
    require(seal['failures'] == 0 and seal['artifact_sha256'], 'Native local suite incomplete')
    cd = next(c for c in cells if c['dataset'] == 'coco_cd')
    methods = {}
    for method in ('noadapt', 'tent', 'tpt', 'mlmp'):
        prefix = f'coco_cd/{method}/'
        r = native.read(prefix + 'result.json', seal['artifact_sha256'][prefix + 'result.json'])
        p = native.read(prefix + 'protocol.json', seal['artifact_sha256'][prefix + 'protocol.json'])
        require(r['status'] == 'completed' and r['full_split'] is True and p['full_split'] is True
                and r['images'] == p['images'] == p['available_images'] == 500
                and r['method'] == p['method'] == method and p['seed'] == 0
                and p['labels_used_for_adaptation'] is False
                and r['protocol_sha256'] == native.hashes[prefix + 'protocol.json']
                and p['checkpoint_sha256'] == cd['checkpoint_sha256']
                and len(p['image_ids']) == len(p['label_ids']) == 500,
                'Native local reference scope differs')
        require(all(re.fullmatch('[0-9]{12}\\.jpg', Path(s).name) for s in p['image_ids'])
                and canonical(['coco:' + Path(s).stem for s in p['image_ids']]) == cd['ordered_ids_sha256'],
                'Native local source identifiers differ')
        # Historical JSON encodes exact integral counts as floats; reject fractional counts.
        v = {}
        for k in ('intersection', 'union'):
            require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 and x == int(x) for x in r[k]),
                    'Invalid native pooled count')
            v[k] = [int(x) for x in r[k]]
        methods[method] = pooled(dict(v, miou_unrounded=r['miou_unrounded']), 3)
    require(max(methods, key=methods.get) == 'tent' and f'{methods["tent"]:.2f}' == '72.20',
            'Strongest historical local reference differs')
    return dict(reported=reported, reported_verification='Supplied reported_baselines.json; no fresh PDF extraction',
                native_cd=methods, native_cd_best='tent', native_cd_root=str(NATIVE_ROOT)), ev, native


def import_suite(root=ROOT, paper=PAPER):
    """Read-only verification; return table-ready data and explicit trust boundaries."""
    ev = Evidence(root)
    require(ev.root == ROOT, 'Only the pinned suite root is admitted')
    prepared = ev.read('SUITE_PREPARED.json', PREPARED_SHA)
    execution = ev.read('EXECUTION_STARTED.json')
    service = service_success(ev.root, prepared, execution)
    require(prepared['datasets'] == list(DATASETS) and prepared['full_arms'] == list(ARMS)
            and prepared['algorithm'] == 'ImageBalancedTRACE:C' and prepared['method_changes'] is False
            and prepared['blocked'] == {} and prepared['coordinates'] == list(COORDINATES)
            and prepared['calibration_images'] == 256 and prepared['calibration_seed'] == 2027
            and prepared['model_seed'] == 0 and prepared['max_unique_trials_per_dataset'] == 80,
            'Prepared suite contract differs')
    verify_runtime(ev, prepared)
    ev.raw('frozen_contract.md', prepared['contract_sha256'])
    inputs = ev.read('inputs.json', prepared['inputs_sha256'], 32 * MIB)
    require(set(inputs['datasets']) == set(DATASETS), 'Prepared input populations differ')
    metadata = inputs['preparation_metadata']
    require(metadata['annotation_pixels_read'] is False and metadata['offline_construction_performed'] is False
            and metadata['historical_development_exposure_retained'] is True
            and metadata['global_never_seen_certified'] is False, 'Input preparation exposure differs')
    # Bind declared panel sources to the freshly checked frozen inventory. External
    # historical sources remain recorded identities; their payloads are not opened.
    for entry in inputs['datasets'].values():
        for panel_name in ('calibration', 'evaluation'):
            panel = entry[panel_name]
            for name, expected in panel['source_sha256'].items():
                actual = prepared['contract_sha256'] if name == 'docs/single-probe-tuning-contract.md' else prepared['runtime_hashes'][name]
                require(expected == actual, 'Panel source inventory hash binding differs')
            for name, record in panel['source_hashes'].items():
                sha(record['sha256'])
                p = Path(name)
                if p.is_relative_to(ev.root):
                    relative = str(p.relative_to(ev.root))
                    require(relative in ev.hashes and record['sha256'] == ev.hashes[relative], 'Frozen panel source binding differs')
    terminal = ev.read('SUITE_SEALED.json')
    state = ev.read('queue_state.json')
    require(terminal['complete'] is True and terminal['datasets'] == list(DATASETS) and terminal['arm_results'] == 32
            and terminal['suite_prepared_sha256'] == PREPARED_SHA and set(terminal['scored_seals']) == set(DATASETS)
            and terminal['waterbirds_evaluation_scope'] == 'development_overlap'
            and terminal['service_exit_certified'] is False and state['phase'] == 'completed'
            and state['pid'] == execution['pid'] and state['arm_results'] == 32
            and state['completed_datasets'] == list(DATASETS), 'Terminal result inventory differs')
    configs, selections, parent, upstream = verify_selections(ev, prepared, terminal, inputs)
    barrier = ev.read('ALL_FULL_PREDICTIONS_SEALED.json', terminal['all_full_prediction_barrier_sha256'])
    require(barrier['complete'] is True and barrier['stage'] == 'full' and barrier['gt_contents_opened'] is False
            and barrier['suite_prepared_sha256'] == PREPARED_SHA
            and set(barrier['prediction_seal_sha256']) == {str(ev.root / 'full' / d) for d in DATASETS},
            'All-population full prediction barrier differs')
    cells = [verify_cell(ev, d, configs[d], inputs['datasets'][d], prepared, barrier, terminal) for d in DATASETS]
    csv_data = ev.raw('full_metrics.csv', terminal['metrics_sha256'], MIB)
    verify_csv(list(csv.DictReader(io.StringIO(csv_data.decode()))), cells, ev.root)
    ev.raw('trials.csv', terminal['trials_sha256'], MIB)
    refs, ref_ev, native = verify_references(paper, cells)
    # A second service snapshot plus read-set check detects drift before publication.
    require(service_success(ev.root, prepared, execution) == service, 'Retained service changed during import')
    evidence = {}
    for reader in (ev, upstream, ref_ev, native):
        for relative, digest in list(reader.hashes.items()):
            target = reader.path(relative)
            h = hashlib.sha256()
            require(target.stat().st_size <= 64 * MIB, 'Metadata grew before publication')
            with target.open('rb') as stream:
                for index in range(1025):
                    block = stream.read(64 * 1024)
                    if not block:
                        break
                    h.update(block)
                else:
                    raise Invalid('Metadata grew beyond final recheck bound')
            require(h.hexdigest() == digest, 'Metadata drift before publication: ' + relative)
            evidence[str(target)] = digest
    return dict(schema_version=1, suite_root=str(ev.root), suite_prepared_sha256=PREPARED_SHA,
        service=service, execution=execution, algorithm=prepared['algorithm'], populations=POPULATIONS,
        arms=list(ARMS), arm_results=32, checkpoint_identity_only=prepared['checkpoint'],
        cells=cells, selections=selections, parent_import=parent, references=refs,
        historical_exposure=prepared['historical_exposure'], continuation=prepared['continuation'],
        preparation_metadata=metadata, source_sha256=prepared['runtime_hashes'], metadata_sha256=evidence,
        importer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing_scope=TIMING_SCOPE, payload_verification=dict(
            basis='Frozen queue runtime verifies all prediction payloads at barrier and before scoring; retained successful root/hash/PID-bound service certifies completion.',
            current_payload_rehashed=False, current_metadata_rehashed=True,
            prediction_checkpoint_rgb_gt_opened=False, per_image_counts_opened=False,
            source_inventory_currently_rehashed=True,
            excluded_current_rehash='Predictions, episode candidate receipts, checkpoint, RGB, GT, external historical source payloads; upstream search certification is inherited, not rerun.'),
        claims=dict(independent_test=False, global_never_seen=False, significance=False,
                    matched_external_comparison=False, separate_control_tuning=False))


def table(label, caption, header, rows, columns, wide=True):
    env = 'table*' if wide else 'table'
    return (f'\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n'
            '\\centering\\small\n' + f'\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\n'
            '\\toprule\n' + header + r' \\' + '\n\\midrule\n' + '\n'.join(rows)
            + '\n\\bottomrule\n\\end{tabular}\n' + f'\\end{{{env}}}\n')


def build_outputs(report):
    """Render only owned snippets; captions describe the scientific protocol."""
    cells = report['cells']
    by = {c['dataset']: c for c in cells}
    rows = [LABELS[c['dataset']] + ' & ' + f'{c["images"]:,}' + ' & '
            + ' & '.join(f'{c["scores"][a]:.2f}' for a in ARMS)
            + f' & {c["delta_confidence_pp"]:+.2f}' + r' \\' for c in cells]
    comparison = table('tab:tuned-full-comparison',
        r'Full-population mIoU (\%, seed 0). Uniform and Confidence use the TRACE-selected '
        r'configuration within each dataset, without separate tuning. All splits retain historical '
        r'development exposure; Waterbirds includes all 256 calibration sources among its 5,322 images. '
        r'$\Delta$ is TRACE minus Confidence in percentage points; Object/Stuff denote COCO Object/COCO-Stuff.',
        r'Dataset & Images & No Adapt & Uniform & Confidence & TRACE & $\Delta$', rows, 'lrrrrrr')
    rows = [LABELS[c['dataset']] + f' & {c["timings"]["image_confidence"]["ms_per_image"]:.2f}'
            + f' & {c["timings"]["image_C"]["ms_per_image"]:.2f} & {c["extra_time_percent"]:+.2f}'
            + r'\%' + r' \\' for c in cells]
    cost = table('tab:tuned-full-cost',
        r'Model-side time per image (ms), excluding the first episode as warmup. '
        r'Timing includes reset, candidate admission, diagnostic views, optimization, and final readout; '
        r'it excludes data loading, disk writes, and offline scoring. Extra time is relative to Confidence '
        r'under the same TRACE-selected configuration.',
        'Dataset & Confidence (ms) & TRACE (ms) & Extra time', rows, 'lrrr')
    rows = [LABELS[c['dataset']] + ' & ' + ' & '.join(str(c['config'][k]) for k in COORDINATES)
            + r' \\' for c in cells]
    configurations = table('tab:tuned-full-configurations',
        r'TRACE-selected configurations: episode size $B$, learning rate, update steps, and maximum '
        r'regions per image $K$. Uniform and Confidence reuse these settings without separate tuning. '
        r'Selection uses 256 calibration images per dataset; the Waterbirds calibration sources overlap '
        r'its full development population. Historical development exposure is retained.',
        r'Dataset & $B$ & Learning rate & Steps & $K$', rows, 'lrrrr')
    refs = report['references']
    published = refs['reported']
    columns = ('voc20', 'voc21', 'pc60', 'coco_object', 'coco_stuff')
    rows = []
    for method in published['methods']:
        values = []
        for d in columns:
            mean, std = published['datasets'][d][method]
            values.append(f'${mean:.2f}$' if std is None else f'${mean:.2f} \\pm {std:.2f}$')
        rows.append(method + ' (R) & ' + ' & '.join(values) + r' \\')
    rows += [r'\midrule', 'TRACE (L) & ' + ' & '.join(f'{by[d]["scores"]["image_C"]:.2f}' for d in columns) + r' \\']
    reference = table('tab:tuned-full-reference',
        r'Published references (R) and local TRACE (L), mIoU (\%). Reported means and standard '
        r'deviations follow Table~5, Original NACLIP, of \citet{noori2025mlmp}; No Adapt has no '
        r'reported dispersion. Local results use seed 0 and TRACE-selected configurations on '
        r'development-exposed populations. Published and local methods retain their respective protocols.',
        'Method & ' + ' & '.join(LABELS[d] for d in columns), rows, 'lrrrrr')
    native = refs['native_cd']
    rows = [name + f' (L, historical native) & {native[key]:.2f}' + r' \\'
            for key, name in (('noadapt', 'No Adapt'), ('tent', 'TENT'), ('tpt', 'TPT'), ('mlmp', 'MLMP'))]
    rows += [r'\midrule', f'TRACE (L, tuned) & {by["coco_cd"]["scores"]["image_C"]:.2f}' + r' \\']
    reference += table('tab:tuned-full-cd-reference',
        r'COCO-CD mIoU (\%) alongside existing local native baselines. TENT is the strongest '
        r'historical local baseline (72.20). Historical baselines and tuned TRACE retain their '
        r'respective backends, prompts, and optimization settings; the historical No Adapt uses its native readout.',
        'Method / protocol & COCO-CD', rows, 'lr')
    outputs = dict(zip(OUTPUTS[:4], (comparison, reference, cost, configurations)))
    provenance = dict(report, generated_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in outputs.items()})
    outputs[OUTPUTS[4]] = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'
    return outputs


def write_outputs(outputs, directory):
    """All artifact edits go through apply_patch, never through source paths."""
    require(set(outputs) == set(OUTPUTS), 'Output ownership differs')
    directory = Path(directory).absolute()
    require(directory == PAPER / 'generated' and directory.resolve() == directory, 'Output destination differs')
    patch = ['*** Begin Patch']
    for name, content in outputs.items():
        target = directory / name
        require(not target.is_symlink(), 'Aliased output')
        if target.exists():
            old = target.read_text()
            if old == content:
                continue
            patch += ['*** Update File: ' + str(target), '@@']
            patch += ['-' + line for line in old.splitlines()]
            patch += ['+' + line for line in content.splitlines()]
        else:
            patch += ['*** Add File: ' + str(target)]
            patch += ['+' + line for line in content.splitlines()]
    patch.append('*** End Patch')
    if len(patch) > 2:
        subprocess.run(['apply_patch'], input='\n'.join(patch) + '\n', text=True, check=True,
                       capture_output=True, timeout=30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write', action='store_true', help='Apply the five owned generated outputs after verification')
    args = parser.parse_args(argv)
    try:
        report = import_suite(args.root)
        outputs = build_outputs(report)
        if args.write:
            write_outputs(outputs, PAPER / 'generated')
        print(json.dumps(dict(status='exported' if args.write else 'verified', populations=8, arm_results=32,
                              outputs=list(outputs), metadata_files=len(report['metadata_sha256']))))
        return 0
    except Pending as error:
        print('PENDING: ' + str(error), file=sys.stderr)
        return 3
    except (Invalid, KeyError, TypeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print('INVALID: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
