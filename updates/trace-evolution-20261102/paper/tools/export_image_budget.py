"""Bounded CPU-only importer, not scoring, mode selection or a manuscript review.

Exit 0: verified/exported; 3: pending; 2: invalid. Failed checks leave outputs
untouched. Prediction arrays and candidate receipts rely on the pinned frozen
queue's final full-payload check and its root/hash/PID-bound successful service.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PAPER = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/home/wwww/hzl/results/trace_image_budget_20261029_0233')
PREPARED_SHA = '579372bf5a4208ad9e1f465698b0d82193ecd8d7c70162b3b1b3c67c84f056c5'
SERVICE = 'trace-image-budget-20261029-0233.service'
ARMS = ('episode_min', 'image_uniform', 'image_confidence', 'image_T', 'image_C', 'image_min')
DATASETS = ('pc60', 'coco_object')
ORDERS = (None, 20261030, 20261031)
OUTPUTS = ('image_budget_comparison.tex', 'image_budget_cost.tex', 'image_budget_provenance.json')
LIMIT = 16 * 1024 * 1024
EXPOSURE = ('Prospective confirmation relative only to documented exclusion inventory coverage; '
            'missing historical records are not certified absent. Not globally unseen raw/source cohorts.')
SCOPE = 'Prospective relative to documented exclusions, not exhaustive never-seen certification'
PLAN_SHA = '8569e1f9a51ee0647fe365eef3be9902b46697b017db2c4990acae1e0388ca03'
TIMING_SCOPE = ('Synchronized CUDA wallclock per independent arm: episodic reset, initial readout, '
    'candidate construction/selection, requested diagnostic views, LayerNorm optimization, final inference '
    'and reconstruction/CPU prediction transfer. Includes accounting hooks. Excludes model setup/loading, '
    'RGB decode/H2D/geometry preparation common to arms, post-run provenance checks, prediction/receipt '
    'disk writes, and offline scoring. Exclude only first complete episode (episode 0) per arm from timing '
    'aggregates; retain every prediction, count and accuracy observation including warmup.')
COUNTERS = tuple(p + '_' + s for p in ('diagnostic_forward', 'eval_forward', 'adapt_forward',
                                      'encoder', 'diagnostic_encoder') for s in ('calls', 'inputs'))


class Invalid(ValueError):
    pass


class Pending(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
        parse_constant=lambda v: (_ for _ in ()).throw(Invalid(f'Nonfinite JSON: {v}')))


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f'Invalid {name}')
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, f'Invalid integer {name}')
    return value


def close(actual, expected, name):
    require(type(actual) in (float, int) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), f'{name} differs')


def child(root, relative):
    path = root / relative
    require(not Path(relative).is_absolute() and path.absolute() == path.resolve()
            and path.resolve().is_relative_to(root.resolve()), 'Path escape/symlink')
    return path


class Evidence:
    """Read explicit metadata once, hash it, and reject drift before publication."""

    def __init__(self, root):
        self.root, self.data, self.hashes = root, {}, {}

    def raw(self, relative, expected=None):
        path = child(self.root, relative)
        if relative not in self.data:
            limit = 64 * 1024 * 1024 if relative == 'frozen_cohorts.json' else LIMIT
            require(path.stat().st_size <= limit, f'Metadata exceeds bound: {relative}')
            require(len(self.data) < 1024, 'Evidence file count exceeds bound')
            value = path.read_bytes()
            require(len(value) <= limit, 'Metadata grew past bound')
            self.data[relative] = value
            self.hashes[relative] = digest(value)
            require(sum(map(len, self.data.values())) <= 128 * 1024 * 1024, 'Evidence byte budget exceeded')
        require(expected is None or self.hashes[relative] == expected, f'Seal/hash mismatch: {relative}')
        return self.data[relative]

    def read(self, relative):
        return decode(self.raw(relative))

    def inventory(self, entries, prefix=''):
        require(isinstance(entries, dict) and 0 < len(entries) <= 512, 'Invalid artifact inventory bound')
        for name, expected in entries.items():
            require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Invalid SHA-256')
            require(not {'predictions', 'episodes'} & set(Path(name).parts)
                    and Path(name).suffix != '.npy', 'Payload scan forbidden')
            self.raw(prefix + name, expected)

    def stable(self):
        for name, expected in self.hashes.items():
            path = child(self.root, name)
            require(path.stat().st_size == len(self.data[name]), f'Source drift: {name}')
            require(digest(path.read_bytes()) == expected, f'Source drift: {name}')


def service_success(service, root, expected_hash, execution):
    fields = ('Id', 'LoadState', 'ActiveState', 'SubState', 'Result', 'MainPID',
              'ExecMainPID', 'ExecMainCode', 'ExecMainStatus', 'ExecStart')
    response = subprocess.run(['systemctl', '--user', 'show', service, '--property=' + ','.join(fields)],
                              capture_output=True, text=True, check=True, timeout=15)
    state = dict(line.split('=', 1) for line in response.stdout.splitlines() if '=' in line)
    require(set(fields) <= state.keys(), 'Service status fields unavailable')
    require(state['Id'] == service and state['LoadState'] == 'loaded', 'Service identity unavailable')
    require(f'--suite {root} ' in state['ExecStart'] and f'--suite-sha256 {expected_hash} ' in state['ExecStart']
            and '-m tools.evolution_confirmation.queue run ' in state['ExecStart'], 'Service root/command binding differs')
    require(integer(execution['pid'], 'execution PID') > 0
            and state['ExecMainPID'] == str(execution['pid']), 'Service execution PID differs')
    if state['MainPID'] != '0' or state['SubState'] in ('running', 'start', 'start-post'):
        raise Pending('Suite service has not reached terminal state')
    require(state['Result'] == 'success' and state['ExecMainCode'] == '1' and state['ExecMainStatus'] == '0'
            and (state['ActiveState'], state['SubState']) in {('active', 'exited'), ('inactive', 'dead')},
            'Service did not exit successfully')
    return state


def entries():
    return [dict(key=f'{stage}/{dataset}/{"sorted" if order is None else order}',
                 stage=stage, dataset=dataset, images=n, order_seed=order, arms=list(ARMS),
                 config=f'configs/{dataset}.json',
                 manifest=f'manifests/{dataset}.{stage}.{"sorted" if order is None else order}.json',
                 output=f'cells/{stage}/{dataset}/{"sorted" if order is None else order}')
            for stage, n in (('calibration', 128), ('confirmation', 256)) for dataset in DATASETS
            for order in ((None,) if stage == 'calibration' else ORDERS)]


def jsonlines(data, limit):
    for index, line in enumerate(data.splitlines()):
        require(index < limit and len(line) <= 256 * 1024, 'JSONL exceeds row/line bound')
        yield decode(line)


def counts(row, classes):
    i, u = row['intersection'], row['union']
    require(len(i) == len(u) == classes and all(type(a) is int and type(b) is int and 0 <= a <= b
            for a, b in zip(i, u)), 'Invalid intersection/union counts')
    return i, u


def pooled(row, classes):
    i, u = counts(row, classes)
    ratios = [a / b for a, b in zip(i, u) if b]
    require(bool(ratios), 'Null pooled mIoU')
    value = 100 * math.fsum(ratios) / len(ratios)
    close(row['miou_unrounded'], value, 'Pooled mIoU')
    return value


def cost_summary(data, manifest, result):
    n, batch, steps = manifest['max_images'], manifest['config']['batch_size'], manifest['config']['steps']
    require(type(batch) is int and 0 < batch < n and integer(steps, 'steps') > 0, 'No non-warmup timing population')
    episodes = (n + batch - 1) // batch
    totals = {a: {k: 0 for k in COUNTERS} for a in ARMS}
    timing = {a: dict(seconds=0., measured_images=0, measured_episodes=0) for a in ARMS}
    seen, candidates = set(), {}
    for row in jsonlines(data, n * len(ARMS)):
        e, a = integer(row['episode'], 'episode'), row['arm']
        require(a in ARMS and e < episodes and (e, a) not in seen, 'Cost episode/arm identity differs')
        seen.add((e, a))
        images = min(batch, n - e * batch)
        require(type(row['images']) is int and row['images'] == images and row['warmup'] is (e == 0)
                and row['steps'] == steps, 'Cost image/warmup/step scope differs')
        order = tuple(reversed(ARMS)) if e % 2 else ARMS
        require(row['order_position'] == order.index(a), 'Alternating arm order differs')
        patches = integer(row['patches'], 'patches')
        require(patches >= images, 'Missing source-image patches')
        identity = row['candidate_count'], row['candidate_sha256'], patches
        require(identity == candidates.setdefault(e, identity), 'Candidate identity differs across arms')
        candidate_count = integer(row['candidate_count'], 'candidate_count')
        require(re.fullmatch('[0-9a-f]{64}', row['candidate_sha256']), 'Invalid candidate digest')
        factor = 0 if a in ('image_uniform', 'image_confidence') else 2 if a.endswith('min') else 1
        require(set(k for k in row if k.endswith(('_calls', '_inputs'))) == set(COUNTERS), 'Counter schema differs')
        for k in COUNTERS:
            totals[a][k] += integer(row[k], k)
        require(row['diagnostic_forward_inputs'] == row['diagnostic_encoder_inputs'] == factor * candidate_count
                and row['diagnostic_forward_calls'] == row['diagnostic_encoder_calls'], 'Actual probe schedule differs')
        views = factor * candidate_count
        view_batch = manifest['config']['trace_view_batch_size']
        require(type(view_batch) is int and view_batch > 0, 'Invalid view batch')
        require(row['diagnostic_forward_calls'] == (views + view_batch - 1) // view_batch,
                'Actual diagnostic calls differ')
        for suffix in ('calls', 'inputs'):
            require(row['encoder_' + suffix] == sum(row[p + '_forward_' + suffix]
                    for p in ('eval', 'diagnostic', 'adapt')), 'Encoder accounting differs')
        require(row['eval_forward_inputs'] == 2 * patches and row['eval_forward_calls'] == 2
                and row['adapt_forward_inputs'] == steps * patches and row['adapt_forward_calls'] == steps,
                'Evaluation/adaptation schedule differs')
        seconds = number(row['seconds'], 'seconds')
        require(seconds > 0, 'Zero measured time')
        if e:
            timing[a]['seconds'] += seconds
            timing[a]['measured_images'] += images
            timing[a]['measured_episodes'] += 1
    require(seen == {(e, a) for e in range(episodes) for a in ARMS}, 'Incomplete cost stream')
    for a in ARMS:
        require(totals[a] == result['costs'][a], 'Canonical cost totals differ')
        t, supplied = timing[a], result['timings'][a]
        require(t['measured_images'] == n - batch and supplied['measured_images'] == n - batch
                and supplied['measured_episodes'] == episodes - 1 and supplied['warmup_episode'] == 0
                and supplied['accuracy_includes_warmup'] is True, 'Timing denominator differs')
        for k, expected in (('seconds', t['seconds']), ('seconds_per_image', t['seconds'] / (n - batch)),
                            ('images_per_second', (n - batch) / t['seconds'])):
            close(supplied[k], expected, 'Timing ' + k)
        totals[a].update(images=n, **t, ms_per_image=1000 * t['seconds'] / (n - batch),
                         diagnostic_inputs_per_image=totals[a]['diagnostic_encoder_inputs'] / n,
                         encoder_inputs_per_image=totals[a]['encoder_inputs'] / n)
    return totals


def runtime(value):
    expected = dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
        deterministic_warn_only=False, deterministic_debug_mode=2, cudnn_deterministic=True, cudnn_benchmark=False)
    require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in expected.items()), 'Runtime determinism drift')


def runtime_inventory(root):
    found, nodes = set(), 0
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [f for f in folders if f not in ('__pycache__', '.pytest_cache', '.git')]
        for name in (*folders, *files):
            nodes += 1
            require(nodes <= 2048, 'Runtime inventory exceeds entry bound')
            path = Path(directory) / name
            require(not path.is_symlink(), 'Runtime source symlink')
        for name in files:
            if Path(name).suffix not in ('.pyc', '.pyo'):
                found.add(str((Path(directory) / name).relative_to(root)))
    return found


def verify_cell(ev, entry, prepared, barriers, selection_link, gate, bundle):
    prefix = entry['output'] + '/'
    manifest = ev.read(entry['manifest'])
    ev.raw(entry['manifest'], entry['manifest_sha256'])
    n, stage, dataset = entry['images'], entry['stage'], entry['dataset']
    config = ev.read(entry['config'])
    parent = ev.read(f'parents/{dataset}/prediction_protocol.json')
    require(manifest['worker'] == 'image_budget_confirmation' and manifest['format_version'] == 1
            and manifest['dataset'] == dataset and manifest['stage'] == stage and manifest['max_images'] == n
            and manifest['seed'] == 0 and manifest['arms'] == list(ARMS) and manifest['arm_order'] == 'alternating'
            and manifest['order_seed'] == entry['order_seed'], 'Manifest scope differs')
    require(manifest['suite_root'] == str(ev.root) and manifest['cohorts_path'] == str(ev.root / 'frozen_cohorts.json')
            and manifest['cohort_sha256'] == prepared['cohort_sha256'], 'Manifest root/cohort differs')
    require(manifest['config'] == config == parent['config'] and manifest['config_path'] == str(ev.root / entry['config'])
            and manifest['config_sha256'] == ev.hashes[entry['config']] and manifest['checkpoint'] == parent['checkpoint'],
            'Inherited config/checkpoint differs')
    require(manifest['source_sha256'] and all(ev.hashes.get(str(Path(k).relative_to(ev.root))) == v
            for k, v in manifest['source_sha256'].items()), 'Manifest frozen-source link differs')
    require(manifest['timing_scope'] == TIMING_SCOPE and manifest['exposure_scope'] == SCOPE, 'Manifest claim scope differs')
    panel = manifest['panel']
    rows = bundle['datasets'][dataset][stage]
    expected_rows = {r['official_id']: r for r in rows}
    require(panel['full_split'] is False and len(panel['inputs']) == len(expected_rows) == n
            and panel['ordered_ids'] == [r['official_id'] for r in panel['inputs']]
            and set(panel['ordered_ids']) == set(expected_rows), 'Panel identity population differs')
    for row in panel['inputs']:
        require(all(row.get(k) == v for k, v in expected_rows[row['official_id']].items()), 'Frozen cohort row differs')
    pred = ev.read(prefix + 'PREDICTIONS_SEALED.json')
    pred_hash = ev.hashes[prefix + 'PREDICTIONS_SEALED.json']
    batch = config['batch_size']
    require(type(batch) is int and 0 < batch < n, 'Invalid batch')
    episodes = (n + batch - 1) // batch
    small = {'prediction_protocol.json', 'initial_state.json', 'prediction_result.json', 'cost.jsonl'}
    inventory = ({f'predictions/{a}/{i:06d}.npy' for a in ARMS for i in range(n)}
                 | {f'episodes/{e:06d}.{a}.json' for a in ARMS for e in range(episodes)} | small)
    require(pred['complete'] is True and pred['gt_contents_opened'] is False and pred['images'] == n
            and pred['arms'] == list(ARMS) and set(pred['artifact_sha256']) == inventory
            and all(re.fullmatch('[0-9a-f]{64}', v) for v in pred['artifact_sha256'].values()), 'Prediction seal scope differs')
    ev.inventory({k: pred['artifact_sha256'][k] for k in small}, prefix)
    protocol = ev.read(prefix + 'prediction_protocol.json')
    selected = selection_link if stage == 'confirmation' else None
    require(protocol['manifest'] == manifest and protocol['manifest_path'] == str(ev.root / entry['manifest'])
            and protocol['manifest_sha256'] == entry['manifest_sha256'] and protocol['selection'] == selected,
            'Prediction manifest/selection link differs')
    runtime(protocol['deterministic_runtime'])
    require(protocol['deterministic_runtime'] == gate['deterministic_runtime']
            and protocol['gpu_lock']['exclusive'] is True and protocol['gpu_lock']['pid'] > 0
            and all(protocol['gpu_lock'][k] == gate['gpu_lock'][k] for k in ('path', 'inode', 'exclusive')),
            'Gate/runtime/lock binding differs')
    score = ev.read(prefix + 'SCORED_SEALED.json')
    require(score['prediction_seal_sha256'] == pred_hash and set(score['artifact_sha256'])
            == {'SCORING_STARTED.json', 'scores_by_episode.jsonl', 'score_inputs.json', 'result.json'}, 'Score seal chain differs')
    ev.inventory(score['artifact_sha256'], prefix)
    started = ev.read(prefix + 'SCORING_STARTED.json')
    require(started == dict(prediction_seal_sha256=pred_hash, stage_barrier=barriers[stage], selection=selected),
            'Scoring stage barrier/selection link differs')
    result, predicted = ev.read(prefix + 'result.json'), ev.read(prefix + 'prediction_result.json')
    require(result['status'] == 'completed' and result['dataset'] == dataset and result['stage'] == stage
            and result['order_seed'] == entry['order_seed'] and result['images'] == n and result['full_split'] is False
            and result['prediction_seal_sha256'] == pred_hash and result['cohort_sha256'] == prepared['cohort_sha256']
            and result['label_contents_first_opened_after_all_stage_predictions_sealed'] is True
            and result['selection'] == selected and result['exposure_scope'] == SCOPE
            and set(result['arm_results']) == set(result['costs']) == set(result['timings']) == set(ARMS), 'Canonical scope differs')
    require(predicted['status'] == 'predicted' and predicted['images'] == n and predicted['full_split'] is False
            and all(result[k] == predicted[k] for k in ('costs', 'timings', 'timing_scope'))
            and result['timing_scope'] == TIMING_SCOPE, 'Canonical/predicted cost/time differs')
    classes = len(panel['classes'])
    require(0 < classes <= 4096, 'Class bound differs')
    sums = {a: [[0] * classes, [0] * classes] for a in ARMS}
    records = 0
    for e, row in enumerate(jsonlines(ev.raw(prefix + 'scores_by_episode.jsonl'), n)):
        require(row['episode'] == e and set(row['arms']) == set(ARMS), 'Score episode coverage differs')
        for a in ARMS:
            i, u = counts(row['arms'][a], classes)
            sums[a] = [[x + y for x, y in zip(old, new)] for old, new in zip(sums[a], (i, u))]
        records += 1
    require(records == episodes, 'Missing scored episodes')
    scores = {}
    for a in ARMS:
        require([result['arm_results'][a]['intersection'], result['arm_results'][a]['union']] == sums[a], 'Pooled episode counts differ')
        scores[a] = pooled(result['arm_results'][a], classes)
    labels = ev.read(prefix + 'score_inputs.json')['label_sha256']
    require(set(labels) == {r['label'] for r in panel['inputs']} and len(labels) == n
            and all(re.fullmatch('[0-9a-f]{64}', h) for h in labels.values()), 'Offline label population differs')
    return dict(key=entry['key'], dataset=dataset, stage=stage, order_seed=entry['order_seed'], images=n,
                scores=scores, margins={a: scores[a] - scores['image_confidence'] for a in ARMS},
                classes=panel['classes'], pooled_counts={a: dict(intersection=sums[a][0], union=sums[a][1]) for a in ARMS},
                timings=result['timings'], costs=cost_summary(ev.raw(prefix + 'cost.jsonl'), manifest, result),
                config=config, manifest_sha256=entry['manifest_sha256'], config_sha256=manifest['config_sha256'],
                prediction_seal_sha256=pred_hash, score_seal_sha256=ev.hashes[prefix + 'SCORED_SEALED.json'])


def verify_selection(selection, cells):
    """Audit recorded candidate statistics only; never choose or switch a mode."""
    require(selection['complete'] is True and selection['selected_mode'] == 'C', 'Frozen global C selection differs')
    calibration = {c['dataset']: c for c in cells if c['stage'] == 'calibration'}
    require(set(calibration) == set(DATASETS) and set(selection['candidates']) == {'T', 'C', 'min'}, 'Calibration scope differs')
    for mode in ('T', 'C', 'min'):
        record = selection['candidates'][mode]
        margins = {d: calibration[d]['margins']['image_' + mode] for d in DATASETS}
        require(set(record['margin_pp']) == set(DATASETS), 'Calibration margins missing')
        for d, value in margins.items():
            close(record['margin_pp'][d], value, 'Calibration margin')
        close(record['worst_dataset_margin_pp'], min(margins.values()), 'Calibration worst margin')
        close(record['macro_margin_pp'], math.fsum(margins.values()) / 2, 'Calibration macro margin')
        require(record['diagnostic_encoder_inputs'] == sum(calibration[d]['costs']['image_' + mode]
                ['diagnostic_encoder_inputs'] for d in DATASETS), 'Calibration cost link differs')


def verify_csv(data, cells, selection_sha):
    require(len(data) <= 1024 * 1024, 'CSV exceeds bound')
    table = csv.DictReader(io.StringIO(data.decode()))
    columns = {'dataset', 'stage', 'order_seed', 'arm', 'images', 'full_split', 'miou_unrounded',
        'margin_vs_image_confidence_pp', 'diagnostic_encoder_inputs', 'selected_global_mode', 'is_selected_probe',
        'manifest_sha256', 'config_sha256', 'selection_sha256', 'prediction_seal_sha256', 'score_seal_sha256',
        'measured_images', 'seconds', 'seconds_per_image', 'images_per_second', 'exposure_scope'}
    require(len(table.fieldnames or []) == len(columns) and set(table.fieldnames) == columns, 'CSV columns differ')
    rows = list(table)
    require(len(rows) == 48, 'Expected exactly 48 CSV rows')
    expected = {(c['dataset'], c['stage'], '' if c['order_seed'] is None else str(c['order_seed']), a): c
                for c in cells for a in ARMS}
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ('dataset', 'stage', 'order_seed', 'arm'))
        require(key in expected and key not in seen and set(row) == columns, 'CSV duplicate/foreign cell-arm link')
        seen.add(key)
        c, arm = expected[key], key[-1]
        fixed = dict(images=str(c['images']), full_split='False', selected_global_mode='C',
            is_selected_probe=str(arm == 'image_C'), selection_sha256=selection_sha, exposure_scope=EXPOSURE,
            diagnostic_encoder_inputs=str(c['costs'][arm]['diagnostic_encoder_inputs']),
            measured_images=str(c['timings'][arm]['measured_images']))
        fixed.update({k: c[k] for k in ('manifest_sha256', 'config_sha256', 'prediction_seal_sha256', 'score_seal_sha256')})
        require(all(row[k] == v for k, v in fixed.items()), 'CSV provenance/count/selection link differs')
        for k, value in (('miou_unrounded', c['scores'][arm]), ('margin_vs_image_confidence_pp', c['margins'][arm]),
                         *((k, c['timings'][arm][k]) for k in ('seconds', 'seconds_per_image', 'images_per_second'))):
            close(float(row[k]), value, 'CSV ' + k)
    require(seen == set(expected), 'CSV population incomplete')


def render(cells):
    labels = ('Old min', 'Image uniform', 'Image confidence', 'Image T', 'Image C', 'Image min')
    scope = ('Global C was sealed from calibration before confirmation; no reselection. '
        'Prospective training-source cohorts under partial documented history only; July candidate pools '
        'remain uncertified. Not independent-unseen or full-split evidence. All orders are dependent '
        'reorderings, not confidence intervals or significance evidence. ')
    def table(caption, heading, rows, columns):
        return '\n'.join([r'\begin{table*}[t]', r'\caption{' + caption + '}', r'\centering\scriptsize',
            r'\begin{tabular}{@{}lll' + 'r' * columns + '@{}}', r'\toprule', heading + r' \\', r'\midrule',
            *rows, r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    def prefix(c):
        return ('PC60' if c['dataset'] == 'pc60' else 'Object') + ' & ' + c['stage'].capitalize() + ' & ' + (
            'Sorted' if c['order_seed'] is None else str(c['order_seed']))
    heading = 'Dataset & Stage & Order & ' + ' & '.join(labels)
    comparison = table(scope + 'Pooled mIoU includes warmup. Calibration128 is separate from '
        'confirmation256; all six arms and three confirmation orders are retained.', heading,
        [prefix(c) + ' & ' + ' & '.join(f'{c["scores"][a]:.6f}' for a in ARMS) + r' \\' for c in cells], 6)
    comparison += table('Signed differences versus image confidence in percentage points. '
        'Retain negative, zero and positive differences; no post-confirmation mode choice.', heading,
        [prefix(c) + ' & ' + ' & '.join(f'{c["margins"][a]:+.6f}' for a in ARMS) + r' \\' for c in cells], 6)
    timing_rows, work_rows = [], []
    for c in cells:
        timing_rows.append(prefix(c) + f' & {c["costs"]["image_C"]["measured_images"]}' + ' & ' + ' & '.join(
            f'{c["costs"][a]["ms_per_image"]:.3f}' for a in ARMS) + r' \\')
        work_rows.append(prefix(c) + f' & {c["images"]}' + ' & ' + ' & '.join(
            f'{c["costs"][a]["diagnostic_inputs_per_image"]:.3f} / '
            f'{c["costs"][a]["encoder_inputs_per_image"]:.3f}' for a in ARMS) + r' \\')
    cost_heading = 'Dataset & Stage & Order & $N$ & ' + ' & '.join(labels)
    cost = table(scope + 'Synchronized model-side ms/image; $N$ excludes only episode 0. '
        'Includes reset, candidates, requested views, adaptation, final inference and CPU prediction transfer. '
        'Excludes model loading, RGB decode, H2D/geometry preparation, disk writes and scoring; '
        'not end-to-end throughput.', cost_heading, timing_rows, 7)
    cost += table('Actual diagnostic encoder inputs / total encoder inputs per source image. '
        '$N$ includes all images including warmup, unlike timing. Calls and all totals are retained in provenance.',
        cost_heading, work_rows, 7)
    return dict(zip(OUTPUTS[:2], (comparison, cost)))


def publish(paper, outputs):
    directory = paper / 'generated'
    require(directory.is_dir() and set(outputs) == set(OUTPUTS), 'Wrong publication directory/output set')
    old = {n: child(directory, n).read_bytes() if (directory / n).exists() else None for n in outputs}
    replaced = []
    with tempfile.TemporaryDirectory(prefix='.image-budget-', dir=directory) as temp:
        stage = Path(temp)
        for n, content in outputs.items():
            (stage / n).write_text(content)
        try:
            for n in outputs:
                os.replace(stage / n, directory / n)
                replaced.append(n)
        except BaseException:
            for n in reversed(replaced):
                if old[n] is None:
                    (directory / n).unlink()
                else:
                    (stage / n).write_bytes(old[n])
                    os.replace(stage / n, directory / n)
            raise


def verify_suite(ev, expected_hash, service):
    ev.raw('SUITE_PREPARED.json', expected_hash)
    prepared = ev.read('SUITE_PREPARED.json')
    require(prepared['worker'] == 'confirmation_queue' and prepared['format_version'] == 1
            and prepared['runtime_root'] == str(ev.root / 'runtime') and prepared['arms'] == list(ARMS), 'Prepared scope differs')
    expected = entries()
    require(len(prepared['cells']) == 8 and all(set(e) == set(x) | {'manifest_sha256'}
            and all(e[k] == v for k, v in x.items()) for e, x in zip(prepared['cells'], expected)), 'Eight-cell admission differs')
    ev.inventory(prepared['runtime_hashes'], 'runtime/')
    require(runtime_inventory(ev.root / 'runtime') == set(prepared['runtime_hashes']), 'Runtime inventory differs')
    ev.inventory(prepared['artifact_sha256'])
    ev.raw('frozen_plan.json', prepared['plan_sha256'])
    require(prepared['plan_sha256'] == PLAN_SHA and prepared['exposure_note'] == EXPOSURE, 'Plan/exposure freeze differs')
    ev.raw('frozen_cohorts.json', prepared['cohort_sha256'])
    bundle = ev.read('frozen_cohorts.json')
    audit = bundle['audit']
    require(all(audit[k] is False for k in ('global_never_seen_certified', 'historical_index_exhaustive',
                'strong_independent_confirmation_supported')), 'Historical uncertified boundary differs')
    history = audit['historical_pool_audit']
    require(history['status'] == 'unresolved' and history['candidate_pool_ids_reconstructed'] is False
            and history['strong_independent_confirmation_supported'] is False
            and history['pc60_unused_capacity']['supports_identifiable_384_image_unused_cohort'] is False,
            'July history uncertified boundary differs')
    for d in DATASETS:
        arrays = {s: bundle['datasets'][d][s] for s in ('calibration', 'confirmation')}
        require(not ({r['official_id'] for r in arrays['calibration']} & {r['official_id'] for r in arrays['confirmation']}),
                'Calibration/confirmation overlap')
        for s, rows in arrays.items():
            require(canonical([r['official_id'] for r in rows]) == bundle['datasets'][d]['selected_ids_sha256'][s],
                    'Cohort ID fingerprint differs')
        parent = ev.read(f'parents/{d}/prediction_protocol.json')
        seal = ev.read(f'parents/{d}/PREDICTIONS_SEALED.json')
        require(seal['complete'] is True and seal['gt_contents_opened'] is False
                and seal['artifact_sha256']['prediction_protocol.json'] == ev.hashes[f'parents/{d}/prediction_protocol.json']
                and parent['checkpoint'] == ev.read(f'parents/{d}/checkpoint_identity.json'), 'Parent seal/checkpoint link differs')
    if not child(ev.root, 'SUITE_SEALED.json').exists():
        state = ev.read('queue_state.json')
        require(state.get('phase') != 'failed', 'Queue failed')
        raise Pending('Whole-suite seal absent; outputs untouched')
    seal = ev.read('SUITE_SEALED.json')
    scope = dict(complete=True, calibration_cells=2, confirmation_cells=6, formal_cells=8, arm_results=48,
                 passed_gt_free_gates=2, full_split=False, selected_mode='C', suite_prepared_sha256=expected_hash,
                 cohort_sha256=prepared['cohort_sha256'], runtime_hashes=prepared['runtime_hashes'], exposure_note=EXPOSURE)
    require(all(seal[k] == v and type(seal[k]) is type(v) for k, v in scope.items()), 'Whole-suite seal scope differs')
    required = {'SUITE_PREPARED.json', 'EXECUTION_STARTED.json', 'SELECTION_SEALED.json', 'execution_children.json',
                'queue_state.json', 'rows.csv', 'ALL_CALIBRATION_PREDICTIONS_SEALED.json',
                'ALL_CONFIRMATION_PREDICTIONS_SEALED.json', *prepared['artifact_sha256']}
    required.update(f'gates/{d}.json' for d in DATASETS)
    required.update(e['output'] + '/' + n for e in prepared['cells']
                    for n in ('PREDICTIONS_SEALED.json', 'SCORED_SEALED.json', 'result.json'))
    require(required <= seal['artifact_sha256'].keys(), 'Whole-suite canonical coverage missing')
    ev.inventory(seal['artifact_sha256'])
    execution = ev.read('EXECUTION_STARTED.json')
    require(execution['suite_prepared_sha256'] == expected_hash, 'Execution prepared link differs')
    observed = service_success(service, ev.root, expected_hash, execution)
    state = ev.read('queue_state.json')
    expected_state = dict(phase='completed', active_cell=None, error=None, passed_gates=2, sealed_predictions=8,
        scored_cells=8, calibration_cells=2, confirmation_cells=6, calibration_predictions=2,
        confirmation_predictions=6, calibration_scores=2, confirmation_scores=6, total_cells=8)
    require(all(state[k] == v for k, v in expected_state.items()), 'Queue not fully complete')
    selection = ev.read('SELECTION_SEALED.json')
    selection_sha = ev.hashes['SELECTION_SEALED.json']
    require(seal['selection_sha256'] == selection_sha and selection['suite_prepared_sha256'] == expected_hash
            and selection['cohort_sha256'] == prepared['cohort_sha256']
            and selection['rule'] == ev.read('frozen_plan.json')['selection'], 'Selection frozen link differs')
    barriers = {}
    for stage in ('calibration', 'confirmation'):
        filename = f'ALL_{stage.upper()}_PREDICTIONS_SEALED.json'
        barrier = ev.read(filename)
        predictions = {e['output']: ev.hashes[e['output'] + '/PREDICTIONS_SEALED.json']
                       for e in prepared['cells'] if e['stage'] == stage}
        require(barrier == dict(complete=True, stage=stage, suite_prepared_sha256=expected_hash,
                prediction_seal_sha256=predictions, gt_contents_opened=False), 'Stage barrier population/link differs')
        barriers[stage] = barrier
    require(selection['calibration_barrier_sha256'] == ev.hashes['ALL_CALIBRATION_PREDICTIONS_SEALED.json']
            and selection['calibration_score_sha256'] == {e['output']: ev.hashes[e['output'] + '/SCORED_SEALED.json']
                for e in prepared['cells'] if e['stage'] == 'calibration'}, 'Selection calibration-only link differs')
    gates = {}
    for e in prepared['cells'][:2]:
        d = e['dataset']
        g = ev.read(f'gates/{d}.json')
        require(g['passed'] is True and g['gt_contents_opened'] is False and g['images'] == 4
                and g['dataset'] == d and g['manifest_sha256'] == e['manifest_sha256'], 'GT-free gate scope differs')
        ev.raw(f'gates/{d}.json.STARTED.json', g['attempt_sha256'])
        require(ev.read(f'gates/{d}.json.STARTED.json') == dict(status='started', manifest_sha256=e['manifest_sha256']),
                'Gate attempt link differs')
        runtime(g['deterministic_runtime'])
        require(len(g['comparisons']) == 4 and all(r['candidate_masks_equal'] is True
            and number(r['max_support_abs_difference'], 'support') <= 1e-5
            and number(r['max_loss_abs_difference'], 'loss') <= 1e-5 and r['changed_label_pixels'] == 0
            for r in g['comparisons']), 'Single-image equivalence gate differs')
        manifest = ev.read(e['manifest'])
        require(g['checkpoint'] == manifest['checkpoint'] and g['source_sha256'] == manifest['source_sha256'],
                'Gate source/checkpoint differs')
        for comparison in g['comparisons']:
            counters = comparison['actual_single_zero_view_counts']
            require(set(counters) == {'image_T', 'image_C', 'image_uniform', 'image_confidence'}, 'Gate arm schedule differs')
            candidates = integer(comparison['candidates'], 'gate candidates')
            for arm, work in counters.items():
                require(set(work) == set(COUNTERS), 'Gate counter schema differs')
                for k in COUNTERS:
                    integer(work[k], 'gate ' + k)
                views = candidates if arm in ('image_T', 'image_C') else 0
                require(work['diagnostic_encoder_inputs'] == work['diagnostic_forward_inputs'] == views
                        and work['diagnostic_encoder_calls'] == work['diagnostic_forward_calls'], 'Gate actual probe count differs')
                for suffix in ('calls', 'inputs'):
                    require(work['encoder_' + suffix] == sum(work[p + '_forward_' + suffix]
                        for p in ('eval', 'diagnostic', 'adapt')), 'Gate encoder accounting differs')
        gates[d] = g
    receipts = ev.read('execution_children.json')
    expected_commands = []
    for e in prepared['cells'][:2]:
        expected_commands.append(('gate', e))
    for stage in ('calibration', 'confirmation'):
        for action in ('predict', 'score'):
            expected_commands.extend((action, e) for e in prepared['cells'] if e['stage'] == stage)
    require(len(receipts) == len(expected_commands) == 18, 'Execution child count differs')
    for receipt, (action, e) in zip(receipts, expected_commands):
        argv = [prepared['python'], '-B', '-m', 'tools.evolution_confirmation.run', action]
        if action != 'score':
            argv += ['--manifest', str(ev.root / e['manifest']), '--manifest-sha256', e['manifest_sha256']]
        argv += ['--output', str(ev.root / (f'gates/{e["dataset"]}.json' if action == 'gate' else e['output']))]
        require(receipt['argv'] == argv and receipt['exit_code'] == 0 and receipt['error'] is None
                and receipt['cpu_only'] is (action == 'score'), 'Child command/stage/exit differs')
        ev.raw(receipt['log'], receipt['log_sha256'])
        ev.raw(receipt['receipt'], receipt['receipt_sha256'])
        require(ev.read(receipt['receipt']) == {k: v for k, v in receipt.items() if k not in ('receipt', 'receipt_sha256')},
                'Child receipt link differs')
    selected = dict(path=str(ev.root / 'SELECTION_SEALED.json'), sha256=selection_sha, selected_mode='C')
    cells = [verify_cell(ev, e, prepared, barriers, selected, gates[e['dataset']], bundle) for e in prepared['cells']]
    verify_selection(selection, cells)
    verify_csv(ev.raw('rows.csv'), cells, selection_sha)
    return prepared, observed, execution, selection, barriers, cells


def run(root=DEFAULT_ROOT, paper=PAPER, expected_hash=PREPARED_SHA, service=SERVICE, check_only=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', "Set CUDA_VISIBLE_DEVICES='' for CPU-only import")
    root, paper = Path(root).resolve(), Path(paper).resolve()
    ev = Evidence(root)
    prepared, observed, execution, selection, barriers, cells = verify_suite(ev, expected_hash, service)
    outputs = render(cells)
    summary = {d: {a: dict(by_order={('sorted' if c['order_seed'] is None else str(c['order_seed'])): c['margins'][a]
                    for c in cells if c['dataset'] == d and c['stage'] == 'confirmation'},
                    minimum=min(c['margins'][a] for c in cells if c['dataset'] == d and c['stage'] == 'confirmation'),
                    maximum=max(c['margins'][a] for c in cells if c['dataset'] == d and c['stage'] == 'confirmation'))
                for a in ARMS} for d in DATASETS}
    contrasts = {c['key']: dict(C_minus_old_min_pp=c['scores']['image_C'] - c['scores']['episode_min'],
                               T_minus_C_pp=c['scores']['image_T'] - c['scores']['image_C']) for c in cells}
    provenance = dict(schema=1, suite_root=str(root), suite_prepared_sha256=expected_hash,
        suite_sealed_sha256=ev.hashes['SUITE_SEALED.json'], service=observed, cells=cells,
        selection=selection, barriers=barriers, confirmation_margin_summary=summary, additional_contrasts=contrasts,
        exposure=EXPOSURE, july_history_certified=False, independent_unseen_certified=False,
        verification_scope='Independently rehashed frozen runtime and explicit bounded artifacts, cohort metadata '
            '(64 MiB cap), manifests, seals, barriers, selection links and 18 execution receipts; recomputed 8 '
            'canonical pooled episode-count results, 48 CSV links, all counters and warmup timing denominators. '
            'No frozen worker executed; no model library imported. Prediction arrays, per-episode candidate '
            'receipts, RGB/label contents and checkpoint weights NOT reread. Payload integrity, original '
            'checkpoint/input checks and temporal GT isolation rely on the pinned frozen queue final '
            'validate_prediction/validate_score full-payload verification before SUITE_SEALED, plus real '
            'root/hash/PID-bound successful terminal service. This is not a new current-array rehash certificate.',
        frozen_payload_check_sources=['runtime/tools/evolution_confirmation/queue.py: run final validation loop',
            'runtime/tools/evolution_confirmation/run.py: verify_prediction_seal full artifact loop'],
        uncertainty='Descriptive only; dependent same-image reorderings, not independent replicates, CI or '
            'significance evidence. Calibration reported separately; global C already selected, not selected '
            'anew. No full-split, external-method superiority, independent-unseen or July-history certification.',
        source_sha256={str(root / k): v for k, v in ev.hashes.items()},
        importer_sha256=digest(Path(__file__).read_bytes()), runtime_hashes=prepared['runtime_hashes'],
        output_sha256={n: digest(v.encode()) for n, v in outputs.items()})
    ev.stable()
    require(runtime_inventory(root / 'runtime') == set(prepared['runtime_hashes']), 'Runtime inventory drift')
    service_success(service, root, expected_hash, execution)
    outputs[OUTPUTS[2]] = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if not check_only:
        publish(paper, outputs)
    return dict(status='ready' if check_only else 'imported', tables_unchanged=check_only,
        outputs=list(OUTPUTS), cells=8, arm_results=48, selected_mode='C', confirmation_margins=summary), 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--paper-root', type=Path, default=PAPER)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        status, code = run(args.suite_root, args.paper_root, check_only=args.check_only)
    except Pending as error:
        status, code = dict(status='pending', reason=str(error), tables_unchanged=True), 3
    except (Invalid, OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        status, code = dict(status='invalid', error=str(error), tables_unchanged=True), 2
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
