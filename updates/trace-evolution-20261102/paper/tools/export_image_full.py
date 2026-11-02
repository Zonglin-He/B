"""CPU metadata importer for the pinned full-split C suite, not a scorer.

Exit 0: verified/exported; 3: pending; 2: invalid. No outputs before every
check passes. Arrays/candidate payloads rely on frozen queue certification
and the real root/hash/PID-bound successful terminal systemd service.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

try:
    from . import export_image_budget as base, export_evolution as reference
except ImportError:
    import export_image_budget as base
    import export_evolution as reference

Invalid, Pending = base.Invalid, base.Pending
require, digest, canonical = base.require, base.digest, base.canonical
integer, number, close = base.integer, base.number, base.close
child, jsonlines, pooled = base.child, base.jsonlines, base.pooled
PAPER = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/home/wwww/hzl/results/trace_image_full_20261029_0333')
PREPARED_SHA = 'e0f7374042eadb65fbc8146de6dfcf35a71151dc1c95c115b0d4cf9c8444edaa'
SERVICE = 'trace-image-full-20261029-0333.service'
PLAN_SHA = '598bc2715e959fc03421f28748ffe706522ec5c7757ca9fedb03aee959952dd8'
SELECTION_SHA = '90fa3a20097698e9b83cafd416a4b05fe7f033dcd8aadb027c970a9788ac11e4'
SELECTION_PREPARED_SHA = base.PREPARED_SHA
DATASETS = ('pc60', 'coco_object', 'voc21', 'voc20', 'coco_cd', 'coco_stuff')
POPULATIONS = dict(zip(DATASETS, (5105, 5000, 1449, 1449, 500, 5000)))
ARMS = ('image_uniform', 'image_confidence', 'image_C')
COUNTERS, TIMING_SCOPE = base.COUNTERS, base.TIMING_SCOPE
EXPOSURE = ('Development-exposed official full splits and inherited per-dataset configurations; '
            'fixed C, not untouched or equal-tuning confirmation.')
PREUPDATE = 'unavailable: no additional encoder forward and no independently gated reconstruction of initial logits'
OUTPUTS = ('image_full_comparison.tex', 'image_full_cost.tex', 'image_full_reference.tex',
           'image_full_provenance.json')
LABELS = dict(zip(DATASETS, ('PC60', 'Object', 'VOC21', 'VOC20', 'COCO-CD', 'Stuff')))
ARM_LABELS = ('Uniform', 'Confidence', 'TRACE(C)')
SMALL = {'prediction_protocol.json', 'initial_state.json', 'prediction_result.json', 'cost.jsonl'}
SCORING = {'SCORING_STARTED.json', 'scores_by_episode.jsonl', 'score_inputs.json', 'result.json'}
INHERITED_TREES = ('trace_ovss/', 'tools/evolution/', 'tools/evolution_confirmation/', 'tools/evolution_followup/')
INHERITED_FILES = ('tools/suite_inputs.py', 'tools/trace_algorithm_variants.py', 'tools/trace_execution_audit.py')
PARENT_ROOT = Path('/home/wwww/hzl/results/trace_self_evolution_20261028_1338')
VOC20_ROOT = Path('/home/wwww/hzl/results/trace_evolution_voc20_cli_recovery_20261028_2312')
PARENT_HASHES = {
    'pc60': ('9a7b1cd20e72c04d6e44db4f6a2f1f6520bcc11ceaeb93035cdeaf961802ef5e', '0b3b9c7dcf128e1229e281b1f9b54627244d7abf3311678203e31b9703c491ac'),
    'coco_object': ('397972602aec47b40d0f705a770583d5c2f45e0e2ff16eb1383c645fc8f4d743', '63e7b5e08963ee82a1a5f519cdf658b07509c68a5baef64f54f3c936e8d74e1e'),
    'voc21': ('9dcfb9d42a8a9ee463c4a6c2f4be805d2e44e6454deaa3f3aa53b7ca4d99afb0', '2f3155446fc285cce4e963e52c41611bae97f3eacdfdc32ee38c8ca2f053c0a1'),
    'voc20': ('709b0502e156e908e52069896892320a2925bb318cacc9e4c765eb3ce232b025', 'e87fe99865d9d07478da842659c6e899a44809ba898daa76f264f820aecd480f'),
    'coco_cd': ('1b08b148344f139dc31e7b7285ab556e12e40fd3de3c1ae468da79f570b25b35', '9fe16315093b2af05744e4008ddb495fa7318ead34ef44bfbbf9384dfff22c34'),
    'coco_stuff': ('f3d82494be65465f44ce49645b8659bfeb601ea599cd4418277e3093265c12ec', '43c16da245e60a58a7ed9dcf3feb7483f284f1410da93d0a2f7d3cea08407842'),
}


def sha256(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA-256')
    return value


class Evidence(base.Evidence):
    def raw(self, relative, expected=None):
        # No importer call may open prediction/candidate payloads, even via a seal.
        require(not {'predictions', 'episodes'} & set(Path(relative).parts)
                and Path(relative).suffix != '.npy', 'Payload scan forbidden')
        return super().raw(relative, expected)


def entries():
    return [dict(key=f'full/{d}/seed0', dataset=d, images=POPULATIONS[d], arms=list(ARMS),
                 config=f'configs/{d}.json', manifest=f'manifests/{d}.json', output=f'cells/full/{d}/seed0')
            for d in DATASETS]


def service_success(service, root, expected_hash, execution, python):
    fields = ('Id', 'LoadState', 'ActiveState', 'SubState', 'Result', 'MainPID', 'ExecMainPID',
              'ExecMainCode', 'ExecMainStatus', 'ExecStart', 'ExecMainStartTimestamp', 'ExecMainExitTimestamp')
    response = subprocess.run(['systemctl', '--user', 'show', service, '--property=' + ','.join(fields)],
                              capture_output=True, text=True, check=True, timeout=15)
    state = dict(line.split('=', 1) for line in response.stdout.splitlines() if '=' in line)
    require(set(fields) <= state.keys() and state['Id'] == service and state['LoadState'] == 'loaded',
            'Service identity unavailable')
    command = (f'argv[]={python} -B -m tools.evolution_image_full.queue run --suite {root} '
               f'--suite-sha256 {expected_hash} ;')
    require(command in state['ExecStart'], 'Service root/command binding differs')
    require(integer(execution['pid'], 'execution PID') > 0
            and state['ExecMainPID'] == str(execution['pid']), 'Service execution PID differs')
    if state['MainPID'] != '0' or state['SubState'] in ('running', 'start', 'start-post'):
        raise Pending('Suite service has not reached terminal state')
    require(state['Result'] == 'success' and state['ExecMainCode'] == '1' and state['ExecMainStatus'] == '0'
            and (state['ActiveState'], state['SubState']) in {('active', 'exited'), ('inactive', 'dead')}
            and bool(state['ExecMainStartTimestamp']) and bool(state['ExecMainExitTimestamp']),
            'Service did not exit successfully')
    return state


def validate_cost(row, arm, cfg):
    require(row['arm'] == arm and arm in ARMS, 'Actual cost arm differs')
    images = integer(row['images'], 'images')
    patches, candidates = integer(row['patches'], 'patches'), integer(row['candidate_count'], 'candidates')
    active = integer(row['active_candidates'], 'active candidates')
    require(images > 0 and patches == images and 0 <= active <= candidates <= images * cfg['trace_max_regions'],
            'Actual candidate/patch count differs')
    sha256(row.get('candidate_sha256', '0' * 64))
    steps = cfg['steps'] if active else 0
    require(type(row['steps']) is int and row['steps'] == steps, 'Actual active-candidate steps differ')
    views = candidates if arm == 'image_C' else 0
    calls = (views + cfg['trace_view_batch_size'] - 1) // cfg['trace_view_batch_size']
    expected = dict(diagnostic_forward_inputs=views, diagnostic_encoder_inputs=views,
        diagnostic_forward_calls=calls, diagnostic_encoder_calls=calls, eval_forward_calls=2,
        eval_forward_inputs=2 * patches, adapt_forward_calls=steps, adapt_forward_inputs=steps * patches,
        encoder_calls=2 + steps + calls, encoder_inputs=(2 + steps) * patches + views)
    require({k for k in row if k.endswith(('_calls', '_inputs'))} == set(COUNTERS), 'Counter schema differs')
    require(all(integer(row[k], k) == v for k, v in expected.items()), 'Actual 0/0/1 schedule differs')
    number(row['seconds'], 'seconds')


def cost_summary(data, manifest, result):
    n, cfg = manifest['max_images'], manifest['config']
    batch = cfg['batch_size']
    episodes = (n + batch - 1) // batch
    rows = list(jsonlines(data, episodes * len(ARMS)))
    require(len(rows) == episodes * len(ARMS), 'Incomplete cost stream')
    totals = {a: {k: 0 for k in COUNTERS} for a in ARMS}
    timings = {a: dict(seconds=0., measured_images=0, measured_episodes=0) for a in ARMS}
    candidate_totals = {a: 0 for a in ARMS}
    for e in range(episodes):
        group = rows[e * len(ARMS):(e + 1) * len(ARMS)]
        order = ARMS if e % 2 == 0 else tuple(reversed(ARMS))
        require([r['arm'] for r in group] == list(order), 'Cost episode/arm order differs')
        require(len({(r['candidate_sha256'], r['candidate_count']) for r in group}) == 1,
                'Candidate identity/count differs across arms')
        for position, row in enumerate(group):
            a = row['arm']
            require(type(row['episode']) is int and row['episode'] == e and row['order_position'] == position
                    and row['warmup'] is (e == 0) and row['images'] == min(batch, n - e * batch),
                    'Cost image/warmup scope differs')
            validate_cost(row, a, cfg)
            require(row['seconds'] > 0, 'Zero measured time')
            candidate_totals[a] += row['candidate_count']
            for k in COUNTERS:
                totals[a][k] += row[k]
            if e:
                t = timings[a]
                t['seconds'] += row['seconds']
                t['measured_images'] += row['images']
                t['measured_episodes'] += 1
    for a in ARMS:
        require(totals[a] == result['costs'][a], 'Canonical actual cost totals differ')
        t, supplied = timings[a], result['timings'][a]
        require(t['measured_images'] == supplied['measured_images'] == n - batch
                and t['measured_episodes'] == supplied['measured_episodes'] == episodes - 1
                and supplied['warmup_episode'] == 0 and supplied['accuracy_includes_warmup'] is True,
                'Timing denominator differs')
        for k, v in (('seconds', t['seconds']), ('seconds_per_image', t['seconds'] / (n - batch)),
                     ('images_per_second', (n - batch) / t['seconds'])):
            close(supplied[k], v, 'Timing ' + k)
        totals[a].update(images=n, **t, candidates=candidate_totals[a],
                         ms_per_image=1000 * t['seconds'] / (n - batch),
                         diagnostic_inputs_per_image=totals[a]['diagnostic_encoder_inputs'] / n,
                         encoder_inputs_per_image=totals[a]['encoder_inputs'] / n)
    return totals


def verify_selection(ev, prepared):
    prefix = 'selection/'
    ev.raw(prefix + 'SUITE_PREPARED.json', SELECTION_PREPARED_SHA)
    ev.raw(prefix + 'SELECTION_SEALED.json', SELECTION_SHA)
    selected_prepared = ev.read(prefix + 'SUITE_PREPARED.json')
    selection, completed = ev.read(prefix + 'SELECTION_SEALED.json'), ev.read(prefix + 'SUITE_SEALED.json')
    require(selection['complete'] is True and selection['selected_mode'] == 'C'
            and selection['suite_prepared_sha256'] == SELECTION_PREPARED_SHA
            and completed['complete'] is True and completed['selected_mode'] == 'C'
            and completed['selection_sha256'] == SELECTION_SHA
            and completed['suite_prepared_sha256'] == SELECTION_PREPARED_SHA
            and completed['formal_cells'] == 8 and completed['arm_results'] == 48,
            'Pinned selected-C proof differs')
    inherited = lambda k: k.startswith(INHERITED_TREES) or k in INHERITED_FILES
    original = {k: v for k, v in selected_prepared['runtime_hashes'].items() if inherited(k)}
    actual = {k: v for k, v in prepared['runtime_hashes'].items() if inherited(k)}
    require(len(original) == 61 and actual == original and set(INHERITED_FILES) <= set(actual)
            and all(any(k.startswith(t) for k in actual) for t in INHERITED_TREES),
            'Selected-C inherited 61-source continuity differs')
    plan = ev.read(prefix + 'frozen_plan.json')
    require(selection['rule'] == plan['selection'] and selection['rule']['tie_band_pp'] == .02,
            'Selection rule differs')
    barrier = ev.read(prefix + 'ALL_CALIBRATION_PREDICTIONS_SEALED.json')
    require(barrier['complete'] is True and barrier['stage'] == 'calibration'
            and barrier['gt_contents_opened'] is False
            and barrier['suite_prepared_sha256'] == SELECTION_PREPARED_SHA
            and selection['calibration_barrier_sha256'] == ev.hashes[prefix + 'ALL_CALIBRATION_PREDICTIONS_SEALED.json'],
            'Selection calibration barrier differs')
    expected = {f'cells/calibration/{d}/sorted' for d in base.DATASETS}
    require(set(selection['calibration_score_sha256']) == set(barrier['prediction_seal_sha256']) == expected,
            'Selection evidence not exact two calibration cells')
    direct = {'SELECTION_SEALED.json', 'SUITE_PREPARED.json', 'frozen_plan.json',
              'ALL_CALIBRATION_PREDICTIONS_SEALED.json'}
    calibration = {}
    for d in base.DATASETS:
        relative = f'cells/calibration/{d}/sorted'
        p = prefix + relative + '/'
        ev.raw(p + 'SCORED_SEALED.json', selection['calibration_score_sha256'][relative])
        seal = ev.read(p + 'SCORED_SEALED.json')
        require(set(seal['artifact_sha256']) == SCORING, 'Calibration score inventory differs')
        ev.inventory(seal['artifact_sha256'], p)
        require(seal['prediction_seal_sha256'] == barrier['prediction_seal_sha256'][relative],
                'Calibration prediction barrier differs')
        r = ev.read(p + 'result.json')
        require(r['stage'] == 'calibration' and r['dataset'] == d and r['images'] == 128
                and r['order_seed'] is None and r['status'] == 'completed', 'Calibration scope differs')
        calibration[d] = r
        direct.update(relative + '/' + n for n in ('SCORED_SEALED.json', 'result.json'))
    for name in direct:
        ev.raw(prefix + name, completed['artifact_sha256'].get(name))
        require(completed['artifact_sha256'].get(name) == ev.hashes[prefix + name], 'Selection completed seal differs')
    candidates = {}
    for mode in ('T', 'C', 'min'):
        a = 'image_' + mode
        margins = {d: pooled(calibration[d]['arm_results'][a], len(calibration[d]['arm_results'][a]['union']))
                   - pooled(calibration[d]['arm_results']['image_confidence'],
                            len(calibration[d]['arm_results']['image_confidence']['union'])) for d in base.DATASETS}
        candidates[mode] = dict(margin_pp=margins, worst_dataset_margin_pp=min(margins.values()),
            macro_margin_pp=math.fsum(margins.values()) / 2,
            diagnostic_encoder_inputs=sum(integer(calibration[d]['costs'][a]['diagnostic_encoder_inputs'], 'calibration cost')
                                          for d in base.DATASETS))
    require(candidates == selection['candidates'], 'Recorded calibration statistics differ')
    best = max(v['worst_dataset_margin_pp'] for v in candidates.values())
    tied = [m for m in ('T', 'C', 'min') if best - candidates[m]['worst_dataset_margin_pp'] <= .02 + 1e-12]
    winner = min(tied, key=lambda m: (candidates[m]['diagnostic_encoder_inputs'],
                                    -candidates[m]['macro_margin_pp'], ('T', 'C', 'min').index(m)))
    require(winner == 'C', 'Pinned calibration selection is not C; no reselection allowed')
    return selection, original


def verify_manifest(ev, e, prepared):
    d, n = e['dataset'], e['images']
    ev.raw(e['manifest'], e['manifest_sha256'])
    m, cfg = ev.read(e['manifest']), ev.read(e['config'])
    parent_path = f'parents/{d}/prediction_protocol.json'
    ev.raw(parent_path, PARENT_HASHES[d][0])
    parent, receipt = ev.read(parent_path), ev.read(f'parents/{d}/parent_receipt.json')
    directory = (VOC20_ROOT if d == 'voc20' else PARENT_ROOT) / 'cells/mechanism' / d
    require(receipt == dict(protocol_path=str(directory / 'prediction_protocol.json'), protocol_sha256=PARENT_HASHES[d][0],
        seal_path=str(directory / 'PREDICTIONS_SEALED.json'), seal_sha256=PARENT_HASHES[d][1], complete=True,
        gt_contents_opened=False, images=n, sealed_protocol_sha256=PARENT_HASHES[d][0]), 'Pinned parent receipt differs')
    require(parent['dataset'] == d and type(parent['seed']) is int and parent['seed'] == 0
            and parent['full_split'] is True and parent['images'] == parent['available_images'] == n,
            'Parent full population differs')
    fixed = dict(format_version=1, worker='image_full', dataset=d, stage='full', seed=0, order_seed=None,
        max_images=n, arms=list(ARMS), arm_order='alternating', suite_root=str(ev.root),
        config_path=str(ev.root / e['config']), config_sha256=ev.hashes[e['config']],
        selection_sha256=SELECTION_SHA, parent_protocol_sha256=PARENT_HASHES[d][0],
        parent_seal_sha256=PARENT_HASHES[d][1], plan_sha256=PLAN_SHA,
        timing_scope=TIMING_SCOPE, exposure_scope=EXPOSURE, preupdate_prediction_status=PREUPDATE)
    require(all(m[k] == v and type(m[k]) is type(v) for k, v in fixed.items()), 'Manifest full admission differs')
    require(m['config'] == cfg == parent['config'] and m['checkpoint'] == parent['checkpoint'],
            'Inherited config/checkpoint differs')
    require(set(cfg) == {'dataset', 'data_subdir', 'data_dir', 'resize', 'patch_size', 'patch_stride',
            'batch_size', 'lr', 'steps', 'trace_max_regions', 'class_extensions', 'trace_view_batch_size'}
            and cfg['resize'] == cfg['patch_size'] == [224, 224] and Path(cfg['data_dir']).is_absolute()
            and type(cfg['class_extensions']) is bool, 'Config schema/preprocessing differs')
    for k in ('steps', 'trace_max_regions', 'trace_view_batch_size', 'patch_stride', 'batch_size'):
        require(integer(cfg[k], k) > 0, 'Invalid config ' + k)
    require(cfg['batch_size'] < n and number(cfg['lr'], 'lr') > 0, 'Invalid batch/lr')
    require(m['source_sha256'] == {str(ev.root / 'runtime' / k): v for k, v in prepared['runtime_hashes'].items()},
            'Manifest full 149-source link differs')
    p = m['panel']
    require(p['full_split'] is True and len(p['inputs']) == len(parent['inputs']) == len(p['ordered_ids']) == n
            and p['inputs'] == parent['inputs'] and p['ordered_ids'] == [r['image'] for r in p['inputs']]
            and p['ordered_ids'] == sorted(set(p['ordered_ids'])), 'Full population/order differs')
    for k in ('classes', 'queries', 'query_to_class', 'reduce_zero_label'):
        require(p[k] == parent[k], 'Inherited vocabulary/label convention differs')
    require(0 < len(p['classes']) <= 4096 and p['queries'] and type(p['reduce_zero_label']) is bool,
            'Invalid vocabulary scope')
    for k in ('image', 'label'):
        require(len({r[k] for r in p['inputs']}) == n and all(Path(r[k]).is_absolute() for r in p['inputs']),
                'Duplicate/nonabsolute full input')
    for r in p['inputs']:
        sha256(r['image_sha256'])
        require(len(r['label_metadata']) == 2 and all(type(x) is int and x >= 0 for x in r['label_metadata']),
                'Invalid label stat metadata')
    return m


def verify_gate(ev, e, m, prepared_sha):
    name = f'gates/{e["dataset"]}.json'
    g = ev.read(name)
    marker = name + '.STARTED.json'
    ev.raw(marker, g['attempt_sha256'])
    require(ev.read(marker) == dict(status='started', manifest_sha256=e['manifest_sha256']), 'Gate attempt differs')
    require(g['passed'] is True and g['gt_contents_opened'] is False and g['images'] == 4
            and g['dataset'] == e['dataset'] and g['manifest_sha256'] == e['manifest_sha256']
            and g['suite_prepared_sha256'] == prepared_sha and g['checkpoint'] == m['checkpoint']
            and g['source_sha256'] == m['source_sha256'] and g['ordered_ids'] == m['panel']['ordered_ids'][:4],
            'GT-free gate identity differs')
    batch = m['config']['batch_size']
    sizes = [min(batch, 4 - i) for i in range(0, 4, batch)]
    require(g['configured_batch_size'] == batch and g['batch_images'] == sizes
            and len(g['comparisons']) == len(sizes), 'Gate batch population differs')
    base.runtime(g['deterministic_runtime'])
    require(g['gpu_lock']['exclusive'] is True and integer(g['gpu_lock']['pid'], 'gate PID') > 0,
            'Gate exclusive lock differs')
    require(sum(c['candidates'] for c in g['comparisons']) > 0, 'Vacuous gate')
    for c, images in zip(g['comparisons'], sizes):
        require(c['candidate_masks_equal'] is True and c['reset_equal'] is True and c['changed_label_pixels'] == 0
                and set(c['actual_arm_counts']) == set(ARMS), 'Gate equivalence differs')
        for k in ('support', 'coefficient', 'loss', 'gradient'):
            require(number(c['max_' + k + '_abs_difference'], k) <= 1e-5, 'Gate numeric difference')
        for a in ARMS:
            r = c['actual_arm_counts'][a]
            require(r['images'] == images and r['candidate_count'] == c['candidates'], 'Gate actual population differs')
            validate_cost(r, a, m['config'])
    return g


def verify_cell(ev, e, m, barrier, gate_hashes, gate, prepared_sha):
    prefix, n = e['output'] + '/', e['images']
    pred = ev.read(prefix + 'PREDICTIONS_SEALED.json')
    pred_hash = ev.hashes[prefix + 'PREDICTIONS_SEALED.json']
    episodes = (n + m['config']['batch_size'] - 1) // m['config']['batch_size']
    expected = ({f'predictions/{a}/{i:06d}.npy' for a in ARMS for i in range(n)}
                | {f'episodes/{i:06d}.{a}.json' for a in ARMS for i in range(episodes)} | SMALL)
    require(pred['complete'] is True and pred['gt_contents_opened'] is False and pred['images'] == n
            and pred['arms'] == list(ARMS) and pred['suite_prepared_sha256'] == prepared_sha
            and set(pred['artifact_sha256']) == expected
            and barrier['prediction_seal_sha256'][e['output']] == pred_hash, 'Prediction seal inventory/barrier differs')
    for h in pred['artifact_sha256'].values():
        sha256(h)
    ev.inventory({k: pred['artifact_sha256'][k] for k in SMALL}, prefix)
    p = ev.read(prefix + 'prediction_protocol.json')
    require(p['manifest'] == m and p['manifest_path'] == str(ev.root / e['manifest'])
            and p['manifest_sha256'] == e['manifest_sha256'] and p['suite_prepared_sha256'] == prepared_sha
            and p['selection_sha256'] == SELECTION_SHA and p['gate_sha256'] == gate_hashes,
            'Prediction protocol/gates binding differs')
    base.runtime(p['deterministic_runtime'])
    require(p['deterministic_runtime'] == gate['deterministic_runtime']
            and p['gpu_lock']['exclusive'] is True and integer(p['gpu_lock']['pid'], 'prediction PID') > 0
            and all(p['gpu_lock'][k] == gate['gpu_lock'][k] for k in ('path', 'inode', 'exclusive')),
            'Prediction/gate runtime/lock differs')
    initial = ev.read(prefix + 'initial_state.json')
    sha256(initial['model_sha256'])
    require(initial['adapted_parameter_names'] and len(set(initial['adapted_parameter_names']))
            == len(initial['adapted_parameter_names']), 'Initial adapted parameter scope differs')
    seal = ev.read(prefix + 'SCORED_SEALED.json')
    require(seal['prediction_seal_sha256'] == pred_hash and set(seal['artifact_sha256']) == SCORING,
            'Score seal scope differs')
    ev.inventory(seal['artifact_sha256'], prefix)
    barrier_sha = ev.hashes['ALL_FULL_PREDICTIONS_SEALED.json']
    require(ev.read(prefix + 'SCORING_STARTED.json') == dict(prediction_seal_sha256=pred_hash,
            barrier_sha256=barrier_sha, suite_prepared_sha256=prepared_sha), 'Scoring barrier/start differs')
    r, predicted = ev.read(prefix + 'result.json'), ev.read(prefix + 'prediction_result.json')
    require(r['status'] == 'completed' and r['dataset'] == e['dataset'] and r['stage'] == 'full'
            and type(r['seed']) is int and r['seed'] == 0 and r['images'] == n and r['full_split'] is True
            and r['prediction_seal_sha256'] == pred_hash and r['barrier_sha256'] == barrier_sha
            and r['selection_sha256'] == SELECTION_SHA and r['exposure_scope'] == EXPOSURE
            and r['label_contents_first_opened_after_all_full_predictions_sealed'] is True
            and set(r['arm_results']) == set(r['costs']) == set(r['timings']) == set(ARMS), 'Canonical full scope differs')
    require(predicted['status'] == 'predicted' and predicted['images'] == n and predicted['full_split'] is True
            and r['timing_scope'] == TIMING_SCOPE and all(r[k] == predicted[k] for k in ('costs', 'timings', 'timing_scope')),
            'Canonical/predicted actual costs/timing differ')
    classes = len(m['panel']['classes'])
    totals = {a: [[0] * classes, [0] * classes] for a in ARMS}
    rows = list(jsonlines(ev.raw(prefix + 'scores_by_episode.jsonl'), episodes))
    require(len(rows) == episodes, 'Incomplete pooled episode stream')
    for i, row in enumerate(rows):
        require(type(row['episode']) is int and row['episode'] == i
                and integer(row['images'], 'scored images') == min(m['config']['batch_size'], n - i * m['config']['batch_size'])
                and set(row['arms']) == set(ARMS), 'Pooled episode population differs')
        for a in ARMS:
            values = base.counts(row['arms'][a], classes)
            totals[a] = [[x + y for x, y in zip(old, new)] for old, new in zip(totals[a], values)]
    scores = {}
    for a in ARMS:
        require(totals[a] == [r['arm_results'][a]['intersection'], r['arm_results'][a]['union']],
                'Pooled episode counts differ')
        scores[a] = pooled(r['arm_results'][a], classes)
    labels = ev.read(prefix + 'score_inputs.json')['label_sha256']
    require(set(labels) == {r['label'] for r in m['panel']['inputs']} and len(labels) == n,
            'Offline label hash population differs')
    for h in labels.values():
        sha256(h)
    return dict(key=e['key'], dataset=e['dataset'], images=n, full_split=True, seed=0, scores=scores,
        margins={a: scores[a] - scores['image_confidence'] for a in ARMS},
        contrasts=dict(C_minus_confidence_pp=scores['image_C'] - scores['image_confidence'],
                       C_minus_uniform_pp=scores['image_C'] - scores['image_uniform']),
        pooled_counts={a: dict(intersection=totals[a][0], union=totals[a][1]) for a in ARMS},
        costs=cost_summary(ev.raw(prefix + 'cost.jsonl'), m, r), timings=r['timings'], config=m['config'],
        checkpoint=m['checkpoint'], classes=m['panel']['classes'],
        ordered_ids_sha256=canonical(m['panel']['ordered_ids']), inputs_sha256=canonical(m['panel']['inputs']),
        manifest_sha256=e['manifest_sha256'], config_sha256=m['config_sha256'],
        prediction_seal_sha256=pred_hash, score_seal_sha256=ev.hashes[prefix + 'SCORED_SEALED.json'],
        initial_model_sha256=initial['model_sha256'])


def verify_csv(data, cells):
    require(len(data) <= 1024 * 1024, 'CSV exceeds bound')
    columns = ('dataset', 'seed', 'arm', 'images', 'full_split', 'miou_unrounded', 'margin_vs_image_confidence_pp',
        'diagnostic_encoder_inputs', 'measured_images', 'seconds', 'seconds_per_image', 'images_per_second',
        'manifest_sha256', 'selection_sha256', 'prediction_seal_sha256', 'score_seal_sha256', 'exposure')
    table = csv.DictReader(io.StringIO(data.decode()))
    require(table.fieldnames == list(columns), 'CSV columns differ')
    rows = list(table)
    require(len(rows) == 18, 'Expected exactly 18 CSV rows')
    for row, (c, a) in zip(rows, ((c, a) for c in cells for a in ARMS)):
        fixed = dict(dataset=c['dataset'], seed='0', arm=a, images=str(c['images']), full_split='True',
            diagnostic_encoder_inputs=str(c['costs'][a]['diagnostic_encoder_inputs']),
            measured_images=str(c['timings'][a]['measured_images']), selection_sha256=SELECTION_SHA, exposure=EXPOSURE,
            **{k: c[k] for k in ('manifest_sha256', 'prediction_seal_sha256', 'score_seal_sha256')})
        require(set(row) == set(columns) and all(row[k] == v for k, v in fixed.items()),
                'CSV canonical population/order/provenance differs')
        for k, v in (('miou_unrounded', c['scores'][a]), ('margin_vs_image_confidence_pp', c['margins'][a]),
                     *((k, c['timings'][a][k]) for k in ('seconds', 'seconds_per_image', 'images_per_second'))):
            close(float(row[k]), v, 'CSV ' + k)


def verify_receipts(ev, prepared, expected_sha):
    execution = ev.read('execution_children.json')
    require(len(execution) == 18 and len(prepared['child_receipts']) == 6, 'Child receipt count differs')
    commands = [(execution[index * 6 + cell], action, e)
                for index, action in enumerate(('gate', 'predict', 'score'))
                for cell, e in enumerate(prepared['cells'])]
    commands = [(r, 'prepare', e) for r, e in zip(prepared['child_receipts'], prepared['cells'])] + commands
    for r, action, e in commands:
        argv = [prepared['python'], '-B', '-m', 'tools.evolution_image_full.run', action, '--suite', str(ev.root)]
        if action != 'prepare':
            argv += ['--suite-sha256', expected_sha]
        argv += ['--dataset', e['dataset']]
        require(r['argv'] == argv and type(r['exit_code']) is int and r['exit_code'] == 0
                and r['error'] is None and r['cpu_only'] is (action in ('prepare', 'score'))
                and r['log'] == f'logs/{action}-{e["dataset"]}.log'
                and r['receipt'] == f'logs/{action}-{e["dataset"]}.receipt.json', 'Child command/exit/stage differs')
        ev.raw(r['log'], r['log_sha256'])
        ev.raw(r['receipt'], r['receipt_sha256'])
        require(ev.read(r['receipt']) == {k: v for k, v in r.items() if k not in ('receipt', 'receipt_sha256')},
                'Child receipt binding differs')
        require(integer(r['retained_bytes'], 'retained log bytes') == len(ev.raw(r['log']))
                and integer(r['output_bytes'], 'output log bytes') >= r['retained_bytes'], 'Child log bytes differ')


def verify_suite(ev, expected_sha, service):
    ev.raw('SUITE_PREPARED.json', expected_sha)
    p = ev.read('SUITE_PREPARED.json')
    require(p['format_version'] == 1 and p['worker'] == 'image_full_queue' and p['suite_root'] == str(ev.root)
            and p['runtime_root'] == str(ev.root / 'runtime') and p['arms'] == list(ARMS)
            and p['selected_mode'] == 'C' and p['selection_sha256'] == SELECTION_SHA and p['exposure_note'] == EXPOSURE,
            'Prepared suite root/scope differs')
    require(len(p['cells']) == 6 and all(set(e) == set(x) | {'manifest_sha256'} and all(e[k] == v for k, v in x.items())
            for e, x in zip(p['cells'], entries())), 'Exact six canonical cells differ')
    require(len(p['runtime_hashes']) == 149, 'Frozen 149-source count differs')
    ev.inventory(p['runtime_hashes'], 'runtime/')
    require(base.runtime_inventory(ev.root / 'runtime') == set(p['runtime_hashes']), 'Frozen source inventory differs')
    ev.inventory(p['artifact_sha256'])
    ev.raw('frozen_plan.json', PLAN_SHA)
    require(p['plan_sha256'] == PLAN_SHA, 'Prepared plan differs')
    plan = ev.read('frozen_plan.json')
    require(plan['datasets'] == list(DATASETS) and plan['populations'] == POPULATIONS and plan['arms'] == list(ARMS)
            and plan['selected_global_mode'] == 'C' and plan['selection_sha256'] == SELECTION_SHA
            and plan['selection_prepared_sha256'] == SELECTION_PREPARED_SHA
            and (plan['formal_cells'], plan['arm_results'], plan['gt_free_gates'], plan['gate_images_per_dataset']) == (6, 18, 6, 4),
            'Pinned full plan scope differs')
    selection, continuity = verify_selection(ev, p)
    manifests = [verify_manifest(ev, e, p) for e in p['cells']]
    require(len({canonical(m['checkpoint']) for m in manifests}) == 1, 'Cross-dataset checkpoint differs')
    if not child(ev.root, 'SUITE_SEALED.json').exists():
        require(ev.read('queue_state.json').get('phase') != 'failed', 'Queue failed')
        raise Pending('Whole-suite seal absent; outputs untouched')
    seal = ev.read('SUITE_SEALED.json')
    fixed = dict(complete=True, full_split=True, formal_cells=6, arm_results=18, passed_gt_free_gates=6,
        selected_mode='C', selection_sha256=SELECTION_SHA, suite_prepared_sha256=expected_sha,
        runtime_hashes=p['runtime_hashes'], exposure_note=EXPOSURE, service_success_verified=False)
    require(all(seal[k] == v and type(seal[k]) is type(v) for k, v in fixed.items()), 'Whole-suite seal scope differs')
    required = {'SUITE_PREPARED.json', 'EXECUTION_STARTED.json', 'execution_children.json', 'queue_state.json', 'rows.csv',
                'ALL_FULL_PREDICTIONS_SEALED.json', *p['artifact_sha256']}
    required.update(f'gates/{d}.json' + suffix for d in DATASETS for suffix in ('', '.STARTED.json'))
    required.update(e['output'] + '/' + name for e in p['cells'] for name in (*SCORING, 'SCORED_SEALED.json', 'PREDICTIONS_SEALED.json'))
    require(required <= seal['artifact_sha256'].keys(), 'Whole-suite canonical coverage missing')
    ev.inventory(seal['artifact_sha256'])
    execution = ev.read('EXECUTION_STARTED.json')
    require(execution['suite_prepared_sha256'] == expected_sha, 'Execution prepared link differs')
    observed = service_success(service, ev.root, expected_sha, execution, p['python'])
    state = ev.read('queue_state.json')
    require(all(state[k] == v and type(state[k]) is type(v) for k, v in dict(phase='completed', active_cell=None,
        passed_gates=6, sealed_predictions=6, scored_cells=6, total_cells=6, total_arm_results=18, error=None).items()),
        'Queue completion counts differ')
    verify_receipts(ev, p, expected_sha)
    barrier = ev.read('ALL_FULL_PREDICTIONS_SEALED.json')
    require(barrier['complete'] is True and barrier['stage'] == 'full' and barrier['gt_contents_opened'] is False
            and barrier['suite_prepared_sha256'] == expected_sha
            and set(barrier['prediction_seal_sha256']) == {e['output'] for e in p['cells']}, 'Global six-cell barrier differs')
    gates = {e['dataset']: verify_gate(ev, e, m, expected_sha) for e, m in zip(p['cells'], manifests)}
    gate_hashes = {d: ev.hashes[f'gates/{d}.json'] for d in DATASETS}
    cells = [verify_cell(ev, e, m, barrier, gate_hashes, gates[e['dataset']], expected_sha)
             for e, m in zip(p['cells'], manifests)]
    require(len({c['initial_model_sha256'] for c in cells}) == 1, 'Initial reset-model identities differ')
    verify_csv(ev.raw('rows.csv'), cells)
    return p, observed, execution, selection, continuity, barrier, cells


def verify_references(paper, cells):
    ev = Evidence(paper)
    reported = ev.read('reported_baselines.json')
    require(Path(reported['source_pdf']).stat().st_size <= 32 * 1024 * 1024, 'Reference PDF exceeds bound')
    fingerprints = {}
    reference.verify_reported(reported, fingerprints)
    native = Evidence(reference.NATIVE_ROOT)
    seal = native.read('SUITE_SEALED.json')
    cd = next(c for c in cells if c['dataset'] == 'coco_cd')
    # Historical native rows stay a separate protocol, never matched controls.
    methods = {}
    for method in ('noadapt', 'tent', 'tpt', 'mlmp'):
        prefix = f'coco_cd/{method}/'
        for name in ('result.json', 'protocol.json'):
            relative = prefix + name
            require(relative in seal['artifact_sha256'], 'Native baseline missing from seal')
            native.raw(relative, seal['artifact_sha256'][relative])
        r, p = native.read(prefix + 'result.json'), native.read(prefix + 'protocol.json')
        require(r['status'] == 'completed' and r['full_split'] is True and p['full_split'] is True
                and r['images'] == p['images'] == p['available_images'] == cd['images']
                and r['method'] == p['method'] == method and p['seed'] == 0
                and p['labels_used_for_adaptation'] is False
                and r['protocol_sha256'] == native.hashes[prefix + 'protocol.json'], 'Native baseline scope differs')
        require(p['checkpoint_sha256'] == cd['checkpoint']['sha256'], 'Native reference checkpoint differs')
        # Compare only recorded identifiers/hashes. No RGB or label files opened.
        require(len(p['image_ids']) == len(p['label_ids']) == cd['images']
                and canonical(p['image_ids']) == cd['ordered_ids_sha256'], 'Native reference full order differs')
        counts = {}
        for k in ('intersection', 'union'):
            require(all(reference.whole_count(v) for v in r[k]), 'Invalid native pooled counts')
            counts[k] = [int(v) for v in r[k]]
        value = pooled(dict(counts, miou_unrounded=r['miou_unrounded']), len(cd['classes']))
        methods[method] = dict(miou_unrounded=value, config=p['config'], method_settings=p['method_settings'],
            protocol_sha256=native.hashes[prefix + 'protocol.json'], result_sha256=native.hashes[prefix + 'result.json'],
            pooled_counts=counts, comparison_scope='Historical local native protocol, not the new three-arm experiment')
    best = max(methods, key=lambda m: methods[m]['miou_unrounded'])
    require(best == 'tent', 'Native best-method identity differs')
    return dict(reported=reported, native_cd=methods, native_cd_best=best,
        native_cd_delta_pp=cd['scores']['image_C'] - methods[best]['miou_unrounded'],
        reported_deltas_pp={c['dataset']: c['scores']['image_C'] - max(v[0] for v in reported['datasets'][c['dataset']].values())
                            for c in cells if c['dataset'] in reported['datasets']},
        reported_pdf_sha256=fingerprints), (ev, native)


def table(label, caption, heading, rows, columns, one_column=False):
    environment = 'table' if one_column else 'table*'
    return '\n'.join(['\\begin{' + environment + '}[t]', r'\caption{' + caption + '}', r'\label{' + label + '}',
        r'\centering\small' + (r'\setlength{\tabcolsep}{4pt}' if one_column else ''),
        r'\begin{tabular}{@{}' + columns + '@{}}', r'\toprule', heading + r' \\',
        r'\midrule', *rows, r'\bottomrule', r'\end{tabular}', '\\end{' + environment + '}', ''])


def render(cells, refs):
    scope = ('Effect of regional weighting on full-split mIoU (seed 0). '
             'Candidates, readouts, and optimization settings are shared within each dataset. ')
    comparison = table('tab:image-full-comparison', scope +
        'Results use inherited configurations and development-exposed splits, as detailed in the setup. '
        'Object/Stuff denote COCO Object/COCO-Stuff. '
        '$\\Delta$ is TRACE(C) minus Confidence in percentage points.',
        r'Dataset & Uniform & Confidence & TRACE(C) & $\Delta$',
        [LABELS[c['dataset']] + ' & ' + ' & '.join(f'{c["scores"][a]:.2f}' for a in ARMS)
         + f' & {c["margins"]["image_C"]:+.2f}' + r' \\'
         for c in cells], 'lrrrr', one_column=True)
    cost = table('tab:image-full-cost', 'Synchronized model-side ms/image for the new full-split runs. '
        '$N_t$ excludes only episode 0 per arm. Includes reset, initial readout, candidates, requested views, '
        'LayerNorm optimization, final inference and CPU prediction transfer. Excludes loading, RGB decode, '
        'H2D/geometry preparation, provenance checks, disk writes and scoring; not end-to-end throughput.',
        r'Dataset & $N_t$ & Uniform & Confidence & TRACE(C)',
        [LABELS[c['dataset']] + f' & {c["costs"]["image_C"]["measured_images"]} & '
         + ' & '.join(f'{c["costs"][a]["ms_per_image"]:.2f}' for a in ARMS) + r' \\' for c in cells], 'lrrrr')
    cost += table('tab:image-full-work', 'Actual diagnostic encoder calls / inputs, including warmup. '
        'Uniform and Confidence compute zero probes; TRACE(C) computes one input per candidate. '
        'Counts are from these full populations, not earlier cohort ratios. Total encoder work is retained in provenance.',
        r'Dataset & $N$ & Candidates & Uniform & Confidence & TRACE(C)',
        [LABELS[c['dataset']] + f' & {c["images"]} & {c["costs"]["image_C"]["candidates"]} & '
         + ' & '.join(f'{c["costs"][a]["diagnostic_encoder_calls"]} / {c["costs"][a]["diagnostic_encoder_inputs"]}'
                      for a in ARMS) + r' \\' for c in cells], 'lrrrrr')
    order = ('voc20', 'voc21', 'pc60', 'coco_object', 'coco_stuff')
    by = {c['dataset']: c for c in cells}
    reported = refs['reported']
    rows = []
    for method in reported['methods']:
        values = []
        for d in order:
            mean, sd = reported['datasets'][d][method]
            values.append(f'${mean:.2f}' + (rf' \pm {sd:.2f}' if sd is not None else '') + '$')
        rows.append(method + ' (R) & ' + ' & '.join(values) + r' \\')
    rows += [r'\midrule', 'TRACE(C) (L) & ' + ' & '.join(f'{by[d]["scores"]["image_C"]:.2f}' for d in order) + r' \\',
             r'$\Delta$ vs. best R (pp) & ' + ' & '.join(f'{refs["reported_deltas_pp"][d]:+.2f}' for d in order) + r' \\']
    ref = table('tab:image-full-reference', r'Segmentation performance alongside published references. '
        r'R: reported means and standard deviations from Table~5 (Original NACLIP) of \citet{noori2025mlmp}; '
        'No Adapt has no reported dispersion. L: local TRACE(C), seed 0. '
        'Published and local methods retain their respective evaluation and optimization protocols; '
        'matched regional-weighting comparisons appear in Table~\\ref{tab:image-full-comparison}.',
        'Method & VOC20 & VOC21 & PC60 & Object & Stuff', rows, 'lrrrrr')
    cd_rows = [dict(noadapt='No Adapt', tent='TENT', tpt='TPT', mlmp='MLMP')[m]
               + f' (L, historical native) & {refs["native_cd"][m]["miou_unrounded"]:.2f}' + r' \\'
               for m in ('noadapt', 'tent', 'tpt', 'mlmp')]
    cd_rows += [r'\midrule', f'TRACE(C) (L, new) & {by["coco_cd"]["scores"]["image_C"]:.2f}' + r' \\',
                r'$\Delta$ vs. native TENT (pp) & ' + f'{refs["native_cd_delta_pp"]:+.2f}' + r' \\']
    ref += table('tab:image-full-cd-reference', 'COCO-CD performance alongside existing local baseline runs. '
        'Methods share the image set and checkpoint but retain their own backends, prompts, and optimization settings. '
        'The baseline No Adapt row uses its native readout. Protocol details are given in Appendix~\\ref{app:native}.',
        'Method / protocol & COCO-CD', cd_rows, 'lr')
    return dict(zip(OUTPUTS[:3], (comparison, cost, ref)))


def publish(paper, outputs):
    # Reuse the rollback-on-failure implementation, with only this output set.
    require(set(outputs) == set(OUTPUTS), 'Wrong full output set')
    directory = paper / 'generated'
    require(directory.is_dir(), 'Missing generated directory')
    old = {n: child(directory, n).read_bytes() if (directory / n).exists() else None for n in outputs}
    replaced = []
    with base.tempfile.TemporaryDirectory(prefix='.image-full-', dir=directory) as temp:
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


def run(root=DEFAULT_ROOT, paper=PAPER, expected_hash=PREPARED_SHA, service=SERVICE, check_only=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', "Set CUDA_VISIBLE_DEVICES='' for CPU-only import")
    root, paper = Path(root).absolute(), Path(paper).absolute()
    require(root == root.resolve() and paper == paper.resolve(), 'Suite/paper aliases rejected')
    ev = Evidence(root)
    prepared, observed, execution, selection, continuity, barrier, cells = verify_suite(ev, expected_hash, service)
    refs, ref_evidence = verify_references(paper, cells)
    outputs = render(cells, refs)
    provenance = dict(schema=1, suite_root=str(root), suite_prepared_sha256=expected_hash,
        suite_sealed_sha256=ev.hashes['SUITE_SEALED.json'], service=observed, cells=cells,
        selection=selection, selection_sha256=SELECTION_SHA, barrier=barrier,
        exposure=EXPOSURE, independent_unseen_certified=False, equal_tuning_certified=False,
        no_adapt_reset_readout_reused=False, references=refs,
        runtime_hashes=prepared['runtime_hashes'], inherited_source_continuity=continuity,
        verified_source_count=len(prepared['runtime_hashes']), inherited_source_count=len(continuity),
        verified_execution_children=18, verified_prepare_children=6,
        verification_scope='CPU-only bounded metadata. Independently rehashed 149 frozen sources and 61 inherited '
            'dependencies, prepared artifacts, selection calibration proof, six exact full populations/orders/configs, '
            'six gates, six prediction seal inventories, global barrier, 24 child receipts, scoring seals, '
            'six canonical results and all 18 CSV rows. Recomputed pooled class counts from every scored episode, '
            'all actual forward/encoder counters and timing excluding only episode 0. Fresh PDF Table 5 extraction '
            'and sealed historical native-reference count recomputation. No model/runtime execution, GPU, new '
            'scoring, labels, RGB, checkpoint weights, NPY or per-episode candidate payload reads. '
            'Payload bytes and original checkpoint/input identity and temporal label isolation rely on the pinned '
            'frozen queue final validate_prediction/validate_score full-payload certification plus genuine '
            'root/hash/PID-bound successful terminal service; NOT a fresh current-payload rehash certificate.',
        frozen_payload_check_sources=['runtime/tools/evolution_image_full/queue.py: run final validation loop',
            'runtime/tools/evolution_image_full/run.py: verify_prediction_seal and validate_score_stream'],
        timing_scope=TIMING_SCOPE, uncertainty='Single deterministic seed and development-exposed splits. '
            'No CI, significance, equal tuning, untouched holdout or matched external-method superiority claim.',
        source_sha256={str(e.root / n): h for e in (ev, *ref_evidence) for n, h in e.hashes.items()},
        importer_sha256=digest(Path(__file__).read_bytes()),
        helper_sha256={str(Path(m.__file__).resolve()): digest(Path(m.__file__).read_bytes()) for m in (base, reference)},
        output_sha256={n: digest(v.encode()) for n, v in outputs.items()})
    for e in (ev, *ref_evidence):
        e.stable()
    for path, expected in refs['reported_pdf_sha256'].items():
        require(reference.sha(Path(path)) == expected, 'Reference PDF drift')
    require(base.runtime_inventory(root / 'runtime') == set(prepared['runtime_hashes']), 'Runtime inventory drift')
    require(service_success(service, root, expected_hash, execution, prepared['python']) == observed, 'Service changed during import')
    outputs[OUTPUTS[-1]] = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if not check_only:
        publish(paper, outputs)
    return dict(status='ready' if check_only else 'imported', tables_unchanged=check_only, cells=6, arm_results=18,
        gates=6, outputs=list(OUTPUTS), C_minus_confidence_pp={c['dataset']: c['margins']['image_C'] for c in cells},
        C_minus_best_reported_pp=refs['reported_deltas_pp'], C_minus_native_cd_tent_pp=refs['native_cd_delta_pp']), 0


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
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        status, code = dict(status='invalid', error=str(error), tables_unchanged=True), 2
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
