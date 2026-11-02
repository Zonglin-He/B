"""Sequential baseline-first pilot on an existing sealed algorithm panel."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


DATASETS = ('voc21', 'pc60', 'coco_object', 'coco_cd')
BASELINES = ('noadapt', 'tent', 'tpt', 'mlmp')
METHOD_UPDATE_STEPS = dict(noadapt=0, native_noadapt=0, legacy_trace_noadapt=0,
                          tent=10, tpt=10, mlmp=10)
TIMING_SCOPE = 'GPU-synchronized adaptation and final inference; excludes loading/scoring/model setup'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def verify_suite(root):
    seal = json.loads((root/'SUITE_SEALED.json').read_text())
    for name, expected in seal['artifact_sha256'].items():
        if sha(root/name) != expected:
            raise ValueError(f'Algorithm evidence seal mismatch: {name}')
    if json.loads((root/'queue_state.json').read_text())['status'] != 'completed':
        raise ValueError('Algorithm queue must finish before this pilot')


def check_source_fingerprint(actual, frozen):
    for key in ('native', 'baseline', 'experiment'):
        if actual[key] != frozen[key]:
            raise ValueError(f'Between-cell source drift: {key}')
    for path, value in actual['controls'].items():
        if frozen['controls'].get(path) != value:
            raise ValueError(f'Between-cell control drift: {path}')


def cost_number(value, name, *, integer=False, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Invalid baseline cost: {name}') from error
    if (isinstance(value, bool) or not math.isfinite(number) or number < 0
            or (positive and number <= 0) or (integer and not number.is_integer())):
        raise ValueError(f'Invalid baseline cost: {name}')
    return int(number) if integer else number


def check_episode_costs(costs, expected, warmup, update_steps):
    batch_size = expected['config']['batch_size']
    count = len(expected['image_ids'])
    episodes = math.ceil(count / batch_size)
    if not count or len(costs) != episodes or type(warmup) is not int or warmup != min(2, episodes - 1):
        raise ValueError('Baseline cost episodes or warmup do not match the protocol')
    for episode, cost in enumerate(costs):
        ids = expected['image_ids'][episode*batch_size:(episode+1)*batch_size]
        if (cost_number(cost['episode'], 'episode', integer=True) != episode
                or cost_number(cost['images'], 'images', integer=True) != len(ids)
                or json.loads(cost['image_ids']) != ids):
            raise ValueError('Baseline cost episode boundary mismatch')
        if cost_number(cost['update_steps'], 'update_steps', integer=True) != update_steps:
            raise ValueError('Baseline adaptation step mismatch')


def recompute_costs(costs, warmup):
    rows = []
    for cost in costs:
        row = {key: cost_number(cost[key], key, integer=True, positive=True)
               for key in ('images', 'patches', 'calls', 'inputs')}
        row.update({key: cost_number(cost[key], key) for key in
                    ('adaptation_seconds', 'inference_seconds')})
        row.update({key: cost_number(cost[key], key, positive=True) for key in
                    ('total_seconds', 'peak_allocated_mib')})
        if (row['patches'] < row['images'] or row['inputs'] < max(row['calls'], row['patches'])
                or not math.isclose(row['total_seconds'],
                                    row['adaptation_seconds'] + row['inference_seconds'],
                                    rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError('Baseline cost components or visual counts are inconsistent')
        rows.append(row)
    measured = rows[warmup:]
    count = sum(row['images'] for row in measured)
    return dict(warmup_episodes=warmup, measured_images=count,
                ms_per_image=1000 * math.fsum(r['total_seconds'] for r in measured) / count,
                adaptation_ms_per_image=1000 * math.fsum(r['adaptation_seconds'] for r in measured) / count,
                inference_ms_per_image=1000 * math.fsum(r['inference_seconds'] for r in measured) / count,
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in measured),
                visual_inputs_per_image=sum(r['inputs'] for r in measured) / count,
                visual_calls_per_image=sum(r['calls'] for r in measured) / count,
                all_visual_inputs=sum(r['inputs'] for r in rows),
                all_visual_calls=sum(r['calls'] for r in rows))


def validate_cell(directory, expected, method, sources, expected_input_hashes=None,
                  expected_environment=None):
    """Validate cell evidence, optionally binding content and environment to its parent."""
    cell = json.loads((directory/'result.json').read_text())
    protocol = json.loads((directory/'protocol.json').read_text())
    for filename, key in (('protocol.json', 'protocol_sha256'),
                          ('per_image_counts.json', 'per_image_counts_sha256'),
                          ('cost.csv', 'cost_sha256')):
        if sha(directory/filename) != cell[key]:
            raise ValueError(f'Cell artifact seal mismatch: {filename}')
    if (cell['status'] != 'completed' or cell['images'] != len(expected['image_ids'])
            or cell['method'] != method):
        raise ValueError('Invalid completed baseline cell')
    for key in ('image_ids', 'label_ids', 'config', 'vocabulary_sha256', 'seed'):
        if protocol[key] != expected[key]:
            raise ValueError(f'Baseline panel mismatch: {key}')
    if expected_input_hashes is not None:
        paths = [path for pair in zip(expected['image_ids'], expected['label_ids']) for path in pair]
        inputs = protocol.get('input_hashes', {})
        if (not isinstance(expected_input_hashes, dict)
                or any(path not in expected_input_hashes for path in paths)
                or any(not isinstance(expected_input_hashes[path], str)
                       or len(expected_input_hashes[path]) != 64
                       or set(expected_input_hashes[path]) - set('0123456789abcdef') for path in paths)
                or inputs.get('sha256') != [expected_input_hashes[path] for path in paths]
                or inputs.get('file_metadata') != expected['file_metadata']):
            raise ValueError('Baseline input content differs from parent input hashes')
    if expected_environment is not None and protocol.get('environment') != expected_environment:
        raise ValueError('Baseline execution environment differs from native suite')
    if protocol.get('timing_scope') != TIMING_SCOPE:
        raise ValueError('Baseline timing scope mismatch')
    check_source_fingerprint(protocol['source_hashes'], sources)
    predictions = json.loads((directory/'per_image_counts.json').read_text())
    if [r['image_id'] for r in predictions] != expected['image_ids']:
        raise ValueError('Baseline prediction cohort mismatch')
    for key in ('intersection', 'union'):
        sums = [sum(r[key][i] for r in predictions) for i in range(len(cell[key]))]
        if sums != cell[key]:
            raise ValueError(f'Prediction counts mismatch: {key}')
    ratios = [i/u for i, u in zip(cell['intersection'], cell['union']) if u > 0]
    miou = 100*math.fsum(ratios)/len(ratios)
    if not math.isclose(miou, cell['miou_unrounded'], abs_tol=1e-9):
        raise ValueError('Baseline mIoU does not equal pooled per-image counts')
    costs = list(csv.DictReader((directory/'cost.csv').open()))
    if method not in METHOD_UPDATE_STEPS:
        raise ValueError(f'Unsupported baseline method: {method}')
    steps = METHOD_UPDATE_STEPS[method]
    for record in (protocol, cell):
        if cost_number(record['method_settings']['update_steps'], 'update_steps', integer=True) != steps:
            raise ValueError('Baseline method update count differs from fixed contract')
    check_episode_costs(costs, expected, cell['warmup_episodes'], steps)
    if protocol.get('warmup_episodes') != cell['warmup_episodes']:
        raise ValueError('Baseline protocol warmup mismatch')
    recomputed = recompute_costs(costs, cell['warmup_episodes'])
    for key, value in recomputed.items():
        actual = cost_number(cell[key], key, integer=isinstance(value, int))
        matches = actual == value if isinstance(value, int) else math.isclose(
            actual, value, rel_tol=1e-10, abs_tol=1e-12)
        if not matches:
            raise ValueError(f'Baseline cost summary mismatch: {key}')
    cell.update(recomputed)
    return cell


def publish(output, rows, algorithm_rows):
    write_csv(output/'baselines.csv', rows)
    comparisons = []
    for candidate in algorithm_rows:
        if candidate['algorithm'] not in ('quality_only', 'selective_01', 'selective_05',
                                          'quality_staged', 'quality_lazy_01', 'quality_lazy_class_01'):
            continue
        options = [r for r in rows if r['dataset'] == candidate['dataset']
                   and r['method'] in BASELINES and r['status'] == 'completed']
        complete = len(options) == len(BASELINES)
        best = max(options, key=lambda r: r['mIoU']) if options else None
        comparisons.append(dict(dataset=candidate['dataset'], algorithm=candidate['algorithm'],
            images=int(candidate['images']), mIoU=float(candidate['mIoU']),
            ms_per_image=float(candidate['ms_per_image']),
            matched_baselines_completed=len(options), comparison_complete=complete,
            best_available_baseline=best['method'] if best else None,
            best_available_mIoU=best['mIoU'] if best else None,
            delta_vs_best_available=float(candidate['mIoU'])-best['mIoU'] if best else None,
            runtime_ratio_vs_best_available=(float(candidate['ms_per_image'])/best['ms_per_image']
                                             if best and best['ms_per_image'] > 0 else None),
            result_path=candidate['result_path'],
            baseline_result_path=best['result_path'] if best else None,
            scope='same sealed IDs/vocabulary/episodes; method-native backends and settings'))
    write_csv(output/'comparison.csv', comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite, output, baseline = (p.resolve() for p in
                               (args.suite_root, args.output, args.baseline_root))
    verify_suite(suite)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Fresh output required; no retries or evidence overwrite')
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        locks = []
        for path in (output/'queue.lock', suite/'queue.lock'):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError(f'GPU already has compute workers; not interrupting: {active}')
        code = output/'code'
        code.mkdir()
        worker_source = Path(__file__).with_name('benchmark_matched_baselines.py')
        for source in (Path(__file__), worker_source):
            shutil.copy2(source, code/source.name)
        worker = code/worker_source.name
        script_sha = sha(worker)
        plan = [(dataset, method) for dataset in DATASETS
                for method in ('native_noadapt', 'legacy_trace_noadapt', *BASELINES)]
        parent_protocol = json.loads((suite/'suite_protocol.json').read_text())
        spec = importlib.util.spec_from_file_location('frozen_baseline_adapter', worker)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        controls = {suite/'suite_protocol.json', suite/'SUITE_SEALED.json',
                    Path.home()/'.cache/clip/ViT-L-14.pt'}
        for dataset, method in plan:
            _, files = adapter.archived_settings(baseline, dataset, method,
                parent_protocol['input_panels'][dataset+'/128']['config'])
            controls.update(files)
        controls = sorted(controls)
        source_fingerprint = adapter.source_snapshot(suite/'runtime', baseline, controls, worker)
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
            suite_root=str(suite), suite_seal_sha256=sha(suite/'SUITE_SEALED.json'),
            package_root=str(suite/'runtime'), baseline_root=str(baseline),
            cells=[dict(dataset=d, method=m) for d, m in plan],
            automatic_promotion=False, automatic_retry=False,
            scripts={p.name: sha(p) for p in code.iterdir() if p.is_file()},
            source_fingerprint=source_fingerprint,
            protocol='Matched images/vocabulary/episode boundaries, method-native settings and backends; single-seed pilot')
        write_json(output/'manifest.json', manifest)
        rows = []
        algorithm_rows = list(csv.DictReader((suite/'algorithm_summary.csv').open()))
        for index, (dataset, method) in enumerate(plan):
            directory = output/dataset/method
            directory.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(worker), '--suite-root', str(suite),
                '--package-root', str(suite/'runtime'), '--baseline-root', str(baseline),
                '--dataset', dataset, '--method', method, '--output', str(directory), '--workers', '2']
            write_json(output/'queue_state.json', dict(status='running', pid=os.getpid(),
                completed=index, total=len(plan), dataset=dataset, method=method, command=command))
            if sha(worker) != script_sha:
                raise RuntimeError('Frozen baseline adapter changed')
            adapter.check_snapshot(source_fingerprint,
                adapter.source_snapshot(suite/'runtime', baseline, controls, worker))
            with (directory.parent/(method+'.launcher.log')).open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=suite/'runtime', pass_fds=tuple(h.fileno() for h in locks))
            row = dict(dataset=dataset, method=method, status='failed', images=128,
                mIoU=None, ms_per_image=None, visual_inputs_per_image=None,
                visual_calls_per_image=None, result_path=str(directory/'result.json'))
            if result.returncode == 0:
                cell = validate_cell(directory, parent_protocol['input_panels'][dataset+'/128'],
                                     method, source_fingerprint)
                row.update(status='completed', mIoU=cell['miou_unrounded'],
                    ms_per_image=cell['ms_per_image'],
                    visual_inputs_per_image=cell.get('visual_inputs_per_image'),
                    visual_calls_per_image=cell.get('visual_calls_per_image'))
            row['returncode'] = result.returncode
            rows.append(row)
            publish(output, rows, algorithm_rows)
            print(f'{index+1}/{len(plan)} {dataset}/{method}: {row["status"]} {row["mIoU"]}', flush=True)
        verify_suite(suite)
        adapter.check_snapshot(source_fingerprint,
            adapter.source_snapshot(suite/'runtime', baseline, controls, worker))
        failures = sum(r['status'] != 'completed' for r in rows)
        write_json(output/'queue_state.json', dict(status='completed_with_failures' if failures else 'completed',
            completed=len(rows), total=len(plan), failures=failures,
            finished_utc=datetime.now(timezone.utc).isoformat()))
        sealed = ['manifest.json', 'baselines.csv', 'comparison.csv', 'queue_state.json']
        sealed += [str(p.relative_to(output)) for p in output.rglob('*')
                   if p.is_file() and p.name in ('protocol.json', 'result.json', 'cost.csv',
                                                 'per_image_counts.json', 'failure.json')]
        write_json(output/'SUITE_SEALED.json', dict(artifact_sha256={p: sha(output/p) for p in sealed},
            automatic_promotion=False, failures=failures))


if __name__ == '__main__':
    main()
