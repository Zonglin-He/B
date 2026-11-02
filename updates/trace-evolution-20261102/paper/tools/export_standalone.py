"""CPU metadata importer for the prelocked six-cell standalone diagnostic.

Run with CUDA_VISIBLE_DEVICES='' and the suite's existing Python environment.
Exit 0: ready/imported; 3: pending, outputs untouched; 2: invalid, outputs untouched.
No model initialization, GPU work, scoring, queue writes, or payload regeneration.
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
import subprocess
import sys
import tempfile
import types


PAPER = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/home/wwww/hzl/results/trace_standalone_followup_20261029_0105')
PREPARED_SHA = '65a9b3d964c68e7b5bae5808feac435a6281941794cb5871d2da5be443c5dab3'
SERVICE = 'trace-standalone-followup-20261029-0105.service'
ARMS = ('Uniform', 'Confidence', 'T', 'C', 'min')
CONDITIONS = ('inherited_sorted', 'common_sorted', 'inherited_regrouped')
DATASETS = ('pc60', 'coco_object')
CONDITION_LABELS = dict(zip(CONDITIONS, ('Retained', 'Common', 'Regrouped')))
DATASET_LABELS = dict(pc60='PC60', coco_object='COCO Object')
OUTPUTS = ('standalone_comparison.tex', 'standalone_cost.tex', 'standalone_provenance.json')
LIMIT = 16 * 1024 * 1024


class Invalid(ValueError):
    pass


class Pending(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def child(root, relative):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Path escape/symlink')
    return path


def read(path):
    require(Path(path).stat().st_size <= LIMIT, f'Metadata exceeds size bound: {path}')
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, f'Duplicate JSON key: {k}')
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(Invalid(f'Nonfinite JSON: {v}')))


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'Invalid {name}')
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, f'Invalid integer {name}')
    return value


def close(actual, expected, name):
    number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), f'{name} differs')


def remember(path, fingerprints, digest=None):
    actual = sha(path)
    require(digest is None or digest == actual, f'Seal/hash mismatch: {path}')
    fingerprints[str(path)] = actual


def check_inventory(root, entries, fingerprints):
    require(isinstance(entries, dict) and entries, 'Empty artifact inventory')
    for relative, digest in entries.items():
        remember(child(root, relative), fingerprints, digest)


def load_queue(root, expected_hash):
    """Only the pinned frozen queue and its read-only validators are invoked."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', "Set CUDA_VISIBLE_DEVICES='' for CPU-only import")
    require(sha(root / 'SUITE_PREPARED.json') == expected_hash, 'Prepared-suite hash mismatch')
    prepared = read(root / 'SUITE_PREPARED.json')
    # Verify the entire bounded source snapshot before executing any helper.
    check_inventory(root / 'runtime', prepared['runtime_hashes'], {})
    path = root / 'runtime/tools/evolution_followup/queue.py'
    module = types.ModuleType('_standalone_frozen_queue')
    module.__file__ = str(path)
    sys.path.insert(0, str(root / 'runtime'))
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    require(module.ROOT == root / 'runtime', 'Frozen queue root differs')
    return module


def service_success(service, root, expected_hash, execution):
    fields = ('Id', 'LoadState', 'ActiveState', 'SubState', 'Result', 'MainPID',
              'ExecMainPID', 'ExecMainCode', 'ExecMainStatus', 'ExecStart')
    response = subprocess.run(['systemctl', '--user', 'show', service, '--property=' + ','.join(fields)],
                              capture_output=True, text=True, check=True, timeout=15)
    state = dict(line.split('=', 1) for line in response.stdout.splitlines() if '=' in line)
    require(set(fields) <= set(state), 'Service status fields unavailable')
    require(state['Id'] == service and state['LoadState'] == 'loaded', 'Service identity unavailable')
    require(f'--suite {root} ' in state['ExecStart'] and f'--suite-sha256 {expected_hash} ' in state['ExecStart']
            and 'tools.evolution_followup.queue run ' in state['ExecStart'], 'Service root/command binding differs')
    require(state['ExecMainPID'] == str(execution['pid']), 'Service execution PID differs')
    if state['MainPID'] != '0' or state['SubState'] in ('running', 'start', 'start-post'):
        raise Pending('Suite service has not reached terminal state')
    require(state['Result'] == 'success' and state['ExecMainCode'] == '1' and state['ExecMainStatus'] == '0'
            and (state['ActiveState'], state['SubState']) in {('active', 'exited'), ('inactive', 'dead')},
            'Service did not exit successfully')
    return state


def lines(path, limit):
    require(path.stat().st_size <= LIMIT, 'JSONL exceeds bounded metadata size')
    with path.open() as stream:
        for index, line in enumerate(stream):
            require(index < limit and len(line) <= 256 * 1024, 'JSONL exceeds row/line bound')
            yield json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(Invalid(v)))


def counts(row, classes):
    i, u = row['intersection'], row['union']
    require(len(i) == len(u) == classes and all(type(a) is int and type(b) is int and 0 <= a <= b
            for a, b in zip(i, u)), 'Invalid intersection/union counts')
    return i, u


def pooled(row, classes):
    i, u = counts(row, classes)
    ratios = [a / b for a, b in zip(i, u) if b]
    require(bool(ratios), 'Null pooled mIoU: no valid union')
    value = 100 * math.fsum(ratios) / len(ratios)
    close(row['miou_unrounded'], value, 'Pooled mIoU')
    return value


def cost_summary(path, manifest, timings):
    batch, steps = manifest['config']['batch_size'], manifest['config']['steps']
    require(type(batch) is int and 0 < batch < 128, 'No non-warmup timing population')
    episodes = (128 + batch - 1) // batch
    counters = ('diagnostic_forward_inputs', 'diagnostic_forward_calls', 'encoder_inputs', 'encoder_calls')
    totals = {a: dict(images=0, measured_images=0, measured_episodes=0, seconds=0.,
                     **{k: 0 for k in counters}) for a in ARMS}
    seen, candidates = set(), {}
    for row in lines(path, 128 * 5):
        e, arm = integer(row['episode'], 'episode'), row['arm']
        require(arm in ARMS and e < episodes and (e, arm) not in seen, 'Cost episode/arm identity differs')
        seen.add((e, arm))
        images = min(batch, 128 - e * batch)
        require(row['images'] == images and row['warmup'] is (e == 0) and row['steps'] == steps,
                'Cost image/warmup/step scope differs')
        order = list(reversed(ARMS)) if e % 2 else list(ARMS)
        require(row['order_position'] == order.index(arm), 'Alternating arm order differs')
        identity = (row['candidate_count'], row['candidate_sha256'], row['episode_sha256'])
        require(identity == candidates.setdefault(e, identity), 'Candidate identity differs across arms')
        n = integer(row['candidate_count'], 'candidate_count')
        factor = 0 if arm in ('Uniform', 'Confidence') else 2 if arm == 'min' else 1
        require(row['diagnostic_forward_inputs'] == row['diagnostic_encoder_inputs'] == factor * n,
                'Actual diagnostic counter schedule differs')
        for k, value in row.items():
            if k.endswith(('_inputs', '_calls')):
                integer(value, k)
        require(row['encoder_inputs'] == row['total_forward_inputs']
                == sum(row[p + '_forward_inputs'] for p in ('eval', 'diagnostic', 'adapt')),
                'Encoder/forward input accounting differs')
        require(row['encoder_calls'] == row['total_forward_calls']
                == sum(row[p + '_forward_calls'] for p in ('eval', 'diagnostic', 'adapt')),
                'Encoder/forward call accounting differs')
        t = totals[arm]
        t['images'] += images
        for k in counters:
            t[k] += integer(row[k], k)
        seconds = number(row['seconds'], 'seconds', positive=True)
        if e:
            t['seconds'] += seconds
            t['measured_images'] += images
            t['measured_episodes'] += 1
    require(seen == {(e, a) for e in range(episodes) for a in ARMS}, 'Incomplete cost stream')
    for arm, t in totals.items():
        timing = timings[arm]
        require(t['images'] == 128 and timing['measured_images'] == t['measured_images'] == 128 - batch
                and timing['measured_episodes'] == t['measured_episodes']
                and timing['warmup_episode'] == 0 and timing['accuracy_includes_warmup'] is True,
                'Timing image denominator differs')
        for key, expected in (('seconds', t['seconds']), ('seconds_per_image', t['seconds'] / t['measured_images']),
                              ('images_per_second', t['measured_images'] / t['seconds'])):
            close(timing[key], expected, key)
        t['ms_per_image'] = 1000 * t['seconds'] / t['measured_images']
        for k in counters:
            t[k + '_per_image'] = t[k] / 128
    return totals


def verify_cell(root, entry, queue, fingerprints):
    directory = child(root, entry['output'])
    manifest = read(child(root, entry['manifest']))
    prediction, scored = (read(directory / n) for n in ('PREDICTIONS_SEALED.json', 'SCORED_SEALED.json'))
    pred_hash = sha(directory / 'PREDICTIONS_SEALED.json')
    worker = queue.worker()
    require(prediction['complete'] is True and prediction['gt_contents_opened'] is False
            and prediction['images'] == 128 and prediction['arms'] == list(ARMS)
            and set(prediction['artifact_sha256']) == worker.prediction_artifacts(manifest),
            'Prediction seal population/scope differs')
    # Queue attests complete payload rehashing. This importer rehashes only small metadata.
    for name in ('prediction_protocol.json', 'initial_state.json', 'prediction_result.json', 'cost.jsonl'):
        remember(directory / name, fingerprints, prediction['artifact_sha256'][name])
    protocol = read(directory / 'prediction_protocol.json')
    require(protocol['manifest'] == manifest and protocol['manifest_sha256'] == entry['manifest_sha256']
            and protocol['manifest_path'] == str(root / entry['manifest'])
            and protocol['selection_seed'] == manifest['selection_seed']
            and protocol['episode_order_seed'] == manifest['episode_order_seed']
            and protocol['ordered_ids'] == manifest['panel']['ordered_ids'], 'Prediction manifest identity differs')
    queue.strict_runtime(protocol['deterministic_runtime'])
    smoke = read(root / f"smoke/{entry['dataset']}.json")
    require(protocol['deterministic_runtime'] == smoke['deterministic_runtime']
            and protocol['gpu_lock']['path'] == str(worker.GPU_LOCK)
            and protocol['gpu_lock']['exclusive'] is True
            and all(protocol['gpu_lock'][k] == smoke['gpu_lock'][k] for k in ('path', 'inode', 'exclusive')),
            'Smoke/runtime/lock identity differs')
    required = {'SCORING_STARTED.json', 'scores_by_episode.jsonl', 'score_inputs.json', 'result.json'}
    require(scored['prediction_seal_sha256'] == pred_hash and set(scored['artifact_sha256']) == required,
            'Scoring seal coverage/chain differs')
    check_inventory(directory, scored['artifact_sha256'], fingerprints)
    require(read(directory / 'SCORING_STARTED.json')['prediction_seal_sha256'] == pred_hash,
            'Scoring-start chain differs')
    result = read(directory / 'result.json')
    require(result['status'] == 'completed' and result['dataset'] == entry['dataset'] and result['seed'] == 0
            and result['images'] == 128 and result['full_split'] is False
            and result['prediction_seal_sha256'] == pred_hash
            and result['label_contents_first_opened_after_all_requested_arms_sealed'] is True
            and result['exposure_note'] == manifest['exposure_note'] == queue.EXPOSURE
            and result['claim_boundary'] == manifest['claim_boundary']
            and set(result['arm_results']) == set(result['timings']) == set(ARMS), 'Canonical result scope differs')
    predicted = read(directory / 'prediction_result.json')
    require(predicted['status'] == 'predicted' and predicted['images'] == 128 and predicted['full_split'] is False
            and predicted['accuracy_includes_warmup'] is True and result['timings'] == predicted['timings']
            and result['timing_scope'] == predicted['timing_scope'] == worker.TIMING_SCOPE,
            'Result timing metadata differs')
    classes = len(manifest['panel']['classes'])
    require(0 < classes <= 4096, 'Class count outside bound')
    sums = {a: [[0] * classes, [0] * classes] for a in ARMS}
    batch, records = manifest['config']['batch_size'], 0
    for e, row in enumerate(lines(directory / 'scores_by_episode.jsonl', 128)):
        require(row['episode'] == e and row['warmup'] is (e == 0) and set(row['arms']) == set(ARMS),
                'Score episode coverage differs')
        for a in ARMS:
            i, u = counts(row['arms'][a], classes)
            sums[a] = [[x + y for x, y in zip(old, new)] for old, new in zip(sums[a], (i, u))]
        records += 1
    require(records == (128 + batch - 1) // batch, 'Missing scored episodes')
    scores = {}
    for a in ARMS:
        metric = result['arm_results'][a]
        require([metric['intersection'], metric['union']] == sums[a], 'Pooled counts differ from sealed episodes')
        scores[a] = pooled(metric, classes)
    require(set(read(directory / 'score_inputs.json')['label_sha256'])
            == {r['label'] for r in manifest['panel']['inputs']}, 'Offline label population differs')
    return dict(dataset=entry['dataset'], condition=entry['condition'], scores=scores,
                delta=scores['min'] - scores['Confidence'], config=manifest['config'],
                timings=result['timings'], costs=cost_summary(directory / 'cost.jsonl', manifest, result['timings']))


def verify_csv(path, cells):
    require(path.stat().st_size < 1024 * 1024, 'CSV exceeds bound')
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    expected = {(c['dataset'], c['condition'], a): c for c in cells for a in ARMS}
    require(len(rows) == 30, 'Expected exactly 30 CSV rows')
    seen = set()
    columns = {'dataset', 'condition', 'arm', 'images', 'full_split', 'miou_unrounded',
               'measured_images', 'seconds', 'seconds_per_image', 'images_per_second'}
    for row in rows:
        require(set(row) == columns, 'CSV columns differ')
        key = row['dataset'], row['condition'], row['arm']
        require(key in expected and key not in seen, 'CSV duplicate/foreign cell-arm link')
        seen.add(key)
        c, arm = expected[key], key[2]
        require(row['images'] == '128' and row['full_split'] == 'False'
                and row['measured_images'] == str(c['timings'][arm]['measured_images']), 'CSV count/split link differs')
        close(float(row['miou_unrounded']), c['scores'][arm], 'CSV mIoU link')
        for name in ('seconds', 'seconds_per_image', 'images_per_second'):
            close(float(row[name]), c['timings'][arm][name], 'CSV timing link')
    require(seen == set(expected), 'CSV population incomplete')


def render(cells):
    def table_body(caption, label, headings, rows):
        return '\n'.join(['\\caption{' + caption + '}', '\\label{' + label + '}',
            r'\centering\small', r'\begin{tabular}{@{}ll' + 'r' * (len(headings) - 2) + '@{}}',
            r'\toprule', ' & '.join(headings) + r' \\', r'\midrule', *rows,
            r'\bottomrule', r'\end{tabular}', ''])
    def table(caption, label, headings, rows):
        return r'\begin{table*}[t]' + '\n' + table_body(caption, label, headings, rows) + r'\end{table*}' + '\n'
    def prefix(c):
        return DATASET_LABELS[c['dataset']] + ' & ' + CONDITION_LABELS[c['condition']]
    rows = []
    for c in cells:
        best = max(c['scores'].values())
        metrics = [('$\\mathbf{' + f'{c["scores"][a]:.2f}' + '}$') if c['scores'][a] == best
                   else f'${c["scores"][a]:.2f}$' for a in ARMS]
        rows.append(prefix(c) + ' & ' + ' & '.join(metrics) + f' & ${c["delta"]:+.2f}$' + r' \\')
    scope = (r'Fixed 128-image diagnostic per dataset using reused, development-exposed evaluation images; '
             r'not full-split or independent generalization evidence. Retained uses inherited settings, '
             r'Common uses prelocked batch 2, LR 0.001, 10 steps, $K=16$, view batch 16, and Regrouped '
             r'changes episode ordering at retained settings. Vocabulary is retained in every condition. ')
    comparison = table(scope + r'Pooled mIoU ($\uparrow$) includes every image, including warmup; '
        r'bold marks the actual unrounded row maximum. $\Delta$ is min minus Confidence in percentage points. '
        r'All conditions were locked before outcomes; no significance, necessity, or full-split claim.',
        'tab:standalone-comparison', ['Dataset', 'Condition', *ARMS[:-1], 'TRACE (min)', r'$\Delta$'], rows)
    time_rows, work_rows = [], []
    for c in cells:
        n = c['timings']['min']['measured_images']
        time_rows.append(prefix(c) + f' & {n}' + ' & ' + ' & '.join(
            f'{c["costs"][a]["ms_per_image"]:.2f}' for a in ARMS) + r' \\')
        work_rows.append(prefix(c) + ' & ' + ' & '.join(
            f'{c["costs"][a]["diagnostic_forward_inputs_per_image"]:.2f} / '
            f'{c["costs"][a]["encoder_inputs_per_image"]:.2f}' for a in ARMS) + r' \\')
    cost = r'\begin{table*}[t]' + '\n' + table_body(scope + r'(a) Model-side synchronized ms/image for independent standalone arms. '
        r'Only episode 0 is excluded from time: $N_t=128-\mathrm{batch}$ is shown explicitly. '
        r'Includes reset, candidate construction, required diagnostics, adaptation, final prediction and CPU '
        r'prediction transfer; excludes setup/model loading, RGB decode, H2D/geometry preparation, artifact '
        r'writes and offline scoring. Not end-to-end throughput or a full-split speedup claim.',
        'tab:standalone-cost', ['Dataset', 'Condition', '$N_t$', *ARMS[:-1], 'TRACE (min)'], time_rows)
    cost += '\n' + r'\par\medskip' + '\n' + table_body(r'(b) Actual diagnostic-forward inputs / total encoder inputs per source image. '
        r'Work denominators are all 128 images, including episode 0, unlike timing. '
        r'Counters are summed from the bounded sealed cost stream, not inferred from nominal view schedules. '
        r'Encoder and diagnostic call totals are retained separately in provenance.',
        'tab:standalone-work', ['Dataset', 'Condition', *ARMS[:-1], 'TRACE (min)'], work_rows)
    cost += r'\end{table*}' + '\n'
    return dict(zip(OUTPUTS[:2], (comparison, cost)))


def publish(paper, outputs):
    directory = paper / 'generated'
    require(directory.is_dir() and set(outputs) == set(OUTPUTS), 'Wrong publication directory/output set')
    old = {n: child(directory, n).read_bytes() if (directory / n).exists() else None for n in outputs}
    replaced = []
    with tempfile.TemporaryDirectory(prefix='.standalone-', dir=directory) as temp:
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
    root, paper = Path(root).resolve(), Path(paper).resolve()
    queue = load_queue(root, expected_hash)
    prepared = queue.verify_prepared(root, expected_hash)
    fingerprints = {str(Path(__file__).resolve()): sha(Path(__file__))}
    check_inventory(root / 'runtime', prepared['runtime_hashes'], fingerprints)
    check_inventory(root, prepared['artifact_sha256'], fingerprints)
    remember(root / 'SUITE_PREPARED.json', fingerprints, expected_hash)
    if not (root / 'SUITE_SEALED.json').exists():
        state = read(root / 'queue_state.json')
        require(state.get('phase') != 'failed', 'Queue failed; no import permitted')
        return dict(status='pending', tables_unchanged=True, queue_state=state,
                    reason='Whole SUITE_SEALED.json absent; no cell scores/payloads imported'), 3
    seal = read(root / 'SUITE_SEALED.json')
    require(seal['complete'] is True and seal['formal_cells'] == 6 and seal['arm_results'] == 30
            and seal['passed_gt_free_gates'] == 2 and seal['full_split'] is False
            and seal['suite_prepared_sha256'] == expected_hash and seal['runtime_hashes'] == prepared['runtime_hashes']
            and seal['payload_verification'] == 'All cell prediction and score payloads rehashed via their linked seals'
            and seal['exposure_note'] == queue.EXPOSURE, 'Whole-suite seal scope differs')
    required = {'SUITE_PREPARED.json', 'EXECUTION_STARTED.json', 'ALL_PREDICTIONS_SEALED.json',
                'rows.csv', 'execution_children.json', 'queue_state.json', *prepared['artifact_sha256']}
    required.update(e['output'] for e in prepared['smokes'])
    required.update(e['output'] + '/' + n for e in prepared['cells']
                    for n in ('PREDICTIONS_SEALED.json', 'SCORED_SEALED.json', 'result.json'))
    require(required <= set(seal['artifact_sha256']), 'Whole-suite seal lacks canonical coverage')
    check_inventory(root, seal['artifact_sha256'], fingerprints)
    remember(root / 'SUITE_SEALED.json', fingerprints)
    execution = read(root / 'EXECUTION_STARTED.json')
    require(execution['suite_sha256'] == expected_hash, 'Execution prepared-root link differs')
    observed = service_success(service, root, expected_hash, execution)
    state = read(root / 'queue_state.json')
    require(state['phase'] == 'completed' and state['passed_smokes'] == 2 and state['sealed_predictions'] == 6
            and state['scored_cells'] == state['total_cells'] == 6 and state['error'] is None, 'Queue not fully complete')
    barrier = read(root / 'ALL_PREDICTIONS_SEALED.json')
    require(barrier['suite_sha256'] == expected_hash and barrier['labels_opened_by_queue'] is False,
            'All-prediction barrier differs')
    gates = {e['key']: queue.validate_smoke(root, e) for e in prepared['smokes']}
    pred_hashes = {e['key']: sha(root / e['output'] / 'PREDICTIONS_SEALED.json') for e in prepared['cells']}
    require(barrier['smoke_sha256'] == gates and barrier['prediction_seal_sha256'] == pred_hashes,
            'Prediction barrier population/link differs')
    receipts = read(root / 'execution_children.json')
    require(len(receipts) == 14 and all(r['exit_code'] == 0 for r in receipts), 'Execution child receipts incomplete')
    cells = [verify_cell(root, e, queue, fingerprints) for e in prepared['cells']]
    verify_csv(root / 'rows.csv', cells)
    outputs = render(cells)
    provenance = dict(schema=1, suite_root=str(root), suite_prepared_sha256=expected_hash,
        suite_sealed_sha256=sha(root / 'SUITE_SEALED.json'), service=observed, cells=cells,
        source_sha256=fingerprints, runtime_hashes=prepared['runtime_hashes'],
        exposure=queue.EXPOSURE, all_predictions_barrier=barrier,
        verification_scope='Pinned frozen queue preparation/smoke validators; runtime/config/manifest/root and small metadata/seal chains rehashed; pooled episode counts, cost counters and CSV links independently recomputed. Prediction NPY payloads, per-episode candidate receipts, raw images/labels and checkpoint weights are NOT reread. Their hash verification relies on the pinned queue whole-suite seal plus root-bound real service successful terminal exit.',
        uncertainty='Descriptive dependent reused-image diagnostic only; no bootstrap, significance, necessity or independent-generalization inference.',
        output_sha256={n: hashlib.sha256(v.encode()).hexdigest() for n, v in outputs.items()})
    for path, digest in fingerprints.items():
        require(sha(path) == digest, f'Source drift during import: {path}')
    service_success(service, root, expected_hash, execution)
    outputs[OUTPUTS[2]] = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if not check_only:
        publish(paper, outputs)
    return dict(status='ready' if check_only else 'imported', tables_unchanged=check_only,
                outputs=list(OUTPUTS), cells=6, arm_results=30,
                deltas={c['dataset'] + '/' + c['condition']: c['delta'] for c in cells}), 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--suite-sha256', default=PREPARED_SHA)
    parser.add_argument('--service', default=SERVICE)
    parser.add_argument('--paper-root', type=Path, default=PAPER)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        status, code = run(args.suite_root, args.paper_root, args.suite_sha256, args.service, args.check_only)
    except Pending as error:
        status, code = dict(status='pending', reason=str(error), tables_unchanged=True), 3
    except (Invalid, OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        status, code = dict(status='invalid', error=str(error), tables_unchanged=True), 2
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
