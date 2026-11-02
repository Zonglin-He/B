"""Freeze and serially execute local native baselines after the mechanism suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType

from tools.evolution_local_baselines import common as c


def now():
    return datetime.now(timezone.utc).isoformat()


def state(suite, phase, **fields):
    c.writable_path(suite, suite / 'queue_state.json')
    c.write_json(suite / 'queue_state.json', dict(phase=phase, pid=os.getpid(), updated_at=now(), **fields), replace=True)


def archive_inventory():
    c.require(c.sha256(c.ARCHIVE / 'source_provenance.json') == c.ARCHIVE_PROVENANCE_SHA, 'Archived provenance pin differs')
    source = c.read_json(c.ARCHIVE / 'source_provenance.json')
    c.require(source['legacy_commit'] == 'ba9f57bf47846ec727c616c5163233a98ca0bf94', 'Wrong archived native commit')
    hashes = dict(source['baseline_sources'])
    hashes.update(c.BASELINE_EXTRA)
    for name, expected in hashes.items():
        c.require(c.sha256(c.BASELINE_ARCHIVE / name) == expected, 'Archived baseline source changed')
    import yaml
    c.require(yaml.safe_load((c.BASELINE_ARCHIVE / 'prompts.yaml').read_text())['prompt_templates'] == c.PROMPTS,
              'Native seven prompt templates differ')
    return hashes, source


def parent_report():
    path = c.ROOT / 'paper_evolution/tools/export_tuned_full.py'
    contents = path.read_bytes()
    module = ModuleType('local_baseline_parent_importer')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(contents, str(path), 'exec'), module.__dict__)
    report = module.import_suite(c.PARENT)
    c.require(hashlib.sha256(contents).hexdigest() == report['importer_sha256'] == c.sha256(path), 'Executed importer changed')
    return report


def prepare(args):
    from tools.evolution_followup.queue import snapshot
    suite = args.output.absolute()
    c.require(suite == suite.resolve() and suite.parent == c.PARENT.parent
              and suite.name.startswith('trace_local_baselines_') and not suite.exists(), 'New isolated output required')
    c.require(c.sha256(c.PARENT / 'SUITE_PREPARED.json') == c.PARENT_SHA
              and c.sha256(c.DEPENDENCY / 'SUITE_PREPARED.json') == c.DEPENDENCY_SHA, 'Wrong parent/dependency')
    native, provenance = archive_inventory()
    report = parent_report()
    c.require(len(report['cells']) == 8 and report['arm_results'] == 32, 'Eight completed TRACE populations required')
    suite.mkdir()
    state(suite, 'preparing')
    inventory = snapshot(c.ROOT, suite / 'runtime')
    dependency = c.read_json(c.DEPENDENCY / 'SUITE_PREPARED.json')
    c.require(all(inventory.get(k) == v for k, v in dependency['runtime_hashes'].items()), 'Frozen mechanism source changed')
    for name in native:
        target = suite / 'baseline_source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.BASELINE_ARCHIVE / name, target)
        c.require(c.sha256(target) == native[name], 'Native copy differs')
    shutil.copyfile(c.ARCHIVE / 'source_provenance.json', suite / 'archive_provenance.json')
    shutil.copyfile(c.ROOT / 'docs/local-baseline-rerun-contract.md', suite / 'frozen_contract.md')
    shutil.copyfile(c.ROOT / 'paper_evolution/tools/export_tuned_full.py', suite / 'parent_importer.py')
    c.require(c.sha256(suite / 'parent_importer.py') == report['importer_sha256'], 'Executed importer copy drift')
    copied, originals, entries = {}, {}, {}
    sources = {f'parents/{name}': c.PARENT / name for name in
               ('SUITE_PREPARED.json', 'SUITE_SEALED.json', 'ALL_CONFIGS_SELECTED.json', 'ALL_FULL_PREDICTIONS_SEALED.json')}
    for dataset in c.DATASETS:
        for name in ('prediction_protocol.json', 'PREDICTIONS_SEALED.json', 'SCORED_SEALED.json', 'result.json'):
            sources[f'parents/{dataset}/{name}'] = c.PARENT / 'full' / dataset / name
        sources[f'parents/{dataset}/SELECTED_CONFIG_SEALED.json'] = c.PARENT / 'selections' / dataset / 'SELECTED_CONFIG_SEALED.json'
    for name, original in sources.items():
        target = suite / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        value = c.sha256(target)
        c.require(value == report['metadata_sha256'].get(str(original)) == c.sha256(original), 'Copy outside verified parent read-set')
        copied[name], originals[name] = value, str(original)
    for dataset in c.DATASETS:
        parent = c.read_json(suite / f'parents/{dataset}/prediction_protocol.json')['manifest']
        path = suite / f'parents/{dataset}/panel.json'
        c.write_json(path, parent['panel'])
        copied[str(path.relative_to(suite))] = c.sha256(path)
        entries[dataset] = dict(panel_path=str(path.relative_to(suite)), panel_sha256=c.sha256(path), config=parent['config'])
    c.write_json(suite / 'PARENT_IMPORT.json', dict(parent_suite=str(c.PARENT), parent_prepared_sha256=c.PARENT_SHA,
        verified_parent_populations=8, verified_parent_full_outcomes=32, metadata_sha256=report['metadata_sha256'],
        parent_original_paths=originals, importer_sha256=report['importer_sha256'],
        payload_scope='Verified metadata/pooled counts; original frozen controller certifies parent payloads'))
    prepared = dict(schema_version=1, algorithm=c.ALGORITHM, jobs=c.JOBS, parameter_search=False, retry=False,
        parent_suite=str(c.PARENT), parent_prepared_sha256=c.PARENT_SHA, dependency_suite=str(c.DEPENDENCY),
        dependency_prepared_sha256=c.DEPENDENCY_SHA, dependency_service=c.DEPENDENCY_SERVICE,
        runtime_hashes=inventory, baseline_hashes=native, entries=entries, checkpoint=dependency['checkpoint'],
        parent_proof_sha256=c.sha256(suite / 'PARENT_IMPORT.json'), parent_artifact_sha256=copied,
        archive_provenance_sha256=c.sha256(suite / 'archive_provenance.json'),
        contract_sha256=c.sha256(suite / 'frozen_contract.md'), python=str(args.python.resolve(strict=True)),
        service_name=args.service, resources=c.RESOURCES, created_at=now(),
        development_exposed=True, waterbirds_development_overlap=256,
        reported_datasets=['voc20', 'voc21', 'pc59', 'pc60', 'coco_object', 'coco_stuff'])
    c.write_json(suite / 'SUITE_PREPARED.json', prepared)
    identity = c.sha256(suite / 'SUITE_PREPARED.json')
    (suite / 'manifests').mkdir()
    for dataset, method in c.JOBS:
        for stage in ('gate', 'full'):
            c.write_json(c.manifest_path(suite, dataset, method, stage),
                         c.expected_manifest(suite, prepared, dataset, method, stage, identity))
    c.verify_prepared(suite, identity)
    state(suite, 'prepared', suite_prepared_sha256=identity, gates=18, full_cells=18, gpu_work_started=False)
    print(dict(suite=str(suite), prepared_sha256=identity, full_cells=18), flush=True)


def service_fields(name):
    fields = (*c.RESOURCES, 'Id', 'LoadState', 'ActiveState', 'SubState', 'MainPID', 'ExecMainPID',
              'ExecMainCode', 'ExecMainStatus', 'Result', 'InvocationID', 'ControlGroup', 'WorkingDirectory', 'ExecStart')
    result = subprocess.run(['systemctl', '--user', 'show', name, '--property=' + ','.join(fields)],
                            text=True, capture_output=True, check=True, timeout=30)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def execution_identity(suite, prepared):
    invocation = os.environ.get('INVOCATION_ID', '')
    command = Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    group = next((x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')), None)
    actual = service_fields(prepared['service_name'])
    c.require(len(invocation) == 32 and command == c.expected_command(suite, prepared)
              and group and group.endswith('/' + prepared['service_name'])
              and actual['ActiveState'] == 'active' and actual['SubState'] == 'running'
              and actual['MainPID'] == actual['ExecMainPID'] == str(os.getpid())
              and actual['InvocationID'] == invocation and actual['ControlGroup'] == group
              and actual['WorkingDirectory'] == str(suite / 'runtime') == str(Path.cwd().resolve())
              and all(actual[k] == v for k, v in c.RESOURCES.items())
              and 'argv[]=' + ' '.join(command) + ' ;' in actual['ExecStart'], 'Actual controller envelope differs')
    return dict(pid=os.getpid(), invocation_id=invocation, command=command, service=prepared['service_name'],
        control_group=group, cwd=str(Path.cwd().resolve()), resources=actual,
        suite_prepared_sha256=prepared['_verified_sha256'], retry_allowed=False, started_at=now())


def bound_service_terminal(suite, prepared, execution):
    actual = service_fields(prepared['service_name'])
    command = 'argv[]=' + ' '.join(c.expected_command(suite, prepared)) + ' ;'
    c.require(actual['Id'] == prepared['service_name'] == execution['service']
              and actual['LoadState'] == 'loaded' and actual['ActiveState'] == 'active'
              and actual['SubState'] == 'exited' and actual['MainPID'] == '0'
              and actual['Result'] == 'success' and actual['ExecMainCode'] == '1' and actual['ExecMainStatus'] == '0'
              and actual['ExecMainPID'] == str(execution['pid']) and actual['InvocationID'] == execution['invocation_id']
              and actual['WorkingDirectory'] == execution['cwd'] == str(suite / 'runtime')
              and actual['ControlGroup'] in ('', execution['control_group']) and command in actual['ExecStart']
              and all(actual[k] == v == execution['resources'][k] for k, v in c.RESOURCES.items())
              and execution['command'] == c.expected_command(suite, prepared)
              and execution['suite_prepared_sha256'] == prepared['_verified_sha256']
              and execution['resources']['ControlGroup'] == execution['control_group']
              and execution['resources']['InvocationID'] == execution['invocation_id']
              and execution['retry_allowed'] is False, 'Actual terminal not certified')
    return actual


def wait_dependency(suite):
    from tools.evolution_allocation import analyze as original
    previous = None
    while True:
        actual = service_fields(c.DEPENDENCY_SERVICE)
        if actual['LoadState'] == 'loaded' and actual['ActiveState'] == 'active' and actual['SubState'] == 'exited':
            state(suite, 'certifying_dependency', dependency=str(c.DEPENDENCY))
            report = original.analyze(c.DEPENDENCY, c.DEPENDENCY_SHA, c.DEPENDENCY_SERVICE)
            c.require(report['full_outcomes'] == 10, 'Dependency mechanism outcomes incomplete')
            c.write_json(suite / 'DEPENDENCY_COMPLETED.json', report)
            return
        c.require(actual['LoadState'] == 'loaded' and actual['ActiveState'] in ('active', 'activating')
                  and actual['SubState'] in ('running', 'start'), 'Upstream failed/not present; no retry')
        if previous != (actual['MainPID'], actual['InvocationID']):
            state(suite, 'waiting_for_mechanism_suite', dependency=str(c.DEPENDENCY), service=c.DEPENDENCY_SERVICE)
            previous = actual['MainPID'], actual['InvocationID']
        time.sleep(30)


def child(suite, prepared, phase, dataset, method):
    c.verify_prepared(suite, prepared['_verified_sha256'], frozen=True)
    path = c.manifest_path(suite, dataset, method, 'gate' if phase == 'gate' else 'full')
    manifest = c.read_json(path)
    logs = suite / 'logs'
    logs.mkdir(exist_ok=True)
    log = c.writable_path(suite, logs / f'{dataset}.{method}.{phase}.log')
    receipt = c.writable_path(suite, logs / f'{dataset}.{method}.{phase}.json')
    c.require(not log.exists() and not receipt.exists(), 'No child retry')
    command = [prepared['python'], '-B', '-m', 'tools.evolution_local_baselines.worker', '--phase', phase,
        '--manifest', str(path), '--manifest-sha256', c.sha256(path), '--output', manifest['output']]
    if phase in ('score', 'verify'):
        command += ['--barrier', str(suite / 'ALL_FULL_PREDICTIONS_SEALED.json')]
    env = dict(os.environ, PYTHONPATH=str(suite / 'runtime'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if phase in ('score', 'verify'):
        env['CUDA_VISIBLE_DEVICES'] = ''
    state(suite, phase, dataset=dataset, method=method, command=command, active_output=manifest['output'])
    started = now()
    with log.open('x') as stream:
        completed = subprocess.run(command, cwd=suite / 'runtime', env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
    c.write_json(receipt, dict(command=command, returncode=completed.returncode, started_at=started, ended_at=now(),
                             log_sha256=c.sha256(log), manifest_sha256=c.sha256(path)))
    c.require(completed.returncode == 0, f'Baseline worker failed; no retry: {receipt}')


def validate_gate(suite, prepared, dataset, method):
    path = c.manifest_path(suite, dataset, method, 'gate')
    output = suite / 'gates' / f'{dataset}.{method}'
    report, seal = c.read_json(output / 'gate.json'), c.read_json(output / 'GATE_PASSED.json')
    c.require(seal == dict(report, gate_sha256=c.sha256(output / 'gate.json')), 'Gate not complete/bound')
    c.validate_gate_report(output, c.read_json(path), report)
    return c.sha256(output / 'GATE_PASSED.json')


def validate_results(suite, prepared):
    from tools.evolution_local_baselines import worker
    rows, seals = [], {}
    for dataset, method in c.JOBS:
        manifest_path = c.manifest_path(suite, dataset, method, 'full')
        manifest = c.read_json(manifest_path)
        output = Path(manifest['output'])
        result = worker.verify_score_seal(output, manifest, c.sha256(manifest_path),
                                          suite / 'ALL_FULL_PREDICTIONS_SEALED.json')
        c.require(result['status'] == 'completed' and result['images'] == c.POPULATIONS[dataset], 'Canonical population incomplete')
        rows.append(dict(dataset=dataset, method=method, images=result['images'],
            mIoU=result['miou_unrounded'], result_path=str(output / 'result.json')))
        seals[f'{dataset}.{method}'] = c.sha256(output / 'SCORED_SEALED.json')
    c.require(len(rows) == 18, 'Eighteen canonical results required')
    return rows, seals


def run(args):
    suite = c.absolute_path(args.suite)
    prepared = c.verify_prepared(suite, args.suite_sha256, frozen=True)
    c.require(not (suite / 'EXECUTION_STARTED.json').exists(), 'No controller retry/resume')
    with c.writable_path(suite, suite / 'queue.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = execution_identity(suite, prepared)
        c.write_json(suite / 'EXECUTION_STARTED.json', execution)
        try:
            wait_dependency(suite)
            gates = {}
            for dataset, method in c.JOBS:
                child(suite, prepared, 'gate', dataset, method)
                gates[f'{dataset}.{method}'] = validate_gate(suite, prepared, dataset, method)
            c.write_json(suite / 'ALL_GATES_PASSED.json', dict(complete=True, suite_prepared_sha256=args.suite_sha256, gates=gates))
            predicted = {}
            from tools.evolution_local_baselines import worker
            for dataset, method in c.JOBS:
                child(suite, prepared, 'predict', dataset, method)
                path = c.manifest_path(suite, dataset, method, 'full')
                manifest = c.read_json(path)
                worker.verify_prediction_seal(Path(manifest['output']), manifest, c.sha256(path))
                predicted[f'{dataset}.{method}'] = c.sha256(Path(manifest['output']) / 'PREDICTIONS_SEALED.json')
            c.write_json(suite / 'ALL_FULL_PREDICTIONS_SEALED.json', dict(complete=True, gt_contents_opened=False,
                suite_prepared_sha256=args.suite_sha256, predictions=predicted))
            for dataset, method in c.JOBS:
                child(suite, prepared, 'score', dataset, method)
                child(suite, prepared, 'verify', dataset, method)
            rows, seals = validate_results(suite, prepared)
            c.verify_prepared(suite, args.suite_sha256, frozen=True)
            with c.writable_path(suite, suite / 'full_metrics.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            c.write_json(suite / 'SUITE_SEALED.json', dict(complete=True, full_outcomes=18,
                suite_prepared_sha256=args.suite_sha256, score_seals=seals,
                full_metrics_sha256=c.sha256(suite / 'full_metrics.csv'),
                prediction_barrier_sha256=c.sha256(suite / 'ALL_FULL_PREDICTIONS_SEALED.json'),
                execution_started_sha256=c.sha256(suite / 'EXECUTION_STARTED.json'),
                dependency_completed_sha256=c.sha256(suite / 'DEPENDENCY_COMPLETED.json'), service_exit_certified=False))
            state(suite, 'completed', full_outcomes=18, service_exit_certified=False)
        except BaseException as error:
            state(suite, 'failed', error=f'{type(error).__name__}: {error}', retry_allowed=False)
            raise


def certify_completed(suite, expected):
    suite = c.absolute_path(suite)
    prepared = c.verify_prepared(suite, expected)
    execution = c.read_json(suite / 'EXECUTION_STARTED.json')
    terminal = bound_service_terminal(suite, prepared, execution)
    sealed, progress = c.read_json(suite / 'SUITE_SEALED.json'), c.read_json(suite / 'queue_state.json')
    c.require(progress['phase'] == 'completed' and progress['pid'] == execution['pid']
              and sealed['complete'] is True and sealed['full_outcomes'] == 18
              and sealed['suite_prepared_sha256'] == expected and sealed['service_exit_certified'] is False
              and sealed['full_metrics_sha256'] == c.sha256(suite / 'full_metrics.csv')
              and sealed['prediction_barrier_sha256'] == c.sha256(suite / 'ALL_FULL_PREDICTIONS_SEALED.json')
              and sealed['execution_started_sha256'] == c.sha256(suite / 'EXECUTION_STARTED.json')
              and sealed['dependency_completed_sha256'] == c.sha256(suite / 'DEPENDENCY_COMPLETED.json'),
              'Final local baseline suite not complete/bound')
    dependency = c.read_json(suite / 'DEPENDENCY_COMPLETED.json')
    c.require(dependency['suite'] == str(c.DEPENDENCY) and dependency['suite_prepared_sha256'] == c.DEPENDENCY_SHA
              and dependency['full_outcomes'] == 10, 'Wrong completed dependency')
    rows, scores = validate_results(suite, prepared)
    c.require(scores == sealed['score_seals'], 'Final canonical score seals differ')
    with (suite / 'full_metrics.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    c.require(len(csv_rows) == 18 and all(
        actual['dataset'] == expected_row['dataset'] and actual['method'] == expected_row['method']
        and int(actual['images']) == expected_row['images'] and float(actual['mIoU']) == expected_row['mIoU']
        and actual['result_path'] == expected_row['result_path'] for actual, expected_row in zip(csv_rows, rows, strict=True)),
        'Canonical CSV differs from verified results')
    c.require(bound_service_terminal(suite, prepared, execution) == terminal, 'Terminal changed during verification')
    return dict(suite=str(suite), suite_prepared_sha256=expected, full_outcomes=18,
        image_method_evaluations=sum(row['images'] for row in rows), rows=rows,
        suite_seal_sha256=c.sha256(suite / 'SUITE_SEALED.json'), service_terminal=terminal,
        service_exit_certified=True, development_exposed=True, independent_test_claim=False,
        comparison_scope='Native fixed settings, local input geometry; not equal tuning budget or external-paper identity')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    setup = sub.add_parser('prepare')
    setup.add_argument('--output', type=Path, required=True)
    setup.add_argument('--service', required=True)
    setup.add_argument('--python', type=Path, default=Path('/home/wwww/miniconda3/envs/mlmp/bin/python'))
    execute = sub.add_parser('run')
    execute.add_argument('--suite', type=Path, required=True)
    execute.add_argument('--suite-sha256', required=True)
    verify = sub.add_parser('certify')
    verify.add_argument('--suite', type=Path, required=True)
    verify.add_argument('--suite-sha256', required=True)
    args = parser.parse_args(argv)
    if args.phase == 'prepare':
        return prepare(args)
    if args.phase == 'run':
        return run(args)
    import json
    print(json.dumps(certify_completed(args.suite, args.suite_sha256), allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
