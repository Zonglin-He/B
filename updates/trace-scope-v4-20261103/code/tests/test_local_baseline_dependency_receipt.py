"""Receipt reuse authenticates metadata/exit without recertifying payloads."""

import builtins
import copy
import csv
from pathlib import Path
import subprocess
import sys

import pytest

from tools.evolution_allocation import common as upstream
from tools.evolution_allocation import queue as upstream_queue
from tools.evolution_local_baselines import common as c
from tools.evolution_local_baselines import dependency_receipt as dr


@pytest.fixture
def certified(tmp_path, monkeypatch):
    root = tmp_path / 'dependency'
    root.mkdir()
    monkeypatch.setattr(c, 'DEPENDENCY', root)
    execution = dict(pid=123)
    c.write_json(root / 'EXECUTION_STARTED.json', execution)
    c.write_json(root / 'queue_state.json', dict(phase='completed', pid=123))
    c.write_json(root / 'SUITE_PREPARED.json', dict(synthetic=True))
    identity = c.sha256(root / 'SUITE_PREPARED.json')
    monkeypatch.setattr(c, 'DEPENDENCY_SHA', identity)
    terminal = dict(LoadState='loaded', ActiveState='active', SubState='exited', MainPID='0',
                    Result='success', ExecMainStatus='0', InvocationID='a'*32)
    monkeypatch.setattr(upstream, 'verify_prepared', lambda *args: dict(_verified_sha256=identity))
    monkeypatch.setattr(upstream_queue, 'bound_service_terminal', lambda *args: terminal.copy())
    links, score_links, rows = {}, {}, []
    for dataset in upstream.DATASETS:
        output = root / 'full' / dataset
        output.mkdir(parents=True)
        c.write_json(output / 'PREDICTIONS_SEALED.json', dict(complete=True))
        links[str(output)] = c.sha256(output / 'PREDICTIONS_SEALED.json')
        results = {arm: dict(miou_unrounded=float(index+1)) for index, arm in enumerate(upstream.ARMS)}
        c.write_json(output / 'result.json', dict(status='completed', images=upstream.POPULATIONS[dataset],
                                                arm_results=results))
        for arm in upstream.ARMS:
            rows.append(dict(dataset=dataset, arm=arm, images=upstream.POPULATIONS[dataset],
                mIoU=results[arm]['miou_unrounded'], result_path=str(output / 'result.json')))
    c.write_json(root / 'ALL_FULL_PREDICTIONS_SEALED.json', dict(complete=True, gt_contents_opened=False,
        suite_prepared_sha256=identity, prediction_seal_sha256=links))
    barrier = c.sha256(root / 'ALL_FULL_PREDICTIONS_SEALED.json')
    for dataset in upstream.DATASETS:
        output = root / 'full' / dataset
        c.write_json(output / 'SCORED_SEALED.json', dict(complete=True, barrier_sha256=barrier,
            prediction_seal_sha256=links[str(output)], artifact_sha256={'result.json': c.sha256(output / 'result.json')}))
        score_links[dataset] = c.sha256(output / 'SCORED_SEALED.json')
    with (root / 'full_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    c.write_json(root / 'SUITE_SEALED.json', dict(complete=True, full_outcomes=10,
        suite_prepared_sha256=identity, datasets=list(upstream.DATASETS), arms=upstream.ARMS,
        execution_started_sha256=c.sha256(root / 'EXECUTION_STARTED.json'),
        full_metrics_sha256=c.sha256(root / 'full_metrics.csv'), barrier_sha256=barrier, score_seals=score_links))
    monkeypatch.setattr(dr, 'SUITE_SEAL_SHA', c.sha256(root / 'SUITE_SEALED.json'))
    receipt = tmp_path / 'old_receipt.json'
    report = dict(suite=str(root), suite_prepared_sha256=identity, full_outcomes=10, rows=rows,
                  service_terminal=terminal.copy())
    c.write_json(receipt, report)
    monkeypatch.setattr(dr, 'RECEIPT', receipt)
    monkeypatch.setattr(dr, 'RECEIPT_SHA', c.sha256(receipt))
    return root, receipt, report, terminal


def test_receipt_reuse_metadata_only(certified, monkeypatch):
    root, receipt, original, _ = certified
    original_bytes = receipt.read_bytes()
    real_open = builtins.open
    real_path_open = Path.open
    def deny_payload(path):
        name = str(path)
        if '/predictions/' in name or name.endswith(('.npy', '.png', '.jpg')) or 'scores_by_' in name or 'per_image_counts' in name:
            raise AssertionError('Payload/GT reread: ' + name)
    def safe_open(path, *args, **kwargs):
        deny_payload(path)
        return real_open(path, *args, **kwargs)
    def safe_path_open(path, *args, **kwargs):
        deny_payload(path)
        return real_path_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', safe_open)
    monkeypatch.setattr(Path, 'open', safe_path_open)
    monkeypatch.setattr(upstream_queue, 'validate_results', lambda *args: (_ for _ in ()).throw(AssertionError('No deep scan')))
    result = dr.verify_completed_receipt()
    assert result['rows'] == original['rows'] and result['full_outcomes'] == 10
    assert result['certification_reuse']['payload_rehashed'] is False
    assert receipt.read_bytes() == original_bytes


@pytest.mark.parametrize('name', ['SUITE_SEALED.json', 'full_metrics.csv', 'EXECUTION_STARTED.json',
    'ALL_FULL_PREDICTIONS_SEALED.json', 'full/coco_object/SCORED_SEALED.json',
    'full/pc60/PREDICTIONS_SEALED.json', 'full/pc60/result.json'])
def test_metadata_drift_rejected(certified, name):
    root, _, _, _ = certified
    (root / name).write_text('{}')
    with pytest.raises((RuntimeError, KeyError)):
        dr.verify_completed_receipt()


def test_receipt_drift_or_different_terminal_rejected(certified):
    _, receipt, _, terminal = certified
    terminal['InvocationID'] = 'b'*32
    with pytest.raises(RuntimeError, match='invocation changed'):
        dr.verify_completed_receipt()
    receipt.write_text('{}')
    with pytest.raises(RuntimeError, match='receipt digest'):
        dr.verify_completed_receipt()


def test_copied_receipt_required_and_authenticated(certified, tmp_path):
    _, receipt, _, _ = certified
    new = tmp_path / 'new_suite'
    new.mkdir()
    with pytest.raises(FileNotFoundError):
        dr.verify_completed_receipt(new)
    (new / 'dependency_receipt.json').write_bytes(receipt.read_bytes())
    assert dr.verify_completed_receipt(new)['full_outcomes'] == 10
    (new / 'dependency_receipt.json').write_text('{}')
    with pytest.raises(RuntimeError, match='receipt digest'):
        dr.verify_completed_receipt(new)


def test_real_upstream_read_set_and_copied_wait_path(tmp_path):
    script = r'''
import builtins
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from tools.evolution_allocation import common as upstream
from tools.evolution_allocation import queue as upstream_queue
from tools.evolution_local_baselines import common as c
from tools.evolution_local_baselines import dependency_receipt as dr
from tools.evolution_local_baselines import queue as q

root, target = c.DEPENDENCY, Path(sys.argv[1])
prepared = c.read_json(root / 'SUITE_PREPARED.json')
proof = c.read_json(root / 'PARENT_IMPORT.json')
receipt = c.read_json(dr.RECEIPT)
allowed = {root / name for name in ('SUITE_PREPARED.json', 'SUITE_SEALED.json',
    'EXECUTION_STARTED.json', 'queue_state.json', 'ALL_FULL_PREDICTIONS_SEALED.json',
    'full_metrics.csv', 'frozen_contract.md', 'PARENT_IMPORT.json')}
allowed.add(dr.RECEIPT)
allowed.update(root / 'runtime' / p for p in prepared['runtime_hashes'])
allowed.update(root / p for p in prepared['parent_artifact_sha256'])
allowed.update(Path(p) for p in proof['parent_metadata_sha256'])
for d in upstream.DATASETS:
    allowed.update(upstream.manifest_path(root, d, stage) for stage in ('gate', 'full'))
    allowed.update(root / 'full' / d / name for name in ('SCORED_SEALED.json', 'PREDICTIONS_SEALED.json', 'result.json'))
target.mkdir()
(target / 'dependency_receipt.json').write_bytes(dr.RECEIPT.read_bytes())
allowed.add(target / 'dependency_receipt.json')
output_files = {target / 'queue_state.json', target / 'queue_state.json.tmp', target / 'DEPENDENCY_COMPLETED.json'}
seen, queries = [], []
original_open, original_io, original_import = builtins.open, io.open, builtins.__import__
original_path_open = Path.open
def check(path, mode):
    path = Path(path).resolve()
    if any(flag in mode for flag in ('w', 'a', 'x', '+')):
        assert path in output_files, ('Unexpected write', path)
    else:
        assert path in allowed, ('Unlisted content read', path)
        seen.append(str(path))
def guarded_open(path, mode='r', *args, **kwargs):
    check(path, mode)
    return original_open(path, mode, *args, **kwargs)
def guarded_io(path, mode='r', *args, **kwargs):
    check(path, mode)
    return original_io(path, mode, *args, **kwargs)
def guarded_path_open(path, mode='r', *args, **kwargs):
    check(path, mode)
    return original_path_open(path, mode, *args, **kwargs)
def guarded_import(name, *args, **kwargs):
    assert name.split('.')[0] not in ('torch', 'numpy', 'PIL'), ('Model/array import', name)
    return original_import(name, *args, **kwargs)
def service(command, **kwargs):
    assert command[:4] == ['systemctl', '--user', 'show', c.DEPENDENCY_SERVICE], command
    queries.append(command)
    return SimpleNamespace(stdout='\n'.join(k+'='+v for k,v in receipt['service_terminal'].items()))
builtins.open, io.open, builtins.__import__ = guarded_open, guarded_io, guarded_import
Path.open = guarded_path_open
subprocess.run = service
q.service_fields = lambda _: receipt['service_terminal'].copy()
# Real upstream.verify_prepared and real bound_service_terminal remain intact.
q.wait_dependency(target)
builtins.open, io.open, builtins.__import__ = original_open, original_io, original_import
Path.open = original_path_open
result = c.read_json(target / 'DEPENDENCY_COMPLETED.json')
assert len(queries) == 2 and len(set(seen)) > 100, (len(queries), len(set(seen)))
assert result['certification_reuse']['payload_rehashed'] is False
assert result['certification_reuse']['prediction_or_label_arrays_decoded'] is False
assert not any(n.split('.')[0] in ('torch', 'numpy', 'PIL') for n in sys.modules)
print(json.dumps(dict(unique_read_paths=len(set(seen)), queries=len(queries), full_outcomes=result['full_outcomes'])))
'''
    import os
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path / 'copied_suite')],
        text=True, capture_output=True, timeout=60,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    import json
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['queries'] == 2 and report['full_outcomes'] == 10
