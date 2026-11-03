"""Synthetic scope, admission and lifecycle tests; no models or dataset reads."""

import copy
import csv
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.evolution_local_baselines import common as c
from tools.evolution_local_baselines import queue as q


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    suite = tmp_path / 'suite'
    runtime, native = suite / 'runtime', suite / 'baseline_source'
    runtime.mkdir(parents=True)
    native.mkdir()
    (runtime / 'native.py').write_text('VALUE = 1\n')
    sources = {'native.py': c.sha256(runtime / 'native.py')}
    (native / 'method.py').write_text('NATIVE = True\n')
    baseline = {'method.py': c.sha256(native / 'method.py')}
    parent, dependency = tmp_path / 'parent', tmp_path / 'dependency'
    parent.mkdir()
    dependency.mkdir()
    monkeypatch.setattr(c, 'PARENT', parent)
    monkeypatch.setattr(c, 'DEPENDENCY', dependency)
    monkeypatch.setattr(c, 'POPULATIONS', dict.fromkeys(c.DATASETS, 4))
    monkeypatch.setattr(c, 'BASELINE_EXTRA', {})
    checkpoint = dict(path=str(tmp_path / 'checkpoint'), sha256='a' * 64)
    Path(checkpoint['path']).write_text('synthetic checkpoint, stat only')
    original_prepared = dict(runtime_hashes=sources, checkpoint=checkpoint)
    c.write_json(parent / 'SUITE_PREPARED.json', original_prepared)
    c.write_json(dependency / 'SUITE_PREPARED.json', original_prepared)
    monkeypatch.setattr(c, 'PARENT_SHA', c.sha256(parent / 'SUITE_PREPARED.json'))
    monkeypatch.setattr(c, 'DEPENDENCY_SHA', c.sha256(dependency / 'SUITE_PREPARED.json'))
    (suite / 'parents').mkdir()
    for name in ('SUITE_SEALED.json', 'ALL_CONFIGS_SELECTED.json', 'ALL_FULL_PREDICTIONS_SEALED.json'):
        c.write_json(parent / name, dict(complete=True))
    entries = {}
    for dataset in c.DATASETS:
        destination = suite / 'parents' / dataset
        destination.mkdir()
        config = dict(dataset=dataset, resize=[224, 224], patch_size=[224, 224], patch_stride=112,
                      class_extensions=True, batch_size=13, lr=.00375, steps=30)
        rows = []
        for i in range(4):
            image, label = tmp_path / f'{dataset}{i}.jpg', tmp_path / f'{dataset}{i}.png'
            image.write_text('synthetic RGB')
            label.write_text('synthetic annotation, stat only')
            rows.append(dict(image=str(image), label=str(label), official_id=str(i), image_sha256='b' * 64))
        panel = dict(inputs=rows, ordered_ids=[str(i) for i in range(4)], classes=['a', 'b'],
                     queries=['a', 'b'], query_to_class=[0, 1], full_split=True, images=4)
        c.write_json(destination / 'panel.json', panel)
        full = parent / 'full' / dataset
        full.mkdir(parents=True)
        c.write_json(full / 'prediction_protocol.json', dict(manifest=dict(config=config, checkpoint=checkpoint, panel=panel)))
        for name in ('PREDICTIONS_SEALED.json', 'SCORED_SEALED.json', 'result.json'):
            c.write_json(full / name, dict(complete=True))
        selection = parent / 'selections' / dataset
        selection.mkdir(parents=True)
        c.write_json(selection / 'SELECTED_CONFIG_SEALED.json', dict(config=config))
        entries[dataset] = dict(config=config, panel_path=f'parents/{dataset}/panel.json',
                                panel_sha256=c.sha256(destination / 'panel.json'))
    originals = c.parent_sources()
    for relative, original in originals.items():
        target = suite / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    artifacts = {relative: c.sha256(suite / relative) for relative in originals}
    artifacts.update({entry['panel_path']: entry['panel_sha256'] for entry in entries.values()})
    (suite / 'parent_importer.py').write_text('SYNTHETIC_IMPORTER = True\n')
    c.write_json(suite / 'PARENT_IMPORT.json', dict(parent_suite=str(parent), parent_prepared_sha256=c.PARENT_SHA,
        verified_parent_populations=8, verified_parent_full_outcomes=32,
        importer_sha256=c.sha256(suite / 'parent_importer.py'),
        metadata_sha256={str(path): c.sha256(path) for path in originals.values()},
        parent_original_paths={key: str(value) for key, value in originals.items()}))
    c.write_json(suite / 'archive_provenance.json', dict(legacy_commit='ba9f57bf47846ec727c616c5163233a98ca0bf94',
                                                       baseline_sources=baseline))
    monkeypatch.setattr(c, 'ARCHIVE_PROVENANCE_SHA', c.sha256(suite / 'archive_provenance.json'))
    (suite / 'frozen_contract.md').write_text('Locked baseline protocol\n')
    prepared = dict(schema_version=1, algorithm=c.ALGORITHM, jobs=[list(j) for j in c.JOBS], parameter_search=False,
        retry=False, parent_suite=str(parent), parent_prepared_sha256=c.PARENT_SHA,
        dependency_suite=str(dependency), dependency_prepared_sha256=c.DEPENDENCY_SHA,
        dependency_service=c.DEPENDENCY_SERVICE, resources=c.RESOURCES, runtime_hashes=sources,
        baseline_hashes=baseline, entries=entries, checkpoint=checkpoint,
        parent_proof_sha256=c.sha256(suite / 'PARENT_IMPORT.json'), parent_artifact_sha256=artifacts,
        archive_provenance_sha256=c.ARCHIVE_PROVENANCE_SHA, contract_sha256=c.sha256(suite / 'frozen_contract.md'),
        python='/python', service_name='trace-local-baselines-test.service')
    c.write_json(suite / 'SUITE_PREPARED.json', prepared)
    identity = c.sha256(suite / 'SUITE_PREPARED.json')
    (suite / 'manifests').mkdir()
    for dataset, method in c.JOBS:
        for stage in ('gate', 'full'):
            c.write_json(c.manifest_path(suite, dataset, method, stage),
                         c.expected_manifest(suite, prepared, dataset, method, stage, identity))
    return suite, prepared, identity


def test_fixed_eighteen_cells():
    assert len(c.JOBS) == len(set(c.JOBS)) == 18
    assert sum(c.POPULATIONS[d] for d, _ in c.JOBS) == 58040
    assert [d for d, m in c.JOBS if m == 'mlmp'] == list(c.DATASETS)
    assert {d for d, m in c.JOBS if m != 'mlmp'} == set(c.MISSING)
    assert 'pc59' not in c.MISSING


@pytest.mark.parametrize('method,updates,losses,calls', [('mlmp', 10, 10, 12), ('noadapt', 0, 0, 1),
    ('tent', 10, 10, 11), ('tpt', 10, 10, 11), ('watt', 70, 5, 141), ('clipartt', 10, 10, 21)])
def test_native_budgets_not_trace_selected(method, updates, losses, calls):
    actual = c.settings('/suite', method, {'path': '/checkpoint'})
    assert actual['batch_size'] == 2 and actual['constructor']['lr'] == .001
    assert actual['expected_update_steps'] == updates
    assert actual['expected_loss_entries'] == losses
    assert actual['expected_visual_calls'] == calls
    assert actual['parameter_search'] is False
    assert ('steps' not in actual['constructor']) == (method == 'watt')


def test_complete_prepared_admitted(frozen):
    suite, _, identity = frozen
    assert c.verify_prepared(suite, identity)['_verified_sha256'] == identity


def upgrade_receipt_schema(suite, prepared):
    from tools.evolution_local_baselines import dependency_receipt as dr
    shutil.copyfile(dr.RECEIPT, suite / 'dependency_receipt.json')
    prepared = dict(prepared, schema_version=2, dependency_receipt_sha256=dr.RECEIPT_SHA,
                    dependency_suite_seal_sha256=dr.SUITE_SEAL_SHA)
    c.write_json(suite / 'SUITE_PREPARED.json', prepared, replace=True)
    identity = c.sha256(suite / 'SUITE_PREPARED.json')
    for dataset, method in c.JOBS:
        for stage in ('gate', 'full'):
            c.write_json(c.manifest_path(suite, dataset, method, stage),
                c.expected_manifest(suite, prepared, dataset, method, stage, identity), replace=True)
    return identity


def test_v2_prepared_binds_certified_dependency_copy(frozen):
    suite, prepared, _ = frozen
    identity = upgrade_receipt_schema(suite, prepared)
    assert c.verify_prepared(suite, identity)['schema_version'] == 2


@pytest.mark.parametrize('damage', ['missing', 'content', 'symlink', 'hardlink'])
def test_v2_dependency_copy_drift_or_alias_rejected(frozen, damage):
    suite, prepared, _ = frozen
    identity = upgrade_receipt_schema(suite, prepared)
    path = suite / 'dependency_receipt.json'
    if damage == 'missing':
        path.unlink()
    elif damage == 'content':
        path.write_text('{}')
    elif damage == 'symlink':
        other = suite / 'elsewhere'
        path.rename(other)
        path.symlink_to(other)
    else:
        (suite / 'elsewhere').hardlink_to(path)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        c.verify_prepared(suite, identity)


@pytest.mark.parametrize('tree,name', [('runtime', 'startup.py'), ('runtime', '__pycache__/native.pyc'),
    ('baseline_source', 'extra.py'), ('baseline_source', '__pycache__/method.pyc')])
def test_inventory_closed(frozen, tree, name):
    suite, _, identity = frozen
    path = suite / tree / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('unrecorded')
    with pytest.raises(RuntimeError):
        c.verify_prepared(suite, identity)


@pytest.mark.parametrize('name', ['runtime/native.py', 'baseline_source/method.py', 'frozen_contract.md',
    'parents/pc59/panel.json', 'parents/pc60/prediction_protocol.json', 'archive_provenance.json'])
def test_source_or_parent_drift_rejected(frozen, name):
    suite, _, identity = frozen
    (suite / name).write_text('changed')
    with pytest.raises((RuntimeError, ValueError)):
        c.verify_prepared(suite, identity)


@pytest.mark.parametrize('field,value', [('seed', True), ('seed', 1), ('schema_version', True),
    ('algorithm', 'TRACE'), ('stage', 'calibration'), ('method', 'image_C')])
def test_manifest_scope_types(frozen, field, value):
    suite, _, _ = frozen
    manifest = c.read_json(c.manifest_path(suite, 'voc20', 'mlmp', 'full'))
    manifest[field] = value
    with pytest.raises(RuntimeError):
        c.validate_manifest(manifest)


@pytest.mark.parametrize('field,value', [('lr', .002), ('batch_size', True), ('batch_size', 13), ('steps', 30)])
def test_no_trace_or_search_settings(frozen, field, value):
    suite, _, identity = frozen
    path = c.manifest_path(suite, 'voc20', 'mlmp', 'full')
    manifest = c.read_json(path)
    manifest['config'][field] = value
    c.write_json(path, manifest, replace=True)
    with pytest.raises(RuntimeError):
        c.verify_prepared(suite, identity)


def test_four_rgb_gate_no_label_loading(frozen):
    suite, _, _ = frozen
    gate = c.read_json(c.manifest_path(suite, 'pc59', 'mlmp', 'gate'))
    full = c.read_json(c.manifest_path(suite, 'pc59', 'mlmp', 'full'))
    assert len(gate['panel']['inputs']) == gate['panel']['images'] == 4
    assert gate['panel']['full_split'] is False
    assert gate['panel']['inputs'] == full['panel']['inputs'][:4]
    assert gate['config']['steps'] == 10
    assert full['config']['steps'] != 30


def test_all_eight_annotation_aliases_guarded(frozen):
    suite, _, _ = frozen
    manifest = c.read_json(c.manifest_path(suite, 'voc20', 'mlmp', 'gate'))
    wb = c.read_json(suite / 'parents/waterbirds_seg/panel.json')['inputs'][0]['label']
    alias = suite / 'alias'
    alias.hardlink_to(wb)
    with pytest.raises(RuntimeError, match='annotation'):
        c.guard_paths(manifest, [alias])


def test_foreign_runtime_rejected(frozen):
    suite, _, identity = frozen
    with pytest.raises(RuntimeError, match='Frozen runtime'):
        c.verify_prepared(suite, identity, frozen=True)


def terminal(suite, prepared):
    prepared = dict(prepared, _verified_sha256='d' * 64)
    name = prepared['service_name']
    invocation = 'a' * 32
    command = c.expected_command(suite, prepared)
    fields = dict(c.RESOURCES, Id=name, LoadState='loaded', ActiveState='active', SubState='exited',
        MainPID='0', ExecMainPID='123', ExecMainCode='1', ExecMainStatus='0', Result='success',
        InvocationID=invocation, ControlGroup='', WorkingDirectory=str(suite / 'runtime'),
        ExecStart='argv[]=' + ' '.join(command) + ' ;')
    startup = dict(fields, ActiveState='active', SubState='running', MainPID='123', ControlGroup='/user/' + name)
    execution = dict(service=name, pid=123, invocation_id=invocation, cwd=str(suite / 'runtime'),
        control_group=startup['ControlGroup'], command=command, resources=startup,
        suite_prepared_sha256=prepared['_verified_sha256'], retry_allowed=False)
    return prepared, fields, execution


def test_real_retained_terminal_accepted(frozen, monkeypatch):
    suite, prepared, _ = frozen
    prepared, fields, execution = terminal(suite, prepared)
    monkeypatch.setattr(q, 'service_fields', lambda _: fields)
    assert q.bound_service_terminal(suite, prepared, execution) == fields


@pytest.mark.parametrize('field,value', [('LoadState', 'not-found'), ('SubState', 'running'),
    ('MainPID', '123'), ('Result', 'exit-code'), ('ExecMainStatus', '1'), ('ExecMainCode', '0'),
    ('InvocationID', 'b' * 32), ('MemoryMax', 'infinity'), ('ControlGroup', '/another/service'),
    ('ExecStart', 'argv[]=/other ;'), ('ExecMainPID', '456')])
def test_success_result_alone_never_terminal(frozen, monkeypatch, field, value):
    suite, prepared, _ = frozen
    prepared, fields, execution = terminal(suite, prepared)
    fields[field] = value
    monkeypatch.setattr(q, 'service_fields', lambda _: fields)
    with pytest.raises(RuntimeError, match='terminal'):
        q.bound_service_terminal(suite, prepared, execution)


def test_failed_dependency_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(q, 'service_fields', lambda _: dict(LoadState='loaded', ActiveState='failed', SubState='failed'))
    with pytest.raises(RuntimeError, match='no retry'):
        q.wait_dependency(tmp_path)


def test_controller_refuses_second_attempt(frozen, monkeypatch):
    suite, prepared, identity = frozen
    monkeypatch.setattr(c, 'verify_prepared', lambda *a, **kw: dict(prepared, _verified_sha256=identity))
    c.write_json(suite / 'EXECUTION_STARTED.json', dict(already_started=True))
    with pytest.raises(RuntimeError, match='No controller retry'):
        q.run(SimpleNamespace(suite=suite, suite_sha256=identity))


def gate_report(suite, method='mlmp'):
    from tools.evolution_followup.standalone import GPU_LOCK
    path = c.manifest_path(suite, 'coco_cd', method, 'gate')
    manifest = c.read_json(path)
    output = Path(manifest['output'])
    output.mkdir(parents=True)
    attempt = dict(phase='gate', manifest_sha256=c.sha256(path), output=str(output), retry_allowed=False)
    c.write_json(output / 'STARTED.json', attempt)
    c.write_json(output.parent / f'{output.name}.STARTED.json', attempt)
    report = dict(passed=True, **{k: manifest[k] for k in ('algorithm', 'dataset', 'method', 'stage', 'seed',
        'suite_prepared_sha256', 'source_sha256', 'baseline_source_sha256', 'checkpoint', 'settings', 'parent_proof_sha256')},
        manifest=manifest, manifest_sha256=c.sha256(path), settings_constructor=manifest['settings']['constructor'],
        images=4, gt_contents_opened=False, configured_batch_size=2, batch_images=[2, 2],
        tolerance=1e-5, changed_labels=0, comparisons=[],
        gpu_lock=dict(path=str(GPU_LOCK), inode=123, pid=456, exclusive=True),
        deterministic_runtime=dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
            deterministic_warn_only=False, deterministic_debug_mode=2, cudnn_deterministic=True,
            cudnn_benchmark=False, sdpa_enabled=dict.fromkeys(('flash', 'mem_efficient', 'math', 'cudnn'), True)))
    reset = dict(snapshot_sha256='a'*64, model_sha256='b'*64, trainable_sha256='c'*64,
        optimizer_sha256='d'*64, ctx_sha256='e'*64 if method == 'tpt' else None,
        optimized_parameter_names=['ctx'] if method == 'tpt' else ['visual.ln.weight', 'visual.ln.bias'],
        native_trainable_parameter_names=['ctx'] if method == 'tpt' else ['visual.ln.weight', 'visual.ln.bias', 'logit_scale'],
        nonoptimized_parameter_names=['weight'] if method == 'tpt' else ['logit_scale', 'weight'])
    for i in range(2):
        budget = manifest['settings']
        cost = dict(**{k: manifest[k] for k in ('algorithm', 'dataset', 'method', 'stage', 'seed')},
            episode=i, images=2, patches=2, warmup=i == 0, seconds=1., native_reset_verified=True,
            gt_contents_opened=False, optimizer_steps=budget['expected_update_steps'],
            loss_entries=budget['expected_loss_entries'], visual_calls=budget['expected_visual_calls'],
            visual_inputs=2*budget['expected_visual_calls'])
        updates = [dict(step=j+1, all_gradients_parameters_Adam_finite=True, gradient_max_abs=0.,
            parameter_max_abs=0., pre_parameter_max_abs=0., gradient_sha256='a'*64,
            pre_parameter_sha256='b'*64, post_parameter_sha256='c'*64, adam_state_sha256='d'*64,
            adam_tensor_devices=['cpu', 'cuda:0']) for j in range(budget['expected_update_steps'])]
        report['comparisons'].append(dict(episode=i, order=['native', 'observed'] if i == 0 else ['observed', 'native'],
            batch_images=2, finite_updates_verified=True, changed_labels=0,
            ordered_ids=manifest['panel']['ordered_ids'][2*i:2*i+2],
            max_abs=dict(loss=0., gradient=0., parameters=0., final_parameters=0.), actual_cost=cost,
            losses=[.1]*budget['expected_loss_entries'], update_receipts=updates, reset_state=copy.deepcopy(reset),
            parity_parameter_names=reset['optimized_parameter_names'].copy()))
    return output, manifest, report


@pytest.mark.parametrize('method', c.METHODS)
def test_gate_real_budget_certificate_scope(frozen, method):
    suite, _, _ = frozen
    output, manifest, report = gate_report(suite, method)
    assert c.validate_gate_report(output, manifest, report) == report


@pytest.mark.parametrize('mutation', ['tolerance', 'changed', 'gradient', 'steps', 'budget', 'source', 'attempt',
    'reset_missing', 'reset_digest', 'reset_unequal', 'reset_member', 'gpu_lock', 'determinism', 'backend',
    'parity_duplicate', 'parity_missing', 'parity_extra'])
def test_gate_certificate_not_just_pass_flag(frozen, mutation):
    suite, _, _ = frozen
    output, manifest, report = gate_report(suite)
    if mutation == 'tolerance':
        report['tolerance'] = 1e-4
    elif mutation == 'changed':
        report['comparisons'][1]['changed_labels'] = 1
    elif mutation == 'gradient':
        report['comparisons'][0]['max_abs']['gradient'] = 1e-4
    elif mutation == 'steps':
        report['comparisons'][0]['update_receipts'].pop()
    elif mutation == 'budget':
        report['comparisons'][0]['actual_cost']['optimizer_steps'] = 9
    elif mutation == 'source':
        report['source_sha256'] = {}
    elif mutation == 'attempt':
        (output.parent / f'{output.name}.STARTED.json').unlink()
    elif mutation == 'reset_missing':
        report['comparisons'][0].pop('reset_state')
    elif mutation == 'reset_digest':
        report['comparisons'][0]['reset_state']['ctx_sha256'] = 'INVALID'
    elif mutation == 'reset_unequal':
        report['comparisons'][0]['reset_state']['model_sha256'] = 'f'*64
    elif mutation == 'reset_member':
        report['comparisons'][0]['reset_state']['optimized_parameter_names'] = ['logit_scale']
    elif mutation == 'gpu_lock':
        report['gpu_lock']['exclusive'] = False
    elif mutation == 'determinism':
        report['deterministic_runtime']['deterministic_warn_only'] = True
    elif mutation == 'backend':
        report['deterministic_runtime']['sdpa_enabled']['flash'] = False
    elif mutation == 'parity_duplicate':
        report['comparisons'][0]['parity_parameter_names'].append('visual.ln.weight')
    elif mutation == 'parity_missing':
        report['comparisons'][0]['parity_parameter_names'].pop()
    elif mutation == 'parity_extra':
        report['comparisons'][0]['parity_parameter_names'].append('unknown.weight')
    with pytest.raises((RuntimeError, FileNotFoundError, KeyError)):
        c.validate_gate_report(output, manifest, report)


def prediction_barrier(suite):
    predictions = {}
    for dataset, method in c.JOBS:
        path = c.manifest_path(suite, dataset, method, 'full')
        manifest = c.read_json(path)
        output = Path(manifest['output'])
        output.mkdir(parents=True)
        c.write_json(output / 'PREDICTIONS_SEALED.json', dict(complete=True, gt_contents_opened=False,
            manifest_sha256=c.sha256(path), images=4, output=str(output),
            **{k: manifest[k] for k in ('suite_prepared_sha256', 'dataset', 'method', 'algorithm', 'stage', 'seed')}))
        predictions[f'{dataset}.{method}'] = c.sha256(output / 'PREDICTIONS_SEALED.json')
    path = suite / 'ALL_FULL_PREDICTIONS_SEALED.json'
    c.write_json(path, dict(complete=True, gt_contents_opened=False,
        suite_prepared_sha256=manifest['suite_prepared_sha256'], predictions=predictions))
    return path, manifest


def test_all_eighteen_prediction_seals_required(frozen):
    suite, _, _ = frozen
    path, manifest = prediction_barrier(suite)
    assert c.verify_barrier(path, manifest) == c.sha256(path)
    barrier = c.read_json(path)
    barrier['predictions'].pop('voc20.mlmp')
    c.write_json(path, barrier, replace=True)
    with pytest.raises(RuntimeError):
        c.verify_barrier(path, manifest)


@pytest.mark.parametrize('field,value', [('images', 3), ('stage', 'gate'), ('method', 'tent'),
    ('suite_prepared_sha256', 'e'*64), ('output', '/elsewhere'), ('gt_contents_opened', True)])
def test_peer_prediction_scope_checked_even_with_new_hash(frozen, field, value):
    suite, _, _ = frozen
    path, manifest = prediction_barrier(suite)
    peer = suite / 'full/voc20.mlmp/PREDICTIONS_SEALED.json'
    sealed = c.read_json(peer)
    sealed[field] = value
    c.write_json(peer, sealed, replace=True)
    barrier = c.read_json(path)
    barrier['predictions']['voc20.mlmp'] = c.sha256(peer)
    c.write_json(path, barrier, replace=True)
    with pytest.raises(RuntimeError):
        c.verify_barrier(path, manifest)


def completed_suite(suite, prepared, identity, monkeypatch):
    rows = [dict(dataset=d, method=m, images=4, mIoU=1.1,
                 result_path=str(suite / 'full' / f'{d}.{m}' / 'result.json')) for d, m in c.JOBS]
    score_seals = {f'{d}.{m}': 'a'*64 for d, m in c.JOBS}
    monkeypatch.setattr(c, 'verify_prepared', lambda *a, **kw: dict(prepared, _verified_sha256=identity))
    monkeypatch.setattr(q, 'bound_service_terminal', lambda *a: dict(real_terminal=True))
    monkeypatch.setattr(q, 'validate_results', lambda *a: (rows, score_seals))
    c.write_json(suite / 'EXECUTION_STARTED.json', dict(pid=123))
    c.write_json(suite / 'queue_state.json', dict(phase='completed', pid=123))
    c.write_json(suite / 'DEPENDENCY_COMPLETED.json', dict(suite=str(c.DEPENDENCY),
        suite_prepared_sha256=c.DEPENDENCY_SHA, full_outcomes=10))
    c.write_json(suite / 'ALL_FULL_PREDICTIONS_SEALED.json', dict(complete=True))
    with (suite / 'full_metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sealed = dict(complete=True, full_outcomes=18, suite_prepared_sha256=identity,
        service_exit_certified=False, score_seals=score_seals)
    for field, file in [('full_metrics_sha256', 'full_metrics.csv'),
        ('prediction_barrier_sha256', 'ALL_FULL_PREDICTIONS_SEALED.json'),
        ('execution_started_sha256', 'EXECUTION_STARTED.json'),
        ('dependency_completed_sha256', 'DEPENDENCY_COMPLETED.json')]:
        sealed[field] = c.sha256(suite / file)
    c.write_json(suite / 'SUITE_SEALED.json', sealed)
    return sealed


def test_post_terminal_completion_preserves_exposure_boundary(frozen, monkeypatch):
    suite, prepared, identity = frozen
    completed_suite(suite, prepared, identity, monkeypatch)
    report = q.certify_completed(suite, identity)
    assert report['full_outcomes'] == 18 and report['service_exit_certified'] is True
    assert report['independent_test_claim'] is False and report['development_exposed'] is True


@pytest.mark.parametrize('field,value', [('full_outcomes', 17), ('complete', False),
    ('service_exit_certified', True), ('score_seals', {}), ('execution_started_sha256', 'a'*64)])
def test_false_completion_receipt_rejected(frozen, monkeypatch, field, value):
    suite, prepared, identity = frozen
    sealed = completed_suite(suite, prepared, identity, monkeypatch)
    sealed[field] = value
    c.write_json(suite / 'SUITE_SEALED.json', sealed, replace=True)
    with pytest.raises(RuntimeError):
        q.certify_completed(suite, identity)
