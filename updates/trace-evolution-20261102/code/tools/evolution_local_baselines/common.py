"""Immutable scope, source and admission checks for eighteen local baselines."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import subprocess

from tools.evolution_allocation.common import (
    absolute_path, canonical_hash, digest_valid, read_json, require, sha256,
    verify_inventory, writable_path, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path('/home/wwww/hzl/results/trace_single_probe_waterbirds_full_20261029_0817')
PARENT_SHA = '4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3'
DEPENDENCY = Path('/home/wwww/hzl/results/trace_allocation_controls_20261029_2055')
DEPENDENCY_SHA = 'c3b9e5d1548655e6225e41f2b8def91007ecb6b4904343388958b14ba2f42dd4'
DEPENDENCY_SERVICE = 'trace-allocation-controls-20261029-2055.service'
ARCHIVE = Path('/home/wwww/hzl/results/reviewer_response_20260929')
BASELINE_ARCHIVE = ARCHIVE / 'baseline_source'
ARCHIVE_PROVENANCE_SHA = 'ab92729b308ccef321882b6736fb769153d51711359b68081ae56413bce68d1b'
BASELINE_EXTRA = {
    'prompts.yaml': '22ce4f928f024acc16195be5509bf53ca39dd5d1e66c18cda7fc860336a6bcdf',
    'LICENCE': '769edb081de98ff7feac12e9a4351fbf84db3b5d8c3498ef6f296df7c66d2a0c',
    'ovss/clip/bpe_simple_vocab_16e6.txt.gz': '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a',
}
ALGORITHM = 'local-native-baseline-reruns:v1'
DATASETS = ('voc20', 'voc21', 'pc59', 'pc60', 'coco_object', 'coco_stuff', 'coco_cd', 'waterbirds_seg')
POPULATIONS = dict(zip(DATASETS, (1449, 1449, 5105, 5105, 5000, 5000, 500, 5322)))
METHODS = ('mlmp', 'noadapt', 'tent', 'tpt', 'watt', 'clipartt')
MISSING = ('coco_cd', 'waterbirds_seg')
JOBS = tuple((d, 'mlmp') for d in DATASETS) + tuple((d, m) for d in MISSING for m in METHODS[1:])
RESOURCES = dict(MemoryMax='12884901888', MemorySwapMax='1073741824', OOMPolicy='stop',
                 Restart='no', RemainAfterExit='yes')
PROMPTS = ['itap of a {}', 'a bad photo of the {}.', 'a origami {}.',
           'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']
MANIFEST_KEYS = {'schema_version', 'algorithm', 'dataset', 'method', 'stage', 'seed',
    'suite_root', 'runtime_root', 'suite_prepared_sha256', 'source_sha256', 'baseline_source_sha256',
    'baseline_root', 'config', 'panel', 'checkpoint', 'settings', 'output', 'parent_proof_sha256'}


def settings(suite, method, checkpoint):
    require(method in METHODS, 'Unknown baseline method')
    constructor = dict(ovss_type='naclip', ovss_backbone=checkpoint['path'], lr=.001, device='cuda')
    losses, updates, forwards = 10, 10, 11
    if method == 'mlmp':
        forwards = 12
        constructor.update(steps=10, vision_outputs=list(range(-1, -19, -1)), alpha_cls=1.,
            prompt_integration='loss', prompt_dir=str(Path(suite) / 'baseline_source/prompts.yaml'))
    elif method == 'watt':
        constructor.update(watt_l=2, watt_m=5, prompt_dir=str(Path(suite) / 'baseline_source/prompts.yaml'))
        losses, updates, forwards = 5, 70, 141
    else:
        constructor['steps'] = 10
        if method == 'tpt':
            constructor['n_ctx'] = 4
        if method == 'clipartt':
            constructor['clipartt_k'] = 3
            forwards = 21
        if method == 'noadapt':
            losses, updates, forwards = 0, 0, 1
    return dict(constructor=constructor, batch_size=2, optimizer='Adam', betas=[.9, .999],
        weight_decay=0., expected_update_steps=updates, expected_loss_entries=losses,
        expected_visual_calls=forwards, prompt_templates=PROMPTS if method in ('mlmp', 'watt') else ['a photo of a {}'],
        episodic_reset=True, parameter_search=False, native_method=method)


def manifest_path(suite, dataset, method, stage):
    require((dataset, method) in JOBS and stage in ('gate', 'full'), 'Unknown local baseline cell')
    return Path(suite) / 'manifests' / f'{dataset}.{method}.{stage}.json'


def expected_manifest(suite, prepared, dataset, method, stage, identity):
    suite = absolute_path(suite)
    require((dataset, method) in JOBS and stage in ('gate', 'full'), 'Unplanned cell')
    entry = prepared['entries'][dataset]
    panel = read_json(suite / entry['panel_path'])
    if stage == 'gate':
        panel = copy.deepcopy(panel)
        panel['inputs'], panel['ordered_ids'], panel['full_split'] = panel['inputs'][:4], panel['ordered_ids'][:4], False
        if 'images' in panel:
            panel['images'] = 4
        if 'input_identity_sha256' in panel:
            panel['input_identity_sha256'] = canonical_hash(panel['inputs'])
    config = copy.deepcopy(entry['config'])
    config.update(batch_size=2, lr=.001, steps=10)
    return dict(schema_version=1, algorithm=ALGORITHM, dataset=dataset, method=method, stage=stage, seed=0,
        suite_root=str(suite), runtime_root=str(suite / 'runtime'), suite_prepared_sha256=identity,
        source_sha256=prepared['runtime_hashes'], baseline_source_sha256=prepared['baseline_hashes'],
        baseline_root=str(suite / 'baseline_source'), config=config, panel=panel,
        checkpoint=prepared['checkpoint'], settings=settings(suite, method, prepared['checkpoint']),
        output=str(suite / ('gates' if stage == 'gate' else 'full') / f'{dataset}.{method}'),
        parent_proof_sha256=prepared['parent_proof_sha256'])


def annotation_inodes(manifest):
    suite = Path(manifest['suite_root'])
    labels = set()
    for dataset in DATASETS:
        panel = read_json(suite / f'parents/{dataset}/panel.json')
        for row in panel['inputs']:
            status = Path(row['label']).stat()
            labels.add((status.st_dev, status.st_ino))
    return labels


def guard_paths(manifest, paths):
    labels = annotation_inodes(manifest)
    for path in paths:
        status = Path(path).stat()
        require((status.st_dev, status.st_ino) not in labels, 'Non-label path aliases a suite annotation')


def parent_sources():
    result = {f'parents/{n}': PARENT / n for n in
              ('SUITE_PREPARED.json', 'SUITE_SEALED.json', 'ALL_CONFIGS_SELECTED.json', 'ALL_FULL_PREDICTIONS_SEALED.json')}
    for dataset in DATASETS:
        for name in ('prediction_protocol.json', 'PREDICTIONS_SEALED.json', 'SCORED_SEALED.json', 'result.json'):
            result[f'parents/{dataset}/{name}'] = PARENT / 'full' / dataset / name
        result[f'parents/{dataset}/SELECTED_CONFIG_SEALED.json'] = PARENT / 'selections' / dataset / 'SELECTED_CONFIG_SEALED.json'
    return result


def verify_prepared(suite, expected, *, frozen=False):
    suite = absolute_path(suite)
    require(digest_valid(expected) and sha256(suite / 'SUITE_PREPARED.json') == expected, 'Wrong prepared identity')
    p = read_json(suite / 'SUITE_PREPARED.json')
    require(p['algorithm'] == ALGORITHM and p['jobs'] == [list(j) for j in JOBS]
            and p['parent_suite'] == str(PARENT) and p['parent_prepared_sha256'] == PARENT_SHA
            and p['dependency_suite'] == str(DEPENDENCY) and p['dependency_prepared_sha256'] == DEPENDENCY_SHA
            and p['dependency_service'] == DEPENDENCY_SERVICE and p['resources'] == RESOURCES
            and p['parameter_search'] is False and p['retry'] is False
            and set(p['entries']) == set(DATASETS), 'Prepared scope differs')
    require(isinstance(p['service_name'], str) and p['service_name'].startswith('trace-local-baselines-')
            and p['service_name'].endswith('.service') and '/' not in p['service_name'], 'Wrong service binding')
    guarded = dict(suite_root=str(suite))
    guard_paths(guarded, [suite / 'frozen_contract.md', suite / 'PARENT_IMPORT.json', suite / 'archive_provenance.json', suite / 'parent_importer.py',
        Path(p['checkpoint']['path']), *[suite / 'runtime' / n for n in p['runtime_hashes']],
        *[suite / 'baseline_source' / n for n in p['baseline_hashes']],
        *[suite / n for n in p['parent_artifact_sha256']]])
    if frozen:
        require(ROOT == suite / 'runtime' and Path.cwd().resolve() == ROOT, 'Frozen runtime/root required')
    verify_inventory(suite / 'runtime', p['runtime_hashes'], exact=True)
    verify_inventory(suite / 'baseline_source', p['baseline_hashes'], exact=True)
    for tree, hashes in ((suite / 'runtime', p['runtime_hashes']), (suite / 'baseline_source', p['baseline_hashes'])):
        require(all((tree / name).stat().st_nlink == 1 for name in hashes), 'Hardlinked frozen source')
    require(sha256(suite / 'frozen_contract.md') == p['contract_sha256']
            and sha256(suite / 'PARENT_IMPORT.json') == p['parent_proof_sha256']
            and sha256(PARENT / 'SUITE_PREPARED.json') == PARENT_SHA, 'Parent/contract drift')
    proof = read_json(suite / 'PARENT_IMPORT.json')
    require(proof['parent_suite'] == str(PARENT) and proof['parent_prepared_sha256'] == PARENT_SHA
            and proof['verified_parent_populations'] == 8 and proof['verified_parent_full_outcomes'] == 32
            and proof['importer_sha256'] == sha256(suite / 'parent_importer.py'),
            'Wrong parent proof')
    originals = parent_sources()
    require(proof['parent_original_paths'] == {k: str(v) for k, v in originals.items()}
            and set(p['parent_artifact_sha256']) == set(originals) | {f'parents/{d}/panel.json' for d in DATASETS},
            'Incomplete canonical parent chain')
    guard_paths(guarded, list(map(Path, proof['metadata_sha256'])))
    for path, value in proof['metadata_sha256'].items():
        require(sha256(path) == value, 'Original parent read-set drift')
    for name, value in p['parent_artifact_sha256'].items():
        require(sha256(suite / name) == value, 'Copied parent artifact drift')
        original = proof['parent_original_paths'].get(name)
        if original:
            require(proof['metadata_sha256'].get(original) == value == sha256(original), 'Parent copy not authenticated by importer')
    original = read_json(PARENT / 'SUITE_PREPARED.json')
    require(all(p['runtime_hashes'].get(k) == v for k, v in original['runtime_hashes'].items()), 'Inherited native source drift')
    dependency = read_json(DEPENDENCY / 'SUITE_PREPARED.json')
    require(sha256(DEPENDENCY / 'SUITE_PREPARED.json') == DEPENDENCY_SHA
            and all(p['runtime_hashes'].get(k) == v for k, v in dependency['runtime_hashes'].items())
            and p['checkpoint'] == original['checkpoint'] == dependency['checkpoint'], 'Frozen dependency/checkpoint drift')
    provenance = read_json(suite / 'archive_provenance.json')
    require(sha256(suite / 'archive_provenance.json') == p['archive_provenance_sha256'] == ARCHIVE_PROVENANCE_SHA
            and provenance['legacy_commit'] == 'ba9f57bf47846ec727c616c5163233a98ca0bf94'
            and p['baseline_hashes'] == dict(provenance['baseline_sources'], **BASELINE_EXTRA),
            'Native baseline source identity differs')
    for dataset, entry in p['entries'].items():
        panel = read_json(suite / entry['panel_path'])
        parent_protocol = read_json(suite / f'parents/{dataset}/prediction_protocol.json')['manifest']
        require(sha256(suite / entry['panel_path']) == entry['panel_sha256']
                and panel == parent_protocol['panel'] and entry['config'] == parent_protocol['config']
                and len(panel['inputs']) == len(panel['ordered_ids']) == POPULATIONS[dataset]
                and panel['full_split'] is True, 'Original full/config/order differs')
    for dataset, method in JOBS:
        for stage in ('gate', 'full'):
            path = manifest_path(suite, dataset, method, stage)
            if (suite / 'manifests').exists():
                manifest = read_json(path)
                validate_manifest(manifest)
                require(canonical_hash(manifest) == canonical_hash(expected_manifest(suite, p, dataset, method, stage, expected)), 'Manifest differs')
    if (suite / 'manifests').exists():
        require({path.name for path in (suite / 'manifests').iterdir()} ==
                {f'{d}.{m}.{s}.json' for d, m in JOBS for s in ('gate', 'full')}, 'Manifest inventory differs')
    p['_verified_sha256'] = expected
    return p


def validate_manifest(m):
    require(set(m) == MANIFEST_KEYS and (m['dataset'], m['method']) in JOBS
            and m['stage'] in ('gate', 'full') and type(m['seed']) is int and m['seed'] == 0
            and type(m['schema_version']) is int and m['schema_version'] == 1
            and m['algorithm'] == ALGORITHM, 'Manifest schema/scope differs')
    require(type(m['config']['batch_size']) is int and m['config']['batch_size'] == 2
            and type(m['config']['steps']) is int and m['config']['steps'] == 10
            and type(m['config']['lr']) is float and m['config']['lr'] == .001,
            'Native baseline fixed settings differ')
    require(m['settings'] == settings(m['suite_root'], m['method'], m['checkpoint']), 'Native settings differ')
    return m


def admission(args):
    path, output = absolute_path(args.manifest), absolute_path(args.output)
    require(sha256(path) == args.manifest_sha256, 'Manifest hash differs')
    m = read_json(path)
    validate_manifest(m)
    suite = absolute_path(m['suite_root'])
    require(path == manifest_path(suite, m['dataset'], m['method'], m['stage'])
            and output == Path(m['output']) and output == suite / ('gates' if m['stage'] == 'gate' else 'full') / f"{m['dataset']}.{m['method']}"
            and ROOT == suite / 'runtime' and Path.cwd().resolve() == ROOT, 'Noncanonical runtime/output')
    # Alias guards precede any source, checkpoint or prediction-content hash.
    paths = [suite / 'SUITE_PREPARED.json', suite / 'PARENT_IMPORT.json', suite / 'frozen_contract.md',
             suite / 'archive_provenance.json', Path(m['checkpoint']['path'])]
    paths += [Path(m['runtime_root']) / n for n in m['source_sha256']]
    paths += [Path(m['baseline_root']) / n for n in m['baseline_source_sha256']]
    paths += [Path(r['image']) for r in m['panel']['inputs']]
    paths += list((suite / 'manifests').glob('*.json'))
    paths += [p for p in (suite / 'parents').rglob('*') if p.is_file()]
    guard_paths(m, paths)
    prepared = verify_prepared(suite, m['suite_prepared_sha256'], frozen=True)
    require(canonical_hash(m) == canonical_hash(expected_manifest(suite, prepared, m['dataset'], m['method'], m['stage'], m['suite_prepared_sha256'])),
            'Unbound manifest')
    writable_path(suite, output)
    return m, output


def expected_command(suite, prepared):
    return [prepared['python'], '-B', '-m', 'tools.evolution_local_baselines.queue', 'run',
            '--suite', str(suite), '--suite-sha256', prepared['_verified_sha256']]


def require_gpu_envelope(manifest):
    suite = Path(manifest['suite_root'])
    execution = read_json(suite / 'EXECUTION_STARTED.json')
    prepared = read_json(suite / 'SUITE_PREPARED.json')
    prepared['_verified_sha256'] = manifest['suite_prepared_sha256']
    invocation = os.environ.get('INVOCATION_ID', '')
    group = next((x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')), None)
    require(os.getppid() == execution['pid'] and len(invocation) == 32 and invocation == execution['invocation_id']
            and group == execution['control_group'] and group.endswith('/' + prepared['service_name'])
            and execution['service'] == prepared['service_name'] and execution['retry_allowed'] is False
            and execution['suite_prepared_sha256'] == manifest['suite_prepared_sha256']
            and execution['command'] == expected_command(suite, prepared), 'Wrong actual controller invocation')
    fields = (*RESOURCES, 'Id', 'ActiveState', 'SubState', 'MainPID', 'ExecMainPID', 'InvocationID', 'ExecStart', 'WorkingDirectory', 'ControlGroup')
    result = subprocess.run(['systemctl', '--user', 'show', prepared['service_name'], '--property=' + ','.join(fields)],
                            text=True, capture_output=True, check=True, timeout=30)
    actual = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    require(all(actual.get(k) == v for k, v in RESOURCES.items()) and actual.get('Id') == prepared['service_name']
            and actual.get('ActiveState') == 'active' and actual.get('SubState') == 'running'
            and actual.get('MainPID') == actual.get('ExecMainPID') == str(execution['pid'])
            and actual.get('InvocationID') == invocation and actual.get('ControlGroup') == group
            and actual.get('WorkingDirectory') == str(suite / 'runtime')
            and 'argv[]=' + ' '.join(execution['command']) + ' ;' in actual.get('ExecStart', ''), 'GPU resource/root/command differs')
    return actual


def verify_gates(manifest):
    suite = Path(manifest['suite_root'])
    boundary = read_json(suite / 'ALL_GATES_PASSED.json')
    require(boundary['complete'] is True and boundary['suite_prepared_sha256'] == manifest['suite_prepared_sha256']
            and set(boundary['gates']) == {f'{d}.{m}' for d, m in JOBS}, 'All eighteen gates required')
    for d, method in JOBS:
        output = suite / 'gates' / f'{d}.{method}'
        guard_paths(manifest, [output / 'GATE_PASSED.json', output / 'gate.json'])
        seal, report = read_json(output / 'GATE_PASSED.json'), read_json(output / 'gate.json')
        gate_manifest = read_json(manifest_path(suite, d, method, 'gate'))
        require(sha256(output / 'GATE_PASSED.json') == boundary['gates'][f'{d}.{method}']
                and seal == dict(report, gate_sha256=sha256(output / 'gate.json')), 'Gate certificate binding differs')
        validate_gate_report(output, gate_manifest, report)
    return boundary['gates']


def validate_gate_report(output, manifest, report):
    from tools.evolution_local_baselines.worker import validate_cost
    output = Path(output)
    path = manifest_path(manifest['suite_root'], manifest['dataset'], manifest['method'], 'gate')
    manifest_sha = sha256(path)
    require(report['passed'] is True and type(report['images']) is int and report['images'] == 4
            and report['gt_contents_opened'] is False and report['manifest_sha256'] == manifest_sha
            and canonical_hash(report['manifest']) == canonical_hash(manifest)
            and all(report[k] == manifest[k] for k in ('algorithm', 'dataset', 'method', 'stage', 'seed',
                'suite_prepared_sha256', 'source_sha256', 'baseline_source_sha256', 'checkpoint', 'settings', 'parent_proof_sha256'))
            and report['settings_constructor'] == manifest['settings']['constructor']
            and report['configured_batch_size'] == 2 and report['batch_images'] == [2, 2]
            and type(report['tolerance']) is float and report['tolerance'] == 1e-5
            and type(report['changed_labels']) is int and report['changed_labels'] == 0
            and len(report['comparisons']) == 2, 'Native gate scope/numerical contract differs')
    require(not (output / 'GATE_FAILED.json').exists(), 'Failed gate cannot pass')
    external = output.parent / f'{output.name}.STARTED.json'
    guard_paths(manifest, [output / 'STARTED.json', external])
    require(read_json(output / 'STARTED.json') == read_json(external)
            == dict(phase='gate', manifest_sha256=manifest_sha, output=str(output), retry_allowed=False),
            'Gate persistent attempt differs')
    validate_runtime_receipt(report)
    nonnegative = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0
    reset = None
    parity_names = None
    for episode, row in enumerate(report['comparisons']):
        require(type(row['episode']) is int and row['episode'] == episode
                and row['order'] == (['native', 'observed'] if episode == 0 else ['observed', 'native'])
                and row['batch_images'] == 2 and row['finite_updates_verified'] is True
                and type(row['changed_labels']) is int and row['changed_labels'] == 0
                and row['ordered_ids'] == manifest['panel']['ordered_ids'][2*episode:2*episode+2]
                and set(row['max_abs']) == {'loss', 'gradient', 'parameters', 'final_parameters'}
                and all(nonnegative(x) and x <= 1e-5 for x in row['max_abs'].values()), 'Native gate comparison differs')
        validate_cost(row['actual_cost'], manifest, episode)
        validate_reset_receipt(row['reset_state'], manifest['method'])
        if reset is None:
            reset = row['reset_state']
        require(row['reset_state'] == reset, 'Native gate reset states differ between episodes')
        names = row['parity_parameter_names']
        optimized = set(reset['optimized_parameter_names'])
        require(isinstance(names, list) and all(isinstance(n, str) for n in names)
                and len(names) == len(set(names)) and optimized <= set(names)
                and set(names) <= optimized | set(reset['nonoptimized_parameter_names'])
                and (manifest['method'] == 'watt' or names == reset['optimized_parameter_names']),
                'Native parity parameter membership differs')
        if parity_names is None:
            parity_names = names
        require(names == parity_names, 'Native parity membership differs between episodes')
        require(len(row['losses']) == manifest['settings']['expected_loss_entries']
                and all(type(x) in (int, float) and math.isfinite(x) for x in row['losses'])
                and len(row['update_receipts']) == manifest['settings']['expected_update_steps'], 'Native gate update budget differs')
        for index, receipt in enumerate(row['update_receipts']):
            require(type(receipt['step']) is int and receipt['step'] == index + 1
                    and receipt['all_gradients_parameters_Adam_finite'] is True
                    and all(nonnegative(receipt[k]) and receipt[k] <= 1e-5 for k in
                        ('gradient_max_abs', 'parameter_max_abs', 'pre_parameter_max_abs'))
                    and all(digest_valid(receipt[k]) for k in ('gradient_sha256', 'pre_parameter_sha256',
                        'post_parameter_sha256', 'adam_state_sha256'))
                    and receipt['adam_tensor_devices'], 'Native gate update trajectory differs')
    return report


def validate_reset_receipt(reset, method):
    hashes = ('snapshot_sha256', 'model_sha256', 'trainable_sha256', 'optimizer_sha256')
    names = ('optimized_parameter_names', 'native_trainable_parameter_names', 'nonoptimized_parameter_names')
    require(set(reset) == set(hashes) | set(names) | {'ctx_sha256'}
            and all(digest_valid(reset[k]) for k in hashes)
            and all(isinstance(reset[k], list) and all(isinstance(n, str) and n for n in reset[k])
                    and len(reset[k]) == len(set(reset[k])) for k in names), 'Invalid native reset proof schema')
    optimized, trainable, untouched = (set(reset[k]) for k in names)
    require(optimized and optimized <= trainable and not optimized & untouched
            and trainable <= optimized | untouched, 'Invalid native reset parameter membership')
    if method == 'tpt':
        require(reset['optimized_parameter_names'] == ['ctx'] and digest_valid(reset['ctx_sha256']), 'Invalid TPT external ctx proof')
    else:
        require(reset['ctx_sha256'] is None and 'ctx' not in optimized | untouched
                and all(n.startswith('visual.') and n.endswith(('.weight', '.bias')) for n in optimized),
                'Invalid native visual LayerNorm membership')


def validate_runtime_receipt(report):
    from tools.evolution_followup.standalone import GPU_LOCK
    lock, runtime = report['gpu_lock'], report['deterministic_runtime']
    require(set(lock) == {'path', 'inode', 'pid', 'exclusive'} and lock['path'] == str(GPU_LOCK)
            and type(lock['inode']) is int and lock['inode'] > 0
            and type(lock['pid']) is int and lock['pid'] > 0 and lock['exclusive'] is True,
            'Missing actual shared GPU lock proof')
    require(set(runtime) == {'cublas_workspace_config', 'deterministic_algorithms', 'deterministic_warn_only',
                'deterministic_debug_mode', 'cudnn_deterministic', 'cudnn_benchmark', 'sdpa_enabled'}
            and runtime['cublas_workspace_config'] == ':4096:8' and runtime['deterministic_algorithms'] is True
            and runtime['deterministic_warn_only'] is False and type(runtime['deterministic_debug_mode']) is int
            and runtime['deterministic_debug_mode'] == 2 and runtime['cudnn_deterministic'] is True
            and runtime['cudnn_benchmark'] is False and set(runtime['sdpa_enabled']) == {'flash', 'mem_efficient', 'math', 'cudnn'}
            and all(type(v) is bool and v for v in runtime['sdpa_enabled'].values()), 'Deterministic default-attention proof differs')


def verify_barrier(path, manifest):
    suite = Path(manifest['suite_root'])
    require(absolute_path(path) == suite / 'ALL_FULL_PREDICTIONS_SEALED.json', 'Noncanonical barrier')
    guard_paths(manifest, [path])
    barrier = read_json(path)
    require(barrier['complete'] is True and barrier['gt_contents_opened'] is False
            and barrier['suite_prepared_sha256'] == manifest['suite_prepared_sha256']
            and set(barrier['predictions']) == {f'{d}.{m}' for d, m in JOBS}, 'All eighteen predictions required')
    for d, method in JOBS:
        output = suite / 'full' / f'{d}.{method}'
        seal_path = output / 'PREDICTIONS_SEALED.json'
        guard_paths(manifest, [seal_path])
        require(sha256(seal_path) == barrier['predictions'][f'{d}.{method}'], 'Peer prediction seal drift')
        seal = read_json(seal_path)
        require(seal['complete'] is True and seal['gt_contents_opened'] is False
                and seal['images'] == POPULATIONS[d]
                and type(seal['images']) is int
                and seal['manifest_sha256'] == sha256(manifest_path(suite, d, method, 'full'))
                and seal['suite_prepared_sha256'] == manifest['suite_prepared_sha256']
                and seal['output'] == str(output) and seal['dataset'] == d and seal['method'] == method
                and seal['algorithm'] == ALGORITHM and seal['stage'] == 'full' and type(seal['seed']) is int and seal['seed'] == 0,
                'Peer prediction scope differs')
    return sha256(path)
