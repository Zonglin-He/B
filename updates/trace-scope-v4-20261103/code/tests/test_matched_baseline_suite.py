import copy
import csv
import hashlib
import json

import pytest

from tools.run_matched_baseline_suite import (
    METHOD_UPDATE_STEPS, TIMING_SCOPE, check_episode_costs, check_source_fingerprint,
    publish, validate_cell, verify_suite,
)


def test_comparison_does_not_use_old_trace_as_acceptance(tmp_path):
    algorithms = [dict(dataset='voc21', algorithm=a, images='128',
                       mIoU='52.0', ms_per_image='25', result_path='candidate.json')
                  for a in ('standard', 'staged_full', 'quality_only', 'selective_01')]
    baselines = [dict(dataset='voc21', method=m, status='completed',
                     mIoU=value, ms_per_image=10, result_path=m+'.json')
                 for m, value in (('noadapt', 40), ('tent', 50), ('tpt', 45), ('mlmp', 51),
                                  ('native_noadapt', 53))]
    publish(tmp_path, baselines, algorithms)
    rows = list(csv.DictReader((tmp_path/'comparison.csv').open()))
    assert {r['algorithm'] for r in rows} == {'quality_only', 'selective_01'}
    assert all(r['best_available_baseline'] == 'mlmp' for r in rows)
    assert all(float(r['delta_vs_best_available']) == 1 for r in rows)
    assert all(r['comparison_complete'] == 'True' for r in rows)


def test_missing_and_failed_baselines_are_not_zero(tmp_path):
    candidate = dict(dataset='pc60', algorithm='quality_only', images='128',
                     mIoU='25', ms_per_image='40', result_path='candidate.json')
    baselines = [dict(dataset='pc60', method='tent', status='failed', mIoU=None,
                     ms_per_image=None, result_path='missing.json'),
                 dict(dataset='pc60', method='mlmp', status='completed', mIoU=27,
                     ms_per_image=20, result_path='mlmp.json')]
    publish(tmp_path, baselines, [candidate])
    row = next(csv.DictReader((tmp_path/'comparison.csv').open()))
    assert row['matched_baselines_completed'] == '1'
    assert row['comparison_complete'] == 'False'
    assert float(row['delta_vs_best_available']) == -2


def test_verify_seal_rejects_modified_parent_evidence(tmp_path):
    state = tmp_path/'queue_state.json'
    state.write_text(json.dumps(dict(status='completed')))
    (tmp_path/'SUITE_SEALED.json').write_text(json.dumps(dict(
        artifact_sha256={'queue_state.json': hashlib.sha256(state.read_bytes()).hexdigest()})))
    verify_suite(tmp_path)
    state.write_text(json.dumps(dict(status='running')))
    with pytest.raises(ValueError, match='seal mismatch'):
        verify_suite(tmp_path)


def test_between_cell_changes_fail():
    frozen = dict(native={'native.py': 'abc'}, baseline={'tent.py': 'def'},
                  experiment={'runner.py': 'ghi'}, controls={'config.json': 'jkl', 'second.json': 'mno'})
    actual = {**frozen, 'controls': {'config.json': 'jkl'}}
    check_source_fingerprint(actual, frozen)
    with pytest.raises(ValueError, match='source drift'):
        check_source_fingerprint({**actual, 'baseline': {'tent.py': 'changed'}}, frozen)
    with pytest.raises(ValueError, match='control drift'):
        check_source_fingerprint({**actual, 'controls': {'config.json': 'changed'}}, frozen)


def test_episode_and_warmup_are_not_trusted_from_worker():
    expected = dict(config={'batch_size': 16}, image_ids=[str(i) for i in range(128)])
    costs = [dict(episode=i, images=16, image_ids=json.dumps(expected['image_ids'][16*i:16*(i+1)]),
                  update_steps=10) for i in range(8)]
    check_episode_costs(costs, expected, 2, 10)
    for bad_costs, warmup in ((costs, 0), ([dict(costs[0], images=128)], 0),
                              ([dict(costs[0], episode=1), *costs[1:]], 2),
                              ([dict(costs[0], images=15), *costs[1:]], 2),
                              ([dict(costs[0], update_steps=1), *costs[1:]], 2)):
        with pytest.raises(ValueError):
            check_episode_costs(bad_costs, expected, warmup, 10)


def test_full_pc60_costs_keep_last_partial_episode():
    expected = dict(config={'batch_size': 9}, image_ids=[str(i) for i in range(5105)])
    costs = [dict(episode=i, images=len(expected['image_ids'][9*i:9*(i+1)]),
                  image_ids=json.dumps(expected['image_ids'][9*i:9*(i+1)]), update_steps=10)
             for i in range(568)]
    check_episode_costs(costs, expected, 2, 10)
    assert costs[-1]['images'] == 2
    with pytest.raises(ValueError):
        check_episode_costs(costs[:-1], expected, 2, 10)


@pytest.fixture
def sealed_cell(tmp_path):
    expected = dict(config={'batch_size': 2}, image_ids=[str(i) for i in range(5)],
                    label_ids=[str(i)+'.mask' for i in range(5)], file_metadata=[[5, 1]] * 10,
                    vocabulary_sha256='vocabulary', seed=0)
    hashes = {path: hashlib.sha256(path.encode()).hexdigest()
              for pair in zip(expected['image_ids'], expected['label_ids']) for path in pair}
    environment = dict(python='python', torch='torch', cuda='cuda', gpu='gpu',
                       gpu_inventory=['gpu, uuid, driver'], cuda_visible_devices=None,
                       float32_matmul_precision='highest', matmul_tf32=False, cudnn_tf32=True)
    sources = dict(native={}, baseline={}, experiment={}, controls={})
    protocol = dict(expected, method_settings={'update_steps': 10}, warmup_episodes=2,
                    timing_scope=TIMING_SCOPE, source_hashes=sources, environment=environment,
                    input_hashes=dict(file_metadata=expected['file_metadata'], sha256=list(hashes.values())))
    predictions = [dict(image_id=image, label_id=label, intersection=[1], union=[2])
                   for image, label in zip(expected['image_ids'], expected['label_ids'])]
    costs = [dict(episode=i, images=len(expected['image_ids'][2*i:2*i+2]),
                  image_ids=json.dumps(expected['image_ids'][2*i:2*i+2]), patches=2,
                  update_steps=10, calls=2, inputs=8, adaptation_seconds=2., inference_seconds=1.,
                  total_seconds=3., peak_allocated_mib=100.) for i in range(3)]
    result = dict(status='completed', images=5, method='mlmp', method_settings={'update_steps': 10},
                  intersection=[5], union=[10], miou_unrounded=50., warmup_episodes=2,
                  measured_images=1, ms_per_image=3000., adaptation_ms_per_image=2000.,
                  inference_ms_per_image=1000., peak_allocated_mib=100.,
                  visual_inputs_per_image=8., visual_calls_per_image=2.,
                  all_visual_inputs=24, all_visual_calls=6)

    def save():
        (tmp_path/'protocol.json').write_text(json.dumps(protocol))
        (tmp_path/'per_image_counts.json').write_text(json.dumps(predictions))
        with (tmp_path/'cost.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(costs[0]))
            writer.writeheader()
            writer.writerows(costs)
        for name, key in (('protocol.json', 'protocol_sha256'),
                          ('per_image_counts.json', 'per_image_counts_sha256'), ('cost.csv', 'cost_sha256')):
            result[key] = hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
        (tmp_path/'result.json').write_text(json.dumps(result))

    save()
    return dict(directory=tmp_path, expected=expected, hashes=hashes, environment=copy.deepcopy(environment),
                sources=sources, protocol=protocol, result=result, costs=costs, save=save)


def validate_fixture(cell, **kwargs):
    cell['save']()
    return validate_cell(cell['directory'], cell['expected'], cell['result']['method'], cell['sources'], **kwargs)


def test_validate_cell_preserves_old_api_and_accepts_parent_bindings(sealed_cell):
    old = validate_fixture(sealed_cell)
    bound = validate_fixture(sealed_cell, expected_input_hashes=sealed_cell['hashes'],
                             expected_environment=sealed_cell['environment'])
    assert old == bound
    assert bound['measured_images'] == 1
    assert bound['ms_per_image'] == 3000
    assert bound['visual_inputs_per_image'] == 8


def test_returned_costs_are_recomputed_even_within_roundoff_tolerance(sealed_cell):
    sealed_cell['result']['ms_per_image'] += 1e-9
    assert validate_fixture(sealed_cell)['ms_per_image'] == 3000.


@pytest.mark.parametrize('change', ['image', 'label', 'reorder', 'missing', 'metadata', 'parent_missing', 'malformed'])
def test_cell_input_hashes_are_bound_to_parent_in_pair_order(sealed_cell, change):
    inputs = sealed_cell['protocol']['input_hashes']
    if change in {'image', 'label'}:
        inputs['sha256'][0 if change == 'image' else 1] = '0' * 64
    elif change == 'reorder':
        inputs['sha256'].reverse()
    elif change == 'missing':
        inputs['sha256'].pop()
    elif change == 'metadata':
        inputs['file_metadata'] = [[99, 1]] * 10
    elif change == 'parent_missing':
        del sealed_cell['hashes'][sealed_cell['expected']['image_ids'][0]]
    else:
        sealed_cell['hashes'][sealed_cell['expected']['image_ids'][0]] = None
        inputs['sha256'][0] = None
    with pytest.raises(ValueError, match='input content'):
        validate_fixture(sealed_cell, expected_input_hashes=sealed_cell['hashes'])


@pytest.mark.parametrize('field', ['python', 'torch', 'cuda', 'gpu', 'gpu_inventory', 'cuda_visible_devices',
                                  'float32_matmul_precision', 'matmul_tf32', 'cudnn_tf32', 'missing'])
def test_cell_environment_is_bound_to_parent(sealed_cell, field):
    actual = sealed_cell['protocol']['environment']
    if field == 'missing':
        del actual['gpu']
    else:
        actual[field] = 'changed'
    with pytest.raises(ValueError, match='environment'):
        validate_fixture(sealed_cell, expected_environment=sealed_cell['environment'])


@pytest.mark.parametrize('method,steps', list(METHOD_UPDATE_STEPS.items()))
def test_step_contract_covers_all_baselines_and_no_update_controls(sealed_cell, method, steps):
    sealed_cell['result']['method'] = method
    for record in (sealed_cell['protocol'], sealed_cell['result']):
        record['method_settings']['update_steps'] = steps
    for cost in sealed_cell['costs']:
        cost['update_steps'] = steps
    validate_fixture(sealed_cell)
    for record in (sealed_cell['protocol'], sealed_cell['result']):
        record['method_settings']['update_steps'] = steps + 1
    for cost in sealed_cell['costs']:
        cost['update_steps'] = steps + 1
    with pytest.raises(ValueError, match='fixed contract'):
        validate_fixture(sealed_cell)


@pytest.mark.parametrize('key', ['warmup_episodes', 'measured_images', 'ms_per_image',
    'adaptation_ms_per_image', 'inference_ms_per_image', 'peak_allocated_mib',
    'visual_inputs_per_image', 'visual_calls_per_image', 'all_visual_inputs', 'all_visual_calls'])
def test_all_exported_cost_summaries_are_recomputed(sealed_cell, key):
    sealed_cell['result'][key] += 1
    with pytest.raises(ValueError):
        validate_fixture(sealed_cell)


@pytest.mark.parametrize('field,value', [
    ('adaptation_seconds', -1), ('inference_seconds', float('nan')), ('total_seconds', -3),
    ('total_seconds', 0), ('total_seconds', float('inf')), ('total_seconds', 4),
    ('peak_allocated_mib', -1), ('calls', 0), ('calls', -1), ('calls', 1.5),
    ('inputs', 1), ('inputs', -8), ('patches', 0), ('patches', 1), ('episode', .5),
    ('images', 1.5), ('update_steps', 10.5)])
@pytest.mark.parametrize('episode', [0, 2])
def test_invalid_episode_costs_include_warmup_and_measured_rows(sealed_cell, field, value, episode):
    sealed_cell['costs'][episode][field] = value
    if field == 'patches' and value == 1 and episode == 2:
        sealed_cell['costs'][episode]['inputs'] = 0
    with pytest.raises(ValueError):
        validate_fixture(sealed_cell)


def test_consistently_negative_timing_does_not_pass_recomputation(sealed_cell):
    for cost in sealed_cell['costs']:
        cost.update(adaptation_seconds=-2, inference_seconds=-1, total_seconds=-3)
    sealed_cell['result'].update(ms_per_image=-3000, adaptation_ms_per_image=-2000,
                                 inference_ms_per_image=-1000)
    with pytest.raises(ValueError, match='Invalid baseline cost'):
        validate_fixture(sealed_cell)


def test_timing_scope_cannot_change(sealed_cell):
    sealed_cell['protocol']['timing_scope'] = 'unsynchronized inference only'
    with pytest.raises(ValueError, match='timing scope'):
        validate_fixture(sealed_cell)
