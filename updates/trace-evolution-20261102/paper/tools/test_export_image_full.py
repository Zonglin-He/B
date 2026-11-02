"""Synthetic metadata tests: no real model, prediction array, label or GPU."""

import copy
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from paper_evolution.tools import export_image_full as export

VERIFY_REFERENCES = export.verify_references
SUBPROCESS_RUN = subprocess.run


class Tests(unittest.TestCase):
    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))

    def raw(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def read(self, name):
        return json.loads((self.root / name).read_bytes())

    def sha(self, name):
        return export.digest((self.root / name).read_bytes())

    def patch(self, name, value):
        patcher = mock.patch.object(export, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def lines(self, name, values):
        self.raw(name, ''.join(json.dumps(v) + '\n' for v in values))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'suite'
        self.paper = Path(self.temp.name) / 'paper'
        self.generated = self.paper / 'generated'
        self.generated.mkdir(parents=True)
        for n in (*export.OUTPUTS, 'unrelated.tex'):
            (self.generated / n).write_text('KEEP ' + n)
        self.before = {p.name: p.read_bytes() for p in self.generated.iterdir()}
        env = mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='')
        env.start()
        self.addCleanup(env.stop)
        self.patch('POPULATIONS', {d: 5 for d in export.DATASETS})
        names = list(export.INHERITED_FILES)
        names += [tree + f'synthetic_{i}.py' for tree in export.INHERITED_TREES for i in range(14)]
        names += ['trace_ovss/extra_a.py', 'trace_ovss/extra_b.py']
        names += [f'other/source_{i}.txt' for i in range(88)]
        for n in names:
            self.raw('runtime/' + n, '# synthetic source: ' + n)
        sources = {n: self.sha('runtime/' + n) for n in names}
        self.make_selection(sources)
        self.plan = dict(datasets=list(export.DATASETS), populations=export.POPULATIONS, arms=list(export.ARMS),
            selected_global_mode='C', selection_sha256=export.SELECTION_SHA,
            selection_prepared_sha256=export.SELECTION_PREPARED_SHA, formal_cells=6, arm_results=18,
            gt_free_gates=6, gate_images_per_dataset=4)
        self.write('frozen_plan.json', self.plan)
        self.patch('PLAN_SHA', self.sha('frozen_plan.json'))
        self.manifests, pins = {}, {}
        self.prepared = dict(format_version=1, worker='image_full_queue', suite_root=str(self.root),
            runtime_root=str(self.root / 'runtime'), arms=list(export.ARMS), selected_mode='C',
            selection_sha256=export.SELECTION_SHA, exposure_note=export.EXPOSURE, runtime_hashes=sources,
            plan_sha256=export.PLAN_SHA, python='/synthetic/python', python_sha256='1' * 64,
            cells=export.entries(), child_receipts=[])
        for e in self.prepared['cells']:
            d = e['dataset']
            cfg = dict(dataset='SyntheticDataset', data_subdir='synthetic', data_dir='/synthetic',
                batch_size=2, lr=.001, steps=2, trace_max_regions=2, trace_view_batch_size=3,
                patch_stride=112, resize=[224, 224], patch_size=[224, 224], class_extensions=False)
            inputs = [dict(image=f'/synthetic/{d}/{i:03d}.jpg', label=f'/synthetic/{d}/{i:03d}.png',
                           image_sha256='1' * 64, label_metadata=[10, 20]) for i in range(5)]
            checkpoint = dict(path='/synthetic/checkpoint.pt', sha256='3' * 64)
            parent = dict(dataset=d, seed=0, full_split=True, images=5, available_images=5,
                inputs=inputs, config=cfg, checkpoint=checkpoint, classes=['present', 'absent'],
                queries=['present', 'absent'], query_to_class=None, reduce_zero_label=False)
            parent_path = f'parents/{d}/prediction_protocol.json'
            self.write(parent_path, parent)
            pins[d] = self.sha(parent_path), '4' * 64
            directory = (export.VOC20_ROOT if d == 'voc20' else export.PARENT_ROOT) / 'cells/mechanism' / d
            self.write(f'parents/{d}/parent_receipt.json', dict(protocol_path=str(directory / 'prediction_protocol.json'),
                protocol_sha256=pins[d][0], seal_path=str(directory / 'PREDICTIONS_SEALED.json'),
                seal_sha256=pins[d][1], complete=True, gt_contents_opened=False, images=5, sealed_protocol_sha256=pins[d][0]))
            self.write(e['config'], cfg)
            manifest = dict(format_version=1, worker='image_full', dataset=d, stage='full', seed=0, order_seed=None,
                max_images=5, arms=list(export.ARMS), arm_order='alternating', suite_root=str(self.root),
                config=cfg, config_path=str(self.root / e['config']), config_sha256=self.sha(e['config']),
                checkpoint=checkpoint, timing_scope=export.TIMING_SCOPE, exposure_scope=export.EXPOSURE,
                preupdate_prediction_status=export.PREUPDATE, plan_sha256=export.PLAN_SHA,
                selection_sha256=export.SELECTION_SHA, parent_protocol_sha256=pins[d][0], parent_seal_sha256=pins[d][1],
                source_sha256={str(self.root / 'runtime' / k): v for k, v in sources.items()},
                panel=dict(inputs=inputs, ordered_ids=[r['image'] for r in inputs], full_split=True,
                           **{k: parent[k] for k in ('classes', 'queries', 'query_to_class', 'reduce_zero_label')}))
            self.write(e['manifest'], manifest)
            e['manifest_sha256'] = self.sha(e['manifest'])
            self.manifests[d] = manifest
            self.prepared['child_receipts'].append(self.receipt('prepare', d))
        self.patch('PARENT_HASHES', pins)
        self.prepared['artifact_sha256'] = {str(p.relative_to(self.root)): self.sha(str(p.relative_to(self.root)))
            for folder in ('parents', 'selection', 'configs', 'manifests', 'logs')
            for p in (self.root / folder).rglob('*') if p.is_file()}
        self.prepared['artifact_sha256']['frozen_plan.json'] = self.sha('frozen_plan.json')
        self.write('SUITE_PREPARED.json', self.prepared)
        self.prepared_sha = self.sha('SUITE_PREPARED.json')
        self.write('EXECUTION_STARTED.json', dict(suite_prepared_sha256=self.prepared_sha, pid=1234))
        self.runtime = dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
            deterministic_warn_only=False, deterministic_debug_mode=2, cudnn_deterministic=True, cudnn_benchmark=False)
        self.lock = dict(path='/synthetic/gpu.lock', inode=12, pid=1235, exclusive=True)
        self.results = {}
        for e in self.prepared['cells']:
            d, m = e['dataset'], self.manifests[e['dataset']]
            marker = f'gates/{d}.json.STARTED.json'
            self.write(marker, dict(status='started', manifest_sha256=e['manifest_sha256']))
            comparison = dict(candidate_masks_equal=True, reset_equal=True, changed_label_pixels=0, candidates=4,
                max_support_abs_difference=0., max_coefficient_abs_difference=0., max_loss_abs_difference=0.,
                max_gradient_abs_difference=0., actual_arm_counts={a: self.cost_row(0, a, 2) for a in export.ARMS})
            self.write(f'gates/{d}.json', dict(passed=True, gt_contents_opened=False, images=4, dataset=d,
                manifest_sha256=e['manifest_sha256'], suite_prepared_sha256=self.prepared_sha,
                attempt_sha256=self.sha(marker), checkpoint=m['checkpoint'], source_sha256=m['source_sha256'],
                ordered_ids=m['panel']['ordered_ids'][:4], configured_batch_size=2, batch_images=[2, 2],
                comparisons=[comparison, comparison], deterministic_runtime=self.runtime, gpu_lock=self.lock))
        self.gate_hashes = {d: self.sha(f'gates/{d}.json') for d in export.DATASETS}
        for e in self.prepared['cells']:
            self.make_cell(e)
        receipts = [self.receipt(action, e['dataset']) for action in ('gate', 'predict', 'score') for e in self.prepared['cells']]
        self.write('execution_children.json', receipts)
        self.write('queue_state.json', dict(phase='completed', active_cell=None, error=None, passed_gates=6,
            sealed_predictions=6, scored_cells=6, total_cells=6, total_arm_results=18))
        self.reseal()
        self.service_overrides = {}
        process = mock.patch.object(export.subprocess, 'run', side_effect=self.systemctl)
        process.start()
        self.addCleanup(process.stop)
        self.refs = dict(reported=dict(methods=['No Adapt', 'TENT', 'TPT', 'WATT', 'CLIPArTT', 'MLMP'],
            datasets={d: {m: [75.91 if m == 'No Adapt' else 83.76, None if m == 'No Adapt' else .01]
                          for m in ('No Adapt', 'TENT', 'TPT', 'WATT', 'CLIPArTT', 'MLMP')}
                      for d in export.DATASETS if d != 'coco_cd'}),
            native_cd={m: dict(miou_unrounded=72.20029853468192) for m in ('noadapt', 'tent', 'tpt', 'mlmp')},
            native_cd_delta_pp=2.78, reported_deltas_pp={d: -.2727808136116 if d == 'voc20' else 1.
                for d in export.DATASETS if d != 'coco_cd'}, reported_pdf_sha256={})
        self.patch('verify_references', lambda paper, cells: (self.refs, ()))

    def receipt(self, action, dataset):
        argv = ['/synthetic/python', '-B', '-m', 'tools.evolution_image_full.run', action, '--suite', str(self.root)]
        if action != 'prepare':
            argv += ['--suite-sha256', self.prepared_sha]
        argv += ['--dataset', dataset]
        log, receipt = f'logs/{action}-{dataset}.log', f'logs/{action}-{dataset}.receipt.json'
        self.raw(log, '')
        r = dict(argv=argv, log=log, log_sha256=self.sha(log), output_bytes=0, retained_bytes=0,
                 exit_code=0, error=None, cpu_only=action in ('prepare', 'score'))
        self.write(receipt, r)
        return dict(r, receipt=receipt, receipt_sha256=self.sha(receipt))

    def make_selection(self, sources):
        self.write('selection/SUITE_PREPARED.json', dict(runtime_hashes=sources))
        self.patch('SELECTION_PREPARED_SHA', self.sha('selection/SUITE_PREPARED.json'))
        rule = dict(tie_band_pp=.02)
        self.write('selection/frozen_plan.json', dict(selection=rule))
        barrier = dict(complete=True, stage='calibration', gt_contents_opened=False,
            suite_prepared_sha256=export.SELECTION_PREPARED_SHA,
            prediction_seal_sha256={f'cells/calibration/{d}/sorted': '5' * 64 for d in export.base.DATASETS})
        self.write('selection/ALL_CALIBRATION_PREDICTIONS_SEALED.json', barrier)
        selection = dict(complete=True, selected_mode='C', suite_prepared_sha256=export.SELECTION_PREPARED_SHA,
            rule=rule, calibration_barrier_sha256=self.sha('selection/ALL_CALIBRATION_PREDICTIONS_SEALED.json'),
            calibration_score_sha256={}, candidates={})
        for d in export.base.DATASETS:
            relative = f'cells/calibration/{d}/sorted'
            prefix = 'selection/' + relative + '/'
            result = dict(status='completed', stage='calibration', dataset=d, images=128, order_seed=None,
                arm_results={a: dict(intersection=[v * 128, 0], union=[12800, 0], miou_unrounded=float(v))
                    for a, v in (('image_confidence', 50), ('image_T', 49), ('image_C', 51), ('image_min', 50))},
                costs={a: dict(diagnostic_encoder_inputs=128 if a != 'image_min' else 256)
                       for a in ('image_T', 'image_C', 'image_min')})
            for name in export.SCORING:
                self.write(prefix + name, result if name == 'result.json' else {})
            self.write(prefix + 'SCORED_SEALED.json', dict(prediction_seal_sha256='5' * 64,
                artifact_sha256={n: self.sha(prefix + n) for n in export.SCORING}))
            selection['calibration_score_sha256'][relative] = self.sha(prefix + 'SCORED_SEALED.json')
        for m, v in (('T', -1.), ('C', 1.), ('min', 0.)):
            selection['candidates'][m] = dict(margin_pp={d: v for d in export.base.DATASETS},
                worst_dataset_margin_pp=v, macro_margin_pp=v, diagnostic_encoder_inputs=256 if m != 'min' else 512)
        self.write('selection/SELECTION_SEALED.json', selection)
        self.patch('SELECTION_SHA', self.sha('selection/SELECTION_SEALED.json'))
        direct = ['SELECTION_SEALED.json', 'SUITE_PREPARED.json', 'frozen_plan.json', 'ALL_CALIBRATION_PREDICTIONS_SEALED.json']
        direct += [f'cells/calibration/{d}/sorted/{n}' for d in export.base.DATASETS for n in ('SCORED_SEALED.json', 'result.json')]
        self.write('selection/SUITE_SEALED.json', dict(complete=True, selected_mode='C', selection_sha256=export.SELECTION_SHA,
            suite_prepared_sha256=export.SELECTION_PREPARED_SHA, formal_cells=8, arm_results=48,
            artifact_sha256={n: self.sha('selection/' + n) for n in direct}))

    def cost_row(self, episode, arm, images):
        candidates = 2 * images
        views = candidates if arm == 'image_C' else 0
        calls = (views + 2) // 3
        order = export.ARMS if episode % 2 == 0 else tuple(reversed(export.ARMS))
        return dict(episode=episode, arm=arm, images=images, patches=images, warmup=episode == 0,
            steps=2, active_candidates=candidates, candidate_count=candidates, candidate_sha256='2' * 64,
            order_position=order.index(arm), seconds=float(episode + export.ARMS.index(arm) + 1),
            diagnostic_forward_inputs=views, diagnostic_encoder_inputs=views,
            diagnostic_forward_calls=calls, diagnostic_encoder_calls=calls,
            eval_forward_calls=2, eval_forward_inputs=2 * images, adapt_forward_calls=2,
            adapt_forward_inputs=2 * images, encoder_calls=4 + calls, encoder_inputs=4 * images + views)

    def make_cell(self, e):
        prefix, d = e['output'] + '/', e['dataset']
        m = self.manifests[d]
        self.write(prefix + 'prediction_protocol.json', dict(manifest=m, manifest_path=str(self.root / e['manifest']),
            manifest_sha256=e['manifest_sha256'], suite_prepared_sha256=self.prepared_sha,
            selection_sha256=export.SELECTION_SHA, gate_sha256=self.gate_hashes,
            deterministic_runtime=self.runtime, gpu_lock=self.lock))
        self.write(prefix + 'initial_state.json', dict(model_sha256='6' * 64, adapted_parameter_names=['visual.ln.weight']))
        costs = [self.cost_row(e, a, min(2, 5 - e * 2)) for e in range(3)
                 for a in (export.ARMS if e % 2 == 0 else tuple(reversed(export.ARMS)))]
        totals = {a: {k: sum(r[k] for r in costs if r['arm'] == a) for k in export.COUNTERS} for a in export.ARMS}
        timings = {}
        for a in export.ARMS:
            seconds = sum(r['seconds'] for r in costs if r['arm'] == a and r['episode'] != 0)
            timings[a] = dict(seconds=seconds, measured_images=3, measured_episodes=2, seconds_per_image=seconds / 3,
                images_per_second=3 / seconds, warmup_episode=0, accuracy_includes_warmup=True)
        self.lines(prefix + 'cost.jsonl', costs)
        values = dict(zip(export.ARMS, (49, 50, 51)))
        self.lines(prefix + 'scores_by_episode.jsonl', [dict(episode=e, images=images,
            arms={a: dict(intersection=[values[a] * images, 0], union=[100 * images, 0]) for a in export.ARMS})
            for e, images in enumerate((2, 2, 1))])
        predicted = dict(status='predicted', images=5, full_split=True, costs=totals, timings=timings, timing_scope=export.TIMING_SCOPE)
        self.write(prefix + 'prediction_result.json', predicted)
        result = dict(predicted, status='completed', dataset=d, stage='full', seed=0, selection_sha256=export.SELECTION_SHA,
            exposure_scope=export.EXPOSURE, label_contents_first_opened_after_all_full_predictions_sealed=True,
            arm_results={a: dict(intersection=[values[a] * 5, 0], union=[500, 0], miou_unrounded=float(values[a])) for a in export.ARMS})
        self.results[d] = result
        self.write(prefix + 'result.json', result)
        self.write(prefix + 'score_inputs.json', dict(label_sha256={r['label']: '7' * 64 for r in m['panel']['inputs']}))

    def reseal(self):
        predictions = {}
        for e in self.prepared['cells']:
            prefix = e['output'] + '/'
            artifacts = {f'predictions/{a}/{i:06d}.npy': '8' * 64 for a in export.ARMS for i in range(5)}
            artifacts.update({f'episodes/{i:06d}.{a}.json': '9' * 64 for a in export.ARMS for i in range(3)})
            artifacts.update({n: self.sha(prefix + n) for n in export.SMALL})
            self.write(prefix + 'PREDICTIONS_SEALED.json', dict(complete=True, gt_contents_opened=False, images=5,
                arms=list(export.ARMS), suite_prepared_sha256=self.prepared_sha, artifact_sha256=artifacts))
            predictions[e['output']] = self.sha(prefix + 'PREDICTIONS_SEALED.json')
        barrier = dict(complete=True, stage='full', gt_contents_opened=False,
                       suite_prepared_sha256=self.prepared_sha, prediction_seal_sha256=predictions)
        self.write('ALL_FULL_PREDICTIONS_SEALED.json', barrier)
        for e in self.prepared['cells']:
            prefix = e['output'] + '/'
            started = dict(prediction_seal_sha256=predictions[e['output']],
                barrier_sha256=self.sha('ALL_FULL_PREDICTIONS_SEALED.json'), suite_prepared_sha256=self.prepared_sha)
            self.write(prefix + 'SCORING_STARTED.json', started)
            r = self.read(prefix + 'result.json')
            r.update({k: v for k, v in started.items() if k != 'suite_prepared_sha256'})
            self.write(prefix + 'result.json', r)
            self.write(prefix + 'SCORED_SEALED.json', dict(prediction_seal_sha256=predictions[e['output']],
                artifact_sha256={n: self.sha(prefix + n) for n in export.SCORING}))
        self.csv_rows()
        self.outer_seal()

    def csv_rows(self):
        columns = ('dataset', 'seed', 'arm', 'images', 'full_split', 'miou_unrounded', 'margin_vs_image_confidence_pp',
            'diagnostic_encoder_inputs', 'measured_images', 'seconds', 'seconds_per_image', 'images_per_second',
            'manifest_sha256', 'selection_sha256', 'prediction_seal_sha256', 'score_seal_sha256', 'exposure')
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for e in self.prepared['cells']:
            prefix = e['output'] + '/'
            r = self.read(prefix + 'result.json')
            for a in export.ARMS:
                writer.writerow(dict(dataset=e['dataset'], seed=0, arm=a, images=5, full_split=True,
                    miou_unrounded=r['arm_results'][a]['miou_unrounded'],
                    margin_vs_image_confidence_pp=r['arm_results'][a]['miou_unrounded'] - r['arm_results']['image_confidence']['miou_unrounded'],
                    diagnostic_encoder_inputs=r['costs'][a]['diagnostic_encoder_inputs'], **{k: r['timings'][a][k]
                        for k in ('measured_images', 'seconds', 'seconds_per_image', 'images_per_second')},
                    manifest_sha256=e['manifest_sha256'], selection_sha256=export.SELECTION_SHA,
                    prediction_seal_sha256=self.sha(prefix + 'PREDICTIONS_SEALED.json'),
                    score_seal_sha256=self.sha(prefix + 'SCORED_SEALED.json'), exposure=export.EXPOSURE))
        self.raw('rows.csv', stream.getvalue())

    def outer_seal(self):
        artifacts = dict(self.prepared['artifact_sha256'])
        for folder in ('cells', 'gates', 'logs'):
            for p in (self.root / folder).rglob('*'):
                if p.is_file():
                    artifacts[str(p.relative_to(self.root))] = self.sha(str(p.relative_to(self.root)))
        artifacts.update({n: self.sha(n) for n in ('SUITE_PREPARED.json', 'EXECUTION_STARTED.json',
            'execution_children.json', 'queue_state.json', 'rows.csv', 'ALL_FULL_PREDICTIONS_SEALED.json')})
        self.write('SUITE_SEALED.json', dict(complete=True, full_split=True, formal_cells=6, arm_results=18,
            passed_gt_free_gates=6, selected_mode='C', selection_sha256=export.SELECTION_SHA,
            suite_prepared_sha256=self.prepared_sha, runtime_hashes=self.prepared['runtime_hashes'],
            exposure_note=export.EXPOSURE, service_success_verified=False, artifact_sha256=artifacts))

    def systemctl(self, argv, **kwargs):
        self.assertEqual(argv[:4], ['systemctl', '--user', 'show', export.SERVICE])
        state = dict(Id=export.SERVICE, LoadState='loaded', ActiveState='active', SubState='exited',
            Result='success', MainPID='0', ExecMainPID='1234', ExecMainCode='1', ExecMainStatus='0',
            ExecMainStartTimestamp='Thu 2026-10-29 03:33:23 CST', ExecMainExitTimestamp='Thu 2026-10-29 06:29:28 CST',
            ExecStart=f'{{ argv[]=/synthetic/python -B -m tools.evolution_image_full.queue run --suite {self.root} '
                      f'--suite-sha256 {self.prepared_sha} ; }}')
        state.update(self.service_overrides)
        return subprocess.CompletedProcess(argv, 0, '\n'.join(k + '=' + v for k, v in state.items()), '')

    def run_import(self, check_only=False):
        return export.run(self.root, self.paper, expected_hash=self.prepared_sha, check_only=check_only)

    def unchanged(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.generated.iterdir()}, self.before)

    def invalid(self, message):
        with self.assertRaisesRegex((export.Invalid, KeyError, ValueError), message):
            self.run_import()
        self.unchanged()

    def test_complete_import_and_full_precision(self):
        status, code = self.run_import()
        self.assertEqual((status['status'], code), ('imported', 0))
        p = json.loads((self.generated / export.OUTPUTS[-1]).read_bytes())
        self.assertEqual(len(p['cells']), 6)
        self.assertEqual(p['verified_source_count'], 149)
        self.assertEqual(p['inherited_source_count'], 61)
        self.assertEqual(p['verified_execution_children'], 18)
        self.assertEqual(p['cells'][0]['costs']['image_C']['measured_images'], 3)
        self.assertEqual(p['cells'][0]['costs']['image_C']['diagnostic_encoder_inputs'], 10)
        comparison = (self.generated / export.OUTPUTS[0]).read_text()
        self.assertIn('\\begin{table}[t]', comparison)
        self.assertIn('Uniform & Confidence & TRACE(C)', comparison)
        self.assertNotIn('image_', comparison)
        reference = (self.generated / export.OUTPUTS[2]).read_text()
        self.assertIn('-0.27', reference)
        self.assertIn(r'\pm 0.01', reference)
        self.assertIn('No Adapt (R)', reference)
        self.assertEqual(p['references']['reported_deltas_pp']['voc20'], -.2727808136116)
        self.assertEqual((self.generated / 'unrelated.tex').read_bytes(), self.before['unrelated.tex'])

    def test_check_only_preserves_all_outputs(self):
        self.assertEqual(self.run_import(True)[0]['status'], 'ready')
        self.unchanged()

    def test_population_and_order_semantics(self):
        e = copy.deepcopy(self.prepared['cells'][0])
        for mutate in (lambda m: m['panel']['inputs'].pop(), lambda m: m['panel']['ordered_ids'].reverse(),
                       lambda m: m.update(max_images=4), lambda m: m['config'].update(steps=99)):
            with self.subTest(mutate=mutate):
                m = copy.deepcopy(self.manifests['pc60'])
                mutate(m)
                self.write(e['manifest'], m)
                e['manifest_sha256'] = self.sha(e['manifest'])
                with self.assertRaises(export.Invalid):
                    export.verify_manifest(export.Evidence(self.root), e, self.prepared)
        self.unchanged()

    def test_manifest_full_source_binding(self):
        e = copy.deepcopy(self.prepared['cells'][0])
        m = copy.deepcopy(self.manifests['pc60'])
        m['source_sha256'].pop(next(iter(m['source_sha256'])))
        self.write(e['manifest'], m)
        e['manifest_sha256'] = self.sha(e['manifest'])
        with self.assertRaisesRegex(export.Invalid, '149-source'):
            export.verify_manifest(export.Evidence(self.root), e, self.prepared)

    def test_61_source_continuity(self):
        prepared = copy.deepcopy(self.prepared)
        prepared['runtime_hashes']['tools/suite_inputs.py'] = '0' * 64
        with self.assertRaisesRegex(export.Invalid, '61-source'):
            export.verify_selection(export.Evidence(self.root), prepared)

    def test_broken_source_fails_without_output(self):
        self.raw('runtime/tools/suite_inputs.py', '# changed')
        self.invalid('Seal/hash mismatch')

    def test_pinned_selection_drift(self):
        self.write('selection/SELECTION_SEALED.json', dict(complete=True, selected_mode='T'))
        self.invalid('Seal/hash mismatch')

    def test_selection_semantics(self):
        p = 'selection/SELECTION_SEALED.json'
        selection = self.read(p)
        selection['selected_mode'] = 'T'
        self.write(p, selection)
        with mock.patch.object(export, 'SELECTION_SHA', self.sha(p)):
            with self.assertRaisesRegex(export.Invalid, 'selected-C'):
                export.verify_selection(export.Evidence(self.root), self.prepared)

    def test_global_barrier_coverage(self):
        b = self.read('ALL_FULL_PREDICTIONS_SEALED.json')
        b['prediction_seal_sha256'].pop(next(iter(b['prediction_seal_sha256'])))
        self.write('ALL_FULL_PREDICTIONS_SEALED.json', b)
        self.outer_seal()
        self.invalid('Global six-cell barrier')

    def test_gate_semantic_failure(self):
        name = 'gates/pc60.json'
        gate = self.read(name)
        gate['comparisons'][0]['max_gradient_abs_difference'] = 1e-4
        self.write(name, gate)
        self.outer_seal()
        self.invalid('Gate numeric')

    def test_gate_zero_probe_violation(self):
        name = 'gates/pc60.json'
        gate = self.read(name)
        gate['comparisons'][0]['actual_arm_counts']['image_uniform']['diagnostic_encoder_inputs'] = 1
        self.write(name, gate)
        self.outer_seal()
        self.invalid('0/0/1')

    def test_actual_cost_schedule(self):
        name = 'cells/full/pc60/seed0/cost.jsonl'
        rows = list(export.jsonlines((self.root / name).read_bytes(), 9))
        rows[0]['diagnostic_encoder_inputs'] = 1
        self.lines(name, rows)
        self.reseal()
        self.invalid('0/0/1')

    def test_candidate_counts_across_arms(self):
        name = 'cells/full/pc60/seed0/cost.jsonl'
        rows = list(export.jsonlines((self.root / name).read_bytes(), 9))
        rows[0]['candidate_count'] -= 1
        self.lines(name, rows)
        self.reseal()
        self.invalid('Candidate identity/count')

    def test_cost_arm_order(self):
        name = 'cells/full/pc60/seed0/cost.jsonl'
        rows = list(export.jsonlines((self.root / name).read_bytes(), 9))
        rows[0], rows[1] = rows[1], rows[0]
        self.lines(name, rows)
        self.reseal()
        self.invalid('Cost episode/arm order')

    def test_timing_excludes_only_episode_zero(self):
        for name in ('prediction_result.json', 'result.json'):
            p = 'cells/full/pc60/seed0/' + name
            r = self.read(p)
            r['timings']['image_C']['measured_images'] = 2
            self.write(p, r)
        self.reseal()
        self.invalid('Timing denominator')

    def test_pooled_counts_not_csv_trust(self):
        p = 'cells/full/pc60/seed0/scores_by_episode.jsonl'
        rows = list(export.jsonlines((self.root / p).read_bytes(), 3))
        rows[0]['arms']['image_C']['intersection'][0] += 1
        self.lines(p, rows)
        self.reseal()
        self.invalid('Pooled episode counts')

    def test_pooled_metric_recomputed(self):
        p = 'cells/full/pc60/seed0/result.json'
        r = self.read(p)
        r['arm_results']['image_C']['miou_unrounded'] = 52.
        self.write(p, r)
        self.reseal()
        self.invalid('Pooled mIoU')

    def test_score_episode_population(self):
        p = 'cells/full/pc60/seed0/scores_by_episode.jsonl'
        rows = list(export.jsonlines((self.root / p).read_bytes(), 3))
        rows[-1]['images'] = 2
        self.lines(p, rows)
        self.reseal()
        self.invalid('Pooled episode population')

    def test_csv_value_duplicate_and_missing(self):
        original = (self.root / 'rows.csv').read_text()
        for change in ('value', 'duplicate', 'missing'):
            with self.subTest(change=change):
                rows = list(csv.reader(io.StringIO(original)))
                if change == 'value':
                    rows[1][5] = '99'
                elif change == 'duplicate':
                    rows[1] = rows[2]
                else:
                    rows.pop()
                stream = io.StringIO()
                csv.writer(stream).writerows(rows)
                self.raw('rows.csv', stream.getvalue())
                self.outer_seal()
                self.invalid('CSV')

    def test_child_failed_or_wrong_stage(self):
        original = self.read('execution_children.json')
        for field, value in (('exit_code', 1), ('cpu_only', True)):
            r = copy.deepcopy(original)
            r[0][field] = value
            self.write('execution_children.json', r)
            self.outer_seal()
            self.invalid('Child command/exit/stage')

    def test_service_terminal_failures(self):
        for field, value in (('Result', 'oom-kill'), ('ExecMainStatus', '9'), ('ExecMainCode', '2'),
                             ('ExecMainPID', '9999'), ('ExecStart', 'wrong'), ('LoadState', 'not-found'),
                             ('ExecMainExitTimestamp', '')):
            with self.subTest(field=field):
                self.service_overrides = {field: value}
                self.invalid('Service')

    def test_service_running_pending(self):
        self.service_overrides = dict(MainPID='1234', SubState='running', ActiveState='active')
        with self.assertRaises(export.Pending):
            self.run_import()
        self.unchanged()

    def test_no_suite_seal_no_output(self):
        (self.root / 'SUITE_SEALED.json').unlink()
        with self.assertRaises(export.Pending):
            self.run_import()
        self.unchanged()

    def test_failed_queue_not_pending(self):
        (self.root / 'SUITE_SEALED.json').unlink()
        self.write('queue_state.json', dict(phase='failed'))
        self.invalid('Queue failed')

    def test_late_source_drift_no_output(self):
        original = export.Evidence.stable
        def drift(ev):
            if ev.root == self.root:
                self.raw('runtime/tools/suite_inputs.py', '# late drift')
            return original(ev)
        with mock.patch.object(export.Evidence, 'stable', drift):
            self.invalid('Source drift')

    def test_publish_rollback(self):
        original = export.os.replace
        calls = 0
        def fail_once(a, b):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError('synthetic publication failure')
            return original(a, b)
        with mock.patch.object(export.os, 'replace', side_effect=fail_once):
            with self.assertRaisesRegex(OSError, 'publication failure'):
                self.run_import()
        self.unchanged()

    def test_payload_read_guard(self):
        for name in ('predictions/a.npy', 'cells/full/pc60/seed0/episodes/000000.image_C.json'):
            with self.assertRaisesRegex(export.Invalid, 'Payload scan forbidden'):
                export.Evidence(self.root).raw(name)

    def test_cpu_guard(self):
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            self.invalid('CPU-only')

    def test_duplicate_json_and_nonfinite_rejected(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(export.Invalid):
                export.base.decode(value)

    def reference_fixture(self):
        native = Path(self.temp.name) / 'native'
        cd = dict(dataset='coco_cd', images=5, checkpoint={'sha256': '3' * 64}, classes=['present', 'absent'],
            scores={'image_C': 75.}, ordered_ids_sha256=export.canonical(self.manifests['coco_cd']['panel']['ordered_ids']))
        artifacts = {}
        for method, value in (('noadapt', 68), ('tent', 72), ('tpt', 69), ('mlmp', 70)):
            prefix = native / 'coco_cd' / method
            prefix.mkdir(parents=True)
            p = dict(full_split=True, method=method, seed=0, images=5, available_images=5,
                labels_used_for_adaptation=False, checkpoint_sha256='3' * 64,
                image_ids=self.manifests['coco_cd']['panel']['ordered_ids'],
                label_ids=[r['label'] for r in self.manifests['coco_cd']['panel']['inputs']],
                config={}, method_settings={'protocol': 'synthetic native'})
            (prefix / 'protocol.json').write_text(json.dumps(p))
            r = dict(status='completed', method=method, full_split=True, images=5,
                protocol_sha256=export.digest((prefix / 'protocol.json').read_bytes()),
                intersection=[float(value), 0.], union=[100., 0.], miou_unrounded=float(value))
            (prefix / 'result.json').write_text(json.dumps(r))
            for name in ('protocol.json', 'result.json'):
                artifacts[f'coco_cd/{method}/{name}'] = export.digest((prefix / name).read_bytes())
        (native / 'SUITE_SEALED.json').write_text(json.dumps(dict(artifact_sha256=artifacts)))
        pdf = Path(self.temp.name) / 'reference.pdf'
        pdf.write_bytes(b'synthetic PDF bytes; extraction is mocked')
        reported = copy.deepcopy(self.refs['reported'])
        reported.update(source_pdf=str(pdf), source_pdf_sha256=export.digest(pdf.read_bytes()))
        (self.paper / 'reported_baselines.json').write_text(json.dumps(reported))
        text = 'Table 5:\n'
        for d, label in (('voc20', 'V20'), ('voc21', 'V21'), ('pc60', 'P60'),
                         ('coco_object', 'COCOObject'), ('coco_stuff', 'COCOStuff')):
            tokens = []
            for method in reported['methods']:
                mean, sd = reported['datasets'][d][method]
                tokens.append(f'{mean:.2f}' + (f' \u00b1 {sd:.2f}' if sd is not None else ''))
            text += label + ' (Original) ' + ' '.join(tokens) + '\n'
        text += 'shown in Table 4'
        return native, cd, text

    def test_reported_pdf_crosscheck_and_native_pooled_references(self):
        native, cd, text = self.reference_fixture()
        with mock.patch.object(export.reference, 'NATIVE_ROOT', native), mock.patch.object(
                export.subprocess, 'run', return_value=subprocess.CompletedProcess(['pdftotext'], 0, text, '')) as process:
            refs, evidence = VERIFY_REFERENCES(self.paper, [cd])
        self.assertEqual(process.call_args.args[0][0], 'pdftotext')
        self.assertEqual(refs['native_cd_best'], 'tent')
        self.assertEqual(refs['native_cd_delta_pp'], 3.)
        self.assertEqual(refs['native_cd']['tent']['miou_unrounded'], 72.)
        self.assertEqual(len(evidence), 2)
        for e in evidence:
            e.stable()

    def test_reference_pdf_disagreement_rejected(self):
        native, cd, text = self.reference_fixture()
        text = text.replace('83.76', '83.77', 1)
        with mock.patch.object(export.reference, 'NATIVE_ROOT', native), mock.patch.object(
                export.subprocess, 'run', return_value=subprocess.CompletedProcess(['pdftotext'], 0, text, '')):
            with self.assertRaisesRegex(export.reference.Invalid, 'Published values differ'):
                VERIFY_REFERENCES(self.paper, [cd])
        self.unchanged()

    def test_native_reference_nonpooled_metric_rejected(self):
        native, cd, text = self.reference_fixture()
        path = native / 'coco_cd/tent/result.json'
        r = json.loads(path.read_bytes())
        r['miou_unrounded'] = 73.
        path.write_text(json.dumps(r))
        seal_path = native / 'SUITE_SEALED.json'
        seal = json.loads(seal_path.read_bytes())
        seal['artifact_sha256']['coco_cd/tent/result.json'] = export.digest(path.read_bytes())
        seal_path.write_text(json.dumps(seal))
        with mock.patch.object(export.reference, 'NATIVE_ROOT', native), mock.patch.object(
                export.subprocess, 'run', return_value=subprocess.CompletedProcess(['pdftotext'], 0, text, '')):
            with self.assertRaisesRegex(export.Invalid, 'Pooled mIoU'):
                VERIFY_REFERENCES(self.paper, [cd])

    def test_reference_failure_no_output(self):
        with mock.patch.object(export, 'verify_references', side_effect=ValueError('reference verification failed')):
            self.invalid('reference verification failed')

    @unittest.skipUnless(shutil.which('pdflatex'), 'Optional TeX layout check requires pdflatex')
    def test_single_column_table_width_and_all_tex_compile(self):
        self.run_import()
        comparison = (self.generated / export.OUTPUTS[0]).read_text()
        tabular = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', comparison, re.S).group()
        source = (r'\documentclass[twocolumn]{article}\usepackage{booktabs,natbib}\begin{document}'
            + r'\setbox0=\hbox{\small\setlength{\tabcolsep}{4pt} ' + tabular
            + r'}\typeout{TABLEWIDTH=\the\wd0;COLUMNWIDTH=\the\columnwidth}'
            + ''.join(r'\input{' + str(self.generated / n) + '}' for n in export.OUTPUTS[:3])
            + r'\begin{thebibliography}{1}\bibitem[Noori et al.(2025)]{noori2025mlmp}Reference.'
              r'\end{thebibliography}\end{document}')
        with tempfile.TemporaryDirectory(prefix='image-full-tex-') as directory:
            result = SUBPROCESS_RUN(['pdflatex', '-halt-on-error', '-interaction=nonstopmode',
                '-jobname=tablecheck', '-output-directory', directory],
                input=source.replace('\n', ' ') + '\n', text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertNotIn('Overfull \\hbox', result.stdout)
        widths = re.search(r'TABLEWIDTH=([0-9.]+)pt;COLUMNWIDTH=([0-9.]+)pt', result.stdout)
        self.assertIsNotNone(widths)
        self.assertLessEqual(float(widths[1]), float(widths[2]))


if __name__ == '__main__':
    unittest.main()
