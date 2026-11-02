"""Synthetic bounded metadata; no real suite, predictions, labels, model or GPU."""

import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location('export_image_budget', Path(__file__).with_name('export_image_budget.py'))
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class Tests(unittest.TestCase):
    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))

    def read(self, name):
        return json.loads((self.root / name).read_bytes())

    def sha(self, name):
        return export.digest((self.root / name).read_bytes())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'suite'
        self.paper = Path(self.temp.name) / 'paper'
        self.generated = self.paper / 'generated'
        self.generated.mkdir(parents=True)
        for name in (*export.OUTPUTS, 'unrelated.tex'):
            (self.generated / name).write_text('KEEP ' + name)
        self.before = {p.name: p.read_bytes() for p in self.generated.iterdir()}
        self.env = mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.write('runtime/minimal.py', 'synthetic frozen source')
        plan = Path(__file__).resolve().parents[2] / 'docs/confirmation-image-budget-plan.json'
        (self.root / 'frozen_plan.json').write_bytes(plan.read_bytes())
        self.bundle = dict(audit=dict(global_never_seen_certified=False, historical_index_exhaustive=False,
            strong_independent_confirmation_supported=False, historical_pool_audit=dict(status='unresolved',
                candidate_pool_ids_reconstructed=False, strong_independent_confirmation_supported=False,
                pc60_unused_capacity=dict(supports_identifiable_384_image_unused_cohort=False))), datasets={})
        for d in export.DATASETS:
            cohorts = {s: [dict(official_id=f'{d}/{s}/{i}', image=f'/synthetic/{d}/{s}/{i}.jpg',
                                label=f'/synthetic/{d}/{s}/{i}.png', image_sha256='1' * 64,
                                label_metadata=dict(contents_read=False)) for i in range(n)]
                       for s, n in (('calibration', 128), ('confirmation', 256))}
            cohorts['selected_ids_sha256'] = {s: export.canonical([r['official_id'] for r in rows])
                                              for s, rows in list(cohorts.items())}
            self.bundle['datasets'][d] = cohorts
        self.write('frozen_cohorts.json', self.bundle)
        self.runtime = dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
            deterministic_warn_only=False, deterministic_debug_mode=2, cudnn_deterministic=True,
            cudnn_benchmark=False, sdpa_enabled=dict(flash=True, mem_efficient=True, math=True, cudnn=True))
        self.lock = dict(path='/synthetic/gpu.lock', inode=12, pid=100, exclusive=True)
        self.prepared = dict(format_version=1, worker='confirmation_queue', runtime_root=str(self.root / 'runtime'),
            runtime_hashes={'minimal.py': self.sha('runtime/minimal.py')}, artifact_sha256={}, cells=export.entries(),
            arms=list(export.ARMS), exposure_note=export.EXPOSURE, plan_sha256=self.sha('frozen_plan.json'),
            cohort_sha256=self.sha('frozen_cohorts.json'), python='/synthetic/python')
        self.results = {}
        self.manifests = {}
        for d in export.DATASETS:
            cfg = dict(batch_size=64, steps=2, trace_view_batch_size=16)
            self.write(f'configs/{d}.json', cfg)
            self.write(f'parents/{d}/prediction_protocol.json', dict(config=cfg, checkpoint={'sha256': '3' * 64}))
            self.write(f'parents/{d}/checkpoint_identity.json', {'sha256': '3' * 64})
            self.write(f'parents/{d}/PREDICTIONS_SEALED.json', dict(complete=True, gt_contents_opened=False,
                artifact_sha256={'prediction_protocol.json': self.sha(f'parents/{d}/prediction_protocol.json')}))
        for entry in self.prepared['cells']:
            self.make_cell(entry)
        frozen = ['frozen_plan.json', 'frozen_cohorts.json']
        frozen += [str(p.relative_to(self.root)) for folder in ('configs', 'parents', 'manifests')
                   for p in (self.root / folder).rglob('*') if p.is_file()]
        self.prepared['artifact_sha256'] = {n: self.sha(n) for n in frozen}
        self.write('SUITE_PREPARED.json', self.prepared)
        self.prepared_sha = self.sha('SUITE_PREPARED.json')
        self.write('EXECUTION_STARTED.json', dict(suite_prepared_sha256=self.prepared_sha, pid=1234))
        self.barriers = {}
        for stage in ('calibration', 'confirmation'):
            barrier = dict(complete=True, stage=stage, suite_prepared_sha256=self.prepared_sha,
                prediction_seal_sha256={e['output']: self.sha(e['output'] + '/PREDICTIONS_SEALED.json')
                    for e in self.prepared['cells'] if e['stage'] == stage}, gt_contents_opened=False)
            self.barriers[stage] = barrier
            self.write(f'ALL_{stage.upper()}_PREDICTIONS_SEALED.json', barrier)
        selection = dict(complete=True, selected_mode='C', cohort_sha256=self.prepared['cohort_sha256'],
            suite_prepared_sha256=self.prepared_sha, rule=self.read('frozen_plan.json')['selection'],
            calibration_barrier_sha256=self.sha('ALL_CALIBRATION_PREDICTIONS_SEALED.json'), candidates={},
            calibration_score_sha256={})
        for m in ('T', 'C', 'min'):
            margin = {'pc60': {'T': -1., 'C': 1., 'min': 0.}[m], 'coco_object': {'T': 3., 'C': 2., 'min': 0.}[m]}
            selection['candidates'][m] = dict(margin_pp=margin, worst_dataset_margin_pp=min(margin.values()),
                macro_margin_pp=sum(margin.values()) / 2,
                diagnostic_encoder_inputs=sum(self.results[e['key']]['costs']['image_' + m]['diagnostic_encoder_inputs']
                    for e in self.prepared['cells'] if e['stage'] == 'calibration'))
        # Calibration scoring is sealed before the recorded selection exists.
        for e in self.prepared['cells'][:2]:
            self.seal_score(e, None)
            selection['calibration_score_sha256'][e['output']] = self.sha(e['output'] + '/SCORED_SEALED.json')
        self.write('SELECTION_SEALED.json', selection)
        self.selected = dict(path=str(self.root / 'SELECTION_SEALED.json'), sha256=self.sha('SELECTION_SEALED.json'), selected_mode='C')
        # Bind confirmation protocols to selection, then refresh its stage barrier.
        for e in self.prepared['cells'][2:]:
            p = e['output'] + '/prediction_protocol.json'
            protocol = self.read(p)
            protocol['selection'] = self.selected
            self.write(p, protocol)
            self.refresh_prediction(e)
        self.barriers['confirmation']['prediction_seal_sha256'] = {
            e['output']: self.sha(e['output'] + '/PREDICTIONS_SEALED.json') for e in self.prepared['cells'][2:]}
        self.write('ALL_CONFIRMATION_PREDICTIONS_SEALED.json', self.barriers['confirmation'])
        for e in self.prepared['cells'][2:]:
            self.seal_score(e, self.selected)
        for e in self.prepared['cells'][:2]:
            d = e['dataset']
            gate_counts = {}
            for arm in ('image_T', 'image_C', 'image_uniform', 'image_confidence'):
                v = 2 if arm in ('image_T', 'image_C') else 0
                gate_counts[arm] = dict(diagnostic_forward_inputs=v, diagnostic_encoder_inputs=v,
                    diagnostic_forward_calls=int(v > 0), diagnostic_encoder_calls=int(v > 0),
                    eval_forward_inputs=2, eval_forward_calls=2, adapt_forward_inputs=2, adapt_forward_calls=2,
                    encoder_inputs=4 + v, encoder_calls=4 + int(v > 0))
            self.write(f'gates/{d}.json.STARTED.json', dict(status='started', manifest_sha256=e['manifest_sha256']))
            self.write(f'gates/{d}.json', dict(passed=True, gt_contents_opened=False, images=4, dataset=d,
                manifest_sha256=e['manifest_sha256'], attempt_sha256=self.sha(f'gates/{d}.json.STARTED.json'),
                deterministic_runtime=self.runtime, gpu_lock=self.lock,
                comparisons=[dict(candidate_masks_equal=True, max_support_abs_difference=0.,
                    max_loss_abs_difference=0., changed_label_pixels=0, candidates=2,
                    actual_single_zero_view_counts=gate_counts)] * 4,
                checkpoint=self.manifests[e['key']]['checkpoint'], source_sha256=self.manifests[e['key']]['source_sha256']))
        commands = [('gate', e) for e in self.prepared['cells'][:2]]
        commands += [(action, e) for stage in ('calibration', 'confirmation') for action in ('predict', 'score')
                     for e in self.prepared['cells'] if e['stage'] == stage]
        receipts = []
        for index, (action, e) in enumerate(commands):
            argv = ['/synthetic/python', '-B', '-m', 'tools.evolution_confirmation.run', action]
            if action != 'score':
                argv += ['--manifest', str(self.root / e['manifest']), '--manifest-sha256', e['manifest_sha256']]
            argv += ['--output', str(self.root / (f'gates/{e["dataset"]}.json' if action == 'gate' else e['output']))]
            self.write(f'logs/{index}.log', '')
            receipt = dict(argv=argv, log=f'logs/{index}.log', log_sha256=self.sha(f'logs/{index}.log'),
                exit_code=0, error=None, cpu_only=action == 'score')
            self.write(f'logs/{index}.receipt.json', receipt)
            receipts.append(dict(**receipt, receipt=f'logs/{index}.receipt.json', receipt_sha256=self.sha(f'logs/{index}.receipt.json')))
        self.write('execution_children.json', receipts)
        self.write('queue_state.json', dict(phase='completed', active_cell=None, error=None, passed_gates=2,
            sealed_predictions=8, scored_cells=8, calibration_cells=2, confirmation_cells=6,
            calibration_predictions=2, confirmation_predictions=6, calibration_scores=2, confirmation_scores=6, total_cells=8))
        self.csv_rows()
        self.seal_suite()
        self.process = mock.patch.object(export.subprocess, 'run', side_effect=self.systemctl)
        self.process.start()
        self.addCleanup(self.process.stop)

    def make_cell(self, e):
        n, cfg = e['images'], self.read(e['config'])
        rows = self.bundle['datasets'][e['dataset']][e['stage']]
        if e['order_seed']:
            rows = list(reversed(rows))
        manifest = dict(format_version=1, worker='image_budget_confirmation', dataset=e['dataset'], stage=e['stage'],
            max_images=n, seed=0, arms=list(export.ARMS), arm_order='alternating', order_seed=e['order_seed'],
            suite_root=str(self.root), cohorts_path=str(self.root / 'frozen_cohorts.json'), cohort_sha256=self.prepared['cohort_sha256'],
            config=cfg, config_path=str(self.root / e['config']), config_sha256=self.sha(e['config']),
            checkpoint={'sha256': '3' * 64}, source_sha256={str(self.root / 'runtime/minimal.py'): self.sha('runtime/minimal.py')},
            timing_scope=export.TIMING_SCOPE, exposure_scope=export.SCOPE,
            panel=dict(full_split=False, inputs=rows, ordered_ids=[r['official_id'] for r in rows], classes=['a', 'absent']))
        self.write(e['manifest'], manifest)
        self.manifests[e['key']] = manifest
        e['manifest_sha256'] = self.sha(e['manifest'])
        prefix = e['output'] + '/'
        self.write(prefix + 'prediction_protocol.json', dict(manifest=manifest, manifest_path=str(self.root / e['manifest']),
            manifest_sha256=e['manifest_sha256'], selection=None, deterministic_runtime=self.runtime, gpu_lock=self.lock))
        self.write(prefix + 'initial_state.json', {'model_sha256': '4' * 64})
        costs, episodes = [], []
        totals = {a: {k: 0 for k in export.COUNTERS} for a in export.ARMS}
        timings = {}
        values = dict(zip(export.ARMS, (55, 49, 50, 49 if e['dataset'] == 'pc60' else 53,
                                       51 if e['dataset'] == 'pc60' else 52, 50)))
        for ep in range(n // 64):
            order = tuple(reversed(export.ARMS)) if ep % 2 else export.ARMS
            for index, a in enumerate(export.ARMS):
                factor = 0 if a in ('image_uniform', 'image_confidence') else 2 if a.endswith('min') else 1
                v = factor * 64
                row = dict(episode=ep, arm=a, images=64, patches=64, warmup=ep == 0, steps=2,
                    order_position=order.index(a), candidate_count=64, candidate_sha256='2' * 64, seconds=float(index + 1),
                    diagnostic_forward_inputs=v, diagnostic_encoder_inputs=v, diagnostic_forward_calls=v // 16,
                    diagnostic_encoder_calls=v // 16, eval_forward_inputs=128, eval_forward_calls=2,
                    adapt_forward_inputs=128, adapt_forward_calls=2, encoder_inputs=256 + v, encoder_calls=4 + v // 16)
                costs.append(row)
                for k in export.COUNTERS:
                    totals[a][k] += row[k]
            episodes.append(dict(episode=ep, arms={a: dict(intersection=[values[a] * 64, 0], union=[6400, 0]) for a in export.ARMS}))
        for index, a in enumerate(export.ARMS):
            seconds = float((n // 64 - 1) * (index + 1))
            timings[a] = dict(measured_images=n - 64, measured_episodes=n // 64 - 1, seconds=seconds,
                seconds_per_image=seconds / (n - 64), images_per_second=(n - 64) / seconds,
                warmup_episode=0, accuracy_includes_warmup=True)
        for name, rows in (('cost.jsonl', costs), ('scores_by_episode.jsonl', episodes)):
            (self.root / (prefix + name)).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        predicted = dict(status='predicted', images=n, full_split=False, timings=timings, costs=totals, timing_scope=export.TIMING_SCOPE)
        self.write(prefix + 'prediction_result.json', predicted)
        result = dict(predicted, status='completed', dataset=e['dataset'], stage=e['stage'], order_seed=e['order_seed'],
            cohort_sha256=self.prepared['cohort_sha256'], exposure_scope=export.SCOPE,
            label_contents_first_opened_after_all_stage_predictions_sealed=True,
            arm_results={a: dict(intersection=[values[a] * n, 0], union=[100 * n, 0], miou_unrounded=float(values[a])) for a in export.ARMS})
        self.results[e['key']] = result
        self.refresh_prediction(e)

    def refresh_prediction(self, e):
        prefix, n = e['output'] + '/', e['images']
        small = {'prediction_protocol.json', 'initial_state.json', 'prediction_result.json', 'cost.jsonl'}
        names = ({f'predictions/{a}/{i:06d}.npy' for a in export.ARMS for i in range(n)}
                 | {f'episodes/{ep:06d}.{a}.json' for a in export.ARMS for ep in range(n // 64)} | small)
        self.write(prefix + 'PREDICTIONS_SEALED.json', dict(complete=True, gt_contents_opened=False, images=n,
            arms=list(export.ARMS), artifact_sha256={name: self.sha(prefix + name) if name in small else '0' * 64 for name in names}))

    def seal_score(self, e, selection):
        prefix = e['output'] + '/'
        result = self.results[e['key']]
        result.update(prediction_seal_sha256=self.sha(prefix + 'PREDICTIONS_SEALED.json'), selection=selection)
        self.write(prefix + 'result.json', result)
        self.write(prefix + 'SCORING_STARTED.json', dict(prediction_seal_sha256=result['prediction_seal_sha256'],
            selection=selection, stage_barrier=self.barriers[e['stage']]))
        self.write(prefix + 'score_inputs.json', dict(label_sha256={r['label']: '5' * 64 for r in self.manifests[e['key']]['panel']['inputs']}))
        self.refresh_score(e)

    def refresh_score(self, e):
        prefix = e['output'] + '/'
        self.write(prefix + 'SCORED_SEALED.json', dict(prediction_seal_sha256=self.sha(prefix + 'PREDICTIONS_SEALED.json'),
            artifact_sha256={n: self.sha(prefix + n) for n in ('SCORING_STARTED.json', 'scores_by_episode.jsonl', 'score_inputs.json', 'result.json')}))

    def csv_rows(self):
        rows = []
        for e in self.prepared['cells']:
            r = self.results[e['key']]
            for a in export.ARMS:
                value = r['arm_results'][a]['miou_unrounded']
                rows.append(dict(dataset=e['dataset'], stage=e['stage'], order_seed=e['order_seed'], arm=a,
                    images=e['images'], full_split=False, miou_unrounded=value, margin_vs_image_confidence_pp=value - 50,
                    diagnostic_encoder_inputs=r['costs'][a]['diagnostic_encoder_inputs'], selected_global_mode='C',
                    is_selected_probe=a == 'image_C', manifest_sha256=e['manifest_sha256'], config_sha256=self.sha(e['config']),
                    selection_sha256=self.sha('SELECTION_SEALED.json'), prediction_seal_sha256=self.sha(e['output'] + '/PREDICTIONS_SEALED.json'),
                    score_seal_sha256=self.sha(e['output'] + '/SCORED_SEALED.json'), exposure_scope=export.EXPOSURE,
                    **{k: r['timings'][a][k] for k in ('measured_images', 'seconds', 'seconds_per_image', 'images_per_second')}))
        self.write_csv(rows)

    def write_csv(self, rows):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (self.root / 'rows.csv').write_text(stream.getvalue())

    def seal_suite(self):
        names = list(self.prepared['artifact_sha256'])
        names += ['SUITE_PREPARED.json', 'EXECUTION_STARTED.json', 'SELECTION_SEALED.json', 'execution_children.json',
                  'queue_state.json', 'rows.csv', 'ALL_CALIBRATION_PREDICTIONS_SEALED.json', 'ALL_CONFIRMATION_PREDICTIONS_SEALED.json']
        names += [f'gates/{d}.json' for d in export.DATASETS]
        names += [e['output'] + '/' + n for e in self.prepared['cells'] for n in ('PREDICTIONS_SEALED.json', 'SCORED_SEALED.json', 'result.json')]
        self.write('SUITE_SEALED.json', dict(complete=True, calibration_cells=2, confirmation_cells=6, formal_cells=8,
            arm_results=48, passed_gt_free_gates=2, full_split=False, selected_mode='C', exposure_note=export.EXPOSURE,
            suite_prepared_sha256=self.prepared_sha, cohort_sha256=self.prepared['cohort_sha256'],
            runtime_hashes=self.prepared['runtime_hashes'], selection_sha256=self.sha('SELECTION_SEALED.json'),
            artifact_sha256={n: self.sha(n) for n in names}))

    def systemctl(self, *args, **kwargs):
        state = dict(Id=export.SERVICE, LoadState='loaded', ActiveState='active', SubState='exited', Result='success',
            MainPID='0', ExecMainPID='1234', ExecMainCode='1', ExecMainStatus='0',
            ExecStart=f'{{ argv[]=/synthetic/python -B -m tools.evolution_confirmation.queue run --suite {self.root} --suite-sha256 {self.prepared_sha} ; }}')
        state.update(getattr(self, 'service_changes', {}))
        return subprocess.CompletedProcess(args[0], 0, stdout='\n'.join(k + '=' + v for k, v in state.items()))

    def run_import(self, **kwargs):
        return export.run(self.root, self.paper, self.prepared_sha, **kwargs)

    def unchanged(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.generated.iterdir()}, self.before)

    def rejects(self, message):
        with self.assertRaisesRegex(export.Invalid, message):
            self.run_import()
        self.unchanged()

    def test_complete_and_no_payload_reads(self):
        # There are deliberately no prediction/episode files in this fixture.
        status, code = self.run_import()
        self.assertEqual((status['status'], code), ('imported', 0))
        provenance = json.loads((self.generated / export.OUTPUTS[2]).read_text())
        self.assertEqual(len(provenance['cells']), 8)
        self.assertFalse(provenance['july_history_certified'])
        self.assertIn('NOT reread', provenance['verification_scope'])
        self.assertLess(provenance['additional_contrasts']['confirmation/pc60/sorted']['C_minus_old_min_pp'], 0)
        self.assertGreater(provenance['additional_contrasts']['confirmation/coco_object/sorted']['T_minus_C_pp'], 0)
        self.assertEqual((self.generated / 'unrelated.tex').read_bytes(), self.before['unrelated.tex'])
        for name in export.OUTPUTS[:2]:
            text = (self.generated / name).read_text()
            for token in ('Calibration', 'Confirmation', '20261030', '20261031', 'Image T', 'Image C', 'Old min'):
                self.assertIn(token, text)

    def test_check_only(self):
        self.assertEqual(self.run_import(check_only=True)[1], 0)
        self.unchanged()

    def test_cpu_guard(self):
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            self.rejects('CPU-only')

    def test_prepared_hash(self):
        (self.root / 'SUITE_PREPARED.json').write_text('{}')
        self.rejects('hash mismatch')

    def test_runtime_drift(self):
        self.write('runtime/minimal.py', 'changed')
        self.rejects('hash mismatch')

    def test_runtime_extra_file(self):
        self.write('runtime/unlisted.py', 'unlisted')
        self.rejects('Runtime inventory differs')

    def test_terminal_failures(self):
        for changes, pattern in (({'ExecMainPID': '999'}, 'PID differs'), ({'LoadState': 'not-found'}, 'identity'),
                ({'ExecStart': f'-m tools.evolution_confirmation.queue run --suite /wrong --suite-sha256 {self.prepared_sha} ;'}, 'root/command'),
                ({'ExecStart': f'-m tools.evolution_confirmation.queue run --suite {self.root} --suite-sha256 wrong ;'}, 'root/command'),
                ({'Result': 'exit-code'}, 'successfully'),
                ({'ExecMainStatus': '1'}, 'successfully')):
            with self.subTest(changes=changes):
                self.service_changes = changes
                self.rejects(pattern)

    def test_live_service_pending(self):
        self.service_changes = dict(MainPID='1234', SubState='running')
        with self.assertRaises(export.Pending):
            self.run_import()
        self.unchanged()

    def test_missing_suite_seal(self):
        (self.root / 'SUITE_SEALED.json').unlink()
        with self.assertRaises(export.Pending):
            self.run_import()
        self.unchanged()

    def test_missing_whole_seal_coverage(self):
        seal = self.read('SUITE_SEALED.json')
        del seal['artifact_sha256']['rows.csv']
        self.write('SUITE_SEALED.json', seal)
        self.rejects('coverage missing')

    def test_csv_failures(self):
        original = list(csv.DictReader(io.StringIO((self.root / 'rows.csv').read_text())))
        for kind in ('missing', 'duplicate', 'sign', 'hash', 'selection', 'timing', 'nan'):
            rows = copy.deepcopy(original)
            if kind == 'missing':
                rows.pop()
            elif kind == 'duplicate':
                rows[-1] = rows[0].copy()
            else:
                key, value = {'sign': ('margin_vs_image_confidence_pp', '-999'), 'hash': ('manifest_sha256', '0' * 64),
                    'selection': ('selected_global_mode', 'T'), 'timing': ('measured_images', '256'),
                    'nan': ('miou_unrounded', 'nan')}[kind]
                rows[0][key] = value
            self.write_csv(rows)
            self.seal_suite()
            with self.subTest(kind=kind):
                self.rejects('CSV')

    def test_stage_barrier_link(self):
        barrier = self.read('ALL_CONFIRMATION_PREDICTIONS_SEALED.json')
        barrier['prediction_seal_sha256'].pop(next(iter(barrier['prediction_seal_sha256'])))
        self.write('ALL_CONFIRMATION_PREDICTIONS_SEALED.json', barrier)
        self.seal_suite()
        self.rejects('barrier population')

    def test_selection_link(self):
        selection = self.read('SELECTION_SEALED.json')
        selection['calibration_score_sha256'] = {}
        self.write('SELECTION_SEALED.json', selection)
        self.seal_suite()
        self.rejects('calibration-only link')

    def test_scoring_barrier(self):
        e = self.prepared['cells'][2]
        name = e['output'] + '/SCORING_STARTED.json'
        started = self.read(name)
        started['stage_barrier'] = self.barriers['calibration']
        self.write(name, started)
        self.refresh_score(e)
        self.seal_suite()
        self.rejects('Scoring stage barrier')

    def test_pooled_counts(self):
        e = self.prepared['cells'][2]
        name = e['output'] + '/result.json'
        result = self.read(name)
        result['arm_results']['image_C']['intersection'][0] += 1
        self.write(name, result)
        self.refresh_score(e)
        self.seal_suite()
        self.rejects('Pooled episode counts')

    def test_gate_actual_counter(self):
        name = 'gates/pc60.json'
        gate = self.read(name)
        gate['comparisons'][0]['actual_single_zero_view_counts']['image_T']['diagnostic_encoder_inputs'] += 1
        self.write(name, gate)
        self.seal_suite()
        self.rejects('Gate actual probe count')

    def test_recorded_selection_statistics(self):
        ev = export.Evidence(self.root)
        *_, cells = export.verify_suite(ev, self.prepared_sha, export.SERVICE)
        record = self.read('SELECTION_SEALED.json')
        record['candidates']['C']['margin_pp']['pc60'] *= -1
        with self.assertRaisesRegex(export.Invalid, 'Calibration margin'):
            export.verify_selection(record, cells)

    def test_deterministic_runtime(self):
        runtime = copy.deepcopy(self.runtime)
        runtime['deterministic_warn_only'] = True
        with self.assertRaisesRegex(export.Invalid, 'Runtime determinism'):
            export.runtime(runtime)

    def test_rounded_miou(self):
        self.assertAlmostEqual(export.pooled(dict(intersection=[1, 0], union=[3, 0], miou_unrounded=100/3), 2), 100/3)
        with self.assertRaisesRegex(export.Invalid, 'Pooled mIoU'):
            export.pooled(dict(intersection=[1], union=[3], miou_unrounded=33.33), 1)

    def test_prediction_inventory(self):
        e = self.prepared['cells'][2]
        name = e['output'] + '/PREDICTIONS_SEALED.json'
        seal = self.read(name)
        del seal['artifact_sha256']['predictions/image_C/000000.npy']
        self.write(name, seal)
        self.seal_suite()
        # The stage barrier must detect the changed inventory's seal first.
        self.rejects('barrier population/link')

    def test_scored_metadata_hash(self):
        e = self.prepared['cells'][2]
        self.write(e['output'] + '/score_inputs.json', dict(label_sha256={}))
        self.rejects('Seal/hash mismatch')

    def test_second_service_observation(self):
        original = self.systemctl
        calls = 0
        def service(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.service_changes = {'ExecMainPID': '999'}
            return original(*args, **kwargs)
        with mock.patch.object(export.subprocess, 'run', side_effect=service):
            self.rejects('PID differs')

    def test_missing_score_episode(self):
        e = self.prepared['cells'][2]
        name = e['output'] + '/scores_by_episode.jsonl'
        path = self.root / name
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        self.refresh_score(e)
        self.seal_suite()
        self.rejects('Missing scored episodes')

    def test_cost_stream_failures(self):
        e = self.prepared['cells'][2]
        manifest = self.manifests[e['key']]
        data = (self.root / e['output'] / 'cost.jsonl').read_bytes()
        original = [json.loads(line) for line in data.splitlines()]
        for kind in ('missing', 'duplicate', 'warmup', 'probe', 'calls', 'candidate', 'nonfinite'):
            rows = copy.deepcopy(original)
            if kind == 'missing':
                rows.pop()
            elif kind == 'duplicate':
                rows[-1] = rows[0].copy()
            elif kind == 'warmup':
                rows[0]['warmup'] = False
            elif kind == 'probe':
                rows[0]['diagnostic_forward_inputs'] += 1
            elif kind == 'calls':
                rows[0]['encoder_calls'] += 1
            elif kind == 'candidate':
                rows[1]['candidate_sha256'] = '6' * 64
            else:
                rows[0]['seconds'] = float('nan')
            changed = '\n'.join(json.dumps(r) for r in rows).encode()
            with self.subTest(kind=kind), self.assertRaises(export.Invalid):
                export.cost_summary(changed, manifest, self.results[e['key']])

    def test_timing_denominator(self):
        e = self.prepared['cells'][2]
        result = copy.deepcopy(self.results[e['key']])
        result['timings']['image_C']['measured_images'] = 256
        with self.assertRaisesRegex(export.Invalid, 'Timing denominator'):
            export.cost_summary((self.root / e['output'] / 'cost.jsonl').read_bytes(), self.manifests[e['key']], result)

    def test_nonfinite_duplicate_json_and_bounds(self):
        for data in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(export.Invalid):
                export.decode(data)
        ev = export.Evidence(self.root)
        with self.assertRaisesRegex(export.Invalid, 'Payload scan forbidden'):
            ev.inventory({'cells/a/predictions/b.npy': '0' * 64})
        with self.assertRaisesRegex(export.Invalid, 'Path escape'):
            ev.raw('../escape')
        with mock.patch.object(export, 'LIMIT', 1), self.assertRaisesRegex(export.Invalid, 'exceeds bound'):
            ev.raw('queue_state.json')

    def test_drift_prevents_publish(self):
        original = export.Evidence.stable
        def mutate(ev):
            self.write('queue_state.json', {'phase': 'changed'})
            original(ev)
        with mock.patch.object(export.Evidence, 'stable', mutate):
            self.rejects('Source drift')

    def test_publish_rollback(self):
        replace = os.replace
        calls = 0
        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('injected publication failure')
            return replace(source, destination)
        with mock.patch.object(export.os, 'replace', side_effect=fail_second), self.assertRaises(OSError):
            self.run_import()
        self.unchanged()


if __name__ == '__main__':
    unittest.main()
