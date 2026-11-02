"""Focused CPU-only regression tests; no experimental runtime or payload reads."""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from . import export_tuned_full as e
except ImportError:
    import export_tuned_full as e


class TunedFullTests(unittest.TestCase):
    def score_fixture(self):
        dataset, cfg = 'voc20', {'batch_size': 32}
        n, classes = e.POPULATIONS[dataset], e.CLASSES[dataset]
        rows = []
        total = 0
        for episode, start in enumerate(range(0, n, cfg['batch_size'])):
            images = min(cfg['batch_size'], n - start)
            total += images
            rows.append(dict(episode=episode, images=images,
                             arms={a: dict(intersection=[images] * classes, union=[2 * images] * classes)
                                   for a in e.ARMS}))
        result = dict(arm_results={a: dict(intersection=[total] * classes, union=[2 * total] * classes,
                                          miou_unrounded=50.) for a in e.ARMS})
        return rows, dataset, cfg, result

    def test_score_stream_includes_partial_tail_and_warmup(self):
        rows, d, cfg, result = self.score_fixture()
        self.assertEqual(rows[-1]['images'], 9)
        self.assertEqual(e.verify_score_stream(iter(rows), d, cfg, result), dict.fromkeys(e.ARMS, 50.))

    def test_wrong_population_count(self):
        rows, d, cfg, result = self.score_fixture()
        rows[-1]['images'] -= 1
        with self.assertRaisesRegex(e.Invalid, 'count/scope'):
            e.verify_score_stream(iter(rows), d, cfg, result)

    def test_missing_episode(self):
        rows, d, cfg, result = self.score_fixture()
        with self.assertRaisesRegex(e.Invalid, 'Incomplete'):
            e.verify_score_stream(iter(rows[:-1]), d, cfg, result)

    def test_wrong_pooled_counts_even_if_same_ratio(self):
        rows, d, cfg, result = self.score_fixture()
        for key in ('intersection', 'union'):
            result['arm_results']['image_C'][key] = [2 * x for x in result['arm_results']['image_C'][key]]
        with self.assertRaisesRegex(e.Invalid, 'Pooled counts'):
            e.verify_score_stream(iter(rows), d, cfg, result)

    def test_wrong_recorded_pooled_score(self):
        with self.assertRaisesRegex(e.Invalid, 'Pooled mIoU'):
            e.pooled(dict(intersection=[2, 0], union=[4, 0], miou_unrounded=25.), 2)

    def test_absent_classes_excluded_from_mean(self):
        self.assertEqual(e.pooled(dict(intersection=[2, 0], union=[4, 0], miou_unrounded=50.), 2), 50.)

    def test_invalid_count_types_and_ranges(self):
        for x, y in ((True, 1), (1.5, 2), (-1, 2), (3, 2)):
            with self.subTest(x=x), self.assertRaises(e.Invalid):
                e.counts(dict(intersection=[x], union=[y]), 1)

    def test_hash_mismatch(self):
        reader = e.Evidence(Path('/tmp'))
        target = Mock()
        target.stat.return_value = SimpleNamespace(st_size=2)
        target.open.return_value = io.BytesIO(b'{}')
        with patch.object(reader, 'path', return_value=target), self.assertRaisesRegex(e.Invalid, 'Hash differs'):
            reader.raw('result.json', '0' * 64)

    def test_oversized_metadata_rejected_before_open(self):
        reader = e.Evidence(Path('/tmp'))
        target = Mock()
        target.stat.return_value = SimpleNamespace(st_size=9 * e.MIB)
        with patch.object(reader, 'path', return_value=target), self.assertRaisesRegex(e.Invalid, 'byte bound'):
            reader.raw('result.json')
        target.open.assert_not_called()

    def test_payload_paths_never_opened(self):
        reader = e.Evidence(Path('/tmp'))
        for path in ('predictions/image_C/000000.npy', 'episodes/000000.image_C.json',
                     'per_image_counts.jsonl', '../result.json', 'checkpoint.pt'):
            with self.subTest(path=path), self.assertRaises(e.Invalid):
                reader.path(path)

    def test_stream_hash_and_row_limit(self):
        reader = e.Evidence(Path('/tmp'))
        data = b'{"episode":0}\n{"episode":1}\n'
        target = Mock()
        target.stat.return_value = SimpleNamespace(st_size=len(data))
        target.open.side_effect = lambda *args: io.BytesIO(data)
        with patch.object(reader, 'path', return_value=target):
            self.assertEqual(len(list(reader.lines('cost.jsonl', hashlib.sha256(data).hexdigest(), 2))), 2)
            with self.assertRaisesRegex(e.Invalid, 'row/byte bound'):
                list(reader.lines('cost.jsonl', hashlib.sha256(data).hexdigest(), 1))
            with self.assertRaisesRegex(e.Invalid, 'Hash differs'):
                list(reader.lines('scores_by_episode.jsonl', '0' * 64, 2))

    def overlap_fixture(self):
        flags = dict(protocol='development_overlap', evaluation_scope='development_overlap',
                     development_overlap=True, allow_waterbirds_development_overlap=True,
                     cal_full_source_id_overlap=256, train_full_source_id_overlap=2458,
                     independent_test_claim=False, global_never_seen_certified=False)

        def panel(n, stage):
            ids = ['source:' + str(i) for i in range(n)]
            return dict(flags, images=n, ordered_ids=ids, available=True, full_split=stage == 'full',
                        stage=stage, classes=['background', 'landbird', 'waterbird'], queries=['q'],
                        original_full_count=5322,
                        inputs=[dict(official_id=i, image='/rgb/' + i, label='/gt/' + i,
                                     image_sha256='a' * 64, label_metadata_details=dict(contents_read=False)) for i in ids])

        return dict(calibration=panel(256, 'calibration'), evaluation=panel(5322, 'full'),
                    selection=dict(flags, historical_development_exposure_retained=True,
                                   annotation_pixels_read=False, size=256, seed=2027),
                    input_audit=dict(flags, historical_selected_full_overlap=512, historical_selected_count=512),
                    config_provenance=dict(kind='historical'))

    def test_waterbirds_overlap_preserved(self):
        exposure = e.verify_exposure('waterbirds_seg', self.overlap_fixture())
        self.assertEqual(exposure['calibration_full_source_id_overlap'], 256)
        self.assertEqual(exposure['evaluation_scope'], 'development_overlap')
        self.assertFalse(exposure['independent_test_claim'])

    def test_wrong_overlap_flag_or_count(self):
        for key, value in (('cal_full_source_id_overlap', 0), ('independent_test_claim', True),
                           ('evaluation_scope', 'test')):
            entry = self.overlap_fixture()
            entry['evaluation'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(e.Invalid, 'overlap disclosure'):
                e.verify_exposure('waterbirds_seg', entry)

    def test_wrong_actual_overlap(self):
        entry = self.overlap_fixture()
        row = entry['calibration']['inputs'][0]
        row['official_id'] = entry['calibration']['ordered_ids'][0] = 'not-in-full'
        with self.assertRaisesRegex(e.Invalid, 'calibration/full overlap'):
            e.verify_exposure('waterbirds_seg', entry)

    def test_csv_requires_exact_32_rows(self):
        with self.assertRaisesRegex(e.Invalid, 'exactly 32'):
            e.verify_csv([{}] * 31, [], e.ROOT)

    def test_noadapt_cannot_have_update_cost(self):
        row = dict.fromkeys(e.COUNTERS, 0)
        row.update(arm='no_adapt', images=2, patches=2, steps=0, candidate_count=0,
                   active_candidates=0, seconds=.1, candidate_sha256='a' * 64,
                   eval_forward_calls=1, eval_forward_inputs=2, encoder_calls=1, encoder_inputs=2)
        e.validate_cost(row, 'no_adapt', dict(trace_max_regions=8))
        row['adapt_forward_calls'] = 1
        with self.assertRaisesRegex(e.Invalid, 'schedule'):
            e.validate_cost(row, 'no_adapt', dict(trace_max_regions=8))

    def test_service_requires_real_success_pid_command(self):
        prepared = dict(python='/python')
        execution = dict(pid=123, suite_prepared_sha256=e.PREPARED_SHA, retry_allowed=False)
        state = dict(Id=e.SERVICE, LoadState='loaded', ActiveState='active', SubState='exited',
                     Result='success', MainPID='0', ExecMainPID='123', ExecMainCode='1', ExecMainStatus='0',
                     ExecMainStartTimestamp='start', ExecMainExitTimestamp='exit',
                     ExecStart=f'argv[]=/python -B -m tools.evolution_tuning.continuation run --suite {e.ROOT} '
                               f'--suite-sha256 {e.PREPARED_SHA} ;')
        for changed, raises in (({}, None), ({'ExecMainStatus': '1'}, e.Invalid),
                                ({'ExecMainPID': '124'}, e.Invalid), ({'ExecStart': 'other suite'}, e.Invalid),
                                ({'MainPID': '123', 'SubState': 'running'}, e.Pending)):
            value = dict(state, **changed)
            response = SimpleNamespace(stdout='\n'.join(k + '=' + v for k, v in value.items()))
            with self.subTest(changed=changed), patch.object(e.subprocess, 'run', return_value=response):
                if raises is None:
                    self.assertEqual(e.service_success(e.ROOT, prepared, execution), value)
                else:
                    with self.assertRaises(raises):
                        e.service_success(e.ROOT, prepared, execution)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(e.Invalid, 'Duplicate JSON'):
            e.decode('{"images":1449,"images":1}')

    def test_invalid_import_never_writes(self):
        with patch.object(e, 'import_suite', side_effect=e.Invalid('wrong hash')), \
                patch.object(e, 'write_outputs') as writer, patch.object(e.sys, 'stderr', io.StringIO()):
            self.assertEqual(e.main(['--write']), 2)
        writer.assert_not_called()

    def test_unique_labels_and_readable_tables(self):
        cells = []
        for d in e.DATASETS:
            cells.append(dict(dataset=d, images=e.POPULATIONS[d], scores=dict.fromkeys(e.ARMS, 50.),
                              delta_confidence_pp=0., extra_time_percent=1.,
                              timings={a: dict(ms_per_image=10.) for a in e.ARMS},
                              config=dict(batch_size=2, lr=.0015, steps=9, trace_max_regions=8)))
        reported = dict(methods=['No Adapt', 'TENT'], datasets={d: {'No Adapt': [45., None], 'TENT': [46., .01]}
                       for d in ('voc20', 'voc21', 'pc60', 'coco_object', 'coco_stuff')})
        report = dict(cells=cells, references=dict(reported=reported, native_cd=dict(noadapt=61.14, tent=72.20, tpt=64.27, mlmp=71.01)))
        outputs = e.build_outputs(report)
        tex = ''.join(outputs[k] for k in e.OUTPUTS[:4])
        self.assertEqual(set(outputs), set(e.OUTPUTS))
        for suffix in ('comparison', 'reference', 'cost', 'configurations', 'cd-reference'):
            self.assertEqual(tex.count('\\label{tab:tuned-full-' + suffix + '}'), 1)
        for raw_arm in e.ARMS:
            self.assertNotIn(raw_arm, tex)
        self.assertNotIn('tab:configurations}', tex)
        self.assertIn('without separate tuning', tex)
        self.assertIn('256 calibration sources', tex)
        self.assertIn('\\begin{table*}', outputs['tuned_full_comparison.tex'])


if __name__ == '__main__':
    unittest.main()
