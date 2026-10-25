import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).parents[1] / 'scripts/export_results.py'
spec = importlib.util.spec_from_file_location('export_results', MODULE)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_roundtrip_dedup_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            suite = source / 'rebuttal_original_test'
            suite.mkdir(parents=True)
            payload = json.dumps({'status': 'completed', 'metrics': {'mIoU': 51.2}})
            (suite / 'result.json').write_text(payload)
            (suite / 'duplicate.json').write_text(payload)
            (suite / 'sample.npz').write_bytes(b'prediction-output')
            (suite / 'model.pt').write_bytes(b'not-for-upload')
            (suite / 'secret.txt').write_text('password=123456789')
            (suite / 'link.json').symlink_to(suite / 'result.json')
            runtime = suite / 'runtime'
            runtime.mkdir()
            (runtime / 'file.json').write_text('{}')
            dest = root / 'export'
            snap = exporter.export(source, dest, part_mib=1)
            self.assertEqual(snap['included_files'], 3)
            self.assertEqual(snap['unique_objects'], 2)
            self.assertEqual(snap['excluded_counts']['sensitive_content_not_uploaded'], 1)
            exporter.verify(dest, full=True)
            out = root / 'restored'
            exporter.restore(dest, out)
            self.assertEqual((out / suite.name / 'result.json').read_text(), payload)
            self.assertEqual((out / suite.name / 'duplicate.json').read_text(), payload)
            self.assertFalse((out / suite.name / 'model.pt').exists())
            self.assertFalse((out / suite.name / 'secret.txt').exists())
            with self.assertRaises(FileExistsError):
                exporter.restore(dest, out)
            with self.assertRaises(ValueError):
                exporter.export(source, dest)

    def test_archive_checksum_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source/suite'
            source.mkdir(parents=True)
            (source / 'result.json').write_text('{}')
            dest = root / 'export'
            snap = exporter.export(source.parent, dest)
            archive = dest / 'assets' / snap['archives'][0]['name']
            with archive.open('ab') as f:
                f.write(b'tampering')
            with self.assertRaises(ValueError):
                exporter.verify(dest)

    def test_export_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                exporter.export(root, root / 'export')

    def test_partitioned_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root / 'source/suite'
            suite.mkdir(parents=True)
            (suite / 'one.npy').write_bytes(b'a' * 700000)
            (suite / 'two.npy').write_bytes(b'b' * 700000)
            dest = root / 'export'
            snap = exporter.export(suite.parent, dest, part_mib=1)
            self.assertEqual(len(snap['archives']), 2)
            exporter.verify(dest, full=True)
            exporter.restore(dest, root / 'restored', suite='suite')
            self.assertEqual((root / 'restored/suite/two.npy').stat().st_size, 700000)

    def test_restore_rechecks_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root / 'source/suite'
            suite.mkdir(parents=True)
            (suite / 'result.json').write_text('{}')
            dest = root / 'export'
            exporter.export(suite.parent, dest)
            exporter.verify(dest, full=True)
            with (dest / 'index/files.csv.gz').open('ab') as f:
                f.write(b'tamper')
            with self.assertRaises(ValueError):
                exporter.restore(dest, root / 'restored')

    def test_non_numeric_credentials_and_encrypted_keys(self):
        self.assertTrue(exporter.has_secret(b'password=example-review-password'))
        self.assertTrue(exporter.has_secret(b'-----BEGIN ENCRYPTED PRIVATE KEY-----'))
        self.assertTrue(exporter.has_secret(b'api_key="fixture-value-not-a-real-key"'))
        self.assertFalse(exporter.has_secret(b'password=None'))

    def test_root_level_evidence_is_included(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            (source / 'summary.csv').write_text('dataset,mIoU\nvoc21,52.16\n')
            snap = exporter.export(source, root / 'export')
            self.assertEqual(snap['included_files'], 1)
            self.assertTrue((root / 'export/results/summary.csv').exists())

    def test_nested_job_outcomes_and_generic_score(self):
        data = json.dumps({'status': 'completed', 'job': {'dataset': 'coco_cd',
            'seed': 2, 'variant': 'full', 'smoke': False}, 'outcomes': [{
                'images': 500, 'full_split': True, 'metrics': {'mIoU': 75.16}}]}).encode()
        row = exporter.result_rows(data, Path('suite/result.json'), 'suite', 'abc')[0]
        self.assertEqual((row['dataset'], row['seed'], row['mIoU']), ('coco_cd', 2, 75.16))
        self.assertTrue(row['full_split'])
        generic = exporter.result_rows(b'{"score":0.7}', Path('suite/result.json'), 'suite', 'abc')[0]
        self.assertEqual(generic['mIoU'], '')
        self.assertEqual(generic['reported_score'], 0.7)

    def test_symlink_restore_parent_has_no_external_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source/suite/new/sub'
            source.mkdir(parents=True)
            (source / 'result.json').write_text('{}')
            dest = root / 'export'
            exporter.export(root / 'source', dest)
            exporter.verify(dest, full=True)
            output, external = root / 'out', root / 'external'
            output.mkdir()
            external.mkdir()
            (output / 'suite').symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                exporter.restore(dest, output)
            self.assertFalse((external / 'new').exists())

    def test_symlink_above_output_has_no_external_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source/suite'
            source.mkdir(parents=True)
            (source / 'result.json').write_text('{}')
            dest = root / 'export'
            exporter.export(root / 'source', dest)
            exporter.verify(dest, full=True)
            external = root / 'external'
            external.mkdir()
            (root / 'link').symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                exporter.restore(dest, root / 'link/new-output')
            self.assertFalse((external / 'new-output').exists())


if __name__ == '__main__':
    unittest.main()
