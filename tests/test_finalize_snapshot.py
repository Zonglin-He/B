import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('finalize_snapshot', SCRIPTS / 'finalize_snapshot.py')
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


class FinalizerTests(unittest.TestCase):
    def test_archive_derived_index_root_capture_and_source_patch_removal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root / 'source/suite'
            suite.mkdir(parents=True)
            data = {'status': 'completed', 'job': {'dataset': 'pc60', 'variant': 'full', 'seed': 1},
                    'outcomes': [{'images': 5105, 'full_split': True, 'metrics': {'mIoU': 30.37}}]}
            (suite / 'result.json').write_text(json.dumps(data))
            (suite / 'source.diff').write_text('implementation source that must not be uploaded')
            ex = finalizer.ex
            dest = root / 'export'
            with mock.patch.object(ex, 'TEXT', ex.TEXT | {'.diff'}):
                ex.export(suite.parent, dest)
            # The live result changes after capture. The index must use archived bytes.
            data['outcomes'][0]['metrics']['mIoU'] = 999
            (suite / 'result.json').write_text(json.dumps(data))
            (suite.parent / 'late-summary.csv').write_text('dataset,mIoU\npc60,30.37\n')
            finalizer.finalize(dest, suite.parent)
            ex.verify(dest, full=True)
            with (dest / 'index/runs.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['dataset'], 'pc60')
            self.assertEqual(float(rows[0]['mIoU']), 30.37)
            self.assertTrue((dest / 'results/late-summary.csv').exists())
            snapshot = json.loads((dest / 'snapshot.json').read_text())
            self.assertEqual(snapshot['excluded_counts']['source_patch_not_uploaded'], 1)
            ex.restore(dest, root / 'restored')
            self.assertFalse((root / 'restored/suite/source.diff').exists())
            self.assertEqual(json.loads((root / 'restored/suite/result.json').read_text())['outcomes'][0]['metrics']['mIoU'], 30.37)


if __name__ == '__main__':
    unittest.main()
