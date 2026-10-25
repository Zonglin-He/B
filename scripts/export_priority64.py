#!/usr/bin/env python3
"""Export only completed priority studies, without modifying live experiments."""

import hashlib
import json
from pathlib import Path
import shutil

import export_results as ex
import finalize_snapshot

PROJECT = Path('/home/wwww/hzl')
REPO = Path(__file__).resolve().parents[1]
PRIORITY = PROJECT / 'results/rebuttal_priority64_repaired_20261025_1940'
ROLE = PROJECT / 'results/rebuttal_role_policy64_repaired_20261025_1938'
STAGE = REPO / '.export-work/priority64-20261025'
DEST = REPO / 'updates/rebuttal64-20261025'


def read(path):
    return json.loads(path.read_text())


def copy_stable(source, destination):
    if source.is_symlink():
        raise ValueError('Symlink not exported: ' + str(source))
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed during capture: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    if ex.digest(source) != hashlib.sha256(payload).hexdigest():
        raise ValueError('Input changed after capture: ' + str(source))
    shutil.copystat(source, destination)


def copy_tree(source, destination):
    for path in sorted(source.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix in ex.TEXT | ex.BINARY:
            copy_stable(path, destination / path.relative_to(source))


def main():
    if STAGE.exists() or (DEST / 'snapshot.json').exists():
        raise FileExistsError('Use a new immutable update destination')
    if not (PRIORITY / 'PRIORITY_RESULTS_READY.json').is_file():
        raise ValueError('Priority GPU studies have not finished')
    STAGE.mkdir(parents=True)
    captured = []
    for root in (PRIORITY, ROLE):
        target = STAGE / root.name
        target.mkdir()
        for name in ('manifest.json', 'manifest_seal.json', 'protocol.json',
                     'PRIORITY_RESULTS_READY.json', 'README.md', 'RESULTS.md',
                     'SUITE_SEALED.json', 'schedule_override.json'):
            if (root / name).is_file():
                copy_stable(root / name, target / name)
        copy_stable(root / 'queue_state.json', target / 'queue_state_at_capture.json')
        if root == PRIORITY:
            cells = [root / 'cells' / job['id'] for job in read(root / 'manifest.json')['local_jobs']]
        else:
            cells = [root / 'cells' / job['id'] for job in read(root / 'manifest.json')['jobs']]
        for cell in cells:
            result = read(cell / 'result.json')
            if result.get('status') != 'completed':
                raise ValueError('Incomplete priority cell: ' + str(cell))
            for outcome in result['outcomes']:
                if ex.digest(Path(outcome['prediction_seal'])) != outcome['prediction_seal_sha256']:
                    raise ValueError('Prediction seal mismatch')
            copy_tree(cell, target / cell.relative_to(root))
            captured.append(dict(path=str(cell), result_sha256=ex.digest(cell / 'result.json'),
                                 images=result['job']['max_images'], smoke=result['job']['smoke']))
        for name in ('tables', 'historical', 'offline_pixel_geometry'):
            if (root / name).is_dir():
                copy_tree(root / name, target / name)
    record = dict(captured_at_utc=ex.utcnow(), original_source=str(PROJECT / 'results'),
                  method='Archived submitted TRACE; new diagnostic controls are labeled separately',
                  privacy='Public upload explicitly authorized by user on 2026-10-25',
                  cells=captured, active_original_continuation_included=False,
                  omitted=['source/runtime code', 'model weights', 'datasets', 'credentials',
                           'in-progress original continuation outputs'],
                  scope='Six completed 64-image cells, four smoke cells, completed offline audits')
    (STAGE / 'capture.json').write_text(json.dumps(record, indent=2) + '\n')
    ex.export(STAGE, DEST)
    finalize_snapshot.finalize(DEST, STAGE)
    ex.verify(DEST, full=True)


if __name__ == '__main__':
    main()
