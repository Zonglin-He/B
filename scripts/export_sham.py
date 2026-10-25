#!/usr/bin/env python3
"""Capture completed sham controls without reading or changing live GPU jobs."""

import argparse
import csv
import json
from pathlib import Path

import export_results as ex
from export_priority64 import copy_stable, copy_tree
import finalize_snapshot


PROJECT = Path('/home/wwww/hzl')
REPO = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'results/rebuttal_sham_priority_20261025_2023'
STAGE = REPO / '.export-work/sham-20261025'
DEST = REPO / 'updates/sham-20261025'
POLICIES = {'native', 'sham_residual', 'sham_all_views'}


def read(path):
    return json.loads(path.read_text())


def checked_hash(path, expected, cell=None):
    if cell is not None and not path.resolve().is_relative_to(cell.resolve()):
        raise ValueError('Sealed artifact is outside its cell: ' + str(path))
    if path.is_symlink() or ex.digest(path) != expected:
        raise ValueError('Artifact checksum mismatch: ' + str(path))


def validate_cell(cell, job, expected_result_sha):
    checked_hash(cell / 'result.json', expected_result_sha)
    result = read(cell / 'result.json')
    if result['status'] != 'completed' or result['job'] != job:
        raise ValueError('Incomplete cell or changed protocol: ' + job['id'])
    count = job['max_images']
    if count != (4 if job['smoke'] else 64) or job['study'] != 'sham':
        raise ValueError('Unexpected evidence population: ' + job['id'])
    seal_path = cell / 'prediction_diagnostic_seal.json'
    checked_hash(seal_path, result['prediction_seal_sha256'])
    seal = read(seal_path)
    if seal['status'] != 'sealed_before_GT' or set(seal['records']) != POLICIES:
        raise ValueError('Missing diagnostic seal or policy: ' + job['id'])
    checked_hash(cell / 'sham_probes.csv', seal['probes_sha256'])
    if not read(cell / 'native_equivalence.json')['exact']:
        raise ValueError('Native behavior equivalence failed')
    for relative, sha in seal['candidate_files'].items():
        checked_hash(cell / relative, sha, cell)
    outcomes = result['outcomes']
    if len(outcomes) != 3 or {o['policy'] for o in outcomes} != POLICIES:
        raise ValueError('Incomplete adaptation outcomes')
    populations = []
    checked_predictions = 0
    for outcome in outcomes:
        if outcome['status'] != 'completed' or outcome['images'] != count or outcome['full_split']:
            raise ValueError('Incorrect outcome population or completion flag')
        path = Path(outcome['prediction_seal'])
        checked_hash(path, outcome['prediction_seal_sha256'], cell)
        prediction_seal = read(path)
        if prediction_seal['labels_in_artifact'] is not False:
            raise ValueError('Prediction payload unexpectedly contains labels')
        records = prediction_seal['records']
        if records != seal['records'][outcome['policy']] or len(records) != count:
            raise ValueError('Prediction and diagnostic records disagree')
        ids = [record['image_id'] for record in records]
        if len(set(ids)) != count or ids != job['image_ids']:
            raise ValueError('Prediction image population differs from manifest')
        populations.append(ids)
        for record in records:
            checked_hash(Path(record['path']), record['sha256'], cell)
            checked_predictions += 1
    if any(ids != populations[0] for ids in populations):
        raise ValueError('Policies were evaluated on different images')
    return dict(id=job['id'], dataset=job['dataset'], images=count,
                smoke=job['smoke'], full_split=False,
                result_sha256=expected_result_sha,
                verified_prediction_files=checked_predictions,
                verified_candidate_files=len(seal['candidate_files']),
                diagnostic_seal_sha256=result['prediction_seal_sha256'])


def main(source=SOURCE, stage=STAGE, dest=DEST):
    if stage.exists() or (dest / 'snapshot.json').exists():
        raise FileExistsError('Use a fresh immutable export destination')
    ready = read(source / 'SHAM_RESULTS_READY.json')
    if ready['status'] != 'completed':
        raise ValueError('Sham results are not complete')
    checked_hash(source / 'manifest.json', read(source / 'manifest_seal.json')['sha256'])
    manifest = read(source / 'manifest.json')
    jobs = [job for job in manifest['jobs'] if job['study'] == 'sham']
    if {job['id'] for job in jobs} != set(ready['results']) or len(jobs) != 4:
        raise ValueError('Completed marker differs from the four-cell protocol')
    reports = [validate_cell(source / 'cells' / job['id'], job, ready['results'][job['id']])
               for job in jobs]
    stage.mkdir(parents=True)
    target = stage / source.name
    target.mkdir()
    for name in ('manifest.json', 'manifest_seal.json', 'SHAM_RESULTS_READY.json', 'README.md'):
        copy_stable(source / name, target / name)
    for job in jobs:
        copy_tree(source / 'cells' / job['id'], target / 'cells' / job['id'])
    copy_tree(source / 'tables', target / 'tables')
    # Result rows must agree with the unrounded formal-only summary being uploaded.
    with (target / 'tables/sham_adaptation.csv').open(newline='') as f:
        table = list(csv.DictReader(f))
    expected = {}
    for job in jobs:
        if not job['smoke']:
            for outcome in read(target / 'cells' / job['id'] / 'result.json')['outcomes']:
                expected[job['dataset'], outcome['policy']] = outcome
    if len(table) != len(expected):
        raise ValueError('Formal summary row count mismatch')
    for row in table:
        outcome = expected.pop((row['dataset'], row['policy']))
        if (int(row['images']) != outcome['images']
                or float(row['mIoU']) != outcome['metrics']['mIoU']
                or float(row['correct_to_wrong_pct']) != outcome['correct_to_wrong_pct']):
            raise ValueError('Summary values disagree with sealed result')
    record = dict(captured_at_utc=ex.utcnow(), source=str(source),
        method='Archived submitted TRACE, not later algorithm variants',
        scope='Two fixed 64-image diagnostic studies and two four-image smoke checks',
        public_upload_authorization='User requested sham upload to public Zonglin-He/B',
        cells=reports, source_modified=False, gpu_work_started=False,
        historical_manifest_also_lists_signals_jobs_not_included=True,
        omitted=['pending or active queue work', 'unrelated copied offline analyses',
                 'source/runtime dependencies', 'dataset images and GT annotations',
                 'model weights', 'credentials'],
        native_fallback='Ineligible sham masks retain native scores in adaptation',
        interpretation='Location specificity does not identify a unique causal source')
    (target / 'capture_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    ex.export(stage, dest)
    finalize_snapshot.finalize(dest, stage)
    ex.verify(dest, full=True)
    snapshot = read(dest / 'snapshot.json')
    if snapshot['excluded_counts']:
        raise ValueError('Unexpected exclusions; inspect export before publication')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--dest', type=Path, default=DEST)
    args = parser.parse_args()
    main(args.source.resolve(), args.stage.resolve(), args.dest.resolve())
