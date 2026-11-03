"""E01/E40: freeze equal search opportunities before any new calibration score."""

from __future__ import annotations

import argparse
import copy
from itertools import product
from pathlib import Path

from tools.evolution_tuning.common import canonical_hash
from .bootstrap import TUNED, read_json, require
from .intake import sha256, write_new

ARMS = ('image_uniform', 'image_confidence', 'image_C')
DATASETS = ('coco_object', 'pc60', 'coco_cd')
COORDINATES = ('batch_size', 'lr', 'steps', 'trace_max_regions')
GRID = dict(batch_size=[2, 8], lr=[.0005, .001, .00375], steps=[5, 30], trace_max_regions=[16, 32])


def candidates(config):
    values = []
    for setting in product(*(GRID[key] for key in COORDINATES)):
        row = copy.deepcopy(config)
        row.update(zip(COORDINATES, setting, strict=True))
        values.append(row)
    values.append(copy.deepcopy(config))  # Same existing calibration-selected anchor for every arm.
    unique = {canonical_hash(row): row for row in values}
    return [unique[key] for key in sorted(unique)]


def select(rows, expected_keys):
    import math
    require(len(rows) == len(expected_keys) and {r['config_sha256'] for r in rows} == set(expected_keys),
            'Equal complete candidate coverage required before selection')
    require(all(r['status'] == 'completed' and type(r['calibration_miou']) in (int, float)
                and math.isfinite(r['calibration_miou']) for r in rows),
            'All frozen calibration trials must complete; no failure skipping')
    # Exact maximum first; deterministic low-update/small-K tie-breaks never use full/test scores.
    return min(rows, key=lambda r: (-r['calibration_miou'], r['config']['steps'],
        r['config']['trace_max_regions'], r['config']['batch_size'], r['config']['lr']))


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output == output.resolve(), 'New E01 protocol directory required')
    prepared_path, inputs_path = TUNED / 'SUITE_PREPARED.json', TUNED / 'inputs.json'
    parent = read_json(prepared_path)
    require(sha256(prepared_path) == '4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3'
            and sha256(inputs_path) == parent['inputs_sha256'], 'Calibration parent/input metadata drift')
    inputs = read_json(inputs_path)['datasets']
    panels, configs, anchors, split_checks = {}, {}, {}, {}
    for dataset in DATASETS:
        entry = inputs[dataset]
        panel = entry['calibration']
        full_path = TUNED / 'full' / f'{dataset}.manifest.json'
        full = read_json(full_path)
        calibration_ids, full_ids = panel['ordered_ids'], full['panel']['ordered_ids']
        require(panel['full_split'] is False and len(calibration_ids) == len(set(calibration_ids)) == 256,
                'Exactly256 training calibration identities required')
        require(not (set(calibration_ids) & set(full_ids)), 'Training calibration/full identity overlap')
        require(full['checkpoint'] == parent['checkpoint'], 'Shared checkpoint differs')
        anchor = full['config']
        anchors[dataset] = anchor
        panels[dataset] = panel
        configs[dataset] = candidates(anchor)
        split_checks[dataset] = dict(calibration_images=256, full_images=len(full_ids), direct_id_overlap=0,
            calibration_panel_sha256=canonical_hash(panel), full_manifest_sha256=sha256(full_path),
            calibration_metadata_origin='Previously verified training-origin panels; historical evaluation exposure retained')
    output.mkdir(parents=True)
    write_new(output / 'PROTOCOL.json', dict(experiments=['E01', 'E40'], stage='protocol_frozen_no_trials',
        datasets=list(DATASETS), arms=list(ARMS), coordinates=list(COORDINATES), grid=GRID,
        anchor_policy='Include prior training-calibration-selected C configuration identically for every arm',
        source_prepared_sha256=sha256(prepared_path), source_inputs_sha256=sha256(inputs_path),
        code_sha256=sha256(__file__), checkpoint=parent['checkpoint'], split_checks=split_checks,
        budget='Equal complete configuration opportunities on identical256 training calibration images per arm',
        equal_compute_time_claim=False, actual_compute_must_be_reported=True,
        time_matched_comparison='Separate E43 protocol; equal grid is not equal GPU time',
        selection='Maximum unrounded calibration pooled mIoU; ties lower steps,K,B,LR; reject missing/failed trials',
        evaluation='Only independently selected frozen configurations; no full/test reselection',
        native_eighteen_cell_queue_unchanged=True, trials_executed=0, gpu_started=False))
    write_new(output / 'calibration_panels.json', panels)
    jobs = [dict(dataset=dataset, arm=arm, config=config, config_sha256=canonical_hash(config))
            for dataset in DATASETS for arm in ARMS for config in configs[dataset]]
    write_new(output / 'trial_grid.json', dict(jobs=jobs, anchors=anchors, all_trials_required=True))
    write_new(output / 'PREPARED.json', dict(executable_queue=False, trials_executed=0, frozen_trials=len(jobs),
        artifact_sha256={n: sha256(output / n) for n in ('PROTOCOL.json', 'calibration_panels.json', 'trial_grid.json')}))
    print(dict(output=str(output), frozen_trial_opportunities=len(jobs), trials_executed=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
