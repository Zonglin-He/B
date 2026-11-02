#!/usr/bin/env python3
"""Read-only publication of the TRACE evolution snapshot authorized 2026-11-02."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import export_results as ex


ROOT = Path('/home/wwww/hzl/results')
SOURCE = Path('/home/wwww/.codex/worktrees/trace-simplification/hzl')
REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'updates/trace-evolution-20261102'
PATTERNS = ('trace_*20261028*', 'trace_*20261029*', 'trace_*20261030*')
SKIP = ex.SKIP_DIRS | {'parents', 'baseline_source', 'calibration_labels',
                      'input_assets', 'gt', 'annotations', 'images', 'checkpoints'}


def read(path):
    return json.loads(path.read_text())


def stable_copy(source, target):
    if source.is_symlink() or not source.is_file():
        raise ValueError('Non-regular publication input: ' + str(source))
    before = source.stat()
    payload = source.read_bytes()
    if ex.has_secret(payload):
        raise ValueError('Credential pattern detected; file not published: ' + str(source))
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during publication: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    if ex.digest(target) != sha:
        raise ValueError('Publication copy mismatch')
    return dict(sha256=sha, bytes=len(payload), original=str(source))


def selected_walk(root):
    paths = sorted({p for pattern in PATTERNS for p in root.glob(pattern)
                    if p.is_dir() or p.suffix == '.json'})
    for suite in paths:
        if suite.is_file():
            yield root, suite, ex.exclusion(suite)
            continue
        for parent, dirs, names in os.walk(suite, followlinks=False):
            keep = []
            for name in sorted(dirs):
                path = Path(parent, name)
                if path.is_symlink():
                    yield suite, path, 'symlink_not_followed'
                elif name in SKIP:
                    yield suite, path, 'runtime_parent_or_dataset_dependency_excluded'
                else:
                    keep.append(name)
            dirs[:] = keep
            for name in sorted(names):
                path = Path(parent, name)
                yield suite, path, ex.exclusion(path)


def source_snapshot():
    raw = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', '-z',
                                   '--cached', '--others', '--exclude-standard'])
    manifest = {}
    for name in sorted(set(raw.decode().split('\0')) - {''}):
        relative = Path(name)
        if relative.parts[0] not in {'trace_ovss', 'tools', 'tests', 'docs', 'paper_evolution'} \
                and name not in {'README.md', 'pyproject.toml', '.gitignore'}:
            continue
        if relative.parts[0] == 'paper_evolution':
            target = DEST / 'paper' / Path(*relative.parts[1:])
        else:
            target = DEST / 'code' / relative
        if (SOURCE / relative).is_file():
            manifest[str(target.relative_to(DEST))] = stable_copy(SOURCE / relative, target)
    for name in ('main.pdf', 'supplement.pdf'):
        target = DEST / 'paper/build' / name
        manifest[str(target.relative_to(DEST))] = stable_copy(SOURCE / 'paper_evolution/build' / name, target)
    frozen = ROOT / 'trace_evolution_review_round05_20261029_tuned'
    packet = read(frozen / 'packet_manifest.json')
    if ex.digest(frozen / 'packet_manifest.json') != '386ae89c92a21329dcef08d21262071dac27e71788da07879170885943c8fb23':
        raise ValueError('Wrong round05 packet')
    for name, expected in packet['files'].items():
        if ex.digest(frozen / name) != expected:
            raise ValueError('Frozen paper mismatch: ' + name)
    manifest['paper-round05/packet_manifest.json'] = stable_copy(
        frozen / 'packet_manifest.json', DEST / 'paper-round05/packet_manifest.json')
    for name in packet['files']:
        target = DEST / 'paper-round05' / name
        manifest[str(target.relative_to(DEST))] = stable_copy(frozen / name, target)
    info = dict(captured_at_utc=ex.utcnow(), source_commit=subprocess.check_output(
        ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_worktree_included=True, files=manifest,
        exclusions=['datasets/sample images', 'model weights', 'private credentials',
                    'untracked ignored build byproducts other than two current PDFs'],
        paper_status='Existing round05 draft, not a new scientifically completed revision',
        prior_review=dict(score=5, scale=10, confidence=4, confidence_scale=5),
        human_authorization='User explicitly requested upload to public Zonglin-He/B on 2026-11-02')
    (DEST / 'source_manifest.json').write_text(json.dumps(info, indent=2) + '\n')


def canonical_tables():
    rows = []
    base = ROOT / 'trace_single_probe_waterbirds_full_20261029_0817'
    for result in sorted((base / 'full').glob('*/result.json')):
        value = read(result)
        assert value['status'] == 'completed' and value['full_split'] is True
        for arm, outcome in value['arm_results'].items():
            rows.append(dict(dataset=value['dataset'], arm=arm, images=value['images'],
                             mIoU=outcome['miou_unrounded'], source=str(result),
                             result_sha256=ex.digest(result)))
    assert len(rows) == 32
    target = DEST / 'tables/tuned_full_32.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    ex.write_csv(target, rows, list(rows[0]))
    stable_copy(ROOT / 'trace_allocation_controls_20261029_2055/full_metrics.csv',
                DEST / 'tables/allocation_full_10.csv')
    trials = []
    for suite in ('trace_single_probe_tuning_20261029_0748', base.name):
        with (ROOT / suite / 'trials.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                trials.append(dict(row, source_suite=suite))
    assert len(trials) == 455 and all(row['status'] == 'completed' for row in trials)
    ex.write_csv(DEST / 'tables/calibration_trials_455.csv', trials, list(trials[0]))


def main():
    if (DEST / 'snapshot.json').exists():
        raise ValueError('Immutable publication snapshot already exists')
    DEST.mkdir(parents=True, exist_ok=True)
    source_snapshot()
    canonical_tables()
    units = {}
    for unit in ('trace-local-baselines-20261030-0008.service',):
        state = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '-p', 'LoadState', '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
            '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'InvocationID'], text=True)
        units[unit] = dict(line.split('=', 1) for line in state.splitlines() if '=' in line)
    (DEST / 'execution_status_at_capture.json').write_text(json.dumps(dict(
        captured_at_utc=ex.utcnow(), services=units,
        baseline=dict(passed_gates=0, formal_predictions=0, new_scores=0,
            failure='Unrelated CFCTTA eager package import; MLMP constructor not entered'),
        latest_human_decision='Import repair, CPU regression, independent audit and fresh single controlled queue approved; not yet executed',
        publication_started_gpu_work=False), indent=2) + '\n')
    ex.walk_results = selected_walk
    original_family = ex.family
    ex.family = lambda name: 'trace_evolution_snapshot_protocols_not_interchangeable' if name.startswith('trace_') else original_family(name)
    ex.TEXT.add('.jsonl')
    old_browse = ex.browse_candidate
    ex.browse_candidate = lambda rel, size: False if 'episodes' in rel.parts or 'trials' in rel.parts else old_browse(rel, size)
    ex.export(ROOT, DEST, part_mib=1200)
    print('EXPORT_COMPLETE; verifying archived objects, not rerunning scientific certification', flush=True)
    ex.verify(DEST, full=True)


if __name__ == '__main__':
    main()
