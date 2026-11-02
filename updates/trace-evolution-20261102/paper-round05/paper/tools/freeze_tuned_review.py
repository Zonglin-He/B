"""Freeze the eight-dataset manuscript for a fresh independent review."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


SUITE = Path('/home/wwww/hzl/results/trace_single_probe_waterbirds_full_20261029_0817')
PREPARED = '4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3'
SERVICE = 'trace-waterbirds-full-20261029-0817.service'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output):
    source = Path(__file__).resolve().parents[1]
    if sha(SUITE / 'SUITE_PREPARED.json') != PREPARED:
        raise ValueError('Wrong experiment identity')
    state = subprocess.run(['systemctl', '--user', 'show', SERVICE,
        '-p', 'LoadState', '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
        '-p', 'Result', '-p', 'ExecMainStatus'], check=True, text=True,
        capture_output=True).stdout
    status = dict(line.split('=', 1) for line in state.splitlines() if '=' in line)
    expected = dict(LoadState='loaded', ActiveState='active', SubState='exited',
                    MainPID='0', Result='success', ExecMainStatus='0')
    if status != expected:
        raise ValueError('Experiment not at verified retained-success terminal')
    for name in ('main.log', 'supplement.log'):
        log = (source / 'build' / name).read_text(errors='replace')
        if any(term in log for term in ('Overfull', 'Float too large',
                'undefined references', 'undefined citations', 'multiply defined')):
            raise ValueError(f'Unresolved LaTeX diagnostic: {name}')
    files = [Path(p) for p in (
        'main.tex', 'appendix.tex', 'supplement.tex', 'preamble.tex',
        'references.bib', 'aaai2027.sty', 'aaai2027.bst', 'reported_baselines.json',
        'build/main.pdf', 'build/supplement.pdf', 'build/main.aux', 'build/main.bbl',
        'build/main.log', 'build/supplement.log',
        'generated/tuned_full_comparison.tex', 'generated/tuned_full_reference.tex',
        'generated/tuned_full_configurations.tex', 'generated/tuned_full_cost.tex',
        'generated/tuned_full_provenance.json',
        'generated/image_budget_comparison.tex', 'generated/image_budget_cost.tex',
        'generated/image_budget_provenance.json',
        'tools/export_tuned_full.py', 'tools/freeze_tuned_review.py',
    )]
    figures = source / 'figure' / 'tuned'
    if not (figures / 'image_full_provenance.json').is_file():
        raise ValueError('Current-configuration figure provenance missing')
    files.extend(p.relative_to(source) for p in sorted(figures.iterdir()) if p.is_file())
    before = {str(p): sha(source / p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    for relative in files:
        target = output / 'paper' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        if sha(target) != before[str(relative)] or sha(source / relative) != before[str(relative)]:
            raise ValueError(f'Freeze drift: {relative}')
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        method='ImageBalancedTRACE global C; one context probe; image-balanced regional entropy',
        suite_prepared_sha256=PREPARED, service=status,
        files={'paper/' + k: v for k, v in before.items()},
        evidence_roots=[str(SUITE),
            '/home/wwww/hzl/results/trace_single_probe_tuning_20261029_0748',
            '/home/wwww/hzl/results/trace_image_budget_20261029_0233',
            '/home/wwww/hzl/results/trace_two_view_full_baselines_20261025/coco_cd'],
        scope='Frozen paper and provenance. No author reviews or desired rating supplied. '
              'Original experiment artifacts remain read-only.')
    (output / 'packet_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), files=len(before),
                         manifest_sha256=sha(output / 'packet_manifest.json')), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args().output)
