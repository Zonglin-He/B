#!/usr/bin/env python3
"""Export result evidence without touching the experiment workspace or GPU."""

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile


TEXT = {'.json', '.csv', '.tsv', '.md', '.txt', '.tex', '.yaml', '.yml',
        '.log', '.stdout', '.xml', '.sha256'}
BINARY = {'.npz', '.npy', '.png', '.jpg', '.jpeg', '.svg', '.pdf', '.db'}
SKIP_DIRS = {'.git', '__pycache__', 'runtime', 'source_snapshot', 'frozen_source',
             'source', 'source_reference', 'vendor', 'code', '_snapshot',
             'snapshots', 'assets', 'models', 'data', 'wandb', 'tensorboard'}
SECRET = re.compile(
    rb'\b(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,}'
    rb'|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b'
    rb'|-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'
    rb'|(?i:password|passwd|sudo_password)\s*[\"\x27]?\s*[:=]\s*[\"\x27]?\d{8,}'
)
ASSIGNMENT = re.compile(
    rb'(?i)\b(?:password|passwd|sudo_password|api[_-]?key|access_token|refresh_token|client_secret)'
    rb'[\"\x27]?\s*[:=]\s*(?:\"([^\"\r\n]+)\"|\x27([^\x27\r\n]+)\x27|([A-Za-z0-9_@!#$%.\-]{4,}))'
)
FIELDS = ['path', 'suite', 'family', 'bytes', 'mtime_ns', 'sha256', 'archive',
          'member', 'git_path', 'disposition', 'observation']


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def has_secret(data):
    lower = data.lower()
    markers = (b'ghp_', b'gho_', b'ghu_', b'ghs_', b'ghr_', b'github_pat_',
               b'sk-', b'akia', b'-----begin ', b'password', b'passwd',
               b'api_key', b'api-key', b'apikey', b'access_token', b'refresh_token', b'client_secret')
    if not any(marker in lower for marker in markers):
        return False
    if SECRET.search(data):
        return True
    placeholders = {b'none', b'null', b'false', b'true', b'redacted', b'placeholder',
                    b'example', b'password', b'passwd', b'api_key', b'access_token'}
    for match in ASSIGNMENT.finditer(data):
        value = next(v for v in match.groups() if v is not None)
        if value.lower() in placeholders or set(value) <= {42}:
            continue
        if value.startswith((b'os.', b'args.', b'config.', b'self.', b'getenv')):
            continue
        return True
    return False


def exclusion(path):
    if path.is_symlink():
        return 'symlink_not_followed'
    if path.name.startswith('.env') or path.name.lower() in {
            'credentials', 'credentials.json', 'hosts.yml', 'id_rsa', 'id_ed25519',
            '.netrc', '.npmrc', '.pypirc'}:
        return 'credential_filename'
    if path.suffix.lower() not in TEXT | BINARY:
        return 'not_a_result_artifact_or_excluded_weight_source_archive'
    return ''


def family(name):
    if name.startswith(('rebuttal_original_', 'rebuttal_priority64_', 'rebuttal_role_policy64_', 'rebuttal_sham_', 'reviewer_followup_',
                        'reviewer_response_', 'reviewer_corruption_', 'trace_runtime_')):
        return 'submitted_trace_reviewer_evidence'
    if name.startswith('rebuttal_backbone_'):
        return 'new_backbone_validation_or_preparation'
    if '20261024' in name or '20261025' in name:
        return 'resubmission_algorithm_variants'
    return 'historical_results_do_not_assume_current_method'


def walk_results(root):
    def priority(p):
        name = p.name
        if name.startswith('rebuttal_original_'):
            return (0, name)
        if name.startswith(('reviewer_followup_', 'reviewer_response_', 'reviewer_corruption_')):
            return (1, name)
        if '20261025' in name or '20261024' in name:
            return (2, name)
        return (3, name)

    for suite in sorted(root.iterdir(), key=priority):
        if suite.is_file() or suite.is_symlink():
            yield root, suite, exclusion(suite)
            continue
        if not suite.is_dir() or suite.is_symlink():
            continue
        if suite.name.startswith('driver_compat_'):
            yield suite, suite, 'non_experiment_driver_runtime'
            continue
        for parent, dirs, names in os.walk(suite, followlinks=False):
            keep = []
            for name in sorted(dirs):
                p = Path(parent, name)
                if p.is_symlink():
                    yield suite, p, 'symlink_not_followed'
                elif name in SKIP_DIRS:
                    yield suite, p, 'source_runtime_or_dependency_directory'
                else:
                    keep.append(name)
            dirs[:] = keep
            for name in sorted(names):
                p = Path(parent, name)
                yield suite, p, exclusion(p)


def scan_file(path):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        return None, 'not_regular_file'
    h, tail, sensitive = hashlib.sha256(), b'', False
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
            if path.suffix.lower() in TEXT | {'.svg'}:
                if has_secret(tail + chunk):
                    sensitive = True
                tail = chunk[-256:]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        return None, 'changed_during_export'
    if sensitive:
        return None, 'sensitive_content_not_uploaded'
    return (h.hexdigest(), after), ''


def browse_candidate(relative, size):
    if size > 1024 * 1024:
        return False
    if any(p in {'candidates', 'predictions', 'per_image', 'drift'} for p in relative.parts):
        return False
    if relative.suffix.lower() in {'.csv', '.tsv', '.md', '.tex', '.yaml', '.yml'}:
        return True
    if relative.suffix.lower() == '.json':
        return any(s in relative.name.lower() for s in (
            'result', 'summary', 'protocol', 'config', 'manifest', 'best',
            'metrics', 'seal', 'environment', 'provenance', 'queue_state', 'checks'))
    return relative.name in {'results.txt', 'summary.txt', 'metrics.txt'}


class ArchiveWriter:
    def __init__(self, assets, max_bytes, start_index=0):
        self.assets, self.limit = assets, max_bytes
        self.start_index = start_index
        self.stream = self.tar = None
        self.size = 0
        self.parts = []
        self.seen = {}

    def close_part(self):
        if self.tar is not None:
            self.tar.close()
            self.gz.close()
            self.stream.close()
            self.parts[-1]['bytes'] = self.path.stat().st_size
            self.parts[-1]['sha256'] = digest(self.path)
            self.tar = self.stream = None

    def add(self, path, sha, st):
        if sha in self.seen:
            return self.seen[sha]
        if self.tar is None or self.size + st.st_size > self.limit:
            self.close_part()
            self.path = self.assets / f'results-objects-{len(self.parts) + 1 + self.start_index:03d}.tar.gz'
            self.stream = self.path.open('wb')
            gz = gzip.GzipFile(filename='', mode='wb', compresslevel=1,
                               fileobj=self.stream, mtime=0)
            self.tar = tarfile.open(fileobj=gz, mode='w|')
            self.gz = gz
            self.size = 0
            self.parts.append({'name': self.path.name, 'objects': 0})
        member = f'objects/{sha}'
        info = tarfile.TarInfo(member)
        info.size, info.mode, info.mtime = st.st_size, 0o644, st.st_mtime
        h = hashlib.sha256()

        class CheckedReader:
            def __init__(self, f):
                self.f = f

            def read(self, n):
                chunk = self.f.read(n)
                h.update(chunk)
                return chunk

        with path.open('rb') as f:
            self.tar.addfile(info, CheckedReader(f))
        after = path.stat()
        if h.hexdigest() != sha or (st.st_size, st.st_mtime_ns) != (
                after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Artifact changed while packing; retry export: {path}')
        self.size += st.st_size
        self.parts[-1]['objects'] += 1
        self.seen[sha] = (self.path.name, member)
        return self.seen[sha]

    def finish(self):
        self.close_part()


def result_rows(data, relative, suite, sha):
    if relative.name not in {'result.json', 'results.json', 'metrics.json', 'best_score.json'}:
        return []
    try:
        x = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return []
    if not isinstance(x, dict):
        return []
    job = x.get('job') if isinstance(x.get('job'), dict) else {}
    outcomes = x.get('outcomes')
    outcomes = outcomes if isinstance(outcomes, list) and outcomes else [x]
    result = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        metrics = outcome.get('metrics') if isinstance(outcome.get('metrics'), dict) else {}
        score = next((v for v in (metrics.get('mIoU'), outcome.get('mIoU'),
                                 outcome.get('miou')) if isinstance(v, (int, float))), None)
        result.append({'path': relative.as_posix(), 'family': family(suite), 'suite': suite,
            'status': outcome.get('status', x.get('status', 'not_declared')),
            'dataset': outcome.get('dataset', x.get('dataset', job.get('dataset', ''))),
            'method': outcome.get('method', x.get('method', x.get('algorithm', ''))),
            'variant': outcome.get('variant', outcome.get('policy', x.get('variant', job.get('variant', '')))),
            'seed': outcome.get('seed', x.get('seed', job.get('seed', ''))),
            'images': outcome.get('images', x.get('images', '')),
            'full_split': outcome.get('full_split', x.get('full_split', '')),
            'smoke': job.get('smoke', x.get('smoke', '')),
            'evidence_scope': x.get('evidence_scope', x.get('evidence', '')),
            'mIoU': score if score is not None else '',
            'reported_score': outcome.get('score', x.get('score', '')), 'sha256': sha})
    return result


def result_row(path, relative, suite, sha):
    if path.name not in {'result.json', 'results.json', 'metrics.json', 'best_score.json'}:
        return None
    if path.stat().st_size > 1024 * 1024:
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        return None
    rows = result_rows(data, relative, suite, sha)
    return rows[0] if rows else None


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def export(root, dest, part_mib=1200):
    root, dest = root.resolve(), dest.resolve()
    if root in dest.parents or dest == root:
        raise ValueError('Destination must be outside the source results tree')
    if (dest / 'snapshot.json').exists():
        raise ValueError('Use a fresh export directory; existing snapshots are immutable')
    dest.mkdir(parents=True, exist_ok=True)
    assets = dest / 'assets'
    index = dest / 'index'
    assets.mkdir(exist_ok=True)
    index.mkdir(exist_ok=True)
    start = utcnow()
    archive = ArchiveWriter(assets, part_mib * 1024 * 1024)
    suites, results = {}, []
    counters, excluded = collections.Counter(), collections.Counter()
    git_bytes = 0
    manifest = index / 'files.csv.gz'
    with gzip.open(manifest, 'wt', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for suite, p, reason in walk_results(root):
            relative = p.relative_to(root)
            suite_name = suite.name if suite != root else '_root_result_files'
            s = suites.setdefault(suite_name, {'suite': suite_name, 'family': family(suite_name),
                'included_files': 0, 'included_bytes': 0, 'excluded_entries': 0})
            row = {'path': relative.as_posix(), 'suite': suite_name,
                   'family': family(suite_name), 'disposition': reason,
                   'observation': 'not_an_atomic_filesystem_snapshot'}
            try:
                st = p.lstat()
                row.update(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
                if not reason:
                    scanned, reason = scan_file(p)
                    if reason == 'changed_during_export':
                        scanned, reason = scan_file(p)
                    if not reason:
                        sha, st = scanned
                        row.update(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
                        name, member = archive.add(p, sha, st)
                        row.update(sha256=sha, archive=name, member=member,
                                   disposition='included', observation='stable_during_hash_and_pack')
                        if browse_candidate(relative, st.st_size) and git_bytes + st.st_size < 180 * 1024**2:
                            target = dest / 'results' / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(p, target)
                            if digest(target) != sha:
                                target.unlink()
                            else:
                                git_bytes += st.st_size
                                row['git_path'] = target.relative_to(dest).as_posix()
                        entry = result_row(p, relative, suite_name, sha)
                        if entry is not None:
                            results.append(entry)
                        s['included_files'] += 1
                        s['included_bytes'] += st.st_size
                        counters['included_files'] += 1
                        counters['included_bytes'] += st.st_size
                    else:
                        row['disposition'] = reason
            except FileNotFoundError:
                row['disposition'] = 'disappeared_during_export'
            if row['disposition'] != 'included':
                excluded[row['disposition']] += 1
                s['excluded_entries'] += 1
            w.writerow(row)
            if sum(counters.values()) and counters['included_files'] % 20000 == 0:
                print(json.dumps({'included_files': counters['included_files'],
                                  'logical_GiB': round(counters['included_bytes'] / 1024**3, 2)}), flush=True)
    archive.finish()
    write_csv(index / 'suites.csv', suites.values(),
              ['suite', 'family', 'included_files', 'included_bytes', 'excluded_entries'])
    fields = ['path', 'family', 'suite', 'status', 'dataset', 'method', 'variant',
              'seed', 'images', 'full_split', 'smoke', 'evidence_scope', 'mIoU',
              'reported_score', 'sha256']
    write_csv(index / 'runs.csv', results, fields)
    state = {}
    for p in root.glob('*/queue_state.json'):
        q = dest / 'results' / p.relative_to(root)
        if not q.exists():
            continue
        try:
            x = json.loads(q.read_text())
            cells = x.get('cells', {})
            counts = collections.Counter(v.get('status', 'unknown') for v in cells.values()) if isinstance(cells, dict) else {}
            state[p.parent.name] = {'status': x.get('status'), 'active': x.get('active'),
                                   'counts': dict(counts), 'total': x.get('total')}
        except (ValueError, AttributeError):
            state[p.parent.name] = {'status': 'unparseable'}
    snapshot = {'started_at_utc': start, 'finished_at_utc': utcnow(),
        'source_root': str(root), 'snapshot_type': 'read_only_live_file_snapshot',
        'atomic': False, 'source_modified': False, 'gpu_work_started': False,
        'included_files': counters['included_files'], 'logical_bytes': counters['included_bytes'],
        'unique_objects': len(archive.seen), 'git_result_bytes': git_bytes,
        'excluded_counts': dict(excluded), 'suite_count': len(suites),
        'run_index_rows': len(results), 'archives': archive.parts,
        'files_manifest_sha256': digest(manifest), 'queue_snapshots': state}
    (dest / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (index / 'archives.json').write_text(json.dumps(archive.parts, indent=2) + '\n')
    print(json.dumps(snapshot, indent=2), flush=True)
    return snapshot


def verify(dest, full=False):
    snapshot = json.loads((dest / 'snapshot.json').read_text())
    if digest(dest / 'index/files.csv.gz') != snapshot['files_manifest_sha256']:
        raise ValueError('File manifest checksum mismatch')
    found = {}
    for item in snapshot['archives']:
        path = dest / 'assets' / item['name']
        if digest(path) != item['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path.name}')
        with tarfile.open(path, 'r|gz') as tar:
            count = 0
            for member in tar:
                if not re.fullmatch(r'objects/[0-9a-f]{64}', member.name) or not member.isfile():
                    raise ValueError('Unsafe archive member')
                if full:
                    h = hashlib.sha256()
                    stream = tar.extractfile(member)
                    tail = b''
                    for chunk in iter(lambda: stream.read(1024**2), b''):
                        h.update(chunk)
                        if has_secret(tail + chunk):
                            raise ValueError('Sensitive-content pattern in archived object')
                        tail = chunk[-256:]
                    if h.hexdigest() != member.name.split('/')[1]:
                        raise ValueError('Object checksum mismatch')
                found[(path.name, member.name)] = member.size
                count += 1
            if count != item['objects']:
                raise ValueError('Archive object count mismatch')
    rows = 0
    with gzip.open(dest / 'index/files.csv.gz', 'rt', newline='') as f:
        for row in csv.DictReader(f):
            if row['disposition'] == 'included':
                if (row['archive'], row['member']) not in found:
                    raise ValueError(f'Missing object: {row["path"]}')
                if row['member'] != f'objects/{row["sha256"]}' or int(row['bytes']) != found[(row['archive'], row['member'])]:
                    raise ValueError(f'Inconsistent manifest metadata: {row["path"]}')
                if row['git_path'] and digest(dest / row['git_path']) != row['sha256']:
                    raise ValueError(f'Browsable copy mismatch: {row["path"]}')
                rows += 1
    if rows != snapshot['included_files']:
        raise ValueError('Manifest row count mismatch')
    result = {'verified': True, 'full_object_hashes': full,
              'full_object_secret_scan': full, 'files': rows,
              'objects': len(found), 'checked_at_utc': utcnow()}
    (dest / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


def restore(dest, output, suite=None):
    if not (dest / 'verification.json').exists():
        raise ValueError('Run verify before restore')
    snapshot = json.loads((dest / 'snapshot.json').read_text())
    if digest(dest / 'index/files.csv.gz') != snapshot['files_manifest_sha256']:
        raise ValueError('File manifest checksum mismatch')
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError('Symlink in restore output ancestry')
    targets = collections.defaultdict(list)
    with gzip.open(dest / 'index/files.csv.gz', 'rt', newline='') as f:
        for row in csv.DictReader(f):
            if row['disposition'] != 'included' or (suite and row['suite'] != suite):
                continue
            relative = PurePosixPath(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe target path')
            path = output / relative
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
            targets[(row['archive'], row['member'])].append(path)
    restored = 0
    output.mkdir(parents=True, exist_ok=True)
    for item in snapshot['archives']:
        path = dest / 'assets' / item['name']
        if digest(path) != item['sha256']:
            raise ValueError('Archive checksum mismatch')
        with tarfile.open(path, 'r|gz') as tar:
            for member in tar:
                paths = targets.get((path.name, member.name), [])
                if not paths:
                    continue
                with tempfile.TemporaryFile() as tmp:
                    stream = tar.extractfile(member)
                    h = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024**2), b''):
                        h.update(chunk)
                        tmp.write(chunk)
                    if h.hexdigest() != member.name.split('/')[1]:
                        raise ValueError('Object checksum mismatch')
                    for target in paths:
                        if any(parent.is_symlink() for parent in target.parents):
                            raise ValueError('Symlink in restore destination')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as f:
                            tmp.seek(0)
                            shutil.copyfileobj(tmp, f)
                        restored += 1
    print(json.dumps({'restored_files': restored, 'suite': suite}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--part-mib', type=int, default=1200)
    p = sub.add_parser('verify')
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--full', action='store_true')
    p = sub.add_parser('restore')
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite')
    args = parser.parse_args()
    if args.command == 'export':
        export(args.source, args.dest, args.part_mib)
    elif args.command == 'verify':
        verify(args.dest, args.full)
    else:
        restore(args.dest, args.output, args.suite)
