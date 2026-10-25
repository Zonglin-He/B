#!/usr/bin/env python3
"""Audit archived bytes, add root evidence, and rebuild schema-aware indices."""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

import export_results as ex


def finalize(dest, source):
    snapshot = json.loads((dest / 'snapshot.json').read_text())
    with gzip.open(dest / 'index/files.csv.gz', 'rt', newline='') as f:
        rows = list(csv.DictReader(f))
    result_names = {'result.json', 'results.json', 'metrics.json', 'best_score.json', 'queue_state.json'}
    want_json = {r['sha256'] for r in rows if Path(r['path']).name in result_names and r['disposition'] == 'included'}
    bans = {}
    for row in rows:
        if row['disposition'] != 'included':
            continue
        reason = ex.exclusion(source / row['path'])
        if reason:
            if Path(row['path']).suffix.lower() in {'.diff', '.patch'}:
                reason = 'source_patch_not_uploaded'
            bans[row['sha256']] = reason
    banned = set(bans)
    metadata, small_json = {}, {}
    for part in snapshot['archives']:
        path = dest / 'assets' / part['name']
        if ex.digest(path) != part['sha256']:
            raise ValueError('Original archive checksum mismatch')
        with tarfile.open(path, 'r|gz') as tar:
            for member in tar:
                if not member.isfile() or not ex.re.fullmatch(r'objects/[0-9a-f]{64}', member.name):
                    raise ValueError('Unsafe archive object')
                sha = member.name.split('/')[1]
                h, tail, sensitive = hashlib.sha256(), b'', False
                chunks = [] if sha in want_json and member.size <= 1024**2 else None
                stream = tar.extractfile(member)
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    h.update(chunk)
                    sensitive = sensitive or ex.has_secret(tail + chunk)
                    tail = chunk[-256:]
                    if chunks is not None:
                        chunks.append(chunk)
                if h.hexdigest() != sha:
                    raise ValueError('Original object checksum mismatch')
                if sensitive:
                    banned.add(sha)
                    bans[sha] = 'sensitive_content_not_uploaded'
                metadata[(part['name'], member.name)] = (member.size, int(member.mtime * 10**9))
                if chunks is not None:
                    small_json[sha] = b''.join(chunks)
        print(json.dumps({'audited': part['name'], 'excluded_objects': len(banned)}), flush=True)

    # Remove excluded objects from the actual release payload, not just its index.
    for part in snapshot['archives']:
        path = dest / 'assets' / part['name']
        part_bans = {member for (name, member) in metadata if name == part['name'] and member.split('/')[1] in banned}
        if not part_bans:
            continue
        tmp = path.with_suffix('.audited.tmp')
        count = 0
        with path.open('rb') as original, tmp.open('wb') as output:
            with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=output, mtime=0) as gz:
                with tarfile.open(fileobj=original, mode='r|gz') as old, tarfile.open(fileobj=gz, mode='w|') as new:
                    for member in old:
                        if member.name in part_bans:
                            continue
                        stream = old.extractfile(member)
                        h = hashlib.sha256()

                        class Reader:
                            def read(self, size):
                                chunk = stream.read(size)
                                h.update(chunk)
                                return chunk

                        new.addfile(member, Reader())
                        if h.hexdigest() != member.name.split('/')[1]:
                            raise ValueError('Repacked object checksum mismatch')
                        count += 1
        os.replace(tmp, path)
        part.update(objects=count, bytes=path.stat().st_size, sha256=ex.digest(path))

    for row in rows:
        if row['disposition'] != 'included':
            continue
        if row['sha256'] in banned:
            if row['git_path']:
                (dest / row['git_path']).unlink(missing_ok=True)
            row.update(disposition=bans[row['sha256']], archive='', member='', git_path='')
        else:
            size, _ = metadata[(row['archive'], row['member'])]
            row.update(bytes=size)

    # The first exporter version omitted top-level result files. Capture them too.
    writer = ex.ArchiveWriter(dest / 'assets', 1200 * 1024**2, len(snapshot['archives']))
    writer.seen = {r['sha256']: (r['archive'], r['member']) for r in rows if r['disposition'] == 'included'}
    known = {r['path'] for r in rows}
    for p in sorted(source.iterdir()):
        if p.is_dir() and not p.is_symlink():
            continue
        if p.name in known:
            continue
        row = {'path': p.name, 'suite': '_root_result_files',
               'family': ex.family('_root_result_files'), 'observation': 'stable_root_capture',
               'disposition': ex.exclusion(p)}
        st = p.lstat()
        row.update(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        if not row['disposition']:
            data = p.read_bytes()
            after = p.stat()
            if (st.st_size, st.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                row['disposition'] = 'changed_during_export'
            elif ex.has_secret(data):
                row['disposition'] = 'sensitive_content_not_uploaded'
            else:
                sha = hashlib.sha256(data).hexdigest()
                with tempfile.TemporaryDirectory() as temp:
                    stable = Path(temp) / p.name
                    stable.write_bytes(data)
                    os.utime(stable, ns=(st.st_atime_ns, st.st_mtime_ns))
                    name, member = writer.add(stable, sha, stable.stat())
                row.update(sha256=sha, archive=name, member=member, disposition='included')
                if ex.browse_candidate(Path(p.name), len(data)):
                    target = dest / 'results' / p.name
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(data)
                    row['git_path'] = target.relative_to(dest).as_posix()
                if p.name in result_names:
                    small_json[sha] = data
        rows.append(row)
    writer.finish()
    snapshot['archives'].extend(writer.parts)

    suites, results, queues = {}, [], {}
    for row in rows:
        s = suites.setdefault(row['suite'], {'suite': row['suite'], 'family': row['family'],
                 'included_files': 0, 'included_bytes': 0, 'excluded_entries': 0})
        if row['disposition'] != 'included':
            s['excluded_entries'] += 1
            continue
        s['included_files'] += 1
        s['included_bytes'] += int(row['bytes'])
        data = small_json.get(row['sha256'])
        if data is not None:
            results.extend(ex.result_rows(data, Path(row['path']), row['suite'], row['sha256']))
            if Path(row['path']).name == 'queue_state.json':
                try:
                    state = json.loads(data)
                    cells = state.get('cells', {})
                    counts = collections.Counter(v.get('status', 'unknown') for v in cells.values()) if isinstance(cells, dict) else {}
                    queues[row['suite']] = {'status': state.get('status'), 'active': state.get('active'),
                                           'counts': dict(counts), 'total': state.get('total')}
                except (ValueError, AttributeError):
                    queues[row['suite']] = {'status': 'unparseable'}
    manifest = dest / 'index/files.csv.gz'
    with gzip.open(manifest, 'wt', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ex.FIELDS)
        w.writeheader()
        w.writerows(rows)
    ex.write_csv(dest / 'index/suites.csv', suites.values(),
                 ['suite', 'family', 'included_files', 'included_bytes', 'excluded_entries'])
    ex.write_csv(dest / 'index/runs.csv', results, ['path', 'family', 'suite', 'status',
        'dataset', 'method', 'variant', 'seed', 'images', 'full_split', 'smoke',
        'evidence_scope', 'mIoU', 'reported_score', 'sha256'])
    included = [r for r in rows if r['disposition'] == 'included']
    snapshot.update(included_files=len(included), logical_bytes=sum(int(r['bytes']) for r in included),
        unique_objects=len({r['sha256'] for r in included}),
        git_result_bytes=sum(int(r['bytes']) for r in included if r.get('git_path')),
        suite_count=len(suites), run_index_rows=len(results), queue_snapshots=queues,
        excluded_counts=dict(collections.Counter(r['disposition'] for r in rows if r['disposition'] != 'included')),
        files_manifest_sha256=ex.digest(manifest), finalized_at_utc=ex.utcnow(),
        source_patch_and_credential_audit=True, run_index_source='captured_archive_bytes')
    (dest / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (dest / 'index/archives.json').write_text(json.dumps(snapshot['archives'], indent=2) + '\n')
    print(json.dumps({'finalized': True, 'files': snapshot['included_files'],
                      'run_rows': len(results), 'excluded_objects': len(banned)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.dest.resolve(), args.source.resolve())
