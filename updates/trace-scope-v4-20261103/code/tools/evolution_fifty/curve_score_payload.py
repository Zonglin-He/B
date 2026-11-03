"""Bounded G03 scoring IO; admission and terminal proof belong to the caller.

New s1/s3 predictions use G's palette/OpenCV geometry. Sealed s5 episode
counts are streamed without reopening s5 predictions or recomputing old CIs.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat

import numpy as np

from . import generalization_score as metric
from . import generalization_score_worker as protected
from . import legacy_score_payload as native_inputs
from .bootstrap import require
from .corruption_curve import same_provenance
from .generalization import ARMS
from .risk_score_payload import annotation_bytes, hard_array, stable_bytes

ROOT = Path(__file__).resolve().parents[2]
RESERVE, OUTPUT_CAP = 16*1024**3, 256*1024**2
LINE_CAP, REUSED_CAP = 128*1024, 16*1024**2


def data_exclusions(source):
    native = source["native"]
    native_inputs.preflight_annotations(dict(annotation_metadata=native["annotation_metadata"], rows=source["rows"]))
    forbidden = {(v["device"], v["inode"]) for v in native["annotation_metadata"].values()}
    annotations = set(forbidden)
    for rows in source["rows"].values():
        for row in rows:
            path = Path(row["image"])
            state = path.lstat()
            require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(state.st_mode)
                    and (state.st_dev, state.st_ino) not in annotations, "G03 RGB/GT roles differ")
            forbidden.add((state.st_dev, state.st_ino))
    return forbidden


def file_identity(path, forbidden, limit, seen=None):
    path = Path(path)
    state = path.lstat()
    inode = (state.st_dev, state.st_ino)
    require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(state.st_mode)
            and state.st_nlink == 1 and inode not in forbidden and 0 < state.st_size <= limit
            and (seen is None or inode not in seen), "G03 payload path/role/extent differs")
    if seen is not None:
        seen.add(inode)
    return [*inode, state.st_size, state.st_mtime_ns]


def preflight_payloads(parent, specs, seals, source):
    """Stat the entire graph before reading any payload, including late items."""
    parent = Path(parent)
    forbidden = data_exclusions(source)
    for root, snapshot in ((ROOT, True), (parent/"runtime", False)):
        if root == ROOT or root.exists():
            members = protected.source_members(root, forbidden, snapshot_source=snapshot)
            forbidden.update(v[:2] for v in members.values())
    identities, seen, names = {}, set(), set()
    require(type(specs) is list and 0 < len(specs) <= 10, "G03 condition population bound differs")
    for spec in specs:
        name, images, batch = spec["id"], spec["images"], spec["config"]["batch_size"]
        require(type(name) is str and name and "/" not in name and name not in (".", "..") and name not in names
                and spec["dataset"] == "voc20" and spec["arms"] == list(ARMS)
                and type(images) is int and 0 < images <= 1449 and type(batch) is int and 0 < batch <= 512
                and len(source["rows"][spec["input_key"]]) == images, "G03 condition identity/geometry differs")
        names.add(name)
        prefix = f"conditions/{name}/"
        expected = {prefix+f"predictions/{arm}/{i:06d}.npy": 2*1024**2 for arm in ARMS for i in range(images)}
        expected.update({prefix+f"episodes/{e:06d}.{arm}.json": 8*1024**2
                         for arm in ARMS for e in range((images+batch-1)//batch)})
        expected.update({prefix+n: limit for n, limit in (("cost.jsonl", 16*1024**2),
                         ("initial_state.json", 2*1024**2), ("prediction_result.json", 2*1024**2))})
        inventory = seals[name]["artifact_sha256"]
        require(set(inventory) == set(expected), "G03 complete payload membership differs")
        for relative, limit in expected.items():
            digest = inventory[relative]
            require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    "G03 declared payload hash differs")
            identities[relative] = file_identity(parent/relative, forbidden, limit, seen)
    return identities


def pinned(parent, relative, digest, forbidden, limit, identities):
    path = Path(parent)/relative
    require(file_identity(path, forbidden, limit) == identities[relative], "G03 frozen payload stat drift")
    return stable_bytes(path, limit, digest, expected_identity=tuple(identities[relative]))


def cost_binding(parent, spec, inventory, forbidden, identities):
    prefix = f"conditions/{spec['id']}/"
    for name, limit in (("cost.jsonl", 16*1024**2), ("initial_state.json", 2*1024**2),
                        ("prediction_result.json", 2*1024**2)):
        pinned(parent, prefix+name, inventory[prefix+name], forbidden, limit, identities)
    summary = protected.condition_cost(Path(parent), spec, inventory, forbidden)
    # The unchanged validator reopens metadata. Close its entire read interval
    # against the prepared pins before allowing any annotation access.
    for name, limit in (("cost.jsonl", 16*1024**2), ("initial_state.json", 2*1024**2),
                        ("prediction_result.json", 2*1024**2)):
        pinned(parent, prefix+name, inventory[prefix+name], forbidden, limit, identities)
    raw, _ = pinned(parent, prefix+"cost.jsonl", inventory[prefix+"cost.jsonl"], forbidden, 16*1024**2, identities)
    rows = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        require(type(row["episode"]) is int and type(row["order_position"]) is int,
                "G03 integer episode/order required")
        digest = row["candidate_sha256"]
        require(digest is None if row["arm"] == "no_adapt" else
                type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "G03 original candidate identity digest differs")
    for episode in range((spec["images"]+spec["config"]["batch_size"]-1)//spec["config"]["batch_size"]):
        adapted = [row for row in rows[episode*3:(episode+1)*3] if row["arm"] != "no_adapt"]
        require(len(adapted) == 2 and adapted[0]["candidate_sha256"] == adapted[1]["candidate_sha256"],
                "G03 paired candidate population differs")
    return summary, rows


@contextmanager
def output_lines(path, budget):
    """Own an exclusive output FD, including persisted hash and successful close."""
    path = Path(path)
    def identity(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink, s.st_mode)
    with path.open("xb+") as stream:
        created, written, hasher = os.fstat(stream.fileno()), 0, hashlib.sha256()
        def observe():
            current = os.fstat(stream.fileno())
            require(identity(current) == identity(path.lstat()) and stat.S_ISREG(current.st_mode)
                    and current.st_nlink == 1 and path.resolve(strict=True) == path
                    and (created.st_dev, created.st_ino) == (current.st_dev, current.st_ino)
                    and current.st_size == written, "G03 counts FD/path/extent drift")
            return identity(current)
        previous = observe()
        def append(value, progress):
            nonlocal written, previous
            line = (json.dumps(value, allow_nan=False)+"\n").encode()
            require(len(line) <= LINE_CAP and type(budget["bytes"]) is int and budget["bytes"] >= 0
                    and budget["bytes"]+len(line) <= OUTPUT_CAP
                    and shutil.disk_usage(path.parent).free >= RESERVE+len(line), "G03 counts output resource bound")
            require(observe() == previous, "G03 counts changed before write")
            require(stream.write(line) == len(line), "Short G03 counts write")
            stream.flush()
            written += len(line)
            budget["bytes"] += len(line)
            hasher.update(line)
            previous = observe()
            progress()
            require(observe() == previous, "G03 counts changed during progress")
        yield append
        stream.flush()
        os.fsync(stream.fileno())
        require(observe() == previous, "G03 counts changed before persisted verification")
        stream.seek(0)
        persisted, total = hashlib.sha256(), 0
        while chunk := stream.read(min(64*1024, written-total+1)):
            total += len(chunk)
            require(total <= written <= OUTPUT_CAP, "G03 persisted output bound")
            persisted.update(chunk)
        require(total == written and persisted.digest() == hasher.digest() and observe() == previous,
                "G03 persisted counts bytes differ")
    require(identity(path.lstat()) == previous and path.resolve(strict=True) == path,
            "G03 counts changed during close")
    append.digest = hasher.hexdigest()


def score_condition(suite, bound, spec, budget, label_hashes, progress):
    parent, source = Path(bound["parent"]), bound["source"]
    # Full cross-condition preflight precedes even the first annotation open.
    identities = preflight_payloads(parent, bound["specs"], bound["seals"], source)
    require(identities == bound["payload_metadata"], "G03 graph changed after preparation")
    forbidden = data_exclusions(source)
    rows, batch = source["rows"][spec["input_key"]], spec["config"]["batch_size"]
    require(len(spec["vocabulary"]["classes"]) == 20 and spec["config"]["resize"] == [224, 224],
            "G03 original class/geometry differs")
    name, inventory = spec["id"], bound["seals"][spec["id"]]["artifact_sha256"]
    cost, raw_rows = cost_binding(parent, spec, inventory, forbidden, identities)
    values = np.zeros(((len(rows)+batch-1)//batch, 3, 2, 20), dtype=np.int64)
    ordered, path = [], Path(suite)/(name+".episode_counts.jsonl")
    with output_lines(path, budget) as append:
        for episode, start in enumerate(range(0, len(rows), batch)):
            current, ids = rows[start:start+batch], [r["official_id"] for r in rows[start:start+batch]]
            ordered.append(ids)
            by_arm = {r["arm"]: r for r in raw_rows[episode*3:(episode+1)*3]}
            for arm in ARMS:
                relative = f"conditions/{name}/episodes/{episode:06d}.{arm}.json"
                raw, _ = pinned(parent, relative, inventory[relative], forbidden, 8*1024**2, identities)
                saved = json.loads(raw)
                require(same_provenance(saved["ordered_ids"], ids) and saved["GT_opened"] is False
                        and all(same_provenance(saved[k], v) for k, v in by_arm[arm].items())
                        and type(saved["losses"]) is list and len(saved["losses"]) == saved["steps"]
                        and all(type(v) in (int, float) and math.isfinite(v) for v in saved["losses"]),
                        "G03 episode/cost/source identity differs")
            gt = []
            for index, row in enumerate(current, start):
                content, digest = annotation_bytes(row)
                labels = metric.decode_annotation(content, 20, spec["vocabulary"]["reduce_zero_label"])
                require(row["label"] not in label_hashes or label_hashes[row["label"]] == digest,
                        "G03 repeated annotation bytes changed")
                label_hashes[row["label"]] = digest
                gt.append(dict(source_id=row["official_id"], label=row["label"], sha256=digest))
                for a, arm in enumerate(ARMS):
                    relative = f"conditions/{name}/predictions/{arm}/{index:06d}.npy"
                    raw, _ = pinned(parent, relative, inventory[relative], forbidden, 2*1024**2, identities)
                    values[episode, a] += metric.counts(hard_array(raw, 20, (224, 224)), labels, 20)
            append(dict(episode=episode, ordered_ids=ids, labels=gt, I_U_by_arm=values[episode].tolist()),
                   lambda: progress(dict(phase="CPU_G03_score", condition=name, completed_sources=min(start+batch, len(rows)),
                                         total_sources=len(rows), GT_opened=True, pid=os.getpid())))
    return values, ordered, cost, path, append.digest


def load_reused_counts(path, digest, identity, ordered, classes=20):
    """Authenticate one compact S5 episode stream, never a per-image count file."""
    path = Path(path)
    require(classes == 20 and 0 < len(ordered) <= 1449 and all(type(ids) is list and ids for ids in ordered),
            "G03 reused original episode layout differs")
    require(file_identity(path, set(), REUSED_CAP) == list(identity), "G03 S5 frozen stream stat differs")
    before = path.lstat()
    require([before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns] == list(identity)
            and before.st_nlink == 1 and stat.S_ISREG(before.st_mode) and path.resolve(strict=True) == path,
            "G03 S5 stat changed between pin validation and open")
    def state(s): return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
    values = np.zeros((len(ordered), 3, 2, classes), dtype=np.int64)
    hasher, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        require(state(os.fstat(stream.fileno())) == state(before), "G03 S5 opened FD differs")
        for episode, ids in enumerate(ordered):
            line = stream.readline(LINE_CAP+1)
            total += len(line)
            require(0 < len(line) <= LINE_CAP and line.endswith(b"\n") and total <= REUSED_CAP,
                    "G03 S5 line resource bound differs")
            hasher.update(line)
            row = json.loads(line)
            require(type(row["episode"]) is int and row["episode"] == episode and row["ordered_ids"] == ids
                    and type(row["labels"]) is list and len(row["labels"]) == len(ids)
                    and [r["source_id"] for r in row["labels"]] == ids, "G03 S5 original episode pairing differs")
            counts = row["I_U_by_arm"]
            require(type(counts) is list and len(counts) == 3 and all(type(a) is list and len(a) == 2
                    and all(type(c) is list and len(c) == classes and all(type(v) is int and 0 <= v <= len(ids)*224**2 for v in c)
                            for c in a) for a in counts), "G03 S5 native integer I/U differs")
            array = np.asarray(counts, dtype=np.int64)
            require(bool((array[:, 0] <= array[:, 1]).all()), "G03 S5 intersection exceeds union")
            values[episode] = array
        require(stream.read(1) == b"" and total == before.st_size and hasher.hexdigest() == digest
                and state(os.fstat(stream.fileno())) == state(before), "G03 S5 stream hash/extent/FD drift")
    require(state(path.lstat()) == state(before) and path.resolve(strict=True) == path, "G03 S5 close identity drift")
    return values
