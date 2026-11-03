"""Bind M01 replay to completed prediction metadata without opening payloads.

This is not runtime admission: the controller must also validate retained
service identity, runtime/input inventories and per-readout payload hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .bootstrap import require


def bound_json(path, digest):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and not path.is_symlink() and path.is_file(), "Aliased replay metadata")
    before = path.stat()
    require(0 < before.st_size <= 32*1024**2, "Replay metadata size outside bound")
    content = path.read_bytes()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(path.stat()) == identity(before)
            and hashlib.sha256(content).hexdigest() == digest, "Replay metadata digest or identity differs")
    return json.loads(content)


def bind_parent(parent, prepared_sha256, barrier_sha256):
    """Return ordered case references; do not infer success from a stale state."""
    parent = Path(parent)
    require(parent.is_absolute() and parent.resolve(strict=True) == parent
            and not parent.is_symlink(), "Canonical replay parent required")
    require(not any((parent/name).exists() for name in ("FAILED.json", "CANCELLED.json")),
            "Failed or cancelled parent cannot admit replay")
    prepared = bound_json(parent/"SUITE_PREPARED.json", prepared_sha256)
    barrier = bound_json(parent/"ALL_RISK_PREDICTIONS_SEALED.json", barrier_sha256)
    require(prepared["root"] == str(parent) and prepared["schema_version"] == 1,
            "Parent suite identity differs")
    require(barrier["complete"] is True and barrier["gt_contents_opened"] is False
            and set(barrier["dataset_seals"]) == set(prepared["datasets"])
            == {"coco_object", "pc60"}, "Incomplete or wrong prediction barrier")
    bound_json(parent/"PROTOCOL.json", prepared["protocol_sha256"])
    datasets = {}
    for dataset, spec in prepared["datasets"].items():
        seal = bound_json(parent/dataset/"PREDICTIONS_SEALED.json", barrier["dataset_seals"][dataset])
        require(seal["complete"] is True and seal["gt_contents_opened"] is False
                and seal["suite_prepared_sha256"] == prepared_sha256
                and seal["dataset_spec"] == spec and seal["candidates"] == spec["candidate_count"],
                "Dataset prediction seal binding differs")
        panel = bound_json(parent/"panels"/f"{dataset}.panel.json", spec["panels_sha256"])
        episodes = bound_json(parent/"panels"/f"{dataset}.episodes.json", spec["episodes_sha256"])
        require(panel["candidate_count"] == len(panel["examples"]) == spec["candidate_count"] > 0
                and panel["config"] == spec["native_config"], "Frozen candidate panel differs")
        steps = spec["native_config"]["steps"]
        require(type(steps) is int and 1 <= steps <= 64, "Invalid native checkpoint count")
        saved_steps = sorted({0, 1, steps})
        groups, seen = {}, set()
        for example in panel["examples"]:
            key = (example["source_id"], example["candidate_id"])
            episode = example["episode"]
            require(key not in seen and type(episode) is int and episode >= 0
                    and str(episode) in episodes, "Duplicate or invalid candidate episode")
            seen.add(key)
            rows = episodes[str(episode)]
            local = example["candidate"]["image_idx"]
            require(type(local) is int and 0 <= local < len(rows)
                    and rows[local] == example["source_input"]
                    and rows[local]["official_id"] == example["source_id"]
                    and example["episode_ids"] == [r["official_id"] for r in rows],
                    "Replay source/candidate order differs")
            groups.setdefault(episode, []).append(example)
        require(set(episodes) == {str(e) for e in groups}
                and len(groups) == spec["episode_count"], "Replay episode coverage differs")
        cases, expected_paths = [], set()
        for episode, examples in groups.items():
            for example in examples:
                case = f"case_{len(cases):06d}"
                checkpoints = []
                for step in saved_steps:
                    labels = f"{case}/step_{step:02d}.labels.npy"
                    snapshot = f"{case}/step_{step:02d}.visual_ln.npz"
                    checkpoints.append(dict(step=step, labels=labels, snapshot=snapshot))
                    expected_paths.update((labels, snapshot))
                trajectory = f"{case}/trajectory.json"
                expected_paths.add(trajectory)
                cases.append(dict(case=case, episode=episode, source_id=example["source_id"],
                    candidate_id=example["candidate_id"], example=example,
                    trajectory=trajectory, checkpoints=checkpoints))
        require(set(seal["artifact_sha256"]) == expected_paths,
                "Sealed prediction inventory missing or unexpected checkpoint")
        require(all(isinstance(v, str) and len(v) == 64
                    and all(c in "0123456789abcdef" for c in v)
                    for v in seal["artifact_sha256"].values()), "Invalid sealed payload digest")
        datasets[dataset] = dict(spec=spec, episodes=episodes, cases=cases,
            artifact_sha256=seal["artifact_sha256"], seal_sha256=barrier["dataset_seals"][dataset])
    return dict(parent=str(parent), prepared_sha256=prepared_sha256, barrier_sha256=barrier_sha256,
        datasets=datasets, payloads_opened=False, labels_read=False, GPU_admitted=False)
