"""Render fifty-card progress only from small, hash-checked evidence receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .bootstrap import read_json, require
from .intake import sha256, write_new


def checked_seal(directory, filename):
    directory = Path(directory).resolve(strict=True)
    seal = read_json(directory / filename)
    require(seal["complete"] is True, "Incomplete evidence seal")
    for name, expected in seal["artifact_sha256"].items():
        path = directory / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.resolve() == path and path.is_relative_to(directory), "Unsafe evidence path")
        require(sha256(path) == expected, "Evidence artifact drift: " + str(path))
    return dict(path=str(directory / filename), sha256=sha256(directory / filename), scope=seal.get("scope"))


def apply_scope(statuses, scope):
    active = scope.get("active_experiments")
    require(type(active) is list and len(active) > 0
            and all(type(key) is str and key in statuses for key in active)
            and len(active) == len(set(active)), "Invalid active experiment scope")
    require(scope.get("unlisted_policy") == "deferred_not_completed",
            "Deferred experiments cannot be counted as completed")
    selected = set(active)
    for key, row in statuses.items():
        row["required"] = key in selected
        row["execution_policy"] = "required" if key in selected else "deferred"
    return [row for row in statuses.values() if row["required"]]


def generate(root, output, scope_path=None):
    root = Path(root).resolve(strict=True)
    checked_seal(root / "intake", "INTAKE_SEALED.json")
    registry = read_json(root / "intake/registry.json")
    cards = registry["experiments"]
    require([r["id"] for r in cards] == [f"E{i:02d}" for i in range(1, 51)]
            and all(r["required"] for r in cards), "Mandatory fifty-card scope changed")
    statuses = {r["id"]: dict(id=r["id"], title=r["title"], required=True,
        status="not_started", evidence=[], remaining="Implement, freeze and execute the supplied card; no inferred completion.")
        for r in cards}
    e01 = root / 'E01_E40_protocol_v1'
    if (e01 / 'PREPARED.json').exists():
        proof = read_json(e01 / 'PREPARED.json')
        require(proof['executable_queue'] is False and proof['trials_executed'] == 0,
                'Protocol preparation cannot imply executed trials')
        for name, expected in proof['artifact_sha256'].items():
            path = e01 / name
            require(path.resolve() == path and path.is_relative_to(e01)
                    and sha256(path) == expected, 'Frozen E01 protocol drift')
        receipt = dict(path=str(e01 / 'PREPARED.json'), sha256=sha256(e01 / 'PREPARED.json'),
                       scope='225 equal-opportunity calibration trials frozen; none executed')
        for key in ('E01', 'E40'):
            statuses[key].update(status='prepared', evidence=[receipt],
                remaining='Worker/controller and real gates/calibration/selection/full evaluation pending; no calibration outcomes yet.')
    if (root / "E02_cache_v1/CACHE_AUDIT_SEALED.json").exists():
        receipt = checked_seal(root / "E02_cache_v1", "CACHE_AUDIT_SEALED.json")
        statuses["E02"].update(status="partial", evidence=[receipt],
            remaining="Cached four-arm assignments/EPS coverage audited on full Object/PC60; actual adaptation/predictions/mIoU/gates pending.")
    if (root / "E08_existing_v1/ANALYSIS_SEALED.json").exists():
        receipt = checked_seal(root / "E08_existing_v1", "ANALYSIS_SEALED.json")
        result = read_json(root / "E08_existing_v1/result.json")
        require(len(result["rows"]) == 10 and result["existing_prediction_contrasts_complete"] is True,
                "E08 existing contrasts incomplete")
        statuses["E08"].update(status="partial", evidence=[receipt],
            remaining="Ten existing contrasts complete; apply estimator to new E02 contrasts once sealed.")
    if (root / "E06_E31_inputs_v1/INPUTS_SEALED.json").exists():
        receipt = checked_seal(root / "E06_E31_inputs_v1", "INPUTS_SEALED.json")
        for key in ("E06", "E31"):
            statuses[key].update(status="partial", evidence=[receipt],
                remaining="Input identity/group audit only; official provenance, test-only inference and scoring pending.")
    if (root / "E30_algebra_v1/CPU_ALGEBRA_SEALED.json").exists():
        receipt = checked_seal(root / "E30_algebra_v1", "CPU_ALGEBRA_SEALED.json")
        statuses["E30"].update(status="partial", evidence=[receipt],
            remaining="Ten CPU algebra checks passed; real-model identity support and final-label parity pending.")
    selected = list(statuses.values())
    scope_receipt = None
    if scope_path is not None:
        scope_path = Path(scope_path).resolve(strict=True)
        scope = read_json(scope_path)
        selected = apply_scope(statuses, scope)
        scope_receipt = dict(path=str(scope_path), sha256=sha256(scope_path), scope_id=scope["scope_id"])
    report = dict(updated_at_utc=datetime.now(timezone.utc).isoformat(), required=len(selected),
        historical_intake_cards=50, deferred=50-len(selected), scope_override=scope_receipt,
        fully_completed=sum(r["status"] == "completed" for r in selected),
        partially_completed=sum(r["status"] == "partial" for r in selected),
        protocols_prepared=sum(r['status'] == 'prepared' for r in selected),
        new_gpu_experiments_started=0, scientific_completion=False,
        scope="Evidence snapshot at generation, not a process monitor", experiments=list(statuses.values()))
    write_new(output, report)
    print(dict(output=str(output), fully_completed=report["fully_completed"], partial=report["partially_completed"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=Path, help="Explicit later human scope; intake remains immutable")
    args = parser.parse_args()
    generate(args.root, args.output, args.scope)
