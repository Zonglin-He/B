"""Unfrozen G03 offline CPU scorer; implementation is not scoring admission.

Future prediction identity is supplied explicitly by the operator on every
phase. Only a fresh independently admitted scorer may open new GT. Synthetic
tests do not establish actual artifact, service, model or scientific closure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import threading
import time

import numpy as np
import torch

from . import corruption_curve_worker as output
from . import curve_analysis as analysis
from . import curve_prediction_worker as prediction
from . import curve_score_payload as payload
from . import generalization_score_worker as protected
from . import spatial_score_worker as shared
from .bootstrap import require
from .risk_score_worker import canonical_child, publish_seal, verify_audit_binding

ROOT = Path(__file__).resolve().parents[2]
BASE = protected.BASE
ANALYSIS = BASE/"G03_ANALYSIS_PROTOCOL_v1.json"
ANALYSIS_SHA = "7655bee2573e1c79b8a4856f9272ce9b45b5ccc025c1c09ca740f2d9b96cca86"
S5_ROOT = BASE/"G_offline_score_20261102_2321"
S5_PREPARED = "ad2804b18158528eefc11525d87d43c696b02ae57f3c2a94291663545d5b9569"
S5_INPUTS = "13f5c33c6377b15402959bc261b66e02b8151d55d2c6c9eca636bfd12b7e3e76"
S5_PROTOCOL = "988998d4fff15a4eb66580dafb7529053994df812929828c3f93c6db5f1468d4"
S5_SEAL = "9356fe54cb3eeeb94e0068f06593efbbc7194ff459581897bc685851b9210048"
S5_ADMISSION = "a61f64bac7df1fb42d4a1dcfb3e62c85d267db87711be2c5e3768313fb0f0c86"
S5_REPORT = "81f3465cf748467bae8caef28cf45734817fefceb8dbeccacd7b9a723181dd49"
S5_UNIT = "trace-generalization-score-20261102-2321.service"
S5_PID, S5_INVOCATION = 2094938, "bccd4c5d1f564e3380b3acadbd42bdd5"
RESERVE, OUTPUT_CAP = 16*1024**3, 256*1024**2
REUSED_CAP, LINE_CAP = 16*1024**2, 128*1024


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
            and threading.active_count() == 1, "G03 scoring requires serialized hidden uninitialized CUDA")


def operator_pins(pins):
    keys = {"root", "prepared_sha256", "unit", "pid", "invocation", "global_sha256",
            "started_sha256", "gate_sha256", "admission_sha256", "report_sha256"}
    require(type(pins) is dict and set(pins) == keys, "Explicit complete G03 prediction operator pins required")
    root = pins["root"]
    require(type(root) is str and PurePosixPath(root).is_absolute() and str(PurePosixPath(root)) == root
            and ".." not in PurePosixPath(root).parts and root != "/", "Canonical G03 parent pin required")
    require(type(pins["unit"]) is str and len(pins["unit"]) <= 128
            and re.fullmatch(r"trace-[a-z0-9][a-z0-9-]*\.service", pins["unit"])
            and type(pins["pid"]) is int and 0 < pins["pid"] < 2**31, "G03 service/PID pins differ")
    for key in keys-{"root", "unit", "pid"}:
        size = 32 if key == "invocation" else 64
        require(type(pins[key]) is str and len(pins[key]) == size
                and all(c in "0123456789abcdef" for c in pins[key]), "G03 identity/hash pin differs")
    return copy.deepcopy(pins)


def parent_terminal(pins):
    pins = operator_pins(pins)
    return prediction.service(pins["unit"], pins["pid"], pins["invocation"])


def reused_terminal():
    return prediction.service(S5_UNIT, S5_PID, S5_INVOCATION)


def role_roots(parent):
    return ((ROOT, True), (parent/"runtime", False), (S5_ROOT/"runtime", False),
            (protected.PARENT/"runtime", False), (prediction.INPUT_ROOT/"runtime", False),
            (prediction.INPUT_ROOT/"reference", False), (output.PARENT/"runtime", False),
            (output.PARENT/"reference", False))


def source_exclusions(source, parent, *, sources=True):
    _, forbidden = output.data_roles(source["native"])
    forbidden |= payload.data_exclusions(source)
    if sources:
        data = set(forbidden)
        for root, snapshot in role_roots(parent):
            forbidden.update(v[:2] for v in protected.source_members(root, data, snapshot_source=snapshot).values())
    return forbidden


def shared_order(source):
    anchor, clean = source["anchor"], source["native"]["rows"]["clean.voc20"]
    prediction.validate_specs(source["specs"], source["rows"], anchor)
    batch = anchor["config"]["batch_size"]
    require(type(batch) is int and 0 < batch <= 512 and anchor["config"]["resize"] == [224, 224]
            and len(anchor["vocabulary"]["classes"]) == 20 and len(clean) == 1449,
            "G03 original geometry/population differs")
    identity = ("official_id", "label", "label_metadata", "label_metadata_details")
    for spec in source["specs"]:
        rows = source["rows"][spec["input_key"]]
        require(len(rows) == len(clean) and all(all(prediction.curve.same_provenance(row[k], original[k])
                for k in identity) for row, original in zip(rows, clean, strict=True)),
                "G03 original source/annotation pairing differs")
    ids = [row["official_id"] for row in clean]
    require(all(type(value) is str and value for value in ids) and len(set(ids)) == len(ids),
            "G03 original source IDs differ")
    return [ids[start:start+batch] for start in range(0, len(ids), batch)]


def s5_specs(source):
    result = []
    for kind in analysis.CURVE_TYPES:
        spec = copy.deepcopy(source["anchor"])
        spec.update(id=f"naclip.ViT-L-14.voc20.{kind}.s5", input_key="corruption."+kind, corruption=kind, severity=5)
        result.append(spec)
    return result


def prediction_graph(parent, specs, forbidden):
    """Stat every expected new cell before even opening parent seal metadata."""
    identities, seen = {}, set()
    for spec in specs:
        name, images, batch = spec["id"], spec["images"], spec["config"]["batch_size"]
        require(type(name) is str and name and "/" not in name and name not in (".", ".."), "Invalid G03 condition ID")
        prefix = f"conditions/{name}/"
        expected = {prefix+f"predictions/{arm}/{i:06d}.npy": 2*1024**2
                    for arm in analysis.ARMS for i in range(images)}
        expected.update({prefix+f"episodes/{e:06d}.{arm}.json": 8*1024**2
                         for arm in analysis.ARMS for e in range((images+batch-1)//batch)})
        expected.update({prefix+n: limit for n, limit in (("cost.jsonl", 16*1024**2),
                         ("initial_state.json", 2*1024**2), ("prediction_result.json", 2*1024**2))})
        for relative, limit in expected.items():
            current = output.file_pin(parent/relative, forbidden, limit, single=True)[:4]
            require(tuple(current[:2]) not in seen and relative not in identities, "G03 payload role collision")
            seen.add(tuple(current[:2]))
            identities[relative] = current
    return identities


def reused_identities(source, forbidden):
    result, seen = {}, set()
    for spec in s5_specs(source):
        name = spec["id"]+".episode_counts.jsonl"
        current = output.file_pin(S5_ROOT/name, forbidden, REUSED_CAP, single=True)[:4]
        require(tuple(current[:2]) not in seen, "G03 reused S5 streams alias")
        seen.add(tuple(current[:2]))
        result[name] = current
    return result


def audit_binding(root, prepared_sha, barrier, forbidden, admission_sha, report_sha, permission):
    binding = barrier["audit_binding"]
    require(binding["admission_path"] == str(root/"ADMITTED.json")
            and binding["report_path"] == str(root/"INDEPENDENT_REPORT.md")
            and binding["admission_sha256"] == admission_sha and binding["report_sha256"] == report_sha,
            "G03 upstream audit paths/hashes differ")
    audit, _ = protected.metadata(root/"ADMITTED.json", admission_sha, forbidden, limit=4*1024**2)
    shared.bounded_content(root/"INDEPENDENT_REPORT.md", report_sha, forbidden, 4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == prepared_sha
            and audit.get("independent_reviewer") and audit.get(permission) is True,
            "G03 upstream permission differs")
    verify_audit_binding(binding)
    return audit


def reused_cost(spec, inventory, forbidden):
    paths = [(protected.PARENT/f"conditions/{spec['id']}/"/name, limit)
             for name, limit in (("cost.jsonl", 16*1024**2), ("initial_state.json", 2*1024**2),
                                 ("prediction_result.json", 2*1024**2))]
    before = [output.file_pin(path, forbidden, limit, single=True) for path, limit in paths]
    cost = protected.condition_cost(protected.PARENT, spec, inventory, forbidden)
    require(before == [output.file_pin(path, forbidden, limit, single=True) for path, limit in paths],
            "G03 reused cost/reset metadata changed during validation")
    return cost


def bind_reused(source, forbidden, identities):
    require(S5_ROOT.resolve(strict=True) == S5_ROOT and not any((S5_ROOT/n).exists()
            for n in ("FAILED.json", "CANCELLED.json")), "G03 reused scoring parent failed")
    prepared, _ = protected.metadata(S5_ROOT/"SUITE_PREPARED.json", S5_PREPARED, forbidden)
    old, _ = protected.metadata(S5_ROOT/"SCORE_INPUTS.json", S5_INPUTS, forbidden)
    protected.metadata(S5_ROOT/"PROTOCOL.json", S5_PROTOCOL, forbidden)
    require(prepared["root"] == str(S5_ROOT) and prepared["schema_version"] == 1
            and prepared["inputs_sha256"] == S5_INPUTS and prepared["protocol_sha256"] == S5_PROTOCOL
            and prepared["parent_prepared_sha256"] == protected.PARENT_SHA
            and old["parent"] == str(protected.PARENT) and old["prepared_sha256"] == protected.PARENT_SHA,
            "G03 reused score/source parent identity differs")
    native = {k: v for k, v in source["native"]["rows"].items() if not k.startswith("S5.")}
    require(prediction.curve.same_provenance(old["inputs"], native), "G03 old/new original input provenance differs")
    protected.prediction.validate_inputs(old["inputs"], old["specs"])
    seal, _ = protected.metadata(S5_ROOT/"ANALYSIS_SEALED.json", S5_SEAL, forbidden)
    require(seal["complete"] is True and seal["GT_opened"] is True and seal["new_GPU"] is False
            and seal["suite_prepared_sha256"] == S5_PREPARED and seal["conditions"] == 19 and seal["cells"] == 57
            and seal["image_method_evaluations"] == 104214 and seal["parent_barrier_sha256"] == old["barrier_sha256"]
            and seal["prospective_analysis_sha256"] == protected.ANALYSIS_SHA,
            "G03 reused score analysis seal differs")
    audit = audit_binding(S5_ROOT, S5_PREPARED, seal, forbidden, S5_ADMISSION, S5_REPORT, "CPU_GT_scoring_allowed")
    require(audit.get("G_primary_GT_scoring_allowed") is True, "G03 reused score lacked G-primary GT permission")
    start, _ = protected.metadata(S5_ROOT/"STARTED.json", seal["started_sha256"], forbidden)
    require(type(start["pid"]) is int and start["pid"] == S5_PID and start["suite_prepared_sha256"] == S5_PREPARED
            and start["audit"] == seal["audit_binding"] and start["parent_barrier_sha256"] == old["barrier_sha256"]
            and start["GT_opened"] is False, "G03 reused score STARTED differs")
    data = source_exclusions(source, Path(source["score_parent"]), sources=False)
    protected.guarded_inventory(S5_ROOT/"runtime", prepared["runtime_hashes"], data)
    gp, _ = protected.metadata(protected.PARENT/"SUITE_PREPARED.json", protected.PARENT_SHA, forbidden,
        expected_identity=protected.PARENT_METADATA_IDENTITIES["SUITE_PREPARED.json"])
    specs, _ = protected.metadata(protected.PARENT/"CONDITIONS.json", gp["conditions_sha256"], forbidden)
    require(prediction.curve.same_provenance(specs, old["specs"]), "G03 sealed G1954 condition metadata differs")
    gseal, _ = protected.metadata(protected.PARENT/"ALL_G_PREDICTIONS_SEALED.json", old["barrier_sha256"], forbidden)
    require(gseal["complete"] is True and gseal["GT_opened"] is False and gseal["cells"] == 57
            and gseal["image_method_evaluations"] == 104214 and gseal["suite_prepared_sha256"] == protected.PARENT_SHA
            and set(gseal["condition_seals"]) == {s["id"] for s in specs}, "G03 G1954 global provenance differs")
    result = {}
    for expected in s5_specs(source):
        matches = [s for s in specs if s["id"] == expected["id"]]
        require(len(matches) == 1 and prediction.curve.same_provenance(matches[0], expected),
                "G03 S5 config/arms/readout/vocabulary/episode grouping differs")
        spec = matches[0]
        child, _ = protected.metadata(protected.PARENT/"conditions"/spec["id"]/"PREDICTIONS_SEALED.json",
                                     gseal["condition_seals"][spec["id"]], forbidden)
        require(prediction.curve.same_provenance(child, old["seals"][spec["id"]])
                and child["complete"] is True and child["GT_opened"] is False
                and child["condition"] == spec and child["suite_prepared_sha256"] == protected.PARENT_SHA,
                "G03 selected S5 prediction seal differs")
        source_rows = old["inputs"][spec["input_key"]]
        prediction.curve.validate_S5_reuse({s["corruption"]: old["inputs"][s["input_key"]]
            for s in s5_specs(source)}, source["native"]["rows"]["clean.voc20"])
        require(len(source_rows) == spec["images"], "G03 reused S5 source cardinality differs")
        name = spec["id"]+".episode_counts.jsonl"
        require(prediction.curve.digest(seal["artifact_sha256"][name]), "G03 compact S5 count hash missing")
        result[spec["corruption"]] = dict(spec=spec, path=str(S5_ROOT/name), sha256=seal["artifact_sha256"][name],
            identity=identities[name], source_prepared_sha256=S5_PREPARED, source_analysis_sha256=S5_SEAL,
            cost=reused_cost(spec, child["artifact_sha256"], forbidden), resources=child.get("resource_observation"))
    return result


def bind_parent(pins):
    pins = operator_pins(pins)
    terminal = parent_terminal(pins)
    old_terminal = reused_terminal()
    source = prediction.source_inputs()  # This constructor deliberately installs no GT-denial hook.
    parent = Path(pins["root"])
    require(parent.resolve(strict=True) == parent and parent != S5_ROOT and not any((parent/n).exists()
            for n in ("FAILED.json", "CANCELLED.json")), "Unsuccessful/colliding G03 prediction parent")
    ordered = shared_order(source)
    data = source_exclusions(source, parent, sources=False)
    forbidden = source_exclusions(source, parent)
    graph = prediction_graph(parent, source["specs"], forbidden)
    metadata_forbidden = forbidden | {tuple(v[:2]) for v in graph.values()}
    reused = reused_identities(source, metadata_forbidden)
    metadata_forbidden |= {tuple(v[:2]) for v in reused.values()}
    prepared, _ = protected.metadata(parent/"SUITE_PREPARED.json", pins["prepared_sha256"], metadata_forbidden)
    saved, _ = protected.metadata(parent/"INPUTS.json", prepared["inputs_sha256"], metadata_forbidden)
    require(prepared["schema_version"] == 1 and prepared["root"] == str(parent)
            and prediction.curve.same_provenance(saved, source), "G03 exact reconstructed source differs")
    prior = prediction.prerequisite_bindings(prepared["prior_V01"], metadata_forbidden)
    require(prediction.curve.same_provenance(prior, prepared["prior_binding"]), "G03 prediction prerequisite drift")
    plan, _ = protected.metadata(parent/"PROTOCOL.json", prepared["protocol_sha256"], metadata_forbidden)
    require(prediction.curve.same_provenance(plan, prediction.protocol(source, prepared["prior_V01"])),
            "G03 prediction protocol differs")
    require(len(source["parent_native_hashes"]) == 68, "G03 exact native68 membership differs")
    shared.native_binding(source["parent_native_hashes"], data)
    protected.guarded_inventory(parent/"runtime", prepared["runtime_hashes"], data)
    barrier, _ = protected.metadata(parent/"ALL_G_PREDICTIONS_SEALED.json", pins["global_sha256"], metadata_forbidden)
    require(barrier["complete"] is True and barrier["GT_opened"] is False
            and barrier["suite_prepared_sha256"] == pins["prepared_sha256"] and barrier["cells"] == 30
            and barrier["image_method_evaluations"] == 43470 and barrier["started_sha256"] == pins["started_sha256"]
            and barrier["gate_binding"]["barrier_sha256"] == pins["gate_sha256"]
            and set(barrier["condition_seals"]) == {s["id"] for s in source["specs"]}
            and barrier["input_seal_sha256"] == source["input_seal_sha256"]
            and prediction.curve.same_provenance(barrier["prior_binding"], prior), "G03 full30-cell global seal differs")
    admitted = audit_binding(parent, pins["prepared_sha256"], barrier, metadata_forbidden,
                            pins["admission_sha256"], pins["report_sha256"], "G03_predictions_allowed")
    start, _ = protected.metadata(parent/"STARTED.json", pins["started_sha256"], metadata_forbidden)
    require(admitted.get("GT_scoring_allowed") is False and start["suite_prepared_sha256"] == pins["prepared_sha256"]
            and type(start["pid"]) is int and start["pid"] == pins["pid"] and start["GT_opened"] is False
            and start["audit"] == barrier["audit_binding"] and prediction.curve.same_provenance(start["prior"], prior),
            "G03 prediction STARTED/permission differs")
    prediction.verify_gates(parent, pins["prepared_sha256"], source, barrier["gate_binding"])
    seals = {}
    for spec in source["specs"]:
        child, _ = protected.metadata(parent/"conditions"/spec["id"]/"PREDICTIONS_SEALED.json",
                                     barrier["condition_seals"][spec["id"]], metadata_forbidden)
        require(child["complete"] is True and child["GT_opened"] is False and child["condition"] == spec
                and child["suite_prepared_sha256"] == pins["prepared_sha256"] and child["images"] == 1449
                and child["image_method_evaluations"] == 4347, "G03 complete condition seal differs")
        seals[spec["id"]] = child
    require(payload.preflight_payloads(parent, source["specs"], seals, source) == graph,
            "G03 complete new payload graph drift")
    costs = {s["id"]: payload.cost_binding(parent, s, seals[s["id"]]["artifact_sha256"], forbidden, graph)[0]
             for s in source["specs"]}
    source_for_reuse = dict(source, score_parent=str(parent))
    reused_binding = bind_reused(source_for_reuse, metadata_forbidden, reused)
    prospective, _ = protected.metadata(ANALYSIS, ANALYSIS_SHA, metadata_forbidden)
    require(prediction.curve.same_provenance(prospective, analysis.protocol()), "G03 prospective analysis protocol differs")
    require(parent_terminal(pins) == terminal and reused_terminal() == old_terminal, "G03 terminal identity drift")
    return dict(parent=str(parent), prepared_sha256=pins["prepared_sha256"], parent_pins=pins, terminal=terminal,
        reused_terminal=old_terminal, source=source, specs=source["specs"], seals=seals, payload_metadata=graph,
        ordered=ordered, costs=costs, reused=reused_binding, analysis_sha256=ANALYSIS_SHA)


def protocol(pins):
    return dict(experiment="G03_offline_CPU_score", parent_pins=operator_pins(pins), analysis_sha256=ANALYSIS_SHA,
        estimator=analysis.protocol(), geometry="Palette indices then OpenCV INTER_NEAREST224, not native PIL",
        S5="Five sealed G2321 compact episode streams only; no S5 predictions/GT reread or old CI overwrite",
        permission="New exact CPU_GT AND G03_GT after all30 predictions and same retained successful exit",
        limits=dict(output_cap_bytes=OUTPUT_CAP, shared_reserve_bytes=RESERVE), GPU_allowed=False,
        failures="Exclusive owned STARTED; protected progress; no resume/retry; global publication is final IO")


def prepare(suite, pins):
    cpu_only()
    bound = bind_parent(pins)
    suite = Path(suite).absolute()
    protected_roots = [r for r, _ in role_roots(Path(bound["parent"]))] + [Path(bound["parent"]), S5_ROOT]
    require(suite.resolve() == suite and not suite.exists() and not any(suite.is_relative_to(r) or r.is_relative_to(suite)
            for r in protected_roots), "Fresh disjoint canonical G03 score suite required")
    require(shutil.disk_usage(suite.parent).free >= RESERVE+OUTPUT_CAP, "G03 score prepare headroom insufficient")
    suite.mkdir()
    inputs_sha = protected.write_prepared_json(suite/"SCORE_INPUTS.json", bound)
    plan_sha = protected.write_prepared_json(suite/"PROTOCOL.json", protocol(pins))
    inventory = protected.guarded_snapshot(ROOT, suite/"runtime", source_exclusions(bound["source"], Path(bound["parent"]), sources=False))
    return protected.write_prepared_json(suite/"SUITE_PREPARED.json", dict(schema_version=1, root=str(suite),
        runtime_hashes=inventory, inputs_sha256=inputs_sha, protocol_sha256=plan_sha,
        parent_pins=operator_pins(pins), GT_opened=False, CPU_scoring_started=False))


def admission(suite, expected_sha, pins):
    cpu_only()
    bound = bind_parent(pins)
    suite = Path(suite).resolve(strict=True)
    forbidden = source_exclusions(bound["source"], Path(bound["parent"]))
    forbidden |= {tuple(v[:2]) for v in bound["payload_metadata"].values()}
    forbidden |= {tuple(v["identity"][:2]) for v in bound["reused"].values()}
    data = source_exclusions(bound["source"], Path(bound["parent"]), sources=False)
    forbidden |= {v[:2] for v in protected.source_members(suite/"runtime", data).values()}
    prepared, _ = protected.metadata(suite/"SUITE_PREPARED.json", expected_sha, forbidden)
    require(prepared["schema_version"] == 1 and prepared["root"] == str(suite)
            and prediction.curve.same_provenance(prepared["parent_pins"], pins)
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong exact G03 scoring runtime")
    inputs, _ = protected.metadata(suite/"SCORE_INPUTS.json", prepared["inputs_sha256"], forbidden)
    plan, _ = protected.metadata(suite/"PROTOCOL.json", prepared["protocol_sha256"], forbidden)
    require(prediction.curve.same_provenance(inputs, bound) and prediction.curve.same_provenance(plan, protocol(pins)),
            "G03 scoring source/protocol drift")
    protected.guarded_inventory(suite/"runtime", prepared["runtime_hashes"], data)
    prediction.general.runtime_imports(suite/"runtime")
    audit, audit_sha = protected.metadata(suite/"ADMITTED.json", forbidden=forbidden, limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("CPU_GT_scoring_allowed") is True
            and audit.get("G03_GT_scoring_allowed") is True and audit.get("GPU_allowed") is False,
            "New exact independent G03 CPU_GT AND G03_GT admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong G03 scoring report path")
    shared.bounded_content(report, audit["report_sha256"], forbidden, 4*1024**2)
    ids = [output.file_pin(p, forbidden, 4*1024**2, single=True)[:4] for p in (suite/"ADMITTED.json", report)]
    binding = dict(admission_path=str(suite/"ADMITTED.json"), admission_sha256=audit_sha, admission_identity=ids[0],
        report_path=str(report), report_sha256=audit["report_sha256"], report_identity=ids[1], independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite, prepared, inputs, binding


def reused_labels(binding, ordered, clean, hashes):
    """Stream the same compact file separately to authenticate label receipts."""
    path, identity = Path(binding["path"]), binding["identity"]
    require(output.file_pin(path, set(), REUSED_CAP, single=True)[:4] == identity, "G03 S5 receipt stat drift")
    before = output.identity(path.lstat())
    require(before[:4] == identity and before[5] == 1 and stat.S_ISREG(before[6])
            and path.resolve(strict=True) == path, "G03 S5 receipt second stat differs from supplied identity")
    by_id = {row["official_id"]: row["label"] for row in clean}
    hasher, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        require(output.identity(os.fstat(stream.fileno())) == before, "G03 S5 receipt FD differs")
        for episode, ids in enumerate(ordered):
            line = stream.readline(LINE_CAP+1)
            total += len(line)
            require(0 < len(line) <= LINE_CAP and line.endswith(b"\n") and total <= REUSED_CAP, "G03 S5 receipt line bound")
            hasher.update(line)
            row = json.loads(line)
            require(type(row["episode"]) is int and row["episode"] == episode and row["ordered_ids"] == ids
                    and type(row["labels"]) is list and len(row["labels"]) == len(ids), "G03 S5 receipt episode differs")
            for source_id, label in zip(ids, row["labels"], strict=True):
                require(set(label) == {"source_id", "label", "sha256"} and label["source_id"] == source_id
                        and label["label"] == by_id[source_id] and prediction.curve.digest(label["sha256"]),
                        "G03 S5 label path/hash/source identity differs")
                name, digest = label["label"], label["sha256"]
                require(name not in hashes or hashes[name] == digest, "G03 S5 label hashes disagree across five streams")
                hashes[name] = digest
        require(stream.read(1) == b"" and total == before[2] and hasher.hexdigest() == binding["sha256"]
                and output.identity(os.fstat(stream.fileno())) == before, "G03 S5 receipt hash/FD/extent differs")
    require(output.identity(path.lstat()) == before, "G03 S5 receipt close drift")


def write_output(suite, name, content, budget, on_create=lambda: None):
    require(type(budget.get("bytes")) is int and 0 <= budget["bytes"] <= OUTPUT_CAP
            and budget["bytes"]+len(content) <= OUTPUT_CAP and shutil.disk_usage(suite).free >= RESERVE+len(content),
            "G03 score output/reserve cap exceeded")
    return output.write_output(suite, name, content, budget, on_create)


def execute(suite, bound):
    parent, source = Path(bound["parent"]), bound["source"]
    require(payload.preflight_payloads(parent, bound["specs"], bound["seals"], source) == bound["payload_metadata"],
            "G03 complete payload graph changed before GT")
    forbidden = source_exclusions(source, parent) | {tuple(v[:2]) for v in bound["payload_metadata"].values()}
    require(reused_identities(source, forbidden) == {Path(v["path"]).name: v["identity"] for v in bound["reused"].values()},
            "G03 all five reused streams changed before GT")
    ordered, labels, budget, artifacts, costs = bound["ordered"], {}, dict(bytes=0), {}, {}
    grid = np.zeros((len(ordered), 5, 3, 3, 2, 20), dtype=np.int64)
    for t, kind in enumerate(analysis.CURVE_TYPES):
        binding = bound["reused"][kind]
        reused_labels(binding, ordered, source["native"]["rows"]["clean.voc20"], labels)
        grid[:, t, 2] = payload.load_reused_counts(Path(binding["path"]), binding["sha256"], binding["identity"], ordered, classes=20)
        costs[binding["spec"]["id"]] = dict(prediction_cost=binding["cost"], resources=binding["resources"],
                                           provenance="Reused sealed G1954 S5 metadata; no new inference")
    for spec in bound["specs"]:
        counts, ids, cost, path, digest = payload.score_condition(suite, bound, spec, budget, labels,
            lambda row: shared.write_json(suite/"queue_state.json", row, replace=True))
        require(ids == ordered and counts.shape == (len(ordered), 3, 2, 20) and counts.dtype == np.int64
                and Path(path) == suite/(spec["id"]+".episode_counts.jsonl") and prediction.curve.digest(digest),
                "G03 scored original episode/schema differs")
        grid[:, analysis.CURVE_TYPES.index(spec["corruption"]), analysis.SEVERITIES.index(spec["severity"])] = counts
        artifacts[Path(path).name] = digest
        costs[spec["id"]] = dict(prediction_cost=cost, resources=bound["seals"][spec["id"]].get("resource_observation"),
                                 provenance="New sealed G03 s1/s3 prediction metadata")
    result, _ = analysis.paired_summary(grid, replicates=10000, chunk=32)
    result.update(ordered_source_episodes=ordered, costs=costs, analysis_sha256=ANALYSIS_SHA,
                  original_source_images=1449, source="L_local", S5_analysis_sha256=S5_SEAL,
                  cost_scope="Raw inherited model time/work and resources are descriptive; no equal-budget claim; offline scoring overhead separate")
    for name, value in (("CURVE_RESULTS.json", result), ("ANNOTATIONS_READ.json", labels), ("REUSED_S5_PROVENANCE.json", bound["reused"])):
        artifacts[name], _ = write_output(suite, name, output.encoded(value), budget)
    identities = {name: output.file_pin(suite/name, forbidden, 64*1024**2, single=True)[:4] for name in artifacts}
    return dict(complete=True, conditions=15, cells=45, new_prediction_conditions=10, new_prediction_cells=30,
        image_method_evaluations=43470, reused_image_method_evaluations=21735, GT_opened=True, new_GPU=False,
        artifact_sha256=artifacts, artifact_identities=identities, output_bytes=budget["bytes"], scientific_completion=False)


def run(suite, expected_sha, pins):
    suite, prepared, bound, audit = admission(suite, expected_sha, pins)
    require(not any((suite/n).exists() for n in ("STARTED.json", "FAILED.json", "ANALYSIS_SEALED.json")), "No G03 score resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP, "G03 score run headroom insufficient")
    owned = []
    try:
        verify_audit_binding(audit)
        started, _ = write_output(suite, "STARTED.json", output.encoded(dict(suite_prepared_sha256=expected_sha,
            audit=audit, pid=os.getpid(), parent_pins=pins, GT_opened=False, automatic_retry=False)),
            dict(bytes=0), lambda: owned.append(True))
        begin = time.perf_counter()
        result = execute(suite, bound)
        require(result["conditions"] == 15 and result["cells"] == 45 and result["new_prediction_cells"] == 30
                and result["image_method_evaluations"] == 43470, "Incomplete G03 scoring population")
        shared.write_json(suite/"queue_state.json", dict(phase="G03_analysis_complete_awaiting_seal_and_exit",
            GT_opened=True, pid=os.getpid()), replace=True)
        require(prediction.curve.same_provenance(bind_parent(pins), bound), "G03 final parent/source/stat drift")
        forbidden = source_exclusions(bound["source"], Path(bound["parent"]))
        forbidden |= {tuple(v[:2]) for v in bound["payload_metadata"].values()}
        forbidden |= {tuple(v["identity"][:2]) for v in bound["reused"].values()}
        protected.guarded_inventory(suite/"runtime", prepared["runtime_hashes"], source_exclusions(bound["source"], Path(bound["parent"]), sources=False))
        for name, digest in (("SUITE_PREPARED.json", expected_sha), ("SCORE_INPUTS.json", prepared["inputs_sha256"]),
                             ("PROTOCOL.json", prepared["protocol_sha256"]), ("STARTED.json", started)):
            protected.metadata(suite/name, digest, forbidden)
        for name, digest in result["artifact_sha256"].items():
            path = canonical_child(suite, name)
            require(output.file_pin(path, forbidden, 64*1024**2, single=True)[:4] == result["artifact_identities"][name],
                    "G03 final output stat drift")
            payload.stable_bytes(path, 64*1024**2, digest,
                                 expected_identity=tuple(result["artifact_identities"][name]))
            require(output.file_pin(path, forbidden, 64*1024**2, single=True)[:4] == result["artifact_identities"][name],
                    "G03 final output identity changed after bound hash")
        prediction.general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        cpu_only()
        publish_seal(suite/"ANALYSIS_SEALED.json", dict(**result, suite_prepared_sha256=expected_sha,
            parent_pins=pins, reused_analysis_sha256=S5_SEAL, prospective_analysis_sha256=ANALYSIS_SHA,
            audit_binding=audit, started_sha256=started, elapsed_seconds=time.perf_counter()-begin))
    except BaseException as error:
        if owned:
            protected.write_prepared_json(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate-admission", "run"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256")
    parser.add_argument("--parent-pins", required=True, help="Explicit operator JSON root/prepared/unit/PID/invocation/start/gate/global/audit/report pins")
    args = parser.parse_args()
    pins = operator_pins(json.loads(args.parent_pins))
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite, pins), GT_opened=False, GPU_started=False))
    else:
        require(args.suite_sha256, "Exact G03 scoring prepared hash required")
        if args.phase == "run":
            run(args.suite, args.suite_sha256, pins)
        else:
            admission(args.suite, args.suite_sha256, pins)
            print(dict(admission_valid=True, GT_opened=False, CUDA_initialized=torch.cuda.is_initialized()))
