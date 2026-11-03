"""M12 protected CPU case controller, unfrozen and not execution permission.

Only exact new qualitative GT/RGB admission permits selected payload reads.
There is no model, inference, upload, publication or paper-editing entry point.
External caps and a manual single attempt remain future operator obligations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import cv2
import torch

from . import case_payload as core
from . import corruption_curve_worker as output
from . import generalization_score_worker as protected
from . import legacy_score_payload as annotations
from . import spatial_score_worker as parent
from .bootstrap import require
from .corruption_curve import digest, same_provenance
from .risk_score_payload import stable_bytes
from .risk_score_worker import canonical_child, publish_seal, verify_audit_binding

ROOT = Path(__file__).resolve().parents[2]
BASE = parent.BASE
SCOPE = BASE/"SCOPE_v4.json"
CASE_PROTOCOL = BASE/"M12_CASE_PROTOCOL_v1.json"
CASE_PROTOCOL_SHA = "397edb0aa6896d0976aad2fabd80c2e1c2498eaabb3284aca39ff555e88d2ed4"
DATASETS = ("coco_object", "pc60")
MODES = ("native", "translated", "global", "confidence")
RESERVE, OUTPUT_CAP = 16*1024**3, 256*1024**2
RESOURCES = dict(MemoryMax=4*1024**3, MemorySwapMax=512*1024**2, CPUQuota_percent=100,
    Nice=10, IO="idle", threads=[1, 1, 1], OOMPolicy="stop", Restart="no", RemainAfterExit=True,
    KillMode="control-group", CUDA_VISIBLE_DEVICES="")


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
            and threading.active_count() == 1 and cv2.getNumThreads() == torch.get_num_threads()
            == torch.get_num_interop_threads() == 1, "M12 requires CPU-only serialized one-thread runtime")


def source_roles(source):
    annotations.preflight_annotations(source)
    labels = {(value["device"], value["inode"]) for value in source["annotation_metadata"].values()}
    rgb = set()
    for rows in source["rows"].values():
        for row in rows:
            current = output.file_pin(Path(row["image"]), labels, 32*1024**2)
            if "image_pins" in source:
                require(current == source["image_pins"][row["image"]], "M12 all-G RGB stat drift")
            rgb.add(tuple(current[:2]))
    require(not labels.intersection(rgb), "M12 RGB/GT roles overlap")
    return labels, rgb


def role_roots(source):
    roots = [(ROOT, True), (parent.PARENT/"runtime", False), (protected.PARENT/"runtime", False)]
    for root in (parent.PARENT/"reference", protected.PARENT/"reference"):
        if root.exists():
            roots.append((root, False))
    return roots


def source_exclusions(source, *, include_sources=True, data_roles=None):
    labels, rgb = source_roles(source if data_roles is None else data_roles)
    data, sources = labels | rgb, set()
    if include_sources:
        for root, snapshot in role_roots(source):
            sources.update(value[:2] for value in protected.source_members(root, data, snapshot_source=snapshot).values())
    return data | sources


def trusted_source():
    """Only the two operator-stat-pinned G documents precede the data registry."""
    root = protected.PARENT
    require(root.is_absolute() and root.resolve(strict=True) == root, "Noncanonical trusted G root")
    pins = {name: output.file_pin(root/name, set(), 128*1024**2, single=True)
            for name in ("SUITE_PREPARED.json", "INPUTS.json")}
    for name, pin in pins.items():
        require(tuple(pin[:4]) == tuple(protected.PARENT_METADATA_IDENTITIES[name]), "M12 G bootstrap stat pin differs")
    prepared, _ = protected.metadata(root/"SUITE_PREPARED.json", protected.PARENT_SHA,
        expected_identity=protected.PARENT_METADATA_IDENTITIES["SUITE_PREPARED.json"])
    inputs, _ = protected.metadata(root/"INPUTS.json", prepared["inputs_sha256"],
        expected_identity=protected.PARENT_METADATA_IDENTITIES["INPUTS.json"])
    for name, pin in pins.items():
        require(output.file_pin(root/name, set(), 128*1024**2, single=True) == pin, "M12 G bootstrap changed during read")
    require(prepared["schema_version"] == 1 and prepared["root"] == str(root)
            and type(inputs) is dict and inputs and all(type(rows) is list and rows for rows in inputs.values()),
            "M12 trusted G source schema differs")
    registry = {}
    for rows in inputs.values():
        for row in rows:
            require(row["label"] not in registry or same_provenance(registry[row["label"]], row["label_metadata_details"]),
                    "M12 G repeated annotation identity differs")
            registry[row["label"]] = copy.deepcopy(row["label_metadata_details"])
    roles = dict(rows=inputs, annotation_metadata=registry)
    labels, _ = source_roles(roles)  # All G clean/VOC/corrupted rows, stat only.
    roles["image_pins"] = {row["image"]: output.file_pin(Path(row["image"]), labels, 32*1024**2)
                           for rows in inputs.values() for row in rows}
    data = source_exclusions(roles, include_sources=False)
    forbidden = source_exclusions(roles)  # Enumerate every source/ref role before any further read.
    for name, pin in pins.items():
        require(tuple(pin[:2]) not in forbidden, "M12 G bootstrap aliases data/source/reference")
    specs, _ = protected.metadata(root/"CONDITIONS.json", prepared["conditions_sha256"], forbidden)
    general = parent.prediction.general
    general.validate_inputs(inputs, specs)
    protected.guarded_inventory(root/"runtime", prepared["runtime_hashes"], data)
    annotations_clean = {row["label"]: copy.deepcopy(row["label_metadata_details"])
                         for key, rows in inputs.items() if key.startswith("clean.") for row in rows}
    checkpoint = next(s["checkpoint"] for s in specs if s["backbone"] == "ViT-L/14")
    selected = []
    for dataset in DATASETS:
        original = next(s for s in specs if s["dataset"] == dataset and s["corruption"] == "original")
        spec = copy.deepcopy(original)
        spec.update(id="M02."+dataset, backbone="ViT-L/14", model_type="naclip", checkpoint=copy.deepcopy(checkpoint),
                    vision_outputs=parent.prediction.visual_layers("ViT-L/14"), modes=list(MODES), preprocessing="native_PIL_bilinear")
        spec.pop("arms")
        selected.append(spec)
    source = dict(parent_input_suite=str(root), parent_prepared_sha256=protected.PARENT_SHA,
        parent_inputs_sha256=prepared["inputs_sha256"], parent_conditions_sha256=prepared["conditions_sha256"],
        rows={d: inputs["clean."+d] for d in DATASETS}, specs=selected, annotation_metadata=annotations_clean,
        G_condition_ids=[s["id"] for s in specs], GT_opened=False, old_predictions_reused=False)
    parent.prediction.validate_inputs(source["rows"], source["specs"])
    return source, roles


def safe_parent_binding():
    terminal = parent.parent_terminal()  # Retained exact success precedes even bootstrap metadata.
    source, roles = trusted_source()
    forbidden = source_exclusions(source, data_roles=roles)
    barrier, _ = protected.metadata(parent.PARENT/"ALL_PREDICTIONS_SEALED.json", parent.PARENT_BARRIER_SHA, forbidden)
    require(set(barrier["dataset_seals"]) == set(DATASETS), "M12 parent child seal membership differs")
    seals = {d: protected.metadata(parent.PARENT/d/"PREDICTIONS_SEALED.json", barrier["dataset_seals"][d], forbidden)[0]
             for d in DATASETS}
    binding = dict(parent=str(parent.PARENT), prepared_sha256=parent.PARENT_SHA, barrier_sha256=parent.PARENT_BARRIER_SHA,
                   terminal=terminal, source=source, seals=seals, analysis_sha256=parent.ANALYSIS_SHA, data_roles=roles)
    authenticate_parent(binding, forbidden)
    return binding


def fixed_cases(source):
    require(set(source["rows"]) == set(DATASETS) and [s["dataset"] for s in source["specs"]] == list(DATASETS),
            "M12 requires both fixed datasets")
    expected = []
    for spec in source["specs"]:
        dataset, batch = spec["dataset"], spec["config"]["batch_size"]
        rows = source["rows"][dataset]
        require(type(batch) is int and 0 < batch <= 512 and len(rows) >= 2
                and spec["images"] == len(rows) and spec["modes"] == list(MODES)
                and spec["preprocessing"] == "native_PIL_bilinear"
                and spec["config"]["resize"] == spec["config"]["patch_size"] == [224, 224],
                "M12 source/spec/geometry differs")
        for index in (0, 1):
            require(type(rows[index]["official_id"]) is str and rows[index]["official_id"], "M12 source ID differs")
            expected.append(dict(dataset=dataset, index=index, official_id=rows[index]["official_id"],
                                 episode=index//batch, batch_idx=index % batch))
    require(same_provenance(core.select_cases(source), expected), "M12 case selection is not fixed original indices0/1")
    return expected


def preflight_selected(binding, cases):
    """Stat the COMPLETE four-case graph, with no RGB/GT/NPY/receipt opens."""
    source, root = binding["source"], Path(binding["parent"])
    labels, rgb = source_roles(binding.get("data_roles", source))
    external = source_exclusions(source, data_roles=binding.get("data_roles")) - (labels | rgb)
    graph, seen = {}, {}
    def add(path, role, limit, sha, forbidden):
        name = str(path)
        current = output.file_pin(path, forbidden, limit, single=role not in ("rgb", "GT"))
        value = dict(role=role, pin=current, limit=limit, sha256=sha)
        require(name not in graph or same_provenance(graph[name], value), "M12 path has conflicting payload roles")
        inode = tuple(current[:2])
        require(inode not in seen or seen[inode] == name, "M12 selected payloads alias another path/role")
        seen[inode], graph[name] = name, value
    for case in cases:
        dataset, index, episode = case["dataset"], case["index"], case["episode"]
        row = source["rows"][dataset][index]
        require(digest(row["image_sha256"]), "M12 source RGB digest is missing")
        add(Path(row["image"]), "rgb", 32*1024**2, row["image_sha256"], external | labels)
        add(Path(row["label"]), "GT", 16*1024**2, None, external | rgb)
        inventory = binding["seals"][dataset]["artifact_sha256"]
        for mode in MODES:
            for relative, role, cap in ((f"{dataset}/predictions/{mode}/{index:06d}.npy", "prediction", 2*1024**2),
                                        (f"{dataset}/episodes/{episode:06d}.{mode}.json", "episode", 8*1024**2)):
                require(digest(inventory[relative]), "M12 selected sealed payload hash missing")
                add(canonical_child(root, relative), role, cap, inventory[relative], external | labels | rgb)
        for name, cap in (("cost.jsonl", 16*1024**2), ("initial_state.json", 2*1024**2), ("prediction_result.json", 2*1024**2)):
            relative = dataset+"/"+name
            require(digest(inventory[relative]), "M12 selected sealed cost/reset hash missing")
            add(canonical_child(root, relative), "cost_reset", cap, inventory[relative], external | labels | rgb)
    return graph


def recheck(graph, names=None):
    for name in graph if names is None else names:
        record = graph[name]
        require(output.file_pin(Path(name), set(), record["limit"], single=record["role"] not in ("rgb", "GT")) == record["pin"],
                "M12 pinned selected payload identity drift")


def pinned_bytes(graph, path):
    name = str(path)
    record = graph[name]
    recheck(graph, (name,))
    raw, sha = stable_bytes(Path(name), record["limit"], record["sha256"], expected_identity=tuple(record["pin"][:4]))
    recheck(graph, (name,))
    return raw, sha


def authenticate_parent(binding, forbidden):
    root, source = Path(binding["parent"]), binding["source"]
    require(root == parent.PARENT and root.resolve(strict=True) == root
            and binding["prepared_sha256"] == parent.PARENT_SHA and binding["barrier_sha256"] == parent.PARENT_BARRIER_SHA
            and not any((root/n).exists() for n in ("FAILED.json", "CANCELLED.json")), "M12 exact successful M02 parent differs")
    prepared, _ = protected.metadata(root/"SUITE_PREPARED.json", parent.PARENT_SHA, forbidden)
    saved, _ = protected.metadata(root/"INPUTS.json", prepared["inputs_sha256"], forbidden)
    require(prepared["schema_version"] == 1 and prepared["root"] == str(root)
            and same_provenance(saved, source) and len(prepared["parent_native_hashes"]) == 68, "M12 original source/native68 differs")
    data = source_exclusions(source, include_sources=False, data_roles=binding.get("data_roles"))
    parent.native_binding(prepared["parent_native_hashes"], data)
    protected.guarded_inventory(root/"runtime", prepared["runtime_hashes"], data)
    protected.metadata(root/"PROTOCOL.json", prepared["protocol_sha256"], forbidden)
    barrier, _ = protected.metadata(root/"ALL_PREDICTIONS_SEALED.json", parent.PARENT_BARRIER_SHA, forbidden)
    require(barrier["complete"] is True and barrier["GT_opened"] is False
            and barrier["suite_prepared_sha256"] == parent.PARENT_SHA and barrier["cells"] == 8
            and barrier["started_sha256"] == parent.PARENT_START_SHA
            and barrier["image_method_evaluations"] == sum(parent.POPULATIONS[d]*4 for d in DATASETS)
            and set(barrier["dataset_seals"]) == set(DATASETS), "M12 parent barrier differs")
    audit = barrier["audit_binding"]
    require(audit["admission_path"] == str(root/"ADMITTED.json") and audit["report_path"] == str(root/"INDEPENDENT_REPORT.md")
            and audit["admission_sha256"] == parent.PARENT_ADMISSION_SHA and audit["report_sha256"] == parent.PARENT_REPORT_SHA,
            "M12 parent audit provenance differs")
    admitted, _ = protected.metadata(root/"ADMITTED.json", parent.PARENT_ADMISSION_SHA, forbidden)
    parent.bounded_content(root/"INDEPENDENT_REPORT.md", parent.PARENT_REPORT_SHA, forbidden, 4*1024**2)
    verify_audit_binding(audit)
    start, _ = protected.metadata(root/"STARTED.json", parent.PARENT_START_SHA, forbidden)
    require(admitted["decision"] == "ADMITTED" and admitted["M02_predictions_allowed"] is True
            and admitted["GT_scoring_allowed"] is False and admitted["suite_prepared_sha256"] == parent.PARENT_SHA
            and start["suite_prepared_sha256"] == parent.PARENT_SHA and type(start["pid"]) is int
            and str(start["pid"]) == binding["terminal"]["ExecMainPID"] and start["GT_opened"] is False
            and same_provenance(start["audit"], audit), "M12 parent STARTED/permission differs")
    parent.prediction.verify_gates(root, parent.PARENT_SHA, barrier["gate_binding"], forbidden)
    for dataset in DATASETS:
        seal, _ = protected.metadata(root/dataset/"PREDICTIONS_SEALED.json", barrier["dataset_seals"][dataset], forbidden)
        require(same_provenance(seal, binding["seals"][dataset]), "M12 parent dataset seal drift")
        spec = next(s for s in source["specs"] if s["dataset"] == dataset)
        images, batch = spec["images"], spec["config"]["batch_size"]
        require(seal["complete"] is True and seal["GT_opened"] is False and seal["dataset"] == dataset
                and seal["suite_prepared_sha256"] == parent.PARENT_SHA and seal["images"] == images == parent.POPULATIONS[dataset]
                and same_provenance(seal["modes"], list(MODES)) and seal["image_method_evaluations"] == images*4,
                "M12 parent full dataset population differs")
        expected = {f"{dataset}/predictions/{m}/{i:06d}.npy" for m in MODES for i in range(images)}
        expected |= {f"{dataset}/episodes/{e:06d}.{m}.json" for m in MODES for e in range((images+batch-1)//batch)}
        expected |= {f"{dataset}/{n}" for n in ("initial_state.json", "cost.jsonl", "prediction_result.json")}
        require(set(seal["artifact_sha256"]) == expected and all(digest(v) for v in seal["artifact_sha256"].values()),
                "M12 parent full eight-cell payload membership differs")
    plan, _ = protected.metadata(parent.ANALYSIS, parent.ANALYSIS_SHA, forbidden)
    require(plan["prediction_prepared_sha256"] == parent.PARENT_SHA and plan["arm_order"] == list(MODES)
            and plan["fixed_contrasts"] == {k: list(v) for k, v in parent.analysis.CONTRASTS.items()}
            and type(plan["uncertainty"]["replicates"]) is int and plan["uncertainty"]["replicates"] == 10000
            and type(plan["uncertainty"]["root_seed"]) is int and plan["uncertainty"]["root_seed"] == 20261102,
            "M12 pinned parent analysis provenance differs")
    require(same_provenance(parent.parent_terminal(), binding["terminal"]), "M12 retained terminal identity drift")


def bind_inputs(scope_sha256):
    require(digest(scope_sha256), "Explicit operator SCOPE_v4 SHA256 required")
    binding = safe_parent_binding()
    source = binding["source"]
    cases = fixed_cases(source)
    graph = preflight_selected(binding, cases)
    forbidden = source_exclusions(source, data_roles=binding["data_roles"]) | {tuple(v["pin"][:2]) for v in graph.values()}
    scope, _ = protected.metadata(SCOPE, scope_sha256, forbidden, limit=4*1024**2)
    prospective, _ = protected.metadata(CASE_PROTOCOL, CASE_PROTOCOL_SHA, forbidden, limit=4*1024**2)
    require(type(scope) is dict and bool(scope) and same_provenance(prospective, core.protocol()),
            "M12 SCOPE_v4/prospective case protocol differs")
    return dict(parent_binding=binding, cases=cases, selected_payloads=graph, scope_sha256=scope_sha256,
                scope=scope, case_protocol_sha256=CASE_PROTOCOL_SHA, case_protocol=prospective)


def protocol(scope_sha256):
    return dict(experiment="M12_fixed_original_cases_only", scope_path=str(SCOPE), scope_sha256=scope_sha256,
        prospective_case_protocol_path=str(CASE_PROTOCOL), prospective_case_protocol_sha256=CASE_PROTOCOL_SHA,
        rendering=core.protocol(), selection="Indices0/1 per dataset in original source order, never GT/score/outcome selection",
        exposure="Known aggregate M02 outcomes already observed; these are descriptive fixed cases, not untouched selection",
        permission="New exact CPU_GT AND M12_qualitative_GT AND RGB_reading permission",
        GPU_allowed=False, MODEL_allowed=False, upload_allowed=False, B_publication_allowed=False, paper_edits=False,
        execution_resources=RESOURCES, limits=dict(output_cap_bytes=OUTPUT_CAP, shared_reserve_bytes=RESERVE),
        failures="Manual once after independent exact artifact review; exclusive owned STARTED, no resume/retry; global seal last")


def prepare(suite, scope_sha256):
    cpu_only()
    bound = bind_inputs(scope_sha256)
    suite = Path(suite).absolute()
    roots = [root for root, _ in role_roots(bound["parent_binding"]["source"])]
    roots.extend((parent.PARENT, protected.PARENT))
    require(suite.resolve() == suite and not suite.exists()
            and not any(suite.is_relative_to(root) or root.is_relative_to(suite) for root in roots),
            "Fresh disjoint canonical M12 suite required")
    require(shutil.disk_usage(suite.parent).free >= RESERVE+OUTPUT_CAP, "M12 prepare disk headroom insufficient")
    suite.mkdir()
    inputs_sha = protected.write_prepared_json(suite/"CASE_INPUTS.json", bound)
    protocol_sha = protected.write_prepared_json(suite/"PROTOCOL.json", protocol(scope_sha256))
    inventory = protected.guarded_snapshot(ROOT, suite/"runtime", source_exclusions(bound["parent_binding"]["source"],
        include_sources=False, data_roles=bound["parent_binding"]["data_roles"]))
    return protected.write_prepared_json(suite/"SUITE_PREPARED.json", dict(schema_version=1, root=str(suite),
        inputs_sha256=inputs_sha, protocol_sha256=protocol_sha, runtime_hashes=inventory,
        scope_sha256=scope_sha256, case_protocol_sha256=CASE_PROTOCOL_SHA, GT_opened=False, RGB_opened=False, MODEL_used=False, GPU_used=False))


def admission(suite, expected_sha, scope_sha256):
    cpu_only()
    bound = bind_inputs(scope_sha256)
    source = bound["parent_binding"]["source"]
    data = source_exclusions(source, include_sources=False, data_roles=bound["parent_binding"]["data_roles"])
    forbidden = source_exclusions(source, data_roles=bound["parent_binding"]["data_roles"]) | {tuple(v["pin"][:2]) for v in bound["selected_payloads"].values()}
    suite = Path(suite).resolve(strict=True)
    forbidden |= {v[:2] for v in protected.source_members(suite/"runtime", data).values()}
    prepared, _ = protected.metadata(suite/"SUITE_PREPARED.json", expected_sha, forbidden)
    require(prepared["schema_version"] == 1 and prepared["root"] == str(suite)
            and prepared["scope_sha256"] == scope_sha256 and prepared["case_protocol_sha256"] == CASE_PROTOCOL_SHA
            and Path(__file__).resolve().is_relative_to(suite/"runtime"), "Wrong exact M12 runtime")
    inputs, _ = protected.metadata(suite/"CASE_INPUTS.json", prepared["inputs_sha256"], forbidden)
    plan, _ = protected.metadata(suite/"PROTOCOL.json", prepared["protocol_sha256"], forbidden)
    require(same_provenance(inputs, bound) and same_provenance(plan, protocol(scope_sha256)), "M12 input/protocol drift")
    protected.guarded_inventory(suite/"runtime", prepared["runtime_hashes"], data)
    parent.prediction.general.runtime_imports(suite/"runtime")
    audit, audit_sha = protected.metadata(suite/"ADMITTED.json", forbidden=forbidden, limit=4*1024**2)
    require(audit["decision"] == "ADMITTED" and audit["suite_prepared_sha256"] == expected_sha
            and audit.get("independent_reviewer") and audit.get("CPU_GT_scoring_allowed") is True
            and audit.get("M12_qualitative_GT_allowed") is True and audit.get("RGB_reading_allowed") is True
            and audit.get("GPU_allowed") is False and audit.get("MODEL_allowed") is False,
            "New exact CPU_GT AND M12 qualitative GT AND RGB admission required")
    report = suite/"INDEPENDENT_REPORT.md"
    require(audit["report_path"] == str(report), "Wrong M12 independent report")
    parent.bounded_content(report, audit["report_sha256"], forbidden, 4*1024**2)
    ids = [output.file_pin(path, forbidden, 4*1024**2, single=True)[:4] for path in (suite/"ADMITTED.json", report)]
    binding = dict(admission_path=str(suite/"ADMITTED.json"), admission_sha256=audit_sha, admission_identity=ids[0],
        report_path=str(report), report_sha256=audit["report_sha256"], report_identity=ids[1], independent_reviewer=audit["independent_reviewer"])
    verify_audit_binding(binding)
    return suite, prepared, inputs, binding


def strict_episode_metadata(records, spec, ids, episode, cases):
    """Apply the closed core's metadata contracts without fabricated pixel inputs."""
    shared = None
    require(type(ids) is list and ids and all(type(v) is str and v for v in ids), "M12 original episode IDs differ")
    for mode in MODES:
        row = records[mode]
        require(row["GT_opened"] is False and row["scientific_identity"] == "M02_spatial_"+mode
                and type(row["episode"]) is int and row["episode"] == episode
                and same_provenance(row["ordered_ids"], ids) and digest(row["candidate_sha256"]),
                "M12 strict original episode/mode identity differs")
        candidates = core.checked_candidates(row, spec)
        require(shared is None or row["candidate_sha256"] == shared, "M12 original candidate digests differ across modes")
        shared = row["candidate_sha256"]
        geometry = row["preservation_geometry"]
        require(type(geometry) is list and len(geometry) == len(candidates), "M12 strict geometry membership differs")
        for index, (candidate, saved) in enumerate(zip(candidates, geometry, strict=True)):
            require(type(saved["candidate_index"]) is int and saved["candidate_index"] == index
                    and type(saved["batch_idx"]) is int and saved["batch_idx"] == candidate["batch_idx"]
                    and saved["mode"] == mode and saved["readout_and_loss_remain_original"] is True,
                    "M12 strict geometry index/batch/mode differs")
            if mode == "translated":
                _, expected = core.translated_candidate(core.candidate_mask(candidate))
                require(all(k in saved and same_provenance(saved[k], value) for k, value in expected.items()),
                        "M12 strict translated geometry differs")
    for case in cases:
        require(type(case["official_id"]) is str and 0 < len(case["official_id"]) <= 256
                and case["official_id"].isascii() and type(case["batch_idx"]) is int
                and 0 <= case["batch_idx"] < len(ids) and ids[case["batch_idx"]] == case["official_id"],
                "M12 selected display/source resource bound differs")
        chosen = next((c for c in records["native"]["candidates"] if c["batch_idx"] == case["batch_idx"]), None)
        if chosen is not None:
            query = spec["vocabulary"]["queries"][chosen["class_idx"]]
            require(type(query) is str and len(query) <= 1024, "M12 query display resource bound differs")


def validate_selected_metadata(bound):
    binding, graph = bound["parent_binding"], bound["selected_payloads"]
    source, root = binding["source"], Path(binding["parent"])
    require(same_provenance(preflight_selected(binding, bound["cases"]), graph), "M12 complete graph drift before metadata/GT")
    forbidden = source_exclusions(source, data_roles=binding.get("data_roles"))
    receipts, costs = {}, {}
    for spec in source["specs"]:
        dataset = spec["dataset"]
        inventory = binding["seals"][dataset]["artifact_sha256"]
        cost_paths = [str(root/dataset/name) for name in ("cost.jsonl", "initial_state.json", "prediction_result.json")]
        for name in cost_paths:
            pinned_bytes(graph, Path(name))
        summary, rows = parent.cost_binding(root, spec, inventory, forbidden)
        recheck(graph, cost_paths)  # Closed metadata validator may reopen; never adopt a replacement inode.
        costs[dataset] = summary
        coverage = {m: dict(candidates=0, EPS_active_candidates=0, sources=0) for m in MODES}
        coverage.update(translated_shiftable=0, translated_unshiftable=0, translated_overlap_sum=0.)
        batch = spec["config"]["batch_size"]
        episodes = sorted({c["episode"] for c in bound["cases"] if c["dataset"] == dataset})
        for episode in episodes:
            ids = [r["official_id"] for r in source["rows"][dataset][episode*batch:(episode+1)*batch]]
            raw_rows = rows[episode*4:(episode+1)*4]
            require(len(raw_rows) == 4 and {row["mode"] for row in raw_rows} == set(MODES)
                    and all(type(row["episode"]) is int and row["episode"] == episode
                            and type(row["order_position"]) is int for row in raw_rows), "M12 raw episode cost identity differs")
            by_mode = {row["mode"]: row for row in raw_rows}
            saved, paths = {}, []
            for mode in MODES:
                path = root/dataset/"episodes"/f"{episode:06d}.{mode}.json"
                raw, _ = pinned_bytes(graph, path)
                record = json.loads(raw)
                require(same_provenance(record["ordered_ids"], ids) and all(k in record
                        and same_provenance(record[k], value) for k, value in by_mode[mode].items()),
                        "M12 typed episode/source/raw-cost identity differs")
                saved[mode], paths = record, paths+[str(path)]
            actual = parent.episode_metadata(root, spec, inventory, forbidden, episode, rows, coverage)
            recheck(graph, (*cost_paths, *paths))
            require(set(actual) == set(MODES) and all(same_provenance(value, ids) for value in actual.values()),
                    "M12 validated four-mode original episode IDs differ")
            strict_episode_metadata(saved, spec, ids, episode,
                [c for c in bound["cases"] if c["dataset"] == dataset and c["episode"] == episode])
            receipts[(dataset, episode)] = saved
    recheck(graph)
    require(same_provenance(parent.parent_terminal(), binding["terminal"]), "M12 terminal drift before first selected content")
    return receipts, costs


def write_output(suite, name, content, budget, on_create=lambda: None):
    require(type(budget.get("bytes")) is int and 0 <= budget["bytes"] <= OUTPUT_CAP
            and budget["bytes"]+len(content) <= OUTPUT_CAP and shutil.disk_usage(suite).free >= RESERVE+len(content),
            "M12 output/reserve budget exceeded")
    return output.write_output(suite, name, content, budget, on_create)


def execute(suite, bound):
    receipts, costs = validate_selected_metadata(bound)
    binding, graph = bound["parent_binding"], bound["selected_payloads"]
    root, source = Path(binding["parent"]), binding["source"]
    budget, artifacts, identities, manifest = dict(bytes=0), {}, {}, []
    for case in bound["cases"]:
        recheck(graph)
        dataset, index = case["dataset"], case["index"]
        spec = next(s for s in source["specs"] if s["dataset"] == dataset)
        row = source["rows"][dataset][index]
        rgb, rgb_sha = pinned_bytes(graph, Path(row["image"]))
        gt, gt_sha = pinned_bytes(graph, Path(row["label"]))
        predictions = {mode: pinned_bytes(graph, root/dataset/"predictions"/mode/f"{index:06d}.npy")[0] for mode in MODES}
        recheck(graph)
        png, metadata = core.render_case(rgb, gt, predictions, receipts[(dataset, case["episode"])], spec, case)
        require(type(png) is bytes and png and type(metadata) is dict, "M12 renderer output schema differs")
        prefix = f"cases/{dataset}.{index:06d}"
        record = dict(case=case, rendering=metadata, source_RGB_sha256=rgb_sha, source_GT_sha256=gt_sha,
            parent_prepared_sha256=parent.PARENT_SHA, parent_barrier_sha256=parent.PARENT_BARRIER_SHA,
            case_protocol_sha256=CASE_PROTOCOL_SHA, scope_sha256=bound["scope_sha256"],
            cost=costs[dataset], resources=binding["seals"][dataset].get("resource_observation"),
            cost_scope="Inherited descriptive M02 raw cost; no equal-budget or causal case claim",
            GPU_used=False, MODEL_used=False, uploaded=False, B_published=False)
        for name, value in ((prefix+".png", png), (prefix+".json", output.encoded(record))):
            artifacts[name], identities[name] = write_output(suite, name, value, budget)
        manifest.append(dict(case=case, PNG=prefix+".png", metadata=prefix+".json"))
        parent.write_json(suite/"queue_state.json", dict(phase="M12_fixed_case_rendering", completed_cases=len(manifest),
            total_cases=4, GT_opened=True, RGB_opened=True, GPU_used=False), replace=True)
    require(len(manifest) == 4, "M12 incomplete fixed cases")
    artifacts["CASES.json"], identities["CASES.json"] = write_output(suite, "CASES.json", output.encoded(dict(
        cases=manifest, protocol=bound["case_protocol"], exposure=protocol(bound["scope_sha256"])["exposure"],
        paper_integration=False, uploaded=False, B_published=False)), budget)
    return dict(complete=True, cases=4, datasets=list(DATASETS), GT_opened=True, RGB_opened=True,
        GPU_used=False, MODEL_used=False, uploaded=False, B_published=False, paper_integration=False,
        artifact_sha256=artifacts, artifact_identities=identities, output_bytes=budget["bytes"])


def run(suite, expected_sha, scope_sha256):
    suite, prepared, bound, audit = admission(suite, expected_sha, scope_sha256)
    require(not any((suite/name).exists() for name in ("STARTED.json", "FAILED.json", "CASES_SEALED.json")), "No M12 resume/retry")
    require(shutil.disk_usage(suite).free >= RESERVE+OUTPUT_CAP, "M12 run headroom insufficient")
    owned = []
    try:
        verify_audit_binding(audit)
        started, _ = write_output(suite, "STARTED.json", output.encoded(dict(suite_prepared_sha256=expected_sha,
            audit=audit, pid=os.getpid(), scope_sha256=scope_sha256, GT_opened=False, RGB_opened=False,
            GPU_used=False, MODEL_used=False, automatic_retry=False)), dict(bytes=0), lambda: owned.append(True))
        begin = time.perf_counter()
        result = execute(suite, bound)
        require(result["cases"] == 4 and result["GPU_used"] is False and result["MODEL_used"] is False, "Incomplete M12 output")
        parent.write_json(suite/"queue_state.json", dict(phase="M12_cases_complete_awaiting_seal_and_exit", cases=4,
            GT_opened=True, RGB_opened=True, GPU_used=False, MODEL_used=False), replace=True)
        require(same_provenance(bind_inputs(scope_sha256), bound), "M12 final parent/input/protocol drift")
        source = bound["parent_binding"]["source"]
        data = source_exclusions(source, include_sources=False, data_roles=bound["parent_binding"]["data_roles"])
        forbidden = source_exclusions(source, data_roles=bound["parent_binding"]["data_roles"]) | {tuple(v["pin"][:2]) for v in bound["selected_payloads"].values()}
        protected.guarded_inventory(suite/"runtime", prepared["runtime_hashes"], data)
        forbidden |= {value[:2] for value in protected.source_members(suite/"runtime", data).values()}
        for name, sha in (("SUITE_PREPARED.json", expected_sha), ("CASE_INPUTS.json", prepared["inputs_sha256"]),
                          ("PROTOCOL.json", prepared["protocol_sha256"]), ("STARTED.json", started)):
            protected.metadata(suite/name, sha, forbidden)
        for name, sha in result["artifact_sha256"].items():
            path = canonical_child(suite, name)
            identity = result["artifact_identities"][name]
            require(output.file_pin(path, forbidden, 64*1024**2, single=True) == identity, "M12 final output stat drift")
            stable_bytes(path, 64*1024**2, sha, expected_identity=tuple(identity[:4]))
            require(output.file_pin(path, forbidden, 64*1024**2, single=True) == identity, "M12 final output stat drift after read")
        parent.prediction.general.runtime_imports(suite/"runtime")
        verify_audit_binding(audit)
        cpu_only()
        publish_seal(suite/"CASES_SEALED.json", dict(**result, suite_prepared_sha256=expected_sha,
            scope_sha256=scope_sha256, case_protocol_sha256=CASE_PROTOCOL_SHA, parent_prepared_sha256=parent.PARENT_SHA,
            parent_barrier_sha256=parent.PARENT_BARRIER_SHA, audit_binding=audit, started_sha256=started,
            elapsed_seconds=time.perf_counter()-begin))
    except BaseException as error:
        if owned:
            protected.write_prepared_json(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate-admission", "run"))
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--suite-sha256")
    parser.add_argument("--scope-sha256", required=True, help="Explicit operator pin of the immutable SCOPE_v4.json")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.phase == "prepare":
        print(dict(prepared_sha256=prepare(args.suite, args.scope_sha256), GT_opened=False, RGB_opened=False, GPU_used=False))
    else:
        require(args.suite_sha256, "Exact M12 prepared SHA256 required")
        if args.phase == "run":
            run(args.suite, args.suite_sha256, args.scope_sha256)
        else:
            admission(args.suite, args.suite_sha256, args.scope_sha256)
            print(dict(admission_valid=True, GT_opened=False, RGB_opened=False, GPU_used=False))
