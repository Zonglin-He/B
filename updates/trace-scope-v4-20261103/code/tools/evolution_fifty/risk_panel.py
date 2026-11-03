"""Freeze TRACE-M01 candidate-risk panels from existing label-blind receipts.

This is input preparation, not an update-risk experiment or model execution.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .bootstrap import TUNED, read_json, require
from .intake import sha256, write_new

DATASETS = ("coco_object", "pc60")
SOURCE_LIMIT, CANDIDATE_LIMIT = 128, 8


def key(*parts):
    return hashlib.sha256("TRACE-M01-v1\0".encode() + "\0".join(parts).encode()).hexdigest()


def select_sources(ids, limit=SOURCE_LIMIT):
    require(len(ids) == len(set(ids)) and type(limit) is int and limit > 0, "Invalid source sample")
    selected = set(sorted(ids, key=lambda value: (key(value), value))[:limit])
    return [value for value in ids if value in selected]


def candidate_identity(row):
    return f"{row['batch_idx']:04d}:{row['class_idx']:04d}:{row['component_idx']:06d}"


def select_candidates(source_id, rows, limit=CANDIDATE_LIMIT):
    require(type(limit) is int and limit > 0 and all("region" in row for row in rows), "Invalid candidates")
    identities = [candidate_identity(row) for row in rows]
    require(len(set(identities)) == len(identities), "Duplicate candidate identity")
    # No confidence, C, GT, or observed outcome enters this sampling priority.
    selected = set(sorted(identities, key=lambda value: (key(source_id, value), value))[:limit])
    return [row for row in rows if candidate_identity(row) in selected]


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "New risk protocol required")
    output.mkdir(parents=True)
    write_new(output / "PROTOCOL.json", dict(experiment="TRACE-M01", datasets=list(DATASETS),
        source_limit=SOURCE_LIMIT, candidate_limit=CANDIDATE_LIMIT,
        selection="Hash-only source/candidate priorities, preserve original episode/candidate order; never GT/outcome selected",
        update="Same initial model/Adam state reset per candidate; original candidate mean entropy with rho=1, no C scaling",
        steps="Record state after1step and after original frozen step count in one reset trajectory",
        config="Original per-dataset B/lr/K/visualLN/queries/input and episode geometry unchanged",
        readout="Original source image; candidate and outside regions separately; ignore255 excluded",
        primary_outcomes=["GT-class mean support delta", "Correct-to-Wrong", "Wrong-to-Correct", "net correct pixels"],
        stratification="Within dataset: predicted benchmark class, confidence bins width0.05, fixed area-fraction bins",
        confidence_bin_width=.05, area_fraction_edges=[0., .01, .05, .2, .5, 1.],
        C_groups="Within-stratum median split; exact-median ties excluded from matched contrast and reported",
        overlap_policy="Report every source/candidate and unmatched stratum; do not merge sparse strata after seeing outcomes",
        uncertainty="Resample whole original episodes with shared draws,10000 replicates; no seed/order generality",
        labels="Not opened during panel preparation; future offline scoring only after all panel trajectories sealed",
        old_17_89_pp="Forbidden as an outcome of this experiment",
        source_code_sha256={str(Path(__file__).resolve()): sha256(__file__),
            str(Path(__file__).with_name("intake.py").resolve()): sha256(Path(__file__).with_name("intake.py"))},
        new_gpu_started=False, status="preparing_inputs_not_experiment_complete"))
    reports, bindings = {}, {}
    for dataset in DATASETS:
        root = TUNED / "full" / dataset
        seal = read_json(root / "PREDICTIONS_SEALED.json")
        require(seal["complete"] is True and seal["gt_contents_opened"] is False, "Unsealed diagnostic source")
        inventory = seal["artifact_sha256"]
        protocol_path = root / "prediction_protocol.json"
        require(sha256(protocol_path) == inventory["prediction_protocol.json"], "Original prediction protocol drift")
        manifest = read_json(protocol_path)["manifest"]
        ids, batch = manifest["panel"]["ordered_ids"], manifest["config"]["batch_size"]
        selected = set(select_sources(ids))
        episodes = sorted({i//batch for i, source in enumerate(ids) if source in selected})
        examples, source_count = [], {source: 0 for source in selected}
        inputs = manifest["panel"]["inputs"]
        source_bindings = {"PREDICTIONS_SEALED.json": sha256(root / "PREDICTIONS_SEALED.json"),
                           "prediction_protocol.json": sha256(protocol_path)}
        for episode in episodes:
            name = f"episodes/{episode:06d}.image_C.json"
            require(sha256(root/name) == inventory[name], "Cached candidate receipt drift")
            source_bindings[name] = inventory[name]
            receipt = read_json(root/name)
            episode_ids = ids[episode*batch:(episode+1)*batch]
            require(receipt["episode"] == episode and receipt["ordered_ids"] == episode_ids, "Episode membership drift")
            for local, source in enumerate(episode_ids):
                if source not in selected:
                    continue
                rows = [r for r in receipt["candidates"] if r["image_idx"] == local]
                sampled = select_candidates(source, rows)
                source_count[source] = len(sampled)
                for row in sampled:
                    region = row["region"]
                    shape = region["shape"]
                    require(region["encoding"] == "packbits-little-hex" and len(shape) == 2
                            and all(type(n) is int and n > 0 for n in shape), "Invalid candidate mask")
                    packed = bytes.fromhex(region["data"])
                    mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
                    area = int(mask[:shape[0]*shape[1]].sum())
                    require(len(packed) == (shape[0]*shape[1]+7)//8
                            and area == region["true_pixels"] == row["area"] > 0,
                            "Candidate mask/area mismatch")
                    require(row["diagnostic_algorithm"] == "C" and row["confidence"] == row["original_support"]
                            and row["reliability"] == row["counterfactual_support"], "Non-native diagnostic source")
                    examples.append(dict(source_id=source, original_source_index=episode*batch+local,
                        episode=episode, episode_ids=episode_ids, candidate_id=candidate_identity(row),
                        candidate=row, source_input=inputs[episode*batch+local]))
        write_new(output / f"{dataset}.panel.json", dict(dataset=dataset, config=manifest["config"],
            selected_source_ids=[value for value in ids if value in selected],
            candidate_count=len(examples), sources_with_no_candidates=sum(v == 0 for v in source_count.values()),
            examples=examples, source_files_sha256=source_bindings,
            original_checkpoint=manifest["checkpoint"], query_to_class=manifest["panel"]["query_to_class"]))
        reports[dataset] = dict(selected_sources=len(selected), candidates=len(examples),
            original_episode_groups=len(episodes), sources_with_no_candidates=sum(v == 0 for v in source_count.values()),
            trajectory_checkpoints=[1, manifest["config"]["steps"]])
        bindings[dataset] = source_bindings
    names = ["PROTOCOL.json", *[f"{d}.panel.json" for d in DATASETS]]
    write_new(output / "PREPARED.json", dict(executable_queue=False, new_gpu_started=False,
        status="input_panel_only", datasets=reports,
        artifact_sha256={name: sha256(output/name) for name in names}))
    print(reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
