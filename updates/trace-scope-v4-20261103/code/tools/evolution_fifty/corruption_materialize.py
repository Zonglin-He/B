"""Fresh, single-attempt CPU VOC20-C generation from pinned upstream bytes.

This creates experimental RGB inputs only, never model predictions or scores.
The historical cache and original projects are read-only and not restored.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from tools.evolution_followup.queue import snapshot
from tools.evolution_tuning.common import write_json
from .bootstrap import read_json, require
from .corruption_parity import UPSTREAM_COMMIT, load_reference, snapshot_reference
from .generalization import CORRUPTIONS
from .intake import sha256, write_new

ROOT = Path(__file__).resolve().parents[2]
MAX_OUTPUT_BYTES = 8*1024**3


def versions():
    return dict({k:importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-image", "Pillow")}, cv2=cv2.__version__)


def generate_image(module, bgr, source_index, corruption):
    require(bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3, "Expected decoded BGR uint8")
    require(type(source_index) is int and source_index >= 0 and corruption in CORRUPTIONS, "Invalid generation condition")
    before = bgr.copy()
    state = np.random.get_state()
    try:
        np.random.seed(source_index)
        degraded = module.corrupt(bgr, severity=5, corruption_name=corruption)
        pixels = cv2.resize(degraded, (224, 224), interpolation=cv2.INTER_LINEAR)
        require(np.array_equal(bgr, before), "Reference corruption mutated shared source pixels")
        require(pixels.dtype == np.uint8 and pixels.shape == (224, 224, 3), "Invalid generated corruption geometry")
        success, content = cv2.imencode(".png", pixels)
        require(success and np.array_equal(cv2.imdecode(content, cv2.IMREAD_COLOR), pixels), "PNG roundtrip changed generated pixels")
        return content.tobytes()
    finally:
        np.random.set_state(state)


def prepare(inputs, output):
    inputs, output = Path(inputs).resolve(strict=True), Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh CPU corruption suite required")
    prepared = read_json(inputs/"INPUTS_PREPARED.json")
    require(sha256(inputs/"CLEAN_INPUTS.json") == prepared["artifact_sha256"]["CLEAN_INPUTS.json"], "Frozen clean input drift")
    rows = read_json(inputs/"CLEAN_INPUTS.json")["voc20"]["panel"]["inputs"]
    require(len(rows) == len({r["official_id"] for r in rows}) == 1449, "VOC20 full population required")
    output.mkdir(parents=True)
    write_new(output/"SOURCE_INPUTS.json", rows)
    ref = snapshot_reference(output)
    write_new(output/"PROTOCOL.json", dict(status="CPU_RGB_input_generation_only", official_source_commit=UPSTREAM_COMMIT,
        source_inputs_sha256=sha256(output/"SOURCE_INPUTS.json"), parent_input_receipt_sha256=sha256(inputs/"INPUTS_PREPARED.json"),
        reference_sha256=ref, corruptions=list(CORRUPTIONS), severity=5, source_images=1449, expected_RGBs=21735,
        reason="Historical cache failed all30 representative exact-upstream regeneration checks; never overwrite or reuse it silently",
        process="cv2 decode originalBGR; independent NumPyseed sample_idx for eachtype; exact upstreamcorrupt BEFORE cv2linear224resize;losslessPNG",
        original_labels="Never opened, warped, decoded or used for selection; same metadata only carried to future offline scoring",
        max_output_bytes=MAX_OUTPUT_BYTES, free_disk_required_bytes=12*1024**3, versions=versions(),
        RNG="Restore NumPy global state after every sample/type including failure; no concurrent threads around seeded generator",
        cache_or_dataset_completion="Input completion only, not GPU gate, prediction, metric or scientific completion",
        automatic_retry=False, resume=False, GPU_allowed=False))
    inventory = snapshot(ROOT, output/"runtime")
    write_new(output/"CPU_PREPARED.json", dict(root=str(output), protocol_sha256=sha256(output/"PROTOCOL.json"),
        runtime_hashes=inventory, source_inputs_sha256=sha256(output/"SOURCE_INPUTS.json"), reference_sha256=ref,
        independent_GPU_admission=False, GPU_allowed=False))
    return sha256(output/"CPU_PREPARED.json")


def run(suite, expected_sha):
    from .risk_worker import forbid_annotations, strict_runtime_inventory
    suite = Path(suite).resolve(strict=True)
    require(sha256(suite/"CPU_PREPARED.json") == expected_sha, "Wrong CPU input generation suite")
    bound = read_json(suite/"CPU_PREPARED.json")
    require(bound["root"] == str(suite) and Path(__file__).resolve().is_relative_to(suite/"runtime")
            and strict_runtime_inventory(suite/"runtime") == bound["runtime_hashes"], "Wrong CPU generator runtime")
    require(sha256(suite/"PROTOCOL.json") == bound["protocol_sha256"]
            and sha256(suite/"SOURCE_INPUTS.json") == bound["source_inputs_sha256"], "CPU generation inputs changed")
    require(strict_runtime_inventory(suite/"reference") ==
            {str(Path(name).relative_to("reference")): digest for name, digest in bound["reference_sha256"].items()},
            "Pinned upstream reference inventory drift")
    protocol = read_json(suite/"PROTOCOL.json")
    require(versions() == protocol["versions"], "Generator package version drift")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU generator must hide all CUDA devices")
    require(not (suite/"STARTED.json").exists() and not (suite/"FAILED.json").exists(), "No CPU generation resume/retry")
    import shutil
    require(shutil.disk_usage(suite).free >= protocol["free_disk_required_bytes"], "Insufficient CPU generation disk budget")
    rows = read_json(suite/"SOURCE_INPUTS.json")
    require(len(rows) == protocol["source_images"]
            and protocol["corruptions"] == list(CORRUPTIONS) and protocol["severity"] == 5
            and protocol["expected_RGBs"] == len(rows)*len(CORRUPTIONS), "Frozen generation population drift")
    forbid_annotations([r["label"] for r in rows])
    write_new(suite/"STARTED.json", dict(pid=os.getpid(), prepared_sha256=expected_sha, GPU_allowed=False, labels_read=False))
    datasets = {kind:[] for kind in CORRUPTIONS}
    total_bytes, completed, started = 0, 0, time.perf_counter()
    try:
        module = load_reference(suite/"reference")
        with (suite/"cost.jsonl").open("x") as costs:
            for index, row in enumerate(rows):
                status = Path(row["label"]).stat()
                require([status.st_size, status.st_mtime_ns] == row["label_metadata"], "Label metadata changed")
                content = Path(row["image"]).read_bytes()
                require(hashlib.sha256(content).hexdigest() == row["image_sha256"], "Original RGB drift")
                bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                require(bgr is not None, "Original RGB decode failed")
                tick = time.perf_counter()
                for kind in CORRUPTIONS:
                    encoded = generate_image(module, bgr, index, kind)
                    total_bytes += len(encoded)
                    require(total_bytes <= MAX_OUTPUT_BYTES, "Generated input disk bound exceeded")
                    target = suite/"rgb"/kind/f"{index:06d}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(encoded)
                    item = copy.deepcopy(row)
                    item.update(image=str(target), image_sha256=hashlib.sha256(encoded).hexdigest(),
                        clean_image=row["image"], clean_image_sha256=row["image_sha256"],
                        corruption=kind, severity=5, original_sample_idx=index,
                        preprocessing="already_corrupted_and_cv2_resized_no_second_resize")
                    datasets[kind].append(item)
                    completed += 1
                elapsed = time.perf_counter()-started
                costs.write(json.dumps(dict(source_index=index, generated_RGBs=15,
                    seconds=time.perf_counter()-tick, elapsed_seconds=elapsed, labels_read=False))+"\n")
                costs.flush()
                write_json(suite/"queue_state.json", dict(phase="generating_RGB_inputs_only", pid=os.getpid(),
                    completed_sources=index+1, total_sources=len(rows), completed_RGBs=completed,
                    total_RGBs=protocol["expected_RGBs"], bytes=total_bytes, elapsed_seconds=elapsed,
                    labels_read=False, GPU_used=False), replace=True)
        require(completed == protocol["expected_RGBs"] and all(len(v) == len(rows) for v in datasets.values()), "Generated population incomplete")
        write_new(suite/"VOC20_C_S5_INPUTS.json", datasets)
        write_new(suite/"INPUTS_SEALED.json", dict(complete=True, scope="fresh_generated_RGB_inputs_only_not_predictions_or_scores",
            prepared_sha256=expected_sha, generated_RGBs=completed, generated_bytes=total_bytes,
            all_RGB_payload_hashes_in="VOC20_C_S5_INPUTS.json", reference_sha256=bound["reference_sha256"],
            artifact_sha256={name:sha256(suite/name) for name in
                ("PROTOCOL.json", "SOURCE_INPUTS.json", "VOC20_C_S5_INPUTS.json", "cost.jsonl")},
            elapsed_seconds=time.perf_counter()-started, labels_read=False, GPU_used=False, scientific_completion=False))
        write_json(suite/"queue_state.json", dict(phase="RGB_inputs_sealed_not_experiment_complete", completed_RGBs=completed,
            pid=os.getpid(), labels_read=False, GPU_used=False, scientific_completion=False), replace=True)
    except BaseException as error:
        write_new(suite/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False,
            completed_RGBs=completed, labels_read=False, GPU_used=False))
        raise
    finally:
        for name in list(sys.modules):
            if name == "_trace_pinned_upstream_corruptions" or name.startswith("_trace_pinned_upstream_corruptions."):
                del sys.modules[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args()
    if args.phase == "prepare":
        require(args.inputs is not None, "Frozen source inputs required")
        print(dict(prepared_sha256=prepare(args.inputs, args.suite), GPU_started=False))
    else:
        require(args.suite_sha256, "CPU prepared SHA required")
        run(args.suite, args.suite_sha256)
