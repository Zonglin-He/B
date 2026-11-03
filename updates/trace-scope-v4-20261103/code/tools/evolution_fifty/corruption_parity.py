"""Bounded CPU regeneration audit against pinned upstream MLMP corruptions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from .bootstrap import read_json, require
from .generalization import CORRUPTIONS
from .intake import sha256, write_new

REPO = Path("/home/wwww/hzl/external/MLMP")
UPSTREAM_COMMIT = "2a887de018b3f2c3b132f69fe806bab12a2d511f"
PREFIX = "utils/imagecorruptions/"


def git_bytes(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def snapshot_reference(output):
    names = git_bytes("ls-tree", "-r", "--name-only", UPSTREAM_COMMIT, PREFIX).decode().splitlines()
    require(len(names) == 8 and set(Path(n).name for n in names) ==
        {"__init__.py", "corruptions.py", "frost1.png", "frost2.png", "frost3.png", "frost4.jpg", "frost5.jpg", "frost6.jpg"},
        "Unexpected pinned upstream corruption inventory")
    inventory = {}
    for name in names:
        relative = Path(name).relative_to(PREFIX)
        require(".." not in relative.parts, "Unsafe reference path")
        content = git_bytes("show", f"{UPSTREAM_COMMIT}:{name}")
        path = output/"reference"/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        inventory[str(path.relative_to(output))] = hashlib.sha256(content).hexdigest()
    return inventory


def load_reference(root):
    name = "_trace_pinned_upstream_corruptions"
    require(not any(key == name or key.startswith(name+".") for key in sys.modules), "Reference namespace already loaded")
    spec = importlib.util.spec_from_file_location(name, root/"__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit(inputs, output):
    inputs, output = Path(inputs).resolve(strict=True), Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh parity audit directory required")
    prepared = read_json(inputs/"INPUTS_PREPARED.json")
    for name in ("PROTOCOL.json", "CLEAN_INPUTS.json", "VOC20_C_S5_INPUTS.json"):
        require(sha256(inputs/name) == prepared["artifact_sha256"][name], "Frozen parity input changed")
    clean = read_json(inputs/"CLEAN_INPUTS.json")["voc20"]["panel"]["inputs"]
    cached = read_json(inputs/"VOC20_C_S5_INPUTS.json")
    indices = [0, len(clean)-1]
    output.mkdir(parents=True)
    reference = snapshot_reference(output)
    write_new(output/"PROTOCOL.json", dict(scope="CPU representative generator parity, not full-population regeneration",
        inputs_prepared_sha256=sha256(inputs/"INPUTS_PREPARED.json"), upstream_commit=UPSTREAM_COMMIT,
        upstream_URL=f"https://github.com/dosowiechi/MLMP/tree/{UPSTREAM_COMMIT}/utils/imagecorruptions",
        reference_sha256=reference, corruption_types=list(CORRUPTIONS), severity=5,
        source_indices=indices, source_ids=[clean[i]["official_id"] for i in indices],
        selection="First and last official source indices, every type, fixed before regeneration; no score/GT/outcome selection",
        process="cv2BGR decode; NumPy seed official sample_idx; pinned upstream corrupt;cv2linear224resize; compare decoded cachedBGR",
        equality="Exact uint8 pixel equality, no tolerance, no mutation/regeneration of cache",
        GT_opened=False, model_loaded=False, GPU_used=False))
    module = load_reference(output/"reference")
    rows = []
    try:
        for index in indices:
            source = clean[index]
            require(sha256(source["image"]) == source["image_sha256"], "Source RGB drift")
            original = cv2.imread(source["image"], cv2.IMREAD_COLOR)
            require(original is not None, "Source RGB decode failed")
            for kind in CORRUPTIONS:
                row = dict(source_index=index, official_id=source["official_id"], corruption=kind)
                state = np.random.get_state()
                try:
                    np.random.seed(index)
                    image = module.corrupt(original, severity=5, corruption_name=kind)
                    generated = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                    bound = cached[kind][index]
                    require(sha256(bound["image"]) == bound["image_sha256"], "Cached parity RGB drift")
                    observed = cv2.imread(bound["image"], cv2.IMREAD_COLOR)
                    require(observed is not None and observed.shape == generated.shape == (224, 224, 3), "Parity shape differs")
                    difference = np.abs(observed.astype(np.int16)-generated.astype(np.int16))
                    row.update(exact_equal=bool(np.array_equal(observed, generated)),
                        changed_channel_values=int(np.count_nonzero(difference)), max_abs=int(difference.max()),
                        observed_pixels_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
                        generated_pixels_sha256=hashlib.sha256(generated.tobytes()).hexdigest())
                except Exception as error:
                    row.update(exact_equal=False, error=f"{type(error).__name__}: {error}")
                finally:
                    np.random.set_state(state)
                rows.append(row)
        result = dict(samples=len(rows), exact_matches=sum(r["exact_equal"] for r in rows),
            all_samples_exact=all(r["exact_equal"] for r in rows), rows=rows,
            versions={package: importlib.metadata.version(package) for package in
                      ("numpy", "scipy", "scikit-image", "Pillow")},
            cv2_version=cv2.__version__, GT_opened=False, model_loaded=False, GPU_used=False,
            scope="Only these30representative source/type pairs; not all21735cache payloads or an exact paper-environment reproduction")
        write_new(output/"RESULT.json", result)
        write_new(output/"ANALYSIS_SEALED.json", dict(complete=True, scope="bounded_CPU_generator_parity_only",
            artifact_sha256={name:sha256(output/name) for name in ("PROTOCOL.json", "RESULT.json")},
            reference_sha256=reference, all_samples_exact=result["all_samples_exact"], new_GPU_started=False))
        return result
    finally:
        for name in list(sys.modules):
            if name == module.__name__ or name.startswith(module.__name__+"."):
                del sys.modules[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.output)
    print({k: result[k] for k in ("samples", "exact_matches", "all_samples_exact", "GT_opened", "GPU_used")})
