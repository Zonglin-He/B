"""G03-only CPU primitives; no IO, queue, admission or model execution.

The future controller must independently bind the pinned upstream reference,
clean source bytes and the existing S5 seal before using these primitives.
Manifest validation checks declared provenance, not actual payload integrity.
"""

from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath

import cv2
import numpy as np

from .bootstrap import require
from .generalization import ARMS, CORRUPTIONS, CURVE_TYPES, cell_matrix

SOURCES = 1449
SEVERITIES = (1,3)
MAX_SOURCE_BYTES = 128*1024**2
MAX_PNG_BYTES = 2*1024**2
PREPROCESSING = "already_corrupted_and_cv2_resized_no_second_resize"


def conditions():
    rows = tuple(dict(id=f"{kind}.s{severity}",corruption=kind,severity=severity)
                 for kind in CURVE_TYPES for severity in SEVERITIES)
    planned = [r for r in cell_matrix() if r["stage"] == "second_stage"]
    require(len(rows) == 10 and len(planned) == 30
            and {(r["corruption"],r["severity"],r["arm"]) for r in planned}
                == {(r["corruption"],r["severity"],arm) for r in rows for arm in ARMS},
            "G03 fixed second-stage matrix differs")
    return rows


def plan():
    return dict(conditions=list(conditions()),source_images=SOURCES,new_RGBs=SOURCES*10,
        future_cells=30,future_image_method_evaluations=SOURCES*30,
        reused_S5_types=list(CURVE_TYPES),new_severities=list(SEVERITIES),
        preprocessing="Original BGR -> pinned corrupt with NumPy seed original index -> OpenCV linear224 -> lossless PNG",
        labels="Unchanged original annotation metadata only; no GT content or selection",
        RNG="Independent original sample index for each type/severity; restore even on failure",
        executable_queue=False,GPU_allowed=False,scientific_completion=False)


def generate_image(module,bgr,source_index,corruption,severity):
    require(isinstance(bgr,np.ndarray) and bgr.dtype == np.uint8 and bgr.ndim == 3
            and bgr.shape[2] == 3 and min(bgr.shape[:2]) > 0 and bgr.nbytes <= MAX_SOURCE_BYTES,
            "G03 expects bounded decoded original BGR uint8")
    require(type(source_index) is int and 0 <= source_index < SOURCES
            and corruption in CURVE_TYPES and type(severity) is int and severity in SEVERITIES,
            "Invalid fixed G03 generation condition")
    # Isolate the shared clean image even if an upstream implementation fails
    # after mutating its input; successful mutation is also rejected.
    pixels = bgr.copy()
    state = np.random.get_state()
    try:
        np.random.seed(source_index)
        degraded = module.corrupt(pixels,severity=severity,corruption_name=corruption)
        require(np.array_equal(pixels,bgr), "G03 reference mutated original input")
        require(isinstance(degraded,np.ndarray) and degraded.dtype == np.uint8
                and degraded.shape == bgr.shape, "G03 reference output differs from original geometry/dtype")
        resized = cv2.resize(degraded,(224,224),interpolation=cv2.INTER_LINEAR)
        ok,encoded = cv2.imencode(".png",resized)
        require(ok and isinstance(encoded,np.ndarray) and encoded.dtype == np.uint8
                and 0 < encoded.nbytes <= MAX_PNG_BYTES, "Invalid bounded G03 PNG encoding")
        require(np.array_equal(cv2.imdecode(encoded,cv2.IMREAD_COLOR),resized), "G03 PNG roundtrip differs")
        return encoded.tobytes()
    finally:
        np.random.set_state(state)


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical_root(value):
    require(type(value) is str and PurePosixPath(value).is_absolute()
            and str(PurePosixPath(value)) == value and ".." not in PurePosixPath(value).parts,
            "G03 manifest root must be lexically canonical absolute path")
    return PurePosixPath(value)


def validate_sources(sources):
    require(type(sources) is list and len(sources) == SOURCES
            and all(type(r) is dict and type(r.get("official_id")) is str for r in sources)
            and len({r["official_id"] for r in sources}) == SOURCES,
            "G03 full ordered clean population required")
    images,labels = set(),set()
    for row in sources:
        require(digest(row.get("image_sha256")) and type(row.get("label_metadata")) is list
                and len(row["label_metadata"]) == 2
                and all(type(v) is int and v >= 0 for v in row["label_metadata"]),
                "G03 original hash/annotation metadata declaration differs")
        image,label = str(canonical_root(row["image"])),str(canonical_root(row["label"]))
        require(image not in images and label not in labels, "G03 duplicate clean source or label path")
        images.add(image)
        labels.add(label)
    require(not images.intersection(labels), "G03 clean RGB/GT pathname roles overlap")
    return images|labels


def same_provenance(actual,expected):
    """JSON provenance is exact in type as well as value; false is not zero."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (all(type(k) is str for k in actual) and all(type(k) is str for k in expected)
                and set(actual) == set(expected)
                and all(same_provenance(actual[k],v) for k,v in expected.items()))
    if type(expected) is list:
        return len(actual) == len(expected) and all(same_provenance(a,e) for a,e in zip(actual,expected,strict=True))
    if type(expected) is float:
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected
    return type(expected) in (str,int,bool,type(None)) and actual == expected


def validate_manifest(manifest,sources,root):
    """No file opens/stat: compare every declared row with its clean source."""
    root = canonical_root(root)
    grid = conditions()
    source_paths = validate_sources(sources)
    require(type(manifest) is dict and set(manifest) == {r["id"] for r in grid}, "G03 condition membership differs")
    seen = set()
    for condition in grid:
        rows = manifest[condition["id"]]
        require(type(rows) is list and len(rows) == SOURCES, "G03 condition population differs")
        for index,(source,row) in enumerate(zip(sources,rows,strict=True)):
            require(type(row) is dict and digest(source.get("image_sha256")) and digest(row.get("image_sha256")),
                    "G03 clean/generated digest declaration differs")
            path = str(root/"rgb"/condition["corruption"]/f"severity_{condition['severity']}"/f"{index:06d}.png")
            expected = copy.deepcopy(source)
            expected.update(image=path,image_sha256=row["image_sha256"],clean_image=source["image"],
                clean_image_sha256=source["image_sha256"],corruption=condition["corruption"],
                severity=condition["severity"],original_sample_idx=index,preprocessing=PREPROCESSING)
            require(same_provenance(row,expected),
                    "G03 source order/provenance/annotation/geometry declaration differs")
            require(path not in seen and path not in source_paths, "G03 duplicate or source-aliased output pathname")
            seen.add(path)
    return dict(valid=True,conditions=len(grid),new_RGBs=len(seen),GT_opened=False,
                payload_hashes_verified=False,filesystem_identity_verified=False)


def validate_S5_reuse(selected,sources):
    """Verify five selected existing S5 metadata lists; never regenerate S5."""
    source_paths = validate_sources(sources)
    require(type(selected) is dict and set(selected) == set(CURVE_TYPES),
            "G03 S5 reuse needs exact five types and full clean population")
    seen = set()
    for kind in CURVE_TYPES:
        require(kind in CORRUPTIONS and type(selected[kind]) is list and len(selected[kind]) == SOURCES,
                "G03 S5 reuse population differs")
        for index,(source,row) in enumerate(zip(sources,selected[kind],strict=True)):
            require(type(source) is dict and type(row) is dict and digest(source.get("image_sha256"))
                    and digest(row.get("image_sha256")) and type(row.get("image")) is str,
                    "G03 S5 reuse digest declaration differs")
            expected = copy.deepcopy(source)
            expected.update(image=row["image"],image_sha256=row["image_sha256"],clean_image=source["image"],
                clean_image_sha256=source["image_sha256"],corruption=kind,severity=5,
                original_sample_idx=index,preprocessing=PREPROCESSING)
            require(same_provenance(row,expected),
                    "G03 S5 reuse source/annotation/provenance differs")
            canonical_root(row["image"])
            require(row["image"] not in seen and row["image"] not in source_paths, "G03 S5 reuse path repeats or aliases source")
            seen.add(row["image"])
    return dict(valid=True,reference_RGBs=len(seen),S5_regenerated=False,payload_or_seal_verified=False)
