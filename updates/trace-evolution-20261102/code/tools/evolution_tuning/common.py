"""Pure selection and evidence helpers; never imports or executes a model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


DATASETS = ("voc20", "voc21", "pc59", "pc60", "coco_object", "coco_stuff", "coco_cd", "waterbirds_seg")
ORDER = ("voc20", "voc21", "pc59", "pc60", "coco_cd", "coco_object", "coco_stuff", "waterbirds_seg")
COORDINATES = ("batch_size", "lr", "steps", "trace_max_regions")
FULL_ARMS = ["no_adapt", "image_uniform", "image_confidence", "image_C"]
ALGORITHM = "ImageBalancedTRACE:C"
MAX_TRIALS = 80
TOLERANCE = .02
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        path = root / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(root)
                and not path.is_symlink() and sha256(path) == expected, f"Artifact/source drift: {name}")


def config_key(config):
    return canonical_hash(config)[:24]


def coarse_values(coordinate, initial):
    if coordinate == "batch_size":
        values = list(range(1, 17))
    elif coordinate == "lr":
        values = [1e-5, 3e-5, .0001, .0003, .0005, .001, .002, .003, .005, .008, .01, .015, .02]
    elif coordinate == "steps":
        values = [1, 3, 5, 8, 12, 16, 24, 32, 48]
    elif coordinate == "trace_max_regions":
        values = [1, 2, 4, 8, 16, 24, 32, 48]
    else:
        raise ValueError(coordinate)
    return sorted(set(values + [initial[coordinate]]))


def fine_values(coordinate, winner):
    value = winner[coordinate]
    if coordinate == "lr":
        return sorted({float(f"{value * factor:.8g}") for factor in (.5, .75, 1, 1.25, 1.5, 2)
                       if 1e-6 <= value * factor <= .04})
    offsets = range(-3, 4) if coordinate == "batch_size" else range(-2, 3)
    limit = 32 if coordinate == "batch_size" else 64
    return sorted({value + offset for offset in offsets if 1 <= value + offset <= limit})


def choose(rows):
    """Global best with one fixed tolerance, not cumulative coordinate losses."""
    valid = [row for row in rows if row["status"] == "completed"]
    require(valid, "No valid completed calibration trial")
    require(all(math.isfinite(row["mIoU"]) and math.isfinite(row["ms_per_image"])
                and row["ms_per_image"] > 0 for row in valid), "Nonfinite selection evidence")
    maximum = max(row["mIoU"] for row in valid)
    eligible = [row for row in valid if row["mIoU"] >= maximum - TOLERANCE]
    return min(eligible, key=lambda r: (r["ms_per_image"], *[r["config"][k] for k in COORDINATES]))


def pooled_miou(intersection, union):
    require(len(intersection) == len(union) and bool(union), "Invalid count dimensions")
    require(all(type(i) is int and type(u) is int and 0 <= i <= u
                for i, u in zip(intersection, union)), "Invalid pooled counts")
    values = [i / u for i, u in zip(intersection, union) if u]
    require(values, "All evaluation classes have zero union")
    return 100 * math.fsum(values) / len(values)
