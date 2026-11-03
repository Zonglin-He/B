"""E06/E31 input identity audit only; no model, RGB or label contents read."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

from .bootstrap import TUNED, read_json, require
from .intake import sha256, write_new

METADATA = Path("/home/wwww/hzl/data/waterbirds_seg/correlation_0.95/images/metadata.csv")
CALIBRATION = TUNED / "trials/waterbirds_seg/000_b92875bdb8566c3f501c600d.manifest.json"


def identity(name):
    path = Path(name)
    require(not path.is_absolute() and len(path.parts) == 2 and ".." not in path.parts,
            "Unsafe/unknown CUB identity")
    return "cub:" + path.with_suffix("").as_posix()


def audit(output):
    output = Path(output).absolute()
    require(not output.exists() and output == output.resolve(), "New input-audit directory required")
    full_path = TUNED / "full/waterbirds_seg.manifest.json"
    full = read_json(full_path)
    calibration = read_json(CALIBRATION)
    require(full["stage"] == "full" and full["dataset"] == "waterbirds_seg"
            and calibration["stage"] == "calibration" and calibration["dataset"] == "waterbirds_seg",
            "Wrong parent manifests")
    source = {str(path): sha256(path) for path in (METADATA, full_path, CALIBRATION)}
    with METADATA.open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 5322 and len({r["img_filename"] for r in rows}) == 5322,
            "Metadata population/identity mismatch")
    full_rows = {r["official_id"]: r for r in full["panel"]["inputs"]}
    require(set(full_rows) == {identity(r["img_filename"]) for r in rows}, "Metadata/full identity mismatch")
    calibration_ids = set(calibration["panel"]["ordered_ids"])
    require(len(calibration_ids) == 256, "Calibration population differs")
    splits = Counter(r["split"] for r in rows)
    require(dict(splits) == {"0": 2458, "1": 296, "2": 2568}, "Unexpected declared split counts")
    test_rows = [r for r in rows if r["split"] == "2"]
    test_ids = {identity(r["img_filename"]) for r in test_rows}
    train_ids = {identity(r["img_filename"]) for r in rows if r["split"] == "0"}
    require(calibration_ids <= train_ids and not (test_ids & calibration_ids), "Calibration/test overlap")
    groups = Counter()
    for row in rows:
        require(row["y"] in ("0", "1") and row["place"] in ("0", "1")
                and int(row["group_id"]) == 2 * int(row["y"]) + int(row["place"]),
                "Unverified class/background/group convention")
        if row["split"] == "2":
            groups[row["group_id"]] += 1
    # Preserve the historical ordered source identity restricted to test only,
    # but do not reuse the mixed-population episodes or adapted predictions.
    ordered = [i for i in full["panel"]["ordered_ids"] if i in test_ids]
    panel = dict(full["panel"], ordered_ids=ordered, inputs=[full_rows[i] for i in ordered],
        images=len(ordered), full_split=True, population_protocol="declared_metadata_split2_test_only")
    require(len(ordered) == 2568, "Test identity count differs")
    for path, expected in source.items():
        require(sha256(path) == expected, "Identity metadata changed during audit")
    output.mkdir(parents=True)
    write_new(output / "test_panel.json", panel)
    write_new(output / "groups.json", {identity(r["img_filename"]):
        dict(y=int(r["y"]), place=int(r["place"]), group_id=int(r["group_id"])) for r in test_rows})
    write_new(output / "INPUT_AUDIT.json", dict(experiments=["E06", "E31"],
        stage="metadata_audited_not_inference", source_sha256=source, declared_splits=dict(splits),
        test_count=2568, calibration_count=256, calibration_test_overlap=0,
        test_group_counts=dict(groups), historical_full_includes_all_test=True,
        untouched_test=False, rgb_contents_read=False, label_contents_read=False,
        episodes_must_be_rebuilt=True, new_inference=False,
        metadata_origin="Existing local Waterbirds metadata; external official-source provenance still to be cross-checked",
        config_source=str(full_path), no_test_selection=True))
    write_new(output / "INPUTS_SEALED.json", dict(complete=True, scope="input identity only; E06/E31 not completed",
        artifact_sha256={n: sha256(output / n) for n in ("test_panel.json", "groups.json", "INPUT_AUDIT.json")}))
    print(dict(output=str(output), test_images=len(ordered), groups=dict(groups), inference_started=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)


if __name__ == "__main__":
    main()
