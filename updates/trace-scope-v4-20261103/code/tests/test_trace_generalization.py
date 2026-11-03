import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools.evolution_fifty.generalization import (
    ARMS, CORRUPTIONS, CURVE_TYPES, cache_key, cell_matrix, checked_cache_row,
    corruption_inputs, protocol, visual_layers, label_inodes, read_rgb_bytes, prepare,
)
from tools.evolution_fifty.generalization_data import GeneralizationRGBDataset
from tools.evolution_fifty.intake import sha256
from trace_ovss.data.datasets import CLIP_MEAN, CLIP_STD
from trace_ovss.models.dense_clip import DenseCLIP


def test_matrix_is_prospective_complete_and_has_no_external_baseline_or_duplicate_curve():
    rows = cell_matrix()
    assert len(rows) == len({r["id"] for r in rows}) == 132
    p = protocol()
    assert p["stage_counts"] == {
        "primary": dict(cells=57, image_method_evaluations=104214),
        "second_stage": dict(cells=30, image_method_evaluations=43470),
        "resource_conditional": dict(cells=45, image_method_evaluations=65205),
    }
    assert set(r["arm"] for r in rows) == set(ARMS)
    primary = [r for r in rows if r["stage"] == "primary"]
    assert sum(r["corruption"] != "original" for r in primary) == 45
    assert len([r for r in primary if r["model_type"] == "sclip"]) == 3
    for kind in CURVE_TYPES:
        for arm in ARMS:
            relevant = [r for r in rows if r["backbone"] == "ViT-L/14"
                        and r["corruption"] == kind and r["arm"] == arm]
            assert sorted(r["severity"] for r in relevant) == [1, 3, 5]
    assert p["search_trials"] == 0 and not p["executable_queue"] and not p["scientific_completion"]
    assert p["representative_risk_extension"]["types"] == ["gaussian_noise", "fog"]
    assert p["corruption_groups"]["weather"] == ("snow", "frost", "fog", "brightness")
    assert p["old_cache_usable_for_GPU"] is False and p["fresh_RGB_seal_required"] is True


def test_layers_are_legal_prospective_three_quarters_not_silently_clipped():
    assert visual_layers("ViT-B/16") == list(range(-1, -10, -1))
    assert visual_layers("ViT-L/14") == list(range(-1, -19, -1))
    assert len(DenseCLIP._resolve_layers(visual_layers("ViT-B/16"), 12)) == 9
    with pytest.raises(ValueError, match="out of range"):
        DenseCLIP._resolve_layers(visual_layers("ViT-L/14"), 12)
    with pytest.raises(ValueError, match="Unsupported"):
        visual_layers("ViT-B/32")


def cache_fixture(tmp_path, kind="gaussian_noise", index=0):
    source = tmp_path/"source.jpg"
    source.write_bytes(b"source RGB identity fixture")
    label = tmp_path/"annotation.png"
    label.write_bytes(b"GT must never be decoded")
    row = dict(official_id="voc:source", image=str(source), image_sha256=sha256(source),
               label=str(label), label_metadata=[label.stat().st_size, label.stat().st_mtime_ns])
    root = tmp_path/"cache"
    key = cache_key(str(source), index, kind, 5, (224, 224), source.stat())
    cached = root/"severity_5"/kind/key[:2]/f"{key}.png"
    cached.parent.mkdir(parents=True)
    bgr = np.zeros((224, 224, 3), np.uint8)
    bgr[:] = (11, 23, 177)
    assert cv2.imwrite(str(cached), bgr)
    raw = dict(dataset="voc20", source_path=str(source), sample_idx=str(index), corruption=kind,
        severity="5", resize="224x224", status="hit", cache_path=str(cached),
        bytes=str(cached.stat().st_size), error="")
    return row, root, raw, bgr


def test_cache_identity_binds_source_index_corruption_metadata_and_exact_payload(tmp_path):
    row, root, raw, _ = cache_fixture(tmp_path)
    frozen = checked_cache_row(raw, row, 0, root)
    assert frozen["image_sha256"] == sha256(raw["cache_path"])
    assert frozen["clean_image"] == row["image"] and frozen["image"] != row["image"]
    assert frozen["official_id"] == row["official_id"] and frozen["label"] == row["label"]
    with pytest.raises(ValueError, match="source/index"):
        checked_cache_row(raw, row, 1, root)
    changed = dict(raw, corruption="fog")
    with pytest.raises(ValueError, match="Cache key"):
        checked_cache_row(changed, row, 0, root)
    Path(row["image"]).write_bytes(b"source changed")
    with pytest.raises(ValueError, match="Cache key"):
        checked_cache_row(raw, row, 0, root)


def test_cache_coverage_rejects_duplicate_and_missing_not_silently_skipping(tmp_path):
    row, root, raw, _ = cache_fixture(tmp_path)
    def write_manifest(records):
        with (root/"manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(raw))
            writer.writeheader()
            writer.writerows(records)
    write_manifest([raw])
    with pytest.raises(ValueError, match="Incomplete"):
        corruption_inputs([row], root)
    write_manifest([raw, raw])
    with pytest.raises(ValueError, match="Duplicate"):
        corruption_inputs([row], root)


def config(tmp_path):
    return dict(dataset="PascalVOC20Dataset", data_dir=str(tmp_path), resize=[224, 224],
                patch_size=[224, 224], patch_stride=112)


def test_corrupted_rgb_reader_has_no_GT_or_clean_pixel_read_and_no_second_resize(tmp_path, monkeypatch):
    row, root, raw, bgr = cache_fixture(tmp_path)
    frozen = checked_cache_row(raw, row, 0, root)
    actual_open = Path.open
    def audited_open(path, *args, **kwargs):
        assert str(path) not in (row["image"], row["label"])
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", audited_open)
    def no_resize(*args, **kwargs):
        pytest.fail("Already resized corruption must not be resized again")
    monkeypatch.setattr(cv2, "resize", no_resize)
    data = GeneralizationRGBDataset(config(tmp_path), [frozen])
    item = data[0]
    expected = (torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float()-CLIP_MEAN)/CLIP_STD
    assert set(item) == {"img_patches", "meta"}
    torch.testing.assert_close(item["img_patches"][0], expected, rtol=0, atol=0)
    assert item["img_patches"].shape == (1, 3, 224, 224)
    assert torch.cuda.is_initialized() is False


def test_reader_hash_guard_and_annotation_inode_alias(tmp_path):
    row, root, raw, _ = cache_fixture(tmp_path)
    frozen = checked_cache_row(raw, row, 0, root)
    data = GeneralizationRGBDataset(config(tmp_path), [frozen])
    Path(frozen["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="RGB drift"):
        data[0]
    alias = tmp_path/"hardlinked.png"
    alias.hardlink_to(Path(row["label"]))
    with pytest.raises(ValueError, match="aliases annotation"):
        GeneralizationRGBDataset(config(tmp_path), [dict(frozen, image=str(alias))])


def test_clean_reader_matches_cv2_not_PIL_resize_and_uses_RGB(tmp_path):
    source = tmp_path/"clean.png"
    bgr = np.random.default_rng(7).integers(0, 256, size=(31, 49, 3), dtype=np.uint8)
    assert cv2.imwrite(str(source), bgr)
    label = tmp_path/"label.invalid"
    label.write_bytes(b"invalidGT")
    row = dict(official_id="source", image=str(source), image_sha256=sha256(source), label=str(label))
    actual = GeneralizationRGBDataset(config(tmp_path), [row])[0]["img_patches"][0]
    expected_rgb = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR)[..., ::-1].copy()
    expected = (torch.from_numpy(expected_rgb).permute(2, 0, 1).float()-CLIP_MEAN)/CLIP_STD
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("hardlink", [False, True])
def test_preparation_annotation_alias_rejected_before_open(tmp_path, monkeypatch, hardlink):
    row, root, raw, _ = cache_fixture(tmp_path)
    path = Path(raw["cache_path"])
    path.unlink()
    if hardlink:
        path.hardlink_to(row["label"])
    else:
        row["label"] = str(path)
        path.write_bytes(b"GT payload must not be opened")
    raw["bytes"] = str(path.stat().st_size)
    deny = label_inodes([row])
    real_open = Path.open
    def guarded_open(p, *args, **kwargs):
        if (p.stat().st_dev, p.stat().st_ino) in deny:
            pytest.fail("Annotation inode reached open before alias rejection")
        return real_open(p, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="aliases annotation"):
        checked_cache_row(raw, row, 0, root)
    with pytest.raises(ValueError, match="aliases annotation"):
        read_rgb_bytes(path, deny)


def test_preparation_rejects_alias_to_another_sources_annotation(tmp_path):
    row, root, raw, _ = cache_fixture(tmp_path)
    other = tmp_path/"other_label"
    other.write_bytes(b"OtherGT")
    path = Path(raw["cache_path"])
    path.unlink()
    path.hardlink_to(other)
    raw["bytes"] = str(path.stat().st_size)
    with pytest.raises(ValueError, match="aliases annotation"):
        checked_cache_row(raw, row, 0, root, forbidden_inodes=label_inodes([dict(label=str(other))]))


def test_rejected_old_cache_cannot_be_prepared_as_standard_inputs_by_default(tmp_path):
    output = tmp_path/"must_not_be_created"
    with pytest.raises(ValueError, match="Historical cache failed generator parity"):
        prepare(output)
    assert not output.exists()
