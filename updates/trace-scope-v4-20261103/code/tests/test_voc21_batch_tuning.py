"""Only batch size changes, and training calibration exclusively selects it."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_simplification import with_batch_size
from run_voc21_batch_tuning import batch_panel, choose_batch, validate_disjoint_panels
import suite_inputs


def trials():
    return [dict(batch_size=value, status="completed", mIoU=40 + value * .1,
                 ms_per_image=1000 / value) for value in range(1, 17)]


def test_batch_override_does_not_mutate_or_change_other_settings():
    config = dict(batch_size=16, lr=.00063, steps=36, trace_max_regions=33,
                  trace_view_batch_size=16)
    panel = dict(config=config, image_ids=["train/a", "train/b"], selection_seed=2027)
    saved = copy.deepcopy(panel)
    updated = batch_panel(panel, 7)
    assert updated["config"] == {**config, "batch_size": 7}
    assert panel == saved and updated["image_ids"] == saved["image_ids"]
    assert with_batch_size(config, None) == config
    with pytest.raises(ValueError):
        with_batch_size(config, 0)


def test_calibration_selection_has_a_global_tolerance_not_a_chaining_rule():
    rows = trials()
    rows[4]["mIoU"] = 60
    rows[6]["mIoU"] = 59.99
    rows[14]["mIoU"] = 59.97
    rows[0]["validation_mIoU"] = 100
    assert choose_batch(rows)["batch_size"] == 7


def test_every_integer_is_tested_and_fatal_failures_are_not_skipped():
    with pytest.raises(RuntimeError, match="all sixteen"):
        choose_batch(trials()[:-1])
    rows = trials()
    rows[2]["status"] = "failed"
    with pytest.raises(RuntimeError, match="Unresolved"):
        choose_batch(rows)
    rows[2] = dict(batch_size=3, status="oom", mIoU=None, ms_per_image=None)
    assert choose_batch(rows)["batch_size"] == 16


def test_training_images_and_labels_cannot_overlap_validation():
    validate_disjoint_panels(dict(image_ids=["train/a"], label_ids=["labels/a"]),
                             dict(image_ids=["val/b"], label_ids=["labels/b"]))
    with pytest.raises(RuntimeError, match="overlaps"):
        validate_disjoint_panels(dict(image_ids=["same"], label_ids=["a"]),
                                 dict(image_ids=["same"], label_ids=["b"]))
    with pytest.raises(RuntimeError, match="annotation"):
        validate_disjoint_panels(dict(image_ids=["train/a"], label_ids=["same"]),
                                 dict(image_ids=["val/b"], label_ids=["same"]))


def test_calibration_input_selection_records_and_uses_seed_2027(tmp_path, monkeypatch):
    from trace_ovss.data import datasets

    pairs = []
    for i in range(24):
        image, label = tmp_path / f"{i:02d}.jpg", tmp_path / f"{i:02d}.png"
        image.touch()
        label.touch()
        pairs.append((image, label))

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.samples = pairs
            self.vocabulary = SimpleNamespace(classes=("a", "b"), extensions=("a", "b"),
                                               extension_to_class=(0, 1))

        def __len__(self):
            return len(pairs)

    monkeypatch.setattr(datasets, "SegmentationPatchDataset", FakeDataset)
    monkeypatch.setattr(suite_inputs, "validate_discovery", lambda data: None)
    root = Path(__file__).resolve().parents[1]
    plan = {"cal": ("voc21", "calibration", 8)}
    panel = suite_inputs.expected_inputs(root, tmp_path, plan, selection_seed=2027)["cal"]
    indices = sorted(np.random.default_rng(2027).permutation(24)[:8])
    assert panel["selection_seed"] == 2027
    assert panel["image_ids"] == [str(pairs[i][0]) for i in indices]
    assert len(panel["file_metadata"]) == 16
