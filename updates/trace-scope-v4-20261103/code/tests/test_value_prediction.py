import json

import pytest
import torch

from tools.evolution_fifty import value_prediction as worker
from test_generalization_runtime import method
from test_image_full_worker import config, batch_meta


def fixture(monkeypatch, dataset="pc60"):
    current, cfg = method(), config(2)
    current.steps = cfg["steps"] = worker.runtime.NATIVE_STEPS[dataset]
    cfg["dataset"] = "COCOObjectDataset" if dataset == "coco_object" else "PascalContext60Dataset"
    rows = [dict(official_id=f"synthetic{i}", image=f"synthetic{i}.rgb") for i in range(3)]
    spec = dict(dataset=dataset, config=cfg, points=list(worker.runtime.points(dataset)),
        backbone="ViT-L/14", model_type="naclip", preprocessing="native_PIL_bilinear",
        vocabulary=dict(classes=current.classes, queries=current.classes, query_to_class=None))
    monkeypatch.setitem(worker.POPULATIONS, dataset, 3)
    def batch(data, start, stop, method, cfg):
        torch.manual_seed(start+11)
        meta = batch_meta(stop-start)
        meta["img_path"] = [r["image"] for r in rows[start:stop]]
        return torch.randn(stop-start,3,32,32), meta, None, torch.arange(stop-start)
    monkeypatch.setattr(worker.prediction, "episode_batch", batch)
    return current, rows, spec


@pytest.mark.parametrize("dataset", ["coco_object", "pc60"])
def test_entire_predeclared_grid_actual_updates_and_source_inventory(tmp_path, monkeypatch, dataset):
    current, rows, spec = fixture(monkeypatch, dataset)
    before = worker.shared.restored_state(current)
    progress = []
    result = worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), progress.append)
    assert result["payloads_complete"] and result["scientific_completion"] is False
    assert result["image_method_evaluations"] == 15 and len(result["artifact_sha256"]) == 15+10+3
    assert worker.shared.restored_state(current) == before and current.steps == spec["config"]["steps"]
    for episode in range(2):
        records = [json.loads((tmp_path/dataset/"episodes"/f"{episode:06d}.{point['id']}.json").read_text())
                   for point in spec["points"]]
        assert len({row["candidate_sha256"] for row in records}) == 1
        assert all(row["optimizer_steps"] == row["requested_steps"] for row in records)
        assert all(row["arm"] == row["point_id"] and row["weighting_arm"] == row["fixed_point"]["arm"] for row in records)
        assert all(row["auxiliary_checkpoint_readouts"] == 0 for row in records)
    assert progress[-1]["completed_sources"] == 3
    assert not list(tmp_path.rglob("*SEALED*")) and not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["population", "taxonomy", "grid", "preprocessing", "queries"])
def test_scope_change_rejects_before_output(tmp_path, monkeypatch, fault):
    current, rows, spec = fixture(monkeypatch)
    if fault == "population": rows = rows[:-1]
    elif fault == "taxonomy": spec["config"]["dataset"] = "COCOObjectDataset"
    elif fault == "grid": spec["points"] = spec["points"][:-1]
    elif fault == "preprocessing": spec["preprocessing"] = "cv2_bilinear"
    else: spec["vocabulary"]["queries"] = list(reversed(current.classes))
    with pytest.raises(ValueError): worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), lambda _:None)
    assert not list(tmp_path.iterdir())


def test_noop_optimizer_never_publishes_predictions(tmp_path, monkeypatch):
    current, rows, spec = fixture(monkeypatch)
    current.optimizer.step = lambda *a, **k:None
    with pytest.raises(ValueError): worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), lambda _:None)
    assert not list(tmp_path.rglob("*.npy")) and not list(tmp_path.rglob("*SEALED*"))


def test_midgrid_failure_keeps_partials_without_seal(tmp_path, monkeypatch):
    current, rows, spec = fixture(monkeypatch)
    original, calls = worker.runtime.run_point, []
    def fail(method, dataset, point, *args, **kwargs):
        calls.append(point["id"])
        if len(calls) == 2: raise RuntimeError("synthetic V01 failure")
        return original(method, dataset, point, *args, **kwargs)
    monkeypatch.setattr(worker.runtime, "run_point", fail)
    with pytest.raises(RuntimeError, match="synthetic V01 failure"):
        worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), lambda _:None)
    assert list(tmp_path.rglob("*.npy")) and not list(tmp_path.rglob("*SEALED*"))


def test_collator_duplicate_IDs_reject_before_point_adaptation(tmp_path,monkeypatch):
    current,rows,spec = fixture(monkeypatch)
    original = worker.prediction.episode_batch
    def collate(*args):
        images,meta,valid,ids = original(*args)
        return images,meta,valid,torch.zeros_like(ids)
    monkeypatch.setattr(worker.prediction,"episode_batch",collate)
    monkeypatch.setattr(worker.runtime,"run_point",lambda *a,**k:pytest.fail("Invalid batch reached V01 point"))
    with pytest.raises(ValueError,match="positional"):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
    assert not list(tmp_path.rglob("*.npy")) and not list(tmp_path.rglob("*SEALED*"))


def test_float_serialized_grid_rejects_before_output(tmp_path,monkeypatch):
    current,rows,spec = fixture(monkeypatch)
    spec["points"][0]["steps"] = float(spec["points"][0]["steps"])
    with pytest.raises(ValueError,match="grid"):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
    assert not list(tmp_path.iterdir())
