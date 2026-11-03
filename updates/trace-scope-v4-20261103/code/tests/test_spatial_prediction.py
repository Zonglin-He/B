import copy
import json

import pytest
import torch

from tools.evolution_fifty import spatial_prediction as worker
from test_spatial_runtime import method
from test_image_full_worker import config,batch_meta


def fixture(tmp_path,monkeypatch):
    current, cfg = method(),config(2)
    rows = [dict(official_id=f"synthetic{i}",image=f"synthetic{i}.rgb") for i in range(3)]
    spec = dict(dataset="coco_object",config=cfg,modes=list(worker.runtime.MODES),
        preprocessing="native_PIL_bilinear",backbone="ViT-L/14",model_type="naclip",
        vocabulary=dict(classes=current.classes,query_to_class=None))
    monkeypatch.setitem(worker.POPULATIONS,"coco_object",3)
    def batch(data,start,stop,method,cfg):
        torch.manual_seed(start+11)
        images = torch.randn(stop-start,3,32,32)
        meta = batch_meta(stop-start)
        meta["img_path"] = [r["image"] for r in rows[start:stop]]
        return images,meta,None,torch.arange(stop-start)
    monkeypatch.setattr(worker.prediction,"episode_batch",batch)
    return current,rows,spec


def test_actual_four_mode_prediction_core_keeps_every_source_and_candidate_identity(tmp_path,monkeypatch):
    current,rows,spec = fixture(tmp_path,monkeypatch)
    states = []
    result = worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),states.append)
    assert result["payloads_complete"] and result["scientific_completion"] is False
    assert result["image_method_evaluations"] == 12 and result["GT_opened"] is False
    assert len(result["artifact_sha256"]) == 12+8+3
    assert states[-1]["completed_sources"] == 3
    for episode in range(2):
        records = [json.loads((tmp_path/"coco_object/episodes"/f"{episode:06d}.{m}.json").read_text())
                   for m in worker.runtime.MODES]
        assert len({r["candidate_sha256"] for r in records}) == 1
        assert all(r["ordered_ids"] == [v["official_id"] for v in rows[episode*2:episode*2+2]] for r in records)
    assert not list(tmp_path.rglob("*SEALED*"))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault",["population","preprocessing","modes"])
def test_spatial_core_rejects_scope_change_before_output(tmp_path,monkeypatch,fault):
    current,rows,spec = fixture(tmp_path,monkeypatch)
    if fault == "population":
        rows = rows[:-1]
    elif fault == "preprocessing":
        spec["preprocessing"] = "cv2_bilinear"
    else:
        spec["modes"] = spec["modes"][:1]
    with pytest.raises(ValueError):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
    assert not list(tmp_path.iterdir())


def test_spatial_core_failure_preserves_partial_payloads_without_seal(tmp_path,monkeypatch):
    current,rows,spec = fixture(tmp_path,monkeypatch)
    original = worker.runtime.run_arm
    def fail(method,mode,*args):
        if mode == "translated":
            raise RuntimeError("synthetic fault")
        return original(method,mode,*args)
    monkeypatch.setattr(worker.runtime,"run_arm",fail)
    with pytest.raises(RuntimeError,match="synthetic fault"):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
    assert (tmp_path/"coco_object/predictions/native/000000.npy").exists()
    assert not list(tmp_path.rglob("*SEALED*"))


def test_full_spatial_core_cannot_claim_updates_for_noop_optimizer(tmp_path,monkeypatch):
    current,rows,spec = fixture(tmp_path,monkeypatch)
    current.optimizer.step = lambda *args,**kwargs: None
    with pytest.raises(ValueError,match="schedule"):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
    assert not list(tmp_path.rglob("*.npy"))
    assert not list(tmp_path.rglob("*SEALED*"))
