import json

import pytest
import torch

from tools.evolution_fifty import factorial_prediction as worker
from test_factorial_runtime import method
from test_image_full_worker import config,batch_meta


def fixture(tmp_path,monkeypatch):
    current,cfg = method(),config(2)
    rows = [dict(official_id=f"synthetic{i}",image=f"synthetic{i}.rgb") for i in range(3)]
    spec = dict(dataset="coco_object",config=cfg,studies=list(worker.STUDIES),factorial_arms=list(worker.runtime.ARMS),
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


def test_full_factorial_stream_keeps_two_studies_four_arms_all_sources_and_receipts(tmp_path,monkeypatch):
    current,rows,spec = fixture(tmp_path,monkeypatch)
    states = []
    result = worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),states.append)
    assert result["payloads_complete"] and result["scientific_completion"] is False
    assert result["image_method_evaluations"] == 24 and result["GT_opened"] is False
    assert len(result["artifact_sha256"]) == 24+16+3
    assert states[-1]["completed_sources"] == 3
    for episode in range(2):
        records = [json.loads((tmp_path/"coco_object/episodes"/f"{episode:06d}.{mode}.json").read_text())
                   for mode in worker.MODES]
        assert len({r["candidate_sha256"] for r in records}) == 1
        for r in records:
            assert r["all_arms_measure_C"] and not r["production_Confidence_timing"]
            assert r["ordered_ids"] == [v["official_id"] for v in rows[episode*2:episode*2+2]]
            expected = r["assignment"]["arms"][r["factorial_arm"]]["active_candidate_ids"]
            assert len(r["native_dtype_budget"]) == len(expected)
    assert not list(tmp_path.rglob("*SEALED*"))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault",["population","preprocessing","studies","arms"])
def test_factorial_core_rejects_scope_drift_before_outputs(tmp_path,monkeypatch,fault):
    current,rows,spec = fixture(tmp_path,monkeypatch)
    if fault == "population":
        rows = rows[:-1]
    elif fault == "preprocessing":
        spec["preprocessing"] = "cv2_bilinear"
    elif fault == "studies":
        spec["studies"] = ["natural"]
    else:
        spec["factorial_arms"] = spec["factorial_arms"][:1]
    with pytest.raises(ValueError):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
    assert not list(tmp_path.iterdir())


def test_partial_factorial_failure_never_publishes_seal_or_resumes(tmp_path,monkeypatch):
    current,rows,spec = fixture(tmp_path,monkeypatch)
    original = worker.runtime.run_arm
    def fail(method,arm,*args):
        if arm == "Dc_Ok":
            raise RuntimeError("synthetic factorial failure")
        return original(method,arm,*args)
    monkeypatch.setattr(worker.runtime,"run_arm",fail)
    with pytest.raises(RuntimeError,match="synthetic factorial"):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
    assert (tmp_path/"coco_object/predictions/natural.Dk_Ok/000000.npy").exists()
    assert not list(tmp_path.rglob("*SEALED*"))
    with pytest.raises(FileExistsError):
        worker.execute_dataset(current,rows,rows,spec,tmp_path,dict(bytes=0),lambda _:None)
