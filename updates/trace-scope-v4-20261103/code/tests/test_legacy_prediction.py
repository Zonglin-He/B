import json

import pytest
import torch

from tools.evolution_fifty import legacy_prediction as worker
from test_legacy_runtime import method, config, batch_meta


def fixture(monkeypatch):
    current, cfg = method(), config(2)
    rows = [dict(official_id=f"synthetic{i}", image=f"synthetic{i}.rgb") for i in range(3)]
    spec = dict(dataset="coco_object", config=cfg, modes=list(worker.runtime.MODES),
                preprocessing="native_PIL_bilinear", backbone="ViT-L/14", model_type="naclip",
                vocabulary=dict(classes=current.classes, queries=current.classes, query_to_class=None))
    monkeypatch.setitem(worker.POPULATIONS, "coco_object", 3)
    def batch(data, start, stop, method, cfg):
        torch.manual_seed(start+11)
        meta = batch_meta(stop-start)
        meta["img_path"] = [r["image"] for r in rows[start:stop]]
        return torch.randn(stop-start,3,32,32), meta, None, torch.arange(stop-start)
    monkeypatch.setattr(worker.prediction, "episode_batch", batch)
    return current, rows, spec


def test_all_four_modes_stream_every_source_and_preserve_candidates(tmp_path, monkeypatch):
    current, rows, spec = fixture(monkeypatch)
    states = []
    result = worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), states.append)
    assert result["payloads_complete"] and not result["scientific_completion"]
    assert result["image_method_evaluations"] == 12 and not result["GT_opened"]
    assert len(result["artifact_sha256"]) == 12+8+3
    for episode in range(2):
        records = [json.loads((tmp_path/"coco_object/episodes"/f"{episode:06d}.{m}.json").read_text())
                   for m in worker.runtime.MODES]
        assert len({r["candidate_sha256"] for r in records}) == 1
        assert all(r["ordered_ids"] == [v["official_id"] for v in rows[episode*2:episode*2+2]] for r in records)
        assert all(r["legacy_archive_sha256"] == worker.runtime.legacy_rho.ARCHIVE_SHA for r in records)
    assert states[-1]["completed_sources"] == 3
    assert not list(tmp_path.rglob("*SEALED*")) and not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["population", "preprocessing", "modes", "taxonomy", "mapping", "queries"])
def test_scope_or_vocabulary_drift_rejects_before_output(tmp_path, monkeypatch, fault):
    current, rows, spec = fixture(monkeypatch)
    if fault == "population": rows = rows[:-1]
    elif fault == "preprocessing": spec["preprocessing"] = "cv2_bilinear"
    elif fault == "modes": spec["modes"] = spec["modes"][:1]
    elif fault == "taxonomy": spec["config"]["dataset"] = "PascalContext60Dataset"
    elif fault == "mapping": spec["vocabulary"]["query_to_class"] = [1,0]
    else: spec["vocabulary"]["queries"] = list(reversed(current.classes))
    with pytest.raises(ValueError):
        worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), lambda _:None)
    assert not list(tmp_path.iterdir())


def test_failure_preserves_partial_outputs_without_seal_or_resume(tmp_path, monkeypatch):
    current, rows, spec = fixture(monkeypatch)
    original = worker.runtime.run_arm
    def fail(method, mode, *args):
        if mode == "legacy.image": raise RuntimeError("synthetic M03 fault")
        return original(method, mode, *args)
    monkeypatch.setattr(worker.runtime, "run_arm", fail)
    with pytest.raises(RuntimeError, match="synthetic M03 fault"):
        worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), lambda _:None)
    assert (tmp_path/"coco_object/predictions/C.image/000000.npy").exists()
    assert not list(tmp_path.rglob("*SEALED*"))


def test_noop_optimizer_cannot_publish_predictions(tmp_path, monkeypatch):
    current, rows, spec = fixture(monkeypatch)
    current.optimizer.step = lambda *args,**kwargs:None
    with pytest.raises(ValueError, match="schedule"):
        worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), lambda _:None)
    assert not list(tmp_path.rglob("*.npy"))


@pytest.mark.parametrize("short,bound", [("coco_object","PascalContext60Dataset"),
                                       ("pc60","COCOObjectDataset")])
def test_consistent_wrong_taxonomy_cannot_claim_other_dataset_before_work(tmp_path, monkeypatch, short, bound):
    base, rows, spec = fixture(monkeypatch)
    current = worker.runtime.LegacyExperimentTRACE(dataset_name=bound, benchmark_classes=base.classes,
        query_to_class=None, classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, prompts=base.prompt_templates, model=base.model,
        tokenize_fn=base.tokenize, device="cpu", view_mode="C")
    current.model.eval()
    spec["dataset"], spec["config"]["dataset"] = short, bound
    monkeypatch.setitem(worker.POPULATIONS, short, 3)
    def forbidden(*args, **kwargs):
        pytest.fail("Wrong dataset namespace reached model/reset/batch work")
    monkeypatch.setattr(current, "reset", forbidden)
    monkeypatch.setattr(current.model, "forward", forbidden)
    monkeypatch.setattr(current.optimizer, "step", forbidden)
    monkeypatch.setattr(worker.prediction, "episode_batch", forbidden)
    with pytest.raises(ValueError, match="vocabulary"):
        worker.execute_dataset(current, rows, rows, spec, tmp_path, dict(bytes=0), lambda _:None)
    assert not list(tmp_path.iterdir())
