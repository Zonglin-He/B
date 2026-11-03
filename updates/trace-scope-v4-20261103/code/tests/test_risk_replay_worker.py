from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.evolution_fifty import risk_replay_worker as worker
from tools.evolution_fifty.intake import sha256, write_new
from tools.evolution_fifty.risk_replay import fixed_ln_state
from tools.evolution_fifty.unit_risk import source_prediction
from test_risk_replay import ReplayMethod


class CPUReplayMethod(ReplayMethod):
    def __init__(self):
        torch.manual_seed(19)
        super().__init__()
        self.device = torch.device("cpu")
        self.work = None

    @contextmanager
    def count_work(self):
        self.work = dict(eval_forward_calls=0, eval_forward_inputs=0, adapt_forward_calls=0)
        try:
            yield self.work
        finally:
            self.work = None

    def evaluate(self, images):
        if self.work is not None:
            self.work["eval_forward_calls"] += 1
            self.work["eval_forward_inputs"] += len(images)
        return super().evaluate(images)


def setup(tmp_path, monkeypatch):
    parent, suite = tmp_path/"parent", tmp_path/"suite"
    suite.mkdir()
    expected = "f"*64
    cfg = dict(resize=[3, 3], patch_size=[3, 3], patch_stride=2, steps=3)
    images = torch.randn((4, 3, 3, 3), generator=torch.Generator().manual_seed(11))
    rows = []
    for i in range(4):
        path = tmp_path/f"rgb{i}.fixture"
        path.write_bytes(f"fixed synthetic RGB{i}".encode())
        rows.append(dict(image=str(path), image_sha256=sha256(path), official_id=f"source{i}",
                         label=str(tmp_path/f"GT{i}_must_not_be_opened")))
    meta = dict(img_path=[r["image"] for r in rows], img_shape=[(3, 3)]*4, patch_grid_shape=[(1, 1)]*4)
    bound = dict(parent=str(parent), datasets={})
    for dataset in ("coco_object", "pc60"):
        native = CPUReplayMethod()
        zero = {n: p.detach().clone() for n, p in native.model.named_parameters() if p.requires_grad}
        inventory, cases = {}, []
        for index in range(2):
            path = parent/dataset/f"case_{index:06d}"
            path.mkdir(parents=True)
            points = []
            for step in (0, 1, 3):
                state = {n: p+step*(index+1)*.1 for n, p in zero.items()}
                snapshot_path = path/f"step_{step:02d}.visual_ln.npz"
                with snapshot_path.open("xb") as stream:
                    np.savez_compressed(stream, **{n: v.numpy() for n, v in state.items()})
                with fixed_ln_state(native, state):
                    labels = source_prediction(native, images, meta, cfg, None, 1)
                label_path = path/f"step_{step:02d}.labels.npy"
                with label_path.open("xb") as stream:
                    np.save(stream, labels, allow_pickle=False)
                a, b = str(snapshot_path.relative_to(parent/dataset)), str(label_path.relative_to(parent/dataset))
                inventory[a], inventory[b] = sha256(snapshot_path), sha256(label_path)
                points.append(dict(step=step, snapshot=a, labels=b, snapshot_sha256=inventory[a], labels_sha256=inventory[b]))
            trajectory = path/"trajectory.json"
            write_new(trajectory, dict(source_id="source1", candidate_id=f"candidate{index}", original_episode=3,
                                      actual_steps=3, gt_contents_opened=False))
            trajectory_name = str(trajectory.relative_to(parent/dataset))
            inventory[trajectory_name] = sha256(trajectory)
            cases.append(dict(case=path.name, episode=3, source_id="source1", candidate_id=f"candidate{index}",
                example=dict(candidate=dict(image_idx=1)), trajectory=trajectory_name, checkpoints=points))
        bound["datasets"][dataset] = dict(spec=dict(native_config=cfg, query_to_class=None,
            queries=["a", "b", "c", "d"], candidate_count=2), episodes={"3":rows}, cases=cases, artifact_sha256=inventory)
    monkeypatch.setattr(worker, "admission", lambda *_: (suite, {}, bound))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=256*1024**3))
    monkeypatch.setattr(worker, "forbid_annotations", lambda labels: pytest.fail("Missing GT deny list") if not labels else None)
    @contextmanager
    def cpu_lock():
        yield dict(path="CPU fixture lock only", exclusive=True)
    monkeypatch.setattr(worker, "gpu_lock", cpu_lock)
    monkeypatch.setattr(worker, "deterministic_runtime", lambda: torch.manual_seed(19))
    monkeypatch.setattr(worker, "build_method", lambda _: CPUReplayMethod())
    monkeypatch.setattr(worker.confirmation, "verify_label_metadata", lambda _: None)
    def data(_cfg, _rows):
        return [dict(img_patches=images[i:i+1], meta={k:v[i] for k,v in meta.items()}) for i in range(4)]
    monkeypatch.setattr(worker.confirmation, "make_data", data)
    return suite, expected, bound


def test_actual_filesystem_replay_stream_dedup_cost_and_all_support_barrier(tmp_path, monkeypatch):
    suite, expected, bound = setup(tmp_path, monkeypatch)
    worker.run(suite, expected)
    barrier = json.loads((suite/"ALL_SUPPORTS_SEALED.json").read_text())
    assert barrier["complete"] and barrier["candidates"] == 4 and not barrier["GT_opened"]
    assert barrier["scientific_completion"] is False
    for dataset, digest in barrier["dataset_seals"].items():
        seal_path = suite/dataset/"SUPPORTS_SEALED.json"
        assert sha256(seal_path) == digest
        seal = json.loads(seal_path.read_text())
        assert seal["candidates"] == 2 and seal["unique_readouts"] == 5
        references = json.loads((suite/dataset/"REFERENCES.json").read_text())
        assert len(references) == 2
        assert references[0]["checkpoints"][0]["reference"] == references[1]["checkpoints"][0]["reference"]
        for case in references:
            for point in case["checkpoints"]:
                ref = point["reference"]
                assert point["exact_changed_labels"] == 0
                assert ref["actual_work"] == dict(eval_forward_calls=1, eval_forward_inputs=4, adapt_forward_calls=0)
        for name, digest in seal["artifact_sha256"].items():
            assert sha256(suite/name) == digest
    assert json.loads((suite/"queue_state.json").read_text())["phase"] == "payloads_complete_awaiting_support_seal_and_exit"
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="No replay resume/retry"):
        worker.run(suite, expected)


def test_tampered_snapshot_in_a_reused_reference_is_not_silently_skipped(tmp_path, monkeypatch):
    suite, expected, bound = setup(tmp_path, monkeypatch)
    item = bound["datasets"]["coco_object"]["cases"][1]["checkpoints"][0]
    path = tmp_path/"parent"/"coco_object"/item["snapshot"]
    path.write_bytes(b"tampered duplicate theta0 snapshot")
    with pytest.raises(ValueError, match="payload changed"):
        worker.run(suite, expected)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_SUPPORTS_SEALED.json").exists()
    assert list((suite/"coco_object"/"supports").glob("*.npz"))


def test_partial_hard_label_failure_preserves_files_and_forbids_barrier_and_retry(tmp_path, monkeypatch):
    suite, expected, bound = setup(tmp_path, monkeypatch)
    point = bound["datasets"]["coco_object"]["cases"][1]["checkpoints"][2]
    path = tmp_path/"parent"/"coco_object"/point["labels"]
    with path.open("wb") as stream:
        np.save(stream, np.zeros((3, 3), np.uint16), allow_pickle=False)
    point["labels_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="changed sealed"):
        worker.run(suite, expected)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_SUPPORTS_SEALED.json").exists()
    assert list((suite/"coco_object"/"supports").glob("*.npz"))
    with pytest.raises(ValueError, match="No replay resume/retry"):
        worker.run(suite, expected)


def test_disk_preflight_before_start_lock_or_model(tmp_path, monkeypatch):
    suite, expected, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=worker.FREE_REQUIRED-1))
    monkeypatch.setattr(worker, "gpu_lock", lambda: pytest.fail("No lock before disk preflight"))
    monkeypatch.setattr(worker, "build_method", lambda _: pytest.fail("No model before disk preflight"))
    with pytest.raises(ValueError, match="storage headroom"):
        worker.run(suite, expected)
    assert not (suite/"STARTED.json").exists()


def test_progress_writer_failure_preserves_failure_and_blocks_all_support_seal(tmp_path, monkeypatch):
    suite, expected, _ = setup(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise TypeError("injected writer failure")
    monkeypatch.setattr(worker, "write_json", fail)
    with pytest.raises(TypeError, match="writer failure"):
        worker.run(suite, expected)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_SUPPORTS_SEALED.json").exists()


def test_sealed_trajectory_identity_is_checked_before_snapshot_readout(tmp_path, monkeypatch):
    suite, expected, bound = setup(tmp_path, monkeypatch)
    source = bound["datasets"]["coco_object"]
    case = source["cases"][0]
    trajectory = tmp_path/"parent"/"coco_object"/case["trajectory"]
    data = json.loads(trajectory.read_text())
    data["source_id"] = "wrong source"
    write_new(tmp_path/"replacement.json", data)
    trajectory.write_bytes((tmp_path/"replacement.json").read_bytes())
    source["artifact_sha256"][case["trajectory"]] = sha256(trajectory)
    monkeypatch.setattr(worker, "replay_one", lambda *args: pytest.fail("Identity drift before readout"))
    with pytest.raises(ValueError, match="trajectory identity"):
        worker.run(suite, expected)


def test_final_progress_failure_never_publishes_global_support_seal(tmp_path, monkeypatch):
    suite, expected, _ = setup(tmp_path, monkeypatch)
    native = worker.write_json
    def fail(path, value, **kwargs):
        if value.get("phase") == "payloads_complete_awaiting_support_seal_and_exit":
            raise RuntimeError("final bookkeeping failed")
        return native(path, value, **kwargs)
    monkeypatch.setattr(worker, "write_json", fail)
    with pytest.raises(RuntimeError, match="bookkeeping"):
        worker.run(suite, expected)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_SUPPORTS_SEALED.json").exists()


def test_execution_lock_exit_failure_never_publishes_global_support_seal(tmp_path, monkeypatch):
    suite, expected, _ = setup(tmp_path, monkeypatch)
    count = 0
    @contextmanager
    def lock():
        nonlocal count
        count += 1
        yield dict(CPU_fixture=True)
        if count == 2:
            raise RuntimeError("execution lock exit failed")
    monkeypatch.setattr(worker, "gpu_lock", lock)
    with pytest.raises(RuntimeError, match="lock exit"):
        worker.run(suite, expected)
    assert (suite/"FAILED.json").exists() and not (suite/"ALL_SUPPORTS_SEALED.json").exists()


def test_shared_disk_headroom_preserved_before_any_support_write(tmp_path, monkeypatch):
    suite, expected, _ = setup(tmp_path, monkeypatch)
    count = 0
    def disk(_):
        nonlocal count
        count += 1
        return SimpleNamespace(free=256*1024**3 if count == 1 else 16*1024**3)
    monkeypatch.setattr(worker.shutil, "disk_usage", disk)
    with pytest.raises(ValueError, match="reserved shared-disk"):
        worker.run(suite, expected)
    assert (suite/"FAILED.json").exists() and not list((suite/"coco_object"/"supports").glob("*.npz"))
