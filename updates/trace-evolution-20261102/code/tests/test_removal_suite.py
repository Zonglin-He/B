"""Frozen removal-suite evidence is checked before accepting completed cells."""

import copy
import fcntl
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_removal_suite import completed_attempt, experiment_plan, validate_artifact
from finalize_removal_suite import control_attempt


def completed_cell(tmp_path):
    expected = dict(dataset="coco_cd", split="evaluation", seed=0, selection_seed=20261024,
        image_ids=["a", "b"], label_ids=["a.png", "b.png"],
        file_metadata=[[1, 2], [3, 4], [5, 6], [7, 8]],
        vocabulary_sha256="vocabulary", available_images=2,
        config=dict(trace_view_batch_size=16, batch_size=2, steps=5))
    protocol = dict(**expected, images=2, full_split=True,
                    source_hashes={"source": "original"}, variant="full")
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    result = dict(status="completed", images=2, full_split=True,
        protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        intersection=[3, 2], union=[4, 4], miou_unrounded=62.5)
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "per_image_counts.json").write_text(json.dumps([
        dict(image_id="a", intersection=[1, 1], union=[2, 2]),
        dict(image_id="b", intersection=[2, 1], union=[2, 2]),
    ]))
    return expected, protocol, result


def test_suite_uses_method_removals_and_only_declared_fast_populations():
    cells = experiment_plan()
    assert len(cells) == 29
    assert len({cell["key"] for cell in cells}) == len(cells)
    assert {cell["dataset"] for cell in cells if cell["stage"] == "full_validation"} == {
        "voc20", "voc21", "pc60", "coco_cd"}
    assert all(cell["dataset"] not in {"cityscapes", "waterbirds_seg"} for cell in cells)
    for cell in cells:
        if cell["stage"] == "ablation":
            assert cell["limit"] == (None if cell["dataset"] == "coco_cd" else 512)


def test_reuse_is_accepted_only_after_exact_source_and_metric_checks(tmp_path):
    expected, protocol, result = completed_cell(tmp_path)
    assert validate_artifact(tmp_path, expected, {"source": "original"}, "full") == result
    with pytest.raises(RuntimeError, match="Algorithm identity"):
        validate_artifact(tmp_path, expected, {"source": "modified"}, "full")
    with pytest.raises(RuntimeError, match="Algorithm identity"):
        validate_artifact(tmp_path, expected, {"source": "original"}, "no_source_tracing")
    changed = copy.deepcopy(expected)
    changed["config"]["trace_view_batch_size"] = 8
    with pytest.raises(RuntimeError, match="batching"):
        validate_artifact(tmp_path, changed, {"source": "original"}, "full")
    changed_result = {**result, "miou_unrounded": 100}
    (tmp_path / "result.json").write_text(json.dumps(changed_result))
    with pytest.raises(RuntimeError, match="mIoU cannot be reproduced"):
        validate_artifact(tmp_path, expected, {"source": "original"}, "full")


def test_scope_and_complete_per_image_records_are_not_inferred_from_a_flag(tmp_path):
    expected, protocol, result = completed_cell(tmp_path)
    records_path = tmp_path / "per_image_counts.json"
    records = json.loads(records_path.read_text())
    records_path.write_text(json.dumps(records[:1]))
    with pytest.raises(RuntimeError, match="incomplete or out of order"):
        validate_artifact(tmp_path, expected, {"source": "original"}, "full")
    records_path.write_text(json.dumps(records))
    (tmp_path / "result.json").write_text(json.dumps({**result, "full_split": False}))
    with pytest.raises(RuntimeError, match="evidence label"):
        validate_artifact(tmp_path, expected, {"source": "original"}, "full")


def test_reuse_rejects_changed_file_identity_and_unknown_environment(tmp_path):
    expected, protocol, result = completed_cell(tmp_path)
    changed = copy.deepcopy(expected)
    changed["file_metadata"][0][0] += 1
    with pytest.raises(RuntimeError, match="input-file identity"):
        validate_artifact(tmp_path, changed, {"source": "original"}, "full")
    with pytest.raises(RuntimeError, match="Execution environment"):
        validate_artifact(tmp_path, expected, {"source": "original"}, "full",
                          environment={"gpu": "a different device"})


def test_resume_adopts_the_earliest_complete_attempt_not_the_best_score(tmp_path):
    for index, score in ((1, 40), (2, 99)):
        directory = tmp_path / f"attempt_{index:03d}"
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps({"mIoU": score}))
    (tmp_path / "attempt_003").mkdir()
    assert completed_attempt(tmp_path).name == "attempt_001"


def test_inherited_queue_lock_survives_loss_of_the_parent_handle(tmp_path):
    lock = (tmp_path / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE, pass_fds=(lock.fileno(),))
    lock.close()
    other = (tmp_path / "queue.lock").open("a")
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.communicate(b"x", timeout=10)
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    other.close()


def test_companion_only_supplies_missing_identity_never_overrides_a_conflict(tmp_path):
    directory = tmp_path / "ablation/coco_cd/full/attempt_001"
    directory.mkdir(parents=True)
    expected, protocol, result = completed_cell(directory)
    companion = tmp_path / "suite_protocol.json"
    companion.write_text(json.dumps({"input_panels": {"case": expected}}))

    def save_protocol(value):
        path = directory / "protocol.json"
        path.write_text(json.dumps(value))
        (directory / "result.json").write_text(json.dumps({**result,
            "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        timestamp = path.stat().st_mtime_ns - 1000
        os.utime(companion, ns=(timestamp, timestamp))

    legacy = {key: value for key, value in protocol.items() if key != "file_metadata"}
    save_protocol(legacy)
    validate_artifact(directory, expected, {"source": "original"}, "full", companion_protocol=companion)
    late = (directory / "protocol.json").stat().st_mtime_ns + 1000
    os.utime(companion, ns=(late, late))
    with pytest.raises(RuntimeError, match="not recorded before"):
        validate_artifact(directory, expected, {"source": "original"}, "full", companion_protocol=companion)
    conflicting = copy.deepcopy(protocol)
    conflicting["file_metadata"][0][0] += 1
    save_protocol(conflicting)
    with pytest.raises(RuntimeError, match="input-file identity changed"):
        validate_artifact(directory, expected, {"source": "original"}, "full", companion_protocol=companion)


def test_finalizer_preserves_incomplete_attempt_and_adopts_completed_control(tmp_path):
    first, needs_run = control_attempt(tmp_path)
    assert needs_run
    (first / "protocol.json").write_text("immutable first attempt")
    original = (first / "protocol.json").read_bytes()
    second, needs_run = control_attempt(tmp_path)
    assert needs_run and second != first
    assert (first / "protocol.json").read_bytes() == original
    (second / "result.json").write_text("completed control")
    adopted, needs_run = control_attempt(tmp_path)
    assert adopted == second and not needs_run
    assert (first / "protocol.json").read_bytes() == original
