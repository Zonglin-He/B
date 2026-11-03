import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_full_baseline_followup as followup


def test_pc60_baselines_first_and_excluded_methods_stay_excluded():
    cells = followup.plan()
    assert len(cells) == 16
    assert all(c["dataset"] == "pc60" for c in cells[:4])
    assert [c["method"] for c in cells[:4]] == ["mlmp", "tent", "tpt", "noadapt"]
    assert all(c["panel"] == c["dataset"] + "/full" for c in cells)
    assert not {"watt", "clipartt"}.intersection(c["method"] for c in cells)


def test_followup_freezes_analysis_and_both_baseline_helpers(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    for name in followup.FILES:
        (code / name).write_text("# frozen\n")
    (tmp_path / "code_seal.json").write_text(json.dumps({n: followup.sha(code / n) for n in followup.FILES}))
    followup.verify_code(tmp_path)
    (code / "benchmark_matched_baselines.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="code changed"):
        followup.verify_code(tmp_path)


def test_changed_early_result_cannot_be_resealed(tmp_path):
    for name in ("result.json", "protocol.json", "cost.csv", "per_image_counts.json"):
        (tmp_path / name).write_text("original")
    frozen = followup.accepted_hashes(tmp_path)
    followup.check_accepted(tmp_path, frozen)
    (tmp_path / "result.json").write_text("changed")
    with pytest.raises(RuntimeError, match="Previously accepted"):
        followup.check_accepted(tmp_path, frozen)


def test_post_analysis_launch_error_cannot_change_completed_state(tmp_path, monkeypatch):
    state = tmp_path / "queue_state.json"
    state.write_text('{"status":"completed"}')
    before = state.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("cannot start child")
    monkeypatch.setattr(followup.subprocess, "run", fail)
    followup.run_analysis(tmp_path, tmp_path / "parent")
    assert state.read_bytes() == before
    assert json.loads((tmp_path / "analysis_status.json").read_text())["status"] == "failed"
    monkeypatch.setattr(followup, "write_json", fail)
    followup.run_analysis(tmp_path, tmp_path / "parent")
    assert state.read_bytes() == before
