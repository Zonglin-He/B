"""CPU startup regression tests; no CUDA initialization or model loading."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.evolution import run_cell as worker


@pytest.fixture(autouse=True)
def restore_global_settings(monkeypatch):
    saved_environment = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    saved = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cuda.flash_sdp_enabled())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA must not initialize in CPU tests"))
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic = saved[2]
        torch.backends.cudnn.benchmark = saved[3]
        torch.backends.cuda.enable_flash_sdp(saved[4])
        if saved_environment is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved_environment


def test_missing_environment_is_fixed_and_actual_flags_are_recorded(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    sdp_before = worker.deterministic_runtime_state()["sdpa_enabled"]
    actual = worker.configure_deterministic_runtime()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert actual == worker.deterministic_runtime_state()
    assert actual["deterministic_algorithms"] is True
    assert actual["deterministic_warn_only"] is False
    assert actual["deterministic_debug_mode"] == 2
    assert actual["cudnn_deterministic"] is True
    assert actual["cudnn_benchmark"] is False
    assert actual["sdpa_enabled"] == sdp_before


def test_exact_existing_environment_is_accepted_and_warning_only_is_disabled(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert worker.configure_deterministic_runtime()["deterministic_warn_only"] is False


@pytest.mark.parametrize("value", ["", ":16:8", "4096:8", ":4096:8 ", "invalid"])
def test_conflicting_environment_fails_before_any_flag_change(monkeypatch, value):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    before = worker.deterministic_runtime_state()
    with pytest.raises(RuntimeError, match="must be exactly"):
        worker.configure_deterministic_runtime()
    assert worker.deterministic_runtime_state() == before


def test_already_initialized_cuda_is_rejected_without_mutation(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    before = worker.deterministic_runtime_state()
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        worker.configure_deterministic_runtime()
    assert worker.deterministic_runtime_state() == before


def test_configuration_does_not_force_math_or_change_tf32(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.enable_flash_sdp(True)
    before = worker.deterministic_runtime_state()["sdpa_enabled"]
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    worker.configure_deterministic_runtime()
    assert worker.deterministic_runtime_state()["sdpa_enabled"] == before
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == tf32


def test_torch_flags_can_be_restored_without_changing_policy(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    before = worker.deterministic_runtime_state()
    worker.configure_deterministic_runtime()
    torch.use_deterministic_algorithms(before["deterministic_algorithms"],
                                       warn_only=before["deterministic_warn_only"])
    torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
    assert worker.deterministic_runtime_state() == before


def test_predict_configures_before_package_import_or_model_setup(monkeypatch):
    calls = []

    def configure():
        calls.append("configured")
        return {}

    class Root:
        def resolve(self):
            assert calls == ["configured"]
            raise RuntimeError("CPU startup sentinel")

    monkeypatch.setattr(worker, "configure_deterministic_runtime", configure)
    with pytest.raises(RuntimeError, match="CPU startup sentinel"):
        worker.predict(SimpleNamespace(package_root=Root()))


def test_score_phase_does_not_configure_or_touch_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "configure_deterministic_runtime",
                        lambda: pytest.fail("Score must not configure CUDA"))
    seen = []
    monkeypatch.setattr(worker, "score", lambda output, **kwargs: seen.append((output, kwargs)))
    worker.main(["--package-root", str(tmp_path), "--data-root", str(tmp_path), "--output", str(tmp_path),
                 "--dataset", "coco_cd", "--phase", "score"])
    assert len(seen) == 1
