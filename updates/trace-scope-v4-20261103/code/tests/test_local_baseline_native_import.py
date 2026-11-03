"""Actual archived package imports, without constructing or forwarding a model."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.evolution_local_baselines import worker

ARCHIVE = Path("/home/wwww/hzl/results/trace_local_baselines_20261030_0008")


@pytest.mark.parametrize("method", list(worker.NATIVE))
@pytest.mark.parametrize("ordering", ["normal", "archive_later"])
def test_real_archived_method_import_in_fresh_cpu_process(method, ordering):
    script = r'''
import importlib.abc
import inspect
import json
from pathlib import Path
import sys
from unittest import mock
import torch
from tools.evolution_local_baselines import worker

root = Path(sys.argv[1])
prepared = worker.read_json(root / "SUITE_PREPARED.json")
method = sys.argv[2]
if sys.argv[3] == "archive_later":
    sys.path[:0] = ["/home/wwww/hzl/results/reviewer_response_20260929/baseline_source", str(root / "baseline_source")]
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cfc_tta" or fullname.startswith("cfc_tta.") or fullname == "adapt.cfctta":
            raise AssertionError("Unrelated CFCTTA import attempted: " + fullname)
sys.meta_path.insert(0, Forbidden())
manifest = dict(baseline_root=str(root / "baseline_source"), method=method,
    baseline_source_sha256=prepared["baseline_hashes"])
assert not torch.cuda.is_initialized()
with mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA")), \
     mock.patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("No model construction")), \
     mock.patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("No forward")):
    cls = worker.load_native_class(manifest)
    again = worker.load_native_class(manifest)
assert cls is again and not torch.cuda.is_initialized()
assert not any(n == "cfc_tta" or n.startswith("cfc_tta.") or n == "adapt.cfctta" for n in sys.modules)
assert sys.modules["adapt"].__spec__.origin is None
assert getattr(sys.modules["adapt"], "__file__", None) is None
assert worker.sha256(root / "baseline_source/adapt/__init__.py") == prepared["baseline_hashes"]["adapt/__init__.py"]
print(json.dumps(dict(method=method, class_file=inspect.getfile(cls), cuda_initialized=False,
    constructed_models=0, forwards=0, cfctta_imported=False)))
'''
    import os
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ARCHIVE), method, ordering],
        text=True, capture_output=True, timeout=60,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["method"] == method and report["constructed_models"] == 0
    assert not report["cuda_initialized"] and not report["cfctta_imported"]


def test_unknown_method_and_hash_drift_fail_before_import(monkeypatch):
    prepared = worker.read_json(ARCHIVE / "SUITE_PREPARED.json")
    manifest = dict(baseline_root=str(ARCHIVE / "baseline_source"), method="cfctta",
                    baseline_source_sha256=prepared["baseline_hashes"].copy())
    monkeypatch.setattr(worker, "verify_native_imports", lambda root: None)
    with pytest.raises(RuntimeError, match="Unknown native"):
        worker.load_native_class(manifest)
    manifest["method"] = "tent"
    manifest["baseline_source_sha256"]["adapt/tent.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source hash"):
        worker.load_native_class(manifest)
