import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tools.evolution import run_cell as shared
from tools.evolution_fifty import risk_worker as worker
from test_unit_risk import AuditMethod, candidate


def test_real_gate_matches_independent_direct_mean_reference_on_CPU(monkeypatch):
    torch.manual_seed(17)
    method = AuditMethod()
    images = torch.randn(2, 3, 4, 4)
    monkeypatch.setattr(worker, "source_prediction", lambda m, x, *args:
                        m.forward_adapt_logits(x)[0, 1].argmax(0).detach().numpy())
    example = dict(candidate=candidate(), source_id="source", candidate_id="id")
    report = worker.real_gate(method, images, example, {}, {}, None, shared.rng_state(method.device))
    assert report["passed"] and report["steps"] == 3 and report["changed_labels"] == 0
    assert max(report[k] for k in ("loss_max_abs", "gradient_max_abs", "parameter_max_abs", "adam_max_abs")) <= 1e-5
    assert shared.restored_state(method)


def test_duplicate_or_missing_candidate_population_rejected():
    row = dict(episode=0, source_id="source", candidate_id="id")
    assert list(worker.groups(dict(examples=[row], candidate_count=1))) == [0]
    with pytest.raises(ValueError):
        worker.groups(dict(examples=[row, copy.deepcopy(row)], candidate_count=2))
    with pytest.raises(ValueError):
        worker.groups(dict(examples=[row], candidate_count=2))


def test_GT_audit_blocks_direct_path_and_inode_alias_in_fresh_CPU_process(tmp_path):
    label, alias = tmp_path/"label.npy", tmp_path/"alias.npy"
    np.save(label, np.ones((2, 2), dtype=np.uint8))
    alias.hardlink_to(label)
    code = '''
import json, sys
from pathlib import Path
import numpy as np
from tools.evolution_fifty.risk_worker import forbid_annotations
label, alias = map(Path, sys.argv[1:])
forbid_annotations([str(label)])
blocked = []
for operation in (lambda: label.read_bytes(), lambda: open(alias, 'rb').read(), lambda: np.load(label)):
    try:
        operation()
    except PermissionError:
        blocked.append(True)
print(json.dumps(blocked))
'''
    child = subprocess.run([sys.executable, "-B", "-c", code, str(label), str(alias)],
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == [True, True, True]


def test_admission_refuses_unfrozen_worker_before_model_work(tmp_path):
    from tools.evolution_fifty.intake import write_new, sha256
    write_new(tmp_path/"SUITE_PREPARED.json", dict(root=str(tmp_path), runtime_hashes={}))
    with pytest.raises(ValueError, match="exact frozen runtime"):
        worker.admission(tmp_path, sha256(tmp_path/"SUITE_PREPARED.json"))


def checkpoint_spec(tmp_path):
    from tools.evolution_fifty.intake import sha256
    path = tmp_path/"checkpoint.pt"
    path.write_bytes(b"known-weights")
    return dict(checkpoint=dict(path=str(path), bytes=path.stat().st_size,
                               mtime_ns=path.stat().st_mtime_ns, sha256=sha256(path)),
                native_config=dict(lr=.001, steps=1, trace_max_regions=4, trace_view_batch_size=16),
                queries=["a", "b"])


@pytest.mark.parametrize("drift", ["missing", "content", "mtime"])
def test_checkpoint_drift_rejected_before_constructor(tmp_path, monkeypatch, drift):
    import os
    spec = checkpoint_spec(tmp_path)
    path = Path(spec["checkpoint"]["path"])
    if drift == "missing":
        path.unlink()
    elif drift == "content":
        path.write_bytes(b"other-weights")
        os.utime(path, ns=(spec["checkpoint"]["mtime_ns"], spec["checkpoint"]["mtime_ns"]))
    else:
        stamp = spec["checkpoint"]["mtime_ns"]+100000
        os.utime(path, ns=(stamp, stamp))
    called = []
    monkeypatch.setattr(worker.confirmation, "ExperimentTRACE", lambda **kwargs: called.append(True))
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.build_method(spec)
    assert called == []


def test_checkpoint_change_inside_constructor_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = checkpoint_spec(tmp_path)
    def constructor(**kwargs):
        Path(spec["checkpoint"]["path"]).write_bytes(b"other-weights")
        return SimpleNamespace(model=torch.nn.Identity())
    monkeypatch.setattr(worker.confirmation, "ExperimentTRACE", constructor)
    with pytest.raises(ValueError, match="Checkpoint"):
        worker.build_method(spec)


def test_same_checkpoint_constructed_twice_and_revalidated_between_phases(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = checkpoint_spec(tmp_path)
    called = []
    def constructor(**kwargs):
        called.append(kwargs["backbone"])
        return SimpleNamespace(model=torch.nn.Identity())
    monkeypatch.setattr(worker.confirmation, "ExperimentTRACE", constructor)
    first = worker.build_method(spec)
    assert first.model.training is False
    Path(spec["checkpoint"]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        worker.build_method(spec)
    assert len(called) == 1


@pytest.mark.parametrize("drift", ["added_package", "removed", "changed", "symlink"])
def test_full_runtime_inventory_rejects_all_drift_classes(tmp_path, drift):
    from tools.evolution_followup.queue import source_inventory
    for name in ("tools", "tests", "trace_ovss"):
        (tmp_path/name).mkdir()
    original = tmp_path/"tools"/"unit_risk.py"
    original.write_text("known = 1\n")
    expected = source_inventory(tmp_path)
    if drift == "added_package":
        package = tmp_path/"tools"/"unit_risk"
        package.mkdir()
        (package/"__init__.py").write_text("shadow = True\n")
    elif drift == "removed":
        original.unlink()
    elif drift == "changed":
        original.write_text("known = 2\n")
    else:
        (tmp_path/"tests"/"alias.py").symlink_to(original)
    if drift == "symlink":
        with pytest.raises((ValueError, RuntimeError), match="symlink"):
            source_inventory(tmp_path)
    else:
        assert source_inventory(tmp_path) != expected


@pytest.mark.parametrize("extra", ["source_package", "sourceless_package", "normal_bytecode_cache", "root_module"])
def test_actual_admission_rejects_import_shadow_package(tmp_path, monkeypatch, extra):
    from tools.evolution_followup.queue import source_inventory
    from tools.evolution_fifty.intake import write_new, sha256
    runtime = tmp_path/"runtime"
    for name in ("tools", "tests", "trace_ovss"):
        (runtime/name).mkdir(parents=True)
    entry = runtime/"tools"/"risk_worker.py"
    entry.write_text("known = 1\n")
    (runtime/"tools"/"unit_risk.py").write_text("known = 2\n")
    inventory = source_inventory(runtime)
    write_new(tmp_path/"PROTOCOL.json", {})
    write_new(tmp_path/"SUITE_PREPARED.json", dict(root=str(tmp_path), runtime_hashes=inventory,
        protocol_sha256=sha256(tmp_path/"PROTOCOL.json"), datasets={}))
    expected_sha = sha256(tmp_path/"SUITE_PREPARED.json")
    write_new(tmp_path/"ADMITTED.json", dict(decision="ADMITTED", suite_prepared_sha256=expected_sha,
        independent_reviewer="synthetic_test_only", new_gpu_allowed=True))
    monkeypatch.setattr(worker, "__file__", str(entry))
    worker.admission(tmp_path, expected_sha)
    if extra == "root_module":
        (runtime/"sitecustomize.py").write_text("untrusted = True\n")
    elif extra == "source_package":
        shadow = runtime/"tools"/"unit_risk"
        shadow.mkdir()
        (shadow/"__init__.py").write_text("shadow = True\n")
    else:
        import py_compile
        source = tmp_path/"untrusted.py"
        source.write_text("untrusted = True\n")
        if extra == "sourceless_package":
            shadow = runtime/"tools"/"unit_risk"
            shadow.mkdir()
            destination = shadow/"__init__.pyc"
        else:
            shadow = runtime/"tools"/"__pycache__"
            shadow.mkdir()
            destination = shadow/"unit_risk.cpython-310.pyc"
        py_compile.compile(str(source), cfile=str(destination), doraise=True)
    with pytest.raises(ValueError, match="inventory membership|bytecode/cache"):
        worker.admission(tmp_path, expected_sha)
