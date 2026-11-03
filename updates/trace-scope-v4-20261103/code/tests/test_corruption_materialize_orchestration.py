import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools.evolution_fifty import corruption_materialize as worker
from tools.evolution_fifty.generalization import CORRUPTIONS
from tools.evolution_fifty.intake import sha256, write_new
from tools.evolution_fifty.risk_worker import strict_runtime_inventory


def fixture_suite(tmp_path, monkeypatch, *, fail_import=False):
    runtime = tmp_path/"runtime"
    runtime.mkdir()
    entry = runtime/"fixture_entry.py"
    entry.write_text("# CPU fixture entry path only\n")
    monkeypatch.setattr(worker, "__file__", str(entry))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    reference = tmp_path/"reference"
    reference.mkdir()
    code = ('raise RuntimeError("injected reference load failure")\n' if fail_import else
            'import numpy as np\ndef corrupt(image, **kwargs):\n'
            '    return np.clip(image.astype(int)+np.random.randint(-10,11,image.shape),0,255).astype(np.uint8)\n')
    (reference/"__init__.py").write_text(code)
    rows = []
    for index in range(2):
        image, label = tmp_path/f"source{index}.png", tmp_path/f"annotation{index}.invalid"
        assert cv2.imwrite(str(image), np.full((37, 53, 3), 50+index, np.uint8))
        label.write_bytes(b"GT must stay closed")
        rows.append(dict(official_id=f"source{index}", image=str(image), image_sha256=sha256(image),
            label=str(label), label_metadata=[label.stat().st_size, label.stat().st_mtime_ns]))
    write_new(tmp_path/"SOURCE_INPUTS.json", rows)
    write_new(tmp_path/"PROTOCOL.json", dict(versions=worker.versions(), free_disk_required_bytes=1,
        source_images=2, expected_RGBs=30, corruptions=list(CORRUPTIONS), severity=5))
    bound = dict(root=str(tmp_path), protocol_sha256=sha256(tmp_path/"PROTOCOL.json"),
        source_inputs_sha256=sha256(tmp_path/"SOURCE_INPUTS.json"), runtime_hashes=strict_runtime_inventory(runtime),
        reference_sha256={"reference/__init__.py":sha256(reference/"__init__.py")})
    write_new(tmp_path/"CPU_PREPARED.json", bound)
    return sha256(tmp_path/"CPU_PREPARED.json")


def test_real_CPU_controller_writes_two_source_progress_all_types_and_input_seal(tmp_path, monkeypatch):
    expected = fixture_suite(tmp_path, monkeypatch)
    writes = []
    actual_writer = worker.write_json
    def observe(path, value, **kwargs):
        actual_writer(path, value, **kwargs)
        writes.append(json.loads(Path(path).read_text()))
    monkeypatch.setattr(worker, "write_json", observe)
    worker.run(tmp_path, expected)
    assert [r.get("completed_sources") for r in writes] == [1, 2, None]
    assert writes[-1]["phase"] == "RGB_inputs_sealed_not_experiment_complete"
    seal = json.loads((tmp_path/"INPUTS_SEALED.json").read_text())
    assert seal["complete"] and seal["generated_RGBs"] == 30
    assert not seal["scientific_completion"] and not seal["labels_read"] and not seal["GPU_used"]
    for name, digest in seal["artifact_sha256"].items():
        assert sha256(tmp_path/name) == digest
    rows = json.loads((tmp_path/"VOC20_C_S5_INPUTS.json").read_text())
    assert set(rows) == set(CORRUPTIONS)
    for kind, images in rows.items():
        assert len(images) == 2
        for index, row in enumerate(images):
            assert row["corruption"] == kind and row["original_sample_idx"] == index
            assert row["image_sha256"] == sha256(row["image"])
            assert cv2.imread(row["image"]).shape == (224, 224, 3)
    with pytest.raises(ValueError, match="No CPU generation resume/retry"):
        worker.run(tmp_path, expected)
    assert torch.cuda.is_initialized() is False


def test_reference_load_failure_is_recorded_and_cannot_resume(tmp_path, monkeypatch):
    expected = fixture_suite(tmp_path, monkeypatch, fail_import=True)
    with pytest.raises(RuntimeError, match="reference load failure"):
        worker.run(tmp_path, expected)
    failure = json.loads((tmp_path/"FAILED.json").read_text())
    assert failure["completed_RGBs"] == 0 and failure["automatic_retry"] is False
    assert not (tmp_path/"rgb").exists() and not (tmp_path/"INPUTS_SEALED.json").exists()
    with pytest.raises(ValueError, match="No CPU generation resume/retry"):
        worker.run(tmp_path, expected)


@pytest.mark.parametrize("extra", ["__pycache__/__init__.cpython-310.pyc", "corruptions/__init__.py", "extra.py"])
def test_unlisted_reference_import_sources_or_bytecode_rejected_before_start(tmp_path, monkeypatch, extra):
    expected = fixture_suite(tmp_path, monkeypatch)
    path = tmp_path/"reference"/extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unlisted executable content")
    with pytest.raises(ValueError, match="bytecode/cache|reference inventory drift"):
        worker.run(tmp_path, expected)
    assert not (tmp_path/"STARTED.json").exists() and not (tmp_path/"INPUTS_SEALED.json").exists()


def test_annotation_inode_guard_runs_before_actual_source_read(tmp_path, monkeypatch):
    expected = fixture_suite(tmp_path, monkeypatch)
    image = tmp_path/"source0.png"
    image.unlink()
    image.hardlink_to(tmp_path/"annotation0.invalid")
    with pytest.raises(PermissionError, match="GT alias content read"):
        worker.run(tmp_path, expected)
    failure = json.loads((tmp_path/"FAILED.json").read_text())
    assert failure["completed_RGBs"] == 0 and not (tmp_path/"rgb").exists()
