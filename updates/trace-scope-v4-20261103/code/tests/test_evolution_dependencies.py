"""Appending a dataset must not acquire GPU work ahead of frozen dependencies."""

import json
from pathlib import Path

import pytest

from tools import self_evolution_controller as controller
from tools.evolution_replay import controller as replay


def test_default_plan_is_unchanged():
    assert len(controller.plan()) == 18
    assert controller.plan()[0]["key"] == "smoke/coco_cd"
    assert controller.plan()[-1]["key"] == "mechanism/coco_stuff"


def test_voc20_append_runs_only_trace_with_full_replays():
    cells = controller.plan(["voc20"], trace_only=True)
    assert [c["key"] for c in cells] == ["smoke/voc20", "mechanism/voc20",
        "repetition/voc20/seed1", "repetition/voc20/seed2"]
    assert all(c["policies"] == ["trace"] for c in cells)
    assert [c["limit"] for c in cells] == [4, None, None, None]


@pytest.fixture
def dependency(tmp_path, monkeypatch):
    parent, output = tmp_path / "parent", tmp_path / "output"
    parent.mkdir()
    output.mkdir()
    controller.write_json(parent / "manifest.json", {"cells": [{}, {}]})
    controller.write_json(parent / "queue_state.json", dict(status="completed", completed=2, total=2))
    controller.write_json(parent / "SUITE_SEALED.json", dict(artifact_sha256={
        "manifest.json": controller.digest(parent / "manifest.json"),
        "queue_state.json": controller.digest(parent / "queue_state.json")}))
    manifest = dict(cells=controller.plan(["voc20"], True), dependencies=[dict(
        path=str(parent), service="parent.service", total=2,
        manifest_sha256=controller.digest(parent / "manifest.json"))])
    service = dict(MainPID="0", Result="success", ActiveState="inactive", SubState="dead")
    monkeypatch.setattr(replay, "service_state", lambda _: service)
    return parent, output, manifest, service


def test_completed_dependency_has_sealed_receipt(dependency):
    parent, output, manifest, _ = dependency
    controller.wait_dependencies(output, manifest)
    receipt = json.loads((output / "dependencies.json").read_text())
    assert receipt[0]["seal_sha256"] == controller.digest(parent / "SUITE_SEALED.json")


def test_wait_does_not_accept_running_service_even_with_completed_status(dependency, monkeypatch):
    _, output, manifest, service = dependency
    service["MainPID"] = "123"
    def stop_wait(_):
        raise TimeoutError("still waiting")
    monkeypatch.setattr(controller.time, "sleep", stop_wait)
    with pytest.raises(TimeoutError):
        controller.wait_dependencies(output, manifest)
    assert json.loads((output / "queue_state.json").read_text())["status"] == "waiting_for_dependencies"
    assert not (output / "dependencies.json").exists()


def test_wait_for_cache_consumer_before_rechecking_parent_quiescence(dependency, monkeypatch):
    parent, output, manifest, service = dependency
    from tools import evolution_completion
    live = manifest["dependencies"][0]
    receipt_parent = dict(live, completion_receipt={"path": "not-yet-admitted", "sha256": "unused"})
    manifest["dependencies"] = [receipt_parent, live]
    service["MainPID"] = "123"
    def forbidden(*args):
        pytest.fail("A live cache consumer may still reference the completed parent's runtime")
    monkeypatch.setattr(evolution_completion, "validate_receipt", forbidden)
    def stop_wait(_):
        raise TimeoutError("waiting for cache consumer")
    monkeypatch.setattr(controller.time, "sleep", stop_wait)
    with pytest.raises(TimeoutError):
        controller.wait_dependencies(output, manifest)
    assert not (output / "dependencies.json").exists()


@pytest.mark.parametrize("damage", ["manifest", "failed", "incomplete", "seal", "count"])
def test_bad_dependency_cannot_start(dependency, damage):
    parent, output, manifest, service = dependency
    if damage == "manifest":
        controller.write_json(parent / "manifest.json", {})
    elif damage == "failed":
        service["Result"] = "oom-kill"
    elif damage == "incomplete":
        controller.write_json(parent / "queue_state.json", dict(status="running"))
    elif damage == "seal":
        controller.write_json(parent / "SUITE_SEALED.json", {"artifact_sha256": {"manifest.json": "bad"}})
    else:
        controller.write_json(parent / "queue_state.json", dict(status="completed", completed=1, total=2))
    with pytest.raises(RuntimeError):
        controller.wait_dependencies(output, manifest)
    assert not (output / "dependencies.json").exists()
