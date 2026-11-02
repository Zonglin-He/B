"""Service-state lookup failures must not launch dependent GPU controls."""

from types import SimpleNamespace

import pytest

from tools.evolution_replay import controller


@pytest.mark.parametrize("returncode,text", [(1, ""), (0, ""),
    (0, "MainPID=0\nResult=success\n"),
    (0, "LoadState=not-found\nMainPID=0\nActiveState=inactive\nSubState=dead\nResult=success\n")])
def test_service_query_fails_closed(monkeypatch, returncode, text):
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=returncode, stdout=text, stderr="synthetic failure"))
    with pytest.raises(RuntimeError):
        controller.service_state("parent.service")


def test_loaded_service_returns_actual_fields(monkeypatch):
    content = "LoadState=loaded\nMainPID=0\nActiveState=inactive\nSubState=dead\nResult=success\n"
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=content, stderr=""))
    assert controller.service_state("parent.service")["Result"] == "success"
