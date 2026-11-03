"""Narrow STARTED ownership regression for new unfrozen M03/M04 scorers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import test_factorial_score_worker as factorial
import test_legacy_score_worker as legacy
import test_value_score_worker as owned_cases


@pytest.mark.parametrize("study",[legacy,factorial],ids=["M03","M04"])
@pytest.mark.parametrize("fault",["fsync","close","close_mutation","fstat","identity"])
def test_exclusive_marker_faults_have_owned_FAILED_and_no_GT(tmp_path,monkeypatch,study,fault):
    # Reuse the already closed V01 failure injector, not its source fixture or
    # permission: the actual respective controller admission and run execute.
    monkeypatch.setattr(owned_cases,"worker",study.worker)
    monkeypatch.setattr(owned_cases,"fixture",study.fixture)
    owned_cases.test_actual_exclusive_STARTED_creation_fault_is_owned_without_GT(tmp_path,monkeypatch,fault)


@pytest.mark.parametrize("study",[legacy,factorial],ids=["M03","M04"])
@pytest.mark.parametrize("fault",["timer","exclusions"])
def test_all_postcreation_prelude_faults_are_owned(tmp_path,monkeypatch,study,fault):
    suite,digest,_=study.fixture(tmp_path,monkeypatch)
    worker=study.worker
    def fail(*_a,**_k):raise RuntimeError("owned prelude failed")
    def denied(*_a,**_k):pytest.fail("Prelude failure reached execute or GT")
    monkeypatch.setattr(worker,"execute",denied)
    monkeypatch.setattr(worker.payload,"annotation_bytes",denied)
    if fault == "timer":monkeypatch.setattr(worker,"time",SimpleNamespace(perf_counter=fail))
    else:
        owner,key=(worker,"source_exclusions") if study is legacy else (worker.payload,"data_exclusions")
        original=getattr(owner,key)
        def exclusions(*args,**kwargs):
            if (suite/"STARTED.json").exists():fail()
            return original(*args,**kwargs)
        monkeypatch.setattr(owner,key,exclusions)
    with pytest.raises(RuntimeError,match="owned prelude"):worker.run(suite,digest)
    assert (suite/"STARTED.json").exists() and (suite/"FAILED.json").exists()
    assert not (suite/"ANALYSIS_SEALED.json").exists()


@pytest.mark.parametrize("study",[legacy,factorial],ids=["M03","M04"])
@pytest.mark.parametrize("kind",["regular","hardlink","symlink","directory"])
def test_foreign_raced_markers_are_not_claimed_or_modified(tmp_path,monkeypatch,study,kind):
    suite,digest,_=study.fixture(tmp_path,monkeypatch)
    worker=study.worker
    marker,victim=suite/"STARTED.json",tmp_path/"foreign_victim"
    victim.write_bytes(b"foreign bytes")
    original,created=Path.open,[]
    def identity(path):
        s=path.lstat()
        return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def opened(path,*args,**kwargs):
        if path == marker and args == ("xb",):
            if kind == "directory":marker.mkdir()
            elif kind == "hardlink":marker.hardlink_to(victim)
            elif kind == "symlink":marker.symlink_to(victim)
            else:
                with original(marker,"xb") as stream:stream.write(b"foreign marker")
            created.append((identity(marker),identity(victim)))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",opened)
    with pytest.raises(OSError):worker.run(suite,digest)
    assert len(created) == 1 and created[0] == (identity(marker),identity(victim))
    assert victim.read_bytes() == b"foreign bytes"
    assert not (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


