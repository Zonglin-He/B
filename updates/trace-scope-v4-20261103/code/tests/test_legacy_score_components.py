import hashlib
import gzip
import io
import sys
import tarfile

import pytest

from tools.evolution_fifty import legacy_score_components as loader


def fixture(monkeypatch,*,fault=None):
    contents = dict(zip(loader.MEMBERS,[
        b"EPS=1e-8\n",
        b"def class_roles_for(dataset,classes):\n return tuple('background' if x=='background' else 'stuff' if x=='road' else 'thing' for x in classes)\ndef extend_class_roles(base,mapping):\n return tuple(base[i] for i in mapping)\n",
        b"from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Score:\n quality:float\n dominance:float\n reliability:float\ndef trace_reliability(target,context,residual,role):\n quality=min(target,context)\n dominance=max(residual-target,0.) if role=='thing' else 0.\n return Score(quality,dominance,quality*(1.-dominance))\n",
        b"# Synthetic indexing component; no models or files\n",
        b"def trace_view_types(role):\n return ('target','residual','counterfactual') if role=='thing' else ('target','counterfactual')\ndef old_oom_iterator():\n raise AssertionError('Historical iterator must never be selected')\n",
    ],strict=True))
    if fault == "compile":contents[loader.MEMBERS[-1]] = b"raise RuntimeError('synthetic component compile failure')\n"
    if fault == "oversize_member":contents[loader.MEMBERS[0]] = b"#"+b"x"*(256*1024)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw,mode="w:gz") as archive:
        for name,content in contents.items():
            if fault == "missing" and name == loader.MEMBERS[-1]:continue
            item = tarfile.TarInfo(name)
            item.size = len(content)
            if fault == "valid_pax":item.pax_headers = {"comment":"x"*1024}
            if fault == "symlink" and name == loader.MEMBERS[0]:
                item.type,item.linkname,item.size = tarfile.SYMTYPE,"outside",0
                archive.addfile(item)
            else:archive.addfile(item,io.BytesIO(content))
            if fault == "duplicate" and name == loader.MEMBERS[0]:archive.addfile(item,io.BytesIO(content))
        other = b"raise AssertionError('unrelated historical engine must not compile')\n"
        info = tarfile.TarInfo("trace_ovss/method/engine.py")
        info.size = len(other)
        archive.addfile(info,io.BytesIO(other))
    content = raw.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(loader,"ARCHIVE_SHA",digest)
    spec = dict(sha256=digest,member_sha256={n:hashlib.sha256(c).hexdigest() for n,c in contents.items()},
        episode_objective_sha256=loader.OBJECTIVE_SHA,old_engine_loaded=False,
        old_optimizer_loaded=False,old_OOM_iterator_selected=False)
    return content,spec


def test_loads_only_five_verified_synthetic_components_from_bytes_no_file_opens(monkeypatch):
    content,spec = fixture(monkeypatch)
    current = set(sys.modules)
    result = loader.load_components_bytes(content,spec)
    assert set(result["modules"]) == {"constants","roles","reliability","indexing","interventions"}
    assert result["archive_sha256"] == spec["sha256"]
    score = result["modules"]["reliability"].trace_reliability(.8,.6,.9,"thing")
    assert score.reliability == pytest.approx(.54)
    assert result["modules"]["roles"].class_roles_for("PascalContext60Dataset",["background","person","road"]) == ("background","thing","stuff")
    assert result["modules"]["interventions"].trace_view_types("stuff") == ("target","counterfactual")
    created = set(sys.modules)-current
    assert created and all(n.startswith("_trace_scoring_pinned_") for n in created)
    assert not any(n.endswith(".engine") for n in created)


@pytest.mark.parametrize("fault",["raw_hash","member_hash","missing","duplicate","symlink","oversize_member","scope","objective","too_large"])
def test_invalid_scope_or_bytes_reject_before_any_namespace_exec(monkeypatch,fault):
    content,spec = fixture(monkeypatch,fault=fault)
    if fault == "raw_hash":content += b"changed"
    elif fault == "member_hash":spec["member_sha256"][loader.MEMBERS[0]] = "0"*64
    elif fault == "scope":spec["old_engine_loaded"] = True
    elif fault == "objective":spec["episode_objective_sha256"] = "0"*64
    elif fault == "too_large":content = b"x"*(1024**2+1)
    before = set(sys.modules)
    with pytest.raises(ValueError):loader.load_components_bytes(content,spec)
    assert set(sys.modules) == before


def test_component_compile_failure_removes_every_partial_private_module(monkeypatch):
    content,spec = fixture(monkeypatch,fault="compile")
    before = set(sys.modules)
    with pytest.raises(RuntimeError,match="synthetic component"):
        loader.load_components_bytes(content,spec)
    assert set(sys.modules) == before


def test_bounded_valid_PAX_metadata_still_loads_all_five_components(monkeypatch):
    content,spec = fixture(monkeypatch,fault="valid_pax")
    assert len(loader.load_components_bytes(content,spec)["modules"]) == 5


@pytest.mark.parametrize("kind",[tarfile.XHDTYPE,tarfile.XGLTYPE,tarfile.GNUTYPE_LONGNAME,tarfile.GNUTYPE_LONGLINK])
@pytest.mark.parametrize("extent",[512*1024,33*1024**2])
def test_extended_headers_reject_before_internal_tar_parsing_or_namespace(monkeypatch,kind,extent):
    content,spec = fixture(monkeypatch)
    item = tarfile.TarInfo("extended-header")
    item.type,item.size = kind,extent
    prefix = item.tobuf(format=tarfile.USTAR_FORMAT)
    raw = prefix+(b"x"*extent if extent < 1024**2 else b"")+gzip.decompress(content)
    content = gzip.compress(raw,mtime=0)
    digest = hashlib.sha256(content).hexdigest()
    spec["sha256"] = digest
    monkeypatch.setattr(loader,"ARCHIVE_SHA",digest)
    def denied(*args,**kwargs):pytest.fail("Unbounded tar parser reached")
    monkeypatch.setattr(loader.tarfile,"open",denied)
    before = set(sys.modules)
    with pytest.raises(ValueError,match="header extent|extended tar header"):
        loader.load_components_bytes(content,spec)
    assert set(sys.modules) == before


def test_actual_expansion_cap_precedes_tar_parser(monkeypatch):
    content,spec = fixture(monkeypatch)
    content = gzip.compress(gzip.decompress(content)+bytes(loader.MAX_EXPANDED),mtime=0)
    spec["sha256"] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(loader,"ARCHIVE_SHA",spec["sha256"])
    monkeypatch.setattr(loader.tarfile,"open",lambda *a,**k:pytest.fail("Expanded archive reached tar parser"))
    with pytest.raises(ValueError,match="gzip expanded extent"):
        loader.load_components_bytes(content,spec)
