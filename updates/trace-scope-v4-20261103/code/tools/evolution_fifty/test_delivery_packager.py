"""Synthetic /tmp packaging tests only, never real publication/admission proof.

Paper archives/reviews, two exporter pins, license text and first-party source
trees below are stubs. No research source is imported from those fixtures, and
no actual result tree, paper archive, RGB/GT/NPY, reference-mutation test,
model, service, Git or network is accessed.
"""
import copy
from dataclasses import replace
import gzip
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from tools.evolution_fifty import delivery_packager as p


def save(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(raw, bytes):
        raw = p.encoded(raw)
    path.write_bytes(raw)
    return dict(path=str(path), sha256=p.sha(raw), bytes=len(raw))


def archive_bytes(members):
    result = io.BytesIO()
    with gzip.GzipFile(fileobj=result, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for name, raw in members:
                header = tarfile.TarInfo(name)
                header.size = len(raw)
                tar.addfile(header, io.BytesIO(raw))
    return result.getvalue()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    work, results = tmp_path/"hzl", tmp_path/"results"
    work.mkdir(); results.mkdir()
    policy = p.Policy(worktree=work, results=results, metadata_count=2, v002_members=6,
        v002_archive_sha256=None, review_sha256={}, disposition_sha256=None)
    limits = replace(p.Limits(), reserve=0)
    for folder in p.CODE_ROOTS:
        save(work/folder/"synthetic.py", b"# Synthetic source, never executed.\n")
    for path in p.NOTICES:
        save(work/path, b"Synthetic source or notice text; no real license assertion.\n")
    save(work/"trace_ovss/resources/classes.txt", b"background\nsynthetic\n")
    metadata = {}
    for name, raw in {"M01_offline_score_20261102_1942/ANALYSIS_SEALED.json":b'{"complete":true}\n',
                      "M01_offline_score_synthetic/DATASET_RESULTS.json":b'{"delta":-2,"CI":null}\n'}.items():
        pin = save(policy.metadata/name, raw)
        metadata[name] = dict(sha256=pin["sha256"], bytes=pin["bytes"], identity=["synthetic old identity"])
    registry = save(policy.paper/"generated/source_provenance.json",
                    dict(source_pins=metadata, results_root=str(policy.metadata)))
    papers = []
    for label in ("draft_v001", "draft_v002"):
        composites = [f"figures/case_{i}.png" for i in range(4)]
        contents = {"main.tex": b"Synthetic original paper. Negative and null retained.\n",
                    "generated/source_provenance.json": Path(registry["path"]).read_bytes(),
                    **{n:b"Synthetic derived composite placeholder; no actual pixels." for n in composites}}
        manifest = dict(label=label, members={n:dict(sha256=p.sha(b), bytes=len(b)) for n,b in contents.items()})
        papers.append(dict(label=label, archive=save(policy.paper/"archives"/(label+".tar.gz"),
            archive_bytes(list(contents.items()))), manifest=save(policy.paper/("FREEZE_"+label+".json"),manifest),
            members=len(contents), tar_prefix="", manifest_member=None, derived_composites=composites))
    reviews = {"v001":save(policy.paper/"reviews/manuscript_review_draft_v001.md",b"Synthetic R1 finding.\n"),
               "v002":save(policy.paper/"reviews/manuscript_review_draft_v002_closure.md",b"Synthetic R1 PARTIAL; no acceptance.\n")}
    disposition = save(work/p.DISPOSITION,b"Synthetic disposition: partial construction, negatives retained.\n")
    code_review = save(work/"docs/scope-v4-delivery-code-independent-20261103.md",
                       b"Synthetic CODE report only; not an actual independent closure.\n")
    document = save(work/"docs/M01-risk-results-20261102.md",b"Synthetic negative and null results.\n")
    spec = dict(schema_version=1, papers=papers, exporter_provenance=registry, reviews=reviews,
        disposition=disposition, code_review=code_review, documents=[document], extra_metadata=[],
        config_paths=[], acknowledgements=copy.deepcopy(p.ACK))
    def plan_pin(specification=None, caps=None):
        plan = p.make_plan(specification or spec,policy,caps or limits)
        return save(results/("plan_"+str(len(list(results.glob('plan_*.json'))))+".json"),plan)
    return dict(work=work, results=results, policy=policy, limits=limits, spec=spec,
                plan_pin=plan_pin, tmp=tmp_path)


def destination(f, suffix="packet"):
    return f["results"]/(p.PREFIX+suffix)


def unpack(raw):
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        return {m.name:tar.extractfile(m).read() for m in tar if m.isfile()}


def test_normal_create_plan_validate_build_and_preserve_bytes(fixture):
    f=fixture
    specpin=save(f["tmp"]/"operator-spec.json",f["spec"])
    created=p.create_plan(specpin,destination(f,"plan"),f["policy"],f["limits"])
    pin={k:created[k] for k in ("path","sha256","bytes")}
    before=set(f["results"].iterdir())
    result=p.validate(pin,f["policy"],f["limits"])
    assert result["valid"] and not result["writes_performed"] and set(f["results"].iterdir())==before
    target=destination(f)
    receipt=p.build(pin,target,f["policy"],f["limits"])
    assert receipt["complete"] is True and receipt["publication_actions_performed"] is False
    assert {x.name for x in target.iterdir()}=={"delivery.tar.gz","MANIFEST.json","LOCAL_VERIFY.json"}
    blobs=unpack((target/"delivery.tar.gz").read_bytes())
    manifest=json.loads((target/"MANIFEST.json").read_bytes())
    assert set(blobs)==set(manifest["archive_member_allowlist"])
    assert blobs["MANIFEST.json"]==(target/"MANIFEST.json").read_bytes()
    for paper in f["spec"]["papers"]:
        assert blobs[f"paper/{paper['label']}/original.tar.gz"]==Path(paper["archive"]["path"]).read_bytes()
    assert blobs["reviews/manuscript_review_draft_v001.md"]==b"Synthetic R1 finding.\n"
    assert blobs["evidence/M01_offline_score_synthetic/DATASET_RESULTS.json"]==b'{"delta":-2,"CI":null}\n'
    assert p.sha((target/"delivery.tar.gz").read_bytes())==receipt["archive"]["sha256"]
    assert "PARTIAL" in blobs["DELIVERY_SCOPE.md"].decode() and "NOT RUN" in blobs["DELIVERY_SCOPE.md"].decode()
    for n in p.NOTICES:
        assert "implementation/current_first_party/"+n in blobs
    with pytest.raises(ValueError,match="already exists"):
        p.build(pin,target,f["policy"],f["limits"])


@pytest.mark.parametrize("key,value",[("local_packaging_authorized",False),
    ("all_referenced_reviews_completed",False),("paper_acceptance_claimed",True),
    ("R1_status","CLOSED"),("publication_actions_authorized",True),
    ("local_packaging_authorized",1)])
def test_requires_explicit_completed_local_partial_ack(fixture,key,value):
    f=fixture;spec=copy.deepcopy(f["spec"]);spec["acknowledgements"][key]=value
    with pytest.raises(ValueError,match="acknowledgements"):
        f["plan_pin"](spec)


def test_actual_production_review_archive_disposition_pins_enforced(fixture):
    f=fixture
    for field in ("v002_archive_sha256","disposition_sha256"):
        policy=replace(f["policy"],**{field:"0"*64})
        with pytest.raises(ValueError,match="pin|archive"):
            p.make_plan(f["spec"],policy,f["limits"])
    policy=replace(f["policy"],review_sha256={"v001":"0"*64,"v002":"0"*64})
    with pytest.raises(ValueError,match="review pin"):
        p.make_plan(f["spec"],policy,f["limits"])


def test_tamper_before_parse_and_metadata_membership(fixture):
    f=fixture;pin=f["plan_pin"]();plan=json.loads(Path(pin["path"]).read_bytes())
    target=Path(plan["source_metadata"][0]["path"]);target.write_bytes(b"not-json same role")
    with pytest.raises(ValueError,match="size|hash"):
        p.validate(pin,f["policy"],f["limits"])
    # A self-consistent plan hash is not permission to change the exporter membership.
    plan["source_metadata"]=[]
    bad=save(f["tmp"]/"bad-plan.json",plan)
    with pytest.raises(ValueError,match="metadata list"):
        p.validate(bad,f["policy"],f["limits"])


@pytest.mark.parametrize("name",["../escape","/absolute","a//b","a/./b","a/../b","a\\b",".env","a\nfile"])
def test_path_traversal_rejected(name):
    with pytest.raises(ValueError):p.relative(name)


def test_paper_undeclared_member_and_traversal_without_extraction(fixture):
    f=fixture;paper=f["spec"]["papers"][0];manifest=Path(paper["manifest"]["path"]).read_bytes()
    contents=list(unpack(Path(paper["archive"]["path"]).read_bytes()).items())
    for name in ("unexpected.txt","../escape"):
        with pytest.raises(ValueError):p.paper_check(archive_bytes(contents+[(name,b"x")]),manifest,paper,f["limits"])
    assert not (f["tmp"]/"escape").exists()


def test_paper_symlink_member_refused(fixture):
    f=fixture;paper=f["spec"]["papers"][0]
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w") as tar:
        m=tarfile.TarInfo("main.tex");m.type=tarfile.SYMTYPE;m.linkname="elsewhere";tar.addfile(m)
    with pytest.raises(ValueError,match="links"):
        p.paper_check(gzip.compress(stream.getvalue()),Path(paper["manifest"]["path"]).read_bytes(),paper,f["limits"])


def test_symlink_parent_and_leaf_refused(fixture):
    f=fixture;plain=f["tmp"]/"plain.json";plain.write_bytes(b"{}")
    link=f["tmp"]/"leaf.json";link.symlink_to(plain)
    with pytest.raises(ValueError):p.read(link,100)
    parent=f["tmp"]/"aliased";parent.symlink_to(f["work"],target_is_directory=True)
    with pytest.raises(OSError):p.read(parent/"README.md",1024)
    (f["work"]/"tests/aliased.py").symlink_to(plain)
    with pytest.raises(ValueError,match="Symlink"):
        p.code_inventory(f["policy"],[])


def test_hardlinks_and_fifo_refused_before_content_open(fixture,monkeypatch):
    f=fixture;pin=f["plan_pin"]();plan=json.loads(Path(pin["path"]).read_bytes())
    leaf=Path(plan["source_metadata"][0]["path"]);os.link(leaf,f["tmp"]/"duplicate")
    opened=[];real=os.open
    def spy(path,*args,**kwargs):
        if path==leaf.name:opened.append(path)
        return real(path,*args,**kwargs)
    monkeypatch.setattr(p.os,"open",spy)
    with pytest.raises(ValueError,match="single-link"):
        p.validate(pin,f["policy"],f["limits"])
    assert opened==[]
    fifo=f["tmp"]/"fifo.json";os.mkfifo(fifo)
    with pytest.raises(ValueError,match="regular"):p.read(fifo,100)


def test_same_bytes_inode_replacement_during_bounded_read(fixture,monkeypatch):
    f=fixture;leaf=f["tmp"]/"stable.json";raw=b'{"value":null}\n';leaf.write_bytes(raw)
    real=p.os.read;changed=[]
    def replace_during_read(fd,n):
        result=real(fd,n)
        if not changed:
            replacement=f["tmp"]/"replacement.json";replacement.write_bytes(raw);replacement.replace(leaf);changed.append(True)
        return result
    monkeypatch.setattr(p.os,"read",replace_during_read)
    with pytest.raises(ValueError,match="FD/path drift"):p.read(leaf,100,p.sha(raw),len(raw))


def test_source_excludes_payload_vendor_hidden_and_credentials(fixture):
    f=fixture
    for name in ("tests/data/raw.npy","tests/vendor/foreign.py","trace_ovss/.env",
                 "tests/fixtures/payload.py","trace_ovss/secrets.py","trace_ovss/weights/model.bin"):
        save(f["work"]/name,b"EXCLUDED")
    names=p.code_inventory(f["policy"],[])
    assert not any(n.endswith((".npy","foreign.py",".env","payload.py","secrets.py",".bin")) for n in names)
    assert "trace_ovss/synthetic.py" in names
    with pytest.raises(ValueError):p.code_inventory(f["policy"],["tests/data/raw.npy"])


def test_source_membership_drift_refused(fixture):
    f=fixture;pin=f["plan_pin"]();save(f["work"]/"tests/new.py",b"# Unexecuted new source\n")
    with pytest.raises(ValueError,match="membership differs"):
        p.validate(pin,f["policy"],f["limits"])


def test_foreign_and_existing_output_never_changed(fixture):
    f=fixture;pin=f["plan_pin"]();foreign=f["tmp"]/"foreign";foreign.mkdir();sentinel=foreign/"keep";sentinel.write_bytes(b"keep")
    with pytest.raises(ValueError,match="Foreign"):
        p.build(pin,foreign,f["policy"],f["limits"])
    existing=destination(f);existing.mkdir();save(existing/"keep",b"keep")
    with pytest.raises(ValueError,match="already exists"):
        p.build(pin,existing,f["policy"],f["limits"])
    assert sentinel.read_bytes()==b"keep" and (existing/"keep").read_bytes()==b"keep"


@pytest.mark.parametrize("field,value",[("code",4),("metadata",4),("total",1),("paper_expanded",32),("package",32),("members",3)])
def test_caps_refuse_without_completed_output(fixture,field,value):
    f=fixture;limits=replace(f["limits"],**{field:value})
    with pytest.raises((ValueError,EOFError)):
        pin=f["plan_pin"](caps=limits)
        p.build(pin,destination(f),f["policy"],limits)
    assert not (destination(f)/"LOCAL_VERIFY.json").exists()


def test_disk_reserve_refuses_before_destination(fixture,monkeypatch):
    f=fixture;pin=f["plan_pin"]()
    monkeypatch.setattr(p.shutil,"disk_usage",lambda _:type("Usage",(),{"free":0})())
    with pytest.raises(ValueError,match="reserve"):
        p.build(pin,destination(f),f["policy"],f["limits"])
    assert not destination(f).exists()


@pytest.mark.parametrize("phase",["delivery.tar.gz","MANIFEST.json","LOCAL_VERIFY.json"])
def test_ordinary_write_failures_are_owned_failed_and_not_resumed(fixture,monkeypatch,phase):
    f=fixture;pin=f["plan_pin"]();original=p.OwnedOutput.write
    def fail(self,name,raw):
        if name==phase:raise OSError("Synthetic ordinary write failure")
        return original(self,name,raw)
    monkeypatch.setattr(p.OwnedOutput,"write",fail)
    with pytest.raises(OSError):p.build(pin,destination(f),f["policy"],f["limits"])
    assert json.loads((destination(f)/"FAILED.json").read_bytes())["complete"] is False
    assert not (destination(f)/"LOCAL_VERIFY.json").exists()
    with pytest.raises(ValueError,match="already exists"):
        p.build(pin,destination(f),f["policy"],f["limits"])


def test_no_project_or_io_action_imports():
    import ast
    tree=ast.parse(Path(p.__file__).read_text())
    names={n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import)}
    assert not names.intersection({"torch","numpy","PIL","cv2","subprocess","socket","requests","urllib"})


def test_full_handoff_and_nonallowlisted_result_payload_refused(fixture):
    f=fixture;spec=copy.deepcopy(f["spec"])
    spec["documents"]=[save(f["work"]/"docs/self-evolution-handoff.md",b"Must not bundle full handoff")]
    with pytest.raises(ValueError,match="handoffs"):
        f["plan_pin"](spec)
    for path in (f["policy"].metadata/"x/predictions/DATASET_RESULTS.json",
                 f["policy"].metadata/"x/cost.jsonl",f["tmp"]/"RESULT.json"):
        with pytest.raises(ValueError):p.metadata_path(dict(path=str(path)),f["policy"])


def test_exporter_registry_must_match_frozen_v002_member(fixture):
    f=fixture;spec=copy.deepcopy(f["spec"])
    path=Path(spec["exporter_provenance"]["path"])
    registry=json.loads(path.read_bytes());registry["new_unreviewed_annotation"]="not in frozen paper"
    spec["exporter_provenance"]=save(path,registry)
    pin=f["plan_pin"](spec)
    with pytest.raises(ValueError,match="exact reviewed v002"):
        p.build(pin,destination(f),f["policy"],f["limits"])
    assert not destination(f).exists()


def test_local_verification_refuses_tamper_and_failed_packet(fixture):
    f=fixture;pin=f["plan_pin"]();target=destination(f)
    p.build(pin,target,f["policy"],f["limits"])
    path=target/"LOCAL_VERIFY.json"
    receipt_pin=dict(path=str(path),sha256=p.sha(path.read_bytes()),bytes=path.stat().st_size)
    assert p.verify_delivery(target,receipt_pin,f["policy"],f["limits"])["complete"]
    save(target/"FAILED.json",dict(complete=False))
    with pytest.raises(ValueError,match="Failed"):
        p.verify_delivery(target,receipt_pin,f["policy"],f["limits"])
    (target/"FAILED.json").unlink()
    with (target/"delivery.tar.gz").open("ab") as stream:stream.write(b"tampered")
    with pytest.raises(ValueError,match="size|hash"):
        p.verify_delivery(target,receipt_pin,f["policy"],f["limits"])


def test_verifier_refuses_self_consistent_extra_tar_member(fixture):
    f=fixture;pin=f["plan_pin"]();target=destination(f)
    p.build(pin,target,f["policy"],f["limits"])
    members=list(unpack((target/"delivery.tar.gz").read_bytes()).items())
    members.append(("unexpected.txt",b"Undeclared harmless text"))
    archive=archive_bytes(members);save(target/"delivery.tar.gz",archive)
    receipt=json.loads((target/"LOCAL_VERIFY.json").read_bytes())
    receipt["archive"]=dict(sha256=p.sha(archive),bytes=len(archive))
    receipt_pin=save(target/"LOCAL_VERIFY.json",receipt)
    with pytest.raises(ValueError,match="type/member"):
        p.verify_delivery(target,receipt_pin,f["policy"],f["limits"])


def test_paper_hidden_trailing_bytes_refused(fixture):
    f=fixture;paper=f["spec"]["papers"][0]
    expanded=gzip.decompress(Path(paper["archive"]["path"]).read_bytes())+b"UNDECLARED"
    with pytest.raises(ValueError,match="trailing paper"):
        p.paper_check(gzip.compress(expanded),Path(paper["manifest"]["path"]).read_bytes(),paper,f["limits"])


def test_read_error_and_close_error_do_not_leak_directory_fds(fixture,monkeypatch):
    f=fixture;leaf=f["tmp"]/"ordinary.json";leaf.write_bytes(b"{}")
    opened=set();targets=set();faults=[];real_open=p.os.open;real_close=p.os.close
    def open_spy(path,flags,*args,**kwargs):
        fd=real_open(path,flags,*args,**kwargs);opened.add(fd)
        if path==leaf.name:targets.add(fd)
        return fd
    def close_spy(fd):
        real_close(fd);opened.discard(fd)
        if fd in targets and not faults:
            targets.discard(fd);faults.append(True);raise OSError("Synthetic close error after actual close")
        targets.discard(fd)
    monkeypatch.setattr(p.os,"open",open_spy);monkeypatch.setattr(p.os,"close",close_spy)
    with pytest.raises(OSError,match="close error"):p.read(leaf,100)
    assert opened==set()
    for _ in range(20):
        with pytest.raises(ValueError,match="hash mismatch"):p.read(leaf,100,"0"*64)
    assert opened==set()


@pytest.mark.parametrize("phase",["delivery.tar.gz","LOCAL_VERIFY.json"])
def test_actual_output_close_errors_fail_and_receipt_cannot_override_failure(fixture,monkeypatch,phase):
    f=fixture;pin=f["plan_pin"]();target=destination(f)
    targets=set();faults=[];real_open=p.os.open;real_close=p.os.close
    def opening(path,flags,*args,**kwargs):
        fd=real_open(path,flags,*args,**kwargs)
        if path==phase and flags & os.O_WRONLY:targets.add(fd)
        return fd
    def closing(fd):
        real_close(fd)
        if fd in targets and not faults:
            targets.discard(fd);faults.append(True);raise OSError("Synthetic actual writer close error")
        targets.discard(fd)
    monkeypatch.setattr(p.os,"open",opening);monkeypatch.setattr(p.os,"close",closing)
    with pytest.raises(OSError,match="writer close"):
        p.build(pin,target,f["policy"],f["limits"])
    assert faults and (target/"FAILED.json").exists()
    if (target/"LOCAL_VERIFY.json").exists():
        path=target/"LOCAL_VERIFY.json";receipt_pin=dict(path=str(path),sha256=p.sha(path.read_bytes()),bytes=path.stat().st_size)
        with pytest.raises(ValueError,match="Failed"):
            p.verify_delivery(target,receipt_pin,f["policy"],f["limits"])


def test_output_context_exit_error_precedes_completion_receipt(fixture,monkeypatch):
    f=fixture;pin=f["plan_pin"]();original=p.OwnedOutput.__exit__;faults=[]
    def exit_error(self,*exc):
        original(self,*exc)
        if not faults:faults.append(True);raise OSError("Synthetic output context exit error")
    monkeypatch.setattr(p.OwnedOutput,"__exit__",exit_error)
    with pytest.raises(OSError,match="context exit"):
        p.build(pin,destination(f),f["policy"],f["limits"])
    assert (destination(f)/"FAILED.json").exists() and not (destination(f)/"LOCAL_VERIFY.json").exists()


def test_final_written_manifest_identity_not_rebased(fixture,monkeypatch):
    f=fixture;pin=f["plan_pin"]();target=destination(f);original=p.read;changed=[]
    def swap(path,*args,**kwargs):
        if Path(path)==target/"MANIFEST.json" and not changed:
            leaf=Path(path);raw=leaf.read_bytes();save(target/"replacement.json",raw)
            (target/"replacement.json").replace(leaf);changed.append(True)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(p,"read",swap)
    with pytest.raises(ValueError,match="before open"):
        p.build(pin,target,f["policy"],f["limits"])
    assert changed and (target/"FAILED.json").exists() and not (target/"LOCAL_VERIFY.json").exists()


def test_cli_dispatch_is_explicit_and_does_not_build_on_validate(fixture,monkeypatch,capsys):
    f=fixture;pin=f["plan_pin"]();calls=[]
    monkeypatch.setattr(p,"validate",lambda value:calls.append(value) or dict(valid=True,writes_performed=False))
    monkeypatch.setattr(p.sys,"argv",["delivery_packager","validate","--plan",pin["path"],"--sha256",pin["sha256"]])
    p.main()
    assert calls==[pin] and json.loads(capsys.readouterr().out)["writes_performed"] is False


def test_additional_new_code_reports_are_explicit_and_old_grades_refused(fixture):
    f=fixture;spec=copy.deepcopy(f["spec"])
    raw=b"Synthetic additional new CODE findings, unchanged and not graded.\n"
    spec["additional_code_reviews"]=[save(f["work"]/"docs/scope-v4-exporter-code-independent-20261103.md",raw)]
    pin=f["plan_pin"](spec);p.build(pin,destination(f),f["policy"],f["limits"])
    blobs=unpack((destination(f)/"delivery.tar.gz").read_bytes())
    assert blobs["reviews/code/scope-v4-exporter-code-independent-20261103.md"]==raw
    spec["additional_code_reviews"]=[save(f["work"]/"docs/round05_grade.md",b"Excluded old grade")]
    with pytest.raises(ValueError,match="old grades"):
        f["plan_pin"](spec)


@pytest.mark.parametrize('action', ['verify', 'build'])
@pytest.mark.parametrize('change', ['failed', 'manifest', 'manifest_same_bytes',
                                    'receipt_same_bytes', 'archive_same_bytes'])
def test_final_verification_rebinds_membership_and_all_leaves(fixture, monkeypatch, action, change):
    f = fixture
    pin = f['plan_pin']()
    target = destination(f)
    if action == 'verify':
        p.build(pin, target, f['policy'], f['limits'])
    original_read, original_verify = p.read, p.verify_delivery
    active, changed = [], []
    def late_change(path, *args, **kwargs):
        result = original_read(path, *args, **kwargs)
        if active and Path(path) == target/'delivery.tar.gz' and not changed:
            changed.append(True)
            if change == 'failed':
                save(target/'FAILED.json', dict(complete=False))
            else:
                name = ('MANIFEST.json' if change.startswith('manifest') else
                        'LOCAL_VERIFY.json' if change.startswith('receipt') else 'delivery.tar.gz')
                leaf = target/name
                raw = leaf.read_bytes() if change.endswith('same_bytes') else b'{"replacement":true}\n'
                replacement = target/'synthetic-replacement'
                save(replacement, raw)
                replacement.replace(leaf)
        return result
    def verifying(*args, **kwargs):
        active.append(True)
        try:
            return original_verify(*args, **kwargs)
        finally:
            active.pop()
    monkeypatch.setattr(p, 'read', late_change)
    monkeypatch.setattr(p, 'verify_delivery', verifying)
    with pytest.raises(ValueError, match='Failed|Final output leaf'):
        if action == 'build':
            p.build(pin, target, f['policy'], f['limits'])
        else:
            receipt_path = target/'LOCAL_VERIFY.json'
            receipt_pin = dict(path=str(receipt_path), sha256=p.sha(receipt_path.read_bytes()),
                               bytes=receipt_path.stat().st_size)
            p.verify_delivery(target, receipt_pin, f['policy'], f['limits'])
    assert changed
    if action == 'build':
        assert (target/'FAILED.json').exists()


def test_directory_fstat_exception_closes_new_child(fixture, monkeypatch):
    folder = fixture['tmp']/'ordinary-directory'
    folder.mkdir()
    opened, children, faults = set(), set(), []
    original_open, original_close, original_fstat = p.os.open, p.os.close, p.os.fstat
    def opening(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        opened.add(fd)
        if path == folder.name:
            children.add(fd)
        return fd
    def closing(fd):
        original_close(fd)
        opened.discard(fd)
        children.discard(fd)
    def stating(fd):
        if fd in children and not faults:
            faults.append(True)
            raise OSError('Synthetic child fstat failure')
        return original_fstat(fd)
    monkeypatch.setattr(p.os, 'open', opening)
    monkeypatch.setattr(p.os, 'close', closing)
    monkeypatch.setattr(p.os, 'fstat', stating)
    with pytest.raises(OSError, match='child fstat'):
        with p.directory(folder):
            pytest.fail('Faulted directory must not yield')
    assert faults and not opened


def test_namespace_change_during_final_leaf_rebind_is_not_adopted(fixture, monkeypatch):
    f = fixture
    target = destination(f)
    p.build(f['plan_pin'](), target, f['policy'], f['limits'])
    leaf = target/'LOCAL_VERIFY.json'
    receipt_pin = dict(path=str(leaf), sha256=p.sha(leaf.read_bytes()), bytes=leaf.stat().st_size)
    original_read, original_observe = p.read, p.observe
    parsed_archive, changed = [], []
    def reading(path, *args, **kwargs):
        result = original_read(path, *args, **kwargs)
        if Path(path) == target/'delivery.tar.gz':
            parsed_archive.append(True)
        return result
    def observing(path, *args, **kwargs):
        result = original_observe(path, *args, **kwargs)
        if parsed_archive and Path(path) == target/'delivery.tar.gz' and not changed:
            changed.append(True)
            manifest = target/'MANIFEST.json'
            replacement = target/'synthetic-replacement'
            save(replacement, manifest.read_bytes())
            replacement.replace(manifest)
        return result
    monkeypatch.setattr(p, 'read', reading)
    monkeypatch.setattr(p, 'observe', observing)
    with pytest.raises(ValueError, match='directory membership drift'):
        p.verify_delivery(target, receipt_pin, f['policy'], f['limits'])
    assert changed


def test_fixed_V01_terminal_metadata_name_is_supported_without_scope_expansion(fixture):
    f = fixture
    relative = 'V01_offline_score_terminal_001.json'
    pin = dict(path=str(f['policy'].metadata/relative), sha256='0'*64, bytes=1)
    assert p.metadata_path(pin, f['policy']) == relative
    for denied in ('V02_offline_score_terminal_001.json',
                   'V01_offline_score/episodes/000000.json',
                   'V01_offline_score/coco_object.episode_counts.jsonl'):
        pin['path'] = str(f['policy'].metadata/denied)
        with pytest.raises(ValueError, match='refused|allowlisted'):
            p.metadata_path(pin, f['policy'])


def c4_seal(f):
    path = f['policy'].metadata/'M01_offline_score_20261102_1942/ANALYSIS_SEALED.json'
    raw = path.read_bytes()
    return dict(path=str(path), sha256=p.sha(raw), bytes=len(raw))


def c4_spec(f, *, linked=True):
    pin = c4_seal(f)
    if linked:
        os.link(pin['path'], f['tmp']/'synthetic-readonly-seal-alias')
    spec = copy.deepcopy(f['spec'])
    spec['readonly_registry_seal_hardlinks'] = [dict(pin, nlink=2)]
    return spec


def c4_change(f, action):
    path = Path(c4_seal(f)['path'])
    before = path.stat()
    if action == 'count':
        os.link(path, f['tmp']/'synthetic-third-seal-link')
    elif action == 'ctime':
        os.chmod(path, before.st_mode ^ 0o100)
        os.chmod(path, before.st_mode)
        assert path.stat().st_ctime_ns != before.st_ctime_ns
        assert path.stat().st_nlink == 2 and path.stat().st_mode == before.st_mode
    else:
        assert action == 'same_bytes_inode'
        replacement = f['tmp']/'synthetic-same-bytes-seal'
        replacement.write_bytes(path.read_bytes())
        os.link(replacement, f['tmp']/'synthetic-new-inode-seal-alias')
        replacement.replace(path)
        assert path.stat().st_nlink == 2 and path.stat().st_ino != before.st_ino


def test_c4_exact_nlink2_registry_opt_in_validates_builds_and_stays_readonly(fixture, monkeypatch):
    f = fixture
    spec = c4_spec(f)
    seal = c4_seal(f)
    before = p.identity(Path(seal['path']).stat())
    pin = f['plan_pin'](spec)
    registry_read, opted_read = [], []
    original_pin_read, original_seal_read = p.pin_read, p._read_readonly_seal
    def reading(pin, *args, **kwargs):
        raw = original_pin_read(pin, *args, **kwargs)
        if pin == spec['exporter_provenance']:
            registry_read.append(True)
        return raw
    def sealed(record, *args, **kwargs):
        assert registry_read and record['role'] == 'sealed_summary_metadata'
        opted_read.append(True)
        return original_seal_read(record, *args, **kwargs)
    monkeypatch.setattr(p, 'pin_read', reading)
    monkeypatch.setattr(p, '_read_readonly_seal', sealed)
    assert p.validate(pin, f['policy'], f['limits'])['writes_performed'] is False
    result = p.build(pin, destination(f), f['policy'], f['limits'])
    assert result['complete'] and opted_read
    assert p.identity(Path(seal['path']).stat()) == before
    assert Path(seal['path']).read_bytes() == b'{"complete":true}\n'
    assert (f['tmp']/'synthetic-readonly-seal-alias').read_bytes() == b'{"complete":true}\n'
    manifest = json.loads((destination(f)/'MANIFEST.json').read_bytes())
    record = next(r for r in manifest['source_records'] if r['pin']['path'] == seal['path'])
    assert record['role'] == 'sealed_summary_metadata' and record['readonly_expected_nlink'] == 2


def test_c4_absent_or_empty_opt_in_keeps_default_single_link_policy(fixture):
    f = fixture
    spec = c4_spec(f)
    for entries in (None, []):
        candidate = copy.deepcopy(spec)
        if entries is None:
            candidate.pop('readonly_registry_seal_hardlinks')
        else:
            candidate['readonly_registry_seal_hardlinks'] = entries
        with pytest.raises(ValueError, match='single-link'):
            p.validate(f['plan_pin'](candidate), f['policy'], f['limits'])
    seal = c4_seal(f)
    for reader in (lambda:p.observe(seal['path'], 1024), lambda:p.read(seal['path'], 1024),
                   lambda:p.pin_read(seal, 1024)):
        with pytest.raises(ValueError, match='single-link'):
            reader()


@pytest.mark.parametrize('value', [None, {}, 'all', [None]])
def test_c4_malformed_opt_in_list_rejected(fixture, value):
    f = fixture
    spec = copy.deepcopy(f['spec'])
    spec['readonly_registry_seal_hardlinks'] = value
    with pytest.raises(ValueError, match='opt-ins|pin'):
        p.validate_spec(spec, f['policy'])


@pytest.mark.parametrize('count', [True, False, 1, 3, 0, -2, 2.0, '2'])
def test_c4_only_exact_integer_two_is_supported(fixture, count):
    f = fixture
    spec = c4_spec(f, linked=False)
    spec['readonly_registry_seal_hardlinks'][0]['nlink'] = count
    with pytest.raises(ValueError, match='integer nlink2'):
        p.validate_spec(spec, f['policy'])


@pytest.mark.parametrize('change', ['sha256', 'bytes', 'unused', 'duplicate', 'extra_field', 'missing_field'])
def test_c4_unused_duplicate_or_mismatched_pins_are_not_ignored(fixture, change):
    f = fixture
    spec = c4_spec(f)
    entry = spec['readonly_registry_seal_hardlinks'][0]
    if change == 'sha256':
        entry['sha256'] = '0'*64
    elif change == 'bytes':
        entry['bytes'] += 1
    elif change == 'unused':
        entry['path'] = str(f['policy'].metadata/'M02_offline_score_20261103_0156/ANALYSIS_SEALED.json')
    elif change == 'duplicate':
        spec['readonly_registry_seal_hardlinks'].append(dict(entry))
    elif change == 'extra_field':
        entry['role'] = 'sealed_summary_metadata'
    else:
        entry.pop('nlink')
    with pytest.raises(ValueError, match='opt-in|pin'):
        f['plan_pin'](spec)


@pytest.mark.parametrize('relative', ['M01_offline_score_synthetic/ANALYSIS_SEALED.json',
    'A01_cached_gain_groups_v1/ANALYSIS_SEALED.json', 'M01_offline_score_20261102_1942/DATASET_RESULTS.json',
    'M01_offline_score_20261102_1942/episodes/ANALYSIS_SEALED.json', '../ANALYSIS_SEALED.json'])
def test_c4_opt_in_geometry_cannot_expand_to_other_metadata(fixture, relative):
    f = fixture
    spec = c4_spec(f, linked=False)
    spec['readonly_registry_seal_hardlinks'][0]['path'] = str(f['policy'].metadata/relative)
    with pytest.raises(ValueError):
        p.validate_spec(spec, f['policy'])


@pytest.mark.parametrize('role', ['code', 'paper', 'review', 'plan', 'output', 'foreign'])
def test_c4_opt_in_cannot_relax_foreign_or_non_registry_roles(fixture, role):
    f = fixture
    spec = c4_spec(f, linked=False)
    paths = dict(code=f['work']/'tests/synthetic.py', paper=Path(spec['papers'][0]['manifest']['path']),
        review=Path(spec['reviews']['v001']['path']), plan=f['results']/'PIN_PLAN.json',
        output=destination(f)/'MANIFEST.json', foreign=f['tmp']/'ANALYSIS_SEALED.json')
    spec['readonly_registry_seal_hardlinks'][0]['path'] = str(paths[role])
    with pytest.raises(ValueError, match='outside fixed results root'):
        p.validate_spec(spec, f['policy'])


def test_c4_nlink1_opt_in_is_unused_permission_not_a_single_link_fallback(fixture):
    f = fixture
    spec = c4_spec(f, linked=False)
    with pytest.raises(ValueError, match='exact-nlink2'):
        p.validate(f['plan_pin'](spec), f['policy'], f['limits'])


def test_c4_opted_symlink_is_still_rejected(fixture, monkeypatch):
    f = fixture
    spec = c4_spec(f)
    pin = f['plan_pin'](spec)
    seal = Path(c4_seal(f)['path'])
    seal.unlink()
    seal.symlink_to(f['tmp']/'synthetic-readonly-seal-alias')
    opened, real = [], p.os.open
    def opening(path, *args, **kwargs):
        if path == seal.name:
            opened.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(p.os, 'open', opening)
    with pytest.raises(ValueError, match='exact-nlink2'):
        p.validate(pin, f['policy'], f['limits'])
    assert not opened


def test_c4_authenticated_registry_mismatch_refuses_opted_content_open(fixture, monkeypatch):
    f = fixture
    spec = c4_spec(f)
    pin = f['plan_pin'](spec)
    plan = json.loads(Path(pin['path']).read_bytes())
    entry = plan['specification']['readonly_registry_seal_hardlinks'][0]
    entry['sha256'] = '0'*64
    next(r for r in plan['source_metadata'] if r['path'] == entry['path'])['sha256'] = '0'*64
    bad = save(f['tmp']/'self-consistent-unbound-plan.json', plan)
    opened, real = [], p.os.open
    def opening(path, *args, **kwargs):
        if path == 'ANALYSIS_SEALED.json':
            opened.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(p.os, 'open', opening)
    with pytest.raises(ValueError, match='Exporter allowlist/pin membership'):
        p.validate(bad, f['policy'], f['limits'])
    assert not opened


def test_c4_registry_authentication_failure_never_opens_opted_content(fixture, monkeypatch):
    f = fixture
    spec = c4_spec(f)
    pin = f['plan_pin'](spec)
    registry = Path(spec['exporter_provenance']['path'])
    raw = registry.read_bytes()
    registry.write_bytes(bytes([raw[0]^1])+raw[1:])
    opened, real = [], p.os.open
    def opening(path, *args, **kwargs):
        if path == 'ANALYSIS_SEALED.json':
            opened.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(p.os, 'open', opening)
    with pytest.raises(ValueError, match='hash mismatch before parse'):
        p.validate(pin, f['policy'], f['limits'])
    assert not opened


@pytest.mark.parametrize('action', ['count', 'ctime', 'same_bytes_inode'])
@pytest.mark.parametrize('stage', ['before_open', 'after_read'])
def test_c4_opted_count_ctime_and_inode_changes_fail_closed(fixture, monkeypatch, action, stage):
    f = fixture
    pin = f['plan_pin'](c4_spec(f))
    original, changed = p._read_readonly_seal, []
    def changing(*args, **kwargs):
        if stage == 'before_open':
            c4_change(f, action)
            changed.append(True)
        raw = original(*args, **kwargs)
        if stage == 'after_read':
            c4_change(f, action)
            changed.append(True)
        return raw
    monkeypatch.setattr(p, '_read_readonly_seal', changing)
    with pytest.raises(ValueError, match='nlink2|before open|Final input identity'):
        p.validate(pin, f['policy'], f['limits'])
    assert changed


def test_c4_nlink_and_ctime_are_bound_during_actual_fd_read(fixture, monkeypatch):
    f = fixture
    pin = f['plan_pin'](c4_spec(f))
    seal_id = p.identity(Path(c4_seal(f)['path']).stat())[:2]
    original, changed = p.os.read, []
    def reading(fd, size):
        raw = original(fd, size)
        if p.identity(p.os.fstat(fd))[:2] == seal_id and not changed:
            c4_change(f, 'count')
            changed.append(True)
        return raw
    monkeypatch.setattr(p.os, 'read', reading)
    with pytest.raises(ValueError, match='Input FD/path drift'):
        p.validate(pin, f['policy'], f['limits'])
    assert changed


@pytest.mark.parametrize('action', ['count', 'ctime', 'same_bytes_inode'])
def test_c4_late_opted_input_changes_fail_before_final_receipt(fixture, monkeypatch, action):
    f = fixture
    pin = f['plan_pin'](c4_spec(f))
    original, changed = p.rebind, []
    def rebinding(*args, **kwargs):
        original(*args, **kwargs)
        if not changed:
            c4_change(f, action)
            changed.append(True)
    monkeypatch.setattr(p, 'rebind', rebinding)
    with pytest.raises(ValueError, match='nlink2|Final input identity'):
        p.build(pin, destination(f), f['policy'], f['limits'])
    assert changed and (destination(f)/'FAILED.json').exists()
    assert not (destination(f)/'LOCAL_VERIFY.json').exists()


@pytest.mark.parametrize('name', ['delivery.tar.gz', 'MANIFEST.json', 'LOCAL_VERIFY.json'])
def test_c4_readonly_input_opt_in_never_allows_output_hardlinks(fixture, name):
    f = fixture
    result = p.build(f['plan_pin'](c4_spec(f)), destination(f), f['policy'], f['limits'])
    os.link(destination(f)/name, f['tmp']/'synthetic-output-alias')
    with pytest.raises(ValueError, match='single-link'):
        p.verify_delivery(destination(f), result['local_receipt_pin'], f['policy'], f['limits'])


def test_c4_wrong_record_role_refused_even_with_matching_grant(fixture):
    f = fixture
    spec = c4_spec(f)
    pin = c4_seal(f)
    grants = p._readonly_seal_pins(spec, f['policy'], [pin])
    for role in ('extra_small_metadata', 'current_code_not_frozen_runtime', 'exporter_registry'):
        record = dict(pin=pin, role=role, cap=f['limits'].metadata, readonly_expected_nlink=2)
        with pytest.raises(ValueError, match='exact registry metadata role'):
            p._read_readonly_seal(record, grants, None, f['policy'])


def test_c4_two_opted_registry_roles_cannot_alias_one_inode(fixture, monkeypatch):
    f = fixture
    first = c4_seal(f)
    second_path = f['policy'].metadata/'M02_offline_score_20261103_0156/ANALYSIS_SEALED.json'
    second_path.parent.mkdir()
    os.link(first['path'], second_path)
    second = dict(first, path=str(second_path))
    spec = copy.deepcopy(f['spec'])
    spec['readonly_registry_seal_hardlinks'] = [dict(pin, nlink=2) for pin in (first, second)]
    registry = dict(results_root=str(f['policy'].metadata), source_pins={
        str(Path(pin['path']).relative_to(f['policy'].metadata)):{k:pin[k] for k in ('sha256', 'bytes')}
        for pin in (first, second)})
    spec['exporter_provenance'] = save(Path(spec['exporter_provenance']['path']), registry)
    for paper in spec['papers']:
        contents = unpack(Path(paper['archive']['path']).read_bytes())
        contents['generated/source_provenance.json'] = p.encoded(registry)
        manifest = dict(label=paper['label'], members={name:dict(sha256=p.sha(raw), bytes=len(raw))
            for name, raw in contents.items()})
        paper['manifest'] = save(Path(paper['manifest']['path']), manifest)
        paper['archive'] = save(Path(paper['archive']['path']), archive_bytes(list(contents.items())))
    pin = f['plan_pin'](spec)
    opened, real = [], p.pin_read
    def reading(pin, *args, **kwargs):
        opened.append(pin['path'])
        return real(pin, *args, **kwargs)
    monkeypatch.setattr(p, 'pin_read', reading)
    with pytest.raises(ValueError, match='Input role size/inode collision'):
        p.validate(pin, f['policy'], f['limits'])
    assert opened == [pin['path']]


def test_c4_actual_seal_bytes_still_must_match_pinned_hash(fixture):
    f = fixture
    pin = f['plan_pin'](c4_spec(f))
    alias = f['tmp']/'synthetic-readonly-seal-alias'
    raw = alias.read_bytes()
    changed = raw.replace(b'true', b'null')
    assert changed != raw and len(changed) == len(raw)
    alias.write_bytes(changed)
    assert alias.stat().st_nlink == 2
    with pytest.raises(ValueError, match='hash mismatch before parse'):
        p.validate(pin, f['policy'], f['limits'])


def test_c4_v002_registry_binding_precedes_exception_authorized_read(fixture, monkeypatch):
    f = fixture
    spec = c4_spec(f)
    registry_path = Path(spec['exporter_provenance']['path'])
    registry = json.loads(registry_path.read_bytes())
    registry['unreviewed_annotation'] = 'Not in frozen v002 registry bytes'
    spec['exporter_provenance'] = save(registry_path, registry)
    pin = f['plan_pin'](spec)
    opened, real = [], p.os.open
    def opening(path, *args, **kwargs):
        if path == 'ANALYSIS_SEALED.json':
            opened.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(p.os, 'open', opening)
    with pytest.raises(ValueError, match='exact reviewed v002 manifest'):
        p.validate(pin, f['policy'], f['limits'])
    assert not opened


def test_c4_all_nine_reported_geometries_are_explicitly_supported(fixture):
    f = fixture
    assert len(p.READONLY_REGISTRY_SEALS) == 9
    spec = copy.deepcopy(f['spec'])
    entries = [dict(path=str(f['policy'].metadata/rel), sha256='0'*64, bytes=1, nlink=2)
               for rel in sorted(p.READONLY_REGISTRY_SEALS)]
    spec['readonly_registry_seal_hardlinks'] = entries
    metadata = [{k:entry[k] for k in ('path', 'sha256', 'bytes')} for entry in entries]
    assert len(p._readonly_seal_pins(spec, f['policy'], metadata)) == 9
    p.validate_spec(spec, f['policy'])


@pytest.mark.parametrize('action', ['validate', 'build'])
def test_c4_fixed_trusted_archive_authenticates_external_manifest_before_seal_read(fixture, monkeypatch, action):
    f = fixture
    spec = c4_spec(f)
    paper = next(paper for paper in spec['papers'] if paper['label'] == 'draft_v002')
    original_archive = Path(paper['archive']['path']).read_bytes()
    policy = replace(f['policy'], v002_archive_sha256=paper['archive']['sha256'])
    seal = c4_seal(f)
    changed_seal = save(Path(seal['path']), b'{"unreviewed_synthetic_seal":true}\n')
    registry_path = Path(spec['exporter_provenance']['path'])
    registry = json.loads(registry_path.read_bytes())
    relative = str(Path(seal['path']).relative_to(policy.metadata))
    registry['source_pins'][relative].update(sha256=changed_seal['sha256'], bytes=changed_seal['bytes'])
    spec['exporter_provenance'] = save(registry_path, registry)
    spec['readonly_registry_seal_hardlinks'] = [dict(changed_seal, nlink=2)]
    manifest_path = Path(paper['manifest']['path'])
    manifest = json.loads(manifest_path.read_bytes())
    manifest['members']['generated/source_provenance.json'] = {
        k:spec['exporter_provenance'][k] for k in ('sha256', 'bytes')}
    paper['manifest'] = save(manifest_path, manifest)
    plan = p.make_plan(spec, policy, f['limits'])
    pin = save(f['tmp']/'self-consistent-invalid-manifest-plan.json', plan)
    opened, real = [], p.os.open
    def opening(path, *args, **kwargs):
        if os.fsdecode(path) == 'ANALYSIS_SEALED.json':
            opened.append(path)
        return real(path, *args, **kwargs)
    monkeypatch.setattr(p.os, 'open', opening)
    with pytest.raises(ValueError, match='Paper tar member hash mismatch'):
        if action == 'validate':
            p.validate(pin, policy, f['limits'])
        else:
            p.build(pin, destination(f), policy, f['limits'])
    assert not opened
    assert Path(paper['archive']['path']).read_bytes() == original_archive
    assert not destination(f).exists()


def test_c4_fixed_trusted_archive_accepts_original_registry_seal(fixture):
    f = fixture
    spec = c4_spec(f)
    paper = next(paper for paper in spec['papers'] if paper['label'] == 'draft_v002')
    policy = replace(f['policy'], v002_archive_sha256=paper['archive']['sha256'])
    seal = Path(c4_seal(f)['path'])
    before = p.identity(seal.stat())
    pin = f['plan_pin'](spec)
    assert p.validate(pin, policy, f['limits'])['valid'] is True
    result = p.build(pin, destination(f), policy, f['limits'])
    assert result['complete'] is True
    assert p.verify_delivery(destination(f), result['local_receipt_pin'], policy, f['limits'])['complete'] is True
    assert p.identity(seal.stat()) == before
