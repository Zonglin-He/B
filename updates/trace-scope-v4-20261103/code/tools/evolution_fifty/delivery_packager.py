"""Bounded local Scope-v4 delivery, not scientific admission or publication.

CLI workflow (only after independent code review and explicit operator pins):
  create-plan --spec PATH --sha256 SHA --destination NEW_PLAN_ROOT
  validate --plan NEW_PLAN_ROOT/PIN_PLAN.json --sha256 PRINTED_PLAN_SHA
  build --plan NEW_PLAN_ROOT/PIN_PLAN.json --sha256 SHA --destination NEW_PACKET_ROOT
  verify --receipt NEW_PACKET_ROOT/LOCAL_VERIFY.json --sha256 SHA --destination NEW_PACKET_ROOT

Both destinations must be new direct children of the configured results parent,
named trace_scope_v4_delivery_20261103_*. Validation writes nothing. No imports
of research code, subprocesses, Git, network, services, data or metric engines.
The caller supplies archive layout/manifest pins; no archive is extracted.

Operator spec schema (all pins: absolute path, sha256, bytes): schema_version=1;
papers=[draft_v001,draft_v002] each with label/archive/manifest/members/tar_prefix/
manifest_member/derived_composites; exporter_provenance; reviews={v001:pin,v002:pin};
disposition; code_review (new raw delivery CODE report); documents=[pins];
extra_metadata=[pins]; config_paths=[explicit extra text config paths];
acknowledgements=ACK; optional additional_code_reviews=[new explicit report pins].
Optional readonly_registry_seal_hardlinks=[{path,sha256,bytes,nlink:2}] permits
only exact authenticated registry seals in READONLY_REGISTRY_SEALS. It grants
readonly input access, not alias immutability or any output/code relaxation.
The four composite paths are manifest-relative. tar_prefix
has no trailing slash; manifest_member is null unless the original tar also
contains the external FREEZE manifest, then give its prefix-relative name.
CODE_ROOTS are enumerated as source only; NOTICES and packaged resource/config
text are included. Additional requirements/config text must be explicit. Source
metadata comes exclusively from the exact registry frozen in v002. No document
or result-tree search is used. Full handoffs/heartbeat prompts are not supported.
See test_delivery_packager.py for a synthetic complete spec, not production pins.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile


VERSION = "trace-scope-v4-20261103"
PREFIX = "trace_scope_v4_delivery_20261103_"
CODE_ROOTS = ("trace_ovss", "tools/evolution", "tools/evolution_fifty", "tests")
TEXT_SUFFIXES = {".py", ".txt", ".yaml", ".yml", ".toml", ".json", ".md"}
NOTICES = ("README.md", "pyproject.toml", "trace_ovss/resources/NOTICE.md",
           "trace_ovss/resources/licenses/NACLIP_LICENSE.txt")
EXCLUDED = {"external", "vendor", "third_party", "dependencies", "node_modules",
            "site-packages", "datasets", "results", "artifacts", "outputs",
            "checkpoints", "weights", "cache", "__pycache__", "fixtures"}
SENSITIVE = {"credentials", "credentials.json", "secrets", "secrets.json",
             "secrets.py", "id_rsa", "id_ed25519", "service-account.json"}
DOCUMENTS = {"docs/"+n for n in (
    "M01-risk-results-20261102.md", "M02-spatial-results-20261103.md",
    "M03-legacy-results-20261103.md", "M04-factorial-results-20261103.md",
    "G-primary-results-20261102.md", "G03-severity-results-20261103.md",
    "V01-value-results-20261103.md")}
DOCUMENTS.add("paper_evolution/scope_v4_integration_outline.md")
DISPOSITION = "docs/scope-v4-paper-review-disposition-20261103.md"
METADATA_NAMES = {"ANALYSIS_SEALED.json", "STARTED.json", "result.json", "RESULT.json",
                  "DATASET_RESULTS.json", "CONDITION_RESULTS.json", "PAIRED_SUMMARIES.json",
                  "CURVE_RESULTS.json"}
READONLY_REGISTRY_SEALS = frozenset({
    "G03_offline_score_20261103_1455/ANALYSIS_SEALED.json",
    "G_offline_score_20261102_2321/ANALYSIS_SEALED.json",
    "M01_offline_score_20261102_1942/ANALYSIS_SEALED.json",
    "M01_offline_score_20261102_1942/coco_object/ANALYSIS_SEALED.json",
    "M01_offline_score_20261102_1942/pc60/ANALYSIS_SEALED.json",
    "M02_offline_score_20261103_0156/ANALYSIS_SEALED.json",
    "M03_offline_score_20261103_0856/ANALYSIS_SEALED.json",
    "M04_offline_score_20261103_0646/ANALYSIS_SEALED.json",
    "V01_offline_score_20261103_1258/ANALYSIS_SEALED.json",
})
ACK = dict(all_referenced_reviews_completed=True, local_packaging_authorized=True,
           publication_actions_authorized=False, paper_acceptance_claimed=False,
           R1_status="PARTIAL", local_mask_subset_generation_recovered=False)
SCOPE = {
    "current_code": "Current explicit first-party roots, not a claim of frozen-runtime identity or dependency closure.",
    "paper": "Original whole archives preserved byte-for-byte; raw reviews/disposition are separate post-freeze additions.",
    "science": "No metric computation, new experiment, positivity/rating filter, GT permission or overall acceptance claim.",
    "R1": "PARTIAL: historical local COCO-CD/Waterbirds mask and subset creation remains unrecovered; not fully reproducible.",
    "excluded": "No datasets, raw RGB/GT, predictions/NPY, weights/models, count/source-feature/transition streams, vendor/dependency trees, old release exports or whole evidence archive.",
    "tests": "Source presence is not execution evidence; six reference-mutation demonstrations remain NOT RUN by this packaging task.",
    "publication": "Local packet only. No Git, network, push, upload or release action; main author handles any later Zonglin-He/B publication.",
    "not_turnkey": "Other first-party tool roots, external dependencies and dataset construction are not silently bundled; this is not turnkey reconstruction.",
    "credentials": "Known sensitive/hidden paths are refused. Explicit code/config membership requires author review; this is not a semantic secret-scanning certification.",
    "verification_contract": "Completion requires the pinned LOCAL_VERIFY.json, exact three-file output membership and successful archive/manifest byte verification. Any FAILED.json invalidates completion even if a receipt was already written before an ordinary close error.",
    "evidence_basis": "Pinned reviewed-exporter metadata and raw review/disposition bytes, not a fresh service/completion/scientific audit. Operator completion acknowledgements are not machine grading of Markdown conclusions.",
}


@dataclass(frozen=True)
class Limits:
    code: int = 2*1024**2
    metadata: int = 8*1024**2
    review: int = 1024**2
    paper_archive: int = 32*1024**2
    paper_member: int = 32*1024**2
    paper_expanded: int = 128*1024**2
    total: int = 192*1024**2
    package: int = 256*1024**2
    members: int = 4096
    reserve: int = 16*1024**3


@dataclass(frozen=True)
class Policy:
    worktree: Path = Path("/home/wwww/.codex/worktrees/trace-simplification/hzl")
    results: Path = Path("/home/wwww/hzl/results")
    metadata_count: int = 36
    v002_members: int = 69
    v002_archive_sha256: str | None = "b72931845b4ae3e843fbd0730eeaa4b729be4c2cbf574ef1170139481e230bbc"
    review_sha256: dict = field(default_factory=lambda: {
        "v001": "514073234adc093e806f34d3f0e9a0c35ada6e999ec16de717f907a7fb641d0d",
        "v002": "a4148ffc476acb6366eb709a0753d013034166aa0fd7559bb884f652355e176f"})
    disposition_sha256: str | None = "965361320254f6781479018184fa893fb6fa4ebd04b6e90bae62664c7a5b09c4"

    @property
    def paper(self):
        return self.worktree/"paper_evolution/scope_v4_20261103"

    @property
    def metadata(self):
        return self.results/"trace_mechanism_20261102"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)+"\n").encode()


def pairs(items):
    result = {}
    for k, v in items:
        require(k not in result, "Duplicate JSON key")
        result[k] = v
    return result


def parsed(raw):
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def relative(value):
    require(type(value) is str and value and len(value) <= 1024, "Invalid member path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and str(p) == value and ".." not in p.parts
            and all(not x.startswith(".") for x in p.parts) and "\\" not in value
            and not any(ord(c) < 32 for c in value), "Unsafe member path")
    return p


def absolute(value):
    require(type(value) is str and value.startswith("/") and len(value) <= 4096,
            "Absolute path required")
    p = Path(value)
    require(str(p) == value and ".." not in p.parts and "\\" not in value
            and not any(ord(c) < 32 for c in value),
            "Noncanonical absolute path")
    return p


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink, s.st_mode)


def directory_id(s):
    return (s.st_dev, s.st_ino, s.st_mode)


def close_all(fds):
    error = None
    for fd in reversed(fds):
        try:
            os.close(fd)
        except OSError as current:
            error = error or current
    if error is not None:
        raise error


@contextmanager
def directory(path):
    """Authenticate every absolute directory component, without following links."""
    p = absolute(str(path))
    require(len(p.parts) <= 64, "Directory depth bound")
    fds, chain = [], []
    try:
        fd = os.open("/", os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        fds.append(fd)
        for name in p.parts[1:]:
            child = os.open(name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
            fds.append(child)
            chain.append((fd, name, directory_id(os.fstat(child))))
            fd = child
        yield fd
        for parent, name, observed in chain:
            require(directory_id(os.stat(name, dir_fd=parent, follow_symlinks=False)) == observed,
                    "Directory path identity drift")
        require(directory_id(p.lstat()) == directory_id(os.fstat(fd)), "Directory root drift")
    finally:
        close_all(fds)


def observe(path, cap):
    return _observe_regular(path, cap, 1)


def _observe_regular(path, cap, nlink):
    path = absolute(str(path))
    with directory(path.parent) as parent:
        s = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == nlink and 0 <= s.st_size <= cap,
                "Not bounded single-link regular input" if nlink == 1 else
                "Not bounded exact-nlink2 readonly registry seal")
        return identity(s)


def read(path, cap, expected_sha=None, expected_bytes=None, expected_identity=None):
    return _read_with_observer(path, cap, expected_sha, expected_bytes, expected_identity, observe)


def _read_with_observer(path, cap, expected_sha, expected_bytes, expected_identity, observer):
    path = absolute(str(path))
    before = observer(path, cap)
    if expected_identity is not None:
        require(before == tuple(expected_identity), "Input changed before open")
    if expected_bytes is not None:
        require(before[2] == expected_bytes, "Input size mismatch")
    with directory(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=parent)
        try:
            require(identity(os.fstat(fd)) == before, "Pre-open input binding drift")
            raw = bytearray()
            while len(raw) <= before[2]:
                block = os.read(fd, min(65536, before[2]+1-len(raw)))
                if not block:
                    break
                raw.extend(block)
            require(len(raw) == before[2] and identity(os.fstat(fd)) == before
                    == identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False)),
                    "Input FD/path drift")
        finally:
            os.close(fd)
    raw = bytes(raw)
    require(expected_sha is None or sha(raw) == expected_sha, "Input hash mismatch before parse")
    return raw, before


def valid_pin(pin):
    require(type(pin) is dict and set(pin) == {"path", "sha256", "bytes"}, "Exact pin fields required")
    absolute(pin["path"])
    require(type(pin["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
            and type(pin["bytes"]) is int and pin["bytes"] >= 0, "Invalid hash/size pin")
    return pin


def pin_read(pin, cap, observed=None):
    valid_pin(pin)
    return read(pin["path"], cap, pin["sha256"], pin["bytes"], observed)[0]


def safe_text_path(name):
    p = relative(name)
    require(not any(x.lower() in EXCLUDED|SENSITIVE or x.lower().startswith(".env") for x in p.parts)
            and p.suffix.lower() in TEXT_SUFFIXES, "Excluded code/config path")
    require("data" not in p.parts or p.parts[:2] == ("trace_ovss", "data"), "Data tree is not code")
    return p


def code_inventory(policy, configs):
    require(type(configs) is list and len(configs) == len(set(configs)), "Explicit unique config paths required")
    selected, visited = set(NOTICES), 0
    for name in configs:
        p = safe_text_path(name)
        require(len(p.parts) == 1 or any(p.is_relative_to(PurePosixPath(r)) for r in CODE_ROOTS),
                "Config outside explicit first-party roots")
        selected.add(name)
    def walk(folder):
        nonlocal visited
        children = []
        with directory(policy.worktree/folder) as fd:
            entries = []
            with os.scandir(fd) as it:
                for entry in it:
                    visited += 1
                    require(visited <= 16384, "Code inventory traversal bound")
                    entries.append(entry)
            for entry in sorted(entries, key=lambda e:e.name):
                name = (PurePosixPath(folder)/entry.name).as_posix()
                if entry.name.startswith(".") or entry.name.lower() in EXCLUDED|SENSITIVE:
                    continue
                if entry.name == "data" and name != "trace_ovss/data":
                    continue
                s = entry.stat(follow_symlinks=False)
                require(not stat.S_ISLNK(s.st_mode), "Symlink in first-party source tree")
                if stat.S_ISDIR(s.st_mode):
                    children.append(name)
                elif (name.endswith(".py") or name.startswith(("trace_ovss/resources/", "trace_ovss/configs/"))
                      and PurePosixPath(name).suffix in TEXT_SUFFIXES):
                    safe_text_path(name)
                    require(stat.S_ISREG(s.st_mode), "Special source member")
                    selected.add(name)
        for name in children:
            walk(name)
    for root in CODE_ROOTS:
        walk(root)
    require(len(selected) <= 2048, "Code member bound")
    for name in selected:
        safe_text_path(name)
    return sorted(selected)


def metadata_path(pin, policy):
    path = absolute(pin["path"])
    require(path.is_relative_to(policy.metadata), "Metadata outside fixed results root")
    rel = relative(path.relative_to(policy.metadata).as_posix())
    require(not any(x.lower() in EXCLUDED|{"predictions", "episodes", "runtime", "reference",
                "rgb", "labels", "annotations", "snapshots", "states"} for x in rel.parts),
            "Payload/runtime metadata role refused")
    require(rel.name in METADATA_NAMES or re.fullmatch(
        r"(?:M[0-9]{2}|G[0-9]{2}|G_primary|G|V01)_.*terminal_[0-9]{3}\.json", rel.name),
        "Not an allowlisted small summary/seal/terminal")
    return rel.as_posix()


def registry_metadata(registry, policy):
    pins = registry["source_pins"]
    require(type(pins) is dict and len(pins) == policy.metadata_count, "Exporter metadata membership count")
    require(registry["results_root"] == str(policy.metadata), "Exporter metadata root differs")
    result = []
    for rel, value in sorted(pins.items()):
        relative(rel)
        pin = dict(path=str(policy.metadata/rel), sha256=value["sha256"], bytes=value["bytes"])
        valid_pin(pin)
        metadata_path(pin, policy)
        result.append(pin)
    return result


def _readonly_seal_pins(spec, policy, metadata=None):
    entries = spec.get("readonly_registry_seal_hardlinks", [])
    require(type(entries) is list and len(entries) <= len(READONLY_REGISTRY_SEALS),
            "Bounded explicit readonly registry seal opt-ins required")
    result = {}
    for entry in entries:
        require(type(entry) is dict and set(entry) == {"path", "sha256", "bytes", "nlink"}
                and type(entry["nlink"]) is int and entry["nlink"] == 2,
                "Exact readonly registry seal pin and integer nlink2 required")
        pin = {k:entry[k] for k in ("path", "sha256", "bytes")}
        valid_pin(pin)
        require(pin["bytes"] > 0 and metadata_path(pin, policy) in READONLY_REGISTRY_SEALS,
                "Unsupported readonly registry seal geometry")
        require(pin["path"] not in result, "Duplicate readonly registry seal opt-in")
        result[pin["path"]] = pin
    if metadata is not None:
        registry = {pin["path"]:pin for pin in metadata}
        require(len(registry) == len(metadata), "Duplicate registry metadata paths")
        for path, pin in result.items():
            require(path in registry and encoded(pin) == encoded(registry[path]),
                    "Unused or mismatched readonly registry seal opt-in")
    return result


def _observe_readonly_seal(pin, cap, policy):
    valid_pin(pin)
    require(metadata_path(pin, policy) in READONLY_REGISTRY_SEALS,
            "Unsupported readonly registry seal geometry")
    observed = _observe_regular(pin["path"], cap, 2)
    require(observed[2] == pin["bytes"], "Readonly registry seal size mismatch")
    return observed


def _readonly_record_pin(record, grants):
    pin = record["pin"]
    require(record["role"] == "sealed_summary_metadata" and pin["path"] in grants
            and encoded(pin) == encoded(grants[pin["path"]])
            and type(record.get("readonly_expected_nlink")) is int
            and record["readonly_expected_nlink"] == 2,
            "Readonly hardlinks are restricted to the exact registry metadata role")
    return pin


def _read_readonly_seal(record, grants, observed, policy):
    pin = _readonly_record_pin(record, grants)
    def observer(path, cap):
        require(str(path) == pin["path"], "Readonly registry seal path mismatch")
        return _observe_readonly_seal(pin, cap, policy)
    return _read_with_observer(pin["path"], record["cap"], pin["sha256"], pin["bytes"], observed, observer)[0]


def validate_spec(spec, policy):
    keys = {"schema_version", "papers", "exporter_provenance",
        "reviews", "disposition", "code_review", "documents", "extra_metadata", "config_paths",
        "acknowledgements"}
    require(type(spec) is dict and keys <= set(spec)
        <= keys|{"additional_code_reviews", "readonly_registry_seal_hardlinks"}
        and type(spec["schema_version"]) is int and spec["schema_version"] == 1,
        "Exact specification schema required")
    _readonly_seal_pins(spec, policy)
    require(type(spec["acknowledgements"]) is dict and encoded(spec["acknowledgements"]) == encoded(ACK),
            "Explicit completed-review/local-only/R1-partial acknowledgements required")
    require(type(spec["papers"]) is list and {p["label"] for p in spec["papers"]} == {"draft_v001", "draft_v002"}
            and len(spec["papers"]) == 2, "Both original whole paper versions required")
    require(type(spec["reviews"]) is dict and set(spec["reviews"]) == {"v001", "v002"}, "Both raw reviews required")
    records = []
    def add(pin, target, role, cap):
        valid_pin(pin)
        require(pin["bytes"] > 0, "Nonempty bounded publication input required")
        relative(target)
        records.append(dict(pin=pin, member=target, role=role, cap=cap))
    for paper in spec["papers"]:
        require(set(paper) == {"label", "archive", "manifest", "members", "tar_prefix",
            "manifest_member", "derived_composites"}, "Exact paper layout required")
        label = paper["label"]
        require(type(paper["members"]) is int and 0 < paper["members"] <= 256, "Paper member count bound")
        if label == "draft_v002":
            require(paper["members"] == policy.v002_members, "Wrong v002 member count")
            require(policy.v002_archive_sha256 is None or paper["archive"]["sha256"] == policy.v002_archive_sha256,
                    "Not the exact reviewed v002 whole archive")
        for kind in ("archive", "manifest"):
            path = absolute(paper[kind]["path"])
            require(path.is_relative_to(policy.paper), "Paper source outside explicit manuscript root")
        require(paper["archive"]["path"].endswith((".tar.gz", ".tgz"))
                and paper["manifest"]["path"].endswith(".json"), "Paper archive/manifest extensions")
        require(type(paper["tar_prefix"]) is str and (not paper["tar_prefix"] or relative(paper["tar_prefix"])),
                "Explicit archive prefix required")
        if paper["manifest_member"] is not None:
            relative(paper["manifest_member"])
        require(type(paper["derived_composites"]) is list and len(paper["derived_composites"]) == 4
                and len(set(paper["derived_composites"])) == 4, "Exactly four authorized derived composites required")
        for name in paper["derived_composites"]:
            p = relative(name)
            require(p.parts[0] == "figures" and p.suffix == ".png", "Only explicit derived figure PNGs")
        add(paper["archive"], f"paper/{label}/original.tar.gz", "original_paper_archive", 0)
        add(paper["manifest"], f"paper/{label}/FREEZE.json", "paper_manifest", 0)
    for label, pin in spec["reviews"].items():
        name = "manuscript_review_draft_v001.md" if label == "v001" else "manuscript_review_draft_v002_closure.md"
        require(pin["path"] == str(policy.paper/"reviews"/name), "Not the explicit raw manuscript review")
        require(not policy.review_sha256 or pin["sha256"] == policy.review_sha256[label], "Wrong actual raw review pin")
        add(pin, "reviews/"+name, "raw_manuscript_review", 0)
    require(spec["disposition"]["path"] == str(policy.worktree/DISPOSITION), "Post-review disposition path")
    require(policy.disposition_sha256 is None or spec["disposition"]["sha256"] == policy.disposition_sha256,
            "Wrong actual post-review disposition pin")
    add(spec["disposition"], "publication/"+DISPOSITION, "post_review_disposition", 0)
    cr = absolute(spec["code_review"]["path"])
    require(cr.parent == policy.worktree/"docs" and cr.suffix == ".md"
            and "delivery" in cr.name and "code" in cr.name, "New delivery CODE raw report required")
    add(spec["code_review"], "reviews/code/"+cr.name, "raw_delivery_code_review", 0)
    extra_reviews = spec.get("additional_code_reviews", [])
    require(type(extra_reviews) is list and len(extra_reviews) <= 8, "Bounded explicit new CODE reports")
    for pin in extra_reviews:
        path = absolute(pin["path"])
        require(path.parent == policy.worktree/"docs" and path.name.endswith("20261103.md")
                and "code" in path.name and any(word in path.name for word in ("scope-v4", "scope_v4", "export", "delivery")),
                "Only explicit new Scope-v4 CODE reports, not old grades")
        add(pin, "reviews/code/"+path.name, "additional_new_raw_code_review", 0)
    require(spec["exporter_provenance"]["path"] == str(policy.paper/"generated/source_provenance.json"),
            "Explicit exporter registry required")
    add(spec["exporter_provenance"], "evidence/exporter_source_provenance.json", "exporter_registry", 0)
    require(type(spec["documents"]) is list and type(spec["extra_metadata"]) is list, "Explicit documents/extra metadata lists")
    for pin in spec["documents"]:
        path = absolute(pin["path"])
        require(path.is_relative_to(policy.worktree) and path.relative_to(policy.worktree).as_posix() in DOCUMENTS,
                "Publication document not allowlisted; full handoffs/heartbeats excluded")
        add(pin, "publication/"+path.relative_to(policy.worktree).as_posix(), "publication_document", 0)
    for pin in spec["extra_metadata"]:
        add(pin, "evidence/"+metadata_path(pin, policy), "extra_small_metadata", 0)
    return records


def make_plan(spec, policy=Policy(), limits=Limits()):
    records = validate_spec(spec, policy)
    registry = parsed(pin_read(spec["exporter_provenance"], limits.metadata))
    metadata = registry_metadata(registry, policy)
    _readonly_seal_pins(spec, policy, metadata)
    code = []
    total = sum(r["pin"]["bytes"] for r in records)+sum(r["bytes"] for r in metadata)
    require(total <= limits.total, "Input total cap before source snapshot")
    for name in code_inventory(policy, spec["config_paths"]):
        observed = observe(policy.worktree/name, limits.code)
        total += observed[2]
        require(total <= limits.total, "Input total cap during source snapshot")
        raw, _ = read(policy.worktree/name, limits.code, expected_identity=observed)
        require(b"\0" not in raw, "Binary source/config refused")
        raw.decode("utf-8")
        code.append(dict(path=str(policy.worktree/name), sha256=sha(raw), bytes=len(raw)))
    return dict(schema_version=1, version=VERSION, worktree=str(policy.worktree),
        results_root=str(policy.metadata), specification=spec, source_metadata=metadata, code=code,
        limits=limits.__dict__, scope=SCOPE)


def plan_records(plan, policy, limits):
    require(type(plan) is dict and set(plan) == {"schema_version", "version", "worktree", "results_root",
        "specification", "source_metadata", "code", "limits", "scope"}, "Exact pin-plan schema")
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1
            and plan["version"] == VERSION and plan["worktree"] == str(policy.worktree)
            and plan["results_root"] == str(policy.metadata) and encoded(plan["scope"]) == encoded(SCOPE)
            and encoded(plan["limits"]) == encoded(limits.__dict__), "Plan policy/root/limits drift")
    records = validate_spec(plan["specification"], policy)
    for r in records:
        r["cap"] = (limits.paper_archive if r["role"] == "original_paper_archive" else
            limits.metadata if r["role"] in ("paper_manifest", "exporter_registry", "extra_small_metadata") else limits.review)
    require(type(plan["source_metadata"]) is list and len(plan["source_metadata"]) == policy.metadata_count,
            "Complete exporter metadata list required")
    for pin in plan["source_metadata"]:
        valid_pin(pin)
        records.append(dict(pin=pin, member="evidence/"+metadata_path(pin, policy), role="sealed_summary_metadata", cap=limits.metadata))
    require(type(plan["code"]) is list, "Explicit code member list required")
    expected = code_inventory(policy, plan["specification"]["config_paths"])
    actual = []
    for pin in plan["code"]:
        valid_pin(pin)
        path = absolute(pin["path"])
        require(path.is_relative_to(policy.worktree), "Code outside worktree")
        name = path.relative_to(policy.worktree).as_posix()
        safe_text_path(name)
        actual.append(name)
        records.append(dict(pin=pin, member="implementation/current_first_party/"+name, role="current_code_not_frozen_runtime", cap=limits.code))
    require(actual == expected, "Current code/config membership differs from pin plan")
    require(len(records)+2 <= limits.members and len({r["member"] for r in records}) == len(records)
            and len({r["pin"]["path"] for r in records}) == len(records), "Duplicate/capped input roles or members")
    require(sum(r["pin"]["bytes"] for r in records) <= limits.total, "Input total cap")
    grants = _readonly_seal_pins(plan["specification"], policy, plan["source_metadata"])
    for record in records:
        if record["pin"]["path"] in grants:
            record["readonly_expected_nlink"] = 2
            _readonly_record_pin(record, grants)
    return records


def paper_check(raw, manifest_raw, paper, limits):
    manifest = parsed(manifest_raw)
    require(manifest.get("label") == paper["label"], "Paper manifest label mismatch")
    members = manifest["members"]
    require(type(members) is dict and len(members) == paper["members"], "Paper manifest membership differs")
    expected, total = {}, 0
    for name, pin in members.items():
        p = relative(name)
        require(type(pin) is dict and set(pin) == {"sha256", "bytes"} and type(pin["bytes"]) is int
                and 0 <= pin["bytes"] <= limits.paper_member and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]),
                "Paper member size/hash pin")
        require(p.suffix in {".tex", ".bib", ".sty", ".bst", ".md", ".py", ".json", ".pdf"}
                or name in paper["derived_composites"], "Unauthorized paper binary/member")
        expected[name] = pin
        total += pin["bytes"]
    require(set(n for n in expected if n.endswith(".png")) == set(paper["derived_composites"]),
            "Paper derived-composite membership differs")
    if paper["manifest_member"] is not None:
        require(paper["manifest_member"] not in expected, "Manifest-member collision")
        expected[paper["manifest_member"]] = dict(bytes=len(manifest_raw), sha256=sha(manifest_raw))
    require(total <= limits.paper_expanded, "Paper declared expanded cap")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(limits.paper_expanded+1)
    require(len(expanded) <= limits.paper_expanded, "Paper decompression cap")
    prefix = paper["tar_prefix"]
    allowed_dirs = set()
    for name in expected:
        full = PurePosixPath(prefix)/name if prefix else PurePosixPath(name)
        allowed_dirs.update(str(x) for x in full.parents if str(x) != ".")
    seen, directories = set(), set()
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
        for i, member in enumerate(archive):
            require(i < 1024, "Paper tar entry bound")
            name = member.name.rstrip("/") if member.isdir() else member.name
            relative(name)
            if member.isdir():
                require(name in allowed_dirs and name not in directories, "Foreign/duplicate paper directory")
                directories.add(name)
                continue
            require(member.isfile() and not member.linkname and member.size <= limits.paper_member,
                    "Paper links/special/large members refused")
            if prefix:
                require(name.startswith(prefix+"/"), "Paper tar prefix mismatch")
                name = name[len(prefix)+1:]
            require(name in expected and name not in seen, "Paper tar member not declared or duplicate")
            seen.add(name)
            pin = expected[name]
            require(member.size == pin["bytes"], "Paper tar member size mismatch")
            with archive.extractfile(member) as stream:
                content = stream.read(limits.paper_member+1)
            require(len(content) == pin["bytes"] and sha(content) == pin["sha256"], "Paper tar member hash mismatch")
        require(not any(expanded[archive.offset:]), "Non-padding trailing paper bytes")
    require(seen == set(expected), "Missing paper archive members")
    return dict(frozen_members=len(members), archive_sha256=sha(raw), original_bytes_preserved=True)


def materialize(plan, plan_raw, policy, limits):
    records = plan_records(plan, policy, limits)
    spec = plan["specification"]
    grants = _readonly_seal_pins(spec, policy, plan["source_metadata"])
    observations, inodes, payloads = {}, set(), {"PIN_PLAN.json": plan_raw}
    papers = {}
    # Every leaf is type/size/inode checked before any input-content open.
    for r in records:
        pin = r["pin"]
        observed = (_observe_readonly_seal(_readonly_record_pin(r, grants), r["cap"], policy)
                    if pin["path"] in grants else observe(pin["path"], r["cap"]))
        require(observed[2] == pin["bytes"] and observed[:2] not in inodes, "Input role size/inode collision")
        inodes.add(observed[:2])
        observations[pin["path"]] = observed
    if grants:
        # Complete the stat/alias barrier first, then authenticate registry bytes
        # before any exception-authorized sealed metadata content is opened.
        r = next(r for r in records if r["role"] == "exporter_registry")
        raw = pin_read(r["pin"], r["cap"], observations[r["pin"]["path"]])
        metadata = registry_metadata(parsed(raw), policy)
        require(encoded(metadata) == encoded(plan["source_metadata"]), "Exporter allowlist/pin membership mismatch")
        grants = _readonly_seal_pins(spec, policy, metadata)
        payloads[r["member"]] = raw
        paper = next(paper for paper in spec["papers"] if paper["label"] == "draft_v002")
        r = next(r for r in records if r["role"] == "paper_manifest" and r["pin"] == paper["manifest"])
        raw = pin_read(r["pin"], r["cap"], observations[r["pin"]["path"]])
        frozen_registry = parsed(raw)["members"].get("generated/source_provenance.json")
        require(frozen_registry == {k:spec["exporter_provenance"][k] for k in ("sha256", "bytes")},
                "Exporter registry not bound to exact reviewed v002 manifest")
        payloads[r["member"]] = raw
        manifest_raw = raw
        r = next(r for r in records if r["role"] == "original_paper_archive" and r["pin"] == paper["archive"])
        raw = pin_read(r["pin"], r["cap"], observations[r["pin"]["path"]])
        papers["draft_v002"] = paper_check(raw, manifest_raw, paper, limits)
        payloads[r["member"]] = raw
    for r in records:
        if r["member"] in payloads:
            continue
        observed = observations[r["pin"]["path"]]
        raw = (_read_readonly_seal(r, grants, observed, policy) if r["pin"]["path"] in grants
               else pin_read(r["pin"], r["cap"], observed))
        if r["role"] == "current_code_not_frozen_runtime":
            require(b"\0" not in raw, "Binary code/config refused")
            raw.decode("utf-8")
        payloads[r["member"]] = raw
    registry = parsed(payloads["evidence/exporter_source_provenance.json"])
    require(encoded(registry_metadata(registry, policy)) == encoded(plan["source_metadata"]),
            "Exporter allowlist/pin membership mismatch")
    for paper in spec["papers"]:
        base = "paper/"+paper["label"]+"/"
        if paper["label"] not in papers:
            papers[paper["label"]] = paper_check(payloads[base+"original.tar.gz"], payloads[base+"FREEZE.json"], paper, limits)
        if paper["label"] == "draft_v002":
            frozen_registry = parsed(payloads[base+"FREEZE.json"])["members"].get("generated/source_provenance.json")
            require(frozen_registry == {k:spec["exporter_provenance"][k] for k in ("sha256", "bytes")},
                    "Exporter registry not bound to exact reviewed v002 manifest")
    payloads["DELIVERY_SCOPE.md"] = ("# Scope-v4 bounded local delivery\n\n"+
        "\n\n".join(k+": "+v for k,v in SCOPE.items())+"\n").encode()
    require(sum(map(len, payloads.values())) <= limits.total, "Payload total cap including plan/scope")
    manifest = dict(schema_version=1, version=VERSION, plan_sha256=sha(plan_raw), scope=SCOPE,
        paper_checks=papers, source_records=records,
        payload_members={n:dict(sha256=sha(b), bytes=len(b)) for n,b in sorted(payloads.items())},
        archive_member_allowlist=sorted([*payloads, "MANIFEST.json"]),
        self_binding="MANIFEST.json and delivery.tar.gz hashes/sizes are bound by external LOCAL_VERIFY.json; no self-hash claim.")
    manifest_raw = encoded(manifest)
    require(len(manifest_raw) <= limits.metadata and len(payloads)+1 <= limits.members, "Manifest cap")
    payloads["MANIFEST.json"] = manifest_raw
    return payloads, observations, manifest


def pack(payloads, limits):
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, content in sorted(payloads.items()):
                relative(name)
                header = tarfile.TarInfo(name)
                header.size, header.mode, header.mtime = len(content), 0o644, 0
                archive.addfile(header, io.BytesIO(content))
                require(raw.tell() <= limits.package, "Compressed packet cap")
    result = raw.getvalue()
    require(len(result) <= limits.package, "Closed compressed packet cap")
    return result


def verify_packet(raw, payloads, limits):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(limits.package+1)
    require(len(expanded) <= limits.package, "Packet expanded cap")
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
        for i, member in enumerate(archive):
            require(i < limits.members, "Packet member cap")
            name = member.name
            relative(name)
            require(member.isfile() and not member.linkname and name in payloads and name not in seen,
                    "Packet member type/membership mismatch")
            seen.add(name)
            expected = payloads[name]
            require(member.size == len(expected), "Packet member size mismatch")
            with archive.extractfile(member) as stream:
                actual = stream.read(len(expected)+1)
            require(actual == expected, "Packet member bytes mismatch")
        require(not any(expanded[archive.offset:]), "Non-padding trailing packet bytes")
    require(seen == set(payloads), "Packet members missing")


class OwnedOutput:
    def __init__(self, destination, policy):
        self.path = absolute(str(destination))
        require(self.path.parent == policy.results and self.path.name.startswith(PREFIX)
                and re.fullmatch(r"[A-Za-z0-9_-]+", self.path.name), "Foreign delivery output root")
        self.policy, self.observed = policy, None

    def __enter__(self):
        self.parent_context = directory(self.path.parent)
        self.parent = self.parent_context.__enter__()
        self.fd = None
        try:
            os.mkdir(self.path.name, 0o700, dir_fd=self.parent)
            self.fd = os.open(self.path.name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=self.parent)
            self.observed = directory_id(os.fstat(self.fd))
            self.check()
            return self
        except BaseException:
            exc = sys.exc_info()
            try:
                if self.fd is not None:
                    os.close(self.fd)
            finally:
                self.parent_context.__exit__(*exc)
            raise

    def check(self):
        require(directory_id(os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)) == self.observed
                == directory_id(os.fstat(self.fd)), "Owned output root replaced")

    def write(self, name, raw):
        require(relative(name).name == name, "Flat owned output required")
        self.check()
        fd = os.open(name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        try:
            before = os.fstat(fd)
            require(before.st_nlink == 1 and stat.S_ISREG(before.st_mode), "Output FD type/link")
            offset = 0
            while offset < len(raw):
                count = os.write(fd, raw[offset:offset+65536])
                require(count > 0, "Short output write")
                offset += count
            os.fsync(fd)
            observed = identity(os.fstat(fd))
            require(observed[2] == len(raw) and observed[:2] == identity(before)[:2]
                    and observed == identity(os.stat(name, dir_fd=self.fd, follow_symlinks=False)), "Output write binding")
        finally:
            os.close(fd)
        os.fsync(self.fd)
        self.check()
        return observed

    def __exit__(self, *exc):
        try:
            self.check()
        finally:
            try:
                os.close(self.fd)
            finally:
                self.parent_context.__exit__(*exc)


def rebind(observations, policy, spec):
    grants = _readonly_seal_pins(spec, policy)
    require(set(grants) <= set(observations), "Unused readonly registry seal opt-in at final rebind")
    for path, expected in observations.items():
        observed = (_observe_readonly_seal(grants[path], expected[2], policy) if path in grants
                    else observe(path, expected[2]))
        require(observed == expected, "Final input identity drift")
    names = code_inventory(policy, spec["config_paths"])
    expected = sorted(str(Path(p).relative_to(policy.worktree)) for p in observations
        if any(Path(p).is_relative_to(policy.worktree/r) for r in CODE_ROOTS)
        and str(Path(p).relative_to(policy.worktree)) in names)
    # Root notices/configs are already bound above; compare code membership separately.
    require([n for n in names if any(PurePosixPath(n).is_relative_to(PurePosixPath(r)) for r in CODE_ROOTS)] == expected,
            "Final code membership drift")


def validate(plan_pin, policy=Policy(), limits=Limits()):
    require(plan_pin["path"].endswith(".json"), "Explicit JSON plan required")
    raw = pin_read(plan_pin, limits.metadata)
    plan = parsed(raw)
    payloads, observations, manifest = materialize(plan, raw, policy, limits)
    rebind(observations, policy, plan["specification"])
    return dict(valid=True, writes_performed=False, plan_sha256=sha(raw),
                members=len(payloads), input_files=len(observations), scope=SCOPE)


def verify_delivery(destination, receipt_pin, policy=Policy(), limits=Limits()):
    """Portable byte verification within a permitted local packet, not new admission."""
    output = OwnedOutput(destination, policy)
    require(receipt_pin["path"] == str(output.path/"LOCAL_VERIFY.json"), "Exact local receipt path required")
    with directory(output.path) as fd:
        directory_state = identity(os.fstat(fd))
        def exact_membership():
            with os.scandir(fd) as entries:
                names = []
                for entry in entries:
                    names.append(entry.name)
                    require(len(names) <= 3, "Failed/foreign output members; not complete")
            require(set(names) == {"LOCAL_VERIFY.json", "MANIFEST.json", "delivery.tar.gz"},
                    "Failed/incomplete output membership")
        exact_membership()
        bindings = {}
        def bound(name, cap, value):
            path = output.path/name
            raw, observed = read(path, cap, value["sha256"], value["bytes"])
            bindings[name] = (cap, observed)
            return raw
        receipt = parsed(bound("LOCAL_VERIFY.json", limits.metadata, receipt_pin))
        require(receipt["complete"] is True and receipt["local_archive_verified"] is True
                and receipt["version"] == VERSION and receipt["publication_actions_performed"] is False
                and encoded(receipt["scope"]) == encoded(SCOPE), "Local receipt contract mismatch")
        manifest_raw = bound("MANIFEST.json", limits.metadata, receipt["manifest"])
        manifest = parsed(manifest_raw)
        require(manifest["plan_sha256"] == receipt["plan_sha256"] and encoded(manifest["scope"]) == encoded(SCOPE),
                "Receipt/manifest binding mismatch")
        expected = manifest["payload_members"]
        require(type(expected) is dict and len(expected)+1 <= limits.members
                and set(expected)|{"MANIFEST.json"} == set(manifest["archive_member_allowlist"])
                and len(manifest["archive_member_allowlist"]) == len(expected)+1 == receipt["member_count"],
                "Packet manifest membership mismatch")
        require({"PIN_PLAN.json", "DELIVERY_SCOPE.md", "paper/draft_v001/original.tar.gz",
                 "paper/draft_v002/original.tar.gz"} <= set(expected), "Required delivery members missing")
        for name, pin in expected.items():
            relative(name)
            require(type(pin) is dict and set(pin) == {"sha256", "bytes"}
                    and type(pin["bytes"]) is int and 0 <= pin["bytes"] <= limits.total,
                    "Invalid packet payload pin")
        require(sum(pin["bytes"] for pin in expected.values()) <= limits.total, "Packet payload total cap")
        raw = bound("delivery.tar.gz", limits.package, receipt["archive"])
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            expanded = stream.read(limits.package+1)
        require(len(expanded) <= limits.package, "Packet expanded cap")
        seen = set()
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            for i, member in enumerate(archive):
                relative(member.name)
                require(i < limits.members and member.isfile() and not member.linkname
                        and member.name in manifest["archive_member_allowlist"] and member.name not in seen,
                        "Packet type/member mismatch")
                seen.add(member.name)
                pin = dict(sha256=sha(manifest_raw), bytes=len(manifest_raw)) if member.name == "MANIFEST.json" else expected[member.name]
                require(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= limits.total
                        and member.size == pin["bytes"], "Packet declared member size")
                with archive.extractfile(member) as stream:
                    content = stream.read(member.size+1)
                require(len(content) == pin["bytes"] and sha(content) == pin["sha256"], "Packet declared member hash")
                if member.name == "PIN_PLAN.json":
                    require(sha(content) == receipt["plan_sha256"], "Packet plan binding mismatch")
            require(not any(expanded[archive.offset:]), "Non-padding trailing packet bytes")
        require(seen == set(manifest["archive_member_allowlist"]), "Missing packet members")
        # Parsing may outlive the original reads; keep their named identities bound.
        exact_membership()
        for name, (cap, observed) in bindings.items():
            require(observe(output.path/name, cap) == observed, "Final output leaf identity drift")
        exact_membership()
        require(identity(os.fstat(fd)) == directory_state, "Final output directory membership drift")
    return receipt


def build(plan_pin, destination, policy=Policy(), limits=Limits()):
    # Reject foreign/existing destinations even before expensive reads; no resume.
    output = OwnedOutput(destination, policy)
    require(plan_pin["path"].endswith(".json"), "Explicit JSON plan required")
    require(not output.path.exists() and not output.path.is_symlink(), "Delivery destination already exists")
    raw = pin_read(plan_pin, limits.metadata)
    plan = parsed(raw)
    payloads, observations, manifest = materialize(plan, raw, policy, limits)
    archive = pack(payloads, limits)
    verify_packet(archive, payloads, limits)
    require(shutil.disk_usage(policy.results).free >= limits.reserve+len(archive)+2*len(payloads["MANIFEST.json"]),
            "Delivery disk reserve")
    try:
        with output:
            archive_id = output.write("delivery.tar.gz", archive)
            manifest_id = output.write("MANIFEST.json", payloads["MANIFEST.json"])
            # Read back the exact just-closed identities, not freshly adopted inodes.
            bound, _ = read(output.path/"delivery.tar.gz", limits.package, sha(archive), len(archive), archive_id)
            verify_packet(bound, payloads, limits)
            read(output.path/"MANIFEST.json", limits.metadata, sha(payloads["MANIFEST.json"]),
                 len(payloads["MANIFEST.json"]), manifest_id)
            rebind(observations, policy, plan["specification"])
        # Output context/close must succeed before the final local verification receipt.
        receipt = encoded(dict(complete=True, version=VERSION, plan_sha256=sha(raw),
            local_archive_verified=True, archive=dict(sha256=sha(archive), bytes=len(archive)),
            manifest=dict(sha256=sha(payloads["MANIFEST.json"]), bytes=len(payloads["MANIFEST.json"])),
            member_count=len(payloads), scope=SCOPE, publication_actions_performed=False))
        with directory(output.path):
            require(directory_id(output.path.lstat()) == output.observed, "Output changed before final receipt")
            read(output.path/"delivery.tar.gz", limits.package, sha(archive), len(archive), archive_id)
            read(output.path/"MANIFEST.json", limits.metadata, sha(payloads["MANIFEST.json"]),
                 len(payloads["MANIFEST.json"]), manifest_id)
            rebind(observations, policy, plan["specification"])
            # Reattach only to this owned directory, never mkdir/reuse arbitrary output.
            with directory(output.path) as fd:
                output.fd = fd
                with directory(output.path.parent) as parent:
                    output.parent = parent
                    receipt_id = output.write("LOCAL_VERIFY.json", receipt)
                    read(output.path/"LOCAL_VERIFY.json", limits.metadata, sha(receipt), len(receipt), receipt_id)
        receipt_pin = dict(path=str(output.path/"LOCAL_VERIFY.json"), sha256=sha(receipt), bytes=len(receipt))
        result = verify_delivery(output.path, receipt_pin, policy, limits)
        # This pin is returned externally, not embedded as an impossible self-hash.
        return dict(result, local_receipt_pin=receipt_pin)
    except BaseException as error:
        if output.observed is not None:
            try:
                with directory(output.path) as fd, directory(output.path.parent) as parent:
                    output.fd, output.parent = fd, parent
                    output.check()
                    output.write("FAILED.json", encoded(dict(complete=False, error_type=type(error).__name__,
                        error=str(error)[:500], no_resume=True)))
            except BaseException:
                pass
        raise


def create_plan(spec_pin, destination, policy=Policy(), limits=Limits()):
    require(spec_pin["path"].endswith(".json"), "Explicit JSON specification required")
    spec = parsed(pin_read(spec_pin, limits.metadata))
    plan = make_plan(spec, policy, limits)
    raw = encoded(plan)
    require(len(raw) <= limits.metadata, "Pin plan cap")
    with OwnedOutput(destination, policy) as output:
        observed = output.write("PIN_PLAN.json", raw)
        read(output.path/"PIN_PLAN.json", limits.metadata, sha(raw), len(raw), observed)
    return dict(path=str(output.path/"PIN_PLAN.json"), sha256=sha(raw), bytes=len(raw), packaging_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create-plan", "validate", "build", "verify"))
    parser.add_argument("--spec")
    parser.add_argument("--plan")
    parser.add_argument("--receipt")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args()
    source = args.spec if args.action == "create-plan" else args.receipt if args.action == "verify" else args.plan
    require(source is not None and str(source).endswith(".json"), "Explicit JSON spec/plan required")
    observed = observe(source, Limits().metadata)
    pin = dict(path=str(absolute(source)), sha256=args.sha256, bytes=observed[2])
    if args.action == "verify":
        require(args.destination is not None, "Local packet root required")
        result = verify_delivery(args.destination, pin)
    elif args.action == "validate":
        require(args.destination is None, "Validation writes no output")
        result = validate(pin)
    else:
        require(args.destination is not None, "Exclusive destination required")
        result = (create_plan if args.action == "create-plan" else build)(pin, args.destination)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
