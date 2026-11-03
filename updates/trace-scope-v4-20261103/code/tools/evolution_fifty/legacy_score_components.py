"""Load only five hash-pinned historical components from already bounded bytes.

No pathname opens or extraction. The future scorer must establish archive file
identity/role exclusion before its bounded read. This module grants no GT or
artifact permission and never invokes the historical engine/optimizer/iterator.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import sys
import tarfile
from types import ModuleType
import zlib

from .bootstrap import require
from .legacy_rho import ARCHIVE_SHA,MEMBERS
from .legacy_runtime import OBJECTIVE_SHA

_NAMESPACE_COUNTER = itertools.count()
MAX_EXPANDED = 32*1024**2


def bounded_tar(content):
    decoder = zlib.decompressobj(16+zlib.MAX_WBITS)
    raw = decoder.decompress(content,MAX_EXPANDED+1)
    require(len(raw) <= MAX_EXPANDED and decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
            "M03 scoring gzip expanded extent or member framing differs")
    offset,headers = 0,0
    while offset < len(raw):
        block = raw[offset:offset+512]
        require(len(block) == 512, "M03 scoring tar header is truncated")
        if block == bytes(512):
            require(not any(raw[offset:]), "M03 scoring tar contains data after its footer")
            return raw
        # frombuf parses one fixed header, without consuming PAX/GNU payloads.
        member = tarfile.TarInfo.frombuf(block,"utf-8","strict")
        headers += 1
        require(headers <= 1000 and 0 <= member.size <= MAX_EXPANDED,
                "M03 scoring raw tar header extent exceeds bound")
        if member.type in (tarfile.XHDTYPE,tarfile.XGLTYPE,tarfile.GNUTYPE_LONGNAME,tarfile.GNUTYPE_LONGLINK):
            require(member.size <= 256*1024, "M03 scoring extended tar header exceeds preparse bound")
        offset += 512+((member.size+511)//512)*512
        require(offset <= len(raw), "M03 scoring tar payload is truncated")
    raise ValueError("M03 scoring tar footer is missing")


def load_components_bytes(content,spec):
    require(type(content) is bytes and 0 < len(content) <= 1024**2
            and hashlib.sha256(content).hexdigest() == spec["sha256"] == ARCHIVE_SHA,
            "M03 scoring archive bytes differ from pinned source")
    expected = spec["member_sha256"]
    require(type(expected) is dict and set(expected) == set(MEMBERS)
            and all(type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                    for v in expected.values()) and spec["episode_objective_sha256"] == OBJECTIVE_SHA
            and all(spec[k] is False for k in ("old_engine_loaded","old_optimizer_loaded","old_OOM_iterator_selected")),
            "M03 scoring archive component scope differs")
    raw_archive = bounded_tar(content)
    selected,seen,total = {},set(),0
    with tarfile.open(fileobj=io.BytesIO(raw_archive),mode="r:") as source:
        for index,member in enumerate(source):
            require(index < 1000 and member.size >= 0, "M03 scoring archive inventory exceeds bound")
            total += member.size
            require(total <= 32*1024**2, "M03 scoring archive declared extent exceeds bound")
            if member.name not in MEMBERS:
                continue
            require(member.name not in seen and member.isfile() and not member.islnk() and not member.issym()
                    and 0 < member.size <= 256*1024, "M03 scoring component missing/duplicate/nonregular/oversized")
            seen.add(member.name)
            handle = source.extractfile(member)
            require(handle is not None, "Unreadable pinned M03 component")
            with handle:
                raw = handle.read(256*1024+1)
            require(len(raw) == member.size and hashlib.sha256(raw).hexdigest() == expected[member.name],
                    "M03 scoring component bytes differ")
            selected[member.name] = raw
    require(set(selected) == set(MEMBERS), "M03 scoring component membership incomplete")
    root = f"_trace_scoring_pinned_{ARCHIVE_SHA[:16]}_{next(_NAMESPACE_COUNTER)}"
    require(not any(n == root or n.startswith(root+".") for n in sys.modules), "M03 private scoring namespace collision")
    loaded,created = {},[]
    try:
        for name in MEMBERS:
            module_name = root+"."+name.removesuffix(".py").replace("/", ".")
            parts = module_name.split(".")
            for length in range(1,len(parts)):
                package_name = ".".join(parts[:length])
                if package_name not in sys.modules:
                    package = ModuleType(package_name)
                    package.__path__ = []
                    sys.modules[package_name] = package
                    created.append(package_name)
            module = ModuleType(module_name)
            module.__file__ = "pinned-bytes:"+ARCHIVE_SHA+":"+name
            module.__package__ = module_name.rpartition(".")[0]
            sys.modules[module_name] = module
            created.append(module_name)
            exec(compile(selected[name],module.__file__,"exec"),module.__dict__)
            loaded[name.rsplit("/",1)[-1].removesuffix(".py")] = module
    except BaseException:
        for name in reversed(created):
            sys.modules.pop(name,None)
        raise
    return dict(modules=loaded,member_sha256=dict(expected),archive_sha256=ARCHIVE_SHA,
                old_engine_loaded=False,old_optimizer_loaded=False,old_OOM_iterator_selected=False)
