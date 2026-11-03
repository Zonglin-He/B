"""Read pinned historical TRACE score/view modules without restoring old projects.

Only score/role/view components are loaded into a private namespace. The old
engine, optimizer, input pipeline and OOM-retry iterator are never selected.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tarfile
from types import ModuleType

from .bootstrap import require
from .intake import sha256

ARCHIVE = Path("/home/wwww/hzl/results/reviewer_response_20260929/source_snapshot_validated.tar.gz")
ARCHIVE_SHA = "d88873fa15ada2751e07d7129cb30e96c94f832fbf4c053d63b10384f6b1dc46"
MEMBERS = ("trace_ovss/method/trace/constants.py", "trace_ovss/data/roles.py",
           "trace_ovss/method/trace/reliability.py", "trace_ovss/method/trace/indexing.py",
           "trace_ovss/method/trace/interventions.py")


def load_components(archive=ARCHIVE):
    archive = Path(archive).resolve(strict=True)
    require(sha256(archive) == ARCHIVE_SHA, "Historical TRACE archive identity mismatch")
    root = "_trace_pinned_"+ARCHIVE_SHA[:16]
    loaded, hashes = {}, {}
    with tarfile.open(archive, "r:gz") as source:
        names = [m.name for m in source.getmembers()]
        for name in MEMBERS:
            require(names.count(name) == 1, "Historical source missing/duplicated")
            member = source.getmember(name)
            require(member.isfile() and member.size <= 256*1024, "Invalid historical source entry")
            handle = source.extractfile(member)
            require(handle is not None, "Unreadable historical source")
            content = handle.read()
            require(len(content) == member.size, "Truncated historical source")
            hashes[name] = hashlib.sha256(content).hexdigest()
            module_name = root+"."+name.removesuffix(".py").replace("/", ".")
            parts = module_name.split(".")
            for length in range(1, len(parts)):
                package_name = ".".join(parts[:length])
                if package_name not in sys.modules:
                    package = ModuleType(package_name)
                    package.__path__ = []
                    sys.modules[package_name] = package
            # Every member is compiled from exact pinned bytes, with relative
            # imports resolving only within this namespace, not trace_ovss.
            module = ModuleType(module_name)
            module.__file__ = str(archive)+":"+name
            module.__package__ = module_name.rpartition(".")[0]
            sys.modules[module_name] = module
            exec(compile(content, module.__file__, "exec"), module.__dict__)
            loaded[name.rsplit("/", 1)[-1].removesuffix(".py")] = module
    require(sha256(archive) == ARCHIVE_SHA, "Historical source changed during loading")
    return dict(modules=loaded, member_sha256=hashes, archive_sha256=ARCHIVE_SHA,
                old_engine_loaded=False, old_optimizer_loaded=False, old_OOM_iterator_selected=False)


def query_roles(components, dataset_name, benchmark_classes, query_to_class):
    roles = components["modules"]["roles"]
    base = roles.class_roles_for(dataset_name, benchmark_classes)
    return base if query_to_class is None else roles.extend_class_roles(base, query_to_class)
