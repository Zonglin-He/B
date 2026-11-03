"""Preserve the supplied cards verbatim and register all fifty as mandatory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


DEPENDENCIES = {
    7: [6], 8: [], 13: [5], 14: [5], 15: [5], 16: [], 20: [],
    31: [6, 7], 32: [7], 39: [5], 40: [1], 43: [1], 44: [1],
    46: [], 47: [], 49: [], 50: [5],
}
FIELDS = (
    "优先级", "要回答的问题", "最小设计", "必须固定/核对", "指标与交付",
    "能支持什么／结果分支", "资源与论文落点", "状态",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def parse_cards(text):
    headings = list(re.finditer(r"(?m)^实验 (\d{2})｜([^\r\n]+)\r?$", text))
    if [int(m[1]) for m in headings] != list(range(1, 51)):
        raise ValueError("Exactly E01 through E50, once each and ordered, are required")
    cards = []
    for index, heading in enumerate(headings):
        stop = headings[index + 1].start() if index < 49 else text.index("建议论证图表组合", heading.end())
        block = text[heading.start():stop].strip()
        fields = {}
        for name in FIELDS:
            match = re.search(r"(?m)^" + re.escape(name) + r"：([^\r\n]+)", block)
            if match is None:
                raise ValueError(f"E{heading[1]} lacks {name}")
            fields[name] = match[1]
        number = int(heading[1])
        cards.append(dict(id=f"E{number:02d}", title=heading[2], required=True,
            original_priority=fields["优先级"], source_fields=fields, source_card=block,
            dependencies=[f"E{i:02d}" for i in DEPENDENCIES.get(number, [])],
            status="not_started", execution_protocol="not_frozen", evidence=[],
            note="User requires every card; priority/conditional wording does not waive execution."))
    return cards


def intake(source, output):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output != output.resolve():
        raise ValueError("A new canonical intake directory is required")
    before = sha256(source)
    cards = parse_cards(source.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    shutil.copyfile(source, output / "user_cards.txt")
    if sha256(source) != before or sha256(output / "user_cards.txt") != before:
        raise ValueError("User card bytes changed during intake")
    registry = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        authorization="Human requested all E01-E50; none optional or dropped by result sign.",
        source_path=str(source), source_sha256=before, experiments=cards,
        total_required=50, completed=0, gpu_experiments_started=0,
        scientific_completion=False, cards_are_not_results=True,
        retained_earlier_baselines="The previously authorized native 18-cell queue remains required separately.")
    write_new(output / "registry.json", registry)
    write_new(output / "INTAKE_SEALED.json", dict(complete=True, scope="intake_only_not_execution",
        artifact_sha256={name: sha256(output / name) for name in ("user_cards.txt", "registry.json")}))
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = intake(args.source, args.output)
    print(json.dumps(dict(output=str(args.output), required=result["total_required"], completed=0)))


if __name__ == "__main__":
    main()
