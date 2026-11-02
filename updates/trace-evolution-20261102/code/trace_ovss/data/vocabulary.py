"""Packaged open-vocabulary class names and prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources


VOCABULARY_FILES = {
    "PascalVOC20Dataset": "cls_voc20.txt",
    "PascalVOC21Dataset": "cls_voc21.txt",
    "PascalContext59Dataset": "cls_context59.txt",
    "PascalContext60Dataset": "cls_context60.txt",
    "COCOObjectDataset": "cls_coco_object.txt",
    "COCOStuffDataset": "cls_coco_stuff.txt",
    "CityscapesDataset": "cls_city_scapes.txt",
    "COCOCDDataset": "coco_cd.txt",
    "WaterbirdsSegDataset": "waterbirds_seg.txt",
}

BACKGROUND_FIRST = frozenset(
    {
        "PascalVOC21Dataset",
        "COCOObjectDataset",
        "PascalContext60Dataset",
        "COCOCDDataset",
        "WaterbirdsSegDataset",
    }
)


@dataclass(frozen=True)
class BenchmarkVocabulary:
    """Canonical labels plus their text-query extensions."""

    classes: tuple[str, ...]
    extensions: tuple[str, ...]
    extension_to_class: tuple[int, ...]


def load_vocabulary(dataset_name: str) -> BenchmarkVocabulary:
    try:
        filename = VOCABULARY_FILES[dataset_name]
    except KeyError as exc:
        raise KeyError(f"No packaged vocabulary for {dataset_name}") from exc

    path = resources.files("trace_ovss.resources.class_names").joinpath(filename)
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = [tuple(part.strip() for part in line.split(",")) for line in lines]
    classes = [group[0] for group in groups]
    if dataset_name in BACKGROUND_FIRST:
        classes[0] = "background"

    extensions = []
    extension_to_class = []
    query_owners = {}
    for class_idx, group in enumerate(groups):
        for query in group:
            key = query.casefold()
            if key in query_owners and query_owners[key] != class_idx:
                previous = classes[query_owners[key]]
                raise ValueError(
                    f"Text query {query!r} maps to both {previous!r} and {classes[class_idx]!r}"
                )
            query_owners[key] = class_idx
        extensions.extend(group)
        extension_to_class.extend([class_idx] * len(group))
    return BenchmarkVocabulary(
        classes=tuple(classes),
        extensions=tuple(extensions),
        extension_to_class=tuple(extension_to_class),
    )


def load_prompt_templates() -> tuple[str, ...]:
    path = resources.files("trace_ovss.resources").joinpath("prompts.txt")
    prompts = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not prompts:
        raise RuntimeError("TRACE prompt template resource is empty")
    return prompts
