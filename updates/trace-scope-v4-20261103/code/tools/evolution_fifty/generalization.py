"""CPU-only prospective inputs for TRACE backbone and corruption evaluation.

No model, checkpoint loading, adaptation, predictions or GT contents are read.
Preparation is not an executable or scientifically completed experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

from .bootstrap import POPULATIONS, TUNED, read_json, require
from .intake import sha256, write_new

CORRUPTION_GROUPS = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
CORRUPTIONS = tuple(c for group in CORRUPTION_GROUPS.values() for c in group)
CURVE_TYPES = ("gaussian_noise", "motion_blur", "fog", "contrast", "jpeg_compression")
ARMS = ("no_adapt", "image_confidence", "image_C")
BACKBONES = {"ViT-B/16": (12, 16, 9), "ViT-L/14": (24, 14, 18)}
CACHE_VERSION = "mlmp-imagecorruptions-resized-bgr-v1"
CACHE = Path("/home/wwww/hzl/data/corruption_cache_s5_v1")
DATASETS = ("coco_object", "pc60", "voc20")


def label_inodes(rows):
    return {(s.st_dev, s.st_ino) for row in rows for s in (Path(row["label"]).stat(),)}


def read_rgb_bytes(path, forbidden):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(), "Noncanonical RGB path")
    before = path.stat()
    require((before.st_dev, before.st_ino) not in forbidden, "RGB preparation aliases annotation")
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with path.open("rb") as stream:
        require(identity(os.fstat(stream.fileno())) == identity(before), "RGB changed before read")
        content = stream.read()
        require(identity(os.fstat(stream.fileno())) == identity(before), "RGB changed during read")
    require(identity(path.stat()) == identity(before), "RGB path changed during read")
    return content


def visual_layers(backbone):
    require(backbone in BACKBONES, "Unsupported prospective backbone")
    depth, _, count = BACKBONES[backbone]
    layers = list(range(-1, -count-1, -1))
    require(all(-depth <= index < 0 for index in layers), "Illegal visual output layer")
    return layers


def cell_matrix():
    rows = []
    def add(stage, backbone, dataset, corruption="original", severity=0, model_type="naclip"):
        for arm in ARMS:
            prefix = "sclip." if model_type == "sclip" else ""
            rows.append(dict(id=f"{prefix}{backbone.replace('/', '-')}.{dataset}.{corruption}.s{severity}.{arm}",
                stage=stage, backbone=backbone, dataset=dataset, corruption=corruption,
                severity=severity, arm=arm, images=POPULATIONS[dataset], model_type=model_type,
                vision_outputs=visual_layers(backbone)))
    for dataset in DATASETS:
        add("primary", "ViT-B/16", dataset)
    add("primary", "ViT-B/16", "voc20", model_type="sclip")
    for corruption in CORRUPTIONS:
        add("primary", "ViT-L/14", "voc20", corruption, 5)
    # Severity 5 is shared with primary, not rerun for the curve.
    for corruption in CURVE_TYPES:
        for severity in (1, 3):
            add("second_stage", "ViT-L/14", "voc20", corruption, severity)
    for corruption in CORRUPTIONS:
        add("resource_conditional", "ViT-B/16", "voc20", corruption, 5)
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate prospective condition")
    return rows


def protocol():
    cells = cell_matrix()
    return dict(experiments=["G01_second_backbone", "G02_corruption", "G03_severity_curve", "G04_second_OVSS"],
        authorization="2026-11-02 actual human: add second backbone/corruption and named SCLIP second OVSS study; no external baseline reproduction",
        priority="M01/M02 and G02 first; G01, M03/M04 next; curves before K/weight-strength/direction grids",
        algorithm="ImageBalancedTRACE:C; production algorithm not modified",
        arms=list(ARMS), seed=0, search_trials=0,
        settings="Transfer each dataset's already frozen L/14 clean configuration to B/16; same settings for Confidence/C",
        corruption_settings="One VOC20 clean configuration across all corruption types/severities; no corruption/test selection",
        historical_exposure="Transferred clean settings have historical development exposure; no independent calibration claim",
        visual_layer_policy="Prospective last three quarters of visual blocks: B/16 last9/12; L/14 last18/24. Not claimed to be MLMP's B/16 layer policy",
        no_adapt_readout="Zero updates, same frozen TRACE prompt/multilevel readout as paired adapted arms; not necessarily published plain NACLIP readout",
        second_OVSS="SCLIP B/16 VOC20clean x3arms; same G01 model/data/config/prompt/layer policy but own candidates, legal SCLIP visual-LN membership; counts reported. Actual official structural parity and real checkpoint gates required",
        single_C="Candidate preservation/context operator/original-position readout/detach/image-region loss unchanged",
        candidate_source="Every condition computes candidates/confidence/C afresh on its own observed input and fixed initial model; no clean-derived masks/scores",
        corruptions=list(CORRUPTIONS), corruption_groups=CORRUPTION_GROUPS,
        main_severity=5, curve_types=list(CURVE_TYPES), curve_severities=[1, 3, 5],
        representative_risk_extension=dict(types=["gaussian_noise", "fog"], severity=5,
            status="later diagnostic protocol required; not 15 repeats of full M01"),
        preprocessing="Official MLMP path: OpenCV BGR decode, corruption seeded by official sample_idx BEFORE resize, OpenCV bilinear224x224, BGR-to-RGB then CLIP normalization",
        labels="Original labels unchanged by corruptions, VOC20 background-to-ignore then OpenCV nearest224x224 ONLY in offline score; official elastic benchmark does not warp GT",
        cache="Historical cache rejected after30representative pinned-generator mismatches; require fresh versioned RGB inputs with reference provenance and sealed hashes",
        old_cache_usable_for_GPU=False, fresh_RGB_seal_required=True,
        cache_parity="Historical hashes freeze observed bytes, not generator equivalence; never reuse the historical cache receipt as standard input admission",
        source_paper="https://arxiv.org/html/2505.21844v2#A4",
        source_backbone="https://arxiv.org/html/2505.21844v2#A7.SS1",
        source_code=["https://github.com/dosowiechi/MLMP/blob/main/utils/mm_transforms.py",
                     "https://github.com/dosowiechi/MLMP/blob/main/utils/segmentation_datasets.py"],
        literature="Published values marked R with exact table/backbone/protocol. New values L; no same-environment/equal-tuning claim",
        metrics=["pooled mIoU per corruption/arm", "macro mean of15 corruption mIoUs",
                 "C-minus-Confidence per corruption and macro mean", "strict positive/tie/negative counts",
                 "macro mean within predefined corruption groups", "actual optimizer/view calls/model/end-to-end time/peak memory"],
        uncertainty="Paired source/episode resampling, shared draws across conditions; fixed settings, no seed/order generality",
        gate="Fresh runtime/checkpoint/input/config freeze and independent exact-artifact audit; real RGB finite loss/gradient/parameter <=1e-5 and0changedlabels, legal14x14 B/16 geometry",
        barriers="All label-free paired predictions in each frozen stage sealed before that stage's GT scoring; no GT in candidate/diagnostic/adaptation",
        resources=dict(single_shared_gpu_lock=True, memory_max_bytes=12*1024**3,
            swap_max_bytes=1024**3, restart="no", OOMPolicy="stop", automatic_retry=False),
        stage_counts={stage: dict(cells=sum(r["stage"] == stage for r in cells),
            image_method_evaluations=sum(r["images"] for r in cells if r["stage"] == stage))
            for stage in ("primary", "second_stage", "resource_conditional")},
        not_authorized=["external TENT/TPT/WATT/CLIPArTT/MLMP reproduction", "old225trial search", "any additional OVSS pipeline beyond named SCLIP", "new natural-shift datasets"],
        executable_queue=False, new_gpu_started=False, scientific_completion=False)


def cache_key(source_path, sample_idx, corruption, severity, target_size, status):
    spec = dict(source_path=str(source_path), source_size=status.st_size,
        source_mtime_ns=status.st_mtime_ns, sample_idx=sample_idx, corruption=corruption,
        severity=severity, target_size=list(target_size), version=CACHE_VERSION)
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_cache_row(raw, source, index, root, *, forbidden_inodes=None):
    require(raw["dataset"] == "voc20" and raw["corruption"] in CORRUPTIONS
            and raw["severity"] == "5" and raw["resize"] == "224x224"
            and raw["status"] in ("hit", "write") and not raw["error"], "Invalid cached corruption record")
    require(raw["source_path"] == source["image"] and int(raw["sample_idx"]) == index,
            "Cache source/index/order mismatch")
    path = Path(raw["cache_path"])
    digest = cache_key(source["image"], index, raw["corruption"], 5, (224, 224), Path(source["image"]).stat())
    expected = root/"severity_5"/raw["corruption"]/digest[:2]/f"{digest}.png"
    require(path == expected and path.is_absolute() and not path.is_symlink()
            and path.resolve(strict=True) == path, "Cache key or canonical path drift")
    require(0 < path.stat().st_size <= 8*1024**2 and path.stat().st_size == int(raw["bytes"]),
            "Corruption cache size drift")
    forbidden = label_inodes([source]) | (forbidden_inodes or set())
    content = read_rgb_bytes(path, forbidden)
    with Image.open(io.BytesIO(content)) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (224, 224),
                "Corruption cache format/geometry mismatch")
        image.load()
    row = copy.deepcopy(source)
    row.update(image=str(path), image_sha256=hashlib.sha256(content).hexdigest(),
        clean_image=source["image"], clean_image_sha256=source["image_sha256"],
        corruption=raw["corruption"], severity=5, original_sample_idx=index,
        preprocessing="already_corrupted_and_cv2_resized_no_second_resize", cache_key=digest)
    return row


def corruption_inputs(sources, root, *, forbidden_inodes=None):
    require(len({r["image"] for r in sources}) == len(sources), "Duplicate clean source")
    by_image = {r["image"]: i for i, r in enumerate(sources)}
    result = {kind: {} for kind in CORRUPTIONS}
    forbidden = label_inodes(sources) | (forbidden_inodes or set())
    with (root/"manifest.csv").open(newline="") as stream:
        for raw in csv.DictReader(stream):
            if raw["dataset"] != "voc20":
                continue
            require(raw["source_path"] in by_image and raw["corruption"] in result,
                    "Unexpected VOC corruption input")
            index = by_image[raw["source_path"]]
            kind = raw["corruption"]
            require(index not in result[kind], "Duplicate VOC corruption input")
            result[kind][index] = checked_cache_row(raw, sources[index], index, root, forbidden_inodes=forbidden)
    require(all(set(rows) == set(range(len(sources))) for rows in result.values()),
            "Incomplete VOC corruption population")
    return {kind: [rows[i] for i in range(len(sources))] for kind, rows in result.items()}


def checked_checkpoint(backbone, *, forbidden_inodes=None):
    from trace_ovss.models._vendor_clip.clip import _MODELS
    require(backbone in BACKBONES, "Unsupported checkpoint")
    url = _MODELS[backbone]
    path = Path.home()/".cache/clip"/url.rsplit("/", 1)[1]
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, "Missing canonical checkpoint")
    before = path.stat()
    require((before.st_dev, before.st_ino) not in (forbidden_inodes or set()), "Checkpoint aliases annotation")
    digest = sha256(path)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(digest == url.split("/")[-2] and identity(before) == identity(after), "Official checkpoint identity drift")
    return dict(path=str(path), sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns,
        official_url=url, backbone=backbone, vision_outputs=visual_layers(backbone))


def prepare(output, *, historical_cache_audit_only=False):
    require(historical_cache_audit_only is True,
            "Historical cache failed generator parity; only explicit CPU historical-cache audit is allowed")
    output = Path(output).absolute()
    require(not output.exists() and output == output.resolve(), "Fresh generalization input directory required")
    output.mkdir(parents=True)
    write_new(output/"PROTOCOL.json", protocol())
    write_new(output/"CELLS.json", cell_matrix())
    try:
        clean, parents, rgb_hashes, manifests = {}, {}, {}, {}
        for dataset in DATASETS:
            root = TUNED/"full"/dataset
            path = root/"prediction_protocol.json"
            seal = read_json(root/"PREDICTIONS_SEALED.json")
            require(seal["complete"] and not seal["gt_contents_opened"]
                    and sha256(path) == seal["artifact_sha256"][path.name], "Original input protocol not sealed")
            manifest = read_json(path)["manifest"]
            manifests[dataset] = manifest
            parents[dataset] = dict(protocol_path=str(path), protocol_sha256=sha256(path),
                prediction_seal_sha256=sha256(root/"PREDICTIONS_SEALED.json"))
        forbidden = label_inodes([row for manifest in manifests.values() for row in manifest["panel"]["inputs"]])
        for dataset, manifest in manifests.items():
            rows = manifest["panel"]["inputs"]
            require(len(rows) == POPULATIONS[dataset]
                    and [r["official_id"] for r in rows] == manifest["panel"]["ordered_ids"], "Clean population/order drift")
            for row in rows:
                if row["image"] not in rgb_hashes:
                    rgb_hashes[row["image"]] = hashlib.sha256(read_rgb_bytes(row["image"], forbidden)).hexdigest()
                require(rgb_hashes[row["image"]] == row["image_sha256"], "Clean RGB content drift")
                status = Path(row["label"]).stat()
                require([status.st_size, status.st_mtime_ns] == row["label_metadata"], "Label metadata drift")
            clean[dataset] = dict(config=manifest["config"], panel=manifest["panel"])
        write_new(output/"CLEAN_INPUTS.json", clean)
        write_new(output/"VOC20_C_S5_INPUTS.json", corruption_inputs(clean["voc20"]["panel"]["inputs"], CACHE, forbidden_inodes=forbidden))
        write_new(output/"CHECKPOINTS.json", {b: checked_checkpoint(b, forbidden_inodes=forbidden) for b in BACKBONES})
        write_new(output/"INPUTS_PREPARED.json", dict(status="CPU_inputs_only_not_executable_not_results",
            experiment_protocol_sha256=sha256(output/"PROTOCOL.json"), parents=parents,
            existing_cache_manifest_sha256=sha256(CACHE/"manifest.csv"),
            existing_cache_completion_metadata_sha256=sha256(CACHE/"complete.json"),
            source_code_sha256={str(Path(__file__).resolve()): sha256(__file__)},
            artifact_sha256={name: sha256(output/name) for name in
                ("PROTOCOL.json", "CELLS.json", "CLEAN_INPUTS.json", "VOC20_C_S5_INPUTS.json", "CHECKPOINTS.json")},
            corrupted_RGB_files_validated=len(CORRUPTIONS)*POPULATIONS["voc20"],
            GT_contents_read=False, model_loaded=False, new_GPU_started=False,
            independent_artifact_admitted=False, cache_generator_parity_verified=False,
            remaining=["official generator parity", "new frozen execution/scoring worker", "independent code/artifact audit", "actual model gates", "predictions/seals/scores/terminal verification"]))
    except BaseException as error:
        write_new(output/"FAILED.json", dict(error=f"{type(error).__name__}: {error}", automatic_retry=False))
        raise
    return read_json(output/"INPUTS_PREPARED.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-cache-audit-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.output, historical_cache_audit_only=args.historical_cache_audit_only)
    print(json.dumps({k: result[k] for k in ("status", "corrupted_RGB_files_validated", "GT_contents_read", "new_GPU_started")}))
