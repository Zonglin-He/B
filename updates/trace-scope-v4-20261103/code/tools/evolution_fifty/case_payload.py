"""Pure M12 descriptive-case selection and PNG rendering, never admission.

Only supplied, separately admitted bytes are decoded. There is no model,
metric-based selection, filesystem input, invented attribution or GT feedback.
"""

from __future__ import annotations

import hashlib
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from trace_ovss.method.probe_geometry import translated_candidate
from tools.evolution_followup.standalone import candidate_identity
from .bootstrap import require
from .corruption_curve import same_provenance
from .risk_score_inputs import decode_annotation
from .risk_score_payload import hard_array

DATASETS = ("coco_object", "pc60")
MODES = ("native", "translated", "global", "confidence")
SHAPE = (224, 224)


def protocol():
    return dict(experiment="M12_descriptive_M02_cases", datasets=list(DATASETS),
        source_indices=[0, 1], cases=4, selection="First two original manifest entries per dataset, fixed before case payload reads; no GT, score, gain, candidate count or visual attractiveness selection",
        candidate="First original confidenceTopK receipt entry belonging to that source; empty source remains empty, no replacement",
        panels=["RGB", "GT", "Confidence prediction", "Native C prediction", "Original candidate overlay", "Translated candidate overlay", "Translated prediction", "Global prediction"],
        geometry="M02 native PIL BILINEAR RGB and NEAREST indexed GT at224; no G OpenCV geometry",
        mask_semantics="Overlays display authenticated original/translated preservation support only, not rendered model perturbations, attention or causal attribution",
        exposure="Registered after M02 aggregate results; source-index rule not chosen by source outcomes; descriptive nonrepresentative examples, not population evidence",
        no_inference=True, no_new_scores=True, no_method_change=True, no_RGB_GT_upload=True,
        permission="New exact independent CPU_GT and M12_qualitative_GT and RGB_reading permission after complete M02 prediction seals and exact successful exit",
        failure="Failclosed, retain empty/negative/uninformative cases and partial evidence; no alternate case selection or automatic retry")


def select_cases(source):
    require(set(source["rows"]) == set(DATASETS) and type(source["specs"]) is list
            and [s["dataset"] for s in source["specs"]] == list(DATASETS), "M12 source dataset identity differs")
    cases = []
    for spec in source["specs"]:
        dataset, rows = spec["dataset"], source["rows"][spec["dataset"]]
        batch = spec["config"]["batch_size"]
        require(type(batch) is int and 0 < batch <= 512 and len(rows) == spec["images"] >= 2
                and spec["modes"] == list(MODES) and spec["config"]["resize"] == [224, 224],
                "M12 fixed original source geometry differs")
        for index in (0, 1):
            name = rows[index]["official_id"]
            require(type(name) is str and 0 < len(name) <= 256 and name.isascii()
                    and all(ord(c) >= 32 for c in name), "M12 source ID format differs")
            cases.append(dict(dataset=dataset, index=index, official_id=name,
                              episode=index//batch, batch_idx=index%batch))
    return cases


def palette(classes):
    require(type(classes) is int and 1 <= classes <= 81, "M12 class palette bound differs")
    colors = np.zeros((256, 3), dtype=np.uint8)
    for index in range(classes):
        digest = hashlib.sha256(f"M12-class-{index}".encode()).digest()
        colors[index] = [48+v%192 for v in digest[:3]]
    colors[255] = [128, 128, 128]
    return colors


def rgb_image(content):
    require(type(content) is bytes and 0 < len(content) <= 32*1024**2, "M12 RGB compressed bound differs")
    with Image.open(io.BytesIO(content)) as image:
        require(image.format in ("JPEG", "PNG", "TIFF", "BMP") and 0 < image.width*image.height <= 16*1024**2,
                "M12 RGB format/decoded bounds differ")
        return np.asarray(image.convert("RGB").resize(SHAPE[::-1], Image.Resampling.BILINEAR)).copy()


def annotation(content, classes, reduce_zero):
    require(type(content) is bytes and 0 < len(content) <= 16*1024**2, "M12 GT byte bound differs")
    with Image.open(io.BytesIO(content)) as image:
        require(image.format in ("PNG", "TIFF") and 0 < image.width*image.height <= 16*1024**2
                and image.mode in ("P", "L", "I", "I;16", "RGB"), "M12 indexed GT format differs")
        values = np.asarray(image)
        if values.ndim == 3:
            require(bool((values == values[:, :, :1]).all()), "M12 colored RGB is not an indexed GT")
    return decode_annotation(content, SHAPE, classes, reduce_zero)


def candidate_mask(candidate):
    region = candidate["region"]
    require(type(region) is dict and region["shape"] == [16, 16]
            and region["encoding"] == "packbits-little-hex" and type(region["data"]) is str
            and len(region["data"]) == 64, "M12 authentic candidate mask encoding differs")
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(region["data"]), dtype=np.uint8), bitorder="little").reshape(16, 16).astype(bool)
    require(type(region["true_pixels"]) is int and type(candidate["area"]) is int
            and int(bits.sum()) == region["true_pixels"] == candidate["area"] > 0,
            "M12 candidate mask area differs")
    return np.repeat(np.repeat(bits, 14, axis=0), 14, axis=1)


def mask_overlay(rgb, mask, color):
    out = rgb.copy()
    out[mask] = ((rgb[mask].astype(np.uint16)+np.asarray(color, dtype=np.uint16))//2).astype(np.uint8)
    return out


def checked_candidates(row, spec):
    values, decoded = row["candidates"], []
    images, queries, maximum = len(row["ordered_ids"]), spec["vocabulary"]["queries"], spec["config"]["trace_max_regions"]
    require(type(values) is list and type(maximum) is int and 0 < maximum <= 256
            and 0 < images <= 512 and len(values) <= images*maximum, "M12 original candidate resource bound differs")
    for item in values:
        require(all(type(item[k]) is int for k in ("batch_idx", "image_idx", "class_idx", "component_idx"))
                and 0 <= item["batch_idx"] == item["image_idx"] < images
                and 0 <= item["class_idx"] < len(queries) and 0 < item["component_idx"] <= 256
                and all(type(item[k]) in (int, float) and math.isfinite(item[k]) and 0 <= item[k] <= 1
                        for k in ("confidence", "reliability")), "M12 original candidate source/query/score differs")
        mask = candidate_mask(item)[::14, ::14].copy()
        decoded.append(dict(item, region=mask))
    require(candidate_identity(decoded) == row["candidate_sha256"], "M12 declared original candidate content differs")
    return values


def wrapped_lines(draw, text, font, maximum):
    """Fit unbroken IDs as well as words, using measured pixels, never clipping."""
    lines, current = [], ""
    for character in text:
        proposal = current+character
        if current and draw.textbbox((0, 0), proposal, font=font)[2] > maximum:
            lines.append(current)
            current = character
        else:
            current = proposal
    if current:
        lines.append(current)
    require(all(draw.textbbox((0, 0), line, font=font)[2] <= maximum for line in lines),
            "M12 display font exceeds bounded line width")
    return lines


def render_case(rgb_bytes, gt_bytes, prediction_bytes_by_mode, episode_receipts_by_mode, spec, case):
    """Return one PNG and full JSON legend/provenance, never a seal or score."""
    require(type(case) is dict and set(case) == {"dataset", "index", "official_id", "episode", "batch_idx"}
            and case["dataset"] == spec["dataset"] in DATASETS
            and type(case["index"]) is int and case["index"] in (0, 1)
            and type(case["episode"]) is int and type(case["batch_idx"]) is int
            and case["episode"] == case["index"]//spec["config"]["batch_size"]
            and case["batch_idx"] == case["index"]%spec["config"]["batch_size"]
            and set(prediction_bytes_by_mode) == set(episode_receipts_by_mode) == set(MODES)
            and spec["modes"] == list(MODES) and spec["config"]["resize"] == [224, 224]
            and spec["preprocessing"] == "native_PIL_bilinear", "M12 supplied case/grid/geometry differs")
    classes = len(spec["vocabulary"]["classes"])
    colors = palette(classes)
    rgb = rgb_image(rgb_bytes)
    gt = annotation(gt_bytes, classes, spec["vocabulary"]["reduce_zero_label"])
    predictions = {mode:hard_array(prediction_bytes_by_mode[mode], classes, SHAPE) for mode in MODES}
    records, original_identity, checked = episode_receipts_by_mode, None, {}
    for mode in MODES:
        row = records[mode]
        require(row["GT_opened"] is False and row["scientific_identity"] == "M02_spatial_"+mode
                and type(row["episode"]) is int and row["episode"] == case["episode"]
                and type(row["ordered_ids"]) is list and 0 <= case["batch_idx"] < len(row["ordered_ids"])
                and row["ordered_ids"][case["batch_idx"]] == case["official_id"]
                and type(row["candidate_sha256"]) is str and len(row["candidate_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["candidate_sha256"]),
                "M12 supplied original episode/mode identity differs")
        require(original_identity is None or row["candidate_sha256"] == original_identity,
                "M12 modes use different original candidates")
        require(same_provenance(row["ordered_ids"], records["native"]["ordered_ids"]),
                "M12 modes have different original source order")
        checked[mode] = checked_candidates(row, spec)
        original_identity = row["candidate_sha256"]
    candidates = checked["native"]
    require(type(candidates) is list and all(type(c["batch_idx"]) is int and 0 <= c["batch_idx"] < len(records["native"]["ordered_ids"])
            for c in candidates), "M12 original candidate source membership differs")
    matches = [(i, c) for i, c in enumerate(candidates) if c["batch_idx"] == case["batch_idx"]]
    mask = shifted = np.zeros(SHAPE, dtype=bool)
    selected = None
    if matches:
        index, item = matches[0]
        mask = candidate_mask(item)
        shifted, geometry = translated_candidate(mask)
        require(len(records["translated"]["preservation_geometry"]) == len(candidates),
                "M12 translated geometry membership differs")
        saved = records["translated"]["preservation_geometry"][index]
        require(all(same_provenance(saved[k], v) for k, v in geometry.items())
                and type(saved["candidate_index"]) is int and saved["candidate_index"] == index
                and type(saved["batch_idx"]) is int and saved["batch_idx"] == case["batch_idx"] and saved["mode"] == "translated",
                "M12 mask overlay does not match original translated receipt")
        require(type(item["class_idx"]) is int and 0 <= item["class_idx"] < len(spec["vocabulary"]["queries"])
                and all(type(item[k]) in (float, int) and math.isfinite(item[k]) and 0 <= item[k] <= 1
                        for k in ("confidence", "reliability")), "M12 candidate query/score receipt differs")
        selected = dict(receipt_index=index, query_index=item["class_idx"], query=spec["vocabulary"]["queries"][item["class_idx"]],
                        confidence=item["confidence"], C=item["reliability"], original_mask_SHA256=hashlib.sha256(mask.tobytes()).hexdigest(),
                        translated_mask_SHA256=hashlib.sha256(shifted.tobytes()).hexdigest(), geometry=geometry)
    tiles = [rgb, colors[gt], colors[predictions["confidence"]], colors[predictions["native"]],
             mask_overlay(rgb, mask, [32, 224, 64]), mask_overlay(rgb, shifted, [224, 32, 192]),
             colors[predictions["translated"]], colors[predictions["global"]]]
    labels = protocol()["panels"]
    width, gap = 960, 16
    font = ImageFont.load_default()
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    heading = f"{case['dataset']} source index {case['index']}: {case['official_id']}"
    require(type(case["official_id"]) is str and 0 < len(case["official_id"]) <= 256
            and case["official_id"].isascii(), "M12 display ID differs")
    paragraphs = [heading, "Fixed first-source examples; descriptive, nonrepresentative; no new inference or score."]
    if selected:
        require(type(selected["query"]) is str and len(selected["query"]) <= 1024, "M12 query display resource bound differs")
        paragraphs.extend([f"Candidate query: {selected['query']!r}",
            f"Receipt #{selected['receipt_index']}; confidence={selected['confidence']:.6g}; C={selected['C']:.6g}. Green: original; magenta: translated."])
    else:
        paragraphs.append("No original candidate for this source; both mask panels show unchanged RGB, with no substitute case.")
    header = [line for text in paragraphs for line in wrapped_lines(measure, text.encode("ascii", "backslashreplace").decode(), font, width-2*gap)]
    footer = wrapped_lines(measure, "Mask overlays show preservation support, not attention or rendered perturbations. Empty candidate sources remain empty.", font, width-2*gap)
    top = 24+16*len(header)
    height = top+512+16*len(footer)+16
    require(height <= 4096, "M12 case canvas height exceeds resource bound")
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(header):
        draw.text((gap, 8+i*16), line, font=font, fill="black")
    for i, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        x, y = gap+(i%4)*240, top+(i//4)*256
        draw.text((x, y), label, font=font, fill="black")
        canvas.paste(Image.fromarray(tile), (x, y+18))
    for i, line in enumerate(footer):
        draw.text((gap, top+512+i*16), line, font=font, fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    png = buffer.getvalue()
    require(len(png) <= 8*1024**2, "M12 PNG output bound exceeded")
    metadata = dict(case=case, panel_order=labels, PNG_size=[width, height], tile_top=top, candidate=selected,
        candidate_status="first original source candidate" if selected else "No original candidate; overlays unchanged RGB, no substitute case",
        original_candidate_identity=original_identity, selection=protocol()["selection"],
        class_legend=[dict(index=i, name=name, RGB=colors[i].tolist()) for i, name in enumerate(spec["vocabulary"]["classes"])],
        ignore255_RGB=colors[255].tolist(), geometry=protocol()["geometry"],
        mask_semantics=protocol()["mask_semantics"], exposure=protocol()["exposure"],
        new_inference=False, new_scores=False, population_effect_or_causality_claim=False)
    return png, metadata
