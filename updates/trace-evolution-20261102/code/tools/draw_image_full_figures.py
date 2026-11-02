"""CPU-only figures from the sealed, selected-C full suite; never run a model.

All result paths are read-only. GT is opened only after all six prediction
seals have been checked. Illustrations are explicitly selected extrema, not
an alternative evaluation population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_cairo import RendererCairo, cairo
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_ovss.data.datasets import CLIP_MEAN, CLIP_STD
from trace_ovss.method.trace.interventions import TraceInterventionMixin

SUITE_HASH = "e0f7374042eadb65fbc8146de6dfcf35a71151dc1c95c115b0d4cf9c8444edaa"
DATASETS = ("coco_cd", "coco_object", "pc60", "voc21", "voc20", "coco_stuff")
POPULATIONS = dict(zip(DATASETS, (500, 5000, 5105, 1449, 1449, 5000)))
DISPLAY = dict(zip(DATASETS, ("COCO-CD", "COCO Object", "PC60", "VOC21", "VOC20", "COCO-Stuff")))
ARMS = ("image_uniform", "image_confidence", "image_C")
WIDTH = 6.875
SINGLE_WIDTH = 3.375
LIMIT = 400
EPS = 1e-8
CYAN = "#007f88"
RED = "#ad3644"
SELECTION_RULE = {
    "bound": "indices 0..min(399,N-1) in frozen official order only",
    "teaser": "Non-background candidates, confidence >= 0.9, native area 16..192; PC60 maximum C, COCO Object minimum C; ties lowest image index then receipt candidate index. No GT used.",
    "segmentation": "For each image choose its largest non-background GT class with >=128 valid pixels (tie lowest category ID); compute that single named-class IoU for both saved arms. Maximum TRACE-minus-Confidence in CD, Object, PC60; minimum in VOC21. Ties lowest image index. Require three strictly positive deltas and one strictly negative delta.",
    "warning": "Post-seal offline GT-selected extrema; not representative, not an aggregate performance estimate, not used to select the method or report overall metrics.",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_region(region):
    """Decode the recorded shape, not an assumed ViT grid size."""
    shape = tuple(region["shape"])
    require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape), "Invalid mask shape")
    require(region["encoding"] == "packbits-little-hex", "Unknown mask encoding")
    packed = np.frombuffer(bytes.fromhex(region["data"]), dtype=np.uint8)
    size = int(np.prod(shape))
    require(len(packed) == (size + 7) // 8, "Invalid packed mask length")
    bits = np.unpackbits(packed, bitorder="little")
    require(not bits[size:].any(), "Nonzero mask padding")
    mask = bits[:size].reshape(shape).astype(bool)
    require(int(mask.sum()) == region["true_pixels"] > 0, "Mask count mismatch")
    return mask


def category(panel, query):
    mapping = panel["query_to_class"]
    require(0 <= query < len(panel["queries"]), "Invalid original query")
    return query if mapping is None else int(mapping[query])


def collapse_query_max(probabilities, mapping):
    """Audit helper: category-max probabilities before category argmax."""
    if mapping is None:
        return probabilities
    require(len(mapping) == probabilities.shape[0], "Query mapping length")
    result = np.full((max(mapping) + 1, *probabilities.shape[1:]), -np.inf)
    np.maximum.at(result, np.asarray(mapping), probabilities)
    return result


def prepare_gt(raw, reduce_zero):
    gt = np.asarray(raw).astype(np.int64, copy=True)
    if gt.ndim == 3:
        gt = gt[..., 0]
    if reduce_zero:
        background, valid = gt == 0, gt != 255
        gt[valid] -= 1
        gt[background] = 255
    return gt


def named_class_iou(prediction, gt, cls):
    require(prediction.shape == gt.shape == (224, 224), "224px matched map required")
    valid = gt != 255
    intersection = int(np.count_nonzero(valid & (prediction == cls) & (gt == cls)))
    union = int(np.count_nonzero(valid & ((prediction == cls) | (gt == cls))))
    return dict(intersection=intersection, union=union, iou=100.0 * intersection / union if union else 0.0)


def context_view(rgb, native_mask):
    """Only reconstruct the input intervention, using the real CPU helper.

    Pixels are RGB 0..255 float32, normalized with the released CLIP constants.
    Return float pixels as well as the lossless nearest-upsampled boolean mask.
    No support is measured here: C always comes from the sealed receipt.
    """
    require(rgb.shape == (224, 224, 3) and native_mask.dtype == np.bool_, "Input/mask schema")
    pixels = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None]
    mean, std = CLIP_MEAN[None], CLIP_STD[None]
    normalized = (pixels - mean) / std
    adapter = SimpleNamespace(_mean=mean, _std=std)
    with torch.inference_mode():
        background = TraceInterventionMixin.make_trace_counterfactual_background(adapter, normalized)
        mask = torch.nn.functional.interpolate(torch.from_numpy(native_mask.astype(np.float32))[None, None],
                                               size=(224, 224), mode="nearest")
        view = normalized * mask + background * (1 - mask)
        require(torch.equal(view[mask.expand_as(view).bool()], normalized[mask.expand_as(view).bool()]),
                "Candidate input pixels changed in normalized space")
        output = (view * std + mean)[0].permute(1, 2, 0).numpy()
    return output, mask[0, 0].numpy().astype(bool)


def service_status(suite):
    command = ["systemctl", "--user", "show", "trace-image-full-20261029-0333.service",
               "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "ExecMainCode",
               "-p", "ExecMainStatus", "-p", "ExecStart", "-p", "RemainAfterExit", "-p", "Restart"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    require(all(fields.get(k) == v for k, v in dict(ActiveState="active", SubState="exited", Result="success",
                ExecMainCode="1", ExecMainStatus="0", RemainAfterExit="yes", Restart="no").items()),
            "Full-suite retained service is not successfully exited")
    require(str(suite) in fields["ExecStart"] and SUITE_HASH in fields["ExecStart"], "Wrong service identity")
    return dict(command=command, properties=fields)


class Evidence:
    def __init__(self, suite):
        self.suite = Path(suite).resolve()
        self.checked = {}
        self.protocols, self.seals, self.results = {}, {}, {}
        self.label_hashes = {}
        self.gt_reads = {d: set() for d in DATASETS}
        self.prediction_reads = {d: set() for d in DATASETS}
        self.receipt_images = {d: set() for d in DATASETS}
        self.all_seals_checked = False
        require(sha(self.suite / "SUITE_PREPARED.json") == SUITE_HASH, "Wrong prepared full suite")
        self.record(self.suite / "SUITE_PREPARED.json")
        complete = self.record(self.suite / "SUITE_SEALED.json")
        barrier = self.record(self.suite / "ALL_FULL_PREDICTIONS_SEALED.json")
        require(complete["complete"] and complete["formal_cells"] == 6 and complete["arm_results"] == 18
                and complete["passed_gt_free_gates"] == 6 and complete["selected_mode"] == "C", "Incomplete full suite")
        require(barrier["complete"] and not barrier["gt_contents_opened"]
                and barrier["suite_prepared_sha256"] == complete["suite_prepared_sha256"] == SUITE_HASH, "Invalid barrier")
        self.seal_service_flag = complete["service_success_verified"]
        self.service = service_status(self.suite)
        for dataset in DATASETS:
            cell = self.cell(dataset)
            relative = str(cell.relative_to(self.suite))
            seal = self.record(cell / "PREDICTIONS_SEALED.json", barrier["prediction_seal_sha256"][relative])
            require(seal["complete"] and seal["gt_contents_opened"] is False
                    and seal["images"] == POPULATIONS[dataset] and tuple(seal["arms"]) == ARMS
                    and seal["suite_prepared_sha256"] == SUITE_HASH, "Invalid prediction seal")
            self.seals[dataset] = seal["artifact_sha256"]
            protocol = self.record(cell / "prediction_protocol.json", self.seals[dataset]["prediction_protocol.json"])
            require(protocol["suite_prepared_sha256"] == SUITE_HASH, "Wrong protocol suite")
            manifest = protocol["manifest"]
            require(manifest["dataset"] == dataset and manifest["max_images"] == POPULATIONS[dataset]
                    and manifest["config"]["resize"] == manifest["config"]["patch_size"] == [224, 224], "Wrong manifest")
            require(len(manifest["panel"]["inputs"]) == POPULATIONS[dataset], "Wrong population")
            self.protocols[dataset] = manifest
            scoring = self.record(cell / "SCORED_SEALED.json", complete["artifact_sha256"][relative + "/SCORED_SEALED.json"])
            require(scoring["prediction_seal_sha256"] == sha(cell / "PREDICTIONS_SEALED.json"), "Unbound score seal")
            self.label_hashes[dataset] = self.record(cell / "score_inputs.json", scoring["artifact_sha256"]["score_inputs.json"])["label_sha256"]
            result = self.record(cell / "result.json", scoring["artifact_sha256"]["result.json"])
            require(result["label_contents_first_opened_after_all_full_predictions_sealed"]
                    and result["barrier_sha256"] == sha(self.suite / "ALL_FULL_PREDICTIONS_SEALED.json")
                    and result["status"] == "completed" and result["images"] == POPULATIONS[dataset]
                    and set(result["arm_results"]) == set(ARMS), "Invalid score/barrier binding")
            for arm in ARMS:
                scores = result["arm_results"][arm]
                inter, union = np.array(scores["intersection"]), np.array(scores["union"])
                pooled = 100 * np.mean(inter[union > 0] / union[union > 0])
                require(abs(pooled - scores["miou_unrounded"]) < 1e-10, "Pooled mIoU mismatch")
            self.results[dataset] = result
        for relative in ("trace_ovss/method/trace/interventions.py", "trace_ovss/method/trace/constants.py",
                         "trace_ovss/data/datasets.py", "trace_ovss/method/image_balanced.py",
                         "trace_ovss/method/trace/candidates.py", "trace_ovss/method/trace/objective.py",
                         "tools/evolution_image_full/run.py", "trace_ovss/cli.py"):
            self.record(self.suite / "runtime" / relative, complete["runtime_hashes"][relative], parse=False)
            self.record(ROOT / relative, complete["runtime_hashes"][relative], parse=False)
        self.all_seals_checked = True

    def cell(self, dataset):
        return self.suite / "cells/full" / dataset / "seed0"

    def record(self, path, expected=None, *, parse=True):
        path = Path(path)
        digest = sha(path)
        require(expected is None or digest == expected, f"Hash drift: {path}")
        self.checked[str(path)] = digest
        return read(path) if parse else digest

    def bounded(self, dataset, index):
        require(self.all_seals_checked, "GT/prediction diagnostics before global six-seal check")
        require(type(index) is int and 0 <= index < min(LIMIT, POPULATIONS[dataset]), "Diagnostic bound exceeded")

    def gt(self, dataset, index):
        self.bounded(dataset, index)
        self.gt_reads[dataset].add(index)
        panel = self.protocols[dataset]["panel"]
        path = Path(panel["inputs"][index]["label"])
        self.record(path, self.label_hashes[dataset][str(path)], parse=False)
        with Image.open(path) as image:
            gt = prepare_gt(image.resize((224, 224), Image.Resampling.NEAREST), panel["reduce_zero_label"])
        require(np.all((gt == 255) | ((gt >= 0) & (gt < len(panel["classes"])))), "GT category range")
        return gt

    def prediction(self, dataset, index, arm):
        self.bounded(dataset, index)
        self.prediction_reads[dataset].add(index)
        relative = f"predictions/{arm}/{index:06d}.npy"
        path = self.cell(dataset) / relative
        self.record(path, self.seals[dataset][relative], parse=False)
        array = np.load(path, allow_pickle=False)
        require(array.shape == (224, 224) and np.issubdtype(array.dtype, np.integer)
                and array.min() >= 0 and array.max() < len(self.protocols[dataset]["panel"]["classes"]), "Prediction category schema")
        return array

    def photograph(self, dataset, index):
        self.bounded(dataset, index)
        entry = self.protocols[dataset]["panel"]["inputs"][index]
        self.record(entry["image"], entry["image_sha256"], parse=False)
        with Image.open(entry["image"]) as image:
            return np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)).copy()

    def receipts(self, dataset):
        manifest = self.protocols[dataset]
        batch = manifest["config"]["batch_size"]
        for episode in range((min(LIMIT, POPULATIONS[dataset]) + batch - 1) // batch):
            relative = f"episodes/{episode:06d}.image_C.json"
            payload = self.record(self.cell(dataset) / relative, self.seals[dataset][relative])
            require(payload["arm"] == "image_C" and payload["diagnostic_forward_inputs"] == payload["candidate_count"], "Wrong probe schedule")
            for candidate_index, row in enumerate(payload["candidates"]):
                index = episode * batch + int(row["image_idx"])
                if index >= LIMIT:
                    continue
                self.bounded(dataset, index)
                self.receipt_images[dataset].add(index)
                require(payload["ordered_ids"][row["image_idx"]] == manifest["panel"]["inputs"][index]["image"], "Image receipt mismatch")
                require(row["batch_idx"] == row["image_idx"] and row["target_support"] is None
                        and row["residual_support"] is None and row["dominance"] is None
                        and row["diagnostic_algorithm"] == "C", "Non-C receipt")
                cls = category(manifest["panel"], row["class_idx"])
                require(cls == row["preupdate_candidate_category_idx"], "Original query/category mismatch")
                mask = decode_region(row["region"])
                require(int(mask.sum()) == row["area"] and row["class_name"] == manifest["panel"]["queries"][row["class_idx"]], "Candidate identity mismatch")
                require(row["reliability"] == row["counterfactual_support"]
                        and 0 <= row["confidence"] <= 1 and 0 <= row["counterfactual_support"] <= 1, "Unmeasured support")
                yield dict(dataset=dataset, image_index=index, episode=episode, candidate_index=candidate_index,
                           receipt=str(self.cell(dataset) / relative), category_id=cls,
                           category_name=manifest["panel"]["classes"][cls], query=manifest["panel"]["queries"][row["class_idx"]],
                           receipt_row=row)


def select_teaser(evidence):
    selected, counts = [], {}
    for dataset, largest in (("pc60", True), ("coco_object", False)):
        eligible = []
        for case in evidence.receipts(dataset):
            row = case["receipt_row"]
            if case["category_name"] != "background" and row["confidence"] >= .9 and 16 <= row["area"] <= 192:
                eligible.append(case)
        require(eligible, f"No eligible teaser candidates: {dataset}")
        counts[dataset] = len(eligible)
        sign = -1 if largest else 1
        selected.append(min(eligible, key=lambda c: (sign * c["receipt_row"]["counterfactual_support"], c["image_index"], c["candidate_index"])))
    return selected, counts


def select_segmentation(evidence):
    selected, counts = [], {}
    for dataset in ("coco_cd", "coco_object", "pc60", "voc21"):
        panel = evidence.protocols[dataset]["panel"]
        choices = []
        for index in range(min(LIMIT, POPULATIONS[dataset])):
            gt = evidence.gt(dataset, index)
            classes, areas = np.unique(gt[gt != 255], return_counts=True)
            eligible = [(int(c), int(n)) for c, n in zip(classes, areas)
                        if n >= 128 and panel["classes"][c] != "background"]
            if not eligible:
                continue
            cls, area = min(eligible, key=lambda pair: (-pair[1], pair[0]))
            a = named_class_iou(evidence.prediction(dataset, index, "image_confidence"), gt, cls)
            b = named_class_iou(evidence.prediction(dataset, index, "image_C"), gt, cls)
            choices.append(dict(dataset=dataset, image_index=index, category_id=cls, category_name=panel["classes"][cls],
                                gt_area=area, confidence=a, trace=b, delta=b["iou"] - a["iou"], negative=dataset == "voc21"))
        require(choices, "No eligible segmentation case")
        sign = 1 if dataset == "voc21" else -1
        case = min(choices, key=lambda c: (sign * c["delta"], c["image_index"]))
        require(case["delta"] < 0 if case["negative"] else case["delta"] > 0, "Case direction not genuine")
        selected.append(case)
        counts[dataset] = len(choices)
    return selected, counts


def configure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False,
                         "axes.linewidth": .6, "lines.linewidth": .7, "patch.linewidth": .6,
                         "text.usetex": False, "savefig.facecolor": "white"})


def save_figure(fig, output, name):
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    require(min(sizes) >= 9, "Figure text smaller than 9pt at native print size")
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    bounds = []
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            bbox = item.get_window_extent(renderer)
            require(bbox.x0 >= canvas.x0 - 1 and bbox.x1 <= canvas.x1 + 1
                    and bbox.y0 >= canvas.y0 - 1 and bbox.y1 <= canvas.y1 + 1, f"Clipped figure text: {item.get_text()}")
            bounds.append((item.get_text(), bbox))
    for i, (first, a) in enumerate(bounds):
        for second, b in bounds[i + 1:]:
            require(min(a.x1, b.x1) - max(a.x0, b.x0) <= 1
                    or min(a.y1, b.y1) - max(a.y0, b.y0) <= 1,
                    f"Overlapping figure text: {first!r}, {second!r}")
    all_text = [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    math_paths = []
    for item in list(fig.findobj(Text)):
        if not item.get_visible() or not item.get_text().startswith("$"):
            continue
        formula, size = item.get_text(), item.get_fontsize()
        prop = FontProperties(size=size)
        glyphs = MathTextParser("path").parse(formula, dpi=72, prop=prop).glyphs
        minimum = min(glyph[1] for glyph in glyphs)
        require(minimum >= 9, f"Math subscript smaller than 9pt: {formula}")
        path = TextPath((0, 0), formula, size=size, usetex=False)
        box = path.get_extents()
        text_box = item.get_window_extent(renderer)
        center = fig.transFigure.inverted().transform(((text_box.x0 + text_box.x1) / 2, (text_box.y0 + text_box.y1) / 2))
        scale_x, scale_y = 1 / (fig.get_figwidth() * 72), 1 / (fig.get_figheight() * 72)
        transform = Affine2D().scale(scale_x, scale_y).translate(
            center[0] - (box.x0 + box.x1) / 2 * scale_x,
            center[1] - (box.y0 + box.y1) / 2 * scale_y)
        # MathText's real glyph outlines stay vector. Outlining only formulae
        # prevents Cairo from wrapping Greek glyph subsets in CID fonts.
        fig.add_artist(PathPatch(path, transform=transform + fig.transFigure, facecolor="black", linewidth=0))
        item.set_visible(False)
        math_paths.append(dict(source=formula, base_pt=size, minimum_glyph_pt=minimum, rendering="MathText vector glyph outlines"))
    # Cairo embeds simple TrueType subsets for our ASCII labels, avoiding the
    # CID font wrappers emitted by Matplotlib's normal Type-42 PDF backend.
    previous_dpi = fig.dpi
    surface = None
    try:
        fig.set_dpi(72)
        surface = cairo.PDFSurface(str(output / f"{name}.pdf"), fig.get_figwidth() * 72, fig.get_figheight() * 72)
        surface.restrict_to_version(cairo.PDF_VERSION_1_5)
        pdf_renderer = RendererCairo(72)
        pdf_renderer.set_context(cairo.Context(surface))
        fig.draw(pdf_renderer)
        pdf_renderer.gc.ctx.show_page()
    finally:
        if surface is not None:
            surface.finish()
        fig.set_dpi(previous_dpi)
    for dpi, suffix in ((150, ""), (72, "_print")):
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", str(dpi),
                        str(output / f"{name}.pdf"), str(output / f"{name}{suffix}")],
                       check=True, capture_output=True)
    fonts = subprocess.run(["pdffonts", str(output / f"{name}.pdf")], check=True, text=True, capture_output=True).stdout
    require("CID" not in fonts and "Type 3" not in fonts and "TrueType" in fonts, "PDF font contract")
    record = dict(width_in=float(fig.get_figwidth()), height_in=float(fig.get_figheight()),
                  minimum_text_pt=min(sizes), minimum_line_pt=.6, external_crop=False,
                  image_panels=[dict(width_in=ax.get_position().width * fig.get_figwidth(),
                                     height_in=ax.get_position().height * fig.get_figheight())
                                for ax in fig.axes if ax.images],
                  text_overlap_check="all visible text bounding boxes checked pairwise before export",
                  renders_dpi=[150, 72], embedded_font_report=fonts,
                  math=math_paths, text=all_text)
    plt.close(fig)
    return record


def photo_axis(fig, box, rgb, mask=None):
    ax = fig.add_axes(box)
    ax.imshow(rgb, interpolation="nearest", vmin=0, vmax=255)
    if mask is not None:
        ax.contour(mask, levels=[.5], colors=[CYAN], linewidths=.8)
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_linewidth(.6)
    return ax


def draw_teaser(evidence, cases, output):
    fig = plt.figure(figsize=(SINGLE_WIDTH, 4.65))
    fig.text(.5, .977, "Initial confidence is not context support", ha="center", va="top", fontsize=9, weight="bold")
    reconstructions = []
    for row, case in enumerate(cases):
        rec = case["receipt_row"]
        rgb = evidence.photograph(case["dataset"], case["image_index"])
        pixels, mask = context_view(rgb, decode_region(rec["region"]))
        y = .52 if row == 0 else .095
        title = f"{DISPLAY[case['dataset']]} / {case['category_name']}"
        fig.text(.04, y + .342, title, fontsize=9, va="bottom")
        for col, array in enumerate((rgb, pixels.clip(0, 255) / 255)):
            photo_axis(fig, [.04 + .49 * col, y, .43, .326], array, mask)
        fig.text(.255, y - .018, f"Confidence: {rec['confidence']:.3f}", ha="center", va="top")
        fig.text(.745, y - .018, f"Probe support: {rec['counterfactual_support']:.3f}", ha="center", va="top")
        prefix = f"image_full_teaser_case{row + 1}"
        Image.fromarray(rgb).save(output / f"{prefix}_input.png")
        Image.fromarray(np.rint(pixels.clip(0, 255)).astype(np.uint8)).save(output / f"{prefix}_context.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"{prefix}_mask.png")
        np.save(output / f"{prefix}_context_float32.npy", pixels, allow_pickle=False)
        reconstructions.append(dict(candidate_pixels_normalized_exact=True,
                                    candidate_rgb_max_error=float(np.max(abs(pixels[mask] - rgb[mask]))),
                                    input_png=f"{prefix}_input.png", context_float32=f"{prefix}_context_float32.npy",
                                    native_shape=rec["region"]["shape"], input_mask_true_pixels=int(mask.sum())))
    fig.text(.255, .913, "Original + candidate", ha="center")
    fig.text(.745, .913, "Context probe", ha="center")
    fig.text(.5, .018, "Cyan: candidate pixels unchanged.", ha="center", va="bottom", fontsize=9)
    return save_figure(fig, output, "image_full_teaser"), reconstructions


def draw_pipeline(output):
    fig = plt.figure(figsize=(WIDTH, 4.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(.5, .975, "TRACE", ha="center", va="top", fontsize=12, weight="bold")
    boxes = ((.018, .045, .305), (.353, .38, .305), (.688, .715, .295))
    titles = ("1  Candidate construction", "2  Context probe", "3  Per-image entropy")
    bodies = (
        ["j: source; b: crop; c: query", r"$R=(b_R,\Omega_R,c_R)$", "4-connected query regions", "Top-K by mean confidence", "Native 16 x 16 token masks"],
        ["Candidate input pixels fixed", "Outside: .35 contrast + (g,b,r)", "One view per candidate", "Original query and tokens", "Frozen initial readout"],
        ["Prompt/token mean entropy", r"$\overline{H}_R(\theta)$", "Detach; normalize per image", "Mean over active images", "Visual LayerNorm affine"],
    )
    for (left, text_x, width), title, lines in zip(boxes, titles, bodies):
        ax.add_patch(Rectangle((left, .50), width, .38, transform=ax.transAxes,
                               facecolor="#f5f7f8", edgecolor="#83949a", linewidth=.7))
        fig.text(text_x, .835, title, fontsize=9, weight="bold", va="top")
        for i, line in enumerate(lines):
            fig.text(text_x, .775 - i * .06, line, fontsize=14 if line.startswith("$") else 9, va="top")
    for x in (.33, .665):
        ax.add_patch(FancyArrowPatch((x - .009, .69), (x + .02, .69), arrowstyle="-|>", mutation_scale=9,
                                    linewidth=.8, color=CYAN, transform=ax.transAxes))
    fig.text(.5, .39, r"$\rho_R=\frac{1}{|\Omega_R|}\sum_{i\in\Omega_R}\pi^0_{b_R,i}(c_R\mid x_R^{\mathrm{ctx}})$", ha="center", fontsize=19)
    fig.text(.5, .245, r"$\alpha_{R\mid j}=\mathrm{sg}(\rho_R)\,/\!\sum_{Q\in\mathcal{R}_j^+}\mathrm{sg}(\rho_Q)$", ha="center", fontsize=19)
    fig.text(.5, .10, r"$\mathcal{L}_{\mathrm{TRACE}}(\theta)=|\mathcal{I}_+|^{-1}\sum_{j\in\mathcal{I}_+}\quad\sum_{R\in\mathcal{R}_j^+}\alpha_{R\mid j}\,\overline{H}_R(\theta)$", ha="center", fontsize=19)
    return save_figure(fig, output, "image_full_pipeline")


def overlay(rgb, labels, cls, valid):
    image = rgb.astype(float) / 255
    result = image * .48
    target = labels == cls
    result[target] = .4 * image[target] + .6 * np.array([0, .72, .76])
    result[~valid] = .86
    return result


def draw_segmentation(evidence, cases, output):
    height = 7.25
    fig = plt.figure(figsize=(WIDTH, height))
    for x, label in zip((.14, .385, .63, .875), ("Input", "Confidence", "TRACE", "GT")):
        fig.text(x, 7.07 / height, label, ha="center", fontsize=10, weight="bold")
    for row, case in enumerate(cases):
        dataset, index, cls = case["dataset"], case["image_index"], case["category_id"]
        rgb, gt = evidence.photograph(dataset, index), evidence.gt(dataset, index)
        a = evidence.prediction(dataset, index, "image_confidence")
        b = evidence.prediction(dataset, index, "image_C")
        title_y = 6.8 - row * 1.67
        image_bottom = title_y - .10 - 1.4
        suffix = ", negative" if case["negative"] else ""
        heading = (f"{DISPLAY[dataset]} / {case['category_name']} | Class IoU: "
                   f"{case['confidence']['iou']:.1f} -> {case['trace']['iou']:.1f} "
                   f"({case['delta']:+.1f} pp{suffix})")
        fig.text(.03, title_y / height, heading,
                 color=RED if case["negative"] else "#23343a", fontsize=9)
        arrays = (rgb, overlay(rgb, a, cls, gt != 255), overlay(rgb, b, cls, gt != 255), overlay(rgb, gt, cls, gt != 255))
        for col, array in enumerate(arrays):
            center = (.14, .385, .63, .875)[col]
            photo_axis(fig, [center - .7 / WIDTH, image_bottom / height, 1.4 / WIDTH, 1.4 / height], array)
        prefix = f"image_full_segmentation_case{row + 1}"
        Image.fromarray(rgb).save(output / f"{prefix}_input.png")
        for label, array in (("confidence", a), ("trace", b), ("gt", gt)):
            np.save(output / f"{prefix}_{label}.npy", array, allow_pickle=False)
    fig.text(.5, .055 / height, "Class IoU: Confidence -> TRACE. Cyan: named class; gray: GT void.", ha="center", fontsize=9)
    return save_figure(fig, output, "image_full_segmentation")


def allocation_rows(evidence):
    rows = []
    for dataset in DATASETS:
        result = evidence.results[dataset]
        values = {a: result["arm_results"][a]["miou_unrounded"] for a in ARMS}
        rows.append(dict(dataset=dataset, images=result["images"], **values,
                         C_minus_confidence=values["image_C"] - values["image_confidence"],
                         C_minus_uniform=values["image_C"] - values["image_uniform"]))
    return rows


def delta_label(value):
    return f"{value:+.3f}"


def draw_allocation(rows, output):
    fig = plt.figure(figsize=(SINGLE_WIDTH, 3.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(.5, .973, "Full-split allocation controls", ha="center", va="top", fontsize=10, weight="bold")
    fig.text(.5, .881, "TRACE - control / pooled mIoU (pp)", ha="center", fontsize=9)
    for x, label, align in ((.045, "Dataset", "left"), (.61, "Confidence", "center"), (.87, "Uniform", "center")):
        fig.text(x, .792, label, ha=align, weight="bold", fontsize=9)
    for y in (.755, .237):
        ax.plot([.04, .965], [y, y], color="#52636b", linewidth=.6, transform=ax.transAxes)
    for index, row in enumerate(rows):
        y = .698 - index * .081
        fig.text(.045, y, DISPLAY[row["dataset"]], va="center", fontsize=9)
        fig.text(.61, y, delta_label(row["C_minus_confidence"]), ha="center", va="center", fontsize=9, color=CYAN)
        fig.text(.87, y, delta_label(row["C_minus_uniform"]), ha="center", va="center", fontsize=9, color="#6c4e77")
    fig.text(.5, .165, "Six complete splits; same candidates.", ha="center", fontsize=9)
    fig.text(.5, .098, "Seed 0; descriptive differences only.", ha="center", fontsize=9)
    fig.text(.5, .031, "COCO-Stuff vs. Confidence: +0.043 pp.", ha="center", fontsize=9)
    return save_figure(fig, output, "image_full_allocation")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("/home/wwww/hzl/results/trace_image_full_20261029_0333"))
    parser.add_argument("--output", type=Path, default=ROOT / "paper_evolution/figure")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output == ROOT / "paper_evolution/figure", "Write only the owned figure asset directory")
    torch.set_num_threads(1)
    configure_style()
    evidence = Evidence(args.suite)
    print("Six prediction seals and full scores verified; CPU-only bounded selection starts.", flush=True)
    teaser, teaser_counts = select_teaser(evidence)
    segmentation, segmentation_counts = select_segmentation(evidence)
    output.mkdir(parents=True, exist_ok=True)
    layouts = {}
    layouts["image_full_teaser"], reconstruction = draw_teaser(evidence, teaser, output)
    layouts["image_full_pipeline"] = draw_pipeline(output)
    layouts["image_full_segmentation"] = draw_segmentation(evidence, segmentation, output)
    rows = allocation_rows(evidence)
    layouts["image_full_allocation"] = draw_allocation(rows, output)
    for case in teaser + segmentation:
        case["image_path"] = evidence.protocols[case["dataset"]]["panel"]["inputs"][case["image_index"]]["image"]
    provenance = dict(schema_version=1, suite=str(evidence.suite), suite_prepared_sha256=SUITE_HASH,
        script_sha256=sha(__file__), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], model_forwards=0,
        selection_rule=SELECTION_RULE, eligible_counts=dict(teaser=teaser_counts, segmentation=segmentation_counts),
        teaser=teaser, segmentation=segmentation, reconstruction=reconstruction, allocation=rows,
        layout=layouts, input_artifact_sha256=evidence.checked, service=evidence.service,
        suite_seal_service_success_verified=evidence.seal_service_flag,
        service_note="Suite seal flag is false by construction; retained service verified independently read-only. No seal changed.",
        mask_note="Actual new receipts are native boolean 16x16 packed little-endian, not 14x14; no resampling before decode.",
        normalization=dict(RGB_scale="0..255 float32", mean=CLIP_MEAN.flatten().tolist(), std=CLIP_STD.flatten().tolist(),
                           formula="z=(x-mu)/sigma; xhat=z*sigma+mu; m=spatial_mean(xhat) per channel; b=clip(cycle_gbr(0.35*(xhat-m)+m),0,255); zC=z*M+((b-mu)/sigma)*(1-M)",
                           mask_upsampling="torch interpolate nearest from recorded native shape to 224x224",
                           display="Probe embedded as floating RGB /255; companion PNG rounded for inspection; float32 NPY preserves reconstruction values."),
        scope="C measured in sealed initial frozen readout at original query and original tokens, never remeasured. Confidence is a matched allocation control, not MLMP. Initial full-map prediction unavailable and never fabricated.",
        gt_conventions="224x224 nearest; ignore 255; reduce-zero only where frozen protocol says so; category predictions already use per-category max over query probabilities before argmax.",
        diagnostic_reads={d: dict(gt_indices=sorted(evidence.gt_reads[d]), prediction_indices=sorted(evidence.prediction_reads[d]),
                                 receipt_image_indices=sorted(evidence.receipt_images[d])) for d in DATASETS},
        bounds_note="One boundary episode may include later images in its JSON; only records with global index <400 enter selection. No per_image_json, full count arrays, model checkpoints or GPU calls loaded.")
    provenance["output_sha256"] = {p.name: sha(p) for p in sorted(output.glob("image_full_*")) if p.suffix in (".pdf", ".png", ".npy")}
    write_json(output / "image_full_provenance.json", provenance)
    print(json.dumps(dict(teaser=[{k: c[k] for k in ("dataset", "image_index", "category_name")} for c in teaser],
                          segmentation=segmentation, allocation=rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
