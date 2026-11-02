"""Draw measured two-view examples and matched segmentation comparisons on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_ovss.data.datasets import CLIP_MEAN, CLIP_STD
from trace_ovss.method.trace.interventions import TraceInterventionMixin


DISPLAY = {"coco_cd": "COCO-CD", "pc60": "PC60", "voc21": "VOC21",
           "coco_object": "COCO Object", "coco_stuff": "COCO-Stuff"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value):
    return str(value).replace("_", r"\_").replace("&", r"\&")


def photograph(protocol, index):
    with Image.open(protocol["inputs"][index]["image"]) as image:
        return np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)).copy()


def view_images(rgb, token_mask):
    original = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
    mean, std = CLIP_MEAN.unsqueeze(0), CLIP_STD.unsqueeze(0)
    normalized = (original - mean) / std
    adapter = SimpleNamespace(_mean=mean, _std=std)
    weak = TraceInterventionMixin.make_trace_weak_image(adapter, normalized)
    shifted = TraceInterventionMixin.make_trace_counterfactual_background(adapter, normalized)
    mask = torch.nn.functional.interpolate(torch.from_numpy(token_mask.astype(np.float32))[None, None],
        size=(224, 224), mode="nearest")
    views = [normalized * mask + context * (1-mask) for context in (weak, shifted)]
    arrays = [((view * std + mean)[0].permute(1, 2, 0).numpy().clip(0, 255) / 255) for view in views]
    return arrays, mask[0, 0].numpy().astype(bool)


def support_cases(suite):
    selected = []
    for dataset in ("coco_object", "pc60"):
        cell = suite / "cells/mechanism" / dataset
        protocol = read(cell / "prediction_protocol.json")
        choices = []
        for path in sorted((cell / "episodes").glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                for i, mask in enumerate(data["masks"]):
                    query = int(data["query_index"][i])
                    cls = protocol["query_to_class"][query] if protocol["query_to_class"] is not None else query
                    if protocol["classes"][cls] == "background" or not 32 <= mask.sum() <= 192 or data["confidence"][i] < .9:
                        continue
                    choices.append(dict(dataset=dataset, image_index=int(data["image_index"][i]),
                        episode=int(path.stem), candidate=i, query=query, semantic_class=cls, name=protocol["classes"][cls],
                        confidence=float(data["confidence"][i]), T=float(data["T"][i]), C=float(data["C"][i]),
                        q=float(data["q"][i]), mask=mask.copy()))
        if not choices:
            raise RuntimeError(f"No eligible illustrative candidate: {dataset}")
        high = None
        for candidate in sorted(choices, key=lambda c: (-c["q"], c["image_index"])):
            gt = gt_labels(protocol, candidate["image_index"])
            mask = np.asarray(Image.fromarray(candidate["mask"]).resize((224, 224), Image.Resampling.NEAREST))
            valid = mask & (gt != 255)
            purity = float(np.mean(gt[valid] == candidate["semantic_class"])) if valid.any() else 0.0
            if purity >= .7:
                high = dict(candidate, offline_gt_purity=purity)
                break
        if high is None:
            raise RuntimeError("No high-support illustration with offline semantic agreement")
        low = min((c for c in choices if c["image_index"] != high["image_index"]),
                  key=lambda c: (c["q"], c["image_index"]))
        selected.extend([(high, protocol), (low, protocol)])
    return selected


def draw_support(suite, output, *, short=False):
    cases = support_cases(suite)
    if short:
        cases = [cases[2], cases[1]]
    fig, axes = plt.subplots(len(cases), 3, figsize=(7.05, 4.5 if short else 7.15))
    records = []
    for row, (case, protocol) in enumerate(cases):
        rgb = photograph(protocol, case["image_index"])
        (preserved, perturbed), mask = view_images(rgb, case["mask"])
        labels = [f"Original confidence: {case['confidence']:.2f}",
                  f"Preserved support: {case['T']:.2f}", f"Perturbed support: {case['C']:.2f}"]
        for col, (ax, image, label) in enumerate(zip(axes[row], (rgb, preserved, perturbed), labels)):
            ax.imshow(image, interpolation="nearest")
            ax.contour(mask, levels=[.5], colors=["#087b8a"], linewidths=.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(label, fontsize=10, labelpad=3)
        axes[row, 0].text(0, 1.01, text(f"{DISPLAY[case['dataset']]}: {case['name']}"),
                          transform=axes[row, 0].transAxes, fontsize=10, va="bottom")
        axes[row, 1].text(.5, 1.01, f"Common support: {case['q']:.2f}", transform=axes[row, 1].transAxes,
                          fontsize=10, va="bottom", ha="center")
        records.append({k: v for k, v in case.items() if k != "mask"})
    fig.subplots_adjust(left=.035, right=.96, bottom=.04, top=.91, wspace=.07, hspace=.42)
    for ax, title in zip(axes[0], ("Input candidate", "Target-preserved view", "Context-perturbed view")):
        position = ax.get_position()
        fig.text((position.x0+position.x1)/2, .975, title, ha="center", va="top", fontsize=10)
    fig.savefig(output / ("two_view_teaser.pdf" if short else "two_view_qualitative.pdf"), dpi=450)
    plt.close(fig)
    return records


def draw_controls(suite, replay, output):
    fig, axes = plt.subplots(3, 2, figsize=(7.05, 6.3))
    labels = {"uniform": "Uniform regions", "target": "Preserved only", "counterfactual": "Perturbed only",
              "mean": "Arithmetic mean", "product": "Product", "pixel_normalized": "Pixel-normalized", "trace": "TRACE"}
    for ax, dataset in zip(axes.flat, DISPLAY):
        result = read(suite / "cells/mechanism" / dataset / "result.json")
        scores = {r["policy"]: r["miou_unrounded"] for r in result["policy_results"] if r["policy"] != "permuted"}
        if dataset in ("pc60", "coco_object"):
            scores.update({r["policy"]: r["miou_unrounded"] for r in read(replay / "cells" / dataset / "result.json")["policy_results"]})
        policies = [p for p in labels if p in scores]
        values = [scores[p]-scores["confidence"] for p in policies]
        for y, (policy, value) in enumerate(zip(policies, values)):
            ax.plot(value, y, "D" if policy == "trace" else "o", ms=5,
                    color="#16616b" if policy == "trace" else "#555555", mfc="#16616b" if policy == "trace" else "white")
            label = f"{value:+.2f}" if abs(value) >= .005 else "0.00"
            ax.annotate(label, (value, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", pad=.2))
        ax.axvline(0, ls="--", lw=.7, color="#888888")
        lo, hi = min(values+[0]), max(values+[0])
        span = max(hi-lo, .4)
        ax.set_xlim(lo-.08*span, hi+.32*span)
        ax.set_yticks(range(len(policies)), [labels[p] for p in policies], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(DISPLAY[dataset], fontsize=10)
        ax.set_xlabel("mIoU change vs. confidence (pp)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    fig.subplots_adjust(left=.19, right=.985, bottom=.065, top=.95, wspace=.86, hspace=.74)
    fig.savefig(output / "allocation_controls.pdf")
    plt.close(fig)


def gt_labels(protocol, index):
    with Image.open(protocol["inputs"][index]["label"]) as image:
        gt = np.asarray(image.resize((224, 224), Image.Resampling.NEAREST)).astype(np.int64)
    if gt.ndim == 3:
        gt = gt[..., 0]
    if protocol["reduce_zero_label"]:
        background, valid = gt == 0, gt != 255
        gt[valid] -= 1
        gt[background] = 255
    return gt


def named_iou(prediction, gt, classes):
    valid = gt != 255
    values = []
    for cls in classes:
        intersection = np.count_nonzero(valid & (prediction == cls) & (gt == cls))
        union = np.count_nonzero(valid & ((prediction == cls) | (gt == cls)))
        values.append(intersection / union if union else 0)
    return 100 * float(np.mean(values))


def segmentation_case(suite, dataset, failure=False):
    cell = suite / "cells/mechanism" / dataset
    protocol = read(cell / "prediction_protocol.json")
    choices = []
    for index in range(min(400, len(protocol["inputs"]))):
        gt = gt_labels(protocol, index)
        names, counts = np.unique(gt[gt != 255], return_counts=True)
        eligible = [(int(n), int(c)) for n, c in zip(names, counts)
                    if c >= 128 and protocol["classes"][n] != "background"]
        eligible.sort(key=lambda pair: (-pair[1], pair[0]))
        if not eligible or (dataset == "coco_object" and len(eligible) < 2):
            continue
        classes = [n for n, _ in eligible[:2 if dataset == "coco_object" else 1]]
        confidence = np.load(cell / "predictions/confidence" / f"{index:06d}.npy", allow_pickle=False)
        trace = np.load(cell / "predictions/trace" / f"{index:06d}.npy", allow_pickle=False)
        a, b = named_iou(confidence, gt, classes), named_iou(trace, gt, classes)
        choices.append(dict(dataset=dataset, image_index=index, classes=classes, names=[protocol["classes"][c] for c in classes],
            confidence_iou=a, trace_iou=b, delta=b-a, failure=failure))
    if not choices:
        raise RuntimeError("No eligible segmentation illustration")
    case = (min if failure else max)(choices, key=lambda c: c["delta"])
    if (failure and case["delta"] >= 0) or (not failure and case["delta"] <= 0):
        raise RuntimeError("Illustrative case does not have its declared direction")
    return case, protocol


def overlay(rgb, prediction, classes):
    value = rgb.astype(float) / 255 * .55
    colors = np.array([[.05, .72, .72], [.96, .61, .12]])
    for index, cls in enumerate(classes):
        mask = prediction == cls
        value[mask] = .5 * rgb[mask] / 255 + .5 * colors[index]
    return value


def draw_segmentation(suite, output):
    cases = [segmentation_case(suite, d, failure=(d == "voc21"))
             for d in ("coco_cd", "coco_object", "pc60", "voc21")]
    fig, axes = plt.subplots(4, 4, figsize=(7.05, 7.2))
    records = []
    for row, (case, protocol) in enumerate(cases):
        index, dataset = case["image_index"], case["dataset"]
        cell = suite / "cells/mechanism" / dataset
        rgb, gt = photograph(protocol, index), gt_labels(protocol, index)
        confidence = np.load(cell / "predictions/confidence" / f"{index:06d}.npy", allow_pickle=False)
        trace = np.load(cell / "predictions/trace" / f"{index:06d}.npy", allow_pickle=False)
        arrays = [rgb, overlay(rgb, confidence, case["classes"]), overlay(rgb, trace, case["classes"]), overlay(rgb, gt, case["classes"])]
        for col, (ax, array) in enumerate(zip(axes[row], arrays)):
            ax.imshow(array, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col in (1, 2):
                metric = case["confidence_iou"] if col == 1 else case["trace_iou"]
                ax.set_xlabel(f"Named-class IoU: {metric:.1f}", fontsize=9, labelpad=3)
        axes[row, 0].set_xlabel(text(" / ".join(case["names"])), fontsize=9, labelpad=3)
        kind = "Failure" if case["failure"] else "Positive example"
        axes[row, 0].text(0, 1.02, text(f"{DISPLAY[dataset]}: {kind}"), transform=axes[row, 0].transAxes, fontsize=9)
        records.append(case)
    fig.subplots_adjust(left=.025, right=.995, bottom=.035, top=.91, wspace=.045, hspace=.38)
    for ax, title in zip(axes[0], ("Input image", "Confidence weighting", "TRACE", "Ground truth")):
        position = ax.get_position()
        fig.text((position.x0+position.x1)/2, .975, title, ha="center", va="top", fontsize=9.5)
    fig.savefig(output / "segmentation_examples.pdf", dpi=450)
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10,
        "axes.linewidth": .7, "lines.linewidth": .8, "xtick.labelsize": 9, "ytick.labelsize": 9})
    output = args.paper / "figure"
    output.mkdir(exist_ok=True)
    support = draw_support(args.suite, output)
    teaser = draw_support(args.suite, output, short=True)
    draw_controls(args.suite, args.replay, output)
    segmentation = draw_segmentation(args.suite, output)
    record = dict(suite=str(args.suite), replay=str(args.replay), support_cases=support, teaser_cases=teaser,
        segmentation_cases=segmentation,
        selection="Illustrative extremes, not random or representative: high-confidence masks32-192 tokens ranked by common support; high-support illustrations additionally require offline semantic GT purity>=0.7, never used in adaptation; named-class IoU extremes among first400 images per dataset.",
        transforms="CPU rendering through the exact intervention helper; displayed scores are stored GPU measurements, not recomputed.",
        segmentation_metric="Mean IoU over the explicitly named one/two ground-truth categories, not dataset or whole-image mIoU.",
        source_sha256={str(p): digest(p) for p in (args.suite / "SUITE_SEALED.json", args.replay / "SUITE_SEALED.json", Path(__file__))})
    (output / "final_figure_provenance.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(support_cases=len(support), segmentation_cases=len(segmentation))), flush=True)


if __name__ == "__main__":
    main()
