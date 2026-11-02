"""Draw current-method diagrams and source-bound development comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


def configure():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": .7,
        "lines.linewidth": 1.0, "text.latex.preamble": r"\usepackage{amsmath}"})


def pipeline(path):
    fig, ax = plt.subplots(figsize=(7.05, 2.8))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    positions = [(1.2, 3.2), (4.8, 3.2), (8.5, 3.2),
                 (1.2, .75), (4.8, .75), (8.5, .75)]
    labels = [
        (r"\textbf{Connected Candidates}", "Initial token-grid prediction", r"Top-$K$ per source image"),
        (r"\textbf{Two Views per Candidate}", "Preserve region, weaken context", "Preserve region, perturb context"),
        (r"\textbf{Shared Initial Encoder}", r"Read $T_R$ and $C_R$", "at original candidate positions"),
        (r"\textbf{Relative Update Weights}", r"$\rho_R=\min(T_R,C_R)$", "Detached; fixed in the episode"),
        (r"\textbf{Region-Normalized Entropy}", r"$\mathcal L=\frac{\sum_R\rho_R\overline H_R}{\sum_R\rho_R}$", "Original input; no labels"),
        (r"\textbf{Adapt and Predict}", "Update visual LayerNorm affine", "Reconstruct pixel segmentation"),
    ]
    for (x, y), lines in zip(positions, labels):
        ax.text(x, y + .23, lines[0], ha="center", va="center", fontsize=10)
        ax.text(x, y - .15, lines[1], ha="center", va="center", fontsize=9.5)
        ax.text(x, y - .5, lines[2], ha="center", va="center", fontsize=9.5)
    for a, b in [((2.65, 3.1), (3.25, 3.1)), ((6.5, 3.1), (6.95, 3.1)),
                 ((2.65, .55), (3.2, .55)), ((6.55, .55), (7, .55))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                    linewidth=1, color="black"))
    ax.plot([8.5, 8.5, 1.2, 1.2], [2.4, 1.8, 1.8, 1.45], color="black", lw=.8)
    ax.add_patch(FancyArrowPatch((1.2, 1.55), (1.2, 1.35), arrowstyle="-|>",
                                mutation_scale=10, linewidth=1, color="black"))
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.04)
    fig.savefig(path)
    plt.close(fig)


def comparisons(root, path):
    with (root / "baselines.csv").open() as handle:
        baselines = list(csv.DictReader(handle))
    with (root / "comparison.csv").open() as handle:
        trace = {row["dataset"]: row for row in csv.DictReader(handle)}
    datasets = ("voc21", "pc60", "coco_object", "coco_cd")
    display = ("VOC21", "PC60", "COCO Object", "COCO-CD")
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.5))
    for ax, dataset, title in zip(axes, datasets, display):
        by_method = {r["method"]: float(r["mIoU"]) for r in baselines if r["dataset"] == dataset}
        values = [by_method[m] for m in ("noadapt", "tent", "tpt", "mlmp")]
        values.append(float(trace[dataset]["mIoU"]))
        bars = ax.bar(np.arange(5), values, width=.7,
                      color=["white"] * 4 + ["#285e59"], edgecolor="black", linewidth=.7)
        for bar, hatch in zip(bars, ("", "//", "..", "xx", "")):
            bar.set_hatch(hatch)
        lower, upper = min(values) - 2, max(values) + 3.3
        ax.set_ylim(lower, upper)
        for x, value in enumerate(values):
            ax.text(x, value + .18, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(5), ("None", "TENT", "TPT", "MLMP", "TRACE"), rotation=60, ha="right")
        ax.set_title(title, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linewidth=.5, alpha=.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"mIoU ($\%$)")
    fig.subplots_adjust(left=.065, right=.995, top=.85, bottom=.29, wspace=.5)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    configure()
    output = args.paper_root / "figure"
    output.mkdir(parents=True, exist_ok=True)
    pipeline(output / "two_view_pipeline.pdf")
    comparisons(args.baseline_root, output / "native_operating_points.pdf")
    sources = [args.baseline_root / n for n in ("baselines.csv", "comparison.csv")]
    provenance = dict(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope="Sealed historical full seed0, method-native backends and inherited settings; not equal tuning or optimization budgets.",
        pipeline="Explicit TwoViewTRACE; no residual/dominance/semantic-role branch.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
