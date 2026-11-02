"""CPU-only sidecar figures for the sealed eight-dataset tuned full suite.

Verify all eight selection/prediction/score metadata chains before opening
bounded diagnostic labels or predictions. Never replay inference or rescore a
full population. Shared rendering and the static pipeline are imported intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import draw_image_full_figures as shared
from tools.evolution_tuning.common import canonical_hash

np, plt, torch = shared.np, shared.plt, shared.torch
require, sha = shared.require, shared.sha
ROOT = shared.ROOT
SUITE = Path("/home/wwww/hzl/results/trace_single_probe_waterbirds_full_20261029_0817")
SUITE_HASH = "4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3"
OUTPUT = ROOT / "paper_evolution/figure/tuned"
SERVICE = "trace-waterbirds-full-20261029-0817.service"
DATASETS = ("voc20", "voc21", "pc59", "pc60", "coco_cd", "coco_object", "coco_stuff", "waterbirds_seg")
POPULATIONS = dict(zip(DATASETS, (1449, 1449, 5105, 5105, 500, 5000, 5000, 5322)))
ARMS = ("no_adapt", "image_uniform", "image_confidence", "image_C")
DISPLAY = {**shared.DISPLAY, "pc59": "PC59", "waterbirds_seg": "Waterbirds*"}
LIMIT = shared.LIMIT
SELECTION_RULE = {**shared.SELECTION_RULE,
    "null_policy": "Keep the selected extremum in provenance; show a null case if the predeclared strict sign fails. Never retune the rule or extend the first400 bound."}


def service_status(suite):
    command = ["systemctl", "--user", "show", SERVICE]
    for key in ("ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus",
                "ExecStart", "RemainAfterExit", "Restart", "MainPID"):
        command.extend(["-p", key])
    result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=30)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    expected = dict(ActiveState="active", SubState="exited", Result="success", ExecMainCode="1",
                    ExecMainStatus="0", RemainAfterExit="yes", Restart="no", MainPID="0")
    require(all(fields.get(k) == v for k, v in expected.items()), "Full service not successfully exited")
    require(str(suite) in fields["ExecStart"] and SUITE_HASH in fields["ExecStart"], "Wrong service identity")
    return dict(command=command, properties=fields)


def validate_scores(result, classes):
    require(set(result["arm_results"]) == set(ARMS), "Missing score arm")
    for arm in ARMS:
        score = result["arm_results"][arm]
        inter, union = np.asarray(score["intersection"]), np.asarray(score["union"])
        require(inter.shape == union.shape == (classes,) and np.issubdtype(inter.dtype, np.integer)
                and np.issubdtype(union.dtype, np.integer) and np.all(inter >= 0)
                and np.all(union >= inter) and np.any(union > 0), "Invalid aggregate counts")
        value = 100 * math.fsum(int(x) / int(y) for x, y in zip(inter, union) if y) / int((union > 0).sum())
        require(math.isfinite(score["miou_unrounded"])
                and abs(value - score["miou_unrounded"]) < 1e-10, "Pooled mIoU mismatch")


class Evidence(shared.Evidence):
    """Adapt the sealed tuning schema, without mutating shared module globals."""

    def __init__(self, suite):
        self.suite = Path(suite).resolve()
        require(self.suite == SUITE, "Only the authorized tuned suite may supply figure data")
        self.checked, self.protocols, self.seals, self.results = {}, {}, {}, {}
        self.label_hashes, self.selections, self.bindings = {}, {}, {}
        self.gt_reads = {d: set() for d in DATASETS}
        self.prediction_reads = {d: set() for d in DATASETS}
        self.receipt_images = {d: set() for d in DATASETS}
        self.prediction_arm_reads = {d: {a: set() for a in ARMS} for d in DATASETS}
        self.all_seals_checked = False
        self.prepared = self.record(self.suite / "SUITE_PREPARED.json", SUITE_HASH)
        complete = self.record(self.suite / "SUITE_SEALED.json")
        require(complete["complete"] is True and set(complete["datasets"]) == set(DATASETS)
                and complete["arm_results"] == 32 and complete["suite_prepared_sha256"] == SUITE_HASH
                and set(complete["scored_seals"]) == set(DATASETS), "Incomplete tuned suite")
        require(tuple(self.prepared["full_arms"]) == ARMS and self.prepared["model_seed"] == 0
                and set(self.prepared["datasets"]) == set(DATASETS)
                and self.prepared["algorithm"] == "ImageBalancedTRACE:C"
                and self.prepared["method_changes"] is False, "Wrong prepared protocol")
        self.seal_service_flag = complete["service_exit_certified"]
        self.service = service_status(self.suite)
        barrier_path = self.suite / "ALL_FULL_PREDICTIONS_SEALED.json"
        barrier = self.record(barrier_path, complete["all_full_prediction_barrier_sha256"])
        barrier_hash = self.checked[str(barrier_path)]
        require(barrier["complete"] is True and barrier["stage"] == "full"
                and barrier["gt_contents_opened"] is False and barrier["suite_prepared_sha256"] == SUITE_HASH
                and set(barrier["prediction_seal_sha256"]) == {str(self.cell(d)) for d in DATASETS}, "Invalid global barrier")
        selection_path = self.suite / "ALL_CONFIGS_SELECTED.json"
        selections = self.record(selection_path, complete["selected_config_sha256"])
        selection_hash = self.checked[str(selection_path)]
        require(selections["complete"] is True and selections["full_evaluation_used_for_selection"] is False
                and set(selections["selections"]) == set(DATASETS), "Invalid selection barrier")
        parent = self.record(self.suite / "PARENT_IMPORT_SEALED.json", complete["parent_import_sha256"])
        reused = set(self.prepared["continuation"]["reused_datasets"])
        require(parent["complete"] is True and set(parent["imported_selections"]) == reused
                and reused == set(DATASETS) - {"waterbirds_seg"}
                and parent["parent_prepared_sha256"] == self.prepared["continuation"]["parent_prepared_sha256"],
                "Invalid imported selection metadata")
        gate = self.record(self.suite / "gates/waterbirds_seg/GATE_PASSED.json", complete["waterbirds_gate_sha256"])
        require(gate["passed"] is True and gate["gt_contents_opened"] is False
                and gate["suite_prepared_sha256"] == SUITE_HASH, "Invalid Waterbirds gate metadata")
        require(complete["waterbirds_evaluation_scope"] == selections["waterbirds_evaluation_scope"]
                == self.prepared["continuation"]["waterbirds_evaluation_scope"] == "development_overlap",
                "Waterbirds scope changed")
        self.record(self.suite / "frozen_contract.md", self.prepared["contract_sha256"], parse=False)
        self.record(self.suite / "inputs.json", self.prepared["inputs_sha256"], parse=False)
        self.record(self.suite / "full_metrics.csv", complete["metrics_sha256"], parse=False)
        self.record(self.suite / "trials.csv", complete["trials_sha256"], parse=False)
        for dataset in DATASETS:
            cell = self.cell(dataset)
            selected_path = self.suite / "selections" / dataset / "SELECTED_CONFIG_SEALED.json"
            selected_hash = selections["selections"][dataset]
            selected = self.record(selected_path, selected_hash)
            require(selected["dataset"] == dataset and selected["suite_prepared_sha256"] == SUITE_HASH
                    and selected["calibration_only"] is True
                    and selected["full_evaluation_used_for_selection"] is False
                    and selected["config"] == selected["selected_trial"]["config"], "Invalid selected configuration")
            if dataset in reused:
                require(parent["imported_selections"][dataset] == selected_hash
                        and selected["selection_reused_without_search"] is True, "Imported selection differs")
            self.selections[dataset] = selected
            seal_path = cell / "PREDICTIONS_SEALED.json"
            prediction_hash = barrier["prediction_seal_sha256"][str(cell)]
            seal = self.record(seal_path, prediction_hash)
            require(seal["complete"] is True and seal["gt_contents_opened"] is False
                    and seal["images"] == POPULATIONS[dataset] and tuple(seal["arms"]) == ARMS
                    and seal["suite_prepared_sha256"] == SUITE_HASH and seal["dataset"] == dataset
                    and seal["stage"] == "full" and seal["seed"] == 0
                    and seal["calibration"] is False and seal["full_split"] is True
                    and seal["output"] == str(cell), "Invalid prediction seal")
            self.seals[dataset] = seal["artifact_sha256"]
            protocol = self.record(cell / "prediction_protocol.json", self.seals[dataset]["prediction_protocol.json"])
            manifest = protocol["manifest"]
            manifest_path = self.suite / "full" / f"{dataset}.manifest.json"
            require(protocol["manifest_path"] == str(manifest_path), "Wrong manifest path")
            require(self.record(manifest_path, seal["manifest_sha256"]) == manifest
                    and protocol["manifest_sha256"] == seal["manifest_sha256"]
                    and protocol["gt_contents_opened"] is False, "Unbound prediction protocol")
            require(manifest["suite_root"] == str(self.suite) and manifest["suite_prepared_sha256"] == SUITE_HASH
                    and manifest["dataset"] == dataset and manifest["stage"] == "full" and manifest["seed"] == 0
                    and manifest["output"] == str(cell) and tuple(manifest["arms"]) == ARMS
                    and manifest["config"] == selected["config"]
                    and manifest["selected_config_path"] == str(selected_path)
                    and manifest["selected_config_sha256"] == selected_hash
                    and manifest["all_selections_sha256"] == selection_hash
                    and manifest["runtime_root"] == str(self.suite / "runtime")
                    and manifest["source_sha256"] == self.prepared["runtime_hashes"]
                    and manifest["checkpoint"] == self.prepared["checkpoint"], "Manifest config/source identity differs")
            panel = manifest["panel"]
            require(manifest["config"]["resize"] == manifest["config"]["patch_size"] == [224, 224]
                    and len(panel["inputs"]) == len(panel["ordered_ids"]) == panel["images"] == POPULATIONS[dataset]
                    and panel["full_split"] is True
                    and panel["input_identity_sha256"] == canonical_hash(panel["inputs"])
                    and panel["ordered_ids"] == [r["official_id"] for r in panel["inputs"]], "Wrong frozen panel/order")
            if dataset == "waterbirds_seg":
                require(panel["evaluation_scope"] == "development_overlap" and panel["development_overlap"] is True
                        and panel["independent_test_claim"] is False, "Waterbirds overlap not disclosed")
            self.protocols[dataset] = manifest
            # Inspect the sealed inventory only. Do not open or stat all payloads.
            batch = manifest["config"]["batch_size"]
            episodes = (POPULATIONS[dataset] + batch - 1) // batch
            expected_names = {f"predictions/{a}/{i:06d}.npy" for a in ARMS for i in range(POPULATIONS[dataset])}
            expected_names.update(f"episodes/{i:06d}.{a}.json" for a in ARMS for i in range(episodes))
            expected_names.update(("prediction_protocol.json", "initial_state.json", "prediction_result.json", "cost.jsonl"))
            require(set(self.seals[dataset]) == expected_names, "Prediction inventory differs")
            score = self.record(cell / "SCORED_SEALED.json", complete["scored_seals"][dataset])
            require(score["complete"] is True and score["prediction_seal_sha256"] == prediction_hash
                    and score["barrier_sha256"] == barrier_hash, "Unbound score seal")
            started = self.record(cell / "SCORING_STARTED.json", score["artifact_sha256"]["SCORING_STARTED.json"])
            labels = self.record(cell / "score_inputs.json", score["artifact_sha256"]["score_inputs.json"])
            result = self.record(cell / "result.json", score["artifact_sha256"]["result.json"])
            for payload in (started, labels, result):
                require(payload["manifest_sha256"] == seal["manifest_sha256"]
                        and payload["barrier_sha256"] == barrier_hash, "Score manifest/barrier mismatch")
            require(started["prediction_seal_sha256"] == result["prediction_seal_sha256"] == prediction_hash
                    and result["status"] == "completed" and result["images"] == POPULATIONS[dataset]
                    and result["dataset"] == dataset and result["stage"] == "full" and result["seed"] == 0
                    and result["calibration"] is False and result["full_split"] is True
                    and result["config"] == selected["config"]
                    and result["label_contents_first_opened_after_stage_predictions_sealed"] is True,
                    "Invalid canonical score scope")
            require(set(labels["label_sha256"]) == {r["label"] for r in panel["inputs"]}, "Label inventory differs")
            validate_scores(result, len(panel["classes"]))
            self.label_hashes[dataset] = labels["label_sha256"]
            self.results[dataset] = result
            self.bindings[dataset] = dict(selection_sha256=selected_hash, manifest_sha256=seal["manifest_sha256"],
                prediction_seal_sha256=prediction_hash, scored_seal_sha256=complete["scored_seals"][dataset],
                source_sha256=manifest["source_sha256"], checkpoint=manifest["checkpoint"], config=manifest["config"],
                result_sha256=score["artifact_sha256"]["result.json"], barrier_sha256=barrier_hash)
        # Validate frozen source files, but never load model/checkpoint payloads.
        for relative, digest in self.prepared["runtime_hashes"].items():
            self.record(self.suite / "runtime" / relative, digest, parse=False)
        for relative in ("tools/draw_image_full_figures.py", "trace_ovss/data/datasets.py",
                         "trace_ovss/method/trace/interventions.py", "trace_ovss/method/trace/constants.py"):
            self.record(ROOT / relative, self.prepared["runtime_hashes"][relative], parse=False)
        self.validate_csv()
        self.all_seals_checked = True

    def cell(self, dataset):
        return self.suite / "full" / dataset

    def bounded(self, dataset, index):
        require(self.all_seals_checked, "GT/prediction diagnostics before global eight-seal check")
        require(dataset in DATASETS and type(index) is int and 0 <= index < min(LIMIT, POPULATIONS[dataset]),
                "Diagnostic bound exceeded")

    def prediction(self, dataset, index, arm):
        require(arm in ARMS, "Unknown prediction arm")
        array = super().prediction(dataset, index, arm)
        self.prediction_arm_reads[dataset][arm].add(index)
        return array

    def validate_csv(self):
        with (self.suite / "full_metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 32 and {(r["dataset"], r["arm"]) for r in rows}
                == {(d, a) for d in DATASETS for a in ARMS}, "Incomplete full metrics table")
        for row in rows:
            dataset, arm = row["dataset"], row["arm"]
            require(float(row["mIoU"]) == self.results[dataset]["arm_results"][arm]["miou_unrounded"]
                    and int(row["images"]) == POPULATIONS[dataset]
                    and json.loads(row["config"]) == self.selections[dataset]["config"]
                    and row["result_path"] == str(self.cell(dataset) / "result.json")
                    and row["independent_test_claim"] == "False", "CSV/canonical result mismatch")

    def receipts(self, dataset):
        manifest = self.protocols[dataset]
        batch, panel = manifest["config"]["batch_size"], manifest["panel"]
        for episode in range((min(LIMIT, POPULATIONS[dataset]) + batch - 1) // batch):
            self.bounded(dataset, episode * batch)
            relative = f"episodes/{episode:06d}.image_C.json"
            path = self.cell(dataset) / relative
            payload = self.record(path, self.seals[dataset][relative])
            start = episode * batch
            require(payload["arm"] == "image_C" and payload["episode"] == episode
                    and payload["dataset"] == dataset and payload["stage"] == "full"
                    and payload["gt_contents_opened"] is False
                    and payload["ordered_ids"] == panel["ordered_ids"][start:start + batch]
                    and payload["diagnostic_forward_inputs"] == payload["candidate_count"] == len(payload["candidates"]),
                    "Invalid measured C receipt")
            for candidate_index, row in enumerate(payload["candidates"]):
                local = row["image_idx"]
                require(type(local) is int and 0 <= local < payload["images"], "Invalid receipt local index")
                index = start + local
                if index >= LIMIT:
                    continue
                self.bounded(dataset, index)
                self.receipt_images[dataset].add(index)
                require(payload["ordered_ids"][local] == panel["inputs"][index]["official_id"], "Image receipt mismatch")
                validate_candidate(row, panel)
                cls = shared.category(panel, row["class_idx"])
                yield dict(dataset=dataset, image_index=index, official_id=panel["ordered_ids"][index],
                    episode=episode, candidate_index=candidate_index, receipt=str(path), receipt_sha256=self.seals[dataset][relative],
                    category_id=cls, category_name=panel["classes"][cls], query=panel["queries"][row["class_idx"]], receipt_row=row)


def validate_candidate(row, panel):
    cls = shared.category(panel, row["class_idx"])
    mask = shared.decode_region(row["region"])
    require(row["batch_idx"] == row["image_idx"] and row["target_support"] is None
            and row["residual_support"] is None and row["dominance"] is None
            and row["diagnostic_algorithm"] == "C" and row["omitted"] is False, "Non-C receipt")
    require(cls == row["preupdate_candidate_category_idx"] and int(mask.sum()) == row["area"]
            and row["class_name"] == panel["queries"][row["class_idx"]], "Candidate identity mismatch")
    require(row["reliability"] == row["quality"] == row["counterfactual_support"]
            and math.isfinite(row["confidence"]) and 0 <= row["confidence"] <= 1
            and math.isfinite(row["counterfactual_support"]) and 0 <= row["counterfactual_support"] <= 1,
            "Unmeasured/nonfinite support")


def signed_extremum(choices, dataset):
    if not choices:
        return dict(dataset=dataset, selected_case=None, extremum=None, sign_requirement_met=False,
                    null_reason="No eligible largest non-background GT class with >=128 valid pixels")
    negative = dataset == "voc21"
    case = min(choices, key=lambda c: ((1 if negative else -1) * c["delta"], c["image_index"]))
    passed = case["delta"] < 0 if negative else case["delta"] > 0
    return dict(dataset=dataset, selected_case=case if passed else None, extremum=case,
                sign_requirement_met=passed, null_reason=None if passed else "Predeclared strict sign not met")


def select_segmentation(evidence):
    selected, counts = [], {}
    for dataset in ("coco_cd", "coco_object", "pc60", "voc21"):
        panel, choices = evidence.protocols[dataset]["panel"], []
        for index in range(min(LIMIT, POPULATIONS[dataset])):
            gt = evidence.gt(dataset, index)
            classes, areas = np.unique(gt[gt != 255], return_counts=True)
            eligible = [(int(c), int(n)) for c, n in zip(classes, areas)
                        if n >= 128 and panel["classes"][c] != "background"]
            if not eligible:
                continue
            cls, area = min(eligible, key=lambda pair: (-pair[1], pair[0]))
            a = shared.named_class_iou(evidence.prediction(dataset, index, "image_confidence"), gt, cls)
            b = shared.named_class_iou(evidence.prediction(dataset, index, "image_C"), gt, cls)
            choices.append(dict(dataset=dataset, image_index=index, category_id=cls, category_name=panel["classes"][cls],
                gt_area=area, confidence=a, trace=b, delta=b["iou"] - a["iou"], negative=dataset == "voc21"))
        selected.append(signed_extremum(choices, dataset))
        counts[dataset] = len(choices)
    return selected, counts


def draw_segmentation(evidence, selections, output):
    if all(s["selected_case"] is not None for s in selections):
        return shared.draw_segmentation(evidence, [s["selected_case"] for s in selections], output)
    # The usual renderer is unchanged; reserve empty rows instead of choosing replacements.
    fig = plt.figure(figsize=(shared.WIDTH, 7.25))
    for x, label in zip((.14, .385, .63, .875), ("Input", "Confidence", "TRACE", "GT")):
        fig.text(x, 7.07 / 7.25, label, ha="center", fontsize=10, weight="bold")
    for row, selection in enumerate(selections):
        case = selection["selected_case"]
        title_y = 6.8 - row * 1.67
        if case is None:
            fig.text(.03, title_y / 7.25, f"{DISPLAY[selection['dataset']]} / no qualifying signed case", fontsize=9)
            fig.text(.5, (title_y - .75) / 7.25, selection["null_reason"], ha="center", fontsize=9)
            continue
        dataset, index, cls = case["dataset"], case["image_index"], case["category_id"]
        rgb, gt = evidence.photograph(dataset, index), evidence.gt(dataset, index)
        a, b = (evidence.prediction(dataset, index, arm) for arm in ("image_confidence", "image_C"))
        suffix = ", negative" if case["negative"] else ""
        fig.text(.03, title_y / 7.25, f"{DISPLAY[dataset]} / {case['category_name']} | Class IoU: "
            f"{case['confidence']['iou']:.1f} -> {case['trace']['iou']:.1f} ({case['delta']:+.1f} pp{suffix})",
            fontsize=9, color=shared.RED if case["negative"] else "#23343a")
        arrays = (rgb, shared.overlay(rgb, a, cls, gt != 255), shared.overlay(rgb, b, cls, gt != 255),
                  shared.overlay(rgb, gt, cls, gt != 255))
        for center, array in zip((.14, .385, .63, .875), arrays):
            shared.photo_axis(fig, [center - .7 / shared.WIDTH, (title_y - 1.5) / 7.25, 1.4 / shared.WIDTH, 1.4 / 7.25], array)
        prefix = f"image_full_segmentation_case{row + 1}"
        shared.Image.fromarray(rgb).save(output / f"{prefix}_input.png")
        for label, array in (("confidence", a), ("trace", b), ("gt", gt)):
            np.save(output / f"{prefix}_{label}.npy", array, allow_pickle=False)
    fig.text(.5, .055 / 7.25, "Class IoU: Confidence -> TRACE. Cyan: named class; gray: GT void.", ha="center", fontsize=9)
    return shared.save_figure(fig, output, "image_full_segmentation")


def draw_pipeline(output):
    try:
        return shared.draw_pipeline(output)
    except ValueError as error:
        if not str(error).startswith("Overlapping figure text:"):
            raise
        # The imported authoring function leaves its figure open on validation
        # failure. Increase spacing only; every formula and font stays intact.
        fig = plt.gcf()
        fig.set_size_inches(shared.WIDTH, 6.0)
        return shared.save_figure(fig, output, "image_full_pipeline")


def allocation_rows(evidence):
    rows = []
    for dataset in DATASETS:
        result = evidence.results[dataset]
        values = {a: result["arm_results"][a]["miou_unrounded"] for a in ARMS}
        rows.append(dict(dataset=dataset, images=result["images"], **values,
            C_minus_no_adapt=values["image_C"] - values["no_adapt"],
            C_minus_uniform=values["image_C"] - values["image_uniform"],
            C_minus_confidence=values["image_C"] - values["image_confidence"]))
    return rows


def draw_allocation(rows, output):
    fig = plt.figure(figsize=(shared.SINGLE_WIDTH, 4.1))
    fig.text(.5, .97, "Regional allocation gains", ha="center", va="top", fontsize=10, weight="bold")
    ax = fig.add_axes([.355, .19, .59, .62])
    ax.axvline(0, color="#7f8588", linewidth=.6)
    colors, controls = ("#666666", "#846398", shared.CYAN), ("no_adapt", "uniform", "confidence")
    for offset, color, control in zip((-.17, 0, .17), colors, controls):
        ax.scatter([r[f"C_minus_{control}"] for r in rows], np.arange(len(rows)) + offset,
                   s=18, color=color, marker="o", zorder=3)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[DISPLAY[r["dataset"]] for r in rows],
           ylim=(len(rows) - .5, -.5))
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=9, pad=3)
    ax.xaxis.set_major_locator(shared.matplotlib.ticker.MaxNLocator(nbins=4))
    ax.grid(axis="x", color="#dedede", linewidth=.6)
    for edge in ("top", "right", "left"):
        ax.spines[edge].set_visible(False)
    for x, color, label in zip((.035, .35, .66), colors, ("No Adapt", "Uniform", "Confidence")):
        fig.text(x, .875, "o", color=color, va="center", fontsize=9)
        fig.text(x + .035, .875, label, va="center", fontsize=9)
    fig.text(.64, .087, "TRACE - control (mIoU pp)", ha="center", fontsize=9)
    fig.text(.5, .025, "*Waterbirds: development overlap.", ha="center", fontsize=9)
    return shared.save_figure(fig, output, "image_full_allocation")


def enrich_case(evidence, case):
    dataset, index = case["dataset"], case["image_index"]
    entry = evidence.protocols[dataset]["panel"]["inputs"][index]
    case.update(official_id=entry["official_id"], image_path=entry["image"], image_sha256=entry["image_sha256"],
                source_binding=evidence.bindings[dataset])
    if "receipt_row" not in case:
        case["label_sha256"] = evidence.label_hashes[dataset][entry["label"]]
        case["prediction_sha256"] = {a: evidence.seals[dataset][f"predictions/{a}/{index:06d}.npy"]
                                     for a in ("image_confidence", "image_C")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output == OUTPUT, "Write only the owned tuned asset directory")
    torch.set_num_threads(1)
    shared.configure_style()
    evidence = Evidence(args.suite)
    print("All eight selection/prediction/score metadata chains verified; bounded CPU selection starts.", flush=True)
    teaser, teaser_counts = shared.select_teaser(evidence)
    segmentation, segmentation_counts = select_segmentation(evidence)
    for case in teaser:
        enrich_case(evidence, case)
    for selection in segmentation:
        if selection["extremum"] is not None:
            enrich_case(evidence, selection["extremum"])
    output.mkdir(parents=True, exist_ok=True)
    layouts = {}
    layouts["image_full_teaser"], reconstruction = shared.draw_teaser(evidence, teaser, output)
    layouts["image_full_pipeline"] = draw_pipeline(output)
    layouts["image_full_segmentation"] = draw_segmentation(evidence, segmentation, output)
    rows = allocation_rows(evidence)
    layouts["image_full_allocation"] = draw_allocation(rows, output)
    provenance = dict(schema_version=1, suite=str(evidence.suite), suite_prepared_sha256=SUITE_HASH,
        invocation=[sys.executable, "-B", "tools/draw_tuned_full_figures.py", "--suite", str(evidence.suite), "--output", str(output)],
        script_sha256=sha(__file__), shared_script_sha256=sha(shared.__file__),
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], model_forwards=0, support_remeasurements=0,
        selection_rule=SELECTION_RULE, eligible_counts=dict(teaser=teaser_counts, segmentation=segmentation_counts),
        teaser=teaser, segmentation=segmentation, reconstruction=reconstruction, allocation=rows,
        layout=layouts, input_artifact_sha256=evidence.checked, source_bindings=evidence.bindings,
        service=evidence.service, suite_seal_service_exit_certified=evidence.seal_service_flag,
        service_note="The suite seal flag is false by construction. Service exited/success is independently verified read-only; no seal modified.",
        verification_scope="All eight selection, prediction inventory, protocol, scoring, label-hash and barrier metadata chains; all 32 pooled scores recomputed from canonical aggregate counts and cross-checked with full_metrics.csv. Frozen runtime hashes verified. This is not a fresh hash/score audit of all prediction payloads.",
        bounded_payload_scope="Only first400 C receipt episodes for PC60/Object, first400 GT for CD/Object/PC60/VOC21, and eligible first400 Confidence/C maps. Every opened RGB/GT/prediction/receipt payload hash checked against its sealed identity before decoding.",
        bounds_note="A final boundary episode may contain indices >=400; they are ignored. No per_image counts, scores_by_episode.jsonl, cost.jsonl, checkpoint, full prediction scan, GPU query or model forward is loaded/executed.",
        pipeline_note="shared.draw_pipeline imported and called unchanged, including all mathematics; source hash matches the prepared runtime snapshot. If its original 4.5in canvas fails text overlap validation, the still-open figure is re-exported at 6.0in height, changing spacing only.",
        normalization=dict(RGB_scale="0..255 float32", mean=shared.CLIP_MEAN.flatten().tolist(), std=shared.CLIP_STD.flatten().tolist(),
            implementation="shared.context_view using the hash-matched released CPU intervention helper; nearest native-mask upsampling; candidate pixels exact in normalized space; no remeasured support"),
        gt_conventions="224x224 nearest; ignore255; reduce-zero only as frozen panel specifies; single named largest nonbackground GT class, not whole-image mIoU; saved category predictions only.",
        claim_boundary="Post-seal GT-selected segmentation extrema are not representative or used for aggregate scores/config selection. Development-exposed protocols; Waterbirds development_overlap is not an independent test. All displayed experimental outputs come only from the authorized tuned suite.",
        diagnostic_reads={d: dict(gt_indices=sorted(evidence.gt_reads[d]), prediction_indices=sorted(evidence.prediction_reads[d]),
            receipt_image_indices=sorted(evidence.receipt_images[d]),
            prediction_arm_indices={a: sorted(v) for a, v in evidence.prediction_arm_reads[d].items()}) for d in DATASETS})
    provenance["output_sha256"] = {p.name: sha(p) for p in sorted(output.glob("image_full_*")) if p.suffix in (".pdf", ".png", ".npy")}
    shared.write_json(output / "image_full_provenance.json", provenance)
    fields = ("dataset", "image_index", "official_id", "category_name", "confidence", "trace", "delta")
    print(json.dumps(dict(teaser=[{k: c[k] for k in ("dataset", "image_index", "official_id", "category_name", "receipt_row")} for c in teaser],
        segmentation=[dict(dataset=s["dataset"], sign_requirement_met=s["sign_requirement_met"],
            selected_case={k: s["selected_case"][k] for k in fields} if s["selected_case"] else None,
            null_reason=s["null_reason"]) for s in segmentation], allocation=rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
