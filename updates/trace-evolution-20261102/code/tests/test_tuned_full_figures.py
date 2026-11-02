"""CPU-only contracts for tuned figures, fixed selection and bounded provenance."""

import ast
import copy
import inspect
import subprocess

import numpy as np
import pytest

from tools import draw_tuned_full_figures as figures


def test_eight_seals_gate_all_diagnostic_access():
    evidence = object.__new__(figures.Evidence)
    evidence.all_seals_checked = False
    with pytest.raises(ValueError, match="global eight-seal"):
        evidence.bounded("pc60", 0)
    evidence.all_seals_checked = True
    for dataset in figures.DATASETS:
        evidence.bounded(dataset, 399)
    for index in (-1, 400, 5000, True, 1.0):
        with pytest.raises(ValueError, match="bound exceeded"):
            evidence.bounded("pc60", index)
    with pytest.raises(ValueError, match="bound exceeded"):
        evidence.bounded("unknown", 0)


def test_aggregate_scores_reject_corruption_and_missing_arm():
    result = dict(arm_results={a: dict(intersection=[1, 2], union=[2, 4], miou_unrounded=50.0)
                               for a in figures.ARMS})
    figures.validate_scores(result, 2)
    bad = copy.deepcopy(result)
    bad["arm_results"]["no_adapt"]["miou_unrounded"] += .01
    with pytest.raises(ValueError, match="mIoU mismatch"):
        figures.validate_scores(bad, 2)
    bad = copy.deepcopy(result)
    bad["arm_results"].pop("image_uniform")
    with pytest.raises(ValueError, match="Missing score arm"):
        figures.validate_scores(bad, 2)


@pytest.mark.parametrize("dataset,deltas,expected", [
    ("coco_cd", [1, 3, 3], 1), ("coco_object", [0, -1], None),
    ("pc60", [0, 0], None), ("voc21", [1, -2, -2], 1), ("voc21", [0, 2], None),
])
def test_signed_extrema_keep_nulls_and_lowest_index_ties(dataset, deltas, expected):
    choices = [dict(image_index=i, delta=d) for i, d in enumerate(deltas)]
    selection = figures.signed_extremum(list(reversed(choices)), dataset)
    assert selection["sign_requirement_met"] == (expected is not None)
    if expected is None:
        assert selection["selected_case"] is None
        assert selection["extremum"] in choices
        assert selection["null_reason"]
    else:
        assert selection["selected_case"]["image_index"] == expected
    assert figures.signed_extremum([], dataset)["extremum"] is None


def test_teaser_uses_unchanged_gt_free_thresholds_and_tiebreaks():
    def case(index, support, confidence=.95, area=16, name="target", candidate=0):
        return dict(image_index=index, candidate_index=candidate, category_name=name,
                    receipt_row=dict(counterfactual_support=support, confidence=confidence, area=area))

    rows = [case(1, .7, candidate=2), case(1, .7, candidate=1), case(0, .1),
            case(2, .99, confidence=.89999), case(3, .001, name="background"),
            case(4, .001, area=15), case(5, .99, area=193), case(6, .99, area=192, confidence=.9)]

    class Fake:
        def receipts(self, dataset):
            return iter(reversed(rows))

        def gt(self, *args):
            pytest.fail("Teaser must never use GT")

    selections, counts = figures.shared.select_teaser(Fake())
    assert selections[0]["image_index"] == 6
    assert selections[1]["image_index"] == 0
    assert counts == dict(pc60=4, coco_object=4)
    tied, _ = figures.shared.select_teaser(type("Fake", (), {"receipts": lambda self, dataset: iter(rows[:2])})())
    assert all(c["candidate_index"] == 1 for c in tied)


def test_segmentation_names_largest_nonbackground_class_before_iou(monkeypatch):
    monkeypatch.setattr(figures, "LIMIT", 2)
    gt = np.zeros((224, 224), dtype=np.int64)
    gt[:2] = 1
    gt[2:4] = 2
    gt[4:5] = 255
    read_indices = []

    class Fake:
        protocols = {d: dict(panel=dict(classes=["background", "one", "two"])) for d in figures.DATASETS}

        def gt(self, dataset, index):
            read_indices.append(index)
            return gt

        def prediction(self, dataset, index, arm):
            # Class 1 and 2 tie in GT area. Lowest category wins even though
            # background is largest and class 2 might tell a different story.
            pred = np.zeros_like(gt)
            if (arm == "image_C") == (dataset != "voc21"):
                pred[:(1 if index == 0 else 2)] = 1
            return pred

    selections, counts = figures.select_segmentation(Fake())
    assert all(s["selected_case"]["image_index"] == 1 for s in selections)
    assert all(s["selected_case"]["category_id"] == 1 for s in selections)
    assert all(s["selected_case"]["gt_area"] == 448 for s in selections)
    assert set(counts.values()) == {2}
    assert max(read_indices) == 1


def test_null_segmentation_renders_without_opening_any_payload(monkeypatch, tmp_path):
    class Fake:
        def photograph(self, *args):
            pytest.fail("Null case must not open an image or substitute another case")

    selections = [figures.signed_extremum([dict(image_index=0, delta=0)], d)
                  for d in ("coco_cd", "coco_object", "pc60", "voc21")]
    captured = []
    monkeypatch.setattr(figures.shared, "save_figure", lambda fig, output, name: captured.append(fig) or name)
    assert figures.draw_segmentation(Fake(), selections, tmp_path) == "image_full_segmentation"
    assert len(captured[0].axes) == 0
    figures.plt.close(captured[0])


@pytest.fixture(scope="module")
def real_evidence():
    if not (figures.SUITE / "SUITE_SEALED.json").exists():
        pytest.skip("Authorized suite not mounted")
    return figures.Evidence(figures.SUITE)


def test_all_eight_metadata_chains_and_32_scores_without_payload_reads(real_evidence):
    evidence = real_evidence
    assert evidence.all_seals_checked
    assert len(evidence.results) == len(evidence.bindings) == 8
    assert all(not indices for indices in evidence.gt_reads.values())
    assert all(not indices for indices in evidence.prediction_reads.values())
    assert not any("/predictions/" in name or name.endswith("scores_by_episode.jsonl") for name in evidence.checked)
    assert len(figures.allocation_rows(evidence)) == 8


@pytest.mark.parametrize("filename,key,value,message", [
    ("ALL_CONFIGS_SELECTED.json", "full_evaluation_used_for_selection", True, "selection barrier"),
    ("ALL_FULL_PREDICTIONS_SEALED.json", "gt_contents_opened", True, "global barrier"),
    ("full/waterbirds_seg/result.json", "barrier_sha256", "0" * 64, "manifest/barrier"),
    ("selections/pc60/SELECTED_CONFIG_SEALED.json", "calibration_only", False, "selected configuration"),
])
def test_metadata_mismatch_stops_before_any_gt_or_prediction(monkeypatch, filename, key, value, message):
    if not figures.SUITE.exists():
        pytest.skip("Authorized suite not mounted")
    original = figures.Evidence.record

    def altered(self, path, *args, **kwargs):
        result = original(self, path, *args, **kwargs)
        if str(path) == str(figures.SUITE / filename):
            result[key] = value
        return result

    monkeypatch.setattr(figures.Evidence, "record", altered)
    monkeypatch.setattr(figures.shared.Image, "open", lambda *a, **k: pytest.fail("GT/image read before validation"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Prediction read before validation"))
    with pytest.raises(ValueError, match=message):
        figures.Evidence(figures.SUITE)


def test_real_assets_hashes_bounds_scores_and_receipts(real_evidence):
    provenance_path = figures.OUTPUT / "image_full_provenance.json"
    if not provenance_path.exists():
        pytest.skip("Generate tuned assets first")
    p = figures.shared.read(provenance_path)
    assert p["suite_prepared_sha256"] == figures.SUITE_HASH
    assert p["script_sha256"] == figures.sha(figures.__file__)
    assert p["shared_script_sha256"] == figures.sha(figures.shared.__file__)
    assert p["model_forwards"] == p["support_remeasurements"] == 0
    assert p["cuda_visible_devices"] == ""
    assert p["selection_rule"]["teaser"] == figures.shared.SELECTION_RULE["teaser"]
    assert p["selection_rule"]["segmentation"] == figures.shared.SELECTION_RULE["segmentation"]
    assert p["allocation"] == figures.allocation_rows(real_evidence)
    assert len(p["allocation"]) == 8 and len(p["segmentation"]) == 4 and len(p["teaser"]) == 2
    for dataset, reads in p["diagnostic_reads"].items():
        for key in ("gt_indices", "prediction_indices", "receipt_image_indices"):
            assert all(0 <= i < 400 for i in reads[key])
        assert reads["prediction_arm_indices"]["no_adapt"] == []
        assert reads["prediction_arm_indices"]["image_uniform"] == []
        if dataset not in ("pc60", "coco_object", "coco_cd", "voc21"):
            assert reads["gt_indices"] == reads["prediction_indices"] == []
    for case in p["teaser"]:
        payload = figures.shared.read(case["receipt"])
        assert figures.sha(case["receipt"]) == case["receipt_sha256"]
        row = payload["candidates"][case["candidate_index"]]
        assert row == case["receipt_row"]
        assert payload["ordered_ids"][row["image_idx"]] == case["official_id"]
        active = [r for r in payload["candidates"] if r["reliability"] > figures.shared.EPS]
        denominator = sum(r["reliability"] for r in active if r["image_idx"] == row["image_idx"])
        expected = row["reliability"] / denominator / len({r["image_idx"] for r in active})
        assert row["objective_coefficient"] == pytest.approx(expected, abs=1e-12)
    for selection in p["segmentation"]:
        case = selection["selected_case"]
        if case is None:
            assert not selection["sign_requirement_met"]
            continue
        assert case == selection["extremum"]
        assert case["delta"] < 0 if case["negative"] else case["delta"] > 0
        assert case["delta"] == case["trace"]["iou"] - case["confidence"]["iou"]
        for label, arm in (("confidence", "image_confidence"), ("trace", "image_C")):
            counts = case[label]
            assert counts["iou"] == 100 * counts["intersection"] / counts["union"]
            assert case["prediction_sha256"][arm] == real_evidence.seals[case["dataset"]][f"predictions/{arm}/{case['image_index']:06d}.npy"]
    for name, digest in p["output_sha256"].items():
        assert figures.sha(figures.OUTPUT / name) == digest
    for layout in p["layout"].values():
        assert layout["minimum_text_pt"] >= 9
        assert not layout["external_crop"]
        assert "pairwise" in layout["text_overlap_check"]
    assert len(p["layout"]["image_full_pipeline"]["math"]) == 5
    assert all(m["minimum_glyph_pt"] >= 9 for m in p["layout"]["image_full_pipeline"]["math"])
    assert figures.shared.SUITE_HASH != figures.SUITE_HASH
    assert len(figures.shared.DATASETS) == 6


def test_pdf_fonts_annotations_and_all_four_arm_scores():
    path = figures.OUTPUT / "image_full_provenance.json"
    if not path.exists():
        pytest.skip("Generate tuned assets first")
    p = figures.shared.read(path)
    for name in ("teaser", "pipeline", "segmentation", "allocation"):
        pdf = figures.OUTPUT / f"image_full_{name}.pdf"
        report = subprocess.run(["pdffonts", str(pdf)], check=True, text=True, capture_output=True).stdout
        assert "CID" not in report and "Type 3" not in report and "TrueType" in report
        for line in report.splitlines()[2:]:
            assert "yes" in line
        exported = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, text=True, capture_output=True).stdout
        exported = " ".join(exported.split())
        if name == "allocation":
            assert "development overlap" in exported
            assert "Regional allocation gains" in exported
            assert all(label in exported for label in ("No Adapt", "Uniform", "Confidence"))
            for row in p["allocation"]:
                assert figures.DISPLAY[row["dataset"]] in exported
                for control in ("no_adapt", "uniform", "confidence"):
                    arm = dict(no_adapt="no_adapt", uniform="image_uniform", confidence="image_confidence")[control]
                    assert row[f"C_minus_{control}"] == row["image_C"] - row[arm]
        elif name == "teaser":
            for case in p["teaser"]:
                for title, key in (("Confidence", "confidence"), ("Probe support", "counterfactual_support")):
                    assert f"{title}: {case['receipt_row'][key]:.3f}" in exported
        elif name == "segmentation":
            for selection in p["segmentation"]:
                case = selection["selected_case"]
                if case:
                    assert f"Class IoU: {case['confidence']['iou']:.1f} -> {case['trace']['iou']:.1f}" in exported


def test_companion_arrays_are_exact_new_suite_predictions_and_cpu_reconstruction(monkeypatch):
    path = figures.OUTPUT / "image_full_provenance.json"
    if not path.exists():
        pytest.skip("Generate tuned assets first")
    p = figures.shared.read(path)
    monkeypatch.setattr(figures.torch.cuda, "is_available", lambda: pytest.fail("No GPU query allowed"))
    for i, selection in enumerate(p["segmentation"], 1):
        case = selection["selected_case"]
        if case:
            for label, arm in (("confidence", "image_confidence"), ("trace", "image_C")):
                assert figures.sha(figures.OUTPUT / f"image_full_segmentation_case{i}_{label}.npy") == case["prediction_sha256"][arm]
    for i, case in enumerate(p["teaser"], 1):
        prefix = figures.OUTPUT / f"image_full_teaser_case{i}"
        with figures.shared.Image.open(str(prefix) + "_input.png") as image:
            rgb = np.asarray(image).copy()
        with figures.shared.Image.open(str(prefix) + "_context.png") as image:
            rounded = np.asarray(image).copy()
        expected, mask = figures.shared.context_view(rgb, figures.shared.decode_region(case["receipt_row"]["region"]))
        np.testing.assert_array_equal(np.load(str(prefix) + "_context_float32.npy", allow_pickle=False), expected)
        np.testing.assert_array_equal(rounded[mask], rgb[mask])
        assert case["receipt_row"]["counterfactual_support"] == case["receipt_row"]["reliability"]


def test_pipeline_math_is_identical_to_imported_static_source():
    path = figures.OUTPUT / "image_full_provenance.json"
    if not path.exists():
        pytest.skip("Generate tuned assets first")
    p = figures.shared.read(path)
    tree = ast.parse(inspect.getsource(figures.shared.draw_pipeline))
    source_math = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 1 and n.value.startswith("$")]
    exported_math = [m["source"] for m in p["layout"]["image_full_pipeline"]["math"]]
    assert sorted(exported_math) == sorted(source_math)
