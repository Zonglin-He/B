"""CPU contracts for selected-C scientific figure values and provenance."""

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from tools import draw_image_full_figures as figures


def packed(mask):
    return dict(shape=list(mask.shape), encoding="packbits-little-hex",
                data=np.packbits(mask.ravel(), bitorder="little").tobytes().hex(), true_pixels=int(mask.sum()))


@pytest.mark.parametrize("shape", [(14, 14), (16, 16), (3, 5)])
def test_native_mask_roundtrip_shape_indices_and_bool(shape):
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[[0, 5, mask.size - 1]] = True
    actual = figures.decode_region(packed(mask))
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, mask)
    assert np.flatnonzero(actual).tolist() == [0, 5, mask.size - 1]


def test_mask_corruption_is_not_silently_repaired():
    mask = np.ones((16, 16), dtype=bool)
    item = packed(mask)
    item["true_pixels"] -= 1
    with pytest.raises(ValueError, match="count mismatch"):
        figures.decode_region(item)
    item = packed(np.ones((3, 5), dtype=bool))
    item["data"] = "ffff"
    with pytest.raises(ValueError, match="padding"):
        figures.decode_region(item)


def test_counterfactual_matches_exact_cpu_formula_and_preserves_candidate(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("No GPU query allowed"))
    rgb = np.random.default_rng(9).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:8, 4:11] = True
    actual, full = figures.context_view(rgb, mask)
    assert full.shape == (224, 224)
    assert full.sum() == mask.sum() * 14 * 14
    pixels = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None]
    mean, std = figures.CLIP_MEAN[None], figures.CLIP_STD[None]
    normalized = (pixels - mean) / std
    recovered = normalized * std + mean
    spatial = recovered.mean((2, 3), keepdim=True)
    background = (.35 * (recovered - spatial) + spatial)[:, [1, 2, 0]].clamp(0, 255)
    m = torch.from_numpy(full.astype(np.float32))[None, None]
    expected = ((normalized * m + ((background - mean) / std) * (1 - m)) * std + mean)[0].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual[full], rgb[full], atol=3.1e-5, rtol=0)
    assert np.mean(abs(actual[~full] - rgb[~full])) > 10


def test_label_conventions_and_named_class_iou_not_whole_image_miou():
    raw = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(figures.prepare_gt(raw, True), [[255, 0, 1, 255]])
    np.testing.assert_array_equal(figures.prepare_gt(raw, False), raw)
    gt = np.zeros((224, 224), dtype=np.int64)
    pred = gt.copy()
    gt[:20] = 1
    pred[:10] = 1
    gt[20:30] = 255
    pred[20:30] = 1
    assert figures.named_class_iou(pred, gt, 1) == dict(intersection=2240, union=4480, iou=50.0)


def test_query_category_max_is_not_query_sum():
    probabilities = np.array([[[.31]], [[.30]], [[.39]]])
    result = figures.collapse_query_max(probabilities, [0, 0, 1])
    assert result[:, 0, 0].tolist() == [.31, .39]
    assert result.argmax(0).item() == 1
    panel = dict(queries=["car", "automobile", "dog"], query_to_class=[0, 0, 1])
    assert figures.category(panel, 1) == 0
    assert figures.category(panel, 2) == 1


def test_all_six_seals_and_bound_required_before_gt():
    evidence = object.__new__(figures.Evidence)
    evidence.all_seals_checked = False
    with pytest.raises(ValueError, match="global six-seal"):
        evidence.bounded("pc60", 0)
    evidence.all_seals_checked = True
    for index in (-1, 400, 5000):
        with pytest.raises(ValueError, match="bound exceeded"):
            evidence.bounded("pc60", index)
    evidence.bounded("pc60", 399)


def test_selection_rule_exposes_gt_extrema_and_negative():
    rules = figures.SELECTION_RULE
    assert "minimum in VOC21" in rules["segmentation"]
    assert "No GT used" in rules["teaser"]
    assert "not representative" in rules["warning"]


def test_final_assets_provenance_annotations_and_scales():
    path = figures.ROOT / "paper_evolution/figure/image_full_provenance.json"
    if not path.exists():
        pytest.skip("Run authoring script to check real sealed assets")
    provenance = json.loads(path.read_text())
    assert provenance["suite_prepared_sha256"] == figures.SUITE_HASH
    assert provenance["model_forwards"] == 0 and provenance["cuda_visible_devices"] == ""
    assert len(provenance["allocation"]) == 6
    assert len(provenance["teaser"]) == 2 and len(provenance["segmentation"]) == 4
    assert sum(c["negative"] for c in provenance["segmentation"]) == 1
    assert all(c["delta"] < 0 if c["negative"] else c["delta"] > 0 for c in provenance["segmentation"])
    for case in provenance["teaser"]:
        rec = case["receipt_row"]
        assert rec["target_support"] is None and rec["residual_support"] is None
        assert rec["region"]["shape"] == [16, 16]
        payload = figures.read(case["receipt"])
        assert payload["candidates"][case["candidate_index"]] == rec
        active = [r for r in payload["candidates"] if r["reliability"] > figures.EPS]
        sources = {r["image_idx"] for r in active}
        denominator = sum(r["reliability"] for r in active if r["image_idx"] == rec["image_idx"])
        expected = rec["reliability"] / denominator / len(sources)
        assert rec["objective_coefficient"] == pytest.approx(expected, abs=1e-12)
    for name, layout in provenance["layout"].items():
        assert layout["minimum_text_pt"] >= 9 and layout["minimum_line_pt"] >= .5
        assert layout["width_in"] == (3.375 if name in ("image_full_teaser", "image_full_allocation") else 6.875)
        assert not layout["external_crop"]
        assert not any("MLMP" in text for text in layout["text"])
    text = " ".join(provenance["layout"]["image_full_allocation"]["text"])
    for row in provenance["allocation"]:
        assert figures.delta_label(row["C_minus_confidence"]) in text
        assert figures.delta_label(row["C_minus_uniform"]) in text
    teaser_text = " ".join(provenance["layout"]["image_full_teaser"]["text"])
    for case in provenance["teaser"]:
        assert f"Confidence: {case['receipt_row']['confidence']:.3f}" in teaser_text
        assert f"Probe support: {case['receipt_row']['counterfactual_support']:.3f}" in teaser_text
    assert "#151" not in teaser_text and "#241" not in teaser_text and "Selected extrema" not in teaser_text
    pipeline = provenance["layout"]["image_full_pipeline"]
    assert "TRACE" in pipeline["text"]
    assert len(pipeline["math"]) == 5
    assert all(m["minimum_glyph_pt"] >= 9 for m in pipeline["math"])
    assert any(r"\rho_R" in m["source"] for m in pipeline["math"])
    assert any(r"\alpha_{R\mid j}" in m["source"] for m in pipeline["math"])
    segmentation_text = " ".join(provenance["layout"]["image_full_segmentation"]["text"])
    for case in provenance["segmentation"]:
        assert f"Class IoU: {case['confidence']['iou']:.1f} -> {case['trace']['iou']:.1f}" in segmentation_text
    # The manuscript's actual 505.89pt text width must leave caption headroom.
    segmentation_layout = provenance["layout"]["image_full_segmentation"]
    assert segmentation_layout["height_in"] / segmentation_layout["width_in"] * 505.89 < 540
    assert len(segmentation_layout["image_panels"]) == 16
    for panel in segmentation_layout["image_panels"]:
        assert panel["width_in"] == pytest.approx(1.4)
        assert panel["height_in"] == pytest.approx(1.4)
    for dataset, reads in provenance["diagnostic_reads"].items():
        for indices in reads.values():
            assert all(0 <= i < 400 for i in indices)
    for name, digest in provenance["output_sha256"].items():
        assert figures.sha(path.parent / name) == digest
    assert "per_image_json" in provenance["bounds_note"]
    for index in (1, 2):
        prefix = path.parent / f"image_full_teaser_case{index}"
        original = np.asarray(figures.Image.open(str(prefix) + "_input.png"))
        context = np.asarray(figures.Image.open(str(prefix) + "_context.png"))
        mask = np.asarray(figures.Image.open(str(prefix) + "_mask.png")) > 0
        np.testing.assert_array_equal(original[mask], context[mask])


def test_real_pdfs_have_embedded_simple_fonts_and_no_type3_or_cid():
    output = figures.ROOT / "paper_evolution/figure"
    for name in ("teaser", "pipeline", "segmentation", "allocation"):
        path = output / f"image_full_{name}.pdf"
        if not path.exists():
            pytest.skip("Generate all four figures first")
        assert path.read_bytes().startswith(b"%PDF-1.5")
        report = subprocess.run(["pdffonts", str(path)], check=True, text=True, capture_output=True).stdout
        assert "CID" not in report and "Type 3" not in report
        fonts = report.splitlines()[2:]
        assert fonts
        for font in fonts:
            assert "TrueType" in font or "Type 1" in font
            assert "yes" in font


def test_pdf_numeric_annotations_are_actual_exported_text():
    output = figures.ROOT / "paper_evolution/figure"
    path = output / "image_full_provenance.json"
    if not path.exists():
        pytest.skip("Generate figures first")
    provenance = figures.read(path)
    for name in ("teaser", "segmentation", "allocation"):
        exported = subprocess.run(["pdftotext", "-layout", str(output / f"image_full_{name}.pdf"), "-"],
                                  check=True, text=True, capture_output=True).stdout
        exported = " ".join(exported.split())
        if name == "teaser":
            for case in provenance["teaser"]:
                assert f"Probe support: {case['receipt_row']['counterfactual_support']:.3f}" in exported
                assert f"Confidence: {case['receipt_row']['confidence']:.3f}" in exported
        elif name == "segmentation":
            for case in provenance["segmentation"]:
                assert f"Class IoU: {case['confidence']['iou']:.1f} -> {case['trace']['iou']:.1f}" in exported
        else:
            for row in provenance["allocation"]:
                assert figures.delta_label(row["C_minus_confidence"]) in exported
                assert figures.delta_label(row["C_minus_uniform"]) in exported
