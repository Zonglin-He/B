import io

import numpy as np
from PIL import Image
import pytest

from tools.evolution_fifty.risk_score_inputs import candidate_image_mask, decode_annotation, initial_region_quality


def candidate():
    mask = np.array([[True, False], [False, True]])
    return dict(batch_idx=1, image_idx=1, area=2, region=dict(shape=[2, 2], true_pixels=2,
        encoding="packbits-little-hex", data=np.packbits(mask.ravel(), bitorder="little").tobytes().hex()))


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def test_original_mask_nearest_projection_preserves_location_and_area_scale():
    c = candidate()
    got = candidate_image_mask(c, [4, 4])
    expected = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], bool)
    np.testing.assert_array_equal(got, expected)
    assert c["area"] == 2 and c["region"]["shape"] == [2, 2]
    c["batch_idx"] = 0
    with pytest.raises(ValueError, match="membership"):
        candidate_image_mask(c, [4, 4])


def test_annotation_nearest_ignore_and_optional_reducezero_match_existing_rule():
    pixels = np.array([[0, 1], [2, 255]], np.uint8)
    standard = decode_annotation(png(pixels), [4, 4], 3, False)
    np.testing.assert_array_equal(standard, np.repeat(np.repeat(pixels, 2, 0), 2, 1))
    reduced = decode_annotation(png(pixels), [2, 2], 2, True)
    np.testing.assert_array_equal(reduced, [[255, 0], [1, 255]])
    with pytest.raises(ValueError, match="outside frozen"):
        decode_annotation(png(pixels), [2, 2], 2, False)


def test_initial_correctness_control_separate_from_update_outcome_and_empty_coverage():
    before = np.array([[0, 1], [0, 1]], np.uint16)
    gt = np.array([[0, 0], [255, 1]], np.uint8)
    result = initial_region_quality(before, gt, np.ones((2, 2), bool), 2)
    assert result["valid_pixels"] == 3 and result["correct_pixels"] == 2
    assert result["fraction_correct"] == pytest.approx(2/3) and result["fixed_bin"] == 2
    empty = initial_region_quality(before, gt, np.array([[0, 0], [1, 0]], bool), 2)
    assert empty["fraction_correct"] is None and empty["fixed_bin"] is None


def test_annotation_and_mask_geometry_failclosed():
    with pytest.raises(ValueError):
        candidate_image_mask(candidate(), [1, 1])
    with pytest.raises(ValueError):
        decode_annotation(png(np.array([[1]], np.uint8)), [100000, 1], 2, False)
    with pytest.raises(ValueError):
        initial_region_quality(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8), np.ones((2, 2)), 2)
