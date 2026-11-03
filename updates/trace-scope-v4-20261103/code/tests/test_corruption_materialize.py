from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.evolution_fifty.corruption_materialize import generate_image


def same_rng(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def test_generation_corrupts_original_resolution_BGR_before_resize_and_restores_RNG():
    bgr = np.random.default_rng(23).integers(0, 255, size=(43, 61, 3), dtype=np.uint8)
    original = bgr.copy()
    expected_calls = []
    def corrupt(pixels, *, severity, corruption_name):
        expected_calls.append((pixels.shape, severity, corruption_name))
        assert np.array_equal(pixels, original)
        return np.clip(pixels.astype(int)+np.random.randint(-20, 21, pixels.shape), 0, 255).astype(np.uint8)
    np.random.seed(101)
    state = np.random.get_state()
    encoded = generate_image(SimpleNamespace(corrupt=corrupt), bgr, 7, "gaussian_noise")
    assert same_rng(state, np.random.get_state())
    assert np.array_equal(bgr, original)
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    np.random.seed(7)
    expected = cv2.resize(corrupt(original, severity=5, corruption_name="gaussian_noise"), (224, 224), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(decoded, expected)
    assert expected_calls == [((43, 61, 3), 5, "gaussian_noise")]*2


def test_same_source_type_has_same_bytes_independent_of_external_RNG_and_prior_types():
    image = np.full((37, 48, 3), 127, np.uint8)
    def corrupt(pixels, **kwargs):
        return np.random.randint(0, 256, pixels.shape, dtype=np.uint8)
    module = SimpleNamespace(corrupt=corrupt)
    first = generate_image(module, image, 5, "gaussian_noise")
    generate_image(module, image, 20, "fog")
    np.random.seed(400)
    assert generate_image(module, image, 5, "gaussian_noise") == first
    assert generate_image(module, image, 6, "gaussian_noise") != first


def test_reference_failure_restores_RNG_and_does_not_invent_output():
    def corrupt(*args, **kwargs):
        np.random.rand(10)
        raise ValueError("injected reference failure")
    state = np.random.get_state()
    with pytest.raises(ValueError, match="reference failure"):
        generate_image(SimpleNamespace(corrupt=corrupt), np.zeros((32, 32, 3), np.uint8), 1, "fog")
    assert same_rng(state, np.random.get_state())


def test_mutating_reference_rejected_not_allowed_to_change_next_corruption_input():
    def corrupt(pixels, **kwargs):
        pixels[0, 0, 0] = 255
        return pixels
    with pytest.raises(ValueError, match="mutated shared source"):
        generate_image(SimpleNamespace(corrupt=corrupt), np.zeros((32, 32, 3), np.uint8), 1, "fog")


@pytest.mark.parametrize("kind,index", [("unknown", 0), ("fog", -1), ("fog", True)])
def test_invalid_condition_rejected(kind, index):
    with pytest.raises(ValueError, match="Invalid generation"):
        generate_image(SimpleNamespace(), np.zeros((32, 32, 3), np.uint8), index, kind)
