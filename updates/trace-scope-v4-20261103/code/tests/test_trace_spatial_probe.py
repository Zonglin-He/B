import copy

import numpy as np
import pytest
import torch

from trace_ovss.method.image_balanced import ImageBalancedTRACE
from trace_ovss.method.spatial_probe import SpatialProbeMixin


class Toy(SpatialProbeMixin):
    def __init__(self, mode):
        self.probe_mode = mode
        self.view_mode = "C"
        self.seen = []

    def make_trace_counterfactual_background(self, image):
        return image*.35

    def _forward_trace_view_batches(self, keys, build):
        if keys:
            views = build(keys)
            self.seen.append(views.clone())
            mean = views.mean(1)
            yield keys, torch.stack((mean, -mean), dim=1)


def inputs():
    image = torch.ones((1, 3, 4, 4))
    masks = [np.zeros((4, 4), dtype=bool) for _ in range(2)]
    masks[0][0:2, 0:2] = True
    masks[1][2:4, 2:4] = True
    rows = [dict(batch_idx=0, image_idx=0, class_idx=0, component_idx=i,
                 confidence=.8, region=mask) for i, mask in enumerate(masks)]
    return image, torch.full((1, 2, 4, 4), .5), rows


def test_native_view_and_support_equal_actual_production_evaluator():
    image, probabilities, rows = inputs()
    ours, native = Toy("native"), Toy("native")
    a = ours.evaluate_trace_views(image, probabilities, rows)
    b = ImageBalancedTRACE.evaluate_trace_views(native, image, probabilities, rows)
    assert [r["reliability"] for r in a] == [r["reliability"] for r in b]
    torch.testing.assert_close(ours.seen[0], native.seen[0], rtol=0, atol=0)


def test_sham_moves_only_preservation_not_readout_or_candidate():
    image, probabilities, rows = inputs()
    original = copy.deepcopy(rows)
    native, sham = Toy("native"), Toy("translated")
    a = native.evaluate_trace_views(image, probabilities, rows)
    b = sham.evaluate_trace_views(image, probabilities, rows)
    assert b[0]["reliability"] < a[0]["reliability"]
    for old, new, receipt in zip(original, b, sham.last_spatial_probe_receipts, strict=True):
        np.testing.assert_array_equal(old["region"], new["region"])
        assert receipt["original_area"] == receipt["shifted_area"] == 4
        assert receipt["intersection"] == 0


def test_global_deduplicates_inputs_but_reads_every_original_candidate():
    image, probabilities, rows = inputs()
    method = Toy("global")
    values = method.evaluate_trace_views(image, probabilities, rows)
    assert len(values) == 2 and method._two_view_counts["diagnostic_views"] == 1
    torch.testing.assert_close(method.seen[0], image*.35)
    assert values[0]["reliability"] == values[1]["reliability"]


def test_confidence_uses_no_diagnostic_inputs():
    image, probabilities, rows = inputs()
    method = Toy("confidence")
    result = method.evaluate_trace_views(image, probabilities, rows)
    assert [r["reliability"] for r in result] == [.8, .8]
    assert not method.seen and method._two_view_counts["diagnostic_views"] == 0


def test_invalid_mode_fails_before_model_work():
    with pytest.raises(ValueError):
        Toy("other").evaluate_trace_views(*inputs())
