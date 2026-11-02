"""The manuscript's two-view entry point has an executable identity."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from trace_ovss.method.two_view import TwoViewTRACE, shared_support
from test_simplified_trace import tiny_trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_shared_support_is_the_joint_upper_bounded_score():
    assert shared_support(.8, .6) == .6
    for pair in ((float("nan"), .5), (-.1, .5), (.5, 1.1)):
        with pytest.raises(ValueError):
            shared_support(*pair)
    rng = np.random.default_rng(2027)
    for target, perturbed in rng.random((100, 2)):
        q = shared_support(target, perturbed)
        assert 0 <= q <= target and q <= perturbed


def test_two_view_entrypoint_matches_frozen_research_execution():
    from trace_algorithm_variants import make_algorithm_class
    from trace_ovss.method.trace import TRACE

    torch.manual_seed(2027)
    method = tiny_trace()
    images = torch.randn(2, 3, 32, 32)
    method.__class__ = make_algorithm_class(TRACE, "quality_only")
    method.estimate_trace_reliability(images)
    reference = method.last_candidate_stats
    losses = method.adapt(images)
    prediction = method.evaluate(images).clone()
    method.__class__ = TwoViewTRACE
    method.reset()
    method.estimate_trace_reliability(images)
    assert len(method.last_candidate_stats) == len(reference)
    assert method.last_reliability_summary["diagnostic_views"] == 2 * len(reference)
    for expected, actual in zip(reference, method.last_candidate_stats):
        for key in ("target_support", "counterfactual_support", "reliability"):
            assert actual[key] == expected[key]
        assert actual["residual_support"] is None and actual["dominance"] is None
    assert method.adapt(images) == losses
    assert torch.equal(method.evaluate(images), prediction)
    assert method.adapt(images, valid_mask=torch.zeros(2, 1, 32, 32)) == []
    assert method.last_reliability_summary["diagnostic_views"] == 0
