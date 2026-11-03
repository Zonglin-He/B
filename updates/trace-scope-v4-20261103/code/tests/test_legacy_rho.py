import numpy as np
import pytest
import torch

from tools.evolution_fifty.legacy_rho import load_components, query_roles
from trace_ovss.method.image_balanced import ImageBalancedTRACE


@pytest.fixture(scope="module")
def legacy():
    return load_components()


def test_exact_old_role_dependent_formula_not_current_role_free_substitute(legacy):
    score = legacy["modules"]["reliability"].trace_reliability
    assert score(.8, .6, .9, "thing").reliability == pytest.approx(.54)
    assert score(.8, .6, None, "stuff").reliability == pytest.approx(.6)
    assert score(.8, .6, None, "background").reliability == pytest.approx(.6)
    with pytest.raises(ValueError):
        score(.8, .6, None, "thing")
    assert legacy["old_engine_loaded"] is False and legacy["old_optimizer_loaded"] is False


def test_real_benchmark_roles_and_query_extensions_from_pinned_source(legacy):
    actual = query_roles(legacy, "PascalContext60Dataset", ["background", "person", "road", "table"], [1, 1, 2, 0, 3])
    assert actual == ("thing", "thing", "stuff", "background", "thing")
    assert query_roles(legacy, "COCOObjectDataset", ["background", "person"], None) == ("background", "thing")


def test_old_role_specific_view_counts_are_not_forced_to_three(legacy):
    kinds = legacy["modules"]["interventions"].trace_view_types
    assert kinds("thing") == ("target", "residual", "counterfactual")
    assert kinds("stuff") == kinds("background") == ("target", "counterfactual")


def test_archived_view_operators_match_current_normalized_pixel_operators(legacy):
    old = legacy["modules"]["interventions"].TraceInterventionMixin
    class Stub:
        _mean = torch.tensor([122.7709, 116.7460, 104.0937])[None,:,None,None]
        _std = torch.tensor([68.5005, 66.6322, 70.3232])[None,:,None,None]
    image = torch.randn(1, 3, 12, 12, generator=torch.Generator().manual_seed(1))
    for name in ("make_trace_weak_image", "make_trace_counterfactual_background"):
        a = getattr(old, name)(Stub(), image)
        b = getattr(ImageBalancedTRACE, name)(Stub(), image)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_unpinned_archive_rejected_before_executing_any_source(tmp_path):
    path = tmp_path/"fake.tar.gz"
    path.write_bytes(b"not the pinned archive")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_components(path)
