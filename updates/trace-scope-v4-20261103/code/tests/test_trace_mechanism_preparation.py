import copy

import numpy as np
import pytest

from tools.evolution_fifty.risk_panel import select_sources, select_candidates
from trace_ovss.method.probe_geometry import translated_candidate


def test_sampling_is_outcome_and_score_blind():
    ids = [f"source{i}" for i in range(30)]
    assert select_sources(ids, 8) == select_sources(ids, 8)
    assert [x for x in ids if x in select_sources(ids, 8)] == select_sources(ids, 8)
    rows = [dict(batch_idx=0, class_idx=i, component_idx=1, region={}, C=i/10,
                 confidence=i/10, outcome=i) for i in range(10)]
    altered = copy.deepcopy(rows)
    for row in altered:
        row.update(C=1-row["C"], confidence=0., outcome=-100)
    identities = lambda x: [r["class_idx"] for r in x]
    assert identities(select_candidates("source", rows, 4)) == identities(select_candidates("source", altered, 4))


def test_translation_is_exact_shape_preserving_and_does_not_mutate_readout():
    mask = np.zeros((8, 9), dtype=bool)
    mask[1:3, 2:5] = True
    mask[3, 2] = True
    original = mask.copy()
    shifted, info = translated_candidate(mask)
    assert info["shiftable"] and info["original_area"] == info["shifted_area"] == 7
    y, x = np.nonzero(mask)
    assert shifted[y+info["dy"], x+info["dx"]].all()
    np.testing.assert_array_equal(mask, original)
    assert info["intersection"] == 0
    assert info["readout_and_loss_remain_original"]


def test_full_frame_cannot_translate_and_is_reported_not_removed():
    mask = np.ones((4, 4), dtype=bool)
    shifted, info = translated_candidate(mask)
    np.testing.assert_array_equal(shifted, mask)
    assert info["shiftable"] is False and info["overlap_of_candidate"] == 1


@pytest.mark.parametrize("mask", [np.zeros((4, 4), dtype=bool), np.ones((4, 4)), np.ones((4,), dtype=bool)])
def test_bad_masks_rejected(mask):
    with pytest.raises(ValueError):
        translated_candidate(mask)
