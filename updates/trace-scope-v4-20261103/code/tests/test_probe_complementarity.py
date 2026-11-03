"""CPU-only algebra, admission, streaming and fail-closed provenance tests."""

import hashlib
import json

import numpy as np
import pytest

from tools.evolution_followup import probe_audit as audit


def test_empty_has_undefined_tv_not_a_fake_zero():
    row = audit.audit_episode([], [])
    assert row["candidates"] == 0
    assert row["active"] == dict(min=0, T_only=0, C_only=0)
    assert row["min_vs_T_only"]["tv"] is None
    stats = audit.summarize([row])
    tv = stats["tv"]["min_vs_T_only"]["all_episodes"]
    assert tv["eligible_episodes"] == 0 and tv["undefined_episodes"] == 1
    assert tv["mean"] is None and tv["median"] is None and tv["fraction_gt_005"] is None


def test_eps_admission_is_strict_and_independent_by_arm():
    row = audit.audit_episode([audit.EPS, 2*audit.EPS, 0], [1, 1, 1])
    assert row["active"] == dict(min=1, T_only=1, C_only=3)
    assert row["min_vs_T_only"]["tv"] == 0
    assert row["min_vs_C_only"]["tv"] == pytest.approx(2/3)
    row = audit.audit_episode([audit.EPS], [1])
    assert row["min_vs_C_only"]["tv"] is None
    assert row["active"]["min"] == 0


def test_identical_probes_have_zero_tv_and_all_ties():
    row = audit.audit_episode([0, .2, .8], [0, .2, .8])
    assert row["limit_all"]["tie"] == 3
    for direction in ("min_vs_T_only", "min_vs_C_only"):
        assert row[direction] == dict(tv=0, same_rank=True, same_multiset=True, same_assignment=True)


def test_one_candidate_magnitude_change_is_not_allocation_change():
    row = audit.audit_episode([.2], [.9])
    assert row["limit_all"]["T_lt_C"] == 1
    assert row["min_vs_C_only"]["tv"] == 0
    stats = audit.summarize([row])
    assert stats["active_episode_counts"]["min"] == dict(zero=0, one=1, at_least_two=0)
    assert stats["tv"]["min_vs_C_only"]["at_least_two_min_active"]["mean"] is None


def test_asymmetric_limiting_and_both_tv_directions():
    row = audit.audit_episode([.2, .8], [.8, .4])
    assert row["limit_all"]["T_lt_C"] == row["limit_all"]["C_lt_T"] == 1
    assert row["both_limiters_present_min_active"]
    assert row["min_vs_T_only"]["tv"] == pytest.approx(2/15)
    assert row["min_vs_C_only"]["tv"] == pytest.approx(1/3)
    swapped = audit.audit_episode([.8, .4], [.2, .8])
    assert swapped["min_vs_T_only"] == row["min_vs_C_only"]
    assert swapped["min_vs_C_only"] == row["min_vs_T_only"]


def test_common_scale_can_leave_allocation_unchanged():
    row = audit.audit_episode([.1, .3], [.2, .6])
    assert row["limit_all"]["T_lt_C"] == 2
    assert row["min_vs_C_only"]["tv"] == pytest.approx(0)


def test_same_rank_does_not_imply_same_allocation():
    row = audit.audit_episode([.2, .8], [.4, .6])
    assert row["min_vs_T_only"]["same_rank"]
    assert row["min_vs_T_only"]["tv"] == pytest.approx(.05)
    row = audit.audit_episode([.2, .8], [.4, .4])
    stats = audit.summarize([row])
    assert stats["tv"]["min_vs_T_only"]["all_episodes"]["same_rank_but_tv_gt_005"] == 1


def test_same_multiset_can_have_different_candidate_assignment():
    a, _ = audit.probabilities([.2, .8])
    b, _ = audit.probabilities([.8, .2])
    result = audit.compare_allocations(a, b)
    assert result["same_multiset"] and not result["same_assignment"]
    assert not result["same_rank"] and result["tv"] == pytest.approx(.6)


def test_rank_preserves_exact_ties_and_multiset_tolerance_is_absolute():
    assert not audit.compare_allocations(np.array([.5, .5]), np.array([.4999999, .5000001]))["same_rank"]
    assert audit.compare_allocations(np.array([.5, .5]), np.array([.4999999, .5000001]))["same_multiset"]
    assert not audit.compare_allocations(np.array([.5, .5]), np.array([.49, .51]))["same_multiset"]


def test_limiting_tolerance_and_inclusive_gap_boundaries():
    row = audit.audit_episode([0, 0, 1e-6, 2e-6, 0, .05],
                             [0, 1e-6, 0, 0, .05, 0])
    assert row["limit_all"] == dict(T_lt_C=1, C_lt_T=2, tie=3, abs_gap_ge_005=2,
                                    T_lt_C_gap_ge_005=1, C_lt_T_gap_ge_005=1)
    stats = audit.tv_summary([dict(tv=.05, same_rank=False, same_multiset=False,
                                  same_assignment=False)])
    assert stats["count_gt_005"] == 0


@pytest.mark.parametrize("t,c", [([np.nan], [1]), ([1], [np.inf]), ([-.1], [1]),
                                ([1.1], [1]), ([[.1]], [[.1]]), ([.1, .2], [.1])])
def test_invalid_vectors_rejected(t, c):
    with pytest.raises(ValueError):
        audit.audit_episode(t, c)


def test_summary_denominators_and_no_independent_candidate_weighting():
    empty = audit.audit_episode([], [])
    small = audit.audit_episode([.2, .8], [.8, .4])
    large = audit.audit_episode([.5]*10, [.5]*10)
    stats = audit.summarize([empty, small, large])
    assert stats["candidates"] == 12 and stats["episodes"] == 3
    tv = stats["tv"]["min_vs_T_only"]["all_episodes"]
    assert tv["eligible_episodes"] == 2 and tv["undefined_episodes"] == 1
    assert tv["mean"] == pytest.approx(small["min_vs_T_only"]["tv"]/2)
    assert tv["fraction_gt_005"] == .5


def make_cache(tmp_path, *, bad_q=False, bad_rho=False):
    cell = tmp_path / "fixture"
    (cell / "episodes").mkdir(parents=True)
    labels = ["trace", "target", "counterfactual"]
    receipt = dict(episode=0, image_indices=[0], dataset_indices=[0], policy_labels=labels,
                   candidate_budget=2, candidate_count=2)
    meta = cell / "episodes/000000.json"
    meta.write_text(json.dumps(receipt))
    t, c = np.array([.2, .8]), np.array([.8, .4])
    q = np.minimum(t, c)
    rho = np.stack([q, t, c])
    if bad_rho:
        rho[1, 0] += .1
    if bad_q:
        q = q + .1
    np.savez_compressed(cell / "episodes/000000.npz", T=t, C=c, q=q, rho=rho,
        masks=np.ones((2, 1, 1), dtype=bool), image_index=np.zeros(2, dtype=int),
        patch_index=np.zeros(2, dtype=int), query_index=np.zeros(2, dtype=int),
        confidence=np.ones(2), donor_candidate_index=np.zeros((3, 2), dtype=int))
    return dict(cell=str(cell), episodes=1, images=1, batch_size=1, labels=labels, budget=2,
                sealed_cache_inventory={"episodes/000000.json": audit.sha256(meta)})


def test_stream_only_decompresses_four_score_arrays(tmp_path, monkeypatch):
    import io
    info = make_cache(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__
    opened = []
    def restricted(self, key):
        opened.append(key)
        assert key in ("T", "C", "q", "rho")
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", restricted)
    handle = io.StringIO()
    rows, observed = audit.stream_cell(info, handle)
    assert opened == ["T", "C", "q", "rho"]
    assert len(rows) == 1 and json.loads(handle.getvalue())["candidates"] == 2
    assert len(observed["consumed_T_C_q_rho_sha256"]) == 64


@pytest.mark.parametrize("kwargs", [dict(bad_q=True), dict(bad_rho=True)])
def test_stream_rejects_changed_q_or_arm_weights(tmp_path, kwargs):
    import io
    with pytest.raises(RuntimeError):
        audit.stream_cell(make_cache(tmp_path, **kwargs), io.StringIO())


def test_stream_rejects_receipt_tampering(tmp_path):
    import io
    info = make_cache(tmp_path)
    (tmp_path / "fixture/episodes/000000.json").write_text("{}")
    with pytest.raises(RuntimeError, match="receipt hash"):
        audit.stream_cell(info, io.StringIO())


def test_checked_rejects_bad_hash_and_path_escape(tmp_path):
    path = tmp_path / "small.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="Seal mismatch"):
        audit.checked(tmp_path, "small.json", {"small.json": "0"*64}, {})
    with pytest.raises(RuntimeError, match="escapes"):
        audit.checked(tmp_path, "../outside", {}, {})
    assert audit.checked(tmp_path, "small.json", {"small.json": hashlib.sha256(b"{}").hexdigest()}, {})


def test_existing_output_never_overwritten_even_before_source_reads(tmp_path):
    with pytest.raises(RuntimeError, match="never overwrite"):
        audit.run(tmp_path / "nonexistent_source", tmp_path)
