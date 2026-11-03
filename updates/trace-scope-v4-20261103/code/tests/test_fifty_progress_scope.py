import copy

import pytest

from tools.evolution_fifty.progress import apply_scope


def test_reduced_scope_preserves_evidence_and_does_not_complete_deferred():
    statuses = {
        "E01": dict(status="prepared", required=True, evidence=["frozen-grid"]),
        "E02": dict(status="partial", required=True, evidence=["cache-audit"]),
        "E40": dict(status="prepared", required=True, evidence=["same-frozen-grid"]),
    }
    before = copy.deepcopy(statuses)
    selected = apply_scope(statuses, dict(active_experiments=["E01", "E02"],
                                        unlisted_policy="deferred_not_completed"))
    assert len(selected) == 2 and all(row["required"] for row in selected)
    assert statuses["E40"]["execution_policy"] == "deferred"
    assert statuses["E40"]["required"] is False
    for key, row in statuses.items():
        assert row["status"] == before[key]["status"]
        assert row["evidence"] == before[key]["evidence"]


@pytest.mark.parametrize("active", [[], ["E99"], ["E01", "E01"], "E01", None, [[]]])
def test_scope_rejects_empty_unknown_duplicate_or_malformed_ids(active):
    with pytest.raises(ValueError):
        apply_scope({"E01": {}}, dict(active_experiments=active,
                                     unlisted_policy="deferred_not_completed"))


def test_scope_rejects_deferred_completion_policy():
    with pytest.raises(ValueError):
        apply_scope({"E01": {}}, dict(active_experiments=["E01"], unlisted_policy="completed"))
