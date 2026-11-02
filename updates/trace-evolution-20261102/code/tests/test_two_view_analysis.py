import pytest
from pathlib import Path
import sys
import json
import hashlib
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tools.analyze_two_view_full import allocation_by_class, validate_vocabulary, require_owned_artifact


def row(episode, cls, t, c):
    return dict(episode=episode, class_idx=cls, target_support=t, counterfactual_support=c,
                reliability=min(t, c))


def test_allocation_is_episode_normalized_not_pixel_or_candidate_weighted():
    stats = allocation_by_class([row(0, 0, .8, .4), row(0, 1, .2, .2),
                                row(1, 1, .1, .1)], (0, 1), 2, 1e-8)
    assert stats[0]["mean_episode_quality_share"] == pytest.approx(1 / 3)
    assert stats[1]["mean_episode_quality_share"] == pytest.approx(2 / 3)
    assert stats[0]["mean_episode_target_share"] == pytest.approx(.4)
    assert stats[0]["cf_limiting_fraction"] == 1.


def test_mapping_and_zero_objective_have_explicit_semantics():
    stats = allocation_by_class([row(0, 0, 0, .4), row(0, 1, 0, .2)], (1, 1), 3, 1e-8)
    assert stats[1]["candidates"] == 2
    assert stats[1]["mean_episode_quality_share"] == 0
    assert stats[0]["mean_target"] is None


def test_non_quality_scores_are_rejected():
    bad = dict(row(0, 0, .8, .4), reliability=.3)
    with pytest.raises(ValueError):
        allocation_by_class([bad], None, 1, 1e-8)


def test_swapped_class_names_are_not_accepted():
    vocab = SimpleNamespace(classes=("a", "b"), extensions=("a", "b"), extension_to_class=(0, 1))
    panel = dict(config=dict(class_extensions=True),
        vocabulary_sha256=hashlib.sha256(json.dumps([vocab.extensions, vocab.extension_to_class]).encode()).hexdigest())
    assert validate_vocabulary(vocab, panel) == (0, 1)
    vocab.extensions = ("b", "a")
    with pytest.raises(ValueError, match="vocabulary differs"):
        validate_vocabulary(vocab, panel)


def test_result_paths_must_belong_to_the_sealed_suite(tmp_path):
    owner = tmp_path / "suite"
    owner.mkdir()
    result = owner / "result.json"
    result.write_text("original")
    (owner / "SUITE_SEALED.json").write_text(json.dumps(dict(artifact_sha256={
        "result.json": hashlib.sha256(result.read_bytes()).hexdigest()})))
    require_owned_artifact(result, owner)
    other = tmp_path / "other.json"
    other.write_text("original")
    with pytest.raises(ValueError, match="outside"):
        require_owned_artifact(other, owner)
    result.write_text("changed")
    with pytest.raises(ValueError, match="owning seal"):
        require_owned_artifact(result, owner)
