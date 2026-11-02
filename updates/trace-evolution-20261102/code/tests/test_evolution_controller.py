"""CPU acceptance tests against real prediction/scoring seals and payloads."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import self_evolution_controller as controller
from tools.evolution.policies import expand_policies
from tools.evolution.run_cell import score, seal_predictions


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def reseal_scoring(bundle):
    path = bundle.directory / "SCORED_SEALED.json"
    seal = read_json(path)
    seal["prediction_seal_sha256"] = controller.digest(
        bundle.directory / "PREDICTIONS_SEALED.json"
    )
    seal["artifact_sha256"] = {
        name: controller.digest(bundle.directory / name)
        for name in seal["artifact_sha256"]
    }
    write_json(path, seal)


def reseal_predictions(bundle):
    # Semantic-corruption cases keep the entire hash chain internally valid.
    path = bundle.directory / "PREDICTIONS_SEALED.json"
    seal = read_json(path)
    seal["artifact_sha256"] = {
        name: controller.digest(bundle.directory / name)
        for name in seal["artifact_sha256"]
    }
    write_json(path, seal)
    result_path = bundle.directory / "result.json"
    result = read_json(result_path)
    result["prediction_seal_sha256"] = controller.digest(path)
    write_json(result_path, result)
    reseal_scoring(bundle)


def verify(bundle):
    return controller.verify_cell(
        bundle.directory, bundle.cell, bundle.panel, bundle.manifest
    )


@pytest.fixture
def bundle(tmp_path):
    suite = tmp_path / "suite" / "tmp"
    cell = dict(key="smoke/voc21", dataset="voc21", seed=0, limit=3,
                policies=["trace", "confidence", "permuted"])
    directory = suite / "cells" / cell["key"]
    directory.mkdir(parents=True)
    source = suite / "runtime" / "trace_ovss" / "resources" / "prompts.txt"
    source.parent.mkdir(parents=True)
    source.write_text("a photo of {}\n")
    data = suite / "data"
    data.mkdir()
    inputs = []
    labels = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    for index in range(3):
        image, label = data / f"{index}.png", data / f"{index}.label.png"
        Image.fromarray(np.full((2, 2, 3), index, dtype=np.uint8)).save(image)
        Image.fromarray(labels).save(label)
        inputs.append(dict(image=str(image), label=str(label),
            image_sha256=controller.digest(image),
            label_metadata=[label.stat().st_size, label.stat().st_mtime_ns]))
    config = dict(batch_size=2, patch_size=[2, 2], patch_stride=1)
    classes = ["first", "second"]
    panel = dict(dataset="voc21", split="evaluation", seed=0, selection_seed=20261024,
        image_ids=[row["image"] for row in inputs],
        label_ids=[row["label"] for row in inputs],
        image_sha256={row["image"]: row["image_sha256"] for row in inputs},
        file_metadata=[[path.stat().st_size, path.stat().st_mtime_ns]
                       for row in inputs for path in (Path(row["image"]), Path(row["label"]))],
        vocabulary_sha256=hashlib.sha256(json.dumps([classes, None]).encode()).hexdigest(),
        config=config, available_images=1449)
    write_json(directory / "expected_panel.json", panel)
    variants = expand_policies(cell["policies"], [0, 1, 2])
    protocol = dict(dataset="voc21", seed=0, full_split=False, inputs=inputs,
        config=config, classes=classes, variants=variants, reduce_zero_label=False,
        data_root=str(data), panel=copy.deepcopy(panel),
        source_sha256={str(source): controller.digest(source)},
        expected_panel_sha256=controller.digest(directory / "expected_panel.json"),
        deterministic_runtime=dict(cublas_workspace_config=":4096:8", deterministic_algorithms=True,
            deterministic_warn_only=False, cudnn_deterministic=True, cudnn_benchmark=False),
        timing_scope="synthetic CPU acceptance fixture")
    write_json(directory / "prediction_protocol.json", protocol)
    write_json(directory / "initial_state.json", {})
    write_json(directory / "checkpoint_identity.json", {})
    write_json(directory / "prediction_result.json", dict(status="predicted", images=3))
    (directory / "cost.jsonl").write_text("{}\n")
    for name in ["pre", *[entry[0] for entry in variants]]:
        target = directory / "predictions" / name
        target.mkdir(parents=True)
        prediction = labels if name == "trace" else 1 - labels
        for index in range(3):
            np.save(target / f"{index:06d}.npy", prediction.astype(np.uint16), allow_pickle=False)
    episodes = directory / "episodes"
    episodes.mkdir()
    for episode, indices in enumerate(([0, 1], [2])):
        write_json(episodes / f"{episode:06d}.json",
                   dict(episode=episode, image_indices=indices, candidate_count=0))
        np.savez_compressed(episodes / f"{episode:06d}.npz", masks=np.empty((0, 2, 2), dtype=bool))
    seal_predictions(directory, protocol)
    score(directory, dataset="voc21", seed=0, data_root=data)
    manifest = dict(image_sha256=copy.deepcopy(panel["image_sha256"]),
        runtime_hashes={str(source.relative_to(suite / "runtime")): controller.digest(source)},
        permutation_seeds=[0, 1, 2])
    return SimpleNamespace(directory=directory, cell=cell, panel=panel, manifest=manifest)


def test_valid_bundle_with_real_seals_and_partial_final_episode_passes(bundle):
    result = verify(bundle)
    assert result["status"] == "completed" and result["images"] == 3
    records = [json.loads(line) for line in
               (bundle.directory / "scores_by_episode.jsonl").read_text().splitlines()]
    assert [row["image_indices"] for row in records] == [[0, 1], [2]]
    assert len(result["policy_results"]) == 5
    assert next(row for row in result["policy_results"] if row["policy"] == "trace")["miou_unrounded"] == 100


def test_nondeterministic_runtime_cannot_be_relabelled_after_resealing(bundle):
    path = bundle.directory / "prediction_protocol.json"
    protocol = read_json(path)
    protocol["deterministic_runtime"]["deterministic_algorithms"] = False
    write_json(path, protocol)
    reseal_predictions(bundle)
    with pytest.raises(RuntimeError, match="deterministic runtime"):
        verify(bundle)


@pytest.mark.parametrize("name", ["PREDICTIONS_SEALED.json", "SCORED_SEALED.json"])
@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_missing_or_malformed_seal_is_rejected(bundle, name, damage):
    path = bundle.directory / name
    if damage == "missing":
        path.unlink()
        error = FileNotFoundError
    else:
        path.write_text("not JSON\n")
        error = json.JSONDecodeError
    with pytest.raises(error):
        verify(bundle)


def test_corrupt_prediction_payload_is_rejected(bundle):
    with (bundle.directory / "predictions/trace/000000.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(RuntimeError, match="Artifact seal mismatch"):
        verify(bundle)


@pytest.mark.parametrize("damage", ["missing", "wrong", "duplicate", "wrong_permutation_seed"])
def test_policy_result_exact_set_is_enforced_after_resealing(bundle, damage):
    path = bundle.directory / "result.json"
    result = read_json(path)
    if damage == "missing":
        result["policy_results"].pop()
    elif damage == "wrong":
        result["policy_results"][0]["policy"] = "uniform"
    elif damage == "duplicate":
        result["policy_results"].append(copy.deepcopy(result["policy_results"][0]))
    else:
        result["policy_results"][-1]["permutation_seed"] = 99
    write_json(path, result)
    reseal_scoring(bundle)
    with pytest.raises(RuntimeError, match="policy results"):
        verify(bundle)


def test_worker_image_hash_drift_is_rejected_after_resealing(bundle):
    path = bundle.directory / "prediction_protocol.json"
    protocol = read_json(path)
    protocol["inputs"][0]["image_sha256"] = "0" * 64
    write_json(path, protocol)
    reseal_predictions(bundle)
    with pytest.raises(RuntimeError, match="images/content differ"):
        verify(bundle)


def test_frozen_parent_image_hash_drift_is_rejected(bundle):
    bundle.manifest["image_sha256"][bundle.panel["image_ids"][0]] = "0" * 64
    with pytest.raises(RuntimeError, match="images/content differ"):
        verify(bundle)


def test_same_count_different_image_is_rejected_after_resealing(bundle):
    path = bundle.directory / "prediction_protocol.json"
    protocol = read_json(path)
    protocol["inputs"][0]["image"] = str(bundle.directory / "unrequested.png")
    assert len(protocol["inputs"]) == len(bundle.panel["image_ids"])
    write_json(path, protocol)
    reseal_predictions(bundle)
    with pytest.raises(RuntimeError, match="images/content differ"):
        verify(bundle)


def test_bad_episode_boundary_is_rejected_after_resealing(bundle):
    path = bundle.directory / "scores_by_episode.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["image_indices"] = [1]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    reseal_scoring(bundle)
    with pytest.raises(RuntimeError, match="episode coverage"):
        verify(bundle)


@pytest.mark.parametrize("damage", ["swap_policies", "pre_uses_adapted_counts"])
def test_policy_labels_cannot_reassign_another_policies_metrics(bundle, damage):
    path = bundle.directory / "result.json"
    result = read_json(path)
    if damage == "swap_policies":
        first, second = result["policy_results"][:2]
        first["policy"], second["policy"] = second["policy"], first["policy"]
    else:
        result["pre_result"] = copy.deepcopy(result["policy_results"][0])
        result["pre_result"].update(policy="pre", permutation_seed=None)
    write_json(path, result)
    reseal_scoring(bundle)
    with pytest.raises(RuntimeError):
        verify(bundle)
