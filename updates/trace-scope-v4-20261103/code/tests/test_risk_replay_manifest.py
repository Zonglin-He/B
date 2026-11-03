import hashlib
import json

import pytest

from tools.evolution_fifty.risk_replay_manifest import bind_parent, bound_json


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def parent_fixture(tmp_path, *, fault=None):
    parent = tmp_path/"parent"
    protocol = write(parent/"PROTOCOL.json", dict(experiment="M01"))
    specs, seals = {}, {}
    for dataset in ("coco_object", "pc60"):
        row = dict(official_id=f"{dataset}:source", image="RGB does not exist", label="GT must not open")
        example = dict(source_id=row["official_id"], candidate_id="candidate", episode=3,
            episode_ids=[row["official_id"]], source_input=row, candidate=dict(image_idx=0))
        cfg = dict(steps=3)
        panel = dict(candidate_count=1, config=cfg, examples=[example])
        if fault == "duplicate" and dataset == "pc60":
            panel["examples"] *= 2
            panel["candidate_count"] = 2
        specs[dataset] = dict(candidate_count=panel["candidate_count"], episode_count=1,
            panels_sha256=write(parent/"panels"/f"{dataset}.panel.json", panel),
            episodes_sha256=write(parent/"panels"/f"{dataset}.episodes.json", {"3":[row]}),
            native_config=cfg)
        paths = {f"case_000000/step_{step:02d}.{suffix}": "1"*64
                 for step in (0, 1, 3) for suffix in ("labels.npy", "visual_ln.npz")}
        paths["case_000000/trajectory.json"] = "2"*64
        if fault == "missing" and dataset == "pc60":
            paths.pop("case_000000/step_03.visual_ln.npz")
        if fault == "extra" and dataset == "pc60":
            paths["../elsewhere.npy"] = "3"*64
        seals[dataset] = dict(complete=True, candidates=panel["candidate_count"],
            gt_contents_opened=False, dataset_spec=specs[dataset], artifact_sha256=paths)
    prepared = write(parent/"SUITE_PREPARED.json", dict(root=str(parent), schema_version=1,
        protocol_sha256=protocol, datasets=specs))
    hashes = {}
    for dataset, seal in seals.items():
        seal["suite_prepared_sha256"] = prepared
        if fault == "GT" and dataset == "pc60":
            seal["gt_contents_opened"] = True
        hashes[dataset] = write(parent/dataset/"PREDICTIONS_SEALED.json", seal)
    barrier = write(parent/"ALL_RISK_PREDICTIONS_SEALED.json", dict(complete=True,
        gt_contents_opened=False, dataset_seals=hashes))
    return parent, prepared, barrier


def test_metadata_binding_requires_no_RGB_GT_model_or_prediction_payload(tmp_path):
    parent, prepared, barrier = parent_fixture(tmp_path)
    result = bind_parent(parent, prepared, barrier)
    assert not result["payloads_opened"] and not result["labels_read"] and not result["GPU_admitted"]
    assert set(result["datasets"]) == {"coco_object", "pc60"}
    for dataset in result["datasets"].values():
        assert [r["step"] for r in dataset["cases"][0]["checkpoints"]] == [0, 1, 3]


@pytest.mark.parametrize("fault", ["duplicate", "missing", "extra", "GT"])
def test_metadata_tampering_cannot_be_hidden_by_rehashing_seals(tmp_path, fault):
    parent, prepared, barrier = parent_fixture(tmp_path, fault=fault)
    with pytest.raises(ValueError):
        bind_parent(parent, prepared, barrier)


@pytest.mark.parametrize("marker", ["FAILED.json", "CANCELLED.json"])
def test_failed_or_cancelled_parent_rejected(tmp_path, marker):
    parent, prepared, barrier = parent_fixture(tmp_path)
    write(parent/marker, {})
    with pytest.raises(ValueError, match="Failed or cancelled"):
        bind_parent(parent, prepared, barrier)


def test_stale_or_rehashed_root_cannot_replace_exact_parent_identity(tmp_path):
    parent, prepared, barrier = parent_fixture(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        bind_parent(parent, "0"*64, barrier)
    with pytest.raises(ValueError, match="digest"):
        bind_parent(parent, prepared, "0"*64)


def test_aliased_metadata_rejected_even_with_matching_bytes(tmp_path):
    actual = tmp_path/"actual.json"
    digest = write(actual, {})
    alias = tmp_path/"alias.json"
    alias.symlink_to(actual)
    with pytest.raises(ValueError, match="Aliased"):
        bound_json(alias, digest)
