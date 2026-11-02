"""Immutable scope and admission checks for the C-allocation controls."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

from tools.evolution_tuning.common import canonical_hash, read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
ALGORITHM = "TRACE:C-allocation-controls:v1"
ARMS = ["image_region", "episode_region", "image_token", "episode_token", "image_shuffled"]
DATASETS = ("coco_object", "pc60")
POPULATIONS = dict(coco_object=5000, pc60=5105)
PERMUTATION_SEED = 20261029
PARENT = Path("/home/wwww/hzl/results/trace_single_probe_waterbirds_full_20261029_0817")
PARENT_SHA = "4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3"
EXPOSURE = "Fixed development-exposed populations; no untouched-test or selection claim"
RESOURCES = dict(MemoryMax="12884901888", MemorySwapMax="1073741824", OOMPolicy="stop",
                 Restart="no", RemainAfterExit="yes")
MANIFEST_KEYS = {"schema_version", "algorithm", "dataset", "stage", "seed", "permutation_seed",
    "arms", "suite_root", "runtime_root", "suite_prepared_sha256", "source_sha256", "config",
    "panel", "checkpoint", "output", "parent_proof_sha256"}


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def absolute_path(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve(), "Noncanonical or aliased path")
    return path


def writable_path(suite, path):
    suite, path = absolute_path(suite), Path(path).absolute()
    require(path.is_relative_to(suite) and path != suite and path.resolve() == path,
            "Writable path escapes suite or follows a symlink")
    if path.exists() and path.is_file():
        require(path.stat().st_nlink == 1, "Writable file has an external hardlink")
    return path


def parent_sources():
    files = {f"parents/{name}": PARENT / name for name in
             ("SUITE_PREPARED.json", "SUITE_SEALED.json", "ALL_CONFIGS_SELECTED.json", "ALL_FULL_PREDICTIONS_SEALED.json")}
    for dataset in DATASETS:
        for name in ("prediction_protocol.json", "PREDICTIONS_SEALED.json", "SCORED_SEALED.json", "result.json"):
            files[f"parents/{dataset}/{name}"] = PARENT / "full" / dataset / name
        files[f"parents/{dataset}/SELECTED_CONFIG_SEALED.json"] = PARENT / "selections" / dataset / "SELECTED_CONFIG_SEALED.json"
    return files


def annotation_inodes(manifest):
    """Stat only, covering both full populations even for a four-image gate."""
    suite = Path(manifest["suite_root"])
    labels = set()
    for dataset in DATASETS:
        path = manifest_path(suite, dataset, "full")
        require(path.resolve() == path and path.is_file(), "Missing canonical peer population")
        peer = read_json(path)
        require(peer["dataset"] == dataset and peer["stage"] == "full"
                and peer["suite_root"] == str(suite), "Wrong peer label-index scope")
        for row in peer["panel"]["inputs"]:
            status = Path(row["label"]).stat()
            labels.add((status.st_dev, status.st_ino))
    return labels


def guard_global_inputs(manifest):
    labels = annotation_inodes(manifest)
    suite = Path(manifest["suite_root"])
    paths = [Path(r["image"]) for r in manifest["panel"]["inputs"]]
    paths += [Path(p) for p in manifest["panel"].get("source_hashes", {})]
    paths += [Path(manifest["checkpoint"]["path"])]
    paths += [Path(manifest["runtime_root"]) / p for p in manifest["source_sha256"]]
    paths += [suite / "SUITE_PREPARED.json", suite / "PARENT_IMPORT.json", suite / "frozen_contract.md"]
    paths += list((suite / "manifests").glob("*.json"))
    paths += [p for p in (suite / "parents").rglob("*") if p.is_file()]
    for path in paths:
        status = path.stat()
        require((status.st_dev, status.st_ino) not in labels, "Non-label input/source aliases a suite annotation")


def require_gpu_envelope(manifest):
    """A GPU child must belong to the one recorded controller invocation."""
    suite = Path(manifest["suite_root"])
    path = suite / "EXECUTION_STARTED.json"
    require(path.is_file(), "GPU worker requires the recorded controller invocation")
    execution, prepared = read_json(path), read_json(suite / "SUITE_PREPARED.json")
    prepared["_verified_sha256"] = manifest["suite_prepared_sha256"]
    invocation = os.environ.get("INVOCATION_ID", "")
    require(len(invocation) == 32 and invocation == execution["invocation_id"]
            and os.getppid() == execution["pid"] and execution["retry_allowed"] is False
            and execution["service"] == prepared["service_name"]
            and execution["suite_prepared_sha256"] == manifest["suite_prepared_sha256"],
            "GPU worker outside the prepared controller invocation")
    group = next((line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::")), None)
    require(group == execution["control_group"] and group.endswith("/" + prepared["service_name"])
            and Path.cwd().resolve() == suite / "runtime", "GPU worker cgroup/root differs")
    from tools.evolution_allocation.queue import expected_command
    fields = (*RESOURCES, "Id", "ActiveState", "SubState", "WorkingDirectory", "InvocationID", "ControlGroup",
              "MainPID", "ExecMainPID", "ExecStart")
    response = subprocess.run(["systemctl", "--user", "show", prepared["service_name"],
        "--property=" + ",".join(fields)], text=True, capture_output=True, check=True, timeout=30)
    actual = dict(line.split("=", 1) for line in response.stdout.splitlines() if "=" in line)
    require(all(actual.get(key) == value for key, value in RESOURCES.items())
            and actual.get("Id") == prepared["service_name"] and actual.get("ActiveState") == "active"
            and actual.get("SubState") == "running" and actual.get("WorkingDirectory") == str(suite / "runtime")
            and actual.get("InvocationID") == invocation and actual.get("ControlGroup") == group
            and actual.get("MainPID") == actual.get("ExecMainPID") == str(execution["pid"])
            and execution["command"] == expected_command(suite, prepared)
            and "argv[]=" + " ".join(execution["command"]) + " ;" in actual.get("ExecStart", ""),
            "GPU child actual resource/controller envelope differs")
    return actual


def verify_inventory(root, inventory, *, exact=False):
    root = absolute_path(root)
    require(isinstance(inventory, dict) and inventory, "Missing file inventory")
    for name, expected in inventory.items():
        relative = Path(name)
        path = root / relative
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
                and path.resolve() == path and path.is_file() and digest_valid(expected)
                and sha256(path) == expected, "Artifact/source drift: " + name)
    if exact:
        actual = set()
        for path in root.rglob("*"):
            require(not path.is_symlink() and path.name not in ("__pycache__", ".pytest_cache")
                    and path.suffix not in (".pyc", ".pyo"), "Unrecorded runtime cache/symlink")
            if path.is_file():
                actual.add(str(path.relative_to(root)))
        require(actual == set(inventory), "Runtime contains extra or missing files")


def manifest_path(suite, dataset, stage):
    require(dataset in DATASETS and stage in ("gate", "full"), "Unknown cell")
    return Path(suite) / "manifests" / f"{dataset}.{stage}.json"


def expected_manifest(suite, prepared, dataset, stage, prepared_sha):
    require(dataset in DATASETS and stage in ("gate", "full"), "Unknown cell")
    entry = prepared["entries"][dataset]
    panel = read_json(Path(suite) / entry["panel_path"])
    if stage == "gate":
        panel = copy.deepcopy(panel)
        panel["inputs"] = panel["inputs"][:4]
        panel["ordered_ids"] = panel["ordered_ids"][:4]
        panel["full_split"] = False
        panel["images"] = 4
        panel["input_identity_sha256"] = canonical_hash(panel["inputs"])
    return dict(schema_version=1, algorithm=ALGORITHM, dataset=dataset, stage=stage, seed=0,
        permutation_seed=PERMUTATION_SEED, arms=ARMS.copy(), suite_root=str(suite),
        runtime_root=str(Path(suite) / "runtime"), suite_prepared_sha256=prepared_sha,
        source_sha256=prepared["runtime_hashes"], config=entry["config"], panel=panel,
        checkpoint=prepared["checkpoint"], output=str(Path(suite) / ("gates" if stage == "gate" else "full") / dataset),
        parent_proof_sha256=prepared["parent_proof_sha256"])


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS, "Wrong manifest schema")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["algorithm"] == ALGORITHM and manifest["dataset"] in DATASETS
            and manifest["stage"] in ("gate", "full"), "Wrong variant/cell identity")
    require(type(manifest["seed"]) is int and manifest["seed"] == 0
            and type(manifest["permutation_seed"]) is int and manifest["permutation_seed"] == PERMUTATION_SEED
            and manifest["arms"] == ARMS, "Seed/arm scope changed")
    suite = absolute_path(manifest["suite_root"])
    require(absolute_path(manifest["runtime_root"]) == suite / "runtime"
            and absolute_path(manifest["output"]) == suite / (
                "gates" if manifest["stage"] == "gate" else "full") / manifest["dataset"], "Wrong runtime/output binding")
    for key in ("suite_prepared_sha256", "parent_proof_sha256"):
        require(digest_valid(manifest[key]), "Missing manifest identity hash")
    cfg, panel = manifest["config"], manifest["panel"]
    require(cfg["dataset"] == ("COCOObjectDataset" if manifest["dataset"] == "coco_object" else "PascalContext60Dataset")
            and cfg["resize"] == cfg["patch_size"] == [224, 224]
            and cfg["class_extensions"] is True and cfg["trace_view_batch_size"] == 16
            and cfg["patch_stride"] == 112, "Fixed data/query/geometry changed")
    require(all(type(cfg[key]) is int and 1 <= cfg[key] <= limit for key, limit in
                (("batch_size", 32), ("steps", 64), ("trace_max_regions", 64)))
            and type(cfg["lr"]) in (int, float) and 1e-6 <= cfg["lr"] <= .04, "Invalid selected configuration")
    rows = panel["inputs"]
    count = 4 if manifest["stage"] == "gate" else POPULATIONS[manifest["dataset"]]
    require(len(rows) == count and panel["ordered_ids"] == [r["official_id"] for r in rows]
            and len(set(panel["ordered_ids"])) == count and len({r["image"] for r in rows}) == count
            and panel["full_split"] is (manifest["stage"] == "full"), "Population/order differs")
    require(panel["classes"] and panel["queries"] and type(panel["reduce_zero_label"]) is bool,
            "Missing category/query convention")
    mapping = panel["query_to_class"]
    require(isinstance(mapping, list) and len(mapping) == len(panel["queries"])
            and all(type(c) is int and 0 <= c < len(panel["classes"]) for c in mapping), "Invalid query mapping")
    for row in rows:
        absolute_path(row["image"])
        absolute_path(row["label"])
        require(digest_valid(row["image_sha256"]) and isinstance(row["official_id"], str)
                and row["official_id"] and isinstance(row["label_metadata"], list)
                and len(row["label_metadata"]) == 2
                and all(type(x) is int and x >= 0 for x in row["label_metadata"]), "Malformed input provenance")
    require(digest_valid(manifest["checkpoint"]["sha256"]), "Missing checkpoint digest")
    absolute_path(manifest["checkpoint"]["path"])


def verify_prepared(suite, expected, *, frozen=False):
    suite = absolute_path(suite)
    require(digest_valid(expected) and sha256(suite / "SUITE_PREPARED.json") == expected, "Prepared identity drift")
    prepared = read_json(suite / "SUITE_PREPARED.json")
    require(prepared["algorithm"] == ALGORITHM and prepared["datasets"] == list(DATASETS)
            and prepared["arms"] == ARMS and prepared["permutation_seed"] == PERMUTATION_SEED
            and prepared["model_seed"] == 0 and prepared["parameter_search"] is False
            and prepared["retry"] is False and prepared["exposure"] == EXPOSURE
            and prepared["parent_suite"] == str(PARENT) and prepared["parent_prepared_sha256"] == PARENT_SHA
            and set(prepared["entries"]) == set(DATASETS), "Prepared protocol/scope changed")
    require(isinstance(prepared["service_name"], str)
            and prepared["service_name"].startswith("trace-allocation-controls-")
            and prepared["service_name"].endswith(".service") and "/" not in prepared["service_name"]
            and prepared["resources"] == RESOURCES, "Service/resource binding differs")
    # Canonical manifests are the prepared metadata trust root. Stat all GT
    # identities before authenticating runtime/provenance payload contents.
    scope_manifest = read_json(manifest_path(suite, DATASETS[0], "full"))
    validate_manifest(scope_manifest)
    guard_global_inputs(scope_manifest)
    require(sha256(suite / "frozen_contract.md") == prepared["contract_sha256"]
            and sha256(suite / "PARENT_IMPORT.json") == prepared["parent_proof_sha256"], "Contract/parent proof drift")
    verify_inventory(suite / "runtime", prepared["runtime_hashes"], exact=True)
    verify_inventory(suite, prepared["parent_artifact_sha256"])
    original_prepared_path = suite / "parents/SUITE_PREPARED.json"
    require(sha256(original_prepared_path) == PARENT_SHA, "Wrong frozen parent source identity")
    original_prepared = read_json(original_prepared_path)
    require(all(prepared["runtime_hashes"].get(key) == value
                for key, value in original_prepared["runtime_hashes"].items()), "Inherited source continuity differs")
    proof = read_json(suite / "PARENT_IMPORT.json")
    require(proof["parent_prepared_sha256"] == PARENT_SHA and proof["verified_parent_full_outcomes"] == 32
            and proof["verified_parent_populations"] == 8 and digest_valid(proof["source_importer_sha256"]),
            "Unverified parent result source")
    originals = parent_sources()
    required = set(originals) | {f"parents/{d}/panel.json" for d in DATASETS}
    require(set(prepared["parent_artifact_sha256"]) == required, "Incomplete parent chain inventory")
    for local, original in originals.items():
        expected_original = proof["parent_metadata_sha256"].get(str(original))
        require(digest_valid(expected_original) and sha256(original) == expected_original
                == prepared["parent_artifact_sha256"][local], "Copied evidence differs from verified fixed parent read-set")
    for dataset, entry in prepared["entries"].items():
        require(entry["panel_path"] == f"parents/{dataset}/panel.json"
                and sha256(suite / entry["panel_path"]) == entry["panel_sha256"], "Input panel binding differs")
        original = read_json(suite / f"parents/{dataset}/prediction_protocol.json")
        selected = read_json(suite / f"parents/{dataset}/SELECTED_CONFIG_SEALED.json")
        require(entry["config"] == selected["config"] == original["manifest"]["config"]
                and prepared["checkpoint"] == original["manifest"]["checkpoint"]
                and read_json(suite / entry["panel_path"]) == original["manifest"]["panel"], "Parent panel/config/checkpoint differs")
        for stage in ("gate", "full"):
            manifest = read_json(manifest_path(suite, dataset, stage))
            validate_manifest(manifest)
            require(manifest == expected_manifest(suite, prepared, dataset, stage, expected), "Frozen cell manifest differs")
    if frozen:
        require(ROOT == suite / "runtime" and Path.cwd().resolve() == ROOT, "Execute only from frozen runtime")
    prepared["_verified_sha256"] = expected
    return prepared


def admission(args):
    path, output = absolute_path(args.manifest), absolute_path(args.output)
    require(digest_valid(args.manifest_sha256) and sha256(path) == args.manifest_sha256, "CLI manifest digest differs")
    manifest = read_json(path)
    validate_manifest(manifest)
    suite = Path(manifest["suite_root"])
    require(path == manifest_path(suite, manifest["dataset"], manifest["stage"])
            and output == Path(manifest["output"]), "CLI manifest/output scope differs")
    guard_global_inputs(manifest)
    verify_prepared(suite, manifest["suite_prepared_sha256"], frozen=True)
    for name, module in list(sys.modules.items()):
        if name == "trace_ovss" or name.startswith(("trace_ovss.", "tools.evolution", "tools.suite_inputs", "tools.trace_")):
            filename = getattr(module, "__file__", None)
            if filename:
                source = Path(filename).resolve()
                require(source.is_relative_to(ROOT) and str(source.relative_to(ROOT)) in manifest["source_sha256"],
                        "Cross-root/unbound executable import: " + name)
    from tools.evolution_tuning import worker as prior_worker
    prior_worker.guard_label_aliases(manifest, runtime=True)
    require(prior_worker.shared.checkpoint_identity(manifest["checkpoint"]["path"]) == manifest["checkpoint"], "Checkpoint drift")
    return manifest, output


def verify_barrier(path, manifest, output, manifest_sha):
    suite = Path(manifest["suite_root"])
    require(absolute_path(path) == suite / "ALL_FULL_PREDICTIONS_SEALED.json"
            and manifest["stage"] == "full", "Wrong barrier path/stage")
    labels = annotation_inodes(manifest)
    status = Path(path).stat()
    require((status.st_dev, status.st_ino) not in labels, "Barrier aliases annotation")
    barrier = read_json(path)
    require(barrier["complete"] is True and barrier["stage"] == "full"
            and barrier["gt_contents_opened"] is False
            and barrier["suite_prepared_sha256"] == manifest["suite_prepared_sha256"], "Incomplete or cross-suite barrier")
    links = barrier["prediction_seal_sha256"]
    require(set(links) == {str(suite / "full" / d) for d in DATASETS}, "Barrier must cover both exact full populations")
    from tools.evolution_allocation.worker import guard_output_labels, verify_prediction_seal
    for dataset in DATASETS:
        peer = suite / "full" / dataset
        require(peer.resolve() == peer, "Aliased barrier peer output")
        expected_path = manifest_path(suite, dataset, "full")
        expected_manifest_value = read_json(expected_path)
        guard_output_labels(peer, expected_manifest_value)
        require(sha256(peer / "PREDICTIONS_SEALED.json") == links[str(peer)], "Barrier peer prediction seal drift")
        actual = verify_prediction_seal(peer, expected_manifest_value, sha256(expected_path), hash_payload=True)
        require(actual == expected_manifest_value, "Barrier peer manifest differs")
    require(read_json(manifest_path(suite, manifest["dataset"], "full")) == manifest
            and sha256(manifest_path(suite, manifest["dataset"], "full")) == manifest_sha
            and str(output) in links, "Barrier own output/manifest differs")
    return sha256(path)
