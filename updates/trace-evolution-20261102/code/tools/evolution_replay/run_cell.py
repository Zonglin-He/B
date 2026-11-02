"""Replay sealed initial diagnostics for product and pixel-normalized controls."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch


DATASETS = {"pc60": 5105, "coco_object": 5000}
VARIANTS = [("product", "product", None), ("pixel_normalized", "pixel_normalized", None)]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_artifacts(root, seal):
    root = Path(root).resolve()
    entries = seal.get("artifact_sha256", {})
    if not entries:
        raise RuntimeError("Empty parent artifact seal")
    for relative, expected in entries.items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise RuntimeError(f"Parent artifact hash mismatch: {relative}")


def suite_root(cell):
    for ancestor in Path(cell).resolve().parents:
        if (ancestor / "MANIFEST_SEALED.json").is_file():
            return ancestor
    raise RuntimeError("Parent cell has no sealed suite manifest")


def verify_parent(cell, package_root, shared):
    cell, package_root = Path(cell).resolve(), Path(package_root).resolve()
    suite = suite_root(cell)
    suite_seal = read(suite / "SUITE_SEALED.json")  # Fail closed while parent is running.
    verify_artifacts(suite, suite_seal)
    manifest = read(suite / "manifest.json")
    if digest(suite / "manifest.json") != read(suite / "MANIFEST_SEALED.json")["manifest_sha256"]:
        raise RuntimeError("Parent manifest hash mismatch")
    seal_relative = str((cell / "PREDICTIONS_SEALED.json").relative_to(suite))
    if suite_seal["artifact_sha256"].get(seal_relative) != digest(cell / "PREDICTIONS_SEALED.json"):
        raise RuntimeError("Parent prediction seal is outside the completed suite chain")
    protocol = shared.verify_prediction_seal(cell)
    if (protocol["dataset"] not in DATASETS or protocol["seed"] != 0 or not protocol["full_split"]
            or len(protocol["inputs"]) != DATASETS[protocol["dataset"]]):
        raise RuntimeError("Only predeclared complete seed-0 PC60/COCO Object cells are eligible")
    batch = protocol["config"]["batch_size"]
    if len(protocol["inputs"]) < batch:
        raise RuntimeError("Parent has no complete first episode")
    if ["trace", "trace", None] not in protocol["variants"]:
        raise RuntimeError("Parent lacks the sealed TRACE reference")
    original_root = Path(protocol["package_root"]).resolve()
    sources = {}
    for original, expected in protocol["source_sha256"].items():
        relative = Path(original).resolve().relative_to(original_root)
        path = package_root / relative
        if digest(path) != expected or manifest["runtime_hashes"].get(str(relative)) != expected:
            raise RuntimeError(f"Replay source differs from parent: {relative}")
        sources[str(path)] = expected
    # Check the COMPLETE package and unchanged old runner tree, not just a subset.
    for prefix in ("trace_ovss", "tools/evolution"):
        expected = {name: value for name, value in manifest["runtime_hashes"].items()
                    if name.startswith(prefix + "/")}
        actual = {str(path.relative_to(package_root)): digest(path)
                  for path in (package_root / prefix).rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")}
        if actual != expected:
            raise RuntimeError(f"Complete source tree changed: {prefix}")
    parent_seal = read(cell / "PREDICTIONS_SEALED.json")
    for episode, start in enumerate(range(0, len(protocol["inputs"]), batch)):
        for suffix in ("json", "npz"):
            if f"episodes/{episode:06d}.{suffix}" not in parent_seal["artifact_sha256"]:
                raise RuntimeError("Parent candidate cache is not completely sealed")
        receipt = read(cell / "episodes" / f"{episode:06d}.json")
        if (receipt["image_indices"] != list(range(start, min(start + batch, len(protocol["inputs"]))))
                or receipt["dataset_indices"] != receipt["image_indices"]
                or receipt["candidate_budget"] != protocol["config"]["trace_max_regions"]):
            raise RuntimeError("Parent episode boundaries or candidate budget differ")
    return protocol, dict(parent_cell=str(cell), parent_suite=str(suite),
        parent_prediction_seal_sha256=digest(cell / "PREDICTIONS_SEALED.json"),
        parent_suite_seal_sha256=digest(suite / "SUITE_SEALED.json"),
        parent_manifest_sha256=digest(suite / "manifest.json")), sources


def load_cache(cell, episode, config, candidate_digest):
    start = episode * config["batch_size"]
    sealed = read(Path(cell) / "PREDICTIONS_SEALED.json")["artifact_sha256"]
    for suffix in ("json", "npz"):
        relative = f"episodes/{episode:06d}.{suffix}"
        if digest(Path(cell) / relative) != sealed.get(relative):
            raise RuntimeError("Parent cache changed after prediction sealing")
    receipt = read(Path(cell) / "episodes" / f"{episode:06d}.json")
    with np.load(Path(cell) / "episodes" / f"{episode:06d}.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in
                  ("masks", "image_index", "patch_index", "query_index", "confidence", "T", "C", "q")}
    count = len(arrays["T"])
    if any(len(value) != count for value in arrays.values()) or count != receipt["candidate_count"]:
        raise RuntimeError("Parent cache array lengths differ")
    if arrays["masks"].dtype != np.bool_ or arrays["masks"].ndim != 3:
        raise RuntimeError("Parent token masks must be boolean grids")
    for name in ("T", "C", "q", "confidence"):
        values = arrays[name]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise RuntimeError("Parent support values are invalid")
    if not np.array_equal(arrays["q"], np.minimum(arrays["T"], arrays["C"])):
        raise RuntimeError("Parent q differs from min(T,C)")
    stats = []
    owners = {}
    for i in range(count):
        image_idx = int(arrays["image_index"][i]) - start
        patch_idx = int(arrays["patch_index"][i])
        if image_idx < 0 or image_idx >= len(receipt["image_indices"]) or patch_idx < 0:
            raise RuntimeError("Invalid parent image/patch ownership")
        mask = arrays["masks"][i]
        if not mask.any():
            raise RuntimeError("Empty parent candidate")
        owners[image_idx] = owners.get(image_idx, 0) + 1
        if owners[image_idx] > config["trace_max_regions"]:
            raise RuntimeError("Parent per-image candidate budget exceeded")
        stats.append(dict(batch_idx=patch_idx, image_idx=image_idx, class_idx=int(arrays["query_index"][i]),
            confidence=float(arrays["confidence"][i]), target_support=float(arrays["T"][i]),
            counterfactual_support=float(arrays["C"][i]), quality=float(arrays["q"][i]),
            reliability=float(arrays["q"][i]), region=mask))
    if candidate_digest(stats) != receipt["candidate_sha256"]:
        raise RuntimeError("Reconstructed candidate fingerprint differs from parent")
    return stats, receipt


def make_replay_class(base, parent_cell, config):
    from tools.evolution.policies import candidate_digest, make_fixed_candidate_class
    from trace_ovss.method.trace.constants import EPS
    from trace_ovss.method.trace.indexing import pack_region_indices
    fixed = make_fixed_candidate_class(base)

    class ReplayTRACE(fixed):
        replay_episode = 0
        replay_policy = "trace"

        @torch.no_grad()
        def diagnose(self, images, valid_mask=None, image_ids=None):
            self.reset()
            stats, receipt = load_cache(parent_cell, self.replay_episode, config, candidate_digest)
            if image_ids is None:
                raise RuntimeError("Cache replay requires explicit source-image ownership")
            for row in stats:
                patch = row["batch_idx"]
                if (patch >= len(images) or int(image_ids[patch]) != row["image_idx"]
                        or not 0 <= row["class_idx"] < len(self.classes)):
                    raise RuntimeError("Cache patch/query ownership differs from current batch")
            self.fixed_stats = stats
            self.fixed_digest = receipt["candidate_sha256"]
            self.fixed_image = images
            self.fixed_indices = pack_region_indices([row["region"] for row in stats], images.device)
            self.replay_episode += 1
            return stats

        def select_policy(self, policy, *, permutation_seed=0, episode=0):
            if policy not in ("trace", "product", "pixel_normalized"):
                raise ValueError("Unsupported cache-replay control")
            if candidate_digest(self.fixed_stats) != self.fixed_digest:
                raise RuntimeError("Parent candidate cache mutated")
            self.replay_policy = policy
            q = np.array([row["quality"] for row in self.fixed_stats], dtype=np.float64)
            if policy == "trace":
                values = q
            elif policy == "product":
                values = np.array([row["target_support"] * row["counterfactual_support"] for row in self.fixed_stats])
            else:
                values = q * np.array([row["region"].sum() for row in self.fixed_stats])
            self.fixed_weights = np.where(q > EPS, values, 0.0)
            self.fixed_donors = np.arange(len(q), dtype=np.int64)

        @torch.no_grad()
        def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
            if images is not self.fixed_image or candidate_digest(self.fixed_stats) != self.fixed_digest:
                raise RuntimeError("Cached episode tensor or candidates changed")
            terms = [dict(batch_idx=row["batch_idx"], indices=indices, rho=float(weight))
                     for row, indices, weight in zip(self.fixed_stats, self.fixed_indices, self.fixed_weights)
                     if row["quality"] > EPS]
            self.last_region_terms = terms
            self.last_candidate_stats = self.fixed_stats
            return {"region_terms": terms}

        def prepare_trace_objective(self, entropy_per_pixel, region_terms):
            if self.replay_policy != "product":
                return super().prepare_trace_objective(entropy_per_pixel, region_terms)
            # Keep parent q>EPS admission, without a second EPS cut on T*C.
            weights = entropy_per_pixel.new_tensor([term["rho"] for term in region_terms])
            denominator = entropy_per_pixel.new_tensor(0.0)
            prepared = []
            for term, coefficient in zip(region_terms, weights):
                prepared.append((int(term["batch_idx"]), term["indices"], coefficient))
                denominator = denominator + coefficient
            return prepared, torch.where(denominator > 0, denominator, torch.ones_like(denominator))

    return ReplayTRACE


def exact_gate(method, data, config, mapping, cell, shared, output):
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    from tools.evolution.policies import state_digest
    batch_size = config["batch_size"]
    batch = shared.image_collate([data[i] for i in range(batch_size)])
    images = batch["img_patches"].to(method.device)
    method.replay_episode = 0
    method.diagnose(images, build_patch_valid_mask(batch["meta"], config["patch_size"], config["patch_stride"], method.device),
                    build_patch_image_ids(batch["meta"], method.device))
    method.reset()
    initial = shared.restored_state(method)
    parent_initial = read(Path(cell) / "episodes/000000.json")["reset_assertions"]["trace"]
    if initial != parent_initial:
        raise RuntimeError("Cached TRACE gate initial model/optimizer hashes differ")
    method.select_policy("trace")
    losses = method.adapt(images)
    predictions = shared.label_predictions(method, images, batch["meta"], config, mapping)
    changed = 0
    for index, prediction in enumerate(predictions):
        parent = np.load(Path(cell) / "predictions/trace" / f"{index:06d}.npy", allow_pickle=False)
        if parent.shape != prediction.shape:
            raise RuntimeError("TRACE gate prediction shape differs")
        changed += int(np.count_nonzero(parent != prediction))
    report = dict(passed=changed == 0, changed_label_pixels=changed, images=batch_size,
                  initial_assertions=initial, loss=losses, candidate_sha256=method.fixed_digest,
                  final_trainable_sha256=state_digest({name: p for name, p in method.model.named_parameters() if p.requires_grad}),
                  gt_contents_opened=False, tolerance=0)
    shared.write_json(Path(output) / "replay_gate.json", report)
    if not report["passed"]:
        raise RuntimeError("Cached TRACE exact-label gate failed; no control predictions started")
    method.reset()
    shared.restored_state(method)
    method.replay_episode = 0


def archive_links(parent_cell, output, protocol, shared):
    from trace_ovss.method.trace.constants import EPS
    for episode, start in enumerate(range(0, len(protocol["inputs"]), protocol["config"]["batch_size"])):
        archive = Path(output) / "episodes" / f"{episode:06d}.npz"
        with np.load(archive, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        arrays["effective_weights"] = arrays.pop("rho")
        arrays["area"] = arrays["masks"].reshape(len(arrays["q"]), -1).sum(1) if len(arrays["q"]) else np.empty(0, dtype=np.int64)
        arrays["parent_active"] = arrays["q"] > EPS
        np.savez_compressed(archive, **arrays)
        raw = Path(parent_cell) / "episodes" / f"{episode:06d}.npz"
        shutil.copyfile(raw, archive.with_name(f"{episode:06d}.parent.npz"))
        shared.write_json(archive.with_name(f"{episode:06d}.cachelink.json"), dict(
            parent_npz=str(raw), parent_npz_sha256=digest(raw),
            parent_json_sha256=digest(raw.with_suffix(".json")),
            parent_candidate_sha256=read(raw.with_suffix(".json"))["candidate_sha256"],
            q="bounded initial min(T,C); unchanged", effective_weights="product or q*token-area; not bounded reliability"))


def load_runtime(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from tools.evolution import run_cell as shared
    if not Path(inspect.getfile(shared)).resolve().is_relative_to(Path(root).resolve()):
        raise RuntimeError("Shared runner imported outside --package-root")
    return shared


def predict(args, shared):
    runtime = shared.configure_deterministic_runtime()
    protocol, binding, sources = verify_parent(args.parent_cell, args.package_root, shared)
    if runtime != protocol["deterministic_runtime"]:
        raise RuntimeError("Deterministic execution settings differ from parent")
    from trace_ovss.data.datasets import SegmentationPatchDataset
    from trace_ovss.method.two_view import TwoViewTRACE
    from trace_ovss.configs import VISION_OUTPUTS
    from tools.evolution.policies import state_digest
    from tools.suite_inputs import validate_discovery, validate_population
    config = copy.deepcopy(protocol["config"])
    data = shared.image_only_dataset_class(SegmentationPatchDataset)(config["dataset"], config["data_dir"],
        resize=config["resize"], patch_size=config["patch_size"], patch_stride=config["patch_stride"], include_auxiliary=False)
    validate_discovery(data)
    validate_population(protocol["dataset"], "evaluation", None, len(data))
    if ([str(image) for image, _ in data.samples] != [row["image"] for row in protocol["inputs"]]
            or [str(label) for _, label in data.samples] != [row["label"] for row in protocol["inputs"]]):
        raise RuntimeError("Discovered source-image/annotation order differs from parent")
    queries = data.vocabulary.extensions if config["class_extensions"] else data.vocabulary.classes
    mapping = data.vocabulary.extension_to_class if config["class_extensions"] else None
    vocab_hash = hashlib.sha256(json.dumps([queries, mapping]).encode()).hexdigest()
    if vocab_hash != protocol["panel"]["vocabulary_sha256"]:
        raise RuntimeError("Vocabulary differs from parent")
    for index, row in enumerate(protocol["inputs"]):
        for offset, path in enumerate(data.samples[index]):
            if [path.stat().st_size, path.stat().st_mtime_ns] != protocol["panel"]["file_metadata"][2 * index + offset]:
                raise RuntimeError("Input metadata differs from parent")
        if digest(row["image"]) != row["image_sha256"]:
            raise RuntimeError("Input image hash differs from parent")
    data.expected_image_hashes = {row["image"]: row["image_sha256"] for row in protocol["inputs"]}
    checkpoint = shared.checkpoint_identity(protocol["checkpoint"]["path"])
    if checkpoint != protocol["checkpoint"] or checkpoint != read(args.parent_cell / "checkpoint_identity.json"):
        raise RuntimeError("Checkpoint identity differs from parent")
    if args.output.exists():
        raise FileExistsError("Replay uses a new cell directory, no retries or overwrite")
    args.output.mkdir(parents=True)
    shutil.copyfile(args.parent_cell / "expected_panel.json", args.output / "expected_panel.json")
    own_sources = {str(path.resolve()): digest(path) for path in Path(__file__).parent.glob("*.py")}
    protocol = {**protocol, "variants": VARIANTS, "parent_replay": binding,
                "source_sha256": {**sources, **own_sources}, "reference_verification": False,
                "predeclared_datasets": list(DATASETS),
                "selection_basis": "PC60 and COCO Object fixed before their parent full-control outcomes; no GT selection",
                "control_definitions": {"product": "T*C on parent q>EPS candidates; no additional product EPS pruning",
                    "pixel_normalized": "q*area on parent q>EPS candidates; weighted token-normalized entropy"},
                "timing_scope": "Cached diagnostic loading plus per-policy updates; no diagnostic forwards; not standalone cost"}
    shared.write_json(args.output / "prediction_protocol.json", protocol)
    shared.write_json(args.output / "checkpoint_identity.json", checkpoint)
    shared.write_json(args.output / "progress.json", dict(status="initializing", images_completed=0))
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    cls = make_replay_class(TwoViewTRACE, args.parent_cell, config)
    method = cls(classes=queries, lr=config["lr"], steps=config["steps"], trace_max_regions=config["trace_max_regions"],
                 trace_view_batch_size=config["trace_view_batch_size"], vision_outputs=VISION_OUTPUTS,
                 device="cuda", backbone=checkpoint["path"])
    initial = read(args.parent_cell / "initial_state.json")
    if (state_digest(method.model.state_dict()) != initial["model_sha256"]
            or state_digest(method.optimizer.state_dict()) != initial["optimizer_sha256"]):
        raise RuntimeError("Full initial model or optimizer identity differs from parent")
    shared.restored_state(method)
    shared.write_json(args.output / "initial_state.json", {**initial, "deterministic_runtime": runtime,
        "environment": {**initial["environment"], "python": sys.version, "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(method.device), **runtime},
        "parent_initial_state_sha256": digest(args.parent_cell / "initial_state.json"),
        "initial_identity_matches_parent": True})
    initial_rng = shared.rng_state(method.device)
    exact_gate(method, data, config, mapping, args.parent_cell, shared, args.output)
    shared.restore_rng(initial_rng, method.device)
    shared.execute_predictions(method, data, list(range(len(data))), config, mapping, VARIANTS, args.output)
    archive_links(args.parent_cell, args.output, protocol, shared)
    if shared.deterministic_runtime_state() != runtime:
        raise RuntimeError("Runtime flags changed")
    if any(digest(path) != expected for path, expected in protocol["source_sha256"].items()):
        raise RuntimeError("Source changed during replay")
    if any(digest(row["image"]) != row["image_sha256"] for row in protocol["inputs"]):
        raise RuntimeError("Input images changed during replay")
    if digest(args.parent_cell / "PREDICTIONS_SEALED.json") != binding["parent_prediction_seal_sha256"]:
        raise RuntimeError("Parent prediction seal changed")
    if digest(Path(binding["parent_suite"]) / "SUITE_SEALED.json") != binding["parent_suite_seal_sha256"]:
        raise RuntimeError("Parent suite seal changed")
    shared.write_json(args.output / "prediction_result.json", dict(status="predicted", images=len(data),
        dataset=protocol["dataset"], seed=protocol["seed"], full_split=True, parent_replay=binding))
    shared.seal_predictions(args.output, protocol)
    seal = read(args.output / "PREDICTIONS_SEALED.json")
    seal["artifact_sha256"]["replay_gate.json"] = digest(args.output / "replay_gate.json")
    shared.write_json(args.output / "PREDICTIONS_SEALED.json", seal)
    shared.write_json(args.output / "progress.json", dict(status="predictions_sealed", images_completed=len(data)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-cell", "package-root", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--phase", choices=("predict", "score", "run"), default="run")
    args = parser.parse_args(argv)
    shared = load_runtime(args.package_root)
    if args.phase in ("predict", "run"):
        # The parent suite must be finished; a busy GPU lock fails without retry.
        read(suite_root(args.parent_cell) / "SUITE_SEALED.json")
        lock = Path("/home/wwww/hzl/results/reviewer_followup_gpu.lock")
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            predict(args, shared)
    if args.phase in ("score", "run"):
        verify_parent(args.parent_cell, args.package_root, shared)
        protocol = shared.verify_prediction_seal(args.output)
        binding = protocol["parent_replay"]
        if (str(args.parent_cell) != binding["parent_cell"]
                or digest(args.parent_cell / "PREDICTIONS_SEALED.json") != binding["parent_prediction_seal_sha256"]):
            raise RuntimeError("Scoring parent binding differs")
        if any(digest(path) != expected for path, expected in protocol["source_sha256"].items()):
            raise RuntimeError("Scoring source differs from sealed replay")
        shared.score(args.output, dataset=protocol["dataset"], seed=protocol["seed"], data_root=protocol["data_root"])


if __name__ == "__main__":
    main()
