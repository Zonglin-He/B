"""Detached, CPU-only allocation objectives; no diagnostics or runtime hooks.

``rho`` is the already computed C score, not a score estimated by this module.
Region means include all prompts. Areas count spatial tokens, not prompts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import operator
import random

import torch

from .trace.constants import EPS


CONTROLS = frozenset({"image_region", "episode_region", "image_token",
                      "episode_token", "image_shuffled"})


@dataclass(frozen=True)
class ShuffleCoverage:
    """Counts over active candidates/sources, not all proposed candidates.

    Eligible means at least two active candidates in the source. Actually
    moved counts nonfixed permutation positions, even if their scores tie.
    Score changes are reported separately. No derangement is forced.
    """

    active_candidates: int
    active_sources: int
    exchange_eligible_candidates: int
    exchange_eligible_sources: int
    actually_moved_candidates: int
    actually_moved_sources: int
    score_changed_candidates: int
    score_changed_sources: int


@dataclass(frozen=True)
class PreparedAllocation:
    """Owned detached cache; reuse only for the same shape/dtype on CPU.

    Each group is (rows, normalizer); rows are (crop, indices, q, weight).
    q is the assigned C score; weight is q or q * spatial token count.
    Treat cached tensors as read-only. Input term edits do not alter the cache.
    """

    control: str
    entropy_shape: tuple
    dtype: torch.dtype
    groups: tuple
    coverage: ShuffleCoverage | None


def _integer(value, name):
    if isinstance(value, bool) or (isinstance(value, torch.Tensor) and (
            value.device.type != "cpu" or value.ndim != 0 or value.dtype == torch.bool)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def keyed_source_permutation(size, *, seed, source_key):
    """Return destination-to-donor local indices using an isolated RNG.

    source_key must be a stable, nonempty dataset/source identity string,
    not an episode-local image_idx. Candidate order within a source must also
    be stable. This mapping is independent of other sources and global RNGs.
    """
    size, seed = _integer(size, "size"), _integer(seed, "seed")
    if size < 0 or not isinstance(source_key, str) or not source_key:
        raise ValueError("size must be nonnegative and source_key a nonempty string")
    payload = json.dumps(["trace-allocation-v1", seed, source_key],
                         ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))
    indices = list(range(size))
    rng.shuffle(indices)
    return tuple(indices)


def _validate_entropy(entropy):
    if not isinstance(entropy, torch.Tensor) or entropy.device.type != "cpu":
        raise ValueError("entropy must be a CPU tensor")
    if not entropy.is_floating_point() or entropy.ndim not in (3, 4):
        raise ValueError("entropy must be floating [crop,H,W] or [prompt,crop,H,W]")
    if any(size == 0 for size in entropy.shape):
        raise ValueError("entropy dimensions must be nonempty")
    if not torch.isfinite(entropy).all() or (entropy < 0).any():
        raise ValueError("entropy must be finite and nonnegative")


def prepare_allocation_control(entropy, region_terms, *, control="image_region",
                               permutation=None, seed=None, source_keys=None):
    """Freeze native region terms and allocation for one fixed control.

    Active set: Python-float rho > EPS and nonempty native indices/mask,
    before area multiplication or permutation. image_idx groups sources;
    batch_idx selects crops. Explicit indices take precedence over masks.

    image_shuffled requires exactly one of:
    * permutation: image_idx -> bijection of local active candidate positions,
      with destination i receiving q[permutation[image_idx][i]];
    * seed plus source_keys: image_idx -> unique stable nonempty string.
    Mappings must cover exactly active sources, including singletons. The
    caller owns provenance/order for an explicit mapping. All other controls
    reject shuffle options. Inputs and global RNGs are never modified.

    All denominators use the native > EPS guard after dtype conversion. Thus
    constant-area cancellation holds only when both denominators pass that
    guard; this matters for scores rounded to EPS or underflowed to zero.
    """
    _validate_entropy(entropy)
    if not isinstance(control, str) or control not in CONTROLS:
        raise ValueError(f"unknown allocation control: {control!r}")
    if control != "image_shuffled" and any(
            option is not None for option in (permutation, seed, source_keys)):
        raise ValueError("shuffle options require image_shuffled")
    by_image, crop_sources, episode_rows = {}, {}, []
    crops, height, width = entropy.shape[-3:]
    for term in region_terms:
        source = _integer(term["image_idx"], "image_idx")
        crop = _integer(term["batch_idx"], "batch_idx")
        if not 0 <= crop < crops:
            raise ValueError("batch_idx out of range")
        if crop in crop_sources and crop_sources[crop] != source:
            raise ValueError("a crop cannot belong to multiple source images")
        crop_sources[crop] = source
        rows = by_image.setdefault(source, [])
        raw_q = term["rho"]
        if isinstance(raw_q, torch.Tensor):
            if raw_q.device.type != "cpu" or raw_q.ndim != 0 or raw_q.is_complex():
                raise ValueError("rho must be a real CPU scalar")
            raw_q = raw_q.detach()
        try:
            q = float(raw_q)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rho must be a real scalar") from exc
        if not math.isfinite(q) or q < 0:
            raise ValueError("rho must be finite and nonnegative")
        indices = term.get("indices")
        if indices is None:
            mask = term["mask"]
            if (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu"
                    or tuple(mask.shape) != (height, width)
                    or not torch.isfinite(mask).all()
                    or not ((mask == 0) | (mask == 1)).all()):
                raise ValueError("mask must be a binary CPU [H,W] tensor")
            indices = mask.to(dtype=torch.bool).flatten().nonzero().flatten()
        if (not isinstance(indices, torch.Tensor) or indices.device.type != "cpu"
                or indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError("indices must be a 1D CPU integer tensor")
        indices = indices.detach().clone()
        if (indices < 0).any() or (indices >= height * width).any():
            raise ValueError("region index out of range")
        if indices.unique().numel() != indices.numel():
            raise ValueError("region indices must be unique spatial tokens")
        # Gate on the original scalar, exactly as native TRACE does.
        if q <= EPS or not indices.numel():
            continue
        score = entropy.new_tensor(q)
        if not torch.isfinite(score):
            raise ValueError("rho is not finite in entropy dtype")
        rows.append((crop, indices, score))
        episode_rows.append((crop, indices, score))
    by_image = {source: rows for source, rows in by_image.items() if rows}
    original_groups = tuple(by_image.values())

    coverage = None
    if control == "image_shuffled":
        if permutation is not None:
            if seed is not None or source_keys is not None:
                raise ValueError("pass permutation OR seed/source_keys")
            mapping = permutation
        else:
            if seed is None or not isinstance(source_keys, Mapping):
                raise ValueError("shuffle requires permutation or seed/source_keys")
            seed = _integer(seed, "seed")
            if set(source_keys) != set(by_image):
                raise ValueError("source_keys must cover exactly active sources")
            keys = list(source_keys.values())
            if (any(not isinstance(key, str) or not key for key in keys)
                    or len(set(keys)) != len(keys)):
                raise ValueError("stable source keys must be nonempty unique strings")
            mapping = {source: keyed_source_permutation(len(rows), seed=seed,
                       source_key=source_keys[source]) for source, rows in by_image.items()}
        if not isinstance(mapping, Mapping) or set(mapping) != set(by_image):
            raise ValueError("permutation must cover exactly active sources")
        eligible = eligible_sources = moved = moved_sources = changed = changed_sources = 0
        for source, rows in by_image.items():
            try:
                order = tuple(_integer(i, "permutation index") for i in mapping[source])
            except TypeError as exc:
                raise ValueError("permutation must be a local bijection") from exc
            if sorted(order) != list(range(len(rows))):
                raise ValueError("permutation must be a local bijection")
            n_moved = sum(i != donor for i, donor in enumerate(order))
            n_changed = sum(not torch.equal(rows[i][2], rows[donor][2])
                            for i, donor in enumerate(order))
            eligible += len(rows) if len(rows) > 1 else 0
            eligible_sources += int(len(rows) > 1)
            moved += n_moved
            moved_sources += int(n_moved > 0)
            changed += n_changed
            changed_sources += int(n_changed > 0)
            by_image[source] = [(crop, indices, rows[donor][2])
                                for (crop, indices, _), donor in zip(rows, order)]
        coverage = ShuffleCoverage(sum(map(len, by_image.values())), len(by_image),
                                   eligible, eligible_sources, moved, moved_sources,
                                   changed, changed_sources)

    groups = list(by_image.values())
    if control.startswith("episode_"):
        groups = [episode_rows] if episode_rows else []
    prepared = []
    for group_idx, rows in enumerate(groups):
        denominator = entropy.new_tensor(0.0)
        weighted = []
        for row_idx, (crop, indices, q) in enumerate(rows):
            weight = q * indices.numel() if control.endswith("_token") else q
            # Preserve native summation order/budget even after reassignment.
            budget = (original_groups[group_idx][row_idx][2]
                      if control == "image_shuffled" else weight)
            denominator = denominator + budget
            weighted.append((crop, indices, q, weight))
        if not torch.isfinite(denominator):
            raise ValueError("allocation denominator overflows entropy dtype")
        normalizer = torch.where(denominator.double() > EPS, denominator,
                                 torch.ones_like(denominator))
        prepared.append((tuple(weighted), normalizer))
    return PreparedAllocation(control, tuple(entropy.shape), entropy.dtype,
                              tuple(prepared), coverage)


def allocation_control_entropy(entropy, region_terms=(), *, control=None,
                               prepared=None, permutation=None, seed=None,
                               source_keys=None):
    """Evaluate a pure allocation loss, or reuse an explicitly prepared cache.

    Image arms average active source objectives; episode arms pool all active
    candidates. Empty input returns differentiable zero. C-score gradients are
    always stopped. The cache cannot be combined with new terms/options.
    """
    if prepared is None:
        prepared = prepare_allocation_control(
            entropy, region_terms, control="image_region" if control is None else control,
            permutation=permutation, seed=seed, source_keys=source_keys)
    else:
        _validate_entropy(entropy)
        if not isinstance(prepared, PreparedAllocation):
            raise ValueError("prepared must be a PreparedAllocation cache")
        if (tuple(region_terms) or any(x is not None for x in (permutation, seed, source_keys))
                or (control is not None and control != prepared.control)):
            raise ValueError("cache cannot be combined with terms or different options")
        if tuple(entropy.shape) != prepared.entropy_shape or entropy.dtype != prepared.dtype:
            raise ValueError("cache requires unchanged entropy shape and dtype")
    total = entropy.sum() * 0.0
    for rows, normalizer in prepared.groups:
        numerator = entropy.sum() * 0.0
        for crop, indices, _, weight in rows:
            if entropy.ndim == 4:
                mean = entropy[:, crop].flatten(1).index_select(1, indices).mean()
            else:
                mean = entropy[crop].flatten().index_select(0, indices).mean()
            numerator = numerator + weight * mean
        total = total + numerator / normalizer
    return total / len(prepared.groups) if prepared.groups else total
