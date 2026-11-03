"""Device-native, episode-local allocation objectives for the fixed C ablation.

Use ``prepare_allocation_control`` once after native C diagnostics, then pass
its cache to ``allocation_control_entropy`` on every update. Native diagnostics
already emit host scalar rho; device rho is rejected instead of silently read
back. Indices/masks may be on CPU or the entropy device. Preparation validates
values and exports small detached allocation receipts once; cached evaluation
only checks structure and tensor versions, without accelerator synchronization.

There is no module/model cache or reset hook. The engine's local prepared
variable must be discarded on reset, as in TRACE.adapt. Passing terms during
cached evaluation binds reuse to that exact episode container; an optional
identity-valued episode_key also guards callers that omit terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
import operator

import torch

from .allocation_controls import CONTROLS, ShuffleCoverage, keyed_source_permutation
from .image_balanced import (
    image_balanced_region_entropy,
    prepare_image_balanced_objective,
)
from .trace.constants import EPS
from .trace.objective import TraceObjectiveMixin

SHUFFLE_SEED = 20261029
__all__ = ["SHUFFLE_SEED", "PreparedAllocation", "prepare_allocation_control",
           "allocation_control_entropy", "allocation_control_metadata"]


@dataclass(frozen=True)
class CandidateAllocation:
    """Receipt for an active candidate; coefficient multiplies its mean h_R.

    Scores are the actual dtype-rounded q values. diagnostic_q retains the
    original host scalar used for admission. token_coefficient additionally
    divides by spatial area and prompt count, including the image average.
    """

    term_index: int
    donor_term_index: int
    source: int
    source_key: str | None
    crop: int
    area: int
    diagnostic_q: float
    original_q: float
    assigned_q: float
    weight: float
    normalizer: float
    coefficient: float
    token_coefficient: float


@dataclass(frozen=True)
class PreparedAllocation:
    """Owned, detached native groups, valid for one episode and tensor layout.

    groups contain ((crop, owned_indices, effective_weight), normalizer).
    All tensors stay on the entropy device. Treat this cache as read-only;
    ordinary in-place edits are detected by tensor version counters.
    """

    control: str
    entropy_shape: tuple
    dtype: torch.dtype
    device: torch.device
    groups: tuple
    records: tuple[CandidateAllocation, ...]
    coverage: ShuffleCoverage | None
    seed: int | None
    _terms: object = field(repr=False, compare=False)
    _episode_key: object = field(repr=False, compare=False)
    _seal: tuple = field(default=(), repr=False, compare=False)


def _integer(value, name):
    if isinstance(value, bool) or (isinstance(value, torch.Tensor) and (
            value.device.type != "cpu" or value.ndim != 0 or value.dtype == torch.bool)):
        raise ValueError(f"{name} must be a host integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a host integer") from exc


def _layout(entropy):
    if (not isinstance(entropy, torch.Tensor) or not entropy.is_floating_point()
            or entropy.ndim not in (3, 4) or any(n == 0 for n in entropy.shape)
            or entropy.device.type not in ("cpu", "cuda")):
        raise ValueError("entropy must be floating nonempty CPU/CUDA [crop,H,W] or [prompt,crop,H,W]")


def _input_tensor(value, device, name):
    if not isinstance(value, torch.Tensor) or value.device not in (torch.device("cpu"), device):
        raise ValueError(f"{name} must be a CPU tensor or on the entropy device")


def _owned_terms(entropy, region_terms):
    crops, height, width = entropy.shape[-3:]
    result, crop_sources = [], {}
    for term in region_terms:
        source = _integer(term["image_idx"], "image_idx")
        crop = _integer(term["batch_idx"], "batch_idx")
        if not 0 <= crop < crops:
            raise ValueError("batch_idx out of range")
        if crop in crop_sources and crop_sources[crop] != source:
            raise ValueError("a crop cannot belong to multiple source images")
        crop_sources[crop] = source
        q = term["rho"]
        if isinstance(q, torch.Tensor):
            if q.device.type != "cpu" or q.ndim != 0 or q.is_complex():
                raise ValueError("rho must be a real host scalar from native diagnostics")
            q = q.detach()
        try:
            q = float(q)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rho must be a real host scalar") from exc
        if not math.isfinite(q) or q < 0:
            raise ValueError("rho must be finite and nonnegative")
        indices = term.get("indices")
        if indices is None:
            mask = term["mask"]
            _input_tensor(mask, entropy.device, "mask")
            if (tuple(mask.shape) != (height, width) or not torch.isfinite(mask).all()
                    or not ((mask == 0) | (mask == 1)).all()):
                raise ValueError("mask must be binary [H,W]")
            indices = mask.to(device=entropy.device, dtype=torch.bool).flatten().nonzero().flatten()
        _input_tensor(indices, entropy.device, "indices")
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be a 1D integer tensor")
        if ((indices < 0).any() or (indices >= height * width).any()
                or indices.unique().numel() != indices.numel()):
            raise ValueError("indices must be unique spatial tokens in range")
        indices = indices.detach().to(device=entropy.device).clone()
        result.append(dict(image_idx=source, batch_idx=crop, rho=q, indices=indices))
    return result


def _fingerprint(cache):
    """Inspect only Python metadata, never tensor contents, on the hot path."""
    tensors = []
    if not isinstance(cache.groups, tuple):
        raise ValueError("malformed cache groups")
    for group in cache.groups:
        if not isinstance(group, tuple) or len(group) != 2:
            raise ValueError("malformed cache group")
        rows, normalizer = group
        if not isinstance(rows, tuple) or not rows:
            raise ValueError("malformed cache rows")
        for crop, indices, weight in rows:
            if (type(crop) is not int or not 0 <= crop < cache.entropy_shape[-3]
                    or not isinstance(indices, torch.Tensor) or indices.ndim != 1
                    or not indices.numel() or indices.dtype not in (torch.int32, torch.int64)):
                raise ValueError("malformed cache indices/crop")
            tensors.extend((indices, weight))
        tensors.append(normalizer)
        for scalar in (normalizer, *(row[2] for row in rows)):
            if (not isinstance(scalar, torch.Tensor) or scalar.shape != torch.Size([])
                    or scalar.dtype != cache.dtype):
                raise ValueError("malformed cache weight/normalizer dtype or shape")
    versions = []
    for tensor in tensors:
        if tensor.device != cache.device or tensor.requires_grad:
            raise ValueError("cache tensors must be detached on the entropy device")
        versions.append((id(tensor), tensor._version))
    return (cache.control, cache.entropy_shape, cache.dtype, cache.device,
            id(cache.groups), id(cache.records), id(cache.coverage), cache.seed,
            id(cache._terms), id(cache._episode_key), tuple(versions))


def prepare_allocation_control(entropy, region_terms, *, control="image_region",
                               seed=None, source_keys=None, episode_key=None):
    """Prepare the five-arm objective and a one-time detached metadata receipt.

    image_shuffled requires unique stable dataset/source strings mapped by
    image_idx, covering exactly active sources (including singletons). Its
    only admitted seed is 20261029; omitting seed selects that fixed seed.
    Other arms reject shuffle options and require no source_keys.

    Admission uses original host q > EPS before area weighting/permutation.
    Native double-precision EPS denominator guards and source/term summation
    order are retained, including dtype underflow and differentiable empties.
    Preparation can synchronize for validation/metadata, never offloads loss
    or entropy. Reuse with an ordinary (not inference_mode) autograd episode.
    """
    _layout(entropy)
    if not isinstance(control, str) or control not in CONTROLS:
        raise ValueError(f"unknown allocation control: {control!r}")
    if control != "image_shuffled" and (seed is not None or source_keys is not None):
        raise ValueError("shuffle options require image_shuffled")
    if not torch.isfinite(entropy).all() or (entropy < 0).any():
        raise ValueError("entropy must be finite and nonnegative at preparation")
    # Freeze iterators as well, while binding engine reuse to the caller's container.
    terms = _owned_terms(entropy, region_terms)
    by_source = {}
    active = []
    for i, term in enumerate(terms):
        group = by_source.setdefault(term["image_idx"], [])
        if term["rho"] > EPS and term["indices"].numel():
            group.append(i)
            active.append(i)
    by_source = {source: rows for source, rows in by_source.items() if rows}
    order = {i: i for i in active}
    if control == "image_shuffled":
        seed = SHUFFLE_SEED if seed is None else _integer(seed, "seed")
        if seed != SHUFFLE_SEED:
            raise ValueError("image_shuffled requires fixed seed 20261029")
        if not isinstance(source_keys, Mapping) or set(source_keys) != set(by_source):
            raise ValueError("source_keys must cover exactly active sources")
        keys = list(source_keys.values())
        if (any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("source_keys must be unique nonempty stable strings")
        for source, indices in by_source.items():
            donors = keyed_source_permutation(len(indices), seed=seed, source_key=source_keys[source])
            order.update((i, indices[donor]) for i, donor in zip(indices, donors))

    episode = control.startswith("episode_")
    native = TraceObjectiveMixin
    if episode:
        group = native.prepare_trace_objective(entropy, terms)
        groups = [group] if group[0] else []
        group_ids = [active] if active else []
    else:
        groups = prepare_image_balanced_objective(entropy, terms)
        group_ids = list(by_source.values())
    original_q = {i: row[2] for (rows, _), ids in zip(groups, group_ids)
                  for i, row in zip(ids, rows)}
    prepared, receipt_tensors, receipt_fields = [], [], []
    prompts = entropy.shape[0] if entropy.ndim == 4 else 1
    for (rows, denominator), ids in zip(groups, group_ids):
        weighted = []
        if control.endswith("_token"):
            denominator = entropy.new_tensor(0.0)
        for (crop, indices, q), i in zip(rows, ids):
            q = original_q[order[i]]
            weight = q * indices.numel() if control.endswith("_token") else q
            weighted.append((crop, indices, weight))
            if control.endswith("_token"):
                denominator = denominator + weight
        if control.endswith("_token"):
            denominator = torch.where(denominator.double() > EPS, denominator,
                                      torch.ones_like(denominator))
        prepared.append((tuple(weighted), denominator))
        for (crop, indices, weight), i in zip(weighted, ids):
            coefficient = weight / denominator / len(groups)
            receipt_tensors.append(torch.stack((original_q[i], original_q[order[i]], weight,
                                                denominator, coefficient,
                                                coefficient / indices.numel() / prompts)))
            receipt_fields.append((i, order[i], terms[i]["image_idx"], crop, indices.numel()))
    # This small one-time receipt transfer never contains entropy or a loss.
    values = torch.stack(receipt_tensors).detach().cpu().tolist() if receipt_tensors else []
    records = []
    for (i, donor, source, crop, area), numbers in zip(receipt_fields, values):
        if not all(math.isfinite(n) for n in numbers):
            raise ValueError("allocation scores or denominator overflow entropy dtype")
        records.append(CandidateAllocation(
            i, donor, source, source_keys[source] if source_keys is not None else None,
            crop, area, terms[i]["rho"], *numbers))
    coverage = None
    if control == "image_shuffled":
        eligible = {s for s, ids in by_source.items() if len(ids) > 1}
        moved = [r for r in records if r.term_index != r.donor_term_index]
        changed = [r for r in records if r.original_q != r.assigned_q]
        coverage = ShuffleCoverage(
            len(records), len(by_source), sum(len(by_source[s]) for s in eligible), len(eligible),
            len(moved), len({r.source for r in moved}), len(changed), len({r.source for r in changed}))
    cache = PreparedAllocation(control, tuple(entropy.shape), entropy.dtype, entropy.device,
                               tuple(prepared), tuple(records), coverage, seed,
                               region_terms, episode_key)
    object.__setattr__(cache, "_seal", _fingerprint(cache))
    return cache


def _validate_cache(entropy, cache, region_terms, control, seed, source_keys, episode_key):
    _layout(entropy)
    if not isinstance(cache, PreparedAllocation):
        raise ValueError("prepared must be a PreparedAllocation cache")
    if (tuple(entropy.shape) != cache.entropy_shape or entropy.dtype != cache.dtype
            or entropy.device != cache.device):
        raise ValueError("cache requires unchanged entropy shape, device and dtype")
    if ((control is not None and control != cache.control) or seed is not None
            or source_keys is not None):
        raise ValueError("cache cannot be combined with different control or shuffle options")
    if region_terms is not cache._terms and not (type(region_terms) is tuple and not region_terms):
        raise ValueError("cache belongs to different episode terms; prepare again after reset")
    if episode_key is not cache._episode_key:
        raise ValueError("cache episode_key mismatch; prepare again after reset")
    try:
        signature = _fingerprint(cache)
    except (TypeError, IndexError, AttributeError, RuntimeError) as exc:
        raise ValueError("malformed allocation cache") from exc
    if signature != cache._seal:
        raise ValueError("allocation cache was modified; prepare again")


def allocation_control_entropy(entropy, region_terms=(), *, control=None,
                               prepared=None, seed=None, source_keys=None,
                               episode_key=None):
    """Native tensor loss; cached calls never read tensor values onto the host.

    Engine hooks may supply the same region_terms container used to prepare.
    Cache-only calls omit terms; if bound to episode_key they must supply the
    identical key. A new episode must always prepare a new cache.
    """
    if prepared is None:
        prepared = prepare_allocation_control(
            entropy, region_terms, control="image_region" if control is None else control,
            seed=seed, source_keys=source_keys, episode_key=episode_key)
    else:
        _validate_cache(entropy, prepared, region_terms, control, seed, source_keys, episode_key)
    if prepared.control.startswith("episode_") and prepared.groups:
        return TraceObjectiveMixin.trace_region_normalized_entropy(
            entropy, (), prepared=prepared.groups[0])
    return image_balanced_region_entropy(entropy, (), prepared=prepared.groups)


def allocation_control_metadata(prepared):
    """Return a fresh JSON-ready receipt, with no tensor copies or device sync.

    coefficient is the effective multiplier of a candidate mean h_R, including
    the active-image average and native EPS guard, not an idealized budget.
    Empty/inactive candidates are excluded; term_index binds back to input.
    """
    if not isinstance(prepared, PreparedAllocation):
        raise ValueError("prepared must be a PreparedAllocation cache")
    if _fingerprint(prepared) != prepared._seal:
        raise ValueError("allocation cache was modified; prepare again")
    return dict(control=prepared.control, seed=prepared.seed,
                entropy_shape=list(prepared.entropy_shape), dtype=str(prepared.dtype),
                device=str(prepared.device), active_candidates=len(prepared.records),
                active_sources=len({r.source for r in prepared.records}),
                normalization_groups=len(prepared.groups),
                candidates=[asdict(r) for r in prepared.records],
                coverage=asdict(prepared.coverage) if prepared.coverage is not None else None)
