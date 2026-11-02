"""CPU-only runtime/oracle identity tests; no models or accelerator work."""

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from trace_ovss.method import allocation_controls as oracle
from trace_ovss.method import allocation_runtime as runtime
from trace_ovss.method.image_balanced import image_balanced_region_entropy
from trace_ovss.method.trace.constants import EPS
from trace_ovss.method.trace.objective import TraceObjectiveMixin


ARMS = sorted(oracle.CONTROLS)
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
KEYS = {701: "pc60/validation/a.jpg", -9: "pc60/validation/b.jpg", 88: "pc60/validation/c.jpg"}


def term(crop, source, q, indices):
    return dict(batch_idx=crop, image_idx=source, rho=q,
                indices=torch.tensor(indices, dtype=torch.long))


def terms():
    # Source order is first occurrence, even when that first term is inactive.
    return [term(3, 88, 0., [0]), term(0, 701, .17, [0, 2, 5]),
            term(1, -9, .73, [1]), term(2, 701, .39, [1, 2, 4, 5]),
            term(3, 88, .29, [0, 3]), term(1, -9, .13, [0, 2, 3, 5]),
            term(0, 701, EPS, [1]), term(3, 88, .75, []),
            term(4, 999, EPS / 2, [1, 2])]


def entropy(prompts=False, dtype=torch.float64):
    shape = (3, 5, 2, 3) if prompts else (5, 2, 3)
    return torch.rand(shape, dtype=dtype, generator=torch.Generator().manual_seed(7),
                      requires_grad=True)


def options(arm, keys=KEYS):
    return dict(seed=runtime.SHUFFLE_SEED, source_keys=keys) if arm == "image_shuffled" else {}


def equal_loss_grad(actual, expected, x, exact=True):
    ga = torch.autograd.grad(actual, x, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, x, retain_graph=True)[0]
    if exact:
        assert torch.equal(actual, expected)
        assert torch.equal(ga, ge)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ga, ge, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("prompts", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_all_arms_cpu_oracle_loss_and_gradient_identity(arm, prompts, dtype):
    x, rows = entropy(prompts, dtype), terms()
    rows[3] = dict(batch_idx=2, image_idx=701, rho=.39,
                   mask=torch.tensor([[0, 1, 1], [0, 1, 1]]))
    rows[1]["mask"] = torch.zeros(2, 3, dtype=torch.bool)  # indices take precedence
    expected = oracle.allocation_control_entropy(x, rows, control=arm, **options(arm))
    cache = runtime.prepare_allocation_control(x, rows, control=arm, **options(arm))
    for value in (runtime.allocation_control_entropy(x, rows, control=arm, **options(arm)),
                  runtime.allocation_control_entropy(x, prepared=cache),
                  runtime.allocation_control_entropy(x, rows, prepared=cache)):
        equal_loss_grad(value, expected, x)
    oracle_cache = oracle.prepare_allocation_control(x, rows, control=arm, **options(arm))
    assert cache.coverage == oracle_cache.coverage
    for (actual, an), (wanted, wn) in zip(cache.groups, oracle_cache.groups, strict=True):
        assert torch.equal(an, wn)
        for (crop, indices, weight), (wcrop, wi, _, ww) in zip(actual, wanted, strict=True):
            assert crop == wcrop and torch.equal(indices, wi) and torch.equal(weight, ww)


@pytest.mark.parametrize("prompts", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [3, 19, 27])
def test_native_baselines_bitwise_with_nonbinary_weights(prompts, dtype, seed):
    x, rows = entropy(prompts, dtype), terms()
    generator = torch.Generator().manual_seed(seed)
    for row in rows:
        row["rho"] = float(torch.rand((), generator=generator, dtype=torch.float64))
    for arm, native in (("image_region", image_balanced_region_entropy),
                        ("episode_region", TraceObjectiveMixin.trace_region_normalized_entropy)):
        cache = runtime.prepare_allocation_control(x, rows, control=arm)
        equal_loss_grad(runtime.allocation_control_entropy(x, prepared=cache), native(x, rows), x)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("q", [0., EPS / 2, EPS, EPS * (1 + 1e-9), EPS * (1 + 1e-6), .5])
def test_eps_activity_precedes_area_and_shuffle_with_native_empty_behavior(arm, dtype, q):
    x = entropy(True, dtype)
    rows = [term(0, 1, q, [0, 1]), term(1, 2, .8, []),
            term(2, 1, EPS, list(range(6)))]
    opts = options(arm, {1: "split/a"} if q > EPS else {})
    cache = runtime.prepare_allocation_control(x, rows, control=arm, **opts)
    expected = oracle.allocation_control_entropy(x, rows, control=arm, **opts)
    equal_loss_grad(runtime.allocation_control_entropy(x, prepared=cache), expected, x)
    assert len(cache.records) == int(q > EPS)
    empty = runtime.prepare_allocation_control(x, [], control=arm, **options(arm, {}))
    value = runtime.allocation_control_entropy(x, prepared=empty)
    assert value.requires_grad and value == 0
    assert not torch.autograd.grad(value, x)[0].count_nonzero()


@pytest.mark.parametrize("arm", ARMS)
def test_receipts_are_actual_effective_coefficients_with_unequal_areas(arm):
    x, rows = entropy(True), terms()
    cache = runtime.prepare_allocation_control(x, rows, control=arm, **options(arm))
    receipt = runtime.allocation_control_metadata(cache)
    json.dumps(receipt, allow_nan=False)
    actual = runtime.allocation_control_entropy(x, prepared=cache)
    reconstructed = x.sum() * 0
    expected_gradient = torch.zeros_like(x)
    for record in receipt["candidates"]:
        row = rows[record["term_index"]]
        selected = x[:, record["crop"]].flatten(1)[:, row["indices"]]
        assert record["area"] == row["indices"].numel()
        assert record["source"] == row["image_idx"]
        assert record["original_q"] == row["rho"]
        assert record["assigned_q"] == rows[record["donor_term_index"]]["rho"]
        expected_weight = record["assigned_q"] * (record["area"] if arm.endswith("_token") else 1)
        assert record["weight"] == expected_weight
        assert record["coefficient"] == record["weight"] / record["normalizer"] / len(cache.groups)
        reconstructed = reconstructed + record["coefficient"] * selected.mean()
        flat = expected_gradient[:, record["crop"]].flatten(1)
        flat[:, row["indices"]] += record["token_coefficient"]
    equal_loss_grad(actual, reconstructed, x, exact=False)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], expected_gradient,
                               rtol=1e-12, atol=1e-12)
    # Receipts are fresh copies, with no mutable aliases into the cache.
    receipt["candidates"][0]["area"] = -1
    assert runtime.allocation_control_metadata(cache)["candidates"][0]["area"] > 0


def test_native_eps_guard_receipts_are_not_idealized_unit_budgets():
    x = torch.tensor([[[2., 4.]]], requires_grad=True)
    rows = [term(0, 1, EPS * (1 + 1e-9), [0, 1])]
    cache = runtime.prepare_allocation_control(x, rows)
    row = runtime.allocation_control_metadata(cache)["candidates"][0]
    assert row["diagnostic_q"] > EPS and row["original_q"] < EPS
    assert row["normalizer"] == 1. and row["coefficient"] < EPS
    equal_loss_grad(runtime.allocation_control_entropy(x, prepared=cache),
                    image_balanced_region_entropy(x, rows), x)
    assert runtime.allocation_control_entropy(x, rows, control="image_token") == 3.


def test_four_factorial_arms_are_distinct_and_constant_area_collapses():
    x = torch.tensor([[[1., 3., 9., 13.]], [[20., 22., 26., 30.]]],
                     dtype=torch.float64, requires_grad=True)
    rows = [term(0, 1, .25, [0]), term(0, 1, .5, [1, 2, 3]), term(1, 2, .125, [0, 1])]
    factorial = [arm for arm in ARMS if arm != "image_shuffled"]
    assert len({float(runtime.allocation_control_entropy(x, rows, control=a)) for a in factorial}) == 4
    rows = [term(0, 1, .125, [0, 1]), term(0, 1, .375, [1, 2]), term(1, 2, .75, [2, 3])]
    for scope in ("image", "episode"):
        equal_loss_grad(runtime.allocation_control_entropy(x, rows, control=scope + "_region"),
                        runtime.allocation_control_entropy(x, rows, control=scope + "_token"), x)
    for suffix in ("region", "token"):
        equal_loss_grad(runtime.allocation_control_entropy(x, rows[:2], control="image_" + suffix),
                        runtime.allocation_control_entropy(x, rows[:2], control="episode_" + suffix), x)


def test_shuffle_permutation_multisets_budgets_coverage_and_rng_isolation():
    x, rows = entropy(), terms()
    py_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
    shuffled = runtime.prepare_allocation_control(x, rows, control="image_shuffled", source_keys=KEYS)
    repeated = runtime.prepare_allocation_control(x, rows, control="image_shuffled", **options("image_shuffled"))
    assert shuffled.records == repeated.records
    assert shuffled.coverage == oracle.prepare_allocation_control(
        x, rows, control="image_shuffled", **options("image_shuffled")).coverage
    native = runtime.prepare_allocation_control(x, rows)
    for source in KEYS:
        records = [r for r in shuffled.records if r.source == source]
        old = [r for r in native.records if r.source == source]
        permutation = oracle.keyed_source_permutation(len(records), seed=20261029, source_key=KEYS[source])
        assert [r.assigned_q for r in records] == [r.original_q for r in [records[i] for i in permutation]]
        assert sorted(r.assigned_q for r in records) == sorted(r.original_q for r in records)
        assert [r.normalizer for r in records] == [r.normalizer for r in old]
        assert sum(r.coefficient for r in records) == pytest.approx(1 / 3)
        subset = [r for r in rows if r["image_idx"] == source]
        isolated = runtime.prepare_allocation_control(x, subset, control="image_shuffled",
                                                      source_keys={source: KEYS[source]})
        assert [r.assigned_q for r in isolated.records] == [r.assigned_q for r in records]
    # Stable source keys, not local image_idx, determine donors.
    remap = {source: i + 100 for i, source in enumerate(KEYS)}
    remapped = [dict(row, image_idx=remap.get(row["image_idx"], 999)) for row in rows]
    other = runtime.prepare_allocation_control(x, remapped, control="image_shuffled",
                                               source_keys={remap[s]: k for s, k in KEYS.items()})
    assert [r.assigned_q for r in shuffled.records] == [r.assigned_q for r in other.records]
    reversed_rows = [r for source in reversed(KEYS) for r in rows if r["image_idx"] == source]
    reversed_cache = runtime.prepare_allocation_control(x, reversed_rows, control="image_shuffled", source_keys=KEYS)
    for source in KEYS:
        assert [r.assigned_q for r in reversed_cache.records if r.source == source] == [
            r.assigned_q for r in shuffled.records if r.source == source]
    assert random.getstate() == py_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


def test_shuffle_ties_singletons_and_fixed_points_not_forced_to_move():
    x = entropy()
    # Discover a stable source identity whose ordinary permutation is a swap.
    swap = next(f"dataset/{i}" for i in range(100) if oracle.keyed_source_permutation(
        2, seed=20261029, source_key=f"dataset/{i}") == (1, 0))
    fixed = next(f"dataset/{i}" for i in range(100) if oracle.keyed_source_permutation(
        2, seed=20261029, source_key=f"dataset/{i}") == (0, 1))
    rows = [term(0, 1, .25, [0]), term(0, 1, .25, [1]), term(1, 2, .5, [0])]
    cache = runtime.prepare_allocation_control(x, rows, control="image_shuffled",
                                               source_keys={1: swap, 2: "dataset/singleton"})
    assert cache.coverage == oracle.ShuffleCoverage(3, 2, 2, 1, 2, 1, 0, 0)
    cache = runtime.prepare_allocation_control(x, rows, control="image_shuffled",
                                               source_keys={1: fixed, 2: "dataset/singleton"})
    assert cache.coverage == oracle.ShuffleCoverage(3, 2, 2, 1, 0, 0, 0, 0)


@pytest.mark.parametrize("arm", ARMS)
def test_detached_owned_cache_no_mutation_and_no_cross_episode_reuse(arm):
    x, rows = entropy(), terms()
    rho = torch.tensor([row["rho"] for row in rows], dtype=torch.float64, requires_grad=True)
    for row, q in zip(rows, rho):
        row["rho"] = q
    snapshots = [row["indices"].clone() for row in rows]
    before_x = x.detach().clone()
    cache = runtime.prepare_allocation_control(x, rows, control=arm, **options(arm))
    saved = runtime.allocation_control_entropy(x, prepared=cache)
    saved.backward()
    assert rho.grad is None and x.grad is not None
    assert torch.equal(x, before_x)
    for row, snapshot in zip(rows, snapshots):
        assert torch.equal(row["indices"], snapshot)
    for group, normalizer in cache.groups:
        assert not normalizer.requires_grad and normalizer.device == x.device
        for _, indices, weight in group:
            assert not weight.requires_grad and weight.device == x.device
            assert all(indices.data_ptr() != row["indices"].data_ptr() for row in rows)
    for row in rows:
        row["indices"].fill_(5)
    with torch.no_grad():
        rho.fill_(.9)
    assert torch.equal(runtime.allocation_control_entropy(x, prepared=cache), saved)
    with pytest.raises(ValueError, match="episode terms"):
        runtime.allocation_control_entropy(x, list(rows), prepared=cache)
    fresh_rows = terms()
    fresh_rows[1]["rho"] = .9
    fresh = runtime.prepare_allocation_control(x, fresh_rows, control=arm, **options(arm))
    assert fresh is not cache and fresh.records != cache.records


def test_explicit_episode_key_for_cache_only_calls():
    x, rows, key = entropy(), terms(), object()
    cache = runtime.prepare_allocation_control(x, rows, episode_key=key)
    for supplied in (None, object()):
        with pytest.raises(ValueError, match="episode_key"):
            runtime.allocation_control_entropy(x, prepared=cache, episode_key=supplied)
    equal_loss_grad(runtime.allocation_control_entropy(x, prepared=cache, episode_key=key),
                    image_balanced_region_entropy(x, rows), x)
    other = runtime.prepare_allocation_control(x, rows, episode_key=object())
    with pytest.raises(ValueError, match="episode_key"):
        runtime.allocation_control_entropy(x, prepared=other, episode_key=key)


@pytest.mark.parametrize("arm", ARMS)
def test_cached_hot_path_and_metadata_never_copy_or_read_tensor_scalars(monkeypatch, arm):
    x, rows = entropy(True), terms()
    cache = runtime.prepare_allocation_control(x, rows, control=arm, **options(arm))

    def forbidden(*args, **kwargs):
        raise AssertionError("hot path attempted host value access or finite reduction")

    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "tolist", "__bool__", "__float__", "numpy"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch, "isfinite", forbidden)
        value = runtime.allocation_control_entropy(x, rows, prepared=cache)
        torch.autograd.grad(value, x)
        runtime.allocation_control_metadata(cache)


@pytest.mark.parametrize("change", ["shape", "dtype", "device", "control", "groups", "records", "coverage"])
def test_malformed_cache_rejected_without_value_sync(change):
    x = entropy()
    cache = runtime.prepare_allocation_control(x, terms())
    modifications = dict(shape=dict(entropy_shape=(1, 2, 3)), dtype=dict(dtype=torch.float32),
                         device=dict(device=torch.device("meta")), control=dict(control="episode_region"),
                         groups=dict(groups=()), records=dict(records=()), coverage=dict(coverage=object()))
    broken = replace(cache, **modifications[change])
    with pytest.raises(ValueError, match="cache"):
        runtime.allocation_control_entropy(x, prepared=broken)


@pytest.mark.parametrize("target", ["indices", "weight", "normalizer"])
def test_mutated_cache_tensor_rejected(target):
    x = entropy()
    cache = runtime.prepare_allocation_control(x, terms())
    rows, norm = cache.groups[0]
    tensor = dict(indices=rows[0][1], weight=rows[0][2], normalizer=norm)[target]
    tensor.fill_(0)
    with pytest.raises(ValueError, match="modified"):
        runtime.allocation_control_entropy(x, prepared=cache)
    with pytest.raises(ValueError, match="modified"):
        runtime.allocation_control_metadata(cache)


def test_invalid_cache_structure_options_and_input_layouts():
    x, rows = entropy(), terms()
    cache = runtime.prepare_allocation_control(x, rows)
    for changed in (x.float(), x[:1], torch.empty(x.shape, dtype=x.dtype, device="meta")):
        with pytest.raises(ValueError, match="entropy|cache"):
            runtime.allocation_control_entropy(changed, prepared=cache)
    for opts in (dict(control="episode_region"), dict(seed=20261029), dict(source_keys={})):
        with pytest.raises(ValueError, match="cache"):
            runtime.allocation_control_entropy(x, prepared=cache, **opts)
    for bad in ((), object()):
        with pytest.raises(ValueError, match="PreparedAllocation"):
            runtime.allocation_control_entropy(x, prepared=bad)
    for groups in ([cache.groups[0]], (((), cache.groups[0][1]),),
                   ((((-1, torch.tensor([0]), torch.tensor(.5)),), torch.tensor(1.)),),
                   (((cache.groups[0][0][0],), torch.ones(1, dtype=x.dtype)),)):
        with pytest.raises(ValueError, match="cache"):
            runtime.allocation_control_entropy(x, prepared=replace(cache, groups=groups))


@pytest.mark.parametrize("control", ["bad", "", None, [], 42])
def test_bad_controls(control):
    with pytest.raises(ValueError, match="unknown allocation control"):
        runtime.prepare_allocation_control(entropy(), [], control=control)


@pytest.mark.parametrize("q", [-1., float("nan"), float("inf"), torch.tensor([.5]),
                              torch.tensor(1j), torch.empty((), device="meta"), "bad"])
def test_invalid_scores(q):
    with pytest.raises(ValueError, match="rho"):
        runtime.prepare_allocation_control(entropy(), [term(0, 1, q, [])])


@pytest.mark.parametrize("field,value", [
    ("batch_idx", -1), ("batch_idx", 5), ("image_idx", 1.5), ("image_idx", True),
    ("indices", torch.tensor([0.])), ("indices", torch.tensor([[0]])),
    ("indices", torch.tensor([-1])), ("indices", torch.tensor([6])),
    ("indices", torch.tensor([0, 0])), ("indices", torch.empty(1, device="meta")),
])
def test_invalid_terms(field, value):
    row = term(0, 1, .5, [0])
    row[field] = value
    with pytest.raises(ValueError):
        runtime.prepare_allocation_control(entropy(), [row])


@pytest.mark.parametrize("mask", [torch.ones(6), torch.full((2, 3), .5),
                                 torch.full((2, 3), float("nan")), torch.empty(2, 3, device="meta")])
def test_invalid_masks(mask):
    with pytest.raises(ValueError, match="mask"):
        runtime.prepare_allocation_control(entropy(), [dict(batch_idx=0, image_idx=1, rho=.5, mask=mask)])


def test_missing_schema_and_conflicting_crop_sources():
    for field in ("batch_idx", "image_idx", "rho", "indices"):
        row = term(0, 1, .5, [0])
        del row[field]
        with pytest.raises(KeyError):
            runtime.prepare_allocation_control(entropy(), [row])
    with pytest.raises(ValueError, match="multiple source"):
        runtime.prepare_allocation_control(entropy(), [term(0, 1, .5, [0]), term(0, 2, .5, [1])])


@pytest.mark.parametrize("x", [None, torch.ones(2, 3), torch.ones(0, 2, 3),
                               torch.ones(1, 2, 3, dtype=torch.long), torch.ones(1, 1, 1, 2, 3),
                               torch.full((1, 2, 3), -1.), torch.full((1, 2, 3), float("inf")),
                               torch.full((1, 2, 3), float("nan"))])
def test_bad_entropy_at_prepare_only(x):
    with pytest.raises(ValueError, match="entropy"):
        runtime.prepare_allocation_control(x, [])


@pytest.mark.parametrize("opts", [dict(), dict(seed=3, source_keys={1: "a"}),
    dict(seed=True, source_keys={1: "a"}), dict(source_keys={}), dict(source_keys={1: ""}),
    dict(source_keys={1: 5}), dict(source_keys={1: "a", 2: "b"})])
def test_invalid_shuffle_provenance(opts):
    with pytest.raises(ValueError):
        runtime.prepare_allocation_control(entropy(), [term(0, 1, .5, [0])],
                                           control="image_shuffled", **opts)


def test_duplicate_keys_nonshuffle_options_and_dtype_overflow():
    x, rows = entropy(), [term(0, 1, .5, [0]), term(1, 2, .25, [0])]
    with pytest.raises(ValueError, match="unique"):
        runtime.prepare_allocation_control(x, rows, control="image_shuffled", source_keys={1: "a", 2: "a"})
    for arm in ARMS:
        if arm == "image_shuffled":
            continue
        for opts in (dict(seed=20261029), dict(source_keys={})):
            with pytest.raises(ValueError, match="shuffle options"):
                runtime.prepare_allocation_control(x, rows, control=arm, **opts)
    for arm, q in (("image_region", 1e40), ("image_token", 3e38)):
        with pytest.raises(ValueError, match="overflow"):
            runtime.prepare_allocation_control(x.float(), [term(0, 1, q, [0, 1, 2])], control=arm)


def test_native_functions_are_reused_directly(monkeypatch):
    calls = []
    prepare_native = runtime.prepare_image_balanced_objective
    loss_native = runtime.image_balanced_region_entropy

    def prepare_spy(*args, **kwargs):
        calls.append("prepare")
        return prepare_native(*args, **kwargs)

    def loss_spy(*args, **kwargs):
        calls.append("loss")
        return loss_native(*args, **kwargs)

    monkeypatch.setattr(runtime, "prepare_image_balanced_objective", prepare_spy)
    monkeypatch.setattr(runtime, "image_balanced_region_entropy", loss_spy)
    x, rows = entropy(), terms()
    cache = runtime.prepare_allocation_control(x, rows)
    runtime.allocation_control_entropy(x, prepared=cache)
    assert calls == ["prepare", "loss"]


def test_normal_package_import_without_model_loading_or_accelerator_initialization():
    code = '''
import torch
assert not torch.cuda.is_initialized()
def forbidden(*args, **kwargs):
    raise AssertionError("model loading or accelerator initialization attempted")
torch.nn.Module.__init__ = forbidden
torch.cuda._lazy_init = forbidden
torch.load = forbidden
import trace_ovss.method.allocation_runtime as runtime
import trace_ovss.models as models
models.load_dense_ovss = forbidden
x = torch.ones(1, 2, 3, requires_grad=True)
cache = runtime.prepare_allocation_control(x, [])
runtime.allocation_control_entropy(x, prepared=cache).backward()
assert not torch.cuda.is_initialized()
print("normal package import, CPU loss and backward: OK")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=60, check=True)
    assert "CPU loss and backward: OK" in result.stdout
    assert not torch.cuda.is_initialized()
