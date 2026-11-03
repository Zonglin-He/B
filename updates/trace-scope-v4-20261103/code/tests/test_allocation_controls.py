"""CPU-only tensor contracts, without importing any model/runtime modules."""

import ast
import copy
import importlib.util
from pathlib import Path
import random
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "trace_ovss" / "method"


def _load_cpu_modules():
    # Bypass package __init__ files, which import the model runtime. Execute
    # native objective functions unchanged, excluding model-dependent classes.
    imports_before = set(sys.modules)
    namespace = "_allocation_controls_cpu_test"
    for name, path in ((namespace, METHOD), (namespace + ".trace", METHOD / "trace")):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    loaded = {}
    for name, path in (("trace.constants", METHOD / "trace/constants.py"),
                       ("trace.objective", METHOD / "trace/objective.py"),
                       ("allocation_controls", METHOD / "allocation_controls.py")):
        spec = importlib.util.spec_from_file_location(namespace + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    native_path = METHOD / "image_balanced.py"
    source = ast.parse(native_path.read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"prepare_image_balanced_objective",
                                   "image_balanced_region_entropy"}]
    assert len(functions) == 2
    native = {"TraceObjectiveMixin": loaded["trace.objective"].TraceObjectiveMixin}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(native_path), "exec"), native)
    model_imports = {name for name in set(sys.modules) - imports_before
                     if name.startswith("trace_ovss.models")}
    return loaded, native, model_imports


MODULES, NATIVE, MODEL_IMPORTS = _load_cpu_modules()
controls = MODULES["allocation_controls"]
EPS = MODULES["trace.constants"].EPS
old_objective = MODULES["trace.objective"].TraceObjectiveMixin.trace_region_normalized_entropy
native_image = NATIVE["image_balanced_region_entropy"]
prepare = controls.prepare_allocation_control
loss = controls.allocation_control_entropy
ARMS = sorted(controls.CONTROLS)
FACTORIAL = [arm for arm in ARMS if arm != "image_shuffled"]


def term(crop, source, q, indices):
    return dict(batch_idx=crop, image_idx=source, rho=q,
                indices=torch.tensor(indices, dtype=torch.long))


def fixture_terms():
    return [term(0, 701, .125, [0, 2, 5]), term(1, -9, .75, [1]),
            term(2, 701, .375, [1, 2, 4, 5]), term(3, 88, .25, [0, 3]),
            term(1, -9, .125, [0, 2, 3, 5]), term(0, 701, EPS, [1]),
            term(3, 88, .75, []), term(4, 999, 0., [0]),
            term(4, 999, EPS / 2, [1, 2])]


def entropy_tensor(prompts=False, dtype=torch.float64):
    shape = (3, 5, 2, 3) if prompts else (5, 2, 3)
    return (torch.arange(torch.tensor(shape).prod().item(), dtype=dtype)
            .reshape(shape).div(8).requires_grad_())


def reference(entropy, terms, arm):
    # Independent tensor formula: select spatial tokens, average prompts,
    # then form one weighted vector quotient per requested normalization unit.
    means, weights, sources = [], [], []
    for row in terms:
        q = float(row["rho"].detach()) if isinstance(row["rho"], torch.Tensor) else row["rho"]
        indices = row.get("indices")
        if indices is None:
            indices = row["mask"].flatten().nonzero().flatten()
        if q <= EPS or not indices.numel():
            continue
        pixels = entropy[:, row["batch_idx"]] if entropy.ndim == 4 else entropy[row["batch_idx"]]
        means.append(pixels.reshape(-1, entropy.shape[-2] * entropy.shape[-1])[:, indices].mean())
        weights.append(q * (len(indices) if arm.endswith("token") else 1))
        sources.append(row["image_idx"] if arm.startswith("image") else 0)
    if not means:
        return entropy.sum() * 0
    h, w = torch.stack(means), entropy.new_tensor(weights)
    quotients = []
    for source in dict.fromkeys(sources):
        select = torch.tensor([s == source for s in sources])
        denominator = w[select].sum()
        divisor = denominator if float(denominator.double()) > EPS else torch.ones_like(denominator)
        quotients.append((h[select] * w[select]).sum() / divisor)
    return torch.stack(quotients).mean()


def assert_loss_and_grad(actual, expected, entropy, *, exact=False):
    ga = torch.autograd.grad(actual, entropy, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, entropy, retain_graph=True)[0]
    if exact:
        assert torch.equal(actual, expected)
        assert torch.equal(ga, ge)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ga, ge, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("prompts", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_identity_irregular_multiple_crops_and_sources(prompts, dtype):
    entropy, terms = entropy_tensor(prompts, dtype), fixture_terms()
    # Include an inactive early term: source order is first occurrence, not
    # first active occurrence. Mask regions and index precedence stay native.
    terms.insert(0, term(3, 88, 0., [1]))
    terms[3] = dict(batch_idx=2, image_idx=701, rho=.375,
                    mask=torch.tensor([[0, 1, 1], [0, 1, 1]]))
    terms[1]["mask"] = torch.zeros(2, 3, dtype=torch.bool)
    expected = native_image(entropy, terms)
    cache = prepare(entropy, terms)
    for actual in (loss(entropy, terms), loss(entropy, prepared=cache)):
        assert_loss_and_grad(actual, expected, entropy, exact=True)
    # Episode grouping must keep original interleaved term order too.
    assert_loss_and_grad(loss(entropy, terms, control="episode_region"),
                         old_objective(entropy, terms), entropy, exact=True)


@pytest.mark.parametrize("prompts", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [3, 8, 19, 27])
def test_native_identity_nonbinary_scores_and_summation_order(prompts, dtype, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (3, 5, 2, 3) if prompts else (5, 2, 3)
    entropy = torch.rand(shape, dtype=dtype, generator=generator, requires_grad=True)
    terms = fixture_terms()
    for row, q in zip(terms, torch.rand(len(terms), dtype=torch.float64, generator=generator)):
        row["rho"] = float(q)
    for arm, native in (("image_region", native_image), ("episode_region", old_objective)):
        assert_loss_and_grad(loss(entropy, terms, control=arm), native(entropy, terms),
                             entropy, exact=True)


@pytest.mark.parametrize("prompts", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("q", [0., EPS / 2, EPS, EPS * (1 + 1e-9), EPS * (1 + 1e-6), .5])
def test_native_identity_eps_rounding_underflow_and_empty(prompts, dtype, q):
    entropy = entropy_tensor(prompts, dtype)
    terms = [term(0, 7, q, [0]), term(1, 70, .8, []), term(2, 7, EPS, [1])]
    for rows in (terms, []):
        expected = native_image(entropy, rows)
        assert_loss_and_grad(loss(entropy, rows), expected, entropy, exact=True)
        cache = prepare(entropy, rows)
        assert len(cache.groups) == int(bool(rows) and q > EPS)
        assert_loss_and_grad(loss(entropy, prepared=cache), expected, entropy, exact=True)


@pytest.mark.parametrize("arm", FACTORIAL)
@pytest.mark.parametrize("prompts", [False, True])
def test_factorial_independent_formulas_losses_and_gradients(arm, prompts):
    entropy, terms = entropy_tensor(prompts), fixture_terms()
    expected = reference(entropy, terms, arm)
    cache = prepare(entropy, terms, control=arm)
    assert_loss_and_grad(loss(entropy, terms, control=arm), expected, entropy)
    assert_loss_and_grad(loss(entropy, prepared=cache), expected, entropy)


def test_four_arms_are_distinct_with_unequal_areas_and_source_counts():
    entropy = torch.tensor([[[1., 3., 9., 13.]], [[20., 22., 26., 30.]]],
                           dtype=torch.float64, requires_grad=True)
    terms = [term(0, 1, .25, [0]), term(0, 1, .5, [1, 2, 3]),
             term(1, 2, .125, [0, 1])]
    values = [loss(entropy, terms, control=arm).item() for arm in FACTORIAL]
    assert len(set(values)) == 4
    for arm in FACTORIAL:
        assert_loss_and_grad(loss(entropy, terms, control=arm), reference(entropy, terms, arm), entropy)


@pytest.mark.parametrize("scope", ["image", "episode"])
def test_constant_area_collapse(scope):
    entropy = entropy_tensor(True)
    terms = [term(0, 3, .125, [0, 2]), term(2, 3, .375, [1, 4]),
             term(1, 4, .75, [1, 5])]
    assert_loss_and_grad(loss(entropy, terms, control=scope + "_region"),
                         loss(entropy, terms, control=scope + "_token"), entropy, exact=True)


@pytest.mark.parametrize("scope", ["image", "episode"])
def test_near_eps_native_guard_limits_constant_area_cancellation(scope):
    # This positive Python scalar passes admission but rounds below EPS in
    # float32. Do not silently repair native normalization to force collapse.
    entropy = torch.tensor([[[2., 4.]]], requires_grad=True)
    rows = [term(0, 1, EPS * (1 + 1e-9), [0, 1])]
    region = loss(entropy, rows, control=scope + "_region")
    token = loss(entropy, rows, control=scope + "_token")
    assert_loss_and_grad(region, native_image(entropy, rows), entropy, exact=True)
    assert region.item() < EPS * 4
    assert token.item() == 3.
    assert len(prepare(entropy, rows, control=scope + "_region").groups) == 1


@pytest.mark.parametrize("arm", FACTORIAL)
def test_single_source_image_episode_collapse_and_singleton_region_token(arm):
    entropy = entropy_tensor(True)
    terms = [term(0, 44, .125, [0]), term(2, 44, .75, [1, 2, 3, 5])]
    counterpart = ("episode" if arm.startswith("image") else "image") + "_" + arm.split("_")[1]
    assert_loss_and_grad(loss(entropy, terms, control=arm),
                         loss(entropy, terms, control=counterpart), entropy, exact=True)
    singleton = terms[:1]
    assert_loss_and_grad(loss(entropy, singleton, control=arm),
                         entropy[:, 0, 0, 0].mean(), entropy, exact=True)


@pytest.mark.parametrize("arm", ARMS)
def test_active_set_precedes_area_or_shuffle_and_empty_sources(arm):
    entropy = entropy_tensor()
    terms = [term(0, 1, EPS, list(range(6))), term(1, 2, .5, []),
             term(2, 3, EPS / 2, list(range(6))), term(3, 4, .25, [0, 1])]
    opts = {"permutation": {4: (0,)}} if arm == "image_shuffled" else {}
    cache = prepare(entropy, terms, control=arm, **opts)
    assert len(cache.groups) == 1 and len(cache.groups[0][0]) == 1
    assert_loss_and_grad(loss(entropy, prepared=cache), entropy[3, 0, :2].mean(), entropy)
    empty_opts = {"permutation": {}} if arm == "image_shuffled" else {}
    zero = loss(entropy, terms[:3], control=arm, **empty_opts)
    assert zero.item() == 0 and zero.requires_grad
    assert not torch.autograd.grad(zero, entropy)[0].count_nonzero()


@pytest.mark.parametrize("arm", ARMS)
def test_scores_are_detached_cache_is_owned_and_inputs_not_mutated(arm):
    entropy = entropy_tensor()
    scores = torch.tensor([.125, .75, .375, .25, .125], requires_grad=True)
    terms = fixture_terms()[:5]
    for row, q in zip(terms, scores):
        row["rho"] = q
    snapshots = [{key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                  for key, value in row.items()} for row in terms]
    before_entropy = entropy.detach().clone()
    opts = {"permutation": {701: (1, 0), -9: (1, 0), 88: (0,)}} if arm == "image_shuffled" else {}
    cache = prepare(entropy, terms, control=arm, **opts)
    direct = loss(entropy, terms, control=arm, **opts)
    cached = loss(entropy, prepared=cache)
    assert_loss_and_grad(cached, direct, entropy, exact=True)
    cached.backward()
    assert scores.grad is None and entropy.grad is not None
    for rows, normalizer in cache.groups:
        assert not normalizer.requires_grad
        for _, indices, q, weight in rows:
            assert not indices.requires_grad and not q.requires_grad and not weight.requires_grad
            assert all(indices.data_ptr() != row["indices"].data_ptr() for row in terms)
    for row, snapshot in zip(terms, snapshots):
        for key in row:
            if isinstance(row[key], torch.Tensor):
                assert torch.equal(row[key].detach(), snapshot[key])
            else:
                assert row[key] == snapshot[key]
    assert torch.equal(entropy.detach(), before_entropy)
    for row in terms:
        row["indices"].fill_(5)
    with torch.no_grad():
        scores.fill_(.9)
    assert torch.equal(loss(entropy, prepared=cache), cached)


def test_shuffle_mapping_invariants_coverage_and_native_region_means():
    entropy, terms = entropy_tensor(True), fixture_terms()
    mapping = {701: [1, 0], -9: [0, 1], 88: [0]}
    original_mapping = copy.deepcopy(mapping)
    shuffled = prepare(entropy, terms, control="image_shuffled", permutation=mapping)
    native = prepare(entropy, terms)
    expected_terms = [dict(row) for row in terms]
    expected_terms[0]["rho"], expected_terms[2]["rho"] = terms[2]["rho"], terms[0]["rho"]
    assert_loss_and_grad(loss(entropy, prepared=shuffled), native_image(entropy, expected_terms),
                         entropy, exact=True)
    for (actual, an), (old, on) in zip(shuffled.groups, native.groups):
        assert sorted(q.item() for _, _, q, _ in actual) == sorted(q.item() for _, _, q, _ in old)
        assert torch.equal(an, on)
        assert sum(q.item() for _, _, q, _ in actual) == sum(q.item() for _, _, q, _ in old)
        for a, b in zip(actual, old):
            assert a[0] == b[0] and torch.equal(a[1], b[1])
    assert shuffled.coverage == controls.ShuffleCoverage(5, 3, 4, 2, 2, 1, 2, 1)
    assert mapping == original_mapping


def test_shuffle_singletons_fixed_points_and_tied_scores_are_not_forced():
    entropy = entropy_tensor()
    terms = [term(0, 1, .25, [0]), term(0, 1, .25, [1]), term(2, 4, .75, [1])]
    tied = prepare(entropy, terms, control="image_shuffled", permutation={1: [1, 0], 4: [0]})
    assert tied.coverage == controls.ShuffleCoverage(3, 2, 2, 1, 2, 1, 0, 0)
    identity = prepare(entropy, terms, control="image_shuffled", permutation={1: [0, 1], 4: [0]})
    assert identity.coverage == controls.ShuffleCoverage(3, 2, 2, 1, 0, 0, 0, 0)
    assert_loss_and_grad(loss(entropy, prepared=tied), native_image(entropy, terms), entropy, exact=True)
    # Some ordinary keyed shuffles are identity. No retry-to-derangement loop.
    assert any(controls.keyed_source_permutation(2, seed=s, source_key="dataset/a") == (0, 1)
               for s in range(20))


def test_keyed_shuffle_determinism_source_isolation_and_no_global_rng_mutation():
    entropy, terms = entropy_tensor(), fixture_terms()
    python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
    keys = {701: "split/a", -9: "split/b", 88: "split/c"}
    kwargs = dict(control="image_shuffled", seed=93, source_keys=keys)
    first, second = prepare(entropy, terms, **kwargs), prepare(entropy, terms, **kwargs)
    assert torch.equal(loss(entropy, prepared=first), loss(entropy, prepared=second))
    explicit = {source: controls.keyed_source_permutation(n, seed=93, source_key=keys[source])
                for source, n in ((701, 2), (-9, 2), (88, 1))}
    assert torch.equal(loss(entropy, prepared=first),
                       loss(entropy, terms, control="image_shuffled", permutation=explicit))
    for group, source in zip(first.groups, keys):
        isolated = prepare(entropy, [r for r in terms if r["image_idx"] == source],
                           control="image_shuffled", seed=93, source_keys={source: keys[source]})
        assert [r[2].item() for r in group[0]] == [r[2].item() for r in isolated.groups[0][0]]
    reversed_sources = [r for source in reversed(keys) for r in terms if r["image_idx"] == source]
    reordered = prepare(entropy, reversed_sources, **kwargs)
    for old, new in zip(first.groups, reversed(reordered.groups)):
        assert [r[2].item() for r in old[0]] == [r[2].item() for r in new[0]]
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


@pytest.mark.parametrize("bad", ["unknown", "IMAGE_REGION", "", None, 1, []])
def test_reject_unknown_controls(bad):
    with pytest.raises(ValueError, match="unknown allocation control"):
        prepare(entropy_tensor(), [], control=bad)


@pytest.mark.parametrize("q", [-1., -EPS, float("nan"), float("inf"), -float("inf"),
                              torch.tensor([.5]), torch.tensor(1 + 2j), "bad"])
def test_reject_invalid_scores_even_if_region_empty(q):
    with pytest.raises(ValueError, match="rho"):
        prepare(entropy_tensor(), [term(0, 1, q, [])])


@pytest.mark.parametrize("indices", [torch.tensor([0., 1.]), torch.tensor([[0]]),
                                    torch.tensor([-1]), torch.tensor([6]), torch.tensor([1, 1])])
def test_reject_invalid_region_indices(indices):
    with pytest.raises(ValueError, match="indices|index"):
        prepare(entropy_tensor(), [dict(batch_idx=0, image_idx=1, rho=.5, indices=indices)])


@pytest.mark.parametrize("mask", [torch.ones(6), torch.full((2, 3), .5),
                                 torch.full((2, 3), float("nan"))])
def test_reject_invalid_masks(mask):
    with pytest.raises(ValueError, match="mask"):
        prepare(entropy_tensor(), [dict(batch_idx=0, image_idx=1, rho=.5, mask=mask)])


@pytest.mark.parametrize("entropy", [torch.ones(2, 3), torch.ones(1, 1, 1, 2, 3),
                                    torch.ones(1, 2, 3, dtype=torch.long), torch.ones(0, 2, 3),
                                    torch.full((1, 2, 3), -1.), torch.full((1, 2, 3), float("inf")),
                                    torch.full((1, 2, 3), float("nan")), None])
def test_reject_invalid_entropy(entropy):
    with pytest.raises(ValueError, match="entropy"):
        prepare(entropy, [])


@pytest.mark.parametrize("options", [dict(), dict(seed=1), dict(source_keys={1: "a"}),
    dict(permutation={}), dict(permutation={1: [0, 0]}), dict(permutation={1: [1]}),
    dict(permutation={1: [0., 1.]}), dict(permutation={1: None}),
    dict(permutation={1: [0, 1], 2: [0]}), dict(permutation={1: [0, 1]}, seed=1),
    dict(seed=1, source_keys={}), dict(seed=1, source_keys={1: ""}),
    dict(seed=1, source_keys={1: 42}), dict(seed=True, source_keys={1: "a"})])
def test_reject_missing_or_malformed_shuffle_provenance(options):
    rows = [term(0, 1, .25, [0]), term(0, 1, .5, [1])]
    with pytest.raises(ValueError):
        prepare(entropy_tensor(), rows, control="image_shuffled", **options)


def test_reject_duplicate_source_keys_and_nonshuffle_options():
    entropy = entropy_tensor()
    rows = [term(0, 1, .25, [0]), term(1, 2, .5, [1])]
    with pytest.raises(ValueError, match="unique"):
        prepare(entropy, rows, control="image_shuffled", seed=1, source_keys={1: "a", 2: "a"})
    for arm in FACTORIAL:
        with pytest.raises(ValueError, match="shuffle options"):
            prepare(entropy, rows, control=arm, permutation={})


def test_reject_invalid_source_crop_identity_and_missing_schema():
    entropy = entropy_tensor()
    for field, value in (("image_idx", 1.5), ("batch_idx", -1), ("batch_idx", 5),
                         ("batch_idx", True), ("image_idx", torch.tensor(True))):
        row = term(0, 1, .5, [0])
        row[field] = value
        with pytest.raises(ValueError):
            prepare(entropy, [row])
    with pytest.raises(ValueError, match="multiple source"):
        prepare(entropy, [term(0, 1, .5, [0]), term(0, 2, .5, [1])])
    for field in ("image_idx", "batch_idx", "rho", "indices"):
        row = term(0, 1, .5, [0])
        del row[field]
        with pytest.raises(KeyError):
            prepare(entropy, [row])


def test_reject_cache_mismatch_and_dtype_overflow():
    entropy, rows = entropy_tensor(), fixture_terms()
    cached = prepare(entropy, rows)
    for options in (dict(control="episode_region"), dict(permutation={}), dict(seed=1),
                    dict(source_keys={})):
        with pytest.raises(ValueError, match="cache"):
            loss(entropy, prepared=cached, **options)
    with pytest.raises(ValueError, match="cache"):
        loss(entropy, rows, prepared=cached)
    for changed in (entropy.float(), entropy[:1]):
        with pytest.raises(ValueError, match="shape and dtype"):
            loss(changed, prepared=cached)
    with pytest.raises(ValueError, match="PreparedAllocation"):
        loss(entropy, prepared=())
    for arm, q in (("image_region", 1e40), ("image_token", 3e38)):
        with pytest.raises(ValueError, match="finite|overflows"):
            prepare(entropy.float(), [term(0, 1, q, [0, 1, 2])], control=arm)


def test_cpu_only_contract_without_initializing_an_accelerator():
    entropy = entropy_tensor()
    with pytest.raises(ValueError, match="CPU"):
        prepare(torch.empty(1, 2, 3, device="meta"), [])
    for field, tensor in (("rho", torch.empty((), device="meta")),
                          ("image_idx", torch.empty((), dtype=torch.long, device="meta")),
                          ("indices", torch.empty(1, dtype=torch.long, device="meta"))):
        row = term(0, 1, .5, [0])
        row[field] = tensor
        with pytest.raises(ValueError):
            prepare(entropy, [row])
    with pytest.raises(ValueError, match="mask"):
        prepare(entropy, [dict(batch_idx=0, image_idx=1, rho=.5,
                               mask=torch.empty(2, 3, device="meta"))])


def test_test_loader_does_not_import_model_modules():
    assert not MODEL_IMPORTS
