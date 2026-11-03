"""Model-side primitives for a separately frozen/admitted G stage.

No CLI, checkpoint download, GT reader or independent GPU launch. Production
adaptation is the existing confirmation.ExperimentTRACE, not a new method.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import math
import time
from types import MethodType

import numpy as np
import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation import run as confirmation
from tools.evolution_followup import standalone as prior
from tools.evolution_image_full import run as image_worker
from trace_ovss.method.trace.constants import EPS
from trace_ovss.models.dense_clip import DenseCLIP
from .bootstrap import require
from .generalization import ARMS, BACKBONES, visual_layers
from .risk_worker import checked_checkpoint, adam_tensors, tensor_delta

COUNTERS = image_worker.COUNTERS
ExperimentTRACE = confirmation.ExperimentTRACE


def legal_parameters(method):
    """Independently enumerate visual LN affine tensors actually used here."""
    model = method.model
    require(isinstance(model, DenseCLIP), "Generalization requires the declared dense CLIP wrapper")
    blocks = model.visual.transformer.resblocks
    last = max(model._resolve_layers(method.vision_outputs, len(blocks)))
    expected = {}
    for name, module in model.visual.named_modules():
        if not isinstance(module, torch.nn.LayerNorm):
            continue
        if name.startswith("transformer.resblocks."):
            index = int(name.split(".")[2])
            if index > last or (index == last and name.endswith(".ln_2") and model.architecture == "reduced"):
                continue
        for key, value in module.named_parameters(recurse=False):
            if key in ("weight", "bias"):
                expected[id(value)] = f"visual.{name}.{key}"
    actual = {id(p):n for n,p in model.named_parameters() if p.requires_grad}
    require(set(expected) == set(actual) and bool(actual), "Used visual-LN membership differs")
    require({id(p) for g in method.optimizer.param_groups for p in g["params"]} == set(expected),
            "Actual optimizer visual-LN membership differs")
    shared.restored_state(method)
    return dict(trainable_names=sorted(actual.values()),
        visual_LN_names=sorted(expected.values()),
        parameter_tensors=len(actual), parameter_elements=sum(p.numel() for p in model.parameters() if p.requires_grad),
        architecture=model.architecture, attention_strategy=model.attention_strategy,
        final_selected_block=last, final_MLP_used=model.architecture != "reduced")


def build_method(spec):
    before = checked_checkpoint(spec["checkpoint"])
    backbone, cfg = spec["backbone"], spec["config"]
    require(backbone in BACKBONES and spec["vision_outputs"] == visual_layers(backbone)
            and spec["model_type"] in ("naclip", "sclip")
            and spec["checkpoint"]["backbone"] == backbone,
            "Generalization model/layer policy differs")
    method = ExperimentTRACE(classes=spec["vocabulary"]["queries"], lr=cfg["lr"], steps=cfg["steps"],
        trace_max_regions=cfg["trace_max_regions"], trace_view_batch_size=cfg["trace_view_batch_size"],
        vision_outputs=spec["vision_outputs"], model_type=spec["model_type"],
        backbone=spec["checkpoint"]["path"], device="cuda", view_mode="C")
    require(checked_checkpoint(spec["checkpoint"]) == before, "Checkpoint changed during G construction")
    method.model.eval()
    method.arm = "image_C"
    depth, patch, _ = BACKBONES[backbone]
    require(len(method.model.visual.transformer.resblocks) == depth and method.model.visual.patch_size == patch,
            "Actual visual depth/patch geometry differs")
    expected = ("vanilla", "csa") if spec["model_type"] == "sclip" else ("reduced", "naclip")
    require((method.model.architecture, method.model.attention_strategy) == expected,
            "Actual spatial-prediction backend differs")
    legal_parameters(method)
    return method


def validate_cost(row, arm, cfg):
    require(arm in ARMS, "Undeclared G arm")
    if arm != "no_adapt":
        image_worker.validate_cost(row, arm, cfg)
        return
    require(row["arm"] == arm and type(row["images"]) is int and row["images"] > 0
            and all(type(row[k]) is int for k in ("patches", "steps", "candidate_count", "active_candidates"))
            and row["images"] == row["patches"] and row["steps"] == row["candidate_count"] == row["active_candidates"] == 0
            and type(row["seconds"]) in (float,int) and math.isfinite(row["seconds"])
            and row["seconds"] >= 0, "Invalid zero-update G cost")
    expected = {k:0 for k in COUNTERS}
    expected.update(eval_forward_calls=1, eval_forward_inputs=row["images"],
                    encoder_calls=1, encoder_inputs=row["images"])
    require(all(type(row[k]) is int and row[k] == v for k,v in expected.items()),
            "NoAdapt performed adaptation, candidate extraction or extra readout")


def run_arm(method, arm, images, valid, ids, meta, cfg, mapping):
    """Timer includes native reset through final readout, not decode/H2D/I/O."""
    require(arm in ARMS and method.model.training is False, "Invalid G arm/model mode")
    if arm != "no_adapt":
        method.arm = arm
    with image_worker.finite_observation(method), complete_optimizer_guard(method), method.count_work() as work:
        shared.synchronize(method.device)
        start = time.perf_counter()
        if arm == "no_adapt":
            method.reset()
            losses = []
        else:
            losses = method.adapt(images, valid_mask=valid, image_ids=ids)
        predictions = shared.label_predictions(method, images, meta, cfg, mapping)
        shared.synchronize(method.device)
        seconds = time.perf_counter()-start
    stats = [] if arm == "no_adapt" else method.last_candidate_stats
    row = dict(arm=arm, images=len(meta["img_path"]), patches=len(images), steps=len(losses),
        seconds=seconds, candidate_count=len(stats), active_candidates=sum(r["reliability"] > EPS for r in stats),
        candidate_sha256=None if arm == "no_adapt" else prior.candidate_identity(stats), **work)
    validate_cost(row, arm, cfg)
    require(all(math.isfinite(value) for value in losses), "Nonfinite G trajectory")
    return predictions, stats, losses, row


def require_complete_gradients(method):
    parameters = {p for p in method.model.parameters() if p.requires_grad}
    optimizer = [p for g in method.optimizer.param_groups for p in g["params"]]
    require(bool(parameters) and len(optimizer) == len(parameters) and set(optimizer) == parameters,
            "G optimizer membership differs from complete trainable set")
    require(all(p.grad is not None and p.grad.shape == p.shape for p in parameters),
            "G update missing a legal visual-LN gradient")
    return parameters


def require_complete_Adam(method):
    parameters = require_complete_gradients(method)
    require(set(method.optimizer.state) == parameters, "G Adam state omits legal visual-LN tensors")
    for p in parameters:
        state = method.optimizer.state[p]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                and state["exp_avg"].shape == state["exp_avg_sq"].shape == p.shape,
                "G Adam state membership/geometry differs")


@contextmanager
def complete_optimizer_guard(method):
    handles = []
    def before_step(*_):
        require_complete_gradients(method)
    try:
        handles.append(method.optimizer.register_step_pre_hook(before_step))
        handles.append(method.optimizer.register_step_post_hook(lambda *_: require_complete_Adam(method)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def direct_objective(entropy, terms):
    """Independent explicit per-image weighted means for numerical gates only."""
    groups = {}
    for term in terms:
        if float(term["rho"]) > EPS and term["indices"].numel():
            groups.setdefault(int(term["image_idx"]), []).append(term)
    total = entropy.sum()*0.
    for rows in groups.values():
        numerator, denominator = entropy.sum()*0., entropy.new_tensor(0.)
        for term in rows:
            x = entropy[:, term["batch_idx"]].flatten(1) if entropy.ndim == 4 else entropy[term["batch_idx"]].flatten()
            value = x.index_select(1 if entropy.ndim == 4 else 0, term["indices"]).mean()
            weight = entropy.new_tensor(float(term["rho"]))
            numerator = numerator+weight*value
            denominator = denominator+weight
        normalizer = torch.where(denominator.double() > EPS, denominator, torch.ones_like(denominator))
        total = total+numerator/normalizer
    return total/len(groups) if groups else total


class DirectReference(image_worker.GateReference):
    """Only the gate's loss/readout reference, never the production worker."""

    arm = "image_C"
    _forward_trace_view_batches = prior.StandaloneTRACE._forward_trace_view_batches
    _count = prior.StandaloneTRACE._count
    count_work = prior.StandaloneTRACE.count_work

    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        if self.arm == "image_C":
            return super().evaluate_trace_views(images, original_probs, candidates)
        require(self.arm == "image_confidence", "Invalid direct G reference")
        self._two_view_counts = dict(diagnostic_views=0, diagnostic_calls=0)
        return [dict(row, original_support=float(row["confidence"]), target_support=None,
            counterfactual_support=None, residual_support=None, dominance=None,
            quality=float(row["confidence"]), reliability=float(row["confidence"]),
            diagnostic_algorithm="Confidence", omitted=False) for row in candidates]

    @staticmethod
    def prepare_trace_objective(entropy, terms):
        return tuple(terms)

    @staticmethod
    def trace_region_normalized_entropy(entropy, terms, *, prepared=None):
        return direct_objective(entropy, terms)


def make_reference(method):
    method.reset()
    reference = DirectReference(classes=method.classes, lr=method.lr, steps=method.steps,
        trace_max_regions=method.trace_max_regions, trace_view_batch_size=method.trace_view_batch_size,
        vision_outputs=method.vision_outputs, prompts=method.prompt_templates, model=method.model,
        tokenize_fn=method.tokenize, device=method.device, view_mode="C")
    reference.model.eval()
    require(torch.equal(reference.text_x, method.text_x), "Direct G reference text embeddings differ")
    return reference


def _capture(method, arm, images, valid, ids, meta, cfg, mapping, rng):
    method.reset()
    shared.restore_rng(rng, method.device)
    steps, frozen = [], [(p,p._version) for p in method.model.parameters() if not p.requires_grad]
    def capture(optimizer, args, kwargs):
        require_complete_Adam(method)
        parameters = {n:p.detach().cpu().clone() for n,p in method.model.named_parameters() if p.requires_grad}
        gradients = {n:p.grad.detach().cpu().clone() for n,p in method.model.named_parameters()
                     if p.requires_grad}
        steps.append(dict(parameters=parameters, gradients=gradients, adam=adam_tensors(method)))
    handle = method.optimizer.register_step_post_hook(capture)
    try:
        predictions, stats, losses, counts = run_arm(method, arm, images, valid, ids, meta, cfg, mapping)
        require(all(p._version == version for p,version in frozen), "Frozen G reference tensor mutated")
        require(len(steps) == len(losses), "G gate optimizer-step receipt differs")
        final = {n:p.detach().cpu().clone() for n,p in method.model.named_parameters() if p.requires_grad}
        return dict(predictions=predictions, stats=copy.deepcopy(stats), losses=losses, steps=steps,
                    final=final, counts=counts)
    finally:
        handle.remove()
        method.reset()
        shared.restored_state(method)


def compare_gate(method, reference, images, valid, ids, meta, cfg, mapping, token_grid):
    """Actual RGB gates use this exact core; CPU tests inject only tiny models."""
    method.reset()
    initial = shared.restored_state(method)
    rng, reports, candidate_digest = shared.rng_state(method.device), {}, None
    for arm in ARMS:
        expected = _capture(reference, arm, images, valid, ids, meta, cfg, mapping, rng)
        actual = _capture(method, arm, images, valid, ids, meta, cfg, mapping, rng)
        require(shared.restored_state(method) == initial == shared.restored_state(reference), "G gate reset differs")
        require(len(expected["predictions"]) == len(actual["predictions"]) == len(meta["img_path"])
                and all(a.shape == b.shape for a,b in zip(expected["predictions"], actual["predictions"], strict=True)),
                "G gate prediction geometry differs")
        changed = sum(int(np.count_nonzero(a != b)) for a,b in zip(expected["predictions"], actual["predictions"], strict=True))
        require(len(expected["losses"]) == len(actual["losses"]), "G gate update count differs")
        maximum = dict(loss=max((abs(a-b) for a,b in zip(expected["losses"],actual["losses"],strict=True)), default=0.),
            final_parameter=tensor_delta(expected["final"], actual["final"]), gradient=0., parameter=0., adam=0., support=0.)
        for left,right in zip(expected["steps"],actual["steps"],strict=True):
            for key,field in (("gradient","gradients"),("parameter","parameters"),("adam","adam")):
                maximum[key] = max(maximum[key], tensor_delta(left[field],right[field]))
        if arm != "no_adapt":
            require(actual["stats"] and actual["counts"]["active_candidates"] > 0
                    and len(actual["losses"]) == cfg["steps"], "Vacuous G gate without finite actual updates")
            digest = prior.candidate_identity(actual["stats"])
            require(digest == prior.candidate_identity(expected["stats"]), "Direct G candidate identity differs")
            require(candidate_digest is None or candidate_digest == digest, "C and Confidence select different G candidates")
            candidate_digest = digest
            for a,b in zip(actual["stats"], expected["stats"], strict=True):
                require(tuple(a["region"].shape) == tuple(token_grid) and tuple(b["region"].shape) == tuple(token_grid),
                        "Actual G candidate token grid differs from backbone")
                for field in ("confidence", "reliability", "counterfactual_support"):
                    require((a[field] is None) == (b[field] is None), "Measured G support membership differs")
                    if a[field] is not None:
                        difference = abs(a[field]-b[field])
                        require(math.isfinite(a[field]) and math.isfinite(b[field]) and math.isfinite(difference),
                                "Nonfinite measured G support or difference")
                        maximum["support"] = max(maximum["support"], difference)
        require(changed == 0 and all(math.isfinite(x) and x <= 1e-5 for x in maximum.values()),
                f"Actual G numerical gate failed: {maximum}, labels={changed}")
        reports[arm] = dict(passed=True, max_abs=maximum, changed_labels=changed,
            actual_counts=actual["counts"], candidate_sha256=actual["counts"]["candidate_sha256"], GT_opened=False)
    return reports


def official_sclip_gate(method, images, source):
    """Exact official final-layer component on current weights/observed RGB.

    The later TRACE9layer readout has its own gate; it is not claimed to be
    the published SCLIP whole-pipeline readout.
    """
    from .sclip_reference import reference_methods, SOURCE_SHA
    require(method.model.architecture == "vanilla" and method.model.attention_strategy == "csa",
            "Official component gate requires SCLIP")
    method.reset()
    initial = shared.restored_state(method)
    reference = copy.deepcopy(method.model.visual).eval()
    for name, function in reference_methods(source).items():
        setattr(reference, name, MethodType(function, reference))
    observed = images.detach().clone().requires_grad_()
    other = images.detach().clone().requires_grad_()
    actual = method.model.encode_dense_layers(observed, (-1,), project=True)[0]
    # Official CLIP.encode_image casts before calling the visual encoder.
    expected = reference(other.to(dtype=reference.conv1.weight.dtype), return_all=True, csa=True)
    image_worker.require_finite([actual, expected], "SCLIP official component features")
    actual_parameters = {n:p for n,p in method.model.visual.named_parameters() if p.requires_grad}
    expected_parameters = {n:p for n,p in reference.named_parameters() if p.requires_grad}
    require(actual_parameters.keys() == expected_parameters.keys(), "Official SCLIP used-LN membership differs")
    left = torch.autograd.grad(actual.float().square().mean(), [observed, *actual_parameters.values()], allow_unused=False)
    right = torch.autograd.grad(expected.float().square().mean(), [other, *expected_parameters.values()], allow_unused=False)
    image_worker.require_finite([*left, *right], "SCLIP official component gradients")
    feature = float((actual.detach().float()-expected.detach().float()).abs().max())
    gradient = max(float((a.detach().float()-b.detach().float()).abs().max()) for a,b in zip(left,right,strict=True))
    text = method.text_x[-1].detach()
    labels_a = (torch.nn.functional.normalize(actual.detach(),dim=-1)[:,1:]@text.T).argmax(-1)
    labels_b = (torch.nn.functional.normalize(expected.detach(),dim=-1)[:,1:]@text.T).argmax(-1)
    changed = int((labels_a != labels_b).sum())
    require(max(feature, gradient) <= 1e-5 and changed == 0, "Actual official SCLIP component parity failed")
    require(shared.restored_state(method) == initial, "Official SCLIP gate changed adaptation state")
    return dict(passed=True, source_sha256=SOURCE_SHA, feature_max_abs=feature,
        input_and_visual_LN_gradient_max_abs=gradient, changed_query_labels=changed,
        input_dtype=str(images.dtype), reference_weight_dtype=str(reference.conv1.weight.dtype),
        actual_RGB_sources=len(images), GT_opened=False, adaptation_steps=0,
        scope="Official final-layer component, not published whole9layer readout equivalence")
