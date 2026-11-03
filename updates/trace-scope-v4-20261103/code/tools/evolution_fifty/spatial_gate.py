"""M02 update parity gate; no model loading, CLI, GT reader or launch.

The reference changes only objective evaluation to an explicit masked mean.
Native C/Confidence additionally compare against the unchanged production path.
An actual RGB use of this helper still requires a newly admitted frozen suite.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from tools.evolution import run_cell as shared
from . import generalization_runtime as native
from . import spatial_runtime as spatial
from .bootstrap import require
from .risk_worker import adam_tensors


class DirectSpatialReference(spatial.SpatialExperimentTRACE):
    @staticmethod
    def prepare_trace_objective(entropy, terms):
        return tuple(terms)

    @staticmethod
    def trace_region_normalized_entropy(entropy, terms, *, prepared=None):
        return native.direct_objective(entropy, terms)


def make_reference(method, *, production=False):
    cls = native.ExperimentTRACE if production else DirectSpatialReference
    reference = cls(classes=method.classes, lr=method.lr, steps=method.steps,
        trace_max_regions=method.trace_max_regions, trace_view_batch_size=method.trace_view_batch_size,
        vision_outputs=method.vision_outputs, prompts=method.prompt_templates, model=method.model,
        tokenize_fn=method.tokenize, device=method.device, view_mode="C")
    reference.model.eval()
    return reference


def capture(method, mode, images, valid, ids, meta, cfg, mapping, rng):
    method.reset()
    shared.restore_rng(rng, method.device)
    records, frozen = [], [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    def after_step(*_):
        native.require_complete_Adam(method)
        selected = {n:p for n,p in method.model.named_parameters() if p.requires_grad}
        records.append(dict(parameters={n:p.detach().cpu().clone() for n,p in selected.items()},
            gradients={n:p.grad.detach().cpu().clone() for n,p in selected.items()}, adam=adam_tensors(method)))
    hook = method.optimizer.register_step_post_hook(after_step)
    try:
        labels, stats, losses, cost, geometry = spatial.run_arm(method, mode, images, valid, ids, meta, cfg, mapping)
        require(all(p._version == version for p,version in frozen), "Frozen spatial gate weight mutated")
        require(len(records) == len(losses), "Spatial gate actual step receipt differs")
        return dict(predictions=labels, stats=copy.deepcopy(stats), losses=losses, steps=records,
            final={n:p.detach().cpu().clone() for n,p in method.model.named_parameters() if p.requires_grad},
            counts=cost, geometry=geometry)
    finally:
        hook.remove()
        method.reset()
        shared.restored_state(method)


def finite_tensor_delta(left, right):
    require(bool(left) and left.keys() == right.keys(), "Spatial gate tensor membership differs")
    maximum = 0.
    for name in left:
        a, b = left[name], right[name]
        require(torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype
                and a.numel() > 0 and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                "Spatial gate nonfinite or incompatible tensor receipt")
        difference = (a.detach().cpu()-b.detach().cpu()).abs()
        require(bool(torch.isfinite(difference).all()), "Spatial gate nonfinite derived tensor difference")
        value = float(difference.max())
        require(math.isfinite(value), "Spatial gate nonfinite tensor maximum")
        maximum = max(maximum,value)
    return maximum


def compare(left, right, cfg):
    require(len(left["steps"]) == len(right["steps"]) == cfg["steps"] > 0
            and left["counts"]["active_candidates"] > 0 and right["counts"]["active_candidates"] > 0,
            "Vacuous spatial gate without actual finite updates")
    require(left["counts"]["candidate_sha256"] == right["counts"]["candidate_sha256"]
            and len(left["losses"]) == len(right["losses"]) == cfg["steps"], "Spatial gate candidate/step identity differs")
    require(len(left["predictions"]) == len(right["predictions"]) > 0
            and all(a.shape == b.shape for a,b in zip(left["predictions"], right["predictions"], strict=True)),
            "Spatial gate output geometry differs")
    changed = sum(int(np.count_nonzero(a != b)) for a,b in zip(left["predictions"],right["predictions"],strict=True))
    require(all(type(value) in (float,int) and math.isfinite(value)
                for record in (left,right) for value in record["losses"]), "Spatial gate nonfinite loss receipt")
    loss_differences = [abs(a-b) for a,b in zip(left["losses"],right["losses"],strict=True)]
    require(all(math.isfinite(v) for v in loss_differences), "Spatial gate nonfinite derived loss difference")
    maximum = dict(loss=max(loss_differences),
        final_parameter=finite_tensor_delta(left["final"],right["final"]), gradient=0., parameter=0., adam=0.)
    for a,b in zip(left["steps"], right["steps"], strict=True):
        for target, key in (("gradient","gradients"),("parameter","parameters"),("adam","adam")):
            maximum[target] = max(maximum[target],finite_tensor_delta(a[key],b[key]))
    require(changed == 0 and all(math.isfinite(v) and v <= 1e-5 for v in maximum.values()), "Spatial gate numerical/label mismatch")
    return dict(passed=True, max_abs=maximum, changed_labels=changed, actual_optimizer_steps=len(left["steps"]))


def compare_gate(method, images, valid, ids, meta, cfg, mapping):
    require(isinstance(method, spatial.SpatialExperimentTRACE), "Spatial gate requires the declared method")
    method.reset()
    initial, rng = shared.restored_state(method), shared.rng_state(method.device)
    direct, production = make_reference(method), make_reference(method, production=True)
    reports, identity = {}, None
    try:
        for mode in spatial.MODES:
            expected = capture(direct,mode,images,valid,ids,meta,cfg,mapping,rng)
            actual = capture(method,mode,images,valid,ids,meta,cfg,mapping,rng)
            report = compare(actual,expected,cfg)
            require(actual["geometry"] == expected["geometry"], "Spatial gate preservation/readout geometry differs")
            require([r["reliability"] for r in actual["stats"]] == [r["reliability"] for r in expected["stats"]],
                    "Spatial gate fixed diagnostic supports differ")
            candidate = actual["counts"]["candidate_sha256"]
            identity = candidate if identity is None else identity
            require(candidate == identity, "Spatial gate changed original candidates across modes")
            if mode in ("native","confidence"):
                arm = "image_C" if mode == "native" else "image_confidence"
                old = native._capture(production,arm,images,valid,ids,meta,cfg,mapping,rng)
                report["unchanged_production_parity"] = compare(actual,old,cfg)
            report.update(candidate_sha256=candidate, geometry=actual["geometry"], GT_opened=False)
            reports[mode] = report
            require(shared.restored_state(method) == shared.restored_state(direct)
                    == shared.restored_state(production) == initial, "Spatial gate exact reset differs")
        return dict(passed=True,modes=reports,atol=1e-5,changed_labels=0,GT_opened=False)
    finally:
        method.reset()
