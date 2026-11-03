"""V01 every-budget actual-step numerical gate, no loader/launcher/GT.

The explicit masked-mean oracle shares production candidates/diagnostics.
Gate-only reset snapshots are never used for primary timing or full runs.
"""

from __future__ import annotations

import copy
import math

from . import value_runtime as runtime
from .bootstrap import require
from .risk_worker import adam_tensors
from .spatial_gate import compare


def capture_point(method, dataset, point, images, valid, ids, meta, cfg, mapping, rng):
    runtime.validate_point(dataset, point)
    runtime.validate_episode(dataset, images, valid, ids, meta, cfg)
    runtime.native.legal_parameters(method)
    method.reset()
    records, resets = [], []
    frozen = [(p, p._version) for p in method.model.parameters() if not p.requires_grad]
    initial = runtime.native.shared.restored_state(method)
    original_reset = method.reset
    inherited = "reset" not in method.__dict__
    def observed_reset(*args, **kwargs):
        resets.append({n:p.detach().cpu().clone() for n,p in method.model.named_parameters() if p.requires_grad})
        return original_reset(*args, **kwargs)
    def after_step(*_):
        runtime.native.require_complete_Adam(method)
        selected = {n:p for n,p in method.model.named_parameters() if p.requires_grad}
        records.append(dict(parameters={n:p.detach().cpu().clone() for n,p in selected.items()},
            gradients={n:p.grad.detach().cpu().clone() for n,p in selected.items()}, adam=adam_tensors(method)))
    hook = method.optimizer.register_step_post_hook(after_step)
    method.reset = observed_reset
    try:
        labels, stats, losses, cost = runtime.run_point(method, dataset, point, images, valid, ids, meta, cfg,
                                                      mapping, initial_rng=rng)
        require(len(resets) == 2, "V01 gate requires native adaptation reset plus final cleanup reset")
        require(len(records) == len(losses) == cost["optimizer_steps"], "V01 gate actual step receipts differ")
        require(all(p._version == version for p,version in frozen), "V01 frozen gate parameter changed")
        require(runtime.native.shared.restored_state(method) == initial and method.steps == cfg["steps"],
                "V01 gate point failed to restore exact state/budget")
        return dict(predictions=labels, stats=copy.deepcopy(stats), losses=losses, steps=records,
            final=resets[-1], counts=cost, gate_only_reset_snapshots=len(resets))
    finally:
        hook.remove()
        if inherited:
            del method.reset
        else:
            method.reset = original_reset
        method.reset()


def compare_gate(method, dataset, images, valid, ids, meta, cfg, mapping):
    require(type(method) is runtime.native.ExperimentTRACE and method.model.training is False,
            "V01 gate requires unchanged production method")
    require(type(method.steps) is int and type(cfg["steps"]) is int
            and method.steps == cfg["steps"] == runtime.NATIVE_STEPS[dataset]
            and method.lr == cfg["lr"] and method.trace_max_regions == cfg["trace_max_regions"]
            and method.trace_view_batch_size == cfg["trace_view_batch_size"],
            "V01 gate fixed non-budget settings differ before reference work")
    runtime.validate_episode(dataset, images, valid, ids, meta, cfg)
    runtime.native.legal_parameters(method)
    method.reset()
    initial, rng = runtime.native.shared.restored_state(method), runtime.native.shared.rng_state(method.device)
    direct = runtime.native.make_reference(method)
    reference_steps, reports, identity = direct.steps, {}, None
    try:
        for point in runtime.points(dataset):
            actual_cfg = copy.deepcopy(cfg)
            actual_cfg["steps"] = direct.steps = point["steps"]
            expected = runtime.native._capture(direct, point["arm"], images, valid, ids, meta, actual_cfg, mapping, rng)
            actual = capture_point(method, dataset, point, images, valid, ids, meta, cfg, mapping, rng)
            report = compare(actual, expected, actual_cfg)
            require(len(actual["stats"]) == len(expected["stats"]), "V01 support membership differs")
            maximum = 0.
            for left,right in zip(actual["stats"], expected["stats"], strict=True):
                for field in ("confidence", "reliability", "counterfactual_support"):
                    require((left[field] is None) == (right[field] is None), "V01 support field membership differs")
                    if left[field] is not None:
                        values = (left[field], right[field], abs(left[field]-right[field]))
                        require(all(math.isfinite(v) for v in values), "Nonfinite V01 gate support")
                        maximum = max(maximum, values[-1])
            require(maximum <= 1e-5, "V01 diagnostic support mismatch")
            candidate = actual["counts"]["candidate_sha256"]
            identity = candidate if identity is None else identity
            require(identity == candidate, "V01 budget points changed initial candidates")
            require(runtime.native.shared.restored_state(method) == runtime.native.shared.restored_state(direct) == initial,
                    "V01 gate conditions did not restore exact theta0/Adam")
            report["max_abs"]["support"] = maximum
            report.update(point=point, actual_counts=actual["counts"], candidate_sha256=candidate,
                GT_opened=False, gate_only_reset_snapshots=actual["gate_only_reset_snapshots"],
                gate_cost_not_primary_value_time=True)
            reports[point["id"]] = report
        return dict(passed=True, dataset=dataset, points=reports, atol=1e-5, changed_labels=0, GT_opened=False,
            reference_scope="Explicit masked-mean arithmetic sharing native candidates/diagnostic, not independent whole algorithm",
            gate_instrumentation="Two pre-reset snapshots include final pre-cleanup parameters; gate time excluded from formal accuracy-time curves")
    finally:
        direct.steps = reference_steps
        method.reset()
