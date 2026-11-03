"""M04 every-step loss/gradient/Adam gate, not artifact or GPU admission.

The direct masked-mean reference shares the production probe and assignment;
native uncrossed natural arms additionally compare to unchanged production.
"""

from __future__ import annotations

import copy

from tools.evolution import run_cell as shared
from . import factorial_runtime as runtime
from . import generalization_runtime as native
from .bootstrap import require
from .risk_worker import adam_tensors
from .spatial_gate import compare


class DirectFactorialReference(runtime.FactorialExperimentTRACE):
    @staticmethod
    def trace_region_normalized_entropy(entropy, terms, *, prepared=None):
        return native.direct_objective(entropy,terms)


def make_reference(method, *, production=False):
    cls = native.ExperimentTRACE if production else DirectFactorialReference
    result = cls(classes=method.classes,lr=method.lr,steps=method.steps,
        trace_max_regions=method.trace_max_regions,trace_view_batch_size=method.trace_view_batch_size,
        vision_outputs=method.vision_outputs,prompts=method.prompt_templates,model=method.model,
        tokenize_fn=method.tokenize,device=method.device,view_mode="C")
    result.model.eval()
    return result


def capture(method, arm, common, images, valid, ids, meta, cfg, mapping, source_ids, rng):
    method.reset()
    shared.restore_rng(rng,method.device)
    records, frozen = [],[(p,p._version) for p in method.model.parameters() if not p.requires_grad]
    def after_step(*_):
        native.require_complete_Adam(method)
        selected = {n:p for n,p in method.model.named_parameters() if p.requires_grad}
        records.append(dict(parameters={n:p.detach().cpu().clone() for n,p in selected.items()},
            gradients={n:p.grad.detach().cpu().clone() for n,p in selected.items()},adam=adam_tensors(method)))
    handle = method.optimizer.register_step_post_hook(after_step)
    try:
        labels,stats,losses,cost,assignment,coefficients = runtime.run_arm(method,arm,common,
            images,valid,ids,meta,cfg,mapping,source_ids)
        require(all(p._version == version for p,version in frozen), "Frozen M04 gate weight changed")
        require(len(records) == cost["optimizer_steps"] == len(losses), "M04 gate step records differ")
        return dict(predictions=labels,stats=copy.deepcopy(stats),losses=losses,steps=records,
            final={n:p.detach().cpu().clone() for n,p in method.model.named_parameters() if p.requires_grad},
            counts=cost,assignment=assignment,coefficients=coefficients)
    finally:
        handle.remove()
        method.reset()
        shared.restored_state(method)


def compare_gate(method, images, valid, ids, meta, cfg, mapping, source_ids):
    require(isinstance(method,runtime.FactorialExperimentTRACE), "M04 gate method differs")
    method.reset()
    initial,rng = shared.restored_state(method),shared.rng_state(method.device)
    direct,production = make_reference(method),make_reference(method,production=True)
    reports,identity = {},None
    try:
        for common in (False,True):
            study = "common_active" if common else "natural"
            reports[study] = {}
            for arm in runtime.ARMS:
                expected = capture(direct,arm,common,images,valid,ids,meta,cfg,mapping,source_ids,rng)
                actual = capture(method,arm,common,images,valid,ids,meta,cfg,mapping,source_ids,rng)
                report = compare(actual,expected,cfg)
                require(actual["assignment"] == expected["assignment"]
                        and actual["coefficients"] == expected["coefficients"], "M04 gate assignment/budget differs")
                candidate = actual["counts"]["candidate_sha256"]
                identity = candidate if identity is None else identity
                require(candidate == identity, "M04 gate changed original candidates across arms/studies")
                if not common and arm in ("Dk_Ok","Dc_Oc"):
                    native_arm = "image_confidence" if arm == "Dk_Ok" else "image_C"
                    other = native._capture(production,native_arm,images,valid,ids,meta,cfg,mapping,rng)
                    report["unchanged_production_parity"] = compare(actual,other,cfg)
                report.update(candidate_sha256=candidate,active_candidates=actual["counts"]["active_candidates"],
                    active_sources=len(actual["assignment"]["arms"][arm]["active_source_ids"]),GT_opened=False)
                reports[study][arm] = report
                require(shared.restored_state(method) == shared.restored_state(direct)
                        == shared.restored_state(production) == initial, "M04 gate exact reset differs")
        return dict(passed=True,studies=reports,atol=1e-5,changed_labels=0,GT_opened=False,
            reference_scope="Independent masked-mean loss arithmetic, shared C diagnostic and assignment")
    finally:
        method.reset()
