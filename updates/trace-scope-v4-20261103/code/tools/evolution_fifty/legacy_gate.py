"""M03 explicit loss arithmetic/every-step gate; no model loader or GPU entry.

This shares the pinned view/score and original candidates. It is not an
independent reproduction of the full historical algorithm or OOM iterator.
"""

from __future__ import annotations

import copy

from tools.evolution import run_cell as shared
from . import legacy_runtime as runtime
from . import generalization_runtime as native
from .bootstrap import require
from .risk_worker import adam_tensors
from .spatial_gate import compare


class DirectLegacyReference(runtime.LegacyExperimentTRACE):
    def trace_region_normalized_entropy(self,entropy,terms,*,prepared=None):
        rows = terms if self.objective_mode == "image" else [dict(term,image_idx=0) for term in terms]
        return native.direct_objective(entropy,rows)


def make_reference(method,*,production=False):
    kwargs = dict(classes=method.classes,lr=method.lr,steps=method.steps,
        trace_max_regions=method.trace_max_regions,trace_view_batch_size=method.trace_view_batch_size,
        vision_outputs=method.vision_outputs,prompts=method.prompt_templates,model=method.model,
        tokenize_fn=method.tokenize,device=method.device,view_mode="C")
    if production:
        result = native.ExperimentTRACE(**kwargs)
    else:
        result = DirectLegacyReference(dataset_name=method.legacy_dataset_name,
            benchmark_classes=method.legacy_benchmark_classes,query_to_class=method.legacy_query_to_class,**kwargs)
    result.model.eval()
    return result


def capture(method,mode,images,valid,ids,meta,cfg,mapping,rng):
    method.reset()
    shared.restore_rng(rng,method.device)
    records,frozen = [],[(p,p._version) for p in method.model.parameters() if not p.requires_grad]
    def after_step(*_):
        native.require_complete_Adam(method)
        selected = {n:p for n,p in method.model.named_parameters() if p.requires_grad}
        records.append(dict(parameters={n:p.detach().cpu().clone() for n,p in selected.items()},
            gradients={n:p.grad.detach().cpu().clone() for n,p in selected.items()},adam=adam_tensors(method)))
    handle = method.optimizer.register_step_post_hook(after_step)
    try:
        labels,stats,losses,cost = runtime.run_arm(method,mode,images,valid,ids,meta,cfg,mapping)
        require(all(p._version == version for p,version in frozen), "M03 frozen gate parameter changed")
        require(len(records) == cost["optimizer_steps"] == len(losses), "M03 actual step receipts differ")
        return dict(predictions=labels,stats=copy.deepcopy(stats),losses=losses,steps=records,counts=cost,
            final={n:p.detach().cpu().clone() for n,p in method.model.named_parameters() if p.requires_grad})
    finally:
        handle.remove()
        method.reset()
        shared.restored_state(method)


def compare_gate(method,images,valid,ids,meta,cfg,mapping):
    require(isinstance(method,runtime.LegacyExperimentTRACE), "M03 gate method differs")
    runtime.validate_episode_inputs(method,images,ids,meta,cfg,mapping)
    method.reset()
    initial,rng = shared.restored_state(method),shared.rng_state(method.device)
    direct,production = make_reference(method),make_reference(method,production=True)
    reports,identity = {},None
    try:
        for mode in runtime.MODES:
            expected = capture(direct,mode,images,valid,ids,meta,cfg,mapping,rng)
            actual = capture(method,mode,images,valid,ids,meta,cfg,mapping,rng)
            report = compare(actual,expected,cfg)
            require([r["reliability"] for r in actual["stats"]] == [r["reliability"] for r in expected["stats"]],
                    "M03 pinned diagnostic supports differ")
            candidate = actual["counts"]["candidate_sha256"]
            identity = candidate if identity is None else identity
            require(candidate == identity, "M03 original candidates changed across conditions")
            if mode == "C.image":
                original = native._capture(production,"image_C",images,valid,ids,meta,cfg,mapping,rng)
                report["unchanged_production_C_image_parity"] = compare(actual,original,cfg)
            report.update(candidate_sha256=candidate,GT_opened=False,
                diagnostic_views=actual["counts"]["diagnostic_forward_inputs"],
                legacy_archive_sha256=method.legacy["archive_sha256"])
            reports[mode] = report
            require(shared.restored_state(method) == shared.restored_state(direct)
                    == shared.restored_state(production) == initial, "M03 gate exact reset differs")
        return dict(passed=True,modes=reports,atol=1e-5,changed_labels=0,GT_opened=False,
            reference_scope="Explicit episode/image masked mean, shared pinned diagnostic/candidates")
    finally:
        method.reset()
