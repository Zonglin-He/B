"""M03 pinned legacy diagnostic versus single C actual-update primitives.

The archive supplies five score/view/role modules only. Existing native engine,
optimizer/candidates/readout stay current, and the fixed no-retry view iterator
is never replaced by the historical OOM iterator. No model or GPU launcher.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import time

import torch

from tools.evolution_confirmation.run import ExperimentTRACE
from trace_ovss.method.trace.objective import TraceObjectiveMixin
from trace_ovss.method.trace.constants import EPS
from . import generalization_runtime as native
from . import legacy_rho
from .bootstrap import require
from .intake import sha256
from .spatial_runtime import observe_updates

OBJECTIVE_SHA = "d1dce2a27de03a371b8d00716609c49676b6a444c6d0f4e39b3c09963c8af589"
SCORES = ("C","legacy")
OBJECTIVES = ("image","episode")
MODES = tuple(score+"."+objective for score in SCORES for objective in OBJECTIVES)


def effective_mapping(mapping,queries,classes):
    if mapping is None:
        require(queries == classes, "M03 implicit query mapping is not identity-sized")
        return tuple(range(classes))
    require(isinstance(mapping,(tuple,list)) and len(mapping) == queries
            and all(type(index) is int and 0 <= index < classes for index in mapping),
            "M03 query mapping requires one bounded integer per query")
    return tuple(mapping)


class LegacyExperimentTRACE(ExperimentTRACE):
    def __init__(self, *, dataset_name, benchmark_classes, query_to_class, **kwargs):
        self.score_mode,self.objective_mode = "C","image"
        classes = kwargs.get("classes")
        require(isinstance(classes,(tuple,list)) and bool(classes)
                and isinstance(benchmark_classes,(tuple,list)) and bool(benchmark_classes)
                and all(isinstance(name,str) and bool(name) for name in (*classes,*benchmark_classes))
                and isinstance(dataset_name,str), "M03 explicit query/benchmark taxonomy required")
        self.legacy_effective_mapping = effective_mapping(query_to_class,len(classes),len(benchmark_classes))
        self.legacy_dataset_name = dataset_name
        self.legacy_benchmark_classes = tuple(benchmark_classes)
        self.legacy_query_to_class = None if query_to_class is None else tuple(query_to_class)
        self.legacy = legacy_rho.load_components()
        self.legacy_query_roles = legacy_rho.query_roles(self.legacy,dataset_name,benchmark_classes,query_to_class)
        require(isinstance(classes,(tuple,list)) and len(classes) == len(self.legacy_query_roles)
                and (query_to_class is not None or list(benchmark_classes) == list(classes)),
                "M03 query/benchmark role mapping differs before model initialization")
        require(sha256(Path(inspect.getfile(TraceObjectiveMixin))) == OBJECTIVE_SHA
                and self.legacy["modules"]["constants"].EPS == EPS,
                "M03 unchanged native episode objective differs from verified archived source")
        super().__init__(**kwargs)
        self.arm = "image_C"
        require(self._forward_trace_view_batches.__func__ is ExperimentTRACE._forward_trace_view_batches,
                "M03 must keep the current fixed failclosed diagnostic iterator")

    def make_trace_weak_image(self,image):
        if self.score_mode == "legacy":
            return self.legacy["modules"]["interventions"].TraceInterventionMixin.make_trace_weak_image(self,image)
        return super().make_trace_weak_image(image)

    def make_trace_counterfactual_background(self,image):
        if self.score_mode == "legacy":
            return self.legacy["modules"]["interventions"].TraceInterventionMixin.make_trace_counterfactual_background(self,image)
        return super().make_trace_counterfactual_background(image)

    @torch.no_grad()
    def evaluate_trace_views(self,images,original_probs,candidates):
        require(self.arm == "image_C" and self.view_mode == "C" and self.score_mode in SCORES,
                "M03 diagnostic source differs")
        if self.score_mode == "C":
            return super().evaluate_trace_views(images,original_probs,candidates)
        annotated = [dict(row,role=self.legacy_query_roles[row["class_idx"]]) for row in candidates]
        component = self.legacy["modules"]["interventions"]
        stats = component.TraceInterventionMixin.evaluate_trace_views(self,images,original_probs,annotated)
        views = sum(len(component.trace_view_types(row["role"])) for row in annotated)
        self._two_view_counts = dict(diagnostic_views=views,
            diagnostic_calls=(views+self.trace_view_batch_size-1)//self.trace_view_batch_size)
        for row in stats:
            require(all(math.isfinite(row[k]) and 0 <= row[k] <= 1
                        for k in ("original_support","target_support","counterfactual_support","quality","dominance","reliability"))
                    and (row["residual_support"] is None if row["role"] != "thing" else
                         math.isfinite(row["residual_support"]) and 0 <= row["residual_support"] <= 1),
                    "M03 missing/nonfinite legacy diagnostic")
            row["diagnostic_algorithm"] = "pinned_legacy_rho"
        return stats

    def prepare_trace_objective(self,entropy,terms):
        if self.objective_mode == "episode":
            return TraceObjectiveMixin.prepare_trace_objective(entropy,terms)
        require(self.objective_mode == "image", "Unknown M03 objective")
        return super().prepare_trace_objective(entropy,terms)

    def trace_region_normalized_entropy(self,entropy,terms,*,prepared=None):
        if self.objective_mode == "episode":
            return TraceObjectiveMixin.trace_region_normalized_entropy(entropy,terms,prepared=prepared)
        require(self.objective_mode == "image", "Unknown M03 objective")
        return super().trace_region_normalized_entropy(entropy,terms,prepared=prepared)


def validate_episode_inputs(method,images,ids,meta,cfg,mapping):
    require(cfg["dataset"] == method.legacy_dataset_name
            and effective_mapping(mapping,len(method.classes),len(method.legacy_benchmark_classes))
                == method.legacy_effective_mapping, "M03 runtime taxonomy/readout mapping differs from bound roles")
    require(tuple(cfg["resize"]) == tuple(cfg["patch_size"]) and len(images) == len(meta["img_path"])
            and torch.is_tensor(ids) and ids.ndim == 1 and len(ids) == len(images)
            and ids.dtype in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)
            and torch.equal(ids.to(dtype=torch.int64),torch.arange(len(images),device=ids.device)),
            "M03 requires positional integer single-crop source IDs before any forward")
    require(method.steps == cfg["steps"] and method.lr == cfg["lr"]
            and method.trace_max_regions == cfg["trace_max_regions"]
            and method.trace_view_batch_size == cfg["trace_view_batch_size"], "M03 non-score settings differ")


def run_arm(method,mode,images,valid,ids,meta,cfg,mapping):
    require(isinstance(method,LegacyExperimentTRACE) and mode in MODES and method.model.training is False,
            "M03 method/mode differs")
    validate_episode_inputs(method,images,ids,meta,cfg,mapping)
    native.legal_parameters(method)
    method.score_mode,method.objective_mode = mode.split(".")
    method.arm = "image_C"
    with observe_updates(method) as updates,native.image_worker.finite_observation(method),native.complete_optimizer_guard(method),method.count_work() as work:
        native.shared.synchronize(method.device)
        start = time.perf_counter()
        losses = method.adapt(images,valid_mask=valid,image_ids=ids)
        labels = native.shared.label_predictions(method,images,meta,cfg,mapping)
        native.shared.synchronize(method.device)
        seconds = time.perf_counter()-start
    stats = method.last_candidate_stats
    active = sum(row["reliability"] > EPS for row in stats)
    steps = cfg["steps"] if active else 0
    component = method.legacy["modules"]["interventions"]
    views = len(stats) if method.score_mode == "C" else sum(len(component.trace_view_types(row["role"])) for row in stats)
    calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
    expected = dict(eval_forward_calls=2,eval_forward_inputs=2*len(images),
        adapt_forward_calls=steps,adapt_forward_inputs=steps*len(images),
        diagnostic_forward_calls=calls,diagnostic_forward_inputs=views,
        diagnostic_encoder_calls=calls,diagnostic_encoder_inputs=views,
        encoder_calls=2+steps+calls,encoder_inputs=(2+steps)*len(images)+views)
    require(updates["count"] == len(losses) == steps
            and all(type(work[k]) is int and work[k] == value for k,value in expected.items())
            and all(math.isfinite(value) for value in losses)
            and all(math.isfinite(row["reliability"]) and 0 <= row["reliability"] <= 1 for row in stats),
            "M03 actual finite forward/update schedule differs")
    return labels,stats,losses,dict(arm=mode,images=len(images),patches=len(images),seconds=seconds,
        steps=steps,optimizer_steps=updates["count"],candidate_count=len(stats),active_candidates=active,
        candidate_sha256=native.prior.candidate_identity(stats),legacy_archive_sha256=method.legacy["archive_sha256"],
        legacy_source_members=method.legacy["member_sha256"],episode_objective_sha256=OBJECTIVE_SHA,
        old_engine_loaded=False,old_optimizer_loaded=False,old_OOM_iterator_selected=False,GT_opened=False,**work)
