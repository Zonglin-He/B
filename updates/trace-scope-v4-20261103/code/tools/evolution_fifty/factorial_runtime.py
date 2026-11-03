"""Unfrozen M04 actual-update primitives; no queue, GPU launch or GT reader.

All crossed arms measure the same production C views before assigning scores.
Their cost is not the diagnostic-free production Confidence cost.
"""

from __future__ import annotations

import copy
import math

import torch

from tools.evolution_confirmation.run import ExperimentTRACE
from trace_ovss.method.score_factorial import ARMS, assign_score_factorial
from trace_ovss.method.trace.constants import EPS
from .bootstrap import require
from . import generalization_runtime as native
from .risk_panel import candidate_identity
from .spatial_runtime import observe_updates


class FactorialExperimentTRACE(ExperimentTRACE):
    def __init__(self, *, factorial_arm="Dc_Oc", common_active=False, **kwargs):
        require(factorial_arm in ARMS and type(common_active) is bool,
                "Unknown M04 assignment arm/study")
        self.factorial_arm, self.common_active = factorial_arm, common_active
        self.source_ids = ()
        self.last_factorial_assignment = None
        self.last_factorial_coefficients = []
        super().__init__(**kwargs)
        self.arm = "image_C"

    @torch.no_grad()
    def evaluate_trace_views(self, images, original_probs, candidates):
        require(self.arm == "image_C" and self.view_mode == "C"
                and self.factorial_arm in ARMS and type(self.common_active) is bool,
                "M04 requires the unchanged production C diagnostic")
        rows = super().evaluate_trace_views(images, original_probs, candidates)
        assignment_rows = []
        for row in rows:
            source = row["image_idx"]
            require(type(source) is int and 0 <= source < len(self.source_ids)
                    and row["quality"] == row["reliability"] == row["counterfactual_support"],
                    "M04 original C/source identity differs")
            assignment_rows.append(dict(source_id=self.source_ids[source], candidate_id=candidate_identity(row),
                area=int(row["area"]), kappa=float(row["confidence"]), C=float(row["reliability"])))
        assignment = assign_score_factorial(assignment_rows, common_active=self.common_active)
        values = assignment["arms"][self.factorial_arm]["assigned_scores"]
        self.last_factorial_assignment = dict(assignment, selected_arm=self.factorial_arm,
            candidates=assignment_rows, all_arms_measure_C=True, GT_opened=False)
        return [dict(row, reliability=0. if value is None else value,
            quality=0. if value is None else value, original_C=row["reliability"],
            factorial_excluded=value is None, diagnostic_algorithm="M04."+self.factorial_arm)
            for row,value in zip(rows,values,strict=True)]

    @torch.no_grad()
    def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
        result = super().estimate_trace_reliability(images,valid_mask,image_ids)
        active = [row for row in self.last_candidate_stats if row["reliability"] > EPS]
        for term,row in zip(result["region_terms"],active,strict=True):
            require(term["image_idx"] == row["image_idx"] and term["rho"] == row["reliability"],
                    "M04 active term/candidate identity differs")
            term["factorial_source_id"] = self.source_ids[row["image_idx"]]
            term["factorial_candidate_id"] = candidate_identity(row)
        return result

    def prepare_trace_objective(self, entropy, terms):
        prepared = super().prepare_trace_objective(entropy, terms)
        # Observe the actual native dtype weights/accumulated denominator, not
        # the assignment helper's ideal float64 budget. No objective mutation.
        records = []
        groups = {}
        for term in terms:
            groups.setdefault(int(term["image_idx"]), []).append(term)
        active_groups = [[term for term in group if float(term["rho"]) > EPS
                          and term["indices"].numel()] for group in groups.values()]
        active_groups = [group for group in active_groups if group]
        require(len(active_groups) == len(prepared), "M04 native objective source count differs")
        for terms_in_source,(entries,normalizer) in zip(active_groups,prepared,strict=True):
            require(len(entries) == len(terms_in_source), "M04 native objective term count differs")
            for term,(batch,indices,rho) in zip(terms_in_source,entries,strict=True):
                require(batch == term["batch_idx"] and torch.equal(indices,term["indices"])
                        and not rho.requires_grad and not normalizer.requires_grad
                        and bool(torch.isfinite(rho)) and bool(torch.isfinite(normalizer)),
                        "M04 objective observation changed membership or finite detached weights")
                coefficient = rho/normalizer/len(prepared)
                require(bool(torch.isfinite(coefficient)), "Nonfinite M04 runtime coefficient")
                records.append(dict(image_idx=term["image_idx"],batch_idx=batch,
                    source_id=term.get("factorial_source_id"),candidate_id=term.get("factorial_candidate_id"),
                    rho=float(rho),normalizer=float(normalizer),dtype=str(rho.dtype),
                    active_sources=len(prepared),coefficient=float(coefficient),
                    coefficient_arithmetic="native_dtype_rho_div_normalizer_div_active_sources"))
        self.last_factorial_coefficients = records
        return prepared


def run_arm(method, arm, common_active, images, valid, ids, meta, cfg, mapping, source_ids):
    require(isinstance(method,FactorialExperimentTRACE) and arm in ARMS
            and type(common_active) is bool and method.model.training is False,
            "Invalid M04 method/arm/study")
    require(tuple(cfg["resize"]) == tuple(cfg["patch_size"])
            and len(images) == len(meta["img_path"]) == len(source_ids)
            and all(isinstance(x,str) and x for x in source_ids)
            and len(set(source_ids)) == len(source_ids), "M04 requires unique single-crop source IDs")
    require(torch.is_tensor(ids) and ids.ndim == 1 and len(ids) == len(images)
            and ids.dtype in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)
            and torch.equal(ids.to(dtype=torch.int64),torch.arange(len(images),device=ids.device)),
            "M04 requires explicit integer positional single-crop IDs 0..N-1; None is not allowed")
    require(method.steps == cfg["steps"] and method.lr == cfg["lr"]
            and method.trace_max_regions == cfg["trace_max_regions"]
            and method.trace_view_batch_size == cfg["trace_view_batch_size"],
            "M04 actual nonassignment configuration differs")
    native.legal_parameters(method)
    method.factorial_arm,method.common_active,method.source_ids = arm,common_active,tuple(source_ids)
    method.last_factorial_assignment = dict(assign_score_factorial([],common_active=common_active),
        selected_arm=arm,candidates=[],all_arms_measure_C=True,GT_opened=False)
    method.last_factorial_coefficients = []
    with observe_updates(method) as updates:
        outputs,stats,losses,cost = native.run_arm(method,"image_C",images,valid,ids,meta,cfg,mapping)
    assignment = method.last_factorial_assignment
    require(assignment is not None and assignment["selected_arm"] == arm
            and len(stats) == assignment["candidate_count"], "Missing M04 actual assignment")
    expected = assignment["arms"][arm]
    require(len(expected["active_candidate_ids"]) == cost["active_candidates"]
            and updates["count"] == cost["steps"] == len(losses)
            and len(method.last_factorial_coefficients) == cost["active_candidates"]
            and all(math.isfinite(float(row["reliability"])) for row in stats),
            "M04 actual admission/update/objective receipts differ")
    require({(r["source_id"],r["candidate_id"]) for r in method.last_factorial_coefficients}
            == set(expected["active_candidate_ids"]), "M04 runtime budget candidate identity differs")
    cost.update(factorial_arm=arm,common_active=common_active,optimizer_steps=updates["count"],
        all_arms_measure_C=True,production_Confidence_timing=False,
        coefficient_observation_in_model_time=True)
    return outputs,stats,losses,cost,copy.deepcopy(assignment),copy.deepcopy(method.last_factorial_coefficients)
