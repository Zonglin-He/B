"""Unfrozen M02 spatial-update primitives, not an admitted experiment runner.

Only the diagnostic preservation region changes. Candidate extraction, original
query/readout/loss regions and image-balanced objective remain production TRACE.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import math
import time

import numpy as np
import torch

from tools.evolution import run_cell as shared
from tools.evolution_confirmation.run import ExperimentTRACE
from tools.evolution_image_full import run as image_worker
from trace_ovss.method.spatial_probe import SpatialProbeMixin
from trace_ovss.method.trace.constants import EPS
from .bootstrap import require
from .generalization_runtime import complete_optimizer_guard, legal_parameters
from .generalization_runtime import require_complete_Adam

MODES = ("native", "translated", "global", "confidence")


@contextmanager
def observe_updates(method):
    """Observe completed Adam calls, not returned losses or parameter movement."""
    require(not method.optimizer.state, "M02 episode must start with reset Adam state")
    observed = dict(count=0)
    def after_step(*_):
        require_complete_Adam(method)
        states = list(method.optimizer.state.values())
        runtime_tensors = [value for state in states for value in state.values()]
        require(all(torch.is_tensor(v) for v in runtime_tensors), "Missing tensor Adam state")
        image_worker.require_finite(runtime_tensors, "M02 observed completed Adam state")
        steps = {}
        for state in states:
            step = state["step"]
            require(step.numel() == 1, "M02 Adam step must be scalar")
            steps.setdefault(step.device, []).append(step.reshape(()))
        require(bool(steps) and all(bool(torch.stack(values).eq(observed["count"]+1).all())
                                  for values in steps.values()), "M02 Adam step progression differs")
        observed["count"] += 1
    handle = method.optimizer.register_step_post_hook(after_step)
    try:
        yield observed
    finally:
        handle.remove()


class SpatialExperimentTRACE(SpatialProbeMixin, ExperimentTRACE):
    def __init__(self, *, probe_mode="native", **kwargs):
        require(probe_mode in MODES, "Unknown spatial update mode")
        self.probe_mode = probe_mode
        super().__init__(**kwargs)
        self.arm = "image_C"


def validate_cost(row, stats, cfg):
    require(row["mode"] in MODES and type(row["images"]) is int and row["images"] > 0
            and row["images"] == row["patches"] and row["candidate_count"] == len(stats)
            and 0 <= len(stats) <= row["images"]*cfg["trace_max_regions"]
            and type(row["seconds"]) in (int,float) and math.isfinite(row["seconds"])
            and row["seconds"] >= 0, "Invalid spatial cost scope")
    require(all(type(row[k]) is int and row[k] >= 0 for k in image_worker.COUNTERS),
            "Invalid actual spatial counters")
    active = sum(r["reliability"] > EPS for r in stats)
    steps = cfg["steps"] if active else 0
    views = (0 if row["mode"] == "confidence" else len({r["batch_idx"] for r in stats})
             if row["mode"] == "global" else len(stats))
    calls = (views+cfg["trace_view_batch_size"]-1)//cfg["trace_view_batch_size"]
    require(row["active_candidates"] == active and row["steps"] == steps
            and type(row["optimizer_steps"]) is int and row["optimizer_steps"] == steps
            and row["eval_forward_calls"] == 2 and row["eval_forward_inputs"] == 2*row["patches"]
            and row["adapt_forward_calls"] == steps and row["adapt_forward_inputs"] == steps*row["patches"]
            and row["diagnostic_forward_calls"] == row["diagnostic_encoder_calls"] == calls
            and row["diagnostic_forward_inputs"] == row["diagnostic_encoder_inputs"] == views
            and row["encoder_calls"] == 2+steps+calls
            and row["encoder_inputs"] == (2+steps)*row["patches"]+views,
            "Actual spatial forward/update schedule differs")


def run_arm(method, mode, images, valid, ids, meta, cfg, mapping):
    require(isinstance(method, SpatialExperimentTRACE) and mode in MODES
            and method.model.training is False, "Invalid spatial method/mode")
    require(tuple(cfg["resize"]) == tuple(cfg["patch_size"]) and len(meta["img_path"]) == len(images),
            "Spatial primitives require declared single-crop source episodes")
    require(method.steps == cfg["steps"] and method.trace_max_regions == cfg["trace_max_regions"]
            and method.trace_view_batch_size == cfg["trace_view_batch_size"], "Spatial actual configuration differs")
    legal_parameters(method)
    method.probe_mode = mode
    method.arm = "image_C"
    method.last_spatial_probe_receipts = []
    with observe_updates(method) as updates, image_worker.finite_observation(method), complete_optimizer_guard(method), method.count_work() as work:
        shared.synchronize(method.device)
        started = time.perf_counter()
        losses = method.adapt(images,valid_mask=valid,image_ids=ids)
        predictions = shared.label_predictions(method,images,meta,cfg,mapping)
        shared.synchronize(method.device)
        seconds = time.perf_counter()-started
    stats = method.last_candidate_stats
    require(all(math.isfinite(x) for x in losses)
            and all(math.isfinite(r["reliability"]) and 0 <= r["reliability"] <= 1 for r in stats),
            "Nonfinite spatial loss/support")
    row = dict(mode=mode, images=len(meta["img_path"]),patches=len(images),steps=len(losses),seconds=seconds,
        optimizer_steps=updates["count"],
        candidate_count=len(stats),active_candidates=sum(r["reliability"] > EPS for r in stats),
        candidate_sha256=image_worker.prior.candidate_identity(stats),**work)
    validate_cost(row,stats,cfg)
    geometry = copy.deepcopy(method.last_spatial_probe_receipts)
    require(len(geometry) == len(stats) and all(g["readout_and_loss_remain_original"] for g in geometry),
            "Missing spatial preservation/readout metadata")
    if mode == "translated":
        require(all(g["original_area"] == g["shifted_area"] for g in geometry),
                "Translated preservation changed area")
    require(len(predictions) == len(meta["img_path"])
            and all(p.shape == tuple(cfg["resize"]) and p.dtype == np.uint16 for p in predictions),
            "Invalid spatial prediction geometry")
    return predictions,stats,losses,row,geometry
