"""V01 fixed-grid model-side execution, not an admitted full experiment.

Only the predeclared update budget changes. The production C/Confidence path,
initial candidates, optimizer, objective and probe are otherwise unchanged.
No GT, time-match selection, model loading or GPU launch is performed here.
"""

from __future__ import annotations

import copy

import torch

from . import generalization_runtime as native
from .bootstrap import require
from .spatial_runtime import observe_updates
from .value_protocol import budget_steps

NATIVE_STEPS = dict(coco_object=30,pc60=7)


def points(dataset):
    require(dataset in NATIVE_STEPS, "V01 dataset is not in the frozen grid")
    return tuple(dict(arm=arm,steps=steps,id=f"{dataset}.{arm}.steps{steps}")
        for arm,values in budget_steps(NATIVE_STEPS[dataset]).items() for steps in values)


def validate_point(dataset, point):
    require(isinstance(point,dict) and set(point) == {"arm","steps","id"}
            and type(point["steps"]) is int and isinstance(point["arm"],str) and isinstance(point["id"],str)
            and point in points(dataset), "V01 point not in the exact predeclared integer grid")


def validate_episode(dataset, images, valid, ids, meta, cfg):
    require(cfg["dataset"] == {"coco_object":"COCOObjectDataset", "pc60":"PascalContext60Dataset"}[dataset]
            and torch.is_tensor(images) and images.ndim == 4 and images.shape[1] == 3 and len(images) > 0
            and tuple(cfg["resize"]) == tuple(cfg["patch_size"]) == tuple(images.shape[-2:])
            and len(meta["img_path"]) == len(images)
            and all(len(meta[key]) == len(images) for key in ("img_shape","patch_shape","patch_grid_shape"))
            and all(tuple(row) == tuple(cfg["resize"]) for key in ("img_shape","patch_shape") for row in meta[key])
            and all(tuple(row) == (1,1) for row in meta["patch_grid_shape"]),
            "V01 requires matching positional single-crop source geometry")
    require(torch.is_tensor(ids) and ids.ndim == 1 and len(ids) == len(images)
            and ids.dtype in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)
            and torch.equal(ids.to(dtype=torch.int64),torch.arange(len(images),device=ids.device)),
            "V01 requires integer positional source IDs before model work")
    require(valid is None or (torch.is_tensor(valid) and valid.dtype == torch.bool
            and tuple(valid.shape) == (len(images),1,*images.shape[-2:])), "V01 valid-mask geometry differs")


def run_point(method,dataset,point,images,valid,ids,meta,cfg,mapping,*,initial_rng):
    validate_point(dataset,point)
    require(type(method) is native.ExperimentTRACE and method.model.training is False,
            "V01 requires the unchanged production C/Confidence implementation")
    require(method.steps == cfg["steps"] == NATIVE_STEPS[dataset] and method.lr == cfg["lr"]
            and method.trace_max_regions == cfg["trace_max_regions"]
            and method.trace_view_batch_size == cfg["trace_view_batch_size"], "V01 fixed non-budget settings differ")
    validate_episode(dataset,images,valid,ids,meta,cfg)
    native.legal_parameters(method)
    before = native.shared.restored_state(method)
    original_steps,original_arm = method.steps,method.arm
    actual_cfg = copy.deepcopy(cfg)
    actual_cfg["steps"] = point["steps"]
    try:
        method.steps = point["steps"]
        native.shared.restore_rng(initial_rng,method.device)
        with observe_updates(method) as observed:
            predictions,stats,losses,cost = native.run_arm(method,point["arm"],images,valid,ids,meta,actual_cfg,mapping)
        expected = point["steps"] if cost["active_candidates"] else 0
        require(observed["count"] == cost["steps"] == len(losses) == expected,
                "V01 actual Adam steps differ from returned losses/declared budget")
        cost.update(point_id=point["id"],dataset=dataset,native_steps=NATIVE_STEPS[dataset],
            requested_steps=point["steps"],optimizer_steps=observed["count"],
            actual_step_observation_in_model_time=True,
            time_scope="Synchronized native reset/candidates/probe/adaptation/finalreadout plus common finite/Adam observation; excludes decode/H2D/receipts/GT",
            auxiliary_checkpoint_readouts=0,GT_opened=False)
        return predictions,stats,losses,cost
    finally:
        method.steps,method.arm = original_steps,original_arm
        method.reset()
        require(native.shared.restored_state(method) == before, "V01 exact theta0/Adam reset failed")
