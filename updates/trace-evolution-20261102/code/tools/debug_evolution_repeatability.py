"""GT-free A/A and A/B replay; never relax the original equivalence gate."""

import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh debug artifact directory")
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(args.runtime), str(args.runtime / "tools")]
    from tools.evolution.run_cell import (image_only_dataset_class, image_collate,
        label_predictions, restored_state, restore_rng, rng_state, write_json)
    from tools.evolution.policies import state_digest
    from trace_ovss.method.trace import TRACE
    from trace_ovss.method.two_view import TwoViewTRACE
    from trace_algorithm_variants import make_algorithm_class
    from trace_ovss.configs import best_config, VISION_OUTPUTS
    from trace_ovss.data.datasets import SegmentationPatchDataset
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask
    from torch.nn.attention import SDPBackend, sdpa_kernel

    config = best_config("coco_cd", "/home/wwww/hzl/data")
    data = image_only_dataset_class(SegmentationPatchDataset)(config["dataset"], config["data_dir"],
        resize=config["resize"], patch_size=config["patch_size"], patch_stride=config["patch_stride"],
        include_auxiliary=False)
    indices = sorted(np.random.default_rng(20261024).permutation(len(data))[:4].tolist())
    batch = image_collate([data[i] for i in indices])
    images = batch["img_patches"].cuda()
    valid = build_patch_valid_mask(batch["meta"], config["patch_size"], config["patch_stride"], "cuda")
    image_ids = build_patch_image_ids(batch["meta"], "cuda")
    method = TwoViewTRACE(classes=data.classes, lr=config["lr"], steps=config["steps"],
        trace_max_regions=config["trace_max_regions"], trace_view_batch_size=config["trace_view_batch_size"],
        vision_outputs=VISION_OUTPUTS, device="cuda")
    original_cls = make_algorithm_class(TRACE, "quality_only")
    state = rng_state(method.device)
    reference = state_digest(method.model_state)
    reports, predictions = [], {}
    for mode in ("default", "deterministic_default", "deterministic_math"):
        deterministic = mode != "default"
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
        context = sdpa_kernel(SDPBackend.MATH) if mode == "deterministic_math" else nullcontext()
        with context:
            for index, cls in enumerate((original_cls, original_cls, TwoViewTRACE, TwoViewTRACE)):
                method.__class__ = cls
                method.reset()
                restored_state(method)
                restore_rng(state, method.device)
                key = f"{mode}_{index}"
                gradients = []
                step = method.optimizer.step

                def traced_step(*a, **kw):
                    gradients.append(state_digest({name: p.grad for name, p in method.model.named_parameters()
                                                   if p.requires_grad and p.grad is not None}))
                    return step(*a, **kw)

                method.optimizer.step = traced_step
                try:
                    losses = method.adapt(images, valid_mask=valid, image_ids=image_ids)
                finally:
                    method.optimizer.step = step
                post = label_predictions(method, images, batch["meta"], config, None)
                predictions[key] = post
                trainable = state_digest({name: p for name, p in method.model.named_parameters() if p.requires_grad})
                if state_digest(method.model_state) != reference:
                    raise RuntimeError("Reset reference weights mutated")
                report = dict(key=key, class_name=cls.__name__, losses=losses, gradient_sha256=gradients,
                    trainable_sha256=trainable, supports=[[r["target_support"], r["counterfactual_support"]]
                                                         for r in method.last_candidate_stats],
                    gt_contents_opened=False, deterministic_algorithms=deterministic,
                    attention_backend="math" if mode == "deterministic_math" else "default")
                reports.append(report)
                write_json(args.output / f"{key}.json", report)
                print(json.dumps(dict(key=key, losses=losses, trainable_sha256=trainable)), flush=True)
    comparisons = []
    for mode in ("default", "deterministic_default", "deterministic_math"):
        group = [r for r in reports if r["key"].startswith(mode)]
        for left, right in ((0, 1), (2, 3), (0, 2)):
            a, b = group[left], group[right]
            pa, pb = predictions[a["key"]], predictions[b["key"]]
            comparisons.append(dict(left=a["key"], right=b["key"],
                max_loss_abs_difference=max(abs(x-y) for x, y in zip(a["losses"], b["losses"])),
                gradients_equal=a["gradient_sha256"] == b["gradient_sha256"],
                trainable_equal=a["trainable_sha256"] == b["trainable_sha256"],
                supports_equal=a["supports"] == b["supports"],
                changed_labels=sum(int((x != y).sum()) for x, y in zip(pa, pb))))
    write_json(args.output / "summary.json", dict(comparisons=comparisons, gt_contents_opened=False))
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    with Path("/home/wwww/hzl/results/reviewer_followup_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
