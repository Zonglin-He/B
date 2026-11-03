"""Actual unit-weight TRACE candidate trajectories, isolated from GT scoring.

The caller owns frozen input admission and output seals. Checkpoint callbacks
may save predictions and visual-LN snapshots, never labels or supervised losses.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.evolution import run_cell as shared
from tools.evolution_image_full.run import finite_observation
from .bootstrap import require


def unit_term(candidate, device):
    """Decode original geometry; confidence/C/reliability never set update size."""
    region = candidate["region"]
    shape = region["shape"]
    require(region["encoding"] == "packbits-little-hex" and type(shape) is list
            and len(shape) == 2 and all(type(x) is int and x > 0 for x in shape),
            "Invalid original candidate mask")
    count = shape[0]*shape[1]
    packed = bytes.fromhex(region["data"])
    require(len(packed) == (count+7)//8, "Wrong packed mask length")
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    require(not bool(bits[count:].any()), "Noncanonical packed-mask padding")
    mask = bits[:count].astype(bool).reshape(shape)
    area = int(mask.sum())
    require(area == region["true_pixels"] == candidate["area"] > 0, "Original candidate area differs")
    require(all(type(candidate[k]) is int and candidate[k] >= 0 for k in ("batch_idx", "image_idx")),
            "Invalid crop/source identity")
    return dict(batch_idx=candidate["batch_idx"], image_idx=candidate["image_idx"], rho=1.,
                indices=torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device))


def parameter_snapshot(method):
    return {name: p.detach().cpu().clone() for name, p in method.model.named_parameters() if p.requires_grad}


def run_candidate(method, images, candidate, checkpoint, *, initial_rng=None):
    """Perform the real optimizer updates, with one checkpointed trajectory.

    Both1step and native-step readouts come from the same trajectory. The
    returned reset receipt verifies model AND Adam state against native theta0.
    Readout callbacks receive no C value and no annotations.
    """
    require(type(method.steps) is int and 1 <= method.steps <= 64, "Unbounded update steps")
    require(method.model.training is False, "Risk audit requires the fixed eval-mode model")
    term = unit_term(candidate, images.device)
    require(term["batch_idx"] < len(images), "Candidate crop absent from actual episode")
    losses, records = [], []
    try:
        method.reset()
        reset_receipt = shared.restored_state(method)
        if initial_rng is not None:
            shared.restore_rng(initial_rng, method.device)
        with finite_observation(method):
            checkpoint(0, method)
            prepared = None
            for step in range(1, method.steps+1):
                logits = method.forward_adapt_logits(images).float()
                require(logits.ndim in (4, 5) and bool(torch.isfinite(logits).all()), "Nonfinite/invalid raw logits")
                entropy = method.trace_softmax_entropy(logits)
                require(entropy.ndim in (3, 4) and bool(torch.isfinite(entropy).all())
                        and tuple(entropy.shape[-2:]) == tuple(candidate["region"]["shape"]),
                        "Entropy geometry differs from frozen original mask")
                if prepared is None:
                    prepared = method.prepare_trace_objective(entropy, [term])
                loss = method.trace_region_normalized_entropy(entropy, [term], prepared=prepared)
                require(loss.ndim == 0 and loss.requires_grad and bool(torch.isfinite(loss)), "Invalid unit update loss")
                loss.backward()
                method.optimizer.step()
                method.optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                if step == 1 or step == method.steps:
                    shared.synchronize(method.device)
                    checkpoint(step, method)
                    records.append(dict(step=step, trainable_sha256=shared.state_digest(
                        {name: p for name, p in method.model.named_parameters() if p.requires_grad}),
                        optimizer_sha256=shared.state_digest(method.optimizer.state_dict())))
        return dict(losses=losses, actual_steps=len(losses), update_weight=1.,
                    score_enters_loss=False, reset=reset_receipt, checkpoints=records)
    finally:
        method.reset()
        shared.restored_state(method)


def source_prediction(method, images, meta, config, mapping, source_index):
    """Use the production reconstruction/query mapping, retaining one source."""
    from trace_ovss.cli import collapse_class_extensions
    from trace_ovss.evaluation import aggregate_patch_logits

    with torch.no_grad():
        logits = method.evaluate(images)
        outputs = aggregate_patch_logits(logits, meta["patch_grid_shape"], meta["img_shape"],
                                         config["patch_size"], config["patch_stride"])
        require(0 <= source_index < len(outputs), "Source index differs from actual reconstruction")
        probabilities = outputs[source_index].softmax(0)
        if mapping is not None:
            probabilities = collapse_class_extensions(probabilities, mapping)
        require(bool(torch.isfinite(probabilities).all()), "Nonfinite reconstructed readout")
        return probabilities.argmax(0).cpu().numpy().astype(np.uint16)
