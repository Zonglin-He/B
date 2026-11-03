"""Change allocation only; never rerun candidate admission or diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch


POLICIES = ("trace", "uniform", "confidence", "target", "counterfactual", "mean", "permuted")


def state_digest(value):
    """Hash state without pickle identity or device-dependent serialization."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), str(item.dtype))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for entry in item:
                visit(entry)
        else:
            digest.update(json.dumps(item, allow_nan=False, sort_keys=True).encode())
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def expand_policies(policies, permutation_seeds):
    if not policies or len(policies) != len(set(policies)) or set(policies) - set(POLICIES):
        raise ValueError("Policies must be unique supported names")
    if (not permutation_seeds or len(permutation_seeds) != len(set(permutation_seeds))
            or any(seed < 0 for seed in permutation_seeds)):
        raise ValueError("Permutation seeds must be unique nonnegative integers")
    return [(f"permuted_{seed}", name, seed) if name == "permuted" else (name, name, None)
            for name in policies for seed in (permutation_seeds if name == "permuted" else [None])]


def policy_weights(stats, policy, *, permutation_seed=0, episode=0):
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    target = np.array([row["target_support"] for row in stats], dtype=np.float64)
    cf = np.array([row["counterfactual_support"] for row in stats], dtype=np.float64)
    confidence = np.array([row["confidence"] for row in stats], dtype=np.float64)
    if any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in (target, cf, confidence)):
        raise ValueError("Supports and confidence must be finite probabilities")
    q = np.minimum(target, cf)
    donor = np.arange(len(stats), dtype=np.int64)
    weights = {"trace": q, "uniform": np.ones_like(q), "confidence": confidence,
               "target": target, "counterfactual": cf, "mean": (target + cf) / 2,
               "permuted": q}[policy]
    if policy == "permuted":
        donor = np.random.default_rng(np.random.SeedSequence([permutation_seed, episode])).permutation(len(stats))
        weights = q[donor]
        if not np.array_equal(np.sort(weights), np.sort(q)):
            raise RuntimeError("Permutation changed the weight multiset")
    return weights.copy(), donor


def candidate_digest(stats):
    digest = hashlib.sha256()
    for row in stats:
        for key in ("batch_idx", "image_idx", "class_idx", "confidence", "target_support", "counterfactual_support"):
            digest.update(json.dumps(row[key], allow_nan=False).encode() + b"\0")
        mask = np.asarray(row["region"], dtype=np.bool_)
        digest.update(str(mask.shape).encode())
        digest.update(mask.tobytes())
    return digest.hexdigest()


def make_fixed_candidate_class(base):
    from trace_ovss.method.trace.constants import EPS
    from trace_ovss.method.trace.indexing import pack_region_indices

    class FixedCandidateTRACE(base):
        @torch.no_grad()
        def diagnose(self, images, valid_mask=None, image_ids=None):
            self.reset()
            super().estimate_trace_reliability(images, valid_mask, image_ids)
            self.fixed_stats = copy.deepcopy(self.last_candidate_stats)
            self.fixed_digest = candidate_digest(self.fixed_stats)
            self.fixed_image = images
            self.fixed_indices = pack_region_indices(
                [row["region"] for row in self.fixed_stats], images.device,
            )
            counts = {}
            for row in self.fixed_stats:
                counts[row["image_idx"]] = counts.get(row["image_idx"], 0) + 1
                if counts[row["image_idx"]] > self.trace_max_regions:
                    raise RuntimeError("Candidate budget exceeded")
            return self.fixed_stats

        def select_policy(self, policy, *, permutation_seed=0, episode=0):
            if candidate_digest(self.fixed_stats) != self.fixed_digest:
                raise RuntimeError("Fixed candidate masks, identities or diagnostics changed")
            self.fixed_weights, self.fixed_donors = policy_weights(
                self.fixed_stats, policy, permutation_seed=permutation_seed, episode=episode,
            )

        @torch.no_grad()
        def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
            if images is not self.fixed_image:
                raise RuntimeError("Cached diagnostics belong to another episode tensor")
            if candidate_digest(self.fixed_stats) != self.fixed_digest:
                raise RuntimeError("Fixed diagnostics mutated")
            terms = [dict(batch_idx=int(row["batch_idx"]), indices=indices, rho=float(rho))
                     for row, indices, rho in zip(self.fixed_stats, self.fixed_indices, self.fixed_weights)
                     if rho > EPS]
            self.last_candidate_stats = [dict(row, reliability=float(rho))
                                         for row, rho in zip(self.fixed_stats, self.fixed_weights)]
            self.last_region_terms = terms
            return {"region_terms": terms}

    return FixedCandidateTRACE
