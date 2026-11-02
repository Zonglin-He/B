"""Candidate-level reliability used by TRACE."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReliabilityScore:
    """The three quantities reported for one TRACE candidate."""

    quality: float
    dominance: float
    reliability: float


def trace_reliability(
    target_support: float,
    counterfactual_support: float,
    residual_support: float,
) -> ReliabilityScore:
    """Compute TRACE reliability from the candidate intervention responses.

    Quality is the weaker of target-preserved and context-perturbed support.
    Reliability is reduced when the response after candidate suppression
    exceeds the response after preserving the candidate. Every class uses
    the same rule.
    """

    target, counterfactual, residual = map(
        float, (target_support, counterfactual_support, residual_support)
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0
               for value in (target, counterfactual, residual)):
        raise ValueError("TRACE supports must be finite probabilities in [0, 1]")
    quality = min(target, counterfactual)
    dominance = max(residual - target, 0.0)
    reliability = quality * (1.0 - dominance)
    return ReliabilityScore(
        quality=quality,
        dominance=dominance,
        reliability=reliability,
    )
