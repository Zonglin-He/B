# Removal-suite independent review

Scope: Method-aligned removal definitions and reproducible sequential execution.
Production algorithm files and manuscript were not modified in this round.

## Method review

The independent conceptual review found the controls coherent, provided their
boundaries are stated precisely: connected components versus class unions is
a candidate-policy comparison that also changes count, extent, and ranking;
uniform weighting retains tracing deliberately; pixel normalization preserves
the prompt average and weighted denominator. Full subset runs cannot be
reconstructed by selecting records from a differently batched full-split run.
Relative reliability allocation, not a common loss-weight scale, is tested.
These distinctions are recorded in `docs/removal-experiments.md`.

The reduced full-validation roster was disclosed and frozen before GPU work.
Scored smoke outcomes did not cause any change to method, configurations,
cohorts, or variant definitions. The post-launch changes below strengthen only
execution and provenance handling, without mutating the running snapshot.

## Engineering review and repairs

The first review identified three P2 paths, without establishing that they
had occurred in the running suite:

1. A coordinator could exit while a worker survived, allowing a duplicate
   worker on resume. New workers inherit the exclusive queue-lock descriptor;
   completed unselected attempts are adopted in deterministic order.
2. An old external result lacked a pre-run input-file identity record. New
   worker protocols record file metadata. Unsealed external reuse is rejected.
   A queued finalizer refreshes Full COCO-CD after the primary suite and removes
   the provisional legacy reference from the audited tables.
3. Environment compatibility was not enforced for cost comparisons. New
   runs freeze and check Python, Torch/CUDA, GPU inventory/driver and precision
   flags. Earlier minimal records remain explicitly insufficient for a fully
   matched cost claim; the audited table flags this limitation.

A follow-up review found two additional P2 paths:

4. A companion suite seal could override conflicting metadata recorded by a
   worker. Present-but-conflicting metadata now always fails; only absent
   legacy metadata can use the original pre-run companion seal.
5. A failed final-control retry could overwrite an incomplete directory.
   Final-control attempts are now separate and immutable, and the earliest
   completed attempt is adopted without a new model run.

The final bounded static review closed both follow-up findings and reported
no additional actionable P1/P2 within those fixes. It did not certify completed
experiments, numerical equivalence, or the absence of every possible defect.

The main agent executed all 92 tests successfully. New coverage checks removal
semantics, gradients and episodic resets, input identity and environment
rejection, companion-seal timing/conflicts, inherited queue-lock lifetime, and
preservation/adoption of finalizer attempts. GPU experiment results and their
completion remain separately recorded in the result directory, not inferred
from tests or code review.
