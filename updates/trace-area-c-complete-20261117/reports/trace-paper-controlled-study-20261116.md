# Current-Method Controlled Paper Study

The actual human explicitly requests all paper experiments and ablations.
The original 26-condition comparison and all earlier studies stay immutable.
These prospective controls address attribution, transfer, context and compute,
not selection of a new test winner. MLMP is never tuned. Matching clean TRACE
parameters remain identical across L14/B16 and the original 15 S5 types.

## Source Cohorts

Four existing training families: COCO Object, PC60, VOC20 and Waterbirds.
Select 64 calibration sources and a target of 256 diagnostic sources per
family by a fixed SHA256 identity ranking, excluding every official evaluation
identity across the eight existing families and any physical image alias.
If fewer new historical-excluded sources exist, disclose history rather than
claiming untouched data; the minimum diagnostic population is 128, rounded
down to a complete multiple of eight. No model outcome influences source IDs.
Waterbirds retains each available metadata group in both disjoint phases.

PC60 training membership and masks come from the Detail author's published
train.pth, SHA1 4bfb49e8c1cefe352df876c9b5434e655c9c1d07. Its source keys must
be disjoint from the previously exported authoritative 5104-image validation
population. The CPU preparation explicitly decodes training masks only and
exports only selected new masks. Old local masks/splits are not overwritten.
Other original training labels remain closed until separate prediction-success
CPU scoring. Training labels and all official evaluation annotations are denied
to every GPU inference process.

All method comparisons share predeclared source blocks of eight. Three fixed
seeds permute only within each block; batches never cross blocks. The same
blocks are shared across arms, backbones and corruption types where applicable.
Uncertainty resamples whole blocks shared by every contrasted arm/order/type.
Reordering the same source is not an independent image or independent seed.

## Attribution Controls

Two parameter regimes are kept separate: fixed official MLMP settings and the
frozen selected clean TRACE settings. Neither is reselected using new outcomes.
The core table contains native MLMP, default area-C, selected area-C, selected
area-C without regional/global/CLS terms, the regional-by-CLS interaction,
mass-matched per-region versus per-pixel allocation, constant-covered,
constant-full-image and spatial-shuffle maps. Candidate controls separately
change semantic alias merging or K=4/16 while leaving the final official
readout unchanged. Different configurations are controls, not new main methods.

Object and PC60 additionally compare the seven original R02 score sources:
Uniform, confidence, C, C/(confidence+1e-8), exp(C-confidence), train-quantile
calibrated confidence, and exp(confidence-destroyed-region-support).
The last control is regional pixel-permutation PLPD, not full DeYO reproduction.
The existing frozen R02 scalar/view definitions are reused and explicitly
adapted to the adopted area-C allocation objective. Original candidates,
targets and optimizer budget are held fixed. Common-active support and each
image's total bonus mass are recorded and controlled. No positive-part clipping
or fitted sigmoid silently replaces the signed-change exponential mapping.

The confidence-to-C distribution map is fitted on separate calibration sources
without GT or outcomes and copied unchanged to the matching B16 controls.
It is distribution calibration, not calibrated correctness or update benefit.
Each score uses the same two authenticated diagnostic view families; standalone
timing charges only required work and never calls shared-view amortization an
end-to-end speedup. Constant or identity destructive views remain, without redraw.

## Time And Context

Time-only matching uses every integer step count from 1 through 20, a fixed
pilot population and at least three predetermined episode arrangements. A match
must satisfy the frozen tolerance using time only; no interpolation or accuracy
selection. If no grid point qualifies, retain NULL. Actual end-to-end boundaries,
diagnostic/update/readout segments, warmup, synchronized timing and peak CUDA
allocation must be reported. Historical NULL studies remain in the paper.

R04 reuses the 126 integrity-verified, outcome-blind constructed targets and
their original/sham/hard/edge contexts. Generated model RGB is input; selected
instance masks/classes/proofs are never model input. Scientific model outcomes
require a new frozen prediction stage and a separate CPU scoring stage after
actual successful prediction exit and seals. Helpful-context as well as harmful
context cases are retained. No Waterbirds-independent-causal-test claim follows.

## Final Evidence And Restrictions

Six official clean TENT conditions remain in the authorized baseline scope,
without parameter search. The matched-block inference study is distinguished
from the already complete three-shuffled-repeat main comparison. Zero-update
and single-source-update reuse must be justified by exact source/configuration
and sealed outputs, not scalar mIoU agreement alone.

One original GPU lock; GPU12GiB RAM/1GiB swap, CPU4GiB RAM/1GiB swap/one CPU;
OOMstop/Restartno; no parallel GPU, dropped caches or old-byte retries. Each
stage requires its frozen sources, actual parent success and exact typed
admission. Real first-four RGB finite-step numerical and label gates remain.
The original single Luna/max monitor is reused only for a newly registered
legal running target. No extra monitor, automation or cron is created.
