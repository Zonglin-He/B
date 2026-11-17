# TRACE-area_C Main Method And Training Calibration

## Actual Human Decision

On November12 the actual human confirms area_C as the revised TRACE main method,
requests tuning across the required datasets including Waterbirds, and asks to
repair its loader. The subsequent correction explicitly says MLMP must NOT be
tuned: use its published/official configuration. This overrides any earlier
proposal of symmetric tuning. Method adoption is a design decision, not proof
of independent superiority, C causality, or final paper completion.

No old method, official source, result, paper, release or raw dataset is replaced.
Original trace automation stays PAUSED, monitors OFF. The new finite queue is
not a recurring monitor and has no blind retry or runtime-cap escalation.

## Fixed Scope And Method

Eight L14 clean datasets: Waterbirds-Seg, COCO Object, PC60, VOC20, VOC21, PC59,
COCO Stuff and COCO-CD. Three B16 clean conditions: Object, PC60 and VOC20.
The original fifteen VOC20 S5 corruptions form ONE calibration family with
one shared parameter tuple, selected by their arithmetic macro. No type-specific
best configurations. This is12 calibration families/26 evaluated conditions.
The old SCLIP negative transfer remains historical; it is not silently converted
to the official native NACLIP implementation or retuned in this queue.

TRACE uses unchanged executed area_C: full-image query entropy + alpha_cls1
CLS entropy + pixel-mean detached C-map times query entropy. Fixed top8 query
components, original one context probe per candidate at episode start,
initial-model C, no alias-merging candidate variant, no refresh or absolute gate.
Seven prompts, official final readout, L14 last18/B16 last9,224square CV2 geometry,
batch2, diagnostic batch4, model/Adam reset per episode. B16 last9 is the existing
declared transfer convention, not a claim that every condition occurs in MLMP's
paper. COCO-CD and Waterbirds use the official algorithm at these fixed settings.

## Prospective Search

Only TRACE searches LR={0.0003,0.001,0.003} x steps={5,10,20}:9 configurations.
K, CLS coefficient, regional coefficient, prompt/readout and batch size are not
searched. MLMP always uses LR0.001/10steps/alpha_cls1 and is excluded from selection.
Report tuned TRACE versus fixed official MLMP, never equal search/wallclock budgets.

For each dataset freeze two128-source, mutually disjoint training panels by
SHA256 identity ranking before any trial result. Exclude the union of all declared
heldout source IDs, physical heldout RGB aliases, historical256 calibration IDs,
and the recent32-source pilot. Shared VOC/COCO-derived populations retain their
dependence; this is not global never-exposed data. Waterbirds panels use balanced
four-metadata-group allocation where capacity permits, redistributing shortages
deterministically without duplicate foregrounds. This is a calibration sampling
policy, not the population distribution or an official balanced split.

Search the nine settings on the first128. Keep the top2 by mean pooled class mIoU
(type arithmetic macro for corruptions). Re-evaluate both on the other128 using
three prospectively fixed source orderings/seeds202611120/121/122, rebuilding B2
episodes. Select the mean over all three orders and all declared types. Ties prefer
fewer steps, then smaller LR. All settings and unfavorable results are retained.
The fixed MLMP baseline is evaluated once in search and in all three confirmation
orders; repeated order evaluation is not baseline hyperparameter tuning.

The finite queue produces69,888 predictions excluding first4 implementation
checks:26conditions x128sources x(11 search arms/orders +10 confirmation arms/orders).
Only training scores select settings. No heldout model evaluation or test metric
is part of this queue. After selection, separate fixed full evaluations are needed.

## Waterbirds Loader Repair

The historical loader recursively read all5322 pairs without a split. The new
loader requires train/val/test explicitly and consumes the pinned segmentation
release metadata:2458/296/2568 respectively. Original full5322 results remain
development-overlap results, not overwritten test-only results. Correcting the
split does not erase previous exposure to its test images.

CPU input audit checks every existing image/mask size, indexed mask encoding,
foreground class0/1 metadata -> pixel labels1/2, background0, unique foreground
IDs, physical RGB identities and exact per-pixel CUB grayscale>127 construction.
The official renderer and metadata are hash-pinned. Any wrong/missing mask is
reconstructed only into a NEW result subdirectory from the corresponding raw CUB
segmentation and metadata; old bytes remain unchanged. RGB generation is not
claimed reproduced. Cross-split Places background reuse is reported separately.
This data-repair audit is allowed to inspect mask content but performs no model
evaluation or accuracy-based sample selection. Prediction processes cannot open
annotation content. Indexed PNG decoding never takes an RGB red channel or
mistakes255 for foreground. Later score uses CV2 nearest224 with label IDs intact.

## Execution And Verification

Use one shared GPU lock,12GiB RAM/1GiB swap,Restart=no,OOM stop. CPU stages use
4GiB RAM/1GiB swap/one thread/hidden CUDA; no512MiB-specific swap cap is imposed.
Every actual process records unit/invocation/PID/cwd/argv/environment/effective
cgroup limits. Each score stage follows its GPU's true successful exit and full
prediction seals. First4 actual training RGBs check native-equivalent losses,
parameters and logits <=1e-5 and zero changed labels. All trial updates are finite.
The pipeline verifies hashes and true retained stage exit before advancing.
Source bytes are frozen/copied before execution. Failure stops and preserves
all bytes; changed code requires a new root/freeze. Keep a10GiB disk reserve.

Completion means all12 training selections and verified reports, not a new full
benchmark result, statistical discovery or final paper release. No new GitHub
upload or automatic final-test execution is authorized by this finite queue.
