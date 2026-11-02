# Local Baseline Rerun Protocol

## Latest Human Authorization

The human now requests local MLMP reruns on all eight paper datasets and all
baseline methods on datasets absent from the MLMP paper. This overrides the
earlier published-value-only restriction. WATT/CLIPArTT are newly included ONLY
for the missing datasets. No baseline search or extra seed is authorized.

The previous reference ledger omitted PC59. Primary PDF SHA256
30911a11e6b890b2bf466adb8e33e6f1ac53b82c50b56efb7d8749cef1c82ac7,
main Table5 explicitly reports NACLIP P59 Original MLMP31.95 (+/-.02).
The six reported current-paper protocols are VOC20/VOC21/PC59/PC60/Object/Stuff.
Only COCO-CD and Waterbirds-Seg lack reported values in that paper. Preserve
the prior frozen manuscript/reviews, but correct this omission in the next draft.

## Eighteen Full Cells

MLMP on VOC20/21(1449each),PC59/60(5105each),Object/Stuff(5000each),CD500,
Waterbirds5322. Additionally NoAdapt,TENT,TPT,WATT,CLIPArTT on CD and Waterbirds.
This is18 distinct dataset-method cells,58040 image-method evaluations.
Existing local historical cells remain archived; none substitutes for a rerun.

Use the original B0817 complete input order, checkpoint, category/query mapping,
224-square resize/crop, stride112 and label/ignore/readout conventions. Use
the archived native MLMP repository implementation, not a TRACE-derived loss
or a renamed matched weighting control. Source archive is clean commit
ba9f57bf47846ec727c616c5163233a98ca0bf94, pinned by its recorded source hashes.
Copy required source/tokenizer/prompts into a NEW frozen suite, never restore
or edit the partially cleaned external checkout or any old frozen runtime.

## Fixed Native Settings

Batch2,lr.001,Adam(.9,.999),weightdecay0,episodic model/optimizer reset,seed0.
TENT/TPT/MLMP/CLIPArTT use10 updates; TPT n_ctx4; CLIPArTT k3.
MLMP uses visual outputs-1..-18,alpha_cls1,loss prompt integration,seven prompts.
WATT retains native l2,m5,seven prompt templates. This entails70 actual Adam
steps and5 reported cycle losses per episode, NOT10 actual updates; record
its141 visual calls versus11 for TENT/TPT and21 for CLIPArTT. MLMP has12
visual calls:10 adaptation passes plus TWO native adaptive-weighted readout
passes (`adaptive_weighted_mean` first obtains per-layer weights, then encodes
again with weighted aggregation). Do not remove or merge either native pass.
NoAdapt performs0 updates and1 visual inference call using native TENT readout.
Prompt differences and native backend behavior are method-specific and retained.
Do not apply TRACE-selected lr/steps/batch or search baseline hyperparameters.
Thus local reruns improve execution provenance, not equal-tuning-budget claims.

## Admission and Execution

First CPU tests and independent code/artifact audit. New retained user service
waits for actual2055 successful terminal, source/config/input-bound suite seal
and released shared GPU lock. Never interrupt, repair or duplicate2055.
One GPU worker,shared lock,12GiB RAM/1GiB swap,OOMstop,Restartno,RemainAfterExit.
The running old snapshot must never be edited. A failure is preserved and stops
the new suite without automatic retry, fallback or changing methods/parameters.

All18 real4-RGB gates precede full prediction. Gates compare instrumented and
direct native execution under identical reset/order/RNG, finite actual updates,
loss/gradient/parameter tolerance1e-5 and zero changed labels. Pure wrappers,
syntax/CPU tests or loaded models do not establish real gate completion.
Observe raw finite outputs before sanitization, gradients before Adam and all
post-step parameters/optimizer states; include TPT's external ctx parameter.
Do not silently modify loss arithmetic or native implementation to pass a gate.
Native WATT also averages and writes all LayerNorm states, including text
LayerNorms outside Adam. Preserve this native behavior and compare its actual
written state in parity evidence; exact untouched-state guards must not reject
the legitimate floating-point averaging roundoff. Episode reset remains exact.

All18 full predictions must seal before any new GT scoring. During prediction
use explicit RGB-only loaders; label metadata may be stat-checked but no bytes
opened. Stream per-image score records after the barrier; pool per-class I/U,
never average per-image IoUs or load huge full/per_image_counts JSONs at once.
Record actual step/encoder inputs/calls, finite losses, reset/optimizer state,
time scope and exact warmup exclusion. Include all images in accuracy/work.

## Completion and Manuscript

Check all18 canonical outcomes,exact populations,manifests,parent/source/native
archive/config/input/prediction/score/barrier seals,cost and real retained-service
terminal loaded active/exited MainPID0 Resultsuccess ExecMainStatus0. A final
artifact seal alone does not certify service exit. Preserve all signs/failures.
Keep local reruns distinct from published means/stds and historical native runs.
Waterbirds calibration/full development overlap and other historical exposure
remain; rerunning baselines does not create untouched evaluation or fix W1/W2.
Wait for both mechanism and baseline suites to finish/verify before integrating
the next manuscript and fresh independent paper review. No inherited old score,
no reviewer shopping,extra backbone,geometry campaign,bootstrap or3seed run.
