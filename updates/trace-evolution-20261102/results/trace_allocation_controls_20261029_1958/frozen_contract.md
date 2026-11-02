# C Allocation Controls: Prospective Bounded Follow-up

## Status and Scope

This is a protocol, not an executed experiment or a new paper result.
Independent round05 reviewed the completed eight-dataset paper at5/10,
confidence4/5, with no identified fatal issue. Its design-isolation and
candidate/weight-pairing questions justify this focused follow-up. Do not
re-review the same submission without new substantive evidence.

No new baseline search, MLMP reproduction, backbone, geometry, bootstrap,
training-seed repeat, transform selection, or production-method change is
authorized by this protocol. Existing completed suites remain immutable.
All results retain their development-exposed interpretation.

## Fixed Sources

Use COCO Object full5000 and PC60 full5105, seed0, exact original ordered IDs,
vocabulary/readouts/checkpoint and per-dataset selected B/lr/steps/K from
`trace_single_probe_waterbirds_full_20261029_0817`. Bind the original parent
prepared SHA256
`4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3`,
selected configurations, prediction protocols and seals in new manifests.
Do not choose datasets, hyperparameters, transformation, thresholds, orders,
metric definitions or permutations again after seeing follow-up scores.

The existing image-region method is the reference, not a candidate to replace
after looking at these full scores. New normalization variants are ablations,
not replacement official-method results. A future substantive method change
would require its own calibration/confirmation decision and distinct identity.

## Five Arms

For fixed active candidates with detached C score q_R>EPS, let h_R be the mean
prompt/token entropy and A_R the actual candidate token count. Candidates,
queries, C scores, reset behavior and optimization configuration stay fixed.

| Arm | Coefficient within a normalization group | Grouping |
| --- | --- | --- |
| image_region | q_R / sum(q) | Normalize per source image; average active images. Current method. |
| episode_region | q_R / sum(q) | One episode-wide normalization. |
| image_token | q_R A_R / sum(q A) | Normalize per source image; average active images. |
| episode_token | q_R A_R / sum(q A) | One episode-wide normalization. |
| image_shuffled | permuted q_R / sum(q) | Current image-region loss after within-source C-score permutation. |

Define activity from original q before multiplying by area or permuting. Empty
and EPS behavior must retain the native valid-input semantics. Shared model
parameters and Adam mean equal coefficient mass is not equal gradient norm.

For shuffled allocation, one stable source-keyed seed20261029 determines a
uniform permutation of active candidates within each image. Preserve the
score multiset, total image coefficient, candidate geometry and original
diagnostic scores. Record assigned versus original scores separately. Do not
force a derangement, select a favorable permutation, shuffle across images,
or apply old role strata (current method has no roles). Singletons remain;
report at-least-two-candidate coverage and actual moved-score/candidate rates.
This is one mechanism permutation, not an independent uncertainty estimate.

Every arm resets model/optimizer and computes the same C diagnostic before
updates; do not replace C with a synthetic distribution. Reuse native candidate
and inference paths. Whether validated diagnostic caching can be used is an
engineering question requiring identity checks, not permission to skip counts.

## Execution Admission

Create a new source snapshot, variant identities, source/parent/config/input
hash-bound manifests and a retained systemd service. Do not alter or restart
any old suite. Pure tensor loss/gradient identity and independent CPU reference
tests must pass first. Independent code audit must complete before GPU work.

Native image_region must pass a real RGB GT-free identity gate against the
original C implementation, including masks/queries/C/loss/gradients/parameters
and outputs under the existing deterministic execution and thresholds. Other
arms need actual finite optimization gates and exact coefficient accounting.
Do not loosen a gate or auto-retry after failure.

Use one worker/shared GPU lock, MemoryMax12GiB, swap1GiB, OOMPolicy stop,
Restart no, RemainAfterExit yes. Stream prediction/count records. Seal all five
arms on both full populations before any new full GT scoring. No test-label
access during candidate admission/diagnosis/updates. Keep all negative/null
results and failures. Verify ten canonical score rows, exact populations,
source/config/input/parent/prediction/score/barrier seals, true service terminal
and actual operation/time accounting before analysis.

## Fixed Analysis

Report all ten pooled mIoU outcomes, not mean per-image IoUs. Report image-minus-
episode contrasts separately for region and token pooling; region-minus-token
contrasts separately for image and episode grouping; their difference of
differences; and native-minus-shuffled paired-allocation contrast. Report actual
cost and eligible/moved permutation coverage. No bootstrap, confidence interval,
significance, untouched-test, or generalized causal claim is implied.

If image normalization does not improve accuracy, retain that outcome and
describe it as a budget convention, not an empirically validated performance
advantage. If shuffled weights perform similarly, do not claim candidate-score
pairing is established. No winning full arm automatically becomes the method.

## Current Implementation Work

Default-model author Ampere `01a18751-fde5-7092-b09c-9cf48ca0ad65` completed only new
`trace_ovss/method/allocation_controls.py` and
`tests/test_allocation_controls.py`, then CLOSED. Main-agent rerun passed all
147 CPU tests. This module intentionally rejects non-CPU entropy tensors and
is currently a CPU reference oracle, NOT a device-native GPU runtime or a
worker-ready implementation. No model was loaded and no scientific experiment
was executed. A device-native worker implementation must be independently
checked against this oracle and native C, with real-model gates before launch.
Do not wire this CPU-only function into GPU adaptation by silently adding loss
offloading or bypassing its validation. Normal package-import/worker integration
is untested. No new queue has been prepared or started. Controller, worker,
freeze/admission checks and an independent audit are still required.

### Device-Native Implementation Addendum

The subsequent device-native implementation is
`trace_ovss/method/allocation_runtime.py`; it does not offload the entropy,
loss or gradients. The pure CPU module remains its independent reference.
Five-arm worker/controller/admission/analysis are in
`tools/evolution_allocation/`. Native image_region calls the original
ImageBalancedTRACE objective directly; the other four controls use detached,
episode-local device-native caches. Both author agents are CLOSED.

The main-agent joined CPU run passed550 tests after targeted code-audit
repairs. These cover source/executed-byte identity, both-population label alias
guards, actual service invocation/resource admission, native tensor identity,
full scoring streams, coefficient/permutation coverage and no-retry semantics.
Independent auditor Singer01a1875f-d7ed-7430-bb45-9f4c89cf0fbb is rechecking
the final three fixes in docs/allocation-controls-code-audit.md. No scientific
result, prepared-suite identity, real RGB model gate or CUDA validation exists
yet. Passing CPU tests alone does not authorize launch.
