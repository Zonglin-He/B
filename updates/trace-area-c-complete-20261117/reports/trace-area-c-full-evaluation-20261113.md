# Fixed Clean Parameters, Full Evaluation

Actual human instruction: finish clean confirmation, then directly evaluate the
complete datasets. Copy each clean L14 configuration to its B16 counterpart and
copy clean VOC20 to all15 VOC20-C S5 conditions. Keep MLMP at paper settings.

## Selection And Scope

The existing eight clean L14 training confirmations select one complete22-field
configuration each. Existing completed predictions/scores are reused with their
actual successful process identities. The current model child is not stopped.
Only the old finite dispatcher is replaced; the old35 source files stay unchanged.
Missing confirmation children run the original frozen implementation and inputs.
Once all eight reports succeed, export CONFIGS.json before any full-test score.
No further128-training-image B16/S5 transfer stages are scheduled.

Full clean L14 populations: Waterbirds test2568, Object5000, PC605105, VOC201449,
VOC211449, PC595105, Stuff5000, COCO-CD balanced test500. B16 populations:
Object5000, PC605105, VOC201449. All15 original VOC20-C S5 types use1449sources
each. VOC/PC/COCO val populations are complete benchmark validation sets, not
unavailable private challenge-test labels. No new benchmark or corruption is added.

Each B16/S5 configuration equals its matching clean L14 configuration in all22
fields, including batch1/2/4/8/16, steps, learning rate and numeric layer fraction.
That fraction maps to each model's depth without rescaling or test tuning.
MLMP remains official commit2a887de018b3f2c3b132f69fe806bab12a2d511f,
lr.001/10steps/B2/alpha_cls1/seven prompts/native readout and episodic reset.
One canonical evaluation order is fixed; every final short batch is retained.

## Inputs And Comparability

All arms use original clean RGB decoded by OpenCV, INTER_LINEAR224 and RGB channel
order. This is the frozen calibration geometry. Old clean native MLMP artifacts
used PIL bilinear resize in native_queue.images, so they cannot be reused here.
Both native NoAdapt and fixed MLMP are freshly evaluated for clean datasets.
The old predictions/results remain preserved and are not silently substituted.

S5 uses the original sealed21735 pre-resized RGB PNGs without a second resize.
Its official MLMP/NoAdapt raw predictions may be reused only with all-child/global
seals, same actual retained successful GPU identity, exact source IDs/vocabulary/
checkpoint/settings, and actual first4 label equality. New score code decodes the
same GT for all arms; old scalar scores are not used as evidence of equivalence.

Waterbirds uses only metadata split2, all2568 sources, and the already completed
official-source pixelwise mask audit. Test/validation labels never select new
parameters. Training/evaluation source IDs and original image inodes cannot
overlap. Historical mixed5322 evaluation and historical B16/corruption search
exposure are disclosed, not erased or described as untouched independent tests.

## Execution And Reporting

Sequential finite queue: finish8clean confirmations -> export configs -> prepare
full inputs -> eight L14 clean -> three B16 clean -> all15 S5 -> final summary.
No cron, monitoring agent or recurring automation is enabled. GPU uses the sole
reviewer_followup_gpu.lock,12GiB RAM/1GiBswap; CPU4GiB/1GiBswap/one thread/hidden
CUDA; OOMstop/Restartno. No unchanged-byte retry, cleanup or cap increase.

Each new prediction stage checks realfirst4 MLMP/default-area_C finite10steps,
loss/parameter/logit differences<=1e-5 and zero label differences. Compressed
prediction chunks bound memory. A separate CPU score requires sealed predictions
and the same actual retained successful prediction invocation. Indexed labels
use cv2 INTER_NEAREST224, ignore255 and the original reduce_zero_label mapping.

Report native NoAdapt, fixed MLMP and selected TRACE area_C with pooled class I/U,
per-class IoU, full counts, exact configs and costs. Paired intervals use10000
shared draws of contiguous16-source dependency unions, which cover every complete
episode for batch1/2/4/8/16. The final incomplete group is retained. For S5, share
the same source-group draws across all15types and take the arithmetic type macro,
never pool pixels across types or subtract independent marginal intervals.

Keep negative, null and undefined results. No best full-test trial selection,
no equally tuned optimum claim, no universal superiority guarantee. One canonical
order is not independent multi-seed evidence. Reused S5 baseline timings include
older observers, so do not claim matched throughput or compute ratios from them.
Final RESULT.json and RESULTS.md are produced only after all26conditions succeed;
FAILED/partial stages remain explicit and cannot be called a complete comparison.
