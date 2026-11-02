# Per-Dataset Single-Probe TRACE Tuning

The user explicitly reopens research after the completed round04 delivery.
Preserve that paper, its6/10 independent review, original sources and old results.
This is a new parameter-search stage, not evidence that the tuned paper was
already reviewed. Work stays on codex/trace-self-evolution. No GitHub push.

## Scope

The original Desktop main_0728.tex and appendix contain eight main datasets:
VOC20,VOC21,PC59,PC60,COCO Object,COCO-Stuff,COCO-CD,Waterbirds-Seg.
All eight are included, not just the six already evaluated for selected C.
Cityscapes,ACDC and second backbone are not automatically added here.
Method is fixed ImageBalancedTRACE globalC with image-normalized region entropy.
Only batch_size,lr,steps,trace_max_regions vary independently per dataset.
View batch16, transforms, candidate policy, readouts, prompts, backbone,
224-square geometry and query-extension flags remain fixed.

## Selection

Use a fixed256-image training-source calibration panel per dataset,seed2027,
and a maximum80 unique configurations per dataset. No full-evaluation score
selects parameters, restarts or extra trials. Historical development exposure
is retained as a boundary, not erased by a new manifest.

Coordinate order: batch_size,lr,steps,K, each coarse followed by fine search.
Start with the inherited configuration, retain every deduplicated trial,
then select from all visited valid configurations. Maximize calibration mIoU;
within0.02pp choose lower measured model-side time, then deterministic config
tuple. Keep original inherited candidate even if proposed changes worsen it.
No guarantee that training selection surpasses every external baseline.

Batch coarse1..16 individually plus inherited value; local fine integers
within3 of the selected batch, bounded1..32. LR coarse
{1e-5,3e-5,1e-4,3e-4,5e-4,.001,.002,.003,.005,.008,.01,.015,.02}
plus inherited; fine winner times{.5,.75,1,1.25,1.5,2}, bounded[1e-6,.04].
Steps coarse{1,3,5,8,12,16,24,32,48}plus inherited; fine+/-1,2,bounded1..64.
K coarse{1,2,4,8,16,24,32,48}plus inherited; fine+/-1,2,bounded1..64.
Each next coordinate starts from the selected preceding configuration.
No trial is silently retried. CUDA OOM/nonfinite numerical failure may be
recorded as an ineligible trial with its complete failure receipt; source,
input, manifest or label-isolation failure stops the affected queue.

Freeze selected configs for all eight before any new full-evaluation scoring.
Then full seed0 arms: exact-reset NoAdapt, Uniform, Confidence, TRACE C.
These controls use the selected configuration and do not search baselines.
No extra MLMP reruns where paper values are already available.
All full predictions seal before any full scoring; calibration predictions
seal before each trial score. Scores pool class intersection/union counts.
Save costs, measured forward/input counts, config, trial CSV, selected seals,
source/input identities and result paths. Distinguish trial/full scores.
The first authorized calibration score seals annotation content hashes;
subsequent trial manifests pin that seal and scoring requires identical label
bytes. Prediction processes never open annotation bytes. Runtime verification
uses a closed inventory including root files and rejects executable caches.

## Waterbirds Boundary

User decision,2026-10-29: use the existing Waterbirds train split for tuning,
keep the original5322-image full, and explicitly report overlap as development
evaluation. This is an opt-in exception ONLY for Waterbirds. Select256 fixed
training examples(seed2027);all256 are also in full. Do not claim an independent
test, source-disjoint calibration or generalization from that full score.
No new compositing or254/2 imbalanced derived calibration is required.

The already-running0748 snapshot remains unchanged and still stops after seven
calibration selections. A fresh continuation must wait for that service's
successful retained exit plus all seven canonical selection seals;recompute
the same winner from recorded calibration results,import exact selected
configs without retuning,and then search ONLY Waterbirds. Full runs use all
eight frozen configs and exact unchanged full populations. All eight full
predictions seal before any full scoring. New tables/manifests must carry
Waterbirds development_overlap and its256-image calibration/full overlap.

The following records the original preparation boundary,not a continuing block
after the user's explicit opt-in:

The original all-matched-pairs full protocol has5322 images. Its existing
train.txt2458 IDs are all included in that population. Do not describe tuning
on those IDs and evaluating all5322 as disjoint. Preserve original evaluation
loader; first seek existing/reproducible source-disjoint calibration inputs.
If unavailable, explicitly block only Waterbirds calibration for a user
decision while other datasets progress. Never relabel evaluation as training.

## Worker Contract For New tools/evolution_tuning

Worker CLI:python -m tools.evolution_tuning.worker --phase predict|score|gate
--manifest PATH --manifest-sha256 SHA --output DIRECTORY;
score additionally--barrier PATH. Calibration/full manifests contain:
schema_version1,algorithm ImageBalancedTRACE:C,suite_root,runtime_root,
dataset,stage(calibration/full/gate),seed0,config,checkpoint,
panel(inputs,ordered_ids,classes,queries,query_to_class,reduce_zero_label),
arms and source_sha256(relative paths to frozen runtime).
Each input row:image,label,image_sha256,label_metadata[size,mtime_ns],official_id.
Calibration arms[image_C]; full arms[no_adapt,image_uniform,image_confidence,image_C].
Predict saves predictions/ARM/INDEX.npy, per-episode candidate receipts,
cost.jsonl, prediction_protocol.json, initial_state.json,
prediction_result.json and PREDICTIONS_SEALED.json with all file hashes.
Separate score verifies barrier links and prediction seal first, then loads
one label/prediction at a time and writes pooled result.json/score_inputs.json/
scores_by_episode.jsonl/SCORED_SEALED.json. Result contains arm_results with
miou_unrounded/intersection/union,images,full_split,costs,timings.
Re-use exact ExperimentTRACE/finite_observation/model-side counting and
RGB-only make_data helpers. Do not change production method math.
Gate compares selected-C experiment to direct ImageBalancedTRACE on four
RGB-only images, checks supports/loss/gradients/parameters/final labels and
one-view actual schedule; persistent attempt marker, no retry.

Controller and input preparation have separate file ownership. Input API:
prepare_inputs(package_root,data_root,output,calibration_size=256,seed=2027)
returns datasets mapping keyed by8datasets, each with config,calibration,
evaluation. calibration/evaluation are panel-like dicts with inputs and
vocabulary fields. Include explicit unavailable/block reason if calibration
cannot be source-disjoint. Do not read label contents during ordinary input
discovery; offline new calibration-data construction, if justified, is
separately documented and never model inference.
