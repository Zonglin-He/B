# Frozen Current-Method Ablation Execution

The controls implement the November16 paper-completion request. They are not
new test-selected main methods. Original MLMP and the adopted area-C source
stay byte-identical. The new wrapper calls the existing area-C objective exactly
for its intact default/selected arms; explicit interventions are separate names.

Fourteen core arms run on L14 Object/PC60/VOC20/Waterbirds and B16 Object/PC60/
VOC20. They are MLMP_paper, area_default, area_selected, no_region, no_global,
no_cls, no_region_no_cls, region_balanced_mass, uniform_covered_mass,
uniform_image_mass, spatial_shuffle_mass, semantic_top8, top4 and top16.
The selected-region/CLS two-by-two factorial shares every other selected field.
The default-area versus MLMP contrast keeps all official optimizer, prompt,
temperature and layer-fraction settings fixed. Selected-area versus no-region
isolates the regional bonus under the selected configuration.

Seven extra source controls on Object/PC60 use the original R02 scalar and
destructive-view definitions, calibrated on disjoint 64-source training probes.
These controls adapt those score sources to area-aware allocation; they do not
silently claim the old per-region TRACE objective or full DeYO reproduction.
Common-active candidate support and per-image effective bonus mass are fixed.
The same L14-fitted confidence-to-C distribution map is reused on B16.

All null maps preserve the effective sum after C_power. Constant-covered keeps
support; constant-image changes support; spatial permutation preserves the full
per-image score multiset. Region-balanced mass divides each region's C_power
coefficient by its area then matches the original image bonus mass. Thus a
pixel/CLS loss-balance change is not mistaken for a spatial-allocation effect.
Removing a loss is an explicit diagnostic, not an eligible searched main method.

Each arm runs three fixed within-eight-source-block arrangements. Batches
retain their actual default/selected size and cannot cross a block. All 256
diagnostic sources per family are official-evaluation-disjoint. Each arm sees
the same source order within a block even when its batch size differs.
Shared probe acquisition is logged separately and charged only as a declared
amortized diagnostic. These timings are not end-to-end speed comparisons.

Only sanitized MODEL_INPUTS metadata enters the GPU process. It has global
class vocabulary but no per-source label, Waterbirds group/label, selected GT
mask or scoring object. Dummy annotation paths are never opened. An inode/path
guard denies all true annotation files and the author training-GT cache.
Predictions and candidate diagnostics seal before a separate CPU GT scorer.

For every backbone/dataset stage, actual first-four RGBs compare default and
selected intact area-C against the existing independently executed original
wrapper on every real step, optimized parameter, final logit and decoded label.
Tolerance remains1e-5 and zero changed labels. Finite tests and reference gates
do not establish effectiveness. A failed gate preserves bytes and stops.

The finite execution uses original GPU lock,12GiB RAM/1GiB swap, no parallel
GPU, no retries, no driver/cache changes and no extra automation. Distinct
typed START and actual LIVE records bind the new stage's real PID/invocation,
cwd/argv/environment and effective caps. CPU preparation validates every required
parent global/child and the same actual retained successful stage identities.
