# Fixed Training Followup: Coverage and Budget

The first32 clean TRAINING outcomes motivated this followup before the v1 S5
scores existed. Native MLMP63.93948; restored global+CLS64.34857; area_C63.79545;
raw-C absolute budget61.65354; full all-region/raw-budget combination62.33561.
These are not validation results, independent replications or proof of superiority.
The old32-image v1 run continues unchanged, including all15 original S5 types.

Only two new fixed controls, both on the same native engine and same32 training
sources, will run after the entire v1 pipeline has actually exited successfully:

- `coverage_native_budget`: all semantic connected regions, refresh at0/5,
  native global pixel entropy+CLS plus area-weighted C bonus; retain native Adam
  step budget. Low C removes its extra regional emphasis, not the native update.
  This is explicitly an ablation of whole-update gating, not a claim that the
  original absolute-gating hypothesis was correct.
- `coverage_relative_budget`: same coverage/objective/refresh, but the step and
  relative image budget use area-weighted min(C/max(p0,EPS),1). Both C and p0
  come from initial frozen weights on the current candidate masks/classes.
  This measures retention under the intervention, not calibrated update benefit
  or causality. There is no fitted threshold, floor, learning-rate grid or best
  test-trial selection.

BaseLR.001,10steps,B2,checkpoint,prompts,vocabulary,readout,source order,RGB and
GT geometry stay fixed. Reuse v1 native baseline counts only after true parent
success and sealed artifact checks; replay its first4 NoAdapt/MLMP predictions
in every condition with zero label differences. All new predictions must seal
and exit0 before separate hidden-CUDA CPU scoring, which authenticates the same
source IDs, episode membership and GT byte hashes against v1 counts. All16
conditions and both new arms will be reported with shared whole-B2 paired draws.

These controls are not independent confirmation because they reuse development
images. Their results cannot be substituted for the original1449-image benchmark.
All v1 results, including unfavorable raw-C budget effects, remain visible.
No running v1 code is edited. New followup snapshots all original v1 critical
source bytes plus its own code. Original automation staysPAUSED and monitorOFF.

Root:
`[revision-results]/TRACE_coverage_followup_train_20261111_2230_v1`.
