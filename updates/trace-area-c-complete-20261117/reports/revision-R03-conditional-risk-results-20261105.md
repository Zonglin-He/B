# R03 Conditional Isolated-Update Risk: Actual Results

## Conclusion

The prespecified primary hard-harm endpoint does not establish incremental
out-of-fold risk prediction from C beyond confidence/query/class/area, either
before or after initial-GT-quality control and with either linear or nonlinear
M0. All24 primary-endpoint pointwise intervals cross zero. This is a null
increment result in the fixed historical cohort/model families, not proof
that C contains no information or that TRACE cannot improve segmentation.
No primary-endpoint setting was selected as a winner.

For the secondary true-class-support proxy, Object nonlinear quality-adjusted
M1 is worse on Brier and log loss with positive pointwise95 intervals.
This secondary result is retained, not repaired or relabeled. It is not hard
prediction harm, and no multiple-comparison correction was applied. All16 AP
increment intervals cross zero. No evidence here supports an independently
validated deployment-time update-reliability claim.

## Actual Design And Coverage

- Exact complete old M01 confidence-TopK isolated rho1 cohort: Object1024 cases
  from128 selected sources/113 original episodes; PC601018 cases from128
  selected sources/105 original episodes. All2042 cases usable,0undefined cases.
- Independent candidate updates were reused, not rerun. C never entered the
  old isolated unit-weight loss. Original final native checkpoints: Object30,PC607.
- Primary harm: final inside-region net_correct_pixels<0. Ties are non-harm.
  Secondary: mean_true_class_support_delta<-.001, not hard harm.
- M0: confidence/query ID/mapped class/area fraction; M1 adds C. Both models
  additionally receive original theta0 inside-region correct/valid fraction
  in the quality-adjusted variants. GT quality is offline only.
- Five original-episode GroupKFold folds, shuffle seed20261105, same assignments
  across models/endpoints/quality/families; source images do not cross groups.
  All standardization/category encoding is training-fold-only.
- Fixed L2 logistic C1/lbfgs/max_iter2000. Nonlinear M0 includes total-degree1..3
  numerical monomials/interactions; M1 adds only linear standardized C.
- Shared10000 whole-original-episode draws within each dataset for both models
  and all variants. Pointwise percentile95, no multiplicity adjustment. All
  48 intervals retain10000 draws and have0undefined draws; none were discarded.
- Deltas below are M1-minus-M0 in original score units, not percentage points.
  Negative Brier/log-loss and positive AP favor adding C. AP is noninterpolated
  average precision, not trapezoidal PR area.

## All Prespecified Results

| Dataset | Endpoint | M0 Family | Initial Quality Control | Delta Brier [95% CI] | Delta Log Loss [95% CI] | Delta AP [95% CI] |
|---|---|---|---|---|---|---|
| coco_object | primary | linear | False | -0.000363891 [-0.002227295, +0.001476814] | -0.001368597 [-0.006348006, +0.003652974] | +0.006182722 [-0.002383033, +0.017273499] |
| coco_object | primary | linear | True | -0.000039992 [-0.001191281, +0.001109536] | -0.000261995 [-0.003357248, +0.002825762] | +0.001437783 [-0.003069125, +0.006227306] |
| coco_object | primary | nonlinear | False | -0.000647677 [-0.002770909, +0.001479009] | +0.001424398 [-0.005710420, +0.009849291] | +0.008549844 [-0.000562282, +0.020378731] |
| coco_object | primary | nonlinear | True | -0.000209981 [-0.001451890, +0.001029048] | +0.002370780 [-0.002293592, +0.008549738] | +0.002421358 [-0.004096733, +0.009903381] |
| coco_object | secondary_support | linear | False | +0.000432403 [-0.001771481, +0.002758147] | -0.000449827 [-0.006439926, +0.005729642] | -0.000821615 [-0.014949602, +0.013265492] |
| coco_object | secondary_support | linear | True | +0.000461869 [-0.000180126, +0.001120567] | +0.001276135 [-0.000573367, +0.003131432] | -0.002199112 [-0.006493193, +0.001275036] |
| coco_object | secondary_support | nonlinear | False | +0.000650500 [-0.001393431, +0.002814828] | +0.000528245 [-0.004950468, +0.006311962] | -0.001377996 [-0.014788141, +0.009347110] |
| coco_object | secondary_support | nonlinear | True | +0.000588862 [+0.000053207, +0.001153076] | +0.001677603 [+0.000292878, +0.003149925] | -0.001662745 [-0.004056063, +0.000806547] |
| pc60 | primary | linear | False | +0.000369794 [-0.000131952, +0.000933721] | +0.000807264 [-0.000332355, +0.002037589] | -0.001793029 [-0.005207467, +0.001129961] |
| pc60 | primary | linear | True | +0.000357462 [-0.000142883, +0.000905719] | +0.000761994 [-0.000342643, +0.001962512] | -0.002672445 [-0.005924339, +0.000062112] |
| pc60 | primary | nonlinear | False | +0.000229945 [-0.000448286, +0.000949743] | +0.000475534 [-0.001027504, +0.002067130] | -0.000697667 [-0.004344277, +0.002782737] |
| pc60 | primary | nonlinear | True | +0.000549095 [-0.000075441, +0.001167724] | +0.001266884 [-0.000161996, +0.002670626] | -0.001968284 [-0.004649079, +0.000539054] |
| pc60 | secondary_support | linear | False | -0.001189138 [-0.004337189, +0.001831764] | -0.002105415 [-0.009332543, +0.004817982] | +0.002025028 [-0.006588742, +0.010512780] |
| pc60 | secondary_support | linear | True | -0.000115605 [-0.000798504, +0.000564866] | +0.000563119 [-0.001517237, +0.002667537] | -0.002009499 [-0.004465820, +0.000435150] |
| pc60 | secondary_support | nonlinear | False | -0.001158311 [-0.004434004, +0.001987822] | -0.001657604 [-0.009109258, +0.005554281] | +0.003223485 [-0.005130722, +0.011565964] |
| pc60 | secondary_support | nonlinear | True | -0.000001189 [-0.000543578, +0.000536447] | +0.000639574 [-0.000959700, +0.002214263] | -0.001659897 [-0.004055420, +0.000589489] |

## Interpretation Limits

This was defined after old M01 aggregate/matched findings were known and before
the new OOF fits/results. The historical128-source confidence-selected cohort
and prior exposure remain; group-held-out prediction is not untouched test
data, independent experimental repeats or full-population generalization.
Intervals condition on the fixed OOF fits, not model refitting or run/order
uncertainty. Finite polynomial controls and L2 parameterization do not constitute
a conditional-independence test. Predicting the effect of an isolated unweighted
update is not the causal effect of C or proof that C-weighting prevents harm.
Old M01 matched reversed points and undefined adjusted CIs remain unchanged.
The old prediction terminal capture still lacks ExecMainCode; its exact limited
historical profile preserves that missing evidence. New R03 has its own complete
actual successful terminal envelope and does not invent the historical field.
The paper should describe context-conditioned self-supervised allocation, not
claim independently established update-risk estimation from these data.

## Actual Execution And Evidence

Independent exact derived admission and main frozen validation preceded one
manual capped CPU run. Actual1/1/1 threads, hidden CUDA,4GiB RAM/512MiB swap,
CPU100,Nice10,idleIO,OOMstop,Restartno,RemainAfterExit,KillMode control-group.
Dispatch and immediate PID/cwd/argv/environment/invocation/effective-cap checks
preceded manual LIVE. New original GT/RGB/NPY/NPZ/model bytes were not read.
Derived scalar/outcome/trajectory JSON was explicitly admitted; do not call
this a zero-derived-read analysis. No GPU or new model inference was used.

Actual service trace-revision-r03-20261105-2130.service:21:52:35->21:58:41CST,
PID/Exec3381719,invocationd799bff48d69488fb1cff5cc81b8ff6d; retained loaded/
active/exited/MainPID0/Code1/Status0/Resultsuccess. Main verified all54 original
output identity/hash pins,47875427B, all2076 original source graph bindings,
11 runtime hashes, protocol/admission/audit/start/LIVE and the global seal.
Worker analysis interval360.651661578s includes its declared IO/projection/fit/bootstrap work; retained service peak memory is not a scientific compute comparison.

```text
PREPARED 53d56c4a8f3cc34619fd0c955f8b226d9214a9732baffd63530472a22cc20d58
SOURCE_SPEC 7439772b0ad5219f5db9f3733e4040eda45cadcb9179d941e8f1e37391340091
PROTOCOL abac35c1e4a806d9889953b54b72642c76a0113f752e8b37c29b473e2d6df4d3
ADMITTED 0a5bb112f8d06278aba5742dce8c0185b607d15c72fa239536d2419e8a63711d
REPORT 6680b08f407be65462ce55d4b1be1f85f171cf44dc3bd279113e0f4795c64b96
GLOBAL 266a1bac04868df67f6c0696fd1533c3deeaa63b46452b1392c6223e2a91e278
TERMINAL 0c226821ef05d34ee50017b20bdf14c755dbfa832611e4650452dab0c498e40c
```

This document only formats the bounded sealed terminal summary. No metric,
fit, bootstrap, source outcome or binary payload was recomputed/read for export.
