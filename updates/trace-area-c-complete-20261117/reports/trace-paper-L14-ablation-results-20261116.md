# Current Area-C: Four L14 Diagnostic Ablations

Main verified all four prediction and score stages: actual same-invocation
retained Code1/Status0/success, all20 outputs per task, all210 exact arm/trial
points recomputed from sealed integer source counts without reopening GT.
This is256 evaluation-disjoint TRAIN diagnostic sources per dataset, three
within-eight-source arrangements; not full-test ablation results.

| Dataset | MLMP paper | Area default | Area selected | No region | No CLS | No global | Uniform image mass | Spatial shuffle mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Object |27.378898|27.355931|29.769773|29.875051|29.125891|21.508077|30.206107|30.180502|
| PC60 |27.366621|27.358463|27.481358|27.514792|27.331583|24.903174|27.537250|27.609752|
| VOC20 |85.157991|85.528056|86.401209|86.039895|83.773817|86.510922|86.047809|85.856465|
| Waterbirds |52.722757|53.527830|56.960559|56.515309|54.662732|54.972364|56.954472|57.042953|

Selected area minus no-region pointwise shared whole-block95 intervals:
Object -0.105278[-0.650409,+0.437585]; PC60 -0.033433[-0.213806,+0.226223];
VOC20 +0.361314[-0.127604,+0.895784]; Waterbirds +0.445249[-0.036985,+0.931406].
All four include zero. The region-specific contribution is not established by
these comparisons. Selected-system gains over MLMP cannot all be assigned to C.

Matched-default area minus MLMP: Object -0.022967[-0.091997,+0.097029],
PC60 -0.008158[-0.052349,+0.075683], VOC20 +0.370065[+0.096199,+0.671853],
Waterbirds +0.805073[+0.547878,+1.073891]. These are pointwise diagnostics,
not multiplicity-adjusted evidence of universal benefit or causal mediation.

Dropping global pixel entropy substantially harms Object/PC60, while dropping
CLS harms VOC20/Waterbirds. Mass-matched uniform/spatial controls remain fully
reported even where their means exceed area_C. No diagnostic winner changes
the adopted configuration, matching B16/S5 settings or fixed MLMP baseline.

Waterbirds diagnostic metadata group counts are134/1/1/120. Worst-group point
delta is+0.226537pp, but3605/10000 bootstrap draws are undefined because a rare
group is absent. The interval is unavailable; draws are not discarded, groups
not rebalanced after outcomes, and this is not classification worst-group accuracy.

## Exact Artifact Bindings

Study TRACE_paper_controls_20261116-1051-v1, PLAN
775c3f245a6282efb441c0bd4db39838a513331ebfd97b316ab72201226981ce.

| Task | Prediction PID/inv | Score PID/inv | RESULT SHA256 |
|---|---|---|---|
| L14_coco_object |1800534/54e5ab25eab8440cac8d18740289c0ab|1830855/6aab5b19327847b893dbb655213a25e7|ba4cea8e289b03cd99c3a2ccbbe4cef68cfeebcddbc29c5e2e76025429a028e5|
| L14_pc60 |1830951/4dad4efb20164a69997b1d7bb108a2a9|1834452/0bf0cdd755564a2ebe27eb5faacd76ab|6f1b6370b6d0f8768c5cc511c9d5cf5730e3fa46dcf6f286e40ee780abb97283|
| L14_voc20 |1834546/8adc04e92cb841f5b26fd89076409ed2|1843159/7d9b68a2337447438bac2dbd4b53578f|5ca0fba405a4551f4e8a8f56c829c5a1ea1723ae39bcb45ee7629859a9163e6b|
| L14_waterbirds_seg |1843215/4250259c91b043b99661eeef1ce66b8d|1850833/1c745394bb1c412cb7de7d2a9bb63860|89835bd0fd19e4d305b3c47ce42aa9366b8b7865b87d5e9e408e8531d33ea5b3|

All arms, contrasts, shared draws, undefineds and provenance remain in each
task/score/RESULT.json. B16 and all remaining paper stages are not completed
by this partial report. Historical PC exposure and prior R03 NULLs remain.
