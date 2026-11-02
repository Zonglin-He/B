# TRACE Tuned Full-Split Claim-Evidence Ledger

Author-side ledger, not an independent review. Current identity is
ImageBalancedTRACE with one common context probe C, source-image-normalized
region entropy, and per-dataset calibration-selected B/lr/steps/K.

## Evidence Identity

- Full continuation: `/home/wwww/hzl/results/trace_single_probe_waterbirds_full_20261029_0817`.
- Prepared SHA256: `4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3`.
- Service `trace-waterbirds-full-20261029-0817.service`: loaded active/exited,
  MainPID 0, Result success, ExecMainStatus 0, verified after completion.
- Queue completed 2026-10-29 18:14:48 Beijing: eight datasets, four settings,
  32 full outcomes. All eight full predictions precede any full scoring.
- Seven calibration selections imported from `trace_single_probe_tuning_20261029_0748`;
  401 configurations there plus 54 Waterbirds configurations = 455.
- Complete populations: VOC20/VOC21 1449 each, PC59/PC60 5105 each,
  Object/Stuff 5000 each, CD 500, Waterbirds 5322.
- Current source method matches the frozen runtime. No formula, geometry,
  backbone, prompt, diagnostic transform, or layer changes during tuning.
- CPU importer and new figure adapter validate their explicit source scope.
  Payload completeness rests on the frozen controller's completion validation;
  do not claim a second full current prediction-payload rehash from metadata alone.

## Full Results

| Dataset | No Adapt | Uniform | Confidence | TRACE C | C - Confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC20 | 79.043103 | 81.933983 | 82.244527 | 83.614345 | 1.369818 |
| VOC21 | 49.661701 | 47.357550 | 50.247292 | 53.380927 | 3.133636 |
| PC59 | 31.591770 | 32.917767 | 33.171805 | 33.586753 | 0.414948 |
| PC60 | 28.950769 | 30.591691 | 30.573158 | 30.782843 | 0.209685 |
| COCO Object | 29.646619 | 27.512130 | 29.361107 | 32.671159 | 3.310052 |
| COCO-Stuff | 20.191588 | 21.800496 | 21.801697 | 21.911536 | 0.109839 |
| COCO-CD | 62.818450 | 60.145202 | 64.659582 | 74.231207 | 9.571625 |
| Waterbirds | 47.458443 | 41.621260 | 44.362656 | 56.077790 | 11.715134 |

All four settings execute independently from reset. Uniform/Confidence are
matched weighting replacements at the TRACE-selected configuration, not
separately tuned competitor methods. Labels are resized to224x224. Metrics
pool class intersection/union counts, not per-image IoU averages. One seed;
no independent seed dispersion, CI, bootstrap, or significance claim.

## Exposure and Comparison Scope

Seven datasets have historical development exposure. Current fixed256
training calibration is distinct from their full populations, but is not
certified free of every earlier historical candidate-pool exposure. Waterbirds
uses the user's explicitly authorized development-overlap protocol:256 chosen
train examples are inside unchanged full5322. It is NOT independent test
evidence. Prediction/score isolation does not erase calibration label exposure.

Published MLMP Table5 values are reported references, not local matched runs:
VOC20 83.76, VOC21 50.78, PC60 27.99, Object28.84, Stuff21.25. Current TRACE
differs by -0.145655,+2.600927,+2.792843,+3.831159,+0.661536pp. Local native
CD TENT72.200299 gives a2.030908pp reference difference; native backends,
settings and episode composition are not matched. No unverified PC59 or
Waterbirds external baseline value is supplied. No new MLMP/WATT/CLIPArTT run.

## Claim Map

| Claim | Evidence | Status / boundary |
| --- | --- | --- |
| One candidate-preserving context input provides the score. | Frozen C implementation, RGB-only gates, recorded view/input counters. | Supported; no suppressed view, role branch, minimum, or dominance. |
| Score is original-query mean probability at original candidate tokens. | Executed evaluator and saved candidate receipts. | Supported; behavioral support, not causal source identification. |
| Region averaging removes a direct area multiplier. | Loss equation and executed tensor objective. | Algebraic property, not equal gradient norms. |
| Source normalization fixes per-image coefficient mass. | Sum of within-image normalized scores is one. | Supported; batch composition and Adam interactions remain. |
| Low absolute support suppresses an image's update. | Positive common rescaling and singleton weight cancel. | Rejected; score allocates relative regional priority. |
| C exceeds matched Confidence on all eight full protocols. |32 canonical pooled outcomes. | Supported descriptive comparison; controls use TRACE-selected settings. |
| Region construction alone explains the improvements. | Same candidate sets across Uniform/Confidence/C; large CD/Object/VOC21 contrasts. | Rejected as a complete explanation. No isolated per-candidate causal claim. |
| C exceeds No Adapt on all eight protocols. | Exact-reset No Adapt in same current full suite. | Supported in specified resized/development setting. |
| C exceeds every reported MLMP point. | VOC20 83.614345 vs83.76. | Rejected; signed difference retained. |
| Single probe halves total runtime. | Only diagnostic input count halves vs two probes; measured times differ. | Rejected. Current full C ms/image:165.99,343.73,186.77,84.43,320.36,159.09,30.56,37.99 in table order. |
| C is universally best among probes. | Earlier fixed-panel T leads Object, episode-min leads PC60. | Rejected; mode selection separate from per-dataset tuning. |
| Figures prove population prevalence or update safety. | Bounded first400 selected examples, both improvement/regression cases. | Rejected; actual current-config examples illustrate only. |

## Narrative and Review Status

Story: unlabeled update choice -> confidence lacks context-change observation
-> one probe plus regional allocation -> fixed-candidate evidence -> measured
accuracy/cost. Detailed runtime history belongs in reproduction material,
not in scientific contributions or repeated defensive captions.

Older two-view and six-split inherited-configuration drafts/reviews remain
archived and are not the current eight-split manuscript's evidence identity.
The new manuscript compiled, rendered, checked fonts/references/overfull,
and was frozen for fresh independent round05 review. The actual rating is
5/10 (borderline reject), confidence4/5, no identified fatal issue in scope.
The reviewer is CLOSED; full reports are in reviews/round05.md/json. No previous
score carries over and the task's scientific stopping condition is NOT met.
See reviews/round05_disposition.md and docs/allocation-controls-contract.md
for bounded follow-up. New factorial/permutation work has no GPU results yet;
it must not be inserted into the current evidence table or claimed to repair
independent-generalization/competitive-selection limitations.

Five-dimension questions and paragraph-role reverse outline are in
`mini_outline.md`. No new GPU experiment, baseline search, GitHub push, or
source-data change is part of this manuscript integration.
