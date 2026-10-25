# Reviewer follow-up experiment handoff

User request: execute the minimal additional evidence panel, excluding Figure 1
and typical failure-case artwork. Latest override: use the original Waterbirds
dataloader for internal alignment, not the strict-test-only wrapper.

## Current execution

- Output: `/home/wwww/hzl/results/reviewer_followup_20261010_123627`
- Entrypoint: `/home/wwww/hzl/tools/reviewer_followup/run_queue.py`
- Queue PID at launch: 313806. Check queue_state.json and the live PID; do not
  treat this handoff's status as a current snapshot.
- 22 GPU jobs: 6 smoke plus 16 regular, fully serial, model seed 0.
- Seven CPU unit tests and all six GPU smoke cells passed. Cached full-policy
  predictions exactly match native TRACE in both smoke datasets. NoAdapt C->W
  was zero. The first regular COCO-CD control completed, then VOC21 started.
- Do not modify frozen sources/configs while this queue runs. Do not retry
  failed cells silently or resume the cancelled WATT/CLIPArTT experiments.

## Workload

1. Full-episode controls on COCO-CD, VOC21, PC60, COCO Object: uniform,
   confidence, horizontal-flip JS consistency, q, q(1-d). Every policy shares
   candidates, initial model, pre-predictions, region-normalized objective,
   lr, steps and batch size. Three-view statistics are cached once per episode;
   all five policies perform their own complete optimizer trajectory.
2. COCO-CD and Waterbirds context group mIoU and foreground identity flip /
   foreground-to-background omission statistics. Methods: NoAdapt, TENT, TPT,
   MLMP, TRACE. Fresh Full TRACE CD predictions from item 1 are reused.
3. COCO Object regrouping: three fixed full-image-set permutations, identical
   model seed and configuration, changing only order and batch composition.
4. Offline same-mask q-risk comparison already completed. Primary isolated
   harmful endpoint stays GT probability drop >= .001. Existing .005/.01/.02
   sensitivity is retained; no threshold is selected to favor TRACE.

Waterbirds uses all 5322 original-loader images (2458 train, 296 val, 2568 test).
All methods use the same population. Report these as internal mixed-population
alignment, not official test-only results. Do not combine with the previous
strict-2568-test baseline table.

## First formal result

COCO-CD, 500 balanced-test images, full released config, seed 0:

| Weight policy | mIoU | Actual correct-to-wrong pixels (%) | Worst-group mIoU |
| --- | ---: | ---: | ---: |
| Uniform | 66.66 | 2.511 | 44.55 |
| Confidence | 67.29 | 2.267 | 44.67 |
| Horizontal-flip consistency | 71.49 | 1.711 | 47.78 |
| Shared quality q | 75.15 | 2.007 | 50.48 |
| Full TRACE | 75.16 | 2.034 | 50.49 |

This supports stronger final segmentation than the simpler first three weight
policies. It does not support a sizable extra dominance contribution, nor the
claim that Full TRACE minimizes correct-to-wrong pixels versus every policy.
Preserve both findings. Current q-risk also fails to beat entropy on three of
the four isolated-candidate datasets; do not relabel it as universal safety.

## Artifacts

- `tables/episode_controls.csv`: completed formal controls only, no smoke rows.
- `tables/context_groups.csv`: completed regular grouped/confusion results.
- `tables/episode_regrouping.csv`: fills when permutation cells complete.
- `offline/q_risk_three_seed.csv`: post-hoc aggregation, not new GPU outcomes.
- `offline/cached_context_groups.csv`: earlier CD TRACE-only variants, clearly
  distinct from this package's newly run formal cells.
- Each policy stores prediction NPZs and a ledger sealed before GT scoring,
  followed by per-image confusion/IU and true top-1 C->W counts. C->W divides
  actual correct-to-wrong pixels by initially correct valid pixels, without a
  tunable probability threshold. Weight controls share this denominator.

## GPU environment

The host's loaded kernel driver is 580.173.02, while installed user libraries
are 580.178.04. CUDA was restored only for experiment processes by extracting
the hash-verified official 580.173.02 NVIDIA redistributable and supplying its
lib directory through LD_LIBRARY_PATH. No reboot, driver module unloading,
system package changes, or desktop interruption was performed. The queue
supplies this environment automatically. System-default nvidia-smi may still
report a mismatch; use the local matching binary/library path recorded in
environment.json for monitoring.
