# COCO-Stuff Coordinate Tuning Evidence

## Experiment Evidence

- Dataset: COCO-Stuff clean validation.
- Search protocol: one parameter at a time on a deterministic 240-image proxy.
- Reranking protocol: the six best unique configurations on 960 images.
- Final verification: the three best reranked configurations on all 5,000 validation images, seed 0.
- The 240-image proxy is divisible by every tested batch size, so each batch-size trial sees the same number of images.

## Full Clean Results

| Rank | Main difference | mIoU | mDice | mAcc | Delta vs MLMP | Runtime |
|---:|---|---:|---:|---:|---:|---:|
| 1 | fragment guard eta 0.1 | 21.66 | 32.37 | 42.21 | +0.41 | 1581.6 s |
| 2 | fragment guard eta 0.0 | 21.64 | 32.34 | 42.20 | +0.39 | 1580.5 s |
| 3 | fragment guard eta 0.5 | 21.63 | 32.32 | 42.36 | +0.38 | 1583.5 s |

The selected full-clean configuration is `batch_size=16`, `lr=0.003`, `steps=24`, `cfc_max_regions=12`, `cfc_max_regions_per_class=10`, `cfc_tau_cf=0.32762`, `cfc_rho_min=0.5`, `cfc_min_region_conf=0`, `alpha_cls=0`, and `cfc_fg_guard_eta=0.1`.

The previous canonical seed-0 result was 21.61 mIoU, so the matched-seed gain was +0.05. A subsequent full-clean three-seed rerun produced 21.66, 21.53, and 21.53, or 21.57 +/- 0.08. This is lower and less stable than the previous canonical configuration at 21.61 +/- 0.02. The tuned seed-0 winner is therefore retained as a rejected search artifact and was not adopted as the canonical COCO-Stuff configuration.

## Protocol Risks

1. Proxy and rerank values are selection evidence only and are not paper-ready benchmark results.
2. Selection on seed 0 introduced measurable tuning bias. The subsequent three-seed verification did not improve the previous canonical mean, so the configuration was rejected.
3. The MLMP value of 21.25 is the existing reference used by the project, not a newly rerun matched-seed baseline in this tuning job.
4. The gain over the previous canonical configuration is small (+0.05 mIoU), despite the broad coordinate search. It should be described as a verified incremental improvement, not a large breakthrough.

## Artifacts

- Coordinate trials: `coordinate_trials.csv`
- Large-proxy rerank: `large_proxy_rerank.csv`
- Full-clean verification: `full_clean_verification.csv`
- Selected configuration: `best_full_config.json`
