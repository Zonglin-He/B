# Candidate and Role Experiment Evidence Pass

Date: 2026-07-17

## Experiment Evidence

### Candidate-oracle 2x2 decomposition

- **Status:** complete.
- **Evidence type:** `newly_run`.
- **Workdir:** `/home/wwww/hzl` (the evaluator enters `external/MLMP`).
- **Environment:** `/home/wwww/miniconda3/envs/mlmp`, CUDA GPU, seed 0.
- **Command family:** `experiments/run_candidate_oracle_decomposition.py --datasets voc20 pc60 coco_object coco_stuff --max-images 240`.
- **Inputs:** deterministic prefixes of the clean validation loaders; 240 original images per dataset.
- **Key configuration:** predicted/GT-component candidates crossed with uniform/TRACE reliability; same dataset-specific optimizer and candidate budgets within a dataset; oracle candidates bypass the prediction-confidence proposal threshold; CLS disabled in all cells.
- **Output artifacts:** `candidate_oracle_4datasets_240/{cells.csv,decomposition.csv,run_info.json,protocol_and_interpretation.md}`.
- **Metrics used in draft:** full-resolution mIoU and changes relative to predicted candidates with uniform reliability.
- **Claim readiness:** `paper_ready` as a diagnostic decomposition; not ready as a deployable-method comparison or oracle upper bound.

### Candidate quality audit

- **Status:** complete.
- **Evidence type:** `newly_run`.
- **Command family:** `experiments/analyze_candidate_oracle_quality.py --datasets voc20 pc60 coco_object coco_stuff --max-images 240`.
- **Output artifacts:** `candidate_oracle_quality_4datasets_240/{candidate_quality_summary.csv,candidate_rows.csv,component_rows.csv,coverage_rows.csv,run_details.json}`.
- **Metrics used in draft:** area-weighted candidate purity and GT-component recall at IoU 0.5.
- **Primary evidence:** oracle purity is 1.0 in every available role; component recall rises strongly on all four datasets, while downstream mIoU remains non-monotonic.
- **Claim readiness:** `paper_ready` for the conclusion that candidate geometry and update policy interact.

### Nine-dataset semantic-role diagnosis

- **Status:** complete.
- **Evidence type:** `newly_run`.
- **Command:** `experiments/run_motivation_m1_m7.py --datasets voc20 voc21 pc59 pc60 coco_object coco_stuff cityscapes coco_cd waterbirds_seg --variants full_soft --max-images 240 --workers 4`.
- **Environment:** `/home/wwww/miniconda3/envs/mlmp`, CUDA GPU, seed 0.
- **Inputs:** deterministic first 240 original images from each clean validation loader. Cityscapes yields 8,640 model patches because of its tiled evaluator.
- **Key configuration:** current full-soft TRACE diagnostics; initially correct means pre-update region accuracy at least 0.5; correct-to-harm means an accuracy decrease greater than 0.05.
- **Aggregation:** unweighted macro average over dataset--role strata, not over regions.
- **Output artifacts:** `role_diagnosis_9datasets_240/region_diagnostics.csv` and `table4c_9dataset_role_expansion/{dataset_role_breakdown.csv,pooled_role_breakdown.csv,table4c_role_diagnosis.tex,provenance.json}`.
- **Metrics used in draft:** preserved support, residual context support, counterfactual stability, correct-to-harm rate, and AUC for inverse reliability/entropy/inverse confidence.
- **Claim readiness:** `paper_ready` for role differences and the boundary of the harm-ranking interpretation; `weaken_claim` for any universal claim that inverse TRACE reliability outperforms entropy.

### Figures and paper integration

- **Status:** complete.
- **Evidence type:** `newly_run_analysis_of_newly_run`.
- **Artifacts:** `candidate_oracle_figure/candidate_oracle_bottleneck.{svg,pdf,png}` and `role_diagnosis_figure/role_specific_source_diagnosis.{svg,pdf,png}`.
- **QA:** both `qa_report.md` files pass editability, venue-size readability, grayscale/color, negative-result retention, and manual clipping checks.
- **Paper:** `cfc_tta_paper_draft/main.tex` compiles to a 17-page PDF without undefined references, undefined citations, or overfull boxes.

## Protocol Risks

1. **Fixed diagnostic subset:** both new experiment families use deterministic 240-image prefixes, not full validation sets. They support mechanism and bottleneck claims, not benchmark-wide performance estimates.
2. **Single seed:** diagnostics use seed 0. This is acceptable for deterministic data-flow analysis but does not estimate optimizer-seed variance.
3. **Oracle information:** GT components and labels are used to diagnose candidate quality. Oracle rows are not deployable and are not performance upper bounds.
4. **Proposal-budget interaction:** the oracle keeps practical region budgets and can omit GT components; it also bypasses the prediction confidence threshold. Both choices are disclosed because they change the interpretation of the oracle.
5. **Joint-update label:** correct-to-harm is observed after adapting a batch of candidates. It is associated with candidate diagnostics but is not a per-candidate causal intervention label.
6. **Role availability:** not every benchmark contains evaluated thing, stuff, and background roles. Macro results therefore use 9 thing strata, 4 stuff strata, and 4 background strata.
7. **Cityscapes tiling:** its 240 images create 8,640 patches and many regions. Dataset--role macro aggregation prevents those regions from dominating, but per-image clustering should be retained in any future uncertainty estimate.
8. **No new 10,000-replicate CI:** the nine-dataset role table is descriptive and has no newly computed bootstrap interval. The existing four-dataset Table 4(c) retains its clustered 10,000-replicate CI and must remain separately scoped.

## Results / Setup Draft

We localize the remaining adaptation bottleneck with a $2\times2$ candidate-oracle diagnostic on fixed 240-image clean subsets. Replacing predicted candidates with ground-truth connected components under the same region budgets raises area-weighted purity to 1.0 and substantially improves component recall on all four datasets. Nevertheless, oracle candidates improve uniform-weight mIoU only on VOC20 (+2.15) and reduce it on PascalContext60, COCO Object, and COCO-Stuff (-0.33, -0.97, and -1.06). Candidate construction is therefore not the sole bottleneck: the candidate set and update allocation interact.

We additionally diagnose 75,762 initially correct candidates across nine clean datasets by semantic role. Background, stuff, and thing have macro correct-to-harm rates of 4.4%, 18.7%, and 18.3%, respectively, and exhibit distinct preserved, suppressed, and counterfactual support profiles. Inverse TRACE reliability outperforms entropy and inverse confidence for background harm ranking (0.728 AUC), whereas entropy or inverse confidence is stronger for thing and stuff. This broader test supports a role-conditioned update-allocation interpretation rather than a universal harmful-update detector claim.

## Remaining Blockers

- No blocker remains for reporting these two requested experiment families.
- The highest-value genuinely missing OVSS-specific experiment is vocabulary-composition stress: benchmark labels versus synonymous, semantically related, and unrelated distractor vocabularies under matched MLMP/TRACE prompts.
- If uncertainty is required for the nine-dataset role table, implement a dataset-aware image-cluster bootstrap rather than resampling 219k regions directly.
