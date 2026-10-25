# Evidence Inventory for the 2026-07-17 Paper Revision

## Figure 5: Fragment False Certainty

- **Evidence type:** `newly_run`.
- **Source:** current-model diagnostic inference under `results/paper_revision_20260717_fragment_mining`.
- **Displayed cases:** VOC20 person, COCO Object orange, COCO-Stuff bowl, and COCO-Stuff refrigerator.
- **Traceability:** `cfc_tta_paper_draft/figure/Q2_fragment_false_certainty_qualitative_evidence.csv` records the source SVG, exact recovered component box, raw confidence, area, low-resolution GT IoU, source scores, and reliability.
- **Protocol risk:** selected high-confidence low-IoU examples illustrate a mechanism; they do not estimate fragment prevalence or aggregate gain.

## Figure 7: Final Segmentation Comparison

- **Evidence type:** `newly_run` for the revised COCO-CD capture; `preexisting_artifact` for the retained matched rows.
- **Source:** matched saved prediction tensors listed in `cfc_tta_paper_draft/figure/TRACE_final_segmentation_comparison_evidence.csv`.
- **Metric alignment:** printed values are target-class IoU, or mean target IoU for the two-class row. They are aligned with the displayed colored masks and ground-truth contours.
- **Revised COCO-CD case:** cat target IoU changes from 0.00 to 35.79; aggregate image mIoU changes from 49.25 to 67.41 but is not printed in the figure.
- **Protocol risk:** cases are deliberately selected positive-delta examples. Aggregate and uncertainty claims must come from the benchmark tables and paired bootstrap, not this figure.

## Expanded Table 4(c) / Controlled Source-Risk Diagnostics

- **Evidence type:** `newly_run_analysis_of_preexisting_run`.
- **Source:** 1,366 initially correct regions from the `full_soft` diagnostic variant in `results/requested_experiments_20260703_1020/07_qualitative_current_method/region_diagnostics.csv`.
- **Controls:** harm AUC is computed within raw-confidence bands; reliability quartiles are formed independently within every dataset and semantic-role stratum before pooling.
- **Uncertainty:** 10,000 image/patch-clustered bootstrap replicates at raw confidence at least 0.90.
- **Primary result:** source-risk AUC is 0.745 with 95% CI 0.596--0.862; entropy AUC is 0.489; the source-risk minus entropy CI is 0.048--0.423.
- **Claim boundary:** the difference from inverse confidence is not conclusive, and AUC establishes association/ranking rather than causal source identification.

## Compilation And QA

- `main.tex` compiles successfully with `latexmk -pdf`.
- No overfull boxes, undefined references, or undefined citations are present in the final log.
- The revised figures and expanded table were visually inspected in the compiled PDF at pages 8, 10, and 11.

## Candidate-Oracle Bottleneck Decomposition

- **Evidence type:** `newly_run` (2026-07-17).
- **Source:** `candidate_oracle_4datasets_240/decomposition.csv` and `candidate_oracle_quality_4datasets_240/candidate_quality_summary.csv`.
- **Protocol:** deterministic first 240 clean validation images from VOC20, PascalContext60, COCO Object, and COCO-Stuff; predicted/GT-component candidates crossed with uniform/TRACE reliability; identical per-dataset region budgets; CLS disabled in all four cells.
- **Primary result:** GT candidates reach purity 1.0 and substantially higher component recall, but oracle-uniform mIoU changes relative to predicted-uniform are +2.15, -0.33, -0.97, and -1.06 points across the four datasets.
- **Claim boundary:** GT candidates are best-case candidate geometry under the retained budget, not deployable methods or performance upper bounds. The result supports candidate/update coupling, not candidate construction as the unique bottleneck.

## Nine-Dataset Semantic-Role Diagnosis

- **Evidence type:** `newly_run` (2026-07-17).
- **Source:** `role_diagnosis_9datasets_240/region_diagnostics.csv` and `table4c_9dataset_role_expansion/pooled_role_breakdown.csv`.
- **Protocol:** first 240 clean validation images from nine datasets; initially correct candidates only; statistics macro-averaged over dataset--role strata. Cityscapes contributes 8,640 tiled patches, but macro aggregation prevents its region count from dominating.
- **Primary result:** background, stuff, and thing have macro correct-to-harm rates of 4.4%, 18.7%, and 18.3%. Inverse reliability AUC is 0.728 for background, but entropy/inverse confidence are stronger for thing and stuff.
- **Claim boundary:** the expanded analysis supports role-specific behavior and a role-conditioned update policy. It does not support a universal claim that TRACE reliability is the best per-candidate harm detector.

## New Figure QA

- `candidate_oracle_figure/qa_report.md`: PASS.
- `role_diagnosis_figure/qa_report.md`: PASS.
- Both figures retain editable text, preserve negative results, and identify the fixed diagnostic-subset protocol.
