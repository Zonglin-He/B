# Image Full Appendix Writing Audit

## Scope and Identity

Authorized appendix author, not an independent reviewer. This document is an
authoring and claim-evidence check, not a paper score or acceptance assessment.
Worktree: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.
Owned files: `paper_evolution/appendix.tex` and this audit only.

The appendix describes **TRACE**, with the main title **TRACE: Context-Probed
Region Allocation for Test-Time Adaptation in Open-Vocabulary Semantic
Segmentation**. Its executed identity is `ImageBalancedTRACE(view_mode="C")`:
one context-perturbed view per selected candidate, detached region support,
within-source-image normalization, then a mean over active source images.
No causal expansion of the name or former two-view method identity is used.

No GPU, model initialization, inference, label scoring, result/source edits,
Git commit, main/preamble/claim-ledger edits, exporter edits, or figure edits
were performed. The Desktop, main checkout, and frozen reviews/archives were
not changed. Other owners' concurrent worktree changes are not reverted.

## Mini-Outline

1. Reproduction protocol: method identity, episode execution, image/crop/token
   definitions, candidate selection, exact C transform, dense readouts, prompt
   entropy, inherited settings, reconstruction, metric, and frozen provenance.
2. Objective algebra: equal source-image coefficient budgets, conditional
   rescaling cancellation, local singleton cancellation, area and gradient
   boundaries, and the absence of batch invariance or semantic safety claims.
3. Global selection: 128-image calibration per training dataset, 256-image
   confirmation per dataset, all six arms and all three confirmation orders,
   pre-confirmation C selection, and incomplete historical-exclusion coverage.
4. Computation: encoder inputs versus calls, synchronized model-side timing,
   full-worker finite checks, warmup denominators, and separate cohort costs.
5. Qualitative evidence: original candidate footprints and measured C, missing
   T and full pre-update maps, actual figure selection and provenance limits.
6. Reference protocols: reported MLMP operating points versus local native CD
   references, protocol mismatch, and the retained VOC20 reference deficit.

All main-requested anchors are retained: `app:protocol`, `app:readout`,
`app:theory`, `app:selection`, `app:cost`, `app:figures`, `app:native`.
The appendix is self-contained and does not depend on removed main equation
labels. Main's live anchors remain `eq:reference`, `eq:context-view`,
`eq:support`, `eq:loss`, `sec:method`, and `sec:allocation`.

## Paragraph Roles and Reverse Outline

| Appendix block | Paragraph roles in reading order | Evidence or explanatory endpoint |
| --- | --- | --- |
| Method identity and execution | Opening: define the single selected method. Method: reset, diagnose, update, reconstruct and define controls. Limitation: deterministic runtime is not replication. | `image_balanced.py`, confirmation/full workers, frozen runtime. |
| Images/crops/tokens/candidates | Definition: distinguish normalization and encoder units. Method: query components and initial confidence. Method: source-level top-K and deterministic ties. | Candidate and loader code; one-crop evaluated configuration is not the general multi-crop definition. |
| Transform and support | Method: exact RGB transform. Method: original-query probability at original tokens. Limitation: geometric preservation is not causal isolation. | Actual contrast 0.35, full-crop channel means, channel cycle `(g,b,r)`, clipping and nearest mask. |
| Dense readouts | Opening: evaluation and adaptation are different readouts. Method: per-layer entropy weights and raw-output combination. Method: prompt-specific adaptation logits. Reproduction: seven prompts and checkpoint identity. | `DenseCLIP.forward` and adapter; last 18 outputs, frozen pretrained logit scale, no CLS loss. |
| Image-normalized optimization | Method: region mean, detached local normalizer, active-image mean. Boundary: empty graph-zero and EPS behavior. Method: exact optimizer/parameter subset. Reproduction: inherited settings only. | Actual objective and engine; no area multiplier or extra scalar. |
| Reconstruction and metric | Method: overlap-average pixel logits. Method: query softmax before category extension maximum. Definition: full populations, background, ignore label and pooled mIoU. | Reconstruction/evaluation sources and frozen full protocols; 224-grid nearest labels, not native-resolution evaluation. |
| Frozen evidence | Reproduction: small set of stable suite/source paths. Protocol: gates, global prediction barrier, separate CPU scoring. Limitation: full-validation development exposure and incomplete parameter origin. | Prepared suite, completed seal, small canonical results, inherited provenance. |
| Theory | Advantage: equal explicit source coefficient mass. Boundary: rescaling and source-local singleton cancellation. Boundary: gradient interactions and non-invariance of joint Adam. | Algebra of the stated finite active-score objective, not an accuracy theorem. |
| Selection | Protocol: training cohorts, orders and six arms. Method: exact worst-dataset/tie rule. Evidence: selected C and retained negative calibration margin. Limitation: not best-arm or globally unseen confirmation. | Sealed training selection and generated cohort tables. |
| Cost | Method: inputs versus batched calls. Boundary: no full min arm or imported speed ratio. Reproduction: actual timing scope/finite checks/warmup. Evidence: distinct full and cohort tables. | New full costs are an exporter-owned dependency; cohort costs retain all arms/orders. |
| Figures | Definition: stored candidate measurement versus rendering. Boundary: unmeasured T and missing pre-update masks. Selection limit: selected examples are not representative frequencies or causal estimates. | Native new-C masks/support/final predictions and figure-owner provenance; no old assets included. |
| References | Evidence: reported MLMP values and negative VOC20 difference. Evidence: new C versus retained native TENT CD with mismatched budgets. Boundary: external operating points are not matched controls. | Existing reported/local sources; no extra baseline run or unsupported No Adapt value. |

The resulting flow is execution -> objective interpretation -> mode selection
-> computation -> illustrative evidence -> external comparison boundaries.
Historical reviewer chronology and audit-log narration are not the appendix's
organizing structure.

## Claim-Evidence Map

| Claim | Evidence | Status and boundary |
| --- | --- | --- |
| The new full method is single-probe, source-image-normalized TRACE. | Frozen `runtime/trace_ovss/method/image_balanced.py`, `tools/evolution_confirmation/run.py`, new full worker; six direct-C gates. | Supported as implementation identity, not a new safety claim. |
| Each active source has equal total objective coefficient mass. | Local normalized weights sum to one; active-source mean is external. | Supported algebraically for fixed finite active sets; gradient norm, accuracy and Adam trajectory are not equalized. |
| Within-image rescaling cancels; a local singleton's support cancels. | Detached numerator/denominator algebra and actual helper grouping by source ID. | Supported conditional on unchanged active sets and numerical behavior. A singleton image can coexist with other multi-region images. |
| C was selected globally before confirmation, using calibration only. | `trace_image_budget_20261029_0233/SELECTION_SEALED.json`, selection barrier, locked rule, full suite's pinned parent. | Supported; no confirmation/full winner substitution. |
| Calibration did not show universally positive C margins. | PC60 C minus Confidence `-0.272957` pp; Object `+0.579561` pp. | Supported and disclosed beside selection rationale. |
| C exceeds image Confidence in both confirmation datasets, all three orders. | `generated/image_budget_comparison.tex` and sealed canonical cohort values. | Supported descriptively; not best-arm, independent-unseen, CI or significance evidence. |
| New C exceeds matched Confidence on all six full splits. | Six completed canonical `cells/full/<dataset>/seed0/result.json` files and 18 arm rows; pooled-count arithmetic independently checked during writing. | Supported for these fixed development-exposed finite panels, not universal robustness. |
| VOC20 does not beat the reported MLMP point. | New C `83.48721918638837`; reported MLMP `83.76`. | Supported negative difference `-0.272781` pp; all-reference-win claim excluded. |
| A single probe reduces diagnostic inputs relative to two with identical candidates. | Actual zero/one/two schedules and candidate accounting. | Supported as diagnostic-input arithmetic, not full work or latency superiority. |
| Full timing includes observation guards and excludes setup/data/scoring. | Full worker's synchronized timer and `finite_observation` placement; inherited timing-scope constant. | Supported; no old cohort ratio transferred to full. |
| No T/full pre-update map is available for C full displays. | Native receipts retain C and masks, T null; `PREUPDATE_STATUS` explicitly unavailable. | Supported absence; no fabricated T, before/after rates or old two-view mechanism import. |
| Reported and native baselines are separate contextual comparisons. | Existing MLMP Table 5 references; retained native CD TENT `72.200299` under LR 0.001/10 steps versus C's inherited settings. | Supported with protocol mismatch; no equal-budget/equal-tuning external claim. |
| No new local No Adapt accuracy is established here. | This task does not certify equivalence of a historical reset readout to the new frozen branch. | Numerical claim excluded pending exporter-owned exact unchanged-readout proof. |

## Verified Canonical Full Values

Values are rounded only for display; every calculation above used canonical
unrounded values. Units: mIoU percent and signed percentage points.

| Dataset | Images | New C | C minus image Confidence |
| --- | ---: | ---: | ---: |
| PC60 | 5105 | 30.693152 | +0.355929 |
| COCO Object | 5000 | 32.594583 | +1.769488 |
| VOC21 | 1449 | 52.621491 | +0.296016 |
| VOC20 | 1449 | 83.487219 | +1.309732 |
| COCO-CD | 500 | 74.982118 | +7.317811 |
| COCO-Stuff | 5000 | 21.836221 | +0.042865 |

These values are not substituted from former full mechanisms, risk tables,
seed replays, product/normalization controls or the old 130-row standalone
study. Old min appears only as an explicit alternate arm in the retained
training-cohort tables and their interpretation.

## Verification Performed

- Read the research-paper-writing skill plus Method, Experiments and paper
  self-review guidance; constructed the mini-outline before prose replacement.
- Read actual objective, candidate, transform, loader, dense readout,
  reconstruction, optimizer and full-worker sources without changing them.
- Read small canonical full results; recomputed all 18 mIoUs from their stored
  pooled class intersection/union arrays and checked the six C/Confidence signs.
  This is arithmetic verification, not rescoring labels or inference.
- Recomputed full prepared SHA-256:
  `e0f7374042eadb65fbc8146de6dfcf35a71151dc1c95c115b0d4cf9c8444edaa`.
- Recomputed completed full seal SHA-256:
  `4c2efc97904573e64f586d59f26ce5d94e58a29ea1ca93fb58c71fb5c1ed1fb2`.
- Recomputed selection seal SHA-256:
  `90fa3a20097698e9b83cafd416a4b05fe7f033dcd8aadb027c970a9788ac11e4`.
- Read the completed queue state: six passed gates, six sealed prediction
  cells, six scored cells, 18 arm results, no error.
- Queried the real retained user service read-only: loaded, active/exited,
  MainPID 0, ExecMainPID 3694645, Result success, code/status 1/0,
  exit `2026-10-29 06:29:28 CST`. The suite's own
  `service_success_verified=false` is not reinterpreted as service evidence.
- Matched 32 small full-suite artifact hashes to the completed suite seal,
  including six gates, six results, per-cell prediction/scoring seals, the
  global prediction barrier and CSV. Matched 11 actually read mathematical,
  loader, readout and worker source files to the executed runtime hashes.
- `git diff --check` passed after the appendix rewrite. No full immutable
  prediction-payload rehash is claimed by this authoring task; the bounded
  full importer and canonical seals own that independent admission.

## Integration Boundaries

The appendix includes inherited `generated/configurations.tex` as settings
only, `generated/image_budget_comparison.tex`, `generated/image_budget_cost.tex`,
and the exporter-owned `generated/image_full_cost.tex`. Main owns inclusion of
the new full comparison/reference tables, not this author. No generated table
has been hand-edited or substituted while waiting for its exporter.

The full exporter subsequently published all three tables and provenance.
Their file hashes were checked against `generated/image_full_provenance.json`;
the prepared/selection identities and all six cells match the intended suite.
Its `no_adapt_reset_readout_reused=false` agrees with this appendix's absence
of a new-method No Adapt score. The appendix never substitutes an old cost
table or silently skips its new full-cost include.

Fermat's final `figure/image_full_provenance.json` was read after publication.
The appendix's figure section now states the actual first-400 selection rules,
native 16x16 mask geometry, exact teaser thresholds/ties, saved support values,
four named-class IoU cases and the offline-GT selection boundary. The teaser
is PC60 sofa image 151 / receipt candidate 158 and Object cake image 241 /
receipt candidate 108. Segmentation cases are CD cat 394, Object dining table
40, PC60 bus 199 and negative VOC21 table 85. No old figure is included.

Main owns the all-assets-ready compile. No successful final manuscript
compilation or rendered integrated-layout inspection is claimed by this task.

Final author-side asset hash check found 33 of 36 figure outputs matching the
then-published provenance. `image_full_pipeline.pdf`,
`image_full_pipeline.png`, and `image_full_pipeline_print.png` differed during
the figure owner's concurrent refresh; the two teaser identities and all four
segmentation identities remained unchanged. Figure-owner reconciliation of
these three pipeline hashes is required before the integrated freeze. This
task does not overwrite their assets/provenance or certify that live snapshot.

Appendix authoring is complete. Final owned TeX SHA-256:
`90cbd5174026be3940b9fb66710b0aae93bc938bf294e50eb44edc15f6cb6a09`.

## Author Self-Check

| Dimension | Question and answer | Disposition |
| --- | --- | --- |
| Contribution | Is the supported contribution a named single-probe allocation method rather than a claimed causal detector? Yes; the execution, source-budget objective and scope are explicit. | Pass as claim discipline, not a novelty/acceptance score. |
| Writing clarity | Can the appendix define all reproduction units/readouts without the old main formulas? Yes; source image, crop, token, query/category, prompts, logits, transform, loss and reconstruction are defined. | Source check passed; final rendered layout belongs to integrated compile. |
| Experimental strength | Are small/mixed/negative findings retained? Yes; PC60 calibration deficit, non-best-arm confirmation, tiny Stuff contrast and VOC20 MLMP deficit remain visible. | No significance or independent-uncertainty interpretation. |
| Evaluation completeness | Are cohort and full evidence separated and exposure disclosed? Yes; all arms/orders are included, July 4096 histories uncertified, official fulls development-exposed. | No exhaustive-history, equal-tuning or external equal-budget claim. |
| Method soundness | Do relative normalization limits follow the actual objective? Yes; source-local singleton/rescaling cancellation, empty behavior, shared gradients and no batch invariance are explicit. | No absolute safety guarantee; finite runtime checks are observation-only. |

Clarity/flow/terminology and claim-support alignment were checked after the
rewrite. The seven required anchors are present exactly once; all nine local
labels are unique, LaTeX environments are balanced, all four table includes
exist, and citation keys are present in the existing bibliography. Notation
uses `rho_R` as the method's sole support and `alpha_{R|j}` for within-image
weights. Source image `j`, prompt `a`, crop `b`, token `i`, selected/active sets
`R_j`/`R_j^+`, active images `I_+`, and `L_TRACE` match Main. Candidates remain
the same triple `(b_R, Omega_R, c_R)`; bookkeeping `j_R=j(b_R)` merely records
the source mapping. The transform is named `Shift`, with the unchanged `(g,b,r)`
channel cycle; no differently named second operation is introduced.
The cohort-only symbol `C_R` is defined once as the same support, not
an additional quantity. No extra `q_R` or global/local-alpha ambiguity remains.
Figure-owner pipeline hash reconciliation and Main's integrated
compilation/render check remain delivery dependencies, not requests for new
experiments. The old frozen review packet is unchanged.
