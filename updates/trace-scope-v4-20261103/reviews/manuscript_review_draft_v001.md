# Fresh Independent Manuscript Review: Frozen draft_v001

## Material Findings

Severity: P1 is a substantive reproducibility/interpretation issue that needs disposition before treating the manuscript as self-contained; P2 is an important presentation/evidence-access issue; P3 is nonblocking editorial work. These are manuscript findings, not experiment failures, CODE findings, GT permissions or a numerical publication rating. Page numbers below are physical PDF pages, counted from1 independently for main and supplement. All file/line references refer ONLY to the authenticated archive, not evolving working files.

### R1 P1: The Data Protocols Behind The Largest Gains Are Not Defined Sufficiently To Reproduce Or Interpret Them

**Locations:** main Sec.4, PDFp.3, main.tex:103-112; Table1/main PDFp.5; supplement A.5/B, PDFpp.2-3, appendix.tex:60-80; references.bib:16-63. Enclosed evidence: generated/tuned_full_provenance.json, cells[dataset=coco_cd] and cells[dataset=waterbirds_seg], especially config and exposure.input_audit/selection.

The text gives names/populations, background conventions and historical exposure, but never defines what COCO-CD abbreviates, the three semantic labels, how its500-source evaluation population and masks were constructed, or what exact segmentation/split construction turns Waterbirds into the reported three-class5,322-source population. Source ID hashes, local loader names and local data-directory paths are useful provenance but do not define an independently reconstructible benchmark protocol. This matters especially because COCO-CD and Waterbirds carry9.57pp and11.72pp of the headline gains.

There is also a more specific exposure detail available in the enclosed record: Waterbirds has train_full_source_id_overlap=2458, whereas cal_full_source_id_overlap=256. The manuscript prominently describes the latter but not the former. These are different relationships. The text already disclaims independent testing; this is not a finding that the256 statement is arithmetically false, that the authors hid exposure intentionally, or that TRACE/CLIP was trained on2,458 images. It is a request to explain what the declared training population and the reported full development population actually are, rather than letting the visible256 count stand in for the entire split relationship.

The archive's bibliography contains VOC/Pascal-Context/COCO/COCO-Stuff/Waterbirds-associated references, but none of those five dataset references is cited in the TeX. The corruptions likewise lack a concrete implementation/version/resource description in the paper beyond their names, severities and provenance hashes. The review did not fetch an external implementation to fill in these missing definitions.

**Required disposition without new experiments:** add a compact dataset/protocol specification and citations: exact source/version and declared split population, label taxonomy/background/ignore rules, existing mask/construction/remapping procedure where applicable, train/calibration/full overlap relationships, and corruption generator/version/parameter resource provenance. Preserve all current populations, scores and exposure disclaimers. Do not replace the development population with a newly chosen test subset or imply a new held-out result. Existing metadata may be surfaced; no new scientific run is requested.

### R2 P1: The Central Legacy Comparison Has A Provenance Identity But Not A Self-Contained Operational Definition

**Locations:** main Sec.5.2, PDFp.4, main.tex:144-145; supplement D.2, PDFpp.3-4, appendix.tex:119-125; Table6/supp PDFp.13. Related normalization statement appendix.tex:54-56 and factorial definition appendix.tex:133-149.

The production C transform is specified with equations, contrast0.35, channel shift and a fixed readout. By contrast, the legacy comparator is described by five archived module names, an archive SHA and a role-dependent three-view/two-view count. The manuscript does not give the actual view transformations, the role-to-view binding and query-to-role rule, or the formula reducing those observations to the legacy reliability value. It also does not explicitly write the episode-normalized objective used in the2x2. Thus a reader cannot reconstruct what causes the Object -0.380pp / PC60 +0.142pp contrast or distinguish plausible but different legacy reliability operators from the paper alone.

A pinned hash is an identity of code, not a scientific definition of its calculation; that code/archive is not itself enclosed in this recoverable manuscript. Calling it a declared component embedding rather than a whole published reproduction is an appropriate restriction, but does not resolve the missing component definition. I do not infer that the implemented comparator is wrong or that it is a full historical-method baseline.

The factorial description is closer to operational, but should also specify whether rank/value permutations are within source image, which direction and deterministic ties are used, and exactly when natural/common-active support is applied relative to assignment and normalization. These details affect the interpretation of distribution-by-order effects.

**Required disposition without changing the experiment:** state the already executed legacy transforms, scoring formula, role/query mapping and both normalization equations, ideally with one short pseudocode block and explicit fixed constants. Define the existing factorial assignment unit/ties/support order. Retain all opposite-sign and null results. Do not redesign the comparator, add a new historical reproduction, tune it, or relabel it as a named whole-method baseline.

### R3 P2: The Fixed Cases Lack An Enclosed Semantic Palette Key, And Their Raster Annotations Are Too Small At Printed Size

**Locations:** main Sec.5.5/PDFp.6, main.tex:188; supplement H, appendix.tex:248-277; supplement Figs.1-4/PDFpp.7-10; generated/M12_asset_bindings.json:1-22; full archive membership.

All four stored composites are present, have the declared source identities/hashes and show actual failures as well as segmentations. Green/magenta overlays are correctly distinguished from attention/actual perturbation images. However, semantic GT/prediction colors have no human-readable class legend on those pages. The caption says their category indices must be interpreted within each dataset, and appendix.tex:277 says full palette mappings are retained in structured case evidence. The enclosed M12_asset_bindings.json contains identifiers, copied image hashes and author-side verification claims, not the class/color maps or full case legends. generated/evidence.json contains ten other sealed-summary groups but no M12 legend. No separate case-legend file is in the58-entry archive.

Consequently, this frozen package does not let a reader map an arbitrary displayed segmentation color to a semantic class, even though it can authenticate the PNG bytes. This is an accessibility/completeness finding about this archive, not a claim that the upstream M12 suite lacks those maps. I did not open that upstream suite to repair the gap.

At0.93textwidth, a960-pixel composite's roughly10-pixel raster panel labels/raw-score header are approximately5pt. They can be deciphered with zoom, and their pixels are not clipped, but are substantially smaller than the surrounding caption/body. The larger photographic/segmentation tiles remain inspectable. The body lists the four confidence/C pairs, so the numeric measurements are not wholly missing from the paper; the lack of a semantic key remains.

**Required disposition:** include the already recorded dataset-specific class/color legend, at least for colors appearing in these fixed cases, and bind any copied structured legends in the next version's manifest. Prefer readable vector/TeX panel labels and an outside-image measurement key while preserving the authenticated composite bytes and all four cases. No new raw-data read, inference, case selection, perturbation simulation or per-case score is requested by this review.

### R4 P3: Supplement Float Order Makes The Evidence Needlessly Difficult To Navigate

**Locations:** supplement D.1/D.2 text on PDFp.3 refers to Tables5/6, which appear on PDFpp.12/13. The M01 tables occupy PDFpp.5,6 and11, separated by four case pages7-10. Source appendix.tex:88-101,113-146,252-275.

All rows are present; I did not find cropped tables or words outside a PDF MediaBox. The issue is not hidden evidence. It is the9-10page jump from a control's definition to its numeric table, and the interruption of the M01 sequence by unrelated cases. The long empty areas on some float pages add navigation cost.

**Nonblocking disposition:** reflow/group the existing supplement tables with their related sections, place the four unchanged cases together near H, and give useful page/section navigation. Preserve all rows, nulls and negative results; do not solve the layout by dropping them or shrinking text further.

## Review Verdict And Intrinsic Scientific Limits

**Review completed. Disposition: material manuscript revision is needed for self-contained reproducibility and interpretation, primarily R1/R2 and then R3.** No score, target grade or desired acceptance outcome is supplied. No resource/model-capacity block prevented this review. The present scientific record supports a bounded operational allocation-rule result, not a validated independent risk estimator or a universal/equal-time TTA advantage. Resolving the manuscript findings does not remove the following intrinsic limits of its fixed evidence.

### L1 Contribution And Novelty

The specified contribution is a confidence-admitted connected-region pool, one particular candidate-preserving context response and detached within-image weighting of mean regional entropy. Its structural budget/cancellation properties are understandable. Relative to the enclosed related work, this reads as a focused allocation design with a conditional empirical analysis, not an established new theory of update utility. The paper has not shown that the C response carries independent candidate-risk information. A null mechanism result does not negate the operational gains, but neither does auditability or a larger artifact package establish methodological novelty.

The related-work section lists the relevant families but gives little explicit explanation of the closest conceptual difference between this rule, confidence/sample-selection weighting, perturbation consistency and the declared archived diagnostic. R2 must be addressed before that distinction can be evaluated well. Because full cited papers are not enclosed and external access was out of scope, I cannot certify priority or say this exact rule is already published. The appropriate correction is a precise novelty statement/relationship using the executed method, not an invented stronger claim or an additional baseline sweep.

### L2 Empirical Advantage Is Conditional On Selection, Geometry And Comparator

All eight selected operating points improve over their matched Confidence controls, including large CD/Waterbirds differences. Those controls are useful for the conditional weighting replacement because candidates/settings/readouts are matched. They are not independently tuned algorithm competitors. Eight protocols also do not constitute eight independent domains: VOC20/21, PC59/60 and Object/Stuff reuse source families/protocol variants. Historical exposure, Waterbirds development overlap and deterministic seed0 are material external-validity constraints, already acknowledged in the paper.

NACLIP B/16 C-minus-Confidence is positive on three clean populations, but Object/PC60 C remain below their own zero-update point estimates. Declared SCLIP loses2.395pp with a negative interval. This is a limitation of the declared transfer design, not evidence about every SCLIP implementation. The published/historical reference tables are appropriately not presented as matched state-of-the-art superiority. The current enclosed evidence cannot support such superiority or global independent-test language. No retuning, extra seeds or baseline expansion is prescribed here.

### L3 Mechanism Results Do Not Identify Candidate Utility Or A Universally Better Normalization

M01 is an isolated unit-weight intervention, not a candidate's contribution within joint Adam. All matched intervals are null with10,000/10,000 undefined draws; the adjusted point sign reversal is descriptive under sparse frozen strata, not proof that adjustment identifies a harmful C effect. The current conclusion that independent risk is unestablished is justified; concluding that C has no independent information would not be.

M02 supports the declared position-preservation contrast on Object, not PC60. M04's positive interaction does not identify every candidate's causal role, and EPS support changes remain relevant. M03 shows opposite diagnostic directions across datasets. Moreover, image-minus-episode differences on PC60 are negative for both C and legacy in Table6, so image balance's coefficient-budget property is not evidence that it universally improves accuracy. The algebra and conditional comparisons should remain separate. Current stronger-than-utility interpretations would not be supportable; no manipulation of the sparse strata, dropped undefined draws or substitute endpoints is justified.

### L4 Corruption And Compute Results Leave The Advertised Stronger Questions Open

The severity5 macro is positive, but four condition point differences are negative and elastic has a negative interval. G03's ten S1/S3 intervals are positive, while four of its five S5 intervals cross zero, with Gaussian-noise/contrast negative points. The after-known-S5 registration and reused S5 namespace are correctly disclosed. This supports those fixed conditions, not universal robustness, a monotonic severity law, an independent S5 replication or familywise discovery.

V01 includes the entire prescribed grid. Both ratios fail the fixed eligibility band and both near-time accuracy comparisons remain null. Same-step or nearest descriptive differences are not equal-time answers. These nulls should remain as they are; changing the grid, interpolating, choosing a favorable timing definition or replacing a null with a descriptive contrast is not a valid manuscript fix. Gate-only/common-observer costs are distinguished from historical warmup-excluded timings. No new compute experiment is requested.

## Actual Claim, Equation And Numeric Checks

The review read the complete frozen main.tex and appendix.tex, all included table sources, their enclosed structured summaries/provenance, and the actual main/supplement PDFs. No manuscript source was edited or authored by this reviewer.

### Claim-Evidence Map

| Main claim | Enclosed evidence actually checked | Assessment |
| --- | --- | --- |
| Eight fixed C-minus-Confidence point gains | Table1; all8 cells of tuned_full_provenance.json | Numbers/signs agree; conditional selection scope required. |
| M01 total2042, final unadjusted -11.707/-28.431pp, quality-adjusted +1.667/+4.881pp with nulls | M01 sealed summaries;32inside rows and both other-region tables | Agrees. All matched undefined counts retained; not independent risk identification. |
| Object spatial positive, PC60 unresolved | M02 all14rows; main mechanism table | Agrees, including both PC60 crossing-zero intervals. |
| Conditional distribution/order interaction | All16M04arms and24contrasts, both active-set studies | Agrees. No universal ordering/magnitude or candidate causality follows. |
| Object single-probe loss/PC60 gain | All18M03mode/contrast rows | Agrees; comparator definition still needs R2. |
| NACLIP positives, SCLIP negative | Four G-clean rows and enclosed summaries | Agrees; exact SCLIP declared-scaffold boundary matters. |
| S5 macro +0.581,11pointwins/4losses and mixed groups | All15G-S5 rows and5group/macro rows | Agrees; negative elastic retained. |
| G03 all ten mild intervals positive, four S5 intervals crossingzero | All45arm rows,15contrasts and Figure2 | Agrees. No fitted severity trend/AUC/new macro is shown. |
| V01 complete grid, both eligible outcomes null | All10grid rows,6matching rows, Figure3 and sealed time_matching | Agrees; same-step time-column n/a is not misused as nearest ratio. |
| Cached groups/events are descriptive, not candidate-local | All32A01group rows,8class subsets,40M05rows | Agrees; empty groups,2-source/1315undefined PC60 group and declining Object correction remain. |
| Four fixed cases, not attention or actual displayed perturbation | Four authenticated PNGs, PDFcasepages, M12bindings and captions | Fixed IDs/layout/negative case retained; missing semantic key is R3. |

The main objective's regional mean and within-active-image normalization are internally consistent as stated. The singleton/rescaling cancellation, area multiplier distinction and non-equal-gradient warning follow from those equations under fixed active sets. Diagnosis and optimization readouts, query/category distinction, EPS threshold and numerical gate tolerance are separately described. This is a paper-level mathematical consistency assessment, not revalidation of runtime precision or model implementation.

An independent formatting/projection check compared **356 generated table rows** with their enclosed JSON numbers, including signed/unsigned estimates, intervals and nulls. This was numeric projection, not mIoU estimation or bootstrap recomputation. Additional assertions checked the central interval-sign/count statements above. I found no material numerical transcription/sign error in the current main claims, inspected tables or two result plots. This is not a claim that every upstream metric calculation or published reference was independently rerun.

## Visual Inspection And Citations

All7main and20supplement pages were rendered at130DPI and individually viewed, not inferred from author QA receipts or code closure. Figure1's flow/formulas and the G03/V01 axes, markers, error bars and labels are readable. No fitted trend/interpolated operating point is visually substituted. The main result tables are legible. Supplement table rows, including null/negative cases, are present; no page has text outside its MediaBox according to the supplemental bounding-box check. The case raster annotation sizes and absent semantic legend are the qualifications in R3, and float navigation is R4.

All17 cited BibTeX keys resolve locally and the extracted PDFs contain no unresolved '??' markers. Dataset entries are present but uncited as noted in R1. I checked internal attribution/scope distinctions, not the full external source papers: the archive does not establish independent verification of MLMP Table5 numbers, every bibliography entry's historical accuracy, current literature completeness or priority. No external web lookup, old paper or old review/rating was used to supplement the archive.

The main/supplement use negative Object-legacy, SCLIP, elastic, severe-corruption uncertainty, M01 degeneracy, A01 sparse groups and V01 unmatched outcomes explicitly rather than relegating them to a missing-data claim. That does not resolve novelty/external validity, but it is not evidence suppression. Published reference rows, local historical baselines and current matched controls are distinguished.

## Five-Dimension Review Answers

1. **Contribution:** a specified relative allocation rule and conditional tests are present. Independent-risk interpretation and a broad novelty/priority conclusion are not established. Clarify the exact conceptual increment; artifact rigor is not itself a new scientific mechanism.
2. **Clarity/reproducibility:** production C, two readouts and structural normalization are mostly self-contained. Dataset construction/splits, legacy view/score semantics and case color decoding need R1-R3. No source rewrite was performed by the reviewer.
3. **Experimental strength:** meaningful fixed-setting gains coexist with small/negative/unresolved conditions. No independently tuned, unseen-test or equal-time superiority follows. The current conservative central numerical statements agree with the enclosed evidence.
4. **Completeness:** all declared fixed studies and signed/null outputs inspected are retained. Remaining scientific gaps are intrinsic under the fixed scope, not an invitation to add favorable experiments or claim unperformed estimands.
5. **Method soundness:** equations define a detached, source-balanced regional entropy objective; they do not guarantee useful gradients or independent candidate safety. M01/M02/M03/M04 do not remove that uncertainty, and the text generally acknowledges it. Missing comparator definitions still impede assessment of the empirical mechanism argument.

## Authentication, Work Performed And Limitations

Exact archive and manifest:

```text
fedcc2bf7c15a630c753a3ab26b8e7a150b8deefc9abfbb46d1485ce8d08f6ed  manuscript_draft_v001.tar.gz
3698241 archive bytes
98198a30e29131863be1f0260d0dab22712109f6927e86b40ca1eaf050b8a049  FREEZE_draft_v001.json
57manifested members /14688860bytes; plus manifest =58regular entries
5951a43a15d7fac357d65ee9d5036252e9c198e788bbb9186faa07c9d848417c  build/main.pdf (7pages)
7d9201479d30f7a1b57ee5aa3a7758fb38eb361bb3c4ab6c1ab527606c8bab6d  build/supplement.pdf (20pages)
f12a3ef2c11bef8d1c4691e6dc2f6c33f354ab499b76b7cbfdf1ff5b0810903a  main.tex
63a4dcd0759799497e6383ec02f6e8b2a0260f10a827d7b5ac257632dc961e42  appendix.tex
eda4079f5c01b41b1b8e3f7538449ba2ddf06272af453956778c8e3df33fb845  generated/evidence.json
096f91f464f98484e4253f95142947229154ca43ab30a49eefbf0c63eb5fb645  generated/source_provenance.json
```

Whole archive SHA/size, manifest SHA, exact membership, every member size/hash and aggregate size were verified BEFORE extraction. Absolute/traversal/noncanonical paths, links, duplicates and nonregular members were rejected. Extraction used exclusive files in a new private0700scratch directory, not tar.extractall. All57manifested files and the original archive were independently rehashed unchanged at the end of the review. Rendering/text/analysis intermediates were only under/tmp/scope-v4-review-v001-pz_4h369. Archived tools were read as needed for context but were not executed to access actual experiment sources or rebuild estimates.

Actual tools: Poppler PDFinfo/pdftoppm/pdftotext; standard-library JSON/HTML/hash/path analysis; direct page viewing. No estimator, bootstrap, model, scientific runtime, service/status poll, GPU/lock, actual GT/RGB/prediction array/weights/count stream or external network was used. Only the four supplied derived case composites were viewed as paper assets. Upstream source pins and terminal assertions were assessed as enclosed provenance, not freshly queried actual service evidence. Prior reviews/grades/targets and historical manuscripts were not read. No other agents were used.

Two reviewer-harness failures are retained here so they cannot be mistaken for paper findings:
- An initial XML ElementTree parse of Poppler bbox output failed at line1791/column88 because its text contains a control character. A standard HTMLParser pass then read all7+20page boxes and found no out-of-MediaBox word. The initial parser failure does not show a broken PDF.
- The first independent table-format assertion incorrectly expected leading plus signs on an unsigned arm CI, e.g.32.671[+31.786,+33.291] instead of32.671[31.786,33.291]. Correcting only the scratch check's signed/unsigned formatting made all356row checks pass. No paper/generated evidence was changed; this was not a scientific discrepancy or an exporter finding.

This report is the sole persistent reviewer output. The frozen archive/manuscript/code/assets and old evidence were not edited. Publication/artifact/GT permission is neither granted nor renewed. This raw draft_v001 verdict and its limitations must remain intact; any later author disposition/revision should identify a new version or append a clearly dated disposition, not overwrite the original assessment.

## Disposition Summary

R1/R2 require substantive specification using the already executed protocols; R3 requires accessible existing legend/measurement presentation; R4 is editorial navigation. Preserve every negative/null result, population, existing grid and bootstrap rule. Intrinsic selection, novelty, identification, transfer and equal-time limits remain after those corrections. **Completed fresh review of authenticated draft_v001; material revision required, no numerical rating or target verdict.**
