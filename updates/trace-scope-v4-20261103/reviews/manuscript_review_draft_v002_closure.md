# Same-Reviewer Narrow Revision Verification: Frozen draft_v002

## Findings And Disposition

**Completed narrow R1-R4 verification, not a second full manuscript review or a publication rating. R1 is PARTIAL, with an unresolved construction-reproducibility limit; R2, R3 and R4 are CORRECTED for their original requested manuscript changes. No additional material regression was confirmed within this revision scope.** The original draft_v001 report, its verdict and its intrinsic limitations remain unchanged. This report does not assert that the paper is fully reproducible, scientifically unrestricted, accepted, or approved for any artifact/data/GT execution.

Severity retains the original meanings: P1 concerns substantive reproducibility/interpretation, P2 evidence accessibility, and P3 editorial navigation. Page references are physical PDF pages, counted independently from 1 for main and supplement. All manuscript file/line references below refer to authenticated draft_v002 archive members, not evolving working files. The author's REVISION_draft_v002.md was read as a disposition map, not accepted as the review verdict.

| Item | Narrow disposition | Remaining boundary |
| --- | --- | --- |
| R1 P1 | PARTIAL: consumed-protocol, label, overlap, citation and corruption documentation corrected | Local COCO-CD/Waterbirds mask and subset construction is still unrecovered; independent reconstruction remains unresolved. |
| R2 P1 | CORRECTED: executed comparator and factorial definitions supplied and source-consistent | Real-arithmetic specification is not a new bitwise runtime reproduction, whole historical-method baseline or causal-utility result. |
| R3 P2 | CORRECTED: enclosed semantic keys and readable vector panel/measurement keys supplied | Existing raster microtext remains small, but is no longer the sole interpretive key; cases remain nonrepresentative and descriptive. |
| R4 P3 | CORRECTED: related tables and controls grouped adjacent to definitions | Substantial float whitespace remains, a nonblocking layout cost rather than missing evidence. |

## R1: Documentation Corrected, Original Construction Still Unresolved

**Locations:** main abstract/main.tex:14 and main PDFp.1; main Sec.4/main.tex:108-109 and PDFp.3; supplement A.6/PDFpp.3-4; generated/protocol_specification_R1.tex:6-97 and :99-172; generated/protocol_specification_R1_provenance.json. Integration is appendix.tex:70.

The revision now defines eight consumed populations, their source counts, evaluation-label counts, background/ignore handling and native loader geometry. VOC20/21, PC59/60 and Object/Stuff are not presented as independent image domains. Dataset references previously present but uncited are now cited. The local cat/dog protocol is explicitly identified as 500 existing background/cat/dog pairs, not an additional official COCO task. Waterbirds is explicitly the full local 5,322-pair development population, not an independently recreated public test split.

The Waterbirds relationship is now stated correctly as C subset T subset F, with full population 5,322, all 2,458 declared training sources in full, and the selected 256 calibration sources nested inside training/full. This agrees with the enclosed retained provenance and with the frozen calibration-selection source. The two overlap counts are not added, and neither is misrepresented as proof of TRACE/CLIP training on those sources. The exact ranking string, seed 2027, ID tie break and final source-path sorting agree with the cited source. The historical 512-source exposure is not erased.

All 15 corruption parameter rows are supplied, including the five S1/S3 families, along with the frozen MLMP-derived source identity, recorded package versions, seed/order, BGR/PIL conversion qualification, corruption-before-resize order, original unwarped labels, and G/G03 OpenCV-nearest versus native PIL-nearest distinction. I independently checked the literal parameter lists of 13 families by static AST inspection, and checked zoom's exact arange expression and elastic's alpha/displacement/interpolation source directly. The six frost resource hashes are declared; the actual source's randint(5) restriction to the first five is disclosed. No frost image resource was reopened or newly authenticated by its pixels in this review. Known-S5 exposure and exact S5 reuse remain explicit.

**The residual is material and is not closed by disclosure.** The original COCO-CD selection/balancing, annotation rasterization/category conversion/overlap policy and possible compositing remain unrecovered. For local Waterbirds, original pairing, mask-writing assignment, crop/threshold/compositing RNG and 5,322-pair subset construction remain unrecovered. A current loader, ordered vocabulary, hash, root name or upstream classification citation cannot reconstruct those missing historical operations.

The revision now says this at the appropriate levels: abstract, main protocol and supplement. Therefore I do not retain the original omission/insufficient-disclosure criticism as an uncorrected wording defect. I do retain the underlying P1 reproducibility limitation and mark R1 PARTIAL rather than CORRECTED. The large local gains can describe the retained populations but cannot establish independently reconstructible public-benchmark gains. This is not evidence of a newly discovered metric failure or data leakage, and is not a request to change masks/splits, select a cleaner subset, or rerun experiments. Full construction recovery has not been demonstrated; this review does not claim it is impossible in principle.

## R2: Executed Definitions Corrected

**Locations:** supplement D.2/PDFpp.10-11 and D.3/PDFpp.12-13; generated/legacy_definition_R2.tex:4-105; generated/factorial_definition_R2.tex:4-63; appendix.tex:124-160. Source bindings: generated/legacy_factorial_definition_R2_provenance.json.

I checked the new operational definitions against 21 specifically cited frozen runtime source files and the five selected archived Python components. Their actual byte hashes and quoted excerpts agree with the enclosed pins; all 45 claim-source routes identify valid source ranges. The archive was read only as a bounded 48,504-byte source container, with five component hashes checked, not imported or executed. The old engine, optimizer and OOM iterator were not run.

The supplied equations and prose agree with the checked implementation on the following points:

- Normalization constants, grayscale coefficients, 9/1/4 average pool with zero-padding/divisor 81, contrast 0.35, cyclic channel permutation and clipping; original-mask target/residual/context view definitions are stated in real arithmetic.
- Original query and original token locations are retained. Things schedule T/R/C, stuff/background T/C. Clipped supports enter q=min(T,C), thing-only d=max(R-T,0), and clipped q(1-d). Initial confidence is not multiplied into that reliability formula. Production C remains one response without the role/residual rule.
- Roles are inherited through evaluation-category mapping, not guessed from query spelling. The exact 30 PC60 thing names match the archived role set; Object background handling, vocabulary flattening and extension-to-category mapping match the frozen sources.
- The prompt/token mean entropy is evaluated on unmodified inputs. Both image-normalized and episode-normalized objectives are explicit, with EPS/empty omission, detached once-at-reset scores and skipped empty updates. The episode_min alias is not incorrectly defined as a minimum diagnostic.
- M04 assignment is within each source, with ascending values and ascending (ordering score, lexical candidate ID) recipients. The crop/query/component ID format and initial confidence/area/query/component top-K ordering are separately defined.
- Common-active restriction uses the raw paired scores before reassignment. Natural-active EPS omission follows assignment. The changed conditioned comparator and the fact that every factorial arm pays for C measurement remain explicit.

The shared numerical qualification explicitly separates ideal algebra from native dtype conversion, retained-order accumulation and the denominator guard. It does not substitute an ideal float64 coefficient calculation for execution or promise equal gradients/time. The former over-simple discussion phrase about halving views was also replaced by the accurate one-versus-two/three schedule description (main.tex:196).

Thus the original missing-definition R2 finding is CORRECTED. This is a manuscript/source consistency conclusion, not a reopening of previously closed CODE audits or an independent new model/gate run. The Object negative / PC60 positive legacy contrast, PC60 negative image-minus-episode contrasts, crossing-zero objective interactions and conditional factorial effects remain in unchanged table bytes and revised prose. Their intrinsic interpretation limits are not removed.

## R3: Semantic And Measurement Keys Corrected

**Locations:** supplement H/PDFp.23; Figs.1-4/PDFpp.24-27; Tables20-21/PDFpp.28-29; appendix.tex:265-309; generated/M12_rendering_legends.json; generated/palette_coco_object.tex and generated/palette_pc60.tex. Projection: tools/project_case_legends.py:12-75.

The previously absent semantic legend is now actually enclosed, not merely asserted to exist upstream. I used stable, no-follow, bounded reads of only the four specifically pinned existing case JSON metadata files. Each exact rendering field equals its corresponding enclosed rendering legend. No raw RGB, GT, prediction array or fresh case generation was used. These JSON reads are distinct from viewing the four already frozen derived PNG paper assets.

The independent projection check verified all 141 evaluation IDs/names/RGB triples: 81 Object and 60 PC60, against both case legends per dataset and the cited ordered vocabulary source with canonical background naming. The printed TeX rows contain each ID exactly once, the intended names and exact integer RGB text, and swatch color decimals round back to those same 8-bit values. Ignore 255 is gray (128,128,128); semantic palette colors and green/magenta geometric overlay roles are explicitly distinguished. Category IDs are not confused with extended query indices.

The actual PDF, not just source markup, contains four readable vector keys. Poppler XML at zoom 1 reports 9pt for each query/confidence/C key and all eight panel-position labels. Their rounded measurements match the authenticated metadata for tv, grass, sofa and flower; all panel orders match. I viewed all four complete case pages and both palette pages at full-page rendered resolution. The caption/label/key regions are not clipped or overlapping.

All four composite PNG hashes are byte-identical to draft_v001. No case was replaced or selected by outcome. In particular, the almost-background Confidence example remains, and the captions retain the distinction between preservation overlays and displayed actual perturbation images/attention. The small original raster annotation is still small, but no longer bears the sole burden of communicating the panel identity or score. These corrections address the original R3 request; they do not establish case representativeness, calibrated reliability or a per-candidate causal effect.

## R4: Navigation Corrected, Whitespace Retained

**Locations:** supplement.tex:10-15 and appendix.tex section clear-page boundaries; actual supplement physical pages6-13 and23-29.

The standalone supplement is full width. M01 inside/outside/whole tables are contiguous on pp.6-8; cases no longer interrupt that sequence. Spatial text/table are together on p.9. M03 definitions and its table occupy pp.10-11, and M04 definitions/arms/contrasts pp.12-13. Four cases are together on pp.24-27, immediately followed by the complete palettes pp.28-29. The original 9-10-page definition/table separation is gone. Some top-positioned floats precede their local section heading, but the evidence and definition are now on the same or adjacent pages rather than distant sections.

There is still substantial whitespace on some float pages (including M01 and model-time pages), and the supplement is now 31 pages. This is a residual editorial cost, not dropped evidence or a clipping finding. All 38 pre-existing generated/figure members are byte-identical, including every old numerical table, plot and case PNG. No null/negative row was removed or shrunk out of the underlying table source to obtain this reflow. R4 is CORRECTED for the requested navigation change; it is not a claim of minimal length or perfect page utilization. The main is truthfully an 8-page PDF with reference continuation, not the previous 7-page artifact.

## Independent Work Performed

1. Verified the exact archive size/hash, bounded decompression, all 70 unique regular canonical relative entries, absence of absolute/traversal/link/nonregular members, exact manifest membership, every file size/hash and the 15,108,468-byte aggregate BEFORE extraction. Extracted with exclusive files into a new private 0700 scratch directory; no extractall or execution of an embedded collection script.
2. Compared the manifests and actual authenticated files against the already authenticated draft_v001 extraction: all 38 original generated/figure members are unchanged; scientific evidence SHA remains eda4079f5c01b41b1b8e3f7538449ba2ddf06272af453956778c8e3df33fb845. The old raw report/archive hashes were separately verified unchanged.
3. Read the main/appendix revision diffs, new definition fragments, provenance and legend projection tool/tests. Authenticated 29 non-review R1 cited files, 21 R2 runtime files, five selected archived source components and four small case JSON metadata files with bounded stable-FD/hash-before-parse checks. The R1 provenance's 30th entry names the original raw review; its exact pin was verified separately and only this reviewer's own original report was consulted as the R1-R4 baseline. No other reviews/grades or historical manuscripts were consulted.
4. Ran the exact extracted tools/test_case_legends.py and tools/test_export_scope_v4.py with hidden CUDA, Python -B, PYTHONDONTWRITEBYTECODE=1, OMP/MKL/OpenBLAS threads1, plugin autoload disabled and pytest cache disabled: **58 passed in 0.69s**. These are projection/synthetic exporter tests, not scientific estimates or a renewed broad CODE audit. The real-data projection/export main functions were not invoked.
5. Ran independent scratch checks, not only author tests: all 141 palette entries/swatches/vocabulary mappings, four actual 9pt vector keys and rounded measurements, 13 literal corruption parameter families plus direct zoom/elastic source inspection, exact 30-name PC60 thing taxonomy, valid source routes/quoted excerpts, and PDF text/page bounds. No estimator or bootstrap was invoked.
6. Rendered all 8 main and31 supplement pages at100DPI and viewed the complete39-page contact-sheet flow. Additionally viewed full-page main pp.1/3 and supplement pp.3/4/6/8/10/11/12/13/24-29 (16 full-page views), focusing on revised disclosures, definitions, dense tables, keys and palettes. This is not falsely described as a second individual full-size39-page reading. Poppler word-box checks found 0 out-of-page words: main5,098; supplement12,164; total17,262. No unresolved ?? was present in extracted PDF text.
7. Rehashed all69 extracted manifest files and the original v002 archive unchanged at the end. Rechecked the preserved v001 archive and raw review hashes. Only this new reviewer report is a persistent worktree output; scratch scripts, rendered pages and text are under /tmp/scope-v4-review-v002-sl318m65 and /tmp.

All validation assertions and the58-test execution above passed. No actual failed revision check or new experiment failure was observed. Capability probes found fitz/pdfplumber/pypdf absent from the selected mlmp interpreter; no package installation was attempted. Existing Poppler rendering, XML font data and HTMLParser bounding-box inspection provided the stated PDF checks. Missing optional Python libraries are a reviewer-environment limitation, not a manuscript defect. Author exploratory failures listed inside provenance are not reclassified as independent findings or repeated here as if independently reproduced.

## Exact Frozen Identities

```text
b72931845b4ae3e843fbd0730eeaa4b729be4c2cbf574ef1170139481e230bbc  manuscript_draft_v002.tar.gz
3873331 archive bytes; 69 manifested files /15108468bytes; 70 total regular archive entries
724f41ceea316ef13efb1fbbfcbb77742da03c462728655cc2c8f41e1cf5bef2  FREEZE_draft_v002.json
bd4f3bfdde7a3a4d7b70318e168510c885901062171fe90f0ff999444828b312  build/main.pdf (8pages)
d02cfeef2c27e0698f571288e742d8662f05d8b61382ba849dea406d1973c328  build/supplement.pdf (31pages)
eaab6acded76434caf64f42fdf81241bd36cae38186f5a8fe01aeaae2b964a78  main.tex
378e561ca3c84ca992dd41178d0389d0173eb9ccae07c92d462248609b2e4351  appendix.tex
1d56db1e9d7b46aa055306776e84e9ef647f7d8206747d4335d30ed9afddf327  supplement.tex
45db35bbc35ad52eea1811ea8c7d7ad375f127b7fac9539df504704a574f947e  generated/protocol_specification_R1.tex
181b5d11f6b55efc63824898ab8ee208f4d4a8f39115933ecfc4ba105ea94d37  generated/protocol_specification_R1_provenance.json
ed96105de5a523f280d1e8fdc24204fc584a555d0f62d21f8a01fafdfd2ae012  generated/legacy_definition_R2.tex
9f4c1a26e9340cd63d447fbef0035597bf88946a9effc366f2b967df2f88029a  generated/factorial_definition_R2.tex
c6d60699e13ddc00a957cdd77faba43a2df54bd3127ed29005f0ff4261e71ffc  generated/legacy_factorial_definition_R2_provenance.json
ac138c1f99f9e69b4481026acaf5d8b166accc5ffb0d0f51a2b7c4d4f9748216  generated/M12_rendering_legends.json
b19780128ce0a21ed2340b372f919a37634d480b1fb7b3e10ef6eb564cc666a8  generated/palette_coco_object.tex
c918426494695dbeb9c088cea88af273ea1c6080dd0e6130c66811f1d1c1e3bc  generated/palette_pc60.tex
6224c0184ca3d39965bb48a6721f8c489a76d4f1c52d9a1ebeaa0b15189d35e1  tools/project_case_legends.py
0d98747da1a4e21b0616afe957f586341d4baf6c94f76d1b20e4b6473ef99f63  tools/test_case_legends.py
eda4079f5c01b41b1b8e3f7538449ba2ddf06272af453956778c8e3df33fb845  generated/evidence.json (unchanged)
fedcc2bf7c15a630c753a3ab26b8e7a150b8deefc9abfbb46d1485ce8d08f6ed  preserved manuscript_draft_v001.tar.gz
514073234adc093e806f34d3f0e9a0c35ada6e999ec16de717f907a7fb641d0d  preserved manuscript_review_draft_v001.md
```

The complete source-member identities, including the 21 runtime and five archived component SHA values, are already enclosed in the authenticated c6d60699... provenance member and were checked as stated above. Paths inside that provenance are source identity routes, not authorization to access adjacent result payloads.

## Retained Scientific And Review Limits

The original L1-L4 remain. This revision does not certify priority/novelty from source pinning; independent candidate-risk information or candidate causality; independently tuned/unseen-test advantage; seed variability; whole-SCLIP/whole-legacy equivalence; universal corruption robustness; multiplicity-adjusted discoveries; or equal-time benefit. M01 degeneracy/undefined draws, PC60 spatial uncertainty, opposite legacy results and negative PC60 image-versus-episode contrasts, negative SCLIP/severe-corruption results, and unmatched fixed-grid compute comparisons remain part of the record. None should be converted to a positive result by wording, selection, changed bins, dropped draws or new operating points.

This followup did not repeat the prior356-row numeric audit, regenerate any plot/figure/case, recompute a metric/count stream/bootstrap, rerun a model/gate, or inspect real RGB/GT/NPY/model/checkpoint content. It queried no service/process/GPU/lock and used no other agent or external network. Upstream Waterbirds classification citations/URLs were assessed as enclosed attribution/context; external generator history and local construction completeness were not independently recovered. Frost resources were authenticated as declared metadata pins, not by opening image bytes. The new legend tool was tested for this bounded projection task, not certified as a general exporter/security interface.

**Final narrow conclusion: R1 PARTIAL with construction reproducibility unresolved; R2 CORRECTED; R3 CORRECTED; R4 CORRECTED with remaining whitespace. No new material regression confirmed within R1-R4. This verification is completed, preserves the original critical report, gives no numerical grade or universal approval, and grants/renews no scientific execution, artifact, GPU, RGB or GT permission.**
