# Selected-C Full-Suite Figure Audit

## Scope and Deliverables

Worktree: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.
Authoring is CPU-only and restricted to `tools/draw_image_full_figures.py`,
`tests/test_image_full_figures.py`, `paper_evolution/figure/image_full_*`, and
this audit. No manuscript/source-result edits, model construction, checkpoint
loading, model forwards, GPU jobs, result mutation, or Git commit were made.
Concurrent manuscript and exporter changes belong to other authors.

| Main asset | Native width x height | Intended placement |
| --- | --- | --- |
| `image_full_teaser.pdf` | 3.375 x 4.65 in | Column width |
| `image_full_pipeline.pdf` | 6.875 x 4.5 in | Text width |
| `image_full_segmentation.pdf` | 6.875 x 7.25 in | Text width |
| `image_full_allocation.pdf` | 3.375 x 3.35 in | Column width |

Every PDF has actual-PDF renders at 150 dpi (`.png`) and 72 dpi
(`_print.png`). Input, intervention, exact float32 intervention, mask, and
selected prediction/GT arrays are companion `image_full_*` assets.
`image_full_provenance.json` is authoritative for exact identities, scalars,
selection rules, read bounds, input hashes, final asset hashes, and layout.

## Admission and Immutability

The only experimental source is
`/home/wwww/hzl/results/trace_image_full_20261029_0333`.
Its required prepared hash is
`e0f7374042eadb65fbc8146de6dfcf35a71151dc1c95c115b0d4cf9c8444edaa`.
All six complete prediction seals are verified against the global prediction
barrier before any GT diagnostic reads. Six score seals bind the results and
`score_inputs.json`; every opened GT file must match its sealed scoring hash.
Every opened prediction and episode receipt must match its prediction-seal
hash. Frozen input RGB hashes and relevant frozen implementation hashes are
checked. The 18 full-split mIoUs are independently recomputed from the small
canonical pooled intersection/union arrays, not per-image count dumps.

The suite seal's `service_success_verified` is false by construction in the
queue source. Independent read-only `systemctl --user show` verifies the exact
suite/hash command, `active/exited`, `Result=success`, `ExecMainCode=1`,
`ExecMainStatus=0`, `RemainAfterExit=yes`, and `Restart=no`. The flag was not
rewritten. The properties and command are preserved in figure provenance.

## Method Identity and Geometry

The publication name is TRACE; the selected implementation is
`ImageBalancedTRACE(view_mode="C")`. The main score is only `rho_R`, the
mean original-query probability on the original candidate token locations
under its one context probe. It is detached. Alpha normalizes active region
weights within each source image; the loss averages active source images.
The vector pipeline uses the main notation `R=(b_R, Omega_R, c_R)`, `rho_R`,
`alpha_(R|j)`, `Hbar_R`, and `I_+`. Source image `j` and input crop `b` are
distinct metadata even though this suite has one 224x224 crop per image.

There is no second diagnostic view, minimum, residual, role/drop table,
dominance, or CLS objective. Confidence is the matched initial-confidence
allocation control, not MLMP. No old two-view examples, scores, or risk values
were imported. An initial full-map prediction is unavailable and was not
fabricated; initial confidence exists only as recorded candidate scalars.

Actual new masks are native **16x16**, losslessly encoded as
`packbits-little-hex`, not 14x14. Decode checks shape, byte length, padding,
boolean dtype, and exact area. A ViT token spans 14x14 input pixels here;
that patch size must not be confused with the 16x16 spatial token grid.

### Exact Input Intervention

RGB is decoded from the frozen actual input path and resized by PIL bilinear
to 224x224, exactly as in the admitted data loader. All tensor arithmetic is
CPU float32. Nominal CLIP constants are
`mu=(122.7709,116.7460,104.0937)` and
`sigma=(68.5005,66.6322,70.3232)` in 0..255 RGB units; provenance stores their
exact float32 values. With `z=(x-mu)/sigma`, the existing intervention helper
first recovers `xhat=z*sigma+mu`, then computes:

```text
m = spatial_mean(xhat), independently per RGB channel
b = clip(cycle_gbr(0.35 * (xhat - m) + m), 0, 255)
z_context = z * M + ((b - mu) / sigma) * (1 - M)
```

`M` is the recorded token mask upsampled with Torch nearest interpolation.
The actual `TraceInterventionMixin.make_trace_counterfactual_background`
helper is called on CPU; no new forward or support measurement occurs.
Candidate values are exactly unchanged in normalized input space. The maximum
RGB denormalization roundoff is 0.00000762939453125 in both displayed cases;
rounded companion PNGs retain candidate RGB bytes exactly. Float32 NPYs retain
the actual reconstruction, without display rounding. Cyan contours identify
the preserved footprint, not new masks inferred from the perturbed view.

## Exact Diagnostic Selection

Only frozen-order indices 0..399 are eligible for case diagnostics. CD,
Object, PC60, and VOC21 each open exactly 400 GTs. No VOC20 or Stuff GT or
per-image prediction is opened. Teaser receipt diagnostics cover only PC60
and Object. A boundary episode JSON can contain later-image records; those
records are discarded before selection, and no later image/GT is opened.
No `per_image_json`, giant count file, or model checkpoint is loaded.

Teaser: use non-background candidates with confidence >=0.9 and native area
16..192. Choose maximum measured C in PC60 and minimum measured C in Object;
ties use lowest image index, then receipt candidate index. GT is not used.
Eligible counts are 257 and 139 respectively. Internal IDs are in provenance,
not publication artwork.

| Teaser | Frozen image index | Native area | Initial confidence | Measured probe support |
| --- | ---: | ---: | ---: | ---: |
| PC60 / sofa | 151 | 32 | 0.9841969013214111 | 0.9975855946540833 |
| COCO Object / cake | 241 | 40 | 0.9835453033447266 | 0.0883842259645462 |

Segmentation: for each image, select its largest non-background GT class with
at least 128 valid pixels; ties choose the lowest category ID. Measure that
one class's IoU for each saved arm. Select maximum TRACE-minus-Confidence in
CD, Object, PC60 and minimum in VOC21; ties choose the lowest image index.
Eligible image counts are 400, 395, 400, 397. Three strictly positive cases
and one strictly negative case are required; no fallback changes this rule.

| Case | Image index / category ID | Confidence class IoU | TRACE class IoU | Difference, pp |
| --- | --- | ---: | ---: | ---: |
| CD / cat | 394 / 1 | 0.000000 | 56.160506 | +56.160506 |
| Object / dining table | 40 / 61 | 1.188463 | 86.801947 | +85.613484 |
| PC60 / bus | 199 / 12 | 44.717968 | 78.060917 | +33.342949 |
| VOC21 / table, negative | 85 / 11 | 84.344628 | 62.868852 | -21.475776 |

These are **post-seal, GT-selected extrema, not representative examples**.
They are not used to select probe mode, hyperparameters, or overall metrics.
The Object dining-table annotation includes much of the displayed meal/plate;
the actual GT is retained rather than relabeled to fit an object intuition.

GT uses 224x224 nearest-neighbor resizing, ignores 255, and reduces label zero
only where the frozen protocol requires it. IoU uses valid pixels for both
intersection and union. Displayed maps are named-class overlays; other
categories are dimmed, and GT void is gray. The saved category maps already
use the maximum query probability per category before category argmax, not
the sum of extension probabilities. Candidate queries map to categories only
for identity/display, never to remeasure their original-query support.

## Full-Split Allocation Values

These use all canonical full-split scores, not the diagnostic 400-image panel.
Both columns are TRACE minus the named matched allocation control, mIoU pp.

| Dataset | Images | Minus Confidence | Minus Uniform |
| --- | ---: | ---: | ---: |
| COCO-CD | 500 | +7.317811 | +8.765389 |
| COCO Object | 5000 | +1.769488 | +2.787515 |
| PC60 | 5105 | +0.355929 | +0.309380 |
| VOC21 | 1449 | +0.296016 | +0.179038 |
| VOC20 | 1449 | +1.309732 | +1.703554 |
| COCO-Stuff | 5000 | +0.042865 | +0.091724 |

Only descriptive seed-0 point differences are claimed. No significance,
bootstrap, candidate independence, calibrated-risk, or causal-benefit claim
is made. The tiny Stuff effect remains explicit. Historical development
exposure and inherited per-dataset configurations are not erased by sealing.

## Publication QA and Integration

All four actual PDFs were rendered and inspected at 150 dpi and native
72 dpi print dimensions. Text bounding boxes are checked pairwise for overlap
and against the page before export. The pipeline's third module now leaves
right-hand padding, and adjacent summation limits are separated.

Ordinary text is >=9pt at native placement. Cairo embeds simple TrueType
subsets; `pdffonts` confirms no Type3 or CID fonts in any PDF. Export is
restricted to PDF 1.5 to match the main TeX inclusion limit. Genuine
MathText Greek/formula glyphs are preserved as vector outlines, not raster
images, to avoid Cairo's CID wrappers for Greek. Formula source and individual
glyph sizes are recorded: smallest math glyph is 9.31pt, including nested
subscripts. No LaTeX trimming or external crop is used. PDF inspection finds
minimum stroked widths 0.6pt (pipeline 0.7pt), and no out-of-page text.
Pipeline and allocation each contain zero raster images.

### Final Height Repair

The segmentation asset was reduced from 495x648pt to **495x522pt** without
dropping or changing any case. All 16 image panels are exactly 1.4x1.4in;
images are not stretched or cropped. Each row's Confidence -> TRACE class
IoUs are combined into its heading, eliminating duplicate under-image labels
and excess vertical spacing. Selection disclosure remains in the manuscript
caption/appendix and this provenance; the compact footer explains direction,
cyan class overlays, and gray GT void.

At the integrated main text width **505.89pt**, height is
`522 / 495 * 505.89 = 533.484pt`, below the requested 540pt graphic bound.
Native 9pt annotations become approximately 9.198pt at that width. Compared
with the reported previous 662.27pt graphic, this removes 128.786pt, exceeding
the reported 65.84pt page overflow by about 62.95pt. A regression test enforces
the <540pt bound and all 16 undistorted square panels.

An isolated draft compilation of the current main document was then run with
job name `image_full_integration_check` and output directory `figure`, keeping
all generated files inside the owned `image_full_*` namespace. It exits 0;
the final log reports an actual requested segmentation size of
**505.89pt x 533.49417pt**, with no `Float too large`, `Overfull`, or PDF-version
inclusion warning. The approximately 0.01pt difference from the geometric
calculation is TeX unit/rounding behavior. No TeX source or primary build output
was changed. This is a geometry integration check, not a bibliography/publication
certification: the independent job name has no bibliography build, so citation
warnings and preexisting underfull paragraphs remain. Its log is
`paper_evolution/figure/image_full_integration_check.log`.

## Verification and Reproduction

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B tools/draw_image_full_figures.py
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B -m pytest -q tests/test_image_full_figures.py \
  tests/test_image_full_worker.py tests/test_image_balanced_trace.py
```

The three test modules pass **128 tests**. Focused coverage includes native
mask encodings/counts/padding, exact intervention formula, unchanged candidate
PNG bytes, source-normalized receipt coefficients, query/category max mapping,
GT conventions, named-class IoU, global-seal/read bounds, actual PDF numeric
annotations, minimum scales, math glyph sizes, embedded font types, aspect
ratio/page-height regression, selected-case signs, and all output hashes.

Claim/evidence self-review: the teaser is a measured response illustration,
not a novelty or causal proof; pipeline notation matches the actual selected
objective; full effectiveness comes only from six complete result cells;
the genuine negative and small Stuff result remain visible; shared gradients
and image-budget normalization do not imply equal gradient norms or calibrated
individual update benefit. No new experiment is implied by figure rendering.

One manuscript wording mismatch remains outside this author's ownership:
the inspected allocation caption says differences against Confidence, whereas
the clearly labeled figure additionally gives differences against Uniform.
The caption should name both controls. This is not a data or figure-height
blocker, and no manuscript text was changed by the figure author.

## Final Hash Handoff

The final figure-provenance SHA-256, after the compact layout and PDF 1.5
export, is `6d335d1729c1f9f6b14d261ae0d19d1c8bca2dac491c7066dc78b35a50fdd434`.
The final segmentation PDF SHA-256 is
`31a8b294fd78afa6935f1eca3dca810b16a91d3474be2a7d3916bee008943742`.
The provenance's `output_sha256` binds all four final PDFs and their renders
and companion arrays; its `script_sha256` matches the current authoring script.
