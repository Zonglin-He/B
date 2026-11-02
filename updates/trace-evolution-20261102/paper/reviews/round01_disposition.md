# Round 01 Disposition

Author response and writing ledger only. The original `round01.md` and `round01.json` are unchanged. This document does not conduct a new manuscript review, assign a score, or predict acceptance.

## Repairs Against the Original Concerns

| Finding | Disposition | Completed repair | Remaining limit |
| --- | --- | --- | --- |
| W1: matched design controls | Accepted; experiments complete within the stated scope | Final importer verifies A18+B2+C4. Five full splits have fixed-candidate Uniform, Confidence, T-only, C-only, Mean, Min, and three score-permutation arms. PC60/Object additionally have Product and q-weighted token-normalization controls. Results replace pending mechanism slots. | A uniform-token entropy experiment and paired episode intervals were not run. The spatial control is explicitly q-weighted, not relabeled uniform-token. Min is not universally best; no universal-necessity claim remains. |
| W2: support and update outcomes | Accepted; all-five descriptive evidence incorporated, causal/calibration claim excluded | All five full transition and stratified summaries are finished, sealed, read, and integrated through the main risk and appendix strata tables. Fixed confidence>=0.9/q<0.25 footprints show lower C-to-W/higher W-to-C on CD/PC60/VOC21, but higher values of both rates on Object/Stuff. Separate class/confidence/area rank-tail associations are policy-dependent. | No probability-drop artifact exists, no mIoU bootstrap is run, and no candidate-independent significance, isolated causal effect, or universal calibrated risk is claimed. |
| W3: numerical execution and replay | Accepted; defined deterministic gate closed | Final A/C full runs and ten seed1/2 versus seed0 prediction-hash checks pass. VOC20 seed0/1/2 each score 83.942514 on all 1,449 images. Runtime, fixed episode order, and Stuff seed0-only coverage are explicit. | Fixed-order deterministic repeats are replay checks, not independent robustness samples. The manuscript reports no robustness SD or uncertainty inferred from exact replay. |
| W4: self-contained readout | Accepted; source-to-formula text repair complete | Checked adapter, dense readout, and final extension collapse. Diagnostic prompt mean, entropy-derived layer weights, raw-layer aggregation before projection, adaptation mean-layer/separate-prompt readout, frozen scale, checkpoint hash, six vocabulary resources, max extension operator, and frozen runtime identities are explicit. Adapter/backbone source hashes match frozen A copies. | This was formula/source inspection, not an additional GPU test or re-execution of Erdos's deep importer. |
| M1: completed versus pending prose | Accepted; repaired | Main narrative is analytical: problem/setup -> R/L operating points -> matched allocation -> spatial normalization -> diagnostic outcomes -> computation/replay. All mechanism/risk slots use completed results; no development chronology remains in main prose. Experiments use two-decimal prose, with detailed aggregator limits concentrated in one comparison paragraph and one concise limitation. | Genuine unrun VOC20 controls and unavailable Stuff replays remain --; these are unrun comparisons, not unfinished experiments. |
| M2: cost heading and scope | Accepted; repaired | Main retains input accounting. The earlier standalone cost snapshot is appendix-only and clearly distinguished from final deterministic execution. | No measured latency speedup is inferred from two views, cached replay, or shared-arm timings. |

## Outcome-Dependent Narrative Changes

- Use all six final deterministic full-split results: VOC20 83.942514; VOC21 52.379176; PC60 30.565482; Object 32.701128; CD 75.104794; Stuff 21.745495.
- R/L comparisons stay separate: five MLMP reported references in the main table, CD native TENT 72.200299 in its own table. No extra MLMP run or equal-budget claim.
- Uniform/Confidence gaps are substantial on CD/Object, small on PC60, and slightly negative on VOC21/Stuff. C-only/Mean can beat Min on PC60/VOC21/Stuff; CD C-only essentially ties Min. Product/spatial results do not override those contrary outcomes.
- Fixed high-confidence/low-q candidates: CD 474, PC60 30,079, VOC21 8,235, Object 34,560, Stuff 34,727. C-to-W is conditional on pre-correct pixels and W-to-C on pre-wrong pixels. Object C-to-W increases 7.102936% -> 8.031121% and W-to-C increases 20.034302% -> 22.988722%; Stuff C-to-W increases 8.929196% -> 9.702714% and W-to-C increases 6.011756% -> 8.593489%. Conditional rates cannot be subtracted as a net gain or a decomposition of full-split mIoU.
- Separate low/high-q stratification compares rank quartiles within predicted-class/confidence/area strata; it does not reuse the fixed high-confidence/low-q threshold group. Earlier aggregate diagnostics were visible before stratification, so this is descriptive rather than prospectively untouched inference.
- The minimum is conservative parameter-free common support, not a performance theorem. q is frozen diagnosis, not realized gain; normalized rho cancels for a single active candidate in the entire episode.

## Figures and Ownership

The full pipeline figure remains. The Introduction uses `two_view_teaser.pdf` with the same grass/dog measurements; the full `two_view_qualitative.pdf` panel and selection metadata are retained in the appendix. `allocation_controls.pdf` plots mIoU deltas versus matched Confidence with separate dataset scales, not absolute scores. `segmentation_examples.pdf` columns are Input / Confidence / TRACE / GT, never MLMP; annotations are IoU of the explicitly named class or mean over explicitly named classes, not whole-image/dataset mIoU. Selection uses illustrative extremes, not representative sampling, and offline labels used for selection do not enter adaptation.

Only the five authorized manuscript/ledger files are edited by this writing task. Exporter, generated outputs, figures, algorithms, result artifacts, original review, and Desktop originals remain outside its write ownership. No manuscript build is run while coordinated outputs are changing.

## Remaining Integration

1. All-five risk/strata tables, separate CD mechanism controls, and appendix seed-replay table are integrated. No pending result slot remains in this writing pass.
2. Exporters, generated numbers, figures, and provenance remain untouched by this writer. The original review files are unchanged.
3. The coordinating author performs build, rendering, freezing, and any fresh review after handoff. This writing task runs no build and assigns no review score.
