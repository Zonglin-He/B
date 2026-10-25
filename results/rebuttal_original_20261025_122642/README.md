# Original TRACE rebuttal evidence

This package executes targeted follow-up experiments for the submitted three-view,
role-aware TRACE. Its reference is the source snapshot and released dataset
configurations archived with `reviewer_followup_20261010_123627`. The recent
two-view and cached-prefix algorithms are not substituted for this method.

## Questions and fixed comparisons

| Evidence | Question | Comparison |
| --- | --- | --- |
| Method ablation | Which complete Method modules matter? | Full; w/o Region Construction; w/o Confidence-Source Tracing; w/o Reliability-Weighted Adaptation |
| Weight refresh | Does a fixed initial score become outdated? | Fixed initial weights; refresh once at the adaptation midpoint, retaining the initial candidate masks |
| Context perturbation | Is the score tied to the default context transform? | Default; context noise; context masking; context color transform, with the keep/drop operator unchanged |
| Candidate selection | Does confidence top-K exclude useful candidates? | Confidence top-K; random K; spatially spread K, plus an all-candidate scoring audit |
| Cached segmentation audit | Does the diagnostic predict actual segmentation deterioration? | Fixed historical probability-drop event, actual region accuracy loss, and majority-correct to majority-wrong, kept as distinct outcomes |

Full-split module ablations use VOC21, PC60, COCO Object, COCO-CD and seeds 0/1/2.
The four archived Full seed-0 results are reused after source, configuration,
count and prediction-seal checks. Alternative module rows run anew because
their older source equivalence has not been established. The five main design
signals and all settings are frozen before observing any new result.

Refresh and context-transform experiments use 128 fixed images on PC60 and
COCO-CD. Refresh is tested on clean data and Gaussian-noise severity 3 applied
before resize. Context-transform comparisons use seeds 0/1/2, keeping each
image's diagnostic transform fixed across adaptation seeds. Candidate selection uses 128 fixed images on PC60 and COCO
Object. The all-candidate audit uses 64 images and does not change K in the
adaptation loss. Sampling is label independent and fixed at seed 20271010.

## Execution and provenance

`run_queue.py prepare --root OUTPUT` extracts the archived runtime and records
all source hashes, exact per-cell configurations, references and commands.
The output must be fresh. The controller runs from its frozen runtime and waits
for the existing VOC21 forward-reduction queue's successful completion before
acquiring the legacy experiment locks. GPU cells and adaptation seeds run
serially. A failed cell stops the queue; no silent retry, tuning expansion or
result-dependent removal takes place.

Smokes check all intervention, selection, refresh and module paths before formal
cells. Native Full equivalence is checked against the unmodified archived
algorithm. Predictions are sealed before GT scoring; GT is not used in candidate
selection, intervention construction, weight assignment or parameter updates.

`offline.py --output OUTPUT/offline` analyzes already collected observations on
CPU and records hashes of all inputs. An isolated candidate probability drop is
not called an actual pixel correct-to-wrong transition. Single-candidate entropy
normalization cancels its positive scalar weight, so isolated tests measure risk
association; full-episode policy comparisons measure allocation utility.

## Presentation

The paper may lead with the strongest supported problem, mechanism and scenario.
All prespecified results remain in the evidence package. Weak baselines may be
shown for context, but relevant strong baselines remain in comparison tables.
No event threshold, eligibility rule, comparator or dataset is chosen after
examining the rank of TRACE. Diagnostic subsets are identified by population,
sample count and sampling seed; they are not described as full validation.

Equivalent engineering changes (fixed-text caching, packed indices, batching)
can be used after a direct reference check. Changes to weight formulas, selected
candidates, adapted parameters or optimization trajectories are algorithm
variants and require their own result labels.
