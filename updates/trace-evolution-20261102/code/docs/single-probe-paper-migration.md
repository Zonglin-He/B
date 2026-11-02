# Single-probe paper migration contract

## Conditional Status

This document plans a possible revision; it does not change the current paper's
method identity or scores. Full-split experiments under `image-full-plan.json`
must finish and verify before any manuscript migration. The last independent
review remains round03, 5/10. Completed train-source cohort evidence alone does
not satisfy the original full-split commitment.

## One Method Identity

If adopted, the main method is the calibration-selected **C** mode of
ImageBalancedTRACE. Its candidate mask preserves the candidate's input pixels;
only pixels outside the candidate receive the fixed contrast/channel-cycle
transform. The mean original-query probability at the original candidate
tokens is the detached region weight. There is no second diagnostic view in
this selected variant, no minimum aggregation, role table, suppressed view,
residual dominance, fragment gate or CLS loss.

For source image j with active candidates R_j, define h_R as its mean
prompt/token entropy. The objective is

```text
L = mean_{active source images j}
      sum_{R in R_j} stopgrad(C_R) h_R
      / sum_{R in R_j} stopgrad(C_R).
```

The per-image denominator separates image budget from how many candidates or
how much total support that image contributes. It does not remove shared
gradient, Adam, batch-composition or prompt/readout effects. Scores and geometry
are fixed within the episode. Low absolute C alone does not suppress the total
normalized objective: common scaling cancels, and a singleton weight cancels
within its source image. The exact EPS/empty-source behavior remains the code's
existing rule, not a new numerical or semantic hyperparameter.

## Narrative

1. Dense confidence is a response on a contextualized representation, not a
   measurement of what happens if the surroundings change.
2. A candidate-preserving context probe supplies a specific additional response
   under a fixed intervention. It is not causal source identification or a
   calibrated estimate of realized update benefit.
3. Assign that support relatively among regions within each image, using
   region means so area does not directly multiply a candidate's loss mass.
4. Average image objectives so the support normalization does not make images
   compete through an episode-global denominator.
5. Evaluate this actual single-probe variant against zero-probe uniform and
   confidence weights under fixed candidates/settings, with actual execution
   counts and clearly scoped model-side timing.

The old two-probe premise cannot simply be reworded as though C measured both
independent target support and invariance. Its old teaser/pipeline and old risk
tables must not illustrate the selected single-probe method under a new name.
New figures must use new full-split predictions, measured C and recorded masks.

## Evidence Placement

- Main effectiveness table: only new algorithm's six verified full-split
  values; MLMP paper values explicitly marked reported-reference operating
  points, with citations and no matched-budget or same-environment inference.
- Main mechanism table: full-split Uniform/Confidence/C results under the new
  source-image objective, all six datasets, preserving negative/null values.
- Main cost: actual zero/one-probe input counts and synchronized time; no
  multiplication of nominal view counts into an assumed wall-time speedup.
- Appendix: original six-arm calibration/confirmation matrix and selection
  rule, three fixed orderings, timing denominators, known exposure inventory,
  and old episode-min comparisons labeled as a distinct algorithm.
- Old full-split controls, risk associations and deterministic replays remain
  historical two-view evidence. They are not silently reused for C.

## Remaining Review Questions

| Question | Evidence required or boundary |
| --- | --- |
| Does C improve complete-split segmentation? | New canonical full C/Uniform/Confidence results; literature values separate. |
| Does it actually remove a diagnostic input? | Per-arm encoder counters; no unused T view built; gates and cost streams. |
| Is image-budget normalization better? | Cohort episode-min/image-min contrast isolates only this change. Mixed findings remain; algebraic budget property is not accuracy superiority. |
| Is C always the best probe? | No. T leads Object in confirmation; global C selected only on calibration, all unselected results retained. |
| Is evidence globally development-independent? | Not certified. July4096 mappings incomplete and full evaluation has earlier exposure; state this in main protocol, not only appendix. |
| Are candidate weights calibrated risks or causal effects? | No. Any new outcome analysis is descriptive and uses GT offline after prediction seals. |
| Is the external-method comparison matched? | Not established by published MLMP values; user excludes new MLMP runs on reported datasets. |

## Review Admission

After verified full completion, archive the previous paper, update one coherent
method identity, redraw actual examples, rerun claim/evidence and reverse-
outline checks, compile/render/fonts/overfull verification, then freeze the new
manuscript. One fresh independent default-model reviewer may evaluate it
without target scores or past scores. Preserve the full report. A score below6
requires a substantive, justified next repair, not another reviewer on the same
evidence. Open validity debts cannot be waived by any score.
