# Initial independent migration audit

Reviewer: Kepler, agent `01a180e1-f9ae-7d42-8b99-1b893b08beb7`.
Date: 2026-10-28. This is a summary of the initial read-only migration audit,
not a score for a revised manuscript, which did not yet exist.

## Findings and dispositions

1. **Method identity (critical).** Original manuscript is role-aware three-view,
   branch production is role-free three-view, and research `quality_only` is
   two-view. Define an explicit executable two-view entry point and rewrite
   all algorithm descriptions against it. Old results cannot transfer by name.
2. **Mechanism attribution.** Native baseline improvements do not isolate the
   formula. Compare fixed candidate/optimizer controls and weight permutations.
   Do not reuse the historical 17.89 pp harmful-update claim for min(T,C).
3. **Fairness and repetition.** TRACE inherits dataset-selected settings,
   whereas native baselines use lr=.001 and ten steps. State this distinction,
   add same-backend/budget controls, and repeat the frozen final setting.
4. **Lazy-probing guarantee.** Omitted normalized coefficient mass is bounded,
   not gradients or mIoU. Staged all-view controls must separate batching
   numerics from skipped diagnostics. Keep this out of the core claim until
   verified prospectively.
5. **Frozen prefix.** Changing trainable layers and exact cache reuse are
   different interventions. No untested combination of lazy and prefix gains.
6. **Paper migration.** Rewrite old abstract, role formulas, pipeline, risk
   tables, removals, appendix configurations and qualitative examples. Old
   suppression examples may only illustrate their original offline experiment,
   not the two-view runtime. Regenerate current-method evidence.

## Initial evidence status

Four complete seed0 two-view splits and sixteen native baseline cells are
sealed historical evidence. They support an operating-point comparison on
those four splits, not a new eight-dataset/three-seed claim. The first review
examined protocol and small result files rather than fully rehashing every
large prediction payload. No acceptance score has been assigned.
