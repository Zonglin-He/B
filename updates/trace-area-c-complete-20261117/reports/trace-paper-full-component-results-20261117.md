# Full-Population Regional Component Results

Both full Object5000 and corrected PC605104 component controls completed3 common-block orders. Main independently verified all34 prediction/score output hashes and all12 exact trial points from integer source I/U, without reopening raw GT. The final PC10000 stored draws and percentile bounds were independently reconciled. These are fixed post-diagnostic full-population controls, not a new untouched/preregistered test or a search for a replacement main variant.

## Complete Paired Results

| Dataset | Images | Full area_C | No region | Delta pp | Lower95 | Upper95 | Groups | Undefined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Object | 5000 | 32.441154 | 32.368551 | +0.072602 | -0.091553 | +0.244233 | 625 | 0 |
| PC60 | 5104 | 28.835501 | 28.892091 | -0.056590 | -0.122765 | +0.010258 | 638 | 0 |

Both95% pointwise paired intervals include zero. Object's C-specific mean gain is small; PC60's mean difference is negative. This does not establish that the regional C term is the source of the full-system advantage over fixed paper MLMP. It also does not prove that C can never help under any operating point. No new main method or parameter is selected from these outcomes.

## Exact Trial Values

| Dataset / arm | Order1 | Order2 | Order3 |
|---|---:|---:|---:|
| Object / area_selected | 32.441154 | 32.441154 | 32.441154 |
| Object / no_region | 32.368551 | 32.368551 | 32.368551 |
| PC60 / area_selected | 28.837549 | 28.793649 | 28.875305 |
| PC60 / no_region | 28.916948 | 28.825199 | 28.934127 |

Every selected field stays fixed except explicit removal of the regional loss. Object uses20steps/B1 and PC605steps/B4. Three arrangements stay within common8-source blocks;10000 identical whole-block draws pair both arms and all arrangements. The intervals condition on these fixed runs, are not independent-initialization uncertainty and are not multiplicity-adjusted.

Object's complete selected-C sourcewise four integer count arrays are identical to the original main C result, not merely equal in scalar mIoU. PC60 uses a new common-block episode design, so its selected-C mean28.835501 need not equal the original globally shuffled main mean28.713333 (official-rounded). It does not replace the main table or retroactively improve its score.

The no-region controls retain much of the selected-system point improvement relative to fixed paper MLMP. They retain selected LR, steps, batch, CLS coefficient/temperature and other operating fields, so the comparison is not an equal-search-budget baseline study or a causal allocation of all gains to one factor.

The diagnostic wrapper computes C views and an extra measurement forward even in the no-region accuracy control. These controls do not measure the standalone speed advantage of eliminating C. The separately completed time-budget study supplies efficiency evidence, with both accuracy intervals crossing zero.

## Exact Evidence

Object prediction:2239423/867754867a7c4ba4a10a77f1f137dcfa;score:2301482/ab642042373149a8b480468964ff958d.
Object score COMPLETED SHA256:24f31e6cabbcb4e201eb33e1c31fdabb3e2bea43ef92948f10b3b2010c1bf2ff.
PC60 prediction:2301687/0ed9e47cd9db4b53849a75c050dd1428;score:2311002/a0bce969d37e477fbf587912bb07f77d.
PC60 score COMPLETED SHA256:7111f93704025c5c1efa6d55b35cda08aebdc0f0cc6ae44718283a26ea7b78b8.
Recovery controller:2204921/5dd116aae835431f8a27479c62a1b9ab actually Code1/Status0/success/MainPID0;all152outputs verified.
Controller COMPLETED SHA256:160b19f44f4603e9b798aeb7fc577960bac72ca4a394b30ffd70d31ff403ac56.
Final COMPLETED SHA256:0e7bdd8113229d3739853b94668cae993d901eb89495a01fbca3097977232af9.
Complete result SHA256:7239c8f9047203ab5c8f76596e2b45692f7a68cdbc3a160fbb61451fb3eb171c.
All96 frozen sources remain unchanged. Old failures, original predictions, main tables, PC exposure, B16 negatives, historical R03 NULLs and all undefined values remain. No test-based retuning or unmodified retry occurred.
