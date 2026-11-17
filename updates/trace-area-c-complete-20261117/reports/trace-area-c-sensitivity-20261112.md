# TRACE Area-C: All-Parameter Sensitivity Before Refinement

## Latest Human Authorization

The actual November12 human rejects LR/steps-only tuning and requests all TRACE
parameters screened for sensitivity first. MLMP remains fixed at the official
L14 .001/10steps/B2/alpha_cls1/seven prompts; B16 uses the already disclosed last9
transfer convention. No MLMP parameter is selected. The previous finite
two-parameter dispatcher was stopped after identity verification; its completed
Waterbirds/Object/PC60 families and VOC20 search are retained, not relabeled as
a completed full queue. Original automation remains PAUSED, monitors OFF.

## Parameter Inventory

The complete22-factor operational search schema is in
`tools/evolution_revision/area_c_sensitivity_design.py`. It includes previously
implicit defaults rather than merely exposing learning rate and steps.

| Factor | Coarse levels |
|---|---|
| Learning rate | .0001,.0003,.001,.003,.01 |
| Adaptation steps | 1,3,5,10,20 |
| Adaptation batch size | 1,2,4 |
| Regional C bonus coefficient | 0,.25,.5,1,2,4 |
| CLS entropy coefficient | 0,.25,.5,1,2 |
| Candidate K | 2,4,8,16,32 |
| Minimum region area (model-grid cells) | 1,2,4,8 |
| Candidate confidence floor | 0,.25,.5,.75 |
| Connected-component neighborhood | 4,8 |
| Counterfactual contrast | 0,.2,.35,.6,1 |
| Counterfactual channel swap | false,true |
| Candidate probability temperature | .5,1,2 |
| Counterfactual C temperature | .5,1,2 |
| C exponent | .25,.5,1,2,4 |
| Pixel-entropy temperature | .5,1,2 |
| CLS-entropy temperature | .5,1,2 |
| Adaptation feature-layer fraction | .25,.5,.75,1 |
| Adaptation prompt-prefix count | 1,3,7 |
| Adam beta1 | 0,.5,.9 |
| Adam beta2 | .9,.99,.999 |
| Adam epsilon | 1e-8,1e-6,1e-4 |
| Adam weight decay | 0,.0001,.01 |

The global pixel coefficient remains1 as the reference for relative loss
coefficients. Diagnostic and final readout retain native layers/seven-prompt
mean/adaptive layer weighting; layer/prompt sweeps affect adaptation only.
Prompt subsets are fixed YAML prefixes, never outcome-selected wording. Keep
initial-model C, fixed candidates within an episode, no refresh, no absolute
gate, no semantic-candidate variant. C is heuristic, not a calibrated benefit
probability. Zero regional/CLS coefficients and an identity counterfactual are
reported ablations but ineligible to silently replace the adopted area-C method.

Checkpoint, vocabulary/aliases,224CV2geometry, decoder/ignore/metric, NACLIP
architecture/attention/Gaussian width, visual LayerNorm parameter membership,
Adam family and episodic reset are comparison/method contracts, not accuracy
search parameters. Resource caps, numeric safety EPS and filesystem parameters
are not tuned against accuracy. Diagnostic forward batch1/4/8 gets a first2RGB
numerical/cost check; keep4 regardless of test accuracy. Historical unused
blur/refresh/absolute-gate knobs are not falsely reported as active area-C knobs.

## Sequential Design

Scope remains8L14clean (including corrected Waterbirds-Seg),3B16clean and the
joint original15VOC20S5 family:12families/26conditions. Do not add benchmarks or
corruption types. All15types share one selected configuration.

1. Screen every family before refinement.78TRACE settings: default,60one-factor
   changes,16predeclared scrambled Sobol joint designs and1identity-probe control.
   Use64outcome-blind selected training sources and2paired episode compositions.
   Waterbirds uses16sources from each metadata group, not a species-sorted prefix.
   NoAdapt once and fixed MLMP in both compositions give264576screen predictions,
   excluding implementation checks and any retained failed partials.
2. Report all parameter response curves, ranges/upside, pairwise differences to
   default and MLMP, episode-composition variability, numerical failures and
   measured model time. A range>=.5pp is an exploratory sensitivity flag, not a
   significance claim.16joint configurations test departure from an additive
   one-factor approximation; these are not estimated Sobol variance indices.
3. Per family, take the6largest one-factor ranges, refine neighboring values
   around the best eligible coarse configuration with32Sobol joint proposals,
   retaining default and best3coarse settings. Maximum36TRACE settings,128training
   sources,2compositions. Bounds do not expand after observing results.
4. Confirm best3refinement configurations plus default on the other128training
   sources,3compositions. Fixed MLMP uses all3same compositions. Select mean pooled
   class mIoU over all declared types/compositions; ties fewer steps then config
   hash. Preserve losers, nulls, failures, undefined intervals and all runtime.

The full coarse Cartesian product is1,417,176,000,000 combinations, not executed.
This is all-factor coverage plus bounded joint search, not exhaustive global
optimization. Refinement max249600 and confirmation max53248predictions; total
max567424excluding first4checks. Deduplication can reduce later counts. This
substantial local GPU budget is finite and serialized, not an equal-search or
equal-wallclock comparison with MLMP.

## Sampling And Uncertainty

Reuse sealed training inputs from the completed input stage of
TRACE_area_C_tuning_20261112-104832-v1. No new full data audit/materialization.
Screen64 is a deterministic subset of search128; refine uses search128; confirm
uses the disjoint confirm128. Prior development exposure, especially completed
Waterbirds/Object/PC60 and VOC20 search, is retained explicitly. This is not a
new untouched test or independent confirmatory sample. No heldout metric selects
parameters, and no final validation/test run is in this queue.

Freeze64/128sources into hash-ranked four-source groups. The two/three episode
orders use different B2pairings within each fixed group; B1/B2/B4 never crosses
group boundaries. This permits10000shared whole-four-source-group bootstrap
draws across arms, compositions and all15types. Do not count orders as additional
independent images. Report16screen/32later groups, undefined draws and pointwise
unadjusted intervals. No marginal-CI subtraction, no post-selection significance
or causal interpretation. Range ranking itself is exploratory.

## Execution

New modules only; official/source/historical methods are not modified. Default
configuration must match actual first4RGB historical area-C loss/parameters/
logits<=1e-5 and zero labels. Official native-equivalent checks are also required.
Model/optimizer resets must restore all Adam settings including its saved state,
not just live LR. MLMP always restores defaults. Every trial checks finite
updates. Numerical configuration failures retain partial predictions/logs and
are marked invalid without retry; a default/MLMP failure or infrastructure/OOM
failure stops the queue. No unmodified retry or cap increase.

GPU single shared lock,12GiB RAM/1GiB swap; CPU4GiB/1GiB/one thread/hiddenCUDA;
Restart=no/OOMstop/10GiBdiskreserve. Each process records exact actual invocation,
PID/cwd/argv/env/effective caps. Every CPU score follows complete prediction seals
and true same-invocation GPU success. Preserve source freezes and all failures.
After all screens, a separate CPU summary ranks sensitivities across families;
after all confirmation stages, a final CPU summary records selected settings.
No new monitor/automation, no GitHub upload or old paper/release modifications.

Sobol sampler implementation: SciPy1.15.3 `scipy.stats.qmc.Sobol`, fixed seeds,
`random_base2` with16or32points before discrete mapping/deduplication. Discrete
mapping does not inherit continuous-space balance guarantees. Reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
