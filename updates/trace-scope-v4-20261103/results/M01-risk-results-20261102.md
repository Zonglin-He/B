# M01 Isolated Candidate Update Risk

## Verified Completion

M01_offline_score_20261102_1942 completed the prospectively frozen analysis and
exited successfully at2026-11-02 19:56:40Beijing. Retained service MainPID0,
ExecMainPID2040310,Code1Status0,Resultsuccess,invocation20122292c96648bb8a3f26450f3b1781.
CPU only, no new adaptation. Global analysis SHA
ab367e2121d3a979aede1b63a6f96e6472f57c0e283505487ebc87ad51839a96.
Main verified runtime/input/protocol/parent/admission/report/STARTED/global/
both dataset seals and all eight named result-file hashes, without reopening
the large support/prediction arrays. Full compact receipt and machine-readable
selected results: M01_offline_score_terminal_001.json.

The exact prospective risk protocol remains SHA
5e028bc62c1da112c9f33f1e52522a47c9b5b15cffa623542f613d8d45234d63.
Object:128sources/113originalepisodes/1024candidates; PC60:128/105/1018.
Candidates come from the original confidence-TopK pool. Each isolated candidate
was updated from theta0 with unit weight, with optimizer/RNG reset. These are
not joint TRACE-C updates or a full-benchmark performance evaluation.

## Primary Inside-Region Finding

Harm means mean true-class support inside the original candidate decreases by
more than0.001. Harm rates are percentages of valid candidates; differences
below are percentage points. C and confidence quintile cuts were frozen before
outcomes. Intervals are10000paired original-episode bootstrap pointwise95%
percentiles, conditional on this historically exposed panel/configuration.

| Dataset / Checkpoint | Low C Q1 Harm | High C Q5 Harm | C Q5 Minus Q1, 95% CI | Confidence Q5 Minus Q1, 95% CI |
| --- | ---: | ---: | --- | --- |
| Object / step1 | 37.561% | 23.902% | -13.659 [-23.994,-3.566] | -2.439 [-12.698,+7.626] |
| Object / final30 | 38.049% | 26.341% | -11.707 [-21.330,-2.168] | +2.927 [-6.352,+11.878] |
| PC60 / step1 | 58.333% | 31.863% | -26.471 [-36.182,-16.553] | -18.137 [-29.972,-6.331] |
| PC60 / final7 | 66.176% | 37.745% | -28.431 [-38.650,-18.214] | -25.490 [-35.249,-15.277] |

These unadjusted contrasts support an association between initial C and lower
isolated-update harm in this selected panel. They do NOT show C has information
independent of initial prediction quality, and two separate confidence/C
intervals do not establish a significant difference between diagnostic strength.

## Fixed Matching And Negative Limitations

Matching controls original confidence width0.05, benchmark class AND original
query, token-area bin, with a second analysis additionally controlling theta0
GT-correctness quartile. The estimator is the equal-weight mean of within-fixed-
stratum high-C minus low-C harm, not a pooled candidate comparison. Initial GT
quality is offline analysis only, never inference or deployment input.

| Dataset / Checkpoint | Score-Controlled Harm Difference | Plus Initial GT Quality |
| --- | ---: | ---: |
| Object / step1 | -9.412pp | -1.708pp |
| Object / final30 | -5.022pp | +1.667pp |
| PC60 / step1 | -2.065pp | +6.020pp |
| PC60 / final7 | -3.426pp | +4.881pp |

Both matching analyses at all four dataset/checkpoint settings have null
intervals:10000of10000 bootstrap
draws are undefined under the frozen rule that every fixed stratum requires
both sides. Do not drop undefined draws, widen/merge bins after outcomes, or
claim precision from these point estimates. This is an actual coverage failure
of the prespecified inferential analysis, not a software failure to retry.

Score-only matching covers494of1024Object candidates in190strata and626of1018
PC60 in192strata. Adding initial GT quality covers298in120 and434in161,
respectively. Exact ties and unmatched coverage remain in the original outputs;
these category counts may overlap and must not be summed as a disjoint partition.

After GT-quality control, the harm point estimates mostly reverse sign; final
mean support-change contrasts also become negative (-0.02005Object/-0.02195PC60).
With null intervals and partial matched coverage, the current result does NOT
validate a stable independent risk signal beyond initial correctness. It also
does not identify C as causally harmful. Preserve this mixed evidence alongside
the unadjusted association. M02/M03/M04 and model/corruption studies remain
separate tests; none can erase or retrospectively change this analysis.

All inside/outside/whole regions, step1/final checkpoints and thresholds0/.001/
.005 are retained in canonical result.json files, not only the favorable rows
above. No final manuscript claim is upgraded from the unadjusted result alone.
