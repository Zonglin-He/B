# TRACE Revision Training Pilot

Training-only, fixed32 sources. Not a full-benchmark superiority result.

| Condition | NoAdapt | MLMP | Region C | +Global | +Area | +Budget | +Refresh/all |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 59.1928 | 63.9395 | 62.2144 | 64.3486 | 63.7954 | 61.6535 | 62.3356 |

Same engine/readout/base LR/step count. Diagnostics and gates change actual compute and update sizes.
Region C is a matched-engine control, NOT a rerun of the historical TRACE system.
All positive/negative/null effects and class coverage are in RESULT.json. No best-test-trial selection.
