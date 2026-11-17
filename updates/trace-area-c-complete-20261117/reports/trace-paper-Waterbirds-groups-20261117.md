# Declared Waterbirds-Seg Test Group Results

Derived from the existing sealed main-benchmark source I/U and explicit metadata groups by the successful new science-summary CPU stage. No raw GT was reopened. The declared local Waterbirds-Seg test population has2568 images,642 in each group. It is a locally rendered segmentation extension, not the standard Waterbirds classification benchmark or an independently untouched causal test. The original local5322 mixed-split results and their development exposure remain historical.

Group IDs follow g=2*y+place with y=0 landbird, y=1 waterbird and binary background place. Segmentation labels are background0/landbird1/waterbird2. The table pools source I/U inside each group and then computes segmentation mIoU; it is not classification accuracy.

| Method | Group0 mean | Group1 mean | Group2 mean | Group3 mean | Mean trial minimum-group mIoU |
|---|---:|---:|---:|---:|---:|
| NoAdapt_official | 47.010556 | 40.457361 | 43.970176 | 30.565679 | 30.565679 |
| MLMP_official | 51.537478 | 46.588098 | 40.523667 | 34.124272 | 34.124272 |
| area_C | 51.329759 | 47.955840 | 41.577192 | 37.624873 | 37.624873 |

Area_C's worst-group segmentation point is3.500601pp above MLMP and7.059195pp above official NoAdapt. These are descriptive differences only: the crossed main update-episode graph does not permit an honest paired group interval from these runs. No image-iid or marginal-CI substitute is supplied.

This full declared test analysis is separate from the new256-source training diagnostic with group counts134/1/1/120. That diagnostic's3605 undefined worst-group draws and missing interval remain unchanged; balanced full-test groups do not retroactively repair its diagnostic interval.

Science summary actual:2239259/fda18e8a56eb4ae7b81b3a1245dbc3c0, Code1/Status0/success/MainPID0.
Science summary COMPLETED SHA256:1c479a3b101c509b99aa5a9f2de79d75bccd9b1e30c02970fe965e1ddc18de03.
Source:[revision-results]/TRACE_paper_finish_recovery_20261116-2200-v2/summary/ALL_RESULTS.json.
No main method, hyperparameter, group definition, raw mask or split was reselected for this export.
