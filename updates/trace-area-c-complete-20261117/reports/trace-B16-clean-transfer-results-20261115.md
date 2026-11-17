# Fixed Clean-Parameter B16 Transfer

All nine B16 arms completed full predictions and separate CPU scores, three
shuffled repeats per arm. Original L14 clean-selected22 fields were copied
unchanged to the matching B16 dataset, as explicitly requested. MLMP remains
fixed to its paper hyperparameters with the declared9-of12-layer architecture
transfer and strict deterministic runtime. This is not an unchanged-L14-shell
claim. No B16 test score was used to retune the selected fields.

| Dataset | Images | NoAdapt | MLMP | TRACE area-C | TRACE minus MLMP pp |
|---|---:|---:|---:|---:|---:|
| COCO Object | 5000 | 29.59 | 30.5267 | 29.7900 | -0.7367 |
| PC60 | 5104 | 28.90 | 29.6133 | 29.2000 | -0.4133 |
| VOC20 | 1449 | 77.59 | 84.1233 | 84.3800 | +0.2567 |

These negative/mixed transfer outcomes are retained, not replaced with new
test-selected settings. Results do not support universal improvement across
backbones. They do not identify a causal explanation for the transfer gap.

Object NoAdapt/MLMP were verified in progress329. The seven newly completed arms
were independently verified in progress330: true same-current-invocation retained
prediction AND score success, all output hashes,21 exact trial points recomputed
from compact I/U, no raw GT reopened. Tiny1e-15 std values are floating arithmetic
on equal rounded trial scores, not meaningful nonzero uncertainty. No paired
confidence interval is inferred from the three-repeat std values.

The inherited PC calibration exposure, Object paper-value discrepancy and mixed
historical runtime origins remain disclosed. All11 clean conditions (8L14+3B16)
are complete, but the15S5/full26-condition comparison is still running. Corruption
uses the matching frozen VOC20 clean configuration, not a corruption-specific
search or interpolation.
