# Full-Population Time-Budget Controls

Both full Object5000 and corrected PC605104 time-budget controls completed predictions and a separate NEW offline CPU score. Main verified all34 original prediction/new score outputs across the two tasks, recomputed all12 exact trial points from sealed integer counts without reopening GT, and verified the new PC repair prepare. Earlier failed score/controller bytes remain separate and unchanged.

Steps were selected only by the complete GT-free training time grid: Object1 and PC6010. MLMP remains0.001/10steps/batch2/seven prompts/native losses/readout. All other TRACE clean fields stay fixed. These efficiency controls do not replace the adopted20step Object or5step PC60 main method.

## Accuracy And Actual Time

| Dataset | MLMP mIoU | TRACE control mIoU | Delta pp | Lower95 | Upper95 | Actual time ratios | <=105% budget | Within5% time |
|---|---:|---:|---:|---:|---:|---|---|---|
| Object5000 | 30.421418 | 30.526836 | +0.105418 | -0.028604 | +0.224800 | 0.907266, 0.891906, 0.892122 | true | false |
| PC605104 | 27.968380 | 27.873757 | -0.094623 | -0.380867 | +0.204728 | 0.965957, 0.958002, 0.960998 | true | true |

Both paired accuracy intervals include zero. This does not establish a matched-budget accuracy improvement over MLMP. Object fits the budget but is about9-11% faster, so it is not a strict near-equal-time comparison; PC60 satisfies the prespecified5% near-time condition. The internal arm name area_time_matched describes the train-time-selected control, not an assertion that a formal test necessarily passed equality.

## Exact Trial Points

| Dataset/arm | Order1 | Order2 | Order3 |
|---|---:|---:|---:|
| Object/MLMP_paper | 30.438529 | 30.415504 | 30.410220 |
| Object/area_time_matched | 30.526836 | 30.526836 | 30.526836 |
| PC60/MLMP_paper | 27.989736 | 27.954037 | 27.961365 |
| PC60/area_time_matched | 27.854724 | 27.775215 | 27.991331 |

## Uncertainty And Scope

10000 shared whole8-source-block draws pair both arms and all three arrangements. Object has625 blocks, PC60638; undefined draws are0 for both. Intervals are pointwise/unadjusted and condition on these fixed runs, not independent initialization seeds. Formal time or accuracy did not reselect steps. Old V01 NULL findings and all historical calibration exposure remain. Different episode designs mean the MLMP means here need not exactly equal the original shuffled main table.

Timing charges synchronized input authentication/decode/transfer, all resets, C probes, updates and final readout/host labels. Model construction is separate. Formal peak CUDA data includes wrapper/reference allocation and allocator history; no intrinsic standalone memory advantage follows. Per-episode timings and peaks remain in sealed prediction EPISODES.json.

## Repair And Provenance

PC60's first score failed on the old5105-list enumeration before GT decode. The new process-local route uses the pre-existing checksum-pinned corrected5104 reference loader and verifies exact source order. The existing successful PC60 prediction was reused; no RGB/model inference, parameter or GT was changed to obtain the new score.

Object prediction:2177168/419d6e3370014640a63ef743ff15bb8f;score:2189895/1772272882554394b800d63588589e0c.
Object score COMPLETED SHA256:5394c2a89c15e4d3cde9fc5be69ffd9ca258fe347c1df65d8701812fca2cb004.
PC60 original prediction:2190105/0f6cc7e7a0d74d0997e09cf8963f066f;newscore:2204962/44122134f14a4383aa9c587346afc340.
PC60 original prediction COMPLETED SHA256:83b8d94ba32bdb805b21f0954226e69115438e900ae02d303f39467eb6bc4efa.
PC60 newscore COMPLETED SHA256:c7faf7d9f684bd50a7fbbad70919e29eb95fdd98a932066cf6ce4622bfcbc0bd.
Mainverification:[revision-results]/R05_Object_main_verified_20261116.json and R05_PC60_main_verified_20261116.json.
Six TENT tasks, full component controls and the new manuscript are separate work; none is declared complete by this report.
