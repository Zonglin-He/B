# V01 Fixed Accuracy-Time Grid: Complete, With No Near-Time Match

## Result And Boundary

The full V01 prediction and separately admitted CPU GT scoring stages actually
completed, and main verified their exact successful retained terminals, provenance,
metadata/stat graphs and final output hashes. This is a complete fixed-grid
readout, not an interrupted experiment or pending admission.

**Neither dataset has an eligible near-time Confidence match to native C under
the predeclared inclusive [0.9,1.1] time-ratio rule. Both near-time accuracy
comparisons are null.** Preserve this coverage limitation. Do not interpolate,
change the grid, relax the threshold or rerun to obtain a match.

In the observed fixed grid, extending Confidence updates did not recover native
C accuracy; its mIoU decreased at the additional fixed budgets. This is a
descriptive finding under these historically exposed configurations, not an
equal-time efficacy, universal allocation, independent-risk or candidate-causal
claim. No algorithm or configuration was selected using these results.

## Full Ten-Point Grid

All model seconds below are original synchronized totals including episode0.
Native class I/U is pooled before class averaging; mIoU and intervals are in
percentage points. These are the complete five points for each dataset.

| Dataset | Method | Steps | Raw Model Seconds | mIoU | Pointwise 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| coco_object | C | 15 | 974.521979 | 32.574919 | [31.748701, 33.165202] |
| coco_object | C | 30 | 1626.462316 | 32.671159 | [31.775615, 33.293350] |
| coco_object | Confidence | 30 | 1343.933863 | 29.361107 | [28.550247, 30.006918] |
| coco_object | Confidence | 60 | 2653.889577 | 25.841277 | [24.941459, 26.624269] |
| coco_object | Confidence | 120 | 5272.454141 | 24.242519 | [23.399270, 24.999832] |
| pc60 | C | 4 | 339.304309 | 30.517284 | [30.002150, 30.950356] |
| pc60 | C | 7 | 447.216557 | 30.782843 | [30.258631, 31.217791] |
| pc60 | Confidence | 7 | 281.003929 | 30.573158 | [30.069028, 30.985610] |
| pc60 | Confidence | 14 | 532.888227 | 30.162636 | [29.652186, 30.598300] |
| pc60 | Confidence | 28 | 1036.378710 | 28.338966 | [27.803782, 28.829106] |

## Fixed Contrasts And Time Selection

Confidence selection uses only total all-episode synchronized model time,
not accuracy. Exact multiplicative distance is monotone-equivalent to absolute
log time ratio; exact ties prefer fewer steps. No interpolation is permitted.
The nearest descriptive contrast remains published even when it is not near-time.

| Dataset | Native C | Same-Step C Minus Confidence | Time-Only Nearest Confidence | Conf/C Time Ratio | C Minus Nearest Confidence | Near-Time Comparison |
| --- | --- | --- | --- | ---: | --- | --- |
| coco_object | steps30 | 3.310052 [2.658825, 3.864745] | steps30 | 0.826293 | 3.310052 [2.658825, 3.864745] | null: outside [0.9,1.1] |
| pc60 | steps7 | 0.209685 [0.061794, 0.349708] | steps14 | 1.191566 | 0.620207 [0.398712, 0.831036] | null: outside [0.9,1.1] |

Object's nearest Confidence point uses about82.63% of native C time; PC60's
nearest uses about119.16%. Their positive descriptive accuracy contrasts must
not be labelled matched-budget improvements. The same-step C/Confidence
accuracy anchors coincide with earlier M02/M03/M04 because the native method,
inputs and settings are shared. These V01 model times were freshly measured;
accuracy agreement is not independent new data or permission to substitute old
time anchors. The V01 bootstrap namespace is distinct, so interval endpoints
need not equal the earlier namespace endpoints.

## Work, Coverage And Resource Accounting

| Dataset | Point | Optimizer Steps | Diagnostic Inputs | Encoder Inputs | Candidates | EPS-Active Candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| coco_object | image_C.steps15 | 5775 | 173503 | 258503 | 173503 | 171161 |
| coco_object | image_C.steps30 | 11550 | 173503 | 333503 | 173503 | 171161 |
| coco_object | image_confidence.steps30 | 11550 | 0 | 160000 | 173503 | 173503 |
| coco_object | image_confidence.steps60 | 23100 | 0 | 310000 | 173503 | 173503 |
| coco_object | image_confidence.steps120 | 46200 | 0 | 610000 | 173503 | 173503 |
| pc60 | image_C.steps4 | 1572 | 101505 | 132135 | 101505 | 101002 |
| pc60 | image_C.steps7 | 2751 | 101505 | 147450 | 101505 | 101002 |
| pc60 | image_confidence.steps7 | 2751 | 0 | 45945 | 101505 | 101505 |
| pc60 | image_confidence.steps14 | 5502 | 0 | 81680 | 101505 | 101505 |
| pc60 | image_confidence.steps28 | 11004 | 0 | 153150 | 101505 | 101505 |

The C diagnostic is charged, not zero-cost. Candidate pools are shared across
all five points, but C and Confidence have different EPS-active support. This
is not a controlled candidate-causal contrast. All5000Object/5105PC60sources
are retained. Exact legal visual-LN membership,98trainable tensors/100352elements,
frozen-parameter hashes, resets, complete actual work and warmup-adjusted
throughput remain in sealed DATASET_RESULTS.json and the parent cost receipts.

Primary timing runs from native episodic reset through candidate construction,
C diagnostic/adaptation and final reconstruction/readout, with the common
finite/Adam observer. Decode, H2D, disk IO, GT, preflight, gate-only auxiliary
snapshots and final cleanup are outside primary model time. Full prediction
has zero auxiliary checkpoint readouts;315paired real gate steps are validation
work, not additional formal curve observations. Warmup-excluded throughput is
separate and was not used to select a point. No fabricated FLOPs or subtraction
of unmeasured excluded components is used.

| Dataset | Combined Dataset Model And IO Wall Seconds | Cumulative Process Peak RSS Bytes | GPU Peak Allocated Bytes |
| --- | ---: | ---: | ---: |
| coco_object | 11981.506489 | 5177577472 | 3931494400 |
| pc60 | 2733.846885 | 5177577472 | 3926007296 |

Dataset walls include model/IO and are not per-arm time-match quantities.
Cumulative RSS is not the cgroup's file-cache-inclusive memory usage. Whole
prediction service duration additionally includes validation gates/preflight/
final binding/cleanup, and the separate112.880224sCPU scoring stage is offline
GT cost. These scopes are not silently mixed into primary adaptation timing.

## Uncertainty And Exposure

All10,000draws share original complete episodes across the five points:
385Object and393PC60episodes. PCG64seed derives from compact ASCII
[20261102,"V01",dataset,"fixed_grid"], SHA256first16bytes little-endian uint128.
There are zero undefined draws for all published point/contrast intervals.
The retained protocol would make any affected interval null rather than drop
undefined draws. Intervals are95%pointwise linear percentiles, without
multiplicity correction, not simultaneous inference.

The fixed settings retain historical WB development overlap and test exposure.
This is not untouched validation, independent calibration, a universal
diagnostic-compute win, or proof that more Confidence updates always harm.
No new NoAdapt claim is inferred from this grid. Native indexed GT uses
PIL NEAREST224, not the G-family palette/OpenCV-nearest metric path.

## Exact Provenance And Completion

Prediction V01_value_20261103_0111:
- Prepared f7632c77f32c288bc5bc414eddb22e9527b4bd0628975d57b0ac1124b3ec3eb4.
- Global cad31c0311e7974b0619506a43033854cba28425a1b9cd0be43fc35de156ebda.
- All10cells/50525evaluations/6256341601artifact bytes.
- Actual retained success12:57:10Beijing, Exec2233193/MainPID0,
  invocation608632d0401a4a08b44b8610b372ba37.
- Main281runtime/native68/54421payloadstat/source/audit/start/gates/
  bothchildren/rawcost-reset checks passed without terminal GT/RGB/NPY rereads.

CPU suite V01_offline_score_20261103_1258:
- Prepared aa10df82aa6d34e97c7a84a6f77261e4622d6e6098ef5abeb4a026dab8595ec0.
- INPUTS 93cc7312b6a6da87fe38d0fb508f1bcf2c8deb3f5a5fbc1baeb576d13a0f1b23.
- Protocol c64432173ba18c4fc9fc33bbebd786210baf531f7abbe05559c8e344ab93b4f4.
- Prospective analysis16f5f2c2deeabc1ddfb3adc73979470ae52181d77270fc675827d32e7fe4f082.
- Independent report63754b79e5728366c48ba6a5d87f419d2f62dd9fbde7fc8a09ada0788597c796;
  admission8c230623ae509c95570418582955d7d9cdd7344b013a4c597d95e5d746f5c1e0.
- Independent162frozenCPUtests30.99s; actual89frozen origins,325runtimefiles/
  5242846bytes/native68/complete parent bindings verified. No actual payload
  read or GT permission borrowed from prediction.
- Main fresh frozen admission passed6.283141s/actual1-1-1threads/CUDAfalse.
- Manual ONCE trace-value-score-20261103-1258.service13:06:50to13:08:51Beijing,
  retainedloadedactive/exited MainPID0 Exec2289252 Code1Status0Resultsuccess,
  SAMEinvocation6faed7224e894aeda289fe6f40635d76.
- Dispatch AND immediate actual argv/exe/cwd/env/invocation/cgroup verification
  in one tool orchestration; effective4GiBRAM/512MiBswap/CPU100/Nice10/idleIO/
  OOMstop/Restartno/RemainAfterExityes/KillModecontrol-group. Logged actual
  cv2/torchintra/inter1/1/1/CUDAfalse; peak1265033216B/swap0.
- Analysis global492e9504b3d7adb2c15400de0dc076bb451416999a30a091f6d5ff7e3aa7b4b4.
- All4newoutput hashes/main325runtime/source/native/parent/stat/audit/start/
  protocol/final analysis binding verified,6718355bytes/112.880224s.
- Receipts V01_prediction_terminal_001.json,
  V01_CPU_score_admission_validation_001.json,
  V01_offline_score_execution_started_001.json,
  V01_offline_score_terminal_001.json.

GT was opened only by the admitted CPU scorer after prediction sealing.
No new GPU inference, algorithm change, time-grid extension, rerun, publication
or paper integration was performed to obtain this readout. G03 is a separate
subsequent prediction stage; its execution is not a completed curve result.
