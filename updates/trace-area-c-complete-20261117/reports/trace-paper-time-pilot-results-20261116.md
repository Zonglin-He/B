# Current-Method Time-Only Integer Pilot

Actual GT-free GPU profiling and the separate CPU selector have both succeeded. Main independently verified all10pilot and5selector output hashes, exact source-block coverage, actual steps, every total as a sum of episode timings, and recomputed both timing-only selections. This report does not assert formal full-test time equality or any matched-time accuracy result.

Each dataset has64 separate training calibration sources,3 fixed episode arrangements and all20 integer TRACE step counts. There are63 measurements per dataset:3 fixed MLMP10step/batch2 runs plus60 TRACE points. The selector takes the largest actual integer fitting MLMP time in every arrangement; it reads no accuracy or GT. No interpolation, extrapolation, test-based step selection or main-method replacement occurs.

Timing includes synchronized input authentication/read/decode, CPU-to-GPU transfer, resets, all C probes, updates, final readout and CPU labels. Model construction is recorded separately from warm-online totals. No diagnostic shared-probe amortization, cache dropping or simultaneous GPU work is used. Allocator reservation history and wrapper/reference allocations are retained; these measurements do not establish an intrinsic standalone memory advantage.

## Frozen Selection

| Task | Selected steps | Feasible integers | Selected time ratios across3 arrangements |
|---|---:|---|---|
| L14_coco_object | 1 | 1 | 0.900944, 0.895098, 0.894168 |
| L14_pc60 | 10 | 1,2,3,4,5,6,7,8,9,10 | 0.962352, 0.956607, 0.969510 |

Object selects1step, not the adopted main-method20steps; PC60 selects10, not the adopted main-method5steps. These are distinct efficiency controls and do not replace the main table. All other selected fields stay unchanged. Old V01 NULLs remain unchanged.

## All Actual Integer Ratios

Ratios below divide each TRACE total by the fixed MLMP total in the same training arrangement.

| Steps | Object order1 | Object order2 | Object order3 | PC60 order1 | PC60 order2 | PC60 order3 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.900944 | 0.895098 | 0.894168 | 0.471519 | 0.470267 | 0.475526 |
| 2 | 1.113887 | 1.106039 | 1.112439 | 0.530549 | 0.528203 | 0.531464 |
| 3 | 1.325581 | 1.310704 | 1.308652 | 0.581615 | 0.582954 | 0.590642 |
| 4 | 1.537526 | 1.523839 | 1.527191 | 0.635688 | 0.632636 | 0.642590 |
| 5 | 1.755352 | 1.744330 | 1.738564 | 0.688553 | 0.694135 | 0.691312 |
| 6 | 1.963527 | 1.952641 | 1.942765 | 0.748377 | 0.741514 | 0.753351 |
| 7 | 2.176322 | 2.143671 | 2.154738 | 0.802893 | 0.797620 | 0.811915 |
| 8 | 2.392365 | 2.364867 | 2.351272 | 0.856796 | 0.848055 | 0.861242 |
| 9 | 2.578600 | 2.571740 | 2.566563 | 0.911182 | 0.899316 | 0.922992 |
| 10 | 2.785246 | 2.775640 | 2.752888 | 0.962352 | 0.956607 | 0.969510 |
| 11 | 2.995878 | 3.003306 | 3.010293 | 1.020292 | 1.008164 | 1.019469 |
| 12 | 3.195814 | 3.232058 | 3.207200 | 1.077392 | 1.067703 | 1.076850 |
| 13 | 3.415254 | 3.435332 | 3.420473 | 1.128153 | 1.120330 | 1.133496 |
| 14 | 3.630893 | 3.609107 | 3.629921 | 1.183780 | 1.175957 | 1.190165 |
| 15 | 3.851414 | 3.805108 | 3.818189 | 1.232533 | 1.228196 | 1.245037 |
| 16 | 4.045659 | 4.038918 | 4.044193 | 1.285625 | 1.280586 | 1.293504 |
| 17 | 4.272698 | 4.237137 | 4.255230 | 1.349868 | 1.336351 | 1.353917 |
| 18 | 4.487902 | 4.453647 | 4.460074 | 1.397440 | 1.390792 | 1.409255 |
| 19 | 4.691207 | 4.653732 | 4.627010 | 1.456505 | 1.445372 | 1.462531 |
| 20 | 4.896212 | 4.856805 | 4.851899 | 1.515760 | 1.503110 | 1.518313 |

## Provenance

Pilot2170267/invc6f010a4a6e849c6898e1547c7fe6bda and selector2177097/inve42ca043345b4100829c7bf9fc72c01c both actually retained Code1/Status0/success/MainPID0.
Pilot completion SHA256:490a68801330f60af96cd6a4af88732cd09a9c416ed96cee293c2f8d3b0cd114.
Selector completion SHA256:f8481921cf0634fa8747e140f5980d1310e2d45209b4160d9b196fda897fb08d.
Root:[revision-results]/TRACE_paper_finish_20261116-1637-v1/time.
The next formal full Object/PC60 runs use exactly these selections. Formal ratios, paired accuracy contrasts and budget/near-time flags must be reported even if matching fails. Full-population accuracy cannot reselect the step count.
