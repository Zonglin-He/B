# G03 Fixed Severity Curves: Complete, Mixed At Severity5

## Actual Result And Limits

The ten new s1/s3 conditions finished all30cells/43470predictions; the separately
admitted CPU scorer then combined them with five authenticated sealed S5
compact-count streams. Main verified the exact successful prediction/scoring
terminals and all13newscoreoutput hashes AND original output identities.
This is the actual complete five-type, three-severity, three-arm curve readout.

For each of the five fixed types at severity1 and3, C minus Confidence has a
positive95%pointwise interval. At reused severity5, motion blur has a positive
interval; Gaussian noise, fog, contrast and JPEG intervals cross zero. Gaussian
noise and contrast have negative S5 point estimates. These findings are not a
universal robustness, monotone-severity, candidate-causal or equal-budget claim.
Do not drop weak/negative S5 points, fit an unregistered slope/AUC, select a
winner, change severity/type membership or rerun to obtain positive outcomes.

## Complete Fifteen Condition Contrasts

Every row uses the original1449VOC20sources. NoAdapt is zero updates with the
same TRACE multi-layer readout, not literature plain NACLIP. All mIoUs and
C-Confidence differences are in percentage points. Severity5 is exact reuse,
not new inference.

| Type | Severity | NoAdapt | Confidence | C | C Minus Confidence | Paired Pointwise95% CI | Undefined Draws |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| gaussian_noise | 1 | 79.102096 | 81.419841 | 83.198586 | 1.778746 | [0.977234, 2.600063] | 0 |
| gaussian_noise | 3 | 76.726263 | 79.837738 | 81.712640 | 1.874902 | [0.969050, 2.811805] | 0 |
| gaussian_noise | 5 | 64.374362 | 68.578659 | 68.298789 | -0.279870 | [-1.637438, 1.032325] | 0 |
| motion_blur | 1 | 79.078664 | 81.661695 | 83.143547 | 1.481852 | [0.862182, 2.119054] | 0 |
| motion_blur | 3 | 79.041112 | 80.168090 | 82.504048 | 2.335958 | [1.346001, 3.350120] | 0 |
| motion_blur | 5 | 75.730215 | 77.999869 | 79.700617 | 1.700748 | [0.531744, 2.996389] | 0 |
| fog | 1 | 77.547148 | 80.455751 | 82.097125 | 1.641375 | [1.033804, 2.316285] | 0 |
| fog | 3 | 76.713859 | 79.997205 | 80.806455 | 0.809250 | [0.014432, 1.563415] | 0 |
| fog | 5 | 73.206492 | 77.846146 | 78.239601 | 0.393455 | [-0.549950, 1.296345] | 0 |
| contrast | 1 | 78.926270 | 81.700603 | 83.039234 | 1.338631 | [0.719275, 2.008146] | 0 |
| contrast | 3 | 78.971794 | 82.205507 | 83.007308 | 0.801801 | [0.093241, 1.512163] | 0 |
| contrast | 5 | 73.476842 | 76.540851 | 75.502544 | -1.038307 | [-2.253165, 0.179159] | 0 |
| jpeg_compression | 1 | 78.438450 | 82.025849 | 83.438152 | 1.412303 | [0.744516, 2.118886] | 0 |
| jpeg_compression | 3 | 78.321783 | 82.379327 | 83.838875 | 1.459548 | [0.712004, 2.220655] | 0 |
| jpeg_compression | 5 | 75.407497 | 79.718510 | 80.010920 | 0.292410 | [-0.406769, 0.957080] | 0 |

## All Forty-Five Arm Intervals

| Type | Severity | Arm | mIoU | Pointwise95% CI | Undefined Draws |
| --- | ---: | --- | ---: | --- | ---: |
| gaussian_noise | 1 | no_adapt | 79.102096 | [77.531713, 80.466815] | 0 |
| gaussian_noise | 1 | image_confidence | 81.419841 | [79.759223, 82.775159] | 0 |
| gaussian_noise | 1 | image_C | 83.198586 | [81.635531, 84.490487] | 0 |
| gaussian_noise | 3 | no_adapt | 76.726263 | [75.108784, 78.126401] | 0 |
| gaussian_noise | 3 | image_confidence | 79.837738 | [78.224512, 81.181883] | 0 |
| gaussian_noise | 3 | image_C | 81.712640 | [80.187134, 82.998950] | 0 |
| gaussian_noise | 5 | no_adapt | 64.374362 | [62.452652, 66.119784] | 0 |
| gaussian_noise | 5 | image_confidence | 68.578659 | [66.380401, 70.555591] | 0 |
| gaussian_noise | 5 | image_C | 68.298789 | [66.099622, 70.266877] | 0 |
| motion_blur | 1 | no_adapt | 79.078664 | [77.534881, 80.434031] | 0 |
| motion_blur | 1 | image_confidence | 81.661695 | [80.084282, 82.982853] | 0 |
| motion_blur | 1 | image_C | 83.143547 | [81.712966, 84.348554] | 0 |
| motion_blur | 3 | no_adapt | 79.041112 | [77.427016, 80.457515] | 0 |
| motion_blur | 3 | image_confidence | 80.168090 | [78.423907, 81.631093] | 0 |
| motion_blur | 3 | image_C | 82.504048 | [80.877651, 83.903176] | 0 |
| motion_blur | 5 | no_adapt | 75.730215 | [73.799408, 77.426156] | 0 |
| motion_blur | 5 | image_confidence | 77.999869 | [75.951944, 79.781210] | 0 |
| motion_blur | 5 | image_C | 79.700617 | [77.765112, 81.369317] | 0 |
| fog | 1 | no_adapt | 77.547148 | [75.997824, 78.874884] | 0 |
| fog | 1 | image_confidence | 80.455751 | [78.896862, 81.746197] | 0 |
| fog | 1 | image_C | 82.097125 | [80.522546, 83.427534] | 0 |
| fog | 3 | no_adapt | 76.713859 | [75.128513, 78.077007] | 0 |
| fog | 3 | image_confidence | 79.997205 | [78.398561, 81.321181] | 0 |
| fog | 3 | image_C | 80.806455 | [79.177099, 82.216462] | 0 |
| fog | 5 | no_adapt | 73.206492 | [71.500922, 74.719040] | 0 |
| fog | 5 | image_confidence | 77.846146 | [76.137335, 79.279777] | 0 |
| fog | 5 | image_C | 78.239601 | [76.405761, 79.861514] | 0 |
| contrast | 1 | no_adapt | 78.926270 | [77.395571, 80.234076] | 0 |
| contrast | 1 | image_confidence | 81.700603 | [80.183056, 82.992683] | 0 |
| contrast | 1 | image_C | 83.039234 | [81.553051, 84.313033] | 0 |
| contrast | 3 | no_adapt | 78.971794 | [77.360900, 80.375375] | 0 |
| contrast | 3 | image_confidence | 82.205507 | [80.664966, 83.437152] | 0 |
| contrast | 3 | image_C | 83.007308 | [81.450438, 84.336183] | 0 |
| contrast | 5 | no_adapt | 73.476842 | [71.584357, 75.173958] | 0 |
| contrast | 5 | image_confidence | 76.540851 | [74.581355, 78.250072] | 0 |
| contrast | 5 | image_C | 75.502544 | [73.296967, 77.447011] | 0 |
| jpeg_compression | 1 | no_adapt | 78.438450 | [76.900120, 79.806103] | 0 |
| jpeg_compression | 1 | image_confidence | 82.025849 | [80.455501, 83.343926] | 0 |
| jpeg_compression | 1 | image_C | 83.438152 | [81.945309, 84.675522] | 0 |
| jpeg_compression | 3 | no_adapt | 78.321783 | [76.790496, 79.606726] | 0 |
| jpeg_compression | 3 | image_confidence | 82.379327 | [80.850877, 83.661595] | 0 |
| jpeg_compression | 3 | image_C | 83.838875 | [82.378471, 85.061320] | 0 |
| jpeg_compression | 5 | no_adapt | 75.407497 | [73.761682, 76.809826] | 0 |
| jpeg_compression | 5 | image_confidence | 79.718510 | [77.999143, 81.154847] | 0 |
| jpeg_compression | 5 | image_C | 80.010920 | [78.267512, 81.525925] | 0 |

The S5 points are numerically the original G1954/G2321 values; their intervals
here use the newly frozen G03 shared-draw namespace. Different interval
endpoints are not independent replication, endpoint selection or an overwrite
of G-primary results/intervals. No contrast between severity effects was
registered or inferred. Individual point intervals are not simultaneous
confidence bands for a curve.

## Descriptive Cost And Resources

| Condition | Raw Model Seconds NoAdapt / Conf / C | C Diagnostic Inputs | Conf / C Adapt Forward Calls | Condition Wall Seconds |
| --- | --- | ---: | --- | ---: |
| naclip.ViT-L-14.voc20.gaussian_noise.s5 | 7.877458 / 230.205609 / 249.352466 | 11592 | 6525 / 6525 | 515.798060 |
| naclip.ViT-L-14.voc20.motion_blur.s5 | 7.834836 / 228.871701 / 247.225650 | 11591 | 6525 / 6525 | 512.447680 |
| naclip.ViT-L-14.voc20.fog.s5 | 7.766873 / 227.119972 / 245.618585 | 11592 | 6525 / 6525 | 508.817140 |
| naclip.ViT-L-14.voc20.contrast.s5 | 7.854948 / 229.226192 / 248.087918 | 11592 | 6525 / 6525 | 513.354046 |
| naclip.ViT-L-14.voc20.jpeg_compression.s5 | 7.870996 / 230.639641 / 248.815978 | 11592 | 6525 / 6525 | 515.743814 |
| naclip.ViT-L-14.voc20.gaussian_noise.s1 | 7.789800 / 225.881711 / 245.949203 | 11583 | 6525 / 6525 | 509.663467 |
| naclip.ViT-L-14.voc20.gaussian_noise.s3 | 7.729163 / 224.790640 / 244.884117 | 11592 | 6525 / 6525 | 507.442799 |
| naclip.ViT-L-14.voc20.motion_blur.s1 | 7.732998 / 224.955046 / 243.678669 | 11585 | 6525 / 6525 | 506.308226 |
| naclip.ViT-L-14.voc20.motion_blur.s3 | 7.734376 / 224.673935 / 244.196519 | 11589 | 6525 / 6525 | 506.405143 |
| naclip.ViT-L-14.voc20.fog.s1 | 7.762058 / 226.070476 / 244.699068 | 11590 | 6525 / 6525 | 508.375835 |
| naclip.ViT-L-14.voc20.fog.s3 | 7.788414 / 226.753799 / 245.245273 | 11592 | 6525 / 6525 | 509.768376 |
| naclip.ViT-L-14.voc20.contrast.s1 | 7.704111 / 225.441514 / 243.203364 | 11582 | 6525 / 6525 | 506.129735 |
| naclip.ViT-L-14.voc20.contrast.s3 | 7.753832 / 225.418671 / 244.610858 | 11590 | 6525 / 6525 | 507.460403 |
| naclip.ViT-L-14.voc20.jpeg_compression.s1 | 7.729756 / 225.721176 / 244.051450 | 11582 | 6525 / 6525 | 507.352290 |
| naclip.ViT-L-14.voc20.jpeg_compression.s3 | 7.713126 / 224.886815 / 243.892099 | 11588 | 6525 / 6525 | 506.131669 |

Complete actual work, warmup-adjusted throughput, exact trainable/frozen
parameter identities and all resource observations remain in sealed
CURVE_RESULTS.json. C diagnostic forwards/inputs are charged. Each adaptive
arm has nineupdates per originalbatch2episode;725episodes include a one-source
tail. NoAdapt has no adaptive updates. Candidate construction and scoring for
corrupted images use their own corrupted input, without clean guidance.

Raw model time and inherited condition wall time are descriptive, not an
equal-time allocation comparison; V01 remains the separate accuracy-time study.
Warmup-adjusted throughput does not replace all-episode raw time. Offline
CPU GT scoring and the177.916272sCPU input-generation cost are separate stages.
S5 resource metadata came from the earlier run; new s1/s3 metadata came from
this run. CPU RSS fields are cumulative and GPU allocated observations do not
establish whole-process peak or temporal equality across runs. No extrapolated
FLOPs or subtraction of unmeasured overhead is used.

## Fixed Estimator And Geometry

Protocol7655bee2573e1c79b8a4856f9272ce9b45b5ccc025c1c09ca740f2d9b96cca86
specifies five original declared types, severities1/3/5 and the three arms.
The actual population is1449sources,725wholeoriginalbatch2episodes with a
single-source tail. Class-wise I/U is pooled within each type/severity before
class averaging. There is no cross-type/severity pooling, macro15, AUC, slope
or score-selected curve.

All10,000originalepisode PCG64draws are shared across all45armcells and15paired
contrasts. Seed is SHA256first16bytes little-endian uint128 of compact ASCII
[20261102,"G03","voc20","severity_curves"]. The reported seed integer is
213738108257582924611655285992217739350. All observed published arm/contrast
intervals have zero undefined draws. The protocol retains undefined draws and
null intervals rather than drop/rebin them. Intervals are95%pointwise linear
percentiles with no multiplicity correction.

Indexed palette GT uses OpenCV INTER_NEAREST224, identical to primary G,
not the native-PIL geometry of M02/M03/M04/V01. Each new source/condition GT
was read once and three predictions processed serially. S5 GT/predictions were
not read again; its five bounded compact streams authenticate original
source IDs/order/episode grouping/label hashes/arms/readout/config and original
G1954prediction/G2321score provenance. No enormous per_image_counts file was
loaded and no old G-primary confidence interval was overwritten.

Types/severities were fixed before these new predictions; the G03 estimator
was frozen after G02S5 outcomes were known. Historical WB development overlap,
configuration selection and test exposure remain disclosed. This is not
untouched testing, independent calibration, a full literature baseline
replication, evidence for SCLIP transfer or a general causal allocation result.

## Exact Execution Evidence

Prediction G03_predictions_20261103_1258:
- Prepared dadb651d45f9d9d60341d61727da56863606a5404dba7de6119bd277e1c0832c.
- Global c8045079f20a8730b33026656fc0d19348a576f855ebd6b86d0ca7049e072a8a.
- All30cells43470evaluations4595771977artifactbytes;10childseals.
- Actual retained success14:35:05Beijing, Exec2289686/MainPID0/Code1Status0,
  invocation74be1990a0a544e0a78a39f5a6a0501b.
- Actual independent prediction artifact audit and main fresh frozen admission
  preceded one manual12GiBRAM/1GiBswap GPU service, shared lock inode54264848.
  Actual argv/exe/cwd/env/invocation/cgroup/caps/OOMstop/Restartno/
  RemainAfterExityes/KillModecontrol-group were verified at dispatch.
- All10condition realfirst4RGBgates passed360pairedactualoptimizersteps,
  allsixfinite maxabs0/zerochangedlabels, gatebarrier
  2912cfc7e7427f7b1ee7f9fb0b85c12a16425faf5db6486a7893ae9a4079a62d.
- Main325runtime/native68/65250payloadstat/source/protocol/audit/start/gates/
  all10children/costreset/fiveS5bindingsverified without terminalGT_RGB_NPYscan.
- Receipt G03_prediction_terminal_001.json SHA
  c98f7ce94374cbfe3c0c8a479dd206e5f63425053903021940e26af5f24cae2a.

Score G03_offline_score_20261103_1455:
- Prepared2d7908f30ec2c3f6826e1ee5c317e5773c5facb6de5b020a9be60db074f62a5a.
- INPUTS05e64c1100532d6d46f8134db9bc81068e5f47f88195539159ccf60ee5c1c14c.
- PROTOCOL1b05b26d77a631cce0f716d0a7c55cc71e7dfcd8750283722edca3c821755572.
- Independent report7eb6bfdfcfaf537b890aa346a240bd8bc5d9c988165efe6a850ad4e2ab3e983d,
  admission780f138b868f29207f0861ef84b57272e18a1d88b2476d3223ff5565d65a2dec.
- Independent147frozenCPUtests8.18s/6reference-mutationdemonstrationsdeselected;
 325runtime5242846B/89genuinefrozenorigins/native68/65250stat/fiveS5streams.
  The incorrect112episode reviewer-harness assumption was corrected to genuine
  frozenbatch2/725episodes, original failure preserved; no frozenbytes changed.
- Main fresh frozen admission27.323912s/actualcv2torchintra-inter1/1/1/
  CUDAfalse/GTfalse/nomarkers/sameunitnotfound/disk126357979136B.
- ManualONCEtrace-curve-score-20261103-1455.service15:08:53to15:11:05Beijing,
  retainedloadedactive/exited MainPID0 Exec2312767 Code1Status0Resultsuccess,
  SAMEinvocation5f04ac05c5c84e90b0eb24d990e894b7.
- Dispatch AND immediate actualargv/exe/cwd/env/invocation/effectivecgroup in
  one tool orchestration:4GiBRAM512MiBswapCPU100Nice10idleIO/OOMstopRestartno/
  RemainAfterExityes/KillModecontrol-group. Logged actual1/1/1/CUDAfalse.
  Retained systemd memory-peak accounting is not used as scientific cost.
- GlobalANALYSIS1958034a312e9c476c2c8f430592b5939ce83fe9740a5059639c4d3ba7f65f3a.
- All13newoutputs/originalidentities+hashes/mainruntime/source/native/parent/
  protocol/audit/start/gates/stat/S5bindings verified:7382274B/102.554681s.
- ReceiptG03_offline_score_terminal_001.json SHA
  d80e661d1045e89a3085fbfdd9f4247539b0c14bc8bebc889245852736feeef5.
  CPUadmission/live receipts: G03_CPU_score_admission_validation_001.json and
  G03_offline_score_execution_started_001.json.

The frozen analysis seal retains its producer's scientific_completion=false
field. This is not a live-running marker: main separately authenticated the
successful terminal and all output original identities/hashes, and records
scientific_score_successfully_completed=true in its terminal receipt.
No raw GT/RGB/weights or prediction arrays were uploaded. No rerun, metric
change, new algorithm/model or paper integration was used for this readout.
