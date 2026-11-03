# G Primary Results: Backbone, Corruption And OVSS Transfer

## Conclusion

C beats same-configuration Confidence in the three tested B/16 NACLIP clean
conditions and in the macro15 corruption average, but does not transfer
positively to the configured SCLIP variant. This is mixed evidence, not a
universal OVSS or corruption-robustness result. The main paper is not revised
until the related mechanism and compute studies are verified.

## Clean Conditions

Values are local mIoU, not literature-report numbers. CI is the paired C minus
Confidence difference, in percentage points.

| Implementation / Dataset | Zero Update | Confidence | C | C - Confidence | Paired Pointwise 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| naclip.ViT-B-16.coco_object.original.s0 | 29.977027 | 27.943375 | 29.951891 | 2.008517 | [1.478399, 2.534738] |
| naclip.ViT-B-16.pc60.original.s0 | 31.004466 | 30.074350 | 30.867324 | 0.792974 | [0.629951, 0.954531] |
| naclip.ViT-B-16.voc20.original.s0 | 80.309546 | 82.547409 | 83.647904 | 1.100496 | [0.334043, 1.924223] |
| sclip.ViT-B-16.voc20.original.s0 | 35.214555 | 33.928200 | 31.533137 | -2.395064 | [-3.512464, -1.344540] |

B/16 Object and PC60 C point estimates remain below their zero-update controls
by -0.025135 and -0.137142 points. C-versus-Confidence improvement is therefore not
equivalent to a net adaptation benefit over the initial model. No paired
C-minus-NoAdapt interval was part of this summarized primary contrast.

The SCLIP point loss is retained. Its C-minus-Confidence interval excludes
zero negatively. No positive cross-OVSS transfer claim is supported. This is
the prospectively specified TRACE multilayer SCLIP configuration (last9/12),
not a reproduced published whole SCLIP pipeline. The official final-layer
feature/input-LN-gradient/query-label gate establishes only its tested
component parity. A negative score does not itself establish a software fault
or authorize new settings, new layers, test tuning or a positive-result rerun.

## VOC20-C Severity 5

Macro15 mIoU: zero update 70.920661, Confidence
74.739450, C 75.320331. The paired C-minus-Confidence difference is
0.580882 points, CI [0.221363, 0.947074].
C has 11 point wins, 0 ties and 4 point losses across the15 predeclared types.
This is an arithmetic mean of15 per-type pooled mIoUs, not one pooled
corruption population and not a claim that all15 differences are positive.

| Corruption | C - Confidence | Paired Pointwise 95% CI |
| --- | ---: | --- |
| brightness | 0.837305 | [0.208381, 1.549317] |
| contrast | -1.038307 | [-2.269035, 0.172012] |
| defocus_blur | 0.710426 | [-0.425762, 1.872453] |
| elastic_transform | -1.006007 | [-1.997762, -0.045643] |
| fog | 0.393455 | [-0.541380, 1.286305] |
| frost | 1.360249 | [0.577925, 2.147457] |
| gaussian_noise | -0.279870 | [-1.604998, 1.022816] |
| glass_blur | 2.350912 | [1.277655, 3.474626] |
| impulse_noise | -0.985453 | [-2.189508, 0.194925] |
| jpeg_compression | 0.292410 | [-0.403943, 0.945888] |
| motion_blur | 1.700748 | [0.522855, 3.012197] |
| pixelate | 1.703222 | [0.838916, 2.595178] |
| shot_noise | 1.011093 | [-0.108821, 2.087937] |
| snow | 0.694813 | [0.014976, 1.365458] |
| zoom_blur | 0.968233 | [-0.187189, 2.075985] |

| Predeclared Group | C - Confidence | Paired Pointwise 95% CI |
| --- | ---: | --- |
| blur | 1.432580 | [0.755297, 2.101281] |
| digital | -0.012171 | [-0.547739, 0.528990] |
| noise | -0.084743 | [-0.907904, 0.716530] |
| weather | 0.821455 | [0.367018, 1.285612] |

Noise and digital group intervals include zero; elastic_transform has a
negative paired interval. Blur and weather have positive intervals under
this fixed protocol. Brightness belongs to weather as declared in advance.
The15 types are not treated as random independent samples; bootstrap draws
resample original source-image episodes jointly across arms/types.

## Scope And Provenance

All19 conditions/57 cells/104214 image-method evaluations have prediction seals
and successful G1954 retained exit before this new independently admitted CPU
scoring phase. NoAdapt is the same TRACE readout with zero updates, not a
literature plain-NACLIP baseline. B/16 uses last9/12 and L/14 last18/24; clean
settings are inherited with historical evaluation exposure, not independently
retuned or untouched. Each corrupted image provides its own candidates,
diagnostic supports, adaptation and prediction; no clean-image guidance.

Uncertainty uses10000 predeclared PCG64 whole-original-episode paired draws,
fixed-seed/configuration pointwise percentile intervals; no multiplicity
correction, new random training seed, seed selection, failed-draw dropping or
post-result bin merging. Intervals are conditional on the existing settings
and sampled episodes, not broad method-selection uncertainty.

CPU scorer2321 ran23:30:16 to23:31:45Beijing, PID2094938/invocation
bccd4c5d1f564e3380b3acadbd42bdd5, retained loadedactive/exited MainPID0
Code1Status0 Resultsuccess. All269 frozen sources, trusted parent metadata,
inputs/protocol/audit/report/STARTED/global and22 child-score hashes passed
main terminal verification; no repeated prediction/RGB/GT payload scan.
The scoring phase took84.648397seconds, produced17919844bytes and used no GPU.

- Suite: /home/wwww/hzl/results/trace_mechanism_20261102/G_offline_score_20261102_2321
- Prepared SHA: ad2804b18158528eefc11525d87d43c696b02ae57f3c2a94291663545d5b9569
- Analysis SHA: 9356fe54cb3eeeb94e0068f06593efbbc7194ff459581897bc685851b9210048
- Receipt: G_primary_offline_score_terminal_001.json
- Exact raw values: CONDITION_RESULTS.json and PAIRED_SUMMARIES.json in the suite

## Claim-Evidence Check

- Claim: C's weighting advantage over Confidence persists on tested NACLIP B/16 settings. Evidence: all three paired intervals positive. Status: supported within these settings, not every backbone.
- Claim: C universally benefits adaptation. Evidence: Object/PC60 zero-update controls are slightly higher than C; SCLIP C is lower. Status: not supported.
- Claim: C helps across every corrupted input type. Evidence:11 point wins/4 losses and mixed group intervals. Status: not supported; only macro15 benefit is established here.
- Claim: C transfers positively across the tested OVSS implementations. Evidence: SCLIP paired negative interval. Status: not supported.
- Claim: These results identify C's causal update utility. Evidence: G measures end-to-end outcomes, not identification; M01 remains mixed. Status: not supported.
