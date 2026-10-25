# Frozen ORIGINAL TRACE: CPU Evidence Interpretation

## Evidence Units and Limits

This is a post-hoc, seed-0 reanalysis of frozen observations, not a new model run or new original-submission result.
Candidate observations are isolated one-step, unit-weight updates. Region-normalized entropy cancels a single positive scalar weight; these observations test association/ranking, not the safety of weighting that one update.
Region accuracy loss is a net regional segmentation change. Majority crossing uses pre >= 0.5 and post < 0.5, conditional on initially majority-correct regions. Neither is a count of individual pixel transitions.
The fixed historical probability-drop event (>= 0.001) is a probability proxy, never actual correct-to-wrong segmentation. No threshold sweep, fitted risk model, or best-cell selection is performed.
Full-episode counters are actual labeled-pixel C->W and W->C, with initially correct and initially wrong denominators respectively. They remain separate from isolated candidate events.
Follow-up candidate_diagnostics.csv has no candidate mask. There is no validated mask-level association to follow-up pixel transitions; no candidate-level pixel C->W is inferred from a region mean or image ID.
Candidate size is cached feature-grid mask area, not original-image pixel area. Initial accuracy is a prediction-purity proxy, not an independently measured class-purity statistic. The size/purity table is a descriptive whole-candidate stratification, NOT within-candidate spatial heterogeneity.
Boundary/interior measurements, within-token variance, and within-candidate sub-region variation are not present in the consumed caches. No such measurements are fabricated, and no follow-up mask reconstruction is attempted.
## Fixed Matching and Overlap

Coarsened exact matching uses confidence width 0.02 and role; strict matching additionally uses class, floor(log2(area)), and initial-accuracy bins of width 0.2. Confidence/accuracy 1.0 are included in their final bins. Strict means extra conditioning, not exact equality of continuous covariates.
Within every stratum with >= 20 candidates, select equal floor(n/4) low/high raw-score tails, sorted deterministically by score then candidate ID. Constant scores and tied nonseparable tails are excluded and counted. Outcomes never select membership. All three scores, three events, four datasets, and both matching schemes are retained.
Positive low-minus-high event-rate gaps agree with 1-score risk; zero/negative gaps are retained. Confidence matching is a within-confidence-bin reference, not proof of information beyond exact confidence. Strict matching uses GT-derived pre-accuracy only for offline conditioning, never adaptation.
Coverage_total uses all source candidates; coverage_event_eligible uses event-eligible candidates; coverage_match_eligible uses candidates also passing common-score/covariate checks. matching_strata.csv retains unsupported strata and all exclusion reasons. Low overlap limits generalization; matching is descriptive and not causal identification.
## Candidate Results

- voc21/region_accuracy_drop: 410/16010 positives; 16010 source candidates, 0 ineligible/missing.
- voc21/region_majority_crossing: 166/11224 positives; 16010 source candidates, 4786 ineligible/missing.
- voc21/historical_probability_drop: 3778/16010 positives; 16010 source candidates, 0 ineligible/missing.
- pc60/region_accuracy_drop: 492/11223 positives; 11223 source candidates, 0 ineligible/missing.
- pc60/region_majority_crossing: 154/3564 positives; 11223 source candidates, 7659 ineligible/missing.
- pc60/historical_probability_drop: 4271/11223 positives; 11223 source candidates, 0 ineligible/missing.
- coco_object/region_accuracy_drop: 814/17515 positives; 17515 source candidates, 0 ineligible/missing.
- coco_object/region_majority_crossing: 368/9430 positives; 17515 source candidates, 8085 ineligible/missing.
- coco_object/historical_probability_drop: 5307/17515 positives; 17515 source candidates, 0 ineligible/missing.
- coco_cd/region_accuracy_drop: 107/1998 positives; 1998 source candidates, 0 ineligible/missing.
- coco_cd/region_majority_crossing: 7/854 positives; 1998 source candidates, 1144 ineligible/missing.
- coco_cd/historical_probability_drop: 884/1998 positives; 1998 source candidates, 0 ineligible/missing.

## Matching: Actual Segmentation Events

Both conditioning schemes and both intervention-derived scores are shown below; the within-confidence-bin reference and historical-proxy contrasts are also retained in matching_summary.csv.

| Dataset | Event | Matching | Raw Score | Low Pos/N | High Pos/N | Low-High (pp) | Matched/Eligible | Retained Pos/Eligible Pos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| voc21 | accuracy loss | confidence/role | quality | 75/3950 | 174/3950 | -2.506 | 7900/16010 (49.34%) | 249/410 |
| voc21 | accuracy loss | strict | quality | 57/2084 | 59/2084 | -0.096 | 4168/16010 (26.03%) | 116/410 |
| voc21 | accuracy loss | confidence/role | reliability | 76/3950 | 174/3950 | -2.481 | 7900/16010 (49.34%) | 250/410 |
| voc21 | accuracy loss | strict | reliability | 57/2084 | 59/2084 | -0.096 | 4168/16010 (26.03%) | 116/410 |
| voc21 | majority crossing | confidence/role | quality | 60/2759 | 27/2759 | +1.196 | 5518/11224 (49.16%) | 87/166 |
| voc21 | majority crossing | strict | quality | 35/2007 | 37/2007 | -0.100 | 4014/11224 (35.76%) | 72/166 |
| voc21 | majority crossing | confidence/role | reliability | 59/2759 | 27/2759 | +1.160 | 5518/11224 (49.16%) | 86/166 |
| voc21 | majority crossing | strict | reliability | 35/2007 | 37/2007 | -0.100 | 4014/11224 (35.76%) | 72/166 |
| pc60 | accuracy loss | confidence/role | quality | 46/2723 | 287/2723 | -8.851 | 5446/11223 (48.53%) | 333/492 |
| pc60 | accuracy loss | strict | quality | 2/278 | 3/278 | -0.360 | 556/11223 (4.95%) | 5/492 |
| pc60 | accuracy loss | confidence/role | reliability | 48/2723 | 285/2723 | -8.704 | 5446/11223 (48.53%) | 333/492 |
| pc60 | accuracy loss | strict | reliability | 2/278 | 3/278 | -0.360 | 556/11223 (4.95%) | 5/492 |
| pc60 | majority crossing | confidence/role | quality | 52/805 | 13/805 | +4.845 | 1610/3564 (45.17%) | 65/154 |
| pc60 | majority crossing | strict | quality | 1/86 | 3/86 | -2.326 | 172/3564 (4.83%) | 4/154 |
| pc60 | majority crossing | confidence/role | reliability | 51/805 | 13/805 | +4.720 | 1610/3564 (45.17%) | 64/154 |
| pc60 | majority crossing | strict | reliability | 1/86 | 3/86 | -2.326 | 172/3564 (4.83%) | 4/154 |
| coco_object | accuracy loss | confidence/role | quality | 153/4309 | 299/4309 | -3.388 | 8618/17515 (49.20%) | 452/814 |
| coco_object | accuracy loss | strict | quality | 109/1604 | 74/1604 | +2.182 | 3208/17515 (18.32%) | 183/814 |
| coco_object | accuracy loss | confidence/role | reliability | 153/4309 | 299/4309 | -3.388 | 8618/17515 (49.20%) | 452/814 |
| coco_object | accuracy loss | strict | reliability | 109/1604 | 74/1604 | +2.182 | 3208/17515 (18.32%) | 183/814 |
| coco_object | majority crossing | confidence/role | quality | 146/2294 | 37/2294 | +4.752 | 4588/9430 (48.65%) | 183/368 |
| coco_object | majority crossing | strict | quality | 73/1506 | 44/1506 | +1.926 | 3012/9430 (31.94%) | 117/368 |
| coco_object | majority crossing | confidence/role | reliability | 145/2294 | 37/2294 | +4.708 | 4588/9430 (48.65%) | 182/368 |
| coco_object | majority crossing | strict | reliability | 73/1506 | 44/1506 | +1.926 | 3012/9430 (31.94%) | 117/368 |
| coco_cd | accuracy loss | confidence/role | quality | 6/450 | 41/450 | -7.778 | 900/1998 (45.05%) | 47/107 |
| coco_cd | accuracy loss | strict | quality | 11/203 | 8/203 | +1.478 | 406/1998 (20.32%) | 19/107 |
| coco_cd | accuracy loss | confidence/role | reliability | 6/450 | 41/450 | -7.778 | 900/1998 (45.05%) | 47/107 |
| coco_cd | accuracy loss | strict | reliability | 11/203 | 8/203 | +1.478 | 406/1998 (20.32%) | 19/107 |
| coco_cd | majority crossing | confidence/role | quality | 6/196 | 0/196 | +3.061 | 392/854 (45.90%) | 6/7 |
| coco_cd | majority crossing | strict | quality | 0/62 | 0/62 | +0.000 | 124/854 (14.52%) | 0/7 |
| coco_cd | majority crossing | confidence/role | reliability | 6/196 | 0/196 | +3.061 | 392/854 (45.90%) | 6/7 |
| coco_cd | majority crossing | strict | reliability | 0/62 | 0/62 | +0.000 | 124/854 (14.52%) | 0/7 |

## All Full-Episode Policies

Dataset mIoU is recomputed after summing confusion matrices, omitting only zero-union classes; it is not the mean of image or episode mIoUs. Rates below use different pre-correct/pre-wrong populations, so compare counts and denominators as well.

| Dataset | Policy | Actual C->W / Initially Correct | Actual W->C / Initially Wrong | mIoU Before -> After |
| --- | --- | --- | --- | --- |
| voc21 | uniform | 1428413/54887050 (2.602%) | 3507683/13805886 (25.407%) | 49.6587 -> 52.1639 |
| voc21 | confidence | 1370198/54887050 (2.496%) | 3278627/13805886 (23.748%) | 49.6587 -> 52.1397 |
| voc21 | augmentation_consistency | 1416436/54887050 (2.581%) | 3527987/13805886 (25.554%) | 49.6587 -> 52.3032 |
| voc21 | quality | 1327830/54887050 (2.419%) | 2852719/13805886 (20.663%) | 49.6587 -> 52.1524 |
| voc21 | full | 1330407/54887050 (2.424%) | 2871950/13805886 (20.802%) | 49.6587 -> 52.1589 |
| pc60 | uniform | 9740222/127657958 (7.630%) | 14396876/128490522 (11.205%) | 28.9524 -> 30.1185 |
| pc60 | confidence | 9567072/127657958 (7.494%) | 14568225/128490522 (11.338%) | 28.9524 -> 30.1343 |
| pc60 | augmentation_consistency | 9827923/127657958 (7.699%) | 14297868/128490522 (11.128%) | 28.9524 -> 30.1604 |
| pc60 | quality | 8392526/127657958 (6.574%) | 15401898/128490522 (11.987%) | 28.9524 -> 30.3876 |
| pc60 | full | 8409685/127657958 (6.588%) | 15410107/128490522 (11.993%) | 28.9524 -> 30.3682 |
| coco_object | uniform | 21158568/162822630 (12.995%) | 35767216/88057370 (40.618%) | 29.1062 -> 29.0653 |
| coco_object | confidence | 18411360/162822630 (11.308%) | 31929192/88057370 (36.260%) | 29.1062 -> 29.9398 |
| coco_object | augmentation_consistency | 20694576/162822630 (12.710%) | 35126377/88057370 (39.890%) | 29.1062 -> 29.2520 |
| coco_object | quality | 13748907/162822630 (8.444%) | 27618503/88057370 (31.364%) | 29.1062 -> 31.9362 |
| coco_object | full | 13679392/162822630 (8.401%) | 27659032/88057370 (31.410%) | 29.1062 -> 31.9972 |
| coco_cd | uniform | 539146/21474111 (2.511%) | 1344215/3613889 (37.196%) | 62.8139 -> 66.6572 |
| coco_cd | confidence | 486779/21474111 (2.267%) | 1383039/3613889 (38.270%) | 62.8139 -> 67.2863 |
| coco_cd | augmentation_consistency | 367502/21474111 (1.711%) | 1878689/3613889 (51.985%) | 62.8139 -> 71.4934 |
| coco_cd | quality | 431092/21474111 (2.007%) | 2362020/3613889 (65.360%) | 62.8139 -> 75.1475 |
| coco_cd | full | 436741/21474111 (2.034%) | 2373282/3613889 (65.671%) | 62.8139 -> 75.1644 |

## Actionable Reading

Use region_accuracy_drop for actual net segmentation degradation and region_majority_crossing for an initially majority-correct population. Report positive counts, event eligibility, and score direction alongside AUROC/AP. AP is average precision, not trapezoidal PR area; both metrics are intentionally blank for one-class cells. All predictors share the same finite-score denominator within each event.
A raw 1-reliability AUROC <= 0.5 does not support universal monotone risk detection. Do not reverse the score after seeing the result. Matched gaps concern only their retained overlap and may disagree with unconditional ranking.
For allocation utility, compare full with uniform/confidence/flip-consistency and quality on each dataset, considering mIoU AND both transition counts. Higher mIoU with more C->W is a trade-off, not uniformly safer adaptation. Full minus quality isolates incremental behavior under the cached paired protocol, not a new adaptation experiment.
- voc21, full minus uniform: mIoU -0.0050 pp; actual C->W -98006 pixels; actual W->C -635733 pixels.
- voc21, full minus confidence: mIoU +0.0191 pp; actual C->W -39791 pixels; actual W->C -406677 pixels.
- voc21, full minus augmentation_consistency: mIoU -0.1443 pp; actual C->W -86029 pixels; actual W->C -656037 pixels.
- voc21, full minus quality: mIoU +0.0065 pp; actual C->W +2577 pixels; actual W->C +19231 pixels.
- pc60, full minus uniform: mIoU +0.2498 pp; actual C->W -1330537 pixels; actual W->C +1013231 pixels.
- pc60, full minus confidence: mIoU +0.2340 pp; actual C->W -1157387 pixels; actual W->C +841882 pixels.
- pc60, full minus augmentation_consistency: mIoU +0.2078 pp; actual C->W -1418238 pixels; actual W->C +1112239 pixels.
- pc60, full minus quality: mIoU -0.0194 pp; actual C->W +17159 pixels; actual W->C +8209 pixels.
- coco_object, full minus uniform: mIoU +2.9320 pp; actual C->W -7479176 pixels; actual W->C -8108184 pixels.
- coco_object, full minus confidence: mIoU +2.0575 pp; actual C->W -4731968 pixels; actual W->C -4270160 pixels.
- coco_object, full minus augmentation_consistency: mIoU +2.7452 pp; actual C->W -7015184 pixels; actual W->C -7467345 pixels.
- coco_object, full minus quality: mIoU +0.0610 pp; actual C->W -69515 pixels; actual W->C +40529 pixels.
- coco_cd, full minus uniform: mIoU +8.5073 pp; actual C->W -102405 pixels; actual W->C +1029067 pixels.
- coco_cd, full minus confidence: mIoU +7.8781 pp; actual C->W -50038 pixels; actual W->C +990243 pixels.
- coco_cd, full minus augmentation_consistency: mIoU +3.6710 pp; actual C->W +69239 pixels; actual W->C +494593 pixels.
- coco_cd, full minus quality: mIoU +0.0170 pp; actual C->W +5649 pixels; actual W->C +11262 pixels.

Raw score/event cells with AUROC <= 0.5: 9 (all retained in risk_metrics.csv).
Sparse majority-crossing cells and unmatched strata cannot establish reliable detection or absence of risk. Descriptive candidates share images; no independence-based p-values, learned-model AUROCs, or unsupported causal claims are reported.
All consumed files have SHA-256 hashes in provenance.json. Episode cache counters/confusions and prediction-seal indices are checked against cached results and paired pre-predictions. Prediction .npz payloads are NOT reopened/rehashed and GT is NOT rescored; this is cache consistency verification, not a fresh end-to-end prediction audit.
