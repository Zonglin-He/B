# Current TRACE Paper Evidence Audit

No new model inference or GT scoring. Intervals, where eligible, use whole actual update-dependency components.
An unavailable interval is not replaced by independent-image resampling or marginal-CI subtraction.

| Condition | TRACE minus MLMP pp | Dependence groups | Largest group | Paired95 |
|---|---:|---:|---:|---|
| L14_pc60 | +0.7310 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_coco_object | +2.0202 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20 | +2.3166 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc21 | +0.3828 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_pc59 | +0.9304 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_coco_stuff | +0.7437 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_waterbirds_seg | +4.1216 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_coco_cd | +4.9856 | 1 | 1.0000 | UNIDENTIFIABLE |
| B16_coco_object | -0.7377 | 1 | 1.0000 | UNIDENTIFIABLE |
| B16_pc60 | -0.4129 | 1 | 1.0000 | UNIDENTIFIABLE |
| B16_voc20 | +0.2552 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_gaussian_noise | +2.0173 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_shot_noise | +2.2303 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_impulse_noise | +1.6139 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_defocus_blur | +0.2692 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_glass_blur | +1.4558 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_motion_blur | +1.4979 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_zoom_blur | +2.5086 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_snow | +2.5829 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_frost | +3.6523 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_fog | +2.9585 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_brightness | +2.4368 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_contrast | +1.9197 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_elastic_transform | +3.1611 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_pixelate | +1.5258 | 1 | 1.0000 | UNIDENTIFIABLE |
| L14_voc20_C_jpeg_compression | +1.7976 | 1 | 1.0000 | UNIDENTIFIABLE |

## Calibration Overlap

pc60: confirm 61/128, refine 61/128, screen 34/64
coco_object: confirm 0/128, refine 0/128, screen 0/64
voc20: confirm 0/128, refine 0/128, screen 0/64
voc21: confirm 0/128, refine 0/128, screen 0/64
pc59: confirm 59/128, refine 62/128, screen 33/64
coco_stuff: confirm 0/128, refine 0/128, screen 0/64
waterbirds_seg: confirm 0/128, refine 0/128, screen 0/64
coco_cd: confirm 0/128, refine 0/128, screen 0/64

The fixed mean benchmark remains valid as a descriptive result with disclosed development exposure.
Whole-group inference on crossed shuffled batch designs may need a new common-block study.
The prior recorded episode times exclude loading/startup and cannot settle equal end-to-end compute.
