# Adopted TRACE Parameters Used In The Full Comparison

These are the actual22-field configurations recorded in the sealed full26 comparison, not newly selected settings. All matching B16 conditions and all15VOC20 S5 types were freshly checked to equal their corresponding L14-clean configuration exactly. MLMP remains official0.001/10steps/batch2/alpha_cls1/seven prompts.

| Field | L14_coco_cd | L14_coco_object | L14_coco_stuff | L14_pc59 | L14_pc60 | L14_voc20 | L14_voc21 | L14_waterbirds_seg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C_power | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| C_temperature | 2 | 1 | 2 | 1 | 1 | 1 | 1 | 1 |
| adam_eps | 0.000001 | 1e-8 | 0.000001 | 1e-8 | 1e-8 | 1e-8 | 1e-8 | 1e-8 |
| batch_size | 8 | 1 | 8 | 4 | 4 | 1 | 2 | 2 |
| beta1 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| beta2 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| candidate_temperature | 2 | 1 | 2 | 1 | 1 | 1 | 1 | 1 |
| channel_swap | true | true | true | true | true | true | true | true |
| cls_temperature | 2 | 1.4142135623730951 | 2 | 1 | 2 | 2 | 1 | 1 |
| cls_weight | 1 | 0.5 | 0.5 | 0.75 | 0.75 | 2 | 0.75 | 1 |
| connectivity | 8 | 4 | 8 | 4 | 4 | 4 | 4 | 4 |
| counterfactual_contrast | 0 | 0.35 | 0 | 0.35 | 0.35 | 0.35 | 0.35 | 0.35 |
| layer_fraction | 0.5 | 0.75 | 0.375 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| lr | 0.01 | 0.005477225575051661 | 0.005477225575051661 | 0.01 | 0.01 | 0.003 | 0.001 | 0.0017320508075688774 |
| min_confidence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| min_region_area | 8 | 1 | 8 | 1 | 1 | 1 | 1 | 1 |
| pixel_temperature | 0.7071067811865476 | 1 | 1 | 1.4142135623730951 | 1 | 1 | 0.7071067811865476 | 1 |
| prompt_count | 7 | 7 | 7 | 3 | 3 | 7 | 7 | 1 |
| region_weight | 4 | 1 | 4 | 1 | 1 | 1 | 2 | 1 |
| steps | 5 | 20 | 10 | 5 | 5 | 8 | 20 | 15 |
| top_k | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| weight_decay | 0 | 0.0001 | 0 | 0 | 0 | 0 | 0 | 0 |

Parameter values retain full JSON float precision. The adaptation layer fraction and prompt count affect the adaptation objective, not a different final readout; the fraction maps to each backbone's actual depth. Candidate minimum area is measured in model feature-grid cells, not raw pixels. C scores are not calibrated correctness or update-benefit probabilities.

The historical search covered every operational factor through bounded single-factor and joint designs, including adaptation batch1/2/4/8/16; it was not an exhaustive Cartesian search. The final policy selected only from corresponding clean L14 training panels. Earlier B16/corruption search exposure and the PC overlap after reference correction are disclosed and cannot be erased by copying parameters. Baseline and TRACE search budgets are unequal.

Source:[revision-results]/TRACE_official_baselines_20261114-2022-v10/summary/RESULT.json.
Source RESULT SHA256:88cb8a34bb21189808598b4a723300e6ced497a04f267f746e3cbe4ee65d3f46.
No setting or method was changed while exporting this table.
