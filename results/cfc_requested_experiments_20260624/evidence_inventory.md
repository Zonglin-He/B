# Evidence Inventory

- `preexisting_artifact` /home/wwww/hzl/results/cfc_p0_voc20_full_20260623_193022: VOC20 overall, High-CFC subset, correct-prediction harm analysis. Risk: Analysis uses serialized region/image metrics from a single run.
- `newly_run_model_eval` /home/wwww/hzl/results/synthetic_cfc_voc_formal_flip_20260624: Formal context-conflict table with flip rate and object mIoU. Risk: formal_flip_metrics_available
- `preexisting_artifact` /home/wwww/hzl/results/synthetic_cfc_voc_optimized_20260624_0920: Fallback context-conflict harmful-update table if formal run is incomplete. Risk: Previous run lacks formal flip/object-mIoU columns.
- `preexisting_artifact` /home/wwww/hzl/results/cfc_p0_coco_cd_formal_20260624_0920: Strong shortcut dataset diagnostic. Risk: Negative result for current CFC-TTA implementation.
- `newly_run_model_eval` /home/wwww/hzl/results/cfc_p0_waterbirds_formal_20260624_0928: Waterbirds-Seg shortcut benchmark. Risk: Full derived set evaluation is available.
- `preexisting_artifact` /home/wwww/hzl/results/cfctta_voc21_best_non_strict_corruptions_20260623_1320/v21_corruptions_full: VOC21-C per-corruption CFC-TTA detail. Risk: Baseline values are transcribed from MLMP paper Appendix Table 21.
- `preexisting_artifact` /home/wwww/hzl/results/cfctta_default_pc_context_corruptions_20260623_1911: PascalContext-C per-corruption CFC-TTA detail. Risk: Baseline values are transcribed from MLMP paper Appendix Tables 22 and 23.
