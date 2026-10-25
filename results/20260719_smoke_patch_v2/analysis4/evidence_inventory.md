# Evidence Inventory

## Newly Run

- Matched patch artifacts: 2 dataset-method cells.
- Patch AUROC uses a deterministic image-level calibration/held-out split.
- Region quartiles use frozen TRACE reliability and a threshold-free correct-to-wrong event.

## Preexisting Artifact

- Small-object component summaries reaggregated from `/home/wwww/hzl/results/final_protocol_suite_20260630/05_small_object_regression`.
- Existing `mean_component_recall` is renamed mCov; no value is recomputed from predictions.

## Protocol Risk

- The calibrated TRACE risk is an analysis score, not the deployed adaptation weight.
- Patch-level units overlap spatially inside sliding-window images; image-level splitting prevents leakage across calibration and held-out sets.
- AUROC cells with no harmful positives or no negatives are reported as missing, not replaced.
- COCO-Stuff lacks CLIPArTT if that public baseline fails under the matched protocol.

## Frozen Selection

```json
{
  "threshold": 0.05,
  "fusion_weight": 0.25,
  "qualifying_datasets": 0,
  "worst_dataset_margin": -Infinity,
  "macro_margin": -Infinity,
  "macro_AUROC": -Infinity,
  "evidence_type": "newly_run_analysis",
  "calibration_fraction": 0.25,
  "heldout_fraction": 0.75,
  "split_unit": "image_path",
  "calibration_salt": "trace-harm-calibration-v1",
  "split_hash": "sha1(image_path|trace-harm-calibration-v1) mod 4; calibration=0",
  "scope": "candidate-covered patches that are correct before adaptation",
  "selection_policy": "maximize qualifying dataset count, then worst-dataset AUROC margin, macro margin, and macro AUROC on TRACE calibration patches",
  "min_positives": 1,
  "min_prevalence": 0.0,
  "risk_fusion": "((1-confidence_margin) + lambda*(1-TRACE_reliability))/(1+lambda)",
  "deployed_method_changed": false
}
```
