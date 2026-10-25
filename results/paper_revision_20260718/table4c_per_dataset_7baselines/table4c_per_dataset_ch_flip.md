# Per-Dataset C->H and Prediction Flip

C->H and Flip are both conditioned on initially correct matched candidate regions. Flip is the incidence of regions with more than 5% changed pixels, not the mean changed-pixel fraction.

| Method | VOC20 C->H | Flip | PC60 C->H | Flip | COCO-CD C->H | Flip | Waterbirds-Seg C->H | Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No Adapt | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| TENT | 27.0% | 40.8% | 18.9% | 52.3% | 0.0% | 30.6% | 3.1% | 71.2% |
| TPT | 48.2% | 57.0% | 42.2% | 75.2% | 9.1% | 49.6% | 10.5% | 79.6% |
| WATT | 32.1% | 45.0% | 70.7% | 85.4% | 46.0% | 61.2% | 69.8% | 90.6% |
| CLIPArTT | 28.7% | 41.4% | 39.9% | 62.1% | 17.2% | 24.6% | 60.7% | 74.3% |
| MLMP | 12.8% | 27.7% | 12.9% | 34.0% | 0.0% | 10.6% | 14.6% | 51.4% |
| CFC-TTA | 14.1% | 29.7% | 11.3% | 44.5% | 1.7% | 22.1% | 6.9% | 60.7% |
