# Per-Dataset Correct-to-Harm

C->H is conditioned on each method's initially correct matched candidate regions.

| Method | VOC20 C->H | PC60 C->H | COCO-CD C->H | Waterbirds-Seg C->H |
|---|---:|---:|---:|---:|
| No Adapt | 0.0% | 0.0% | 0.0% | 0.0% |
| TENT | 27.0% | 18.9% | 0.0% | 3.1% |
| TPT | 48.2% | 42.2% | 9.1% | 10.5% |
| WATT | 32.1% | 70.7% | 46.0% | 69.8% |
| CLIPArTT | 28.7% | 39.9% | 17.2% | 60.7% |
| MLMP | 12.8% | 12.9% | 0.0% | 14.6% |
| CFC-TTA | 14.1% | 11.3% | 1.7% | 6.9% |
