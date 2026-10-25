# Candidate-oracle decomposition with baseline anchors

The baseline rows are reference anchors on the same deterministic image prefix. They are not additional cells of the 2x2 factorial: the factorial disables CLS, while complete MLMP and CFC-TTA retain their global term.

| Dataset | NoAdapt | TENT | MLMP | CFC-TTA | Pred+uniform | Pred+CFC weight | Oracle+uniform | Oracle+CFC weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| voc20 | 75.79 | 70.89 | 81.96 | 83.49 | 77.84 | 78.71 | 79.99 | 78.04 |
| pc60 | 27.47 | 25.68 | 28.40 | 28.83 | 29.88 | 30.29 | 29.55 | 29.88 |
| coco_object | 27.67 | 25.10 | 28.72 | 28.96 | 26.39 | 25.20 | 25.42 | 25.93 |
| coco_stuff | 18.90 | 17.72 | 19.33 | 20.21 | 17.73 | 17.85 | 16.67 | 17.79 |
