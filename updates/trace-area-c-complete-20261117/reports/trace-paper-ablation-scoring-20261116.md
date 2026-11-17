# Separate Current-Method CPU Scoring

The calibration fit consumes only sealed GT-free source/candidate/confidence/C
rows after the actual same-invocation GPU success. It fits the existing R02
unsupervised midrank quantile map, never correctness calibration. L14 maps are
copied unchanged to B16. Diagnostic sources are excluded from the map fit.

Each ablation task needs a NEW exactCPU_GT typed START after every prediction
is sealed and its real retained GPU process has successfully exited. The CPU
uses hidden CUDA, one thread,4GiB RAM/1GiB swap and no model/checkpoint. Source
mask identities, vocabulary, official annotation/resize/ignore processing and
integer upstream intersection/union metrics are verified. The input is the
evaluation-disjoint training diagnostic cohort, not a held-out benchmark claim.

Every actual episode must match the frozen three within-eight-source orders,
batch size and step count. All arm contrasts share the same10000 whole-block
draws across arrangements. The region-by-CLS interaction is calculated inside
the same paired draws; independent marginal intervals are never subtracted.
Intervals are pointwise, not multiplicity-adjusted or uncertainty over model
initialization seeds. All negative/null/undefined outcomes remain present.

Waterbirds adds segmentation mIoU for each metadata group and the minimum
group mIoU contrast. It is not bird-classification worst-group accuracy.
Missing-group draws are retained as undefined, not discarded or imputed.
Shared-probe segment timings are not standalone end-to-end efficiency claims.

No diagnostic winner changes the adopted area-C parameters, MLMP settings,
matching B16/S5 parameters, original full26 results or protected old paper.
