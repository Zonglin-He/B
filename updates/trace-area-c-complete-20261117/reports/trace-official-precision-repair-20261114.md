# TRACE Identity Arithmetic Repair

V5 repaired the genuine cross-batch reference-capture error but the real RGB gate
still failed on its first batch. It was insufficient and is not reported as a
successful repair. Its original source snapshot, failure bytes and full journals
are retained. The difference was loss0.0332624912, logits19.515625,
parameters0.0102033652 and1889 decoded pixels after ten steps.

A separate four-RGB diagnosis ran once with zero GT access. Initial parameters,
candidate masks, C maps, first training logits, first CLS logits and first loss
all matched exactly. The first two-image trajectory diverged during subsequent
FP16 updates; the second matched exactly. This isolates the discrepancy beyond
the candidate/initial-forward stages, not a benchmark-score explanation.

The configurable wrapper previously inserted division by unit temperatures,
multiplication by unit coefficients, and unit powers. Their real-number formula
is unchanged, but the extra backward nodes can alter the order/rounding of
accumulated FP16 gradients. The new wrapper omits these operations exactly when
their parameters are1, retaining every nonunit setting's formula. Model dtype,
22 selected fields, candidates, readout, optimizer and official baseline code are
unchanged. The real first-four check still compares against NativeTraceRevision
for ten steps at1e-5 and exactly zero changed labels. No result may be claimed
unless this actual check passes; the diagnosis itself is not a scientific gate.

The fresh finite queue reuses only the two original successful corrected PC60
baseline stages, not either failed TRACE attempt. All remaining conditions and
arms retain their order. PC TRACE calibration overlap and historical exposure
remain disclosed. No new test-based selection, monitoring or automation is added.
