# Sham Priority Insertion

Explicit scheduling update: do not wait for the second backbone. Preserve the
running original worker and insert sham studies at its normal completion.

## Execution Order

1. Let COCO Object w/o Confidence-Source Tracing finish its 5,000-image run.
2. Four-image sham smoke checks on COCO-CD and COCO Object.
3. Fixed 64-image COCO-CD sham control.
4. Fixed 64-image COCO Object sham control.
5. Resume the existing original coordinator in place, without restarting or
   repeating completed original jobs.
6. The original coordinator launches its unchanged second-backbone suite.
7. Only unrelated VOC20 support-distribution collection remains behind backbone.

The same frozen measurement code, formulas, images, mask seed and dataset
configuration are reused from rebuttal_evidence_extensions_20261025_2012.
Only dispatch order changes. Native TRACE, sham residual with native quality,
and all-view sham placement share the same selected candidates and update
settings. All completed/no-effect/unfavorable results are retained.

## Non-Interference and Recovery

Only the original coordinator PID receives SIGSTOP; its live GPU child receives
no stop or termination signal. The stopped coordinator retains the original
queue/GPU locks. The priority worker starts only after normal child completion
and original result validation; required lock reservations are verified before
each sham cell. An unexpected coordinator resume aborts the priority GPU child.

The coordinator resumes on sham success, failure or handled interruption. An
independent guard also handles unexpected priority-controller death: it stops
only a verified priority child, then resumes the same original coordinator.
No model/paper edits, K/steps/lr changes, formal-result overwrites or silent
retries are involved. The old waiting extension controller has been superseded
before it dispatched any GPU cell, so it cannot later repeat these studies.

## Artifacts

- queue_state.json and queue.log: current execution, not anticipated results.
- handoff.json: verified process identities and saved pre-handoff queue states.
- SHAM_RESULTS_READY.json: written only after both formal sham cells complete.
- tables/sham_adaptation.csv: mIoU and actual top-1 correct-to-wrong transitions.
- tables/sham_location_summary.csv: paired location effects and sham coverage.
- cells/...: frozen protocols, all predictions, mask and response evidence.
- original_coordinator_resumed.json: in-place original continuation record.

The copied offline/ directory contains already completed CPU analysis, with
its original provenance intact. It is not a fresh GPU result.
