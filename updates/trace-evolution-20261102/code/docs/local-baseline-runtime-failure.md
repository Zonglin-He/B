# Local Baseline Runtime Failure

## Observed Terminal State

The exact first baseline attempt is preserved, not resumed or edited:

- Suite: `/home/wwww/hzl/results/trace_local_baselines_20261030_0008`.
- Prepared SHA256: `11803f984bdd34fb0b78d925bbcf252f61485be3bc4d43f3eb3eb9fbad7d4515`.
- Service: `trace-local-baselines-20261030-0008.service`.
- Controller PID: 690901; invocation: `9acee57420c1427fafdad156e53a4f6d`.
- Started 2026-10-30 00:16:08 Beijing; stopped at 00:19:26.
- Actual retained state: loaded, failed/failed, MainPID 0, ExecMainPID 690901,
  ExecMainCode 1, ExecMainStatus 1, Result exit-code. This is not an OOM.
- Queue phase: failed; retry_allowed false.

The source/artifact audit and 753 passing CPU tests did not establish a real
native import or GPU gate pass. Preserve that audit as the historical admission
record; this document adds the subsequent actual runtime failure.

## Direct Cause And Scope

The first `voc20.mlmp` gate child exited with return code 1. Its traceback ends:

```text
baseline_source/adapt/__init__.py:9: from .cfctta import CFCTTA
baseline_source/adapt/cfctta.py:13: from cfc_tta.method import ...
ModuleNotFoundError: No module named 'cfc_tta'
```

`build_method` imports `adapt.mlmp` through Python's ordinary package importer.
Before that submodule is returned, the archived package initializer eagerly
imports all its registered methods, including the unrelated CFCTTA extension.
That extension assumes a project-root `cfc_tta` dependency which is outside the
authorized baseline snapshot. The MLMP class constructor was not reached.
There is no measured native loss/gradient/parameter mismatch in this attempt.
Do not describe this as a numerical-equivalence failure or invent such deltas.

Passed native gates: 0/18. Formal full predictions: 0/18. New scores: 0/18.
The gate failure occurs after RGB-only input checks; no full prediction stage
or GT scoring stage began. The shared lock and deterministic-runtime setup
precede the failing import, so do not equate that setup with model execution.

Small primary evidence, all in the immutable failed suite:

- `EXECUTION_STARTED.json` and `queue_state.json`.
- `logs/voc20.mlmp.gate.json` and `logs/voc20.mlmp.gate.log`.
- `gates/voc20.mlmp.STARTED.json` and `gates/voc20.mlmp/GATE_FAILED.json`.

The log receipt binds log SHA256
`0354f5e2f7b9b91a5f1b26b00ed6d8fc2085508e481dd017f0ff838b6a4df02e`
and gate manifest SHA256
`638ff10946df9f005dab1070d537ceaecef17500ae82933d6060e98402a0fea1`.

## Completed Dependency And Unchanged Evidence

`DEPENDENCY_COMPLETED.json` was written before the gate child started. It
certifies the source-bound 2055 mechanism suite, ten canonical outcomes and
the genuine retained successful service terminal. Do not repeat that large
payload certification merely to monitor this failed baseline attempt.
The receipt SHA256 is
`23dbd7be5a46b205194f68a71d332c7592f60cbec874d87e592d12edb6c09496`.

No new all18 gate barrier, full-prediction barrier or completed-suite seal
exists. Original TRACE calibration/full results and all old positive, negative
and null evidence remain unchanged. No manuscript or paper-review update is
justified by this engineering failure.

## Human Decision Required

One asynchronous choice has been sent: allow a minimal native-method import
isolation repair, CPU regressions of the actual archived package import,
independent code/artifact audit and a newly frozen single-attempt queue; or
limit work to CPU diagnosis. Until an actual human answer permits the former,
do not perform additional GPU work, restart the unit, rerun a gate, or choose
a new directory to bypass the stopped attempt. A heartbeat is not approval.

A potential repair must not restore the unrelated CFCTTA project, mutate old
archive/frozen files, change baseline math/configuration, or loosen native
loss/gradient/parameter tolerance 1e-5 and zero changed labels. The complete
authorized scope remains MLMP on eight datasets plus five other baselines on
CD/Waterbirds: eighteen full cells. Scientific completion is still pending.

## CPU Import Diagnosis, 2026-10-30 04:18 Beijing

A separate CPU-only process successfully imported the five pinned archived
method files directly with `importlib.util.spec_from_file_location`: MLMP,
TENT, TPT, WATT and CLIPArTT. Each method-file hash was checked against the
failed suite's prepared native inventory before import. CUDA was hidden;
`torch.cuda._lazy_init` and `torch.nn.Module._call_impl` were trapped to raise.
No native class was instantiated, no model forward ran, and CUDA remained
uninitialized. Neither `adapt`'s aggregator nor CFCTTA/cfc_tta was imported.

This narrows the observed failure to the eager package initializer rather
than a missing dependency required to import those five method files. It is
not a worker repair, completed dependency-isolation implementation, real gate,
constructor/forward test or GPU admission. It does not establish that no
additional runtime issue remains. No source, manifest, result or marker in the
failed suite was modified, and no new queue or service was prepared or started.
Do not repeat this probe without a new diagnostic question; actual approval
for an audited new controlled attempt remains outstanding.
