# Queue-recovery code review

This is an engineering audit, not manuscript review round02. The original
manuscript review score and report remain unchanged.

## Failure and completion evidence

The original main suite completed all18 cells at21:15 Beijing on2026-10-28.
Its transient systemd unit was garbage-collected. Both dependent controllers
failed closed before starting experimental cells because service identity
was unavailable. No successful exit is inferred from a not-found unit's
default Result=success. All failed directories remain untouched.

`tools/evolution_completion.py` deeply reverified all18 cells using their
frozen verifier, including complete prediction/scoring seals, dataset count
aggregation, input hashes and sources, then checked runtime process absence.
The explicit receipt is
`/home/wwww/hzl/results/trace_evolution_recovery_audit_20261028_2215/completion.json`.
Its basis is artifact completion plus quiescence; process exit status is
explicitly uncertified. A receipt is opt-in and hashed into a new manifest,
never an automatic fallback for failed service lookup.

## Independent findings and repairs

Reviewer agent: `01a182db-b01f-7581-875a-ad7115c72575`.
No P1 findings. The initial60-test review found two P2s:

1. Absolute command matching missed relative Python scripts launched from
   runtime cwd. Fixed by checking cwd for same-user Python/relative .py
   commands. Access failures for relevant processes fail closed; unrelated
   nondumpable daemons are not treated as Python workers.
2. Replay receipt admission checked receipt-to-parent binding but omitted the
   downstream frozen parent manifest hash. Fixed by testing that hash before
   accepting the explicit receipt.

Live verification also exposed a legitimate dependency interaction: replay B
references completed A/runtime as its package root. C must wait for B before
rechecking A quiescence. Fixed the dependency ordering without weakening
quiescence admission. An intermediate C controller failed before any cell;
it is preserved, and C recovery2 uses a fresh snapshot. Preparation binds
receipt metadata without claiming quiescence; execution still requires it.

## Verification and limits

Combined targeted CPU tests passed110 cases after all repairs. Independent
final review reran93 dependency/completion tests with no failures or skips,
and found no remaining P1/P2 in this bounded code scope. Real18-cell audit and subsequent receipt
validation passed. New B's experimental worker is byte-identical to old B;
new C's method sources and VOC20 input panels match old C. Both new services
retain real terminal state with RemainAfterExit=yes and have Restart=no.

The original failed process exit cannot be reconstructed from default unit
properties. Artifact completeness is certified instead; it is not silently
relabelled as historical service success. No scientific formula, settings,
threshold, baseline source or experimental outcome was changed by recovery.
