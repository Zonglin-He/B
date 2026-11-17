# Reboot Report Binding Repair

V9 collected37 same-invocation journal/controller proof bindings and verified all
18 completed scoring arms, then its CPU prepare failed while writing the final
report: PreparationOutput.root is the inputs child, but REBOOT_RECOVERY.json is
at the pipeline root. This is a report-path error, not a failed scientific result.
No new GPU model or GT decode ran. Its source snapshot, bytes and full logs stay.

The new wrapper changes only this report binding to the explicitly active root,
adds a regression test and uses a fresh root/unit once. All historical proofs,
identity vocabulary conversion, scientific method/settings, strict real-RGB gate
and GPU/CPU limits remain unchanged. No old transient unit is recreated.
