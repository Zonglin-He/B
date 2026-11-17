# Reboot Recovery and Extension Vocabulary Repair

The actual user requested resuming after shutdown. The current boot is
5443cba565054ea9809275424308b300, begun November14 at14:07 CST. The loaded NVIDIA
module and native nvidia-smi agree on580.178.04. Guarded cuda-run strips the old
580.173.02 compatibility override on this driver; no system packages or global
environment are changed. Actual model RGB gates must still validate fresh CUDA.

Before shutdown V8 completed all three arms on VOC21,PC59,COCO Stuff. Together
with retained V4/V7 cells there are18 successful scoring arms, six complete L14
clean conditions. V8 failed on Waterbirds NoAdapt at13:10 before shutdown: the
extension vocabulary used null for an identity query/class map, but the official
collapse adapter required integer indices. No prediction chunks or score exist
for that failed arm. Old failure bytes, source snapshots and logs are preserved.

Only Waterbirds and COCO-CD, whose exact queries equal their exact classes,
convert null mapping to explicit identity[0,1,2]. This does not change vocabulary,
class IDs, RGB/GT geometry, masks, selected parameters, objective or readout
semantics. Nonidentity or alias lists are never inferred; official benchmark
maps remain unchanged. CPU prepare records the exact representational adapter.

Old transient systemd units disappeared across reboot. They are NOT recreated,
and no current retained success is claimed for them. Reuse instead requires an
explicit historical recovery proof: original STARTED/COMPLETED PID/invocation,
fsynced controller verification showing same actual retained exit0 before reboot,
persistent journal COMPLETE from that same PID/unit/invocation/UID/argv on a
different boot, no FAILED/CANCELLED, and every completed output hash. All proof
bytes are copied and frozen in a new recovery manifest. The historical actual
observation is identified as historical, not manufactured current systemd state.
Missing or inconsistent evidence fails closed. Fresh stages still require the
normal current same-actual-retained-success checks and output seals.

The new finite queue starts at Waterbirds NoAdapt and skips the18 fixed successful
arms. It preserves all selected22 fields, official model/loss/paper parameters,
strict deterministic runtime, shared clean parameters for B16/15S5, original
complete VOC20 RNG predecessor, GT isolation and GPU/CPU caps. It does not restore
an in-memory model or pretend to resume an interrupted batch. First real RGB
gates are rerun on the new process; full successful conditions are not repeated.

Object paper-value discrepancy and PC calibration exposure remain open/disclosed.
The existing Luna max monitor is reused after the new legal stage is registered.
No duplicate monitor, cron, automation or old publication change is made.
