# Official Baseline Repeatability Repair

V7 completed ten scoring arms: all three arms on PC60,Object,VOC20, and VOC21
NoAdapt. On November14 at05:24:10 CST its VOC21 MLMP prediction stage failed
the four-RGB repeated-trajectory check; the dispatcher failed at05:24:12. Luna
reported this at05:25. No VOC21 MLMP prediction chunks or score were completed.
This is not OOM and is not a scientific performance result. Failed source,
output bytes, identities and full journal remain untouched.

V7 configured strict deterministic execution only for TRACE. Baseline processes
still used the upstream seed-only routine. The new version extends strict
deterministic algorithms/CUBLAS_WORKSPACE_CONFIG=:4096:8 to every newly executed
GPU arm before CUDA initialization. This is an explicitly declared runtime
extension, NOT an unchanged-upstream-default runtime claim. The actual official
model, loss, optimizer, learning rate,batch2,steps10,seven prompts and readout
are unchanged. There is no parameter search or paper-score-based selection.
The actual VOC21 first-four gate must still pass1e-5/zero labels. Failures now
include arm,batch and numerical differences; no threshold is relaxed.

The ten successful prior scoring arms are fixed in advance for reuse by their
original absolute paths. New CPU prepare verifies true retained success for both
prediction and score, all output hashes, and independently recomputes every trial
point from compact source I/U without decoding GT. Their original runtime
metadata is preserved, not relabeled as strict execution. Old results are not
overwritten and successful tests are not repeated to select favorable trials.
The first S5 condition carries the complete V7 clean VOC20 RNG stream; subsequent
S5 conditions carry the immediately preceding condition as before.

Current file-reported results before fresh verification: PC60 MLMP27.9833,
TRACE28.7133; Object MLMP30.42,TRACE32.44; VOC20 MLMP83.8033,TRACE86.12.
Object baseline values remain above paper by1.23pp(NoAdapt) and1.58pp(MLMP),
so full paper parity is NOT established for Object. This discrepancy is retained
as open and must not be hidden or tuned away. PC TRACE calibration overlaps the
correct reference validation split: development comparison, not untouched test.

The finite queue starts at the uncompleted VOC21 MLMP arm, followed by its TRACE
arm and remaining clean,B16,and15S5 conditions. Monitoring remains read-only.
The main agent owns repair and scientific verification. No new cron or automation
is created and the original trace automation remains PAUSED.
