# Waterbirds Development Tuning And Eight-Dataset Full Continuation

Waterbirds uses the user's explicitly authorized development-overlap protocol.
Its256 calibration examples(seed2027) are from existingtrain2458 and are also
inside unchangedfull5322. Results are NOT independent test performance.
No synthetic calibration images or modified original dataset files are used.

Prepared SHA256:4f3e54226f99230af48b2105372300d06f31651515324b3ab99dbe138ff3cfa3.
Code commit34b58a5;431 joint CPU tests passed,independent code audit closed.
Service:trace-waterbirds-full-20261029-0817.service.
Dependency:trace-single-probe-tuning-20261029-0748.service and its seven
canonical calibration selections. This continuation waits without GPU work.

After upstream retained-success and full selection verification,import exact
seven winners without repeating their searches. Search ONLY Waterbirds using
the same maximum80-trial B/lr/steps/K grid. All eight configs freeze before
seed0full. Each full uses NoAdapt,Uniform,Confidence,and TRACE C:32 rows.
All eight full predictions seal before any full scoring.

Original full counts:VOC20/21 each1449,PC59/60 each5105,Object/Stuff each5000,
COCO-CD500,Waterbirds5322. Model/math/geometry are unchanged from0748.
Full result CSV labelsWaterbirds development_overlap and256overlappingIDs.
All negative/null results remain;no test-selected extensions or silent retry.

0748 completion is not completion of this new continuation. Check actual
service terminal plus canonical32results and all seals. Both services use
12GiBRAM/1GiBswap,RemainAfterExityes,Restartno,OOMPolicystop and shared GPU lock.
