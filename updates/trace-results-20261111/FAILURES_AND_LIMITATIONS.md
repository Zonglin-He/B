# Failures, Corrections And Boundaries

- Original CPU scoring preparation failed on checkpoint schema comparison; failed bytes were retained.
- B16 direct GPU first launch failed CUDA804 before predictions. A service-local existing driver compatibility wrapper was used for the changed successful run; no global driver repair claimed.
- The earlier B16 PIL-nearest scoring is retained as superseded; corrected CV2-nearest scores are canonical. No model rerun or favorable-trial selection occurred.
- S5 remains unscored in this snapshot. Prediction completion is not mIoU.
- TRACE image_C is named explicitly; no highest-scoring variant is selected. Native MLMP uses its original loss/prompts/readout.
- Full native-system settings, budgets and episode grouping differ. The nine point comparisons have no direct TRACE-minus-MLMP paired interval.
- R00 uses shared whole-original-episode draws, and a15-type arithmetic macro; undefined outcomes remain.
- R03 preserves all16 models and shared10000 whole-episode OOF draws; it is not causal or independent-run evidence.
- Historical selection/development exposure, geometry differences, R1 PARTIAL, negative SCLIP transfer and both old V01 NULL comparisons remain.
- No new Windows manuscript was provided or reviewed. This release contains results, not a new manuscript.
