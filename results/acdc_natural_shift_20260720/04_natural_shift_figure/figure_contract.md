# Figure Contract

"
        "## Core Conclusion
"
        "On the official ACDC validation split, TRACE is marginally above MLMP overall "
        "and improves mIoU in each natural weather condition under a fixed Cityscapes-transfer configuration.

"
        "## Evidence Hierarchy
"
        "- Primary: newly run ACDC pooled and per-condition results in the parent result directory.
"
        "- Supporting: official ACDC validation image counts from the extracted manifest.
"
        "- Context: CLIPArTT was attempted at the frozen common resolution and is reported as OOM.

"
        "## Chart Archetype
"
        "- Type: robustness and generalization slice.
"
        "- Layout: two-panel quantitative comparison.

"
        "## Panel Mapping
"
        "- (a) pooled ACDC comparison across all locally runnable baselines.
"
        "- (b) fog/night/rain/snow slices for No Adapt, MLMP, and TRACE.

"
        "## Reviewer Risk
"
        "The absolute TRACE--MLMP gap is small; zero-based axes and per-weather labels avoid magnification. "
        "The experiment is one seed with Cityscapes-selected TRACE parameters, so it supports representative "
        "natural-shift transfer rather than a broad robustness or efficiency claim.

"
        "## Export Bundle
"
        "- Script: experiments/plot_acdc_natural_shift.py
"
        "- Data: acdc_overall_plot_data.csv and acdc_condition_plot_data.csv
"
        "- Vector: acdc_natural_shift.svg
"
        