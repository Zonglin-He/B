# ACDC Natural-Shift Figure QA

| Check | Status | Evidence |
|---|---|---|
| Core conclusion visible | Pass | Pooled result and all four weather slices are separated into two panels. |
| Source data traceable | Pass | `acdc_overall_plot_data.csv` and `acdc_condition_plot_data.csv`. |
| Sample counts shown | Pass | Fog 100, Night 106, Rain 100, Snow 100. |
| Missing baseline disclosed | Pass | CLIPArTT is explicitly marked OOM. |
| Axis integrity | Pass | Both mIoU axes start at zero. |
| Color and grayscale safety | Pass | Luminance differences plus hatch patterns distinguish methods. |
| Minimum text size | Pass | Plot text is 9 pt or larger. |
| AAAI width | Pass | 6.47 inches, below the 6.5-inch double-column limit. |
| Editable vector text | Pass | SVG contains 40 editable text elements. |
| PDF fonts | Pass | Type-1 Helvetica; no Type 3 or CID/Identity-H fonts. |
| Raster resolution | Pass | PNG export is 2587 x 1011 pixels at 400 dpi. |
| Manual visual overlap check | Pass | Titles, labels, deltas, legend, and OOM annotation do not overlap data. |

The figure reports one-seed results. No error bars are drawn, and the caption must
state that the weather slices are derived from the same full-run per-image
statistics.
