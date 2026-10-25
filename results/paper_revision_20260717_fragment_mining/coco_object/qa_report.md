# Qualitative Case Figure QA

## Figure Contract
- Core conclusion: CFC-TTA assigns source-aware reliability that preserves stable stuff/background regions, downweights context/fragment false certainty, and leaves interpretable failure cases.
- Figure archetype: error diagnosis / qualitative result composite.
- Evidence chain: cases are selected from live CFC-TTA diagnostic forward passes; metadata is in `qualitative_case_metadata.csv`.
- Reviewer risk: case selection is top-scored and illustrative, not a statistical estimate; aggregate claims should cite the CSV summaries and benchmark tables.

## QA Checklist
- Source data delivered: pass
- Editable SVG delivered: pass
- PNG delivered at 300 dpi setting: pass
- Data invention check: pass, no manual case values were invented.
- Axis truncation: not applicable, image panels have no quantitative axes.
- Color policy: pass, low-saturation overlays plus text labels; not rainbow/jet.
- Negative cases included: pass

## Case Counts
- Context False Certainty: 8
- Fragment False Certainty: 8
- Stuff / Background Preserved: 8
- Failure Case: 8

## Dataset Passes
- coco_object: images=224, patches=224
