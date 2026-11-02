# TRACE Paper Evolution

This directory owns the new paper sources and manuscript-local generated artifacts only. Original desktop sources, external code, results, and experiments are not edited. No model loading, GPU work, or new MLMP baseline run is needed to build this manuscript.

## Files

- `main.tex`: main paper with included appendix for whole-document reading.
- `appendix.tex`: shared included appendix, intentionally not a standalone LaTeX document.
- `supplement.tex`: standalone supplementary document wrapper; reads main auxiliary references when available.
- `preamble.tex`: shared AAAI preamble and TRACE macro. AAAI style chooses its bibliography style internally; do not add another `bibliographystyle` command.
- `mini_outline.md`: narrative and paragraph-role/reverse-outline map.
- `claim_evidence.md`: internal author claim/evidence ledger, unresolved gates, and consistency checklist. No independent review score.
- `reported_baselines.json`: MLMP main Table 5 values with source provenance.
- `tools/export_tuned_full.py`: CPU-only importer for the eight completed, calibration-tuned full-split evaluations.
- `generated/tuned_full_*.tex`: current weighting comparisons, separately attributed references, selected configurations, and actual computation.
- `generated/tuned_full_provenance.json`: current verified result sources and verification scope.
- `figure/tuned/image_full_*`: current-configuration teaser, pipeline, allocation comparison, segmentation examples, and provenance. Old two-view and old-configuration figures are not current evidence.
- `tools/freeze_tuned_review.py`: freezes the compiled current manuscript and current assets without supplying a target rating or old reviews.
- `aaai2027.sty`, `aaai2027.bst`, `references.bib`: copied dependencies from the authorized local paper draft, originals unchanged.

## Build

Run from this directory. The exporter does not require PyTorch and performs no inference.

```bash
CUDA_VISIBLE_DEVICES='' python3 -B tools/export_tuned_full.py
cd build
TEXINPUTS=.:..: pdflatex -interaction=nonstopmode -halt-on-error ../main.tex
BIBINPUTS=.:..: BSTINPUTS=.:..: bibtex main
TEXINPUTS=.:..: pdflatex -interaction=nonstopmode -halt-on-error ../main.tex
TEXINPUTS=.:..: pdflatex -interaction=nonstopmode -halt-on-error ../main.tex
TEXINPUTS=.:..: pdflatex -interaction=nonstopmode -halt-on-error ../supplement.tex
BIBINPUTS=.:..: BSTINPUTS=.:..: bibtex supplement
TEXINPUTS=.:..: pdflatex -interaction=nonstopmode -halt-on-error ../supplement.tex
TEXINPUTS=.:..: pdflatex -interaction=nonstopmode -halt-on-error ../supplement.tex
```

Build the main document first to provide cross-document section/table references for the supplement. Compile from `build/` with its local directory first in TEXINPUTS so older root-level auxiliary files cannot shadow the new bibliography. Neither compilation requires external results after the generated tables have been created. The temporary MLMP text extraction is not a build input; the local PDF is used by the table exporter only for its provenance hash. Main and supplementary PDF outputs and compilation byproducts remain under `build/`. The coordinating author owns the final render inspection.

## Scope

The current paper describes ImageBalancedTRACE with a single global context probe C. Eight completed full-split comparisons use configurations selected on fixed256 training-source calibration panels. Historical development exposure remains; Waterbirds additionally has256 calibration/full overlaps and is explicitly development evaluation. All32 full outcomes and455 calibration configurations are retained. Historical two-view results and earlier six-dataset configurations are not substituted for the tuned results. See `claim_evidence.md` for evidence boundaries and `mini_outline.md` for the scientific narrative. This revision does not inherit the frozen round04 manuscript's review score.
