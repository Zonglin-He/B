# Completed-paper preflight

Date: 2026-10-29, Beijing time. This is author-side technical verification,
not the independent scientific review and not an acceptance assessment.

## Complete evidence

- A:18/18, B:2/2, C:4/4 cells verified by the final importer, including
  full official evaluation counts and prediction/scoring/source seals.
- A's service exit code is unavailable after transient-unit collection.
  Its explicitly scoped completion receipt certifies complete artifacts and
  process quiescence, not a reconstructed exit code. B/C retain real successful
  terminal states. All failed predecessor roots remain untouched.
- Five complete-split transition and stratification analyses finished and
  sealed. Positive, null and contrary results are included in the manuscript.
- Seeds1/2 are exact deterministic fixed-order replays, not independent
  uncertainty measurements. Stuff has only seed0.

## Verification

`python3 -B -m pytest -q tests/test_evolution* tests/test_two_view* paper_evolution/tools/test_export_evolution.py`

Result:151 passed,2.35 seconds. The focused exporter/allocation/strata subset
also passed33 tests. No model inference or GPU workload was needed.

`python3 -B paper_evolution/tools/build.py`

Result:both combined paper and standalone supplement compile without overfull
boxes, undefined references/citations, or duplicate labels. The combined PDF
has18 pages including references and appendix; the standalone supplement has7.
The current task does not impose a final submission page limit.

Both final PDFs were rendered using Poppler and all pages inspected. New probe
and segmentation figures use actual new-method artifacts and disclose selected
cases. Main experimental floats are flushed before Discussion so they do not
follow the Conclusion. The working draft retains some sparse float pages;
this is not final camera-ready pagination. No clipping or figure-text overlap
was observed. Font listings show embedded Type1 fonts, no Type3/CID fonts.
The main paper retains the anonymous kit notice. The standalone supplement
uses the kit's nocopyright command to avoid overflowing its title-page notice.

## Frozen review packet

`/home/wwww/hzl/results/trace_evolution_review_round02_20261029_0015`

The38-file packet includes rendered PDFs, TeX, tables, figures, source
provenance and a checksum index, excluding internal author outlines/reviews.

- Combined PDF SHA256:
  `1c48c7f1d80d77dae4211dee568dc26d39873ce254a547f1ea1cfdc140f13167`.
- Standalone supplement SHA256:
  `cf12b916c4bb5a7b24fa9514f2951f4d300507629a67322be89ac78781628f4e`.

The independent reviewer received neither any earlier score nor a desired
score. Its report will be retained verbatim; useful weaknesses determine the
next repair, not a request to raise its score.
