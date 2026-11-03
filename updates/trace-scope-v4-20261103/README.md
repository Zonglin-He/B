# TRACE SCOPE_v4 Follow-Up, 2026-11-03

This is a distinct, bounded follow-up to the November 2 archive. It brings
together the completed fixed studies, the revised manuscript, original new
review reports and a current first-party implementation snapshot. Historical
releases and failed or superseded artifacts are not overwritten.

## Manuscript And Review

- [Revised main paper](paper/main.pdf)
- [Revised supplement](paper/supplement.pdf)
- [Original independent full review](reviews/manuscript_review_draft_v001.md)
- [Same-reviewer revision verification](reviews/manuscript_review_draft_v002_closure.md)
- [Author post-review disposition](publication/scope-v4-paper-review-disposition-20261103.md)

The full review required revision. The subsequent narrow verification records
R2, R3 and R4 as corrected and **R1 as PARTIAL**, not passed. Historical local
COCO-CD and Waterbirds mask/subset construction remains unrecovered. The
manuscript documents the recoverable protocol, nested populations and exposure
history, but does not claim a fully reconstructible public-benchmark replication.
Neither report supplies an acceptance rating or guarantees publication.

The original v001 and v002 whole manuscript archives, including their original
freeze manifests, are preserved byte-for-byte inside the local delivery packet.
Raw reviews and the post-review disposition are separate additions; they do not
retroactively change the reviewed manuscript bytes.

## Scientific Interpretation

- **M01 risk:** the independent candidate-risk interpretation is not established.
  The prespecified GT-quality matching reverses the unadjusted direction, while
  its bootstrap intervals are undefined. Undefined draws are retained, not
  dropped or repaired with post-result bins.
- **M02 spatial control:** Object supports the spatial contrast; both PC60
  spatial intervals cross zero. This is not candidate-level causal evidence.
- **M03 legacy components:** the old multi-view component construction exceeds
  single-C accuracy on Object, with the reverse direction on PC60. This is a
  component comparison in the current engine, not reproduction of the entire
  published historical system.
- **M04 ranking and distribution:** the crossed contrasts and interactions are
  conditional on the paired configurations. They do not establish universally
  superior ranking, magnitude calibration or independent risk allocation.
- **Generalization:** the paired B/16 C-versus-Confidence findings do not imply
  dominance over NoAdapt. The specified SCLIP transfer is negative; its local
  multilayer readout is not the complete published SCLIP pipeline.
- **G03 severity:** the five fixed corruption types at severities 1 and 3 have
  positive pointwise C-versus-Confidence intervals. At severity 5, only motion
  blur has a positive interval; the other four cross zero, including negative
  Gaussian-noise and contrast point estimates. Severity-5 results are exact
  reuse, not newly inferred independent replications.
- **V01 compute allocation:** both prespecified near-time comparisons are NULL
  because no grid point meets the time tolerance. Same-step and nearest-time
  descriptive contrasts are not equal-budget benefits. There is no interpolation
  or outcome-based grid selection.
- **Cached groups and cases:** source-group and class-subset readouts are not
  candidate-local causal effects. Sparse PC60 same-order coverage remains
  insufficient. The four fixed M12 examples are descriptive, not representative
  evidence or attention/perturbation/causal demonstrations; the weak example is
  retained.

The confidence intervals are paired whole-original-episode, pointwise intervals
with no multiplicity adjustment. Historical evaluation exposure and Waterbirds
development overlap remain explicit. Native PIL-nearest and G-study
OpenCV-nearest geometry differ. These studies are not an untouched, independently
tuned, multiseed comparison, and do not prove universal shift robustness.

## Delivery Boundary

The bounded packet contains the two original manuscript archives, explicit raw
reviews and disposition, allowlisted sealed summary/receipt metadata, result
documents, notices and current first-party source. The implementation snapshot
is not a claim that all frozen experimental runtimes or dependencies are
included. Source presence does not prove execution of every included test or
optional method. The six scientific reference-mutation demonstrations remain
NOT RUN by this delivery task.

No raw dataset RGB, ground-truth masks, prediction arrays, weights, per-image
count streams, external dependency trees, credentials or old release payloads
are newly exported. Four previously reviewed derived M12 composites are included
only as manuscript assets. Dataset and model acquisition/construction are not
made turnkey by this package.

Local packaging verification, exact input admission and remote release-byte
verification are distinct records. A CODE review is not input admission; a
local completion receipt is not proof of remote publication. The release
manifest and final publication receipt identify the actually delivered bytes.

## Download And Verify

The [distinct release](https://github.com/Zonglin-He/B/releases/tag/trace-scope-v4-20261103)
provides `delivery.tar.gz`, both revised PDFs, the
[publication manifest](PUBLICATION_MANIFEST.json) and [asset checksums](SHA256SUMS).
The 8,926,222-byte delivery archive has SHA-256
`43c6693e580fc6d609fd8379c1eaaa1a38bb03e43b5a536c7b7455e4e77aa0a4`.
Its 317 members include both complete original manuscript archives.

```bash
gh release download trace-scope-v4-20261103 --repo Zonglin-He/B \
  --dir scope-v4-assets
cd scope-v4-assets
sha256sum -c SHA256SUMS
```

Browsable [current source](code/), [sealed summaries](evidence/),
[result notes](results/), [exact input admission](audit/input_reviews/scope-v4-delivery-plan-independent-002-20261103.md),
[local terminal receipt](audit/BUILD_TERMINAL.json) and
[prior DENIED admission](audit/input_reviews/scope-v4-delivery-plan-independent-20261103.md)
are included in this update. The original DENIED plan and earlier failed code
versions remain in [audit history](audit/); later successful verification does
not relabel those artifacts as successful.

The local build used one manually dispatched CPU service with 4 GiB RAM,
512 MiB swap and one CPU quota. Its process identity and effective limits were
verified before releasing the build gate; it genuinely exited successfully.
These resource limits and systemd accounting are operational records, not
scientific runtime or efficiency measurements. No GPU or model was imported.

## Historical Archive

The [November 2 update](../trace-evolution-20261102/README.md) and older releases
retain their original scope and review state. They must not be interpreted as
the revised SCOPE_v4 manuscript or as evidence that optional experiments were
completed. This follow-up does not replace them.
