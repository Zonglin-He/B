# Pascal Context Reference Correction

The first official-source NoAdapt replay on the previous local PC60 population
gave26.87 for each of three trials, versus paper24.95. This did not establish
paper alignment. Investigation then found a substantive data mismatch, not a
reason to tune MLMP hyperparameters.

## Reference Provenance

MMSeg1.2.2 directs users to the Detail API and trainval_merged.json. The local JSON
was0bytes; its primary Azure URL now returns409/public-access-disabled. The Detail
author's PyTorch-Encoding repository publishes a precomputed validation cache at
https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth and a fixed SHA1
`ebedc94247ec616c57b9a2df15091784826a7b0c`. The complete913687510-byte download
matches that checksum. Author source commit is
`a37b74efc1e9aaaba0158d0103b11d9bceae7b23`.

The author cache is created by Detail.getMask followed by the same sorted60-label
mapping used by the MMSeg converter. Its image membership and masks are used as
the source-defined reference, not selected using any new evaluation score.
This is a published author cache, not a claim that the unavailable raw JSON was
downloaded or reconstructed. Restricted Torch loading allows only PIL.Image.Image
in addition to the default safe types. No arbitrary pickle fallback is used.

## Verified Difference

The previous local population has5105 images, the published author cache5104,
with only2551 shared IDs. There are2554 extra local IDs and2553 missing reference
IDs. Among the2551 common images,2549 raw masks differ:36784444 raw-coordinate
pixels and10189534 nearest224 model-coordinate pixels. Thus a mere count check
or a common scalar NoAdapt check was insufficient.

All5104 reference masks and a sorted validation-ID list were exported separately
without changing the old data tree. Export root:
`PC_OFFICIAL_REFERENCE_EXPORT_20261113-1600-v3`; completion SHA256
`8f103dec10571bf851b37385c08c4ce82bfd4a2806c6a6ae891e574925504569`.
The two earlier reference audits failed on assumptions that the author cache had
5105 masks and that its IDs were a subset of the local IDs; their full original
source, errors, and logs are retained. They are not scientific scores.

## New Evaluation

Preserve the previous26-condition task scope. PC59/PC60 L14 and PC60 B16 now use
the full source-defined5104-ID reference split and reference masks, with unchanged
official transforms, shuffle,3trials, model/factory/readout, and metric code.
All other populations stay unchanged. No existing local-protocol baseline or
partial wrong-split MLMP output is pasted into the new comparison.

The wrong-split V3 controller and active PC60 MLMP child were intentionally stopped
with all bytes/logs retained. No completed run or data file is overwritten. The
new version is independently frozen and run once; no unchanged-byte retry.

TRACE retains all22 previously chosen fields, copied unchanged to the matching
second backbone and corruptions. The old PC calibration partition overlaps the
correct reference validation partition. Explicit per-panel overlap lists are
exported; these TRACE results must not be called untouched held-out PC tests.
No label-based calibration is performed for official MLMP or NoAdapt. No baseline
or TRACE parameters are selected using the corrected validation outcomes.

Source-correct baselines remain a reproduction attempt, not a promise of a target
number. Report all trials and remaining differences honestly. The real MMCV/Detail
extensions and Cython were installed in isolated prefixes; active environments,
old publications, old results, monitors, and automation settings remain unchanged.
