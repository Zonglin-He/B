# Standalone Worker Code Audit

## Verdict

**No concrete P1/P2 correctness or protocol bug was established in the audited worker or its tests.** There are no ordered P1/P2 findings and no source fix is prescribed on the evidence obtained. This is a bounded code audit, not GPU validation, experiment authorization, manuscript review, or an acceptance assessment.

Audit date: 2026-10-29, Asia/Shanghai. Repository root: `/home/wwww/.codex/worktrees/trace-simplification/hzl`.

Audited source snapshots, SHA-256:

- `tools/evolution_followup/standalone.py`: `b31a5d7f40e63380ba0a2e85acfad0530dc6e758237ed8de07182169af29ad5c`
- `tests/test_standalone_probes.py`: `95a372a1b73534e9564e0d090da868aa2fb616dd09a14ac6fef3cbbe6af6523c`

Only the new worker and tests were assessed for findings. The README, inherited method/adapter, input/model contracts, and shared `run_cell.py` were inspected to resolve behavior. No manuscript review contents were read and no paper scores were assigned. No checkpoint was loaded, no download or service was started, and no GPU experiment or experiment configuration was changed. Tiny randomly initialized CPU test models are the only models executed. The only persistent workspace file written by this audit is this report.

## Checks and Evidence

| Area | Result and exact source locations |
| --- | --- |
| Actual 0/1/2-view execution | `standalone.py:117-121,158-170,173-217`: `min` delegates to native two-view evaluation; T/C request only their respective view; Uniform/Confidence request no diagnostic forwards or view helpers. Runtime checks at `412-414` compare requested diagnostic inputs against both wrapper and encoder counts. Tests at `66-155,176-185` execute and count these paths, including exact diagnostic input pixels and empty admission. |
| Missing and inactive supports | `standalone.py:124-137,211-216`: required supports reject missing/nonfinite/out-of-range values; unused supports remain null; zero/EPS-inactive weights are not floored or fabricated. The inherited estimator excludes weights at or below EPS. Tests at `188-214` cover helper-level missing support and inactive Confidence adaptation. |
| Reset, initialization, candidate identity | `standalone.py:380-382,392-411,437-438`: initial trainable/optimizer state is checked; RNG is restored per arm; native `adapt()` resets inside the timed interval; post-arm reset is checked against the initial state. Candidate identity includes batch/image/class, confidence, area/component, mask shape and bits (`140-150`), not measured supports. Frozen parameter version checks are at `415`. Native reset/optimizer behavior is in `trace/engine.py:132-141,194-200` and `adapter.py:77-101`. Tests at `105-121,365-375` check cross-arm candidates and ordering. |
| LayerNorm and loss equivalence | `StandaloneTRACE` does not override initialization, admission, adaptation, or the region entropy objective (`standalone.py:153-217`). Native `trace/engine.py:82-106,149-174` selects visual LayerNorm parameters and applies the inherited Adam update; `trace/objective.py:14-59` computes reliability-weighted mean-region entropy with the inherited normalizer. `test_standalone_probes.py:158-173` compares native-min losses/predictions and method identities and checks the loss formula. No new optimization rule was found. |
| Counters and timing denominators | `standalone.py:219-249` counts disjoint eval/diagnostic/adaptation calls and actual image-encoder inputs using the visual stem hook. Totals at `421-424` do not add diagnostic calls twice. Synchronization/timing at `399-406` includes native reset, admission, requested diagnostics, adaptation, final reconstruction and CPU transfer, but not receipt writes or reset-state hashing. `256-267` excludes episode zero and divides by actual remaining image counts, including a partial final episode; one-episode rates are null. Tests at `75-102,266-277,365-382` cover the accounting identities and denominators. |
| Source/config/checkpoint/panel binding | `standalone.py:270-282,293-315,318-352,367-371,482-491` binds local source/resources, worker/test files, resolved config bytes and values, official checkpoint identity, selection/order seeds, ordered RGB IDs and hashes, vocabulary and label metadata. Startup rebuilds the panel and requires exact equality. Post-run source/runtime/config/RGB/manifest checks occur at `518-522`. Official checkpoint hashing is inherited from `run_cell.py:116-127`; it does not load or download the checkpoint. |
| Label isolation and all-arm seal | `standalone.py:293-306,387` uses the inherited RGB-only loader; preparation enumerates label paths/stat metadata, not label contents. `443-478` requires the complete requested prediction/receipt inventory and verifies its hashes. `score()` calls that verifier at `531`, before label bytes are first opened at `549`. Every requested arm is checked, not just the first scored arm. Tests at `280-299,338-398` guard label-byte reads during prediction and reject a missing later-arm prediction before scoring starts. This is an application boundary, not an OS sandbox. |
| Memory lifetime/streaming | `standalone.py:158-170,201-208` consumes diagnostic logits in bounded chunks and retains scalar supports, not all diagnostic outputs. `385-439` handles one episode at a time; predictions are held for one arm/episode, written and released, not accumulated over the panel. `costs` contains bounded scalar receipts only. Scoring at `542-568` retains one image's GT/predictions and per-class counts. Native-min context caches are per call. This does not establish a measured GPU peak-memory bound. |
| Lock/context startup | `standalone.py:355-364,493-507` acquires the shared nonblocking lock before deterministic runtime configuration, seeding and model construction; `finally` releases it. `run_cell.py:44-55` rejects an already initialized PyTorch CUDA context before configuring runtime. The lock test at `322-335` covers contention, inode receipt and release without CUDA. See the import-time qualification below. |

## Initialization Qualification

The absolute statement in `standalone.py:3` that no CUDA work occurs on import is broader than the inherited import behavior. Importing `TwoViewTRACE` at `standalone.py:39` reaches the vendor CLIP `load()` definition, whose default argument evaluates `torch.cuda.is_available()` (`trace_ovss/models/_vendor_clip/clip.py:94`). That availability query occurs before `gpu_lock()`.

A fresh CPU-only process replaced `torch.cuda.is_available` with a recorder returning false, then imported the worker. It recorded exactly one availability query; `torch.cuda.is_initialized()` remained false. No real availability query or GPU operation was needed for this probe. This establishes an early query, **not** model loading, GPU allocation, or demonstrated context initialization. It is not classified as a P1/P2 blocker on that evidence.

Minimal clarification: distinguish "no model loading, CUDA allocation or PyTorch CUDA-context initialization" from "no CUDA API queries." If the intended lock contract forbids even driver discovery before locking, the inherited eager default must be made lazy in a separately authorized source change; acquire the predict lock before importing that path, and keep prepare/score independent of that eager import. Add a fresh-subprocess import sentinel test rather than relying on `torch.cuda.is_initialized()` alone.

## CPU Validation

All test invocations used `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, one-thread BLAS/OpenMP limits, `python -B`, `-p no:cacheprovider`, and a 120-second timeout. No test source was edited.

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_standalone_probes.py
# 34 passed in 1.77s

python -B -m pytest -q -p no:cacheprovider \
  tests/test_simplified_trace.py tests/test_trace.py tests/test_two_view_entrypoint.py
# 34 passed in 1.48s
```

Environment: Python 3.13.12; PyTorch 2.7.0+cu128. The separate fresh-process environment check reported `cuda_initialized=False`. These are CPU regression results, not real-checkpoint or CUDA evidence.

## Remaining Validation Gaps

1. **Top-level rejection paths are not exercised end to end.** `test_standalone_probes.py:338-361` constructs a synthetic protocol and calls `execute_predictions()`/`seal_predictions()` directly, bypassing `prepare()` and `predict()`. The fixture also omits the real checkpoint/config-file fields. Before GPU execution, add CPU monkeypatch tests proving that changed source/config/checkpoint/panel/digest and a contended lock fail before the model constructor, and that an exception after lock acquisition releases the lock. This is missing regression coverage, not an observed bypass in `predict()`.
2. **Real geometry and complete state trajectories remain only partly covered.** The new integration fixture uses 32x32 images with one patch per image (`44-63,338-344`). Add a CPU multi-patch, padded-valid-mask episode with distinct image IDs and an incomplete final batch. Explicitly compare reset model/optimizer/RNG states across all arm orders. Test all-inactive T/C/min weights and a deliberately absent required runtime support, not only helper validation and zero Confidence. Existing passing tests do not establish these cases independently.
3. **Allocation equivalence should include gradients for every arm.** Native-min equivalence and the scalar loss formula are covered (`158-173`); the new suite does not explicitly compare every arm's inherited loss gradients and updated LayerNorm tensors to an independent weighted-region reference. A focused CPU test would close this gap without a new model or scientific parameter.
4. **GPU properties remain unvalidated.** No conclusion is established about official ViT-L/14 loading/execution, deterministic CUDA-kernel support, multi-patch peak memory, actual CUDA forward counts, synchronization accuracy, throughput, thermal/cache order effects, or real-data accuracy. OOM is intentionally fail-closed (`158-170`; test `217-227`), not automatically rebatched. Any later GPU run still requires separate authorization and its reviewed frozen manifest.
5. **File integrity is not tamper-proof attestation.** The code hashes the artifacts and checks the declared inventory; it does not authenticate seals against an adversary rewriting both files and hashes. The label boundary likewise does not protect against unrelated injected code. Within the documented local-worker trust model, no concrete all-arm-seal-before-label-read violation was found.

Final disposition: no P1/P2 repair is required by this audit. Preserve the distinction between passing bounded CPU checks and the still-unvalidated GPU/protocol integration cases above.
