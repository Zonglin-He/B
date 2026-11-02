# Single-Probe Continuation: Independent CPU Code Audit

## C1 Closure Recheck: 2026-10-29

**C1: CLOSED for the source identities below.** No remaining demonstrated
P1/P2 from this continuation audit. This recheck covered only C1's path
normalization and its regression; the earlier report below is historical and
its open disposition is superseded. No broader re-audit or launch certification.

`tools/evolution_tuning/continuation.py:40-44` now expands user paths and resolves
`parent_suite`, `python`, `source_root`, and `data_root` strictly to absolute
canonical paths. The future output path is expanded and made absolute without
requiring it to exist. `prepare():51` calls normalization before parent
verification or suite creation. The persisted parent/Python strings therefore
come from normalized arguments and no longer change meaning in frozen-runtime
cwd. Missing input paths also fail before the new suite is created.

The new CPU fixture supplies relative parent/Python/source/data/output paths,
checks the normalized identities, changes cwd, and confirms the parent/Python
paths still resolve to the intended existing objects. Static inspection verifies
the helper is invoked before the real preparation side effects; the regression
itself calls the helper, not real input preparation.

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest -q -p no:cacheprovider tests/test_evolution_tuning_continuation.py -k relative_parent_and_python
1 passed, 16 deselected in 0.02s
```

```text
tools/evolution_tuning/continuation.py
48c4a0bd5c60c4009bd9279d27c8fd1e05b481971ddc55a7e7d4ca8dc3be59c3
tests/test_evolution_tuning_continuation.py
45dd88a06f43f2ae1567f64de95273c295d8be79e5838ec19ee858ed55b9f7dd
```

Only this audit document was edited. No GPU, real preparation, service/parent
operation, history scan, implementation/test change, commit or delegation.
The reported `0816` preparation was not inspected, changed or launched; its
CPU-only/unstarted retention and later fresh snapshot are the main user's
disposition, not execution evidence from this recheck.

## Archived Initial Audit

Audit completed 2026-10-29 against the source identities below. This is a bounded
code/protocol integration audit, not a method/manuscript review or acceptance
score. **One open P2; no P1 demonstrated.** Only this document was edited.

## Finding

### C1 [P2] Accepted Relative Dependency Paths Break After The Required Cwd Change

Locations: `tools/evolution_tuning/continuation.py:44`, `:55`, `:71`, `:78`,
`:110-115`.

`prepare()` accepts a relative `--parent-suite`: `queue.verify()` makes its own
local absolute copy, and the subsequent input read works relative to the
preparation cwd. The continuation nevertheless persists
`str(args.parent_suite)` unchanged. `run()` must execute from the new frozen
runtime, so `wait_for_parent()` resolves that stored relative path against a
different cwd. It raises `FileNotFoundError` instead of waiting for/importing
the intended parent, after writing the non-retryable execution-start marker.
The stored custom `--python` path has the same normalization omission at `:71`.

Concrete counterexample: from a directory containing `parent/`, prepare with
`--parent-suite parent --output /absolute/fresh`, then execute the prescribed
run from `/absolute/fresh/runtime`. Preparation stores `parent`; waiting tries
`/absolute/fresh/runtime/parent/queue_state.json`, not the original parent's
state. This is an accepted CLI input, not malformed parent evidence.

CPU reproduction called actual `continuation.prepare()` in a temporary fixture.
Input preparation, snapshot copying, checkpoint identity and parent verification
were replaced by synthetic providers, with real fixture input/state/contract
files. Preparation succeeded and stored `parent`. After changing cwd to its
fixture runtime, the actual stored-path JSON lookup failed as above. No real
preparation, parent suite, system service or model was accessed.

Feasible fix: normalize the parent suite to a canonical absolute path before
verification, all reads and persistence; likewise resolve the configured Python
executable before freezing it. Alternatively reject relative paths explicitly
before creating the suite. Add a prepare-to-runtime-cwd regression. Existing
absolute-path invocations and the default absolute Python are not affected;
using absolute arguments is a bounded operational workaround, not a code fix.

## Verified Behavior

- **Running parent remains untouched.** Continuation code reads the parent;
  search, imported selections and new full outputs are written under the new
  suite. The audited `queue.py` and `worker.py` hashes equal the earlier closed
  audit identities. No old-queue or production-method edit was made by this
  auditor, and the running `0748` suite/service was not inspected or operated.
- **Parent completion gate:** `parent_ready():97-107` requires a retained,
  loaded, active/exited service with MainPID 0, Result success, ExecMainStatus 0,
  and `awaiting_all_eight_calibrations`. A running active service with positive
  PID waits; missing/failed/bad-terminal cases reject. No GPU dispatch occurs
  in `run()` before the wait and parent import return.
- **Parent source/settings and input equivalence:** preparation checks the
  seven calibration identities and all eight full population identities before
  freezing new inputs. The comparison includes ordered rows/IDs, vocabulary,
  mapping, zero-label policy, full indicator and input identity, excluding
  deliberately rebound provenance. Existing parent method/worker/search bytes,
  checkpoint identity and inherited settings must match in `unchanged_method()`.
- **Seven selections are imported, not retuned.** `read_parent_selections()`
  verifies the canonical terminal selection hashes, all seven gate hashes,
  CSV/inventory counts and unique keys, exact trial locations/manifests, and
  bound completed results/scored seals. It validates ineligible receipts,
  reconstructs completed scores/times, and recomputes the global winner using
  the existing fixed tolerance/time/config tuple rule. The chosen config and
  trial must match the parent's selection. Imported seals retain parent
  identities and explicitly mark reuse without search.
- **Only Waterbirds searches:** after import, `run():252-261` gates Waterbirds
  and calls `queue.tune()` exactly for `waterbirds_seg`. It does not call tune
  on the seven imported datasets. Their calibration scores are not inserted
  into full metrics; every full result comes from a new full output directory.
- **Eight selections precede full:** `full_stage():206-222` requires the exact
  eight selection keys, writes the all-config seal, and predicts all eight
  datasets with the exact selected configs and four declared arms before
  constructing the full prediction barrier. Its scoring loop follows that
  barrier. Completion requires 32 arm rows, and the final suite seal links the
  parent import, selections, barrier, scored outputs, metrics and WB trial CSV.
- **Overlap is explicit:** Waterbirds full manifests contain the input panel's
  development-overlap metadata. The all-selection/final seals and all four
  Waterbirds full rows carry the development scope; rows report 256 calibration
  sources in full and `independent_test_claim=False`. This checks truthful
  code/data labeling, not scientific validity or superiority of any score.

## Input Opt-In Integration

Reviewed the completed opt-in implementation and tests, not just the controller
call signature. `prepare_inputs(..., allow_waterbirds_development_overlap=True)`
exists, while omission/False retains the prior blocked behavior. The flag is a
strict bool, and opt-in requires exactly 256 examples with seed 2027.

`_select_waterbirds_development_overlap()` preserves source/path/inode and
RGB/label alias guards, requires training identities and physical inputs/label
metadata to match the original full rows, and selects from existing training
sources using the declared deterministic rank. The full population is not
reduced. Other datasets continue through the disjoint selector unchanged.

The fixture using synthetic 5322-full/2458-train Waterbirds populations verifies
the exact 256 chosen training IDs, repeatability, full membership/order, all
other-seven entries unchanged when toggling the flag, and overlap metadata in
calibration/evaluation panels, selection metadata and input audit. Strict-bool,
size/seed, label-read and identity-corruption cases are covered. No new P1/P2
input-opt-in defect was demonstrated. Input-author closure was reported by the
main user; no author's code was edited by this auditor.

## Verification And Limits

Independent final focused CPU run:

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/wwww/miniconda3/envs/mlmp/bin/python -B -m pytest -q -p no:cacheprovider tests/test_evolution_tuning_continuation.py tests/test_evolution_tuning_inputs.py
58 passed in 5.23s
```

The 58 cases comprise 16 continuation and 42 input cases. The main user's
reported broader 430-test run was not independently repeated here.

The parent-import fixtures execute the actual importer and verify seven config
reuse, CSV metric-drift rejection and selection-seal drift rejection. They
substitute `queue.verify()` and `queue.validate_result()` with synthetic
providers, use one completed trial per dataset, and are not a replay of actual
parent results or a comprehensive multi-trial winner test. The full-stage test
mocks subprocesses/barrier creation and checks dispatch ordering; unchanged
worker seal/barrier behavior was covered in the preceding audit. These limits
are not additional demonstrated bugs.

No real prepare, GPU/checkpoint execution, service status query, restart,
prediction-history scan, old-output mutation, commit or agent delegation was
performed. No claim is made that the real parent has completed, that any new
suite is prepared, or that this audit authorizes a launch. Audit work is complete;
C1 is the remaining code disposition for the owning controller author.

## Source Identity

Continuation implementation, input implementation and input tests were stable
across inspection/completion. Continuation tests gained the main author's
parent-import fixtures during review; the final run used the identity below.

```text
tools/evolution_tuning/continuation.py
687894f99b51dca6653050ac3dc2505f9e2fcb6216eb501bfbbbe91ab40771f8
tests/test_evolution_tuning_continuation.py
d1922e7eedec47b6653217c152f2f8b2e1f279ee0260d143c0b3eacfa85cb2a4
tools/evolution_tuning/inputs.py
872cfd4bf30e606e37eb17e0da3fba66bddda574c62a560acbabfe1dab242cdd
tests/test_evolution_tuning_inputs.py
38ec71e3def37875f05c7d24faa4b0447a41fc696de26bdb01e3573a8a9cfd5a
tools/evolution_tuning/queue.py
20e07e6263c3d47c20ae825c5ac0cde7984de5cdf923c31fdb6fc73ccd7cee1d
tools/evolution_tuning/worker.py
259fbb901b164eed6ca3c211db612947afdf96b7bd8259df47a3c35605767755
docs/single-probe-tuning-contract.md
765b0b13fc00de16357806e4bf923db2792033b3c7e9f842e3b759f6fff7fbaf
```
