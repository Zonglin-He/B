# Reviewed Single-Probe TRACE: Reproduction And Evidence

This companion was added after review. It does not modify the reviewed PDFs,
method, configurations or evidence. It is not a complete portable data release.

## Evaluated Environment

Canonical full-suite initial_state.json records:

- GPU:NVIDIA GeForce RTX5090.
- Python3.10.13, GCC11.2.0 build.
- PyTorch2.7.0+cu128.
- NACLIP ViT-L/14, last18 visual outputs, seven fixed prompts.
- Official OpenAI ViT-L-14 checkpoint SHA256:
  b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836.

Deterministic default attention dispatch is retained. The runtime sets
CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic PyTorch/cuDNN operations and
disabled cuDNN benchmarking. Model seed0 is not multiple independent trials.

## Build The Reviewed Documents

From the extracted packet's paper directory, with pdflatex and bibtex on PATH:

```bash
python tools/build.py
```

Outputs are build/main.pdf (combined paper plus appendix) and
build/supplement.pdf (standalone appendix). No inference or table export runs
as part of this build. Do not regenerate evidence tables with an older exporter.
The archived original PDFs, rather than byte-identical TeX recompilation on a
different TeX installation, define the exact reviewed visual artifact.

## Method And Evaluation Entry Points

The actual selected class is ImageBalancedTRACE(view_mode="C") in
trace_ovss/method/image_balanced.py. The full worker/controller are
tools/evolution_image_full/run.py and queue.py. The ordinary legacy method
default is not a substitute for this explicitly selected experimental class.
The frozen full plan is docs/image-full-plan.json in the branch.

On the original host, the existing completed experiment can be verified
without GPU work or rewriting outputs:

```bash
CUDA_VISIBLE_DEVICES='' python -B paper_evolution/tools/export_image_full.py --check-only
```

The controller's interfaces for a separately prepared experiment are:

```bash
python -m tools.evolution_image_full.queue prepare --source-root <checkout> --plan <locked-plan.json> --python <python-executable> --output <new-suite-directory>
python -m tools.evolution_image_full.queue verify --suite <prepared-suite> --suite-sha256 <new-prepared-sha256>
```

These are interface examples, not commands executed at delivery. Preparation
requires the pinned selection and inherited protocol artifacts named by the
plan. Running GPU work additionally requires the frozen runtime, its run
subcommand, exclusive shared GPU lock and the bounded systemd configuration
recorded in the prepared suite. Never rerun or overwrite the completed suite.
Moving data/weights or changing paths requires a new input manifest and new
prepared hash, not editing old frozen records or bypassing validation.

## Evidence

- Full18 outcomes: results/trace_image_full_20261029_0333 under the main
  /home/wwww/hzl checkout; exact root is in the publication provenance.
- Full selection parent:trace_image_budget_20261029_0233, eight cells/48 arms.
- Generated image_full_provenance.json retains exact scores, per-class pooled
  counts, configs, timing totals/denominators and actual encoder counters.
- Figure/image_full_provenance.json binds new saved candidate/segmentation
  cases, GT-free teaser selection and offline qualitative extreme selection.
- Raw review round04.md/json and separate author disposition are preserved.

The delivery bundle contains manuscript sources/assets, generated small
tables/provenance, actual frozen runtime source and result CSV. It does not
contain licensed/downloadable checkpoint weights, public datasets or the large
prediction population. Dependencies and input locations remain tied to the
recorded environment. A turnkey portable release needs additional packaging;
it is not certified here.

## Interpretation

R denotes paper-reported values, L local. Published MLMP values are not new
same-environment reruns. Current reset No Adapt is not certified. C uses one
diagnostic input per candidate; batching changes call counts. Timing is
model-side, not complete end-to-end throughput; only episode0 is excluded
from time, not accuracy or work. Current scores use224x224 resized labels,
not native-resolution evaluation. Historical development exposure remains
disclosed. See the PDF and raw review for the complete scope boundaries.
