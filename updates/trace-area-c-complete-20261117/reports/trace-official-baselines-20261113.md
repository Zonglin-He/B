# Official Baselines, November 13 Correction

Authorization: actual human requires official MLMP and No Adapt implementations,
and authorizes installing missing dependencies. Do not tune baseline parameters
to bring scores closer to published numbers. Preserve every old result and failure.

## Fixed Protocol

- Pin MLMP upstream `2a887de018b3f2c3b132f69fe806bab12a2d511f`; no upstream edits.
- Use the upstream dataset classes, complete official splits, class extensions,
  RGB transforms, corruption generation at severity5, collation, shuffled DataLoader,
  reconstruction, query-to-class reduction, and metric functions.
- Substitute only machine-specific image/annotation directory prefixes. This host
  keeps COCO RGB in data/coco/val2017 and Object masks in a separate coco_mlmp
  tree, not a common .data/images/val2017 hierarchy. Full IDs and resolved annotation
  paths must agree with the preserved manifest; suffixes and transforms stay official.
- Preserve all113 Object queries, including both background and foreground bed.
- NoAdapt uses official TENT with adaptation disabled, empty prompt_dir (the
  reference prompt), and the native final-layer readout. MLMP-zero-update is
  a different historical arm, never relabelled as this NoAdapt.
- MLMP uses Adam lr0.001, batch2, tensteps, alpha_cls1, allseven official prompts,
  final18 L14 layers, model/optimizer reset per episode. Three shuffled trials,
  global seed0 set once for each dataset/method. Corruption stages retain the
  RNG state of the previous condition, matching the upstream outer loop.
- Four loader workers,224square RGB/GT geometry, stride112, pooled official metrics.
  Record individual rounded-trial values and the mean/std of allthree, never besttrial.
- Official RGB pipeline is unchanged. Only LoadAnnotations is deferred during
  prediction: supply an all-ignore dummy map used by the geometric packer, prohibit
  real annotation opens, seal complete predictions, verify actual retained process
  success, then run the original LoadAnnotations and metrics in a CPU score stage.
- Dependencies are installed in an isolated prefix. Build the real MMCV2.1.0 CPU
  extension against the installed PyTorch; no fake ops or silent operator substitutes.
  Baselines do not call MMCV CUDA kernels. Keep the active environment unchanged.
- Hash checkpoints incrementally, not through the64MiB metadata reader. A failed
  pre-model attempt exposed this adapter error; preserve it and its original
  frozen bytes. Streaming verification does not change model or cgroup settings.

## Scope and Boundaries

Finite queue covers the existing26conditions:8L14clean,3B16clean,15VOC20-C types.
Start PC60 and Object to check the unexplained/genuine protocol differences, then
the remaining clean conditions and fixed backbone/corruption transfers. Each arm
has a prediction stage followed by separately isolated CPU scoring. Each stage
is run once under a new frozen source plan; no unchanged-byte automatic retries.

Keep all22 clean-selected area_C parameters fixed. Recompute its predictions under
the common official vocabulary and input protocol, without further tuning. Copy
matching clean parameters exactly to B16 and S5. Existing local-protocol results
are preserved, not pasted into the new comparison or used to select parameters.

Waterbirds and COCO-CD are not upstream paper datasets: use their corrected full
manifest splits with the same native methods and official transforms, and label
them declared data extensions. B16 uses the existing final75-percent-layer transfer
(9of12), not an unchanged L14 shell script. No strict paper claim for these extensions.

Use allthree repetitions for reported mean/std and retain compact source counts.
Random shuffled episodes cross fixed contiguous groups, so the old contiguous16
bootstrap is not transplanted into this protocol. No fabricated paired CI or
independent marginal-CI subtraction. Keep historical development exposure explicit.

Paper differences above0.2pp are diagnostic flags only, never retry, search,
exclusion, or stopping criteria. Inspect remaining data/environment differences
when a discrepancy remains; do not force agreement. CPU4GiB/1GiBswap/one thread,
GPU12GiB/1GiBswap, OOMstop, Restartno, same exclusive GPUlock. No monitoring agent,
cron, automation activation, upload, old-publication edit, or cache deletion.
