# Human-review response map

Source: the five official reviews in the user-supplied 2026-10-25 attachment
`f6b8def3-e107-4b79-b065-e4944da448cc`. This maps evidence to the actual objections;
it is not a rebuttal draft or a replacement for the submitted method.

| Reviewer | Point | Evidence/action |
| --- | --- | --- |
| MiQ4 | Published baseline values versus local TRACE | Existing matched_released local runs, frozen data/vocabulary/preprocessing and method-specific configurations. Explicit provenance and sample/selection/label-access disclosure. |
| MiQ4 | Reliability versus confidence, correctness and size | Existing image-disjoint crossfit and matching; new cached region-accuracy/majority-crossing audit with matching coverage; actual full-episode pixel transitions. |
| MiQ4 | GT-probability drop need not change segmentation | Distinct actual reconstructed pixel C->W/W->C and region-accuracy endpoints; do not rename a probability decrease a label flip. |
| MiQ4 | Clean single-backbone generalization | Existing independent corruption-before-resize panel; audit archived ViT-B/16 source correspondence before using its mixed results. |
| MiQ4 | Uncommon object-context combinations | Existing aligned/conflicting and four-group results, with original Waterbirds population stated. |
| MiQ4 | Role rules not ablated | Existing q-all / dominance-all / role-aware full-split comparison; q still attenuates stuff/background. |
| MiQ4 | Runtime and memory absent | Existing measured native per-dataset results and controlled K scaling; disclose timing boundary, device, precision and settings. |
| qytD | Frozen initial scores may become outdated | New fixed-mask frozen-versus-midpoint-refresh clean/shift comparison and time-series diagnostic rankings. |
| qytD | Stuff/background context unpenalized | Explain shared q attenuation; existing all-dominance control tests the additional residual rule. |
| qytD | Confidence top-K excludes useful candidates | New fixed-K selection controls and all-candidate audit on label-independent subsets. |
| qytD | Connected candidates too coarse | Existing connected-versus-class-wide module ablation; descriptive candidate-size/purity analysis where caches support it, with no fabricated within-region statistics. |
| qytD | Multi-view work at each step | Correct once-per-episode computation; report measured number of diagnostic inputs and actual optimizer steps. |
| qytD | Ablation and context perturbation lack repeats | New module ablations on four full datasets and context perturbations on two fixed subsets, seeds 0/1/2, reporting paired effects. |
| CMmW | Candidate/region formulation may explain gains | Complete-episode identical-candidate weight controls already compare uniform, confidence, flip consistency, q and Full; repeat module ablations. |
| CMmW | Singleton normalized weight cancels | Explicitly accept the algebra: isolated updates evaluate risk association; multi-candidate episode controls evaluate allocation benefit. |
| CMmW | Cross-view probabilities reflect artifacts | Existing identity, sham shift, mask erosion/dilation, weakening and recognition controls; behavioral dependence interpretation and new Phi alternatives. |
| CMmW | Artificial perturbation choice | New Phi Gaussian noise, context masking and fixed color-statistic transform, keeping W fixed; distinguish this from existing W sensitivity. |
| a9ZV | Figure 2 incomplete | Inspect final uploaded PDF separately; record figure repair as presentation work. |
| a9ZV | Additional backbone | Existing ViT-B/16 experiments need exact source/config provenance audit; no selective winner-only presentation. |
| a9ZV | VOC20/COCO-Stuff gains small | Existing local reruns, source-risk prevalence, candidate roles and per-class outcomes; explain bounded scenario dependence, not a universal win. |
| a9ZV | Latency/FLOPs | Existing actual latency/memory/work counts; no unsupported theoretical FLOPs or real-time claim. |
| Lk2c | Real-time cost | Hardware-specific measured end-to-end/adaptation latency and memory, once-per-episode batched diagnostics; latency is not removed by calling forwards parallel. |
| Lk2c | Noise, masking, style alternatives | New context Phi controls, clearly label color-statistic style transfer rather than learned neural stylization. |
| Lk2c | Pipeline illustration artifacts | Presentation repair, independent of algorithm evidence; no new scientific inference from a redrawn diagram. |

Primary response evidence is full-episode mIoU and real pixel transitions, then
conditional diagnostic evidence and perturbation controls. Quantitative cost and
protocol details answer measurement omissions directly. A learned crossfit
detector's AUROC cannot be called the AUROC of raw TRACE reliability.

The archived Full-versus-q mIoU effects are +0.006, -0.019, +0.061 and +0.017 on
VOC21/PC60/COCO Object/COCO-CD. The evidence supports the combined multi-view
quality mechanism; it does not establish a large independent dominance gain.
Keep the published method fixed while testing the scope of each design claim.
