# TRACE

**Tracing Confidence Sources for Reliable Test-Time Adaptation in
Open-Vocabulary Semantic Segmentation**

TRACE checks where a candidate's confidence comes from before allowing that
candidate to drive test-time adaptation. It builds connected candidates from
the dense OVSS prediction, evaluates candidate-specific target-preserved,
target-suppressed, and context-perturbed views, and assigns a detached
reliability weight to one region-normalized entropy objective.

## Method

For each candidate \(R\), TRACE measures target support \(s_{\mathrm{tar}}\)
and counterfactual support \(s_{\mathrm{cf}}\). Their shared quality is

\[
q_R = \min(s_{\mathrm{tar}}, s_{\mathrm{cf}}).
\]

Every candidate also measures residual support \(s_{\mathrm{res}}\).
Residual support that exceeds target support reduces its reliability:

\[
d_R = \max(s_{\mathrm{res}} - s_{\mathrm{tar}}, 0), \qquad
\rho_R = q_R(1-d_R).
\]

The same three views and formula apply to every class. New vocabularies need
no thing/stuff/background role map. Detached candidate weights allocate the
relative coefficients of region-mean entropy terms. Their absolute scale
cancels in the normalized objective.

## Installation

```bash
cd TRACE
python -m venv .venv
source .venv/bin/activate
pip install -e .

export TRACE_DATA_ROOT=/path/to/datasets
```

TRACE contains its own dense NaCLIP forward, prompt ensemble, public-dataset
loader, overlapping-patch aggregation, metrics, and adaptation loop. The
standard CLIP checkpoint is downloaded to `~/.cache/clip` on first use; set
`TRACE_CHECKPOINT_CACHE` to choose another cache directory.

The vendored CLIP/NaCLIP backbone definitions are MIT licensed. Attribution
and the license text are included under `trace_ovss/resources/`.

## Evaluation

The inherited dataset configurations in
[`trace_ovss/configs/trace.py`](trace_ovss/configs/trace.py) contain the frozen
dataset-specific values of \(K\), batch size, learning rate, and adaptation
steps. This branch evaluates a simplified method; historical main-table scores
must not be presented as measurements of this implementation.

```bash
trace-ovss \
  --dataset voc20 \
  --output outputs/main_table \
  --seed 0 \
  --trials 1
```

Use `--print-config` to inspect a resolved configuration without loading a
model:

```bash
trace-ovss --dataset coco_stuff --print-config
```

Supported dataset aliases are `voc20`, `voc21`, `pc59`, `pc60`,
`coco_object`, `coco_stuff`, `cityscapes`, `coco_cd`, and
`waterbirds_seg`.

## Sample Data

One licensed COCO-Stuff validation image and its converted semantic mask are
included under `sample_data/coco_stuff/`. This pair is intended only for a
one-image input-pipeline smoke check; it is not a reproduction subset and must
not be used to report benchmark metrics. Its source and license information are
recorded in [`sample_data/ATTRIBUTION.md`](sample_data/ATTRIBUTION.md).

To run TRACE on the included pair:

```bash
trace-ovss \
  --dataset coco_stuff \
  --data-root sample_data \
  --output outputs/sample \
  --workers 0 \
  --max-batches 1
```

The expected directories below are relative to `TRACE_DATA_ROOT`:

| Alias | Directory |
|---|---|
| `voc20`, `voc21`, `pc59`, `pc60` | `VOCdevkit/VOC2012` |
| `coco_object`, `coco_stuff` | `coco_stuff` |
| `cityscapes` | `cityscapes` |
| `coco_cd` | `coco_cd_ovss` |
| `waterbirds_seg` | `waterbirds_seg/correlation_0.95` |

## Runtime Implementation

TRACE streams diagnostic views in bounded batches and caches episode-invariant
region indices and loss weights. Normal inference does not retain layer-weight
histories. Individual layer weights can still be requested through
`forward_eval_logits(..., save_weights=True)` for analysis.

The ordinary attention blocks use PyTorch scaled-dot-product attention with
zero dropout; the NaCLIP neighborhood and SCLIP correlation rules retain their
own calculations. The default diagnostic view batch is 16 across datasets.
GPU half-precision results can differ from the previous manual attention
implementation. Fixed-input accuracy/cost measurements are recorded below;
the new implementation must be evaluated before replacing any historical score.

See [the simplification record](docs/simplification.md) for the method change,
checks, profiling protocol, review resolutions, and evidence scope.

## Repository Layout

- `trace_ovss/method/trace/`: candidates, interventions, reliability, and loss.
- `trace_ovss/models/`: dense CLIP/NaCLIP model flow.
- `trace_ovss/data/`: benchmark datasets and vocabularies.
- `trace_ovss/evaluation/`: patch reconstruction and segmentation metrics.
- `trace_ovss/configs/trace.py`: released best configurations.
- `trace_ovss/cli.py`: clean-validation entry point.
- `sample_data/`: one licensed COCO-Stuff image/mask pair for a smoke check.

Full datasets, checkpoints, generated results, and paper drafts are
intentionally excluded from the repository.

The COCO Object vocabulary removes `bed` from the background extensions: the
same text query cannot belong to both background and the bed class. All
comparisons using this corrected vocabulary must rerun every method with the
same queries; legacy COCO Object scores belong to the previous protocol.
