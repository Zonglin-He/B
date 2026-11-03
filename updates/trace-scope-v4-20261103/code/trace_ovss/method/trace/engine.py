"""End-to-end TRACE test-time adaptation engine."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.optim as optim

from trace_ovss.data.vocabulary import load_prompt_templates
from trace_ovss.models import DenseCLIP, load_dense_ovss

from ..adapter import ModelAdapterMixin
from .candidates import TraceCandidateMixin
from .estimator import TraceEstimatorMixin
from .interventions import TraceInterventionMixin
from .objective import TraceObjectiveMixin


class TRACE(
    TraceEstimatorMixin,
    TraceObjectiveMixin,
    TraceCandidateMixin,
    TraceInterventionMixin,
    ModelAdapterMixin,
):
    """Tracing Confidence Sources for reliable OVSS adaptation."""

    def __init__(
        self,
        *,
        classes: Sequence[str],
        lr: float,
        steps: int,
        trace_max_regions: int,
        trace_view_batch_size: int,
        vision_outputs=tuple(range(-1, -19, -1)),
        model_type: str = "naclip",
        backbone: str = "ViT-L/14",
        prompts: Sequence[str] | None = None,
        checkpoint_cache: str | Path | None = None,
        model=None,
        tokenize_fn=None,
        runtime_calculation: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        self.classes = tuple(classes)
        if not self.classes:
            raise ValueError("TRACE requires at least one text class")
        self.lr = float(lr)
        self.steps = int(steps)
        self.trace_max_regions = int(trace_max_regions)
        self.trace_view_batch_size = int(trace_view_batch_size)
        self.vision_outputs = tuple(int(index) for index in vision_outputs)
        self.runtime = bool(runtime_calculation)
        self.device = torch.device(device)

        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.trace_max_regions <= 0:
            raise ValueError("trace_max_regions must be positive")
        if self.trace_view_batch_size <= 0:
            raise ValueError("trace_view_batch_size must be positive")

        if model is None:
            model, tokenize_fn = load_dense_ovss(
                model_type,
                backbone,
                device=self.device,
                download_root=checkpoint_cache,
            )
        if tokenize_fn is None:
            raise ValueError("tokenize_fn is required when injecting a model")
        self.model = model.to(self.device)
        self.tokenize = tokenize_fn
        self.prompt_templates = tuple(prompts or load_prompt_templates())

        self.model.requires_grad_(False)
        self.set_ln_grads(self.model.visual)
        if isinstance(self.model, DenseCLIP):
            blocks = self.model.visual.transformer.resblocks
            last_selected = max(self.model._resolve_layers(self.vision_outputs, len(blocks)))
            for block in blocks[last_selected + 1:]:
                block.requires_grad_(False)
            if self.model.architecture == "reduced":
                # The last selected output omits its MLP, including this norm.
                blocks[last_selected].ln_2.requires_grad_(False)
        parameters, self.adapted_parameter_names = self.collect_ln_params(
            self.model.visual
        )
        if not parameters:
            raise RuntimeError("No visual LayerNorm affine parameters found")
        self.optimizer = optim.Adam(
            parameters,
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model,
            self.optimizer,
        )

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes,
                self.prompt_templates,
                average=True,
            )

        self._mean = torch.tensor(
            [122.7709, 116.7460, 104.0937],
            device=self.device,
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            [68.5005, 66.6322, 70.3232],
            device=self.device,
        ).view(1, 3, 1, 1)
        self.record_candidate_proposals = False
        self.last_candidate_proposals = []
        self.last_candidate_stats = []
        self.last_region_terms = []
        self.last_reliability_summary = {}
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def adapt(self, images, valid_mask=None, image_ids=None):
        """Reset episodically, trace confidence sources, and update LayerNorm."""

        self.reset()
        started = time.perf_counter()
        region_terms = self.estimate_trace_reliability(
            images,
            valid_mask=valid_mask,
            image_ids=image_ids,
        )["region_terms"]
        if not region_terms:
            if self.runtime:
                self.adapt_times.append(time.perf_counter() - started)
            return []

        losses = []
        prepared_objective = None
        for _ in range(self.steps):
            logits = torch.nan_to_num(
                self.forward_adapt_logits(images).float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
            entropy = torch.nan_to_num(
                self.trace_softmax_entropy(logits),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if prepared_objective is None:
                prepared_objective = self.prepare_trace_objective(entropy, region_terms)
            loss = self.trace_region_normalized_entropy(
                entropy,
                region_terms,
                prepared=prepared_objective,
            )
            if not loss.requires_grad:
                break
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            losses.append(loss.detach())

        losses = torch.stack(losses).cpu().tolist() if losses else []

        if self.runtime:
            self.adapt_times.append(time.perf_counter() - started)
        return losses

    @torch.no_grad()
    def evaluate(self, images):
        started = time.perf_counter()
        logits = self.forward_eval_logits(
            images,
            interpolate=True,
            save_weights=False,
        )
        if self.runtime:
            self.eval_times.append(time.perf_counter() - started)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
