"""Research-only frozen-prefix adaptation with optional exact prefix reuse.

Freezing early visual LayerNorms changes the adaptation algorithm. Cache
equivalence is only to the SAME frozen-prefix algorithm, not full-LN TRACE.
Use ``make_cached_prefix_class(TRACE, algorithm)`` with the unwrapped engine.
K, steps, batches, LR, prompts, selected layers and aggregation are inherited.
The depth-6 uncached control isolates caching from the adaptation-depth change.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import weakref

import torch
from torch import nn

from trace_ovss.models.dense_clip import DenseCLIP

if __package__:
    from .trace_algorithm_variants import make_algorithm_class
else:
    from trace_algorithm_variants import make_algorithm_class


CACHED_PREFIX_SETTINGS = {
    "quality_tail6": {"tail_depth": 6, "cache_enabled": True},
    "quality_tail12": {"tail_depth": 12, "cache_enabled": True},
    "quality_tail0": {"tail_depth": 0, "cache_enabled": True},
    "quality_tail6_uncached": {"tail_depth": 6, "cache_enabled": False},
}
CACHED_PREFIX_DEFINITIONS = {
    name: (
        "Complete two-view quality_only; freeze ln_pre and all visual blocks "
        f"except LayerNorm affine in the last {settings['tail_depth']} executed "
        "blocks; always adapt ln_post. The final reduced block's ln_2 stays "
        "frozen. Preserve all selected raw outputs and production aggregation. "
        + (
            "Cache only frozen prefix tokens and early selected RAW outputs "
            "for the identical original-image tensor within one episode, "
            "including proposal, adaptation and final evaluation. Diagnostic "
            "pixel interventions always run the full encoder."
            if settings["cache_enabled"] else
            "No caching on any forward. This has exactly the same trainable "
            "parameters as quality_tail6 and isolates cache equivalence."
        )
        + " Freezing changes the algorithm; caching is exact only conditional "
        "on that frozen prefix, not equivalent to full-LN adaptation."
    )
    for name, settings in CACHED_PREFIX_SETTINGS.items()
}


@dataclass
class _PrefixCache:
    key: tuple
    tokens: torch.Tensor | None
    raw_outputs: tuple[torch.Tensor, ...]


class CachedPrefixDenseCLIP(DenseCLIP):
    """Reuse a detached frozen prefix, never trainable tail or projected outputs.

    Caching is opt-in for one tensor via ``bind_original``. Object identity,
    mutation version, layout, frozen-parameter versions and autocast policy
    must match. Clones, slices and interventions never inherit another tensor's
    cache. Inference tensors without mutation counters take the uncached path.
    ``clear_cache`` releases features and the weak original-image reference.

    Actual work remains observable at conv1 and every block's ln_1, including
    a first cache build under a training forward. No block-call estimates are
    substituted for execution; cache_stats reports only reuse and storage.
    """

    def __init__(self, source, *, vision_outputs, tail_depth, cache_enabled=True):
        if not isinstance(source, DenseCLIP) or isinstance(source, CachedPrefixDenseCLIP):
            raise TypeError("Expected an unwrapped DenseCLIP encoder")
        if not hasattr(source.visual, "patch_size"):
            raise TypeError("Cached-prefix adaptation requires a CLIP ViT")
        if isinstance(tail_depth, bool) or not isinstance(tail_depth, int):
            raise ValueError("tail_depth must be an integer")
        selected = source._resolve_layers(
            vision_outputs, len(source.visual.transformer.resblocks),
        )
        executed_depth = max(selected) + 1
        if not 0 <= tail_depth < executed_depth:
            raise ValueError("tail_depth must be >= 0 and < the executed visual depth")
        super().__init__(source.backbone, architecture=source.architecture,
                         attention_strategy=source.attention_strategy,
                         gaussian_std=source.gaussian_std)
        self.training = source.training
        self.weights_track = source.weights_track
        self._neighborhood_cache = source._neighborhood_cache
        self.selected_layers = selected
        self.executed_depth = executed_depth
        self.tail_depth = tail_depth
        self.prefix_depth = executed_depth - tail_depth
        self.cache_enabled = bool(cache_enabled)
        self._original_ref = None
        self._prefix_cache = None
        self._cache_disabled_depth = 0
        self._cache_builds = 0
        self._cache_hits = 0

        self.requires_grad_(False)
        for block in self.visual.transformer.resblocks[self.prefix_depth:executed_depth]:
            for module in block.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
        self.visual.ln_post.requires_grad_(True)
        if self.architecture == "reduced":
            self.visual.transformer.resblocks[executed_depth - 1].ln_2.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None

    def clear_cache(self, *, reset_stats=False):
        self._prefix_cache = None
        self._original_ref = None
        if reset_stats:
            self._cache_builds = self._cache_hits = 0

    def bind_original(self, images):
        """Declare the complete original batch, not individual diagnostic views."""
        if not self.cache_enabled or self._cache_disabled_depth:
            return
        if images.requires_grad:
            raise ValueError("Frozen-prefix caching does not support image gradients")
        if self._original_ref is not None and self._original_ref() is images:
            return
        self.clear_cache()
        owner = weakref.ref(self)

        def release(_):
            model = owner()
            if model is not None:
                model.clear_cache()

        self._original_ref = weakref.ref(images, release)

    @contextmanager
    def cache_disabled(self):
        """Force exact production pixel forwards, without replacing the cache."""
        self._cache_disabled_depth += 1
        try:
            yield
        finally:
            self._cache_disabled_depth -= 1

    @property
    def cache_stats(self):
        cached = self._prefix_cache
        tensors = (() if cached is None else cached.raw_outputs + (
            (cached.tokens,) if cached.tokens is not None else ()
        ))
        return dict(cache_builds=self._cache_builds, cache_hits=self._cache_hits,
                    cache_bytes=sum(t.numel() * t.element_size() for t in tensors))

    def _prefix_parameters(self):
        yield self.visual.class_embedding
        yield self.visual.positional_embedding
        yield from self.visual.conv1.parameters()
        yield from self.visual.ln_pre.parameters()
        for block in self.visual.transformer.resblocks[:self.prefix_depth]:
            yield from block.parameters()

    def _cache_key(self, image):
        try:
            version = image._version
        except RuntimeError:
            return None
        parameters = tuple(self._prefix_parameters())
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("The cached prefix must stay frozen")
        autocast = torch.is_autocast_enabled(image.device.type)
        return (
            version, tuple(image.shape), tuple(image.stride()), image.storage_offset(),
            image.data_ptr(), image.device, image.dtype, self.training,
            self.architecture, self.attention_strategy, self.gaussian_std,
            autocast, torch.get_autocast_dtype(image.device.type) if autocast else None,
            tuple((id(p), p._version, p.data_ptr(), p.device, p.dtype) for p in parameters),
        )

    def _stem(self, image):
        visual = self.visual
        image = image.to(dtype=visual.conv1.weight.dtype)
        _, _, image_h, image_w = image.shape
        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1)
        position = (
            visual.positional_embedding
            if x.shape[1] == visual.positional_embedding.shape[0] else
            visual.interpolate_pos_encoding(x, image_h, image_w).squeeze(0)
        )
        x = visual.ln_pre(x + position.to(x.dtype))
        return x.permute(1, 0, 2)

    def _blocks(self, x, start, stop, grid):
        outputs = []
        last_selected = self.executed_depth - 1
        for index in range(start, stop):
            block = self.visual.transformer.resblocks[index]
            normalized = block.ln_1(x)
            strategy = self.attention_strategy if index == last_selected else "vanilla"
            attention = self._attention(strategy, block.attn, normalized, grid)
            if index < last_selected or self.architecture != "reduced":
                propagated = x + attention
                propagated = propagated + block.mlp(block.ln_2(propagated))
            if index in self.selected_layers:
                feature = attention if self.architecture == "reduced" else propagated
                outputs.append(feature.permute(1, 0, 2))
            if index < last_selected:
                x = propagated
        return x, tuple(outputs)

    def encode_dense_layers(self, image, output_layers, *, project=True):
        selected = self._resolve_layers(output_layers, len(self.visual.transformer.resblocks))
        if selected != self.selected_layers:
            raise ValueError("Selected visual layers must match the frozen-prefix configuration")
        eligible = (
            self.cache_enabled and not self._cache_disabled_depth
            and self._original_ref is not None and self._original_ref() is image
            and not image.requires_grad
        )
        key = self._cache_key(image) if eligible else None
        if key is None:
            return super().encode_dense_layers(image, selected, project=project)

        grid = tuple(size // self.visual.patch_size for size in image.shape[-2:])
        cached = self._prefix_cache
        if cached is None or cached.key != key:
            self._prefix_cache = None
            # Inference-mode proposal callers must not leave inference tensors
            # that autograd cannot save when the tail is subsequently trained.
            with torch.inference_mode(False), torch.no_grad():
                tokens, outputs = self._blocks(self._stem(image), 0, self.prefix_depth, grid)
                cached = _PrefixCache(
                    key, tokens.detach() if self.tail_depth else None,
                    tuple(output.detach() for output in outputs),
                )
            self._prefix_cache = cached
            self._cache_builds += 1
        else:
            self._cache_hits += 1

        _, tail_outputs = self._blocks(cached.tokens, self.prefix_depth, self.executed_depth, grid)
        outputs = torch.stack(cached.raw_outputs + tail_outputs, dim=0)
        return self._project_visual(outputs) if project else outputs


def make_cached_prefix_class(base, algorithm):
    """Wrap the unmodified TRACE class with complete quality_only and tail LN."""
    if algorithm not in CACHED_PREFIX_SETTINGS:
        raise ValueError(f"Unknown cached-prefix algorithm: {algorithm}")
    settings = dict(CACHED_PREFIX_SETTINGS[algorithm])
    quality_base = make_algorithm_class(base, "quality_only")

    class CachedPrefixTRACE(quality_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cached_prefix_algorithm = algorithm
            self.cached_prefix_settings = dict(settings)
            self.model = CachedPrefixDenseCLIP(
                self.model, vision_outputs=self.vision_outputs, **settings,
            )
            parameters, self.adapted_parameter_names = self.collect_ln_params(self.model.visual)
            self.optimizer = type(self.optimizer)(parameters, **self.optimizer.defaults)
            self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
                self.model, self.optimizer,
            )

        def reset(self):
            self.model.clear_cache(reset_stats=True)
            super().reset()

        @torch.no_grad()
        def estimate_trace_reliability(self, images, valid_mask=None, image_ids=None):
            self.model.bind_original(images)
            try:
                return super().estimate_trace_reliability(images, valid_mask, image_ids)
            except BaseException:
                self.model.clear_cache()
                raise

        @torch.no_grad()
        def evaluate_trace_views(self, images, original_probs, candidates):
            with self.model.cache_disabled():
                return super().evaluate_trace_views(images, original_probs, candidates)

        def forward_adapt_logits(self, images):
            self.model.bind_original(images)
            return super().forward_adapt_logits(images)

        def adapt(self, images, valid_mask=None, image_ids=None):
            try:
                losses = super().adapt(images, valid_mask, image_ids)
            except BaseException:
                self.model.clear_cache()
                raise
            if not losses:
                self.model.clear_cache()
            return losses

        @torch.no_grad()
        def evaluate(self, images):
            try:
                return super().evaluate(images)
            finally:
                self.model.clear_cache()

    return CachedPrefixTRACE
