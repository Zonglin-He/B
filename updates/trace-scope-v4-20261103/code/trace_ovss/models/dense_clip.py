"""Standalone dense CLIP/NaCLIP inference used by TRACE.

The checkpoint and tokenizer are standard CLIP components. This module owns
the OVSS-specific flow: selected visual layers, dense patch--text logits, and
entropy-weighted layer aggregation. No external benchmark repository is
imported at runtime.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ._vendor_clip import load as load_clip
from ._vendor_clip import tokenize


class DenseCLIP(nn.Module):
    """Expose CLIP spatial tokens as dense open-vocabulary logits."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        architecture: str = "reduced",
        attention_strategy: str = "naclip",
        gaussian_std: float = 5.0,
    ) -> None:
        super().__init__()
        if architecture not in {"reduced", "vanilla"}:
            raise ValueError(f"Unsupported visual architecture: {architecture}")
        if attention_strategy not in {
            "naclip",
            "csa",
            "vanilla",
        }:
            raise ValueError(
                f"Unsupported attention strategy: {attention_strategy}"
            )
        if gaussian_std <= 0 and attention_strategy == "naclip":
            raise ValueError("NaCLIP neighborhood attention needs gaussian_std > 0")

        self.backbone = backbone
        self.architecture = architecture
        self.attention_strategy = attention_strategy
        self.gaussian_std = float(gaussian_std)
        self.weights_track: list[torch.Tensor] = []
        self._neighborhood_cache: dict[tuple, torch.Tensor] = {}

    @property
    def visual(self):
        return self.backbone.visual

    @property
    def transformer(self):
        return self.backbone.transformer

    @property
    def ln_final(self):
        return self.backbone.ln_final

    @property
    def token_embedding(self):
        return self.backbone.token_embedding

    @property
    def dtype(self):
        return self.backbone.dtype

    def encode_text(self, text):
        return self.backbone.encode_text(text)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        text_ensemble: bool = False,
        *,
        vision_outputs: Iterable[int] = (-1,),
        interpolate: bool = False,
        vision_out_type: str = "mean",
        save_weights: bool = False,
    ):
        """Return dense logits with shape ``templates x batch x classes x H x W``."""

        if text_ensemble:
            text_features = text
        else:
            text_features = self.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(0)

        raw_layers = self.encode_dense_layers(
            image,
            vision_outputs,
            project=False,
        )

        if vision_out_type == "mean":
            image_features = self._project_visual(raw_layers.mean(dim=0))
        elif vision_out_type == "adaptive_weighted_mean":
            layer_features = self._project_visual(raw_layers)
            layer_features = F.normalize(layer_features, dim=-1)
            layer_logits = self._similarity(layer_features, text_features)
            patch_logits = self._tokens_to_grid(layer_logits[..., 1:, :], image)
            entropy = -(
                patch_logits.softmax(dim=3)
                * patch_logits.log_softmax(dim=3)
            ).sum(dim=3)
            weights = F.softmax(-entropy.mean(dim=(1, 3, 4)), dim=0)
            if save_weights:
                self.weights_track.append(weights.detach().cpu())
            image_features = self._project_visual(
                (
                    raw_layers
                    * weights[:, :, None, None].to(raw_layers.dtype)
                ).sum(dim=0)
            )
        else:
            raise ValueError(
                "vision_out_type must be 'mean' or "
                "'adaptive_weighted_mean'"
            )

        image_features = F.normalize(image_features, dim=-1)
        logits = self._similarity(image_features, text_features)
        logits = self._tokens_to_grid(logits[..., 1:, :], image)
        if interpolate:
            templates, batch, classes, height, width = logits.shape
            logits = F.interpolate(
                logits.reshape(templates * batch, classes, height, width),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(
                templates,
                batch,
                classes,
                image.shape[-2],
                image.shape[-1],
            )
        return logits, image_features, text_features

    def encode_dense_layers(
        self,
        image: torch.Tensor,
        output_layers: Iterable[int],
        *,
        project: bool = True,
    ) -> torch.Tensor:
        """Run the visual encoder and collect the requested spatial layers."""

        visual = self.visual
        if not hasattr(visual, "patch_size"):
            raise TypeError("TRACE currently supports CLIP ViT backbones only")

        image = image.to(dtype=visual.conv1.weight.dtype)
        _, _, image_h, image_w = image.shape
        grid = (image_h // visual.patch_size, image_w // visual.patch_size)

        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1)
        if x.shape[1] == visual.positional_embedding.shape[0]:
            position = visual.positional_embedding
        else:
            position = visual.interpolate_pos_encoding(
                x,
                image_h,
                image_w,
            ).squeeze(0)
        x = visual.ln_pre(x + position.to(x.dtype))
        x = x.permute(1, 0, 2)

        blocks = visual.transformer.resblocks
        selected = self._resolve_layers(output_layers, len(blocks))
        last_selected = max(selected)
        outputs = []
        for layer_idx, block in enumerate(blocks):
            if layer_idx > last_selected:
                break
            normalized = block.ln_1(x)
            strategy = (
                self.attention_strategy
                if layer_idx == last_selected
                else "vanilla"
            )
            attention = self._attention(
                strategy,
                block.attn,
                normalized,
                grid,
            )
            # The final reduced output uses attention alone and has no next
            # block consuming the residual/MLP branch.
            if layer_idx < last_selected or self.architecture != "reduced":
                propagated = x + attention
                propagated = propagated + block.mlp(block.ln_2(propagated))

            if layer_idx in selected:
                feature = (
                    attention
                    if self.architecture == "reduced"
                    else propagated
                )
                outputs.append(feature.permute(1, 0, 2))
            if layer_idx < last_selected:
                x = propagated

        if not outputs:
            raise RuntimeError("No visual layer was selected")
        outputs = torch.stack(outputs, dim=0)
        return self._project_visual(outputs) if project else outputs

    def _project_visual(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            # Preserve the released NaCLIP numerics: selected layers are
            # projected independently before being stacked for weighting.
            return torch.stack(
                [
                    self.visual.ln_post(layer) @ self.visual.proj
                    for layer in features
                ],
                dim=0,
            )
        return self.visual.ln_post(features) @ self.visual.proj

    def _attention(
        self,
        strategy: str,
        layer: nn.MultiheadAttention,
        x: torch.Tensor,
        grid: tuple[int, int],
    ) -> torch.Tensor:
        tokens, batch, width = x.shape
        heads = layer.num_heads
        head_dim = width // heads
        scale = head_dim**-0.5
        q, k, v = F.linear(
            x,
            layer.in_proj_weight,
            layer.in_proj_bias,
        ).chunk(3, dim=-1)

        def split_heads(value):
            return (
                value.contiguous()
                .view(tokens, batch * heads, head_dim)
                .transpose(0, 1)
            )

        if strategy == "naclip":
            # These attention rules never consume the query projection.
            k, v = (split_heads(value) for value in (k, v))
        else:
            q, k, v = (split_heads(value) for value in (q, k, v))
        if strategy == "vanilla":
            output = F.scaled_dot_product_attention(
                q.reshape(batch, heads, tokens, head_dim),
                k.reshape(batch, heads, tokens, head_dim),
                v.reshape(batch, heads, tokens, head_dim),
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            ).reshape(batch * heads, tokens, head_dim)
            output = output.transpose(0, 1).contiguous().view(tokens, batch, width)
            return layer.out_proj(output)
        elif strategy == "csa":
            q_weights = torch.bmm(q, q.transpose(1, 2)) * scale
            k_weights = torch.bmm(k, k.transpose(1, 2)) * scale
            weights = q_weights.softmax(dim=-1) + k_weights.softmax(dim=-1)
        elif strategy == "naclip":
            neighborhood = self._neighborhood(
                grid,
                dtype=x.dtype,
                device=x.device,
            )
            weights = torch.bmm(k, k.transpose(1, 2)) * scale
            weights = weights + neighborhood
            weights = weights.softmax(dim=-1)
        else:
            raise ValueError(f"Unknown attention strategy: {strategy}")

        output = torch.bmm(weights, v)
        output = (
            output.transpose(0, 1)
            .contiguous()
            .view(tokens, batch, width)
        )
        return layer.out_proj(output)

    def _neighborhood(
        self,
        grid: tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = (grid, dtype, device.type, device.index)
        cached = self._neighborhood_cache.get(key)
        if cached is not None:
            return cached

        height, width = grid
        window_h, window_w = 2 * height - 1, 2 * width - 1
        constant = 1.0 / (self.gaussian_std * math.sqrt(2.0))
        axes = []
        for size in (window_h, window_w):
            start = -(size - 1) / 2.0
            axes.append(
                torch.linspace(
                    start * constant,
                    (start + size - 1) * constant,
                    size,
                )
            )
        mesh = torch.meshgrid(*axes, indexing="ij")
        window = torch.exp(-(torch.stack(mesh) ** 2).sum(dim=0))

        identity = torch.einsum(
            "ij,kl->ijkl",
            torch.eye(height),
            torch.eye(width),
        )
        identity = identity.permute(0, 3, 1, 2).contiguous()
        addition = F.conv2d(
            identity.view(-1, height, width).unsqueeze(1),
            window.unsqueeze(0).unsqueeze(1),
            padding="same",
        ).squeeze(1)
        addition = addition.view(height * width, height * width)
        addition = torch.vstack(
            (torch.zeros((1, height * width)), addition)
        )
        addition = torch.hstack(
            (torch.zeros((height * width + 1, 1)), addition)
        )
        cached = addition.unsqueeze(0).to(device=device, dtype=dtype)
        self._neighborhood_cache[key] = cached
        return cached

    def _similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.backbone.logit_scale.exp()
        if image_features.ndim == 4:
            return scale * torch.einsum(
                "lbsd,tcd->ltbsc",
                image_features,
                text_features,
            )
        return scale * torch.einsum(
            "bsd,tcd->tbsc",
            image_features,
            text_features,
        )

    def _tokens_to_grid(
        self,
        logits: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        patch_size = int(self.visual.patch_size)
        grid_h = image.shape[-2] // patch_size
        grid_w = image.shape[-1] // patch_size
        if grid_h * grid_w != logits.shape[-2]:
            raise ValueError(
                "Dense token count does not match the visual patch grid: "
                f"{logits.shape[-2]} versus {grid_h}x{grid_w}"
            )
        if logits.ndim == 5:
            layers, templates, batch, _, classes = logits.shape
            return logits.permute(0, 1, 2, 4, 3).reshape(
                layers,
                templates,
                batch,
                classes,
                grid_h,
                grid_w,
            )
        templates, batch, _, classes = logits.shape
        return logits.permute(0, 1, 3, 2).reshape(
            templates,
            batch,
            classes,
            grid_h,
            grid_w,
        )

    @staticmethod
    def _resolve_layers(
        output_layers: Iterable[int],
        layer_count: int,
    ) -> tuple[int, ...]:
        resolved = tuple(
            layer_count + int(index) if int(index) < 0 else int(index)
            for index in output_layers
        )
        if not resolved:
            raise ValueError("At least one visual output layer is required")
        invalid = sorted(
            index for index in set(resolved) if not 0 <= index < layer_count
        )
        if invalid:
            raise ValueError(
                f"Visual layer indices out of range for {layer_count} blocks: "
                f"{invalid}"
            )
        return tuple(sorted(set(resolved)))


def load_dense_ovss(
    model_type: str = "naclip",
    backbone: str = "ViT-L/14",
    *,
    device: str | torch.device = "cpu",
    download_root: str | Path | None = None,
) -> tuple[DenseCLIP, object]:
    """Load a standalone dense CLIP family model and tokenizer."""

    policies = {
        "clip": ("vanilla", "vanilla"),
        "sclip": ("vanilla", "csa"),
        "naclip": ("reduced", "naclip"),
    }
    try:
        architecture, attention = policies[model_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported OVSS model type: {model_type}") from exc

    backbone_model, _ = load_clip(
        backbone,
        device=device,
        jit=False,
        download_root=(
            str(Path(download_root).expanduser())
            if download_root is not None
            else None
        ),
    )
    model = DenseCLIP(
        backbone_model,
        architecture=architecture,
        attention_strategy=attention,
    ).to(device)
    return model, tokenize
