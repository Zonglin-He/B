"""The fused vanilla path implements the same attention and input gradients."""

import pytest
import torch
import torch.nn.functional as F

from trace_ovss.models._vendor_clip.model import CLIP
from trace_ovss.models.dense_clip import DenseCLIP


def reference_attention(layer, x):
    tokens, batch, width = x.shape
    heads = layer.num_heads
    d = width // heads
    q, k, v = F.linear(x, layer.in_proj_weight, layer.in_proj_bias).chunk(3, -1)
    q, k, v = [a.contiguous().view(tokens, batch * heads, d).transpose(0, 1) for a in (q, k, v)]
    output = ((q * d**-.5) @ k.transpose(1, 2)).softmax(-1) @ v
    output = output.transpose(0, 1).contiguous().view(tokens, batch, width)
    return layer.out_proj(output)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
@pytest.mark.parametrize("batch,heads", [(1, 1), (3, 4)])
def test_fused_attention_matches_reference_values_and_gradients(dtype, tolerance, batch, heads):
    torch.manual_seed(19)
    backbone = CLIP(embed_dim=16, image_resolution=32, vision_layers=2,
                    vision_width=64, vision_patch_size=16, context_length=8,
                    vocab_size=32, transformer_width=64, transformer_heads=1,
                    transformer_layers=1)
    model = DenseCLIP(backbone).to(dtype=dtype)
    layer = torch.nn.MultiheadAttention(64, heads, dropout=0).to(dtype=dtype)
    inputs = torch.randn(5, batch, 64, dtype=dtype, requires_grad=True)
    expected = reference_attention(layer, inputs)
    actual = model._attention("vanilla", layer, inputs, (2, 2))
    variables = (inputs, layer.in_proj_weight, layer.in_proj_bias, layer.out_proj.weight, layer.out_proj.bias)
    expected_grad = torch.autograd.grad(expected.square().mean(), variables)
    actual_grad = torch.autograd.grad(actual.square().mean(), variables)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
