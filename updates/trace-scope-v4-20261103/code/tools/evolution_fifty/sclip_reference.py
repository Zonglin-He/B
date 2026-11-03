"""Bounded CPU structural/numeric check against pinned official SCLIP methods.

Uses random tiny ViT weights, not a real checkpoint/model gate or full score.
Only exact AST method bodies are loaded; no upstream project is restored.
"""

from __future__ import annotations

import argparse
import ast
import copy
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from trace_ovss.models._vendor_clip.model import CLIP
from trace_ovss.models.dense_clip import DenseCLIP
from .bootstrap import require
from .intake import sha256, write_new

SOURCE_COMMIT = "3608360267b6130c1ef18090d7289f17c771cb90"
SOURCE_SHA = "f90c1d6fd73009009dd92bd7dcf0e02db361314746c224fd0805da52af21788e"
SOURCE = Path("/home/wwww/hzl/results/trace_mechanism_20261102/G04_SCLIP_source_v1/model.py")


def reference_methods(source=SOURCE):
    require(sha256(source) == SOURCE_SHA, "Official SCLIP source identity mismatch")
    tree = ast.parse(Path(source).read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VisionTransformer"]
    require(len(classes) == 1, "Official visual class missing/duplicated")
    wanted = ("custom_attn", "forward", "interpolate_pos_encoding")
    nodes = [copy.deepcopy(node) for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    require(len(nodes) == len(wanted) and {n.name for n in nodes} == set(wanted), "Official method set differs")
    namespace = dict(torch=torch, nn=nn, np=np, math=math, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return {name:namespace[name] for name in wanted}


def max_gradient_difference(a, b):
    require(a.keys() == b.keys(), "Tiny visual parameter membership differs")
    maximum = 0.
    for name in a:
        left, right = a[name].grad, b[name].grad
        require((left is None) == (right is None), "Tiny visual gradient membership differs")
        if left is not None:
            maximum = max(maximum, float((left-right).abs().max()))
    return maximum


def audit(output):
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output, "Fresh SCLIP CPU reference result required")
    output.mkdir(parents=True)
    write_new(output/"PROTOCOL.json", dict(scope="CPU tiny random ViT structural/numeric parity only",
        source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA,
        source_url=f"https://github.com/wangf3014/SCLIP/blob/{SOURCE_COMMIT}/clip/model.py",
        extracted_methods=["custom_attn", "forward", "interpolate_pos_encoding"],
        source_code_sha256=sha256(__file__), shapes=[[32,32],[48,32]],
        model="Random12block width64/patch16/embed16 visual model; same weights in both paths",
        comparison="Actual local vanilla/csa DenseCLIP final selected projected features vs exact official SCLIP forward body on same visual backbone",
        gradient="Square-mean feature loss; input and every used visual parameter, no GT",
        threshold=1e-5, changed_labels=0, real_checkpoint_loaded=False, GPU_used=False,
        not_proven=["actual B/16 checkpoint equivalence", "formal whole9layerfusion equivalence to published SCLIP", "dataset accuracy", "GPU artifact admission"]))
    methods = reference_methods()
    rows = []
    for shape in ((32,32),(48,32)):
        torch.manual_seed(71)
        backbone = CLIP(embed_dim=16, image_resolution=32, vision_layers=12, vision_width=64,
            vision_patch_size=16, context_length=8, vocab_size=32, transformer_width=64,
            transformer_heads=1, transformer_layers=1)
        model = DenseCLIP(backbone, architecture="vanilla", attention_strategy="csa").eval()
        reference = copy.deepcopy(backbone.visual).eval()
        reference.custom_attn = MethodType(methods["custom_attn"], reference)
        reference.interpolate_pos_encoding = MethodType(methods["interpolate_pos_encoding"], reference)
        reference.forward = MethodType(methods["forward"], reference)
        image = torch.randn(2,3,*shape, requires_grad=True)
        other = image.detach().clone().requires_grad_()
        actual = model.encode_dense_layers(image, (-1,), project=True)[0]
        expected = reference(other, return_all=True, csa=True)
        actual.square().mean().backward()
        expected.square().mean().backward()
        raw = float((actual-expected).abs().max())
        gradients = max_gradient_difference(dict(backbone.visual.named_parameters()), dict(reference.named_parameters()))
        input_grad = float((image.grad-other.grad).abs().max())
        text = torch.randn(16,3)
        labels = int(((F.normalize(actual.detach(),dim=-1)@text).argmax(-1)
                     != (F.normalize(expected.detach(),dim=-1)@text).argmax(-1)).sum())
        passed = max(raw, gradients, input_grad) <= 1e-5 and labels == 0
        rows.append(dict(shape=list(shape), feature_max_abs=raw, parameter_gradient_max_abs=gradients,
            input_gradient_max_abs=input_grad, changed_synthetic_labels=labels, passed=passed))
    require(not torch.cuda.is_initialized(), "SCLIP reference audit must not initialize CUDA")
    write_new(output/"RESULT.json", dict(all_passed=all(r["passed"] for r in rows), rows=rows,
        real_checkpoint_loaded=False, GT_read=False, GPU_used=False,
        scope="Final-layer structural parity on random CPU tiny models only; actual full B/16 gates still required"))
    write_new(output/"ANALYSIS_SEALED.json", dict(complete=True, scope="CPU component audit, not experiment results",
        artifact_sha256={name:sha256(output/name) for name in ("PROTOCOL.json", "RESULT.json")},
        real_GPU_or_model_admission=False))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(audit(parser.parse_args().output))
