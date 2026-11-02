"""Measure the attention-kernel change at identical weights, inputs and loss."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from benchmark_simplification import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", type=Path, required=True)
    parser.add_argument("--fused-root", type=Path, required=True)
    parser.add_argument("--input-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.manual_root))
    from trace_ovss.data.datasets import SegmentationPatchDataset, trace_collate
    from trace_ovss.cli import build_patch_image_ids, build_patch_valid_mask, set_global_seed
    from trace_ovss.method.trace import TRACE

    protocol = json.loads(args.input_protocol.read_text())
    config = protocol["config"]
    dataset = SegmentationPatchDataset(config["dataset"], config["data_dir"],
        resize=config["resize"], patch_size=config["patch_size"],
        patch_stride=config["patch_stride"], include_auxiliary=False)
    lookup = {str(image): i for i, (image, _) in enumerate(dataset.samples)}
    image_ids = protocol["image_ids"][:2]
    data = trace_collate([dataset[lookup[image]] for image in image_ids])
    vocabulary = dataset.vocabulary
    classes = vocabulary.extensions if config["class_extensions"] else vocabulary.classes
    set_global_seed(0)
    method = TRACE(classes=classes, lr=config["lr"], steps=config["steps"],
                   trace_max_regions=config["trace_max_regions"],
                   trace_view_batch_size=16, device="cuda")
    source = args.fused_root / "trace_ovss/models/dense_clip.py"
    spec = importlib.util.spec_from_file_location("trace_ovss.models.fused_probe", source)
    fused_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused_module)
    manual_attention = method.model._attention
    fused_attention = fused_module.DenseCLIP._attention.__get__(method.model)
    images = data["img_patches"].cuda()
    valid = build_patch_valid_mask(data["meta"], config["patch_size"], config["patch_stride"], "cuda")
    ids = build_patch_image_ids(data["meta"], "cuda")
    terms = method.estimate_trace_reliability(images, valid_mask=valid, image_ids=ids)["region_terms"]
    parameters = [p for p in method.model.parameters() if p.requires_grad]
    outputs, gradients = [], []
    for attention in (manual_attention, fused_attention):
        method.model._attention = attention
        with torch.no_grad():
            outputs.append(method.forward_eval_logits(images, interpolate=False).float())
        entropy = method.trace_softmax_entropy(method.forward_adapt_logits(images).float())
        loss = method.trace_region_normalized_entropy(entropy, terms)
        grads = torch.autograd.grad(loss, parameters)
        gradients.append(torch.cat([gradient.detach().float().flatten() for gradient in grads]))
    delta = outputs[1] - outputs[0]
    gdelta = gradients[1] - gradients[0]
    result = dict(device=str(torch.cuda.get_device_name()), torch_version=torch.__version__,
                  image_ids=image_ids, input_protocol=str(args.input_protocol),
                  input_protocol_sha256=hashlib.sha256(args.input_protocol.read_bytes()).hexdigest(),
                  fused_attention_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  max_absolute_logit_error=float(delta.abs().max()),
                  mean_absolute_logit_error=float(delta.abs().mean()),
                  token_argmax_agreement=float(outputs[0].argmax(1).eq(outputs[1].argmax(1)).float().mean()),
                  max_absolute_gradient_error=float(gdelta.abs().max()),
                  relative_gradient_l2_error=float(gdelta.norm()/gradients[0].norm().clamp_min(1e-12)),
                  fixed_candidate_terms=len(terms),
                  scope="Same pretrained weights and text cache; fixed manual reliability terms isolate kernel effects. No optimizer step or GT-dependent selection.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
