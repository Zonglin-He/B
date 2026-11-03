"""E30 CPU algebra checks of the production objective, not a real-model gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import torch

from trace_ovss.method.image_balanced import image_balanced_region_entropy
from trace_ovss.method.trace.constants import EPS
from .intake import sha256, write_new


def value_and_gradient(entropy, terms):
    x = entropy.detach().clone().requires_grad_()
    value = image_balanced_region_entropy(x, terms)
    return value.detach(), torch.autograd.grad(value, x)[0]


def checks(dtype):
    entropy = torch.linspace(.1, 2., 32, dtype=dtype).reshape(2, 4, 4)
    terms = [dict(image_idx=0, batch_idx=0, rho=.2, indices=torch.tensor([0, 1, 2])),
             dict(image_idx=1, batch_idx=1, rho=.9, indices=torch.tensor([3, 4, 5, 6]))]
    loss, gradient = value_and_gradient(entropy, terms)
    uniform = [dict(t, rho=1.) for t in terms]
    expected_loss, expected_gradient = value_and_gradient(entropy, uniform)
    rows = [dict(check="singleton_weight_cancellation", dtype=str(dtype),
        loss_abs_error=float(abs(loss-expected_loss)),
        gradient_max_abs_error=float((gradient-expected_gradient).abs().max()))]
    multi = terms + [dict(image_idx=0, batch_idx=0, rho=.7, indices=torch.tensor([7, 8]))]
    base_loss, base_gradient = value_and_gradient(entropy, multi)
    for scale in (.25, 2., 8.):
        a, b = value_and_gradient(entropy, [dict(t, rho=t["rho"]*scale) for t in multi])
        rows.append(dict(check="common_positive_scaling_unchanged_membership", scale=scale,
            dtype=str(dtype), loss_abs_error=float(abs(a-base_loss)),
            gradient_max_abs_error=float((b-base_gradient).abs().max())))
    inactive = [dict(t, rho=EPS) for t in terms]
    a, b = value_and_gradient(entropy, inactive)
    rows.append(dict(check="all_at_EPS_inactive_differentiable_zero", dtype=str(dtype),
        loss_abs_error=float(abs(a)), gradient_max_abs_error=float(b.abs().max())))
    for row in rows:
        if row["loss_abs_error"] > 1e-5 or row["gradient_max_abs_error"] > 1e-5:
            raise ValueError("Production objective algebra mismatch: " + repr(row))
    return rows


def run(output):
    output = Path(output).absolute()
    if output.exists() or output != output.resolve():
        raise ValueError("New canonical E30 output required")
    output.mkdir(parents=True)
    files = [Path(__file__), Path(__file__).parents[2] / "trace_ovss/method/image_balanced.py",
             Path(__file__).parents[2] / "trace_ovss/method/trace/objective.py"]
    write_new(output / "PROTOCOL.json", dict(experiment="E30", device="cpu", tolerance=1e-5,
        dtypes=["torch.float32", "torch.float64"], tests=["singleton", "scaling", "EPS"],
        source_sha256={str(p): sha256(p) for p in files},
        real_model_identity_and_label_checks="pending; not established by this algebra test"))
    with mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA")), \
            mock.patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("No model forward")):
        rows = checks(torch.float32) + checks(torch.float64)
    write_new(output / "result.json", dict(experiment="E30", status="partial_cpu_algebra_passed",
        real_model_gate_passed=False, final_labels_compared=False, rows=rows))
    write_new(output / "CPU_ALGEBRA_SEALED.json", dict(complete=True, scope="CPU algebra only",
        artifact_sha256={n: sha256(output / n) for n in ("PROTOCOL.json", "result.json")}))
    print(dict(output=str(output), cpu_checks=len(rows), real_model_gate_passed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
