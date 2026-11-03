import copy

import numpy as np
import pytest
import torch

from tools.evolution_fifty import generalization_runtime as runtime
from trace_ovss.method.image_balanced import image_balanced_region_entropy
from test_simplified_trace import tiny_trace
from test_image_full_worker import config, batch_meta


def method(backend="naclip"):
    base = tiny_trace(architecture="vanilla" if backend == "sclip" else "reduced")
    base.model.attention_strategy = "csa" if backend == "sclip" else "naclip"
    current = runtime.ExperimentTRACE(classes=base.classes, lr=base.lr, steps=base.steps,
        trace_max_regions=base.trace_max_regions, trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs, prompts=base.prompt_templates,
        model=base.model, tokenize_fn=base.tokenize, device="cpu", view_mode="C")
    current.model.eval()
    return current


@pytest.mark.parametrize("input_dtype",[torch.float32,torch.float16])
@pytest.mark.parametrize("batch",[1,3])
def test_official_sclip_entry_preserves_native_mixed_precision_and_original_input_gradient(input_dtype,batch):
    from trace_ovss.models._vendor_clip.model import convert_weights
    from tools.evolution_fifty.sclip_reference import SOURCE
    base = method("sclip")
    convert_weights(base.model.backbone)
    current = runtime.ExperimentTRACE(classes=base.classes,lr=base.lr,steps=base.steps,
        trace_max_regions=base.trace_max_regions,trace_view_batch_size=base.trace_view_batch_size,
        vision_outputs=base.vision_outputs,prompts=base.prompt_templates,model=base.model,
        tokenize_fn=base.tokenize,device="cpu",view_mode="C")
    current.model.eval()
    image = torch.randn(batch,3,32,32).to(input_dtype)
    assert current.model.visual.conv1.weight.dtype == torch.float16
    assert current.model.visual.ln_pre.weight.dtype == torch.float32
    before = runtime.shared.restored_state(current)
    report = runtime.official_sclip_gate(current,image,SOURCE)
    assert report["feature_max_abs"] == report["input_and_visual_LN_gradient_max_abs"] == 0
    assert report["changed_query_labels"] == 0
    assert report["input_dtype"] == str(input_dtype) and report["reference_weight_dtype"] == "torch.float16"
    assert current.model.visual.conv1.weight.dtype == torch.float16
    assert current.model.visual.ln_pre.weight.dtype == torch.float32
    assert image.dtype == input_dtype and not image.requires_grad
    assert runtime.shared.restored_state(current) == before
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["naclip", "sclip"])
@pytest.mark.parametrize("batch", [1, 4])
def test_direct_reference_actual_tiny_model_loss_gradient_parameter_Adam_label_parity(backend, batch):
    current = method(backend)
    reference = runtime.make_reference(current)
    result = runtime.compare_gate(current, reference, torch.randn(batch, 3, 32, 32), None,
        torch.arange(batch), batch_meta(batch), config(batch), None, (2, 2))
    assert set(result) == set(runtime.ARMS)
    for arm, row in result.items():
        assert row["passed"] and row["changed_labels"] == 0
        assert max(row["max_abs"].values()) <= 1e-5
        assert row["actual_counts"]["steps"] == (0 if arm == "no_adapt" else 2)
    runtime.shared.restored_state(current)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("backend", ["naclip", "sclip"])
@pytest.mark.parametrize("arm", runtime.ARMS)
def test_three_arms_have_exact_real_forward_schedules_and_reset(backend, arm):
    current = method(backend)
    before = runtime.shared.restored_state(current)
    images = torch.randn(3, 3, 32, 32)
    predictions, stats, losses, counts = runtime.run_arm(current, arm, images, None,
        torch.arange(3), batch_meta(3), config(3), None)
    assert len(predictions) == 3 and all(x.shape == (32, 32) for x in predictions)
    if arm == "no_adapt":
        assert stats == losses == [] and counts["candidate_sha256"] is None
        assert counts["encoder_calls"] == counts["eval_forward_calls"] == 1
        assert runtime.shared.restored_state(current) == before
    else:
        assert stats and len(losses) == 2
        assert counts["diagnostic_encoder_inputs"] == (len(stats) if arm == "image_C" else 0)
    current.reset()
    assert runtime.shared.restored_state(current) == before


@pytest.mark.parametrize("backend", ["naclip", "sclip"])
def test_legitimate_used_visual_membership_does_not_force_equal_counts(backend):
    current = method(backend)
    record = runtime.legal_parameters(current)
    assert record["final_MLP_used"] == (backend == "sclip")
    final = current.model.visual.transformer.resblocks[-1].ln_2
    assert final.weight.requires_grad == (backend == "sclip")
    foreign = current.model.token_embedding.weight
    foreign.requires_grad_(True)
    with pytest.raises(ValueError, match="membership differs"):
        runtime.legal_parameters(current)


@pytest.mark.parametrize("shape", [(3, 2, 4, 4), (2, 4, 4)])
def test_explicit_reference_matches_native_source_balancing_and_EPS(shape):
    torch.manual_seed(3)
    entropy = torch.randn(*shape, requires_grad=True)
    terms = [dict(image_idx=source, batch_idx=source, rho=rho, indices=torch.tensor(indices))
        for source,rho,indices in ((0,.5,[0, 1]), (0,.2,[2, 3, 4]), (1,.1,[5]), (1,0.,[2]), (1,1e-8,[7]))]
    expected = image_balanced_region_entropy(entropy, terms)
    actual = runtime.direct_objective(entropy, terms)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(expected, entropy)[0],
        torch.autograd.grad(actual, entropy)[0], rtol=0, atol=0)


def test_gate_rejects_nonfinite_before_any_optimizer_step_and_cleans_hooks():
    current = method()
    reference = runtime.make_reference(current)
    parameter = next(p for p in current.model.parameters() if p.requires_grad)
    handle = parameter.register_hook(lambda gradient: gradient*float("nan"))
    try:
        with pytest.raises(RuntimeError, match="Nonfinite"):
            runtime.compare_gate(current, reference, torch.randn(2, 3, 32, 32), None,
                torch.arange(2), batch_meta(2), config(2), None, (2, 2))
    finally:
        handle.remove()
    assert not current.model._forward_hooks
    assert not current.optimizer._optimizer_step_post_hooks
    assert not reference.optimizer._optimizer_step_post_hooks


def test_gate_rejects_wrong_backbone_token_grid_and_vacuous_support():
    current = method()
    reference = runtime.make_reference(current)
    images = torch.randn(2, 3, 32, 32)
    with pytest.raises(ValueError, match="token grid"):
        runtime.compare_gate(current, reference, images, None, torch.arange(2), batch_meta(2), config(2), None, (14,14))
    with pytest.raises(ValueError, match="Vacuous"):
        runtime.compare_gate(current, reference, images, torch.zeros(2, 1, 32, 32, dtype=torch.bool),
            torch.arange(2), batch_meta(2), config(2), None, (2,2))


@pytest.mark.parametrize("counter", ["adapt_forward_calls", "encoder_calls", "diagnostic_encoder_inputs"])
def test_no_adapt_rejects_hidden_diagnostic_or_update_compute(counter):
    current = method()
    *_, row = runtime.run_arm(current, "no_adapt", torch.randn(2, 3, 32, 32), None,
        torch.arange(2), batch_meta(2), config(2), None)
    row[counter] += 1
    with pytest.raises(ValueError, match="NoAdapt performed"):
        runtime.validate_cost(row, "no_adapt", config(2))


def test_builder_rejects_layer_policy_or_checkpoint_drift_before_construction(monkeypatch):
    monkeypatch.setattr(runtime, "checked_checkpoint", lambda *_: ("known", "weights"))
    monkeypatch.setattr(runtime, "ExperimentTRACE", lambda **_: pytest.fail("Invalid policy reached constructor"))
    spec = dict(backbone="ViT-B/16", config={}, vision_outputs=list(range(-1,-19,-1)),
        model_type="naclip", checkpoint=dict(backbone="ViT-B/16"))
    with pytest.raises(ValueError, match="layer policy"):
        runtime.build_method(spec)
    monkeypatch.setattr(runtime, "checked_checkpoint", lambda *_: (_ for _ in ()).throw(ValueError("Checkpoint drift")))
    with pytest.raises(ValueError, match="Checkpoint drift"):
        runtime.build_method(spec)


def test_official_SCLIP_component_core_uses_real_methods_but_only_tiny_CPU_model():
    from tools.evolution_fifty.sclip_reference import SOURCE
    current = method("sclip")
    report = runtime.official_sclip_gate(current, torch.randn(2,3,32,32), SOURCE)
    assert report["passed"] and report["changed_query_labels"] == report["adaptation_steps"] == 0
    assert report["feature_max_abs"] == report["input_and_visual_LN_gradient_max_abs"] == 0.
    assert not torch.cuda.is_initialized()


def test_common_graph_disconnection_cannot_silently_drop_legal_LN_gradients():
    current = method()
    reference = runtime.make_reference(current)
    handle = current.model.visual.transformer.resblocks[-1].ln_1.register_forward_hook(
        lambda module,args,output: output.detach())
    try:
        with pytest.raises(ValueError, match="missing a legal visual-LN gradient"):
            runtime.compare_gate(current,reference,torch.randn(2,3,32,32),None,
                torch.arange(2),batch_meta(2),config(2),None,(2,2))
    finally:
        handle.remove()
    assert not current.optimizer._optimizer_step_pre_hooks
    assert not reference.optimizer._optimizer_step_pre_hooks


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_gate_rejects_nonfinite_candidate_support_even_when_updates_are_finite(bad):
    from types import MethodType
    current = method()
    reference = runtime.make_reference(current)
    original = current.adapt
    def corrupt(self,*args,**kwargs):
        losses = original(*args,**kwargs)
        if self.arm == "image_C":
            self.last_candidate_stats[0]["counterfactual_support"] = bad
        return losses
    current.adapt = MethodType(corrupt,current)
    with pytest.raises(ValueError, match="Nonfinite measured"):
        runtime.compare_gate(current,reference,torch.randn(2,3,32,32),None,
            torch.arange(2),batch_meta(2),config(2),None,(2,2))
