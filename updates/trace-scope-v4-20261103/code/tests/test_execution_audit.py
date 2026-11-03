"""CPU-only actual-work accounting, OOM fallback and audit lifetime tests."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.trace_execution_audit import ExecutionAudit
from test_cached_prefix import WorkCounter, tiny_method


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Execution-audit tests must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def assert_snapshot(snapshot):
    json.dumps(snapshot, allow_nan=False)
    for key, value in snapshot.items():
        if key.endswith("_by_layer"):
            values = json.loads(value)
            assert len(values) == snapshot["encoder_blocks"]
            assert all(type(x) is int and x >= 0 for x in values)
            assert sum(values) == snapshot[key.removesuffix("_by_layer")]
        else:
            assert type(value) is int and value >= 0
    for key in ("calls", "inputs", "block_calls", "block_inputs"):
        assert snapshot[key] == snapshot["successful_" + key] + snapshot["failed_" + key]
    for key in ("block_calls_by_layer", "block_inputs_by_layer"):
        assert json.loads(snapshot[key]) == [a + b for a, b in zip(
            json.loads(snapshot["successful_" + key]), json.loads(snapshot["failed_" + key]))]


@pytest.mark.parametrize("algorithm", ["quality_tail6", "quality_tail6_uncached", "quality_tail0"])
@pytest.mark.parametrize("failure", ["none", "before_blocks", "middle_block", "build_views"])
def test_real_episode_keeps_failed_work_and_successful_36_step_identity(monkeypatch, algorithm, failure):
    method = tiny_method(algorithm, steps=36)
    images = torch.randn(2, 3, 32, 32)
    independent = WorkCounter(method.model)
    failed_sizes, handles = [], []
    with ExecutionAudit(method) as audit:
        def fail_once(module, args, output):
            if method.model._cache_disabled_depth and not failed_sizes:
                batch = int(args[0].shape[0 if failure == "before_blocks" else 1])
                failed_sizes.append(batch)
                raise torch.cuda.OutOfMemoryError("CPU-injected diagnostic OOM")

        if failure == "before_blocks":
            handles.append(method.model.visual.conv1.register_forward_hook(fail_once))
        elif failure == "middle_block":
            handles.append(method.model.visual.transformer.resblocks[3].ln_1.register_forward_hook(fail_once))
        elif failure == "build_views":
            original = method.make_trace_weak_image

            def build(image):
                if not failed_sizes:
                    failed_sizes.append(len(image))
                    raise torch.cuda.OutOfMemoryError("CPU-injected view-building OOM")
                return original(image)

            monkeypatch.setattr(method, "make_trace_weak_image", build)
        try:
            losses = method.adapt(images)
            prediction = method.evaluate(images)
        finally:
            for handle in handles:
                handle.remove()
            independent.close()
        snapshot = audit.snapshot()
        assert len(losses) == 36
        assert torch.isfinite(prediction).all()
        assert_snapshot(snapshot)
        assert snapshot["calls"] == len(independent.stems)
        assert snapshot["inputs"] == sum(independent.stems)
        assert json.loads(snapshot["block_calls_by_layer"]) == [len(v) for v in independent.blocks]
        assert json.loads(snapshot["block_inputs_by_layer"]) == [sum(v) for v in independent.blocks]
        summary = method.last_reliability_summary
        assert snapshot["diagnostic_calls"] == summary["diagnostic_calls"]
        assert snapshot["diagnostic_calls"] > 0
        diagnostic = summary["diagnostic_views"]
        complete_inputs = diagnostic + len(images) * 38
        complete_calls = snapshot["diagnostic_calls"] + 38
        cached = algorithm != "quality_tail6_uncached"
        tail = 0 if algorithm == "quality_tail0" else 6
        prefix = 8 - tail
        stem_inputs = diagnostic + len(images) if cached else complete_inputs
        stem_calls = snapshot["diagnostic_calls"] + 1 if cached else complete_calls
        assert snapshot["successful_inputs"] == stem_inputs
        assert snapshot["successful_calls"] == stem_calls
        wanted_inputs = [stem_inputs] * prefix + [complete_inputs] * tail if cached else [complete_inputs] * 8
        wanted_calls = [stem_calls] * prefix + [complete_calls] * tail if cached else [complete_calls] * 8
        assert json.loads(snapshot["successful_block_inputs_by_layer"]) == wanted_inputs
        assert json.loads(snapshot["successful_block_calls_by_layer"]) == wanted_calls
        failed_forward = failure in {"before_blocks", "middle_block"}
        assert snapshot["failed_forward_attempts"] == int(failed_forward)
        assert snapshot["failed_calls"] == int(failed_forward)
        assert snapshot["failed_inputs"] == (failed_sizes[0] if failed_forward else 0)
        expected_failed = [failed_sizes[0]] * 4 + [0] * 4 if failure == "middle_block" else [0] * 8
        assert json.loads(snapshot["failed_block_inputs_by_layer"]) == expected_failed
        expected_calls = [1] * 4 + [0] * 4 if failure == "middle_block" else [0] * 8
        assert json.loads(snapshot["failed_block_calls_by_layer"]) == expected_calls
        audit.reset()
        empty = audit.snapshot()
        assert_snapshot(empty)
        assert all(value == 0 for key, value in empty.items()
                   if key != "encoder_blocks" and not key.endswith("_by_layer"))
        assert snapshot["inputs"] > 0


class NestedMethod:
    def __init__(self):
        self.model = SimpleNamespace(visual=SimpleNamespace(
            conv1=nn.Identity(), transformer=SimpleNamespace(resblocks=[
                SimpleNamespace(ln_1=nn.Identity()) for _ in range(3)])))

    def forward_eval_logits(self, image, *, nested=False, fail=False, recover=False):
        self.model.visual.conv1(image)
        if nested:
            try:
                self.forward_eval_logits(image, fail=fail)
            except torch.cuda.OutOfMemoryError:
                if not recover:
                    raise
        for block in self.model.visual.transformer.resblocks:
            block.ln_1(image.permute(1, 0, 2))
        if fail and not nested:
            raise torch.cuda.OutOfMemoryError("nested failure")
        return image

    def evaluate_trace_views(self, image, *, nested_scope=False, **kwargs):
        if nested_scope:
            self.evaluate_trace_views(image, **kwargs)
        return self.forward_eval_logits(image, **kwargs)


def test_nested_diagnostic_scopes_and_proposal_inference_exclusion():
    method = NestedMethod()
    image = torch.ones(2, 3, 4)
    with ExecutionAudit(method) as audit:
        method.forward_eval_logits(image)
        method.evaluate_trace_views(image, nested_scope=True)
        method.forward_eval_logits(image)
        snapshot = audit.snapshot()
    assert_snapshot(snapshot)
    assert snapshot["calls"] == 4
    assert snapshot["diagnostic_calls"] == 2
    assert snapshot["failed_forward_attempts"] == 0


@pytest.mark.parametrize("recover", [False, True])
def test_nested_forward_failure_is_not_double_counted(recover):
    method = NestedMethod()
    image = torch.ones(2, 3, 4)
    with ExecutionAudit(method) as audit:
        if recover:
            method.evaluate_trace_views(image, nested=True, fail=True, recover=True)
        else:
            with pytest.raises(torch.cuda.OutOfMemoryError, match="nested failure"):
                method.evaluate_trace_views(image, nested=True, fail=True)
        snapshot = audit.snapshot()
        assert_snapshot(snapshot)
        assert snapshot["failed_forward_attempts"] == 1
        assert snapshot["failed_calls"] == (1 if recover else 2)
        assert snapshot["diagnostic_calls"] == int(recover)
        assert snapshot["failed_block_calls"] == 3
        assert snapshot["successful_block_calls"] == (3 if recover else 0)
        audit.reset()
        method.forward_eval_logits(image)
        assert audit.snapshot()["failed_forward_attempts"] == 0


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_reset_or_close_inside_forward_is_rejected_without_losing_counts(operation):
    method = NestedMethod()
    image = torch.ones(2, 3, 4)
    with ExecutionAudit(method) as audit:
        def change(module, args, output):
            with pytest.raises(RuntimeError, match="active invocation"):
                getattr(audit, operation)()
        handle = method.model.visual.conv1.register_forward_hook(change)
        try:
            method.evaluate_trace_views(image)
        finally:
            handle.remove()
        assert audit.snapshot()["inputs"] == 2
        assert audit.snapshot()["diagnostic_calls"] == 1


def test_exception_before_encoder_has_zero_work_and_scope_is_restored():
    method = NestedMethod()
    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("before encoder")
    method.forward_eval_logits = fail
    with ExecutionAudit(method) as audit:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="before encoder"):
            method.evaluate_trace_views(torch.ones(2, 3, 4))
        snapshot = audit.snapshot()
        assert_snapshot(snapshot)
        assert snapshot["failed_forward_attempts"] == 1
        assert snapshot["failed_inputs"] == snapshot["failed_block_inputs"] == 0
        assert snapshot["diagnostic_calls"] == 0
        audit.reset()
    assert method.forward_eval_logits is fail


def test_non_oom_exception_propagates_without_being_reclassified():
    method = NestedMethod()
    error = ValueError("not an OOM")
    def fail(*args, **kwargs):
        raise error
    method.forward_eval_logits = fail
    with ExecutionAudit(method) as audit:
        with pytest.raises(ValueError) as raised:
            method.evaluate_trace_views(torch.ones(2, 3, 4))
        assert raised.value is error
        assert audit.snapshot()["failed_forward_attempts"] == 0
        audit.reset()


def test_close_restores_methods_hooks_and_snapshot_stays_detached():
    method = NestedMethod()
    before = dict(vars(method))
    hooks = method.model.visual.conv1._forward_pre_hooks
    with ExecutionAudit(method) as audit:
        with pytest.raises(RuntimeError, match="already has"):
            ExecutionAudit(method)
        method.evaluate_trace_views(torch.ones(2, 3, 4))
        snapshot = audit.snapshot()
        snapshot["inputs"] = -1
        assert audit.snapshot()["inputs"] == 2
        assert len(hooks) == 1
    assert vars(method) == before
    assert not hooks
    assert all(not block.ln_1._forward_pre_hooks for block in method.model.visual.transformer.resblocks)
    saved = audit.snapshot()
    method.evaluate_trace_views(torch.ones(2, 3, 4))
    assert audit.snapshot() == saved
    audit.close()
    with pytest.raises(RuntimeError, match="closed"):
        audit.reset()
    with pytest.raises(RuntimeError, match="closed"):
        audit.__enter__()
