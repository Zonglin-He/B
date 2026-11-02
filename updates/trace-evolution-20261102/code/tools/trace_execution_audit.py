"""Observe encoder work, including partial forwards abandoned by OOM fallback.

``ExecutionAudit(method)`` instruments conv1 and every block's ln_1. Call
``reset()`` between episodes, ``snapshot()`` after adaptation and evaluation,
and ``close()`` in a finally block (or use the audit as a context manager).

Totals include every observed attempt. ``failed_*`` contains only work inside
``forward_eval_logits`` invocations that raise ``torch.cuda.OutOfMemoryError``.
An OOM while building pixel views never enters that wrapper, so it contributes
no failed forward attempt or encoder work; the caller's wall time still includes
it. Other exceptions propagate without being classified as recovered OOMs.
``successful_*`` means total minus that recorded failure subset, not proof that
an episode completed. Hooks count module entries, not FLOPs or completed blocks.

Nested scopes and forwards are supported without double-counting failed work.
Use one active audit per method, with serial execution. Reset/close during an
instrumented invocation is rejected rather than silently splitting its counts.
"""

from __future__ import annotations

from functools import wraps
import json

import torch


_MISSING = object()
_OWNER_ATTRIBUTE = "_trace_execution_audit"
_SCALARS = ("calls", "inputs")
_VECTORS = ("block_calls_by_layer", "block_inputs_by_layer")


def _empty_counts(depth):
    return dict(calls=0, inputs=0, block_calls_by_layer=[0] * depth,
                block_inputs_by_layer=[0] * depth)


def _copy_counts(counts):
    return {key: value.copy() if isinstance(value, list) else value
            for key, value in counts.items()}


class ExecutionAudit:
    """Per-episode stem/block counters and successful diagnostic batch counts.

    Snapshot keys are ``calls``, ``inputs``, ``block_calls``, ``block_inputs``,
    ``block_calls_by_layer``, and ``block_inputs_by_layer``, each also available
    with ``failed_`` and ``successful_`` prefixes. Vectors are JSON strings;
    all other values, including ``encoder_blocks``, ``failed_forward_attempts``
    and ``diagnostic_calls``, are Python integers. ``diagnostic_calls`` counts
    successful outermost eval forwards inside ``evaluate_trace_views`` only.
    A nested propagation of the same OOM is one failed forward attempt.
    """

    def __init__(self, method):
        if getattr(method, _OWNER_ATTRIBUTE, None) is not None:
            raise RuntimeError("This method already has an active execution audit")
        visual = method.model.visual
        blocks = tuple(visual.transformer.resblocks)
        self.method = method
        self.encoder_blocks = len(blocks)
        self._handles = []
        self._wrapped = {}
        self._original_attributes = {}
        self._forward_depth = 0
        self._diagnostic_depth = 0
        self._seen_ooms = []
        self._closed = False
        self.reset()

        original_eval = method.forward_eval_logits
        original_views = method.evaluate_trace_views
        for name in ("forward_eval_logits", "evaluate_trace_views"):
            self._original_attributes[name] = vars(method).get(name, _MISSING)

        @wraps(original_eval)
        def forward_eval(*args, **kwargs):
            outermost = self._forward_depth == 0
            diagnostic = self._diagnostic_depth > 0
            before = _copy_counts(self._total)
            failed_before = _copy_counts(self._failed)
            self._forward_depth += 1
            try:
                result = original_eval(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as error:
                self._record_failure(before, failed_before)
                if not any(error is previous for previous in self._seen_ooms):
                    self._failed_forward_attempts += 1
                    self._seen_ooms.append(error)
                raise
            else:
                if outermost and diagnostic:
                    self._diagnostic_calls += 1
                return result
            finally:
                self._forward_depth -= 1
                if outermost:
                    self._seen_ooms.clear()

        @wraps(original_views)
        def evaluate_views(*args, **kwargs):
            self._diagnostic_depth += 1
            try:
                return original_views(*args, **kwargs)
            finally:
                self._diagnostic_depth -= 1

        self._wrapped.update(forward_eval_logits=forward_eval,
                             evaluate_trace_views=evaluate_views)
        try:
            self._handles.append(visual.conv1.register_forward_pre_hook(self._count_stem))
            for index, block in enumerate(blocks):
                self._handles.append(block.ln_1.register_forward_pre_hook(self._block_hook(index)))
            for name, wrapper in self._wrapped.items():
                setattr(method, name, wrapper)
            setattr(method, _OWNER_ATTRIBUTE, self)
        except BaseException:
            self.close()
            raise

    def _count_stem(self, module, args):
        self._total["calls"] += 1
        self._total["inputs"] += int(args[0].shape[0])

    def _block_hook(self, index):
        def count(module, args):
            self._total["block_calls_by_layer"][index] += 1
            self._total["block_inputs_by_layer"][index] += int(args[0].shape[1])
        return count

    def _record_failure(self, before, failed_before):
        # Nested failures may already account for part of this invocation.
        for key in _SCALARS:
            self._failed[key] += (self._total[key] - before[key]
                                  - (self._failed[key] - failed_before[key]))
        for key in _VECTORS:
            for index in range(self.encoder_blocks):
                self._failed[key][index] += (
                    self._total[key][index] - before[key][index]
                    - (self._failed[key][index] - failed_before[key][index]))

    def _require_idle(self):
        if self._forward_depth or self._diagnostic_depth:
            raise RuntimeError("Cannot reset or close an execution audit during an active invocation")

    def reset(self):
        """Clear all episode counters; does not reset the model or its cache."""
        self._require_idle()
        if self._closed:
            raise RuntimeError("Cannot reset a closed execution audit")
        self._total = _empty_counts(self.encoder_blocks)
        self._failed = _empty_counts(self.encoder_blocks)
        self._failed_forward_attempts = 0
        self._diagnostic_calls = 0
        self._seen_ooms.clear()

    def snapshot(self):
        """Return detached JSON-compatible counters without changing any state."""
        successful = {key: self._total[key] - self._failed[key] for key in _SCALARS}
        successful.update({key: [total - failed for total, failed in zip(
            self._total[key], self._failed[key])] for key in _VECTORS})
        result = dict(encoder_blocks=self.encoder_blocks,
                      failed_forward_attempts=self._failed_forward_attempts,
                      diagnostic_calls=self._diagnostic_calls)
        for prefix, counts in (("", self._total), ("failed_", self._failed),
                               ("successful_", successful)):
            result.update({prefix + key: counts[key] for key in _SCALARS})
            for key in _VECTORS:
                result[prefix + key] = json.dumps(counts[key])
                result[prefix + key.removesuffix("_by_layer")] = sum(counts[key])
        return result

    def close(self):
        """Remove hooks and restore original instance/class method resolution."""
        if self._closed:
            return
        self._require_idle()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for name, wrapper in self._wrapped.items():
            if vars(self.method).get(name, _MISSING) is not wrapper:
                continue
            original = self._original_attributes[name]
            if original is _MISSING:
                delattr(self.method, name)
            else:
                setattr(self.method, name, original)
        if getattr(self.method, _OWNER_ATTRIBUTE, None) is self:
            delattr(self.method, _OWNER_ATTRIBUTE)
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot enter a closed execution audit")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
