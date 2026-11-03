"""Fixed-state M01 readout helpers; no adaptation, annotations or executable queue.

These helpers do not grant artifact admission. A future replay controller must
bind the parent prediction barrier, snapshot/prediction hashes and RGB inputs
before calling them; GT support extraction belongs to the later offline phase.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import zipfile

import numpy as np
import torch

from tools.evolution import run_cell as shared
from .bootstrap import require


def decode_ln_snapshot(content, expected_sha256, parameters):
    """Validate the exact saved trainable membership before copying any tensor."""
    require(isinstance(content, bytes) and 0 < len(content) <= 64*1024**2,
            "Invalid or oversized LN snapshot")
    require(hashlib.sha256(content).hexdigest() == expected_sha256,
            "LN snapshot digest differs")
    require(bool(parameters), "Missing expected trainable LN parameters")
    expected_bytes = {n: p.numel()*p.element_size() for n, p in parameters.items()}
    require(sum(expected_bytes.values()) <= 16*1024**2, "Trainable replay state exceeds byte budget")
    with zipfile.ZipFile(io.BytesIO(content)) as zipped:
        members = zipped.infolist()
        require(len(members) == len(parameters)
                and {m.filename for m in members} == {n+".npy" for n in parameters},
                "LN ZIP membership differs")
        for member in members:
            name = member.filename[:-4]
            parameter = parameters[name]
            require(member.file_size <= expected_bytes[name]+4096,
                    "LN member uncompressed size exceeds expected budget")
            with zipped.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "Unsupported snapshot NPY version")
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                require(shape == tuple(parameter.shape) and not fortran and not dtype.hasobject
                        and dtype == parameter.detach().cpu().numpy().dtype
                        and member.file_size == stream.tell()+expected_bytes[name],
                        "LN member header/size differs before allocation")
    output = {}
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files))
                and set(archive.files) == set(parameters), "LN snapshot membership differs")
        for name, parameter in parameters.items():
            value = archive[name]
            expected_dtype = parameter.detach().cpu().numpy().dtype
            require(tuple(value.shape) == tuple(parameter.shape)
                    and value.dtype == expected_dtype and np.issubdtype(value.dtype, np.floating)
                    and bool(np.isfinite(value).all()), "LN snapshot shape/dtype/finite check failed")
            output[name] = torch.from_numpy(value.copy())
    return output


@contextmanager
def fixed_ln_state(method, snapshot):
    """Restore LN/Adam, and retire the method on ANY readout/integrity failure.

    Frozen tensors are byte-digested as well as version-checked. They are not
    backed up: a failed method must be discarded, never caught and reused.
    These integrity costs must be measured separately by a future controller.
    """
    require(not getattr(method, "_trace_replay_invalid", False), "Replay method retired after failure")
    require(not getattr(method, "_trace_replay_guard_active", False), "Nested replay guards forbidden")
    model = method.model
    require(all(not module.training for module in model.modules()), "Readout model must be eval-only")
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(bool(parameters) and set(snapshot) == set(parameters), "Replay trainable membership differs")
    require(all(p.grad is None for p in model.parameters()), "Readout starts with existing gradients")
    for name, parameter in parameters.items():
        value = snapshot[name]
        require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                and value.dtype == parameter.dtype and bool(torch.isfinite(value).all()),
                "Replay snapshot tensor differs")
    original = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer = method.optimizer
    adam_original = copy.deepcopy(optimizer.state_dict())
    adam_before = shared.state_digest(method.optimizer.state_dict())
    def structure():
        return (tuple((n, id(p), p.requires_grad, p.shape, p.dtype, p.device)
                      for n, p in model.named_parameters(remove_duplicate=False)),
                tuple((n, id(v), v.shape, v.dtype, v.device)
                      for n, v in model.named_buffers(remove_duplicate=False)),
                tuple((n, id(m), m.training) for n, m in model.named_modules(remove_duplicate=False)),
                tuple(tuple(id(p) for p in group["params"]) for group in optimizer.param_groups),
                tuple(sorted(id(p) for p in optimizer.state)))
    initial_structure = structure()
    def frozen_digest():
        return shared.state_digest(dict(
            frozen={n: p for n, p in model.named_parameters() if n not in parameters},
            buffers=dict(model.named_buffers())))
    frozen_before = frozen_digest()
    failure = None
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(snapshot[name].to(device=parameter.device))
        versions = {name: p._version for name, p in model.named_parameters()}
        buffer_versions = {name: value._version for name, value in model.named_buffers()}
        method._trace_replay_guard_active = True
        with torch.no_grad():
            yield
        require(method.model is model and method.optimizer is optimizer and structure() == initial_structure,
                "Readout mutated model/optimizer structure or training state")
        require(all(p._version == versions[name] for name, p in model.named_parameters())
                and all(v._version == buffer_versions[name] for name, v in model.named_buffers()),
                "Fixed-state readout mutated model tensors")
        require(frozen_digest() == frozen_before, "Readout mutated frozen tensor content")
        require(all(torch.equal(p.detach(), snapshot[name].to(device=p.device))
                    for name, p in parameters.items()), "Readout changed the loaded LN state")
        require(shared.state_digest(optimizer.state_dict()) == adam_before,
                "Fixed-state readout mutated Adam state")
        require(all(p.grad is None for p in model.parameters()), "Readout created parameter gradients")
    except BaseException as error:
        failure = error
        method._trace_replay_invalid = True
    finally:
        method._trace_replay_guard_active = False
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(original[name])
            if shared.state_digest(optimizer.state_dict()) != adam_before:
                optimizer.load_state_dict(adam_original)
            for parameter in model.parameters():
                parameter.grad = None
            require(shared.state_digest(optimizer.state_dict()) == adam_before,
                    "Replay Adam cleanup failed")
        except BaseException as cleanup:
            method._trace_replay_invalid = True
            if failure is None:
                failure = cleanup
            else:
                # Python3.10 has no BaseException.add_note; preserve original.
                failure.replay_cleanup_failure = f"{type(cleanup).__name__}: {cleanup}"
        if failure is not None:
            raise failure


def validate_geometry(logits, meta, config):
    grids, shapes = meta["patch_grid_shape"], meta["img_shape"]
    require(len(grids) == len(shapes) > 0, "Readout metadata cardinality differs")
    patch = config["patch_size"]
    stride = config["patch_stride"]
    require(len(patch) == 2 and all(type(v) is int and v > 0 for v in patch)
            and type(stride) is int and 0 < stride <= min(patch), "Invalid patch/stride geometry")
    total = 0
    for grid, shape in zip(grids, shapes, strict=True):
        require(len(grid) == len(shape) == 2
                and all(type(v) is int and v > 0 for v in (*grid, *shape)), "Invalid source geometry")
        require(all((g-1)*stride+p >= s and (g == 1 or (g-1)*stride < s)
                    for g, s, p in zip(grid, shape, patch, strict=True)), "Patch grid leaves uncovered or empty crop")
        total += grid[0]*grid[1]
    require(total == logits.shape[0] and tuple(logits.shape[-2:]) == tuple(patch),
            "Readout patch count/spatial shape differs")


def source_support(method, images, meta, config, mapping, source_index):
    """Same production aggregation/softmax/max-extension collapse as hard readout.

Collapsed class support need not sum to one: extensions are MAX-reduced,
not summed or re-normalized. Do not change this when measuring GT support.
"""
    from trace_ovss.cli import collapse_class_extensions
    from trace_ovss.evaluation import aggregate_patch_logits

    require(not getattr(method, "_trace_replay_invalid", False), "Replay method retired after failure")
    require(getattr(method, "_trace_replay_guard_active", False), "Readout requires an active fixed-state guard")
    require(type(source_index) is int and 0 <= source_index < len(meta["img_shape"]),
            "Invalid readout source index")
    with torch.no_grad():
        logits = method.evaluate(images)
        require(logits.ndim == 4 and bool(torch.isfinite(logits).all()), "Invalid readout logits")
        validate_geometry(logits, meta, config)
        if mapping is not None:
            require(len(mapping) == logits.shape[1]
                    and all(type(v) is int and 0 <= v < logits.shape[1] for v in mapping)
                    and set(mapping) == set(range(len(set(mapping)))), "Invalid benchmark query mapping")
        outputs = aggregate_patch_logits(logits, meta["patch_grid_shape"], meta["img_shape"],
                                         config["patch_size"], config["patch_stride"])
        probabilities = outputs[source_index].softmax(0)
        if mapping is not None:
            probabilities = collapse_class_extensions(probabilities, mapping)
        require(bool(torch.isfinite(probabilities).all())
                and bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                "Invalid reconstructed support")
        return probabilities.cpu().numpy()


def replay_one(method, images, meta, config, mapping, source_index, snapshot, sealed_labels):
    """Reject hard-label mismatch before returning any fixed-state soft support."""
    expected = np.asarray(sealed_labels)
    require(expected.ndim == 2 and np.issubdtype(expected.dtype, np.integer),
            "Sealed integer hard prediction required")
    with fixed_ln_state(method, snapshot):
        probabilities = source_support(method, images, meta, config, mapping, source_index)
        labels = probabilities.argmax(0)
        require(labels.shape == expected.shape and np.array_equal(labels, expected),
                "Fixed-state replay changed sealed hard labels")
    return probabilities


def offline_true_support(probabilities, gt):
    """Offline extraction after parent trajectory seal and per-readout hard gate.

    The independently admitted controller owns phase and provenance enforcement;
    this function neither reads files nor sends GT back to model/readout.
    """
    values, gt = np.asarray(probabilities), np.asarray(gt)
    require(values.ndim == 3 and np.issubdtype(values.dtype, np.floating)
            and gt.ndim == 2 and values.shape[1:] == gt.shape
            and np.issubdtype(gt.dtype, np.integer), "Offline GT/support geometry or dtype differs")
    require(bool(np.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()),
            "Invalid offline class support")
    valid = gt != 255
    require(bool(((gt[valid] >= 0) & (gt[valid] < len(values))).all()), "Offline GT class outside mapping")
    selected = np.full(gt.shape, np.nan, dtype=values.dtype)
    y, x = np.nonzero(valid)
    selected[y, x] = values[gt[y, x], y, x]
    return selected
