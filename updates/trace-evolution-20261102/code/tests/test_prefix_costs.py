"""Independent accounting checks for partial-encoder execution."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_algorithm_efficiency_suite import validate_block_costs, worker_command


def measurements(*, tail=6, cached=True, steps=36):
    patches, diagnostic, depth = 16, 100, 24
    stem = diagnostic + patches * (1 if cached else steps + 2)
    full = diagnostic + patches * (steps + 2)
    inputs = [stem] * (depth - tail) + [full] * tail if cached else [full] * depth
    diagnostic_calls = 7
    calls = [diagnostic_calls + 1] * (depth - tail) + [diagnostic_calls + steps + 2] * tail if cached else [diagnostic_calls + steps + 2] * depth
    row = dict(diagnostic_views=diagnostic, update_steps=steps, inputs=stem,
               calls=diagnostic_calls + (1 if cached else steps + 2))
    raw = dict(encoder_blocks=depth, patches=patches, block_inputs=sum(inputs),
               block_calls=sum(calls), block_inputs_by_layer=json.dumps(inputs),
               block_calls_by_layer=json.dumps(calls), diagnostic_calls=diagnostic_calls)
    return raw, row


@pytest.mark.parametrize("tail", [0, 6, 12])
def test_cache_charges_prefix_once_and_all_tail_steps(tail):
    raw, row = measurements(tail=tail)
    result = validate_block_costs(raw, row, f"quality_tail{tail}")
    assert result["block_inputs"] == sum(json.loads(raw["block_inputs_by_layer"]))
    assert result["block_inputs"] > 0


def test_uncached_control_pays_full_encoder_every_step():
    raw, row = measurements(cached=False)
    validate_block_costs(raw, row, "quality_tail6_uncached")


def test_fresh_final_prefix_is_charged_not_hidden():
    raw, row = measurements()
    row["inputs"] += 16
    row["calls"] += 1
    values = json.loads(raw["block_inputs_by_layer"])
    values[:18] = [row["inputs"]] * 18
    raw.update(block_inputs=sum(values), block_inputs_by_layer=json.dumps(values))
    calls = json.loads(raw["block_calls_by_layer"])
    calls[:18] = [row["calls"]] * 18
    raw.update(block_calls=sum(calls), block_calls_by_layer=json.dumps(calls))
    validate_block_costs(raw, row, "quality_tail6")


@pytest.mark.parametrize("bad", ["sum", "depth", "missing_tail_step", "free_prefix", "invented_stem"])
def test_inconsistent_accounting_is_rejected(bad):
    raw, row = measurements()
    values = json.loads(raw["block_inputs_by_layer"])
    if bad == "sum":
        raw["block_inputs"] -= 1
    elif bad == "depth":
        raw["encoder_blocks"] = 23
    elif bad == "missing_tail_step":
        values[-1] -= 16
        raw.update(block_inputs=sum(values), block_inputs_by_layer=json.dumps(values))
    elif bad == "free_prefix":
        values[:18] = [0] * 18
        raw.update(block_inputs=sum(values), block_inputs_by_layer=json.dumps(values))
    else:
        row["inputs"] -= 1
    with pytest.raises(RuntimeError):
        validate_block_costs(raw, row, "quality_tail6")


@pytest.mark.parametrize("bad", ["zero", "tail", "prefix"])
def test_missing_block_calls_are_rejected(bad):
    raw, row = measurements()
    calls = json.loads(raw["block_calls_by_layer"])
    if bad == "zero":
        calls = [0] * 24
    else:
        calls[-1 if bad == "tail" else 0] -= 1
    raw.update(block_calls=sum(calls), block_calls_by_layer=json.dumps(calls))
    with pytest.raises(RuntimeError):
        validate_block_costs(raw, row, "quality_tail6")


def test_failed_diagnostic_work_is_separated_and_included_in_total():
    raw, row = measurements()
    row["inputs"] += 8
    row["calls"] += 1
    failed_inputs, failed_calls = [8, 8] + [0] * 22, [1, 1] + [0] * 22
    for key, failures in (("block_inputs", failed_inputs), ("block_calls", failed_calls)):
        values = [a + b for a, b in zip(json.loads(raw[key + "_by_layer"]), failures)]
        raw.update({key: sum(values), key + "_by_layer": json.dumps(values),
                    "failed_" + key: sum(failures), "failed_" + key + "_by_layer": json.dumps(failures)})
    raw.update(failed_calls=1, failed_inputs=8, failed_forward_attempts=1)
    costs = validate_block_costs(raw, row, "quality_tail6")
    assert costs["failed_inputs"] == 8
    assert costs["block_inputs"] == raw["block_inputs"]
    raw["failed_block_inputs"] = 0
    with pytest.raises(RuntimeError):
        validate_block_costs(raw, row, "quality_tail6")


def test_live_parameter_inventory_not_only_declared_names():
    from types import SimpleNamespace
    import torch
    from torch import nn
    from benchmark_simplification import adaptation_parameter_record
    from run_algorithm_efficiency_suite import validate_parameter_record

    visual = nn.Module()
    visual.ln_post = nn.LayerNorm(4)
    visual.transformer = nn.Module()
    visual.transformer.resblocks = nn.ModuleList([nn.Linear(4, 4)])
    visual.transformer.requires_grad_(False)
    model = nn.Module()
    model.visual = visual
    names = ["visual.ln_post.weight", "visual.ln_post.bias"]
    method = SimpleNamespace(model=model, adapted_parameter_names=names,
                             optimizer=torch.optim.Adam(visual.ln_post.parameters()))
    record = adaptation_parameter_record(method, "quality_tail0")
    validate_parameter_record(record, names)
    assert record["trainable_parameters"] == 8
    with pytest.raises(RuntimeError):
        validate_parameter_record(dict(record, trainable_parameters=0), names)
    visual.ln_post.requires_grad_(False)
    with pytest.raises(RuntimeError):
        adaptation_parameter_record(method, "quality_tail0")
    visual.ln_post.requires_grad_(True)
    visual.transformer.requires_grad_(True)
    with pytest.raises(RuntimeError):
        adaptation_parameter_record(method, "quality_tail0")
    visual.transformer.requires_grad_(False)
    method.optimizer = torch.optim.Adam([visual.ln_post.weight])
    with pytest.raises(RuntimeError):
        adaptation_parameter_record(method, "quality_tail0")


def test_calibration_worker_has_no_budget_override():
    config = dict(data_root="/data", data_dirs={})
    cell = dict(dataset="voc21", split="calibration", limit=128,
                algorithm="quality_tail6", key="calibration/voc21/quality_tail6")
    command = worker_command(Path("/runtime"), config, cell, Path("/output"))
    assert command[command.index("--split") + 1] == "calibration"
    assert command[command.index("--max-images") + 1] == "128"
    assert not {"--steps", "--lr", "--max-regions", "--batch-size", "--view-batch-size"}.intersection(command)
    cell.update(limit=None, split="evaluation")
    assert "--max-images" not in worker_command(Path("/runtime"), config, cell, Path("/output"))
