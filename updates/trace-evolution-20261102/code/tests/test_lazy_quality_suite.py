from pathlib import Path
import sys
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_lazy_quality_suite as suite
from run_algorithm_efficiency_suite import worker_command, experiment_hashes


def test_plan_is_method_fixed_with_two_staging_controls():
    cells = suite.plan()
    assert len(cells) == len({c["key"] for c in cells}) == 20
    assert all(c["dataset"] == "pc60" and c["limit"] == 4 for c in cells[:4])
    assert {c["algorithm"] for c in cells} == {
        "quality_only", "quality_staged", "quality_lazy_01", "quality_lazy_class_01"}
    assert all(c["limit"] == 128 and c["variant"] == "full" for c in cells[4:])


def test_lazy_worker_never_changes_optimizer_or_candidate_budget(tmp_path):
    for cell in suite.plan():
        cmd = worker_command(tmp_path, dict(data_root="/data", data_dirs={}), cell, tmp_path / "cell")
        assert not {"--steps", "--lr", "--batch-size", "--max-regions"}.intersection(cmd)
        assert cmd[cmd.index("--algorithm") + 1] == cell["algorithm"]


def test_snapshot_and_worker_hashes_cover_new_algorithm():
    paths = suite.tool_dependencies(Path(suite.__file__).parent, (suite.CONTROLLER,))
    assert {"trace_lazy_quality.py", "trace_algorithm_variants.py", "benchmark_simplification.py"}.issubset(paths)
    manifest = dict(tool_hashes={p: p for p in paths})
    for name in suite.ALGORITHMS[1:]:
        assert set(experiment_hashes(manifest, name)) == {
            "benchmark_simplification.py", "trace_ablation_variants.py", "trace_algorithm_variants.py",
            "trace_lazy_quality.py", "trace_execution_audit.py"}


def test_other_sealed_baseline_panel_is_not_accepted(tmp_path):
    pilot, baselines = tmp_path / "pilot", tmp_path / "baselines"
    pilot.mkdir()
    baselines.mkdir()
    (pilot / "SUITE_SEALED.json").write_text("seal")
    (pilot / "suite_protocol.json").write_text(json.dumps(dict(input_panels={
        "pc60/128": dict(image_ids=list(range(128)))})))
    record = dict(suite_root=str(pilot), package_root=str(pilot / "runtime"),
                  suite_seal_sha256=suite.digest(pilot / "SUITE_SEALED.json"))
    path = baselines / "manifest.json"
    path.write_text(json.dumps(record))
    table = baselines / "baselines.csv"
    table.write_text("dataset,images\npc60,128\n")
    suite.verify_baseline_binding(pilot, baselines)
    path.write_text(json.dumps(dict(record, suite_root=str(tmp_path / "different"))))
    with pytest.raises(RuntimeError, match="not bound"):
        suite.verify_baseline_binding(pilot, baselines)
    path.write_text(json.dumps(record))
    table.write_text("dataset,images\npc60,5105\n")
    with pytest.raises(RuntimeError, match="populations differ"):
        suite.verify_baseline_binding(pilot, baselines)
