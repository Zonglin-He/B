"""Synthetic sealed-input tests; no active suite, model, or GPU is accessed."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("export_evolution", Path(__file__).with_name("export_evolution.py"))
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)

MOCK_CONTROLLER = '''import hashlib, json
def verify_runtime(root, manifest):
    assert hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == json.loads((root / "MANIFEST_SEALED.json").read_text())["manifest_sha256"]
def verify_cell(directory, cell, panel, manifest):
    for name in ("PREDICTIONS_SEALED.json", "SCORED_SEALED.json"):
        for relative, digest in json.loads((directory/name).read_text())["artifact_sha256"].items():
            assert hashlib.sha256((directory/relative).read_bytes()).hexdigest() == digest
    assert json.loads((directory/"expected_panel.json").read_text()) == panel
    return json.loads((directory/"result.json").read_text())
'''


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def result(dataset, policies=export.POLICIES, full=50.0):
    rows = []
    for policy in policies:
        for seed in (export.PERMUTATION_SEEDS if policy == "permuted" else (None,)):
            value = full if policy == "trace" else 51.0 + (seed or 0)
            rows.append(dict(policy=policy, permutation_seed=seed, miou_unrounded=value))
    return dict(status="completed", dataset=dataset, seed=0, images=1, full_split=True,
                pre_result=dict(miou_unrounded=40.), policy_results=rows)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.suite, self.paper = self.base / "suite", self.base / "paper"
        self.generated = self.paper / "generated"
        self.generated.mkdir(parents=True)
        for name in ("reported_comparison.tex", "cd_local.tex", "comparison_summary.tex", "mechanism_status.tex", "cost_snapshot.tex"):
            (self.generated / name).write_text("KEEP " + name)
        controller = self.suite / "runtime/tools/self_evolution_controller.py"
        controller.parent.mkdir(parents=True)
        controller.write_text(MOCK_CONTROLLER)
        cells = []
        for d in (*export.CORE, "coco_stuff"):
            for phase in ("smoke", "mechanism"):
                cells.append(dict(key=f"{phase}/{d}", dataset=d, seed=0,
                                  limit=4 if phase == "smoke" else None, policies=list(export.POLICIES)))
        for d in export.CORE:
            for seed in (1, 2):
                cells.append(dict(key=f"repetition/{d}/seed{seed}", dataset=d, seed=seed, limit=None, policies=["trace"]))
        config = dict(dataset="mock", resize=[224,224], patch_size=[224,224], patch_stride=112,
                      batch_size=1, class_extensions=False, data_dir="mock")
        panel = dict(image_ids=["image"], label_ids=["label"], vocabulary_sha256="vocab",
                     file_metadata=[[1,2]], available_images=1, config=config)
        self.manifest = dict(cells=cells, input_panels={c["key"]: panel for c in cells},
            runtime_hashes=export.runtime_hashes(self.suite / "runtime"),
            method="trace_ovss.method.two_view.TwoViewTRACE", rho="min(T,C)",
            permutation_seeds=[0,1,2], image_sha256={"image":"hash"})
        self.seal_manifest()
        self.reported = dict(methods=["MLMP", "TENT"], datasets={d: {"MLMP":[55.0,.02], "TENT":[53.0,.01]}
                                 for d in ("voc21", "pc60", "coco_object", "coco_stuff")})
        write(self.paper / "reported_baselines.json", self.reported)

    def seal_manifest(self):
        write(self.suite / "manifest.json", self.manifest)
        write(self.suite / "MANIFEST_SEALED.json", dict(manifest_sha256=export.sha(self.suite / "manifest.json")))

    def complete(self, dataset, policies=None):
        cell = next(c for c in self.manifest["cells"] if c["key"] == f"mechanism/{dataset}")
        directory = self.suite / "cells" / cell["key"]
        records = {"result.json":result(dataset, policies or export.POLICIES),
                   "expected_panel.json":export.expected_panel(self.manifest, cell),
                   "prediction_protocol.json":dict(deterministic_runtime={"deterministic_algorithms":True}, source_sha256={}),
                   "initial_state.json":{}}
        for name, value in records.items():
            write(directory / name, value)
        hashes = {name:export.sha(directory / name) for name in records}
        for name in ("PREDICTIONS_SEALED.json", "SCORED_SEALED.json"):
            write(directory / name, dict(artifact_sha256=hashes))

    def snapshot(self):
        return {p.name:p.read_bytes() for p in self.generated.iterdir()}

    def ready(self):
        for d in export.CORE:
            self.complete(d)

    def test_pending_preserves_all_outputs(self):
        self.complete("coco_cd")
        before = self.snapshot()
        status, code = export.run(self.suite, self.paper)
        self.assertEqual((status["status"], code), ("pending", 3))
        self.assertEqual(len(status["missing"]), 3)
        self.assertEqual(before, self.snapshot())

    def test_pending_cli_exit_code_and_explanation(self):
        before = self.snapshot()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = export.main(["--suite-root", str(self.suite), "--paper-root", str(self.paper)])
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "pending")
        self.assertEqual(before, self.snapshot())

    def test_policy_set_is_a_manifest_gate(self):
        self.manifest["cells"][0]["policies"].remove("mean")
        self.seal_manifest()
        before = self.snapshot()
        with self.assertRaisesRegex(export.Invalid, "policy set"):
            export.run(self.suite, self.paper)
        self.assertEqual(before, self.snapshot())

    def test_completed_policy_set_is_also_required(self):
        self.ready()
        self.complete("pc60", ["trace"])
        before = self.snapshot()
        with self.assertRaisesRegex(export.Invalid, "Policy results"):
            export.run(self.suite, self.paper)
        self.assertEqual(before, self.snapshot())

    def test_negative_deltas_and_published_reference(self):
        outputs, values = export.render({d:result(d) for d in export.CORE}, self.reported,
                                       dict(noadapt=40.,tent=52.,tpt=41.,mlmp=49.))
        self.assertEqual(values["published_deltas"]["voc21"], -5.)
        self.assertEqual(values["cd_local_delta"], -2.)
        text = "".join(outputs.values())
        self.assertNotIn("++", text)
        self.assertNotIn("+-", text)
        self.assertIn("-5.00", text)
        self.assertIn("-2.00", text)
        self.assertNotIn(r"\mathbf{50.00}", outputs["reported_comparison.tex"])
        self.assertIn(r"\mathbf{55.00", outputs["reported_comparison.tex"])
        self.assertNotIn("COCO-Stuff", outputs["reported_comparison.tex"])
        self.assertIn(r"52.00\pm0.82", outputs["mechanism_status.tex"])

    def test_check_only_writes_nothing(self):
        self.ready()
        before = self.snapshot()
        with mock.patch.object(export, "native_cd", return_value=dict(noadapt=40.,tent=52.,tpt=41.,mlmp=49.)):
            status, code = export.run(self.suite, self.paper, check_only=True)
        self.assertEqual((status["status"], code), ("ready", 0))
        self.assertEqual(before, self.snapshot())

    def test_complete_bundle_keeps_historical_cost_untouched(self):
        self.ready()
        before = (self.generated / "cost_snapshot.tex").read_bytes()
        with mock.patch.object(export, "native_cd", return_value=dict(noadapt=40.,tent=52.,tpt=41.,mlmp=49.)):
            status, code = export.run(self.suite, self.paper)
        self.assertEqual((status["status"], code), ("imported", 0))
        self.assertEqual(before, (self.generated / "cost_snapshot.tex").read_bytes())
        provenance = export.read_json(self.generated / "evolution_provenance.json")
        self.assertIn("not independent", provenance["replay_policy"])
        self.assertEqual(len(provenance["verified_cells"]), 4)

    def test_stuff_column_requires_full_result(self):
        inputs = {d:result(d) for d in export.CORE}
        inputs["coco_stuff"] = result("coco_stuff")
        outputs, _ = export.render(inputs, self.reported,
                                   dict(noadapt=40., tent=52., tpt=41., mlmp=49.))
        self.assertIn("COCO-Stuff", outputs["reported_comparison.tex"])

    def test_bundle_write_failure_rolls_back(self):
        before = self.snapshot()
        replace = export.os.replace
        calls = 0
        def failing_replace(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("mock publication failure")
            return replace(source, target)
        with mock.patch.object(export.os, "replace", side_effect=failing_replace):
            with self.assertRaises(OSError):
                export.publish(self.paper, {"reported_comparison.tex":"new", "cd_local.tex":"new"})
        self.assertEqual(before, self.snapshot())

    def test_native_cd_requires_sealed_matching_panels(self):
        root = self.base / "native"
        panel = export.expected_panel(self.manifest, self.manifest["cells"][0])
        hashes = {}
        for method in ("noadapt", "tent", "tpt", "mlmp"):
            directory = root / "coco_cd" / method
            protocol = dict(panel, method=method, full_split=True, seed=0, labels_used_for_adaptation=False)
            write(directory / "protocol.json", protocol)
            records = dict(status="completed", full_split=True, images=1, method=method,
                           protocol_sha256=export.sha(directory / "protocol.json"),
                           intersection=[40], union=[100], miou_unrounded=40.)
            write(directory / "result.json", records)
            for name in ("protocol.json", "result.json"):
                path = directory / name
                hashes[str(path.relative_to(root))] = export.sha(path)
        write(root / "SUITE_SEALED.json", dict(artifact_sha256=hashes))
        with mock.patch.object(export, "NATIVE_ROOT", root):
            self.assertEqual(export.native_cd(panel, {}), dict(noadapt=40.,tent=40.,tpt=40.,mlmp=40.))
            changed = dict(panel, config=dict(panel["config"], batch_size=2))
            with self.assertRaisesRegex(export.Invalid, "episode mismatch"):
                export.native_cd(changed, {})

    def test_runtime_tampering_is_rejected_before_import(self):
        (self.suite / "runtime/tools/self_evolution_controller.py").write_text("raise AssertionError('executed')")
        with self.assertRaisesRegex(export.Invalid, "runtime hash"):
            export.run(self.suite, self.paper)

    def test_deep_verifier_rejects_tampered_cell(self):
        self.ready()
        path = self.suite / "cells/mechanism/pc60/result.json"
        path.write_text("{}")
        before = self.snapshot()
        with self.assertRaisesRegex(export.Invalid, "Deep cell verification"):
            export.run(self.suite, self.paper)
        self.assertEqual(before, self.snapshot())

    def test_voc20_manifest_accepts_only_four_trace_cells(self):
        self.manifest["cells"] = [dict(key=key, dataset="voc20", seed=seed, limit=limit, policies=["trace"])
            for key, seed, limit in (("smoke/voc20", 0, 4), ("mechanism/voc20", 0, None),
                ("repetition/voc20/seed1", 1, None), ("repetition/voc20/seed2", 2, None))]
        panel = next(iter(self.manifest["input_panels"].values()))
        self.manifest["input_panels"] = {c["key"]:panel for c in self.manifest["cells"]}
        self.seal_manifest()
        export.verify_manifest(self.suite, voc20=True)
        self.manifest["cells"][1]["policies"].append("uniform")
        self.seal_manifest()
        with self.assertRaisesRegex(export.Invalid, "policy set"):
            export.verify_manifest(self.suite, voc20=True)

    def test_final_import_needs_both_explicit_paths(self):
        before = self.snapshot()
        for kwargs in (dict(product_suite_root=self.base / "B"), dict(voc20_suite_root=self.base / "C")):
            with self.assertRaisesRegex(export.Invalid, "both explicit"):
                export.run(self.suite, self.paper, **kwargs)
        self.assertEqual(before, self.snapshot())

    def test_suite_index_rejects_incomplete_status_and_tampering(self):
        write(self.suite / "queue_state.json", dict(status="completed", completed=18, total=18))
        entries = {n:export.sha(self.suite / n) for n in ("manifest.json", "queue_state.json")}
        write(self.suite / "SUITE_SEALED.json", dict(artifact_sha256=entries))
        export.verify_index(self.suite, 18, {})
        with self.assertRaisesRegex(export.Invalid, "entire plan"):
            export.verify_index(self.suite, 4, {})
        write(self.suite / "queue_state.json", dict(status="completed", completed=4, total=4))
        with self.assertRaisesRegex(export.Invalid, "seal mismatch"):
            export.verify_index(self.suite, 4, {})

    def test_voc20_null_controls_best_baseline_and_negative_differences(self):
        inputs = {d:result(d) for d in (*export.CORE, "coco_stuff")}
        inputs["voc20"] = result("voc20", ["trace"], full=54.)
        reported = json.loads(json.dumps(self.reported))
        reported["datasets"]["voc20"] = {"MLMP":[53., 0.], "TENT":[56., .01]}
        reported["datasets"]["pc60"]["MLMP"][1] = None
        for d in export.CORE:
            for row in inputs[d]["policy_results"]:
                row["miou_unrounded"] = 49.
                if row["policy"] == "trace":
                    row["miou_unrounded"] = 50.
                if row["policy"] == ("counterfactual" if d == "pc60" else "mean"):
                    row["miou_unrounded"] = 52.
        controls = {d:result(d, export.APPEND_POLICIES) for d in ("pc60", "coco_object")}
        outputs, values = export.render(inputs, reported, dict(noadapt=40.,tent=45.,tpt=41.,mlmp=49.), controls)
        self.assertEqual(values["published_best"]["voc20"], 56.)
        self.assertEqual(values["published_deltas"]["voc20"], -2.)
        self.assertEqual(values["min_minus_controls"]["pc60"]["counterfactual"], -2.)
        self.assertIn("noori2025mlmp", outputs["reported_comparison.tex"])
        self.assertIn(r"\mathbf{56.00\pm0.01}", outputs["reported_comparison.tex"])
        self.assertIn(r"$\mathbf{55.00}$", outputs["reported_comparison.tex"])
        self.assertNotIn(r"55.00\pm0.00", outputs["reported_comparison.tex"])
        mechanism = outputs["mechanism_status.tex"]
        self.assertIn("Method & VOC20 & VOC21 & PC60 & COCO Object & COCO-Stuff", mechanism)
        for name in ("Uniform", "Confidence", "T-only", "C-only", "Mean", "Shuffled", "Product", "Pixel-normalized"):
            row = next(line for line in mechanism.splitlines() if line.startswith(name + " &"))
            self.assertEqual(row.split(" & ")[1], "--")
        cf = next(line for line in mechanism.splitlines() if line.startswith("C-only &"))
        mean = next(line for line in mechanism.splitlines() if line.startswith("Mean &"))
        self.assertIn(r"\mathbf{52.00}", cf)
        self.assertIn(r"\mathbf{52.00}", mean)
        product = next(line for line in mechanism.splitlines() if line.startswith("Product &"))
        self.assertTrue(product.endswith(r"& -- \\"))
        self.assertIn("COCO-CD", outputs["cd_mechanism.tex"])

    def test_policy_score_null_duplicate_and_nonfinite_are_invalid(self):
        for value in (None, float("nan"), float("inf"), -1., 101.):
            record = result("voc20", ["trace"])
            record["policy_results"][0]["miou_unrounded"] = value
            with self.assertRaisesRegex(export.Invalid, "Invalid mIoU"):
                export.policy_scores(record, ["trace"])
        record = result("voc20", ["trace"])
        record["policy_results"] *= 2
        with self.assertRaisesRegex(export.Invalid, "duplicated"):
            export.policy_scores(record, ["trace"])

    def test_current_readout_and_literature_deltas_are_separate(self):
        inputs = {d:result(d) for d in export.CORE}
        inputs["pc60"]["pre_result"]["miou_unrounded"] = 60.
        outputs, values = export.render(inputs, self.reported,
                                       dict(noadapt=40.,tent=52.,tpt=41.,mlmp=49.))
        self.assertEqual(values["adaptation_deltas"]["pc60"], -10.)
        self.assertEqual(values["published_deltas"]["pc60"], -5.)
        self.assertIn("No Adapt, current readout (L)", outputs["reported_comparison.tex"])
        self.assertIn(r"\mathbf{60.00}", outputs["reported_comparison.tex"])
        self.assertIn("-10.00", outputs["reported_comparison.tex"])
        self.assertIn("current No Adapt", outputs["cd_local.tex"])
        del inputs["pc60"]["pre_result"]
        with self.assertRaisesRegex(export.Invalid, "current-readout"):
            export.render(inputs, self.reported, dict(noadapt=40.,tent=52.,tpt=41.,mlmp=49.))

    def test_historical_float_counts_must_be_integral_and_finite(self):
        for value in (0, 40, 40., 17361324.):
            self.assertTrue(export.whole_count(value))
        for value in (True, None, .5, -1., float("nan"), float("inf")):
            self.assertFalse(export.whole_count(value))
        self.assertEqual(export.signed(-.0001), "-0.00")
        self.assertEqual(export.signed(-.1265579544776898), "-0.13")

    def test_pdf_extraction_checks_all_six_methods_and_null_sd(self):
        pdf = self.base / "reference.pdf"
        pdf.write_bytes(b"mock PDF")
        reported = dict(source_pdf=str(pdf), source_pdf_sha256=export.sha(pdf),
            methods=["No Adapt", "TENT", "TPT", "WATT", "CLIPArTT", "MLMP"], datasets={})
        lines = ["Table 5: NACLIP"]
        for d, label in (("voc20", "V20"), ("voc21", "V21"), ("pc60", "P60"),
                         ("coco_object", "COCOObject"), ("coco_stuff", "COCOStuff")):
            values = [[75.91,None],[77.,.04],[75.93,.01],[57.73,.06],[72.77,.14],[83.76,0.]]
            reported["datasets"][d] = dict(zip(reported["methods"], values))
            lines.append(label + " (Original) 75.91 77.00 \u00b10.04 75.93 \u00b10.01 57.73 \u00b10.06 72.77 \u00b10.14 83.76 \u00b10.00")
        with mock.patch.object(export.subprocess, "run", return_value=mock.Mock(stdout="\n".join(lines))):
            fingerprints = {}
            export.verify_reported(reported, fingerprints)
            self.assertEqual(fingerprints[str(pdf)], export.sha(pdf))
            reported["datasets"]["voc20"]["No Adapt"][1] = 0.
            with self.assertRaisesRegex(export.Invalid, "voc20"):
                export.verify_reported(reported, {})

    def test_replays_are_not_independent_uncertainty(self):
        completed = {"voc20":result("voc20", ["trace"], full=83.94251405671336),
                     "coco_stuff":result("coco_stuff")}
        checks = {f"repetition/voc20/seed{s}":dict(miou=83.94251405671336) for s in (1,2)}
        table = export.render_replays(completed, checks)
        self.assertEqual(table.count("83.94251406"), 3)
        self.assertIn("not independent", table)
        self.assertNotIn(r"\pm", table)
        self.assertIn(r"COCO-Stuff & $50.00000000$ & -- & -- & -- & --", table)

    def append_fixture(self):
        self.complete("pc60")
        parent = self.suite / "cells/mechanism/pc60"
        directory = self.base / "B/cells/pc60"
        panel = export.read_json(parent / "expected_panel.json")
        config = panel["config"]
        initial = dict(model_sha256="model", optimizer_sha256="optimizer")
        write(parent / "initial_state.json", initial)
        write(self.suite / "SUITE_SEALED.json", {})
        record = result("pc60", export.APPEND_POLICIES)
        metric = dict(policy="pre", policy_label="pre", permutation_seed=None,
                      intersection=[40], union=[100], miou_unrounded=40.)
        record["pre_result"] = metric
        parent_result = export.read_json(parent / "result.json")
        parent_result["pre_result"] = metric
        write(parent / "result.json", parent_result)
        for row in record["policy_results"]:
            row.update(policy_label=row["policy"], intersection=[51], union=[100])
        protocol = dict(panel=panel, inputs=[dict(image="image",label="label")], config=config,
            checkpoint={"sha256":"checkpoint"}, classes=["class"], queries=["class"], query_to_class=[0],
            reduce_zero_label=False, deterministic_runtime={"deterministic_algorithms":True},
            dataset="pc60", seed=0, full_split=True)
        write(parent / "prediction_protocol.json", protocol)
        write(parent / "episodes/000000.json", dict(candidate_sha256="candidate"))
        write(parent / "episodes/000000.npz", {})
        parent_hashes = {f"episodes/000000.{s}":export.sha(parent / f"episodes/000000.{s}") for s in ("json","npz")}
        write(parent / "PREDICTIONS_SEALED.json", dict(artifact_sha256=parent_hashes))
        binding = dict(parent_cell=str(parent), parent_suite=str(self.suite),
            parent_prediction_seal_sha256=export.sha(parent / "PREDICTIONS_SEALED.json"),
            parent_suite_seal_sha256=export.sha(self.suite / "SUITE_SEALED.json"),
            parent_manifest_sha256=export.sha(self.suite / "manifest.json"))
        protocol.update(expected_panel_sha256=export.sha(parent / "expected_panel.json"),
            parent_replay=binding, source_sha256={str(self.suite / "runtime/tools/self_evolution_controller.py"):
                self.manifest["runtime_hashes"]["tools/self_evolution_controller.py"]},
            variants=[[p,p,None] for p in export.APPEND_POLICIES])
        initial.update(parent_initial_state_sha256=export.sha(parent / "initial_state.json"),
                       initial_identity_matches_parent=True)
        gate = dict(passed=True, changed_label_pixels=0, tolerance=0, gt_contents_opened=False, images=1)
        records = {"prediction_protocol.json":protocol, "expected_panel.json":panel, "initial_state.json":initial,
            "prediction_result.json":{}, "checkpoint_identity.json":{}, "replay_gate.json":gate,
            "episodes/000000.npz":{}, "episodes/000000.json":{}, "episodes/000000.parent.npz":{},
            "episodes/000000.cachelink.json":dict(parent_npz=str(parent / "episodes/000000.npz"),
                parent_npz_sha256=parent_hashes["episodes/000000.npz"],
                parent_json_sha256=parent_hashes["episodes/000000.json"], parent_candidate_sha256="candidate")}
        for name in ("pre", *export.APPEND_POLICIES):
            records[f"predictions/{name}/000000.npy"] = {}
        for name, value in records.items():
            write(directory / name, value)
        self.seal_append(directory, record)
        return directory, parent, panel

    def seal_append(self, directory, record=None):
        scored_names = {"result.json", "score_inputs.json", "scores_by_episode.jsonl"}
        entries = {str(p.relative_to(directory)):export.sha(p) for p in directory.rglob("*") if p.is_file()
                   and p.name not in scored_names | {"PREDICTIONS_SEALED.json", "SCORED_SEALED.json"}}
        write(directory / "PREDICTIONS_SEALED.json", dict(artifact_sha256=entries, complete=True,
                                                       gt_contents_opened=False, images=1))
        record = record or export.read_json(directory / "result.json")
        record.update(prediction_seal_sha256=export.sha(directory / "PREDICTIONS_SEALED.json"),
                      label_contents_first_opened_after_prediction_seal=True)
        write(directory / "result.json", record)
        write(directory / "score_inputs.json", dict(label_sha256={"label":"hash"}))
        policies = {r["policy"]:dict(intersection=r["intersection"],union=r["union"])
                    for r in [record["pre_result"], *record["policy_results"]]}
        write(directory / "scores_by_episode.jsonl", dict(episode=0,image_indices=[0],policies=policies))
        write(directory / "SCORED_SEALED.json", dict(artifact_sha256={n:export.sha(directory/n) for n in scored_names},
            prediction_seal_sha256=record["prediction_seal_sha256"]))

    def verify_append_fixture(self, directory, parent, panel):
        def verify_artifacts(root, seal):
            for relative, digest in seal["artifact_sha256"].items():
                self.assertEqual(export.sha(export.checked_path(root, relative)), digest)
            return set(seal["artifact_sha256"])
        return export.verify_append_cell(directory, "pc60", panel, parent, self.manifest,
                                         mock.Mock(verify_artifacts=verify_artifacts), {})

    def test_append_controls_reuse_verified_parent_and_pooled_counts(self):
        directory, parent, panel = self.append_fixture()
        checked = self.verify_append_fixture(directory, parent, panel)
        self.assertEqual(export.policy_scores(checked, export.APPEND_POLICIES),
                         {("product",None):51., ("pixel_normalized",None):51.})

    def test_append_rejects_resealed_wrong_identity_gate_null_and_metric(self):
        directory, parent, panel = self.append_fixture()
        cases = [("prediction_protocol.json", "parent_replay", {}, "parent identity"),
                 ("replay_gate.json", "changed_label_pixels", 1, "replay gate"),
                 ("initial_state.json", "model_sha256", "other", "initial state"),
                 ("result.json", "dataset", "voc20", "result identity")]
        for name, key, value, error in cases:
            original = export.read_json(directory / name)
            write(directory / name, dict(original, **{key:value}))
            self.seal_append(directory)
            with self.assertRaisesRegex(export.Invalid, error):
                self.verify_append_fixture(directory, parent, panel)
            write(directory / name, original)
            self.seal_append(directory)
        record = export.read_json(directory / "result.json")
        record["policy_results"][0]["miou_unrounded"] = None
        self.seal_append(directory, record)
        with self.assertRaisesRegex(export.Invalid, "Invalid mIoU"):
            self.verify_append_fixture(directory, parent, panel)

    def test_append_rejects_resealed_wrong_counts_and_duplicate_policy(self):
        directory, parent, panel = self.append_fixture()
        record = export.read_json(directory / "result.json")
        record["policy_results"][0]["miou_unrounded"] = 52.
        self.seal_append(directory, record)
        with self.assertRaisesRegex(export.Invalid, "pooled mIoU"):
            self.verify_append_fixture(directory, parent, panel)
        record["policy_results"][0]["miou_unrounded"] = 51.
        record["policy_results"][1]["policy"] = "product"
        self.seal_append(directory, record)
        with self.assertRaisesRegex(export.Invalid, "duplicated"):
            self.verify_append_fixture(directory, parent, panel)

    def test_append_rejects_resealed_foreign_source_and_cache(self):
        directory, parent, panel = self.append_fixture()
        protocol = export.read_json(directory / "prediction_protocol.json")
        write(directory / "prediction_protocol.json", dict(protocol, source_sha256={"/foreign/worker.py":"hash"}))
        self.seal_append(directory)
        with self.assertRaisesRegex(export.Invalid, "worker source"):
            self.verify_append_fixture(directory, parent, panel)
        write(directory / "prediction_protocol.json", protocol)
        path = directory / "episodes/000000.cachelink.json"
        link = export.read_json(path)
        write(path, dict(link, parent_candidate_sha256="other"))
        self.seal_append(directory)
        with self.assertRaisesRegex(export.Invalid, "cache identity"):
            self.verify_append_fixture(directory, parent, panel)

    def test_cli_passes_explicit_suite_paths_and_check_only(self):
        with mock.patch.object(export, "run", return_value=({"status":"ready"},0)) as run:
            with redirect_stdout(io.StringIO()):
                code = export.main(["--suite-root", str(self.suite), "--paper-root", str(self.paper),
                    "--product-suite-root", str(self.base / "B"), "--voc20-suite-root", str(self.base / "C"),
                    "--check-only"])
        self.assertEqual(code, 0)
        run.assert_called_once_with(self.suite, self.paper, True,
                                    product_suite_root=self.base / "B", voc20_suite_root=self.base / "C")

    def test_generated_bundle_hashes_and_repeat_import_are_reproducible(self):
        self.ready()
        with mock.patch.object(export, "native_cd", return_value=dict(noadapt=40.,tent=52.,tpt=41.,mlmp=49.)):
            export.run(self.suite, self.paper)
            first = self.snapshot()
            export.run(self.suite, self.paper)
        self.assertEqual(first, self.snapshot())
        provenance = export.read_json(self.generated / "evolution_provenance.json")
        self.assertEqual(provenance["verified_cell_count"], 4)
        for name, digest in provenance["output_sha256"].items():
            self.assertEqual(export.sha(self.generated / name), digest)

    def test_source_change_before_publication_preserves_old_outputs(self):
        self.ready()
        before = self.snapshot()
        def changed_source(*args):
            write(self.paper / "reported_baselines.json", {})
            return dict(noadapt=40.,tent=52.,tpt=41.,mlmp=49.)
        with mock.patch.object(export, "native_cd", side_effect=changed_source):
            with self.assertRaisesRegex(export.Invalid, "Source changed during import"):
                export.run(self.suite, self.paper)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
