import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from tools.evolution_fifty import risk_score_worker as worker
from tools.evolution_fifty.intake import sha256, write_new
from tools.evolution_fifty.risk_score_payload import support_array, hard_array, annotation_bytes
from tools.evolution_fifty.risk_summary import score_groups, summary
from tools.evolution_fifty.risk_matching import membership, summarize


def archive(array, name="probabilities"):
    stream = io.BytesIO()
    np.savez_compressed(stream, **{name:array})
    return stream.getvalue()


def labels_content(labels):
    stream = io.BytesIO()
    np.save(stream, labels, allow_pickle=False)
    return stream.getvalue()


def fixture(tmp_path, monkeypatch):
    replay, parent, suite = (tmp_path/k for k in ("replay", "parent", "score"))
    suite.mkdir()
    bound = dict(replay=str(replay), original=dict(parent=str(parent), datasets={}), references={},
        risk_protocol=dict(regions=["inside", "outside", "whole"], datasets={}, checkpoints={}),
        candidates=8, barrier_sha256="b"*64, risk_protocol_sha256="c"*64,
        terminal=dict(MainPID="0", Result="success", ExecMainStatus="0"))
    for dataset in ("coco_object", "pc60"):
        cases, refs, rows = [], [], []
        for i in range(4):
            source, episode = f"source{i//2}", 10+i//2
            gtpath = tmp_path/f"{dataset}_{source}.png"
            if not gtpath.exists():
                gt = np.zeros((3, 3), np.uint8)
                gt[0, 0] = 255
                Image.fromarray(gt).save(gtpath)
            state = gtpath.stat()
            source_row = dict(label=str(gtpath), label_metadata=[state.st_size, state.st_mtime_ns],
                label_metadata_details=dict(size_bytes=state.st_size, mtime_ns=state.st_mtime_ns,
                    device=state.st_dev, inode=state.st_ino, contents_read=False))
            high = i % 2 == 1
            c = .9 if high else .1
            candidate = dict(batch_idx=0, image_idx=0, class_idx=0, area=9,
                region=dict(shape=[3, 3], encoding="packbits-little-hex", data="ff01", true_pixels=9))
            case = dict(case=f"case_{i:06d}", episode=episode, source_id=source, candidate_id=f"candidate{i}",
                example=dict(candidate=candidate, source_input=source_row))
            points = []
            for step in (0, 1, 3):
                p = .6 if step == 0 else .7 if high else .4
                support = np.stack([np.full((3, 3), p, np.float32), np.full((3, 3), 1-p, np.float32)])
                hard = support.argmax(0).astype(np.uint16)
                name = f"{dataset}/supports/{source}_{'initial' if step == 0 else f'{i}_{step}'}.npz"
                path = replay/name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(archive(support))
                labelname = f"{case['case']}/step_{step:02d}.labels.npy"
                labelpath = parent/dataset/labelname
                labelpath.parent.mkdir(parents=True, exist_ok=True)
                labelpath.write_bytes(labels_content(hard))
                ref = dict(support=name, sha256=sha256(path), shape=[2, 3, 3], probabilities_dtype="float32",
                    hard_labels_sha256=hashlib.sha256(hard.tobytes()).hexdigest())
                points.append(dict(step=step, labels=labelname, labels_sha256=sha256(labelpath),
                    snapshot_sha256=f"theta0-{source}" if step == 0 else f"{i}-{step}", reference=ref))
            refs.append(dict(case=case["case"], episode=episode, source_id=source,
                candidate_id=case["candidate_id"], checkpoints=points))
            cases.append(case)
            rows.append(dict(case=case["case"], episode=episode, source_id=source, candidate_id=case["candidate_id"],
                C=c, confidence=.9, benchmark_class=0, area=9, shape=[3, 3]))
        bound["original"]["datasets"][dataset] = dict(spec=dict(native_config=dict(resize=[3, 3], steps=3),
            queries=["a", "b"], query_to_class=None), cases=cases)
        bound["references"][dataset] = refs
        bound["risk_protocol"]["datasets"][dataset] = dict(rows=rows, score_groups=score_groups(rows))
        bound["risk_protocol"]["checkpoints"][dataset] = [1, 3]
    report = suite/"INDEPENDENT_REPORT.md"
    report.write_text("Synthetic audit record only, not actual artifact admission\n")
    write_new(suite/"ADMITTED.json", dict(synthetic_fixture=True))
    audit_binding = dict(admission_path=str(suite/"ADMITTED.json"), admission_sha256=sha256(suite/"ADMITTED.json"),
        report_path=str(report), report_sha256=sha256(report), independent_reviewer="CPU fixture")
    for key, path in (("admission_identity", suite/"ADMITTED.json"), ("report_identity", report)):
        s = path.stat()
        audit_binding[key] = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]
    monkeypatch.setattr(worker, "admission", lambda *_: (suite, bound, audit_binding))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=32*1024**3))
    monkeypatch.setattr(worker, "summary", lambda *a: summary(*a, replicates=24))
    monkeypatch.setattr(worker, "matched_summary", lambda *a: summarize(*a, replicates=24))
    return suite, bound


def test_cpu_filesystem_end_to_end_with_exact_consequences_and_no_GPU(tmp_path, monkeypatch):
    suite, bound = fixture(tmp_path, monkeypatch)
    reads = []
    native = worker.annotation_bytes
    def observe(row):
        assert (suite/"STARTED.json").exists()
        reads.append(row["label"])
        return native(row)
    monkeypatch.setattr(worker, "annotation_bytes", observe)
    worker.run(suite, "f"*64)
    seal = json.loads((suite/"ANALYSIS_SEALED.json").read_text())
    assert seal["complete"] and seal["candidates"] == 8 and not seal["new_GPU"]
    assert len(reads) == 4 and len(set(reads)) == 4
    assert seal["started_sha256"] == sha256(suite/"STARTED.json")
    assert seal["audit_binding"] == json.loads((suite/"STARTED.json").read_text())["audit_binding"]
    assert seal["audit_binding"]["admission_sha256"] == sha256(suite/"ADMITTED.json")
    for dataset, digest in seal["dataset_seals"].items():
        assert sha256(suite/dataset/"ANALYSIS_SEALED.json") == digest
        result = json.loads((suite/dataset/"result.json").read_text())
        metrics = result["checkpoints"]["1"]["inside"]
        high = metrics["score_quintiles"]["groups"]["C_Q5"]["metrics"]
        # Exact ties at the first cut go to the higher bin; Q1 stays empty.
        assert metrics["score_quintiles"]["groups"]["C_Q1"]["metrics"]["support_harm_rate_threshold_0.001"]["value"] is None
        low = metrics["score_quintiles"]["groups"]["C_Q2"]["metrics"]
        assert high["support_harm_rate_threshold_0.001"]["value"] == 0
        assert low["support_harm_rate_threshold_0.001"]["value"] == 100
        matched = metrics["matched"]["initial_scores"]["contrasts"]
        assert matched["support_harm_rate_threshold_0.001"]["value"] == -100
        assert matched["candidate_mean_support_change"]["value"] == pytest.approx(.3)
        local = json.loads((suite/dataset/"ANALYSIS_SEALED.json").read_text())
        for name, actual in local["artifact_sha256"].items():
            assert sha256(suite/dataset/name) == actual
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="No scoring resume/retry"):
        worker.run(suite, "f"*64)


def test_admission_failure_blocks_all_GT_and_outputs(tmp_path, monkeypatch):
    suite, _ = fixture(tmp_path, monkeypatch)
    def deny(*_):
        raise ValueError("Independent CPU scoring not admitted")
    monkeypatch.setattr(worker, "admission", deny)
    monkeypatch.setattr(worker, "annotation_bytes", lambda *_: pytest.fail("GT before admission"))
    with pytest.raises(ValueError, match="not admitted"):
        worker.run(suite, "f"*64)
    assert not (suite/"STARTED.json").exists() and not (suite/"coco_object").exists()


def test_prepare_requires_true_support_exit_before_metadata_or_GT(tmp_path, monkeypatch):
    def running():
        raise ValueError("Exact support replay has not successfully exited")
    monkeypatch.setattr(worker, "replay_terminal", running)
    monkeypatch.setattr(worker, "metadata_bind", lambda *_: pytest.fail("No metadata freeze while still running"))
    with pytest.raises(ValueError, match="not successfully exited"):
        worker.prepare(tmp_path/"future")
    assert not (tmp_path/"future").exists()


@pytest.mark.parametrize("failure", ["support", "duplicate_hard", "annotation"])
def test_partial_input_failure_no_global_seal_and_no_retry(tmp_path, monkeypatch, failure):
    suite, bound = fixture(tmp_path, monkeypatch)
    if failure == "support":
        point = bound["references"]["coco_object"][1]["checkpoints"][1]
        (Path(bound["replay"])/point["reference"]["support"]).write_bytes(b"tamper")
    elif failure == "duplicate_hard":
        point = bound["references"]["coco_object"][1]["checkpoints"][0]
        (Path(bound["original"]["parent"])/"coco_object"/point["labels"]).write_bytes(b"tamper duplicated theta0")
    else:
        source = bound["original"]["datasets"]["coco_object"]["cases"][2]["example"]["source_input"]
        Path(source["label"]).write_bytes(b"modified label")
    with pytest.raises(ValueError):
        worker.run(suite, "f"*64)
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
    assert (suite/"coco_object"/"candidate_outcomes.jsonl").exists()
    with pytest.raises(ValueError, match="No scoring resume/retry"):
        worker.run(suite, "f"*64)


def test_final_bookkeeping_failure_blocks_global_seal(tmp_path, monkeypatch):
    suite, _ = fixture(tmp_path, monkeypatch)
    native = worker.write_json
    def fail(path, data, **kwargs):
        if data["phase"] == "analysis_complete_awaiting_seal_and_exit":
            raise RuntimeError("injected final progress failure")
        return native(path, data, **kwargs)
    monkeypatch.setattr(worker, "write_json", fail)
    with pytest.raises(RuntimeError, match="final progress"):
        worker.run(suite, "f"*64)
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()


def test_real_terminal_requires_invocation_zeroPID_and_exited(monkeypatch):
    expected = dict(LoadState="loaded", ActiveState="active", SubState="exited", MainPID="0",
        ExecMainPID="2012362", ExecMainCode="1", ExecMainStatus="0", Result="success", InvocationID=worker.INVOCATION)
    for changed in (dict(SubState="running", MainPID="2012362"), dict(InvocationID="old"), dict(ExecMainStatus="1")):
        actual = dict(expected, **changed)
        monkeypatch.setattr(worker.subprocess, "run", lambda *a, _d=actual, **kw: SimpleNamespace(
            stdout="\n".join(f"{k}={v}" for k,v in _d.items())))
        with pytest.raises(ValueError, match="not successfully exited"):
            worker.replay_terminal()
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="\n".join(f"{k}={v}" for k,v in expected.items())))
    assert worker.replay_terminal() == expected


def test_support_no_renormalization_and_header_bounds():
    data = np.full((2, 3, 3), .2, dtype=np.float32)
    ref = dict(shape=[2, 3, 3], probabilities_dtype="float32")
    assert np.array_equal(support_array(archive(data), ref, 2, [3, 3]), data)
    for bad in (archive(data, "extra"), archive(data.astype(np.float64)), archive(data[:, :, :2]),
                archive(np.full_like(data, np.nan)), archive(data.astype(object))):
        with pytest.raises((ValueError, zipfile.BadZipFile)):
            support_array(bad, ref, 2, [3, 3])
    with pytest.raises(ValueError, match="size differs"):
        hard_array(labels_content(np.zeros((3, 3), np.uint16))+b"trailing", 2, [3, 3])


def test_oversized_NPY_header_rejected_before_numpy_load(monkeypatch):
    fake = io.BytesIO()
    np.lib.format.write_array_header_1_0(fake, dict(descr="<f4", fortran_order=False, shape=(2, 3, 9000000)))
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive_file:
        archive_file.writestr("probabilities.npy", fake.getvalue())
    monkeypatch.setattr(np, "load", lambda *_a, **_kw: pytest.fail("Allocated before header admission"))
    with pytest.raises(ValueError, match="geometry/dtype/size"):
        support_array(zipped.getvalue(), dict(shape=[2, 3, 3], probabilities_dtype="float32"), 2, [3, 3])


def test_query_extension_quality_ties_and_missing_strata_not_merged():
    rows = [dict(episode=i//2, source_id=f"s{i//2}", confidence=.9, benchmark_class=0,
        area=1, shape=[2, 2], C=c) for i,c in enumerate((.1, .9, .1, .9))]
    assert membership(rows)["matched_candidates"] == 4
    separate_queries = membership(rows, query_indices=[0, 1, 0, 1])
    assert separate_queries["matched_candidates"] == 0
    assert len(separate_queries["exact_median_ties"]) == 4
    quality = [dict(fraction_correct=q) for q in (None, .2, .8, .8)]
    matching = membership(rows, quality)
    assert matching["undefined_initial_quality"] == [0]
    assert matching["matched_candidates"] == 2
    assert matching["unmatched_candidates"] == [1]


def test_sparse_cluster_matching_retains_undefined_draws_and_null_interval():
    rows = [dict(episode=i, source_id=f"s{i}", confidence=.9, benchmark_class=0,
        area=1, shape=[2, 2], C=.1 if i == 0 else .9) for i in range(2)]
    outcome = dict(counts=dict(valid=1, pre_correct=1, pre_wrong=0, correct_to_wrong=0, wrong_to_correct=0),
        mean_true_class_support_delta=0., support_delta_sum=0.)
    from tools.evolution_fifty.pixel_transitions import FIELDS
    outcome["counts"] = dict(zip(FIELDS, [1, 1, 0, 0, 0]))
    result = summarize(rows, [outcome, outcome], membership(rows), "fixture", replicates=100)
    value = result["contrasts"]["candidate_mean_support_change"]
    assert value["value"] == 0 and value["undefined_replicates"] > 0
    assert value["lower95"] is None and value["upper95"] is None and value["no_replicate_dropped"]


def test_real_admission_rejects_replay_only_audit_and_parent_terminal_failure(tmp_path, monkeypatch):
    actual_admission = worker.admission
    suite, bound = fixture(tmp_path, monkeypatch)
    runtime = suite/"runtime"
    runtime.mkdir()
    monkeypatch.setattr(worker, "__file__", str(runtime/"tools/evolution_fifty/risk_score_worker.py"))
    monkeypatch.setattr(worker, "strict_runtime_inventory", lambda _: {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    write_new(suite/"SCORE_INPUTS.json", bound)
    write_new(suite/"PROTOCOL.json", dict(CPU_only=True))
    write_new(suite/"SUITE_PREPARED.json", dict(root=str(suite), runtime_hashes={},
        inputs_sha256=sha256(suite/"SCORE_INPUTS.json"), protocol_sha256=sha256(suite/"PROTOCOL.json")))
    digest = sha256(suite/"SUITE_PREPARED.json")
    report = suite/"INDEPENDENT_REPORT.md"
    audit = dict(decision="ADMITTED", suite_prepared_sha256=digest, independent_reviewer="CPU fixture",
        report_path=str(report), report_sha256=sha256(report), label_free_support_replay_allowed=True)
    (suite/"ADMITTED.json").write_text(json.dumps(audit))
    monkeypatch.setattr(worker, "replay_terminal", lambda: pytest.fail("Replay-only admission must fail before parent lookup"))
    with pytest.raises(ValueError, match="CPU scoring admission"):
        actual_admission(suite, digest)
    audit["CPU_GT_scoring_allowed"] = True
    (suite/"ADMITTED.json").write_text(json.dumps(audit))
    def running():
        raise ValueError("Exact replay still running")
    monkeypatch.setattr(worker, "replay_terminal", running)
    monkeypatch.setattr(worker, "metadata_bind", lambda *_: pytest.fail("Running parent cannot pass metadata"))
    with pytest.raises(ValueError, match="still running"):
        actual_admission(suite, digest)
    monkeypatch.setattr(worker, "replay_terminal", lambda: bound["terminal"])
    monkeypatch.setattr(worker, "metadata_bind", lambda *_: bound)
    admitted = actual_admission(suite, digest)
    assert admitted[:2] == (suite, bound)
    assert admitted[2]["admission_sha256"] == sha256(suite/"ADMITTED.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        actual_admission(suite, digest)
    # A hardlink swap after forbidden-inode stat must fail before reading it.
    import os
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    stable = worker.stable_bytes
    native_open = Path.open
    report_opens = []
    def watched_open(path, *args, **kwargs):
        if path == report:
            report_opens.append(path)
        return native_open(path, *args, **kwargs)
    def swap(path, *args, **kwargs):
        if path == report:
            report.unlink()
            label = bound["original"]["datasets"]["coco_object"]["cases"][0]["example"]["source_input"]["label"]
            os.link(label, report)
        return stable(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", watched_open)
    monkeypatch.setattr(worker, "stable_bytes", swap)
    with pytest.raises(ValueError, match="Frozen opened-file identity"):
        actual_admission(suite, digest)
    assert not report_opens and not (suite/"STARTED.json").exists()


def test_final_seal_close_failure_cannot_publish_complete_marker(tmp_path, monkeypatch):
    suite, _ = fixture(tmp_path, monkeypatch)
    original_open = Path.open
    class FailedClose:
        def __init__(self, stream):
            self.stream = stream
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            self.stream.__exit__(*args)
            raise OSError("staging close failed after complete bytes")
    def intercepted(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == suite/".ANALYSIS_SEALED.json.pending" and args and args[0] == "x":
            return FailedClose(stream)
        return stream
    monkeypatch.setattr(Path, "open", intercepted)
    with pytest.raises(OSError, match="staging close failed"):
        worker.run(suite, "f"*64)
    assert (suite/"FAILED.json").exists() and not (suite/"ANALYSIS_SEALED.json").exists()
    assert json.loads((suite/".ANALYSIS_SEALED.json.pending").read_text())["complete"]


def test_audit_drift_during_scoring_is_detected_before_final_seal(tmp_path, monkeypatch):
    suite, _ = fixture(tmp_path, monkeypatch)
    execute = worker.execute
    def drift(*args):
        result = execute(*args)
        (suite/"ADMITTED.json").write_text('{"different_reviewer":true}')
        return result
    monkeypatch.setattr(worker, "execute", drift)
    with pytest.raises(ValueError, match="digest differs|opened-file identity"):
        worker.run(suite, "f"*64)
    assert not (suite/"ANALYSIS_SEALED.json").exists() and (suite/"FAILED.json").exists()
    started = json.loads((suite/"STARTED.json").read_text())
    assert started["audit_binding"]["admission_sha256"] != sha256(suite/"ADMITTED.json")


def test_annotation_opened_descriptor_is_checked_before_any_byte_read(tmp_path, monkeypatch):
    from tools.evolution_fifty.risk_score_payload import annotation_bytes
    path, alternate = tmp_path/"frozen.png", tmp_path/"different.png"
    path.write_bytes(b"AAAA")
    alternate.write_bytes(b"BBBB")
    state = path.stat()
    row = dict(label=str(path), label_metadata=[state.st_size, state.st_mtime_ns], label_metadata_details=dict(
        size_bytes=state.st_size, mtime_ns=state.st_mtime_ns, device=state.st_dev, inode=state.st_ino, contents_read=False))
    original_open = Path.open
    class WatchRead:
        def __init__(self, stream):
            self.stream = stream
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def read(self, *_):
            pytest.fail("Read wrong inode before fstat validation")
    def intercepted(target, *args, **kwargs):
        return WatchRead(original_open(alternate, *args, **kwargs)) if target == path else original_open(target, *args, **kwargs)
    monkeypatch.setattr(Path, "open", intercepted)
    with pytest.raises(ValueError, match="descriptor identity"):
        annotation_bytes(row)


@pytest.mark.parametrize("method", [zipfile.ZIP_BZIP2, zipfile.ZIP_DEFLATED])
def test_forged_zip_size_cannot_force_unbounded_decompression(method, monkeypatch):
    import struct
    import zlib
    output = io.BytesIO()
    data = labels_content(np.ones((1, 1, 1), dtype=np.float32))+b"x"*(8*1024**2)
    with zipfile.ZipFile(output, "w", compression=method) as archive_file:
        archive_file.writestr("probabilities.npy", data)
    content = bytearray(output.getvalue())
    central = content.index(b"PK\x01\x02")
    struct.pack_into("<I", content, central+24, 132)
    original = zlib.decompressobj
    actual_output = []
    class WatchInflater:
        def __init__(self, real):
            self.real = real
        def __getattr__(self, name):
            return getattr(self.real, name)
        def decompress(self, source, maximum=0):
            assert 0 < maximum <= 4101
            result = self.real.decompress(source, maximum)
            actual_output.append(len(result))
            return result
    monkeypatch.setattr(zlib, "decompressobj", lambda *args: WatchInflater(original(*args)))
    with pytest.raises(ValueError, match="ZIP inventory|Actual decompression"):
        support_array(bytes(content), dict(shape=[1, 1, 1], probabilities_dtype="float32"), 1, [1, 1])
    assert not actual_output or max(actual_output) <= 4101


def test_declared_NPY_header_length_checked_before_header_body(monkeypatch):
    import struct
    from tools.evolution_fifty.risk_score_payload import npy_header
    class WatchHeader(io.BytesIO):
        def read(self, n=-1):
            assert 0 < n <= 8
            return super().read(n)
    data = b"\x93NUMPY\x02\x00"+struct.pack("<I", 8*1024**2)+b"unused"
    with pytest.raises(ValueError, match="pre-read bound"):
        npy_header(WatchHeader(data), [1, 1], "float32", len(data))


def test_runtime_inventory_has_no_snapshot_time_cache_exemptions(tmp_path):
    runtime = tmp_path/"runtime"
    runtime.mkdir()
    (runtime/"worker.py").write_text("# synthetic source\n")
    expected = {"worker.py":sha256(runtime/"worker.py")}
    assert worker.strict_runtime_inventory(runtime) == expected
    (runtime/"unexpected.py").write_text("# unapproved namespace file\n")
    assert worker.strict_runtime_inventory(runtime) != expected
    cache = runtime/"__pycache__"
    cache.mkdir()
    (cache/"worker.pyc").write_bytes(b"unapproved bytecode")
    with pytest.raises(ValueError, match="bytecode/cache"):
        worker.strict_runtime_inventory(runtime)


def test_complete_metadata_reference_graph_without_payload_or_GT_reads(tmp_path, monkeypatch):
    suite, bound = fixture(tmp_path, monkeypatch)
    replay, parent = Path(bound["replay"]), Path(bound["original"]["parent"])
    original = bound["original"]
    original.update(prepared_sha256="a"*64, barrier_sha256="b"*64)
    stored, unique, seals = 0, 0, {}
    for dataset, source in original["datasets"].items():
        source["spec"]["native_config"].update(dataset={"coco_object":"COCOObjectDataset", "pc60":"PascalContext60Dataset"}[dataset],
            resize=[224, 224], patch_size=[224, 224])
        source["artifact_sha256"] = {}
        inventory = {}
        for case, ref in zip(source["cases"], bound["references"][dataset], strict=True):
            case["checkpoints"] = []
            candidate = case["example"]["candidate"]
            candidate.update(confidence=.9, counterfactual_support=.9 if len(case["candidate_id"]) and int(case["case"][-1]) % 2 else .1,
                preupdate_candidate_category_idx=0)
            for point in ref["checkpoints"]:
                point["snapshot"] = f"{case['case']}/step_{point['step']:02d}.npz"
                source["artifact_sha256"][point["snapshot"]] = point["snapshot_sha256"]
                source["artifact_sha256"][point["labels"]] = point["labels_sha256"]
                case["checkpoints"].append({k:v for k,v in point.items() if k != "reference"})
                point["exact_changed_labels"] = 0
                key = hashlib.sha256(json.dumps(dict(dataset=dataset, original_episode=case["episode"],
                    source_id=case["source_id"], snapshot_sha256=point["snapshot_sha256"]), sort_keys=True).encode()).hexdigest()
                refdata = point["reference"]
                oldpath = replay/refdata["support"]
                name = f"{dataset}/supports/{key}.npz"
                target = replay/name
                if not target.exists():
                    target.write_bytes(oldpath.read_bytes())
                refdata["support"] = name
                inventory[name] = refdata["sha256"]
                point["exact_changed_labels"] = 0
        write_new(replay/dataset/"REFERENCES.json", bound["references"][dataset])
        refsha = sha256(replay/dataset/"REFERENCES.json")
        unique += len(inventory)
        stored += sum((replay/p).stat().st_size for p in inventory)
        inventory[f"{dataset}/REFERENCES.json"] = refsha
        write_new(replay/dataset/"SUPPORTS_SEALED.json", dict(complete=True, GT_opened=False,
            suite_prepared_sha256="f"*64, candidates=4, references_sha256=refsha,
            unique_readouts=len(inventory)-1, artifact_sha256=inventory))
        seals[dataset] = sha256(replay/dataset/"SUPPORTS_SEALED.json")
    # bind_parent() returns the pre-enriched graph in production.
    original_raw = json.loads(json.dumps(original))
    for source in original_raw["datasets"].values():
        for case in source["cases"]:
            for point in case["checkpoints"]:
                del point["snapshot_sha256"], point["labels_sha256"]
    write_new(replay/"PROTOCOL.json", dict(GT_opened=False))
    write_new(replay/"REPLAY_INPUTS.json", original)
    prepared = dict(root=str(replay), schema_version=1, runtime_hashes={},
        protocol_sha256=sha256(replay/"PROTOCOL.json"), inputs_sha256=sha256(replay/"REPLAY_INPUTS.json"))
    write_new(replay/"SUITE_PREPARED.json", prepared)
    digest = sha256(replay/"SUITE_PREPARED.json")
    for dataset in seals:
        path = replay/dataset/"SUPPORTS_SEALED.json"
        data = json.loads(path.read_text())
        data["suite_prepared_sha256"] = digest
        path.write_text(json.dumps(data))
        seals[dataset] = sha256(path)
    write_new(replay/"ALL_SUPPORTS_SEALED.json", dict(complete=True, GT_opened=False, no_adaptation=True,
        suite_prepared_sha256=digest, dataset_seals=seals, candidates=8, stored_bytes=stored))
    risk = bound["risk_protocol"]
    risk.update(parent_prepared_sha256=original["prepared_sha256"], parent_barrier_sha256=original["barrier_sha256"], source_code_sha256={})
    riskpath = tmp_path/"risk_protocol.json"
    write_new(riskpath, risk)
    monkeypatch.setattr(worker, "RISK_PROTOCOL_SHA", sha256(riskpath))
    monkeypatch.setattr(worker, "strict_runtime_inventory", lambda *_: {})
    monkeypatch.setattr(worker, "bind_parent", lambda *_: json.loads(json.dumps(original_raw)))
    monkeypatch.setattr(worker, "replay_terminal", lambda: bound["terminal"])
    monkeypatch.setattr(worker, "CANDIDATES", 8)
    monkeypatch.setattr(worker, "UNIQUE_READOUTS", unique)
    monkeypatch.setattr(worker, "annotation_bytes", lambda *_: pytest.fail("Metadata must not read GT"))
    monkeypatch.setattr(worker, "support_array", lambda *_: pytest.fail("Metadata must not decode supports"))
    monkeypatch.setattr(worker, "hard_array", lambda *_: pytest.fail("Metadata must not decode labels"))
    actual = worker.metadata_bind(replay, digest, riskpath)
    assert actual["candidates"] == 8 and actual["unique_readouts"] == unique and not actual["GT_opened"]
    first = replay/"coco_object"/"REFERENCES.json"
    altered = json.loads(first.read_text())
    altered[0]["source_id"] = "wrong source"
    first.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="digest or identity"):
        worker.metadata_bind(replay, digest, riskpath)
