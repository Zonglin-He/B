"""Package the exact reviewed paper, raw reviews and bounded source evidence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_evolution"
PACKET = Path("/home/wwww/hzl/results/trace_evolution_review_round04_20261029_0703")
SUITE = Path("/home/wwww/hzl/results/trace_image_full_20261029_0333")
OUTPUT = Path("/home/wwww/hzl/results/trace_evolution_delivery_20261029_0715")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = sha(source)
    shutil.copy2(source, destination)
    if sha(source) != before or sha(destination) != before:
        raise RuntimeError(f"Copy identity failure: {source}")


def main():
    manifest = json.loads((PACKET / "packet_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(PACKET / name) != expected:
            raise RuntimeError(f"Reviewed packet drift: {name}")
    review = json.loads((PAPER / "reviews/round04.json").read_text())
    if review["overall_score"] != 6 or review["fatal_validity_issue_evidenced"]:
        raise RuntimeError("Unexpected raw review decision")
    expected_reviews = {
        "round04.md": "097e72b4ef47d6beadccc66a01ec4486800d78eff0dcc333bdcbd317734c7dd4",
        "round04.json": "732f1740f74a08b6b96b9165ceb8b343810849a644b952824d56c273d9c2f946",
    }
    for name, expected in expected_reviews.items():
        if sha(PAPER / "reviews" / name) != expected:
            raise RuntimeError("Raw review modified after completion")
    status = subprocess.check_output([
        "systemctl", "--user", "show", "trace-image-full-20261029-0333.service",
        "--property=LoadState,ActiveState,SubState,MainPID,ExecMainPID,Result,ExecMainCode,ExecMainStatus",
    ], text=True)
    fields = dict(line.split("=", 1) for line in status.splitlines() if "=" in line)
    expected = dict(LoadState="loaded", ActiveState="active", SubState="exited",
                    MainPID="0", ExecMainPID="3694645", Result="success",
                    ExecMainCode="1", ExecMainStatus="0")
    if fields != expected:
        raise RuntimeError(f"Completed service differs: {fields}")
    state = json.loads((SUITE / "queue_state.json").read_text())
    if state["phase"] != "completed" or state.get("error") is not None:
        raise RuntimeError("Full suite is not successfully complete")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in manifest["files"]:
        copy(PACKET / name, OUTPUT / "reviewed" / name)
    copy(PACKET / "packet_manifest.json", OUTPUT / "reviewed/packet_manifest.json")
    for path in sorted((PAPER / "reviews").glob("round*")):
        if path.is_file() and path.suffix in (".md", ".json"):
            copy(path, OUTPUT / "reviews" / path.name)
    copy(PAPER / "REPRODUCTION.md", OUTPUT / "REPRODUCTION.md")
    prepared = json.loads((SUITE / "SUITE_PREPARED.json").read_text())
    for name, expected in prepared["runtime_hashes"].items():
        source = SUITE / "runtime" / name
        if sha(source) != expected:
            raise RuntimeError(f"Executed runtime drift: {name}")
        copy(source, OUTPUT / "runtime" / name)
    for name in ("SUITE_PREPARED.json", "SUITE_SEALED.json", "ALL_FULL_PREDICTIONS_SEALED.json",
                 "EXECUTION_STARTED.json", "execution_children.json", "queue_state.json",
                 "frozen_plan.json", "rows.csv"):
        copy(SUITE / name, OUTPUT / "results/full" / name)
    for path in sorted((SUITE / "configs").glob("*.json")):
        copy(path, OUTPUT / "results/full/configs" / path.name)
    cohort = Path("/home/wwww/hzl/results/trace_image_budget_20261029_0233")
    for name in ("rows.csv", "SUITE_PREPARED.json", "SUITE_SEALED.json", "SELECTION_SEALED.json"):
        copy(cohort / name, OUTPUT / "results/calibration_confirmation" / name)
    for name in ("image-full-completion-audit.md", "image-full-main-writing-audit.md",
                 "image-full-appendix-writing-audit.md", "image-full-figure-audit.md"):
        copy(ROOT / "docs" / name, OUTPUT / "author_checks" / name)
    receipt = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        scope="Author delivery receipt, not a new independent review or experiment.",
        reviewer="01a184c1-91fa-7b81-a323-c3480cbb103a", reviewer_closed=True,
        score=6, recommendation="weak_accept", confidence=4,
        fatal_validity_issue_evidenced=False,
        reviewed_packet_manifest_sha256=sha(PACKET / "packet_manifest.json"),
        unchanged_review_sha256=expected_reviews, reviewed_files_verified=len(manifest["files"]),
        source_files_verified=len(prepared["runtime_hashes"]), service_terminal=fields,
        all_promised_experiments_complete=True, additional_gpu_jobs_started=False,
        remaining_weaknesses="See raw review and separate round04_disposition.md; not all experimentally closed.",
        included="Reviewed TeX/PDF/assets, raw review history, small result/provenance records and frozen runtime source.",
        excluded="Datasets, checkpoint weights, full prediction population, turnkey portable data release.",
    )
    (OUTPUT / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    hashes = {str(path.relative_to(OUTPUT)): sha(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file()}
    (OUTPUT / "delivery_manifest.json").write_text(json.dumps(dict(files=hashes), indent=2) + "\n")
    archive = OUTPUT.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(OUTPUT, arcname=OUTPUT.name)
    print(json.dumps(dict(output=str(OUTPUT), archive=str(archive), archive_bytes=archive.stat().st_size,
                          archive_sha256=sha(archive), files=len(hashes), receipt=receipt), indent=2))


if __name__ == "__main__":
    main()
