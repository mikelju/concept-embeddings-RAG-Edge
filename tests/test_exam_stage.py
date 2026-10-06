"""Phase 07 pod stage on hand-built examples (plan increment 5b): the probe's skip decision, the
fixed order and the per-system completion marker the laptop downloads on (D9). No pod, no spend."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from edge_rag.pod import exam_stage

ROOT = Path(__file__).resolve().parents[1]


def find_bash() -> str:
    """Git Bash on Windows (System32 bash.exe is WSL), plain bash elsewhere."""
    git_bash = (
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
    )
    if os.name == "nt" and git_bash.exists():
        return str(git_bash)
    found = shutil.which("bash")
    assert found, "bash is needed for the pod scripts"
    return found


def test_fixed_order_is_the_spec_order():
    assert exam_stage.ORDER == (
        "setup",
        "gliner",
        "g-l",
        "c3",
        "j-strong",
        "g-r2",
        "j-rrf3",
        "within-j-strong",
    )
    assert exam_stage.GA1_ORDER == ("setup", "g-a1")
    assert "setup" not in exam_stage.PROBE and "c3" not in exam_stage.PROBE
    assert exam_stage.PROBE["gliner"] == exam_stage.PROBE["g-l"] == 1_000
    assert exam_stage.PROBE["j-strong"] == exam_stage.PROBE["within-j-strong"] == 50


def test_probe_runs_an_item_that_fits_the_remaining_cut():
    # 50 questions in 10 s -> 1,451 in 290.2 s, 0.0597 USD at 0.74 USD/h; 1 h spent of 2.27 USD.
    result = exam_stage.decide(10, 50, 1_451, cut_usd=2.27, spent_seconds=3_600, rate_per_hr=0.74)
    assert result["run"] and result["state"] == "run"
    assert result["projected_seconds"] == pytest.approx(290.2)
    assert result["remaining_usd"] == pytest.approx(1.53)


def test_probe_skips_an_item_that_does_not_fit_and_counts_time_spent():
    # 50 questions in 100 s -> 2,902 s, 0.5965 USD: fits with nothing spent, not after 2.5 h.
    fits = exam_stage.decide(100, 50, 1_451, cut_usd=2.27, spent_seconds=0, rate_per_hr=0.74)
    late = exam_stage.decide(100, 50, 1_451, cut_usd=2.27, spent_seconds=9_000, rate_per_hr=0.74)
    assert fits["run"]
    assert not late["run"] and late["state"] == "not run (cut)"
    assert late["remaining_usd"] == pytest.approx(2.27 - 1.85)
    with pytest.raises(ValueError):
        exam_stage.decide(1, 0, 10, cut_usd=1, spent_seconds=0, rate_per_hr=1)


def test_completion_marker_names_every_file_and_detects_tampering(tmp_path):
    (tmp_path / "g-l").mkdir()
    (tmp_path / "g-l" / "rankings.jsonl.gz").write_bytes(b"abc")
    (tmp_path / "g-l" / "manifest.json").write_text("{}", "utf-8")
    assert not exam_stage.is_complete(tmp_path, "g-l")
    body = exam_stage.mark(tmp_path, "g-l", ["g-l/rankings.jsonl.gz", "g-l/manifest.json"])
    assert sorted(body["files"]) == ["g-l/manifest.json", "g-l/rankings.jsonl.gz"]
    assert exam_stage.is_complete(tmp_path, "g-l")
    with pytest.raises(FileExistsError):
        exam_stage.mark(tmp_path, "g-l", ["g-l/manifest.json"])
    (tmp_path / "g-l" / "rankings.jsonl.gz").write_bytes(b"abd")
    assert not exam_stage.is_complete(tmp_path, "g-l")


STUB_ITEMS = """
item_total() { case "$1" in g-l) echo 1000000000 ;; *) echo 100 ;; esac; }
stub() { echo "ran $1 limit=${2:-full}" >> "$CALLS"; echo "$1 ${2:-full}" > "$3/out.txt"; }
item_setup() { stub setup "$1" "$2"; }
item_gliner() { stub gliner "$1" "$2"; }
item_g-l() { stub g-l "$1" "$2"; }
item_c3() { stub c3 "$1" "$2"; }
item_j-strong() { stub j-strong "$1" "$2"; }
item_g-r2() { [ -z "$1" ] && return 1; stub g-r2 "$1" "$2"; }
item_j-rrf3() { stub j-rrf3 "$1" "$2"; }
item_within-j-strong() { stub within-j-strong "$1" "$2"; }
"""


def posix(path: Path) -> str:
    return path.as_posix()


def test_stage_script_order_probe_skip_markers_and_laptop_fetch(tmp_path):
    bash = find_bash()
    pod, laptop = tmp_path / "pod", tmp_path / "laptop"
    items = tmp_path / "items.sh"
    items.write_text(STUB_ITEMS, "utf-8", newline="\n")
    log, calls = tmp_path / "pod.log", tmp_path / "calls.txt"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "STAGE_PYTHON": posix(Path(sys.executable)),
        "POD_COST_PER_HR": "0.74",
        "HARD_CUT_USD": "2.27",
        "OUT": posix(pod),
        "LOG": posix(log),
        "SAMPLES": posix(tmp_path / "samples"),
        "PROGRESS": posix(tmp_path / "progress"),
        "ITEMS_LIB": posix(items),
        "CALLS": posix(calls),
        "WATCH_S": "1",
    }
    done = subprocess.run(  # noqa: S603
        [bash, "scripts/pod_exam.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    lines = log.read_text("utf-8").splitlines()
    events = [" ".join(line.split()[:2]) for line in lines if line.startswith("ITEM_")]
    assert events == [
        "ITEM_START setup",
        "ITEM_DONE setup",
        "ITEM_START gliner",
        "ITEM_DONE gliner",
        "ITEM_NOT_RUN_CUT g-l",
        "ITEM_START c3",
        "ITEM_DONE c3",
        "ITEM_START j-strong",
        "ITEM_DONE j-strong",
        "ITEM_START g-r2",
        "ITEM_FAIL g-r2",
        "ITEM_START j-rrf3",
        "ITEM_DONE j-rrf3",
        "ITEM_START within-j-strong",
        "ITEM_DONE within-j-strong",
    ]
    assert lines[-1].startswith("STAGE_DONE all")
    ran = calls.read_text("utf-8").splitlines()
    assert ran[:4] == [
        "ran setup limit=full",
        "ran gliner limit=1000",
        "ran gliner limit=full",
        "ran g-l limit=1000",
    ]
    assert "ran j-strong limit=50" in ran and "ran c3 limit=50" not in ran
    not_run = json.loads((pod / "g-l.not-run.json").read_text("utf-8"))
    assert not_run["state"] == "not run (cut)" and not_run["total_count"] == 1_000_000_000
    for item in ("setup", "gliner", "c3", "j-strong", "j-rrf3", "within-j-strong"):
        assert exam_stage.is_complete(pod, item)
    for item in ("g-l", "g-r2"):
        assert not (pod / f"{item}.complete.json").exists()
    assert (pod / "sha256sums.txt").exists()

    # Laptop side (D9): local stand-ins for ssh and scp; every complete item and the not-run
    # record are downloaded and accepted on their sha256; the failed item is not.
    remote_ls = tmp_path / "remote_ls.sh"
    remote_ls.write_text(
        f'cd "{posix(pod)}" && ls -1 $1 2> /dev/null; true\n', "utf-8", newline="\n"
    )
    remote_get = tmp_path / "remote_get.sh"
    remote_get.write_text(f'cp -r "{posix(pod)}/$1" "$2/"\n', "utf-8", newline="\n")
    remote_log = tmp_path / "remote_log.sh"
    remote_log.write_text(f'cat "{posix(log)}"\n', "utf-8", newline="\n")
    fetch_env = {
        **env,
        "LOCAL_OUT": posix(laptop),
        "REMOTE_LS": posix(remote_ls),
        "REMOTE_GET": posix(remote_get),
        "REMOTE_LOG_CMD": posix(remote_log),
        "POLL_S": "0",
    }
    fetched = subprocess.run(  # noqa: S603
        [bash, "scripts/exam_fetch.sh"],
        cwd=ROOT,
        env=fetch_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert fetched.returncode == 0, fetched.stderr
    fetch_log = (laptop / "fetch.log").read_text("utf-8")
    for item in ("setup", "gliner", "c3", "j-strong", "j-rrf3", "within-j-strong"):
        assert exam_stage.is_complete(laptop, item)
        assert f"FETCHED {item} " in fetch_log
    assert "FETCHED_NOT_RUN g-l" in fetch_log and (laptop / "g-l.not-run.json").exists()
    assert not (laptop / "g-r2").exists() and "FETCH_END" in fetch_log
