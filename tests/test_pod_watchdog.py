"""C9 on a stub: the in-pod watchdog and the laptop backup timer, no real pod and no spend."""

import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from edge_rag import backup_timer

ROOT = Path(__file__).resolve().parents[1]
POD_ID = "stub-pod-7"
KEY = "stub-key-not-a-secret"


def find_bash() -> str:
    """Git Bash on Windows (System32 bash.exe is WSL), plain bash elsewhere."""
    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    git_bash = program_files / "Git" / "bin" / "bash.exe"
    if os.name == "nt" and git_bash.exists():
        return str(git_bash)
    found = shutil.which("bash")
    assert found, "bash is needed for the pod scripts"
    return found


class FakeRunPod:
    """Local GraphQL stub: `myself { pods { id } }` and `podTerminate`, every request recorded."""

    def __init__(self, pods: list[str]):
        self.pods = list(pods)
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                query = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["query"]
                fake.requests.append({"query": query, "auth": self.headers["Authorization"]})
                if "podTerminate" in query:
                    fake.pods = [p for p in fake.pods if f'"{p}"' not in query]
                    data: dict = {"podTerminate": None}
                else:
                    data = {"myself": {"pods": [{"id": p} for p in fake.pods]}}
                body = json.dumps({"data": data}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/graphql"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def terminations(self) -> list[dict]:
        return [r for r in self.requests if "podTerminate" in r["query"]]


@pytest.fixture
def runpod():
    fakes: list[FakeRunPod] = []

    def make(pods):
        fakes.append(FakeRunPod(pods))
        return fakes[-1]

    yield make
    for fake in fakes:
        fake.server.shutdown()


def run_stub_pod(
    tmp_path: Path, cut_usd: str, runpodctl_exit: int, graphql_url: str, work_s: int = 20
) -> Path:
    """A stub stage script sourcing scripts/pod_watchdog.sh, with a fake runpodctl on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_ctl = bin_dir / "runpodctl"
    fake_ctl.write_text(
        "#!/usr/bin/env bash\n"
        'flushed=no; [ -f "$OUT/sha256sums.txt" ] && flushed=yes\n'
        'echo "$* flushed=$flushed" >> "$CALLS"\n'
        f"exit {runpodctl_exit}\n",
        newline="\n",
    )
    stage = tmp_path / "stage.sh"
    stage.write_text(
        "set -euo pipefail\n"
        f'source "{(ROOT / "scripts" / "pod_watchdog.sh").as_posix()}"\n'
        'stop_work() { kill "$(cat "$OUT/work.pid")"; }\n'
        'flush() { (cd "$OUT" && sha256sum part-*.jsonl > sha256sums.txt); }\n'
        'mkdir -p "$OUT"\n'
        "watchdog_start\n"
        'echo "{}" > "$OUT/part-1.jsonl"\n'
        f"sleep {work_s} &\n"
        'echo $! > "$OUT/work.pid"\n'
        'wait $! || { echo "STAGE_FAIL work" >> "$LOG"; exit 1; }\n'
        'echo "STAGE_DONE all" >> "$LOG"\n',
        newline="\n",
    )
    env = dict(
        os.environ,
        PATH=f"{bin_dir.as_posix()}{os.pathsep}{os.environ['PATH']}",
        POD_COST_PER_HR="3.6",  # 0.001 USD per second
        HARD_CUT_USD=cut_usd,
        WATCH_S="1",
        RUNPOD_POD_ID=POD_ID,
        RUNPOD_API_KEY=KEY,
        RUNPOD_GRAPHQL_URL=graphql_url,
        OUT=(tmp_path / "out").as_posix(),
        LOG=(tmp_path / "pod.log").as_posix(),
        SAMPLES=(tmp_path / "pod.samples").as_posix(),
        PROGRESS=(tmp_path / "pod.progress").as_posix(),
        CALLS=(tmp_path / "calls.txt").as_posix(),
    )
    # capture_output also waits for the watchdog, which holds the pipes until it ends
    subprocess.run(  # noqa: S603
        [find_bash(), stage.as_posix()], env=env, cwd=tmp_path, capture_output=True, timeout=60
    )
    log = tmp_path / "pod.log"
    return log


def markers(log: Path) -> list[str]:
    return [line.split()[0] for line in log.read_text().splitlines()]


def test_cut_stops_work_flushes_then_removes_the_pod(tmp_path, runpod):
    fake = runpod([POD_ID])
    log = run_stub_pod(tmp_path, cut_usd="0.002", runpodctl_exit=0, graphql_url=fake.url)
    seen = markers(log)
    assert seen.index("STOP_CUT") < seen.index("FLUSHED") < seen.index("TERMINATE_TRY")
    assert "STAGE_FAIL" in seen and "STAGE_DONE" not in seen
    assert (tmp_path / "out" / "sha256sums.txt").read_text().strip().endswith("part-1.jsonl")
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == [f"remove pod {POD_ID} flushed=yes"]
    assert fake.requests == []
    assert KEY not in log.read_text()


def test_cut_falls_back_to_graphql_when_runpodctl_fails(tmp_path, runpod):
    fake = runpod([POD_ID])
    log = run_stub_pod(tmp_path, cut_usd="0.002", runpodctl_exit=1, graphql_url=fake.url)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == [f"remove pod {POD_ID} flushed=yes", f"pod delete {POD_ID} flushed=yes"]
    assert [r["auth"] for r in fake.terminations()] == [f"Bearer {KEY}"]
    assert f'podId: "{POD_ID}"' in fake.terminations()[0]["query"]
    assert fake.pods == []
    assert "TERMINATE_FAIL" not in markers(log)
    assert KEY not in log.read_text()


def test_work_ending_before_the_cut_leaves_the_pod_alone(tmp_path, runpod):
    fake = runpod([POD_ID])
    log = run_stub_pod(tmp_path, cut_usd="1000", runpodctl_exit=0, graphql_url=fake.url, work_s=2)
    seen = markers(log)
    assert "STAGE_DONE" in seen and "STOP_CUT" not in seen
    assert not (tmp_path / "calls.txt").exists()
    assert fake.requests == []


def test_backup_timer_terminates_a_pod_that_is_still_there(runpod, capsys, monkeypatch):
    fake = runpod([POD_ID, "other-pod"])
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RUNPOD_GRAPHQL_URL", fake.url)
    args = ["--pod-id", POD_ID, "--cut-usd", "0.0005", "--rate", "3.6", "--extra-s", "0.2"]
    started = time.time()
    assert backup_timer.main(args) == 0
    assert time.time() - started >= 0.7  # 0.5 s cut plus 0.2 s extra
    assert len(fake.terminations()) == 1
    assert f'podId: "{POD_ID}"' in fake.terminations()[0]["query"]
    assert all(r["auth"] == f"Bearer {KEY}" for r in fake.requests)
    assert fake.pods == ["other-pod"]
    assert KEY not in capsys.readouterr().out


def test_backup_timer_stays_silent_when_the_pod_is_gone(runpod, capsys, monkeypatch):
    fake = runpod(["other-pod"])
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RUNPOD_GRAPHQL_URL", fake.url)
    args = ["--pod-id", POD_ID, "--cut-usd", "0", "--rate", "1", "--extra-s", "0"]
    assert backup_timer.main(args) == 0
    assert len(fake.requests) == 1 and fake.terminations() == []
    assert "already gone" in capsys.readouterr().out
