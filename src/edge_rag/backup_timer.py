"""Laptop backup timer for a pod's hard cut (Phase 07 spec C9, plan increment 1).

Started on the laptop when the pod is created: it waits the pod's cut (cut USD over the offered
rate) plus 5 min, then reads `myself { pods { id } }` and calls `podTerminate` only if the pod
still exists. The key is `RUNPOD_API_KEY` from the environment and is never printed.
    uv run python -m edge_rag.backup_timer --pod-id <id> --cut-usd 2.27 --rate 0.74
"""

import argparse
import json
import os
import time
import urllib.request

API_URL = "https://api.runpod.io/graphql"
EXTRA_S = 300.0
RETRIES = 4
RETRY_S = 30.0


def graphql(query: str, url: str, key: str) -> dict:
    """One GraphQL call; raises on HTTP or GraphQL errors."""
    request = urllib.request.Request(  # noqa: S310 - fixed API URL, or a local stub in tests
        url,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        body = json.loads(response.read())
    if body.get("errors"):
        raise RuntimeError(f"GraphQL errors: {body['errors']}")
    return body["data"]


def pod_exists(pod_id: str, url: str, key: str) -> bool:
    pods = graphql("query { myself { pods { id } } }", url, key)["myself"]["pods"] or []
    return any(pod["id"] == pod_id for pod in pods)


def backup(pod_id: str, wait_s: float, url: str, key: str, sleep=time.sleep) -> bool:
    """Wait, then terminate the pod if it still exists; True when podTerminate was called."""
    sleep(wait_s)
    for attempt in range(RETRIES):  # a network blip at expiry must not leave the pod billing
        try:
            return _terminate_if_alive(pod_id, url, key)
        except OSError as error:
            print(f"[WARN] attempt {attempt + 1} failed: {type(error).__name__}")
            sleep(RETRY_S)
    return _terminate_if_alive(pod_id, url, key)


def _terminate_if_alive(pod_id: str, url: str, key: str) -> bool:
    if not pod_exists(pod_id, url, key):
        print(f"[OK] pod {pod_id} already gone; backup timer silent")
        return False
    graphql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}', url, key)
    print(f"[WARN] pod {pod_id} still existed at the backup expiry; podTerminate called")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--cut-usd", type=float, required=True)
    parser.add_argument("--rate", type=float, required=True, help="offered costPerHr, USD/h")
    parser.add_argument("--extra-s", type=float, default=EXTRA_S)
    args = parser.parse_args(argv)
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("[ERROR] RUNPOD_API_KEY is not set")
        return 2
    wait_s = args.cut_usd / args.rate * 3600 + args.extra_s
    url = os.environ.get("RUNPOD_GRAPHQL_URL", API_URL)
    print(f"[OK] backup timer for pod {args.pod_id}: {wait_s:.0f} s")
    backup(args.pod_id, wait_s, url, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
