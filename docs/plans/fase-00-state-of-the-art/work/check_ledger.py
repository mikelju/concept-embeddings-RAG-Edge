"""Merge and validate the Phase 00 ledgers (spec C1, C3).

Usage: python check_ledger.py
Reads work/ledger-cheap.json and work/ledger-expensive.json, writes ../ledger.json sorted by id,
prints problems and counts, and, if ../survey.md exists, the verified share of the record ids
cited there (every backticked id such as `c-...` or `e-...`).
Exit code 1 if any problem is found.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
INPUTS = ["ledger-cheap.json", "ledger-expensive.json"]
REQUIRED = [
    "id", "system", "family", "cost_class", "benchmark", "setting", "corpus_size", "split",
    "metric", "k_or_budget", "value", "unit", "source", "self_reported", "verified",
    "why_unverified", "llm_in_loop", "offline_cost", "online_cost", "latency", "code_url",
    "weights_public", "licence", "notes", "why_missing",
]
# Fields that must never be null, whatever why_missing says.
NEVER_NULL = ["id", "system", "family", "cost_class", "benchmark", "setting", "metric", "value",
              "unit", "source", "verified"]


def load():
    records = []
    for name in INPUTS:
        data = json.loads((HERE / name).read_text(encoding="utf-8"))
        records.extend(data)
    return records


def check(records):
    problems = []
    ids = Counter(r.get("id") for r in records)
    for rid, n in ids.items():
        if n > 1:
            problems.append(f"duplicate id {rid} ({n} times)")
    for r in records:
        rid = r.get("id")
        for key in REQUIRED:
            if key not in r:
                problems.append(f"{rid}: missing key {key}")
        for key in NEVER_NULL:
            if r.get(key) in (None, ""):
                problems.append(f"{rid}: {key} is empty")
        src = r.get("source") or {}
        if not src.get("locator"):
            problems.append(f"{rid}: source has no locator")
        why = r.get("why_missing") or {}
        for key in REQUIRED:
            if key in ("why_missing", "why_unverified"):
                continue
            if key in r and r[key] in (None, "") and key not in NEVER_NULL and not why.get(key):
                problems.append(f"{rid}: {key} is null without why_missing")
        if r.get("verified") is False and not r.get("why_unverified"):
            problems.append(f"{rid}: verified false without why_unverified")
        if r.get("verified") is True and r.get("why_unverified") not in (None, ""):
            problems.append(f"{rid}: verified true but why_unverified is set")
    return problems


def print_counts(title, counter):
    print(title)
    for key, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"  {key}: {n}")


def cited_share(records):
    survey = PHASE / "survey.md"
    if not survey.exists():
        return
    by_id = {r["id"]: r for r in records}
    cited = sorted(set(re.findall(r"`([ce]-[a-z0-9][a-z0-9._-]*)`", survey.read_text(encoding="utf-8"))))
    unknown = [c for c in cited if c not in by_id]
    known = [c for c in cited if c in by_id]
    verified = [c for c in known if by_id[c]["verified"] is True]
    print("Ids cited in survey.md")
    print(f"  cited: {len(cited)}, in ledger: {len(known)}, unknown: {len(unknown)}")
    for c in unknown:
        print(f"  UNKNOWN id: {c}")
    if known:
        share = 100.0 * len(verified) / len(known)
        print(f"  verified: {len(verified)} of {len(known)} = {share:.1f} percent (C3 requires >= 80)")
        for c in known:
            if by_id[c]["verified"] is not True:
                print(f"  not verified: {c}")
    return unknown


def main():
    records = sorted(load(), key=lambda r: r.get("id") or "")
    problems = check(records)
    (PHASE / "ledger.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Records: {len(records)}, written to ledger.json")
    print(f"Problems: {len(problems)}")
    for p in problems:
        print("  " + p)
    print_counts("By cost class", Counter(r["cost_class"] for r in records))
    print_counts("By family", Counter(r["family"] for r in records))
    print_counts("By benchmark", Counter(r["benchmark"] for r in records))
    print_counts("By verified", Counter(str(r["verified"]) for r in records))
    print_counts("By self_reported", Counter(str(r["self_reported"]) for r in records))
    print_counts("By metric", Counter(r["metric"] for r in records))
    unknown = cited_share(records)
    sys.exit(1 if problems or unknown else 0)


if __name__ == "__main__":
    main()
