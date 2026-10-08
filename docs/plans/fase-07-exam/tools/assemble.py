"""Deviation 07.4: assemble data/phase07/exam-rankings without touching src/ or scripts/."""
import hashlib, json, shutil, time
from pathlib import Path
from edge_rag import config, scoring
from edge_rag.exam_laptop import read_split, rrf4_lists, write_system

P7 = config.DATA_DIR / "phase07"
LAP, POD, OUT = P7 / "test" / "laptop", P7 / "pod", P7 / "exam-rankings"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text("utf-8"))

def upstream(src: Path) -> dict:
    """Every recorded sha256 for this file; all must agree with the file."""
    rec = {}
    if src.is_relative_to(LAP):
        m = src.parent / src.name.replace(".jsonl.gz", ".complete.json")
        rec[str(m.relative_to(P7))] = load(m)["sha256"]
        man = src.parent / "manifest.json"
        if man.exists() and src.name in load(man):
            rec[str(man.relative_to(P7))] = load(man)[src.name]
    else:
        rel = src.relative_to(POD).as_posix()
        stage = POD / f"{rel.split('/')[0]}.complete.json"
        rec[str(stage.relative_to(P7))] = load(stage)["files"][rel]
        man = load(src.parent / src.name.replace(".jsonl.gz", ".manifest.json"))
        rec[str((src.parent / src.name.replace('.jsonl.gz', '.manifest.json')).relative_to(P7))] = (
            man.get("sha256") or man["outputs"][src.name])
    return rec

PLAN = {
    "pooled": {"dense": LAP / "pooled/dense.jsonl.gz", "bm25": LAP / "pooled/bm25.jsonl.gz",
               "hop": LAP / "pooled/hop.jsonl.gz", "p10-b": LAP / "pooled/p10-b.jsonl.gz",
               "p14": LAP / "pooled/p14.jsonl.gz", "G-L": POD / "g-l/g-l.jsonl.gz",
               "G-R": POD / "j-strong/g-r.jsonl.gz", "j-rrf4": POD / "j-strong/j-rrf4.jsonl.gz",
               "G-R2": POD / "g-r2/g-r2.jsonl.gz", "j-rrf3": POD / "j-rrf3/j-rrf3.jsonl.gz",
               "G-A1": POD / "g-a1/rankings/qasper/g-a1.jsonl.gz"},
    "within": {"dense": LAP / "within/dense.jsonl.gz", "bm25": LAP / "within/bm25.jsonl.gz",
               "within-j-strong": POD / "within-j-strong/within-j-strong.jsonl.gz"},
}
assert not OUT.exists(), OUT
log = {}
for setting, systems in PLAN.items():
    (OUT / setting).mkdir(parents=True)
    for name, src in systems.items():
        rec, have = upstream(src), sha(src)
        assert set(rec.values()) == {have}, (src, rec, have)
        dst = OUT / setting / f"{name}.jsonl.gz"
        shutil.copyfile(src, dst)
        assert sha(dst) == have, dst
        (OUT / setting / f"{name}.complete.json").write_text(json.dumps({
            "system": name, "sha256": have, "copied_from": src.relative_to(P7).as_posix(),
            "upstream_sha256": rec, "deviation": "07.4",
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n", "utf-8")
        log[f"{setting}/{name}"] = have
        print(f"[OK] {setting}/{name} {have} <- {src.relative_to(P7).as_posix()} ({len(rec)} records agree)")

_, questions = read_split(P7 / "test")
qids = [q.qid for q in questions]
r = lambda n: scoring.read_rankings(OUT / "pooled" / f"{n}.jsonl.gz")
inputs = {n: log[f"pooled/{n}"] for n in ("dense", "bm25", "G-L", "hop")}
fused = rrf4_lists(r("dense"), r("bm25"), r("G-L"), r("hop"))
marker = write_system(OUT / "pooled", "rrf4", fused, qids,
                      {"deviation": "07.4", "inputs_sha256": inputs})
print(f"[OK] pooled/rrf4 {marker['sha256']} ({len(fused)} rankings, {len(qids)} questions)")
pod_first = load(POD / "j-strong/j-rrf4.manifest.json")["first_stage"]
print("[OK] pod j-rrf4 first-stage inputs equal rrf4 inputs:",
      pod_first == {"dense": inputs["dense"], "bm25": inputs["bm25"], "g-l": inputs["G-L"], "hop": inputs["hop"]})
