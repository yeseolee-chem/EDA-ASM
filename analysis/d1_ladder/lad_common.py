#!/usr/bin/env python3
"""D1-REF·LADDER common paths, constants and helpers (docs/specs/D1LADDER.md). sbatch only.

  python lad_common.py start     # spec 1 start conditions 0-1 .. 0-4 -> results/start_report.json (exit 3 = STOP)

Writes only under $LAD (= $D1LADDER_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/d1_ladder) and analysis/d1_ladder/results
(spec 15); labels_d1.json, the D1 run / job folders, d1_pair, d1_hf and $G2 scratch and results are read only.
Start-condition decisions (user 2026-10-22: every branch = the recommended option, reported):
  0-2  analysis/d1_pair differs from 2d745d23 only by the D1-G2 Phase T root-argument additions (d1_features.py
       runroot / mergeroot, fu2/atoms.py root ...; insertions only, defaults verified by T-1 / T-2; fu2_common.py,
       tokens.py, models.py, train.py byte-identical): accepted as 'same except the validated root arguments'.
  0-3  d1-g2 HEAD did not contain the D1-HF final results; d1-hf (6b9fbeb2) was merged into d1-ladder (2e45b774), so
       analysis/d1_hf/results equals 6b9fbeb2 byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
LAD = Path(os.environ.get("D1LADDER_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder"))
sys.path.insert(0, str(REPO / "analysis" / "d1_pair"))
import d1pair_common as DP  # noqa: E402              (load_d1_labels, D1_JOBS, DPP, SCRATCH, G1_ROOT_D1F, ...)

HF_DIR = REPO / "analysis" / "d1_hf"                   # D1-HF code and results (6b9fbeb2; read only)
HF_RES = HF_DIR / "results"
HF_SCR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_hf")          # D1-HF scratch: inputs/<T>/{hf,g1}/*.inp, runs/{dft,g1}/<T>
G2_DIR = REPO / "analysis" / "d1_g2"                   # D1-G2 code and results (read only)
G2 = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_g2")              # $G2 (read only)
P_DIR = REPO / "analysis" / "d1_pair"                  # D1PAIR / FU2 code (read only)
SEED_V = 20261023                                      # spec 3 V-1, V-2, V-5
SEED_M = 20261019                                      # spec 4-3, 11
KCAL = 627.5094740631
COMMITS = dict(d1pair="2d745d23", d1hf="6b9fbeb2")
D1HF_LABELS = HF_RES / "labels_d1_hf.parquet"


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def assert_path(p):
    """writes only under $LAD or analysis/d1_ladder/results (spec 15); exit 3 otherwise."""
    q = Path(os.path.abspath(p))
    for root in (LAD, RES):
        try:
            q.relative_to(Path(os.path.abspath(root)))
            return q
        except ValueError:
            continue
    stop(f"write path {q} outside $LAD / analysis/d1_ladder/results")


def write_atomic(path, data):
    path = assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if callable(data):
        data(tmp)
    else:
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    return write_atomic(path, json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout


def targets():
    """the 9,266 non-D'' rows of labels_d1.json (= D1-HF / D1-G2 targets), job_id order."""
    return sorted((d for d in DP.load_d1_labels() if d["skeleton"] not in DP.DPP), key=lambda d: d["job_id"])


def cmd_start():
    rep, bad = {}, []
    # 0-1 D1-G2 Phase T committed and passed
    t = json.loads((G2_DIR / "results" / "T_report.json").read_text())
    tracked = git("ls-files", "analysis/d1_g2/results/T_report.json").strip()
    rep["0-1"] = dict(T_all_pass=t.get("all_pass"), F_ready=t.get("F_ready"), committed=bool(tracked))
    if not (t.get("all_pass") and tracked):
        bad.append("0-1")
    # 0-2 analysis/d1_pair = 2d745d23 except the validated root-argument additions
    names = git("diff", "--name-only", COMMITS["d1pair"], "HEAD", "--", "analysis/d1_pair").split()
    num = git("diff", "--numstat", COMMITS["d1pair"], "HEAD", "--", "analysis/d1_pair").split("\n")
    dels = sum(int(x.split()[1]) for x in num if x.strip())
    allowed = {"analysis/d1_pair/d1_features.py", "analysis/d1_pair/fu2/atoms.py"}
    core = ["fu2/fu2_common.py", "fu2/tokens.py", "fu2/models.py", "fu2/train.py", "d1pair_common.py"]
    same_core = {f: not git("diff", "--name-only", COMMITS["d1pair"], "HEAD", "--", f"analysis/d1_pair/{f}").strip()
                 for f in core}
    rep["0-2"] = dict(changed=names, deletions=dels, core_identical=same_core,
                      decision="accepted: only the D1-G2 Phase T root-argument additions (T-1 / T-2 passed)")
    if set(names) - allowed or dels or not all(same_core.values()):
        bad.append("0-2")
    # 0-3 D1-HF results = 6b9fbeb2
    blob = git("rev-parse", f"{COMMITS['d1hf']}:analysis/d1_hf/results/labels_d1_hf.parquet").strip()
    here = git("hash-object", str(D1HF_LABELS)).strip()
    rdiff = git("diff", "--name-only", COMMITS["d1hf"], "HEAD", "--", "analysis/d1_hf/results").split()
    rep["0-3"] = dict(labels_d1_hf_sha256=sha256(D1HF_LABELS), git_blob=here, blob_6b9fbeb2=blob,
                      results_diff_vs_6b9fbeb2=rdiff)
    if blob != here or rdiff:
        bad.append("0-3")
    # 0-4 labels_d1.json (load_d1_labels checks sha256 and 9,496 status-ok rows; exits 3 otherwise)
    lab = DP.load_d1_labels()
    rep["0-4"] = dict(n=len(lab), sha256=DP.D1_LABELS_SHA, n_targets=len(targets()))
    rep.update(fail=bad, stop=bool(bad), git_head=git("rev-parse", "--short", "HEAD").strip())
    write_json(RES / "start_report.json", rep)
    print(json.dumps(rep, indent=1, default=str), flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["start"]:
        sys.exit(cmd_start())
    raise SystemExit(__doc__)
