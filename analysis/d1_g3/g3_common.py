#!/usr/bin/env python3
"""D1-G3 common paths, constants and helpers (docs/specs/D1G3.md). sbatch only.

Writes only under $G3 (= $D1G3_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/d1_g3) and analysis/d1_g3/results (spec 8);
D1LADDER ($LAD, analysis/d1_ladder/results), D1-G2 ($G2, analysis/d1_g2/results), D1-HF and the D1 run / job folders
are read only. Trained models and label files are never changed and nothing is retrained (spec 8).
Quota note (2026-10-24: 293 / 300 GB): keep ORCA outputs small (no .gbw / densities kept, outputs gzipped).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
G3 = Path(os.environ.get("D1G3_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_g3"))
LAD = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_ladder")             # D1LADDER scratch (read only)
LAD_DIR = REPO / "analysis" / "d1_ladder"                  # D1LADDER code (read only, imported)
LAD_RES = LAD_DIR / "results"                              # D1LADDER results (read only)
G2 = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_g2")                  # D1-G2 scratch (read only)
G2_RES = REPO / "analysis" / "d1_g2" / "results"           # D1-G2 results (read only; C_compare.csv)
HF_DIR = REPO / "analysis" / "d1_hf"                       # D1-HF code (runner, build) and results (read only)
HF_SCR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_hf")              # D1-HF scratch (read only: inputs/<T>/hf templates)
P_DIR = REPO / "analysis" / "d1_pair"                      # D1PAIR / FU2 code (read only)
METHODS = ("g3h", "g3r")                                   # spec 0: HF-D3BJ/6-31G* SMD, r2SCAN-3c SMD
SEED_S = 20261024                                          # spec 2 sample
SEED_BOOT = 20261019                                       # spec 6 E-3 bootstrap
KCAL = 627.5094740631
sys.path.insert(0, str(LAD_DIR))
import lad_common as L  # noqa: E402                       (DP = d1pair_common, targets(), D1LADDER paths)

DP = L.DP


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def assert_path(p):
    """writes only under $G3 or analysis/d1_g3/results (spec 8); exit 3 otherwise."""
    q = Path(os.path.abspath(p))
    for root in (G3, RES):
        try:
            q.relative_to(Path(os.path.abspath(root)))
            return q
        except ValueError:
            continue
    stop(f"write path {q} outside $G3 / analysis/d1_g3/results")


def write_atomic(path, data):
    """atomic write with a per-process temporary name (concurrent writers never share a .tmp)."""
    path = assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if callable(data):
        data(tmp)
    else:
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_json(path, obj):
    return write_atomic(path, json.dumps(obj, indent=1, default=_jsonable))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def targets():
    """the 9,266 non-D'' rows of labels_d1.json (job_id order)."""
    return L.targets()


def record_decision(key, text):
    """spec 11: append a decision to results/DECISIONS.md (idempotent per key)."""
    p = RES / "DECISIONS.md"
    old = p.read_text() if p.is_file() else "# D1-G3 결정 기록 (명령서 11절: 추천안을 고르고 적는다)\n\n"
    line = f"- **{key}**: {text}\n"
    if f"- **{key}**:" not in old:
        write_atomic(p, old + line)
