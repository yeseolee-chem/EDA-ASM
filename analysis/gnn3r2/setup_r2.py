#!/usr/bin/env python3
"""GNN3R2 2-2 (docs/specs/GNN3R2.md): copy the GNN3R code, reference results and scratch files -> results/COPY.json.

    python analysis/gnn3r2/setup_r2.py code       # analysis/gnn3r/*.py + jobs/ (not delete_old.py) -> analysis/gnn3r2/,
                                                  # 7 GNN3R result files -> analysis/gnn3r2/ref_gnn3r/
    python analysis/gnn3r2/setup_r2.py scratch    # $GR/A/rowattr_r.pkl -> $GR2/ref_gnn3r/rowattr_r_gnn3r.pkl,
                                                  # $GR/B/p4_cache.pkl -> $GR2/B/p4_cache.pkl (after S0 1-5: $GR2 empty)

Standard library only; any CPU env. Copies are byte copies (tmp file + os.replace); every copy is checked by sha256
against its source. An existing copy with the source sha256 is kept. Exit 0 all equal, 3 = STOP (sha256 mismatch,
missing source, or a copy would overwrite a different file).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                   # analysis/gnn3r2
REPO = HERE.parents[1]
SRC = REPO / "analysis" / "gnn3r"
REF = HERE / "ref_gnn3r"
GR = Path("/gpfs/tmp_cpu2/yeseo1ee/gnn3r")
GR2 = Path("/gpfs/tmp_cpu2/yeseo1ee/gnn3r2")
EXCLUDE = {"delete_old.py"}
RESULT_FILES = ("F_pairs_r.csv", "B_row_status.csv", "B_reactions.csv", "S_folds_r.csv", "eval/metrics.csv",
                "RUN_REPORT_GNN3R.md", "DELETE_old_gnn3.json")
ROWATTR_SHA = "f0e7fe41001a933d20af756be5db724eb6db33ef40b36109c88ed2657b7e81e3"
COPY_JSON = HERE / "results" / "COPY.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(src: Path, dst: Path) -> dict:
    s = sha256(src)
    if dst.exists():
        d = sha256(dst)
        if d != s:
            return dict(src=str(src), dst=str(dst), src_sha256=s, dst_sha256=d, ok=False, note="existing copy differs")
        return dict(src=str(src), dst=str(dst), src_sha256=s, dst_sha256=d, ok=True, note="kept")
    dst.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.")
    os.close(handle)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    d = sha256(dst)
    return dict(src=str(src), dst=str(dst), src_sha256=s, dst_sha256=d, ok=d == s, note="copied")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(handle, "w") as out:
        out.write(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def main(part: str) -> int:
    rec = json.loads(COPY_JSON.read_text()) if COPY_JSON.is_file() else {}
    rows = []
    if part == "code":
        for p in sorted(SRC.glob("*.py")) + sorted((SRC / "jobs").glob("*")):
            if p.name in EXCLUDE or not p.is_file():
                continue
            rows.append(copy(p, HERE / p.relative_to(SRC)))
        for name in RESULT_FILES:
            rows.append(copy(SRC / "results" / name, REF / Path(name).name))
    else:
        r = copy(GR / "A" / "rowattr_r.pkl", GR2 / "ref_gnn3r" / "rowattr_r_gnn3r.pkl")
        r["want_sha256"] = ROWATTR_SHA
        r["ok"] = r["ok"] and r["src_sha256"] == ROWATTR_SHA
        rows += [r, copy(GR / "B" / "p4_cache.pkl", GR2 / "B" / "p4_cache.pkl")]
    rec[part] = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), slurm_job=os.environ.get("SLURM_JOB_ID"),
                     files=rows, all_equal=all(r["ok"] for r in rows))
    rec["all_equal"] = all(v["all_equal"] for k, v in rec.items() if isinstance(v, dict))
    write_json(COPY_JSON, rec)
    print(json.dumps(dict(part=part, n=len(rows), all_equal=rec[part]["all_equal"],
                          bad=[r for r in rows if not r["ok"]]), indent=1))
    return 0 if rec[part]["all_equal"] else 3


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("code", "scratch"):
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
