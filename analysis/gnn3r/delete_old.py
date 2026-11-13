#!/usr/bin/env python3
"""GNN3R clean-up after Phase R (user 2026-11-13: "기존 gnn3는 모두 끝난 후 혼돈방지를 위해 삭제할것").

    python analysis/gnn3r/delete_old.py          reactot or env_d1fu2, CPU; after the final GNN3R commit / push

Deletes the old GNN3 scratch (/gpfs/tmp_cpu2/yeseo1ee/gnn3: GNN3_CLEAN dev / equivalence / unit runs and logs) and the
GNN3R preparation folder (/gpfs/tmp_cpu2/yeseo1ee/gr_prep). The old worktree and branch (eda-asm-prediction-gnn3,
gnn3-clean local + origin) are removed with git on the login node. The gnn3/ package code stays: GNN3R imports it.
Writes results/DELETE_old_gnn3.json (sizes before, absent after). Exit 0 done, 3 = a path is a symlink / missing check.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

TARGETS = (Path("/gpfs/tmp_cpu2/yeseo1ee/gnn3"), Path("/gpfs/tmp_cpu2/yeseo1ee/gr_prep"))


def tree_size(root: Path) -> dict:
    n, b = 0, 0
    for folder, _dirs, names in os.walk(root):
        for name in names:
            p = Path(folder) / name
            n += 1
            b += 0 if p.is_symlink() else p.stat().st_size
    return dict(path=str(root), n_files=n, bytes=b, mb=round(b / 1e6, 1))


def main() -> int:
    rec = {}
    for p in TARGETS:
        if p.is_symlink() or os.path.realpath(p) != str(p) or p == C.GR:
            print(f"STOP: {p} is a symlink / not its own real path / the GNN3R scratch")
            return 3
        rec[str(p)] = dict(before=tree_size(p) if p.is_dir() else None, existed=p.is_dir())
    for p in TARGETS:
        if p.is_dir():
            shutil.rmtree(p)
        rec[str(p)]["absent_after"] = not p.exists()
    out = dict(request="기존 gnn3는 모두 끝난 후 혼돈방지를 위해 삭제할것 (user 2026-11-13)",
               written=time.strftime("%Y-%m-%d %H:%M:%S"), slurm_job=os.environ.get("SLURM_JOB_ID"), deleted=rec,
               git_on_login=["git worktree remove /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3",
                             "git branch -D gnn3-clean", "git push origin --delete gnn3-clean"],
               kept=dict(gnn3_package="gnn3/ in branch gnn3r (imported by analysis/gnn3r)",
                         history="gnn3-clean dd36bc0a is an ancestor of gnn3r: no commit is lost"),
               all_absent=all(v["absent_after"] for v in rec.values()))
    C.write_json(C.RES / "DELETE_old_gnn3.json", out)
    print(json.dumps(out, indent=1))
    return 0 if out["all_absent"] else 3


if __name__ == "__main__":
    sys.exit(main())
