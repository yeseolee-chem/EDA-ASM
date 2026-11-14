#!/usr/bin/env python3
"""GNN3R2 Phase X (docs/specs/GNN3R2.md 11, D-13): delete the previous GNN3R scratch after GNN3R2 passed and was pushed.

    python analysis/gnn3r2/delete_gnn3r.py        any CPU env (gnn3/jobs/run.slurm), submitted from the gnn3r2 worktree

Conditions (any false -> exit 3, nothing deleted): every GNN3R2 report passed (S0, COPY, A, B, B_label, G, S, S-5, D,
K, E, W); RUN_REPORT_GNN3R2.md and gnn3r2.json (no missing path) are in origin/gnn3r2; analysis/gnn3r2 and the spec have
no uncommitted change; HEAD = origin/gnn3r2; origin/gnn3r is an ancestor of origin/gnn3r2.
Deletes only $GR = /gpfs/tmp_cpu2/yeseo1ee/gnn3r (A, B, cv, final .pt, logs). The sizes are written to
results/DELETE_old_gnn3r.json before rmtree; a finished record is not rewritten on resubmit. The git steps (worktree
remove, branch -D gnn3r, push origin --delete gnn3r, git rm -r analysis/gnn3r) run on the login node afterwards and are
listed in the record.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

TARGETS = (C.GR_GNN3R,)
REPORTS = ("S0_start.json", "COPY.json", "A_report.json", "B_report.json", "B_label_report.json", "G_report.json",
           "S_splits_r.json", "S5_report.json", "D_check.json", "cv/K_check.json", "eval/E_check.json",
           "final/W_check.json")
PUSHED_FILES = ("analysis/gnn3r2/results/RUN_REPORT_GNN3R2.md", "analysis/gnn3r2/results/gnn3r2.json",
                "analysis/gnn3r2/results/F_pairs_r.csv", "docs/specs/GNN3R2.md")
RECORD = C.RES / "DELETE_old_gnn3r.json"
REQUEST = "이전 GNN3R은 새 결과가 모두 통과하고 push된 뒤 지운다 (GNN3R2 D-13; 사용자 방침 2026-11-13: 혼동 방지)"


def git(*a):
    r = subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def tree_size(root: Path) -> dict:
    n, b = 0, 0
    for folder, _dirs, names in os.walk(root):
        for name in names:
            p = Path(folder) / name
            n += 1
            b += 0 if p.is_symlink() else p.stat().st_size
    return dict(path=str(root), n_files=n, bytes=b, gb=round(b / 1e9, 3))


def ok_report(name: str) -> bool:
    p = C.RES / name
    if not p.is_file():
        return False
    r = C.read_json(p)
    return bool(r.get("all_pass") if "all_pass" in r else r.get("all_equal"))


def main() -> int:
    prev = C.read_json(RECORD) if RECORD.is_file() else None
    if prev and prev.get("done") and prev.get("all_absent") and not any(p.exists() for p in TARGETS):
        print(f"Phase X done ({RECORD}); not rewritten")
        return 0
    reports = {n: ok_report(n) for n in REPORTS}
    head, origin = git("rev-parse", "HEAD")[1], git("rev-parse", "origin/gnn3r2")[1]
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=all", "--", "analysis/gnn3r2",
                            "docs/specs/GNN3R2.md")[1].splitlines()
             if l.strip() and not l.endswith("analysis/gnn3r2/results/DELETE_old_gnn3r.json")]
    in_origin = {f: git("cat-file", "-e", f"origin/gnn3r2:{f}")[0] == 0 for f in PUSHED_FILES}
    man = C.RES / "gnn3r2.json"
    n_missing = C.read_json(man)["header"]["n_missing_paths"] if man.is_file() else None
    kept_history = git("merge-base", "--is-ancestor", "origin/gnn3r", "origin/gnn3r2")[0] == 0
    cond = dict(reports=reports, head=head, origin_gnn3r2=origin, head_is_origin_gnn3r2=bool(head) and head == origin,
                uncommitted=dirty[:20], pushed_files=in_origin, manifest_missing_paths=n_missing,
                origin_gnn3r=git("rev-parse", "origin/gnn3r")[1], gnn3r_ancestor_of_gnn3r2=kept_history)
    gr2 = Path(os.path.realpath(C.GR))
    safe = {str(p): (not p.is_symlink() and os.path.realpath(p) == str(p) and p != gr2 and p not in gr2.parents)
            for p in TARGETS}
    go = (all(reports.values()) and cond["head_is_origin_gnn3r2"] and not dirty and all(in_origin.values())
          and n_missing == 0 and kept_history and all(safe.values()))
    out = dict(request=REQUEST, spec="docs/specs/GNN3R2.md 11", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), conditions=cond, paths_safe=safe, deleted={},
               git_on_login=["git worktree remove /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3r",
                             "git branch -D gnn3r", "git push origin --delete gnn3r",
                             "git rm -r analysis/gnn3r (branch gnn3r2)"],
               kept=dict(specs="docs/specs/GNN3R.md, docs/specs/gnn3r_ref/ (read by the copied code)",
                         gnn3_package="gnn3/", reference_copies="analysis/gnn3r2/ref_gnn3r/, $GR2/ref_gnn3r/",
                         prep_logs="/gpfs/tmp_cpu2/yeseo1ee/gr2_prep (GNN3R2 S0 / copy logs; not part of spec 11)",
                         history="origin/gnn3r is an ancestor of origin/gnn3r2: no commit is lost"),
               all_absent=False, done=False)
    if go:
        old = (prev or {}).get("deleted", {})
        for p in TARGETS:                                # sizes first (kept from an earlier partial run)
            before = tree_size(p) if p.is_dir() else None
            if old.get(str(p), {}).get("before"):
                before = old[str(p)]["before"]
            out["deleted"][str(p)] = dict(before=before, existed_now=p.is_dir())
        C.write_json(RECORD, out)
        for p in TARGETS:
            if p.is_dir():
                shutil.rmtree(p)
            out["deleted"][str(p)]["absent_after"] = not p.exists()
        out["all_absent"] = all(v["absent_after"] for v in out["deleted"].values())
        out["done"] = out["all_absent"]
        out["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.write_json(RECORD, out)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if out["done"] else 3


if __name__ == "__main__":
    sys.exit(main())
