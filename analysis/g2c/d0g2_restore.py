#!/usr/bin/env python3
"""D0-G2 restore after the partial deletion of 2026-10-30 (cleanup job 1054016 removed $D0G2 from 11:10:42 until the
user cancelled it at 11:12:13: "D0-G2는 필요할 것으로 보여서 삭제 멈춤. 필요하다면 삭제된것만 다시 계산").
D1 run env, sbatch only (GEO1B DECISIONS 0-D0).   python d0g2_restore.py scan

Needs $D0G2/G/labels_d0_g2.json, regenerated (deterministically, from the D0 labels and profiles) by `d0g2.py manifest`.
Per D0 target reaction:
  kept_done : g2_final/<rxn_id>/ has .done, ts.xyz, rel1.xyz, rel2.xyz, result.json, ts.hess and work/;
  kept_fail : it has a .fail_* marker (and no .done);
  partial   : the folder exists but is neither, or a directory inside it was modified after CUTOFF (rm was inside it
              when it was cancelled) -> moved, never deleted, to $D0G2/restore/partial/<rxn_id>;
  deleted   : no folder.
Every kept folder must agree with GEO1 Phase Z (analysis/geo/results/Z_status.csv: ok <-> .done, fail <-> .fail_*).
restore_list.csv (job_id) = deleted + partial -> `d0g2.py pool $D0G2/restore/restore_list.csv <n_par> 8 <i> <n>`
(the pool skips every reaction whose final folder has a marker, so only these are computed).
Writes $D0G2/restore/{restore_list.csv, scan.json}."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

D0G2 = Path(os.environ.get("D0G2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d0_g2"))
REPO = Path(__file__).resolve().parents[2]
Z_STATUS = REPO / "analysis" / "geo" / "results" / "Z_status.csv"
NEED = ("ts.xyz", "rel1.xyz", "rel2.xyz", "result.json", "ts.hess")
CUTOFF = time.mktime(time.strptime("2026-10-30 11:00:00", "%Y-%m-%d %H:%M:%S"))   # before the cleanup started


def touched_after_cutoff(f):
    for dp, dn, _ in os.walk(f):
        if os.stat(dp).st_mtime >= CUTOFF:
            return True
    return False


def scan():
    labs = json.loads((D0G2 / "G" / "labels_d0_g2.json").read_text())
    with open(Z_STATUS, newline="") as fh:
        zs = {int(r["rxn_id"]): r for r in csv.DictReader(fh)}
    out = D0G2 / "restore"
    (out / "partial").mkdir(parents=True, exist_ok=True)
    rows, cnt, mism, moved = [], {}, [], []
    for d in labs:
        rid, T = int(d["rxn_id"]), d["job_id"]
        f = D0G2 / "g2_final" / str(rid)
        if not f.exists():
            st = "deleted"
        else:
            done = (f / ".done").is_file()
            fails = sorted(p.name for p in f.glob(".fail_*"))
            if touched_after_cutoff(f):
                st = "partial"
            elif done and all((f / n).is_file() for n in NEED) and (f / "work").is_dir():
                st = "kept_done"
            elif fails and not done:
                st = "kept_fail"
            else:
                st = "partial"
        z = zs.get(rid, {}).get("status", "")
        if (st == "kept_done" and z != "ok") or (st == "kept_fail" and not z.startswith("fail")):
            mism.append(dict(rxn_id=rid, folder=st, z_status=z))
        if st == "partial":
            dest = out / "partial" / str(rid)
            if dest.exists():
                sys.exit(f"STOP: {dest} exists")
            shutil.move(str(f), str(dest))
            moved.append(rid)
        cnt[st] = cnt.get(st, 0) + 1
        if st in ("deleted", "partial"):
            rows.append(dict(job_id=T, rxn_id=rid, z_status=z, z_core_h=zs.get(rid, {}).get("core_h", "")))
    tmp = out / f"restore_list.csv.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["job_id", "rxn_id", "z_status", "z_core_h"])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out / "restore_list.csv")
    core_h = sum(float(r["z_core_h"]) for r in rows if r["z_core_h"] not in ("", None))
    rep = dict(n_targets=len(labs), by_state=cnt, moved_partial=moved, n_restore=len(rows),
               restore_z_status=dict((s, sum(r["z_status"] == s for r in rows)) for s in sorted({r["z_status"] for r in rows})),
               restore_core_h_at_geo1=core_h, mismatch_with_Z_status=mism[:50], n_mismatch=len(mism),
               cutoff="2026-10-30 11:00:00", written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = out / f"scan.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(rep, indent=1))
    os.replace(tmp, out / "scan.json")
    print(json.dumps(rep, indent=1), flush=True)
    return 3 if mism else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["scan"]:
        sys.exit(scan())
    raise SystemExit(__doc__)
