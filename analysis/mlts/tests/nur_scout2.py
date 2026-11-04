#!/usr/bin/env python3
"""tests/nur_scout2.py — DEPLOY-V2 (MLTS) Nurion READ-ONLY SCOUT 2 (mapping, tags, retries, atom order). Sandbox only.

  small                 every D1 ledger summary tar ($D1/home01/d1_ledger/summary/*.small.tar = the top-level small
                        files of every job of the 23 result tars; not the uploaded tars): per job the top-level names,
                        retry / env-fix markers, queue attempts, run_stage attempts.tsv, ts_result.json keys, and for
                        D1 targets ts.xyz vs the label TS ($LAD/dft_final/<rxn>/ts.xyz: element order + coordinates).
  order <tar> [N]       the first N jobs (default all) of one result tar (opened read only, random access): archive.tgz
                        and its nested ts.tar in memory; every label-level ORCA file of the TS tree written to a
                        node-local $TMPDIR dir (deleted) and parsed: element order and coordinates of every step vs the
                        job's ts.xyz, last OptTS step vs ts.xyz, .hess coordinates vs ts.xyz.
Prints one JSON object; writes nothing on GPFS."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import mlt_common as C  # noqa: E402
import orca_io as O  # noqa: E402

LEDGER = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_ledger/summary")
TOL = 1e-6


def parse_xyz_text(t):
    ln = t.splitlines()
    n = int(ln[0].split()[0])
    sym, pos = [], []
    for s in ln[2:2 + n]:
        q = s.split()
        sym.append(q[0])
        pos.append([float(q[1]), float(q[2]), float(q[3])])
    return sym, np.array(pos)


def target_jobs():
    T = C.targets()
    return {j: int(r) for j, r in zip(T.job_id, T.rxn_id)}


def cmd_small():
    tj = target_jobs()
    out = dict(tars=0, jobs=0, top_names=Counter(), markers=Counter(), marker_jobs=defaultdict(list),
               queue_attempt_starts=Counter(), attempts_tsv_rows=Counter(), attempts_tsv_head=None,
               ts_result_keys=None, ts_result_sample={}, label_cmp=Counter(), label_cmp_bad=[], k_dip=Counter(),
               logs=0, attempts_files=0, non_job_members=Counter())
    for tp in sorted(LEDGER.glob("*.small.tar")):
        out["tars"] += 1
        jobs = defaultdict(dict)
        with tarfile.open(tp, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                p = m.name
                q = p.split("/")
                if q[0] == "jobs" and len(q) == 3:
                    jobs[q[1]][q[2]] = tf.extractfile(m).read()
                elif q[0] == "attempts":
                    out["attempts_files"] += 1
                    txt = tf.extractfile(m).read().decode(errors="replace")
                    out["queue_attempt_starts"][sum(1 for x in txt.splitlines() if x.startswith("start"))] += 1
                elif q[0] == "logs":
                    out["logs"] += 1
                else:
                    out["non_job_members"]["/".join(q[:2])] += 1
        for j, files in jobs.items():
            out["jobs"] += 1
            for n in files:
                out["top_names"][re.sub(r"T\d{5}", "<J>", n)] += 1
                if n.startswith(".") and not n.startswith((".done_", ".packed", ".progress")):
                    out["markers"][n] += 1
                    if len(out["marker_jobs"][n]) < 12:
                        out["marker_jobs"][n].append(j)
            if "attempts.tsv" in files:
                rows = files["attempts.tsv"].decode(errors="replace").splitlines()
                if out["attempts_tsv_head"] is None:
                    out["attempts_tsv_head"] = rows[:4]
                n_ts = sum(1 for r in rows if "ts" in r.split("\t"))
                out["attempts_tsv_rows"][f"ts_rows={n_ts}"] += 1
            if "ts_result.json" in files:
                try:
                    tr = json.loads(files["ts_result.json"])
                    if out["ts_result_keys"] is None:
                        out["ts_result_keys"] = sorted(tr)
                        out["ts_result_sample"] = {k: v for k, v in tr.items()
                                                   if isinstance(v, (str, int, float, bool)) and len(str(v)) < 160}
                    out["k_dip"][str(tr.get("reactant_index_dipole"))] += 1
                except Exception as e:                       # noqa: BLE001
                    out["k_dip"][f"err {type(e).__name__}"] += 1
            if j in tj and "ts.xyz" in files:
                s1, p1 = parse_xyz_text(files["ts.xyz"].decode())
                lp = C.label_ts(tj[j])
                if not lp.exists():
                    out["label_cmp"]["no_label_file"] += 1
                    continue
                s2, p2 = C.read_xyz(lp)
                if s1 != s2:
                    out["label_cmp"]["element_order_differs"] += 1
                    out["label_cmp_bad"].append(j)
                elif float(np.abs(p1 - np.array(p2)).max()) > TOL:
                    out["label_cmp"]["coords_differ"] += 1
                    out["label_cmp_bad"].append(j)
                else:
                    out["label_cmp"]["identical"] += 1
            elif j in tj:
                out["label_cmp"]["target_without_ts_xyz"] += 1
    out["n_targets"] = len(tj)
    out["label_cmp_bad"] = out["label_cmp_bad"][:20]
    out["top_names"] = dict(out["top_names"].most_common(60))
    return out


def _bang(echo):
    return " ".join(" ".join(ln.split("#")[0].split()) for ln in echo.splitlines() if ln.strip().startswith("!"))


def cmd_order(tar, nmax):
    tj = target_jobs()
    res = dict(tar=str(tar), jobs={})
    tmp = Path(tempfile.mkdtemp(prefix="nurscout2_", dir=os.environ.get("TMPDIR", "/tmp")))
    try:
        with tarfile.open(tar, "r:") as tf:
            by = defaultdict(dict)
            order = []
            for m in tf:
                q = m.name.split("/")
                if q[0] == "jobs" and len(q) == 3 and m.isfile():
                    if q[1] not in by:
                        if nmax and len(order) >= nmax:
                            break
                        order.append(q[1])
                    by[q[1]][q[2]] = m
            for j in order:
                mem = by[j]
                r = dict(target=j in tj, top=sorted(mem))
                if "ts.xyz" not in mem or "archive.tgz" not in mem:
                    res["jobs"][j] = r
                    continue
                ts_sym, ts_pos = parse_xyz_text(tf.extractfile(mem["ts.xyz"]).read().decode())
                arc = tarfile.open(fileobj=io.BytesIO(tf.extractfile(mem["archive.tgz"]).read()), mode="r:gz")
                names = [m.name for m in arc.getmembers() if m.isfile()]
                r["archive_top"] = dict(Counter(n.split("/")[0] for n in names))
                inner = []
                if "ts.tar" in names:
                    tt = tarfile.open(fileobj=io.BytesIO(arc.extractfile("ts.tar").read()), mode="r:")
                    inner = [(m, tt) for m in tt.getmembers() if m.isfile()]
                r["ts_layout"] = "ts.tar" if inner else ("ts/" if any(n.startswith("ts/") for n in names) else "none")
                inner += [(m, arc) for m in arc.getmembers() if m.isfile() and m.name.startswith(("ref/", "ts/"))
                          and not (inner and m.name.startswith("ts/"))]
                jd = tmp / j
                files = []
                for m, t in inner:
                    n = m.name
                    if not (n.endswith("_orca.out") or n.endswith(".hess") or n.endswith("_trj.xyz")):
                        continue
                    dst = jd / n.replace("/", "__")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(t.extractfile(m).read())
                    files.append((n, dst))
                rows = []
                for n, dst in files:
                    row = dict(f=re.sub(r"T\d{5}", "<J>", n)[-95:])
                    if n.endswith("_orca.out"):
                        echo = O.input_echo(dst)
                        ok, why = O.label_level(echo)
                        row.update(bang=_bang(echo)[:90], label=ok)
                        if not ok:
                            rows.append(row)
                            continue
                        steps, info = O.read_out_steps(dst)
                        row.update(n_steps=info["n_steps"], n_cyc=info["n_opt_cycles"], term=info["terminated_normally"])
                        if steps:
                            same_el = steps[0]["symbols"] == ts_sym
                            row["same_elements_as_ts"] = same_el
                            if same_el:
                                d = [float(np.abs(s["pos_A"] - ts_pos).max()) for s in steps]
                                row["last_vs_ts_A"] = d[-1]
                                row["min_vs_ts_A"] = [min(d), int(np.argmin(d)) + 1]
                                X = steps[-1]["pos_A"] - steps[-1]["pos_A"].mean(0)
                                Y = ts_pos - ts_pos.mean(0)
                                row["last_vs_ts_centered_A"] = float(np.abs(X - Y).max())
                                U, _, Vt = np.linalg.svd(X.T @ Y)
                                D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
                                row["last_vs_ts_kabsch_A"] = float(np.abs(X @ U @ D @ Vt - Y).max())
                                trj = dst.with_name(dst.name[:-4] + "_trj.xyz")
                                row["trj_present"] = trj.exists()
                    elif n.endswith(".hess"):
                        try:
                            h = O.read_hess(dst)
                            same_el = h["symbols"] == ts_sym
                            row["same_elements_as_ts"] = same_el
                            if same_el:
                                row["hess_vs_ts_A"] = float(np.abs(h["pos_bohr"] * C.BOHR2A - ts_pos).max())
                        except Exception as e:               # noqa: BLE001
                            row["hess_err"] = f"{type(e).__name__}: {e}"[:80]
                    else:
                        continue
                    rows.append(row)
                r["files"] = rows
                if j in tj:
                    s2, p2 = C.read_xyz(C.label_ts(tj[j]))
                    r["ts_xyz_eq_label"] = bool(s2 == ts_sym and np.abs(np.array(p2) - ts_pos).max() <= TOL)
                res["jobs"][j] = r
                shutil.rmtree(jd, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "small":
        out = cmd_small()
    elif a[0] == "order":
        out = cmd_order(Path(a[1]), int(a[2]) if len(a) > 2 else 0)
    else:
        raise SystemExit(__doc__)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
