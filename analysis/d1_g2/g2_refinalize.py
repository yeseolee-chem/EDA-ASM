#!/usr/bin/env python3
"""D1-G2 Phase G repair (docs/specs/D1G2.md 2, 6): reactions whose final folder is marked .fail_error_* because of a
wrapper exception in g2_autode.finalize. Run through sbatch only (jobs/g2.slurm, G2ENV=d1), AFTER every Phase G pool
job has ended (g2_autode.py is not edited while the pools run: each reaction subprocess re-reads it).

  python g2_refinalize.py run <cores>

Bug (found in the first 640 reactions, 2026-10-22): g2_autode._graph used fr.bond_matrix(...) as the adjacency, but
it returns (adj, D), so every reaction whose G2 TS atom order differs from the label (the remap path) ended as
.fail_error_TypeError. Here g2_autode._graph is replaced in this process by the corrected function (B, _ =
fr.bond_matrix(...)); nothing else changes. finalize runs again from the stage outputs kept in $G2/jobs/<T>/ (ts.xyz,
ts_result.json, stereo / ref results, ref_*.xyz, .done_<stage>): autodE, stereo and ref are NOT rerun (spec 2: one run
per reaction); the XTB2 ALPB(water) Freq at the final TS is recomputed as in finalize. The old final folder moves to
$G2/g2_final_prev/<rxn_id>.<time> (never deleted); result.json gets refinalized_from.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_autode as GA  # noqa: E402                  (its import-time setup: D1 code paths, autodE patches)

G, pc = GA.G, GA.pc


def _graph_fixed(fr, syms, xyz, extra_edges=(), tags=None):
    """= g2_autode._graph with the bond matrix unpacked."""
    import networkx as nx
    B, _ = fr.bond_matrix(syms, np.asarray(xyz, dtype=float))
    g = nx.Graph()
    for i, s in enumerate(syms):
        g.add_node(i, lab=(s, None if tags is None else tags[i]))
    n = len(syms)
    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j]:
                g.add_edge(i, j, f=False)
    for i, j in extra_edges:
        g.add_edge(i, j, f=True)
    return g


GA._graph = _graph_fixed                                  # remap() looks _graph up at call time


def cmd_run(cores):
    cores = int(cores)
    lab = {int(d["rxn_id"]): d for d in G.targets()}
    todo = sorted(int(p.parent.name) for p in G.G2_FINAL.glob("*/.fail_error_*") if p.parent.name.isdigit())
    print(f"[refinalize] {len(todo)} folders with .fail_error_*", flush=True)
    os.environ["SLURM_CPUS_PER_TASK"] = os.environ["OMP_NUM_THREADS"] = str(cores)
    out = {}
    for rid in todo:
        L = lab[rid]
        T, kept, old = L["job_id"], G.G2_JOBS / L["job_id"], G.G2_FINAL / str(rid)
        if not all((kept / f).is_file() for f in (".done_ts", ".done_stereo", ".done_ref", "ts.xyz", "ts_result.json",
                                                  "ref_dipole.xyz", "ref_dipolarophile.xyz")):
            out[T] = "kept_stage_outputs_incomplete"
            print(f"  {T} ({rid}): {out[T]}", flush=True)
            continue
        prev = json.loads((old / "result.json").read_text()) if (old / "result.json").is_file() else {}
        local = Path(os.environ.get("TMPDIR", "/tmp")) / f"d1g2_re_{os.getpid()}_{T}"
        cfg = yaml.safe_load(GA.CONFIG.read_text())
        cfg["scratch"] = str(local)
        job = pc.load_job(T, G.MANIFEST)
        jd = pc.job_dir(cfg, T)
        try:
            jd.mkdir(parents=True, exist_ok=True)
            for p in kept.iterdir():
                if p.is_file() and p.name != "inputs.tar.gz":
                    shutil.copy(p, jd / p.name)
            dst = G.assert_path(G.G2 / "g2_final_prev" / f"{rid}.{time.strftime('%Y%m%d_%H%M%S')}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, dst)
            times = {k: v for k, v in (prev.get("stage_wall_s") or {}).items() if k != "freq"}
            os.chdir(local)
            st = GA.finalize(cfg, job, jd, L, cores, G.G2_FINAL, times)
            f = G.G2_FINAL / str(rid) / "result.json"
            r = json.loads(f.read_text())
            r["refinalized_from"] = dict(prev_folder=str(dst), prev_fail_reasons=prev.get("fail_reasons"),
                                         note="finalize rerun from $G2/jobs/<T> with the fixed _graph (2026-10-22); "
                                              "autodE / stereo / ref not rerun")
            tmp = f.with_name("result.json.tmp")
            tmp.write_text(json.dumps(r, indent=1, default=pc._jsonable))
            os.replace(tmp, f)
            out[T] = st
        except Exception as e:                                   # noqa: BLE001
            out[T] = f"refinalize_error {type(e).__name__}: {e}"
        finally:
            os.chdir("/")
            shutil.rmtree(local, ignore_errors=True)
        print(f"  {T} ({rid}): {out[T]}", flush=True)
    vc = pd.Series(list(out.values()), dtype=object).value_counts().to_dict()
    G.write_json(G.RES / "G_refinalize.json", dict(n=len(todo), result=out, counts=vc, written=time.strftime("%F %T")))
    print(f"[refinalize] {vc}", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"] and len(sys.argv) == 3:
        sys.exit(cmd_run(sys.argv[2]))
    raise SystemExit(__doc__)
