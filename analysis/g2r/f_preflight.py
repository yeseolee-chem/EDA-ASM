#!/usr/bin/env python3
"""D1-G2R 8-1 start assertion / F-2 record: run by jobs/g_g2r.slurm before every F / T command (venv5).

    python ../g2r/f_preflight.py <tag>      -> analysis/g2r/results/F_tol/<tag>.json; exit 0 ok, 3 STOP

Reads the values in three interpreters: this job process, a multiprocessing spawn worker (as ext_features.run_pool /
d1_features.run_ids / fu2 atoms use) and a subprocess with the feat_lf child settings (ESPLEY_PARTITION=redetect,
D1PAIR_PHASE=FROOT; feat_lf.lf_env copies os.environ). Each must have: sitecustomize = analysis/g2r/site_g2r, the Si
entries (D1PAIR_SI_PATCH=1, fragmenter COV 1.11 / Z 14, rev5_common BONDI 2.10 / GFN2_VALENCE 4), GATE_E_EH = 1e-4,
TOL_DIP_AU = 1e-2, B1_TOL_OVERRIDE empty and every other tolerance at its ext_features / rev5_common value (spec 0-3).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE / "site_g2r" / "sitecustomize.py"
WANT = dict(GATE_E_EH=1e-4, TOL_DIP_AU=1e-2, SCAN_GATE_EH=1e-6, GATE_S_BLOCK=1e-10, GATE_ORTHO=1e-8, TOL_DIST_A=1e-9,
            TOL_GEDT=1e-8, TOL_TRACE=2e-3, TOL_Q_JSON=5e-4, TOL_EPS_EV=5e-3, TOL_FUKUI=2e-3, TOL_HESS=1e-6,
            TOL_HESS_GEOM_A=1e-4, B1_TOL_OVERRIDE={}, si=["1", 1.11, 14, 2.10, 4], G2R_TOL_PATCH="1")


def probe():
    import ext_features as ef
    import fragmenter as fr
    import rev5_common as r5
    sc = sys.modules.get("sitecustomize")
    return dict(pid=os.getpid(), executable=sys.executable, sitecustomize=getattr(sc, "__file__", None),
                G2R_TOL_PATCH=os.environ.get("G2R_TOL_PATCH"), ESPLEY_PARTITION=os.environ.get("ESPLEY_PARTITION"),
                GATE_E_EH=r5.GATE_E_EH, TOL_DIP_AU=ef.TOL_DIP_AU, SCAN_GATE_EH=r5.SCAN_GATE_EH,
                GATE_S_BLOCK=r5.GATE_S_BLOCK, GATE_ORTHO=r5.GATE_ORTHO, TOL_DIST_A=ef.TOL_DIST_A, TOL_GEDT=ef.TOL_GEDT,
                TOL_TRACE=ef.TOL_TRACE, TOL_Q_JSON=ef.TOL_Q_JSON, TOL_EPS_EV=ef.TOL_EPS_EV, TOL_FUKUI=ef.TOL_FUKUI,
                TOL_HESS=ef.TOL_HESS, TOL_HESS_GEOM_A=ef.TOL_HESS_GEOM_A, B1_TOL_OVERRIDE=dict(ef.B1_TOL_OVERRIDE),
                si=[os.environ.get("D1PAIR_SI_PATCH"), fr.COV.get("Si"), fr.Z.get("Si"), r5.BONDI.get("Si"),
                    r5.GFN2_VALENCE.get("Si")],
                ext_features_file=ef.__file__, rev5_common_file=r5.__file__)


def check(p):
    bad = [k for k, v in WANT.items() if p.get(k) != v]
    sc = p.get("sitecustomize")
    if not sc or Path(sc).resolve() != SITE.resolve():
        bad.append("sitecustomize")
    return bad


def main(tag):
    import multiprocessing as mp
    res = {"job": probe()}
    with mp.get_context("spawn").Pool(1) as pool:
        res["spawn_worker"] = pool.apply(probe)
    env = dict(os.environ, ESPLEY_PARTITION="redetect", D1PAIR_PHASE="FROOT")
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_probe"], env=env, capture_output=True,
                       text=True)
    try:
        res["child_subprocess"] = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:                                                        # noqa: BLE001
        res["child_subprocess"] = dict(error=f"{type(e).__name__}: {e}", stdout=r.stdout[-2000:], stderr=r.stderr[-2000:])
    bad = {k: check(v) for k, v in res.items()}
    ok = not any(bad.values())
    rec = dict(spec="docs/specs/D1G2R.md 8-1 / F-2", tag=tag, written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
               pythonpath=os.environ.get("PYTHONPATH"), want=WANT, tolerances=res["job"], probes=res, mismatches=bad,
               ok=ok)
    sys.path.insert(0, str(HERE))
    import g2r_common as C
    C.write_json(C.RES / "F_tol" / f"{tag}.json", rec)
    print(f"[f_preflight] {tag}: ok={ok} GATE_E_EH={res['job'].get('GATE_E_EH')} TOL_DIP_AU={res['job'].get('TOL_DIP_AU')} "
          f"si={res['job'].get('si')} mismatches={bad}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    if sys.argv[1:] == ["_probe"]:
        print(json.dumps(probe()))
        sys.exit(0)
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
