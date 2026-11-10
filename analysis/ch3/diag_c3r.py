#!/usr/bin/env python3
"""diag_c3r.py — diagnostic for the C-3′ STOP of 11-10 (not a spec phase; nothing here is a CH3 result).

  python diag_c3r.py null     fold 0, 10-target mode, the C-3′ rules with every seed shifted by +1000 (tuning 20262019,
                              finals 20262019 + 0..4; = deploy null_gc.py rule) on the regenerated lf inputs
                              -> $CH3/diag/c3r_null/, compared like C-3′ with the deploy K fold 0 and with the C-3′
                              run -> $CH3/diag/c3r_null.json: the seed-only spread of the C-3′ quantities
                              (chosen cell, best epoch, fold-0 test para MAE per target).
GPU (env_d1fu2, RTX 3090), jobs/gpu_ch3.slurm. Writes only $CH3/diag.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402
import train_ch3 as TC  # noqa: E402

DC3, FT, F, TT = TC.DC3, TC.FT, TC.F, TC.TT
DIAG = C3.CH3 / "diag"
SHIFT = 1000


def rel(a, b):
    return {t: (a[t] - b[t]) / b[t] for t in C3.TARGETS10}


def cmd_null():
    FT.require_gpu()
    D = DC3.get_data("lf")
    DC3.use_targets(C3.TARGETS10)
    seed0 = F.SEED
    F.SEED = seed0 + SHIFT                       # run_unit reads F.SEED at call time (tuning fits, gate seeds, .pt names)
    seeds = [F.SEED + i for i in range(5)]
    u = dict(rung="lf", arm="S", model="TF", fold=0)
    root = DIAG / "c3r_null"
    rec = TC.run_unit(D, u, C3.TARGETS10, root=root, seeds=seeds)
    F.SEED = seed0
    name = TC.uname(u)
    null = pd.read_parquet(root / f"{name}.parquet")
    c3r = pd.read_parquet(C3.C3R / f"{name}.parquet")
    K = pd.read_parquet(TC.DEP_K_PRED)
    dep = K[K.fold == 0]
    m_null, n = TT.para_mae(null, D.P)
    m_c3r, _ = TT.para_mae(c3r, D.P)
    m_dep, _ = TT.para_mae(dep, D.P)
    kr = C3.read_json(TC.DEP_K_REPORT)["units"]["D__lf__S__TF__f0"]
    c3 = C3.read_json(C3.RES / "C3r_regress.json")
    out = dict(rule=f"seeds + {SHIFT} (tuning {seed0 + SHIFT}, finals {seeds}), same inputs / code as C-3′",
               gpu=FT.versions()["gpu"], n_test_para=n,
               chosen=dict(null=rec["chosen"], c3r=c3.get("chosen"), deploy=kr["chosen"]),
               tune=dict(null=[dict(cell=t["cell"], val=t["val"], best_epoch=t["best_epoch"]) for t in rec["tune"]],
                         c3r=c3.get("tune"), deploy=kr.get("tune")),
               para_mae=dict(null=m_null, c3r=m_c3r, deploy=m_dep),
               rel=dict(null_vs_deploy=rel(m_null, m_dep), c3r_vs_deploy=rel(m_c3r, m_dep),
                        null_vs_c3r=rel(m_null, m_c3r)),
               n_targets_within_2pct=dict(null_vs_deploy=int(sum(abs(v) <= 0.02 for v in rel(m_null, m_dep).values())),
                                          c3r_vs_deploy=int(sum(abs(v) <= 0.02 for v in rel(m_c3r, m_dep).values())),
                                          null_vs_c3r=int(sum(abs(v) <= 0.02 for v in rel(m_null, m_c3r).values()))),
               best_epoch_diff=dict(null_vs_deploy=int(rec["chosen"]["best_epoch"]) - int(kr["chosen"]["best_epoch"]),
                                    null_vs_c3r=int(rec["chosen"]["best_epoch"]) - int(c3["chosen"]["best_epoch"])),
               wall_seconds=rec["wall_seconds"], written=C3.now())
    C3.write_json(DIAG / "c3r_null.json", out)
    print(json.dumps({k: out[k] for k in ("chosen", "n_targets_within_2pct", "best_epoch_diff")}, indent=1, default=str))
    print(json.dumps(out["rel"], indent=1))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["null"]:
        sys.exit(cmd_null())
    print(__doc__)
    sys.exit(2)
