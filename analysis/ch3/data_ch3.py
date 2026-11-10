#!/usr/bin/env python3
"""data_ch3.py — CH3-MODEL 5: Ch3Data(DeployData) and the target-mode switch (docs/specs/CH3_MODEL.md 0-2, 5).

Inputs = the regenerated deploy inputs (spec 3): this module sets DEP_SCRATCH=$CH3/lf and
DEP_RESULTS=analysis/ch3/results/lf_inputs before dep_common is imported (another value already in the environment =
STOP), so data_lf.DeployData('lf') reads $CH3/lf/fu2_lf/tokens, $CH3/lf/X182_lf.parquet and the lf_inputs/fu2_lf reports.

Ch3Data = DeployData('lf') (same token arrays, usable pairs, split lists) with y = y10 @ M (strain, inter, solv,
barrier; M = ch3_common.M(), 10 x 4); the 10 deploy targets stay in .y10.

use_targets(targets): the thin wrapper of spec 5 (a). fu2/train.py fit / loss_fn / predict read
fu2_common.TARGETS / I_BAR / I_CH9 / I_PRIM as module constants at call time; this sets them to the deploy values (10
targets: I_BAR 9, I_CH9 0..8, I_PRIM the 9 primary targets) or the ch3 values (TARGETS3, I_BAR 3, I_CH9 [0, 1, 2],
I_PRIM [0, 1, 2, 3]). data_lf / dep_common are imported first, so their TARGETS assertions pass on the original
constants; no deploy / fu2 file is changed.

  python data_ch3.py summary     usable pairs / folds / sha256 of Ch3Data (CPU, env_d1fu2)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402


def _bind_env():
    want = dict(DEP_SCRATCH=C3.LF, DEP_RESULTS=C3.LF_RES)
    for k, v in want.items():
        cur = os.environ.get(k)
        if cur is None:
            os.environ[k] = str(v)
        elif Path(cur).resolve() != Path(v).resolve():
            C3.stop(f"{k}={cur} in the environment, CH3 needs {v} (spec 3)")
    if "dep_common" in sys.modules:
        dc = sys.modules["dep_common"]
        if Path(dc.DEP).resolve() != C3.LF.resolve() or Path(dc.RES).resolve() != C3.LF_RES.resolve():
            C3.stop(f"dep_common imported before data_ch3 with DEP {dc.DEP} / RES {dc.RES}")


_bind_env()
sys.path.insert(0, str(C3.DEPLOY_DIR))
import dep_common as DC  # noqa: E402
import data_lf as DL  # noqa: E402

FT, Mo, F = DL.FT, DL.Mo, DL.F
if Path(DC.DEP).resolve() != C3.LF.resolve() or Path(DC.RES).resolve() != C3.LF_RES.resolve():
    C3.stop(f"dep_common paths DEP {DC.DEP} / RES {DC.RES} are not the CH3 regeneration paths")

MODE10 = dict(TARGETS=list(C3.TARGETS10), I_BAR=9, I_CH9=list(range(9)), I_PRIM=[0, 1, 2, 3, 4, 5, 6, 7, 9])
MODE3 = dict(TARGETS=list(C3.TARGETS3), I_BAR=C3.I_BAR3, I_CH9=list(C3.I_CH9_3), I_PRIM=list(C3.I_PRIM3))
ORIG = dict(TARGETS=list(F.TARGETS), I_BAR=F.I_BAR, I_CH9=list(F.I_CH9), I_PRIM=list(F.I_PRIM))
if ORIG != MODE10 or list(DL.TARGETS) != list(C3.TARGETS10):
    C3.stop(f"fu2_common constants at import {ORIG} != the deploy 10-target values {MODE10}")


def use_targets(targets):
    """set the fu2_common target constants for the next fits (spec 5 (a)); returns the mode dict."""
    targets = list(targets)
    if targets == list(C3.TARGETS10):
        m = MODE10
    elif targets == list(C3.TARGETS3):
        m = MODE3
    else:
        C3.stop(f"unknown target list {targets}")
    F.TARGETS = list(m["TARGETS"])
    F.I_BAR = int(m["I_BAR"])
    F.I_CH9 = list(m["I_CH9"])
    F.I_PRIM = list(m["I_PRIM"])
    return current_mode()


def current_mode():
    return dict(TARGETS=list(F.TARGETS), I_BAR=int(F.I_BAR), I_CH9=list(F.I_CH9), I_PRIM=list(F.I_PRIM))


class Ch3Data(DL.DeployData):
    """DeployData('lf') with y = y10 @ M (spec 5); rung 'lf3' (unit names D__lf3__S__TF__*)."""

    def __init__(self, labels=DL.DEFAULT_LABELS):
        super().__init__("lf", labels)
        self.y10 = self.y
        self.M = C3.M()
        self.y = self.y10 @ self.M
        self.rung = C3.RUNG3
        self.targets = list(C3.TARGETS3)
        self.sha = dict(self.sha, conversion_M=C3.sha256_text(json.dumps(C3.M_ROWS)))


_DATA: dict = {}


def get_data(kind="ch3"):
    """'ch3' -> Ch3Data, 'lf' -> DeployData('lf') (C-3′ / C-D), cached per process."""
    if kind not in _DATA:
        if kind == "ch3":
            _DATA[kind] = Ch3Data()
        elif kind == "lf":
            _DATA[kind] = DL.DeployData("lf")
        else:
            C3.stop(f"unknown data kind {kind}")
    return _DATA[kind]


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["summary"]:
        D = get_data("ch3")
        import numpy as np
        fin = {t: int(np.isfinite(D.y[:, j]).sum()) for j, t in enumerate(C3.TARGETS3)}
        out = dict(n_tokens=int(len(D.rids)), n_usable=int(D.usable_rxn.sum()), n_pairs_P=len(D.pair_list),
                   pair_list_sha=D.pair_list_sha, finite_y=fin,
                   folds={k: {key: int(len(v)) for key, v in D.split(k).items() if key in DL.FOLD_KEYS}
                          for k in range(DL.NF)}, all=int(len(D.split("all")["train"])), sha=D.sha)
        print(json.dumps(out, indent=1))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
