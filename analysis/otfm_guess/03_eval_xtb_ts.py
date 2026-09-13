#!/usr/bin/env python3
"""SPEC18 Step 3 — evaluate xTB TS guess quality vs Coley DFT-TS.

Compares three geometries per rxn (only where all three exist):
  - xTB path TS (this stage's output)
  - React-OT (RP-mode) generated TS (from spec17rev2 Step 7)
  - Coley DFT reference TS (from reactant_complex/*.npz)

Reports Kabsch RMSD, forming-bond distance errors (dd1, dd2) with
mean bias / |Δ| median / std. GATE-3 is a BRANCHING gate — the
verdict decides whether the GUESS-mode training is worth attempting.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"
PROF = PREV / "coley_profiles" / "full_dataset_profiles"


def read_xyz(path: Path):
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    a = A - A.mean(0)
    b = B - B.mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((a @ R.T - b) ** 2).sum(1).mean()))


def formed_bonds(rid: int):
    """Filename-parsed forming-bond indices from Coley's TS filename."""
    d = PROF / str(rid)
    ts = [f for f in d.iterdir()
          if f.name.startswith("TS_") and f.name != "TS_imag_mode.xyz"]
    if not ts:
        return None
    m = re.search(r"_(\d+)-(\d+)_(\d+)-(\d+)\.xyz$", ts[0].name)
    if not m:
        return None
    return (
        (int(m.group(1)), int(m.group(2))),
        (int(m.group(3)), int(m.group(4))),
    )


def main() -> int:
    mode = os.environ.get("XTB_MODE", "pilot")
    res_csv = BASE / "artifacts" / f"xtb_path_{mode}.csv"
    if not res_csv.exists():
        print(f"missing {res_csv} — run 02b_merge_shards first", file=sys.stderr)
        return 1
    res = pd.read_csv(res_csv)
    ok_ids = res[res.status == "ok"].rxn_id.tolist()
    gq = pd.read_csv(PREV / "artifacts" / "generation_quality.csv").set_index("rxn_id")

    rows = []
    for rid in ok_ids:
        npz_path = PREV / "data" / "reactant_complex" / f"rxn_{rid:04d}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        TS = npz["TS"]
        xtb_path = BASE / "xtb_runs" / f"rxn_{rid:04d}" / "ts_xtb.npy"
        if not xtb_path.exists():
            continue
        xtb = np.load(xtb_path)
        fb = formed_bonds(rid)
        if fb is None:
            continue
        (a1, a2), (b1, b2) = fb
        r = dict(
            rxn_id=rid,
            rmsd_xtb=kabsch_rmsd(xtb, TS),
            dd1_xtb=float(np.linalg.norm(xtb[a1] - xtb[a2])
                          - np.linalg.norm(TS[a1] - TS[a2])),
            dd2_xtb=float(np.linalg.norm(xtb[b1] - xtb[b2])
                          - np.linalg.norm(TS[b1] - TS[b2])),
        )
        if rid in gq.index:
            for col in ("rmsd", "dd1", "dd2"):
                if col in gq.columns:
                    r[f"{col}_otfm_rp"] = gq.at[rid, col]
        rows.append(r)

    E = pd.DataFrame(rows)
    out = BASE / "artifacts" / f"xtb_ts_quality_{mode}.csv"
    E.to_csv(out, index=False)
    print(f"평가된 rxn: {len(E)}  ->  {out}")
    if len(E) == 0:
        return 1

    print(f"\n{'':>14}{'xTB path':>12}{'OTFM(RP)':>12}")
    med_x = E.rmsd_xtb.median()
    p95_x = float(np.percentile(E.rmsd_xtb, 95))
    med_o = E.get("rmsd_otfm_rp", pd.Series(dtype=float)).median()
    p95_o = (float(np.percentile(E.rmsd_otfm_rp, 95))
             if "rmsd_otfm_rp" in E and E.rmsd_otfm_rp.notna().any() else float("nan"))
    print(f"{'RMSD 중앙':>14}{med_x:>12.3f}{med_o:>12.3f}")
    print(f"{'RMSD p95':>14}{p95_x:>12.3f}{p95_o:>12.3f}")

    for c in ("dd1", "dd2"):
        x = E[f"{c}_xtb"]
        o = E.get(f"{c}_otfm_rp", pd.Series(dtype=float)).dropna()
        print(f"{c+' |Δ| 중앙':>14}{x.abs().median():>12.3f}"
              f"{(o.abs().median() if len(o) else float('nan')):>12.3f}")
        print(f"{c+' 편향':>14}{x.mean():>12.3f}"
              f"{(o.mean() if len(o) else float('nan')):>12.3f}")
        print(f"{c+' std':>14}{x.std():>12.3f}"
              f"{(o.std() if len(o) else float('nan')):>12.3f}")

    dd_xtb = pd.concat([E.dd1_xtb, E.dd2_xtb]).abs().median()
    dd_rp_series = pd.concat([
        E.get("dd1_otfm_rp", pd.Series(dtype=float)),
        E.get("dd2_otfm_rp", pd.Series(dtype=float)),
    ]).dropna()
    dd_rp = dd_rp_series.abs().median() if len(dd_rp_series) else float("nan")

    if not np.isfinite(dd_rp):
        verdict = "no OTFM(RP) comparison available"
    elif dd_xtb < dd_rp:
        verdict = "xTB alone better than OTFM-RP"
    else:
        verdict = "xTB worse — GUESS mode may not help"
    (BASE / "artifacts" / "GATE3_STATUS.txt").write_text(
        f"dd_xtb_median={dd_xtb:.4f}\n"
        f"dd_otfm_rp_median={dd_rp:.4f}\n"
        f"{verdict}\n"
    )
    print(f"\n{verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
