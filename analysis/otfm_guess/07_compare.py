#!/usr/bin/env python3
"""SPEC18 Step 7 — three-way comparison: xTB path / OTFM-RP / OTFM-GUESS.

For every rxn where all three geometries exist, report:
  - Kabsch RMSD (vs Coley DFT-TS)
  - forming-bond distance error dd1, dd2 with mean bias / |Δ| median / std
  - percentile summary of |Δ| distribution
"""
from __future__ import annotations

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
        syms.append(t[0]); xyz.append([float(v) for v in t[1:4]])
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
    # xTB TS positions (from Step 2 npy) — use only rxns that succeeded
    xtb_res = pd.read_csv(BASE / "artifacts" / "xtb_path_full.csv").set_index("rxn_id")
    ok_xtb = set(xtb_res[xtb_res.status == "ok"].index)

    # OTFM RP-mode from spec17rev2 Step 9 output
    gq_rp = pd.read_csv(PREV / "artifacts" / "generation_quality.csv").set_index("rxn_id")

    # OTFM GUESS-mode generated xyz (Step 6 of this stage)
    guess_dir = BASE / "generated" / "guess"

    rows = []
    for rid in sorted(gq_rp.index):
        # need DFT reference
        npz_path = PREV / "data" / "reactant_complex" / f"rxn_{rid:04d}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        TS = npz["TS"]

        # need forming bond indices
        fb = formed_bonds(rid)
        if fb is None:
            continue
        (a1, a2), (b1, b2) = fb
        ref_d1 = float(np.linalg.norm(TS[a1] - TS[a2]))
        ref_d2 = float(np.linalg.norm(TS[b1] - TS[b2]))

        row = dict(rxn_id=rid)

        # xTB
        if rid in ok_xtb:
            xtb_npy = BASE / "xtb_runs" / f"rxn_{rid:04d}" / "ts_xtb.npy"
            if xtb_npy.exists():
                xtb = np.load(xtb_npy)
                row["rmsd_xtb"] = kabsch_rmsd(xtb, TS)
                row["dd1_xtb"] = float(np.linalg.norm(xtb[a1] - xtb[a2])) - ref_d1
                row["dd2_xtb"] = float(np.linalg.norm(xtb[b1] - xtb[b2])) - ref_d2

        # OTFM RP
        if "rmsd" in gq_rp.columns and rid in gq_rp.index:
            row["rmsd_otfm_rp"] = gq_rp.at[rid, "rmsd"]
            if "dd1" in gq_rp.columns:
                row["dd1_otfm_rp"] = gq_rp.at[rid, "dd1"]
            if "dd2" in gq_rp.columns:
                row["dd2_otfm_rp"] = gq_rp.at[rid, "dd2"]

        # OTFM GUESS
        guess_xyz = guess_dir / f"rxn_{rid:04d}.xyz"
        if guess_xyz.exists():
            _, g_pos = read_xyz(guess_xyz)
            row["rmsd_otfm_guess"] = kabsch_rmsd(g_pos, TS)
            row["dd1_otfm_guess"] = float(np.linalg.norm(g_pos[a1] - g_pos[a2])) - ref_d1
            row["dd2_otfm_guess"] = float(np.linalg.norm(g_pos[b1] - g_pos[b2])) - ref_d2

        if any(k.startswith("rmsd_") for k in row):
            rows.append(row)

    C = pd.DataFrame(rows)
    out = BASE / "artifacts" / "compare_three.csv"
    C.to_csv(out, index=False)
    print(f"평가 대상 rxn: {len(C)}  ->  {out}")

    header = f"\n{'':>18}{'xTB path':>12}{'OTFM(RP)':>12}{'OTFM(GUESS)':>14}"
    print(header)

    def _stat(series):
        if series.dropna().empty:
            return float("nan")
        return series.dropna()

    def _fmt(x):
        return f"{x:>12.3f}" if isinstance(x, float) and np.isfinite(x) else f"{'--':>12}"

    def _fmt_g(x):
        return f"{x:>14.3f}" if isinstance(x, float) and np.isfinite(x) else f"{'--':>14}"

    for label, col in (("RMSD median", "rmsd"),):
        x = C.get(f"{col}_xtb")
        r = C.get(f"{col}_otfm_rp")
        g = C.get(f"{col}_otfm_guess")
        vx = float(x.median()) if x is not None else float("nan")
        vr = float(r.median()) if r is not None else float("nan")
        vg = float(g.median()) if g is not None else float("nan")
        print(f"{label:>18}{_fmt(vx)}{_fmt(vr)}{_fmt_g(vg)}")

    for c in ("dd1", "dd2"):
        for label, agg in (("|Δ| median", lambda s: s.abs().median()),
                          ("bias",       lambda s: s.mean()),
                          ("std",        lambda s: s.std())):
            x = C.get(f"{c}_xtb")
            r = C.get(f"{c}_otfm_rp")
            g = C.get(f"{c}_otfm_guess")
            vx = float(agg(x)) if x is not None and x.notna().any() else float("nan")
            vr = float(agg(r)) if r is not None and r.notna().any() else float("nan")
            vg = float(agg(g)) if g is not None and g.notna().any() else float("nan")
            print(f"{(c+' '+label):>18}{_fmt(vx)}{_fmt(vr)}{_fmt_g(vg)}")

    # GATE-7 verdicts
    dd_rp = pd.concat([C.get("dd1_otfm_rp", pd.Series(dtype=float)),
                       C.get("dd2_otfm_rp", pd.Series(dtype=float))]).dropna()
    dd_g = pd.concat([C.get("dd1_otfm_guess", pd.Series(dtype=float)),
                      C.get("dd2_otfm_guess", pd.Series(dtype=float))]).dropna()
    dd_x = pd.concat([C.get("dd1_xtb", pd.Series(dtype=float)),
                      C.get("dd2_xtb", pd.Series(dtype=float))]).dropna()

    def med(s): return float(s.abs().median()) if len(s) else float("nan")
    def std(s): return float(s.std()) if len(s) else float("nan")

    v_med = {"xtb": med(dd_x), "rp": med(dd_rp), "guess": med(dd_g)}
    v_std = {"xtb": std(dd_x), "rp": std(dd_rp), "guess": std(dd_g)}
    print(f"\n[GATE-7 summary]")
    print(f"  |Δ| median  xtb={v_med['xtb']:.4f}  rp={v_med['rp']:.4f}  guess={v_med['guess']:.4f}")
    print(f"  std         xtb={v_std['xtb']:.4f}  rp={v_std['rp']:.4f}  guess={v_std['guess']:.4f}")

    verdict = []
    if np.isfinite(v_med["guess"]) and np.isfinite(v_med["rp"]):
        verdict.append("guess<rp" if v_med["guess"] < v_med["rp"] else "guess>=rp")
    if np.isfinite(v_std["guess"]) and np.isfinite(v_std["rp"]):
        verdict.append("std_guess<rp" if v_std["guess"] < v_std["rp"] else "std_guess>=rp")
    if np.isfinite(v_med["guess"]) and np.isfinite(v_med["xtb"]):
        verdict.append("guess<xtb" if v_med["guess"] < v_med["xtb"] else "guess>=xtb")

    (BASE / "artifacts" / "GATE7_STATUS.txt").write_text(
        f"n={len(C)}\n"
        f"|delta|_median_xtb={v_med['xtb']:.4f}\n"
        f"|delta|_median_rp={v_med['rp']:.4f}\n"
        f"|delta|_median_guess={v_med['guess']:.4f}\n"
        f"std_xtb={v_std['xtb']:.4f}\n"
        f"std_rp={v_std['rp']:.4f}\n"
        f"std_guess={v_std['guess']:.4f}\n"
        f"verdict={','.join(verdict)}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
