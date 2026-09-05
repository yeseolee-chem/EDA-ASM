#!/usr/bin/env python3
"""SPEC17rev2 Step 9 — generated-TS quality vs Coley DFT reference.

Reads generated/crossfit/rxn_NNNN.xyz, compares to Coley TS_*.xyz on:
  - Kabsch RMSD (Å)
  - forming-bond distances d1, d2 (parsed from TS filename `_a-b_c-d.xyz`)
  - Δd1, Δd2 = generated − reference (Å)

GATE-8 targets (spec §9):
  |Δd_forming| median < 0.05  -> PASS
                     0.05..0.15 -> WARN (interpret in Step 10)
                    > 0.15   -> FAIL (AM1-baseline territory)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train")
PROF = BASE / "coley_profiles/full_dataset_profiles"
GEN = BASE / "generated/crossfit"

for _d in ("artifacts", "data", "ckpt", "generated", "logs", "figures"):
    (BASE / _d).mkdir(parents=True, exist_ok=True)


def read_xyz(path: Path):
    lines = path.read_text().split("\n")
    n = int(lines[0])
    syms, xyz = [], []
    for l in lines[2:2 + n]:
        t = l.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    a, b = A - A.mean(0), B - B.mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((a @ R.T - b) ** 2).sum(1).mean()))


def main() -> int:
    if not GEN.exists():
        print(f"[FATAL] {GEN} missing — run Step 7", file=sys.stderr)
        return 1

    rows = []
    gens = sorted(GEN.glob("rxn_*.xyz"))
    print(f"evaluating {len(gens)} generated TS")

    for g in gens:
        rid = int(g.stem.split("_")[1])
        d = PROF / str(rid)
        ts_files = [f for f in d.iterdir()
                    if f.name.startswith("TS_") and f.name != "TS_imag_mode.xyz"]
        if not ts_files:
            continue
        ts_ref = ts_files[0]
        m = re.search(r"_(\d+)-(\d+)_(\d+)-(\d+)\.xyz$", ts_ref.name)
        ref_s, ref_x = read_xyz(ts_ref)
        gen_s, gen_x = read_xyz(g)
        if ref_s != gen_s:
            rows.append(dict(rxn_id=rid, ok=False, reason="atom order mismatch"))
            continue
        r = dict(rxn_id=rid, ok=True, rmsd=kabsch_rmsd(ref_x, gen_x))
        if m:
            a1, a2, b1, b2 = (int(x) for x in m.groups())
            r["ref_d1"] = float(np.linalg.norm(ref_x[a1] - ref_x[a2]))
            r["ref_d2"] = float(np.linalg.norm(ref_x[b1] - ref_x[b2]))
            r["gen_d1"] = float(np.linalg.norm(gen_x[a1] - gen_x[a2]))
            r["gen_d2"] = float(np.linalg.norm(gen_x[b1] - gen_x[b2]))
            r["dd1"] = r["gen_d1"] - r["ref_d1"]
            r["dd2"] = r["gen_d2"] - r["ref_d2"]
        rows.append(r)

    E = pd.DataFrame(rows)
    E.to_csv(BASE / "artifacts" / "generation_quality.csv", index=False)
    ok = E[E.ok]
    if len(ok) == 0:
        print("[FATAL] no evaluable rows", file=sys.stderr)
        return 1

    print(f"\nRMSD (Å):")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q}: {np.percentile(ok.rmsd, q):.4f}")
    print(f"  mean {ok.rmsd.mean():.4f}   max {ok.rmsd.max():.4f}")

    dd_stats = {}
    print("\nforming-bond distance error (gen − ref, Å):")
    for c in ("dd1", "dd2"):
        v = ok[c].dropna()
        stats = dict(
            mean=float(v.mean()),
            abs_median=float(v.abs().median()),
            abs_p95=float(np.percentile(v.abs(), 95)),
        )
        dd_stats[c] = stats
        print(f"  {c}: mean {stats['mean']:+.4f}   "
              f"|Δ| median {stats['abs_median']:.4f}   "
              f"p95 {stats['abs_p95']:.4f}")

    med_max = max(dd_stats[c]["abs_median"] for c in ("dd1", "dd2"))
    if med_max < 0.05:
        gate = "PASS"
    elif med_max < 0.15:
        gate = "WARN"
    else:
        gate = "FAIL"
    (BASE / "artifacts" / "GATE8_STATUS.txt").write_text(
        f"{gate}\n"
        f"n_evaluable={len(ok)}\n"
        f"rmsd_median={float(np.median(ok.rmsd)):.6f}\n"
        f"dd1_abs_median={dd_stats['dd1']['abs_median']:.6f}\n"
        f"dd2_abs_median={dd_stats['dd2']['abs_median']:.6f}\n"
    )
    print(f"\n=== GATE-8 {gate}  (max |Δ_forming| median = {med_max:.4f} Å) ===")
    print("AM1 baseline for reference: 0.144 / 0.173 Å (Espley).")
    return 0 if gate != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
