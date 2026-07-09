"""Full audit of the 4 raw datasets: for each reaction, check whether R
supports EXACTLY 2 fragments (not 1, not 3+). Report pass rate per family
and save the passing reaction lists as parquets.

Family-specific criteria:
  - dipolar: r0_*.xyz AND r1_*.xyz both exist, NO r2_*.xyz or higher.
  - rgd1:    Rsmiles has EXACTLY 2 dot-separated parts.
  - qmrxn20 (e2/sn2): substrate xyz exists at reactant-conformers/
             AND its element sequence matches the first n_sub atoms of the
             bound complex (reactant-complex-constrained-conformers).
"""
from __future__ import annotations
import re
from collections import Counter
from glob import glob
from pathlib import Path

import ase.io
import h5py
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
DIP_ROOT = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
QMR_TS = RAW / "QMrxn20" / "transition-states"
QMR_RC = RAW / "QMrxn20" / "reactant-complex-constrained-conformers"
QMR_SUB = RAW / "QMrxn20" / "reactant-conformers"
RGD1_H5 = RAW / "rgd1" / "RGD1_CHNO.h5"

OUT_DIR = REPO / "outputs/2frag_audit"


# --------------------------- dipolar ---------------------------
def audit_dipolar():
    print("[dipolar]", flush=True)
    rows = []
    n = 0
    for d in sorted(DIP_ROOT.iterdir()):
        if not d.is_dir(): continue
        try:
            idx = int(d.name)
        except ValueError:
            continue
        n += 1
        r_files = sorted([f.name for f in d.iterdir()
                          if f.name.startswith("r") and f.name.endswith(".xyz")
                          and re.match(r"^r\d+_", f.name)])
        r0 = [f for f in r_files if f.startswith("r0_") and "_alt" not in f]
        r1 = [f for f in r_files if f.startswith("r1_") and "_alt" not in f]
        r2_or_higher = [f for f in r_files if re.match(r"^r[2-9]_", f)]
        passing = bool(r0) and bool(r1) and not r2_or_higher
        rows.append({
            "reaction_id": f"dipolar_{idx:06d}",
            "raw_idx": idx,
            "passing": passing,
            "n_reactant_files": (1 if r0 else 0) + (1 if r1 else 0) + len(r2_or_higher),
            "reason": (
                "" if passing
                else ("no_r0" if not r0
                      else "no_r1" if not r1
                      else f"has_r{2+r2_or_higher.index(r2_or_higher[0])}")),
        })
    df = pd.DataFrame(rows)
    print(f"  total: {n}  passing: {int(df.passing.sum())} ({df.passing.mean()*100:.1f}%)")
    return df


# --------------------------- rgd1 ---------------------------
def audit_rgd1():
    print("[rgd1]", flush=True)
    rows = []
    with h5py.File(RGD1_H5, "r") as f:
        keys = list(f.keys())
        n = len(keys)
        print(f"  reading {n} h5 keys...", flush=True)
        for k in keys:
            try:
                rs = f[k]["Rsmiles"][()]
                rs = rs.decode() if isinstance(rs, bytes) else rs
                parts = rs.split(".")
                passing = len(parts) == 2
                rows.append({
                    "reaction_id": f"rgd1_{k}",
                    "raw_key": k,
                    "passing": passing,
                    "n_smiles_parts": len(parts),
                    "reason": "" if passing else f"n_parts={len(parts)}",
                })
            except Exception as exc:
                rows.append({
                    "reaction_id": f"rgd1_{k}",
                    "raw_key": k,
                    "passing": False,
                    "n_smiles_parts": 0,
                    "reason": f"err:{exc}",
                })
    df = pd.DataFrame(rows)
    print(f"  total: {n}  passing: {int(df.passing.sum())} ({df.passing.mean()*100:.1f}%)")
    return df


# --------------------------- qmrxn20 ---------------------------
def _substrate_label(label: str) -> str:
    return "_".join(label.split("_")[:-1]) + "_0"


def _substrate_z(sub_label: str):
    sub_dir = QMR_SUB / sub_label
    if not sub_dir.exists(): return None
    xyz = sub_dir / "00.xyz"
    if not xyz.exists():
        xyz = next(iter(sub_dir.glob("*.xyz")), None)
    if xyz is None or not xyz.exists(): return None
    return np.asarray(ase.io.read(str(xyz)).get_atomic_numbers(), int)


def _complex_z(subfam: str, label: str):
    rc = QMR_RC / subfam / label
    if not rc.exists(): return None
    xyz = rc / "00.xyz"
    if not xyz.exists():
        xyz = next(iter(rc.glob("*.xyz")), None)
    if xyz is None or not xyz.exists(): return None
    return np.asarray(ase.io.read(str(xyz)).get_atomic_numbers(), int)


def audit_qmrxn20(subfam: str):
    print(f"[qmrxn20_{subfam}]", flush=True)
    rows = []
    labels = [Path(p).stem for p in sorted(glob(str(QMR_TS / subfam / "*.xyz")))]
    for label in labels:
        sub_label = _substrate_label(label)
        z_sub = _substrate_z(sub_label)
        z_comp = _complex_z(subfam, label)
        if z_sub is None:
            reason = "no_substrate_file"
            passing = False
        elif z_comp is None:
            reason = "no_complex_file"
            passing = False
        elif len(z_sub) >= len(z_comp):
            reason = f"substrate_n>=complex_n ({len(z_sub)},{len(z_comp)})"
            passing = False
        elif not np.array_equal(z_sub, z_comp[:len(z_sub)]):
            reason = "z_seq_mismatch"
            passing = False
        else:
            passing = True
            reason = ""
        rows.append({
            "reaction_id": f"qmrxn20_{subfam}_{label}",
            "label": label,
            "substrate_label": sub_label,
            "n_substrate": int(len(z_sub)) if z_sub is not None else 0,
            "n_complex": int(len(z_comp)) if z_comp is not None else 0,
            "passing": passing,
            "reason": reason,
        })
    df = pd.DataFrame(rows)
    print(f"  total: {len(rows)}  passing: {int(df.passing.sum())} ({df.passing.mean()*100:.1f}%)")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dip = audit_dipolar()
    dip.to_parquet(OUT_DIR / "audit_dipolar.parquet", index=False)
    dip[dip.passing].drop(columns=["passing", "reason"]).to_parquet(
        OUT_DIR / "passing_dipolar.parquet", index=False)

    r_e2 = audit_qmrxn20("e2")
    r_e2.to_parquet(OUT_DIR / "audit_qmrxn20_e2.parquet", index=False)
    r_e2[r_e2.passing].drop(columns=["passing", "reason"]).to_parquet(
        OUT_DIR / "passing_qmrxn20_e2.parquet", index=False)

    r_sn2 = audit_qmrxn20("sn2")
    r_sn2.to_parquet(OUT_DIR / "audit_qmrxn20_sn2.parquet", index=False)
    r_sn2[r_sn2.passing].drop(columns=["passing", "reason"]).to_parquet(
        OUT_DIR / "passing_qmrxn20_sn2.parquet", index=False)

    rg = audit_rgd1()
    rg.to_parquet(OUT_DIR / "audit_rgd1.parquet", index=False)
    rg[rg.passing].drop(columns=["passing", "reason"]).to_parquet(
        OUT_DIR / "passing_rgd1.parquet", index=False)

    print(f"\nWrote to {OUT_DIR}/")
    print(f"  audit_*.parquet         — full audit rows (with pass/fail reasons)")
    print(f"  passing_*.parquet       — only reactions that pass 2-fragment check")

    total = len(dip) + len(r_e2) + len(r_sn2) + len(rg)
    passing = int(dip.passing.sum() + r_e2.passing.sum() + r_sn2.passing.sum() + rg.passing.sum())
    print(f"\nGRAND TOTAL: {passing} / {total} passing ({passing/total*100:.1f}%)")


if __name__ == "__main__":
    main()
