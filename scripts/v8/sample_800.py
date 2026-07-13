"""Phase 1 - sample 200 reactions from each of 4 families = 800 total (v8 cohort).

Source: RAW XYZ files under /gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/
Copies R.xyz + TS.xyz + P.xyz to outputs/v8_review/raw_geoms/{rid}/

Cohort saved to outputs/v8_review/cohort_v8.parquet with columns:
  reaction_id, family, n_atoms_R, n_atoms_TS, n_atoms_P
"""
from __future__ import annotations
import random, shutil, sys
from pathlib import Path

import ase.io
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
OUT = REPO / "outputs/v8_review/raw_geoms"
COHORT = REPO / "outputs/v8_review/cohort_v8.parquet"

SEED = 42
N_PER_FAMILY = 200


def _single(dir_path: Path, pattern: str):
    m = sorted(dir_path.glob(pattern))
    return m[0] if m else None


def load_dipolar(idx: int):
    d = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles" / str(idx)
    if not d.exists(): return None
    r0 = _single(d, "r0_*.xyz")
    r1 = _single(d, "r1_*.xyz")
    ts = _single(d, "TS_imag_mode.xyz") or _single(d, "TS_imag_mode_*.xyz") or _single(d, "TS_*.xyz")
    p0 = _single(d, "p0_*.xyz")
    if not (r0 and ts and p0):
        return None
    R_at = ase.io.read(str(r0))
    if r1:
        R_at = R_at + ase.io.read(str(r1))
    TS_at = ase.io.read(str(ts))
    P_at = ase.io.read(str(p0))
    return R_at, TS_at, P_at


def load_qmrxn20(subfam: str, label: str):
    root = RAW / "QMrxn20"
    ts_p = root / "transition-states" / subfam / f"{label}.xyz"
    if not ts_p.exists(): return None
    TS_at = ase.io.read(str(ts_p))
    rc = root / "reactant-complex-constrained-conformers" / subfam / label
    r_p = rc / "00.xyz"
    if not r_p.exists():
        r_p = next(iter(rc.glob("*.xyz")), None) if rc.exists() else None
    R_at = ase.io.read(str(r_p)) if r_p else None
    parts = label.split("_")
    if subfam == "e2":
        plabel = "_".join(parts[:4] + ["0", "0"])
    else:
        plabel = "_".join(parts[:4] + ["0", parts[5]])
    pd_dir = root / "product-conformers" / subfam / plabel
    p_path = pd_dir / "00.xyz"
    if not p_path.exists() and pd_dir.exists():
        p_path = next(iter(pd_dir.glob("*.xyz")), None)
    P_at = ase.io.read(str(p_path)) if p_path else None
    if R_at is None or P_at is None: return None
    return R_at, TS_at, P_at


def load_rgd1(rid: str):
    d = RAW / "rgd1" / "extracted_xyz" / rid
    if not d.exists():
        return None
    if not all((d / f"{s}.xyz").exists() for s in ("R", "TS", "P")):
        return None
    return (ase.io.read(str(d / "R.xyz")),
            ase.io.read(str(d / "TS.xyz")),
            ase.io.read(str(d / "P.xyz")))


def write_xyz(atoms, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(path), atoms, plain=True)


def main():
    rng = random.Random(SEED)
    all_rows = []

    # dipolar: indices 0..5999 in full_dataset_profiles
    dip_root = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
    dip_ids = sorted([int(p.name) for p in dip_root.iterdir() if p.is_dir() and p.name.isdigit()])
    rng.shuffle(dip_ids)
    n_ok = 0
    for i in dip_ids:
        if n_ok >= N_PER_FAMILY: break
        triple = load_dipolar(i)
        if triple is None: continue
        R_at, TS_at, P_at = triple
        rid = f"dipolar_{i:06d}"
        out_dir = OUT / rid
        write_xyz(R_at,  out_dir / "R.xyz")
        write_xyz(TS_at, out_dir / "TS.xyz")
        write_xyz(P_at,  out_dir / "P.xyz")
        all_rows.append({"reaction_id": rid, "family": "dipolar",
                         "n_atoms_R": len(R_at), "n_atoms_TS": len(TS_at), "n_atoms_P": len(P_at)})
        n_ok += 1
    print(f"dipolar: sampled {n_ok}/{N_PER_FAMILY}")

    # qmrxn20 e2 and sn2
    for subfam in ("e2", "sn2"):
        ts_root = RAW / "QMrxn20" / "transition-states" / subfam
        labels = sorted([p.stem for p in ts_root.glob("*.xyz")])
        rng.shuffle(labels)
        n_ok = 0
        for lab in labels:
            if n_ok >= N_PER_FAMILY: break
            triple = load_qmrxn20(subfam, lab)
            if triple is None: continue
            R_at, TS_at, P_at = triple
            rid = f"qmrxn20_{subfam}_{lab}"
            out_dir = OUT / rid
            write_xyz(R_at,  out_dir / "R.xyz")
            write_xyz(TS_at, out_dir / "TS.xyz")
            write_xyz(P_at,  out_dir / "P.xyz")
            all_rows.append({"reaction_id": rid, "family": f"qmrxn20_{subfam}",
                             "n_atoms_R": len(R_at), "n_atoms_TS": len(TS_at), "n_atoms_P": len(P_at)})
            n_ok += 1
        print(f"qmrxn20_{subfam}: sampled {n_ok}/{N_PER_FAMILY}")

    # rgd1: enumerate all extracted; sample if abundant, or extract more from h5
    rgd_root = RAW / "rgd1" / "extracted_xyz"
    rgd_ids = sorted([p.name for p in rgd_root.iterdir() if p.is_dir()])
    rng.shuffle(rgd_ids)
    n_ok = 0
    for rid in rgd_ids:
        if n_ok >= N_PER_FAMILY: break
        triple = load_rgd1(rid)
        if triple is None: continue
        R_at, TS_at, P_at = triple
        out_dir = OUT / rid
        write_xyz(R_at,  out_dir / "R.xyz")
        write_xyz(TS_at, out_dir / "TS.xyz")
        write_xyz(P_at,  out_dir / "P.xyz")
        all_rows.append({"reaction_id": rid, "family": "rgd1",
                         "n_atoms_R": len(R_at), "n_atoms_TS": len(TS_at), "n_atoms_P": len(P_at)})
        n_ok += 1
    print(f"rgd1: sampled {n_ok}/{N_PER_FAMILY}")

    df = pd.DataFrame(all_rows)
    df.to_parquet(COHORT, index=False)
    print(f"\nwrote {COHORT}  total = {len(df)}  ({dict(df.family.value_counts())})")


if __name__ == "__main__":
    main()
