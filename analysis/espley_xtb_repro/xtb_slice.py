#!/usr/bin/env python3
"""xtb_slice.py — GFN2-xTB SPE on 5 species per rxn, over a slice of accepted labels.

Usage:
    python xtb_slice.py <slice_id> <n_slices> <output_parquet>

Writes a partial parquet with one row per rxn processed. Rxns where xTB SPE
fails are recorded with NaN in xtb_* columns (not dropped).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/build_strategy/graph")
import fragmenter as fr  # noqa: E402
from tblite.interface import Calculator  # noqa: E402

BOHR = 1.8897259886
EH2KCAL = 627.5094740631

LABELS_PATH = Path("/home1/yeseo1ee/projects/eda-asm-prediction/labels_all.json")
PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")


def xtb_spe(syms, xyz, charge=0):
    """GFN2-xTB single-point energy (Hartree). Tries ALPB water; else gas phase."""
    Z = np.array([fr.Z[s] for s in syms], dtype=np.int64)
    c = Calculator("GFN2-xTB", Z, np.asarray(xyz) * BOHR, charge=int(charge))
    c.set("verbosity", 0)
    try:
        c.set("solvation", "alpb", "water")
    except Exception:
        pass
    return float(c.singlepoint().get("energy"))


def process_one(label):
    """Return dict for one rxn; NaN xtb_* if compute fails; None if unrecoverable."""
    rid = int(label["rxn_id"])
    q1 = int(label.get("charge1", 0) or 0)
    q2 = int(label.get("charge2", 0) or 0)

    row = {
        "rxn_id": rid,
        "charge1": q1, "charge2": q2,
        "role1": label.get("role1"), "role2": label.get("role2"),
        "n_atoms": label.get("n_atoms"), "n_f1": label.get("n_f1"), "n_f2": label.get("n_f2"),
        "flag_foreign_bond": label.get("flag_foreign_bond"),
        "flag_async": label.get("flag_async"),
        "alt_used": label.get("alt_used"),
    }

    # ---- DFT targets from labels_all ----
    e_ab = label["e_ab_eh"]
    e_f1r = label["e_frag1_rel_eh"]; e_f2r = label["e_frag2_rel_eh"]
    row["dft_barrier_kcal"] = (e_ab - e_f1r - e_f2r) * EH2KCAL
    row["dft_d1_kcal"] = label["d1_kcal"]
    row["dft_d2_kcal"] = label["d2_kcal"]
    row["dft_e_bond_kcal"] = label["e_bond_kcal"]
    for ch in ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"):
        row["dft_" + ch] = label[ch]

    # ---- Load Coley geometries ----
    pdir = PROF / str(rid)
    try:
        rel1_syms, rel1_xyz = fr.read_xyz(pdir / label["rel1_file"])
        rel2_syms, rel2_xyz = fr.read_xyz(pdir / label["rel2_file"])
        ts_syms,   ts_xyz   = fr.read_xyz(pdir / label["ts_file"])
    except Exception as e:
        row["xtb_status"] = f"geom_load_fail:{type(e).__name__}"
        return row

    # ---- Fragment partition (graph rule) ----
    # labels_all: rel1_file = dipole reactant. Call partition with rel1 as r0 first,
    # then verify composition & swap if needed.
    try:
        P = fr.partition((ts_syms, ts_xyz), (rel1_syms, rel1_xyz), (rel2_syms, rel2_xyz))
    except Exception as e:
        row["xtb_status"] = f"partition_error:{type(e).__name__}"
        return row
    if P.status != "ok":
        row["xtb_status"] = f"partition:{P.status}"
        return row
    A_idx = sorted(P.A); B_idx = sorted(P.B)
    if Counter(ts_syms[i] for i in A_idx) != Counter(rel1_syms):
        A_idx, B_idx = B_idx, A_idx

    dist1_syms = [ts_syms[i] for i in A_idx]
    dist1_xyz = ts_xyz[A_idx]
    dist2_syms = [ts_syms[i] for i in B_idx]
    dist2_xyz = ts_xyz[B_idx]

    # ---- 5 SPE ----
    try:
        E_gs1 = xtb_spe(rel1_syms, rel1_xyz, q1)
        E_gs2 = xtb_spe(rel2_syms, rel2_xyz, q2)
        E_dist1 = xtb_spe(dist1_syms, dist1_xyz, q1)
        E_dist2 = xtb_spe(dist2_syms, dist2_xyz, q2)
        E_ts = xtb_spe(ts_syms, ts_xyz, q1 + q2)
    except Exception as e:
        row["xtb_status"] = f"spe_fail:{type(e).__name__}:{str(e)[:80]}"
        return row

    row["xtb_e_gs1_eh"] = E_gs1
    row["xtb_e_gs2_eh"] = E_gs2
    row["xtb_e_dist1_eh"] = E_dist1
    row["xtb_e_dist2_eh"] = E_dist2
    row["xtb_e_ts_eh"] = E_ts

    e_barrier = (E_ts - E_gs1 - E_gs2) * EH2KCAL
    dist_dp = (E_dist1 - E_gs1) * EH2KCAL
    dist_di = (E_dist2 - E_gs2) * EH2KCAL
    sum_dist = dist_dp + dist_di
    interaction = sum_dist - e_barrier

    row["xtb_e_barrier_kcal"] = e_barrier
    row["xtb_dist_dipole_kcal"] = dist_dp
    row["xtb_dist_dipolarophile_kcal"] = dist_di
    row["xtb_sum_distortion_kcal"] = sum_dist
    row["xtb_interaction_kcal"] = interaction
    row["xtb_status"] = "ok"
    return row


def main():
    slice_id = int(sys.argv[1])
    n_slices = int(sys.argv[2])
    out_path = Path(sys.argv[3])

    all_labels = json.load(open(LABELS_PATH))
    accepted = [d for d in all_labels if d.get("status") == "ok"]
    accepted.sort(key=lambda d: d["rxn_id"])
    total = len(accepted)
    per = math.ceil(total / n_slices)
    start = slice_id * per
    end = min(start + per, total)
    my_labels = accepted[start:end]
    print(f"[slice {slice_id}/{n_slices}] total accepted={total}  processing {start}..{end}  ({len(my_labels)} rxns)", flush=True)

    rows = []
    for i, label in enumerate(my_labels):
        r = process_one(label)
        rows.append(r)
        if (i + 1) % 25 == 0:
            n_ok = sum(1 for x in rows if x.get("xtb_status") == "ok")
            print(f"[slice {slice_id}] {i+1}/{len(my_labels)}  ok={n_ok}", flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[slice {slice_id}] wrote {len(df)} rows to {out_path}", flush=True)
    print(f"[slice {slice_id}] status counts: {dict(df['xtb_status'].value_counts())}", flush=True)


if __name__ == "__main__":
    main()
