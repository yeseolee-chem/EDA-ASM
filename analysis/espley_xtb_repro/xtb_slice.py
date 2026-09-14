#!/usr/bin/env python3
"""xtb_slice.py — GFN2-xTB(ALPB water) SPE on 5 species per rxn + Espley-style features.

Usage:
    python xtb_slice.py <slice_id> <n_slices> <output_parquet>

Changes vs v1 (2026-09-14 review):
  * accepted set = labels_all.json minus EXCLUDE (5,260 rxns), not status=="ok" (5,093)
  * solvation: ALPB water is REQUIRED (tblite: calc.add("alpb-solvation", "water"));
    v1 called calc.set("solvation", ...) which raises TypeError and was silently
    swallowed -> every SPE ran in the gas phase
  * interaction = E_ts - E_dist1 - E_dist2  (v1 had the opposite sign)
  * extra features analogous to Espley 2024 Table S2: Mulliken charges of the 4
    reacting atoms (TS, distorted fragments, relaxed reactants), reacting distances
    (formed_d1/d2 from labels), their difference; all cheap by-products
  * every row carries status / flags from labels_all and an explicit xtb_solvation tag
Rxns where xTB fails are recorded with NaN xtb_* columns (not dropped).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for cand in (HERE,
             Path("/gpfs/home1/yeseo1ee/projects/b3lyp_eda_bundle/scripts"),
             Path("/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/build_strategy/graph")):
    if (cand / "fragmenter.py").is_file():
        sys.path.insert(0, str(cand))
        break
import fragmenter as fr  # noqa: E402
from tblite.interface import Calculator  # noqa: E402

BOHR = 1.8897259886
EH2KCAL = 627.5094740631

LABELS_PATH = Path(os.environ.get("ESPLEY_LABELS", "/home1/yeseo1ee/projects/eda-asm-prediction/labels_all.json"))
PROF = Path(os.environ.get("ESPLEY_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))

# agreed exclusions (2026-09-14): 3 foreign-bond TSs + 2 with oi_dft > 0 (variational violation)
EXCLUDE = {3090, 3766, 4252, 3400, 5783}
SOLVENT = "water"


def xtb_spe(syms, xyz, charge=0):
    """GFN2-xTB / ALPB(water) single point. Returns (energy_Eh, mulliken_charges)."""
    Z = np.array([fr.Z[s] for s in syms], dtype=np.int64)
    c = Calculator("GFN2-xTB", Z, np.asarray(xyz, dtype=float) * BOHR, charge=int(charge))
    c.set("verbosity", 0)
    c.add("alpb-solvation", SOLVENT)          # raises if unavailable -> spe_fail (never silent)
    res = c.singlepoint()
    return float(res.get("energy")), np.asarray(res.get("charges"), dtype=float)


def process_one(label):
    rid = int(label["rxn_id"])
    q1 = int(label.get("charge1", 0) or 0)
    q2 = int(label.get("charge2", 0) or 0)
    row = {
        "rxn_id": rid, "charge1": q1, "charge2": q2,
        "role1": label.get("role1"), "role2": label.get("role2"),
        "n_atoms": label.get("n_atoms"), "n_f1": label.get("n_f1"), "n_f2": label.get("n_f2"),
        "label_status": label.get("status"),
        "flag_foreign_bond": bool(label.get("flag_foreign_bond", False)),
        "flag_async": bool(label.get("flag_async", False)),
        "flag_negative_strain": bool(label["d1_kcal"] < -0.5 or label["d2_kcal"] < -0.5),
        "flag_elst_positive": bool(label["elst_dft"] > 0),
        "alt_used": label.get("alt_used"),
        "xtb_solvation": f"alpb-{SOLVENT}",
    }
    # ---- DFT targets ----
    e_ab, f1d, f2d, f1r, f2r = (label[k] for k in ("e_ab_eh", "e_frag1_dist_eh", "e_frag2_dist_eh",
                                                   "e_frag1_rel_eh", "e_frag2_rel_eh"))
    row["dft_barrier_kcal"] = (e_ab - f1r - f2r) * EH2KCAL
    row["dft_eint_spe_kcal"] = (e_ab - f1d - f2d) * EH2KCAL      # ASM-consistent interaction
    row["dft_d1_kcal"] = label["d1_kcal"]
    row["dft_d2_kcal"] = label["d2_kcal"]
    row["dft_e_bond_kcal"] = label["e_bond_kcal"]                # EDA Bond Energy (= 6-channel sum)
    for ch in ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"):
        row["dft_" + ch] = label[ch]
    # reacting distances (SMILES-defined forming bonds, from labels)
    d1, d2 = float(label["formed_d1"]), float(label["formed_d2"])
    row["reacting_distance_0_ts"] = min(d1, d2)
    row["reacting_distance_1_ts"] = max(d1, d2)
    row["reacting_distance_diff_ts"] = abs(d1 - d2)

    # ---- geometries ----
    pdir = PROF / str(rid)
    try:
        rel1 = fr.read_xyz(pdir / label["rel1_file"])
        rel2 = fr.read_xyz(pdir / label["rel2_file"])
        ts = fr.read_xyz(pdir / label["ts_file"])
    except Exception as e:
        row["xtb_status"] = f"geom_load_fail:{type(e).__name__}"
        return row
    ts_syms, ts_xyz = ts

    # ---- partition (graph rule); A = image of rel1 (= dipole) by construction ----
    P = fr.partition(ts, rel1, rel2)
    if P.status != "ok":
        row["xtb_status"] = f"partition:{P.status}"
        return row
    A_idx, B_idx = sorted(P.A), sorted(P.B)
    inv0 = {v: k for k, v in P.map0.items()}   # TS idx -> rel1 heavy idx
    inv1 = {v: k for k, v in P.map1.items()}   # TS idx -> rel2 heavy idx

    # reacting atoms: forming bonds in TS indices, bond 1 = shorter
    pairs = [tuple(map(int, p.split("-"))) for p in str(label["formed_pairs_ts"]).split()]
    D = fr.dist_matrix(ts_xyz)
    pairs.sort(key=lambda p: D[p[0], p[1]])
    react = []   # [(dip_atom_ts, dph_atom_ts)] for bond 1, bond 2
    for i, j in pairs:
        if (i in P.A) == (j in P.A):
            row["xtb_status"] = "forming_pair_not_across_partition"
            return row
        react.append((i, j) if i in P.A else (j, i))

    # ---- 5 SPEs ----
    try:
        E_gs1, q_gs1 = xtb_spe(*rel1, q1)
        E_gs2, q_gs2 = xtb_spe(*rel2, q2)
        E_d1, q_d1 = xtb_spe([ts_syms[i] for i in A_idx], ts_xyz[A_idx], q1)
        E_d2, q_d2 = xtb_spe([ts_syms[i] for i in B_idx], ts_xyz[B_idx], q2)
        E_ts, q_ts = xtb_spe(ts_syms, ts_xyz, q1 + q2)
    except Exception as e:
        row["xtb_status"] = f"spe_fail:{type(e).__name__}:{str(e)[:80]}"
        return row

    row.update(xtb_e_gs1_eh=E_gs1, xtb_e_gs2_eh=E_gs2, xtb_e_dist1_eh=E_d1, xtb_e_dist2_eh=E_d2, xtb_e_ts_eh=E_ts)
    e_barrier = (E_ts - E_gs1 - E_gs2) * EH2KCAL
    dist_dp = (E_d1 - E_gs1) * EH2KCAL
    dist_di = (E_d2 - E_gs2) * EH2KCAL
    row["xtb_e_barrier_kcal"] = e_barrier
    row["xtb_dist_dipole_kcal"] = dist_dp
    row["xtb_dist_dipolarophile_kcal"] = dist_di
    row["xtb_sum_distortion_kcal"] = dist_dp + dist_di
    row["xtb_interaction_kcal"] = (E_ts - E_d1 - E_d2) * EH2KCAL       # = e_barrier - sum_distortion

    # ---- Mulliken charges at the 4 reacting atoms (Espley Table S2 analogue) ----
    posA = {a: k for k, a in enumerate(A_idx)}; posB = {b: k for k, b in enumerate(B_idx)}
    for n, (ia, ib) in enumerate(react, start=1):
        row[f"q_ts_dip_{n}"] = float(q_ts[ia]);            row[f"q_ts_dph_{n}"] = float(q_ts[ib])
        row[f"q_dist_dip_{n}"] = float(q_d1[posA[ia]]);    row[f"q_dist_dph_{n}"] = float(q_d2[posB[ib]])
        row[f"q_gs_dip_{n}"] = float(q_gs1[inv0[ia]]);     row[f"q_gs_dph_{n}"] = float(q_gs2[inv1[ib]])
    row["q_ts_dipole_sum"] = float(q_ts[A_idx].sum())      # charge transfer at TS
    row["xtb_status"] = "ok"
    return row


def main():
    slice_id, n_slices, out_path = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    labels = json.load(open(LABELS_PATH))
    accepted = sorted((d for d in labels if int(d["rxn_id"]) not in EXCLUDE), key=lambda d: d["rxn_id"])
    total = len(accepted)
    per = math.ceil(total / n_slices)
    start, end = slice_id * per, min((slice_id + 1) * per, total)
    mine = accepted[start:end]
    print(f"[slice {slice_id}/{n_slices}] accepted={total} (labels {len(labels)} - exclude {len(EXCLUDE)})  "
          f"processing {start}..{end} ({len(mine)} rxns)", flush=True)
    rows = []
    for i, label in enumerate(mine):
        try:
            rows.append(process_one(label))
        except Exception as e:  # never lose the slice
            rows.append(dict(rxn_id=int(label["rxn_id"]), xtb_status=f"error:{type(e).__name__}:{str(e)[:80]}"))
        if (i + 1) % 25 == 0:
            print(f"[slice {slice_id}] {i+1}/{len(mine)} ok={sum(r.get('xtb_status') == 'ok' for r in rows)}", flush=True)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[slice {slice_id}] wrote {len(df)} rows -> {out_path}; status: {dict(df['xtb_status'].value_counts())}", flush=True)


if __name__ == "__main__":
    main()
