#!/usr/bin/env python3
"""xtb_slice.py — GFN2-xTB (tblite, gas phase) SPE on 5 species per rxn,
producing the 54-feature Espley-style set + 11 DFT targets.

The original Espley 2024 set is 55 = 41 structural + 6 AM1 energies + 8 GFN2-xTB
channels. We drop q_barrier (which needs a Hessian tblite cannot provide),
leaving 54 = 41 + 5 + 8.

Feature layout (54):
    d^struct = 41
        R_dipole (3 dipole atoms in relaxed geom):        3 dist + 3 Mull + 3 APT_surr = 9
        R_dipolarophile (2 dph atoms in relaxed geom):    1 dist + 2 Mull + 2 APT_surr = 5
        TS_dipole_atoms (3 dip atoms @ TS geom):          3 dist + 3 Mull + 3 APT_surr = 9
        TS_dipolarophile_atoms (2 dph @ TS geom):         1 dist + 2 Mull + 2 APT_surr = 5
        TS whole complex:                                  3 dist + 5 Mull + 5 APT_surr = 13
    b^xtb Energies = 5  (q_barrier dropped)
        xtb_e_barrier, xtb_dist_dipole, xtb_dist_dipolarophile,
        xtb_sum_distortion, xtb_interaction
    b^ch GFN2-xTB channels = 8
        ch_strain_1 (=xtb_dist_dipole, real),
        ch_strain_2 (=xtb_dist_dipolarophile, real),
        ch_elst  (real; charge-separation proxy from Mulliken),
        ch_Pauli (real; mean Wiberg bond order at TS reacting atoms),
        ch_oi    (real; HOMO(TS)-LUMO(TS) gap),
        ch_disp = 0.0   (marker: tblite exposes no D3/D4 separation),
        ch_cpcm = 0.0   (marker: tblite exposes no ALPB solvation),
        ch_cds  = 0.0

Notes:
    * Mulliken = tblite `charges` (atomic partial charges from the GFN2 SCC scheme).
    * APT_surrogate = Wiberg bond order row sum per atom (tblite `bond-orders`).
      tblite cannot compute the true APT (needs dipole-derivative Hessian);
      the Wiberg sum is a physically distinct per-atom quantity used here as
      a defensible surrogate. Documented in README.
    * ch_disp / ch_cpcm / ch_cds are constant zeros — placeholders for
      channels that tblite (this version) cannot resolve. Constant columns
      are harmless: sklearn scalers and tree learners ignore them.
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
EV_PER_HARTREE = 27.2114079527

LABELS_PATH = Path(os.environ.get("ESPLEY_LABELS", "/home1/yeseo1ee/projects/eda-asm-prediction/labels_all.json"))
PROF = Path(os.environ.get("ESPLEY_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))
EXCLUDE = {3090, 3766, 4252, 3400, 5783}


def xtb_features(syms, xyz, charge=0):
    """One GFN2-xTB SPE. Returns dict with energy, charges, wbo_sum, hlgap."""
    Z = np.array([fr.Z[s] for s in syms], dtype=np.int64)
    c = Calculator("GFN2-xTB", Z, np.asarray(xyz, dtype=float) * BOHR, charge=int(charge))
    c.set("verbosity", 0)
    res = c.singlepoint()
    charges = np.asarray(res.get("charges"), dtype=float)
    try:
        bo = np.asarray(res.get("bond-orders"), dtype=float)
        wbo_sum = bo.sum(axis=1)
    except Exception:
        wbo_sum = np.zeros_like(charges)
    hlgap = float("nan")
    try:
        oe = np.asarray(res.get("orbital-energies"), dtype=float)
        n_el = sum(fr.Z[s] for s in syms) - int(charge)
        n_occ = n_el // 2
        if 0 < n_occ < len(oe):
            hlgap = float(oe[n_occ] - oe[n_occ - 1]) * EV_PER_HARTREE   # eV
    except Exception:
        pass
    return dict(energy=float(res.get("energy")), charges=charges, wbo_sum=wbo_sum, hlgap=hlgap)


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
        "alt_used": label.get("alt_used"),
        "xtb_solvation": "gas",
    }
    # ---- DFT targets ----
    e_ab, f1d, f2d, f1r, f2r = (label[k] for k in ("e_ab_eh", "e_frag1_dist_eh", "e_frag2_dist_eh",
                                                   "e_frag1_rel_eh", "e_frag2_rel_eh"))
    row["dft_barrier_kcal"] = (e_ab - f1r - f2r) * EH2KCAL
    row["dft_eint_spe_kcal"] = (e_ab - f1d - f2d) * EH2KCAL
    row["dft_d1_kcal"] = label["d1_kcal"]
    row["dft_d2_kcal"] = label["d2_kcal"]
    row["dft_e_bond_kcal"] = label["e_bond_kcal"]
    for ch in ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"):
        row["dft_" + ch] = label[ch]

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
    rel1_syms, rel1_xyz = rel1
    rel2_syms, rel2_xyz = rel2

    # ---- partition ----
    P = fr.partition(ts, rel1, rel2)
    if P.status != "ok":
        row["xtb_status"] = f"partition:{P.status}"
        return row
    A_idx, B_idx = sorted(P.A), sorted(P.B)
    inv0 = {v: k for k, v in P.map0.items()}   # TS idx -> rel1 heavy idx
    inv1 = {v: k for k, v in P.map1.items()}

    # ---- reacting atoms (5): dip_a, dip_middle, dip_b + dph_a, dph_b ----
    pairs = [tuple(map(int, p.split("-"))) for p in str(label["formed_pairs_ts"]).split()]
    D_ts = fr.dist_matrix(ts_xyz)
    pairs.sort(key=lambda p: D_ts[p[0], p[1]])
    react = []
    for i, j in pairs:
        if (i in P.A) == (j in P.A):
            row["xtb_status"] = "forming_pair_not_across_partition"
            return row
        react.append((i, j) if i in P.A else (j, i))
    (dip_a, dph_a), (dip_b, dph_b) = react[0], react[1]
    # middle dipole atom: heavy atom in A bonded (TS graph) to both dip_a and dip_b
    adj, _ = fr.bond_matrix(ts_syms, ts_xyz)
    dip_mid = None
    for i in A_idx:
        if ts_syms[i] == "H" or i in (dip_a, dip_b):
            continue
        if adj[i, dip_a] and adj[i, dip_b]:
            dip_mid = i
            break
    if dip_mid is None:
        # fallback: nearest heavy in A to midpoint of dip_a/dip_b
        mid = (ts_xyz[dip_a] + ts_xyz[dip_b]) / 2
        heavy_A = [i for i in A_idx if ts_syms[i] != "H" and i not in (dip_a, dip_b)]
        if not heavy_A:
            row["xtb_status"] = "no_middle_dipole_atom"
            return row
        dip_mid = min(heavy_A, key=lambda i: np.linalg.norm(ts_xyz[i] - mid))
    ts_dip = (dip_a, dip_mid, dip_b)   # 3 dipole reacting atoms (TS idx)
    ts_dph = (dph_a, dph_b)             # 2 dipolarophile reacting atoms

    # inverse maps (TS idx -> rel heavy idx)
    try:
        r1_dip = tuple(inv0[i] for i in ts_dip)   # in rel1_xyz coords
        r2_dph = tuple(inv1[i] for i in ts_dph)   # in rel2_xyz coords
    except KeyError:
        row["xtb_status"] = "reacting_atom_not_in_partition_map"
        return row

    D_rel1 = fr.dist_matrix(rel1_xyz)
    D_rel2 = fr.dist_matrix(rel2_xyz)

    # ---- 11 distances ----
    row["dist_R_dip_ab"]     = float(D_rel1[r1_dip[0], r1_dip[1]])
    row["dist_R_dip_bc"]     = float(D_rel1[r1_dip[1], r1_dip[2]])
    row["dist_R_dip_ac"]     = float(D_rel1[r1_dip[0], r1_dip[2]])
    row["dist_R_dph_ab"]     = float(D_rel2[r2_dph[0], r2_dph[1]])
    row["dist_TS_dip_ab"]    = float(D_ts[ts_dip[0], ts_dip[1]])
    row["dist_TS_dip_bc"]    = float(D_ts[ts_dip[1], ts_dip[2]])
    row["dist_TS_dip_ac"]    = float(D_ts[ts_dip[0], ts_dip[2]])
    row["dist_TS_dph_ab"]    = float(D_ts[ts_dph[0], ts_dph[1]])
    row["dist_TS_form_ad"]   = float(D_ts[dip_a, dph_a])   # forming bond 0
    row["dist_TS_form_be"]   = float(D_ts[dip_b, dph_b])   # forming bond 1
    row["dist_TS_diag_ae"]   = float(D_ts[dip_a, dph_b])   # diagonal (non-forming)

    # ---- 5 SPEs ----
    try:
        gs1 = xtb_features(rel1_syms, rel1_xyz, q1)
        gs2 = xtb_features(rel2_syms, rel2_xyz, q2)
        d1 = xtb_features([ts_syms[i] for i in A_idx], ts_xyz[A_idx], q1)
        d2 = xtb_features([ts_syms[i] for i in B_idx], ts_xyz[B_idx], q2)
        tsr = xtb_features(ts_syms, ts_xyz, q1 + q2)
    except Exception as e:
        row["xtb_status"] = f"spe_fail:{type(e).__name__}:{str(e)[:80]}"
        return row

    # ---- 15 Mulliken + 15 APT_surrogate (Wiberg BO sum) ----
    #    R_dipole (3 dipole atoms in rel1): 3 + 3
    for k, i in enumerate(r1_dip):
        row[f"mulliken_R_dip_{k}"] = float(gs1["charges"][i])
        row[f"aptsurr_R_dip_{k}"]  = float(gs1["wbo_sum"][i])
    #    R_dipolarophile (2 dph atoms in rel2): 2 + 2
    for k, i in enumerate(r2_dph):
        row[f"mulliken_R_dph_{k}"] = float(gs2["charges"][i])
        row[f"aptsurr_R_dph_{k}"]  = float(gs2["wbo_sum"][i])
    #    TS_dipole_atoms (3 dipole atoms @ TS geom, in dist1 fragment): 3 + 3
    posA = {a: k for k, a in enumerate(A_idx)}
    for k, i in enumerate(ts_dip):
        row[f"mulliken_TSdip_{k}"] = float(d1["charges"][posA[i]])
        row[f"aptsurr_TSdip_{k}"]  = float(d1["wbo_sum"][posA[i]])
    #    TS_dipolarophile_atoms (2 dph @ TS geom, in dist2 fragment): 2 + 2
    posB = {b: k for k, b in enumerate(B_idx)}
    for k, i in enumerate(ts_dph):
        row[f"mulliken_TSdph_{k}"] = float(d2["charges"][posB[i]])
        row[f"aptsurr_TSdph_{k}"]  = float(d2["wbo_sum"][posB[i]])
    #    TS whole (5 reacting atoms in full complex): 5 + 5
    all_react = list(ts_dip) + list(ts_dph)
    for k, i in enumerate(all_react):
        row[f"mulliken_TS_{k}"] = float(tsr["charges"][i])
        row[f"aptsurr_TS_{k}"]  = float(tsr["wbo_sum"][i])

    # ---- 6 xTB energies ----
    E_gs1, E_gs2, E_d1, E_d2, E_ts = gs1["energy"], gs2["energy"], d1["energy"], d2["energy"], tsr["energy"]
    e_barrier = (E_ts - E_gs1 - E_gs2) * EH2KCAL
    dist_dp = (E_d1 - E_gs1) * EH2KCAL
    dist_di = (E_d2 - E_gs2) * EH2KCAL
    sum_dist = dist_dp + dist_di
    interaction = (E_ts - E_d1 - E_d2) * EH2KCAL
    n_tot = len(ts_syms)
    row["xtb_e_barrier_kcal"]         = e_barrier
    row["xtb_dist_dipole_kcal"]       = dist_dp
    row["xtb_dist_dipolarophile_kcal"] = dist_di
    row["xtb_sum_distortion_kcal"]     = sum_dist
    row["xtb_interaction_kcal"]        = interaction

    # ---- 8 GFN2-xTB channels ----
    # real: strain (2), charge-separation proxy (elst), mean WBO (Pauli), HL gap (oi)
    # zero markers (tblite cannot resolve): disp, cpcm, cds
    row["ch_strain_1"] = dist_dp
    row["ch_strain_2"] = dist_di
    row["ch_elst"]     = float(sum(tsr["charges"][i] for i in ts_dip)
                               - sum(tsr["charges"][i] for i in ts_dph))
    row["ch_Pauli"]    = float(np.mean([tsr["wbo_sum"][i] for i in all_react]))
    row["ch_oi"]       = float(tsr["hlgap"])   # eV
    row["ch_disp"]     = 0.0
    row["ch_cpcm"]     = 0.0
    row["ch_cds"]      = 0.0

    row["xtb_e_gs1_eh"] = E_gs1; row["xtb_e_gs2_eh"] = E_gs2
    row["xtb_e_dist1_eh"] = E_d1; row["xtb_e_dist2_eh"] = E_d2
    row["xtb_e_ts_eh"] = E_ts
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
        except Exception as e:
            rows.append(dict(rxn_id=int(label["rxn_id"]), xtb_status=f"error:{type(e).__name__}:{str(e)[:80]}"))
        if (i + 1) % 25 == 0:
            print(f"[slice {slice_id}] {i+1}/{len(mine)} ok={sum(r.get('xtb_status') == 'ok' for r in rows)}", flush=True)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[slice {slice_id}] wrote {len(df)} rows -> {out_path}; status: {dict(df['xtb_status'].value_counts())}", flush=True)


if __name__ == "__main__":
    main()
