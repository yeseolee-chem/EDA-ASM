#!/usr/bin/env python3
"""xtb_slice.py (v4) — GFN2-xTB / ALPB(water) features on Coley DFT geometries, 5,260 rxns.

Usage: python xtb_slice.py <slice_id> <n_slices> <output_parquet>

Layout (68 feature columns, mirrors the 2026-08-27 "55" design with AM1 -> GFN2):
  D_STRUCT41 = 11 dist + 15 mulliken_* + 15 wbo_valence_*      (Espley Table S2 analogue)
  B_XTB5     = xtb_e_barrier / dist_dipole / dist_dipolarophile / sum_distortion / interaction
               (ALPB total energies; q_barrier omitted — no Hessian)
  B_CH8      = b_strain_1, b_strain_2, b_elst, b_pauli, b_oi, b_disp, b_cpcm, b_cds
               one REAL calculated value per EDA channel (see below)
  AUX15      = b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts, gap_dip, gap_dph,
               mu_ts, mu_dip, mu_dph, dmu_complexation, dsasa_{H,C,N,O,F,Cl,Br} (7)
               -> 11 + 7 = 18 columns; the ML feature set AUX18 uses all of them

B_CH8 definitions (Δ = TS − dist1 − dist2 unless stated; all with ALPB water):
  b_strain_1/2  gas-part strain of each fragment = Δ(E_total − G_solv) between the TS-geometry
                fragment and its relaxed reference. Distinct from xtb_dist_* (which is the full
                ALPB strain); the difference is the solvation contribution to strain.
  b_elst        frozen-fragment monopole Coulomb, Σ_{i∈A,j∈B} q_i q_j / r_ij × 332.0637,
                with q from SEPARATE GFN2/ALPB single points on the isolated fragments at the
                TS geometry (i.e. densities are frozen, not relaxed in the complex).
  b_pauli       Δ(repulsion energy) — GFN2's classical repulsion term.
  b_oi          Δ(EHT band-structure energy) = Δ(SCC − isotropic ES − anisotropic ES
                − anisotropic XC − dispersion − G_solv) — the one-electron/orbital part.
  b_disp        inter-fragment D3(BJ)/B3LYP dispersion at the TS geometry. This is the SAME
                quantity as the DFT disp channel (verified MAE 0.000 kcal/mol) — the gate in
                §4 of the SPEC checks it.
  b_cpcm        Δ(G_elec) — the ALPB Born/dielectric term (analogue of Delta CPCM Dielectric).
  b_cds         Δ(G_sasa + G_hb + G_shift) — the ALPB non-polar/SASA term (analogue of
                Delta SMD CDS correction). Weakest of the eight; per-element dsasa_* in AUX
                carries the Σ σ_k A_k functional form that a single scalar cannot.

SINGLE ENGINE: every GFN2 quantity comes from the xtb binary (one program, one version).
Each single point yields, in one run: the term-wise SUMMARY block, the `charges` file (Mulliken
partial charges, 8 decimals), the "Wiberg/Mayer (AO) data" table (per-atom total Wiberg valence),
the molecular dipole and the HOMO-LUMO gap. tblite is NOT used.
Note: xtb's pairwise Wiberg bond orders are printed only above ~0.1, so pairwise WBO features
(wbo_inter, wbo_form_*) are deliberately NOT part of this feature set; the per-atom valences
(exact) and the forming-bond distances carry that information instead.

Dependencies: xtb binary >= 6.7 (env XTB_BIN), dftd3, morfeus-ml, numpy, pandas, pyarrow, networkx.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for cand in (HERE, Path("/gpfs/home1/yeseo1ee/projects/b3lyp_eda_bundle/scripts")):
    if (cand / "fragmenter.py").is_file():
        sys.path.insert(0, str(cand))
        break
import fragmenter as fr  # noqa: E402
from dftd3.interface import DispersionModel, RationalDampingParam  # noqa: E402
from morfeus import SASA  # noqa: E402

BOHR = 1.8897259886
EH2KCAL = 627.5094740631
EH2EV = 27.211386245988
AU2DEBYE = 2.541746473
COULOMB = 332.0637130  # kcal·Å / (mol·e²)
SOLVENT = "water"
ELEMENTS = ("H", "C", "N", "O", "F", "Cl", "Br")

LABELS_PATH = Path(os.environ.get("ESPLEY_LABELS", "/home1/yeseo1ee/projects/eda-asm-prediction/labels_all.json"))
PROF = Path(os.environ.get("ESPLEY_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))
XTB_BIN = os.environ.get("XTB_BIN", "xtb")
EXCLUDE = {3090, 3766, 4252, 3400, 5783}

_TERMS = [("total", r"::\s+total energy"), ("scc", r"::\s+SCC energy"),
          ("ies", r"->\s+isotropic ES"), ("aes", r"->\s+anisotropic ES"),
          ("axc", r"->\s+anisotropic XC"), ("disp_d4", r"->\s+dispersion"),
          ("gsolv", r"->\s+Gsolv"), ("gelec", r"->\s+Gelec"), ("gsasa", r"->\s+Gsasa"),
          ("ghb", r"->\s+Ghb"), ("gshift", r"->\s+Gshift"), ("rep", r"::\s+repulsion energy")]
PAT = {k: re.compile(rf"{p}\s+(-?\d+\.\d+)\s+Eh") for k, p in _TERMS}
GAP_RE = re.compile(r"::\s+HOMO-LUMO gap\s+(-?\d+\.\d+)\s+eV")
DIP_RE = re.compile(r"full:\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+(\d+\.\d+)")
WBO_ROW_RE = re.compile(r"^\s+(\d+)\s+\d+\s+[A-Za-z]{1,2}\s+(-?\d+\.\d+)\s+--", re.M)


# ----------------------------------------------------------------------------- engines
def xtb_sp(syms, xyz, charge=0, solvent=SOLVENT):
    """One xtb GFN2 single point. Returns every quantity this pipeline needs.

    keys: term-wise energies (total, scc, ies, aes, axc, disp_d4, gsolv, gelec, gsasa, ghb,
    gshift, rep, gas, eht), charges (Mulliken, n), valence (per-atom total Wiberg, n),
    gap (eV), mu (Debye).
    """
    n = len(syms)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "m.xyz"), "w") as f:
            f.write(f"{n}\n\n")
            for s_, x in zip(syms, np.asarray(xyz, dtype=float)):
                f.write(f"{s_} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}\n")
        cmd = [XTB_BIN, "m.xyz", "--gfn", "2", "--sp", "--chrg", str(int(charge))]
        if solvent:
            cmd += ["--alpb", solvent]
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=1800,
                           env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("XTB_THREADS", "1")))
        out = r.stdout
        qfile = os.path.join(d, "charges")
        q = np.loadtxt(qfile).reshape(-1) if os.path.isfile(qfile) else np.full(n, np.nan)
    t = {}
    for k, pat in PAT.items():
        m = pat.search(out)
        t[k] = float(m.group(1)) if m else np.nan
    if not np.isfinite(t["total"]) or q.size != n or not np.all(np.isfinite(q)):
        raise RuntimeError("xtb parse fail (energies or charges)")
    t["gas"] = t["total"] - t["gsolv"]                      # non-solvation part of the solvated SPE
    t["eht"] = t["scc"] - (t["ies"] + t["aes"] + t["axc"] + t["disp_d4"] + t["gsolv"])
    val = np.full(n, np.nan)
    for m in WBO_ROW_RE.finditer(out):
        i = int(m.group(1))
        if 1 <= i <= n:
            val[i - 1] = float(m.group(2))
    if not np.all(np.isfinite(val)):
        raise RuntimeError("xtb parse fail (Wiberg table)")
    g = GAP_RE.search(out); dp = DIP_RE.search(out)
    t.update(charges=q, valence=val,
             gap=float(g.group(1)) if g else np.nan,
             mu=float(dp.group(1)) if dp else np.nan)
    return t


def d3_energy(syms, xyz):
    m = DispersionModel(np.array([fr.Z[s] for s in syms]), np.asarray(xyz, dtype=float) * BOHR)
    return float(m.get_dispersion(RationalDampingParam(method="b3lyp"), grad=False)["energy"])


def sasa_by_element(syms, xyz):
    s = SASA(syms, np.asarray(xyz, dtype=float))           # morfeus: atom_areas is a 1-indexed dict
    per = np.array([float(s.atom_areas[i + 1]) for i in range(len(syms))], dtype=float)
    out = {el: 0.0 for el in ELEMENTS}
    for sym, a in zip(syms, per):
        out[sym] = out.get(sym, 0.0) + float(a)
    return out


# ----------------------------------------------------------------------------- one reaction
def process_one(label):
    rid = int(label["rxn_id"])
    q1, q2 = int(label.get("charge1", 0) or 0), int(label.get("charge2", 0) or 0)
    row = {"rxn_id": rid, "charge1": q1, "charge2": q2, "role1": label.get("role1"), "role2": label.get("role2"),
           "n_atoms": label.get("n_atoms"), "n_f1": label.get("n_f1"), "n_f2": label.get("n_f2"),
           "label_status": label.get("status"), "alt_used": label.get("alt_used"),
           "flag_foreign_bond": bool(label.get("flag_foreign_bond", False)),
           "flag_async": bool(label.get("flag_async", False)),
           "flag_negative_strain": bool(label["d1_kcal"] < -0.5 or label["d2_kcal"] < -0.5),
           "xtb_solvation": f"alpb-{SOLVENT}"}
    # ---- DFT targets
    e_ab, f1d, f2d, f1r, f2r = (label[k] for k in ("e_ab_eh", "e_frag1_dist_eh", "e_frag2_dist_eh",
                                                   "e_frag1_rel_eh", "e_frag2_rel_eh"))
    row["dft_barrier_kcal"] = (e_ab - f1r - f2r) * EH2KCAL
    row["dft_eint_spe_kcal"] = (e_ab - f1d - f2d) * EH2KCAL
    row["dft_d1_kcal"], row["dft_d2_kcal"], row["dft_e_bond_kcal"] = label["d1_kcal"], label["d2_kcal"], label["e_bond_kcal"]
    for ch in ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"):
        row["dft_" + ch] = label[ch]

    # ---- geometries + partition
    pdir = PROF / str(rid)
    try:
        rel1, rel2, ts = (fr.read_xyz(pdir / label[k]) for k in ("rel1_file", "rel2_file", "ts_file"))
    except Exception as e:
        row["xtb_status"] = f"geom_load_fail:{type(e).__name__}"; return row
    ts_syms, ts_xyz = ts
    P = fr.partition(ts, rel1, rel2)
    if P.status != "ok":
        row["xtb_status"] = f"partition:{P.status}"; return row
    A_idx, B_idx = sorted(P.A), sorted(P.B)
    inv0 = {v: k for k, v in P.map0.items()}; inv1 = {v: k for k, v in P.map1.items()}
    posA = {a: k for k, a in enumerate(A_idx)}; posB = {b: k for k, b in enumerate(B_idx)}
    fA = ([ts_syms[i] for i in A_idx], ts_xyz[A_idx]); fB = ([ts_syms[i] for i in B_idx], ts_xyz[B_idx])

    # ---- reacting atoms (dip_a, dip_mid, dip_b | dph_a, dph_b); bond 1 = shorter forming bond
    pairs = [tuple(map(int, p.split("-"))) for p in str(label["formed_pairs_ts"]).split()]
    D_ts = fr.dist_matrix(ts_xyz)
    pairs.sort(key=lambda p: D_ts[p[0], p[1]])
    react = []
    for i, j in pairs:
        if (i in P.A) == (j in P.A):
            row["xtb_status"] = "forming_pair_not_across_partition"; return row
        react.append((i, j) if i in P.A else (j, i))
    (dip_a, dph_a), (dip_b, dph_b) = react
    adj, _ = fr.bond_matrix(ts_syms, ts_xyz)
    dip_mid = next((i for i in A_idx if ts_syms[i] != "H" and i not in (dip_a, dip_b)
                    and adj[i, dip_a] and adj[i, dip_b]), None)
    if dip_mid is None:
        mid = (ts_xyz[dip_a] + ts_xyz[dip_b]) / 2
        cands = [i for i in A_idx if ts_syms[i] != "H" and i not in (dip_a, dip_b)]
        if not cands:
            row["xtb_status"] = "no_middle_dipole_atom"; return row
        dip_mid = min(cands, key=lambda i: np.linalg.norm(ts_xyz[i] - mid))
    ts_dip, ts_dph = (dip_a, dip_mid, dip_b), (dph_a, dph_b)
    try:
        r1_dip = tuple(inv0[i] for i in ts_dip); r2_dph = tuple(inv1[i] for i in ts_dph)
    except KeyError:
        row["xtb_status"] = "reacting_atom_not_in_partition_map"; return row
    D1, D2 = fr.dist_matrix(rel1[1]), fr.dist_matrix(rel2[1])

    # ---- 11 distances
    row.update(dist_R_dip_ab=float(D1[r1_dip[0], r1_dip[1]]), dist_R_dip_bc=float(D1[r1_dip[1], r1_dip[2]]),
               dist_R_dip_ac=float(D1[r1_dip[0], r1_dip[2]]), dist_R_dph_ab=float(D2[r2_dph[0], r2_dph[1]]),
               dist_TS_dip_ab=float(D_ts[ts_dip[0], ts_dip[1]]), dist_TS_dip_bc=float(D_ts[ts_dip[1], ts_dip[2]]),
               dist_TS_dip_ac=float(D_ts[ts_dip[0], ts_dip[2]]), dist_TS_dph_ab=float(D_ts[ts_dph[0], ts_dph[1]]),
               dist_TS_form_ad=float(D_ts[dip_a, dph_a]), dist_TS_form_be=float(D_ts[dip_b, dph_b]),
               dist_TS_diag_ae=float(D_ts[dip_a, dph_b]))

    # ---- 5 xtb single points (ALPB): energies + charges + Wiberg valences + gap + dipole
    try:
        T = {"gs1": xtb_sp(*rel1, q1), "gs2": xtb_sp(*rel2, q2),
             "d1": xtb_sp(*fA, q1), "d2": xtb_sp(*fB, q2),
             "ts": xtb_sp(ts_syms, ts_xyz, q1 + q2)}
        p_gs1, p_gs2, p_d1, p_d2, p_ts = T["gs1"], T["gs2"], T["d1"], T["d2"], T["ts"]
    except Exception as e:
        row["xtb_status"] = f"spe_fail:{type(e).__name__}:{str(e)[:80]}"; return row
    dl = lambda k: (T["ts"][k] - T["d1"][k] - T["d2"][k]) * EH2KCAL      # noqa: E731

    # ---- 15 Mulliken + 15 Wiberg valence
    for k, i in enumerate(r1_dip):
        row[f"mulliken_R_dip_{k}"], row[f"wbo_valence_R_dip_{k}"] = float(p_gs1["charges"][i]), float(p_gs1["valence"][i])
    for k, i in enumerate(r2_dph):
        row[f"mulliken_R_dph_{k}"], row[f"wbo_valence_R_dph_{k}"] = float(p_gs2["charges"][i]), float(p_gs2["valence"][i])
    for k, i in enumerate(ts_dip):
        row[f"mulliken_TSdip_{k}"], row[f"wbo_valence_TSdip_{k}"] = float(p_d1["charges"][posA[i]]), float(p_d1["valence"][posA[i]])
    for k, i in enumerate(ts_dph):
        row[f"mulliken_TSdph_{k}"], row[f"wbo_valence_TSdph_{k}"] = float(p_d2["charges"][posB[i]]), float(p_d2["valence"][posB[i]])
    for k, i in enumerate(list(ts_dip) + list(ts_dph)):
        row[f"mulliken_TS_{k}"], row[f"wbo_valence_TS_{k}"] = float(p_ts["charges"][i]), float(p_ts["valence"][i])

    # ---- B_XTB5 (ALPB total energies)
    tot = {k: T[k]["total"] for k in T}
    row["xtb_e_barrier_kcal"] = (tot["ts"] - tot["gs1"] - tot["gs2"]) * EH2KCAL
    row["xtb_dist_dipole_kcal"] = (tot["d1"] - tot["gs1"]) * EH2KCAL
    row["xtb_dist_dipolarophile_kcal"] = (tot["d2"] - tot["gs2"]) * EH2KCAL
    row["xtb_sum_distortion_kcal"] = row["xtb_dist_dipole_kcal"] + row["xtb_dist_dipolarophile_kcal"]
    row["xtb_interaction_kcal"] = (tot["ts"] - tot["d1"] - tot["d2"]) * EH2KCAL
    for k in T:
        row[f"xtb_e_{k}_eh"] = tot[k]

    # ---- B_CH8 : one real calculated value per EDA channel
    row["b_strain_1"] = (T["d1"]["gas"] - T["gs1"]["gas"]) * EH2KCAL
    row["b_strain_2"] = (T["d2"]["gas"] - T["gs2"]["gas"]) * EH2KCAL
    qA, qB = p_d1["charges"], p_d2["charges"]                            # frozen isolated-fragment densities
    row["b_elst"] = float(COULOMB * sum(qA[posA[i]] * qB[posB[j]] / D_ts[i, j] for i in A_idx for j in B_idx))
    row["b_pauli"] = dl("rep")
    row["b_oi"] = dl("eht")
    row["b_disp"] = (d3_energy(ts_syms, ts_xyz) - d3_energy(*fA) - d3_energy(*fB)) * EH2KCAL
    row["b_cpcm"] = dl("gelec")
    row["b_cds"] = dl("gsasa") + dl("ghb") + dl("gshift")

    # ---- AUX18
    row["b_elst_scc"] = dl("ies") + dl("aes")
    row["b_disp_d4"] = dl("disp_d4")
    row["b_axc"] = dl("axc")
    row["b_ct"] = float(p_ts["charges"][A_idx].sum() - q1)               # inter-fragment charge transfer
    try:
        s_ts, s_a, s_b = sasa_by_element(ts_syms, ts_xyz), sasa_by_element(*fA), sasa_by_element(*fB)
    except Exception as e:
        row["xtb_status"] = f"sasa_fail:{type(e).__name__}:{str(e)[:60]}"; return row
    for el in ELEMENTS:
        row[f"dsasa_{el}"] = s_ts[el] - s_a[el] - s_b[el]

    # ---- bookkeeping extras (not features)
    row["gap_ts"], row["mu_ts"] = p_ts["gap"], p_ts["mu"]
    row["gap_dip"], row["gap_dph"] = p_d1["gap"], p_d2["gap"]
    row["mu_dip"], row["mu_dph"] = p_d1["mu"], p_d2["mu"]
    row["dmu_complexation"] = p_ts["mu"] - p_d1["mu"] - p_d2["mu"]
    row["xtb_status"] = "ok"
    return row


def main():
    slice_id, n_slices, out_path = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    labels = json.load(open(LABELS_PATH))
    accepted = sorted((d for d in labels if int(d["rxn_id"]) not in EXCLUDE), key=lambda d: d["rxn_id"])
    per = math.ceil(len(accepted) / n_slices)
    mine = accepted[slice_id * per: min((slice_id + 1) * per, len(accepted))]
    print(f"[slice {slice_id}/{n_slices}] accepted={len(accepted)} (labels {len(labels)} - exclude {len(EXCLUDE)}) "
          f"processing {len(mine)} rxns; xtb={XTB_BIN}", flush=True)
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
