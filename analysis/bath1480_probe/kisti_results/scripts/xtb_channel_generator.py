#!/usr/bin/env python3
"""
xtb_channel_generator.py — per-channel xTB counterparts (input A) for all 7 channels.

Route-1 decomposition from the xtb binary's term-wise printout (GFN2 + ALPB water),
supermolecular differences over (ts, di_dist, dp_dist); strain from relaxed fragments;
disp from the exact D3BJ oracle (2-body, functional parameters configurable).

Channel map (established route-1 assignment):
    elst_xtb   = D(isotropic ES + anisotropic ES)
    pauli_xtb  = D(repulsion + anisotropic XC)
    oi_xtb     = D(EHT)   where EHT = SCC - isoES - anisoES - anisoXC - dispD4 - Gsolv
    disp_xtb   = D3BJ oracle: E(ts) - E(di_dist) - E(dp_dist)   [reproduces the DFT label]
    cpcm_xtb   = D(Gelec)            [ALPB Born electrostatic]
    cds_xtb    = D(Gsasa + Ghb)
    strain_1_xtb = E(di_dist) - E(di_gs)     strain_2_xtb = E(dp_dist) - E(dp_gs)
Bookkeeping: dispd4_xtb = D(dispersion, GFN2-D4), gshift_xtb = D(Gshift),
    eint_total_xtb = D(total)  with hard gate  |eint_total - sum(terms)| < 1e-6 Eh.

Same manifest as xtb_feature_generator.py. All energies kcal/mol.

Usage:
    python xtb_channel_generator.py --manifest manifest.csv --out channels_xtb.parquet \
        [--xtb /path/to/xtb] [--solvent water] [--workers 8] \
        [--d3-params 1.0,1.9889,0.3981,4.4211]      # s6,s8,a1,a2 (B3LYP-D3BJ default)
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

H2K = 627.5094740631
ANG2BOHR = 1.8897259886
Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Cl':17,'Br':35,'I':53,'S':16,'P':15}

TERMS = {  # printout label -> key
    "total energy": "total", "SCC energy": "scc",
    "-> isotropic ES": "iso_es", "-> anisotropic ES": "aniso_es",
    "-> anisotropic XC": "aniso_xc", "-> dispersion": "disp_d4",
    "-> Gsolv": "gsolv", "-> Gelec": "gelec", "-> Gsasa": "gsasa",
    "-> Ghb": "ghb", "-> Gshift": "gshift", "repulsion energy": "rep",
}
_LINE = re.compile(r"::\s+(" + "|".join(re.escape(k) for k in TERMS) + r")\s+(-?\d+\.\d+)\s+Eh")


def read_xyz(p):
    L = Path(p).read_text().splitlines(); n = int(L[0].split()[0])
    num, xyz = [], []
    for ln in L[2:2+n]:
        t = ln.split(); num.append(Z[t[0]]); xyz.append([float(x) for x in t[1:4]])
    return np.array(num), np.array(xyz)


def xtb_terms(xyz_path, charge, xtb_bin, solvent):
    with tempfile.TemporaryDirectory() as td:
        cmd = [xtb_bin, str(Path(xyz_path).resolve()), "--gfn", "2", "--sp",
               "--chrg", str(int(charge))]
        if solvent and solvent != "none":
            cmd += ["--alpb", solvent]
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=600)
    vals = {}
    for m in _LINE.finditer(r.stdout):
        vals[TERMS[m.group(1)]] = float(m.group(2))   # last occurrence wins
    need = {"total", "scc", "iso_es", "aniso_es", "aniso_xc", "disp_d4", "rep"}
    if not need <= vals.keys():
        raise RuntimeError(f"xtb parse missing {need - vals.keys()} for {xyz_path}")
    for k in ("gsolv", "gelec", "gsasa", "ghb", "gshift"):
        vals.setdefault(k, 0.0)
    vals["eht"] = (vals["scc"] - vals["iso_es"] - vals["aniso_es"]
                   - vals["aniso_xc"] - vals["disp_d4"] - vals["gsolv"])
    return vals


def d3bj(xyz_path, params):
    from dftd3.interface import RationalDampingParam, DispersionModel
    num, xyz = read_xyz(xyz_path)
    s6, s8, a1, a2 = params
    m = DispersionModel(num, xyz * ANG2BOHR)
    return m.get_dispersion(
        RationalDampingParam(s6=s6, s8=s8, a1=a1, a2=a2, s9=0.0), grad=False)["energy"]


def one_reaction(args):
    row, xtb_bin, solvent, d3p = args
    out = {"reaction_number": int(row["reaction_number"]), "ok": True, "error": ""}
    try:
        qdi = float(row.get("charge_di", 0) or 0); qdp = float(row.get("charge_dp", 0) or 0)
        T = {tag: xtb_terms(row[f"{tag}_xyz"], q, xtb_bin, solvent)
             for tag, q in (("di_gs", qdi), ("dp_gs", qdp), ("di_dist", qdi),
                            ("dp_dist", qdp), ("ts", qdi + qdp))}
        D = lambda k: (T["ts"][k] - T["di_dist"][k] - T["dp_dist"][k]) * H2K

        out["elst_xtb"]   = D("iso_es") + D("aniso_es")
        out["pauli_xtb"]  = D("rep") + D("aniso_xc")
        out["oi_xtb"]     = D("eht")
        out["cpcm_xtb"]   = D("gelec")
        out["cds_xtb"]    = D("gsasa") + D("ghb")
        out["dispd4_xtb"] = D("disp_d4")
        out["gshift_xtb"] = D("gshift")
        out["eint_total_xtb"] = D("total")
        out["strain_1_xtb"] = (T["di_dist"]["total"] - T["di_gs"]["total"]) * H2K
        out["strain_2_xtb"] = (T["dp_dist"]["total"] - T["dp_gs"]["total"]) * H2K

        # telescoping hard gate (construction-exact up to print precision)
        resid = out["eint_total_xtb"] - (out["elst_xtb"] + out["pauli_xtb"] + out["oi_xtb"]
                 + out["cpcm_xtb"] + out["cds_xtb"] + out["dispd4_xtb"] + out["gshift_xtb"])
        out["telescope_residual_kcal"] = resid
        if abs(resid) > 1e-6 * H2K:
            out["ok"] = False; out["error"] = f"telescoping residual {resid:.2e} kcal"

        # D3BJ oracle (the disp channel counterpart == the DFT label itself)
        out["disp_xtb"] = (d3bj(row["ts_xyz"], d3p) - d3bj(row["di_dist_xyz"], d3p)
                           - d3bj(row["dp_dist_xyz"], d3p)) * H2K
    except Exception as exc:
        out["ok"] = False; out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="channels_xtb.parquet")
    ap.add_argument("--xtb", default="xtb")
    ap.add_argument("--solvent", default="water")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--d3-params", default="1.0,1.9889,0.3981,4.4211")
    a = ap.parse_args()
    d3p = tuple(float(x) for x in a.d3_params.split(","))

    man = pd.read_csv(a.manifest)
    tasks = [(row, a.xtb, a.solvent, d3p) for _, row in man.iterrows()]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one_reaction, tasks))
    else:
        rows = [one_reaction(t) for t in tasks]

    ok = [r for r in rows if r["ok"]]; bad = [r for r in rows if not r["ok"]]
    df = pd.DataFrame(ok).drop(columns=["ok", "error"])
    df.to_parquet(a.out, index=False)
    print(f"[ok] {a.out}: {len(df)} rows | FAIL {len(bad)}")
    if len(df):
        print(f"     max |telescope residual| = {df.telescope_residual_kcal.abs().max():.2e} kcal/mol")
    for b in bad[:10]:
        print(f"   FAIL rxn {b['reaction_number']}: {b['error']}")
    if bad:
        pd.DataFrame(bad).to_csv(Path(a.out).with_suffix(".failures.csv"), index=False)


if __name__ == "__main__":
    main()
