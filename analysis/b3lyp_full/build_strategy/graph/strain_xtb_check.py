#!/usr/bin/env python3
"""strain_xtb_check.py (optional) — GFN2-xTB estimate of the strain labels at stake.

Shows what the OLD rules would have written as fragment strain for the reactions where
they disagree with the graph rule. Gas-phase GFN2-xTB single points via tblite; numbers
are magnitude estimates (expect agreement to ~0.1 kcal/mol across tblite versions).

Expected (kcal/mol):
  3865 / 4069 / 4727  dipole strain  79.0 / 90.2 / 92.2     (ring-opened sydnone vs intact reference)
  5930                dipole strain  23.9                   (looks normal -> undetectable by outlier filters)
  4327  r0 fragment: correct 15.8 ; self_cyclo reference 42.8 ; E(r1)-E(r0) = -27.0
  baseline (300 random neutral rxns, 600 fragments): median 11.2, max 38.7
"""
import collections
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tblite.interface import Calculator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragmenter as fr  # noqa: E402

PROF = Path(os.environ["EDA_PROF"])
META = Path(os.environ["EDA_BASE"]) / "build_strategy" / "graph" / "artifacts" / "input_meta.csv"
BOHR = 1.8897259886


def E(syms, xyz, charge=0):
    c = Calculator("GFN2-xTB", np.array([fr.Z[s] for s in syms]), np.asarray(xyz) * BOHR, charge=charge)
    c.set("verbosity", 0)
    return c.singlepoint().get("energy") * 627.5095


def load(rid):
    d = PROF / str(rid)
    ts_fn = sorted(f for f in os.listdir(d) if f.startswith("TS_") and f != "TS_imag_mode.xyz")[0]
    rfs = sorted(f for f in os.listdir(d) if re.match(r"^r\d+_", f) and "_alt" not in f)
    return fr.read_xyz(d / ts_fn), {f: fr.read_xyz(d / f) for f in rfs}


print("== old primary accepts / graph rule rejects (strain vs Coley r-file, kcal/mol)")
for rid in [3865, 4069, 4727, 5930]:
    (syms, xyz), R = load(rid)
    for comp in fr.geometric_components(syms, xyz):          # what the primary rule would split
        idx = sorted(comp); fsy = [syms[i] for i in idx]
        rf = [k for k, (rs, _) in R.items() if collections.Counter(rs) == collections.Counter(fsy)][0]
        print(f"  {rid} {rf:>22s}  strain = {E(fsy, xyz[idx]) - E(*R[rf]):6.1f}")

print("== isomer pair 4327: r0 fragment strain with correct (r0) vs self_cyclo (r1) reference")
m = pd.read_csv(META).set_index("rxn_id")
(syms, xyz), R = load(4327)
A = sorted(map(int, m.loc[4327, "A_idx"].split()))
eA = E([syms[i] for i in A], xyz[A]); e0 = E(*R["r0_C7H10N2O.xyz"]); e1 = E(*R["r1_C7H10N2O.xyz"])
print(f"  correct {eA - e0:.1f}   self_cyclo {eA - e1:.1f}   E(r1)-E(r0) = {e1 - e0:.1f}")

print("== baseline: 300 random neutral rxns")
rng = np.random.default_rng(0)
ok = m[(m.charge1 == 0) & (m.charge2 == 0) & (~m.flag_foreign_bond)]
vals = []
for rid in rng.choice(ok.index.values, 300, replace=False):
    (syms, xyz), R = load(int(rid)); row = ok.loc[rid]
    A = sorted(map(int, row.A_idx.split())); B = sorted(set(range(len(syms))) - set(A))
    ra = row.rel1_file.replace("_alt", ""); rb = row.rel2_file.replace("_alt", "")
    vals += [E([syms[i] for i in A], xyz[A]) - E(*R[ra]), E([syms[i] for i in B], xyz[B]) - E(*R[rb])]
v = np.array(vals)
print("  quantiles:", {q: round(float(np.quantile(v, q)), 1) for q in [0, .05, .5, .95, .99, 1]})
