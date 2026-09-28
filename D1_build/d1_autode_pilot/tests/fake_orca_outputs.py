#!/usr/bin/env python3
"""LOCAL TEST ONLY: write minimal ORCA-like outputs for a replay job from labels_all numbers,
so assemble.py's glue (parser call, audit view, gates, promotion) can be exercised without ORCA.
Formats only mimic the regexes used by stage3_parse / smd_relabel_audit; the real formats are
validated on the cluster by the D0_replay jobs (they must reproduce labels_all)."""
import json, re, sys
from pathlib import Path
jd, rid, labels = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
L = {r["rxn_id"]: r for r in json.load(open(labels))}[rid]
sp = jd / "sp"
HDR = "Program Version 6.1.1  -   RELEASE  -\nSolvent:   WATER\n Epsilon ...  78.3550\nutilizes the SMD solvation module\n"
TAIL = "SMD CDS free energy correction energy :  -0.001 Eh\nFINAL SINGLE POINT ENERGY {e:.10f}\n****ORCA TERMINATED NORMALLY****\n"
E = dict(eda=L["e_ab_eh"], eda_frag1=L["e_frag1_dist_ghost_eh"], eda_frag2=L["e_frag2_dist_ghost_eh"],
         frag1_dist=L["e_frag1_dist_eh"], frag2_dist=L["e_frag2_dist_eh"], frag1_rel=L["e_frag1_rel_eh"],
         frag2_rel=L["e_frag2_rel_eh"])
raw = L["eda_raw_eh"]
names = [("Bond Energy", "bond"), ("Orbital Energy", "orb"), ("Electrostatic Energy", "elst"), ("Pauli Energy", "pauli"),
         ("Delta E^0(XC)", "xc"), ("Delta Dispersion", "disp"), ("Delta CPCM Dielectric", "cpcm"),
         ("Delta SMD CDS correction", "smd_cds")]
tab = "Energy Term                Hartree      Kcal/mol\n-----------------------------------------\n" + \
      "\n".join(f"  {n:<26}{raw[k]:14.8f}{raw[k]*627.5094740631:14.5f}" for n, k in names) + "\n-----------------------------------------\n"
for k, e in E.items():
    (sp / f"{k}.out").write_text(HDR + (tab if k == "eda" else "") + TAIL.format(e=e))
# ORCA-generated fragment inputs: real atoms of fragment f, ghosts of the other
eda = (sp / "eda.inp").read_text()
atoms = re.findall(r"^\s*([A-Z][a-z]?)\((\d)\)\s+(\S+)\s+(\S+)\s+(\S+)", eda, re.M)
for f, g in (("1", "2"), ("2", "1")):
    q = re.search(rf"FRAG{f}_C\s+(-?\d+)", eda).group(1)
    lines = [f"  {s}({lab}) {x} {y} {z}" if lab == f else f"  {s}:({lab}) {x} {y} {z}" for s, lab, x, y, z in atoms]
    (sp / f"eda_frag{f}.inp").write_text(f"! B3LYP D3BJ def2-TZVP SMD(water) NoSym TightSCF\n* xyz {q} 1\n" + "\n".join(lines) + "\n*\n")
print("fake outputs written for", jd)
