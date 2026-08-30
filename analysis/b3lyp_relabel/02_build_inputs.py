#!/usr/bin/env python3
"""SPEC15 Step 2 — build up to 5 ORCA inputs per rxn.

  eda.inp           complex EDA (existing rxns skip — reuse spec14 result)
  frag1_dist.inp    distorted fragment 1, BARE (no ghost atoms)
  frag2_dist.inp    distorted fragment 2, BARE
  frag1_rel.inp     relaxed fragment 1, at ds3 r*.xyz coords
  frag2_rel.inp     relaxed fragment 2, at ds3 r*.xyz coords

Critical: distorted fragment SPEs must be BARE. eda_frag*.out from the EDA
run include ghost atoms (BSSE correction) so they cannot be used as bare
fragment energies. That's why we generate separate frag*_dist.inp here.
"""
import os
import re
import collections
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")
GEOM = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")

COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
       "Cl": 1.02, "Br": 1.20, "S": 1.05, "I": 1.39}
FACTOR = 1.25

EDA_HEADER = (
    "! B3LYP D3BJ def2-TZVP CPCM(water) NoSym EDA TightSCF\n"
    "%maxcore 3500\n\n%cpcm\n  smd true\n  smdsolvent \"water\"\nend\n\n"
    "%eda\n"
    "  FRAG1 \"B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF\"\n"
    "  FRAG2 \"B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF\"\n"
    "  FRAG1_C 0\n  FRAG1_M 1\n  FRAG2_C 0\n  FRAG2_M 1\nend\n\n* xyz 0 1\n"
)
FRAG_HEADER = (
    "! B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF\n"
    "%maxcore 3500\n\n%cpcm\n  smd true\n  smdsolvent \"water\"\nend\n\n* xyz 0 1\n"
)


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    syms, xyz = [], []
    for l in lines[2:2 + n]:
        t = l.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


def read_ts(rid):
    d = PROF / str(int(rid))
    if not d.exists():
        return None
    cands = [f for f in os.listdir(d)
             if f.startswith("TS_") and f != "TS_imag_mode.xyz" and f.endswith(".xyz")]
    if not cands:
        return None
    fn = sorted(cands)[0]
    m = re.search(r"_(\d+)-(\d+)_(\d+)-(\d+)\.xyz$", fn)
    if not m:
        return None
    syms, xyz = read_xyz(d / fn)
    a1, a2, b1, b2 = map(int, m.groups())
    formed = [tuple(sorted((a1, a2))), tuple(sorted((b1, b2)))]
    return syms, xyz, formed, fn


def split_fragments(syms, xyz, formed):
    n = len(syms)
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    formed_set = set(formed)
    adj = collections.defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in formed_set:
                continue
            cutoff = FACTOR * (COV.get(syms[i], 0.8) + COV.get(syms[j], 0.8))
            if D[i, j] < cutoff:
                adj[i].append(j); adj[j].append(i)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v); comp.append(v); stack.extend(adj[v])
        comps.append(sorted(comp))
    return comps


def formula(syms):
    c = collections.Counter(syms)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def match_reactants(rid, frag_syms):
    """Match ds3 r*.xyz files to fragment compositions."""
    d = PROF / str(int(rid))
    rs = sorted(f for f in os.listdir(d)
                if re.match(r"^r\d+_", f) and f.endswith(".xyz") and "_alt" not in f)
    cand = {}
    for f in rs:
        s, x = read_xyz(d / f)
        cand[formula(s)] = (f, s, x)
    out = []
    for fs in frag_syms:
        key = formula(fs)
        if key not in cand:
            return None
        out.append(cand[key])
    if out[0][0] == out[1][0]:
        return None  # ambiguous (symmetric reaction, same file matches both)
    return out


def write_block(header, syms, xyz, labels=None):
    body = ""
    for i, s in enumerate(syms):
        tag = f"{s}({labels[i]})" if labels else s
        body += f"  {tag:<6}{xyz[i, 0]:16.8f}{xyz[i, 1]:14.8f}{xyz[i, 2]:14.8f}\n"
    return header + body + "*\n"


def main():
    sel = pd.read_csv(BASE / "artifacts" / "sample.csv")
    meta, fails = [], []
    for _, row in sel.iterrows():
        rid = int(row["rxn_id"])
        r = read_ts(rid)
        if r is None:
            fails.append((rid, "TS file/regex fail")); continue
        syms, xyz, formed, fn = r

        comps = split_fragments(syms, xyz, formed)
        if len(comps) != 2:
            fails.append((rid, f"fragment split = {len(comps)}")); continue
        f1_idx, f2_idx = comps
        frag_syms = [[syms[i] for i in f1_idx], [syms[i] for i in f2_idx]]

        rr = match_reactants(rid, frag_syms)
        if rr is None:
            fails.append((rid, "reactant matching failed")); continue

        out = BASE / "inputs" / f"rxn_{rid:04d}"
        out.mkdir(parents=True, exist_ok=True)

        # Complex EDA (skip if reusing spec14 result — 04_parse handles fallback)
        lab = [1 if i in set(f1_idx) else 2 for i in range(len(syms))]
        (out / "eda.inp").write_text(write_block(EDA_HEADER, syms, xyz, lab))

        # Distorted fragments (BARE)
        for k, idxs in enumerate([f1_idx, f2_idx], start=1):
            (out / f"frag{k}_dist.inp").write_text(
                write_block(FRAG_HEADER, [syms[i] for i in idxs], xyz[idxs]))

        # Relaxed fragments (ds3 r*.xyz coords)
        for k, (rfn, rsyms, rxyz) in enumerate(rr, start=1):
            (out / f"frag{k}_rel.inp").write_text(
                write_block(FRAG_HEADER, rsyms, rxyz))

        d1 = float(np.linalg.norm(xyz[formed[0][0]] - xyz[formed[0][1]]))
        d2 = float(np.linalg.norm(xyz[formed[1][0]] - xyz[formed[1][1]]))
        meta.append(dict(rxn_id=rid, ts_file=fn, n_atoms=len(syms),
                         n_f1=len(f1_idx), n_f2=len(f2_idx),
                         rel1_file=rr[0][0], rel2_file=rr[1][0],
                         formed_d1=d1, formed_d2=d2,
                         eda_exists=bool(row["eda_exists"])))

    pd.DataFrame(meta).to_csv(BASE / "artifacts" / "input_meta.csv", index=False)
    n_ok, n_tot = len(meta), len(sel)
    rate = n_ok / max(n_tot, 1)
    print(f"input build: {n_ok}/{n_tot} ({rate*100:.1f}%)")
    for rid, reason in fails[:10]:
        print(f"  fail rxn {rid}: {reason}")

    # eda.inp parity vs spec14 (first common rxn)
    parity = ""
    for _, row in sel.iterrows():
        if row["eda_exists"]:
            rid = int(row["rxn_id"])
            new = BASE / "inputs" / f"rxn_{rid:04d}" / "eda.inp"
            old = GEOM / "results_218" / f"rxn_{rid:04d}" / "eda.inp"
            if new.exists() and old.exists():
                new_h = new.read_text()[:new.read_text().find("* xyz 0 1\n")]
                old_h = old.read_text()[:old.read_text().find("* xyz 0 1\n")]
                parity = f"parity rxn_{rid:04d} header: {'IDENTICAL' if new_h == old_h else 'DIFFER'}"
                print(parity)
                break

    status = "PASS" if rate >= 0.98 else "FAIL"
    with open(BASE / "artifacts" / "GATE2_STATUS.txt", "w") as f:
        f.write(f"{status} built={n_ok}/{n_tot} rate={rate:.3f} {parity}\n")
        for rid, reason in fails:
            f.write(f"  rxn {rid}: {reason}\n")
    print(f"=== GATE-2 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
