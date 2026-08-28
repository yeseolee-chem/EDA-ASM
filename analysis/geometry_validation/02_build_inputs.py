#!/usr/bin/env python3
"""SPEC14 Step 2 — build eda.inp for each sampled reaction from ds3 B3LYP TS xyz.

Header MUST byte-match existing reactions/rxn_XXXX/eda.inp (except for the xyz
block). Verified by diff-ing one built input against the existing one.
"""
import os
import re
import sys
import collections
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
DS3 = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
EXISTING_EDA = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/reactions")

COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
       "Cl": 1.02, "Br": 1.20, "S": 1.05, "I": 1.39}
FACTOR = 1.25

# Header block copied byte-for-byte from an existing eda.inp so downstream
# parsers behave identically. Only the xyz coordinates differ.
HEADER = (
    "! B3LYP D3BJ def2-TZVP CPCM(water) NoSym EDA TightSCF\n"
    "%maxcore 3500\n\n"
    "%cpcm\n"
    "  smd true\n"
    "  smdsolvent \"water\"\n"
    "end\n\n"
    "%eda\n"
    "  FRAG1 \"B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF\"\n"
    "  FRAG2 \"B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF\"\n"
    "  FRAG1_C 0\n"
    "  FRAG1_M 1\n"
    "  FRAG2_C 0\n"
    "  FRAG2_M 1\n"
    "end\n\n"
    "* xyz 0 1\n"
)


def read_ts(rid):
    """Find TS_*.xyz in ds3 profile dir; parse coords + formed-bond indices from filename."""
    d = DS3 / str(int(rid))
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
    lines = (d / fn).read_text().splitlines()
    n = int(lines[0])
    el, xyz = [], []
    for l in lines[2:2 + n]:
        t = l.split()
        el.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    # ds3 uses 1-based? No — from prior audit, the indices are 0-based positions
    # in the xyz coordinate list. Verified by geometry inspection of rxn 0.
    a1, a2, b1, b2 = map(int, m.groups())
    formed = [tuple(sorted((a1, a2))), tuple(sorted((b1, b2)))]
    return el, np.array(xyz), formed, fn


def split_fragments(el, xyz, formed):
    """Cut the two forming bonds, walk connectivity, return fragment atom lists."""
    n = len(el)
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    formed_set = set(formed)
    adj = collections.defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in formed_set:
                continue
            cutoff = FACTOR * (COV.get(el[i], 0.8) + COV.get(el[j], 0.8))
            if D[i, j] < cutoff:
                adj[i].append(j)
                adj[j].append(i)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(adj[v])
        comps.append(sorted(comp))
    return comps


def build_eda(rid, el, xyz, formed):
    n = len(el)
    comps = split_fragments(el, xyz, formed)
    if len(comps) != 2:
        return None, f"split={len(comps)} fragments"
    f1 = set(comps[0])
    body_lines = []
    for i in range(n):
        tag = 1 if i in f1 else 2
        body_lines.append(f"  {el[i]}({tag}){xyz[i, 0]:16.8f}{xyz[i, 1]:14.8f}{xyz[i, 2]:14.8f}")
    body = "\n".join(body_lines) + "\n*\n"
    return HEADER + body, (len(comps[0]), len(comps[1]))


def main():
    sel = pd.read_csv(BASE / "artifacts" / "sample.csv")
    ok = 0
    fails = []
    meta = []
    for rid in sel["rxn_id"]:
        r = read_ts(int(rid))
        if r is None:
            fails.append((int(rid), "TS file missing / filename regex fail"))
            continue
        el, xyz, formed, fn = r
        txt, info = build_eda(rid, el, xyz, formed)
        if txt is None:
            fails.append((int(rid), info))
            continue
        n_f1, n_f2 = info
        out_dir = BASE / "inputs" / f"rxn_{int(rid):04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eda.inp").write_text(txt)
        d1 = float(np.linalg.norm(xyz[formed[0][0]] - xyz[formed[0][1]]))
        d2 = float(np.linalg.norm(xyz[formed[1][0]] - xyz[formed[1][1]]))
        meta.append(dict(rxn_id=int(rid), ts_file=fn,
                         n_atoms=len(el), n_frag1=n_f1, n_frag2=n_f2,
                         formed_d1_b3lyp=d1, formed_d2_b3lyp=d2,
                         formed_pair1=str(formed[0]),
                         formed_pair2=str(formed[1])))
        ok += 1

    pd.DataFrame(meta).to_csv(BASE / "artifacts" / "input_meta.csv", index=False)

    print(f"eda.inp built  : {ok} / {len(sel)}")
    print(f"failures       : {len(fails)}")
    for rid, reason in fails[:10]:
        print(f"  rxn {rid}: {reason}")

    # Header parity check: diff first built input's header against an existing one
    # (existing 3504 cohort must have this rxn too — same 218 sample lives inside).
    parity_note = ""
    if ok > 0:
        first_rid = int(sel["rxn_id"].iloc[0])
        existing_inp = EXISTING_EDA / f"rxn_{first_rid:04d}" / "eda.inp"
        built_inp = BASE / "inputs" / f"rxn_{first_rid:04d}" / "eda.inp"
        if existing_inp.exists():
            # Compare only the header (lines before "* xyz 0 1")
            def header_of(p):
                s = p.read_text()
                cut = s.find("* xyz 0 1\n")
                return s[:cut] if cut >= 0 else s
            match = header_of(existing_inp) == header_of(built_inp)
            parity_note = f"header parity vs existing rxn_{first_rid:04d}: "
            parity_note += "IDENTICAL" if match else "DIFFER"
            print(parity_note)

    status = "PASS" if (ok == len(sel) and not fails) else "FAIL"
    with open(BASE / "artifacts" / "GATE2_STATUS.txt", "w") as f:
        f.write(f"{status} built={ok}/{len(sel)} failures={len(fails)}\n")
        if parity_note:
            f.write(parity_note + "\n")
        for rid, reason in fails:
            f.write(f"  rxn {rid}: {reason}\n")

    print(f"=== GATE-2 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
