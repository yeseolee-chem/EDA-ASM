#!/usr/bin/env python3
"""SPEC16rev Step 1 — build 5 ORCA inputs per rxn (Coley-only source).

Same logic as spec15's 02_build_inputs.py (validated 218/218). Extends to
full 5269 Coley set. Reactant coordinates taken DIRECTLY from Coley r*.xyz
(Option A — no re-optimization, per SPEC §2.2).

GATE-1: build success rate ≥ 97%.
"""
import os
import re
import collections
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")
PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")

COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
       "Cl": 1.02, "Br": 1.20, "S": 1.05, "I": 1.39}
FACTOR = 1.25

EDA_HEADER = (
    "! B3LYP D3BJ def2-TZVP CPCM(water) NoSym EDA TightSCF\n"
    "%maxcore 3500\n\n"
    "%pal nprocs 5 end\n\n"
    "%cpcm\n  smd true\n  smdsolvent \"water\"\nend\n\n"
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
            r_i = COV.get(syms[i])
            r_j = COV.get(syms[j])
            if r_i is None or r_j is None:
                continue
            if D[i, j] < FACTOR * (r_i + r_j):
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
        return None  # ambiguous (symmetric reaction)
    return out


def write_block(header, syms, xyz, labels=None):
    body = ""
    for i, s in enumerate(syms):
        tag = f"{s}({labels[i]})" if labels else s
        body += f"  {tag:<6}{xyz[i, 0]:16.8f}{xyz[i, 1]:14.8f}{xyz[i, 2]:14.8f}\n"
    return header + body + "*\n"


def main():
    csv = pd.read_csv(CSV)
    csv["rxn_id"] = csv["rxn_id"].astype(int)
    rxn_ids = sorted(csv["rxn_id"].tolist())
    print(f"Coley reactions: {len(rxn_ids)}")

    meta = []
    fails = []
    fail_types = collections.Counter()

    for rid in rxn_ids:
        r = read_ts(rid)
        if r is None:
            fails.append((rid, "ts_file"))
            fail_types["ts_file"] += 1
            continue
        syms, xyz, formed, fn = r

        # Unknown element check
        unknown = [s for s in syms if s not in COV]
        if unknown:
            fails.append((rid, f"unknown_element:{unknown[0]}"))
            fail_types["unknown_element"] += 1
            continue

        comps = split_fragments(syms, xyz, formed)
        if len(comps) != 2:
            fails.append((rid, f"fragment_split:{len(comps)}"))
            fail_types["fragment_split"] += 1
            continue
        f1_idx, f2_idx = comps
        frag_syms = [[syms[i] for i in f1_idx], [syms[i] for i in f2_idx]]

        rr = match_reactants(rid, frag_syms)
        if rr is None:
            fails.append((rid, "reactant_match"))
            fail_types["reactant_match"] += 1
            continue

        out = BASE / "inputs" / f"rxn_{rid:04d}"
        out.mkdir(parents=True, exist_ok=True)
        lab = [1 if i in set(f1_idx) else 2 for i in range(len(syms))]
        (out / "eda.inp").write_text(write_block(EDA_HEADER, syms, xyz, lab))
        for k, idxs in enumerate([f1_idx, f2_idx], start=1):
            (out / f"frag{k}_dist.inp").write_text(
                write_block(FRAG_HEADER, [syms[i] for i in idxs], xyz[idxs]))
        for k, (rfn, rsyms, rxyz) in enumerate(rr, start=1):
            (out / f"frag{k}_rel.inp").write_text(
                write_block(FRAG_HEADER, rsyms, rxyz))

        d1 = float(np.linalg.norm(xyz[formed[0][0]] - xyz[formed[0][1]]))
        d2 = float(np.linalg.norm(xyz[formed[1][0]] - xyz[formed[1][1]]))
        meta.append(dict(rxn_id=rid, ts_file=fn, n_atoms=len(syms),
                         n_f1=len(f1_idx), n_f2=len(f2_idx),
                         rel1_file=rr[0][0], rel2_file=rr[1][0],
                         formed_d1=d1, formed_d2=d2))

    pd.DataFrame(meta).to_csv(BASE / "artifacts" / "input_meta.csv", index=False)
    pd.DataFrame(fails, columns=["rxn_id", "reason"]).to_csv(
        BASE / "artifacts" / "build_failures.csv", index=False)

    n_ok = len(meta)
    n_tot = len(rxn_ids)
    rate = n_ok / n_tot
    print(f"\nbuild success: {n_ok}/{n_tot} ({rate*100:.2f}%)")
    print("failure breakdown:")
    for k, v in fail_types.most_common():
        print(f"  {k}: {v}")

    status = "PASS" if rate >= 0.97 else "FAIL"
    with open(BASE / "artifacts" / "GATE1_STATUS.txt", "w") as f:
        f.write(f"{status} built={n_ok}/{n_tot} rate={rate:.4f}\n")
        for k, v in fail_types.most_common():
            f.write(f"  {k}: {v}\n")

    print(f"=== GATE-1 {status} ===")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
