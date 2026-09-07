#!/usr/bin/env python3
"""Five-layer input builder for spec16rev b3lyp_full (5269 Coley rxns).

Consolidates the primary rule (from 01_build_inputs.py) + the four
recovery strategies (previously scattered across viz/recover_*.py)
into one self-contained script.

Strategy chain (first-winner-per-rxn):
  1. primary       — geom factor 1.25, strict reactant match      (5262 wins)
  2. self_cyclo    — geom factor 1.25, same-reactant allowed      (4)
  3. factor_1_20   — geom factor 1.20, strict match               (1)
  4. smiles_map    — SMILES atom-map k → TS atom index (k-1)      (1)
  5. heavy_prefix  — rA heavy sequence == TS heavy prefix; H by NN (1)

Reference outcome: 5269/5269 built. Non-primary rxns:
  self_cyclo   : 4327, 4328, 4329, 4330
  factor_1_20  : 4252
  smiles_map   : 3090
  heavy_prefix : 3766

The primary geom rule intentionally does NOT explicitly exclude the
TS filename's forming bonds — forming bonds at the TS are already
too long (≥ 1.25 × sum-of-covalent-radii) so they're excluded by
the distance cutoff anyway. Excluding them "correctly" (0-indexed)
breaks a few rxns whose filename annotations are misleading.

Idempotent: skips rxns whose 5 .inp files already exist.
"""
import argparse
import collections
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")
PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")

COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
       "Cl": 1.02, "Br": 1.20, "S": 1.05, "I": 1.39}
COV_FACTOR_PRIMARY = 1.25
COV_FACTOR_FALLBACK = 1.20

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


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    syms, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


def formula(syms):
    c = collections.Counter(syms)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def read_ts(rid):
    pdir = PROF / str(int(rid))
    if not pdir.is_dir():
        return None
    cands = [f for f in os.listdir(pdir)
             if f.startswith("TS_") and f != "TS_imag_mode.xyz"
             and f.endswith(".xyz")]
    if not cands:
        return None
    fn = sorted(cands)[0]
    m = re.search(r"_(\d+)-(\d+)_(\d+)-(\d+)\.xyz$", fn)
    if not m:
        return None
    syms, xyz = read_xyz(pdir / fn)
    a1, a2, b1, b2 = map(int, m.groups())
    # 1-indexed pairs (parsed from filename)
    formed_1idx = [tuple(sorted((a1, a2))), tuple(sorted((b1, b2)))]
    return syms, xyz, formed_1idx, fn


def list_reactant_files(rid):
    pdir = PROF / str(int(rid))
    if not pdir.is_dir():
        return []
    rs = sorted(f for f in os.listdir(pdir)
                if re.match(r"^r\d+_", f) and f.endswith(".xyz")
                and "_alt" not in f)
    return [(f, *read_xyz(pdir / f)) for f in rs]


# ---------------------------------------------------------------------
# Geometric fragmentation
# ---------------------------------------------------------------------

def split_fragments_geom(syms, xyz, factor):
    """Connected components using covalent-radii adjacency.
    Forming bonds are NOT explicitly excluded (matches reference primary)."""
    n = len(syms)
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    adj = collections.defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            r_i = COV.get(syms[i]); r_j = COV.get(syms[j])
            if r_i is None or r_j is None:
                continue
            if D[i, j] < factor * (r_i + r_j):
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


# ---------------------------------------------------------------------
# Reactant matching
# ---------------------------------------------------------------------

def match_reactants_strict(rfiles, frag_syms_list):
    """Match by formula. Dict-overwrite (matches reference primary):
    if two reactant files have same formula, last one wins.
    If both fragments matched the same file → None (self-cyclo reject)."""
    cand = {}
    for rf in rfiles:
        cand[formula(rf[1])] = rf   # dict overwrite (reference behavior)
    out = []
    for fs in frag_syms_list:
        key = formula(fs)
        if key not in cand:
            return None
        out.append(cand[key])
    if out[0][0] == out[1][0]:
        return None
    return out


def match_reactants_self_cyclo(rfiles, frag_syms_list):
    """Same as strict but allow both fragments to point at same file."""
    cand = {}
    for rf in rfiles:
        cand[formula(rf[1])] = rf
    out = []
    for fs in frag_syms_list:
        key = formula(fs)
        if key not in cand:
            return None
        out.append(cand[key])
    return out


# ---------------------------------------------------------------------
# Five strategies
# ---------------------------------------------------------------------

def try_primary(syms, xyz, rfiles):
    if [s for s in syms if s not in COV]:
        return None, "unknown_element"
    comps = split_fragments_geom(syms, xyz, COV_FACTOR_PRIMARY)
    if len(comps) != 2:
        return None, f"fragment_split:{len(comps)}"
    f1, f2 = comps
    frag_syms = [[syms[i] for i in f1], [syms[i] for i in f2]]
    rr = match_reactants_strict(rfiles, frag_syms)
    if rr is None:
        return None, "reactant_match"
    return (f1, f2, rr), "primary"


def try_self_cyclo(syms, xyz, rfiles):
    if [s for s in syms if s not in COV]:
        return None, "unknown_element"
    comps = split_fragments_geom(syms, xyz, COV_FACTOR_PRIMARY)
    if len(comps) != 2:
        return None, f"fragment_split:{len(comps)}"
    f1, f2 = comps
    frag_syms = [[syms[i] for i in f1], [syms[i] for i in f2]]
    rr = match_reactants_self_cyclo(rfiles, frag_syms)
    if rr is None:
        return None, "reactant_match"
    return (f1, f2, rr), "self_cyclo"


def try_factor_1_20(syms, xyz, rfiles):
    if [s for s in syms if s not in COV]:
        return None, "unknown_element"
    comps = split_fragments_geom(syms, xyz, COV_FACTOR_FALLBACK)
    if len(comps) != 2:
        return None, f"fragment_split:{len(comps)}"
    f1, f2 = comps
    frag_syms = [[syms[i] for i in f1], [syms[i] for i in f2]]
    rr = match_reactants_strict(rfiles, frag_syms)
    if rr is None:
        return None, "reactant_match"
    return (f1, f2, rr), "factor_1_20"


SMILES_ATOM_RE = re.compile(r"\[[^]]*:(\d+)\]")


def try_smiles_map(syms, xyz, rfiles, rxn_smiles):
    if not isinstance(rxn_smiles, str):
        return None, "no_smiles"
    reactants = rxn_smiles.split(">>")[0].split(".")
    if len(reactants) != 2:
        return None, f"smiles_n_reactants:{len(reactants)}"
    maps_per_r = [[int(m) for m in SMILES_ATOM_RE.findall(r)] for r in reactants]
    n_ts = len(syms)
    f1 = [m - 1 for m in maps_per_r[0] if 0 < m <= n_ts]
    f2 = [m - 1 for m in maps_per_r[1] if 0 < m <= n_ts]
    if len(set(f1) & set(f2)):
        return None, "smiles_overlap"
    n_heavy = sum(1 for s in syms if s != "H")
    if len(f1) + len(f2) != n_heavy:
        return None, "smiles_heavy_count_mismatch"
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    f1s, f2s = set(f1), set(f2)
    for i, s in enumerate(syms):
        if s != "H":
            continue
        heavy = [(j, D[i, j]) for j in range(n_ts) if syms[j] != "H"]
        heavy.sort(key=lambda t: t[1])
        (f1s if heavy[0][0] in f1s else f2s).add(i)
    f1i = sorted(f1s); f2i = sorted(f2s)
    frag_syms = [[syms[i] for i in f1i], [syms[i] for i in f2i]]
    rr = match_reactants_strict(rfiles, frag_syms)
    if rr is None:
        return None, "reactant_match"
    return (f1i, f2i, rr), "smiles_map"


def try_heavy_prefix(syms, xyz, rfiles):
    if len(rfiles) != 2:
        return None, f"reactant_count:{len(rfiles)}"
    n_ts = len(syms)
    ts_heavy_idx = [i for i, s in enumerate(syms) if s != "H"]
    ts_heavy_syms = [syms[i] for i in ts_heavy_idx]
    (rA_fn, rA_syms, rA_xyz), (rB_fn, rB_syms, rB_xyz) = rfiles
    rA_heavy = [s for s in rA_syms if s != "H"]
    rB_heavy = [s for s in rB_syms if s != "H"]
    if len(rA_heavy) + len(rB_heavy) != len(ts_heavy_syms):
        return None, "heavy_count_mismatch"
    if ts_heavy_syms[:len(rA_heavy)] == rA_heavy:
        f1_heavy = ts_heavy_idx[:len(rA_heavy)]
        f2_heavy = ts_heavy_idx[len(rA_heavy):]
        expected_rest = rB_heavy
        r_for_frag1 = (rA_fn, rA_syms, rA_xyz)
        r_for_frag2 = (rB_fn, rB_syms, rB_xyz)
    elif ts_heavy_syms[:len(rB_heavy)] == rB_heavy:
        f1_heavy = ts_heavy_idx[:len(rB_heavy)]
        f2_heavy = ts_heavy_idx[len(rB_heavy):]
        expected_rest = rA_heavy
        r_for_frag1 = (rB_fn, rB_syms, rB_xyz)
        r_for_frag2 = (rA_fn, rA_syms, rA_xyz)
    else:
        return None, "no_prefix_match"
    rest_actual = [syms[i] for i in f2_heavy]
    if collections.Counter(rest_actual) != collections.Counter(expected_rest):
        return None, "remainder_composition_mismatch"
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    f1s, f2s = set(f1_heavy), set(f2_heavy)
    for i, s in enumerate(syms):
        if s != "H":
            continue
        heavy = [(j, D[i, j]) for j in range(n_ts) if syms[j] != "H"]
        heavy.sort(key=lambda t: t[1])
        (f1s if heavy[0][0] in f1s else f2s).add(i)
    f1i = sorted(f1s); f2i = sorted(f2s)
    if len(f1i) != len(r_for_frag1[1]) or len(f2i) != len(r_for_frag2[1]):
        return None, "h_assignment_size_mismatch"
    return (f1i, f2i, [r_for_frag1, r_for_frag2]), "heavy_prefix"


# ---------------------------------------------------------------------
# Emit inputs
# ---------------------------------------------------------------------

def write_block(header, syms, xyz, labels=None):
    body = ""
    for i, s in enumerate(syms):
        tag = f"{s}({labels[i]})" if labels else s
        body += (f"  {tag:<6}"
                 f"{xyz[i, 0]:16.8f}"
                 f"{xyz[i, 1]:14.8f}"
                 f"{xyz[i, 2]:14.8f}\n")
    return header + body + "*\n"


def emit(rdir, syms, xyz, f1_idx, f2_idx, rr):
    rdir.mkdir(parents=True, exist_ok=True)
    lab = [1 if i in set(f1_idx) else 2 for i in range(len(syms))]
    (rdir / "eda.inp").write_text(write_block(EDA_HEADER, syms, xyz, lab))
    for k, idxs in enumerate([f1_idx, f2_idx], start=1):
        (rdir / f"frag{k}_dist.inp").write_text(
            write_block(FRAG_HEADER, [syms[i] for i in idxs], xyz[idxs]))
    for k, (rfn, rsyms, rxyz) in enumerate(rr, start=1):
        (rdir / f"frag{k}_rel.inp").write_text(
            write_block(FRAG_HEADER, rsyms, rxyz))


def rxn_already_built(rdir):
    return all((rdir / f).is_file()
               for f in ["eda.inp", "frag1_dist.inp", "frag2_dist.inp",
                         "frag1_rel.inp", "frag2_rel.inp"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

STRATEGIES = [
    ("primary",      try_primary),
    ("self_cyclo",   try_self_cyclo),
    ("factor_1_20",  try_factor_1_20),
    ("smiles_map",   try_smiles_map),
    ("heavy_prefix", try_heavy_prefix),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report method-per-rxn without writing .inp files.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write rxn_NNNN/ directories (default: BASE/inputs).")
    args = parser.parse_args()

    csv = pd.read_csv(CSV)
    csv["rxn_id"] = csv["rxn_id"].astype(int)
    rxn_ids = sorted(csv["rxn_id"].tolist())
    smiles_by_rxn = dict(zip(csv["rxn_id"], csv["rxn_smiles"]))
    print(f"Coley rxns: {len(rxn_ids)}", file=sys.stderr)

    out_root = Path(args.out_dir) if args.out_dir else BASE / "inputs"
    out_root.mkdir(parents=True, exist_ok=True)

    metas = []
    fails = []
    method_counts = collections.Counter()

    for rid in rxn_ids:
        rdir = out_root / f"rxn_{rid:04d}"
        if rxn_already_built(rdir) and not args.dry_run:
            method_counts["already_built"] += 1
            continue

        ts = read_ts(rid)
        if ts is None:
            fails.append((rid, "no_ts")); continue
        syms, xyz, formed_1idx, ts_fn = ts

        try:
            rfiles = list_reactant_files(rid)
        except FileNotFoundError:
            fails.append((rid, "no_profile_dir")); continue

        # Five-layer chain
        result = None; method = None; last_reason = None
        for name, fn in STRATEGIES:
            if name == "smiles_map":
                res, msg = fn(syms, xyz, rfiles, smiles_by_rxn.get(rid, ""))
            else:
                res, msg = fn(syms, xyz, rfiles)
            if res is not None:
                result = res; method = msg
                break
            last_reason = msg

        if result is None:
            fails.append((rid, f"all_layers_failed:{last_reason}"))
            continue

        f1_idx, f2_idx, rr = result
        if not args.dry_run:
            emit(rdir, syms, xyz, f1_idx, f2_idx, rr)

        d1 = float(np.linalg.norm(xyz[formed_1idx[0][0] - 1] - xyz[formed_1idx[0][1] - 1]))
        d2 = float(np.linalg.norm(xyz[formed_1idx[1][0] - 1] - xyz[formed_1idx[1][1] - 1]))
        metas.append(dict(
            rxn_id=rid, ts_file=ts_fn, n_atoms=len(syms),
            n_f1=len(f1_idx), n_f2=len(f2_idx),
            rel1_file=rr[0][0], rel2_file=rr[1][0],
            formed_d1=d1, formed_d2=d2,
            recovery_method=method,
        ))
        method_counts[method] += 1

    art_dir = BASE / "build_strategy" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metas).to_csv(art_dir / "input_meta.csv", index=False)
    pd.DataFrame(fails, columns=["rxn_id", "reason"]).to_csv(
        art_dir / "build_failures.csv", index=False)

    n_ok = sum(1 for _ in metas) + method_counts["already_built"]
    print(f"\nBUILD SUMMARY: {n_ok}/{len(rxn_ids)} = {n_ok/len(rxn_ids)*100:.2f}%",
          file=sys.stderr)
    for k, v in method_counts.most_common():
        print(f"  {k}: {v}", file=sys.stderr)
    for r, reason in fails:
        print(f"  fail rxn_{r:04d}: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
