#!/usr/bin/env python3
"""audit_extra.py — reproduce the physical-consistency numbers from the builder metadata.

Reads  $EDA_BASE/build_strategy/graph/artifacts/input_meta.csv  (from build_inputs_graph.py --dry-run)
and the Coley profiles, and checks every number against EXPECTED. Exit code 1 if any gate fails.

Sections
  A. partition counts, flags, roles, charges, checksums
  B. imaginary mode (TS_imag_mode.xyz): fragment approach dominates; both forming bonds co-move
  C. strain reference: Coley G_act is defined against r*_alt.xyz whenever it exists
  D. the old 5-layer chain (build_strategy/build_inputs.py) run as-is, compared with the graph rule
  E. the old primary build (artifacts/build_failures.csv of 01_build_inputs.py) vs graph rule
"""
import collections
import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragmenter as fr  # noqa: E402

BASE = Path(os.environ["EDA_BASE"])
PROF = Path(os.environ["EDA_PROF"])
CSV = Path(os.environ["EDA_CSV"])
REPO = Path(os.environ.get("EDA_REPO", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"))  # EDA-ASM checkout

EXPECTED = dict(
    built=5265, rejected=[3865, 4069, 4727, 5930], foreign=[3090, 3766, 4252], n_async=231,
    charged=260, charge_dipolarophile={0: 5005, -2: 199, 1: 61}, alt_used=3037,
    frag1_from_r0=2235, md5_A_idx="28604d59295f3f3d86e280d0c6201fcb",
    md5_rel_charge="96e42a9272dda961b7e634a039c58b13",
    f1_shortest=5255, mode_same_direction=5263, mode_not_sync=[3312, 3926], intra_dominates=0,
    gact_alt_match=3037, gact_orig_match_noalt=2228, alt_abs_gt1=1868, alt_abs_gt3=1016, alt_higher=2103,
    chain=dict(primary=5182, factor_1_20=61, smiles_map=14, heavy_prefix=5, self_cyclo=4, FAIL=3),
    chain_fail=[3710, 3923, 4584], chain_partition_agree=5262,
    old_primary_fail=[3090, 3766, 4252, 4327, 4328, 4329, 4330], common_members=5258,
)
results = {}


def gate(name, got, exp):
    ok = (got == exp)
    results[name] = ok
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got {got}  expected {exp}")


def read_frames(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0]); step = n + 2
    return [np.array([[float(v) for v in l.split()[1:4]] for l in lines[k + 2:k + 2 + n]])
            for k in range(0, len(lines) - step + 1, step)]


# ----------------------------------------------------------------------------- A
print("A. builder metadata")
m = pd.read_csv(BASE / "build_strategy" / "graph" / "artifacts" / "input_meta.csv").sort_values("rxn_id")
f = pd.read_csv(BASE / "build_strategy" / "graph" / "artifacts" / "build_failures.csv")
gate("built", len(m), EXPECTED["built"])
gate("rejected", sorted(f.rxn_id.tolist()), EXPECTED["rejected"])
gate("foreign_bond ids", sorted(m[m.flag_foreign_bond].rxn_id.tolist()), EXPECTED["foreign"])
gate("async (d2>3.0)", int(m.flag_async.sum()), EXPECTED["n_async"])
gate("charged rxns", int(((m.charge1 != 0) | (m.charge2 != 0)).sum()), EXPECTED["charged"])
gate("dipolarophile charge", {int(k): int(v) for k, v in m.charge2.value_counts().items()}, EXPECTED["charge_dipolarophile"])
gate("dipole charge all 0", int((m.charge1 == 0).sum()), EXPECTED["built"])
gate("roles", (m.role1 == "dipole").sum() == len(m) and (m.role2 == "dipolarophile").sum() == len(m), True)
gate("alt_used", int(m.alt_used.sum()), EXPECTED["alt_used"])
gate("frag1 taken from r0", int(m.rel1_file.str.startswith("r0").sum()), EXPECTED["frag1_from_r0"])
gate("md5 of partitions", hashlib.md5("\n".join(f"{r.rxn_id}:{r.A_idx}" for r in m.itertuples()).encode()).hexdigest(), EXPECTED["md5_A_idx"])
gate("md5 of rel files+charges", hashlib.md5("\n".join(f"{r.rxn_id}:{r.rel1_file}:{r.rel2_file}:{r.charge1}:{r.charge2}" for r in m.itertuples()).encode()).hexdigest(), EXPECTED["md5_rel_charge"])

# ----------------------------------------------------------------------------- B + C
print("B. imaginary mode / C. strain reference")
full = pd.read_csv(CSV).set_index("rxn_id")
f1_shortest = same_dir = intra_dom = 0; not_sync = []
alt_match = orig_match_noalt = 0; alt_dE = []
for r in m.itertuples():
    d = PROF / str(r.rxn_id)
    ts_fn = sorted(x for x in os.listdir(d) if x.startswith("TS_") and x != "TS_imag_mode.xyz")[0]
    syms, xyz = fr.read_xyz(d / ts_fn)
    adj, D = fr.bond_matrix(syms, xyz)
    A = set(map(int, r.A_idx.split()))
    fp = sorted(({tuple(sorted(map(int, p.split("-")))) for p in r.formed_pairs_ts.split()}), key=lambda p: D[p])
    heavy = [i for i, s in enumerate(syms) if s != "H"]
    inter = [(i, j) for i in heavy for j in heavy if i < j and ((i in A) != (j in A))]
    if min(inter, key=lambda p: D[p]) == fp[0]:
        f1_shortest += 1
    F = np.array(read_frames(d / "TS_imag_mode.xyz"))
    k = int(np.argmax(np.linalg.norm((F - F[0]).reshape(len(F), -1), axis=1)))
    dd = lambda i, j: np.linalg.norm(F[k, i] - F[k, j]) - np.linalg.norm(F[0, i] - F[0, j])
    d1, d2 = dd(*fp[0]), dd(*fp[1])
    if np.sign(d1) == np.sign(d2):
        same_dir += 1
    else:
        not_sync.append(int(r.rxn_id))
    max_inter = max(abs(dd(i, j)) for i, j in inter if D[i, j] < 3.5)
    max_intra = max(abs(dd(i, j)) for i in heavy for j in heavy if i < j and ((i in A) == (j in A)) and adj[i, j])
    intra_dom += int(max_intra > max_inter)
    # G_act reference
    en = {l.split(",")[0].strip(): (float(l.split(",")[2]), float(l.split(",")[4]))
          for l in (d / "energies.csv").read_text().splitlines()[2:] if l.strip()}
    G = lambda key: en[key][1] + en[key][0]
    rk = sorted(x for x in en if re.match(r"^r\d+_", x) and not x.endswith("_alt"))
    tsk = [x for x in en if x.startswith("TS_")][0]
    has_alt = any(x + "_alt" in en for x in rk)
    ga_orig = (G(tsk) - sum(G(x) for x in rk)) * 627.5095
    ga_alt = (G(tsk) - sum(G(x + "_alt") if x + "_alt" in en else G(x) for x in rk)) * 627.5095
    if has_alt:
        alt_match += int(abs(full.loc[r.rxn_id, "G_act"] - ga_alt) < 0.01)
        for x in rk:
            if x + "_alt" in en:
                alt_dE.append((en[x + "_alt"][1] - en[x][1]) * 627.5095)
    else:
        orig_match_noalt += int(abs(full.loc[r.rxn_id, "G_act"] - ga_orig) < 0.01)
alt_dE = np.array(alt_dE)
gate("forming bond 1 is the shortest inter-fragment contact", f1_shortest, EXPECTED["f1_shortest"])
gate("both forming bonds move in the same direction", same_dir, EXPECTED["mode_same_direction"])
gate("  exceptions", sorted(not_sync), EXPECTED["mode_not_sync"])
gate("intra-fragment bond dominates the mode", intra_dom, EXPECTED["intra_dominates"])
gate("G_act reproduced with _alt reference (rxns with _alt)", alt_match, EXPECTED["gact_alt_match"])
gate("G_act reproduced with orig reference (rxns without _alt)", orig_match_noalt, EXPECTED["gact_orig_match_noalt"])
gate("|E_alt-E_orig| > 1 kcal/mol", int((np.abs(alt_dE) > 1).sum()), EXPECTED["alt_abs_gt1"])
gate("|E_alt-E_orig| > 3 kcal/mol", int((np.abs(alt_dE) > 3).sum()), EXPECTED["alt_abs_gt3"])
gate("E_alt higher than E_orig", int((alt_dE > 0).sum()), EXPECTED["alt_higher"])
print(f"     E_alt-E_orig range: {alt_dE.min():.2f} .. {alt_dE.max():.2f} kcal/mol (expected -9.92 .. 18.30)")

# ----------------------------------------------------------------------------- D
print("D. old 5-layer chain run as-is (build_strategy/build_inputs.py)")
chain_py = REPO / "analysis" / "b3lyp_full" / "build_strategy" / "build_inputs.py"
if chain_py.is_file():
    sys.argv = ["x"]
    spec = importlib.util.spec_from_file_location("chain", chain_py)
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
    ch.PROF = PROF
    smiles = dict(zip(full.index, full.rxn_smiles))
    counts = collections.Counter(); fails = []; agree = 0; rel_self = {}
    A_by_rid = {r.rxn_id: frozenset(map(int, r.A_idx.split())) for r in m.itertuples()}
    nat = {r.rxn_id: r.n_atoms for r in m.itertuples()}
    for rid in sorted(full.index):
        syms, xyz, _, _ = ch.read_ts(rid)
        rfiles = ch.list_reactant_files(rid)
        res = None
        for name, fn in ch.STRATEGIES:
            res, msg = (fn(syms, xyz, rfiles, smiles[rid]) if name == "smiles_map" else fn(syms, xyz, rfiles))
            if res is not None:
                counts[msg] += 1; break
        if res is None:
            counts["FAIL"] += 1; fails.append(rid); continue
        if rid in A_by_rid:
            f1, f2, rr = res
            agree += int({frozenset(f1), frozenset(f2)} == {A_by_rid[rid], frozenset(range(nat[rid])) - A_by_rid[rid]})
            if msg == "self_cyclo":
                rel_self[rid] = [x[0] for x in rr]
    gate("chain method counts", dict(counts), EXPECTED["chain"])
    gate("chain FAIL ids", sorted(fails), EXPECTED["chain_fail"])
    gate("chain partitions identical to graph rule (over graph-built rxns)", agree, EXPECTED["chain_partition_agree"])
    print(f"     self_cyclo reference files: {rel_self}   (both fragments -> the same r1 file = wrong strain reference)")
else:
    print("  (skipped: build_strategy/build_inputs.py not found at EDA_REPO)")

# ----------------------------------------------------------------------------- E
print("E. old primary build vs graph rule")
old_csv = BASE / "artifacts" / "build_failures.csv"          # written by 01_build_inputs.py (old primary)
if old_csv.is_file():
    gate("old primary failures (01_build_inputs.py artifacts)", sorted(pd.read_csv(old_csv).rxn_id.tolist()), EXPECTED["old_primary_fail"])
old = set(EXPECTED["old_primary_fail"])
graph_excl = set(EXPECTED["rejected"]) | set(EXPECTED["foreign"])
gate("only old primary excludes (false rejects)", sorted(old - graph_excl), [4327, 4328, 4329, 4330])
gate("only graph rule excludes (old primary false accepts)", sorted(graph_excl - old), [3865, 4069, 4727, 5930])
gate("members in common", 5269 - len(old | graph_excl), EXPECTED["common_members"])

bad = [k for k, v in results.items() if not v]
print("\nALL GATES PASSED" if not bad else f"\nFAILED GATES: {bad}")
sys.exit(1 if bad else 0)
