#!/usr/bin/env python3
"""build_inputs_graph.py — single-rule ORCA input builder for spec16rev b3lyp_full.

Replaces the 5-layer chain in build_strategy/build_inputs.py with ONE rule
(fragmenter.partition) plus explicit gates. Writes the same 5 files per rxn
(eda.inp, frag{1,2}_dist.inp, frag{1,2}_rel.inp) so 02_submit.sh / 03_parse.py
are unchanged.

What is different from the 5-layer builder
------------------------------------------
  * fragments come from graph embedding, not from a distance threshold alone
      -> isomeric reactant pairs get their own relaxed reference (4327-4330)
      -> late TSs with a forming bond shorter than 1.25*sum(r_cov) are fine
      -> TSs whose fragment connectivity != reactant connectivity are REJECTED
         (3865, 4069, 4727: sydnone N-O broken; 5930: intra-ylide O-C formed)
  * charge per fragment and total charge come from the reaction SMILES
      (260 rxns are charged: 199 x (-2), 61 x (+1)); even-electron gate per fragment
  * frag1 = dipole, frag2 = dipolarophile, always (role from the SMILES: the reactant that
      puts 3 atoms into the new 5-ring). File order r0/r1 is arbitrary (r0 = dipole in 42%),
      and the old primary build had frag1 = dipole in only 68% -> d1/d2 were mixed targets.
  * --alt-ref prefer : use r{k}_alt.xyz as the strain reference when it exists
      (this is the reference Coley's G_act is defined against, 3,037 rxns)
  * flags in input_meta.csv: foreign_bond (inter-fragment covalent contact that is not a
      SMILES forming bond -> TS of a different reaction: 3090, 3766, 4252),
      async (2nd forming bond > 3.0 A), d_form1/d_form2, n_maps

Usage
-----
  python build_inputs_graph.py [--dry-run] [--alt-ref never|prefer] [--out-dir DIR]
                               [--start I --end J] [--exclude-foreign]
"""
import argparse
import collections
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragmenter as fr  # noqa: E402

# ---------------------------------------------------------------- paths / headers (as in build_strategy)
BASE = Path(os.environ.get("EDA_BASE", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full"))
PROF = Path(os.environ.get("EDA_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))
CSV = Path(os.environ.get("EDA_CSV", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv"))

LEVEL = "B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF"
SOLV = '%cpcm\n  smd true\n  smdsolvent "water"\nend\n\n'


def eda_header(q1: int, q2: int) -> str:
    return (f"! {LEVEL.replace('TightSCF', 'EDA TightSCF')}\n"
            "%maxcore 3500\n\n%pal nprocs 5 end\n\n" + SOLV +
            "%eda\n"
            f'  FRAG1 "{LEVEL}"\n'
            f'  FRAG2 "{LEVEL}"\n'
            f"  FRAG1_C {q1}\n  FRAG1_M 1\n  FRAG2_C {q2}\n  FRAG2_M 1\nend\n\n"
            f"* xyz {q1 + q2} 1\n")


def frag_header(q: int) -> str:
    return f"! {LEVEL}\n%maxcore 3500\n\n" + SOLV + f"* xyz {q} 1\n"


# ---------------------------------------------------------------- I/O
def read_ts(rid):
    pdir = PROF / str(int(rid))
    cands = [f for f in os.listdir(pdir)
             if f.startswith("TS_") and f != "TS_imag_mode.xyz" and f.endswith(".xyz")]
    if not cands:
        return None
    fn = sorted(cands)[0]
    syms, xyz = fr.read_xyz(pdir / fn)
    return syms, xyz, fn


def reactant_files(rid):
    pdir = PROF / str(int(rid))
    rs = sorted(f for f in os.listdir(pdir)
                if re.match(r"^r\d+_", f) and f.endswith(".xyz") and "_alt" not in f)
    alts = {f: f[:-4] + "_alt.xyz" for f in rs if (pdir / (f[:-4] + "_alt.xyz")).is_file()}
    return rs, alts


def write_block(header, syms, xyz, labels=None):
    body = ""
    for i, s in enumerate(syms):
        tag = f"{s}({labels[i]})" if labels else s
        body += f"  {tag:<6}{xyz[i, 0]:16.8f}{xyz[i, 1]:14.8f}{xyz[i, 2]:14.8f}\n"
    return header + body + "*\n"


def emit(rdir, syms, xyz, A, B, rel, charges):
    rdir.mkdir(parents=True, exist_ok=True)
    q1, q2 = charges
    lab = [1 if i in A else 2 for i in range(len(syms))]
    (rdir / "eda.inp").write_text(write_block(eda_header(q1, q2), syms, xyz, lab))
    for k, (idx, q) in enumerate([(sorted(A), q1), (sorted(B), q2)], start=1):
        (rdir / f"frag{k}_dist.inp").write_text(
            write_block(frag_header(q), [syms[i] for i in idx], xyz[idx]))
    for k, ((rsyms, rxyz), q) in enumerate(zip(rel, (q1, q2)), start=1):
        (rdir / f"frag{k}_rel.inp").write_text(write_block(frag_header(q), rsyms, rxyz))


def already_built(rdir):
    return all((rdir / f).is_file() for f in
               ["eda.inp", "frag1_dist.inp", "frag2_dist.inp", "frag1_rel.inp", "frag2_rel.inp"])


# ---------------------------------------------------------------- one reaction
def build_one(rid, rxn_smiles, alt_ref: str, frag_order: str = "dipole-first"):
    """Returns (meta dict | None, failure reason | None)."""
    ts = read_ts(rid)
    if ts is None:
        return None, "no_ts"
    syms, xyz, ts_fn = ts
    rs, alts = reactant_files(rid)
    if len(rs) != 2:
        return None, f"reactant_count:{len(rs)}"
    pdir = PROF / str(int(rid))
    r0 = fr.read_xyz(pdir / rs[0]); r1 = fr.read_xyz(pdir / rs[1])

    # ---- the rule
    P = fr.partition((syms, xyz), r0, r1)
    if P.status != "ok":
        return None, f"partition:{P.status}"

    # ---- SMILES-side: charges + forming bonds
    rmols, formed, broken = fr.smiles_reactants_and_formed(rxn_smiles)
    assign = fr.match_smiles_to_reactants(rmols, r0, r1)
    if assign is None:
        return None, "smiles_reactant_mismatch"
    q1, q2 = fr.formal_charges(rmols, assign)

    # ---- roles: frag1 = dipole (3 ring atoms), frag2 = dipolarophile (2 ring atoms)
    dip = fr.dipole_smiles_index(rxn_smiles)
    if dip is None:
        return None, "no_3plus2_ring_in_product"
    r0_is_dipole = (assign[0] == dip[0])
    if frag_order == "dipole-first" and not r0_is_dipole:
        # swap everything that is indexed by fragment
        P = fr.Partition(P.status, P.B, P.A, P.map1, P.map0, P.diag)
        r0, r1 = r1, r0
        rs = [rs[1], rs[0]]
        assign = (assign[1], assign[0])
        q1, q2 = q2, q1
        r0_is_dipole = True
    role1, role2 = ("dipole", "dipolarophile") if r0_is_dipole else ("dipolarophile", "dipole")

    for k, (idx, q) in enumerate([(P.A, q1), (P.B, q2)], start=1):
        if fr.n_electrons([syms[i] for i in idx], q) % 2:
            return None, f"odd_electrons_frag{k}"
    audit = fr.audit_forming_bonds(P, (syms, xyz), r0, r1, rxn_smiles)
    if audit is None:
        return None, "forming_bond_map_failed"

    # ---- strain reference geometries (optionally the stereo-compatible _alt conformer)
    rel_files = []
    for f in rs:
        use = alts.get(f) if (alt_ref == "prefer" and f in alts) else f
        rel_files.append(use)
    rel = [fr.read_xyz(pdir / f) for f in rel_files]
    for k, (rf, r_orig) in enumerate(zip(rel, (r0, r1)), start=1):
        if collections.Counter(rf[0]) != collections.Counter(r_orig[0]):
            return None, f"alt_composition_mismatch_frag{k}"

    d1, d2 = audit.d_formed[0], audit.d_formed[-1]
    meta = dict(
        rxn_id=rid, ts_file=ts_fn, n_atoms=len(syms), n_f1=len(P.A), n_f2=len(P.B),
        rel1_file=rel_files[0], rel2_file=rel_files[1], role1=role1, role2=role2,
        charge1=q1, charge2=q2,
        formed_d1=d1, formed_d2=d2, n_formed=len(formed), n_broken=len(broken),
        n_foreign=len(audit.foreign),
        formed_pairs_ts=" ".join(f"{i}-{j}" for i, j in sorted(audit.formed_ts)),
        foreign_pairs=";".join(f"{syms[i]}{i}-{syms[j]}{j}:{d:.3f}" for i, j, d in audit.foreign),
        flag_foreign_bond=len(audit.foreign) > 0, flag_async=d2 > 3.0,
        alt_used=int(any(f.endswith("_alt.xyz") for f in rel_files)),
        n_maps=P.diag["n_maps"], recovery_method="graph",
        A_idx=" ".join(map(str, sorted(P.A))),
    )
    return (meta, (syms, xyz, P.A, P.B, rel, (q1, q2))), None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--alt-ref", choices=["never", "prefer"], default="prefer")
    ap.add_argument("--frag-order", choices=["dipole-first", "file"], default="dipole-first",
                    help="dipole-first: frag1/d1 = dipole, frag2/d2 = dipolarophile (default); file: frag1 = r0")
    ap.add_argument("--exclude-foreign", action="store_true",
                    help="do not write inputs for rxns flagged foreign_bond")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--meta-suffix", default="")
    args = ap.parse_args()

    csv_df = pd.read_csv(CSV)
    csv_df["rxn_id"] = csv_df["rxn_id"].astype(int)
    rxn_ids = sorted(csv_df["rxn_id"].tolist())[args.start:args.end]
    smiles = dict(zip(csv_df["rxn_id"], csv_df["rxn_smiles"]))
    out_root = Path(args.out_dir) if args.out_dir else BASE / "inputs"
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    metas, fails, counts = [], [], collections.Counter()
    for k, rid in enumerate(rxn_ids, 1):
        rdir = out_root / f"rxn_{rid:04d}"
        if not args.dry_run and already_built(rdir):
            counts["already_built"] += 1
            continue
        try:
            res, reason = build_one(rid, smiles.get(rid, ""), args.alt_ref, args.frag_order)
        except Exception as e:  # keep the batch alive, record the error
            res, reason = None, f"error:{type(e).__name__}:{str(e)[:60]}"
        if res is None:
            fails.append((rid, reason)); counts["fail"] += 1
            continue
        meta, payload = res
        if args.exclude_foreign and meta["flag_foreign_bond"]:
            fails.append((rid, "excluded:foreign_bond")); counts["excluded_foreign"] += 1
            continue
        if not args.dry_run:
            emit(rdir, *payload)
        metas.append(meta); counts["built"] += 1
        if k % 500 == 0:
            print(f"  {k}/{len(rxn_ids)} {dict(counts)}", file=sys.stderr, flush=True)

    art = BASE / "build_strategy" / "graph" / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metas).to_csv(art / f"input_meta{args.meta_suffix}.csv", index=False)
    pd.DataFrame(fails, columns=["rxn_id", "reason"]).to_csv(
        art / f"build_failures{args.meta_suffix}.csv", index=False)

    n_ok = counts["built"] + counts["already_built"]
    print(f"\nBUILD SUMMARY: {n_ok}/{len(rxn_ids)} = {100 * n_ok / max(1, len(rxn_ids)):.2f}%",
          file=sys.stderr)
    for kk, v in counts.most_common():
        print(f"  {kk}: {v}", file=sys.stderr)
    for rid, reason in fails:
        print(f"  fail rxn_{rid:04d}: {reason}", file=sys.stderr)
    if metas:
        m = pd.DataFrame(metas)
        print(f"  flagged foreign_bond: {int(m.flag_foreign_bond.sum())}  async(d2>3.0): {int(m.flag_async.sum())}"
              f"  charged: {int(((m.charge1 != 0) | (m.charge2 != 0)).sum())}  alt_used: {int(m.alt_used.sum())}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
