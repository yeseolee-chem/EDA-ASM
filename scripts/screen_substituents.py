"""Substituent-level fragment screening for Hammett-style learning.

Reuses reports/fragment_screen/per_reaction_smiles.parquet (whole canonical
SMILES for each reaction's R-frame) and computes two complementary
substituent decompositions:

  1. R-group (scaffold/sidechain) decomposition
     scaffold = Bemis-Murcko; substituents = `Chem.ReplaceCore(mol, scaffold)`
     fragments. Each substituent's attachment-point isotope is normalized
     to 0 so the same R group at different scaffold positions collapses
     to one count (Hammett-relevant).

  2. BRICS leaves
     `BRICS.BRICSDecompose(mol, keepNonLeafNodes=False)`. Leaf SMILES
     carry RDKit attachment-type codes (e.g. `[14*]c1...`); we keep the
     codes since they encode the bond environment, but we also report a
     "stripped" variant where all `[N*]` are replaced by `[*]`.

For each variant we report unique fragments and how many appear >= N times
(default N=5).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
PER_RXN = REPO / "reports" / "fragment_screen" / "per_reaction_smiles.parquet"
OUT_DIR = REPO / "reports" / "fragment_screen"

_ISO_RE = re.compile(r"\[\d+\*\]")


def _strip_isotope(smi: str) -> str:
    """Replace every [N*] attachment dummy with [*] (position-agnostic)."""
    return _ISO_RE.sub("[*]", smi)


def _decompose_one(args):
    rxn_id, whole_smi = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem import BRICS
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    out = {"reaction_id": rxn_id, "rgroups": [], "rgroups_norm": [],
           "brics": [], "brics_norm": []}
    if not whole_smi:
        return out
    mol = Chem.MolFromSmiles(whole_smi)
    if mol is None:
        return out
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        return out

    # ---- R-group decomposition ----
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is not None and scaf.GetNumAtoms() > 0:
            side = Chem.ReplaceCore(mol, scaf)
            if side is not None:
                for f in Chem.GetMolFrags(side, asMols=True):
                    s = Chem.MolToSmiles(f)
                    out["rgroups"].append(s)
                    out["rgroups_norm"].append(_strip_isotope(s))
    except Exception:
        pass

    # ---- BRICS ----
    try:
        leaves = list(BRICS.BRICSDecompose(
            mol, returnMols=False, keepNonLeafNodes=False, minFragmentSize=1,
        ))
        for s in leaves:
            out["brics"].append(s)
            out["brics_norm"].append(_strip_isotope(s))
    except Exception:
        pass

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    df = pd.read_parquet(PER_RXN)
    df = df.dropna(subset=["whole_smiles"]).reset_index(drop=True)
    print(f"[main] {len(df)} reactions with whole SMILES", flush=True)

    inputs = list(zip(df["reaction_id"], df["whole_smiles"]))
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        results = pool.map(_decompose_one, inputs, chunksize=200)
    elapsed = time.time() - t0
    print(f"[main] decomposition done in {elapsed:.1f}s", flush=True)

    rgroup_counts: Counter[str] = Counter()
    rgroup_norm_counts: Counter[str] = Counter()
    brics_counts: Counter[str] = Counter()
    brics_norm_counts: Counter[str] = Counter()
    per_rxn_rows: list[dict] = []

    for r in results:
        for s in r["rgroups"]:
            rgroup_counts[s] += 1
        for s in r["rgroups_norm"]:
            rgroup_norm_counts[s] += 1
        for s in r["brics"]:
            brics_counts[s] += 1
        for s in r["brics_norm"]:
            brics_norm_counts[s] += 1
        per_rxn_rows.append({
            "reaction_id": r["reaction_id"],
            "n_rgroups": len(r["rgroups"]),
            "rgroups": ";".join(r["rgroups"]),
            "rgroups_norm": ";".join(r["rgroups_norm"]),
            "n_brics_leaves": len(r["brics"]),
            "brics": ";".join(r["brics"]),
            "brics_norm": ";".join(r["brics_norm"]),
        })

    pd.DataFrame(per_rxn_rows).to_parquet(
        args.out_dir / "per_reaction_substituents.parquet", index=False)

    def dump(counter: Counter[str], stem: str):
        d = (pd.DataFrame(counter.items(), columns=["smiles", "count"])
                .sort_values("count", ascending=False, ignore_index=True))
        d.to_parquet(args.out_dir / f"{stem}_counts.parquet", index=False)
        return d

    rg = dump(rgroup_counts, "rgroup_isotope")
    rgn = dump(rgroup_norm_counts, "rgroup_stripped")
    br = dump(brics_counts, "brics_isotope")
    brn = dump(brics_norm_counts, "brics_stripped")

    T = args.threshold

    def stats(d: pd.DataFrame, label: str):
        n_unique = len(d)
        n_thresh = int((d["count"] >= T).sum())
        coverage_sum = int(d.loc[d["count"] >= T, "count"].sum())
        total_occ = int(d["count"].sum())
        print(f"\n[{label}]")
        print(f"  unique fragments         : {n_unique}")
        print(f"  unique with count >= {T:<2} : {n_thresh}")
        print(f"  occurrences in those     : {coverage_sum} / {total_occ}")
        if n_unique:
            print("  top 15:")
            for smi, c in d.head(15).itertuples(index=False, name=None):
                print(f"    {c:>5}  {smi}")
        return {"label": label, "unique": n_unique,
                "unique_ge_threshold": n_thresh,
                "occurrences_covered": coverage_sum,
                "occurrences_total": total_occ}

    print("\n========== SUBSTITUENT SUMMARY ==========")
    print(f"reactions analyzed : {len(df)}")

    summary = {
        "threshold": T,
        "n_reactions": len(df),
        "elapsed_sec": round(elapsed, 1),
        "rgroup_isotope_labeled": stats(rg, "R-group (isotope-labeled)"),
        "rgroup_position_agnostic": stats(rgn, "R-group ([*] stripped)"),
        "brics_isotope_labeled": stats(br, "BRICS leaves (isotope-labeled)"),
        "brics_position_agnostic": stats(brn, "BRICS leaves ([*] stripped)"),
    }
    with open(args.out_dir / "substituent_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\noutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
