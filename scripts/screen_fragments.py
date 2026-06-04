"""Hammett-style fragment screening over all Halo8 reactions.

For each reaction we pull the R (frame 0) geometry from its source .db,
let RDKit's `rdDetermineBonds` perceive bond orders, then extract:

    - whole_canonical:   canonical SMILES of the reactant molecule
    - murcko_scaffold:   Bemis-Murcko scaffold canonical SMILES
    - generic_scaffold:  generic (atom-blind) Murcko scaffold

We aggregate counts of each across reactions and report how many distinct
fragments appear >= N times (default N=5).

10 source dbs are processed in parallel (one worker per db).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import time
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
DB_DIR = REPO / "data" / "Halo8"
INDEX_PARQUET = REPO / "data" / "halo8_index" / "index.parquet"
OUT_DIR = REPO / "reports" / "fragment_screen"

# Suppress rdkit warnings (lots of valence noise for radicals etc.)
warnings.filterwarnings("ignore")
os.environ.setdefault("RDKIT_DISABLE_DEPRECATION_WARNINGS", "1")


def _z_to_sym(z: int) -> str:
    from ase.data import chemical_symbols
    return chemical_symbols[int(z)]


def _xyz_block(numbers: np.ndarray, positions: np.ndarray) -> str:
    lines = [str(len(numbers)), ""]
    for z, p in zip(numbers, positions):
        lines.append(f"{_z_to_sym(int(z))} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    return "\n".join(lines)


def _decompose(numbers: np.ndarray, positions: np.ndarray, charge: int):
    """Return (whole_smi, murcko_smi, generic_smi) or (None, None, None)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromXYZBlock(_xyz_block(numbers, positions))
        if mol is None:
            return (None, None, None)
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception:
        return (None, None, None)

    try:
        whole = Chem.MolToSmiles(mol)
    except Exception:
        whole = None

    murcko = None
    generic = None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is not None and scaf.GetNumAtoms() > 0:
            murcko = Chem.MolToSmiles(scaf)
            gen = MurckoScaffold.MakeScaffoldGeneric(scaf)
            generic = Chem.MolToSmiles(gen)
    except Exception:
        pass

    return (whole, murcko, generic)


def _process_db(args):
    db_path, charge_map = args
    """Stream R-frames (frame index 0) from one db, build fragments."""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Filter to rows whose dand_id ends with "_0" (literal underscore).
    # Pattern key bytes: ..._0","old_dand_id"... — underscores in LIKE are
    # escaped with backslash via ESCAPE '\'.
    sql = (
        "SELECT id, CAST(substr(data, 9, 200) AS TEXT) AS js, "
        "       numbers, positions, charge, natoms "
        "FROM systems "
        r"WHERE CAST(substr(data, 9, 200) AS TEXT) "
        r"      LIKE '%\_0\",\"old\_dand\_id%' ESCAPE '\'"
    )
    cur.execute(sql)

    whole_counts: Counter[str] = Counter()
    murcko_counts: Counter[str] = Counter()
    generic_counts: Counter[str] = Counter()

    per_reaction = []  # list of dicts (also dumped to per-db parquet)
    n_seen = 0
    n_failed = 0
    n_missing = 0

    while True:
        rows = cur.fetchmany(500)
        if not rows:
            break
        for rid, js, nums_blob, pos_blob, charge, natoms in rows:
            # Extract dand_id from the prefix string.
            # js looks like: {"dand_id":"<id>_0","old_dand_id":...
            try:
                # Cheap parse: find the value between the first pair of quotes
                # after "dand_id":"
                start = js.index('"dand_id":"') + len('"dand_id":"')
                end = js.index('"', start)
                dand_id = js[start:end]
            except ValueError:
                continue
            traj_id = dand_id.rsplit("_", 1)[0]
            chg = int(round(float(charge_map.get(traj_id, charge or 0.0))))
            n_seen += 1

            nums = np.frombuffer(nums_blob, dtype=np.int32)
            pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(-1, 3)

            whole, murcko, generic = _decompose(nums, pos, chg)
            if whole is None:
                n_failed += 1
            if whole:
                whole_counts[whole] += 1
            if murcko:
                murcko_counts[murcko] += 1
            if generic:
                generic_counts[generic] += 1

            per_reaction.append({
                "reaction_id": traj_id,
                "natoms": int(natoms),
                "whole_smiles": whole,
                "murcko_scaffold": murcko,
                "generic_scaffold": generic,
            })
    con.close()

    return {
        "db": db_path.name,
        "n_seen": n_seen,
        "n_failed": n_failed,
        "n_missing": n_missing,
        "whole": dict(whole_counts),
        "murcko": dict(murcko_counts),
        "generic": dict(generic_counts),
        "per_reaction": per_reaction,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5,
                    help="report fragment counts >= threshold")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-reaction charges (for rdDetermineBonds)
    idx = pd.read_parquet(INDEX_PARQUET)
    charge_map = dict(zip(idx["reaction_id"], idx["total_charge"]))
    print(f"[main] {len(idx)} reactions in index", flush=True)

    dbs = sorted(DB_DIR.glob("Halo_*.db"))
    print(f"[main] {len(dbs)} dbs; spawning {args.workers} workers", flush=True)

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        results = pool.map(_process_db, [(p, charge_map) for p in dbs])
    elapsed = time.time() - t0
    print(f"[main] all dbs done in {elapsed:.1f}s", flush=True)

    # Aggregate
    whole_total: Counter[str] = Counter()
    murcko_total: Counter[str] = Counter()
    generic_total: Counter[str] = Counter()
    per_rxn_rows: list[dict] = []
    n_seen = n_failed = 0
    for r in results:
        print(f"[main] {r['db']}: seen={r['n_seen']} failed={r['n_failed']}",
              flush=True)
        n_seen += r["n_seen"]
        n_failed += r["n_failed"]
        for smi, c in r["whole"].items():
            whole_total[smi] += c
        for smi, c in r["murcko"].items():
            murcko_total[smi] += c
        for smi, c in r["generic"].items():
            generic_total[smi] += c
        per_rxn_rows.extend(r["per_reaction"])

    # Persist
    per_rxn = pd.DataFrame(per_rxn_rows)
    per_rxn.to_parquet(args.out_dir / "per_reaction_smiles.parquet", index=False)

    def dump_counts(counter: Counter[str], stem: str):
        df = (pd.DataFrame(counter.items(), columns=["smiles", "count"])
                .sort_values("count", ascending=False, ignore_index=True))
        df.to_parquet(args.out_dir / f"{stem}_counts.parquet", index=False)
        return df

    whole_df = dump_counts(whole_total, "whole")
    murcko_df = dump_counts(murcko_total, "murcko")
    generic_df = dump_counts(generic_total, "generic")

    # Report
    T = args.threshold

    def stats(df: pd.DataFrame, label: str):
        n_unique = len(df)
        n_thresh = int((df["count"] >= T).sum())
        coverage = int(df.loc[df["count"] >= T, "count"].sum())
        total = int(df["count"].sum())
        print(f"\n[{label}]")
        print(f"  unique fragments        : {n_unique}")
        print(f"  unique with count >= {T:<2}: {n_thresh}")
        print(f"  reactions covered (sum) : {coverage} / {total}")
        if n_unique:
            print("  top 10:")
            for smi, c in df.head(10).itertuples(index=False, name=None):
                print(f"    {c:>5}  {smi}")
        return {"label": label, "unique": n_unique,
                "unique_ge_threshold": n_thresh, "covered": coverage,
                "total": total}

    print("\n========== SUMMARY ==========")
    print(f"reactions seen   : {n_seen}")
    print(f"bond-perception failures: {n_failed}")

    summary = {
        "threshold": T,
        "n_reactions_seen": n_seen,
        "n_bond_perception_failures": n_failed,
        "elapsed_sec": round(elapsed, 1),
    }
    summary["whole"] = stats(whole_df, "whole canonical SMILES")
    summary["murcko"] = stats(murcko_df, "Bemis-Murcko scaffolds")
    summary["generic"] = stats(generic_df, "generic (atom-blind) scaffolds")

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\noutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
