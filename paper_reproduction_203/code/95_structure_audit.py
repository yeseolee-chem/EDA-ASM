#!/usr/bin/env python3
"""
Structure quality audit for 203 rxn subset.

For each rxn, compute:
  - d_forming_min: minimum heavy-atom distance between fragment_1 and fragment_2
                   in the TS geometry (Angstrom)
  - Stuyver ΔE‡ / ΔG‡ (if available in raw dataset profile file)
  - correlation with DFT barrier (from labels_5channel_paper.parquet)
  - verdict per rxn (contaminated if d_forming_min < 1.8 Å or > 3.2 Å)

Output:
  results/structure_audit.csv   per-rxn audit table with verdict
  results/contaminated_rxns.csv rxns failing structure gate (for exclusion)
  results/clean_rxns.csv        rxns passing structure gate
  results/structure_audit_summary.txt   histogram + statistics
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paper's typical [3+2] forming bond range
D_MIN_ANGSTROM = 1.8
D_MAX_ANGSTROM = 3.2


def parse_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    elems, coords = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        elems.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elems, np.array(coords)


def try_stuyver_barrier(rid: str) -> float | None:
    """Try to read Stuyver's ΔE‡ (kcal/mol) from raw dataset profile.

    Path: /gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/
             full_dataset_profiles/<idx>/  (contains e.g. r0, r1, TS, P xyz + JSON)

    Returns None if not accessible.
    """
    idx = rid.split("_")[1].lstrip("0") or "0"
    root = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
    d = root / idx
    if not d.is_dir():
        return None
    # look for JSON / metadata; Stuyver ships energies in various formats
    for cand in d.glob("*.json"):
        try:
            j = json.loads(cand.read_text())
            for key in ("dE_activation", "delta_E_activation", "activation_energy", "barrier", "ea_forward"):
                if key in j:
                    v = float(j[key])
                    # heuristic: if abs value > 1, assume kcal/mol; if < 1, assume Hartree
                    return v * 627.509 if abs(v) < 1 else v
        except Exception:
            continue
    return None


def process_rxn(rid: str, geom_root: Path, labels_df: pd.DataFrame) -> dict:
    d = geom_root / rid
    meta = json.loads((d / "meta.json").read_text())
    ts_elems, ts_coords = parse_xyz(d / "TS.xyz")

    frag1 = meta["orca_frag1_ts_idx_0based"]
    frag2 = meta["orca_frag2_ts_idx_0based"]

    # Consider heavy atoms only (H excluded, following common convention for forming bonds)
    heavy_1 = [i for i in frag1 if ts_elems[i] != "H"]
    heavy_2 = [i for i in frag2 if ts_elems[i] != "H"]

    c1 = ts_coords[heavy_1]
    c2 = ts_coords[heavy_2]
    dmat = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1)
    d_forming_min = float(dmat.min())

    # Second smallest distance (informative for [3+2] which has 2 forming bonds)
    flat = dmat.flatten()
    flat.sort()
    d_forming_2nd = float(flat[1]) if len(flat) > 1 else float("nan")

    # DFT barrier from labels
    row = labels_df[labels_df["reaction_id"] == rid].iloc[0]
    barrier_dft = float(row["e_barrier_dft"])
    distortion_dft = float(row["distortion_dft"])
    interaction_dft = float(row["interaction_dft"])

    stuyver_barrier = try_stuyver_barrier(rid)

    # Verdict
    if d_forming_min < D_MIN_ANGSTROM:
        verdict = "contaminated_bonded"
    elif d_forming_min > D_MAX_ANGSTROM:
        verdict = "contaminated_too_far"
    else:
        verdict = "clean"

    return {
        "reaction_id": rid,
        "reaction_number": meta["reaction_number"],
        "n_atoms_ts": meta["n_atoms_ts"],
        "n_heavy_1": len(heavy_1),
        "n_heavy_2": len(heavy_2),
        "d_forming_min": d_forming_min,
        "d_forming_2nd": d_forming_2nd,
        "barrier_dft": barrier_dft,
        "distortion_dft": distortion_dft,
        "interaction_dft": interaction_dft,
        "stuyver_barrier": stuyver_barrier if stuyver_barrier is not None else float("nan"),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent.parent)
    args = ap.parse_args()

    ROOT = args.root
    manifest = ROOT / "MANIFEST.txt"
    rxns = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            rxns.append(parts[1])

    labels = pd.read_parquet(ROOT / "labels" / "labels_5channel_paper.parquet")

    rows = [process_rxn(rid, ROOT / "geometries", labels) for rid in rxns]
    df = pd.DataFrame(rows)

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "structure_audit.csv", index=False)
    df[df["verdict"] == "clean"][["reaction_number", "reaction_id"]].to_csv(
        out / "clean_rxns.csv", index=False)
    df[df["verdict"] != "clean"][["reaction_number", "reaction_id", "d_forming_min", "verdict"]].to_csv(
        out / "contaminated_rxns.csv", index=False)

    # Summary
    n = len(df)
    n_clean = (df["verdict"] == "clean").sum()
    n_bonded = (df["verdict"] == "contaminated_bonded").sum()
    n_far = (df["verdict"] == "contaminated_too_far").sum()

    lines = []
    lines.append(f"=== Structure Audit ({n} rxns) ===\n")
    lines.append(f"Gate: d_forming_min ∈ [{D_MIN_ANGSTROM}, {D_MAX_ANGSTROM}] Å\n")
    lines.append(f"  clean:                 {n_clean}  ({100*n_clean/n:.1f}%)")
    lines.append(f"  contaminated_bonded:   {n_bonded}  ({100*n_bonded/n:.1f}%, d < {D_MIN_ANGSTROM} Å)")
    lines.append(f"  contaminated_too_far:  {n_far}    ({100*n_far/n:.1f}%, d > {D_MAX_ANGSTROM} Å)")

    lines.append(f"\n=== d_forming_min histogram ===")
    bins = [0, 1.2, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 5.0, 10.0]
    hist, edges = np.histogram(df["d_forming_min"], bins=bins)
    for i, count in enumerate(hist):
        lines.append(f"  {edges[i]:.1f} - {edges[i+1]:.1f} Å : {count}")

    lines.append(f"\n=== Barrier stats ===")
    lines.append(f"  ALL   (n={n}):   mean barrier {df['barrier_dft'].mean():.2f}, negative: {(df['barrier_dft']<0).sum()}/{n} ({100*(df['barrier_dft']<0).sum()/n:.0f}%)")
    for verdict in ("clean", "contaminated_bonded", "contaminated_too_far"):
        sub = df[df["verdict"] == verdict]
        if len(sub) > 0:
            lines.append(f"  {verdict:<25} (n={len(sub)}): mean barrier {sub['barrier_dft'].mean():+.2f}, negative: {(sub['barrier_dft']<0).sum()}/{len(sub)}, min={sub['barrier_dft'].min():+.2f}, max={sub['barrier_dft'].max():+.2f}")

    lines.append(f"\n=== Stuyver barrier availability ===")
    n_stuyver = df["stuyver_barrier"].notna().sum()
    lines.append(f"  Stuyver barrier parsed: {n_stuyver} / {n}")
    if n_stuyver > 0:
        sub = df.dropna(subset=["stuyver_barrier"])
        corr_all = sub[["barrier_dft", "stuyver_barrier"]].corr().iloc[0, 1]
        lines.append(f"  Correlation (all n={len(sub)}):   r = {corr_all:.3f}")
        sub_clean = sub[sub["verdict"] == "clean"]
        if len(sub_clean) > 5:
            corr_clean = sub_clean[["barrier_dft", "stuyver_barrier"]].corr().iloc[0, 1]
            lines.append(f"  Correlation (clean n={len(sub_clean)}): r = {corr_clean:.3f}")

    summary = "\n".join(lines)
    (out / "structure_audit_summary.txt").write_text(summary + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
