#!/usr/bin/env python3
"""Cross-check our EDA-NOCV bond energies against Espley's interaction_energies_dft.

Espley's convention:
    interaction_energies_dft = sum_distortion_energies_dft - e_barrier_dft
    (i.e., positive value = interaction stabilization magnitude)

Bickelhaupt convention (ORCA output):
    e_bond_kcal = ΔE_int (negative for attractive)

Relationship (in the ideal case):
    Espley_interaction ≈ -1 × our_bond_energy

We compute:
    diff = our_bond_kcal - (-Espley_interaction_dft)
         = our_bond_kcal + Espley_interaction_dft

MAE(diff) across all reactions should be small (< 3 kcal/mol) if methods are
consistent. Larger MAE flags potential issues (basis, solvation, SCF, etc.).

Usage (inside sbatch):
    python verify_vs_espley.py \\
        results/tt_eda_5channel.parquet \\
        data/espley_reference.parquet \\
        results/verification_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 4:
        print("usage: verify_vs_espley.py <our_parquet> <espley_parquet> <report_md>", file=sys.stderr)
        sys.exit(1)
    our_path, espley_path, report_path = map(Path, sys.argv[1:4])

    ours = pd.read_parquet(our_path)
    espley = pd.read_parquet(espley_path)
    n_our = len(ours)
    n_ref = len(espley)

    # Inner join on rxn_id
    ok = ours[ours['terminated_normally'] == True]
    joined = ok.merge(espley, on='rxn_id', how='inner')
    n_joined = len(joined)

    # Compute diff (both in kcal/mol)
    joined['diff'] = joined['e_bond_kcal'] + joined['interaction_energies_dft']
    diff = joined['diff'].dropna()
    mae = diff.abs().mean() if len(diff) else float('nan')
    rmse = np.sqrt((diff ** 2).mean()) if len(diff) else float('nan')

    # Large-disagreement subset
    large = joined[joined['diff'].abs() > 5.0].sort_values('diff', key=abs, ascending=False)

    lines = [
        "# tt EDA-NOCV vs Espley Reference — Verification Report",
        "",
        f"- Our results (attempted): {n_our}",
        f"- Our results (terminated normally): {len(ok)}",
        f"- Espley reference rows: {n_ref}",
        f"- Joined on rxn_id: **{n_joined}**",
        "",
        "## Sign-flip Bond Energy vs Espley interaction_energy",
        "",
        f"- MAE(our_e_bond_kcal + espley_interaction_dft) = **{mae:.3f} kcal/mol**",
        f"- RMSE                                          = **{rmse:.3f} kcal/mol**",
        "",
        "Small MAE (< 3 kcal/mol) → methods consistent (accounting for the sign flip).",
        "Larger MAE flags basis / solvation / SCF differences.",
        "",
        "## Reactions with |diff| > 5 kcal/mol",
        "",
        f"Count: **{len(large)}**",
        "",
    ]
    if len(large):
        lines.append("| rxn_id | our_e_bond | espley_interaction_dft | diff (kcal/mol) | n_atoms |")
        lines.append("|---:|---:|---:|---:|---:|")
        # Include n_atoms if present
        show = large.head(30).copy()
        if 'n_atoms' not in show.columns:
            show['n_atoms'] = -1
        for _, r in show.iterrows():
            lines.append(f"| {int(r['rxn_id'])} | {r['e_bond_kcal']:.2f} | {r['interaction_energies_dft']:.2f} | {r['diff']:.2f} | {int(r.get('n_atoms', -1))} |")

    lines.extend([
        "",
        "## Failures (our runs)",
        "",
        f"- N reactions failed to terminate normally: {(~ours['terminated_normally'].fillna(False)).sum()}",
        "",
        "## Channels summary (means, kcal/mol, all successful)",
        "",
    ])
    for c in ['e_bond_kcal', 'e_pauli_kcal', 'e_elst_kcal', 'e_orb_kcal',
              'e_xc_kcal', 'e_disp_kcal', 'e_cpcm_kcal', 'e_smd_cds_kcal']:
        if c in ok.columns:
            s = ok[c].dropna()
            lines.append(f"- {c}: mean={s.mean():.2f}  range=[{s.min():.2f}, {s.max():.2f}]  n={len(s)}")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"wrote {report_path}")
    print(f"joined: {n_joined}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")


if __name__ == "__main__":
    main()
