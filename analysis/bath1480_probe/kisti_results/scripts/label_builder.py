#!/usr/bin/env python3
"""
label_builder.py — merge ORCA 6-channel EDA labels into the paper's ds3 dataframe.

Input
  1. --eda-root : directory containing ~3510 ORCA runs, one eda.out per reaction
  2. --pkl      : the paper's manual_tt_solvent.pkl  (3510 x 54, join on reaction_number)

Output
  --out (default manual_tt_7ch.pkl) : original 54 columns
        + 6 new targets : elst_dft, pauli_dft, oi_dft, disp_dft, cpcm_dft, cds_dft
        + bookkeeping   : eint_dft (Bond Energy), sum_residual_kcal, n_loose_scf_exits,
                          n_atoms_frag1/2, elements_frag1/2
  --audit (json) : coverage, gate stats, protocol-gap correlation report

Strain channels = the paper's distortion_energy_1_dft / distortion_energy_2_dft
(wB97X-D/def2-TZVP/IEFPCM) — kept as-is. Interaction sub-channels come from ORCA
B3LYP-D3BJ/def2-TZVP/CPCM+SMD. Mixed protocol accepted for proof-of-concept;
the audit quantifies the gap via corr(eint_dft, -interaction_energies_dft).

Column contract: new targets end in `_dft` so f_select.py / hyp_tuning.py /
ml_analysis.py extend automatically. Bookkeeping columns deliberately do NOT
end in `_dft`; drop DROP_BEFORE_ML before feature selection.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orca_eda_parser import parse_eda_out  # noqa: E402

TARGETS_NEW = ["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
BOOKKEEPING = ["eint_dft", "sum_residual_kcal", "n_loose_scf_exits",
               "n_atoms_frag1", "n_atoms_frag2", "elements_frag1", "elements_frag2"]
DROP_BEFORE_ML = BOOKKEEPING  # eint_dft is an identity (sum of 6), not a target


def extract_key(path: Path, key_regex: str):
    """Pull reaction_number from the run directory name (last regex match wins)."""
    m = re.findall(key_regex, path.parent.name)
    return int(m[-1]) if m else None


def build(eda_root, glob, pkl, out, audit_path, key_regex):
    files = sorted(Path(eda_root).glob(glob))
    if not files:
        sys.exit(f"[fatal] no files matched {eda_root}/{glob}")

    rows, fails, keyless = [], [], []
    for f in files:
        key = extract_key(f, key_regex)
        if key is None:
            keyless.append(str(f)); continue
        r = parse_eda_out(f)
        r["reaction_number"] = key
        (rows if r["ok"] else fails).append(r)

    eda = pd.DataFrame(rows)
    if eda.empty:
        sys.exit("[fatal] every parse failed; first failures:\n"
                 + "\n".join(str(x.get("source")) for x in fails[:10]))

    dup = eda["reaction_number"].duplicated(keep=False)
    if dup.any():
        sys.exit(f"[fatal] duplicate reaction_number after key extraction: "
                 f"{sorted(eda.loc[dup,'reaction_number'].unique())[:10]} ... "
                 f"fix --key-regex (current: {key_regex!r})")

    keep = ["reaction_number"] + TARGETS_NEW + BOOKKEEPING
    eda = eda[[c for c in keep if c in eda.columns]]

    base = pd.read_pickle(pkl)
    merged = base.merge(eda, on="reaction_number", how="inner", validate="1:1")

    audit = {
        "n_files_matched": len(files),
        "n_parsed_ok": len(rows),
        "n_parse_failed": len(fails),
        "n_keyless_paths": len(keyless),
        "n_base_rows": len(base),
        "n_merged": len(merged),
        "n_base_without_eda": int(len(base) - len(merged)),
        "n_eda_without_base": int(len(eda) - len(merged)),
        "failed_sources": [x.get("source") for x in fails][:50],
        "keyless_paths": keyless[:50],
    }

    audit["sum_residual_abs_max_kcal"] = float(merged["sum_residual_kcal"].abs().max())
    audit["n_loose_scf_total"] = int(merged["n_loose_scf_exits"].sum())
    audit["frac_runs_with_loose_scf"] = float((merged["n_loose_scf_exits"] > 0).mean())

    # paper stores interaction POSITIVE (= -dE_int)
    x = -merged["interaction_energies_dft"].to_numpy()
    y = merged["eint_dft"].to_numpy()
    audit["protocol_gap"] = {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "mae_kcal": float(np.mean(np.abs(x - y))),
        "mean_offset_kcal": float(np.mean(y - x)),
    }

    # NOTE: N/O heuristic CANNOT discriminate di/dp for this cohort
    # (measured: frag1 88.3% / frag2 78.8% carry N or O). Recorded as info only.
    # All 6 new channels are fragment-swap invariant; strain uses the paper's
    # own di/dp assignment, so no ML column depends on our BFS frag roles.
    audit["frac_frag1_with_NO"] = float(
        merged["elements_frag1"].str.contains("N|O", regex=True).mean())
    audit["frac_frag2_with_NO"] = float(
        merged["elements_frag2"].str.contains("N|O", regex=True).mean())
    audit["role_warning"] = None

    audit["channel_stats_kcal"] = {
        c: {"mean": float(merged[c].mean()), "min": float(merged[c].min()),
            "max": float(merged[c].max())} for c in TARGETS_NEW}

    merged.to_pickle(out)
    Path(audit_path).write_text(json.dumps(audit, indent=2))

    print(f"[ok] {out}: {merged.shape[0]} rows x {merged.shape[1]} cols (+{len(TARGETS_NEW)} targets)")
    print(f"[ok] audit -> {audit_path}")
    for k in ("n_parse_failed", "n_base_without_eda", "frac_runs_with_loose_scf"):
        print(f"     {k}: {audit[k]}")
    pg = audit["protocol_gap"]
    print(f"     protocol gap r = {pg['pearson_r']:.4f}, MAE = {pg['mae_kcal']:.2f} kcal/mol")
    if audit["role_warning"]:
        print(f"[WARN] {audit['role_warning']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda-root", required=True)
    ap.add_argument("--glob", default="*/eda.out")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", default="manual_tt_7ch.pkl")
    ap.add_argument("--audit", default="label_audit.json")
    ap.add_argument("--key-regex", default=r"(\d+)",
                    help="regex to pull reaction_number from the run dir name")
    a = ap.parse_args()
    build(a.eda_root, a.glob, a.pkl, a.out, a.audit, a.key_regex)
