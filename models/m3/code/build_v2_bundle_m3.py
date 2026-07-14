"""Build the m3 (xtb_geom6_plus) Δ-learning bundle per SPEC_xtb_descriptor_expansion §5.3-5.4.

Starts from the existing m2 bundle (features_v6_delta_xtb_geom6.pt, 25-d descriptors
= 6 geom + 19 xtb) and appends the 6 new SPEC scalars from xtb_extra.parquet:

    xtb_gap, xtb_mu, xtb_omega, xtb_dipole, xtb_qpol, xtb_dwbo_interfrag

Performs the SPEC §5.4 redundancy audit: any new column with |Pearson r| > 0.95
against an existing m2 column is dropped (these are linear/near-linear
combinations already present, e.g. xtb_gap ↔ gap_complex). Audit is
printed and persisted to the sidecar metadata file.

Output: analysis/exp_6arm_redesign_v2/bundles/features_v6_delta_xtb_geom6_plus.pt
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
M2_BUNDLE = HERE / "bundles/features_v6_delta_xtb_geom6.pt"
M2_FAM = HERE / "bundles/features_v6_delta_xtb_geom6.families.json"
EXTRA_CACHE_DEFAULT = HERE / "xtb_cache/xtb_extra.parquet"
EXTRA_CACHE_V2 = HERE / "xtb_cache/xtb_extra_v2.parquet"
EXTRA_CACHE = EXTRA_CACHE_V2 if EXTRA_CACHE_V2.exists() else EXTRA_CACHE_DEFAULT
# Output suffix tracks which cache was used (so re-runs don't clobber old artifacts).
CACHE_TAG = "v2" if EXTRA_CACHE == EXTRA_CACHE_V2 else "v1"
OUT_PT = HERE / f"bundles/features_v6_delta_xtb_geom6_plus_{CACHE_TAG}.pt" if CACHE_TAG == "v2" else HERE / "bundles/features_v6_delta_xtb_geom6_plus.pt"
OUT_FAM = HERE / f"bundles/features_v6_delta_xtb_geom6_plus_{CACHE_TAG}.families.json" if CACHE_TAG == "v2" else HERE / "bundles/features_v6_delta_xtb_geom6_plus.families.json"
META_JSON = HERE / f"bundles/features_v6_delta_xtb_geom6_plus_{CACHE_TAG}.meta.json" if CACHE_TAG == "v2" else HERE / "bundles/features_v6_delta_xtb_geom6_plus.meta.json"

# Existing m2 descriptor column order (must match build_v2_bundles.py:
#   geom6[0..5] then XTB_FEATURE_NAMES[0..18]).
GEOM6_NAMES = ["d1_rmsd_RT", "d2_rmsd_PT", "d3_pauli", "d4_invR", "d5_disp", "d6_natoms"]
XTB_FEATURE_NAMES = [
    "E_int_kcal", "E_complex_kcal", "E_fragA_kcal", "E_fragB_kcal",
    "dipole_complex_norm", "dipole_fragA_norm", "dipole_fragB_norm",
    "dipole_int",
    "HOMO_complex", "LUMO_complex", "gap_complex",
    "HOMO_fragA", "LUMO_fragA", "gap_fragA",
    "HOMO_fragB", "LUMO_fragB", "gap_fragB",
    "sum_q_A_frag_atoms", "n_atoms",
]
M2_COL_NAMES = GEOM6_NAMES + XTB_FEATURE_NAMES  # length 25

EXTRA_COLS = [
    "xtb_gap", "xtb_mu", "xtb_omega",
    "xtb_dipole", "xtb_qpol", "xtb_dwbo_interfrag",
]
REDUNDANCY_THRESHOLD = 0.95


def main() -> None:
    if not M2_BUNDLE.exists():
        sys.exit(f"[FATAL] m2 bundle missing: {M2_BUNDLE}; run build_v2_bundles.py first")
    if not EXTRA_CACHE.exists():
        sys.exit(f"[FATAL] xtb_extra cache missing: {EXTRA_CACHE}; run build_xtb_extra_cache.py first")

    print(f"[m3] loading m2 bundle: {M2_BUNDLE}")
    bundle = torch.load(M2_BUNDLE, map_location="cpu", weights_only=False)
    N = len(bundle["reaction_ids"])
    m2_desc = bundle["descriptors"].numpy().astype(np.float32)
    assert m2_desc.shape == (N, len(M2_COL_NAMES)), (
        f"unexpected m2 descriptor shape {m2_desc.shape} vs {len(M2_COL_NAMES)}"
    )
    print(f"[m3] m2 descriptors shape: {m2_desc.shape}")

    extra = pd.read_parquet(EXTRA_CACHE).set_index("rid")
    missing = [r for r in bundle["reaction_ids"] if r not in extra.index]
    if missing:
        print(f"[m3] WARN: {len(missing)} bundle rids missing from xtb_extra cache; "
              f"NaN-imputing with column means.")

    extra_mat = np.full((N, len(EXTRA_COLS)), np.nan, dtype=np.float32)
    for i, rid in enumerate(bundle["reaction_ids"]):
        if rid in extra.index:
            row = extra.loc[rid]
            extra_mat[i] = np.array([row.get(k, np.nan) for k in EXTRA_COLS], dtype=np.float32)

    # NaN-impute with column means (matches m2 build's policy).
    col_mean = np.nanmean(extra_mat, axis=0)
    nan_mask = np.isnan(extra_mat)
    extra_mat[nan_mask] = np.broadcast_to(col_mean, extra_mat.shape)[nan_mask]

    # Standardise (zero-mean, unit-std). Same policy as m2.
    extra_std = (extra_mat - extra_mat.mean(axis=0, keepdims=True)) / (
        extra_mat.std(axis=0, keepdims=True) + 1e-6
    )

    # SPEC §5.4 redundancy audit — Pearson r against every m2 column.
    print(f"\n[m3] redundancy audit (Pearson |r| vs existing m2 columns, threshold {REDUNDANCY_THRESHOLD}):")
    drop_idx: list[int] = []
    audit_rows: list[dict] = []
    for j, name in enumerate(EXTRA_COLS):
        col = extra_std[:, j]
        if np.std(col) < 1e-9:
            print(f"  {name:24s} : zero variance → drop")
            drop_idx.append(j)
            audit_rows.append({"name": name, "drop": True, "reason": "zero_variance"})
            continue
        max_r = 0.0; max_partner = ""
        for k, m2name in enumerate(M2_COL_NAMES):
            m2col = m2_desc[:, k]
            if np.std(m2col) < 1e-9:
                continue
            r = float(np.corrcoef(col, m2col)[0, 1])
            if abs(r) > abs(max_r):
                max_r = r; max_partner = m2name
        flag = "DROP" if abs(max_r) >= REDUNDANCY_THRESHOLD else "keep"
        print(f"  {name:24s} : max|r|={max_r:+.3f} (with {max_partner})  → {flag}")
        audit_rows.append({"name": name, "max_abs_r": abs(max_r),
                           "partner": max_partner, "drop": abs(max_r) >= REDUNDANCY_THRESHOLD})
        if abs(max_r) >= REDUNDANCY_THRESHOLD:
            drop_idx.append(j)

    keep_idx = [j for j in range(len(EXTRA_COLS)) if j not in drop_idx]
    kept_names = [EXTRA_COLS[j] for j in keep_idx]
    print(f"\n[m3] kept {len(keep_idx)}/{len(EXTRA_COLS)} new columns: {kept_names}")

    if not keep_idx:
        sys.exit("[FATAL] redundancy audit removed all new columns; nothing new to add.")

    extra_kept = extra_std[:, keep_idx]
    full_desc = np.concatenate([m2_desc, extra_kept], axis=1).astype(np.float32)
    full_names = M2_COL_NAMES + kept_names
    print(f"[m3] final descriptor shape: {full_desc.shape}  (col names list length {len(full_names)})")

    # Sanity: no NaN/Inf
    if not np.isfinite(full_desc).all():
        n_bad = (~np.isfinite(full_desc)).sum()
        sys.exit(f"[FATAL] {n_bad} non-finite values in final descriptor matrix")

    out_obj = dict(bundle)
    out_obj["descriptors"] = torch.from_numpy(full_desc)
    torch.save(out_obj, OUT_PT)
    if M2_FAM.exists() and not OUT_FAM.exists():
        shutil.copy(M2_FAM, OUT_FAM)

    meta = {
        "bundle": OUT_PT.name,
        "descriptor_dim": int(full_desc.shape[1]),
        "column_names": full_names,
        "redundancy_audit": audit_rows,
        "kept_new_columns": kept_names,
        "dropped_new_columns": [EXTRA_COLS[j] for j in drop_idx],
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
    }
    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"[m3] wrote {OUT_PT}")
    print(f"[m3] wrote {META_JSON}")


if __name__ == "__main__":
    main()
