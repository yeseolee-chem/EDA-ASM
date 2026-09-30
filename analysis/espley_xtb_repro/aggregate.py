#!/usr/bin/env python3
"""aggregate.py — merge xtb_slice.py slice parquets into one feature parquet with completeness gates.

  python aggregate.py [--geom dft|g1|g2] [--slices DIR] [--out PARQUET]
Defaults ($ESPLEY_OUT = /gpfs/tmp_cpu2/yeseo1ee/espley_xtb):
  dft -> slices_g0/ -> xtb_features_g0.parquet      g1 -> slices_g1/ -> xtb_features_g1.parquet   (g2 likewise)
Gates: N_SLICES slice files, all 5,260 rxns once, every row tagged with --geom, no NaN derived column on ok rows;
dft: >= 99% xtb_status ok. g1/g2: >= 1 ok row, no <geom>_missing row (Phase 1 incomplete); other non-ok rows
(g1_fail:*, partition_mismatch_g1, ...) are allowed.
An existing output is skipped only while all N_SLICES slices exist and none is newer than it; otherwise (a slice
redone) it is moved to <out>.stale — so a failed rebuild leaves no stale merge for dependent jobs — and rebuilt.
The parquet is written atomically (after the gates).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
TAG = {"dft": "g0", "g1": "g1", "g2": "g2"}
N_SLICES = int(os.environ.get("ESPLEY_NSLICES", "18"))
N_EXPECTED = 5260          # 5,265 labels − 5 excluded (3090, 3766, 4252, 3400, 5783)


def derive(df):
    """Add dft_c_ghost_kcal and is_charged (in place; also used by verify_geom_dft.py on raw slices)."""
    # c_ghost = ghost-reference correction (BSSE + cavity) = ghost − own-basis = −(e_bond − eint_spe).
    # Closed budget (method ①): barrier = d1 + d2 + e_bond + c_ghost  (exact).
    # d1 + d2 + e_bond alone mixes two reference frames (method ②) — NOT a barrier target.
    df["dft_c_ghost_kcal"] = df["dft_barrier_kcal"] - (df["dft_d1_kcal"] + df["dft_d2_kcal"] + df["dft_e_bond_kcal"])
    df["is_charged"] = (df["charge2"].fillna(0).astype(int) != 0).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", choices=tuple(TAG), default="dft")
    ap.add_argument("--slices", type=Path, default=None, help="default $ESPLEY_OUT/slices_<g0|g1|g2>")
    ap.add_argument("--out", type=Path, default=None, help="default $ESPLEY_OUT/xtb_features_<g0|g1|g2>.parquet")
    a = ap.parse_args()
    slices = a.slices or ROOT / f"slices_{TAG[a.geom]}"
    out = a.out or ROOT / f"xtb_features_{TAG[a.geom]}.parquet"
    paths = sorted(slices.glob("slice_*.parquet"))
    if out.exists():
        t_out = out.stat().st_mtime
        newer = [p.name for p in paths if p.stat().st_mtime > t_out]
        if len(paths) == N_SLICES and not newer:
            print(f"{out} exists and no slice is newer — skip"); return
        stale = out.with_name(out.name + ".stale")
        os.replace(out, stale)
        print(f"{out} is stale ({len(paths)} slice files, newer than it: {newer}) — moved to {stale.name}, rebuilding")
    if len(paths) != N_SLICES:
        sys.exit(f"GATE: expected {N_SLICES} slice files in {slices}, found {len(paths)}: {[p.name for p in paths]}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values("rxn_id").reset_index(drop=True)
    if len(df) != N_EXPECTED or df.rxn_id.nunique() != N_EXPECTED:
        sys.exit(f"GATE: expected {N_EXPECTED} unique rxns, got {len(df)} rows / {df.rxn_id.nunique()} unique")
    geoms = dict(df["geom"].value_counts(dropna=False)) if "geom" in df else {"<no geom column>": len(df)}
    if geoms != {a.geom: N_EXPECTED}:
        sys.exit(f"GATE: slices in {slices} are not all geom={a.geom}: {geoms}")

    derive(df)
    ok_mask = df["xtb_status"] == "ok"
    for col in ("dft_c_ghost_kcal",):
        nan_ok = int(df.loc[ok_mask, col].isna().sum())
        if nan_ok:
            sys.exit(f"GATE: derived column {col} has {nan_ok} NaN in ok rows")
    gap_mean = float(df.loc[ok_mask, "dft_c_ghost_kcal"].mean())
    gap_sd = float(df.loc[ok_mask, "dft_c_ghost_kcal"].std())
    print(f"derived: dft_c_ghost_kcal (mean {gap_mean:+.3f}, sd {gap_sd:.3f}) / is_charged  ({int(df.is_charged.sum())} rxns)")

    st = dict(df["xtb_status"].value_counts())
    print(f"geom={a.geom}: {len(df)} rows from {len(paths)} slices in {slices}")
    print(f"xtb_status: {st}")
    ok = df[ok_mask]
    print(f"ok={len(ok)}  solvation tags: {dict(ok.xtb_solvation.value_counts())}")
    print(f"ok rows with NaN in any xtb_*/q_* column: {int(ok.filter(regex='^(xtb_|q_)').isna().any(axis=1).sum())}")
    if a.geom == "dft" and len(ok) < 0.99 * N_EXPECTED:
        sys.exit(f"GATE: xTB success rate {len(ok)/N_EXPECTED:.3%} < 99% — inspect failures before ML")
    if a.geom != "dft":
        has_geom = ~df["xtb_status"].str.startswith(f"{a.geom}_")
        print(f"{a.geom} geometry available: {int(has_geom.sum())}; ok among them {int((ok_mask & has_geom).sum())}")
        if not len(ok):
            sys.exit(f"GATE: no ok row for geom={a.geom} — geometry root missing or empty?")
        n_miss = int((df["xtb_status"] == f"{a.geom}_missing").sum())
        if n_miss:
            sys.exit(f"GATE: {n_miss} rxns {a.geom}_missing — Phase 1 incomplete; delete the slices holding them, "
                     f"rerun them after Phase 1 finishes")

    tmp = out.with_name(out.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
