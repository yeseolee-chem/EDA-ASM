#!/usr/bin/env python3
"""d1c_aggregate.py — SPEC_D1_ALL Phase C: merge the D1 xtb_slice.py slices into $ESPLEY_OUT/xtb_features_g1.parquet.

  python d1c_aggregate.py            (inside phase_c_step.slurm agg: phase_c_env.sh sets ESPLEY_OUT / ESPLEY_LABELS)

aggregate.py with N_EXPECTED = the D1 label count instead of the D0 5,260 (its main() cannot be called for D1); the
gates, the derived columns (aggregate.derive) and the stale rule are the same:
  * D1_N_SLICES slice files, every D1 label rxn exactly once, every row geom g1, >= 1 ok row, no g1_missing row,
    no NaN dft_c_ghost_kcal on ok rows;
  * an existing output is kept only while every slice exists and none is newer; otherwise it moves to .stale.
The output is the ext_features.py input (rev5_common.FEAT_G1 under the D1 ESPLEY_OUT) — the SPEC output with the
job_id key (xtb_features_d1_g1.parquet) is written by d1c_merge.py.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "espley_xtb_repro"))
import aggregate as AG  # noqa: E402
import rev5_common as R5  # noqa: E402


def main():
    root = Path(os.environ["ESPLEY_OUT"])
    if root != R5.D1_XTB_ROOT:
        sys.exit(f"STOP: ESPLEY_OUT={root} is not the D1 root {R5.D1_XTB_ROOT} (source phase_c_env.sh)")
    labels = json.loads(Path(os.environ["ESPLEY_LABELS"]).read_text())
    n_exp = len(labels)
    slices, out = root / "slices_g1", root / "xtb_features_g1.parquet"
    paths = sorted(slices.glob("slice_*.parquet"))
    if out.exists():
        newer = [p.name for p in paths if p.stat().st_mtime > out.stat().st_mtime]
        if len(paths) == R5.D1_N_SLICES and not newer:
            print(f"{out} exists and no slice is newer — skip"); return
        os.replace(out, out.with_name(out.name + ".stale"))
        print(f"{out} is stale ({len(paths)} slices, newer: {newer}) — moved to .stale, rebuilding")
    if len(paths) != R5.D1_N_SLICES:
        sys.exit(f"GATE: expected {R5.D1_N_SLICES} slice files in {slices}, found {len(paths)}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values("rxn_id").reset_index(drop=True)
    want = sorted(int(d["rxn_id"]) for d in labels)
    if len(df) != n_exp or df.rxn_id.astype(int).tolist() != want:
        sys.exit(f"GATE: expected the {n_exp} D1 label rxns once each, got {len(df)} rows / {df.rxn_id.nunique()} unique")
    if dict(df["geom"].value_counts(dropna=False)) != {"g1": n_exp}:
        sys.exit(f"GATE: not all rows geom=g1: {dict(df['geom'].value_counts(dropna=False))}")
    ok = df["xtb_status"] == "ok"
    if not ok.any():
        sys.exit("GATE: no ok row")
    n_miss = int((df["xtb_status"] == "g1_missing").sum())
    if n_miss:
        sys.exit(f"GATE: {n_miss} rows g1_missing — G1 incomplete; delete their slices and rerun after G1 finishes")
    AG.derive(df)
    if int(df.loc[ok, "dft_c_ghost_kcal"].isna().sum()):
        sys.exit("GATE: NaN dft_c_ghost_kcal on ok rows")
    print(f"{len(df)} rows from {len(paths)} slices; xtb_status {dict(df.xtb_status.value_counts())}")
    R5.write_atomic(out, lambda t: df.to_parquet(t, index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
