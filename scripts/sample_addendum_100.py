"""Sample 100 additional reactions following the multi-axis stratification
recommendation:

  * Floor for rare patterns:   P0 +8, P1 +8, P3 +20
  * T1x P2_CLOSED deficit fix: P2 +32 (28 from T1x, 4 from halogen sources)
  * Proportional fill:         P4 +10, P5 +22
  Total: +100

Outputs:
  * outputs/phase1/additional_selected.csv  — the 100 new rows
  * appends to outputs/phase1/selected_reactions.csv (with addendum=True flag)
  * outputs/stage5a/addendum_100_summary.json  — quick sanity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SEED = 4242

ALLOCATION = {
    "P0_BIMOL": 8,
    "P1_OPEN": 8,
    "P3_TETHER": 20,
    "P2_CLOSED": 32,
    "P4_DISSOC": 10,
    "P5_HSHIFT": 22,
}
T1X_P2_TARGET = 28  # of the 32 P2 slots, take this many from T1x


def main() -> None:
    rng = np.random.default_rng(SEED)
    sel_csv = ROOT / "outputs/phase1/selected_reactions.csv"
    idx_pq = ROOT / "data/halo8_index/index.parquet"
    bc_pq = ROOT / "data/halo8_index/bond_changes_all.parquet"
    pop_json = ROOT / "outputs/stage5a/population_classification.json"
    add_csv = ROOT / "outputs/phase1/additional_selected.csv"

    pop = pd.DataFrame(json.loads(pop_json.read_text()))
    idx = pd.read_parquet(idx_pq)
    bc = pd.read_parquet(bc_pq)[
        ["reaction_id", "bonds_broken", "bonds_formed", "n_bond_changes",
         "n_components_R"]
    ]
    existing = pd.read_csv(sel_csv)
    already = set(existing["reaction_id"])

    cand = pop[~pop["rxn_id"].isin(already)].copy()
    # Drop overlapping columns from idx before merging to keep names clean
    idx_to_merge = idx.drop(columns=[c for c in ("source", "n_atoms_max",
                                                  "n_heavy_atoms")
                                     if c in idx.columns and c in cand.columns])
    cand = cand.merge(idx_to_merge, left_on="rxn_id", right_on="reaction_id",
                      how="left")
    cand = cand.merge(bc, on="reaction_id", how="left")

    selected_rows: list[pd.DataFrame] = []
    for pat, n_target in ALLOCATION.items():
        pool = cand[cand["pattern"] == pat]
        if pat == "P2_CLOSED":
            t1x_pool = pool[pool["source"] == "T1x"]
            n_t1x = min(T1X_P2_TARGET, n_target, len(t1x_pool))
            seed_t1x = int(rng.integers(1, 1_000_000))
            picks_t1x = t1x_pool.sample(n=n_t1x, random_state=seed_t1x)
            n_rem = n_target - n_t1x
            other_pool = pool[pool["source"] != "T1x"]
            if n_rem > 0 and len(other_pool) > 0:
                seed_other = int(rng.integers(1, 1_000_000))
                picks_other = other_pool.sample(n=min(n_rem, len(other_pool)),
                                                random_state=seed_other)
            else:
                picks_other = pool.iloc[0:0]
            picks = pd.concat([picks_t1x, picks_other])
        else:
            n = min(n_target, len(pool))
            if n == len(pool):
                picks = pool
            else:
                seed = int(rng.integers(1, 1_000_000))
                picks = pool.sample(n=n, random_state=seed)
        selected_rows.append(picks)

    new_df = pd.concat(selected_rows, ignore_index=True)
    assert len(new_df) == 100, len(new_df)

    # Build a row schema that matches selected_reactions.csv
    # selected_reactions.csv columns:
    sel_cols = list(existing.columns)
    new_rows = []
    for _, r in new_df.iterrows():
        new_rows.append({
            "reaction_id": r["reaction_id"],
            "source": r["source"],
            "n_atoms_max": int(r["n_atoms_max"]),
            "n_heavy_atoms": int(r["n_heavy_atoms"]),
            "atomic_numbers": str(r["atomic_numbers"]),
            "n_snapshots": int(r["n_snapshots"]),
            "frame_index_first": int(r["frame_index_first"]),
            "frame_index_last": int(r["frame_index_last"]),
            "ts_frame_idx": int(r["ts_frame_idx"]),
            "ts_position_in_sorted": int(r["ts_position_in_sorted"]),
            "energy_R": float(r["energy_R"]),
            "energy_TS": float(r["energy_TS"]),
            "energy_P": float(r["energy_P"]),
            "activation_energy": float(r["activation_energy"]),
            "ea_relative_to_P": float(r["ea_relative_to_P"]),
            "total_charge": float(r["total_charge"]),
            "formula": r["formula"],
            "source_db_idx": int(r["source_db_idx"]),
            "short_traj": bool(r["short_traj"]),
            "interior_ts": bool(r["interior_ts"]),
            "bonds_broken": str(r.get("bonds_broken", "[]")),
            "bonds_formed": str(r.get("bonds_formed", "[]")),
            "n_bond_changes": int(r.get("n_bond_changes", 0) or 0),
            "n_components_R": int(r.get("n_components_R", 1) or 1),
            "bond_change_bin": "addendum",
            "ea_tertile": "addendum",
            "cell_label": f"addendum|pattern={r['pattern']}",
            "seed": SEED,
        })
    new_csv_df = pd.DataFrame(new_rows, columns=sel_cols)
    new_csv_df.to_csv(add_csv, index=False)
    print(f"wrote {add_csv} with {len(new_csv_df)} rows")

    # Append to selected_reactions.csv (idempotent: only append rows whose
    # reaction_id is not already in the file).
    not_in = ~new_csv_df["reaction_id"].isin(set(existing["reaction_id"]))
    appended = new_csv_df[not_in]
    pd.concat([existing, appended], ignore_index=True).to_csv(sel_csv, index=False)
    print(f"appended {len(appended)} rows to {sel_csv} "
          f"(now {len(existing) + len(appended)} total)")

    # Summary
    print("\nNew 100 by (pattern, source):")
    summary = (new_csv_df.assign(pattern=new_df["pattern"].values)
               .groupby(["pattern", "source"]).size().unstack(fill_value=0))
    print(summary.to_string())
    out = {
        "n_new": len(new_csv_df),
        "allocation": ALLOCATION,
        "seed": SEED,
        "by_pattern": new_df["pattern"].value_counts().to_dict(),
        "by_pattern_source": summary.stack().to_dict() if not summary.empty else {},
        "new_reaction_ids": new_csv_df["reaction_id"].tolist(),
    }
    out_path = ROOT / "outputs/stage5a/addendum_100_summary.json"
    out_path.write_text(json.dumps(
        out,
        indent=2,
        default=lambda o: int(o) if hasattr(o, '__int__') else str(o),
    ))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
