"""spec21 step 1 — join our 400 with Stuyver's 5269 by source_id (G21-A).

Reads:
  - Ref Comparison/spec18r1_espley_s1_labels_fix/results/labels_2ch_400dipolar.pkl
  - Ref Comparison/spec19_espley_s2_structures/logs/discovery.json  (source_id + rxn_smiles per row)
  - /gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv

Writes:
  - results/cohort_joined.parquet   (400 rows, one per our reaction)
  - results/stuyver_full.parquet    (5269 rows, verbatim copy of Stuyver csv)
  - logs/gates.log                  (G21-A + G21-D lines)
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec21_cohort_bias_diagnosis"
STAGE1_PKL = REPO / "Ref Comparison/spec18r1_espley_s1_labels_fix/results/labels_2ch_400dipolar.pkl"
DISCOVERY_JSON = REPO / "Ref Comparison/spec19_espley_s2_structures/logs/discovery.json"
STUYVER_CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")

OUT_JOINED = STAGE / "results/cohort_joined.parquet"
OUT_STUYVER = STAGE / "results/stuyver_full.parquet"
GATES_LOG = STAGE / "logs/gates.log"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(fh, msg):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def main() -> int:
    STAGE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    STAGE.joinpath("results").mkdir(exist_ok=True)

    with open(GATES_LOG, "w") as fh:
        _log(fh, "=== spec21 gates ===")
        _log(fh, f"[env] python={platform.python_version()} pandas={pd.__version__}")

        # G21-D — Stuyver reference availability
        if not STUYVER_CSV.exists():
            _log(fh, f"[G21-D FAIL] Stuyver csv missing at {STUYVER_CSV}")
            raise RuntimeError("G21-D FAIL")
        stuyver_sha = _sha(STUYVER_CSV)
        stuyver = pd.read_csv(STUYVER_CSV)
        _log(fh, f"[G21-D PASS] Stuyver full_dataset.csv present (sha256 {stuyver_sha[:16]}…, n={len(stuyver)})")
        stuyver.to_parquet(OUT_STUYVER, index=False)

        # Load Stage 1 + Stage 2 discovery
        st1 = pd.read_pickle(STAGE1_PKL)
        with open(DISCOVERY_JSON) as jf:
            disc = json.load(jf)
        disc_recs = {r["reaction_id"]: r for r in disc["records"]}
        _log(fh, f"[load] Stage1 n={len(st1)}, discovery n={len(disc_recs)}")

        rows = []
        for _, row in st1.iterrows():
            rid = str(row["reaction_id"])
            rec = disc_recs.get(rid)
            if rec is None:
                raise RuntimeError(f"discovery.json missing record for {rid}")
            rows.append({
                "reaction_number": int(row["reaction_number"]),
                "reaction_id":     rid,
                "sub_source":      str(row["sub_source"]),
                "source_id":       int(rec["source_id"]),
                "rxn_smiles_ours": rec["rxn_smiles"],
                "ts_xyz_path":     rec["paths"]["ts_xyz"]["path"],
                "e_barrier_dft":   float(row["e_barrier_dft"]),
                "sum_distortion_energies_dft": float(row["sum_distortion_energies_dft"]),
                "interaction_energies_dft":    float(row["interaction_energies_dft"]),
            })
        ours = pd.DataFrame(rows)
        _log(fh, f"[our] cohort n={len(ours)}; sub_source counts: "
                 f"{dict(ours['sub_source'].value_counts())}")

        # G21-A part 1 — every source_id is in Stuyver
        stuyver_by_id = stuyver.set_index("rxn_id")
        missing_ids = [s for s in ours["source_id"] if s not in stuyver_by_id.index]
        if missing_ids:
            _log(fh, f"[G21-A FAIL] {len(missing_ids)} source_ids not found in Stuyver csv: {missing_ids[:10]}")
            raise RuntimeError("G21-A FAIL: missing source_ids")

        joined = ours.merge(
            stuyver_by_id[["rxn_smiles", "solvent", "temp", "G_act", "G_r"]],
            left_on="source_id", right_index=True, how="left",
            suffixes=("_ours", "_stuyver"),
        )
        joined = joined.rename(columns={
            "rxn_smiles":         "rxn_smiles_stuyver",
        })
        _log(fh, f"[join] joined shape={joined.shape}")

        # G21-A part 2 — string-for-string SMILES match
        mismatches = joined[joined["rxn_smiles_ours"] != joined["rxn_smiles_stuyver"]]
        if len(mismatches):
            _log(fh, f"[G21-A FAIL] {len(mismatches)} reactions have non-matching SMILES")
            _log(fh, mismatches[["reaction_id", "source_id",
                                  "rxn_smiles_ours", "rxn_smiles_stuyver"]].head(5).to_string())
            raise RuntimeError("G21-A FAIL: SMILES mismatch")
        _log(fh, f"[G21-A PASS] all 400 source_ids present in Stuyver and rxn_smiles matches string-for-string")

        joined.to_parquet(OUT_JOINED, index=False)
        _log(fh, f"[write] {OUT_JOINED}  n={len(joined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
