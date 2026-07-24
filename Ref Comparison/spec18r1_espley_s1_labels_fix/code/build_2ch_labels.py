"""spec18r1 Stage 1 build — DIPOLAR-400 → Espley 2-channel DIAS schema.

Fixes Rev 0's F1 (parquet destroys per-row dicts) and F7 (unstable
reaction_number). Writes a pickle for the paper's scripts to consume,
and a dict-column-stripped parquet for human inspection.

Reads:
  outputs/spec16_orca/labels/dipolar_400_merged.parquet

Writes:
  Ref Comparison/spec18r1_espley_s1_labels_fix/results/labels_2ch_400dipolar.pkl
  Ref Comparison/spec18r1_espley_s1_labels_fix/results/labels_2ch_400dipolar.INSPECTION_ONLY.parquet

All correctness gates run in verify_artifact.py against the reloaded
pickle, not against the in-memory frame.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SRC_PARQUET = REPO / "outputs/spec16_orca/labels/dipolar_400_merged.parquet"
STAGE = REPO / "Ref Comparison/spec18r1_espley_s1_labels_fix"
OUT_PKL = STAGE / "results/labels_2ch_400dipolar.pkl"
OUT_INSPECTION = STAGE / "results/labels_2ch_400dipolar.INSPECTION_ONLY.parquet"
BUILD_LOG = STAGE / "logs/build.log"

COHORT_N = 400
REQUIRED_COLS = [
    "reaction_id", "family", "source",
    "pauli_kcal", "elst_kcal", "orb_kcal", "disp_kcal",
    "strain_kcal", "strain_A_kcal", "strain_B_kcal",
    "int_eda_kcal", "act_kcal",
]


def _log(fh, msg: str) -> None:
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def load_source(fh) -> pd.DataFrame:
    _log(fh, f"[env] python={platform.python_version()} pandas={pd.__version__} numpy={np.__version__}")
    _log(fh, f"[load] {SRC_PARQUET}")
    df = pd.read_parquet(SRC_PARQUET)
    _log(fh, f"[load] shape={df.shape}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Required source columns absent (halting per §5): {missing}")
    _log(fh, f"[load] all required columns present")

    if len(df) != COHORT_N:
        raise RuntimeError(f"source parquet row count {len(df)} != {COHORT_N}")

    fams = df["family"].value_counts().to_dict()
    if set(fams.keys()) != {"dipolar"}:
        raise RuntimeError(f"non-dipolar rows in source: {fams}")
    _log(fh, f"[load] family composition: {fams}")

    # F7: sort by reaction_id BEFORE assigning reaction_number.
    df = df.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    _log(fh, f"[sort] mergesort by reaction_id — reaction_number will follow this order")
    return df


def build(df: pd.DataFrame, fh) -> pd.DataFrame:
    # Use the source-recorded int_eda_kcal for the paper-compatible interaction
    # column (which is what their scripts expect). The independent 4-channel
    # re-sum is used only for gate checks in verify_artifact.py.
    e_int_eda = df["int_eda_kcal"].astype(np.float64)

    out = pd.DataFrame(index=df.index)
    out["reaction_number"] = np.arange(len(df), dtype=np.int32)
    out["sum_distortion_energies_dft"] = df["strain_kcal"].astype(np.float64).values
    # SIGN FLIP: paper stores interaction as sum_dist − barrier = −E_int^EDA.
    out["interaction_energies_dft"] = (-e_int_eda).astype(np.float64).values
    out["e_barrier_dft"] = (df["strain_kcal"] + e_int_eda).astype(np.float64).values

    # Per-fragment strain dict. Key format `{reaction_number}_{fragment_index}`,
    # fragment index 1-based. _clean_dist_contr parses the leading int as
    # reaction_number.
    contribs = []
    for rxn_num, sA, sB in zip(
        out["reaction_number"].tolist(),
        df["strain_A_kcal"].astype(np.float64).values,
        df["strain_B_kcal"].astype(np.float64).values,
    ):
        contribs.append({f"{rxn_num}_1": float(sA), f"{rxn_num}_2": float(sB)})
    out["distortion_contributions_dft"] = contribs

    # Provenance (not consumed by their scripts)
    out["family"] = df["family"].values
    out["reaction_id"] = df["reaction_id"].values
    out["act_kcal_source"] = df["act_kcal"].astype(np.float64).values
    out["sub_source"] = df["source"].values

    _log(fh, f"[build] columns={list(out.columns)}")
    _log(fh, f"[build] dtypes:\n{out.dtypes}")
    return out


def write_artifacts(out: pd.DataFrame, fh) -> None:
    # Pickle (artifact of record).
    tmp_pkl = OUT_PKL.with_suffix(".pkl.tmp")
    out.to_pickle(tmp_pkl)
    tmp_pkl.replace(OUT_PKL)
    size_pkl = OUT_PKL.stat().st_size
    _log(fh, f"[write] {OUT_PKL}  size={size_pkl} bytes")

    # Inspection parquet (dict column DROPPED so no one confuses it for the artifact).
    inspection = out.drop(columns=["distortion_contributions_dft"])
    tmp_pq = OUT_INSPECTION.with_suffix(".parquet.tmp")
    inspection.to_parquet(tmp_pq, index=False)
    tmp_pq.replace(OUT_INSPECTION)
    size_pq = OUT_INSPECTION.stat().st_size
    _log(fh, f"[write] {OUT_INSPECTION}  size={size_pq} bytes")


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "logs").mkdir(exist_ok=True)
    (STAGE / "results").mkdir(exist_ok=True)

    with open(BUILD_LOG, "w") as fh:
        _log(fh, "=== spec18r1 Stage 1 build ===")
        df = load_source(fh)
        out = build(df, fh)
        write_artifacts(out, fh)
        _log(fh, "=== build OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
