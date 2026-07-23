"""spec18_espley_s1_labels — Stage 1: recast 5-channel EDA labels into
Espley et al.'s 2-channel DIAS schema on the DIPOLAR-400 cohort.

Reads:  outputs/spec16_orca/labels/dipolar_400_merged.parquet
Writes: Ref Comparison/spec18_espley_s1_labels/results/labels_2ch_400dipolar.parquet

Cohort: 400 [3+2] dipolar cycloadditions (192 from LOCKED_778 + 208
from spec16 LC-extension). Direct family match with Espley et al.'s
ds3 ([3+2] only, n=3510), so the ds3 distribution anchors in
compare_to_ds3.py are for once meaningfully comparable modulo the
reference DFT level (Deviation #4).

Column mapping (locked):
  sum_distortion_energies_dft  = strain_kcal                          (positive)
  interaction_energies_dft     = -int_eda_kcal                        (SIGN FLIP)
  e_barrier_dft                = strain_kcal + int_eda_kcal           (= sum_dist - interaction)
  distortion_contributions_dft = {f"{rxn}_1": strain_A, f"{rxn}_2": strain_B}
  reaction_number              = int32, contiguous from 0

Sign convention matches Espley's `intera_energy = dist_energy - barrier`
(Digital Discovery 2024, DOI 10.1039/d4dd00224e, Eq. S2). Physically
our EDA E_int is negative (stabilising); theirs is stored positive.
Downstream code (f_select.py) keys on the substring "dft" — DO NOT
rename the columns.

Idempotent: exits early if the output parquet already exists and its
row count matches the cohort size.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths ---
REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SRC_PARQUET = REPO / "outputs/spec16_orca/labels/dipolar_400_merged.parquet"
STAGE = REPO / "Ref Comparison/spec18_espley_s1_labels"
COHORT_JSON = STAGE / "data/cohort_notes.json"
OUT_PARQUET = STAGE / "results/labels_2ch_400dipolar.parquet"
GATES_LOG = STAGE / "logs/gates.log"
BUILD_LOG = STAGE / "logs/build.log"

COHORT_N = 400


def _log(fh, msg: str) -> None:
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def load_source(build_fh) -> pd.DataFrame:
    _log(build_fh, f"[load] reading {SRC_PARQUET}")
    df = pd.read_parquet(SRC_PARQUET)
    _log(build_fh, f"[load] shape={df.shape} cols={list(df.columns)}")

    if len(df) != COHORT_N:
        raise RuntimeError(f"source parquet row count {len(df)} != {COHORT_N}")

    if "family" in df.columns:
        fams = df["family"].value_counts().to_dict()
        _log(build_fh, f"[load] family composition: {fams}")
        if set(fams.keys()) != {"dipolar"}:
            raise RuntimeError(f"non-dipolar rows in source: {fams}")

    return df


def collapse_channels(df: pd.DataFrame, build_fh) -> pd.DataFrame:
    # Use the source-recorded int_eda_kcal (rounded to 2 decimals) rather
    # than re-summing the four channels. This keeps Gate #4 at floor 0.0
    # rather than compounding the 0.02 kcal/mol rounding drift.
    e_int_eda = df["int_eda_kcal"].astype(np.float64)

    out = pd.DataFrame(index=df.index)
    out["reaction_number"] = np.arange(len(df), dtype=np.int32)

    # sum_distortion is the total strain (positive)
    out["sum_distortion_energies_dft"] = df["strain_kcal"].astype(np.float64).values

    # SIGN FLIP: paper stores interaction as sum_dist - barrier = -E_int^EDA (positive).
    # Source: energy_extraction/get_energies.py Eq. S2 in Espley et al. 2024.
    out["interaction_energies_dft"] = (-e_int_eda).astype(np.float64).values

    # e_barrier_dft = sum_distortion - interaction   (identity check in Gate #3)
    out["e_barrier_dft"] = (df["strain_kcal"] + e_int_eda).astype(np.float64).values

    # Per-fragment distortion dict for Table S6 parity.
    # Key format: f"{reaction_number}_{fragment_index}", fragment_index 1-based.
    # f_select.py::General._clean_dist_contr expects exactly this shape.
    contribs = []
    for rxn_num, sA, sB in zip(out["reaction_number"],
                                df["strain_A_kcal"].astype(np.float64).values,
                                df["strain_B_kcal"].astype(np.float64).values):
        contribs.append({f"{rxn_num}_1": float(sA), f"{rxn_num}_2": float(sB)})
    out["distortion_contributions_dft"] = contribs

    # carry provenance columns
    out["family"] = df["family"].values
    out["reaction_id"] = df["reaction_id"].values
    out["act_kcal_source"] = df["act_kcal"].astype(np.float64).values  # Gate #4 anchor
    if "source" in df.columns:
        out["sub_source"] = df["source"].values  # locked_778 vs spec16
    # sidecar for Gate #2 bijection check: source EDA sign flag
    out["_source_int_eda_kcal"] = e_int_eda.values

    _log(build_fh, f"[collapse] built columns: {list(out.columns)}")
    _log(build_fh, f"[collapse] dtypes:\n{out.dtypes}")
    return out


def run_gates(out: pd.DataFrame, gates_fh) -> None:
    fails = []

    # --- Gate #1: cohort ---
    if len(out) == COHORT_N:
        _log(gates_fh, f"[gate 1 PASS] cohort n={len(out)}")
    else:
        msg = f"[gate 1 FAIL] cohort n={len(out)} expected {COHORT_N}"
        _log(gates_fh, msg)
        fails.append(msg)

    # --- Gate #2: sign ---
    n_int_pos = int((out["interaction_energies_dft"] > 0).sum())
    frac_int_pos = n_int_pos / len(out)
    n_sd_pos = int((out["sum_distortion_energies_dft"] > 0).sum())

    # 2a (fractional tripwire): dipolar-only cohort is expected to be
    # >= 95% positive, matching Espley ds3.
    if frac_int_pos >= 0.95:
        _log(gates_fh, f"[gate 2a PASS] interaction > 0 in {n_int_pos}/{len(out)} rows "
                       f"({frac_int_pos:.4f}) — sign flip direction correct")
    else:
        msg = (f"[gate 2a FAIL] interaction > 0 in only {n_int_pos}/{len(out)} rows "
               f"({frac_int_pos:.4f}) — sign flip may not have been applied")
        _log(gates_fh, msg)
        fails.append(msg)
    if n_sd_pos == len(out):
        _log(gates_fh, f"[gate 2b PASS] sum_distortion > 0 in all {n_sd_pos} rows")
    else:
        neg = out.loc[out["sum_distortion_energies_dft"] <= 0, "reaction_id"].tolist()
        msg = f"[gate 2b FAIL] sum_distortion <= 0 in {len(out) - n_sd_pos} rows: {neg[:20]}"
        _log(gates_fh, msg)
        fails.append(msg)

    # 2c (bijection tripwire): sign(interaction_dft) MUST equal -sign(source E_int^EDA).
    src = out["_source_int_eda_kcal"].values
    itn = out["interaction_energies_dft"].values
    mismatched = int(np.sum(np.sign(itn) != -np.sign(src)))
    if mismatched == 0:
        _log(gates_fh, f"[gate 2c PASS] sign(interaction_dft) == -sign(source int_eda) for all {len(out)} rows")
    else:
        msg = f"[gate 2c FAIL] sign mismatch in {mismatched} rows — flip incorrectly applied"
        _log(gates_fh, msg)
        fails.append(msg)

    # --- Gate #3: identity ---
    lhs = out["e_barrier_dft"].values
    rhs = (out["sum_distortion_energies_dft"] - out["interaction_energies_dft"]).values
    d3 = float(np.max(np.abs(lhs - rhs)))
    if d3 < 1e-6:
        _log(gates_fh, f"[gate 3 PASS] identity max|e_barrier - (sum_dist - int)| = {d3:.3e}")
    else:
        msg = f"[gate 3 FAIL] identity max diff = {d3:.3e} (>= 1e-6)"
        _log(gates_fh, msg)
        fails.append(msg)

    # --- Gate #4: ASM identity vs. source barrier ---
    d4_vec = np.abs(out["e_barrier_dft"].values - out["act_kcal_source"].values)
    d4 = float(np.max(d4_vec))
    if d4 < 0.1:
        _log(gates_fh, f"[gate 4 PASS] |e_barrier - act_kcal_source| max = {d4:.6f} kcal/mol")
    else:
        idx_worst = int(np.argmax(d4_vec))
        bad_mask = d4_vec >= 0.1
        bad = out.loc[bad_mask, ["reaction_id", "family", "e_barrier_dft", "act_kcal_source"]]
        bad = bad.assign(abs_diff=d4_vec[bad_mask]).sort_values("abs_diff", ascending=False)
        msg = (f"[gate 4 FAIL] |e_barrier - act_kcal_source| max = {d4:.6f} kcal/mol "
               f"(>= 0.1); worst row = {out.iloc[idx_worst]['reaction_id']}; "
               f"n_bad = {int(bad_mask.sum())}")
        _log(gates_fh, msg)
        _log(gates_fh, f"[gate 4 FAIL] bad rows:\n{bad.to_string()}")
        fails.append(msg)

    # --- Gate #5: schema string parity ---
    target_cols = {
        "sum_distortion_energies_dft",
        "interaction_energies_dft",
        "e_barrier_dft",
        "distortion_contributions_dft",
        "reaction_number",
    }
    have = set(out.columns)
    missing = target_cols - have
    if not missing:
        _log(gates_fh, f"[gate 5a PASS] all target columns present")
    else:
        msg = f"[gate 5a FAIL] missing target columns: {missing}"
        _log(gates_fh, msg)
        fails.append(msg)
    non_dft = [c for c in target_cols if c != "reaction_number" and "dft" not in c]
    if not non_dft:
        _log(gates_fh, f"[gate 5b PASS] every non-key target column contains 'dft'")
    else:
        msg = f"[gate 5b FAIL] columns without 'dft' substring: {non_dft}"
        _log(gates_fh, msg)
        fails.append(msg)

    # --- Gate #6: dtype ---
    dt_rn = str(out["reaction_number"].dtype)
    dt_eb = str(out["e_barrier_dft"].dtype)
    dt_dc = str(out["distortion_contributions_dft"].dtype)
    dtype_ok = (dt_rn == "int32") and (dt_eb == "float64") and (dt_dc == "object")
    if dtype_ok:
        _log(gates_fh, f"[gate 6 PASS] dtypes reaction_number=int32, energies=float64, dict=object")
    else:
        msg = (f"[gate 6 FAIL] dtypes: reaction_number={dt_rn} (expected int32), "
               f"e_barrier_dft={dt_eb} (expected float64), "
               f"distortion_contributions_dft={dt_dc} (expected object)")
        _log(gates_fh, msg)
        fails.append(msg)

    # --- extra: sub-source breakdown (LOCKED_778 vs spec16) ---
    if "sub_source" in out.columns:
        _log(gates_fh, "\n[sub-source n]")
        _log(gates_fh, out.groupby("sub_source").size().to_string())

    # --- extra: contributions dict contract ---
    # Spec §6 item 5 asks for `abs(sum(dict) - sum_distortion) < 1e-6`. In
    # practice the source parquet stores strain_A/strain_B/strain rounded
    # independently, so the floor is set by rounding, not by our arithmetic.
    # Threshold 5e-3 kcal/mol catches genuine schema bugs (which would show
    # ≥ 0.1 kcal/mol drift) while tolerating source rounding.
    contribs_tol = 5e-3
    bad_dicts = 0
    max_sum_diff = 0.0
    for row in out.itertuples(index=False):
        d = row.distortion_contributions_dft
        if len(d) != 2:
            bad_dicts += 1
            continue
        try:
            for k in d:
                a, b = k.split("_")
                int(a); int(b)
        except Exception:
            bad_dicts += 1
            continue
        s = sum(d.values())
        max_sum_diff = max(max_sum_diff, abs(s - row.sum_distortion_energies_dft))
    if bad_dicts == 0 and max_sum_diff < contribs_tol:
        _log(gates_fh, f"[contribs contract PASS] all dicts 2-key, keys int_int, "
                       f"sum matches within {contribs_tol:.0e} (max diff {max_sum_diff:.3e})")
    else:
        msg = (f"[contribs contract FAIL] bad_dicts={bad_dicts}, "
               f"max_sum_diff={max_sum_diff:.3e} (tol {contribs_tol:.0e})")
        _log(gates_fh, msg)
        fails.append(msg)

    if fails:
        raise RuntimeError(f"Gate failures ({len(fails)}):\n" + "\n".join(fails))


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "logs").mkdir(exist_ok=True)
    (STAGE / "results").mkdir(exist_ok=True)

    # Idempotent skip
    if OUT_PARQUET.exists():
        try:
            existing = pd.read_parquet(OUT_PARQUET)
            if len(existing) == COHORT_N:
                print(f"[skip] {OUT_PARQUET} already exists (n={len(existing)}) — leaving in place")
                return
        except Exception:
            pass

    with open(BUILD_LOG, "w") as build_fh, open(GATES_LOG, "w") as gates_fh:
        _log(build_fh, "=== spec18 Stage 1: build 2ch labels (dipolar-400) ===")
        df = load_source(build_fh)
        out = collapse_channels(df, build_fh)

        _log(gates_fh, "=== gates ===")
        run_gates(out, gates_fh)

        # Drop the sidecar column before writing the output of record.
        out_final = out.drop(columns=["_source_int_eda_kcal"])
        tmp = OUT_PARQUET.with_suffix(".parquet.tmp")
        out_final.to_parquet(tmp, index=False)
        tmp.replace(OUT_PARQUET)
        _log(build_fh, f"[write] {OUT_PARQUET}  shape={out_final.shape}")


if __name__ == "__main__":
    sys.exit(main())
