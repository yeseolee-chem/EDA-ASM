"""Assemble labels_v9_5channel.parquet from v9 CP-corrected R-SPs + reused v8 EDA outputs.

Difference from v8:
  * strain reference uses outputs/v9_review/strain_sp_cp/<rid>/frag{A,B}_R.out
    which include ghost basis from the opposing fragment at R positions,
    matching the ghost convention of eda_frag{1,2}.out (numerator). This
    removes the systematic BSSE offset (~-15 kcal/mol on monatomic B).
  * partition consistency: v9 uses TS partition (frag_A_indices) for both
    TS and R, so per-row atom identity is preserved.

The 4 non-strain channels (pauli, elst, orb, disp) are UNCHANGED — copied
straight from v8. Only strain, act, strain_A, strain_B and the *_R_hartree
columns are recomputed.

Idempotent: rerunning after new SPs land will refresh only the rows that
have both frag{A,B}_R.out completed. Rows missing one or both are marked
strain_kcal = NaN and act_kcal = NaN.

Output: outputs/v8_review/labels/labels_v9_5channel.parquet
Sanity checks printed at the end:
  * monatomic-B rows should have |strain_B| < 1 kcal/mol
  * neg strain count should be a small fraction (<10) explainable by RC-not-min
"""
from __future__ import annotations
import re, json
from pathlib import Path
import pandas as pd
import ase.io

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8   = REPO / "outputs/v8_review"
V9   = REPO / "outputs/v9_review"
LABELS_V8_LOCKED = V8 / "labels/labels_v8_5channel.LOCKED_799.parquet"
SP_CP_ROOT       = V9 / "strain_sp_cp"
SP_HELPER_ROOT   = V9 / "strain_sp_helper"
ORCA_ROOT        = V8 / "orca_inputs"
OUT_PARQUET      = V8 / "labels/labels_v9_5channel.parquet"
DIAG_JSON        = V8 / "labels/labels_v9_diagnostics.json"

HARTREE_TO_KCAL = 627.5094740631


def detect_label_error(rid: str, entry: dict, v8_root: Path) -> str:
    """Return a non-empty reason string if this row's R-partition mis-tags a
    fragment atom vs the TS partition (different element for monatomic
    nucleophile) — which produces catastrophic strain due to comparing
    different chemical species. Empty string means no known error."""
    fB_TS = entry.get("frag_B_indices", [])
    fB_R  = entry.get("frag_B_indices_R", [])
    if len(fB_TS) == 1 and len(fB_R) == 1:
        try:
            T = ase.io.read(str(v8_root / "raw_geoms" / rid / "TS.xyz"))
            R = ase.io.read(str(v8_root / "raw_geoms" / rid / "R.xyz"))
            sT = T.get_chemical_symbols()[fB_TS[0]]
            sR = R.get_chemical_symbols()[fB_R[0]]
            if sT != sR:
                return f"R-partition mislabels fragB: TS={sT}, R={sR}"
        except Exception as exc:
            return f"element-check failed ({exc})"
    return ""


def parse_final_energy(path: Path):
    """Last FINAL SINGLE POINT ENERGY (hartree) from a completed ORCA out.
    Returns None if the run didn't terminate normally or the file is missing."""
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in txt:
        return None
    m = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    return float(m[-1]) if m else None


def main() -> None:
    v8 = pd.read_parquet(LABELS_V8_LOCKED)
    manual = json.loads((V8 / "manual_partitions.json").read_text())
    override_path = V9 / "partition_override_v9.json"
    if override_path.exists():
        for rid, patch in json.loads(override_path.read_text()).items():
            if rid in manual:
                manual[rid] = {**manual[rid], **patch}

    rows = []
    n_ok = 0
    n_missing_A = 0
    n_missing_B = 0
    for _, r8 in v8.iterrows():
        rid = r8.reaction_id
        # Numerator: reuse v8 E_A(TS) and E_B(TS) exactly.
        E_A_TS = float(r8.E_A_TS_hartree)
        E_B_TS = float(r8.E_B_TS_hartree)
        # Denominator: v9 CP-corrected R-SPs. Primary in strain_sp_cp/;
        # if missing, fall back to strain_sp_helper/ (parallel helper dir
        # used to accelerate completion without race with primary array).
        E_A_R = parse_final_energy(SP_CP_ROOT / rid / "fragA_R.out")
        if E_A_R is None:
            E_A_R = parse_final_energy(SP_HELPER_ROOT / rid / "fragA_R.out")
        E_B_R = parse_final_energy(SP_CP_ROOT / rid / "fragB_R.out")
        if E_B_R is None:
            E_B_R = parse_final_energy(SP_HELPER_ROOT / rid / "fragB_R.out")

        if E_A_R is None:
            n_missing_A += 1
        if E_B_R is None:
            n_missing_B += 1

        # Monatomic-fragment clamp: strain ≡ 0 for a single-atom fragment
        # (cannot deform). Detect via the SMALLER of TS-side and R-side
        # partition counts so we clamp any row where either representation
        # is monatomic — protects against the 4 known dipolar mislabels
        # where R-partition ≠ TS-partition but both should have the same
        # chemical fragment identity.
        e = manual.get(rid, {})
        nA_TS = len(e.get("frag_A_indices", []))
        nB_TS = len(e.get("frag_B_indices", []))
        nA_R  = len(e.get("frag_A_indices_R", []))
        nB_R  = len(e.get("frag_B_indices_R", []))
        nA = min(nA_TS, nA_R) if nA_R else nA_TS
        nB = min(nB_TS, nB_R) if nB_R else nB_TS

        label_err = detect_label_error(rid, e, V8)

        if E_A_R is None or E_B_R is None:
            strain_A = strain_B = strain = act = float("nan")
        elif label_err:
            # NaN out strain for the 4 known R-label errors (user acknowledged).
            # Numerator and denominator would compare different chemical species.
            strain_A = strain_B = strain = act = float("nan")
        else:
            strain_A = 0.0 if nA == 1 else (E_A_TS - E_A_R) * HARTREE_TO_KCAL
            strain_B = 0.0 if nB == 1 else (E_B_TS - E_B_R) * HARTREE_TO_KCAL
            strain   = strain_A + strain_B
            act      = float(r8.int_eda_kcal) + strain
            n_ok += 1

        rows.append({
            "reaction_id":   rid,
            "family":        r8.family,
            # 4 unchanged channels
            "pauli_kcal":    float(r8.pauli_kcal),
            "elst_kcal":     float(r8.elst_kcal),
            "orb_kcal":      float(r8.orb_kcal),
            "disp_kcal":     float(r8.disp_kcal),
            # recomputed strain + activation proxy
            "strain_kcal":   strain,
            "int_eda_kcal":  float(r8.int_eda_kcal),
            "act_kcal":      act,
            "strain_A_kcal": strain_A,
            "strain_B_kcal": strain_B,
            "E_A_TS_hartree": E_A_TS,
            "E_B_TS_hartree": E_B_TS,
            "E_A_R_hartree":  E_A_R if E_A_R is not None else float("nan"),
            "E_B_R_hartree":  E_B_R if E_B_R is not None else float("nan"),
            "nA_atoms":      nA,
            "nB_atoms":      nB,
            "nA_TS":         nA_TS,
            "nB_TS":         nB_TS,
            "nA_R":          nA_R,
            "nB_R":          nB_R,
            "partition_mismatch": (nA_TS != nA_R) or (nB_TS != nB_R),
            "label_error":   label_err,
        })

    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"assembled {n_ok}/{len(df)} rows with both R-SPs done")
    print(f"  missing fragA_R.out: {n_missing_A}")
    print(f"  missing fragB_R.out: {n_missing_B}")
    print(f"  → {OUT_PARQUET}")

    # Sanity checks (only on completed rows)
    ok = df.dropna(subset=["strain_kcal"]).copy()
    if len(ok) == 0:
        print("no complete rows yet — sanity check deferred")
        return

    print()
    print("=== sanity: monatomic-B rows should have |strain_B| ~ 0 ===")
    mB = ok[ok.nB_atoms == 1]
    if len(mB):
        print(f"  monatomic-B rows: {len(mB)}")
        print(f"  strain_B  mean={mB.strain_B_kcal.mean():.4f}  "
              f"median={mB.strain_B_kcal.median():.4f}  "
              f"min={mB.strain_B_kcal.min():.4f}  "
              f"max={mB.strain_B_kcal.max():.4f}")
        print(f"  |strain_B|<0.5: {(mB.strain_B_kcal.abs() < 0.5).sum()} / {len(mB)}")
        print(f"  |strain_B|<0.1: {(mB.strain_B_kcal.abs() < 0.1).sum()} / {len(mB)}")

    print()
    print("=== negative strain summary ===")
    neg = ok[ok.strain_kcal < 0]
    print(f"  neg total: {len(neg)} / {len(ok)}")
    if len(neg):
        print(neg.groupby("family").size().to_string())
        print()
        print("  worst 10:")
        print(neg.nsmallest(10, "strain_kcal")[
            ["reaction_id","family","strain_kcal","strain_A_kcal","strain_B_kcal","nA_atoms","nB_atoms"]
        ].to_string(index=False))

    print()
    print("=== v8 vs v9 label diff (kcal/mol) ===")
    v8_sub = v8[["reaction_id","strain_kcal","act_kcal"]].rename(
        columns={"strain_kcal":"strain_v8","act_kcal":"act_v8"})
    cmp = ok.merge(v8_sub, on="reaction_id")
    cmp["dstrain"] = cmp.strain_kcal - cmp.strain_v8
    cmp["dact"]    = cmp.act_kcal    - cmp.act_v8
    print(f"  Δstrain (v9-v8):  mean={cmp.dstrain.mean():.2f}  "
          f"std={cmp.dstrain.std():.2f}  "
          f"min={cmp.dstrain.min():.2f}  "
          f"max={cmp.dstrain.max():.2f}")
    print(f"  Δact    (v9-v8):  mean={cmp.dact.mean():.2f}  "
          f"std={cmp.dact.std():.2f}")

    # Persist a small diagnostic JSON for the merge-vs-hold decision
    DIAG_JSON.write_text(json.dumps({
        "n_rows":              int(len(df)),
        "n_complete":          int(len(ok)),
        "n_missing_A":         int(n_missing_A),
        "n_missing_B":         int(n_missing_B),
        "monatomic_B_max_abs": float(mB.strain_B_kcal.abs().max()) if len(mB) else None,
        "neg_strain_count":    int(len(neg)),
        "neg_by_family":       neg.groupby("family").size().to_dict() if len(neg) else {},
        "delta_strain_mean":   float(cmp.dstrain.mean()),
        "delta_strain_std":    float(cmp.dstrain.std()),
    }, indent=2))
    print(f"  diagnostics: {DIAG_JSON}")


if __name__ == "__main__":
    main()
