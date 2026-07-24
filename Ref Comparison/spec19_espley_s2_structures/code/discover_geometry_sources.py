"""spec19 Stage 2 Step 0 — locate every input needed to assemble the five
DIAS structures per reaction on the dipolar-400 cohort.

Emits `logs/discovery.json`. HALTS if any critical path is missing for
any reaction (per §3 blocking requirement).

Critical inputs per reaction:
  - TS coordinates (.xyz)
  - Fragment atom-index assignment (used to compute strain_A/strain_B)
  - Fragment charge + multiplicity
  - Relaxed fragment A/B geometry

Non-critical (informational):
  - Atom-mapped rxn_smiles (for common_atoms; missing OK, downgrades G2-E)

Two sub-sources (192 locked_778 + 208 spec16) have DIFFERENT provenance:

                       locked_778                              spec16
TS.xyz                 outputs/v8_review/raw_geoms/{rid}/     outputs/manual_labels/{rid}/
R.xyz                  same (R.xyz)                            same (R.xyz)
frag assignment (TS)   parse (1)/(2) in                        header of
                       outputs/v8_review/orca_inputs/{rid}/     outputs/manual_labels/{rid}/
                       eda.inp                                  frag_A_TS.xyz
r_A / r_B              geometry inside                         outputs/spec16_orca_strain/inputs/
                       outputs/v8_review/strain_sp/{rid}/       {rid}__f{A,B}/opt.xyz
                       frag{A,B}_R.inp                          (isolated fragment opt)
charge/mult            labels/orca/orca_eda_charges_v9.parquet  same
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec19_espley_s2_structures"
STAGE1_PKL = REPO / "Ref Comparison/spec18r1_espley_s1_labels_fix/results/labels_2ch_400dipolar.pkl"

ORCA_CHARGES = REPO / "labels/orca/orca_eda_charges_v9.parquet"
STUYVER_CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")

# locked_778 paths
LOCKED_RAWGEOMS = REPO / "outputs/v8_review/raw_geoms"
LOCKED_ORCA_IN  = REPO / "outputs/v8_review/orca_inputs"
LOCKED_STRAIN   = REPO / "outputs/v9_review/strain_sp_cp"   # CP-corrected (v9) primary
LOCKED_STRAIN_ALT = REPO / "outputs/v8_review/strain_sp"    # v8 fallback

# spec16 paths
SPEC16_MANUAL   = REPO / "outputs/manual_labels"
SPEC16_ORCA_IN  = REPO / "outputs/spec16_orca/inputs"
SPEC16_STRAIN   = REPO / "outputs/spec16_orca_strain/inputs"

OUT_DISCOVERY = STAGE / "logs/discovery.json"
BUILD_LOG = STAGE / "logs/build.log"


def _log(fh, msg: str) -> None:
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def load_stage1(fh) -> pd.DataFrame:
    _log(fh, f"[env] python={platform.python_version()} pandas={pd.__version__}")
    _log(fh, f"[G2-0] loading Stage 1 pickle: {STAGE1_PKL}")
    df = pd.read_pickle(STAGE1_PKL)
    if df.shape != (400, 9):
        raise RuntimeError(f"G2-0 FAIL: Stage 1 shape {df.shape} != (400, 9)")
    v = df["distortion_contributions_dft"].iloc[0]
    if not (isinstance(v, dict) and len(v) == 2):
        raise RuntimeError(f"G2-0 FAIL: first dict is not shape-2 dict: {v!r}")
    _log(fh, f"[G2-0 PASS-in-reactot] shape=(400, 9); first dict={v}")
    _log(fh, "[G2-0 DEFERRED] round-trip test in pandas==2.1.1 env NOT executed "
             "(no espley_repro env on this HPC). Recorded for Stage 5.")
    return df


def load_charges(fh) -> pd.DataFrame:
    _log(fh, f"[charges] loading {ORCA_CHARGES}")
    d = pd.read_parquet(ORCA_CHARGES)
    _log(fh, f"[charges] shape={d.shape}; cols={list(d.columns)}")
    return d.set_index("reaction_id")


def load_smiles(fh) -> dict:
    _log(fh, f"[smiles] loading {STUYVER_CSV}")
    d = pd.read_csv(STUYVER_CSV)
    _log(fh, f"[smiles] shape={d.shape}")
    return dict(zip(d["rxn_id"].astype(int), d["rxn_smiles"].astype(str)))


def rid_to_source_id(rid: str) -> int:
    # dipolar_000017 → 17
    return int(rid.split("_")[1])


def resolve_paths(rid: str, sub_source: str) -> dict:
    """Return a dict of resolved paths + existence flags."""
    if sub_source == "spec16":
        ts_xyz     = SPEC16_MANUAL / rid / "TS.xyz"
        r_xyz      = SPEC16_MANUAL / rid / "R.xyz"
        frag_A_TS  = SPEC16_MANUAL / rid / "frag_A_TS.xyz"
        frag_B_TS  = SPEC16_MANUAL / rid / "frag_B_TS.xyz"
        eda_inp    = SPEC16_ORCA_IN / rid / "eda.inp"
        r_A_source = SPEC16_STRAIN / f"{rid}__fA" / "opt.xyz"
        r_B_source = SPEC16_STRAIN / f"{rid}__fB" / "opt.xyz"
        r_A_kind   = "opt.xyz (isolated fragment optimized at BLYP-D3BJ/def2-TZVP)"
        r_B_kind   = "opt.xyz (isolated fragment optimized at BLYP-D3BJ/def2-TZVP)"
    elif sub_source == "locked_778":
        ts_xyz     = LOCKED_RAWGEOMS / rid / "TS.xyz"
        r_xyz      = LOCKED_RAWGEOMS / rid / "R.xyz"
        frag_A_TS  = None   # not present; use eda.inp instead
        frag_B_TS  = None
        eda_inp    = LOCKED_ORCA_IN / rid / "eda.inp"
        r_A_source = LOCKED_STRAIN / rid / "fragA_R.inp"
        r_B_source = LOCKED_STRAIN / rid / "fragB_R.inp"
        if not r_A_source.exists() or not r_B_source.exists():
            r_A_source = LOCKED_STRAIN_ALT / rid / "fragA_R.inp"
            r_B_source = LOCKED_STRAIN_ALT / rid / "fragB_R.inp"
            r_A_kind = r_B_kind = "fragA_R.inp geometry (R.xyz atom subset; NOT independently optimized; v8_review/strain_sp fallback)"
        else:
            r_A_kind = r_B_kind = "fragA_R.inp geometry (R.xyz atom subset; NOT independently optimized; v9_review/strain_sp_cp)"
    else:
        raise ValueError(f"unknown sub_source: {sub_source}")

    def _pack(label: str, p) -> dict:
        return {
            "label": label,
            "path": str(p) if p else None,
            "exists": bool(p and Path(p).exists()),
        }

    resolved = {
        "ts_xyz":     _pack("TS full geometry", ts_xyz),
        "r_xyz":      _pack("R full geometry (both fragments)", r_xyz),
        "eda_inp":    _pack("EDA input with (1)/(2) fragment labels", eda_inp),
        "r_A_source": _pack(f"r_A source: {r_A_kind}", r_A_source),
        "r_B_source": _pack(f"r_B source: {r_B_kind}", r_B_source),
    }
    if frag_A_TS is not None:
        resolved["frag_A_TS"] = _pack("frag A of TS with atom indices in header", frag_A_TS)
        resolved["frag_B_TS"] = _pack("frag B of TS with atom indices in header", frag_B_TS)
    return resolved


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "logs").mkdir(exist_ok=True)

    with open(BUILD_LOG, "w") as fh:
        _log(fh, "=== spec19 Stage 2 Step 0 (discovery) ===")
        df = load_stage1(fh)
        charges = load_charges(fh)
        smiles_map = load_smiles(fh)

        records = []
        missing_critical = []
        missing_smiles = []

        for _, row in df.iterrows():
            rid = str(row["reaction_id"])
            sub = str(row["sub_source"])
            rn = int(row["reaction_number"])
            paths = resolve_paths(rid, sub)

            # Charge / mult: locked_778 → orca_eda_charges_v9.parquet (has 783 rows);
            #                spec16    → meta.json in spec16_orca/inputs/{rid}/
            charge_info = None
            if sub == "locked_778":
                if rid not in charges.index:
                    missing_critical.append((rid, "orca_eda_charges_v9 missing"))
                else:
                    r = charges.loc[rid]
                    charge_info = {
                        "total_charge": int(r["total_charge"]),
                        "fragment_charge_a": int(r["fragment_charge_a"]),
                        "fragment_charge_b": int(r["fragment_charge_b"]),
                        "fragment_mult_a": int(r["fragment_mult_a"]),
                        "fragment_mult_b": int(r["fragment_mult_b"]),
                        "source": "labels/orca/orca_eda_charges_v9.parquet",
                    }
            else:  # spec16
                meta_path = SPEC16_ORCA_IN / rid / "meta.json"
                if not meta_path.exists():
                    missing_critical.append((rid, f"meta.json missing at {meta_path}"))
                else:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    try:
                        charge_info = {
                            "total_charge": int(meta["total_C"]),
                            "fragment_charge_a": int(meta["frag1_C"]),
                            "fragment_charge_b": int(meta["frag2_C"]),
                            "fragment_mult_a": int(meta["frag1_M"]),
                            "fragment_mult_b": int(meta["frag2_M"]),
                            "source": f"outputs/spec16_orca/inputs/{rid}/meta.json",
                        }
                    except KeyError as e:
                        missing_critical.append((rid, f"meta.json missing key {e}"))

            # SMILES (informational)
            source_id = rid_to_source_id(rid)
            rxn_smiles = smiles_map.get(source_id)
            if rxn_smiles is None:
                missing_smiles.append(rid)

            # Critical existence
            for key in ("ts_xyz", "r_xyz", "eda_inp", "r_A_source", "r_B_source"):
                if not paths[key]["exists"]:
                    missing_critical.append((rid, f"{key} missing at {paths[key]['path']}"))

            records.append({
                "reaction_number": rn,
                "reaction_id": rid,
                "sub_source": sub,
                "source_id": source_id,
                "paths": paths,
                "charge_info": charge_info,
                "rxn_smiles": rxn_smiles,
            })

        _log(fh, f"[summary] n_records={len(records)}")
        _log(fh, f"[summary] n_missing_critical={len(missing_critical)}")
        _log(fh, f"[summary] n_missing_smiles (informational)={len(missing_smiles)}")

        summary = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "n_records": len(records),
            "n_missing_critical": len(missing_critical),
            "n_missing_smiles": len(missing_smiles),
            "missing_critical_head": missing_critical[:20],
            "missing_smiles_head": missing_smiles[:20],
            "note_locked778_relaxed":
                "For 192 locked_778 reactions, r_A/r_B are R.xyz atom subsets "
                "(reactant-complex geometry, NOT independently optimized). "
                "This matches the strain_A/strain_B label computation but "
                "diverges from spec16's fully-optimized isolated fragments. "
                "Recorded as Deviation #8.",
        }
        with open(OUT_DISCOVERY, "w") as jf:
            json.dump({"summary": summary, "records": records}, jf, indent=2, default=str)
        _log(fh, f"[write] {OUT_DISCOVERY}")

        if missing_critical:
            _log(fh, "[HALT] missing critical inputs — Stage 2 blocked per §3")
            for entry in missing_critical[:10]:
                _log(fh, f"       {entry}")
            raise RuntimeError(f"Discovery HALT: {len(missing_critical)} missing critical inputs")

        _log(fh, "=== discovery OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
