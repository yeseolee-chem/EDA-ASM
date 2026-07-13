"""Assemble the ep29-NequIP vs MACE-OFF backbone-comparison report.

Reads the four ``summary.json`` files (B0 train_cv, M1 train_cv, B0 LC, M1 LC)
for each backbone, plus the labels parquet (to compute per-component label std),
and emits the comparison table specified in ASR_Backbone_Comparison_Spec_v1.0
§7 / §8 to a single JSON.

Output: ``outputs/asr_v1/backbone_comparison/summary.json`` with:
  - per-component MAE table (component, ep29 B0/M1, MACE-OFF B0/M1, label std)
  - learning-curve table (N_train, overall MAE for each model × backbone)
  - provenance: feature_dim per backbone, fold-seed match assertion,
    mace-torch version (from manifest sidecar if present),
    a "decision" string per spec §8.

No retraining happens here — this script is a pure aggregator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


_COMPONENTS = (
    "E_strain_kcal", "Pauli_kcal", "V_elst_kcal", "E_orb_kcal", "E_disp_kcal",
)


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _maybe_read_json(p: Path) -> Optional[dict]:
    try:
        return _read_json(p)
    except FileNotFoundError:
        return None


def _label_std(labels_parquet: Path, family: str = "dipolar") -> dict[str, float]:
    df = pd.read_parquet(labels_parquet)
    df = df[df["family"] == family].reset_index(drop=True)
    return {c: float(df[c].std(ddof=1)) for c in _COMPONENTS}


def _fmt(vals: list[float]) -> list[float]:
    return [round(float(v), 4) for v in vals]


def _per_component_block(summary: Optional[dict]) -> Optional[dict]:
    if summary is None:
        return None
    return {
        "per_component_mae_kcal_mean": _fmt(summary["per_component_mae_kcal_mean"]),
        "per_component_mae_kcal_std":  _fmt(summary["per_component_mae_kcal_std"]),
        "overall_mae_kcal_mean": round(float(summary["overall_mae_kcal_mean"]), 4),
        "overall_mae_kcal_std":  round(float(summary["overall_mae_kcal_std"]), 4),
        "feature_dim": int(summary["feature_dim"]),
        "n_reactions": int(summary["n_reactions"]),
        "cv_seed": int(summary["config"]["cv"]["seed"]),
        "ensemble_base_seed": int(summary["config"]["ensemble"]["base_seed"]),
    }


def _learning_curve_block(summary: Optional[dict]) -> Optional[list[dict]]:
    if summary is None:
        return None
    out = []
    for r in summary["curve"]:
        out.append({
            "N_train": int(r["N_train"]),
            "overall_mae_kcal_mean": round(float(r["overall_mae_kcal_mean"]), 4),
            "overall_mae_kcal_std":  round(float(r["overall_mae_kcal_std"]), 4),
            "per_component_mae_kcal_mean": _fmt(r["per_component_mae_kcal_mean"]),
        })
    return out


def _decision(ep29_overall: Optional[float],
              mace_overall: Optional[float],
              ep29_disp: Optional[float],
              mace_disp: Optional[float],
              ep29_orb: Optional[float],
              mace_orb: Optional[float]) -> str:
    """Apply the spec §1 / §8 decision criteria to the best (M1) result."""
    if mace_overall is None or ep29_overall is None:
        return "indeterminate: missing M1 summary on one side"
    gap = ep29_overall - mace_overall
    # §1: ≤ 7-8 kcal/mol overall is a clear win; ~ep29 ± fold std is a tie.
    if mace_overall <= 8.0 and gap >= 2.0:
        return ("MACE-OFF wins overall: adopt as backbone, re-baseline, "
                "then consider AL/Hammett")
    # Component-only win.
    disp_gain = ((ep29_disp or 0) - (mace_disp or 0))
    orb_gain = ((ep29_orb or 0) - (mace_orb or 0))
    if (disp_gain >= 1.0 or orb_gain >= 1.0) and gap >= 0:
        return ("MACE-OFF wins on hard components only "
                "(E_disp / E_orb): adopt and report component-wise gain")
    if abs(gap) < 1.0:
        return ("MACE-OFF ≈ ep29 (within ~1 kcal/mol of overall MAE): "
                "feature quality is NOT the bottleneck — investigate "
                "label count, label noise, or task framing; do NOT resume "
                "NequIP pretraining")
    return f"inconclusive: overall gap {gap:+.2f} kcal/mol — inspect components manually"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep29-root", default="outputs/asr_v1",
                    help="root containing b0/, m1/, learning_curve_b0/, "
                         "learning_curve_m1/")
    ap.add_argument("--maceoff-root", default="outputs/asr_v1",
                    help="root containing maceoff_b0/, maceoff_m1/, "
                         "maceoff_learning_curve_b0/, maceoff_learning_curve_m1/")
    ap.add_argument("--ep29-b0-dir", default="b0")
    ap.add_argument("--ep29-m1-dir", default="m1")
    ap.add_argument("--ep29-lc-b0-dir", default="learning_curve_b0")
    ap.add_argument("--ep29-lc-m1-dir", default="learning_curve_m1")
    ap.add_argument("--maceoff-b0-dir", default="maceoff_b0")
    ap.add_argument("--maceoff-m1-dir", default="maceoff_m1")
    ap.add_argument("--maceoff-lc-b0-dir", default="maceoff_learning_curve_b0")
    ap.add_argument("--maceoff-lc-m1-dir", default="maceoff_learning_curve_m1")
    ap.add_argument("--maceoff-manifest", default=None,
                    help="path to the feature-cache manifest sidecar "
                         "(default: inferred from MACE-OFF M1 config)")
    ap.add_argument("--labels-parquet", default="ADF_250/adf_outputs/parsed/asr_labels.parquet")
    ap.add_argument("--family", default="dipolar")
    ap.add_argument("--out", default="outputs/asr_v1/backbone_comparison/summary.json")
    args = ap.parse_args()

    ep29 = Path(args.ep29_root)
    mace = Path(args.maceoff_root)

    ep29_b0 = _maybe_read_json(ep29 / args.ep29_b0_dir / "summary.json")
    ep29_m1 = _maybe_read_json(ep29 / args.ep29_m1_dir / "summary.json")
    ep29_lc_b0 = _maybe_read_json(ep29 / args.ep29_lc_b0_dir / "summary.json")
    ep29_lc_m1 = _maybe_read_json(ep29 / args.ep29_lc_m1_dir / "summary.json")

    mace_b0 = _maybe_read_json(mace / args.maceoff_b0_dir / "summary.json")
    mace_m1 = _maybe_read_json(mace / args.maceoff_m1_dir / "summary.json")
    mace_lc_b0 = _maybe_read_json(mace / args.maceoff_lc_b0_dir / "summary.json")
    mace_lc_m1 = _maybe_read_json(mace / args.maceoff_lc_m1_dir / "summary.json")

    label_std = _label_std(Path(args.labels_parquet), family=args.family)

    # ---- Build per-component table ------------------------------------------------
    per_component_table: list[dict[str, Any]] = []
    for ci, comp in enumerate(_COMPONENTS):
        row = {"component": comp, "label_std_kcal": round(label_std[comp], 4)}
        for tag, summary in (
            ("ep29_b0", ep29_b0), ("ep29_m1", ep29_m1),
            ("maceoff_b0", mace_b0), ("maceoff_m1", mace_m1),
        ):
            if summary is None:
                row[tag] = None
                continue
            row[tag] = {
                "mae_mean": round(float(summary["per_component_mae_kcal_mean"][ci]), 4),
                "mae_std":  round(float(summary["per_component_mae_kcal_std"][ci]), 4),
            }
        per_component_table.append(row)

    # ---- Fold-seed parity assertion ----------------------------------------------
    seeds_seen = {
        "ep29_b0_cv_seed":    ep29_b0 and ep29_b0["config"]["cv"]["seed"],
        "ep29_m1_cv_seed":    ep29_m1 and ep29_m1["config"]["cv"]["seed"],
        "maceoff_b0_cv_seed": mace_b0 and mace_b0["config"]["cv"]["seed"],
        "maceoff_m1_cv_seed": mace_m1 and mace_m1["config"]["cv"]["seed"],
    }
    nonnull_seeds = {k: v for k, v in seeds_seen.items() if v is not None}
    seeds_match = len(set(nonnull_seeds.values())) <= 1

    # Cross-check N (both must be the same dataset).
    n_seen = {
        "ep29_b0":    ep29_b0 and ep29_b0["n_reactions"],
        "ep29_m1":    ep29_m1 and ep29_m1["n_reactions"],
        "maceoff_b0": mace_b0 and mace_b0["n_reactions"],
        "maceoff_m1": mace_m1 and mace_m1["n_reactions"],
    }
    nonnull_n = {k: v for k, v in n_seen.items() if v is not None}
    n_match = len(set(nonnull_n.values())) <= 1

    # ---- MACE-OFF manifest (feature_dim, mace-torch version) ---------------------
    mace_manifest_path: Optional[Path] = None
    if args.maceoff_manifest:
        mace_manifest_path = Path(args.maceoff_manifest)
    elif mace_m1 is not None:
        cache = Path(mace_m1["config"]["feature_cache"])
        mace_manifest_path = cache.with_suffix(".manifest.json")
    mace_manifest = _maybe_read_json(mace_manifest_path) if mace_manifest_path else None

    # ---- Decision per §8 ---------------------------------------------------------
    def _comp_idx(name: str) -> int:
        return _COMPONENTS.index(name)
    decision = _decision(
        ep29_overall=ep29_m1 and float(ep29_m1["overall_mae_kcal_mean"]),
        mace_overall=mace_m1 and float(mace_m1["overall_mae_kcal_mean"]),
        ep29_disp=ep29_m1 and float(ep29_m1["per_component_mae_kcal_mean"][_comp_idx("E_disp_kcal")]),
        mace_disp=mace_m1 and float(mace_m1["per_component_mae_kcal_mean"][_comp_idx("E_disp_kcal")]),
        ep29_orb=ep29_m1 and float(ep29_m1["per_component_mae_kcal_mean"][_comp_idx("E_orb_kcal")]),
        mace_orb=mace_m1 and float(mace_m1["per_component_mae_kcal_mean"][_comp_idx("E_orb_kcal")]),
    )

    # ---- Assemble ----------------------------------------------------------------
    out_payload = {
        "spec": "ASR_Backbone_Comparison_Spec_v1.0",
        "component_order": list(_COMPONENTS),
        "per_component_table": per_component_table,
        "learning_curve": {
            "ep29_b0":    _learning_curve_block(ep29_lc_b0),
            "ep29_m1":    _learning_curve_block(ep29_lc_m1),
            "maceoff_b0": _learning_curve_block(mace_lc_b0),
            "maceoff_m1": _learning_curve_block(mace_lc_m1),
        },
        "controlled_comparison": {
            "cv_seeds": seeds_seen,
            "fold_seed_match": bool(seeds_match),
            "n_reactions_per_run": n_seen,
            "n_reactions_match": bool(n_match),
        },
        "backbone_summary": {
            "ep29": _per_component_block(ep29_m1),
            "maceoff": _per_component_block(mace_m1),
            "maceoff_manifest": mace_manifest,
        },
        "decision": decision,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"[compare_backbones] wrote {out_path}")
    print(f"[compare_backbones] fold-seed match: {seeds_match}  "
          f"({nonnull_seeds})")
    print(f"[compare_backbones] decision: {decision}")
    print()
    print("[compare_backbones] per-component table (kcal/mol):")
    hdr = ("component", "label_std", "ep29 B0", "ep29 M1", "MACE B0", "MACE M1")
    print("  " + " | ".join(f"{c:>14s}" for c in hdr))
    for row in per_component_table:
        def fmt_cell(v):
            if v is None:
                return "       -      "
            return f"{v['mae_mean']:>6.2f}±{v['mae_std']:<5.2f}"
        print("  " + " | ".join((
            f"{row['component']:>14s}",
            f"{row['label_std_kcal']:>14.2f}",
            f"{fmt_cell(row['ep29_b0']):>14s}",
            f"{fmt_cell(row['ep29_m1']):>14s}",
            f"{fmt_cell(row['maceoff_b0']):>14s}",
            f"{fmt_cell(row['maceoff_m1']):>14s}",
        )))


if __name__ == "__main__":
    main()
