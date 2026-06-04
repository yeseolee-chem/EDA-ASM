"""Stage 3.10 — Build the unified Phase 1 HDF5 dataset.

Reads:
- outputs/phase1/.tmp/<reaction_id>.npz (per-reaction 5-point bundle)
- fragments_final.json
- bond_changes.json
- selected_reactions.csv

Writes:
- outputs/phase1/phase1_output.h5
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .logging_setup import get_logger, log_header
from .paths import (
    BOND_CHANGES_JSON,
    FRAGMENTS_FINAL_JSON,
    PHASE1_H5,
    SELECTED_CSV,
    TMP_DIR,
    ensure_dirs,
)

H5_GZIP = 4  # spec section 7: compress to keep file size reasonable


def _str_dtype(values: list[str]) -> np.ndarray:
    return np.array(values, dtype=h5py.string_dtype(encoding="utf-8"))


def _attrs_for_reaction(rid: str, frag: dict, bonds: dict, selected_row: pd.Series) -> dict:
    attrs = {
        "source": str(selected_row["source"]),
        "case": str(frag["case"]),
        "n_atoms": int(selected_row["n_atoms_max"]),
        "total_charge": float(selected_row["total_charge"]),
        "multiplicity": 1,
        "halo8_activation_energy": float(selected_row["activation_energy"]),
        "halo8_energy_R": float(selected_row["energy_R"]),
        "halo8_energy_TS": float(selected_row["energy_TS"]),
        "halo8_energy_P": float(selected_row["energy_P"]),
        "halo8_n_snapshots": int(selected_row["n_snapshots"]),
        "halo8_ts_frame_idx": int(selected_row["ts_frame_idx"]),
        "halo8_formula": str(selected_row["formula"]),
        "frag1_smiles": frag.get("frag1_smiles") or "",
        "frag2_smiles": frag.get("frag2_smiles") or "",
        "frag1_charge": int(frag.get("frag1_charge", 0)),
        "frag2_charge": int(frag.get("frag2_charge", 0)),
        "frag1_multiplicity": int(frag.get("frag1_multiplicity", 1)),
        "frag2_multiplicity": int(frag.get("frag2_multiplicity", 1)),
        "frag1_formula": str(frag.get("frag1_formula", "")),
        "frag2_formula": str(frag.get("frag2_formula", "")),
        "review_status": str(frag.get("review_status", "auto_accepted")),
        "auto_confidence": float(frag.get("auto_confidence") or 0.0),
        "rationale": str(frag.get("rationale", "")),
    }
    return attrs


def run(
    fragments_final: Path | None = None,
    selected_csv: Path | None = None,
    bond_changes_json: Path | None = None,
    output_h5: Path | None = None,
) -> Path:
    ensure_dirs()
    log = get_logger("phase1.stage3_10")
    log_header(log, "3.10 HDF5 integration")
    if fragments_final is None:
        fragments_final = FRAGMENTS_FINAL_JSON
    if selected_csv is None:
        selected_csv = SELECTED_CSV
    if bond_changes_json is None:
        bond_changes_json = BOND_CHANGES_JSON
    if output_h5 is None:
        output_h5 = PHASE1_H5

    frags = json.loads(fragments_final.read_text())
    selected = pd.read_csv(selected_csv).set_index("reaction_id")
    bonds = json.loads(bond_changes_json.read_text())
    log.info("Will write %d reactions to %s", len(frags), output_h5)

    written = 0
    skipped: list[str] = []
    with h5py.File(output_h5, "w") as h5:
        meta = h5.create_group("metadata")
        meta.attrs["creation_date"] = dt.datetime.now(dt.timezone.utc).isoformat()
        meta.attrs["seed"] = int(selected["seed"].iloc[0]) if "seed" in selected.columns else -1
        meta.attrs["n_reactions"] = len(frags)
        meta.attrs["adf_version"] = "tbd-phase0"

        rgroup = h5.create_group("reactions")
        for rid, frag in frags.items():
            npz_path = TMP_DIR / f"{rid}.npz"
            if not npz_path.exists():
                log.warning("missing npz for %s — skip", rid)
                skipped.append(rid)
                continue
            if rid not in selected.index:
                log.warning("not in selected csv: %s", rid)
                skipped.append(rid)
                continue

            with np.load(npz_path, allow_pickle=True) as data:
                bundle = {k: data[k] for k in data.files}

            grp = rgroup.create_group(rid)
            sel_row = selected.loc[rid]
            for k, v in _attrs_for_reaction(rid, frag, bonds.get(rid, {}), sel_row).items():
                grp.attrs[k] = v

            grp.create_dataset("numbers", data=bundle["numbers"].astype(np.int32))
            grp.create_dataset("coords_5pts", data=bundle["coords_5pts"].astype(np.float64), compression="gzip", compression_opts=H5_GZIP)
            grp.create_dataset("energies_5pts", data=bundle["energies_5pts"].astype(np.float64))
            grp.create_dataset("forces_5pts", data=bundle["forces_5pts"].astype(np.float32), compression="gzip", compression_opts=H5_GZIP)
            grp.create_dataset("homo_5pts", data=bundle["homo_5pts"].astype(np.float64))
            grp.create_dataset("lumo_5pts", data=bundle["lumo_5pts"].astype(np.float64))
            grp.create_dataset("homo_idx_5pts", data=bundle["homo_idx_5pts"].astype(np.int32))
            grp.create_dataset("lumo_idx_5pts", data=bundle["lumo_idx_5pts"].astype(np.int32))
            grp.create_dataset("mulliken_5pts", data=bundle["mulliken_5pts"].astype(np.float32), compression="gzip", compression_opts=H5_GZIP)
            grp.create_dataset("lowdin_5pts", data=bundle["lowdin_5pts"].astype(np.float32), compression="gzip", compression_opts=H5_GZIP)
            grp.create_dataset("dipole_5pts", data=bundle["dipole_5pts"].astype(np.float64))
            grp.create_dataset("dispersion_5pts", data=bundle["dispersion_5pts"].astype(np.float64))
            grp.create_dataset("zeta_values", data=bundle["zeta_values"].astype(np.float64))
            grp.create_dataset("frame_indices", data=bundle["frame_indices"].astype(np.int32))
            grp.create_dataset("dand_ids", data=_str_dtype([str(x) for x in bundle["dand_ids"]]))

            grp.create_dataset("frag1_atoms", data=np.array(frag["frag1_atoms"], dtype=np.int32))
            grp.create_dataset("frag2_atoms", data=np.array(frag["frag2_atoms"], dtype=np.int32))
            h_caps = frag.get("h_caps", [])
            if h_caps:
                grp.create_dataset(
                    "h_caps",
                    data=np.array([h["h_position"] for h in h_caps], dtype=np.float64),
                )
                grp.create_dataset(
                    "h_caps_attached_to",
                    data=np.array([h["attached_to_atom"] for h in h_caps], dtype=np.int32),
                )
                grp.create_dataset(
                    "h_caps_from_bond",
                    data=np.array([h["from_broken_bond"] for h in h_caps], dtype=np.int32),
                )
            else:
                grp.create_dataset("h_caps", data=np.zeros((0, 3), dtype=np.float64))
                grp.create_dataset("h_caps_attached_to", data=np.zeros((0,), dtype=np.int32))

            bd = bonds.get(rid, {})
            grp.create_dataset(
                "bonds_broken",
                data=np.array(bd.get("bonds_broken", []), dtype=np.int32).reshape(-1, 2)
                if bd.get("bonds_broken")
                else np.zeros((0, 2), dtype=np.int32),
            )
            grp.create_dataset(
                "bonds_formed",
                data=np.array(bd.get("bonds_formed", []), dtype=np.int32).reshape(-1, 2)
                if bd.get("bonds_formed")
                else np.zeros((0, 2), dtype=np.int32),
            )

            written += 1

        # sampling_metadata: replicate the selected csv as a structured table.
        sm = h5.create_group("sampling_metadata")
        for col in selected.columns:
            try:
                vals = selected[col].to_numpy()
                if vals.dtype.kind in ("O", "U"):
                    sm.create_dataset(col, data=_str_dtype([str(x) for x in vals]))
                else:
                    sm.create_dataset(col, data=vals)
            except Exception as e:  # noqa: BLE001
                log.warning("sampling_metadata column %s failed: %s", col, e)
        sm.create_dataset("reaction_id", data=_str_dtype([str(x) for x in selected.index]))
    log.info("Wrote %s with %d reactions (skipped=%d)", output_h5, written, len(skipped))
    return output_h5
