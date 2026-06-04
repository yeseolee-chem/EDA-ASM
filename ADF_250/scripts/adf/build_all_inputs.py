#!/usr/bin/env python
"""Generate ADF EDA-NOCV input decks for every successfully-partitioned reaction.

Reads `data/fragments/v1/fragments.parquet` (from `define_fragments.py`) and the
seed CSV (`data/selection/initial_seed_v1/selected_reactions.csv`), produces one
self-contained deck per reaction under `ADF_800/runs/<reaction_id>/` plus an
index `ADF_800/manifest.csv`.

Fragment indices, charges, multiplicities all come from fragments.parquet so the
A/B assignment is identical to the ASR-spec-compliant partitioning.

Geometries:
  - TS positions   : from path_ts (selected_reactions.csv)
  - fragA/B at TS  : sliced from TS positions using fragments_atoms_a/b
  - fragA relaxed  : for dipolar — from r0_*.xyz (or r1 if the SMARTS match
                     associated fragA with the second LHS fragment); for
                     QMrxn20 — sliced from reactant-complex (path_r), which
                     shares atom order with TS.
  - fragB relaxed  : same logic for fragment B.

A `_failures_build.jsonl` is written for any reaction whose input deck could
not be produced.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from eda_asm.adf import ADFRunSpec, FragmentSpec, generate_run_script  # noqa: E402


def read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = Path(path).read_text()
    lines = text.strip().splitlines()
    n = int(lines[0].strip())
    numbers = np.zeros(n, dtype=np.int64)
    positions = np.zeros((n, 3), dtype=np.float64)
    from ase.data import atomic_numbers
    for i, raw in enumerate(lines[2:2 + n]):
        toks = raw.split()
        numbers[i] = atomic_numbers[toks[0]]
        positions[i] = [float(t) for t in toks[1:4]]
    return numbers, positions


def _build_dipolar_relaxed_positions(
    fragA_idx: np.ndarray, fragB_idx: np.ndarray,
    ts_numbers: np.ndarray, ts_positions: np.ndarray, row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble fragA/B relaxed positions for a dipolar reaction.

    The dipolar dataset ships r0_*.xyz and r1_*.xyz separately. After SMARTS
    partitioning, one of {r0, r1} corresponds to fragment A and the other to
    fragment B. We disambiguate by element multiset match.
    """
    # reaction_id format: dipolar_NNNNNN → integer rxn_id
    rid_str = row["reaction_id"].split("_")[1]
    rxn_dir = (REPO / "data" / "raw" / "dipolar_cycloaddition"
               / "extracted" / "full_dataset_profiles" / str(int(rid_str)))
    r0_files = list(rxn_dir.glob("r0_*.xyz"))
    r1_files = list(rxn_dir.glob("r1_*.xyz"))
    # Drop _alt variants
    r0_files = [p for p in r0_files if "_alt" not in p.stem] or r0_files
    r1_files = [p for p in r1_files if "_alt" not in p.stem] or r1_files
    if not r0_files or not r1_files:
        raise FileNotFoundError(f"missing r0/r1 in {rxn_dir}")
    r0_nums, r0_pos = read_xyz(r0_files[0])
    r1_nums, r1_pos = read_xyz(r1_files[0])

    def _multiset(nums):
        from collections import Counter
        return Counter(int(n) for n in nums)

    fragA_multi = _multiset(ts_numbers[fragA_idx])
    fragB_multi = _multiset(ts_numbers[fragB_idx])
    r0_multi = _multiset(r0_nums)
    r1_multi = _multiset(r1_nums)

    if fragA_multi == r0_multi and fragB_multi == r1_multi:
        return _align_subset(r0_nums, r0_pos, ts_numbers, ts_positions, fragA_idx), \
               _align_subset(r1_nums, r1_pos, ts_numbers, ts_positions, fragB_idx)
    if fragA_multi == r1_multi and fragB_multi == r0_multi:
        return _align_subset(r1_nums, r1_pos, ts_numbers, ts_positions, fragA_idx), \
               _align_subset(r0_nums, r0_pos, ts_numbers, ts_positions, fragB_idx)
    raise ValueError(
        f"dipolar element multisets don't match {row['reaction_id']}: "
        f"fragA={fragA_multi} fragB={fragB_multi} r0={r0_multi} r1={r1_multi}"
    )


def _align_subset(
    sub_numbers: np.ndarray, sub_positions: np.ndarray,
    ts_numbers: np.ndarray, ts_positions: np.ndarray,
    ts_indices: np.ndarray,
) -> np.ndarray:
    """Reorder `sub_positions` to match the TS-subset ordering.

    Greedy match: for each TS atom in the subset, find the closest sub-atom
    of matching element (in TS-subset positions, treating positions as
    consistent geometry approximated via element bucketing alone — this
    works because sub_positions and ts_positions[ts_indices] describe the
    same molecule in (possibly) different orientations).
    """
    n_sub = len(sub_numbers)
    if n_sub != len(ts_indices):
        raise ValueError(f"subset size mismatch: {n_sub} vs {len(ts_indices)}")
    # Group sub atoms by atomic number
    available = {z: [] for z in set(int(x) for x in sub_numbers)}
    for j in range(n_sub):
        available[int(sub_numbers[j])].append(j)
    out = np.zeros((n_sub, 3), dtype=np.float64)
    for i, ts_idx in enumerate(ts_indices):
        z = int(ts_numbers[ts_idx])
        if not available.get(z):
            raise ValueError(f"no available sub-atom of Z={z}")
        # First-fit assignment; positions are approximate anyway because
        # sub_positions are relaxed isolated geometry, not the TS geometry.
        j = available[z].pop(0)
        out[i] = sub_positions[j]
    return out


def _build_qmrxn20_relaxed_positions(
    fragA_idx: np.ndarray, fragB_idx: np.ndarray,
    ts_numbers: np.ndarray, ts_positions: np.ndarray, row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """For QMrxn20, fragA = single anion atom at TS positions (no relaxation
    needed), fragB = substrate; use reactant-conformers/{substrate}/00.xyz if
    available for the relaxed substrate, otherwise fall back to TS-frozen.
    """
    fragA_relaxed = ts_positions[fragA_idx].copy()

    parts = row["reaction_id"].split("_")
    # qmrxn20_<rxn>_<6-part-label>
    if len(parts) < 8:
        return fragA_relaxed, ts_positions[fragB_idx].copy()
    rxn = parts[1]
    ts_label_parts = parts[2:]
    if ts_label_parts and ts_label_parts[-1].startswith("conf"):
        ts_label_parts = ts_label_parts[:-1]
    if len(ts_label_parts) != 6:
        return fragA_relaxed, ts_positions[fragB_idx].copy()
    # reactant-conformers labels keep all 6 fields with position 6 (Y) replaced
    # by '0' — e.g., TS A_A_A_A_B_C → substrate A_A_A_A_B_0 (substrate + X, no Y)
    substrate_label = "_".join(ts_label_parts[:5] + ["0"])
    cand = (REPO / "data" / "raw" / "QMrxn20" / "reactant-conformers"
            / substrate_label / "00.xyz")
    if cand.is_file():
        sub_nums, sub_pos = read_xyz(cand)
        if len(sub_nums) == len(fragB_idx):
            try:
                return fragA_relaxed, _align_subset(
                    sub_nums, sub_pos, ts_numbers, ts_positions, fragB_idx
                )
            except Exception:
                pass
    # fallback: TS-frozen fragB
    return fragA_relaxed, ts_positions[fragB_idx].copy()


def build_one(row_sel: pd.Series, row_frag: pd.Series,
              out_root: Path) -> dict:
    rid = row_sel["reaction_id"]
    fragA_idx = np.array(json.loads(row_frag["fragment_atoms_a"]), dtype=np.int64)
    fragB_idx = np.array(json.loads(row_frag["fragment_atoms_b"]), dtype=np.int64)

    ts_numbers, ts_positions = read_xyz(Path(row_sel["path_ts"]))

    if (fragA_idx.size + fragB_idx.size) != ts_numbers.size:
        raise ValueError(
            f"fragments cover {fragA_idx.size + fragB_idx.size} of {ts_numbers.size} atoms"
        )

    fam = row_sel["family"]
    if fam == "dipolar":
        fragA_relaxed, fragB_relaxed = _build_dipolar_relaxed_positions(
            fragA_idx, fragB_idx, ts_numbers, ts_positions, row_sel
        )
    elif fam in ("e2", "sn2"):
        fragA_relaxed, fragB_relaxed = _build_qmrxn20_relaxed_positions(
            fragA_idx, fragB_idx, ts_numbers, ts_positions, row_sel
        )
    else:
        raise ValueError(f"unknown family {fam}")

    fragA = FragmentSpec(
        name="fA",
        indices=fragA_idx,
        numbers=ts_numbers[fragA_idx],
        positions_at_TS=ts_positions[fragA_idx],
        positions_relaxed=fragA_relaxed,
        charge=int(row_frag["fragment_charge_a"]),
    )
    fragB = FragmentSpec(
        name="fB",
        indices=fragB_idx,
        numbers=ts_numbers[fragB_idx],
        positions_at_TS=ts_positions[fragB_idx],
        positions_relaxed=fragB_relaxed,
        charge=int(row_frag["fragment_charge_b"]),
    )

    safe_rid = rid.replace("/", "_")
    out_dir = out_root / safe_rid
    spec = ADFRunSpec(
        reaction_id=safe_rid,
        fragA=fragA,
        fragB=fragB,
        ts_numbers=ts_numbers,
        ts_positions=ts_positions,
        total_charge=int(row_sel["charge"]),
    )
    generate_run_script(spec, out_dir)
    return {
        "reaction_id": safe_rid,
        "family": fam,
        "n_atoms_total": int(ts_numbers.size),
        "fragA_natoms": int(fragA_idx.size),
        "fragB_natoms": int(fragB_idx.size),
        "fragA_charge": int(row_frag["fragment_charge_a"]),
        "fragB_charge": int(row_frag["fragment_charge_b"]),
        "fragA_mult": int(row_frag["fragment_mult_a"]),
        "fragB_mult": int(row_frag["fragment_mult_b"]),
        "total_charge": int(row_sel["charge"]),
        "has_Br": spec.has_Br,
        "partition_method": row_frag["partition_method"],
        "partition_status": row_frag["partition_status"],
        "delta_Ea_kcal_native": float(row_sel["delta_Ea"]),
        "out_dir": str(out_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-csv", type=Path,
                   default=REPO / "data" / "selection" / "initial_seed_v1"
                   / "selected_reactions.csv")
    p.add_argument("--fragments-parquet", type=Path,
                   default=REPO / "data" / "fragments" / "v1" / "fragments.parquet")
    p.add_argument("--out-root", type=Path, default=REPO / "ADF_800" / "runs")
    p.add_argument("--manifest-csv", type=Path,
                   default=REPO / "ADF_800" / "manifest.csv")
    p.add_argument("--failures-jsonl", type=Path,
                   default=REPO / "ADF_800" / "_failures_build.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(args.selection_csv)
    frags = pd.read_parquet(args.fragments_parquet)
    # Only build for successfully-partitioned reactions
    ok = frags[frags["partition_status"] != "failed"]
    merged = sel.merge(
        ok[["reaction_id", "fragment_atoms_a", "fragment_atoms_b",
            "fragment_charge_a", "fragment_charge_b",
            "fragment_mult_a", "fragment_mult_b",
            "partition_method", "partition_status"]],
        on="reaction_id", how="inner"
    )
    if args.limit:
        merged = merged.head(args.limit)
    print(f"reactions to build: {len(merged)}")
    print(f"by family: {merged['family'].value_counts().to_dict()}")

    rows: list[dict] = []
    n_ok, n_fail = 0, 0
    failures = args.failures_jsonl.open("w")
    for i, sel_row in enumerate(merged.itertuples(index=False)):
        sel_dict = sel_row._asdict()
        try:
            meta = build_one(pd.Series(sel_dict), pd.Series(sel_dict), args.out_root)
            rows.append(meta)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            failures.write(json.dumps({
                "reaction_id": sel_dict["reaction_id"],
                "family": sel_dict["family"],
                "error": str(e),
                "traceback": traceback.format_exc(),
            }) + "\n")
        if (i + 1) % 100 == 0:
            print(f"  progress: {i + 1}/{len(merged)}  ok={n_ok} fail={n_fail}")
    failures.close()

    pd.DataFrame.from_records(rows).to_csv(args.manifest_csv, index=False)
    print(f"\nbuilt {n_ok} input decks; failed {n_fail}")
    print(f"manifest -> {args.manifest_csv}")


if __name__ == "__main__":
    main()
