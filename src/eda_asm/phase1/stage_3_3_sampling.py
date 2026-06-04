"""Stage 3.3 — Stratified sampling of 400 reactions.

Stratification grid (multiplicative):
- Source: T1x : Halo_F : Halo_Cl : Halo_Br = 190 : 70 : 70 : 70
- Heavy atoms: {5, 6, 7, 8} bins, equal share (25% each)
- Bond changes: {2-3, 4-6} bins, 60% / 40%
- Activation energy tertiles within source: low / mid / high, equal share (33%)

Cells whose population is too small are filled to the population, and the
shortfall is redistributed to neighbouring cells (priority: same source >
same heavy-atom > same bond-change bin).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_setup import get_logger, log_header
from .paths import (
    BOND_CHANGES_PARQUET,
    INDEX_PARQUET,
    SELECTED_CSV,
    BOND_CHANGES_JSON,
    ensure_dirs,
)

DEFAULT_SEED = 42

SOURCE_TARGETS = {
    "T1x": 190,
    "Halo_F": 70,
    "Halo_Cl": 70,
    "Halo_Br": 70,
}
HEAVY_BINS = [5, 6, 7, 8]            # exact heavy-atom counts
BOND_CHANGE_BINS = [(2, 3), (4, 6)]  # inclusive ranges
BOND_CHANGE_RATIOS = [0.6, 0.4]
EA_TERTILE_RATIOS = [1 / 3, 1 / 3, 1 / 3]


@dataclass(frozen=True)
class Cell:
    source: str
    heavy: int
    bond_bin: int   # 0 = 2-3, 1 = 4-6
    ea_tertile: int  # 0 = low, 1 = mid, 2 = high

    def label(self) -> str:
        bb = "2-3" if self.bond_bin == 0 else "4-6"
        eb = ("low", "mid", "high")[self.ea_tertile]
        return f"{self.source}|h={self.heavy}|bond={bb}|ea={eb}"


def _bond_bin(n_changes: int) -> int | None:
    for i, (lo, hi) in enumerate(BOND_CHANGE_BINS):
        if lo <= n_changes <= hi:
            return i
    return None


def _ea_tertile(ea: float, edges: tuple[float, float]) -> int:
    if ea < edges[0]:
        return 0
    if ea < edges[1]:
        return 1
    return 2


def _quotas() -> dict[Cell, int]:
    """Compute target count per (source, heavy, bond, tertile) cell."""
    quotas: dict[Cell, int] = {}
    for src, n_src in SOURCE_TARGETS.items():
        for heavy in HEAVY_BINS:
            for bi, br in enumerate(BOND_CHANGE_RATIOS):
                for ti, tr in enumerate(EA_TERTILE_RATIOS):
                    q = n_src * (1 / len(HEAVY_BINS)) * br * tr
                    quotas[Cell(src, heavy, bi, ti)] = q  # keep float; round later
    # Largest-remainder rounding per source so per-source totals match exactly.
    out: dict[Cell, int] = {}
    by_src: dict[str, list[Cell]] = defaultdict(list)
    for c in quotas:
        by_src[c.source].append(c)
    for src, cells in by_src.items():
        floats = np.array([quotas[c] for c in cells])
        ints = np.floor(floats).astype(int)
        remainder = SOURCE_TARGETS[src] - ints.sum()
        if remainder > 0:
            order = np.argsort(-(floats - ints))
            for k in order[:remainder]:
                ints[k] += 1
        for c, n in zip(cells, ints):
            out[c] = int(n)
    return out


def _build_bucketed(
    df: pd.DataFrame,
) -> tuple[dict[Cell, list[int]], dict[str, tuple[float, float]]]:
    """Return cell -> list of dataframe row indices, and per-source EA tertile edges."""
    df = df.reset_index(drop=True)
    tertile_edges: dict[str, tuple[float, float]] = {}
    for src in SOURCE_TARGETS:
        sub = df[df["source"] == src]["activation_energy"].to_numpy()
        if len(sub) == 0:
            tertile_edges[src] = (0.0, 0.0)
        else:
            tertile_edges[src] = (
                float(np.quantile(sub, 1 / 3)),
                float(np.quantile(sub, 2 / 3)),
            )
    bucket: dict[Cell, list[int]] = defaultdict(list)
    for i, row in df.iterrows():
        if row["source"] not in SOURCE_TARGETS:
            continue
        if int(row["n_heavy_atoms"]) not in HEAVY_BINS:
            continue
        bb = _bond_bin(int(row["n_bond_changes"]))
        if bb is None:
            continue
        ed = tertile_edges[row["source"]]
        if ed[0] == 0.0 and ed[1] == 0.0:
            continue
        et = _ea_tertile(float(row["activation_energy"]), ed)
        cell = Cell(row["source"], int(row["n_heavy_atoms"]), bb, et)
        bucket[cell].append(int(i))
    return bucket, tertile_edges


def _neighbor_cells(cell: Cell) -> list[Cell]:
    """Spec priority: same source > same heavy > same bond-bin.

    Returns candidates ordered by closeness (closest first), excluding `cell`.
    """
    out: list[Cell] = []
    for heavy in HEAVY_BINS:
        for bi in range(len(BOND_CHANGE_BINS)):
            for ti in range(3):
                c = Cell(cell.source, heavy, bi, ti)
                if c == cell:
                    continue
                # closeness score: same heavy + same bond bin + same tertile
                score = int(heavy == cell.heavy) + int(bi == cell.bond_bin) + int(ti == cell.ea_tertile)
                out.append((-score, c))
    out.sort(key=lambda x: x[0])
    return [c for _, c in out]


def _sample_with_redistribution(
    bucket: dict[Cell, list[int]],
    quotas: dict[Cell, int],
    rng: np.random.Generator,
    log,
) -> tuple[set[int], dict[Cell, dict]]:
    chosen: set[int] = set()
    cell_log: dict[Cell, dict] = {}
    deficit: dict[Cell, int] = {}
    # Pass 1: take what each cell can supply.
    for cell, q in quotas.items():
        pool = bucket.get(cell, [])
        avail = len(pool)
        take = min(avail, q)
        if take > 0:
            picks = rng.choice(pool, size=take, replace=False).tolist()
            chosen.update(picks)
        if take < q:
            deficit[cell] = q - take
        cell_log[cell] = {
            "quota": q,
            "available": avail,
            "taken_pass1": take,
            "filled_from_neighbors": 0,
        }
    # Pass 2: redistribute deficits to neighboring cells (same source preferred).
    for cell, missing in deficit.items():
        for nb in _neighbor_cells(cell):
            if missing <= 0:
                break
            pool = [i for i in bucket.get(nb, []) if i not in chosen]
            if not pool:
                continue
            take = min(len(pool), missing)
            picks = rng.choice(pool, size=take, replace=False).tolist()
            chosen.update(picks)
            missing -= take
            cell_log[cell]["filled_from_neighbors"] += take
            cell_log.setdefault(nb, {}).setdefault("borrowed_for", []).append(
                {"cell": cell.label(), "n": take}
            )
        if missing > 0:
            cell_log[cell]["unfilled"] = missing
            log.warning("Cell %s still short by %d after redistribution", cell.label(), missing)
    return chosen, cell_log


def run(
    *,
    seed: int = DEFAULT_SEED,
    index_parquet: Path | None = None,
    bond_changes_parquet: Path | None = None,
    output_csv: Path | None = None,
    output_bond_json: Path | None = None,
) -> dict:
    ensure_dirs()
    log = get_logger("phase1.stage3_3")
    log_header(log, "3.3 Stratified sampling", seed=seed)
    if index_parquet is None:
        index_parquet = INDEX_PARQUET
    if bond_changes_parquet is None:
        bond_changes_parquet = BOND_CHANGES_PARQUET
    if output_csv is None:
        output_csv = SELECTED_CSV
    if output_bond_json is None:
        output_bond_json = BOND_CHANGES_JSON

    idx = pd.read_parquet(index_parquet)
    bc = pd.read_parquet(bond_changes_parquet)

    df = idx.merge(
        bc[
            [
                "reaction_id",
                "bonds_broken",
                "bonds_formed",
                "n_bond_changes",
                "n_components_R",
            ]
        ],
        on="reaction_id",
        how="inner",
    )
    log.info("Trajectories with bond-change info: %d", len(df))

    # Mandatory filters per spec: must have at least one bond change, must have
    # an interior TS, and must have enough frames to support 5-point extraction.
    pre_n = len(df)
    df = df[df["n_bond_changes"] >= 1]
    df = df[df["interior_ts"]]
    df = df[df["n_snapshots"] >= 5]
    log.info("After filters (n_bond_changes>=1, interior_ts, n_snapshots>=5): %d (-%d)", len(df), pre_n - len(df))

    # Restrict to the four allowed sources up front.
    df = df[df["source"].isin(SOURCE_TARGETS)].reset_index(drop=True)
    log.info("After source filter: %d trajectories across sources:\n%s", len(df), df["source"].value_counts().to_string())

    quotas = _quotas()
    bucket, tertile_edges = _build_bucketed(df)
    log.info("Per-source EA tertile edges (eV): %s", tertile_edges)

    rng = np.random.default_rng(seed)
    chosen_idx, cell_log = _sample_with_redistribution(bucket, quotas, rng, log)
    log.info("Initial picks: %d (target 400)", len(chosen_idx))

    # Final top-up if still short (rare): draw from any remaining matching source.
    target_total = sum(SOURCE_TARGETS.values())
    if len(chosen_idx) < target_total:
        per_source_target = SOURCE_TARGETS
        per_source_have = {s: 0 for s in SOURCE_TARGETS}
        for i in chosen_idx:
            per_source_have[df.iloc[i]["source"]] += 1
        for src, want in per_source_target.items():
            need = want - per_source_have[src]
            if need <= 0:
                continue
            pool = [i for i in df.index[df["source"] == src] if i not in chosen_idx]
            if not pool:
                log.error("Source %s short by %d and no fallback rows available — STOP", src, need)
                raise RuntimeError(f"sampling shortfall: {src} missing {need}")
            take = min(need, len(pool))
            picks = rng.choice(pool, size=take, replace=False).tolist()
            chosen_idx.update(picks)
            log.info("Top-up for %s: +%d", src, take)

    if len(chosen_idx) != target_total:
        log.error("Sampled %d != %d, STOP", len(chosen_idx), target_total)
        raise RuntimeError(f"could not produce exactly {target_total} samples")

    selected = df.loc[sorted(chosen_idx)].copy().reset_index(drop=True)

    # Annotate with the stratification cell each sample falls into.
    cell_labels = []
    bond_bin_labels = []
    ea_tertile_labels = []
    for _, row in selected.iterrows():
        bb = _bond_bin(int(row["n_bond_changes"]))
        ed = tertile_edges[row["source"]]
        et = _ea_tertile(float(row["activation_energy"]), ed)
        cell_labels.append(Cell(row["source"], int(row["n_heavy_atoms"]), bb, et).label())
        bond_bin_labels.append("2-3" if bb == 0 else "4-6")
        ea_tertile_labels.append(("low", "mid", "high")[et])
    selected["bond_change_bin"] = bond_bin_labels
    selected["ea_tertile"] = ea_tertile_labels
    selected["cell_label"] = cell_labels
    selected["seed"] = seed

    # Persist
    selected.to_csv(output_csv, index=False)
    log.info("Wrote %s with %d rows", output_csv, len(selected))

    # Write a bond_changes.json keyed by reaction_id for downstream stages.
    bond_changes_payload = {
        r["reaction_id"]: {
            "bonds_broken": list(r["bonds_broken"]),
            "bonds_formed": list(r["bonds_formed"]),
            "n_bond_changes": int(r["n_bond_changes"]),
            "n_components_R": int(r["n_components_R"]),
        }
        for r in bc[bc["reaction_id"].isin(selected["reaction_id"])].to_dict(orient="records")
    }
    with open(output_bond_json, "w") as f:
        json.dump(bond_changes_payload, f, indent=2, default=_json_default)
    log.info("Wrote %s with %d entries", output_bond_json, len(bond_changes_payload))

    # Validation: marginal distributions vs targets.
    marginals = _marginal_report(selected)
    log.info("Source distribution:\n%s", marginals["source"].to_string())
    log.info("Heavy atoms:\n%s", marginals["heavy"].to_string())
    log.info("Bond bin:\n%s", marginals["bond_bin"].to_string())
    log.info("EA tertile:\n%s", marginals["ea_tertile"].to_string())

    return {
        "selected": selected,
        "tertile_edges": tertile_edges,
        "quotas": {c.label(): q for c, q in quotas.items()},
        "cell_log": {c.label(): v for c, v in cell_log.items()},
        "marginals": marginals,
        "population_df": df,
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def _marginal_report(selected: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "source": selected["source"].value_counts().sort_index(),
        "heavy": selected["n_heavy_atoms"].value_counts().sort_index(),
        "bond_bin": selected["bond_change_bin"].value_counts().sort_index(),
        "ea_tertile": selected["ea_tertile"].value_counts().sort_index(),
    }
