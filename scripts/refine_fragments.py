"""Refine fragment partitions: tighten cutoffs for unreviewed reactions, and
replace the ones that cannot be split into exactly 2 fragments (drawn from the
raw dataset pool). Reviewed reactions are never touched.

Outputs (all under outputs/frag_review/):
  - manual_partitions.json                  : updated in place (reviewed kept,
                                              unreviewed refit or removed,
                                              replacements added).
  - refinement_report.json                  : per-reaction disposition + stats.
  - cohort_v7.parquet                       : new 789-row labels manifest
                                              (columns: reaction_id, family, source).
  - replacements_need_features.json         : list of newly drawn reaction_ids
                                              that need MACE-OFF23_medium features
                                              computed before ORCA input gen.

The script does NOT compute MACE features for new reactions — that's a separate
sbatch (scripts/refine_mace_extract.sh).

Rule for "acceptable" partition:
  - connected components at TS geometry with cutoff = natural_cutoffs × mult
  - Sweep mult from 1.20 down in 0.05 steps; take the LARGEST mult that yields
    exactly 2 components. This maximises retained real bonds while forcing the
    split. Minimum mult = 0.60 (below that even real covalent bonds break).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from glob import glob
from pathlib import Path
from typing import Optional

import ase
import ase.io
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
from ase.data import chemical_symbols
from ase.neighborlist import build_neighbor_list, natural_cutoffs

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")

FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
LABELS_PQ_V6 = REPO / "labels/adf/adf_labels_v6_multifamily.parquet"
MANUAL_PART = REPO / "outputs/frag_review/manual_partitions.json"

OUT_DIR = REPO / "outputs/frag_review"
REPORT_JSON = OUT_DIR / "refinement_report.json"
COHORT_V7_PQ = OUT_DIR / "cohort_v7.parquet"
REPLACE_FEAT_JSON = OUT_DIR / "replacements_need_features.json"

DIP_ROOT = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
QMR_ROOT = RAW / "QMrxn20"
QMR_TS = QMR_ROOT / "transition-states"
QMR_RC = QMR_ROOT / "reactant-complex-constrained-conformers"
QMR_SUB = QMR_ROOT / "reactant-conformers"  # substrate alone (n_sub atoms)
RGD1_H5 = RAW / "rgd1" / "RGD1_CHNO.h5"

MULT_MAX = 1.20
MULT_MIN = 0.60
MULT_STEP = 0.05
MAX_DRAW_ATTEMPTS = 500  # per replacement — very generous

# Per-family geometry preference:
# - dipolar / rgd1: use R (well-separated reactants) — visualisation and CC both
# - qmrxn20 e2/sn2: R is bound but atom order is [substrate ⊕ nucleophile],
#   so we split by substrate atom count (not by CC).
FAM_GEOM = {"dipolar": "R", "rgd1": "R",
            "qmrxn20_e2": "R", "qmrxn20_sn2": "R"}


def _qmrxn20_substrate_atoms(rid: str):
    """Return ase.Atoms for the substrate alone, or None if not found."""
    label = "_".join(rid.split("_")[2:])            # e.g. A_A_A_A_B_A
    sub_label = "_".join(label.split("_")[:-1]) + "_0"
    sub_dir = QMR_SUB / sub_label
    if not sub_dir.exists():
        return None
    xyz = sub_dir / "00.xyz"
    if not xyz.exists():
        xyz = next(iter(sub_dir.glob("*.xyz")), None)
    if xyz is None or not xyz.exists():
        return None
    return ase.io.read(str(xyz))


def qmrxn20_R_split(rid: str, z: np.ndarray) -> Optional[tuple[list[int], list[int]]]:
    """Substrate/nucleophile split for qmrxn20 R (bound complex).

    Bound-complex atom order is [substrate ⊕ nucleophile]; substrate size is
    read from reactant-conformers/. Sanity-checks that the substrate's element
    sequence matches the first n_sub atoms of the bound complex.
    """
    sub_atoms = _qmrxn20_substrate_atoms(rid)
    if sub_atoms is None:
        return None
    n_sub = len(sub_atoms)
    if not (1 <= n_sub < len(z)):
        return None
    z_sub = np.asarray(sub_atoms.get_atomic_numbers(), int)
    if not np.array_equal(z_sub, z[:n_sub]):
        return None
    return list(range(n_sub)), list(range(n_sub, len(z)))


def _connected_two(z, pos, mult):
    """Return (comp_a, comp_b) sorted-by-size if exactly 2 components at mult, else None."""
    atoms = ase.Atoms(numbers=[int(x) for x in z], positions=pos)
    cutoffs = [c * mult for c in natural_cutoffs(atoms)]
    nl = build_neighbor_list(atoms, cutoffs, self_interaction=False, bothways=True)
    g = nx.Graph()
    g.add_nodes_from(range(len(z)))
    for i in range(len(z)):
        idx, _ = nl.get_neighbors(i)
        for j in idx:
            g.add_edge(i, int(j))
    comps = [sorted(c) for c in nx.connected_components(g)]
    if len(comps) != 2:
        return None
    comps.sort(key=lambda c: -len(c))
    return comps[0], comps[1]


def find_2frag(z, pos) -> Optional[tuple[list[int], list[int], float]]:
    """Sweep mult from MULT_MAX → MULT_MIN; return (idx_A, idx_B, mult) at first hit."""
    m = MULT_MAX
    while m >= MULT_MIN - 1e-9:
        res = _connected_two(z, pos, m)
        if res is not None:
            return list(res[0]), list(res[1]), float(m)
        m -= MULT_STEP
    return None


def load_geom_from_features(rid: str, geom: str = "TS"):
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    if geom not in d:
        raise KeyError(f"{geom} not in .pt for {rid}")
    z = np.asarray(d[geom]["z"], dtype=int)
    pos = np.asarray(d[geom]["pos"], dtype=float)
    return z, pos


def load_ts_from_features(rid: str):
    return load_geom_from_features(rid, "TS")


def _compose_dipolar_R(r0_atoms, r1_atoms, gap=5.0):
    """r0 + r1 with r1 translated to clear r0 by `gap` Å (fixes overlap bug)."""
    p0 = r0_atoms.get_positions()
    p1 = r1_atoms.get_positions()
    p1 = p1 - p1.mean(axis=0)
    p0_ctr = p0 - p0.mean(axis=0)
    r0_r = np.linalg.norm(p0_ctr, axis=1).max()
    r1_r = np.linalg.norm(p1, axis=1).max()
    p1 = p1 + p0.mean(axis=0) + np.array([r0_r + r1_r + gap, 0.0, 0.0])
    z = np.concatenate([np.asarray(r0_atoms.get_atomic_numbers(), int),
                        np.asarray(r1_atoms.get_atomic_numbers(), int)])
    pos = np.vstack([p0, p1])
    return z, pos


def load_geom_from_raw(rid: str, family: str, geom: str = "TS"):
    """Load requested geometry (R or TS) directly from raw dataset."""
    if family == "dipolar":
        idx = int(rid.rsplit("_", 1)[-1])
        d = DIP_ROOT / str(idx)
        if geom == "TS":
            cand = [f for f in d.glob("TS_*.xyz") if "imag_mode" in f.name]
            if not cand:
                cand = list(d.glob("TS_*.xyz"))
            if not cand:
                return None
            atoms = ase.io.read(str(cand[0]))
            return np.asarray(atoms.get_atomic_numbers(), int), atoms.get_positions()
        # R: compose r0 + r1 with spatial offset so they don't overlap.
        r0p = next(iter(d.glob("r0_*.xyz")), None)
        r1p = next(iter(d.glob("r1_*.xyz")), None)
        if r0p is None:
            return None
        r0 = ase.io.read(str(r0p))
        if r1p is None:
            return np.asarray(r0.get_atomic_numbers(), int), r0.get_positions()
        r1 = ase.io.read(str(r1p))
        return _compose_dipolar_R(r0, r1)
    if family in ("qmrxn20_e2", "qmrxn20_sn2"):
        subfam = "e2" if "e2" in family else "sn2"
        label = "_".join(rid.split("_")[2:])
        if geom == "TS":
            p = QMR_TS / subfam / f"{label}.xyz"
        else:  # R = reactant-complex 00.xyz
            rc = QMR_RC / subfam / label
            p = rc / "00.xyz"
            if not p.exists() and rc.exists():
                p = next(iter(rc.glob("*.xyz")), None)
        if p is None or not p.exists():
            return None
        atoms = ase.io.read(str(p))
        return np.asarray(atoms.get_atomic_numbers(), int), atoms.get_positions()
    if family == "rgd1":
        key = "_".join(rid.split("_")[1:])
        with h5py.File(RGD1_H5, "r") as f:
            if key not in f:
                return None
            g = f[key]
            z = np.asarray(g["elements"], int)
            geom_key = "TSG" if geom == "TS" else "RG"
            pos = np.asarray(g[geom_key], float)
        return z, pos
    raise ValueError(f"unknown family {family}")


def load_ts_from_raw(rid: str, family: str):
    return load_geom_from_raw(rid, family, "TS")


def raw_pool_ids(family: str) -> list[str]:
    """Enumerate all reaction_ids available in the raw dataset for a given family."""
    if family == "dipolar":
        return [f"dipolar_{int(d):06d}" for d in sorted(os.listdir(DIP_ROOT))
                if (DIP_ROOT / d).is_dir() and d.isdigit()]
    if family in ("qmrxn20_e2", "qmrxn20_sn2"):
        subfam = "e2" if "e2" in family else "sn2"
        stems = [Path(p).stem for p in sorted(glob(str(QMR_TS / subfam / "*.xyz")))]
        return [f"qmrxn20_{subfam}_{s}" for s in stems]
    if family == "rgd1":
        with h5py.File(RGD1_H5, "r") as f:
            keys = list(f.keys())
        return [f"rgd1_{k}" for k in keys]
    raise ValueError(f"unknown family {family}")


def _rgd1_smiles_is_bimolecular(rid: str) -> bool:
    """Return True iff the Rsmiles for this rgd1 reaction has exactly 2 dot-parts."""
    try:
        key = "_".join(rid.split("_")[1:])
        with h5py.File(RGD1_H5, "r") as f:
            rs = f[key]["Rsmiles"][()]
        rs = rs.decode() if isinstance(rs, bytes) else rs
        return len(rs.split(".")) == 2
    except Exception:
        return False


def try_candidate(rid: str, family: str, geom: str = "TS") -> Optional[dict]:
    """Load raw geom, try to 2-frag; return entry dict on success."""
    loaded = load_geom_from_raw(rid, family, geom)
    if loaded is None:
        return None
    z, pos = loaded
    if len(z) < 4:
        return None
    # RGD1: reject reactions that aren't cleanly bimolecular per Rsmiles.
    if family == "rgd1" and not _rgd1_smiles_is_bimolecular(rid):
        return None
    # QMrxn20 at R geometry: use substrate-based split.
    if family in ("qmrxn20_e2", "qmrxn20_sn2") and geom == "R":
        r_split = qmrxn20_R_split(rid, z)
        if r_split is None:
            return None
        idx_A, idx_B = r_split
        mult = 0.0  # sentinel — no CC mult used
    else:
        res = find_2frag(z, pos)
        if res is None:
            return None
        idx_A, idx_B, mult = res
    # Also load the "other" geometry to include in the .pt file
    other_geom = "R" if geom == "TS" else "TS"
    other = load_geom_from_raw(rid, family, other_geom)
    entry = {
        "reaction_id": rid,
        "family": family,
        "n_atoms": int(len(z)),
        "frag_A_indices": idx_A,
        "frag_B_indices": idx_B,
        "mult_used": mult,
        "z": z.tolist(),
        "pos": pos.tolist(),
        "geom_used": geom,
    }
    if other is not None and len(other[0]) == len(z):
        entry["other_z"] = other[0].tolist()
        entry["other_pos"] = other[1].tolist()
        entry["other_geom"] = other_geom
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", choices=["R", "TS", "family"], default="family",
                    help="Geometry basis. 'family' = per-family map (default): "
                         "dipolar/rgd1→R, qmrxn20→TS.")
    ap.add_argument("--include-reviewed", action="store_true",
                    help="Also re-refit reviewed reactions (keeps reviewed=True flag).")
    args = ap.parse_args()
    print(f"[refine] geom mode = {args.geom}", flush=True)
    print(f"[refine] include_reviewed = {args.include_reviewed}", flush=True)
    if args.geom == "family":
        print(f"[refine] FAM_GEOM = {FAM_GEOM}", flush=True)

    def geom_for(fam):
        if args.geom == "family":
            return FAM_GEOM[fam]
        return args.geom

    random.seed(42)
    # 1) Load: v6 for canonical family quotas + cohort_v7 (if present) for current state
    v6 = pd.read_parquet(LABELS_PQ_V6)
    quotas = v6.family.value_counts().to_dict()
    print(f"[refine] v6 quotas: {quotas}", flush=True)
    v7 = REPO / "outputs/frag_review/cohort_v7.parquet"
    if v7.exists():
        labels = pd.read_parquet(v7)
        print(f"[refine] using cohort_v7 as current state ({len(labels)} rows)", flush=True)
    else:
        labels = v6
        print(f"[refine] no cohort_v7 → using v6 ({len(labels)} rows)", flush=True)
    fam_of = dict(zip(labels.reaction_id, labels.family))
    # Also union with v6 in case any reviewed IDs no longer live in v7.
    for rid, fam in zip(v6.reaction_id, v6.family):
        fam_of.setdefault(rid, fam)
    with open(MANUAL_PART) as f:
        manual = json.load(f)

    all_current = list(labels.reaction_id)
    reviewed = {rid for rid, v in manual.items() if v.get("reviewed") and not v.get("discard")}
    discarded = {rid for rid, v in manual.items() if v.get("discard")}
    print(f"current cohort: {len(all_current)}  reviewed: {len(reviewed)}  discarded: {len(discarded)}", flush=True)

    # 2) For unreviewed reactions currently in cohort, try tighter cutoffs.
    n_ref = n_kept_as_is = n_needs_replace = 0
    kept: list[str] = []
    disposition = {}
    for rid in all_current:
        fam = fam_of[rid]
        geom = geom_for(fam)
        was_reviewed = rid in reviewed
        if was_reviewed and not args.include_reviewed:
            kept.append(rid)
            disposition[rid] = {"status": "kept_reviewed"}
            continue
        if rid in discarded:
            n_needs_replace += 1
            disposition[rid] = {"status": "needs_replace", "reason": "user_discarded"}
            continue
        try:
            z, pos = load_geom_from_features(rid, geom)
        except (FileNotFoundError, KeyError) as exc:
            n_needs_replace += 1
            disposition[rid] = {"status": "needs_replace", "reason": f"no_{geom}: {exc}"}
            continue

        # Family-specific fragmentation strategy
        if fam in ("qmrxn20_e2", "qmrxn20_sn2") and geom == "R":
            r_split = qmrxn20_R_split(rid, z)
            if r_split is None:
                n_needs_replace += 1
                disposition[rid] = {"status": "needs_replace",
                                     "reason": "qmrxn20_no_substrate_file_or_z_mismatch"}
                continue
            idx_A, idx_B = r_split
            mult = None
            method = "substrate_split"
        else:
            res = find_2frag(z, pos)
            if res is None:
                n_needs_replace += 1
                disposition[rid] = {"status": "needs_replace",
                                     "reason": f"no_2frag_at_{geom}_mult>={MULT_MIN}"}
                continue
            idx_A, idx_B, mult = res
            method = f"cc_mult={mult:.2f}"

        n_ref += 1
        note_prefix = "auto-refit (was reviewed)" if was_reviewed else "auto"
        manual[rid] = {
            "frag_A_indices": idx_A,
            "frag_B_indices": idx_B,
            "reviewed": was_reviewed,
            "note": f"{note_prefix} {geom} {method}",
        }
        kept.append(rid)
        disposition[rid] = {"status": "re-refit" if was_reviewed else "refit",
                             "geom": geom, "method": method,
                             "n_A": len(idx_A), "n_B": len(idx_B)}
    print(f"unreviewed disposition: refit={n_ref}  needs_replace={n_needs_replace}", flush=True)

    # Compute per-family shortfall relative to v6 quotas
    kept_by_family = {f: 0 for f in quotas}
    for rid in kept:
        kept_by_family[fam_of[rid]] += 1
    to_replace_by_family = {f: quotas[f] - kept_by_family[f] for f in quotas}
    print(f"kept per family: {kept_by_family}", flush=True)
    print(f"replacements needed per family (to reach v6 quotas): {to_replace_by_family}", flush=True)

    # 3) Draw replacements from raw pool per family, skipping current cohort.
    used_ids = set(all_current)  # never draw an ID already used
    pool_by_family = {}
    for fam in to_replace_by_family:
        if to_replace_by_family[fam] == 0:
            pool_by_family[fam] = []
            continue
        pool = raw_pool_ids(fam)
        pool = [rid for rid in pool if rid not in used_ids]
        random.shuffle(pool)
        pool_by_family[fam] = pool
        print(f"raw pool {fam}: available={len(pool)}", flush=True)

    new_entries: list[dict] = []  # {rid, family, ...}
    for fam, n_need in to_replace_by_family.items():
        if n_need <= 0:
            continue
        pool = pool_by_family[fam]
        fam_geom = geom_for(fam)
        n_drawn = 0
        n_tried = 0
        for cand in pool:
            if n_drawn >= n_need or n_tried >= MAX_DRAW_ATTEMPTS:
                break
            n_tried += 1
            entry = try_candidate(cand, fam, geom=fam_geom)
            if entry is None:
                continue
            new_entries.append(entry)
            used_ids.add(cand)
            n_drawn += 1
        print(f"drew {n_drawn}/{n_need} replacements for {fam} @{fam_geom} (tried {n_tried})", flush=True)
        if n_drawn < n_need:
            print(f"[WARN] {fam}: could not fill {n_need - n_drawn} slots", flush=True)

    # 4) Update manual_partitions.json: drop old un-replaceable IDs, add new ones.
    dropped_ids = [rid for rid in all_current
                   if rid not in reviewed and disposition[rid]["status"] == "needs_replace"]
    for rid in dropped_ids:
        manual.pop(rid, None)

    new_cohort = list(kept)  # reviewed + refit
    for e in new_entries:
        new_cohort.append(e["reaction_id"])
        manual[e["reaction_id"]] = {
            "frag_A_indices": e["frag_A_indices"],
            "frag_B_indices": e["frag_B_indices"],
            "reviewed": False,
            "note": f"replacement {e['geom_used']} mult={e['mult_used']:.2f}",
        }

    print(f"new cohort size: {len(new_cohort)}  (target {len(all_current)})", flush=True)

    # 5) Write outputs.
    with open(MANUAL_PART, "w") as f:
        json.dump(manual, f, indent=1)

    # cohort_v7 parquet: reaction_id + family + source
    fam_of_new = {e["reaction_id"]: e["family"] for e in new_entries}
    rows = []
    for rid in new_cohort:
        if rid in fam_of:
            src = "kept_reviewed" if rid in reviewed else "refit"
            rows.append({"reaction_id": rid, "family": fam_of[rid], "source": src})
        else:
            rows.append({"reaction_id": rid, "family": fam_of_new[rid], "source": "replacement"})
    pd.DataFrame(rows).to_parquet(COHORT_V7_PQ, index=False)

    # replacements_need_features.json: new IDs + raw z + raw pos (for stage-2 MACE)
    need_feat = [{
        "reaction_id": e["reaction_id"],
        "family": e["family"],
        "n_atoms": e["n_atoms"],
        "z": e["z"],
        "pos": e["pos"],
        "frag_A_indices": e["frag_A_indices"],
        "frag_B_indices": e["frag_B_indices"],
        "mult_used": e["mult_used"],
    } for e in new_entries]
    with open(REPLACE_FEAT_JSON, "w") as f:
        json.dump(need_feat, f)

    # refinement_report.json
    report = {
        "reviewed_count": len(reviewed),
        "refit_count": n_ref,
        "needs_replace_count": n_needs_replace,
        "drawn_replacements": len(new_entries),
        "unfilled_slots": n_needs_replace - len(new_entries),
        "per_family_replaced": {fam: sum(1 for e in new_entries if e["family"] == fam)
                                for fam in to_replace_by_family},
        "dropped_ids": dropped_ids,
        "disposition": disposition,
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=1)

    print("---")
    print(f"manual_partitions.json  ->  {MANUAL_PART}")
    print(f"cohort_v7.parquet       ->  {COHORT_V7_PQ}")
    print(f"replacements_need_features.json -> {REPLACE_FEAT_JSON}  ({len(need_feat)} entries)")
    print(f"refinement_report.json  ->  {REPORT_JSON}")


if __name__ == "__main__":
    main()
