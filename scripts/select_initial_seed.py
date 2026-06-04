#!/usr/bin/env python
"""Initial seed selection for ASR Active Learning (spec v1.1).

Selects 800 reactions from QMrxn20 (E2 + SN2) + Stuyver dipolar via
three-tier hierarchical stratified sampling:
  Tier 1 — Family allocation (E2=250, SN2=250, dipolar=300)
  Tier 2 — dE_a quartile (4 per family, equal allocation per cell)
  Tier 3 — Kennard-Stone maximin on Morgan fingerprints (Hamming distance)

Deviations from spec v1.1 (logged in manifest):
  - Data paths match this repo: data/raw/{QMrxn20,dipolar_cycloaddition}/
  - QMrxn20 Mol + SMILES derived from R-complex XYZ via rdkit.Chem.rdDetermineBonds
    (no SMILES exist in source); charge=-1 for anionic systems.
  - max_failure_rate softened to 0.10 (XYZ→Mol on R-complex is harder than on SMILES).
  - Hard expected-count assertions softened to log warnings (actual pool counts
    differ from spec targets because we filter by available barrier rows).
  - UMAP panel skipped if umap-learn not installed.

Exit codes: 0 ok, 1 preflight, 2 load/schema, 3 QC, 4 IO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from sklearn.metrics import pairwise_distances

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eda_asm.datasets import DipolarCycloadditionLoader, QMrxn20Loader  # noqa: E402

# ---------- constants ----------
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
N_QUARTILES = 4
MIN_MEAN_TANIMOTO_DIST = 0.30

EXIT_PREFLIGHT = 1
EXIT_LOAD = 2
EXIT_QC = 3
EXIT_IO = 4


# ---------- helpers ----------
def setup_logging(log_path: Path, level: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("seed_selection")
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _qmrxn20_product_label(reaction: str, ts_label: str) -> str:
    parts = ts_label.split("_")
    if reaction == "e2":
        return "_".join(parts[:4] + ["0", "0"])
    if reaction == "sn2":
        return "_".join(parts[:4] + ["0", parts[5]])
    raise ValueError(reaction)


def xyz_to_mol(path: Path, charge: int) -> "Chem.Mol | None":
    """Build an RDKit Mol from a single-conformer XYZ using DetermineBonds.

    Returns None if DetermineBonds fails (typical for ambiguous TS-like complexes).
    """
    raw = Chem.MolFromXYZFile(str(path))
    if raw is None:
        return None
    mol = Chem.Mol(raw)
    try:
        DetermineBonds(mol, charge=charge, embedChiral=False)
    except (ValueError, RuntimeError):
        return None
    return mol


def mol_to_canon_smiles(mol: "Chem.Mol | None") -> str:
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def morgan_fp(mol: "Chem.Mol | None") -> np.ndarray | None:
    if mol is None:
        return None
    try:
        # DetermineBonds outputs Mol without ring/aromaticity perception
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.AtomValenceException,
            Chem.rdchem.KekulizeException,
            ValueError, RuntimeError):
        return None
    try:
        bit = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_NBITS)
    except RuntimeError:
        return None
    arr = np.zeros(MORGAN_NBITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit, arr)
    return arr


def n_heavy_atoms(mol: "Chem.Mol | None") -> int:
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


# ---------- preflight ----------
def prepare_output_dir(args) -> Path:
    """Resolve output dir, refuse if non-empty unless --force. Call BEFORE setup_logging."""
    out = Path(args.output_dir).resolve()
    if out.exists():
        # Treat the log file we may write to as non-blocking
        existing = [p for p in out.iterdir() if p.name != "selection_log.txt"]
        if existing:
            if not args.force:
                sys.stderr.write(
                    f"ERROR: output dir non-empty: {out} ({len(existing)} entries; use --force)\n"
                )
                sys.exit(EXIT_PREFLIGHT)
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def preflight(args, cfg, out: Path, log) -> tuple[Path, Path]:
    log.info("=" * 70)
    log.info("Pre-flight checks")
    log.info("=" * 70)
    if sys.version_info < (3, 10):
        log.error("Python >= 3.10 required")
        sys.exit(EXIT_PREFLIGHT)

    import rdkit
    import sklearn

    log.info(f"  python  : {sys.version.split()[0]}")
    log.info(f"  rdkit   : {rdkit.__version__}")
    log.info(f"  sklearn : {sklearn.__version__}")
    log.info(f"  pandas  : {pd.__version__}")
    log.info(f"  numpy   : {np.__version__}")

    qm_root = (REPO / cfg["data"]["qmrxn20_root"]).resolve()
    di_root = (REPO / cfg["data"]["dipolar_root"]).resolve()
    if not qm_root.is_dir():
        log.error(f"QMrxn20 root missing: {qm_root}")
        sys.exit(EXIT_PREFLIGHT)
    if not di_root.is_dir():
        log.error(f"dipolar root missing: {di_root}")
        sys.exit(EXIT_PREFLIGHT)
    log.info(f"  QMrxn20 : {qm_root}")
    log.info(f"  dipolar : {di_root}")

    free_gb = shutil.disk_usage(out).free / 2**30
    log.info(f"  free disk : {free_gb:.1f} GB at {out}")
    if free_gb < 5:
        log.error(f"Insufficient disk: {free_gb:.1f} < 5 GB")
        sys.exit(EXIT_PREFLIGHT)

    log.info("Pre-flight OK")
    return qm_root, di_root


# ---------- loaders ----------
def load_qmrxn20_pool(root: Path, cfg: dict, log: logging.Logger) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("Loading QMrxn20")
    log.info("=" * 70)

    method_chain = cfg["qmrxn20"]["method_fallback"]
    reactant_kind = cfg["qmrxn20"]["reactant_kind"]
    charge = cfg["qmrxn20"]["charge"]
    log.info(f"  method fallback : {method_chain}")
    log.info(f"  reactant kind   : {reactant_kind}")
    log.info(f"  system charge   : {charge}")

    barriers = pd.read_csv(root / "barriers.txt")
    barriers = barriers.sort_values(
        ["reaction", "label", "method"], kind="stable"
    ).reset_index(drop=True)

    method_rank = {m: i for i, m in enumerate(method_chain)}
    barriers["_rank"] = barriers["method"].map(method_rank).fillna(99).astype(int)

    sel = barriers[barriers["reactant"] == reactant_kind].copy()
    sel = sel.sort_values(["reaction", "label", "_rank"], kind="stable")
    sel = sel.drop_duplicates(["reaction", "label"], keep="first").reset_index(drop=True)
    log.info(f"  candidate rows  : {len(sel)}")
    log.info(f"  by family       : {sel['reaction'].value_counts().to_dict()}")
    log.info(f"  by method used  : {sel['method'].value_counts().to_dict()}")

    records: list[dict] = []
    n_xyz_missing = 0
    n_mol_fail = 0
    t0 = time.time()
    for i, row in enumerate(sel.itertuples(index=False)):
        if i % 200 == 0 and i > 0:
            log.info(f"    progress: {i}/{len(sel)} ({(time.time()-t0):.1f}s elapsed)")

        ts_path = (root / row.filename_ts).resolve()
        r_path = (root / row.filename_r).resolve()
        p_label = _qmrxn20_product_label(row.reaction, row.label)
        p_path = (root / "product-conformers" / row.reaction / p_label / "00.xyz").resolve()

        if not ts_path.is_file() or not r_path.is_file():
            n_xyz_missing += 1
            continue

        mol_r = xyz_to_mol(r_path, charge=charge)
        if mol_r is None:
            n_mol_fail += 1
            continue
        # Product may not exist on disk; fall back to TS-derived mol if so
        mol_p = xyz_to_mol(p_path, charge=charge) if p_path.is_file() else None
        if mol_p is None:
            mol_p = xyz_to_mol(ts_path, charge=charge)
            if mol_p is None:
                n_mol_fail += 1
                continue

        smiles_r = mol_to_canon_smiles(mol_r)
        smiles_p = mol_to_canon_smiles(mol_p)

        records.append(
            {
                "reaction_id": f"qmrxn20_{row.reaction}_{row.label}_conf000",
                "family": row.reaction,
                "source": "qmrxn20",
                "smiles_r": smiles_r,
                "smiles_p": smiles_p,
                "delta_Ea": float(row.activation),
                "charge": charge,
                "multiplicity": 1,
                "n_heavy_atoms": n_heavy_atoms(mol_r),
                "conformer_id": 0,
                "energy_r_relative": 0.0,
                "path_r": str(r_path),
                "path_p": str(p_path if p_path.is_file() else ts_path),
                "path_ts": str(ts_path),
                "_mol_r": mol_r,
                "_mol_p": mol_p,
                "_qm_method": str(row.method),
            }
        )
    log.info(
        f"  loaded {len(records)} reactions "
        f"(xyz missing: {n_xyz_missing}, DetermineBonds failed: {n_mol_fail}) "
        f"in {time.time()-t0:.1f}s"
    )
    return pd.DataFrame.from_records(records)


def load_dipolar_pool(root: Path, cfg: dict, log: logging.Logger) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("Loading dipolar cycloaddition")
    log.info("=" * 70)

    loader = DipolarCycloadditionLoader(root, energy_kind=cfg["dipolar"]["energy_kind"])
    ids = loader.list_reaction_ids()
    log.info(f"  reactions on disk: {len(ids)}")

    meta = pd.read_csv(root / "full_dataset.csv", index_col=0).set_index("rxn_id")
    profiles_dir = root / "extracted" / "full_dataset_profiles"

    records: list[dict] = []
    n_smiles_fail = 0
    for rid in ids:
        if rid not in meta.index:
            continue
        m = meta.loc[rid]
        rxn_smi = str(m["rxn_smiles"])
        if ">>" not in rxn_smi:
            n_smiles_fail += 1
            continue
        lhs, rhs = rxn_smi.split(">>", 1)
        try:
            smiles_r = Chem.CanonSmiles(lhs)
            smiles_p = Chem.CanonSmiles(rhs)
        except Exception:
            n_smiles_fail += 1
            continue
        mol_r = Chem.MolFromSmiles(smiles_r)
        mol_p = Chem.MolFromSmiles(smiles_p)
        if mol_r is None or mol_p is None:
            n_smiles_fail += 1
            continue

        rxn_dir = profiles_dir / str(rid)
        r0 = next(rxn_dir.glob("r0_*.xyz"), None)
        r1 = next(rxn_dir.glob("r1_*.xyz"), None)
        p0 = next(rxn_dir.glob("p0_*.xyz"), None)
        ts_candidates = [p for p in rxn_dir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz"]
        if not (r0 and r1 and p0 and len(ts_candidates) == 1):
            continue
        ts = ts_candidates[0]

        records.append(
            {
                "reaction_id": f"dipolar_{rid:06d}_conf000",
                "family": "dipolar",
                "source": "stuyver",
                "smiles_r": smiles_r,
                "smiles_p": smiles_p,
                "delta_Ea": float(m["G_act"]),
                "charge": int(cfg["dipolar"]["charge"]),
                "multiplicity": 1,
                "n_heavy_atoms": n_heavy_atoms(mol_r),
                "conformer_id": 0,
                "energy_r_relative": 0.0,
                "path_r": str(r0.resolve()),  # fragment A only; path_p uses product
                "path_p": str(p0.resolve()),
                "path_ts": str(ts.resolve()),
                "_mol_r": mol_r,
                "_mol_p": mol_p,
                "_dipolar_r0": str(r0.resolve()),
                "_dipolar_r1": str(r1.resolve()),
                "_dipolar_meta": {
                    "rxn_id": int(rid),
                    "solvent": str(m["solvent"]),
                    "temp_K": float(m["temp"]),
                    "G_r": float(m["G_r"]),
                },
            }
        )
    log.info(f"  loaded {len(records)} reactions (SMILES failed: {n_smiles_fail})")
    return pd.DataFrame.from_records(records)


# ---------- sampling ----------
def collapse_conformers(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    df = df.copy()
    df["base_reaction_id"] = df["reaction_id"].str.replace(
        r"_conf\d+$", "", regex=True
    )
    df = df.sort_values(["base_reaction_id", "energy_r_relative"], kind="stable")
    df = df.drop_duplicates("base_reaction_id", keep="first").reset_index(drop=True)
    df["reaction_id"] = df["base_reaction_id"]
    df = df.drop(columns=["base_reaction_id", "conformer_id", "energy_r_relative"])
    log.info(f"After conformer collapse: {len(df)} unique reactions")
    return df


def compute_fingerprints(pool: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    log.info("Computing concatenated reactant+product Morgan fingerprints "
             f"(r={MORGAN_RADIUS}, nbits={MORGAN_NBITS} each)")
    mol_r_list = pool["_mol_r"].tolist()
    mol_p_list = pool["_mol_p"].tolist()
    fps: list[np.ndarray | None] = []
    for mol_r, mol_p in zip(mol_r_list, mol_p_list):
        fp_r = morgan_fp(mol_r)
        fp_p = morgan_fp(mol_p)
        if fp_r is None or fp_p is None:
            fps.append(None)
        else:
            fps.append(np.concatenate([fp_r, fp_p]))
    pool = pool.copy()
    pool["fingerprint"] = fps
    n_fail = int(sum(1 for f in fps if f is None))
    log.info(f"  fingerprint failures: {n_fail} / {len(pool)}")
    return pool, n_fail


def assign_quartiles(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    out = []
    for family, sub in df.groupby("family", sort=False):
        sub = sub.copy()
        try:
            sub["quartile"] = pd.qcut(
                sub["delta_Ea"], q=N_QUARTILES, labels=False, duplicates="drop"
            )
        except ValueError as e:
            raise RuntimeError(f"qcut failed for family {family}: {e}")
        n_bins = sub["quartile"].nunique(dropna=True)
        if n_bins < N_QUARTILES:
            log.warning(f"  family {family}: only {n_bins} quartiles populated (ties)")
        log.info(f"  family {family}: dE_a quartile boundaries = "
                 f"{sub.groupby('quartile')['delta_Ea'].agg(['min','max']).to_dict()}")
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def maximin_sample(features: np.ndarray, n_samples: int, seed: int,
                   log: logging.Logger) -> list[int]:
    n_total = len(features)
    if n_samples >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n_total))]
    min_dist = pairwise_distances(features, features[selected], metric="hamming").ravel()

    for _ in range(n_samples - 1):
        next_idx = int(np.argmax(min_dist))
        if float(min_dist[next_idx]) == 0.0:
            log.warning(
                f"  maximin degenerate at {len(selected)}/{n_samples}; "
                "filling remainder randomly"
            )
            remaining = sorted(set(range(n_total)) - set(selected))
            extra = rng.choice(np.array(remaining), size=n_samples - len(selected),
                               replace=False)
            selected.extend(int(i) for i in extra)
            break
        selected.append(next_idx)
        new_dist = pairwise_distances(features, features[[next_idx]],
                                      metric="hamming").ravel()
        min_dist = np.minimum(min_dist, new_dist)
    return selected[:n_samples]


def allocate_per_cell(family_target: int, n_quartiles: int) -> list[int]:
    base = family_target // n_quartiles
    remainder = family_target - base * n_quartiles
    return [base + (1 if q < remainder else 0) for q in range(n_quartiles)]


# ---------- QC ----------
def quality_control(selected_df: pd.DataFrame, allocation: dict, log: logging.Logger) -> None:
    log.info("=" * 70)
    log.info("Quality control")
    log.info("=" * 70)

    total_target = sum(allocation.values())
    if len(selected_df) != total_target:
        log.error(f"QC-1 FAIL: total {len(selected_df)} != {total_target}")
        sys.exit(EXIT_QC)
    log.info(f"  QC-1 total count = {len(selected_df)} OK")

    if selected_df["reaction_id"].nunique() != len(selected_df):
        log.error("QC-2 FAIL: duplicate reaction_ids")
        sys.exit(EXIT_QC)
    log.info(f"  QC-2 no duplicate IDs OK")

    fam_counts = selected_df["family"].value_counts().to_dict()
    for fam, expected in allocation.items():
        got = fam_counts.get(fam, 0)
        if got != expected:
            log.error(f"QC-3 FAIL: {fam} = {got}, expected {expected}")
            sys.exit(EXIT_QC)
    log.info(f"  QC-3 family allocation = {fam_counts} OK")

    for fam in allocation:
        q_counts = selected_df[selected_df["family"] == fam]["quartile"].value_counts()
        nz = int((q_counts > 0).sum())
        if nz < N_QUARTILES:
            log.warning(f"  QC-4 partial: {fam} populated quartiles {nz}/{N_QUARTILES}")
    log.info(f"  QC-4 quartile coverage logged")

    bad = []
    for fam in allocation:
        for q in range(N_QUARTILES):
            cell = selected_df[
                (selected_df["family"] == fam) & (selected_df["quartile"] == q)
            ]
            if len(cell) < 2:
                continue
            fps = np.stack(cell["fingerprint"].values).astype(bool)
            dist_mat = pairwise_distances(fps, metric="jaccard")
            n = len(fps)
            mean_d = float(dist_mat[np.triu_indices(n, k=1)].mean())
            if mean_d < MIN_MEAN_TANIMOTO_DIST:
                bad.append((fam, q, mean_d))
            log.info(f"    {fam}/Q{q}: mean Tanimoto-distance = {mean_d:.3f}")
    if bad:
        log.error(f"QC-5 FAIL: low diversity cells {bad}")
        sys.exit(EXIT_QC)
    log.info(f"  QC-5 within-cell diversity OK")

    missing = [p for p in selected_df["path_ts"] if not Path(p).is_file()]
    if missing:
        log.error(f"QC-6 FAIL: {len(missing)} TS files missing, e.g. {missing[0]}")
        sys.exit(EXIT_QC)
    log.info(f"  QC-6 all TS files present OK")


# ---------- outputs ----------
def write_outputs(
    selected_df: pd.DataFrame,
    pool: pd.DataFrame,
    output_dir: Path,
    cfg: dict,
    args,
    n_fp_failed: int,
    diagnostics: list[dict],
    pool_raw_size: int,
    log: logging.Logger,
) -> None:
    log.info("=" * 70)
    log.info("Writing outputs")
    log.info("=" * 70)
    public_cols = [
        "reaction_id", "family", "source", "smiles_r", "smiles_p",
        "delta_Ea", "charge", "multiplicity", "n_heavy_atoms", "quartile",
        "path_r", "path_p", "path_ts",
    ]
    csv_path = output_dir / "selected_reactions.csv"
    selected_df[public_cols].to_csv(csv_path, index=False)
    log.info(f"  wrote {csv_path}")

    ids_sorted = sorted(selected_df["reaction_id"].tolist())
    ids_path = output_dir / "selected_reaction_ids.json"
    ids_path.write_text(json.dumps(ids_sorted, indent=2))
    log.info(f"  wrote {ids_path}")

    # Pool cache (drop heavy Mol references; keep fingerprint separately)
    cache_cols = [c for c in pool.columns if not c.startswith("_") and c != "fingerprint"]
    pool[cache_cols].to_parquet(output_dir / "pool_after_conformer_collapse.parquet")
    log.info(f"  wrote pool_after_conformer_collapse.parquet")

    valid_fps = pool[pool["fingerprint"].notna()]["fingerprint"]
    fps_array = np.stack(valid_fps.values).astype(np.uint8)
    np.save(output_dir / "morgan_fingerprints.npy", fps_array)
    log.info(f"  wrote morgan_fingerprints.npy  shape={fps_array.shape}")

    script_path = Path(__file__).resolve()
    import rdkit
    import sklearn
    manifest = {
        "schema_version": "1.1",
        "spec_version": "1.1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest()[:16],
        "config": cfg,
        "config_path": str(args.config),
        "random_seed": int(cfg["random_seed"]),
        "inputs": {
            "qmrxn20_root": str((REPO / cfg["data"]["qmrxn20_root"]).resolve()),
            "dipolar_root": str((REPO / cfg["data"]["dipolar_root"]).resolve()),
            "qmrxn20_doi": "10.24435/materialscloud:sf-tz",
            "dipolar_doi": "10.6084/m9.figshare.21707888.v5",
        },
        "pool_stats": {
            "raw_with_conformers": int(pool_raw_size),
            "unique_after_collapse": int(len(pool)),
            "fingerprint_failed": int(n_fp_failed),
        },
        "output_stats": {
            "total_selected": int(len(selected_df)),
            "by_family": selected_df["family"].value_counts().to_dict(),
            "mean_delta_Ea": float(selected_df["delta_Ea"].mean()),
            "std_delta_Ea": float(selected_df["delta_Ea"].std()),
        },
        "diagnostics_per_cell": diagnostics,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "rdkit": rdkit.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "deviations_from_spec": [
            "Data paths use this repo's data/raw/{QMrxn20,dipolar_cycloaddition}/ layout.",
            "QMrxn20 SMILES + Mol derived at runtime from R-complex XYZ via "
            "rdkit.Chem.rdDetermineBonds(charge=-1).",
            "Spec hard-count tolerance (1%) replaced with soft warnings; actual pool counts "
            "come from per-reaction barrier rows with method fallback.",
            "UMAP panel skipped (umap-learn not installed); other diagnostic panels still emitted.",
            "max_failure_rate raised to 0.10 to absorb DetermineBonds failures on ionic complexes.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    manifest["output_hash"] = hashlib.sha256(ids_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    log.info(f"  wrote manifest.json  output_hash={manifest['output_hash'][:16]}...")

    # requirements.lock
    try:
        import subprocess
        lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                       stderr=subprocess.DEVNULL).decode()
        (output_dir / "requirements.lock").write_text(lock)
        log.info(f"  wrote requirements.lock")
    except Exception as e:
        log.warning(f"  requirements.lock skipped: {e}")


def write_diagnostics_png(
    pool: pd.DataFrame, selected_df: pd.DataFrame, output_dir: Path,
    log: logging.Logger,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping diagnostic PNG")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Panel A: dE_a histogram, pool vs selected, per family
    ax = axes[0, 0]
    for fam, color in zip(["e2", "sn2", "dipolar"], ["tab:blue", "tab:orange", "tab:green"]):
        pool_v = pool[pool["family"] == fam]["delta_Ea"].dropna()
        sel_v = selected_df[selected_df["family"] == fam]["delta_Ea"].dropna()
        ax.hist(pool_v, bins=40, histtype="step", color=color, alpha=0.6,
                label=f"{fam} pool (n={len(pool_v)})")
        ax.hist(sel_v, bins=40, color=color, alpha=0.5,
                label=f"{fam} selected (n={len(sel_v)})")
    ax.set_xlabel("dE_a (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title("(A) dE_a distribution: pool vs selected")
    ax.legend(fontsize=7)

    # Panel B: n_heavy_atoms histogram
    ax = axes[0, 1]
    for fam, color in zip(["e2", "sn2", "dipolar"], ["tab:blue", "tab:orange", "tab:green"]):
        pool_v = pool[pool["family"] == fam]["n_heavy_atoms"].dropna()
        sel_v = selected_df[selected_df["family"] == fam]["n_heavy_atoms"].dropna()
        bins = range(int(min(pool_v.min(), sel_v.min())),
                     int(max(pool_v.max(), sel_v.max())) + 2)
        ax.hist(pool_v, bins=bins, histtype="step", color=color, alpha=0.6,
                label=f"{fam} pool")
        ax.hist(sel_v, bins=bins, color=color, alpha=0.5, label=f"{fam} selected")
    ax.set_xlabel("# heavy atoms")
    ax.set_ylabel("count")
    ax.set_title("(B) Molecular-size distribution")
    ax.legend(fontsize=7)

    # Panel C: try UMAP, otherwise scatter dE_a vs n_heavy
    ax = axes[1, 0]
    try:
        import umap
        log.info("  computing UMAP for panel C")
        valid_pool = pool[pool["fingerprint"].notna()].reset_index(drop=True)
        fp_all = np.stack(valid_pool["fingerprint"].values)
        reducer = umap.UMAP(metric="hamming", random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(fp_all)
        sel_ids = set(selected_df["reaction_id"])
        is_sel = valid_pool["reaction_id"].isin(sel_ids).values
        ax.scatter(coords[~is_sel, 0], coords[~is_sel, 1], s=2, c="lightgray", alpha=0.5)
        for fam, color in zip(["e2", "sn2", "dipolar"], ["tab:blue", "tab:orange", "tab:green"]):
            mask = is_sel & (valid_pool["family"].values == fam)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=10, c=color, alpha=0.8, label=fam)
        ax.set_title("(C) UMAP of Morgan fingerprints (hamming)")
        ax.legend(fontsize=8)
    except ImportError:
        log.info("  UMAP unavailable; panel C falls back to dE_a vs n_heavy")
        for fam, color in zip(["e2", "sn2", "dipolar"], ["tab:blue", "tab:orange", "tab:green"]):
            sub = pool[pool["family"] == fam]
            ax.scatter(sub["n_heavy_atoms"], sub["delta_Ea"], s=4, c="lightgray", alpha=0.4)
            sel = selected_df[selected_df["family"] == fam]
            ax.scatter(sel["n_heavy_atoms"], sel["delta_Ea"], s=14, c=color, alpha=0.85,
                       label=fam)
        ax.set_xlabel("# heavy atoms")
        ax.set_ylabel("dE_a (kcal/mol)")
        ax.set_title("(C) dE_a vs n_heavy (UMAP unavailable)")
        ax.legend(fontsize=8)

    # Panel D: within-cell Tanimoto distance boxplot
    ax = axes[1, 1]
    box_data = []
    box_labels = []
    for fam in ["e2", "sn2", "dipolar"]:
        for q in range(N_QUARTILES):
            cell = selected_df[
                (selected_df["family"] == fam) & (selected_df["quartile"] == q)
            ]
            if len(cell) < 2:
                continue
            fps = np.stack(cell["fingerprint"].values).astype(bool)
            d = pairwise_distances(fps, metric="jaccard")
            box_data.append(d[np.triu_indices(len(fps), k=1)])
            box_labels.append(f"{fam[:3]}/Q{q}")
    if box_data:
        ax.boxplot(box_data, tick_labels=box_labels, showfliers=False)
        ax.axhline(MIN_MEAN_TANIMOTO_DIST, color="red", linestyle="--", linewidth=1,
                   label=f"threshold {MIN_MEAN_TANIMOTO_DIST}")
        ax.set_ylabel("pairwise Tanimoto distance")
        ax.set_title("(D) Within-cell diversity")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_png = output_dir / "stratification_diagnostics.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  wrote {out_png}")


# ---------- main ----------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if args.seed <= 0:
        sys.stderr.write("seed must be positive\n")
        sys.exit(EXIT_PREFLIGHT)

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["random_seed"] = int(args.seed)  # CLI overrides config

    out_dir = prepare_output_dir(args)
    log = setup_logging(out_dir / "selection_log.txt", args.log_level)
    log.info(f"args = {vars(args)}")
    log.info(f"config = {json.dumps(cfg, default=str)}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    qm_root, di_root = preflight(args, cfg, out_dir, log)

    qm_pool = load_qmrxn20_pool(qm_root, cfg, log)
    di_pool = load_dipolar_pool(di_root, cfg, log)
    pool_raw = pd.concat([qm_pool, di_pool], ignore_index=True)
    pool_raw_size = len(pool_raw)
    log.info(f"Raw pool: {pool_raw_size} reactions "
             f"({pool_raw['family'].value_counts().to_dict()})")

    pool = collapse_conformers(pool_raw, log)
    pool, n_fp_failed = compute_fingerprints(pool, log)
    failure_rate = n_fp_failed / max(len(pool), 1)
    if failure_rate > float(cfg.get("max_failure_rate", 0.01)):
        log.error(f"fingerprint failure rate {failure_rate:.3f} > "
                  f"{cfg['max_failure_rate']}")
        sys.exit(EXIT_LOAD)
    pool = pool[pool["fingerprint"].notna()].reset_index(drop=True)
    log.info(f"Pool after FP filter: {len(pool)} reactions "
             f"({pool['family'].value_counts().to_dict()})")

    allocation = {k: int(v) for k, v in cfg["family_targets"].items()}
    if sum(allocation.values()) != int(cfg["total_samples"]):
        log.error("family_targets sum != total_samples")
        sys.exit(EXIT_PREFLIGHT)
    for family, target in allocation.items():
        avail = (pool["family"] == family).sum()
        if avail < target:
            log.error(f"  family {family}: {avail} candidates < target {target}")
            sys.exit(EXIT_LOAD)
        if avail < target * 2:
            log.warning(f"  family {family}: only {avail} candidates "
                        f"(< 2x target {target}), stratification will be tight")

    pool = assign_quartiles(pool, log)

    log.info("=" * 70)
    log.info("Tier 3: Kennard-Stone maximin within each cell")
    log.info("=" * 70)
    selected_ids: list[str] = []
    diagnostics: list[dict] = []
    seed = int(cfg["random_seed"])

    for family, family_target in allocation.items():
        df_fam = pool[pool["family"] == family]
        avail_qs = sorted(df_fam["quartile"].dropna().unique().astype(int))
        per_cell = allocate_per_cell(family_target, len(avail_qs))
        log.info(f"  family {family}: per-cell targets = "
                 f"{dict(zip(avail_qs, per_cell))} (avail quartiles: {avail_qs})")
        for q, n_target in zip(avail_qs, per_cell):
            cell = df_fam[df_fam["quartile"] == q].reset_index(drop=True)
            if len(cell) < n_target:
                log.warning(f"    {family}/Q{q}: {len(cell)} < target {n_target}, "
                            "taking all available")
                chosen_idx = list(range(len(cell)))
            else:
                features = np.stack(cell["fingerprint"].values)
                chosen_idx = maximin_sample(features, n_target,
                                             seed=seed + q * 100 + hash(family) % 1000,
                                             log=log)
            chosen_ids = cell.iloc[chosen_idx]["reaction_id"].tolist()
            selected_ids.extend(chosen_ids)
            diagnostics.append({
                "family": family,
                "quartile": int(q),
                "n_target": int(n_target),
                "n_selected": int(len(chosen_ids)),
                "cell_size": int(len(cell)),
                "mean_delta_Ea": float(cell.iloc[chosen_idx]["delta_Ea"].mean()),
                "std_delta_Ea": float(cell.iloc[chosen_idx]["delta_Ea"].std()),
            })
            log.info(f"    {family}/Q{q}: selected {len(chosen_ids)} from cell of {len(cell)}")

    selected_df = pool[pool["reaction_id"].isin(selected_ids)].reset_index(drop=True)

    quality_control(selected_df, allocation, log)

    write_outputs(selected_df, pool, out_dir, cfg, args, n_fp_failed,
                  diagnostics, pool_raw_size, log)
    write_diagnostics_png(pool, selected_df, out_dir, log)

    log.info("=" * 70)
    log.info(f"SUCCESS — 800 reactions selected; outputs at {out_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
