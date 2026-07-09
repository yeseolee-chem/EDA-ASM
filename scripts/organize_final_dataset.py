"""Organize all files for the 776-reaction v7 dataset into a single
self-contained folder for archival / sharing.

Final location:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_final_776_v7/

Copies (does NOT symlink) so the archive is fully portable:
  - Raw dataset files (r0/r1/TS/P xyz per reaction)
  - Fragment definitions (orca_inp_partitions.json + viewer)
  - ORCA EDA calculation outputs (eda.out, eda_frag{1,2}.out + property/inp)
  - ORCA strain optimization outputs (opt.out + property/inp)
  - Labels parquet (orca_eda_labels_v7.parquet)
  - Cohort manifest (cohort_v7.parquet)
  - Scripts used to generate everything

Emits:
  - README.md + PROVENANCE.md
"""
from __future__ import annotations
import json, shutil
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
EDA_ROOT = REPO / "outputs/orca_eda/inputs"
STRAIN_ROOT = REPO / "outputs/orca_strain/inputs"
COHORT_V7 = REPO / "outputs/frag_review/cohort_v7.parquet"
LABELS_V7 = REPO / "labels/orca/orca_eda_labels_v7.parquet"
ORCA_INP_PART = REPO / "outputs/frag_review/orca_inp_partitions.json"
VIEWER = REPO / "outputs/orca_eda/fragmentation_viewer.html"

OUT_ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/final_776_v7")


def _safe_copy(src, dst):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    shutil.copy2(src, dst)
    return True


def _copy_raw_for_reaction(rid: str, family: str, target_root: Path):
    """Copy raw XYZ files for one reaction to target_root/raw_data/<family>/<rid>/."""
    dst = target_root / "raw_data" / family / rid
    dst.mkdir(parents=True, exist_ok=True)

    if family == "dipolar":
        try:
            idx = int(rid.split("_")[-1])
        except ValueError:
            return 0
        src_dir = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles" / str(idx)
        if not src_dir.exists():
            return 0
        n = 0
        for pat in ("r0_*.xyz", "r1_*.xyz", "TS_imag_mode.xyz", "TS_*.xyz", "p0_*.xyz"):
            for f in src_dir.glob(pat):
                if _safe_copy(f, dst / f.name):
                    n += 1
        return n

    if family in ("qmrxn20_e2", "qmrxn20_sn2"):
        subfam = "e2" if family.endswith("e2") else "sn2"
        label = "_".join(rid.split("_")[2:])
        # TS
        ts = RAW / "QMrxn20" / "transition-states" / subfam / f"{label}.xyz"
        _safe_copy(ts, dst / "ts.xyz")
        # Reactant complex
        rc_dir = RAW / "QMrxn20" / "reactant-complex-constrained-conformers" / subfam / label
        if rc_dir.exists():
            for cand in ["00.xyz"] + [p.name for p in rc_dir.glob("*.xyz")]:
                p = rc_dir / cand
                if p.exists():
                    _safe_copy(p, dst / "reactant_complex.xyz"); break
        # Substrate
        sub_label = "_".join(label.split("_")[:-1]) + "_0"
        sub_dir = RAW / "QMrxn20" / "reactant-conformers" / sub_label
        if sub_dir.exists():
            for cand in ["00.xyz"] + [p.name for p in sub_dir.glob("*.xyz")]:
                p = sub_dir / cand
                if p.exists():
                    _safe_copy(p, dst / "substrate.xyz"); break
        return 4

    if family == "rgd1":
        rgd_dir = RAW / "rgd1" / "extracted_xyz" / rid
        if not rgd_dir.exists():
            return 0
        n = 0
        for name in ("R.xyz", "TS.xyz", "P.xyz"):
            if _safe_copy(rgd_dir / name, dst / name):
                n += 1
        return n

    return 0


def _copy_eda_for_reaction(rid: str, target_root: Path):
    src = EDA_ROOT / rid
    if not src.is_dir():
        return 0
    dst = target_root / "orca_eda" / rid
    n = 0
    for pat in ["eda.inp", "eda.out", "eda.property.txt",
                "eda_frag1.inp", "eda_frag1.out", "eda_frag1.property.txt",
                "eda_frag2.inp", "eda_frag2.out", "eda_frag2.property.txt",
                "meta.json"]:
        f = src / pat
        if _safe_copy(f, dst / pat):
            n += 1
    return n


def _copy_strain_for_reaction(rid: str, target_root: Path):
    n = 0
    for tag in ("fA", "fB"):
        src = STRAIN_ROOT / f"{rid}__{tag}"
        if not src.is_dir():
            continue
        dst_tag = "fragA" if tag == "fA" else "fragB"
        dst = target_root / "orca_strain" / rid / dst_tag
        for pat in ["opt.inp", "opt.out", "opt.property.txt"]:
            f = src / pat
            if _safe_copy(f, dst / pat):
                n += 1
    return n


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"target: {OUT_ROOT}", flush=True)

    # 1) Core parquet + JSON files
    _safe_copy(COHORT_V7, OUT_ROOT / "cohort_v7.parquet")
    if LABELS_V7.exists():
        _safe_copy(LABELS_V7, OUT_ROOT / "labels" / "orca_eda_labels_v7.parquet")
    _safe_copy(ORCA_INP_PART, OUT_ROOT / "fragmentation" / "orca_inp_partitions.json")
    _safe_copy(VIEWER, OUT_ROOT / "fragmentation" / "fragmentation_viewer.html")

    # 2) Scripts (for reproducibility)
    scripts_src = REPO / "scripts"
    scripts_dst = OUT_ROOT / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "make_orca_eda_inputs.py",
        "gen_strain_inputs.py", "gen_strain_inputs.sh",
        "parse_orca_5channel.py", "parse_orca_5channel.sh",
        "build_bundles_v7.py", "build_bundles_v7.sh",
        "run_orca_eda_array.sh", "run_orca_strain_array.sh",
        "organize_final_dataset.py", "organize_final_dataset.sh",
    ]:
        _safe_copy(scripts_src / name, scripts_dst / name)

    # 3) Iterate cohort, copy raw + EDA + strain per reaction
    if not COHORT_V7.exists():
        print(f"WARN: cohort_v7.parquet missing at {COHORT_V7}")
        return
    c7 = pd.read_parquet(COHORT_V7)
    n_reactions = 0
    n_raw = n_eda = n_strain = 0
    for row in c7.itertuples(index=False):
        rid = row.reaction_id
        fam = row.family
        n_reactions += 1
        n_raw += _copy_raw_for_reaction(rid, fam, OUT_ROOT)
        n_eda += _copy_eda_for_reaction(rid, OUT_ROOT)
        n_strain += _copy_strain_for_reaction(rid, OUT_ROOT)
        if n_reactions % 100 == 0:
            print(f"  processed {n_reactions} reactions...", flush=True)
    print(f"reactions: {n_reactions}  raw_files: {n_raw}  eda_files: {n_eda}  strain_files: {n_strain}")

    # 4) README + PROVENANCE
    readme = OUT_ROOT / "README.md"
    readme.write_text(f"""# EDA-ASM 776 Reaction Dataset (v7)

Self-contained dataset for {n_reactions} reactions across 4 chemistry families:
- dipolar cycloaddition
- qmrxn20 e2 elimination
- qmrxn20 sn2 substitution
- rgd1 (bimolecular subset)

## Files
- `cohort_v7.parquet`       — 776-row manifest (reaction_id, family, source)
- `labels/`                 — 5-channel EDA labels (Pauli+ΔXC, V_elst, E_orb, E_disp, E_strain, all kcal/mol)
- `raw_data/`               — original XYZ geometries (r0, r1, TS, P per reaction)
- `fragmentation/`          — Fragment A/B atom index definitions used in ORCA input
- `orca_eda/`               — ORCA 6.1 EDA-NOCV calculation outputs (BLYP-D3BJ / def2-TZVP)
- `orca_strain/`            — Per-fragment geometry optimizations for strain calculation
- `scripts/`                — Python + sbatch scripts used to build everything

## 5-Channel Labels

| Column | Formula | Units |
|---|---|---|
| Pauli_kcal    | ORCA Pauli + ΔE^0(XC)                    | kcal/mol |
| V_elst_kcal   | ORCA Electrostatic Energy                | kcal/mol |
| E_orb_kcal    | ORCA Orbital Energy                      | kcal/mol |
| E_disp_kcal   | ORCA Delta Dispersion                    | kcal/mol |
| E_strain_kcal | Σ E(frag@TS) − E(frag_relaxed)           | kcal/mol |

Sum of all 5 channels = activation energy Ea_ASM.

## Provenance
See PROVENANCE.md for step-by-step reproduction.
""")

    prov = OUT_ROOT / "PROVENANCE.md"
    prov.write_text("""# Provenance

## Method
- Level: BLYP D3BJ def2-TZVP TightSCF NoSym (ORCA 6.1, serial)
- Fragmentation: manual review of automatic (SMILES/CC-based) partitions
- 5 channels per ADF convention (Pauli fold-in of ΔE^0(XC))

## Pipeline
1. `make_orca_eda_inputs.py`     → 776 eda.inp (fragment tagged: elem(1)/elem(2))
2. `run_orca_eda_array.sh`       → 776 EDA-NOCV single-points at TS
3. `gen_strain_inputs.py`        → 1552 opt.inp (each fragment at TS-frozen geom)
4. `run_orca_strain_array.sh`    → 1552 fragment opts → E_relaxed per fragment
5. `parse_orca_5channel.py`      → 5-channel parquet
6. `build_bundles_v7.py`         → m1/m2/m3 training bundles

Monatomic nucleophile fragments (QMrxn20 fragment B) do NOT undergo geom opt
(cannot); E_strain contribution set to 0 for these (chemically correct).

## Fragment Identity
See fragmentation/orca_inp_partitions.json for TS-native fragment atom indices.
""")

    print(f"\nDataset ready at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
