# Paper reproduction bundle — 203-rxn snapshot

Self-contained bundle to reproduce **Espley et al. 2024** (*Digital Discovery*, DOI [10.1039/D4DD00224E](https://doi.org/10.1039/D4DD00224E)) — "Distortion/Interaction Analysis via Machine Learning" — on our 203-reaction subset of the 400-rxn dipolar cycloaddition cohort.

## What's needed on the remote PC

Only Gaussian 16 (or 09) with AM1 support. No other DFT / QM software required.

## Quick start

```bash
# On the remote PC with Gaussian installed:
cd paper_reproduction_203/
scripts/run_all_am1_windows.bat        # Windows
# OR
bash scripts/run_all_am1_linux.sh              # Linux serial
bash scripts/run_all_am1_linux.sh --parallel 8 # Linux 8-way parallel

# Verify all 1015 AM1 jobs finished:
python scripts/verify_am1_results.py
```

Return the entire `paper_reproduction_203/` folder (with the new `.log` files) to HPC. The Grayson ML pipeline then runs on HPC with:

```bash
python code/60_import_am1_results.py     # not yet implemented
bash code/70_run_grayson_pipeline.sh     # sbatch, not yet implemented
```

## Snapshot

- **Snapshot date**: 2026-08-04
- **Snapshot count**: 203 / 400 reactions fully complete (5/5 ORCA jobs done)
- **Full 400-rxn snapshot**: separate future bundle after full chain completes
- **Selection criterion**: `spe_R_A + spe_R_B + eda + spe_dA + spe_dB` all ORCA-TERMINATED-NORMALLY

## Paper protocol reproduction map

| Paper step | Our implementation |
|---|---|
| DFT target labels (ωB97X-D/def2-TZVP + IEFPCM(water)) | ORCA 6.1.1 wB97X-D3BJ/def2-TZVP + CPCM(water), EDA-NOCV |
| AM1 SPE features on same geometries | Gaussian AM1 `#P AM1 Freq NoSymm` (this bundle) |
| 5-atom labels (reaction centre atoms) | User-labeled via port 5578 Flask viz app (400/400 confirmed) |
| Fragment A/B partition | User-manual review (192 v8_review + 208 manual_labels; 0 conflict) |
| Fragment type (dipole vs dipolarophile) | User-manual (`is_A_dipole` overrides pkg auto for 42% of rxns) |
| Grayson pipeline stages 3-6 | `grayson_code/` mirror (diassep, energy_extraction, feature_extraction, feature_selection, hyperparameter_tuning, machine_learning) |

## Folder structure

```
paper_reproduction_203/
├── README.md                        ← this file
├── MANIFEST.txt                     ← 203 rxn_number ↔ reaction_id table + snapshot timestamp
├── snapshot.pkl                     ← programmatic snapshot metadata (rxn_ids, protocol, timestamp)
├── labels/                          ← DFT targets (ORCA-parsed)
│   ├── labels_2ch_paper.parquet     ← distortion_dft, interaction_dft, e_barrier_dft (kcal/mol)
│   ├── labels_2ch_paper.csv         ← same, CSV
│   ├── labels_5channel_paper.parquet ← full audit: 5 EDA channels + per-fragment strains + Hartree energies
│   └── label_schema.json            ← label ranges + provenance
├── geometries/dipolar_XXXXXX/       ← 203 rxn dirs
│   ├── reactant_1.xyz               ← relaxed FRAG1 (reordered to match distorted atom order)
│   ├── reactant_2.xyz               ← relaxed FRAG2 (reordered)
│   ├── reactant_1_distorted.xyz     ← FRAG1 at TS geometry
│   ├── reactant_2_distorted.xyz     ← FRAG2 at TS geometry
│   ├── TS.xyz                       ← TS complex (frag tags stripped)
│   └── meta.json                    ← per-rxn: types, ORCA/pkg swap flags, user picks, perm tables
├── gaussian_inputs/                 ← 1015 .gjf files, Grayson-compatible naming
│   ├── ts/       ts_<rn>.gjf                          (203)
│   ├── gs/       gs_<rn>_reactant_[12].gjf            (406)
│   └── dist_gs/  dist_<rn>_reactant_[12].gjf          (406)
├── grayson_pkls/                    ← Grayson f_extract.py input files
│   ├── common_atoms.pkl             ← {'<rn>': {'di':[3], 'dp':[2], 'reacting':[4], 'name':'ts_<rn>'}} (1-based TS indices)
│   ├── mapping.pkl                  ← {'<rn>': {'reactant_1': {ts_1based: r_1based}, 'reactant_2': {...}}}
│   └── mol_types.pkl                ← {'<rn>': {'reactant_1': 'di'/'dp', 'reactant_2': 'di'/'dp'}}
├── grayson_code/                    ← mirror of github.com/the-grayson-group/distortion-interaction_ML
│   ├── diassep/                     (TS fragment splitter — for reference, not used since we have geometries already)
│   ├── energy_extraction/           (barrier extractor)
│   ├── feature_extraction/          (Morfeus + cclib feature extractor; core script)
│   ├── feature_selection/           (correlation + variance filter)
│   ├── hyperparameter_tuning/       (sklearn HP tuning; NN branch broken in Keras 3)
│   └── machine_learning/            (5-seed × 4-model training + evaluation)
├── code/                            ← our bundle-building + label parsing
│   ├── 10_parse_dft_labels.py       ← spec23_paper_setup ORCA .out → 2ch labels parquet
│   ├── 20_prepare_geometries.py     ← ORCA .inp → clean xyz + reorder + meta.json
│   ├── 30_generate_am1_inputs.py    ← xyz → 1015 .gjf files (Grayson-compatible naming)
│   ├── 40_convert_5atom_to_grayson.py ← user 5-atom picks → common_atoms/mapping/mol_types pkls
│   └── 50_verify_bundle.py          ← end-to-end 25-check validator (all passing)
└── scripts/                         ← user-facing execution scripts
    ├── run_all_am1_windows.bat      ← Windows batch runner (skip-if-done)
    ├── run_all_am1_linux.sh         ← Linux runner (--parallel N option)
    └── verify_am1_results.py        ← post-run completion + APT-charge parse check
```

## Bundle regeneration (HPC-side)

If you want to rebuild this bundle from source:

```bash
# From repo root, with reactot conda env:
cd paper_reproduction_203/code/
python 10_parse_dft_labels.py                # DFT labels from spec23_paper_setup/
python 20_prepare_geometries.py              # xyz + meta.json + Kabsch reordering
python 30_generate_am1_inputs.py             # 1015 .gjf files
python 40_convert_5atom_to_grayson.py        # 3 pkl files
python 50_verify_bundle.py                   # 25 sanity checks
```

Dependencies: `pandas`, `pyarrow`, `numpy` (all in reactot env).

## Fragment identity resolution (important audit trail)

Three orthogonal naming conventions exist and are all reconciled inside `geometries/*/meta.json`:

1. **ORCA filename** (`spe_R_A` vs `spe_R_B` file paths): naming determined by the spec23 input-generator script from Stuyver r0/r1 filename order. For **50/203 rxns** this filename swaps the actual fragment content (i.e., `spe_R_A.inp` holds FRAG2 atoms).
2. **eda.inp fragment tags** (`(1)` vs `(2)` on each atom): defines which atoms belong to FRAG1 vs FRAG2 in the EDA-NOCV calculation. This is the ground truth used by ORCA. Matches 5-atom package `ts_idx_A/ts_idx_B` for 100% of 203 rxns (no pkg swap).
3. **Fragment type** (dipole vs dipolarophile): the 5-atom package's `type_A/type_B` is auto-derived from SMILES/RDKit and **disagrees with the user's manual review for 86/203 (42%) of rxns**. User's `is_A_dipole` flag (derived from where they placed dipole picks) is the authoritative label used here.

`meta.json` per rxn records all three so any downstream tool can re-derive whichever it needs.

## Labels — expected ranges (203-rxn subset)

```
distortion_dft  (total):    4.4 - 175.1 kcal/mol   mean 46.9
distortion_dipole:         -0.2 -  97.9            mean 26.1
distortion_dipolarophile:   0.3 -  85.1            mean 20.7
interaction_dft:         -175.7 - -10.4            mean -65.2
e_barrier_dft:            -42.3 -  42.1            mean -18.3
Bond Energy vs channel sum residual: max 0.02 kcal/mol
```

For paper's ds3 (dr3, N=730), Table 1 reports:
- Dipole distortion range 1.0 – 41.3 kcal/mol
- Dipolarophile distortion range 0.1 – 35.3 kcal/mol
- Interaction range -43.6 – -7.9 kcal/mol

Our ranges are wider (larger max) because our 400-rxn subset includes reactions with higher strain than the paper's cohort.

## Citation

Espley, S. G.; Allsop, S. S.; Buttar, D.; Tomasi, S.; Grayson, M. N. *Digital Discovery* **2024**, *3*, 2479–2486. [DOI:10.1039/D4DD00224E](https://doi.org/10.1039/D4DD00224E)

Data protocol matches Stuyver, T.; Jorner, K.; Coley, C. W. *Sci. Data* **2023**, *10*, 66 (figshare 21707888).
