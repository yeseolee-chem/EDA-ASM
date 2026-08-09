# VERSION.md

## Package

- **Name**: tt_eda_kisti_package
- **Version**: v1.0
- **Built**: 2026-08-06
- **Built by**: eda-asm-prediction repo, yeseolele@gmail.com
- **Origin HPC**: gate1 (172.16.10.36)

## Data provenance

- **Source archive**: `data_archive_files.zip`
  - URL: https://researchdata.bath.ac.uk/1480/1/data_archive_files.zip
  - DOI: 10.15125/BATH-01480
  - License: CC-BY 4.0
  - Size: 4,302,424,907 B (4.006 GB)
  - MD5 (base64): `q/qQxvnwAl0WObvXQ75T9g==`
  - MD5 (hex): `abfa90c6f9f0025d1639bbd743be53f6`
- **Paper**: Espley, S. G.; Allsop, S. et al. Digital Discovery 2025.
  DOI: 10.1039/D4DD00224E
- **Code repo (Espley)**: https://github.com/the-grayson-group/distortion-interaction_ML
  - Files used from repo: `diassep/diassep.py`, `energy_extraction/get_energies.py`,
    `feature_extraction/features/tt_solvent_features.pkl` (for filter definition
    and reference values)

## Method (matches Espley 2024 SPE protocol)

- Functional: **B3LYP**
- Dispersion: **D3(BJ)** (Grimme-D3 with Becke-Johnson damping)
- Basis: **def2-TZVP**
- Solvation: **CPCM with SMD parametrization, solvent = water**
- Software: **ORCA 6.x** (backwards-compatible with 5.0.4+ for SMD)
- TS geometry: AM1 opt+freq with implicit solvent (from Espley's
  `optimisations/am1_ts/ts_<N>.out`), NOT re-optimized at DFT
- Fragment partition: `diassep.py` adjacency-matrix mode applied to AM1 TS
  imaginary vibration mode
- Fragment charges/multiplicities: (0, 1) + (0, 1) for both fragments
  (all tt reactions confirmed neutral closed-shell singlet)

## Filter chain

- Starting: 3,980 unique reactions in `three_two_cycloaddition/` (rxn_id 0..5973)
- After "complete 5-file set" filter (am1_ts + splits/dft × 4): 3,662
- After "reaction_number in Espley's ML dataset" filter: 3,504
- After "diassep partition matches Espley frag counts": **3,504** (same;
  100% partition match on Espley subset)

Note: 6 Espley reactions omitted from our set due to diassep failures
(multiple imag freqs or no adjacency changes at AM1 TS). These are
`ts_1615, ts_2122, ts_2452, ts_3218, ts_3340, ts_3538`.

## Output channels (per reaction, from ORCA output)

| Column | Description |
|---|---|
| `e_bond_kcal` | Total interaction energy (= sum of decomposition, negative = attractive) |
| `e_pauli_kcal` | Pauli repulsion (positive) |
| `e_elst_kcal` | Electrostatic (dispersionless, Bickelhaupt-Baerends) |
| `e_orb_kcal` | Orbital (from ETS-NOCV) |
| `e_xc_kcal` | Δ E⁰(XC) (exchange-correlation deformation) |
| `e_disp_kcal` | Dispersion (D3BJ) |
| `e_cpcm_kcal` | Δ CPCM dielectric (solvation electrostatic) |
| `e_smd_cds_kcal` | Δ SMD CDS correction (solvation non-electrostatic) |
| `nocv_top5` | Top-5 NOCV eigenvalue pairs + their DE_k/DT_k/DV_k |
| `terminated_normally` | Bool: reaction converged (grep "ORCA TERMINATED NORMALLY") |

## Sign convention warning

Our `e_bond_kcal` uses standard Bickelhaupt-Baerends convention:
attractive interaction is negative. Espley's `interaction_energies_dft`
uses inverted convention (positive magnitude, computed as
`dist_energy - barrier`).

Relationship: `Espley_interaction_dft ≈ -1 × our_e_bond_kcal` (within
method noise).

The `verify_vs_espley.py` script computes MAE of `our_e_bond + Espley_interaction`;
small MAE (< 3 kcal/mol) confirms consistent implementation.

## Reproducibility

To rebuild from source archive (if package data is lost):

1. Download the Bath 1480 zip (SHA1/MD5 above).
2. Install `cclib`, `xyz_py`, `molml`, `numpy`, `pandas`.
3. Run `scripts/build_manifest.py` (not shipped by default — see
   originating repo `eda-asm-prediction/analysis/bath1480_probe/`).
4. The generated `data/rxn_XXXX/eda.inp` files must be bit-identical.

## Compute cost estimate

- Per-reaction wall (ORCA 6.x on 4 CPU cores):
  - 15-20 atoms: 5-7 min
  - 25-45 atoms: 10-30 min
  - 50-91 atoms: 30 min - 4 h
- Total: 3,504 reactions × ~20 min avg × 4 cores = ~4,700 core-hours

- On KISTI Nurion student allocation (typical 40-160 core concurrent limit):
  - 40 concurrent × 4 cores/rxn = 10 rxns simultaneous → ~120 h wall
  - 160 concurrent × 4 cores/rxn = 40 rxns simultaneous → ~30 h wall
  - Adjust `--array=0-3503%K` to match your allocation.
