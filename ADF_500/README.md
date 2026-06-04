# ADF_500 — ASR spec ADF calculation archive

Snapshot of all files related to the 500-reaction BP86/D3BJ/TZ2P/Good ADF
ASR-spec batch (per `ASR_Fragmentation_Spec.md`).

Generated: 2026-05-22.

## Folder layout

```
ADF_500/
├── README.md                this file
├── scripts/                 spec runner + batch driver + recovery helper
│   ├── run_asr_spec.py            Per-reaction BP86/Good ADF runner (11 jobs/rxn)
│   ├── run_asr_spec_batch.sh      xargs -P 4 batch launcher
│   ├── run_stage5a_fragmentation.py    Stage 5a fragmentation
│   └── recover_asr_results.py     Re-parse spec JSONs from /tmp rkfs
├── cohort/                  reaction-set definition
│   ├── selected_reactions.csv     500 reaction IDs + halo8 metadata
│   ├── asr_spec_rxn_list.txt      500 IDs (one per line, used by batch)
│   └── db_idx_map.json            {rxn_id → halo8 source_db_idx}
├── stage5a/                 fragmentation classification (input to runner)
│   ├── fragmentation_summary.json     500 rxn → pattern + n_fragments
│   ├── review_log.json            Dashboard review status (minimal rebuild)
│   ├── frames_cache.pkl           Cached Halo8 frames (R/TS/P × 500)
│   └── per_reaction/<rxn_id>/result.json    fragment assignments per rxn
├── results/                 ADF result JSONs (Spec section 7 schema)
│   └── <rxn_id>.json              500 files
├── logs/                    per-reaction job stdout/stderr
│   └── <rxn_id>.log
├── docs/
│   └── ASR_Fragmentation_Spec.md  reference spec
└── raw_rkfs/                AMS workdirs with .rkf binary files
    └── <rxn_id>/  (≈32 GB total, **239 / 486 workdirs** preserved)
        ├── whole_R, whole_TS, whole_P      whole-system SP
        ├── frag_<role>_R/TS/P              fragment SP × 3 ζ
        ├── frag_<role>_opt                 fragment geometry optimization
        └── EDA_R, EDA_TS, EDA_P            coupled ETSNOCV EDA × 3 ζ

⚠️ raw_rkfs/ is partial (49 %). GPFS user quota was exceeded mid-archival
   and the remaining 247 workdirs were lost from /tmp before the move
   completed. Result JSONs in `results/` are complete (500/500) and contain
   all spec section-7 fields needed for downstream ML, so this is an audit
   limitation only, not a data-content loss.
```

## ADF settings (`scripts/run_asr_spec.py`)

| Setting             | Value             |
|---------------------|-------------------|
| Functional          | BP86 + D3(BJ)     |
| Basis               | TZ2P, frozen core None |
| Relativity          | ZORA scalar       |
| Integration         | Becke Good        |
| SCF threshold       | 1e-6              |
| Max SCF iterations  | 200               |
| Symmetry            | NoSym             |
| Spin                | Unrestricted iff any fragment has multiplicity > 1 |

11 ADF jobs per reaction × 500 = 5500 SCF calculations (plus 1 reaction with
3-fragment override using 18 jobs).

## Result schema (`results/<rxn_id>.json`)

Per `ASR_Fragmentation_Spec.md` section 7. Top-level keys:

- `reaction_id`, `schema_version`
- `halo8_meta`         source / n_heavy_atoms / frame indices / Halo8 Ea
- `pattern`            P0_BIMOL | P1_OPEN | P2_CLOSED | P3_TETHER | P4_DISSOC | P5_HSHIFT
- `fragmentation`      fragments[] + spin_signs + total_spin_polarization + coupling
- `adf_settings`       methodology fingerprint
- `irc_points`         R / TS / P frame_idx + ADF total energy (kcal/mol)
- `fragment_opt_energy_kcal`    {role → E (kcal)}
- `asr_vector_kcal`    {R, TS, P} × {strain, elst, Pauli, oi, disp} in kcal/mol
- `consistency`        sum_components_TS, ΔE_DFT_TS, diff_TS, S2_per_fragment
- `diagnostics`        10 spec checks (PASS/FAIL)
- `scf_convergence_per_job`     {job_name → bool}
- `status_at_queue`    AUTO_ACCEPT_CANDIDATE | MANUAL_REVIEW_REQUIRED | FAILED
- `manual_review_reasons`       list of failed-diagnostic codes

## Final status summary (500 rxns)

| Status                  | Count  |
|-------------------------|--------|
| MANUAL_REVIEW_REQUIRED  | 312*   |
| RECOVERED_FROM_RKF      | 179**  |
| AUTO_ACCEPT_CANDIDATE   | 8      |
| FAILED                  | 0***   |

\* Dominant reason `sum_consistency`: spec sum-check (`|Σ ASR_i − ΔE‡_DFT| < 2`)
  natively applies to bimolecular dissociation only; P5/P2 intramolecular
  reactions naturally fail this. Re-evaluate downstream as needed.

\*\* `RECOVERED_FROM_RKF`: 179 result JSONs were re-parsed from raw rkfs
  after an `outputs/asr_spec/` directory loss event. Their data is identical
  to a normal completion but lacks the `wall_time_min` field.

\*\*\* 1 reaction (`T1x_C3H5NO2_rxn01106`) initially FAILED in 2-fragment
  setup; was rerun with user-specified 3-fragment split (O atom as own
  fragment, NH triplet, O triplet, BS-singlet coupling). Final status:
  MANUAL_REVIEW_REQUIRED with `sum_consistency` + `Pauli_range` (expected
  for 3-body extension of 2-body diagnostics).

Top diagnostic-fail reasons across the 312:
- 298× sum_consistency · 15× Pauli_range · 14× scf · 13× FRAGMENTATION_3_BODIES
- 13× strain_sign · 2× oi_sign · 1× FRAGMENTATION_4_BODIES · 1× wall_time

## How to re-run a single reaction

```bash
source $HOME/ams2026.103/amsbashrc.sh
export PYTHONPATH=src:.
NSCM=1 $AMSBIN/amspython scripts/run_asr_spec.py --rxn_id <RXN_ID>
```

Result lands in `results/<RXN_ID>.json`. The runner reads cohort/db_idx_map.json
to find Halo8 source DB, then reads stage5a/per_reaction/<rxn>/result.json for
the fragment assignment.

## Re-derive everything from raw_rkfs

If only `raw_rkfs/` survives, use `scripts/recover_asr_results.py` to rebuild
`results/<rxn_id>.json` for every workdir.

## Units

All ADF energies in **kcal/mol**.
Halo8 reference energies in **eV** (preserved as-is in `irc_points.energy_eV_halo8`).
Geometries in **Å**.
