# 400-rxn 5-atom labeling dataset

**400 [3+2] dipolar cycloaddition reactions** from Stuyver / Jorner / Coley 2023
(figshare 21707888 v5), each manually labeled with the **5 reacting atoms**
(3 dipole atoms + 2 dipolarophile atoms) plus **fragment A/B assignment**.
Self-contained: the files in this folder alone reproduce every label.

## Contents

```
400_rxns_5_atoms_labelling_dataset/
├── README.md                       ← this file
├── 5_atom_picks.json               ← primary source of truth (user-labeled)
├── fragment_assignment.json        ← per-rxn fragA/B atom sets on TS complex
├── manifest.csv                    ← rn ↔ reaction_id ↔ sub_source + counts
├── reconstruct_and_verify.py       ← loads everything + runs consistency checks
└── structures/
    └── rxn_XXXX/                   ← 400 dirs (rxn_0000 … rxn_0399)
        ├── ts.xyz                  ← TS complex (required, all 400)
        ├── d_A.xyz                 ← distorted fragA at TS geometry (all 400)
        ├── d_B.xyz                 ← distorted fragB at TS geometry (all 400)
        ├── r_A.xyz                 ← relaxed isolated fragA (282 rxns; optional)
        └── r_B.xyz                 ← relaxed isolated fragB (282 rxns; optional)
```

## Provenance

| item | source |
|---|---|
| geometries (TS + fragments) | Stuyver et al. 2023 figshare 21707888 v5 → spec19 pipeline |
| fragment A/B assignment | RDKit atom-mapping + subgraph match on TS connectivity (spec19) |
| 5-atom picks | user-labeled via Flask viz app on port 5578 (2026-08-01 → 2026-08-04) |
| electronic-structure labels these support | wB97X-D/def2-TZVP + CPCM(water) SPE + EDA-NOCV (paper setup, in progress at spec23_paper_setup/) |

## Conventions

### `5_atom_picks.json`
```
{ "dipolar_000009": {
    "reaction_id": "dipolar_000009",
    "rn": 0,
    "is_A_dipole": false,              # if true, fragA=dipole & fragB=dipolarophile; else swapped
    "di_ts":  [8, 11, 1],              # 3 dipole atom indices in TS complex order
    "dp_ts":  [4, 5],                  # 2 dipolarophile atom indices in TS complex order
    "di_local_in_r": [4, 7, 1],        # same 3 atoms indexed in isolated R fragment
    "dp_local_in_r": [0, 1],           # same 2 atoms indexed in isolated R fragment
    "locked": true,                    # user confirmed
    "source": "user",
    "saved_at": "2026-08-04T10:27:30..." }
  ...
}
```
- **Indices are 0-based** and match XYZ file atom order.
- `di_1` (middle dipole atom) is the central atom of the dipole (e.g. N of C-N-C, C of C-N-N).
- The 5 picks always live on the same fragment side determined by `is_A_dipole`.

### `fragment_assignment.json`
```
{ "dipolar_000009": {
    "reaction_number": 0,
    "reaction_id":     "dipolar_000009",
    "sub_source":      "spec16",       # or "locked_778"
    "n_atoms_ts": 23, "n_atoms_A": 12, "n_atoms_B": 11,
    "ts_idx_A":  [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22],  # A atoms in ts.xyz order
    "ts_idx_B":  [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14],        # B atoms in ts.xyz order
    "type_A":    "dipolarophile",      # or "dipole"
    "type_B":    "dipole",
    "charge_total": 0, "charge_A": 0, "charge_B": 0,
    "mult_A": 1, "mult_B": 1,
    "r_A_provenance": "opt.xyz_isolated_fragment_opt", ... }
  ...
}
```
- `ts_idx_A ∪ ts_idx_B == {0..n_atoms_ts-1}` (partition).
- `d_A.xyz` = ts.xyz atoms in `ts_idx_A` order (frozen at TS geometry).
- `d_B.xyz` = ts.xyz atoms in `ts_idx_B` order (frozen at TS geometry).
- `r_A.xyz` / `r_B.xyz` exist for 282 rxns whose isolated fragments were separately optimized
  (`r_A_provenance == "opt.xyz_isolated_fragment_opt"`); the other 118 have the same atom
  composition but no relaxed-geometry file (`R.xyz_atom_subset_NOT_optimized`).
- For label extraction (strain / interaction), **only `ts.xyz` + `d_A.xyz` + `d_B.xyz` are required**;
  the r_A/r_B files exist purely for reference/visualization.

### `manifest.csv`
Flat CSV with per-rxn: reaction_number, reaction_id, sub_source, natoms_{ts,A,B},
charge_{total,A,B}, mult_{A,B}, ts_idx_A (str-list), ts_idx_B (str-list),
r_A_provenance, r_B_provenance.

## Reproduce & verify

```bash
python3 reconstruct_and_verify.py             # loads all + consistency checks (should print "400/400 pass")
python3 reconstruct_and_verify.py --dump labels.pkl   # also dumps full merged dict
```

Consistency checks performed:
- Every 5-atom pick index lies inside the correct fragment (dipole picks in dipole fragment, etc.)
- Every pick index is within `[0, n_atoms_ts)`
- ts.xyz / d_A.xyz / d_B.xyz exist and atom counts match manifest
- r_A.xyz / r_B.xyz — verified only if present (optional)

## Coverage summary

| metric | count |
|---|---|
| total reactions | 400 |
| locked (user-confirmed picks) | 400 |
| `is_A_dipole=True` | 206 |
| `is_A_dipole=False` | 194 |
| sub_source `spec16` (auto-derived, user-confirmed) | 208 |
| sub_source `locked_778` (pre-labeled from earlier lock file) | 192 |

## Related work (for context)

- Grayson group Espley et al. 2024 (`doi:10.1039/D4DD00224E`, distortion/interaction ML)
  used a similar 5-atom convention on a smaller subset. Our 400-rxn pipeline mirrors their
  wB97X-D/def2-TZVP + CPCM(water) SPE strategy.
- ORCA EDA-NOCV outputs for these 400 reactions live at
  `/gpfs/tmp_cpu2/yeseo1ee/spec23_paper_setup/{reaction_id}/` (not tracked in git — regenerable).
