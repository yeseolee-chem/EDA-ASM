# D1 autodE pilot — report

engine: `orca`; jobs: 14 (D1 10, D0_control 2, D0_replay 2)

## Acceptance

| criterion | value | pass |
|---|---|---|
| A1 D1 TS rows with an ok label >= 8/10 | 8/10 | PASS |
| A2 every produced label passes all gates | 0 with failed gates | PASS |
| A3 D0_replay reproduces labels_all (9 targets, |Δ| <= 0.1 kcal/mol) | max |Δ| = 0.0030 over 2/2 | PASS |
| A4 Coley-port decision == Coley's alt choice on the replay controls | [True, True] | PASS |
| A5 D0_control |Δbarrier| <= 1.0 (diagnose if not; not a hard stop) | max 2.01 over 2/2 | FAIL |

## Jobs

| job_id | kind | ts_id | core | n_atoms | last_stage | fail_stage | imag_cm | d_form_1 | d_form_2 | ts_cfg | reac_cfg | frozen | alt_used | status | coreh_total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J00 | D1 | D1-2823-r1 | A5 | 38.00 | label |  | -312.57 | 2.24 | 2.29 |  |  | 0 | 0.00 | ok | 11.15 |
| J01 | D1 | D1-3422-r2 | B2 | 40.00 | label |  | -453.04 | 2.17 | 2.55 |  |  | 0 | 0.00 | ok | 22.82 |
| J02 | D1 | D1-3449-r2 | B3 | 29.00 | label |  | -450.49 | 2.17 | 2.56 |  |  | 0 | 0.00 | ok | 8.79 |
| J03 | D1 | D1-3499-r1 | C2 |  |  | ts |  |  |  |  |  | 0 |  |  | 1.93 |
| J04 | D1 | D1-3556-r1 | C4 | 24.00 | label |  | -358.22 | 2.28 | 2.35 | t | c | 1 | 1.00 | ok | 5.64 |
| J05 | D1 | D1-3667-r1 | C8 | 48.00 | label |  | -360.15 | 2.19 | 2.31 |  |  | 0 | 0.00 | ok | 19.05 |
| J06 | D1 | D1-3693-r1 | C9 | 32.00 | label |  | -438.08 | 2.20 | 2.37 |  |  | 0 | 0.00 | ok | 9.42 |
| J07 | D1 | D1-3842-r1 | D5 | 35.00 |  | ts | -360.69 | 2.17 | 2.33 |  |  | 0 |  |  | 8.30 |
| J08 | D1 | D1-3740-r1 | D1 | 17.00 | label |  | -432.33 | 2.14 | 2.16 |  |  | 0 | 0.00 | ok | 3.04 |
| J09 | D1 | D1-1417-r1 | A3 | 31.00 | label |  | -295.80 | 2.14 | 2.76 | c | c | 0 | 0.00 | ok | 11.82 |
| J10 | D0_control | D0-0020 | D0 | 25.00 | label |  | -186.75 | 2.23 | 2.67 | t | c | 1 | 1.00 | ok | 7.75 |
| J11 | D0_control | D0-0105 | D0 | 27.00 | label |  | -392.82 | 1.91 | 2.72 |  |  | 0 | 0.00 | ok | 17.70 |
| J12 | D0_replay | D0-0020 | D0 | 25.00 | label |  |  | 2.19 | 2.61 | t | c | 1 | 1.00 | ok | 2.63 |
| J13 | D0_replay | D0-0105 | D0 | 27.00 | label |  |  | 1.96 | 2.61 |  |  | 0 | 0.00 | ok | 3.56 |

### Failures

- **J03** (D1-3499-r1, C2) at `ts`: autodE found no transition state
- **J07** (D1-3842-r1, D5) at `ts`: TS check failed: one_imag

## Labels (kcal/mol, method ①)

| job | ts_id | barrier | d1 | d2 | elst | pauli | oi | disp | cpcm | cds | autodE ΔE‡(sp) | autodE ΔG‡ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J00 | D1-2823-r1 | 13.37 | 17.92 | 5.94 | -38.91 | 74.24 | -37.63 | -7.37 | 0.07 | -1.01 | 13.35 | 26.47 |
| J01 | D1-3422-r2 | 11.98 | 15.89 | 5.52 | -31.12 | 63.09 | -33.51 | -5.82 | 0.57 | -2.70 | 11.91 | 23.56 |
| J02 | D1-3449-r2 | 11.58 | 15.76 | 5.43 | -30.76 | 61.50 | -32.53 | -5.75 | 0.54 | -2.65 | 11.58 | 22.64 |
| J04 | D1-3556-r1 | 11.99 | 19.41 | 5.12 | -42.53 | 79.90 | -42.35 | -7.77 | 2.35 | -2.39 | 6.44 | 18.50 |
| J05 | D1-3667-r1 | 12.47 | 17.04 | 3.49 | -39.32 | 73.67 | -35.97 | -7.82 | 0.83 | 0.18 | 12.44 | 25.02 |
| J06 | D1-3693-r1 | 11.53 | 16.66 | 5.01 | -31.79 | 62.35 | -34.16 | -5.27 | 1.53 | -2.90 | 11.55 | 20.15 |
| J08 | D1-3740-r1 | 13.99 | 13.59 | 11.65 | -53.80 | 99.90 | -56.07 | -6.46 | 8.18 | -3.36 | 13.92 | 26.20 |
| J09 | D1-1417-r1 | -3.30 | 10.24 | 4.51 | -27.22 | 73.92 | -44.86 | -11.28 | -3.49 | -4.12 | -3.22 | 11.11 |
| J10 | D0-0020 | -8.84 | 8.91 | 10.98 | -39.68 | 78.41 | -45.61 | -12.70 | -7.55 | -2.70 | -13.53 | 1.94 |
| J11 | D0-0105 | 9.38 | 16.53 | 11.54 | -57.52 | 130.57 | -72.42 | -11.14 | -4.50 | -3.21 | 9.39 | 22.50 |
| J12 | D0-0020 | -8.48 | 10.12 | 12.00 | -48.31 | 87.73 | -49.92 | -12.68 | -6.23 | -2.68 |  |  |
| J13 | D0-0105 | 11.39 | 19.22 | 11.04 | -60.94 | 120.86 | -68.55 | -11.15 | 4.18 | -3.14 |  |  |

## D0 controls (label − labels_all)

| job | kind | rxn | Δbarrier | Δd1 | Δd2 | Δelst | Δpauli | Δoi | Δdisp | Δcpcm | Δcds | Δd_form max (Å) | ΔG‡ − Coley G_act | alt (Coley / ours) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J10 | D0_control | D0-0020 | -0.366 | -1.208 | -1.016 | 8.632 | -9.324 | 4.313 | -0.021 | -1.325 | -0.023 | 0.059 | -4.22 | 1.0 / 1.0 |
| J11 | D0_control | D0-0105 | -2.007 | -2.694 | 0.500 | 3.423 | 9.701 | -3.874 | 0.010 | -8.674 | -0.067 | 0.117 | -2.09 | 0.0 / 0.0 |
| J12 | D0_replay | D0-0020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  | 1.0 / 1.0 |
| J13 | D0_replay | D0-0105 | -0.000 | -0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  | 0.0 / 0.0 |

## Dipole configuration and reference choice

| job | core | TS cfg | reactant cfg | frozen dihedrals | to_run | alt used | Coley-port reason | lowest-compatible rule would use alt |
|---|---|---|---|---|---|---|---|---|
| J04 | C4 | t | c | 1 | True | 1.0 | constrained; RMSD 0.718 > 0.05 A | True |
| J09 | A3 | c | c | 0 | False | 0.0 | xtb gate: dE_xtb -0.001 >= -0.1 kcal/mol | False |
| J10 | D0 | t | c | 1 | True | 1.0 | constrained; RMSD 0.441 > 0.05 A | True |
| J12 | D0 | t | c | 1 | True | 1.0 | constrained; RMSD 0.441 > 0.05 A | True |

## Cost

core-hours per D1 TS row: mean 11.5, median 10.3, max 22.8 (stages: ts 8.2, ref 0.5, sp 2.8)

extrapolation to 2846 backbone TS rows: 32,633 core-hours; at MaxJobs=10 × 8 cores ≈ 17 days wall (mean-based; the max job sets the tail).
