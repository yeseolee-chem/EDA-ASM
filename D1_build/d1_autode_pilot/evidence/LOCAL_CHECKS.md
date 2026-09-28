# Checks run before hand-over (sandbox, no ORCA; 2026-09-29)

| check | how | result |
|---|---|---|
| autodE writes the intended ORCA inputs | `pilot_common.configure_autode` + `Calculation.generate_input` (autodE 1.4.5, no run) | `! OptTS Freq B3LYP RIJCOSX D3BJ def2-SVP def2/J CPCM(Water)` + `%geom Calc_Hess true Recalc_Hess 30 Trust -0.1 MaxIter 100 end` + `%cpcm smd true SMDsolvent "water" end`; SP uses def2-TZVP |
| TS configuration tags on real TS geometries | `tests/validate_config_on_coley.py` on all 5,260 D0 ok reactions | 0 errors; 0 undefined tags; `_alt` matches TS in 2,777/2,940 (94.5 %) → `cfg_validation_D0.csv` |
| imaginary-mode forming-bond share | Coley `TS_imag_mode.xyz`, all 5,260 | min 0.546 (gate 0.5) |
| regio (crossed pairing) | all 5,260 | 4 < 0 → flag only |
| xtb 6.7.1 scan output = Coley parser | `tests/xtb_scan_format.py` | 60 `unbiased energy:` + 60 `bias energy:` lines |
| input parity with the repo D0 builder | `tests/parity_d0.py --rxn 0 20 105 3232` | 20/20 files byte-identical |
| replay chain ts→ref→inputs | J12/J13 (D0 rxn 20/105) | inputs identical to the repo builder; Coley-port decision = Coley's (xtb stand-in for DFT) |
| label glue | `tests/fake_orca_outputs.py` (synthetic outputs from labels_all energies) + `assemble.py` | labels_all reproduced to 3e-6 kcal/mol; all gates true; D0 audit `check()` no problems |
| bash stage logic | `run_stage` / `run_sp` with a fake ORCA | done/skip/fail markers and timing as intended |

Not testable here (S0 does it on the cluster): autodE parsing of ORCA 6.1.1 outputs; real ORCA EDA
output formats through `smd_relabel_audit.check`; parity against the D0 input files on disk.
