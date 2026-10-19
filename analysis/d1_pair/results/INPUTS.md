# D1PAIR rev 9 — inputs (Phase 0, §1)

written 2026-10-19 10:08 by `d1_phase_p.py inputs` (compute node)

| input | path | sha256 | expected | ok |
|---|---|---|---|---|
| D1 labels | `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/results/labels_d1.json` | `1eaa417102e20813beea9ba190ef8140233a4a9b0b1d9a69afac16aeaeb210a7` | `1eaa417102e20813beea9ba190ef8140233a4a9b0b1d9a69afac16aeaeb210a7` | yes |
| D1 status | `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/results/d1_status.csv` | `a9df873b5021fb11e0d2bd2b2ba1b012abad9e7981f32b676172ec269cabfcf7` | `a9df873b5021fb11e0d2bd2b2ba1b012abad9e7981f32b676172ec269cabfcf7` | yes |
| D0 labels (repo root) | `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1pair/labels_all.json` | `62c0e7845045d0810627e88f1f52a3f353059bd78ac12ed89f8cd0f053a08753` | `62c0e7845045d0810627e88f1f52a3f353059bd78ac12ed89f8cd0f053a08753` | yes |
| D0 features X182_all | `/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/xtb_r6/X182_all.parquet` | `e79791b76ea6257cb9d19f0f7e6cd2a16d467b6d2fe9c40ca205f1717f8be254` | `e79791b76ea6257cb9d19f0f7e6cd2a16d467b6d2fe9c40ca205f1717f8be254` | yes |
| D0 targets S_targets | `/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/S/S_targets.parquet` | `6b02630fd73dd37d6fdb9792063e3094e8e8d8b2cd1353bc13b1ad91b90272bf` | `6b02630fd73dd37d6fdb9792063e3094e8e8d8b2cd1353bc13b1ad91b90272bf` | yes |
| D0 rows S_splits_appendixA | `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1pair/analysis/espley_xtb_repro/results_rev6/S_splits_appendixA.csv` | `b08a700fb47d32fca7e7c9086328a8073a7e87e8830cf8817fa149f5fc777a78` | `b08a700fb47d32fca7e7c9086328a8073a7e87e8830cf8817fa149f5fc777a78` | yes |

- D0 `labels_all.json`: 5265 rows (the D1 interim file `a2df136b…` had 9,459 rows; this one is not that file).
- D1 `labels_d1.json`: 9496 rows, all status ok (checked by `d1pair_common.load_d1_labels`).

## <D1_JOBS>

No unpacked job folder with `T00001/ts.xyz` and `T00001/sp/eda.inp` exists on the server (the Nurion job folders are in the 23 bundle tars of `D1_build/nurion/`, `sp/eda.inp` inside each `jobs/<T>/archive.tgz`; 58 rows remain as UBAI folders in `d1_scratch/jobs/`). Phase 0 STOP -> user decision 2026-10-19: extract read-only into scratch (`d1_extract_jobs.py`).

- `<D1_JOBS>` = `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs`
- extracted ok rows: 9496 / 9496; sources {'nurion': 9438, 'ubai': 58}
- ts.xyz / ref_dipole.xyz / ref_dipolarophile.xyz byte-identical to `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/results/geometries/<T>/`: 9496 / 9496
- label.json job_id / rxn_id = labels_d1.json: 9496; file fields equal: 9496
- `MANIFEST.tsv` sha256 `ad8757efa51d712da32e67c63471b72f8d67ce8348ba537543d46ca7fd2dbbac`
- the 8659 G1 / pair job folders with all of ts.xyz, ref_dipole.xyz, ref_dipolarophile.xyz, label.json, sp/eda.inp: 8659 (missing: [])
