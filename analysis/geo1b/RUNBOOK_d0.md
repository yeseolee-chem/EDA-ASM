# RUNBOOK — GEO1b D0 preparation (design step 1-2, results/D0D1_design_investigation.json)

All compute is sbatch (CLAUDE.md): `--time=48:00:00` comes from jobs/cpu.slurm. Before every sbatch:
`squeue -u $USER -h | wc -l` (< 20) and `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"` (take the partition with idle CPUs).
Logs: /gpfs/tmp_cpu2/yeseo1ee/geo1b/logs/<job-name>_<jobid>.out.

## LAB: d0_labels.py (design step 1: LADDER Q rule on the D0 labels; decisions results/DECISIONS_LAB.md)

Needs nothing new: labels_all.json, full_dataset.csv, the Coley profiles and label_true/work/inputs (no G2c
dependency), so it can run at once, before d0_pairs.py. Env: conda reactot through `jobs/d0lab.slurm` (also
`--time=48:00:00`; the D1 venv of jobs/cpu.slurm has no pyarrow for Q_keys_d0.parquet).

    cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/geo1b
    sbatch -p <cpu1|cpu2> -J gb_d0probe jobs/d0lab.slurm d0_labels.py probe 0 1 2 5 3312 3323   # optional smoke test
    sbatch -p <cpu1|cpu2> jobs/d0lab.slurm d0_labels.py run

probe prints the keys, states and Q-a of the listed rxn_ids as JSON and writes nothing (exit 0 = keys built, Q-a ok).
run: pass = `=== exit 0` in the log and `results/Q_d0_report.json` with `all_pass` true, `stop` false and every
`gates.*.ok` true. Outputs: `$GEO1B/d0/labels_d0_q.json` (5,260 rows), `$GEO1B/d0/Q_keys_d0.parquet`,
`$GEO1B/d0/Q_audit_d0.csv`; their sha256 are in the report `outputs`. Look at `design_key_comparison`,
`cfg_all_key_comparison`, `audit.nonalt_dipole_replacements_across_cfg` (cross-rotamer replacements of non-alt
dipoles), `reassembly`, `hygiene` and gate `Q-2s` (references with an `|err` key, `cr.unkeyed`) before step 2.
Key rule (code geo1b-d0q-2, DECISIONS_LAB L-key-*): `<kind>:<SMILES>` + `|alt` + `|cfg:` on alt dipole references only
+ `|s3d:` on every reference. Run d0_labels.py with this rule before the first d0_pairs.py build and GEO1b D.

A bond-graph difference within a key or within one SMILES (gates `Q-2` / `Q-2p`; `Q-2p.graph_classes` lists the
majority and minority references), or an `|err` reference whose bond graph is not its SMILES graph (`Q-2m`), is a
STOP for a decision (design step 1 and refs_q.py; DECISIONS_LAB L-q2). Nothing valid is written in that case.

Rerun: the same command (idempotent: unchanged inputs, code, main-checkout labels_all.json / stage3_parse.py and an
unchanged (path, size, mtime) digest of the reference xyz and frag{k}_rel.inp/.out give "up to date", exit 0).
`run --force` recomputes everything. Exit 3 = STOP: `stop_reason` in the report, the outputs are renamed `*.stale`
(never valid), a Q-a .. Q-e failure also leaves `$GEO1B/d0/labels_d0_q.REJECTED.json`. Any rerun that changes
labels_d0_q.json must be followed by a d0_pairs.py rerun. Cost: one job, 2 cores, estimated under 1 h (about 10.5k
reference geometries through RDKit / networkx, about 650 MB of frag_rel.out read, about 31.5k file stats).

GEO2b P (design step 8) reads `Q_keys_d0.parquet` next to `$LAD/Q_keys.parquet`: D0 keys are within D0 and never equal
a D1 key (every D0 key has `|s3d:` or `|err:`), so the union is a plain concatenation and refs='min' stays per
dataset. Its `constrained` column = `alt` = "the key carries a dihedral state" (the role D1 `constrained` has in the
key, not D1's rotatability flag); read the states from `alt`, `fixed_state` (state in the key), `cfg_state` and
`stereo3d` (DECISIONS_LAB L-constrained, L-namespace).

## PAIR: d0_pairs.py (design step 2: D0 components, anchor-tree pairs, folds)

Needs: `$GEO1B/d0/labels_d0_q.json` + `results/Q_d0_report.json` from a d0_labels.py pass (`all_pass` true,
`stop` false, and the labels sha256 under `outputs` = the file now on disk; anything else is a STOP). Run it after
`select_g2c.py build` has finished (g2c worktree `analysis/g2c/results/C_select.csv` and `C_select.json` and
`$G2C/d0/g2_final` all exist), so the anchor priority uses the G2c TSs. While the build is running (only part of
that output exists) the run STOPs. Run before the build has written anything, it uses the D0-G2 orig folders
(`g2_source` = `d0_g2_orig`, `preliminary` true, `status` "ok_preliminary"): GEO1b `data.py` refuses that report, so
rerun after the build. The folds stay the same (they never depend on G2 availability, nor on labels_d0_q.json: the
leaf choice reads the Coley labels of labels_all), only the pairs change.

    cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/geo1b
    sbatch -p <cpu1|cpu2> -J gb_d0pairs --cpus-per-task=2 --mem=8G jobs/cpu.slurm d0_pairs.py build

Pass = `=== exit 0` in the log and `results/D0_pairs_report.json` with `stop` false, `g2_source` "g2c" and
`g2_info.c_select_sha256` = sha256 of the current C_select.csv (GEO1b data.py checks both; a rebuilt G2c selection
needs a d0_pairs.py rerun). Look at `components.dipole_fold_span` (a dipole molecule in several folds only where twins
force it), `twins.by_kind` / `n_m1_any` and `pairs.n_ok_m1_twin`.
Outputs: `$GEO1B/d0/d0_folds.csv` (read `twin_of` with `dtype=str, keep_default_na=False`), `$GEO1B/d0/D0_pairs.csv`,
`$GEO1B/d0/d0_components.csv`, `$GEO1B/d0/d0_twins.csv` (every D0 reaction with a stereo-free D1 twin: all twins and
their T1 folds, kind, m1 flag; tests.py U-9b and the GEO2b T1D0 split read it); their sha256 are in the report `outputs`.

Rerun: the same command (idempotent; unchanged inputs give byte-identical outputs). Exit 3 = STOP, the reason is in
the report `fails`; no output is written on a STOP. If `d0_folds.csv` would change, the run STOPs (frozen folds).
Use `jobs/cpu.slurm d0_pairs.py build --force` only when no GEO1b D / R output has used the old d0_folds.csv.
A twin conflict (D1 twins in several T1 folds that the isomeric match cannot settle; report `twins.conflicts`) is a
STOP for a user decision. Only if the user accepts the lowest-D1-rxn_id rule: add `--accept-twin-conflicts` to this
and every later run (without it the next run STOPs again); the accepted reactions stay listed (kind 'conflict').
Cost: well under 1 core-h (RDKit canonical SMILES of about 34k strings, a few dict passes).

## GEO1b D -> baseline -> U (GEO1B.md 3-1 / 3-2 with decisions A-G2c and D0D1; order = DECISIONS_GB GB-order)

The order is required; each later step STOPs or FAILs without the output of the earlier one:
d0_labels.py pass -> `select_g2c.py build` (g2c worktree) -> d0_pairs.py build (`g2_source` "g2c") -> D (`data.py`)
-> baseline (`data.py baseline`) -> U (`tests.py synth`, `tests.py data`) -> R (`r_launch.py`, GEO1B.md 3-3).
Pick the partition with idle capacity first (`sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`;
`sinfo -p gpu3,gpu4,gpu5 -o "%.9P %.6t %.10G %C"`).

    cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/geo1b
    sbatch -p <cpu1|cpu2> --cpus-per-task=4 --mem=32G -J gb_data jobs/cpu.slurm data.py
    sbatch -p <cpu1|cpu2> --dependency=afterok:<gb_data job id> -J gb_baseline jobs/cpu.slurm data.py baseline
    sbatch -p <gpu partition> --gres=gpu:1 -J gb_u_synth jobs/gpu.slurm tests.py synth
    sbatch -p <gpu partition> --gres=gpu:1 --dependency=afterok:<gb_baseline job id> -J gb_u_data jobs/gpu.slurm tests.py data

D: pass = `=== exit 0` and `results/D_dataset_report.json` against the GEO1B.md 3-1 table read with A-G2c (the G2
counts are the G2c counts, DECISIONS_GB GB-g2c-field / GB-g2-count); data.py also needs `$GEO1B/d0/d0_folds.csv`,
`d0_twins.csv` and a passing `results/D0_pairs_report.json`. The optional data.py negative test (DECISIONS_GB
GB-negtest) runs before this real build.
baseline: writes `results/G2RAW_baseline.json` (G2c-raw primary metric, frozen before training; its `records_sha256` =
sha256 of `$GEO1B/data/records.pkl`). tests.py U-8b and evaluate.py (E1) STOP / FAIL without it or with another
records sha. Rerun after a rebuilt records.pkl; `data.py baseline --force` only before any training has used it.
U: every check PASS except U-7 (report only). U-8b also runs the single-run regression (DECISIONS_GB GB-U8r), so
`/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final` must still exist.
