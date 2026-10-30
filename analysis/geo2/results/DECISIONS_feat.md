# GEO2 decisions — role FEAT (g3c.py, feat_geo.py, jobs/geo2.slurm)

Every deviation from a verbatim copy of `analysis/d1_g3/g3_common.py` / `feat.py` / `jobs/g3.slurm` and every
judgment call, with file:line (line numbers at the time of writing). The runtime copy of the decision texts is the
`DECISIONS` dict of `feat_geo.py` (keys `feat_geo/<key>` in `results/DECISIONS.md`). Nothing was run (login node).

## g3c.py (copy of g3_common.py, spec 6-1)

- g3c.py:23 — `G3 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2"))`. The variable keeps the
  name `G3` so that every `C.G3` in the copied feat code stays unchanged; it now means `$GEO2`.
- g3c.py:22 — `RES = HERE / "results"` is byte-identical to g3_common.py; because the file lives in `analysis/geo2`
  it already resolves to `analysis/geo2/results` (spec 6-1 "RES -> analysis/geo2/results"), so no code change.
- g3c.py:32 — `METHODS = ("geo",)`.
- g3c.py:1-10, 56, 104 — text only: module docstring, the `assert_path` STOP message (`$GEO2 / analysis/geo2/results`)
  and the header written when `results/DECISIONS.md` does not exist yet (`# GEO2 결정 기록 ...` instead of
  `# D1-G3 결정 기록 ...`). LAD, LAD_DIR, LAD_RES, G2, G2_RES, HF_DIR, HF_SCR, P_DIR, SEED_S, SEED_BOOT, KCAL and every
  function body are unchanged.

## jobs/geo2.slurm (copy of g3.slurm)

- jobs/geo2.slurm:2, 6, 15-18, 20-21 — `--job-name=geo2`, default log `$GEO2/logs`, `G3ENV` -> `GEO2ENV`,
  `D1G3_SCRATCH` -> `GEO2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo2`, `D1G3_JOB_T0` -> `GEO2_JOB_T0`, echo tag `[geo2]`;
  usage comment. The command loop (`:::` / `;;;`) is unchanged.

## feat_geo.py (copy of feat.py, spec 6-1 table)

- feat_geo.py:1-80 — module docstring rewritten for GEO2 (targets, CLI, preconditions); no D1-G3 Part A / Phase O /
  U-1 text left.
- feat_geo.py:109 — the method constant is named `GEO` (= `"geo"`), not `M`: inside `order_g` / `g_one` `M` is the
  ORCA-module namespace of feat.py (kept verbatim), so a module-level `M` would be shadowed and confusing.
- feat_geo.py:110 — `T0` reads `GEO2_JOB_T0` (geo2.slurm) instead of `D1G3_JOB_T0`; `D1HF_JOB_T0` fallback kept.
- feat_geo.py:113-117 — spec 6-2 STOP thresholds replace D1-G3's single `RATE_MAX = 0.05` (D1-G3 spec 12): G order
  0.5 %, G Freq 1 %, B6 1 %, F-1 / Q-a / T-a 1 % (taken from `g0f0.ORDER_MAX / FREQ_MAX / B6_MAX / F0_RATE_MAX`),
  H run failures 1 %, H input check 1 %, rel-energy gate 0.5 % at 1e-6 Eh, pilot 50 reactions / 2,500 core-h.
- feat_geo.py:132 — byte-budget variable renamed `GEO2_G_BUDGET_GB` (default 2 GB, unchanged meaning: per `run_pool`
  call = per G slice). With 4 slices ~1.2 GB per slice is expected (LADDER G0: 5.28 GB for 9,266 folders), so the
  runbook uses 4 slices; 1-2 slices would hit the budget STOP. `run_pool` itself is verbatim.
- feat_geo.py:143-151 — `CODE_REL` lists `analysis/geo2/feat_geo.py`, `g3c.py`, `jobs/geo2.slurm` instead of the d1_g3
  files; the rest of the list is unchanged (T currency, F-10 / F-16).
- feat_geo.py:152, 319-336 — runtime decision keys carry the prefix `feat_geo/` in `results/DECISIONS.md` (other roles
  write there too; avoids key collisions with their `F-1`-style keys). Same fcntl lock rule (F-11).
- feat_geo.py:154-278 — `DECISIONS`: removed F-3, F-14, U-2f-1..6 (Part A / Phase O / U-1); texts of F-1, F-2, F-4, F-5,
  F-6, F-7, F-10, F-11, F-15, F-16 adapted to geo; new GEO-1 .. GEO-10 (below). F-7b, F-8, F-9, F-12, F-13 kept.
- feat_geo.py:299-303 — H plan files at `results/hf_geo/H_batches/` and checks at `results/hf_geo/M_checks/` (=
  phase_fh's `results/H_batches`, `results/M_checks` under the hf_geo results root that `D1HF_RES` already points to),
  base list `$GEO2/hf_geo/H_list.csv`.
- feat_geo.py:438-483 (GEO-1) — `geo_state` / `geo_base` replace `part_a`, `base_of`, `o_state`, `need_o`, `u1_check`,
  `in_scope`, `u1_skip`: base = every `lad_common.targets()` row with `$GEO2/geo_final/<rxn_id>/.done`, job_id order;
  needs `results/Z_report.json` with `stop: false` (missing = exit 2, stop = exit 3) and `Z_status.csv` ok/kept rows
  = `Z_report.n_folders` = the `.done` markers (else exit 3: folders changed after Z). G2-fallback folders are `.done`
  and therefore in the base (spec 0-2 / 4-2: deployable fallback).
- feat_geo.py:487-528 — `env_d1` / `env_reactot` check `GEO2_SCRATCH` / `GEO2ENV` (geo2.slurm) instead of
  `D1G3_SCRATCH` / `G3ENV`; otherwise identical.
- feat_geo.py:557-780 — `order_g`, `g_one`, `tree_bytes`, `run_pool` verbatim (only the budget STOP message names
  `GEO2_G_BUDGET_GB`).
- feat_geo.py:782-841 — `g_scan`: gate = spec 6-2 G (atom-order failures > 0.5 %, Freq + unexpected errors > 1 %,
  G0-5) over the scanned reactions, plus stale copies; D1-G3's combined > 5 % rule, `min_fail_to_stop` (U-2f-3) and
  `g0_thresholds_info` removed; n_imag 0 / >= 2 recorded, not failures (spec 6-2).
- feat_geo.py:848-882 (GEO-2) — `g <i|A> <n> [retry]` (= g0f0 g0 slicing `base[i::n]` in job_id order). A slice runs
  only its reactions through the unchanged `run_pool`, writes only its folders and `F_steps/g_<i>of<n>.json`, exits 3
  on stale copies (or the run_pool canary / abort / budget STOPs) and 2 while reactions are open. The spec 6-2 G rate
  gate is evaluated on the whole base by `g_report` (review fix R-2, GEO-11) and rechecked by f_slice / f_* / p /
  report, not per slice (a slice rate is not the spec rate).
  `retry` moves only this slice's `.fail_freq_*` / `.fail_error_*` / stale folders (explicit only, as feat.py).
- feat_geo.py:904-1030 (F-4, GEO-7) — `f_summary` gate = spec 6-2 F with D1LADDER F0-3 rates over the G `.done`
  reactions: F-1, B6, Q-a, T-a each > 1 % = STOP, plus the unchanged FU2 verdicts, mergeroot stop, counts, layout.
  ~~Judgment: F-1 is counted without the B6-only rows~~ — **superseded by review fix R-1 below** (F-1 = g0f0 rule,
  B6-only rows included). D1-G3's combined rate (base reactions without a token) is reported as information
  (`gate.combined_info`).
- feat_geo.py:1036-1061 — `f_prior`: the F precondition (env, base, T, G2 inputs, whole-base G gate, F-15 currency),
  computed once per process (f_rest runs five steps).
- feat_geo.py:1064-1088 — `f_root`: the F-15 move-aside + `root_fp.json` + `ids_all.csv` of feat.py `cmd_f`, now done by
  each f_slice task under the fcntl lock `$GEO2/locks/F_geo.lock`, so the array tasks never write those shared files
  concurrently (the first task after a root change moves the old outputs aside; the others find them current).
- feat_geo.py:1106-1256 (F-5) — F split as g0f0 F0: `f_slice <i> <n>` (runroot slice i of n), `f_merge`,
  `f_atoms <i> <n>`, `f_gate` (one retry pass while `.err.json` remain and no retry of the current root exited 0),
  `f_tokens`, `f_report`, `f_rest` (merge, atoms 0 1, gate, tokens; stops at the first non-zero; always f_report).
  Child commands and environments (`feat_env`, `atoms_env_m`) are feat.py's. The retry record carries the sha256 of
  `root_fp.json` (feat_geo.py:1171-1187) so a retry of an older root does not suppress the retry of a new one.
- feat_geo.py:1285-1287 (GEO-6) — **Judgment:** spec 6-2 H "실행 실패 > 1 %" = (H_status `fail` + `gate_fail` +
  `eda_table_unexplained` + `parse_fail`) / finished. D1-G2 counted only the gate statuses and reported `fail` (SCF /
  ORCA) separately; "실행 실패" covers both (D1-G2 H had 0 of each over 8,616 reactions).
- feat_geo.py:1328-1361 (GEO-5) — new rel-energy gate (spec 6-1): `e_frag1_rel_eh`, `e_frag2_rel_eh` of geo ok rows
  vs `analysis/d1_g2/results/labels_d1_hf_g2.parquet` ok rows, <= 1e-6 Eh per reaction; STOP if > 0.5 % of the
  comparable reactions differ. **Judgment:** comparable = geo ok and D1-G2 ok; a geo reaction without a D1-G2 ok row is
  reported (`n_not_in_reference`), not counted as a mismatch; no comparable reaction while geo has ok rows = STOP.
- feat_geo.py:1364-1430 — `h_summary`: thresholds of GEO-6 / input check 1 % / rel gate added; the job-id sets are
  built once (feat.py rebuilt `set(jobs)` per row: O(n^2) at 8,600 reactions); otherwise feat.py's checks (stale runs,
  labels json rows = ok rows, open / absent, core-h).
- feat_geo.py:1514-1588 (F-6) — `h_inputs <W>` = phase_fh h_inputs on the geo base: geominputs once, plan (nb = W x
  ceil(n / (1000 W)), (n_atoms, job_id) round robin, worker k owns j % W == k), plan.json written last. **Judgment:**
  the input-check rule (geominputs status != ok > 1 % of the base, D1-G2 rule) is applied here already (exit 3 before
  any plan / HF run) and again in h_summary.
- feat_geo.py:1571 (spec 6-1 pilot) — pilot = the first 50 inputs-ok base reactions by job_id (`H_pilot.csv`), listed in
  the plan and part of the batches (h_batch skips them as `.done`).
- feat_geo.py:1591-1670 (GEO-4) — `h_pilot`: run.py on the 50, `_hfparse` for statuses / energies, core-h from
  timing.tsv (serial ORCA; the same measure as D1-G2's 1,611 core-h). ~~Judgment: projection = max(spec rule,
  D1-G2-calibrated estimate)~~ — **superseded by review fix R-3 below** (STOP on the spec projection only; the
  calibrated estimate is information / warning). The rel-energy gate is applied to the pilot
  with the same 0.5 % rule (one mismatch in 50 = STOP: identical inputs, so a mismatch means a broken HF path).
  Pilot run failures are reported only; the 1 % rule is applied by h_batch / h_report. An unfinished pilot = exit 2.
  "Proceed automatically when <= 2,500" = `h_batch` requires a passing `results/H_pilot.json` of the current plan
  (feat_geo.py:1502-1510) and is submitted `afterok` on the pilot job.
- feat_geo.py:1673-1717 — `h_batch <k> <W>` = phase_fh h_batch (pre- and post-run cumulative check over this plan's
  M_checks, GEO-6 rule) plus the pilot gate.
- feat_geo.py:1720-1734 — `h_build` = `build.py allgeom <base list> g2` (feat.py's list form, not phase_fh's `-`), so
  H_status.csv holds the base reactions only.
- feat_geo.py:1737-1778 — `h_report` -> `results/H_report.json` (phase_fh h_report content on h_summary + checks +
  workers + pilot).
- feat_geo.py:1790-1795, 1893-1896 (GEO-8) — **Judgment:** `inputs.RUNGS['geo']` is still registered at run time (spec
  6-1); if LADA has fixed `RUNGS['geo']` in inputs.py (spec 8-1) the two entries are compared by `str` values (feat.py
  compared the dicts directly, which would fail on a Path / str spelling of the same path). Different = STOP.
- feat_geo.py:1870-1976 — `cmd_p`: `cmd_p` of feat.py; Part A / O / U-1 removed; the G gate is added to the G / F / H
  precondition loop (feat.py relied on f_summary's counts only); `--accept` passed to f_summary and recorded.
- feat_geo.py:388-393 (GEO-7) — **Judgment:** `p --accept=F-1[,B6]` / `report --accept=...` move only those rate
  STOPs to `accepted_stops` after an explicit user decision (LADDER `--accept-f0-stop`, D1-G2 `F_DECISION.md`
  precedents: both earlier rungs stopped at the F 1 % rule and the user accepted). Default = STOP. Without it an
  acceptance would need a code edit, which changes the CODE_REL sha256 and forces T-2 / T-3 to rerun.
- feat_geo.py:1999-2145 (GEO-9, spec 7) — new `p_report`. Universe = D1PAIR `results/S_folds.csv` (LADDER P8); P_r =
  both ends `usable` in the rung table (LADDER inputs.py `_pairs`). P_dft / P_g2 are recomputed from
  `$LAD/inputs/{dft,g2}.parquet` (sha256 = P_build / P_report) and must equal LADDER's `sets/P_dft.csv`, `P_g2.csv`
  (sha256 = P_report and identical pair_id sets; checked on the login node today: consistent). P_geo from
  `$GEO2/inputs/geo.parquet` (current passing P_build_geo.json). Outputs `results/P_geo_sets.json`,
  `results/P_geo_report.json` (counts by ref_kind / para / P_main reference / fold_T1, sha256 = sha256 of
  `"\n".join(sorted ids)` without trailing newline, P_g2-not-in-P_geo pairs with per-end reasons:
  `no_geo_folder:<Z status>`, `G:<fail reason>`, `no_tokens` / `no_x182` / `no_h`). **Judgment:** the STOP
  `|P_23| < 0.98 |P_g2|` is applied to the full sets and to the P_main reference subsets (the eval_geo2 universe;
  spec 0-4 counts P_g2 = 6,401 there); either failing = STOP. **Extra output:** `$GEO2/inputs/sets/P_{geo,23,023}.csv`
  in LADDER `SET_COLS` format, so a loader that expects `inputs/sets/P_<rung>.csv` can be pointed at `$GEO2` (LADA).
- feat_geo.py:2149-2470 — T-2 / T-3 (`t_check`, `fresh_tdir`, `cmd_t2_g`, `t2_body`, `cmd_t2_f`,
  `inputs_vs_templates`, `t3_body`, `cmd_t3`, `_hfparse`) verbatim except the spec reference text (`GEO2.md 5`) and
  the work folders `$GEO2/t2`, `$GEO2/t3` (through `C.G3`). `_hfparse` is also used by h_pilot.
- feat_geo.py:2471-2546 — `report`: Part A / O / U-1 blocks and the S_pairs.csv pair counts removed; reports Z, T, G,
  F, H, H pilot, H_report, P, the spec 7 pair report and core-h (timing + allocation by step group).
- feat_geo.py:2548-2575 — CLI: `g <i|A|@I> <n> [retry]`, `f_slice|f_atoms <i|A|@I> <n>`, `f_rest`, `f_merge`,
  `f_gate`, `f_tokens`, `f_report`, `h_inputs <W>`, `h_pilot`, `h_batch <k|A|@I> <W>`, `h_build`, `h_report`,
  `p [--accept=...]`, `p_report`, `report [--accept=...]`, `t2_g [--rerun]`, `t2_f`, `t3 [--rerun]`, `_hfparse`.

## Runbook judgment calls (RUNBOOK_feat.md)

- RUNBOOK_feat.md §4 — H workers use `--cpus-per-task=48 --mem=180G` (D1-G2's proven H setting, MaxRSS 13 GB at 48
  workers) instead of the spec 6-3 example `32 / 64G`; W = 5 (D1-G2: W 5, nb 10). Spec 6-3: "실제 job 파일 이름과
  인자는 복사한 코드에 맞춘다".
- RUNBOOK_feat.md §4 — `h_inputs` and `h_pilot` share one 48-core job (`:::`), saving a MaxSubmit slot; the pilot STOP
  (exit 3) kills the `afterok` worker array (`--kill-on-invalid-dep=yes`).
- RUNBOOK_feat.md §6 — P job runs `p ';;;' p_report ';;;' report` so that `F_report.json` is always rewritten.

## Review fixes (2026-10-30; line numbers after the fixes)

- **R-1 F-1 rule = g0f0 (review finding 1, major).** feat_geo.py:1121-1141 — `f1_fail = dict(feat_fail)`: every G
  `.done` reaction without a finite 182-feature row in `X182_d1_geo`, B6-only rows included, exactly as
  `analysis/d1_ladder/g0f0.py:1299-1301` (f0_report; the 2026-10-22 fixer change) and spec 6-2 F "F-1(182열 non-finite)
  > 1 %". The earlier default exclusion of B6-only rows applied, up front, the reading that LADDER used only through a
  user decision after its F0 STOP (`F0_report.json` stop true, `F0_DECISION.md`, `inputs.py build dft
  --accept-f0-stop`), so a geo F-1 rate > 1 % would have passed silently. The B6-only count is kept as information
  (`features.n_F1_without_B6_only`, `gate.F1_without_B6_only`, feat_geo.py:1129, 1231-1233; = g0f0 `n_without_B6_only`).
  The looser reading is applied only by `p --accept=F-1` / `report --accept=F-1` after an explicit user decision (GEO-7).
  Runtime texts `F-4 gate` (feat_geo.py:189-199) and RUNBOOK_feat.md §5 (f_rest row) updated.
- **R-2 B6 gated at G, G-vs-F agreement restored (review finding 2, minor).** New command `g_report [--accept=B6]`
  (feat_geo.py:908-1063, decision GEO-11 at feat_geo.py:292-301) = LADDER g0_report: after all G slices, a whole-base
  g_scan plus the child `_gcheck` (feat_geo.py:926-960 = g0f0 `_g0check` with the root `$GEO2/geo_feat`; the per-folder
  work is `g0f0._g0c_init` / `_g0c_one` unchanged = ts_fragments + `ext_features.block_b6`) -> `results/G_report.json`,
  `$GEO2/G_geo/{gate_ids.csv, gate.json}`. STOP = spec 6-2 G: atom-order failures (order gate + ts_fragments, as
  g0_report counts them) > 0.5 % of the base, Freq / error > 1 % of the base, B6 not computable > 1 % of the `.done`
  folders, stale copies. `f_prior` (feat_geo.py:1244-1278) now needs a current passing G_report (current = same base
  sha256, same root_fp sha256 over the G `result.json` files, gate.json unchanged; missing / old = exit 2, STOP = exit 3),
  so a B6 STOP surfaces before f_slice. `f_summary` (feat_geo.py:1154-1167) restores g0f0 f0_report's STOP "B6 status
  differs between G and F" on the rows whose B6 block ran at F. `cmd_p` (feat_geo.py:2107-2134) and `report`
  (feat_geo.py:2727-2758) include the G report (P precondition loop; F_G_status.csv gains `ts_fragments` / `b6_check`
  columns). **Judgment:** only the B6 rate STOP of g_report can be accepted (`--accept=B6`, GEO-7), mirroring the F
  acceptance; order / Freq / stale cannot. CLI feat_geo.py:2799-2805. RUNBOOK_feat.md §1, §5 (new JGR job between the G
  array and f_slice, 16 cpus / 32G as LADDER g0_report: 43 s for 9,266 folders), §7 (B2 = 10 tasks, total 18).
- **R-3 pilot STOP on the spec projection only (review finding 3, minor).** feat_geo.py:1864-1895 — `used = proj` =
  pilot mean core-h per reaction x inputs-ok reactions (spec 6-1); STOP if > 2,500 or not finite (no finished pilot
  reaction). The D1-G2-calibrated estimate stays in `H_pilot.json` `projection.d1g2_calibrated` as information; if it
  alone exceeds 2,500, `H_pilot.json` `warnings` and a runtime line `feat_geo/GEO-4 pilot warning (plan <sha12>)` in
  `results/DECISIONS.md` record it and H proceeds (spec line 11: decision criteria unchanged). Runtime text `GEO-4 pilot`
  and RUNBOOK_feat.md §4 (h_pilot row) updated.
- **Not FEAT-owned (review findings 4-6):** `analysis/d1_ladder/inputs.py` (RUNGS['geo'] edit makes the dft / g1 / g2
  P_build fingerprints stale through the inputs.py sha256), `train_atoms.py` / `train_xgb.py` (stage-3 gate reads the
  LADDER result snapshot in the training process; CODE_SHA change freezes stage 1 / 2 caches and `train_xgb.py check
  --stage 1|2` would overwrite `L_xgb_report.json`). Verified as described against the code; left to the LADA / LADB
  roles. FEAT code does not call `inputs._build_ok` / `read_table` / `read_set` / `p_report(rung)` (p_report checks
  the LADDER tables by `P_build_<r>.json` outputs sha256 and `P_report.json` only, feat_geo.py:2282-2289), so FEAT is
  not blocked by finding 4.
