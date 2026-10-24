#!/usr/bin/env python3
r"""g3_eval.py — D1-G3 Phase E (docs/specs/D1G3.md 6; decision rules 9; STOP 12 E) and Phase R (spec 10: the Korean
report results/SUMMARY_G3.md). sbatch only. Nothing in this file was run when it was written (login node): every path is
untested until it runs inside a job.

CLI (submit from analysis/d1_g3 through jobs/g3.slurm with G3ENV=reactot, decision EV-0; exit 0 = done, 2 = an input is
missing / its phase is not finished (resubmit after it; nothing written), 3 = STOP (spec 12 E pair lists, an input
consistency check, or an earlier phase that ended in a STOP))
  python g3_eval.py run [--workers N] [--no-figures]
        E-1 .. E-6 + the spec 9 decision -> results/E_geometry.csv, E_noise.csv, E_endtoend.csv, E_recovery.csv,
        E_cost.csv, B_energies.csv, results/figures/figG3_1_noise, figG3_2_endtoend, figG3_3_partB (.png, .pdf, .csv)
        and results/E_report.json (written last). N = rmsd.py worker processes (default $SLURM_CPUS_PER_TASK).
        Deterministic: a rerun recomputes and atomically rewrites every output.
  python g3_eval.py summary
        Phase R -> results/SUMMARY_G3.md (Korean; every number is read from E_report.json, the E_*.csv / B_energies.csv
        files and the phase reports; needs a current E_report.json: its outputs sha256 = the files, else exit 2).

sbatch (login node: one-shot commands only; first `squeue -u $USER -h -r | wc -l` (N + 1 <= 20) and
`sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`):
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_g3
  sbatch -p cpu1,cpu2 -J g3_E --time=48:00:00 --cpus-per-task=8 --mem=32G --export=ALL,G3ENV=reactot \
      --dependency=afterok:$JE3 --kill-on-invalid-dep=yes jobs/g3.slurm g3_eval.py run ::: g3_eval.py summary
  (JE3 = the apply.py e3 job, which itself waits for opt.py assemble -> feat.py p / report.)
  (log $G3/logs/g3_E_<jobid>.out; ' ::: ' = summary only if run exited 0.) Rerun = the same line.

Inputs (read only; missing -> exit 2)
  results/start_report.json (sample.py start; stop false; its 0-1 sha256 of E2_report.json / L_predictions.parquet and
      0-2 sha256 of C_compare.csv must equal the files now)
  results/S_pairs.csv  pair_id, a_rxn, b_rxn, fold, class (same | different), dipole, a_job, b_job, core, fam
  results/S_rxns.csv   rxn_id, job_id, dipole, core, ts_class, roles, part_a, part_b, partA_pairs, partA_classes, partA_ends
  results/S_report.json  outputs sha256 of the two CSVs; inputs sha256 of analysis/d1_pair/results/S_folds.csv
  results/O_status.csv (opt.py assemble) rxn_id, job_id, method (g3h | g3r | partB), status (ok | fail:<reason>), reason,
      ts_status (ok | fail | error), ts_reason, imag_freqs_cm, n_imag, shape_kept, rmsd_start_A, dform_start_A, core_h,
      core_h_refs, attempt_used, n_atoms, dipole, classes, phase, part (A | B), converged_to_label, dE_kcal, e_sp_eh,
      e_label_eh, sp_mode;  results/O_refs.csv mid, method, ..., status, core_h;  results/O_report.json (complete true,
      stop false -> else exit 2 / 3; outputs sha256 of O_status.csv / O_refs.csv);  results/O_estimate.json (reported)
  $G3/<m>_final/<rxn_id>/{.done | .fail_*, ts.xyz}   (marker must agree with O_status status)
  $G3/partB/<rxn_id>/result.json  (rmsd_label_A, dform_label_A, same_as_g2_start, rmsd_g2_A, core_h_optts, core_h_sp)
  results/F_report.json (feat.py report; complete true, stop false -> else exit 2 / 3)
  results/P_build_<m>.json (all_pass, outputs[$G3/inputs/<m>.parquet] = its sha256, usable = the usable rxn_ids) and
      $G3/inputs/<m>.parquet (D1LADDER make_table layout: rxn_id, usable, H_<t> for the 9 H targets)
  $G3/apply/predictions.parquet + apply_report.json (apply.py e3; the contract this file reads, decision EV-7): one row
      per (input_level, model_rung, arm, model, target, pair_id); names as apply.py writes them (OUT_COLS), with the
      accepted alternatives in [ ]:
        pair_id      str, a results/S_pairs.csv pair
        fold         int, the T1 fold of the unit that predicted it (must equal S_pairs fold: no leak)
        input_level  dft | g1 | g2 | g3h | g3r (the rung table the features / H came from; dft, g2, g3h, g3r required,
                     g1 used when present)                         [input, input_rung]
        model_rung   dft | g2 (the unit's training rung)           [train_rung; rung when an input column exists]
        arm          S | SH
        model        AW | GNN | TF (unit rows: yhat = 5-seed mean as in L_predictions), REP (rep_source = the unit
                     model; checked = the representative used here), LINE (checked = the LINE computed here);
                     XGB rows are ignored (no model files, spec T-4)
        target       the 10 TARGETS (c_ghost ignored here)
        yhat         kcal/mol (SH: base + model output, as L_predictions)
        optional     y (checked = F_pairs_cr2.csv dd_<t>, 1e-9), base (SH / LINE: checked = beta x ΔΔ_HF, 1e-6; S: 0),
                     rep_source [source]; other columns (class, beta, yhat_seed_sd, dipole, core) are not read
      apply_report.json: outputs.predictions_sha256 = the parquet, self_check ok; results/T_1.json .. T_5.json passed
      (T_4.json all_pass)
  D1LADDER (read only): results/P_report.json (inputs.p_report(): all_pass), $LAD/inputs/{dft,g1,g2}.parquet (sha256 =
      P_report rungs.<r>.table_sha256), $LAD/inputs/sets/P_012.csv (inputs.read_set), $LAD/F_pairs_cr2.csv
      (inputs.read_labels, sha256 = P_report labels), results/E2_report.json (representatives, line_betas, l_report),
      results/L_predictions.parquet (diagonal check)
  D1-G2 / D1PAIR / D1 (read only): analysis/d1_g2/results/C_compare.csv (cross-check), $G2/g2_final/<rxn>/ts.xyz,
      $D1PAIR/g1_final/<rxn>/ts.xyz, D1 job copies <job>/<ts_file>, labels_d1.json (formed_pairs_ts, ts_file) via
      g3_common.targets(); D1 validation analysis/rmsd.py + stats.py and config_val.yaml (the D1 runtime code copy).
  docs/specs/D1G3.md (the spec 9 / 10 sentences used verbatim are checked to be in it; else STOP)

Outputs (results/; decisions EV-0 .. EV-14 below (prefix EV so that they are not read as the spec metrics E-1 .. E-6),
also in results/DECISIONS.md (written when 'run' starts) and E_report.json)
  E_geometry.csv   scope, class, level, n_rxn, n_bonds, dform_abs_median, dform_abs_p90, dform_abs_mean,
                   dform_short_signed_median, dform_long_signed_median, dform_rxn_max_median, dform_rxn_max_p90,
                   rmsd_heavy_median, rmsd_heavy_p90, rmsd_heavy_mean_finite, n_rmsd_inf, n_same_ts, same_ts_frac,
                   same_ts_ci_lo, same_ts_ci_hi, n_pairs, ddd_abs_median, ddd_abs_p90, ddd_abs_mean,
                   ddd_short_abs_median, ddd_long_abs_median   (E-1; per-reaction values: E_report.json e1_detail)
  E_noise.csv      scope, class, level, channel, n_pairs, noise_median, noise_mean, noise_p90, signed_median,
                   dH_dft_abs_median   (E-2: |ΔΔ_HF(level) − ΔΔ_HF(dft)|, para pairs)
  E_endtoend.csv   scope, class, family, train_rung, arm, role, model (REP | REP_atom | LINE), source, target, input,
                   n_pairs, n_cores, mae, mae_ci_lo, mae_ci_hi, rmse, mean_err, note   (E-3)
  E_recovery.csv   scope, class, method, target, family, model, source, n_pairs, n_cores, mae_g2, mae_g3, mae_dft,
                   gap_g2_dft, recovery, recovery_ci_lo, recovery_ci_hi, n_boot_valid, note   (E-4)
  E_cost.csv       method, class, n_rxn, n_ts_ok, n_ts_fail, n_ts_error, ts_success, ts_success_ci_lo, ts_success_ci_hi,
                   n_shape_kept, shape_kept_rate, shape_kept_ci_lo, shape_kept_ci_hi, n_final_done, final_done_rate,
                   n_retry, ts_core_h_median, ts_core_h_mean, ts_core_h_sum, ts_core_h_per_success,
                   ref_core_h_rxn_median, ref_core_h_rxn_mean, ref_core_h_rxn_sum, n_ref_units, n_ref_units_ok,
                   ref_unit_core_h_median, ref_unit_core_h_mean, ref_unit_core_h_sum, label_ts_core_h,
                   ts_core_h_median_over_label   (E-5; method partB: Part B OptTS + SP cost)
  B_energies.csv   rxn_id, job_id, dipole, ts_class, part_a, status, reason, ts_status, ts_reason, resolved,
                   converged_to_label, dE_kcal, dE_eff_kcal, bin (lt_m05 | abs_le_05 | gt_05 | ''), sp_mode,
                   rmsd_label_A, dform_label_A, same_as_g2_start, rmsd_g2_A, n_imag, imag_freqs_cm, core_h,
                   core_h_optts, core_h_sp, e_sp_eh, e_label_eh   (E-6; distribution by dipole in E_report.json partB)
  E_report.json    provenance (inputs sha256, env, code), checks, pair lists, representatives used, e1_detail, partB,
                   spec9 (per method criteria, verdict, recommendation), decisions, outputs sha256, stop false
  figures/figG3_1_noise, figG3_2_endtoend, figG3_3_partB (.png, .pdf, .csv)
  SUMMARY_G3.md    (summary)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # nothing compiled into the read-only trees (D1 copy, d1_ladder, d1_pair, d1_g2)

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# D1 validation rule first (= c_compare.py order), so that 'fragmenter' resolves to the D1 copy with Si
VAL = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation")
FRAG_EXPECTED = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/analysis/espley_xtb_repro/fragmenter.py")
sys.path.insert(0, str(VAL / "analysis"))
import rmsd as rm  # noqa: E402          D1 validation 6.4: mapped_heavy_rmsd, same_ts (+ rm.fr = fragmenter.read_xyz)
import stats as vstats  # noqa: E402     D1 validation 6.5: wilson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g3_common as C  # noqa: E402     (puts analysis/d1_ladder and analysis/d1_pair on sys.path)

DP, L = C.DP, C.L
import evaluate as EV  # noqa: E402     D1LADDER: TARGETS, PRIMARY9, line_predict, N_BOOT, SEED, PRED, MODEL_ORDER, _clean
import inputs as IN  # noqa: E402       D1LADDER Phase P: p_report, read_set, read_labels, table_path, H_TARGETS
import d1_eval as E9  # noqa: E402      D1PAIR rev 9: boot_diff, mae, rmse
sys.path.append(str(C.REPO / "analysis" / "d1_g2"))
import g2_common as G2C  # noqa: E402   (D1_REPO: the D1 runtime code copy; constants only)

# ---------------------------------------------------------------- constants
CODE_VERSION = "d1-g3-e1"
SPEC = C.REPO / "docs" / "specs" / "D1G3.md"
START = C.RES / "start_report.json"
S_PAIRS, S_RXNS, S_REPORT = C.RES / "S_pairs.csv", C.RES / "S_rxns.csv", C.RES / "S_report.json"
O_STATUS, O_REFS, O_REPORT, O_ESTIMATE = (C.RES / "O_status.csv", C.RES / "O_refs.csv", C.RES / "O_report.json",
                                          C.RES / "O_estimate.json")
F_REPORT = C.RES / "F_report.json"
T_JSON = {k: C.RES / f"T_{k}.json" for k in (1, 2, 3, 4, 5)}
PRED = C.G3 / "apply" / "predictions.parquet"
APPLY_REPORT = C.G3 / "apply" / "apply_report.json"
FINAL = {m: C.G3 / f"{m}_final" for m in C.METHODS}
PARTB = C.G3 / "partB"
G2_FINAL = C.G2 / "g2_final"
G1_FINAL = DP.G1_ROOT_D1F
D1_JOBS = DP.D1_JOBS
C_COMPARE = C.G2_RES / "C_compare.csv"
E2_REPORT = C.LAD_RES / "E2_report.json"
CFG_VAL = VAL / "config_val.yaml"
OUT = dict(geometry=C.RES / "E_geometry.csv", noise=C.RES / "E_noise.csv", endtoend=C.RES / "E_endtoend.csv",
           recovery=C.RES / "E_recovery.csv", cost=C.RES / "E_cost.csv", partB=C.RES / "B_energies.csv")
E_REPORT = C.RES / "E_report.json"
SUMMARY = C.RES / "SUMMARY_G3.md"
FIGDIR = C.RES / "figures"
FIGS = ("figG3_1_noise", "figG3_2_endtoend", "figG3_3_partB")

LEVELS = ("dft", "g1", "g2") + tuple(C.METHODS)           # spec 6: dft (0), g1, g2 (D1LADDER inputs), g3h, g3r
LAD_RUNGS = ("dft", "g1", "g2")
CLASSES = ("same", "different")                            # spec 2 / 6 (mixed is not sampled)
SCOPE_LEVELS = {"common": LEVELS, **{f"own_{m}": LAD_RUNGS + (m,) for m in C.METHODS}}   # decision EV-1
MODEL_RUNGS = ("dft", "g2")
ARMS = ("S", "SH")
FAMILIES = (("dft", "SH"), ("dft", "S"), ("g2", "SH"), ("g2", "S"))      # spec 6 E-3
ROLE = {"dft": "main", "g2": "reference"}                                # 'dft·SH, dft·S(주), g2·SH, g2·S(참고)'
ATOM = ("AW", "GNN", "TF")
XGB = "XGB"
T9 = list(EV.PRIMARY9)                                     # d1 d2 elst pauli oi disp cpcm cds barrier (spec 6 E-2 list)
TIDX = {t: i for i, t in enumerate(EV.TARGETS)}
H_COLS = [f"H_{t}" for t in T9]
REC_TARGETS = ("oi", "cpcm", "barrier")                    # spec 6 E-4
FIG2_TARGETS = ("oi", "cpcm", "barrier")
FIG2_INPUTS = ("g2",) + tuple(C.METHODS) + ("dft",)        # spec 6 figure order 'g2·g3h·g3r·dft'
RMSD_A, DFORM_A = 0.10, 0.02                               # D1 validation 6.4 (config_val prereg.same_ts; checked)
LABEL_TS_CORE_H = 8.2                                      # spec 6 E-5 / 9 ③ (pilot mean core-h per label TS)
CRIT = dict(rec_oi=0.8, mae_oi=0.20, ts_success=0.9, shape_kept=0.9, core_h_median=2.0)   # spec 9 ① ② ③
PARTB_CUT, PARTB_FRAC = 0.5, 0.2                           # spec 6 E-6 / 9 Part B
N_BOOT, SEED = EV.N_BOOT, C.SEED_BOOT                      # spec 6 E-3: 2,000, 20261019
TOL_Y, TOL_DIAG, TOL_BASE, TOL_CC, TOL_BOOT = 1e-9, 1e-3, 1e-6, 1e-9, 1e-12
VERDICT_KO = {"go": "본격 진행 권고 (go)", "partial": "부분", "impossible": "불가"}
METHOD_KO = {"g3h": "g3h(HF-D3BJ/6-31G*, SMD water)", "g3r": "g3r(r2SCAN-3c, SMD water)"}
LEVEL_SLOT = {"dft": 0, "g1": 1, "g2": 2, "g3h": 3, "g3r": 4}     # fixed palette slots (decision EV-13)
LEVEL_MARK = {"dft": "o", "g1": "s", "g2": "^", "g3h": "D", "g3r": "v"}
LEVEL_LABEL = {"dft": "dft (label DFT TS)", "g1": "g1 (xTB, from DFT TS)", "g2": "g2 (xTB, from SMILES)",
               "g3h": "g3h (HF-D3BJ/6-31G*)", "g3r": "g3r (r2SCAN-3c)"}
# spec sentences used verbatim (checked against docs/specs/D1G3.md at run time; STOP if one is not there)
SPEC_TEXT = {
    "9-scope": "Part A는 방법마다 `same` 분류에서 판정한다.",
    "9-go": "**본격 진행 권고 (go):** 아래 셋을 모두 만족.",
    "9-1": "① E-4 oi 회복률 ≥ 0.8, 그리고 dft·SH 대표의 oi para MAE ≤ 0.20",
    "9-2": "② TS 성공률 ≥ 0.9, 형태 유지율 ≥ 0.9",
    "9-3": "③ TS 하나당 코어시간 중앙값 ≤ 2.0 (라벨 TS 단계 평균 8.2의 약 25 %)",
    "9-partial": "**부분:** ①만 만족 → 정확도는 되지만 비용이나 안정성이 문제다.",
    "9-impossible": "**불가:** ① 불만족.",
    "9-both": "두 방법이 모두 go이면 코어시간이 적은 쪽을 권고한다.",
    "9-B-high": 'ΔE‡ < −0.5인 반응 비율이 0.2 이상이면 "라벨 TS가 최저 TS가 아닌 경우가 많다. TS 형태 탐색을 라벨과 '
                '입력 모두에서 강화해야 한다"고 쓴다.',
    "9-B-low": '0.2 미만이면 "G2가 더 높은 TS를 고르는 것이 주원인이다. 입력 쪽 TS 형태 탐색을 강화해야 한다"고 쓴다.',
    "6-label": "라벨 TS 단계(파일럿 평균 8.2 코어시간/반응)",
    "10-first3": "맨 앞 세 문장: 두 방법의 판정(go / 부분 / 불가)과 근거 수치, Part B 비율, 권고.",
    "10-lim1": "표본이 작아 CI가 넓다.",
    "10-lim2": "학습한 모델에 다른 구조 수준의 입력을 넣었다(분포 이동). 이것이 시험의 목적이다.",
    "10-lim3": "Part B는 G2 형태 TS 하나를 다듬은 것이라 전체 형태 탐색은 아니다.",
    "10-lim4": "기준 구조는 G2에서 출발했다.",
    "12-E": "같은 표에서 비교하는 입력 사이 짝 목록 불일치",
}
B_SENT_HIGH = "라벨 TS가 최저 TS가 아닌 경우가 많다. TS 형태 탐색을 라벨과 입력 모두에서 강화해야 한다"
B_SENT_LOW = "G2가 더 높은 TS를 고르는 것이 주원인이다. 입력 쪽 TS 형태 탐색을 강화해야 한다"
assert f'"{B_SENT_HIGH}"' in SPEC_TEXT["9-B-high"] and f'"{B_SENT_LOW}"' in SPEC_TEXT["9-B-low"]

DECISIONS = {
    "EV-0": "환경 = conda reactot (jobs/g3.slurm G3ENV=reactot): D1 검증 rmsd.py(networkx)와 parquet(pyarrow), scipy "
           "(stats.wilson), matplotlib이 모두 있다. env_espley에는 networkx가 없고 D1 venv에는 pyarrow가 없다. "
           "d1_eval.boot_diff는 numpy만 쓰므로 E1(env_espley)과 같은 난수열·같은 값이다.",
    "EV-1": "분류 = S_pairs class(짝), 반응 분류 = S_rxns ts_class(모든 Part A 반응에서 짝 분류와 같음을 확인). 범위(scope): "
           "'common' = 다섯 입력(dft, g1, g2, g3h, g3r)이 모두 있는 짝 = 나란히 놓는 표와 그림의 목록(명령서 6 '같은 짝에 "
           "대해 나란히'); 'own_<m>' = dft, g1, g2, <m>이 있는 짝 = 방법 <m>의 명령서 9 판정 목록('방법마다 판정': 한 "
           "방법의 실패가 다른 방법의 판정 표본을 바꾸지 않게, EV-11). 각 범위 안에서 모든 입력은 같은 짝 목록이고, 아니면 "
           "STOP(12 E). E-1의 '있음' = 그 수준의 TS 구조 파일(.done 폴더; "
           "g3는 $G3/<m>_final/<rxn>/.done의 ts.xyz, $G3/O/units의 실패 구조는 쓰지 않음); E-2–E-4의 '있음' = rung 입력표 "
           "usable(dft·g1·g2 = P_012라 표본 전부, g3 = P_build_<m> usable).",
    "EV-2": "형성 결합 식별 = 라벨 DFT TS에서 짧은 쪽 / 긴 쪽 형성 결합(C_compare dform_dft_short/_long 선례; 원자 번호 "
           "순서는 짝의 두 반응 사이에서 같은 결합을 가리킨다는 보장이 없다). 반응 단위 |Δd| = 두 결합을 모아(2n개) 중앙값·90 "
           "%·평균, 반응별 max도; 짝 단위 Δdd_k = (d_a,k − d_a,k,DFT) − (d_b,k − d_b,k,DFT), k = short, long, |Δdd|를 두 결합 "
           "모아 중앙값·90 %·평균.",
    "EV-3": "무거운 원자 RMSD = D1 검증 rmsd.mapped_heavy_rmsd(입력 TS, 라벨 TS); 같은 TS = rmsd.same_ts(입력, 라벨, 형성 "
           "거리, 0.10 Å, 0.02 Å)(인자 순서 = c_compare.py: a = 비교 구조, b = 라벨 TS). RMSD 중앙값·90 %는 inf(동형 없음)를 "
           "넣은 채 inverted_cdf, 평균은 유한값만, n_inf 따로(C-8 선례). g1·g2 값은 C_compare.csv(same_ts_g1_dft / "
           "rmsd_heavy_g1_dft, same_ts_dft / rmsd_heavy_dft)와 비교해 다르면 STOP. 비율의 95 % CI = D1 검증 stats.wilson.",
    "EV-4": "HF 값 = rung 입력표의 H_<t>(h = cr, 기준 재조립값 = SH 입력과 LINE이 쓰는 값; 6개 EDA 채널은 H = H0); ΔΔ_HF = "
           "H(a) − H(b). dft·g1·g2 표는 경로로 읽고 sha256 = P_report rungs.<r>.table_sha256, g3 표는 sha256 = "
           "P_build_<m>.json outputs. inputs.p_report(rung) / read_table / pairs_frame은 이 작업 트리에서 쓸 수 없다(지문 "
           "키가 d1ladder 작업 트리 절대 경로; 기존 코드 문제로 보고).",
    "EV-5": "E-3 모델 = E2_report representatives(rung·arm·타깃) 그대로. 대표가 XGB인 칸은 모델 파일이 없어(T-4) 빼고 n 0 "
           "행으로 남긴다; 설명용 REP_atom = 같은 칸에서 E2 mean_val_kcal_mae가 가장 낮은 원자 모델(AW, GNN, TF; 동률 순서 "
           "AW → GNN → TF, D1LADDER 13-1), apply.py가 그 행을 냈을 때만. 역할: dft·SH, dft·S = 주, g2·SH, g2·S = 참고. 9 타깃 "
           "= PRIMARY9(c_ghost는 HF 짝이 없어 명령서 E-2 목록에 없음, 보고하지 않음). LINE = E2_report line_betas[rung][fold] "
           "(= 그 fold SH unit의 β, E1-5에서 확인됨) × ΔΔ_HF(입력), arm 무관 1행.",
    "EV-6": "para MAE CI = d1_eval.boot_diff(e, 0, core, 2,000, seed 20261019)(D1LADDER E1 para CI와 같은 호출). 표본 짝은 "
           "모두 S_folds P_main·reference·pos p(확인). y = $LAD/F_pairs_cr2.csv dd_<t>(sha256 = P_report labels).",
    "EV-7": "apply.py 계약(이 파일 머리말의 열; apply.py OUT_COLS의 input_level·model_rung·rep_source를 그대로 읽는다): "
           "E-3의 REP는 apply.py unit 행(AW, GNN, TF)에서 E2 대표 모델을 고르고, LINE은 여기서 계산한다(apply.py REP·LINE "
           "행은 대조만). 검사: fold = S_pairs fold(누출 없음), y = 라벨(1e-9), 대각선(input = model rung) yhat = "
           "L_predictions yhat(≤ 1e-3 = T-4 허용 오차), SH base·LINE yhat = β × ΔΔ_HF(≤ 1e-6; 입력표에서 usable이 아닌 "
           "짝의 apply 행이 있으면 두 파일의 usable 판단이 다른 것이라 STOP), S base = 0, REP 행의 "
           "rep_source = 여기서 쓰는 모델, apply_report.json outputs sha256 = parquet, T_1–T_5 통과. 어긋나면 STOP. 필요한 "
           "입력(dft, g2, g3h, g3r)이 통째로 없으면 exit 2(apply 미완), g1은 있으면 쓴다. 짝 목록 안의 짝이 빠지면 STOP(12 E).",
    "EV-8": "회복률 = (MAE_g2 − MAE_g3) / (MAE_g2 − MAE_dft), dft·SH 대표(E2: oi GNN, cpcm GNN, barrier AW), 같은 범위·분류의 "
           "같은 짝. 분모 ≤ 0 또는 NaN이면 NaN(판정 ① 불만족). CI(설명용) = boot_diff와 같은 core 재표집(같은 난수열; 매번 "
           "g3 MAE CI가 boot_diff 값과 같은지 확인)에서 분모 > 0인 재표집의 비율 백분위.",
    "EV-9": "E-5 TS 성공률 = ts_status ok / (ok + fail), 분류의 Part A TS 단위; 파이프라인 오류(error)는 분모에서 빼고 따로 "
           "센다(명령서 12 O가 따로 다룸). 형태 유지율 = shape_kept True / ok TS. TS 하나당 코어시간 = TS 단위 core_h(1·2차 "
           "시도 합), ok + fail 단위의 중앙값·평균; 성공 TS당 = 끝난 모든 TS 단위(ok, fail, error) core_h 합 / ok. 기준 구조 코어시간: 반응당 = O_status core_h_refs"
           "(공유 분자 단위를 구성원 수로 나눔), 분자 단위당 = O_refs.csv(분류 'all'). 라벨 TS 단계 = 8.2 코어시간/반응(명령서 "
           "6 E-5). 비율 CI = stats.wilson.",
    "EV-10": "Part B: 결과 있음(resolved) = Part B 단위 ts_status ok(gate 통과 후 converged_to_label이거나 SP 성공). "
            "converged_to_label 반응은 ΔE‡ = 0(같은 TS)으로 |ΔE‡| ≤ 0.5에 넣는다. 비율의 분모 = 결과 있는 반응(명령서 9 Part B "
            "규칙의 비율도 이것); SP를 계산한 반응만의 비율도 함께. 실패·오류는 목록과 수로 보고.",
    "EV-11": "명령서 9 판정: 방법 <m>마다 'own_<m>' 범위(dft, g1, g2, <m> 모두 usable인 짝)의 same 분류('방법마다 same "
             "분류에서 판정'; 'common'은 다른 방법의 실패까지 걸러 판정이 다른 방법에 좌우되므로 나란히 놓는 표·그림에만 쓴다). "
             "① = 회복률_oi ≥ 0.8 그리고 MAE(dft·SH 대표, 입력 g3) oi ≤ 0.20; ② = "
            "same Part A 반응의 TS 성공률 ≥ 0.9, 형태 유지율 ≥ 0.9; ③ = same TS 코어시간 중앙값 ≤ 2.0. 값이 NaN이면 불만족. "
            "go = ①②③, 부분 = ①이고 ②·③ 중 하나라도 불만족('①만 만족'을 ①은 되고 나머지가 안 되는 경우로 읽음), 불가 = ① "
            "불만족. 둘 다 go면 ③ 중앙값이 적은 쪽(같으면 평균, 그다음 METHODS 순서).",
    "EV-12": "판정은 결과를 보기 전에 고정된 9절 문구를 실행 때 명령서에서 확인한다(없으면 STOP); Part B 해석 문장은 명령서 9절 "
            "따옴표 안 문장 그대로(+ 마침표).",
    "EV-13": "그림 = d1_figures.style()(rev 9 / LADDER 형식, recessive grid) + 검증된 기준 범주 팔레트(d1_figures.SLOT)의 "
            "고정 칸(입력별: dft 1, g1 2, g2 3, g3h 4, g3r 5; 그림 사이에서 같은 입력 = 같은 색) + 표지 모양 보조 부호화, 패널마다 "
            "y축 하나. 글자는 영어(CJK 글꼴 가정 없음). figG3_1은 로그 y(채널 크기 차이), figG3_3은 ±20 kcal/mol 밖 값을 끝 "
            "칸에 넣고 표시. 팔레트 검증 스크립트(node)는 로그인 노드 규칙상 돌리지 않았다(같은 8색이 기준 팔레트로 검증됨).",
    "EV-14": "SUMMARY_G3.md는 E_report.json, E_*.csv, B_energies.csv, 단계 보고 파일(start, S, T_1–T_5, O, O_estimate, F)의 "
             "값만 옮긴다(반올림 표시; 요청 표본 수 = S_report requested, 라벨 TS 코어시간 = E_cost.csv label_ts_core_h); "
             "T_4.json(apply.py)은 pass / verdict만 읽는다. start_report 0-3의 strict_reading(명령서 0-3 엄격 읽기는 STOP "
             "조건, 결정 0-3으로 진행)과 T_2 deviation, O_report deviations를 이탈로 적는다. E_report.json outputs sha256이 "
             "지금 파일과 다르면 exit 2(run 다시).",
}


class NotReady(Exception):
    pass


# ---------------------------------------------------------------- small helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def not_ready(msg):
    raise NotReady(msg)


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fin(x):
    return _f(x) is not None


def _tri(v):
    """True / False / None from a bool, a 'True' / 'False' string or a missing value."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes"):
        return True
    if s in ("false", "0", "0.0", "no"):
        return False
    return None


def _q(x, p, method="linear"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.quantile(x, p, method=method)) if len(x) else float("nan")


def _mean_fin(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def _med_fin(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def wilson(k, n):
    lo, hi = vstats.wilson(int(k), int(n))
    return lo, hi


def frac(k, n):
    return float(k) / float(n) if n else float("nan")


def need(*paths):
    miss = [str(p) for p in paths if not Path(p).is_file()]
    if miss:
        not_ready(f"input files missing: {miss}")


def git_head():
    r = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def env_rec():
    return dict(host=socket.gethostname(), python=platform.python_version(), executable=sys.executable,
                numpy=np.__version__, pandas=pd.__version__, slurm_job=os.environ.get("SLURM_JOB_ID"),
                cpus=os.environ.get("SLURM_CPUS_PER_TASK"), g3env=os.environ.get("G3ENV"),
                conda_env=os.environ.get("CONDA_DEFAULT_ENV"), g3_scratch=str(C.G3), git_head=git_head())


def record_decision(key, text):
    """g3_common.record_decision under an fcntl lock ($G3/locks/DECISIONS.lock, shared by every d1_g3 script):
    concurrent jobs never lose each other's DECISIONS.md lines (g3_common.py itself is not edited)."""
    import fcntl
    lk = C.assert_path(C.G3 / "locks" / "DECISIONS.lock")
    lk.parent.mkdir(parents=True, exist_ok=True)
    with open(lk, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            C.record_decision(key, text)
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def write_csv(path, df):
    return C.write_atomic(path, lambda t: df.to_csv(t, index=False))


def write_json(path, obj):
    return C.write_json(path, EV._clean(obj))


def _marker(d):
    """'.done' | '.fail_<reason>' | None (no folder / no marker)."""
    try:
        names = os.listdir(d)
    except OSError:
        return None
    if ".done" in names:
        return ".done"
    f = sorted(n for n in names if n.startswith(".fail_"))
    return f[0] if f else None


def _pairs_of(d):
    """label formed_pairs_ts ('3-7 5-6', 0-based label atom order) -> [(3, 7), (5, 6)] (c_compare.label_pairs)."""
    return [tuple(int(a) for a in p.split("-")) for p in str(d["formed_pairs_ts"]).split()]


def _dist(x, p):
    X = np.asarray(x[1], float)
    return float(np.linalg.norm(X[p[0]] - X[p[1]]))


# ---------------------------------------------------------------- input loading and checks
def check_modules():
    rec, bad = {}, []
    mods = dict(evaluate=(EV, C.LAD_DIR / "evaluate.py"), inputs=(IN, C.LAD_DIR / "inputs.py"),
                d1_eval=(E9, C.P_DIR / "d1_eval.py"), rmsd=(rm, VAL / "analysis" / "rmsd.py"),
                stats=(vstats, VAL / "analysis" / "stats.py"), fragmenter=(rm.fr, FRAG_EXPECTED))
    for k, (m, want) in mods.items():
        got = Path(m.__file__).resolve()
        rec[k] = dict(file=str(got), sha256=C.sha256(got))
        if got != Path(want).resolve():
            bad.append(f"{k} imported from {got}, not {want}")
    if VAL.resolve() != (G2C.D1_REPO / "D1_build" / "validation").resolve():
        bad.append(f"VAL {VAL} != g2_common.D1_REPO/D1_build/validation")
    if rm.fr.COV.get("Si") != 1.11:
        bad.append("D1 fragmenter without Si 1.11 A")
    import yaml
    thr = yaml.safe_load(CFG_VAL.read_text())["prereg"]["same_ts"]
    sig = inspect.signature(rm.same_ts).parameters
    rec["same_ts"] = dict(config_val=thr, defaults=dict(rmsd_A=sig["rmsd_A"].default, dform_A=sig["dform_A"].default),
                          used=dict(rmsd_A=RMSD_A, dform_A=DFORM_A), config_val_sha256=C.sha256(CFG_VAL))
    if not (float(thr["rmsd_A"]) == sig["rmsd_A"].default == RMSD_A and
            float(thr["dform_A"]) == sig["dform_A"].default == DFORM_A):
        bad.append(f"same-TS thresholds differ: {rec['same_ts']}")
    if T9 != list(IN.H_TARGETS) or len(T9) != 9:
        bad.append(f"PRIMARY9 {T9} != inputs.H_TARGETS {IN.H_TARGETS}")
    if N_BOOT != 2000 or SEED != 20261019 or EV.SEED != SEED or DP.SEED != SEED or DP.N_BOOT != N_BOOT:
        bad.append(f"bootstrap constants {N_BOOT} / {SEED} / {EV.SEED} / {DP.SEED} / {DP.N_BOOT}")
    if bad:
        C.stop("modules / rule: " + "; ".join(bad))
    return rec


def check_spec():
    txt = SPEC.read_text(encoding="utf-8")
    miss = [k for k, s in SPEC_TEXT.items() if s not in txt]
    if miss:
        C.stop(f"spec sentences not found verbatim in {SPEC}: {miss} (the spec 9 / 10 rules this file applies changed)")
    return dict(path=str(SPEC), sha256=C.sha256(SPEC), n_checked=len(SPEC_TEXT))


def load_start():
    need(START)
    st = C.read_json(START)
    if st.get("stop") or st.get("fail"):
        C.stop(f"{START}: start conditions failed ({(st.get('fail') or [])[:3]})")
    files = ((st.get("0-1") or {}).get("files") or {})
    want = {E2_REPORT: (files.get("E2_report.json") or {}).get("sha256"),
            EV.PRED: (files.get("L_predictions.parquet") or {}).get("sha256"),
            C_COMPARE: ((st.get("0-2") or {}).get("c_compare") or {}).get("sha256")}
    bad = [f"{p}: sha256 != start_report" for p, s in want.items() if not s or not Path(p).is_file() or C.sha256(p) != s]
    if bad:
        C.stop("D1LADDER / D1-G2 inputs changed since sample.py start: " + "; ".join(bad))
    xgb = ((st.get("0-1") or {}).get("representatives") or {}).get("xgb")
    return dict(sha256=C.sha256(START), xgb_representatives=xgb), {str(p): s for p, s in want.items()}


def load_s():
    need(S_PAIRS, S_RXNS, S_REPORT)
    rep = C.read_json(S_REPORT)
    for p in (S_PAIRS, S_RXNS):
        if (rep.get("outputs") or {}).get(str(p)) != C.sha256(p):
            C.stop(f"{p} sha256 != S_report.json outputs")
    SP = pd.read_csv(S_PAIRS, dtype={"pair_id": str, "a_job": str, "b_job": str, "core": str, "dipole": str,
                                     "class": str, "fam": str})
    for c in ("pair_id", "a_rxn", "b_rxn", "fold", "class", "dipole", "core"):
        if c not in SP.columns:
            C.stop(f"{S_PAIRS}: column {c} missing")
    SP["a_rxn"], SP["b_rxn"], SP["fold"] = SP.a_rxn.astype(int), SP.b_rxn.astype(int), SP.fold.astype(int)
    if SP.pair_id.duplicated().any() or not set(SP["class"]) <= set(CLASSES):
        C.stop(f"{S_PAIRS}: duplicated pair_id or classes {sorted(set(SP['class']))} outside {CLASSES}")
    SP = SP.sort_values("pair_id").reset_index(drop=True)
    SR = pd.read_csv(S_RXNS, dtype=str, keep_default_na=False)
    for c in ("rxn_id", "job_id", "dipole", "ts_class", "part_a", "part_b"):
        if c not in SR.columns:
            C.stop(f"{S_RXNS}: column {c} missing")
    SR["rxn_id"] = SR.rxn_id.astype(int)
    SR["part_a"], SR["part_b"] = SR.part_a.map(_tri), SR.part_b.map(_tri)
    SR = SR.set_index("rxn_id")
    A = sorted(set(SP.a_rxn) | set(SP.b_rxn))
    if sorted(SR.index[SR.part_a == True]) != A:                                        # noqa: E712
        C.stop("S_rxns.csv part_a != S_pairs a_rxn | b_rxn")
    cls_of = {}
    for a_, b_, k in zip(SP.a_rxn, SP.b_rxn, SP["class"]):
        for r in (int(a_), int(b_)):
            if cls_of.setdefault(r, k) != k:
                C.stop(f"reaction {r} is in a '{cls_of[r]}' and a '{k}' pair")
    bad = [r for r in A if SR.loc[r, "ts_class"] != cls_of[r]]
    if bad:
        C.stop(f"{len(bad)} Part A reactions whose S_rxns ts_class != their pair class (e.g. {bad[:3]})")
    # the sample pairs are S_folds P_main reference para pairs with the same ends / core / fold (spec 2)
    sfs = (rep.get("inputs") or {}).get(str(IN.S_FOLDS))
    if sfs != C.sha256(IN.S_FOLDS):
        C.stop(f"{IN.S_FOLDS} sha256 != S_report.json inputs")
    S = pd.read_csv(IN.S_FOLDS, dtype={"pair_id": str, "core": str}).set_index("pair_id")
    X = S.loc[SP.pair_id.to_numpy()]
    ok = ((X.pos.astype(str).to_numpy() == "p") & (X.ref_kind.astype(str).to_numpy() == "reference")
          & X.in_P_main.map(_tri).to_numpy(dtype=bool) & (X.a_rxn.to_numpy() == SP.a_rxn.to_numpy())
          & (X.b_rxn.to_numpy() == SP.b_rxn.to_numpy()) & (X.core.astype(str).to_numpy() == SP.core.to_numpy())
          & (X.fold_T1.astype(int).to_numpy() == SP.fold.to_numpy()))
    if not ok.all():
        C.stop(f"{int((~ok).sum())} S_pairs rows are not S_folds P_main reference para pairs with the same ends / core "
               f"/ fold (e.g. {SP.pair_id[~ok].tolist()[:3]})")
    B = sorted(SR.index[SR.part_b == True])                                             # noqa: E712
    info = dict(n_pairs=int(len(SP)), pairs_by_class={k: int((SP["class"] == k).sum()) for k in CLASSES},
                n_part_a=len(A), n_part_b=len(B), s_report_sha256=C.sha256(S_REPORT))
    return SP, SR, A, B, cls_of, info, {str(p): C.sha256(p) for p in (S_PAIRS, S_RXNS, S_REPORT, IN.S_FOLDS)}


def load_o(A, B):
    need(O_STATUS, O_REFS, O_REPORT)
    rep = C.read_json(O_REPORT)
    if rep.get("stop"):
        C.stop(f"Phase O ended in a STOP ({O_REPORT}: {rep.get('stop_reasons')})")
    if not rep.get("complete"):
        not_ready(f"{O_REPORT}: Phase O not complete ({rep.get('n_incomplete')} open; rerun opt.py assemble)")
    for p in (O_STATUS, O_REFS):
        if (rep.get("outputs") or {}).get(str(p)) != C.sha256(p):
            C.stop(f"{p} sha256 != O_report.json outputs")
    OS = pd.read_csv(O_STATUS, dtype={"job_id": str, "method": str, "status": str, "reason": str, "ts_status": str,
                                      "ts_reason": str, "dipole": str, "classes": str, "phase": str, "part": str,
                                      "sp_mode": str, "imag_freqs_cm": str})
    OS["rxn_id"] = OS.rxn_id.astype(int)
    OR = pd.read_csv(O_REFS, dtype={"mid": str, "method": str, "status": str, "reason": str, "key": str,
                                    "unit_key": str})
    bad = []
    for m in C.METHODS:
        q = OS[(OS.method == m) & (OS.part == "A")]
        if q.rxn_id.duplicated().any() or sorted(q.rxn_id) != A:
            bad.append(f"{m}: O_status Part A rows != the {len(A)} Part A reactions")
    qb = OS[OS.method == "partB"]
    if qb.rxn_id.duplicated().any() or sorted(qb.rxn_id) != B:
        bad.append(f"partB: O_status rows != the {len(B)} Part B reactions")
    if bad:
        C.stop("O_status.csv: " + "; ".join(bad))
    est = C.read_json(O_ESTIMATE) if O_ESTIMATE.is_file() else None
    files = {str(p): C.sha256(p) for p in (O_STATUS, O_REFS, O_REPORT) + ((O_ESTIMATE,) if est else ())}
    return OS, OR, rep, est, files


def load_f():
    need(F_REPORT)
    rep = C.read_json(F_REPORT)
    if rep.get("stop"):
        C.stop(f"Phase F ended in a STOP ({F_REPORT}: {rep.get('stop_reasons')})")
    if not rep.get("complete"):
        not_ready(f"{F_REPORT}: Phase F not complete ({(rep.get('incomplete') or [])[:5]})")
    T, usable, files = {}, {}, {str(F_REPORT): C.sha256(F_REPORT)}
    for m in C.METHODS:
        pb, tab = C.RES / f"P_build_{m}.json", C.G3 / "inputs" / f"{m}.parquet"
        need(pb, tab)
        b = C.read_json(pb)
        sha = C.sha256(tab)
        if not b.get("all_pass") or (b.get("outputs") or {}).get(str(tab)) != sha:
            C.stop(f"{pb}: all_pass {b.get('all_pass')} or {tab} sha256 != its outputs")
        X = pd.read_parquet(tab, columns=["rxn_id", "usable"] + H_COLS)
        X["rxn_id"] = X.rxn_id.astype(int)
        X = X.set_index("rxn_id")
        u = set(int(r) for r in X.index[X.usable.astype(bool).to_numpy()])
        if u != set(int(r) for r in (b.get("usable") or [])):
            C.stop(f"{tab}: usable rows != {pb.name} usable")
        T[m], usable[m] = X, u
        files.update({str(pb): C.sha256(pb), str(tab): sha})
    return T, usable, rep, files


def load_ladder():
    rep_p = IN.p_report()                                    # all_pass, not running (exit 2 / 3 inside otherwise)
    T, usable, files = {}, {}, {str(IN.P_REPORT): C.sha256(IN.P_REPORT)}
    for r in LAD_RUNGS:
        p = IN.table_path(r)
        need(p)
        want = ((rep_p.get("rungs") or {}).get(r) or {}).get("table_sha256")
        sha = C.sha256(p)
        if sha != want:
            C.stop(f"{p} sha256 {sha[:16]} != P_report rungs.{r}.table_sha256")
        X = pd.read_parquet(p, columns=["rxn_id", "usable"] + H_COLS)
        X["rxn_id"] = X.rxn_id.astype(int)
        X = X.set_index("rxn_id")
        T[r] = X
        usable[r] = set(int(x) for x in X.index[X.usable.astype(bool).to_numpy()])
        files[str(p)] = sha
    p012 = set(IN.read_set("P_012", rep_p).pair_id.astype(str))
    Y, ysha = IN.read_labels(IN.DEFAULT_LABELS)
    lab = (rep_p.get("labels") or {}).get(str(IN.DEFAULT_LABELS)) or {}
    if not lab.get("ids_ok") or lab.get("sha256") != ysha:
        C.stop(f"{IN.DEFAULT_LABELS}: sha256 / ids differ from P_report labels")
    Y = Y.set_index("pair_id")
    E2 = C.read_json(E2_REPORT)
    if (E2.get("l_report") or {}).get("predictions_sha256") != C.sha256(EV.PRED):
        C.stop(f"{EV.PRED} sha256 != E2_report.json l_report.predictions_sha256")
    files.update({str(IN.DEFAULT_LABELS): ysha, str(E2_REPORT): C.sha256(E2_REPORT), str(EV.PRED): C.sha256(EV.PRED)})
    return T, usable, p012, Y, E2, files


def rep_choices(E2):
    """{(rung, arm, target): rep (E2 model), rep_atom (XGB cells: best atom model by the E2 validation means)}."""
    out, bad = {}, []
    for r, a in FAMILIES:
        for t in T9:
            rec = (((E2.get("representatives") or {}).get(r) or {}).get(a) or {}).get(t) or {}
            m, means = rec.get("model"), rec.get("mean_val_kcal_mae") or {}
            if m not in ATOM + (XGB,):
                bad.append(f"{r}|{a}|{t}: {m}")
                continue
            alt = sorted((x for x in ATOM if _fin(means.get(x))), key=lambda x: (float(means[x]), EV.MODEL_ORDER[x]))
            out[(r, a, t)] = dict(rep=m, rep_atom=(alt[0] if alt else None) if m == XGB else None,
                                  means={k: _f(v) for k, v in means.items()})
    if bad:
        C.stop(f"E2_report representatives missing / unknown: {bad[:6]}")
    return out


def line_betas(E2):
    """{rung: {fold: beta [len(TARGETS)] in TARGETS order}} from E2_report line_betas (= the SH units' beta, E1-5)."""
    out = {}
    for r in MODEL_RUNGS:
        out[r] = {}
        for k in range(EV.NF):
            b = (((E2.get("line_betas") or {}).get(r) or {}).get(str(k))) or {}
            v = [_f(b.get(t)) for t in EV.TARGETS]
            if any(x is None for x in v):
                C.stop(f"E2_report line_betas[{r}][{k}] incomplete / not finite")
            out[r][k] = np.asarray(v, float)
    return out


# ---------------------------------------------------------------- E-1 geometry
def geom_one(t):
    """one Part A reaction: forming distances, DFT-short / long differences, mapped heavy RMSD and same-TS vs the label
    DFT TS for every level with a structure (errors recorded; the caller STOPs on any)."""
    out = dict(rxn_id=t["rxn_id"], levels={}, errors=[])
    try:
        dft = rm.fr.read_xyz(t["dft"])
        pairs = [tuple(p) for p in t["pairs"]]
        dd = [_dist(dft, p) for p in pairs]
    except Exception as e:                                       # noqa: BLE001
        out["errors"].append(f"dft: {type(e).__name__}: {e}")
        return out
    ks = 0 if dd[0] <= dd[1] else 1
    kl = 1 - ks
    out.update(d_dft=dd, short_pair=ks, n_atoms=len(dft[0]))
    out["levels"]["dft"] = dict(d=dd, dform_short=0.0, dform_long=0.0, rmsd=0.0, form_diff_max=0.0, same_ts=True)
    for lv, p in t["paths"].items():
        try:
            g = rm.fr.read_xyz(p)
            if list(g[0]) != list(dft[0]):
                raise ValueError("element order differs from the label TS")
            d = [_dist(g, q) for q in pairs]
            s, r, x = rm.same_ts(g, dft, d, dd, RMSD_A, DFORM_A)
            out["levels"][lv] = dict(d=d, dform_short=d[ks] - dd[ks], dform_long=d[kl] - dd[kl], rmsd=float(r),
                                     form_diff_max=float(x), same_ts=bool(s))
        except Exception as e:                                   # noqa: BLE001
            out["errors"].append(f"{lv}: {type(e).__name__}: {e}")
    return out


def geom_tasks(A, SR, OS, lab):
    tasks, bad = [], []
    for rid in A:
        d = lab.get(rid)
        if d is None or d["job_id"] != SR.loc[rid, "job_id"]:
            bad.append(f"{rid}: not a target label row with job {SR.loc[rid, 'job_id']}")
            continue
        pairs = _pairs_of(d)
        if len(pairs) != 2:
            bad.append(f"{rid}: formed_pairs_ts {d['formed_pairs_ts']!r}")
            continue
        paths = {}
        g1d, g2d = G1_FINAL / str(rid), G2_FINAL / str(rid)
        if _marker(g1d) == ".done" and (g1d / "ts.xyz").is_file():
            paths["g1"] = str(g1d / "ts.xyz")
        if _marker(g2d) == ".done" and (g2d / "ts.xyz").is_file():
            paths["g2"] = str(g2d / "ts.xyz")
        else:
            bad.append(f"{rid}: G2 final folder without .done / ts.xyz")
        for m in C.METHODS:
            fd = FINAL[m] / str(rid)
            mk = _marker(fd)
            st = OS[(OS.method == m) & (OS.part == "A") & (OS.rxn_id == rid)].status.iloc[0]
            if (mk == ".done") != (str(st) == "ok") or mk is None:
                bad.append(f"{m} {rid}: marker {mk} vs O_status status {st!r} (Phase O folder changed after assemble)")
            if mk == ".done":
                if not (fd / "ts.xyz").is_file():
                    bad.append(f"{m} {rid}: .done without ts.xyz")
                else:
                    paths[m] = str(fd / "ts.xyz")
        tasks.append(dict(rxn_id=int(rid), job_id=d["job_id"], dft=str(D1_JOBS / d["job_id"] / d["ts_file"]),
                          pairs=[list(p) for p in pairs], paths=paths))
    if bad:
        C.stop(f"E-1 inputs: {len(bad)} problems: " + "; ".join(bad[:8]))
    return tasks


def run_geometry(tasks, workers):
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as ex:
            res = list(ex.map(geom_one, tasks, chunksize=4))
    else:
        res = [geom_one(t) for t in tasks]
    err = [f"{r['rxn_id']}: {e}" for r in res for e in r["errors"]]
    if err:
        C.stop(f"E-1: {len(err)} structures unreadable / not in label atom order: {err[:6]}")
    return {r["rxn_id"]: r for r in res}


def cc_check(G):
    """g1 / g2 vs the label TS = C_compare.csv (same_ts_g1_dft / rmsd_heavy_g1_dft, same_ts_dft / rmsd_heavy_dft)."""
    cc = pd.read_csv(C_COMPARE, usecols=["rxn_id", "same_ts_dft", "rmsd_heavy_dft", "same_ts_g1_dft",
                                         "rmsd_heavy_g1_dft"]).set_index("rxn_id")
    cols = {"g1": ("same_ts_g1_dft", "rmsd_heavy_g1_dft"), "g2": ("same_ts_dft", "rmsd_heavy_dft")}
    rec, bad = {}, []
    for lv, (cs, cr) in cols.items():
        n, dmax = 0, 0.0
        for rid, g in G.items():
            v = g["levels"].get(lv)
            if v is None:
                continue
            if rid not in cc.index:
                bad.append(f"{lv} {rid}: not in C_compare")
                continue
            s0, r0 = _tri(cc.loc[rid, cs]), float(cc.loc[rid, cr])
            n += 1
            if math.isnan(r0):
                bad.append(f"{lv} {rid}: C_compare {cr} is empty")
                continue
            if s0 is not v["same_ts"]:
                bad.append(f"{lv} {rid}: same_ts {v['same_ts']} vs C_compare {s0}")
            if math.isinf(r0) or math.isinf(v["rmsd"]):
                if r0 != v["rmsd"]:
                    bad.append(f"{lv} {rid}: rmsd {v['rmsd']} vs C_compare {r0}")
            else:
                dmax = max(dmax, abs(r0 - v["rmsd"]))
        rec[lv] = dict(n=n, rmsd_max_abs_diff=dmax, columns=[cs, cr])
        if dmax > TOL_CC:
            bad.append(f"{lv}: rmsd differs from C_compare by {dmax:.3g} A")
    if bad:
        C.stop("E-1 cross-check with C_compare.csv failed (inputs changed?): " + "; ".join(bad[:6]))
    return rec


def e1_tables(G, SP, cls_of):
    rows, lists, detail = [], {}, []
    for rid in sorted(G):
        g = G[rid]
        for lv, v in g["levels"].items():
            detail.append(dict(rxn_id=rid, cls=cls_of[rid], level=lv, d1=v["d"][0], d2=v["d"][1],
                               dform_short=v["dform_short"], dform_long=v["dform_long"], rmsd_heavy=v["rmsd"],
                               form_diff_max=v["form_diff_max"], same_ts=v["same_ts"], short_pair=g["short_pair"]))
    for scope, lvls in SCOPE_LEVELS.items():
        avail = {r for r, g in G.items() if all(lv in g["levels"] for lv in lvls)}
        for cls in CLASSES:
            R = sorted(r for r in avail if cls_of[r] == cls)
            Pp = SP[(SP["class"] == cls) & SP.a_rxn.isin(avail) & SP.b_rxn.isin(avail)]
            lists[f"{scope}|{cls}"] = dict(reactions=R, pairs=Pp.pair_id.tolist())
            for lv in lvls:
                V = [G[r]["levels"][lv] for r in R]
                if len(V) != len(R):
                    C.stop(f"E-1 {scope} {cls} {lv}: reaction list differs between inputs (spec 12 E)")
                sh = np.array([v["dform_short"] for v in V], float)
                lo = np.array([v["dform_long"] for v in V], float)
                ab = np.abs(np.concatenate([sh, lo]))
                mx = np.maximum(np.abs(sh), np.abs(lo))
                rr = np.array([v["rmsd"] for v in V], float)
                ns = sum(bool(v["same_ts"]) for v in V)
                ci = wilson(ns, len(V)) if V else (float("nan"), float("nan"))
                dsh = np.array([G[a]["levels"][lv]["dform_short"] - G[b]["levels"][lv]["dform_short"]
                                for a, b in zip(Pp.a_rxn, Pp.b_rxn)], float)
                dlo = np.array([G[a]["levels"][lv]["dform_long"] - G[b]["levels"][lv]["dform_long"]
                                for a, b in zip(Pp.a_rxn, Pp.b_rxn)], float)
                dab = np.abs(np.concatenate([dsh, dlo]))
                rows.append(dict(
                    scope=scope, **{"class": cls}, level=lv, n_rxn=len(R), n_bonds=int(len(ab)),
                    dform_abs_median=_q(ab, 0.5), dform_abs_p90=_q(ab, 0.9), dform_abs_mean=_mean_fin(ab),
                    dform_short_signed_median=_q(sh, 0.5), dform_long_signed_median=_q(lo, 0.5),
                    dform_rxn_max_median=_q(mx, 0.5), dform_rxn_max_p90=_q(mx, 0.9),
                    rmsd_heavy_median=_q(rr, 0.5, "inverted_cdf"), rmsd_heavy_p90=_q(rr, 0.9, "inverted_cdf"),
                    rmsd_heavy_mean_finite=_mean_fin(rr), n_rmsd_inf=int(np.isinf(rr).sum()), n_same_ts=int(ns),
                    same_ts_frac=frac(ns, len(V)), same_ts_ci_lo=ci[0], same_ts_ci_hi=ci[1], n_pairs=int(len(Pp)),
                    ddd_abs_median=_q(dab, 0.5), ddd_abs_p90=_q(dab, 0.9), ddd_abs_mean=_mean_fin(dab),
                    ddd_short_abs_median=_q(np.abs(dsh), 0.5), ddd_long_abs_median=_q(np.abs(dlo), 0.5)))
    return pd.DataFrame(rows), lists, detail


# ---------------------------------------------------------------- pair lists for E-2 .. E-4
def model_lists(SP, usable):
    """{(scope, class): pair frame (pair_id order)}: both ends usable in every input of the scope (decision EV-1)."""
    out = {}
    for scope, lvls in SCOPE_LEVELS.items():
        ok = set.intersection(*[usable[lv] for lv in lvls])
        for cls in CLASSES:
            Pp = SP[(SP["class"] == cls) & SP.a_rxn.isin(ok) & SP.b_rxn.isin(ok)].sort_values("pair_id")
            out[(scope, cls)] = Pp.reset_index(drop=True)
    return out


def dH(T, lv, Pp):
    """ΔΔ_HF = H(a) − H(b) [n, 9] (T9 order) of input lv on the pairs of Pp; non-finite = STOP (usable rows have H)."""
    X = T[lv]
    a = X.loc[Pp.a_rxn.to_numpy(), H_COLS].to_numpy(float)
    b = X.loc[Pp.b_rxn.to_numpy(), H_COLS].to_numpy(float)
    d = a - b
    if not np.isfinite(d).all():
        C.stop(f"non-finite H values of usable reactions in the {lv} table")
    return d


# ---------------------------------------------------------------- E-2 noise
def e2_table(T, ML):
    rows = []
    for (scope, cls), Pp in ML.items():
        D0 = dH(T, "dft", Pp)
        for lv in SCOPE_LEVELS[scope]:
            D = dH(T, lv, Pp)
            for j, t in enumerate(T9):
                x = D[:, j] - D0[:, j]
                rows.append(dict(scope=scope, **{"class": cls}, level=lv, channel=t, n_pairs=int(len(Pp)),
                                 noise_median=_q(np.abs(x), 0.5), noise_mean=_mean_fin(np.abs(x)),
                                 noise_p90=_q(np.abs(x), 0.9), signed_median=_q(x, 0.5),
                                 dH_dft_abs_median=_q(np.abs(D0[:, j]), 0.5)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- predictions (apply.py)
class Pred:
    KEY = ["input", "train_rung", "arm", "model", "target"]

    def __init__(self, SP, Y):
        need(PRED)
        P = pd.read_parquet(PRED)
        cols = set(P.columns)
        alias = {}                                           # apply.py names -> the names used here (decision EV-7)
        for want, alts in (("input", ("input_level", "input_rung")), ("train_rung", ("model_rung",)),
                           ("source", ("rep_source",))):
            hit = next((x for x in alts if x in cols), None) if want not in cols else None
            if hit is not None:
                alias[hit] = want
        if "train_rung" not in cols and "train_rung" not in alias.values() and "rung" in cols \
                and ("input" in cols or "input" in alias.values()):
            alias["rung"] = "train_rung"                    # 'rung' = the unit's rung only next to an input column
        P = P.rename(columns=alias)
        miss = [c for c in ["pair_id", "fold"] + self.KEY + ["yhat"] if c not in P.columns]
        if miss:
            C.stop(f"{PRED}: columns {miss} missing (contract in the g3_eval.py header; have {sorted(cols)})")
        ids = set(SP.pair_id)
        P["pair_id"] = P.pair_id.astype(str)
        for c in self.KEY:
            P[c] = P[c].astype(str)
        n_all = len(P)
        n_xgb = int((P.model == XGB).sum())
        P = P[P.pair_id.isin(ids) & P.target.isin(T9) & P.train_rung.isin(MODEL_RUNGS) & P.arm.isin(ARMS)
              & (P.model != XGB)].copy()
        P["fold"] = P.fold.astype(int)
        self.inputs = sorted(set(P.input))
        need_in = ["dft", "g2"] + list(C.METHODS)
        absent = [x for x in need_in if x not in self.inputs]
        if absent:
            not_ready(f"{PRED} has no rows for inputs {absent} (apply.py not finished?)")
        if not set(self.inputs) <= set(LEVELS):
            C.stop(f"{PRED}: unknown inputs {sorted(set(self.inputs) - set(LEVELS))}")
        if P.duplicated(self.KEY + ["pair_id"]).any():
            C.stop(f"{PRED}: duplicated (input, train_rung, arm, model, target, pair_id) rows")
        fold = dict(zip(SP.pair_id, SP.fold))
        bad = int((P.fold.to_numpy() != P.pair_id.map(fold).to_numpy()).sum())
        if bad:
            C.stop(f"{PRED}: {bad} rows whose fold != S_pairs fold (spec 6 E-3: own T1 fold unit, no leak)")
        ydev = None
        if "y" in P.columns:
            ydev = 0.0
            for t in T9:
                mk = (P.target == t).to_numpy()
                if mk.any():
                    yl = Y.loc[P.pair_id[mk].tolist(), f"dd_{t}"].to_numpy(float)
                    ydev = max(ydev, float(np.max(np.abs(P["y"].to_numpy(float)[mk] - yl))))
            if not ydev <= TOL_Y:
                C.stop(f"{PRED}: y differs from {IN.DEFAULT_LABELS} by {ydev:.3g} kcal/mol")
        self.has_source = "source" in P.columns
        self.has_base = "base" in P.columns
        self.idx, self.src, self.base = {}, {}, {}
        for k, g in P.groupby(self.KEY, sort=False):
            g = g.set_index("pair_id")
            self.idx[k] = g["yhat"].astype(float)
            if self.has_source:
                self.src[k] = g["source"].astype(str)
            if self.has_base:
                self.base[k] = g["base"].astype(float)
        self.info = dict(path=str(PRED), sha256=C.sha256(PRED), n_rows_file=n_all, n_rows_used=int(len(P)),
                         n_xgb_rows_ignored=n_xgb, columns=sorted(cols), aliases=alias, inputs=self.inputs,
                         models=sorted(set(P.model)), y_max_abs_diff=ydev, has_source=self.has_source,
                         has_base=self.has_base)

    def get(self, X, r, a, model, t, ids, kind):
        """yhat on ids for unit model `model` (or the REP rows when kind = REP); None = no rows for that model (REP_atom
        only); a missing pair = STOP (spec 12 E)."""
        key = (X, r, a, model, t)
        s = self.idx.get(key)
        if s is None and kind == "REP":
            key = (X, r, a, "REP", t)
            s = self.idx.get(key)
            if s is not None and self.has_source:
                bad = sorted(set(self.src[key]) - {model})
                if bad:
                    C.stop(f"{PRED}: REP rows of {X} {r}·{a} {t} have source {bad}, E2 representative is {model}")
        if s is None:
            if kind == "REP":
                C.stop(f"{PRED}: no rows for input {X}, {r}·{a}, model {model} (the E2 representative), target {t}")
            return None
        miss = [p for p in ids if p not in s.index]
        if miss:
            C.stop(f"spec 12 E: {len(miss)} pairs of the comparison list have no prediction for input {X}, {r}·{a}, "
                   f"{model}, {t} (e.g. {miss[:3]})")
        return s.loc[list(ids)].to_numpy(float)


def check_diag(pred, SP, reps):
    """input = train rung: apply.py yhat = L_predictions yhat (<= 1e-3, the T-4 tolerance) for every unit model there."""
    Lp = pd.read_parquet(EV.PRED, columns=["pair_id", "rung", "arm", "model", "target", "fold", "yhat"])
    Lp["pair_id"] = Lp.pair_id.astype(str)
    Lp = Lp[Lp.pair_id.isin(set(SP.pair_id)) & Lp.rung.isin(MODEL_RUNGS) & Lp.model.isin(ATOM) & Lp.target.isin(T9)]
    n, dmax, bad, fold_bad = 0, 0.0, [], 0
    fold = dict(zip(SP.pair_id, SP.fold))
    fold_bad = int((Lp.fold.astype(int).to_numpy() != Lp.pair_id.map(fold).to_numpy()).sum())
    for (r, a, m, t), g in Lp.groupby(["rung", "arm", "model", "target"]):
        s = pred.idx.get((r, r, a, m, t))
        if s is None and reps[(r, a, t)]["rep"] == m:
            s = pred.idx.get((r, r, a, "REP", t))
        if s is None:
            continue
        common = [p for p in g.pair_id if p in s.index]
        if not common:
            continue
        d = np.abs(s.loc[common].to_numpy(float) - g.set_index("pair_id").yhat.loc[common].to_numpy(float))
        n += len(common)
        dmax = max(dmax, float(d.max()))
        if d.max() > TOL_DIAG:
            bad.append(f"{r}·{a} {m} {t}: max |apply - L_predictions| {d.max():.3g}")
    if fold_bad:
        bad.append(f"{fold_bad} L_predictions rows of sample pairs with a fold != S_pairs fold")
    if bad:
        C.stop("diagonal check (apply.py vs L_predictions.parquet): " + "; ".join(bad[:6]))
    return dict(n_compared=n, max_abs_diff=dmax, tol=TOL_DIAG)


def line_pred(T, betas, r, lv, Pp):
    """LINE yhat [n, len(TARGETS)] = beta(rung r, pair fold) x ΔΔ_HF(input lv) (evaluate.line_predict; c_ghost 0)."""
    D = dH(T, lv, Pp)
    D10 = np.zeros((len(Pp), len(EV.TARGETS)))
    for j, t in enumerate(T9):
        D10[:, TIDX[t]] = D[:, j]
    out = np.full_like(D10, np.nan)
    folds = Pp.fold.to_numpy(int)
    for k in np.unique(folds):
        mk = folds == k
        out[mk] = EV.line_predict(betas[r][int(k)], D10[mk])
    return out


def check_base(pred, T, betas, SP, usable):
    """apply.py baselines vs the LINE computed here (<= 1e-6): every SH row's base and every LINE row's yhat (both
    arms) = beta(rung, fold) x ΔΔ_HF(input); every S unit row's base = 0. An SH / LINE row for a pair that is not usable
    in that input's table is a STOP (the two usability rules differ)."""
    rec = dict(n_sh_base=0, sh_base_max_abs_diff=0.0, n_line=0, line_max_abs_diff=0.0, n_s_base=0,
               s_base_max_abs=0.0, tol=TOL_BASE, has_base=pred.has_base)
    for (X, r, a, m, t), s in pred.idx.items():
        if m == "LINE":
            vals, what = s, ("n_line", "line_max_abs_diff")
        elif pred.has_base and a == "SH":
            vals, what = pred.base[(X, r, a, m, t)], ("n_sh_base", "sh_base_max_abs_diff")
        elif pred.has_base and a == "S":
            b = np.abs(pred.base[(X, r, a, m, t)].to_numpy(float))
            rec["n_s_base"] += len(b)
            rec["s_base_max_abs"] = max(rec["s_base_max_abs"], float(b.max()) if len(b) else 0.0)
            continue
        else:
            continue
        Pp = SP[SP.pair_id.isin(vals.index) & SP.a_rxn.isin(usable[X]) & SP.b_rxn.isin(usable[X])]
        extra = sorted(set(vals.index) - set(Pp.pair_id))         # EV-7: no apply row is left unchecked
        if extra:
            C.stop(f"{PRED}: {m} {X} {r}·{a} {t}: {len(extra)} rows for pairs that are not usable in the {X} input table "
                   f"(e.g. {extra[:3]}): apply.py usability != the table 'usable' column")
        if not len(Pp):
            continue
        ln = line_pred(T, betas, r, X, Pp)[:, TIDX[t]]
        d = np.abs(vals.loc[Pp.pair_id.tolist()].to_numpy(float) - ln)
        rec[what[0]] += len(d)
        rec[what[1]] = max(rec[what[1]], float(np.max(d)))
        if not np.max(d) <= TOL_BASE:
            C.stop(f"{PRED}: {m} {X} {r}·{a} {t}: baseline differs from beta x ΔΔ_HF by {np.max(d):.3g} kcal/mol")
    if not rec["s_base_max_abs"] <= TOL_BASE:
        C.stop(f"{PRED}: S rows with a non-zero base ({rec['s_base_max_abs']:.3g})")
    return rec


def check_rep_rows(pred, reps):
    """apply.py REP rows (if any) name the model this file uses: the E2 representative, or for an XGB representative
    the atom model with the lowest E2 validation mean (apply.py A-4 = decision EV-5)."""
    n, bad = 0, []
    for (X, r, a, m, t), s in pred.idx.items():
        if m != "REP" or not pred.has_source:
            continue
        ch = reps[(r, a, t)]
        want = ch["rep"] if ch["rep"] != XGB else ch["rep_atom"]
        got = sorted(set(pred.src[(X, r, a, m, t)]))
        n += 1
        if got != [want]:
            bad.append(f"{X} {r}·{a} {t}: REP rows from {got}, expected {want}")
        if want in ATOM and (X, r, a, want, t) in pred.idx:
            u = pred.idx[(X, r, a, want, t)]
            common = [p for p in s.index if p in u.index]
            if common and np.max(np.abs(s.loc[common].to_numpy(float) - u.loc[common].to_numpy(float))) > 0:
                bad.append(f"{X} {r}·{a} {t}: REP yhat != {want} rows")
    if bad:
        C.stop(f"{PRED}: REP rows inconsistent with the representatives: " + "; ".join(bad[:6]))
    return dict(n_rep_blocks=n)


def check_apply_report():
    """$G3/apply/apply_report.json written by the same apply.py run (outputs sha256 = predictions.parquet)."""
    need(APPLY_REPORT)
    ar = C.read_json(APPLY_REPORT)
    if (ar.get("outputs") or {}).get("predictions_sha256") != C.sha256(PRED):
        C.stop(f"{PRED} sha256 != {APPLY_REPORT} outputs (rerun apply.py e3)")
    sc = ar.get("self_check") or {}
    if sc and not sc.get("ok"):
        C.stop(f"{APPLY_REPORT}: self_check not ok")
    return dict(path=str(APPLY_REPORT), sha256=C.sha256(APPLY_REPORT), self_check_ok=sc.get("ok"),
                common_pairs_n=(ar.get("common_pairs") or {}).get("n"), xgb_replaced=ar.get("xgb_replaced"),
                common_ids=sorted(str(x) for x in ((ar.get("common_pairs") or {}).get("ids") or [])))


def check_t():
    """spec 12 T: every Phase T record passed (T_4.json uses all_pass); a missing record = not ready."""
    need(*T_JSON.values())
    out, bad = {}, []
    for k, p in T_JSON.items():
        r = C.read_json(p)
        ok = bool(r.get("pass")) if "pass" in r else bool(r.get("all_pass"))
        out[f"T-{k}"] = dict(passed=ok, verdict=r.get("verdict"), sha256=C.sha256(p))
        if not ok and r.get("verdict") in ("running", "not_ready"):
            not_ready(f"{p}: verdict {r.get('verdict')}")
        if not ok:
            bad.append(f"T-{k}")
    if bad:
        C.stop(f"Phase T records not passed (spec 12 T): {bad}")
    return out


# ---------------------------------------------------------------- E-3 end-to-end, E-4 recovery
def score(e, cores):
    n = len(e)
    if n == 0:
        return dict(n_pairs=0, n_cores=0, mae=float("nan"), mae_ci_lo=float("nan"), mae_ci_hi=float("nan"),
                    rmse=float("nan"), mean_err=float("nan"))
    d, lo, hi, _ = E9.boot_diff(np.asarray(e, float), np.zeros(n), cores, n_boot=N_BOOT, seed=SEED)
    return dict(n_pairs=int(n), n_cores=int(len(np.unique(cores))), mae=d, mae_ci_lo=lo, mae_ci_hi=hi,
                rmse=float(E9.rmse(np.asarray(e, float))), mean_err=float(np.mean(e)))


def e3_table(ML, pred, reps, T, betas, Y):
    rows, ERR = [], {}
    for (scope, cls), Pp in ML.items():
        ids, cores = Pp.pair_id.tolist(), Pp.core.to_numpy()
        inputs = [x for x in SCOPE_LEVELS[scope] if x in pred.inputs]
        Yp = Y.loc[ids] if ids else None
        for r, a in FAMILIES:
            for t in T9:
                ch = reps[(r, a, t)]
                y = Yp[f"dd_{t}"].to_numpy(float) if ids else np.zeros(0)
                for kind in ("REP", "REP_atom"):
                    src = ch["rep"] if kind == "REP" else ch["rep_atom"]
                    if src is None:
                        continue
                    for X in inputs:
                        base = dict(scope=scope, **{"class": cls}, family=f"{r}·{a}", train_rung=r, arm=a, role=ROLE[r],
                                    model=kind, source=src, target=t, input=X)
                        if kind == "REP" and src == XGB:
                            rows.append(dict(base, **score([], cores),
                                             note="XGB representative: no model file (spec T-4), left out"))
                            continue
                        if not ids:
                            rows.append(dict(base, **score([], cores), note="no pairs"))
                            continue
                        yh = pred.get(X, r, a, src, t, ids, kind)
                        if yh is None:
                            rows.append(dict(base, **score([], cores), note=f"no apply.py rows for {src}"))
                            continue
                        e = yh - y
                        ERR[(scope, cls, r, a, kind, t, X)] = e
                        rows.append(dict(base, **score(e, cores),
                                         note="descriptive: best atom model for an XGB representative" if
                                         kind == "REP_atom" else ""))
        for r in MODEL_RUNGS:
            for X in inputs:
                pr = line_pred(T, betas, r, X, Pp) if ids else None
                for t in T9:
                    base = dict(scope=scope, **{"class": cls}, family=f"LINE·{r}", train_rung=r, arm="-", role=ROLE[r],
                                model="LINE", source=f"E2 line_betas[{r}]", target=t, input=X)
                    if not ids:
                        rows.append(dict(base, **score([], cores), note="no pairs"))
                        continue
                    e = pr[:, TIDX[t]] - Yp[f"dd_{t}"].to_numpy(float)
                    ERR[(scope, cls, r, "-", "LINE", t, X)] = e
                    rows.append(dict(base, **score(e, cores), note=""))
    return pd.DataFrame(rows), ERR


def boot_ratio(eg2, eg3, ed, cores):
    """recovery CI on the boot_diff core resamples (same rng stream: default_rng(SEED), N_BOOT draws of k cores)."""
    uc, inv = np.unique(cores, return_inverse=True)
    k = len(uc)
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(N_BOOT)])
    cnt = np.bincount(inv, minlength=k)
    num = w @ cnt
    m = [(w @ np.bincount(inv, np.abs(e), minlength=k)) / num for e in (eg2, eg3, ed)]
    den = m[0] - m[2]
    ok = den > 0
    r = (m[0][ok] - m[1][ok]) / den[ok]
    lo, hi = (np.percentile(r, [2.5, 97.5]) if ok.any() else (float("nan"), float("nan")))
    g3 = np.percentile(m[1], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), (float(g3[0]), float(g3[1]))


def e4_table(ML, ERR, E3, reps):
    rows, dev = [], 0.0
    ix = E3.set_index(["scope", "class", "family", "model", "target", "input"])
    for (scope, cls), Pp in ML.items():
        cores = Pp.core.to_numpy()
        for t in REC_TARGETS:
            ch = reps[("dft", "SH", t)]
            kind = "REP" if ch["rep"] != XGB else "REP_atom"
            src = ch["rep"] if kind == "REP" else ch["rep_atom"]
            for m in [x for x in C.METHODS if x in SCOPE_LEVELS[scope]]:
                base = dict(scope=scope, **{"class": cls}, method=m, target=t, family="dft·SH", model=kind, source=src,
                            n_pairs=int(len(Pp)), n_cores=int(len(np.unique(cores))))
                ks = [(scope, cls, "dft", "SH", kind, t, X) for X in ("g2", m, "dft")]
                if any(k not in ERR for k in ks) or not len(Pp):
                    rows.append(dict(base, mae_g2=float("nan"), mae_g3=float("nan"), mae_dft=float("nan"),
                                     gap_g2_dft=float("nan"), recovery=float("nan"), recovery_ci_lo=float("nan"),
                                     recovery_ci_hi=float("nan"), n_boot_valid=0, note="no pairs / no predictions"))
                    continue
                eg2, eg3, ed = (ERR[k] for k in ks)
                mg2, mg3, md = (float(E9.mae(e)) for e in (eg2, eg3, ed))
                gap = mg2 - md
                rec = (mg2 - mg3) / gap if gap > 0 else float("nan")
                lo, hi, nv, g3ci = boot_ratio(eg2, eg3, ed, cores)
                ref = ix.loc[(scope, cls, "dft·SH", kind, t, m)]
                dev = max(dev, abs(g3ci[0] - float(ref.mae_ci_lo)), abs(g3ci[1] - float(ref.mae_ci_hi)),
                          abs(mg3 - float(ref.mae)))
                rows.append(dict(base, mae_g2=mg2, mae_g3=mg3, mae_dft=md, gap_g2_dft=gap, recovery=rec,
                                 recovery_ci_lo=lo, recovery_ci_hi=hi, n_boot_valid=nv,
                                 note="" if gap > 0 else "MAE_g2 <= MAE_dft: recovery undefined"))
    if dev > TOL_BOOT:
        C.stop(f"E-4 resampling differs from d1_eval.boot_diff (max {dev:.3g}): the recovery CI would use other resamples")
    return pd.DataFrame(rows), dict(boot_selfcheck_max_abs_diff=dev, tol=TOL_BOOT)


# ---------------------------------------------------------------- E-5 cost / success, E-6 Part B
def _stats(x):
    x = np.asarray([v for v in (_f(u) for u in x) if v is not None], float)
    if not len(x):
        return float("nan"), float("nan"), 0.0
    return float(np.median(x)), float(x.mean()), float(x.sum())


def e5_table(OS, OR, cls_of, B):
    rows = []
    for m in C.METHODS:
        Aq = OS[(OS.method == m) & (OS.part == "A")]
        for cls in CLASSES + ("all",):
            q = Aq if cls == "all" else Aq[Aq.rxn_id.map(cls_of) == cls]
            st = q.ts_status.fillna("").astype(str)
            ok, fail, err = st == "ok", st == "fail", st == "error"
            n_ok, n_fail, n_err = int(ok.sum()), int(fail.sum()), int(err.sum())
            kept = [_tri(v) for v in q.shape_kept[ok]]
            n_kept = sum(1 for v in kept if v is True)
            med, mean, tot = _stats(q.core_h[ok | fail])
            _, _, tot_all = _stats(q.core_h)
            rmed, rmean, rtot = _stats(q.core_h_refs)
            sci = wilson(n_ok, n_ok + n_fail) if n_ok + n_fail else (float("nan"), float("nan"))
            kci = wilson(n_kept, n_ok) if n_ok else (float("nan"), float("nan"))
            row = dict(method=m, **{"class": cls}, n_rxn=int(len(q)), n_ts_ok=n_ok, n_ts_fail=n_fail, n_ts_error=n_err,
                       ts_success=frac(n_ok, n_ok + n_fail), ts_success_ci_lo=sci[0], ts_success_ci_hi=sci[1],
                       n_shape_kept=n_kept, shape_kept_rate=frac(n_kept, n_ok), shape_kept_ci_lo=kci[0],
                       shape_kept_ci_hi=kci[1], n_final_done=int((q.status.astype(str) == "ok").sum()),
                       final_done_rate=frac(int((q.status.astype(str) == "ok").sum()), len(q)),
                       n_retry=int((pd.to_numeric(q.attempt_used, errors="coerce") == 2).sum()),
                       ts_core_h_median=med, ts_core_h_mean=mean, ts_core_h_sum=tot,
                       ts_core_h_per_success=(tot_all / n_ok) if n_ok else float("nan"),
                       ref_core_h_rxn_median=rmed, ref_core_h_rxn_mean=rmean, ref_core_h_rxn_sum=rtot,
                       label_ts_core_h=LABEL_TS_CORE_H, ts_core_h_median_over_label=med / LABEL_TS_CORE_H)
            if cls == "all":
                u = OR[OR.method == m]
                umed, umean, utot = _stats(u.core_h)
                row.update(n_ref_units=int(len(u)), n_ref_units_ok=int((u.status.astype(str) == "ok").sum()),
                           ref_unit_core_h_median=umed, ref_unit_core_h_mean=umean, ref_unit_core_h_sum=utot)
            rows.append(row)
    if len(B):
        med, mean, tot = _stats(B.core_h)
        ok = B.ts_status.astype(str) == "ok"
        rows.append(dict(method="partB", **{"class": "all"}, n_rxn=int(len(B)), n_ts_ok=int(ok.sum()),
                         n_ts_fail=int((B.ts_status.astype(str) == "fail").sum()),
                         n_ts_error=int((B.ts_status.astype(str) == "error").sum()),
                         ts_success=frac(int(ok.sum()), int((B.ts_status.astype(str).isin(["ok", "fail"])).sum())),
                         ts_core_h_median=_stats(B.core_h_optts)[0], ts_core_h_mean=_stats(B.core_h_optts)[1],
                         ts_core_h_sum=_stats(B.core_h_optts)[2], ref_core_h_rxn_median=_stats(B.core_h_sp)[0],
                         ref_core_h_rxn_mean=_stats(B.core_h_sp)[1], ref_core_h_rxn_sum=_stats(B.core_h_sp)[2],
                         ts_core_h_per_success=(tot / int(ok.sum())) if int(ok.sum()) else float("nan"),
                         label_ts_core_h=LABEL_TS_CORE_H,
                         ts_core_h_median_over_label=_stats(B.core_h_optts)[0] / LABEL_TS_CORE_H))
    cols = ["method", "class", "n_rxn", "n_ts_ok", "n_ts_fail", "n_ts_error", "ts_success", "ts_success_ci_lo",
            "ts_success_ci_hi", "n_shape_kept", "shape_kept_rate", "shape_kept_ci_lo", "shape_kept_ci_hi",
            "n_final_done", "final_done_rate", "n_retry", "ts_core_h_median", "ts_core_h_mean", "ts_core_h_sum",
            "ts_core_h_per_success", "ref_core_h_rxn_median", "ref_core_h_rxn_mean", "ref_core_h_rxn_sum",
            "n_ref_units", "n_ref_units_ok", "ref_unit_core_h_median", "ref_unit_core_h_mean", "ref_unit_core_h_sum",
            "label_ts_core_h", "ts_core_h_median_over_label"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _bin(v):
    if v is None:
        return ""
    return "lt_m05" if v < -PARTB_CUT else ("gt_05" if v > PARTB_CUT else "abs_le_05")


def b_table(OS, SR, B):
    rows, bad = [], []
    Bq = OS[OS.method == "partB"].set_index("rxn_id")
    for rid in B:
        r = Bq.loc[rid]
        p = PARTB / str(rid) / "result.json"
        if not p.is_file():
            not_ready(f"{p} missing (opt.py assemble)")
        res = C.read_json(p)
        tss = str(r.ts_status) if isinstance(r.ts_status, str) else ""
        resolved = tss == "ok"
        conv = _tri(r.converged_to_label)
        dE = _f(r.dE_kcal)
        if resolved and conv is not True and dE is None:
            bad.append(f"{rid}: ok, not converged_to_label, but no dE_kcal")
        eff = (0.0 if conv is True else dE) if resolved else None
        rows.append(dict(rxn_id=int(rid), job_id=SR.loc[rid, "job_id"], dipole=SR.loc[rid, "dipole"],
                         ts_class=SR.loc[rid, "ts_class"], part_a=bool(SR.loc[rid, "part_a"]), status=r.status,
                         reason=r.reason, ts_status=tss, ts_reason=r.ts_reason, resolved=resolved,
                         converged_to_label=conv, dE_kcal=dE, dE_eff_kcal=eff, bin=_bin(eff), sp_mode=r.sp_mode,
                         rmsd_label_A=_f(res.get("rmsd_label_A")), dform_label_A=_f(res.get("dform_label_A")),
                         same_as_g2_start=_tri(res.get("same_as_g2_start")), rmsd_g2_A=_f(res.get("rmsd_g2_A")),
                         n_imag=_f(r.n_imag), imag_freqs_cm=r.imag_freqs_cm, core_h=_f(r.core_h),
                         core_h_optts=_f(res.get("core_h_optts")), core_h_sp=_f(res.get("core_h_sp")),
                         e_sp_eh=_f(r.e_sp_eh), e_label_eh=_f(r.e_label_eh)))
    if bad:
        C.stop("Part B results inconsistent: " + "; ".join(bad[:6]))
    if not rows:
        C.stop("no Part B reaction in S_rxns.csv (spec 2 Part B sample is empty)")
    return pd.DataFrame(rows)


def partb_summary(Bt):
    def grp(q):
        res = q[q.resolved.astype(bool)]
        n_res = int(len(res))
        conv = int((res.converged_to_label == True).sum())                              # noqa: E712
        sp = res[res.converged_to_label != True]                                        # noqa: E712
        k = {b: int((res.bin == b).sum()) for b in ("lt_m05", "abs_le_05", "gt_05")}
        ksp = {b: int((sp.bin == b).sum()) for b in ("lt_m05", "abs_le_05", "gt_05")}
        ci = wilson(k["lt_m05"], n_res) if n_res else (float("nan"), float("nan"))
        d = sp.dE_kcal.astype(float).to_numpy()
        return dict(n=int(len(q)), n_resolved=n_res, n_unresolved=int(len(q) - n_res),
                    n_fail=int((q.ts_status == "fail").sum()), n_error=int((q.ts_status == "error").sum()),
                    n_converged_to_label=conv, n_sp=int(len(sp)), n_lt_m05=k["lt_m05"], n_abs_le_05=k["abs_le_05"],
                    n_gt_05=k["gt_05"], frac_lt_m05=frac(k["lt_m05"], n_res), frac_lt_m05_ci=list(ci),
                    frac_abs_le_05=frac(k["abs_le_05"], n_res), frac_gt_05=frac(k["gt_05"], n_res),
                    frac_converged_to_label=frac(conv, n_res), sp_only=dict(
                        n=int(len(sp)), n_lt_m05=ksp["lt_m05"], n_abs_le_05=ksp["abs_le_05"], n_gt_05=ksp["gt_05"],
                        frac_lt_m05=frac(ksp["lt_m05"], len(sp)), dE_median=_med_fin(d), dE_mean=_mean_fin(d),
                        dE_min=float(np.min(d)) if len(d) else float("nan"),
                        dE_max=float(np.max(d)) if len(d) else float("nan")),
                    unresolved=q.loc[~q.resolved.astype(bool), ["rxn_id", "ts_status", "reason"]].to_dict("records"))
    out = dict(all=grp(Bt), by_dipole={d: grp(q) for d, q in sorted(Bt.groupby("dipole"), key=lambda x: str(x[0]))})
    a = out["all"]
    if a["n_resolved"]:
        high = a["frac_lt_m05"] >= PARTB_FRAC
        out["interpretation"] = dict(frac_lt_m05=a["frac_lt_m05"], n_lt_m05=a["n_lt_m05"], n_resolved=a["n_resolved"],
                                     threshold=PARTB_FRAC, rule="9-B-high" if high else "9-B-low",
                                     sentence=(B_SENT_HIGH if high else B_SENT_LOW) + ".")
    else:
        out["interpretation"] = dict(frac_lt_m05=None, n_resolved=0, threshold=PARTB_FRAC, rule=None, sentence=None,
                                     note="no resolved Part B reaction: the spec 9 sentence cannot be chosen")
    out["rule"] = DECISIONS["EV-10"]
    return out


# ---------------------------------------------------------------- spec 9 decision
def spec9(E3, E4, E5):
    out = {}
    for m in C.METHODS:
        sc = f"own_{m}"                                    # EV-11: each method on its own pair list
        r = E4[(E4.scope == sc) & (E4["class"] == "same") & (E4.method == m) & (E4.target == "oi")]
        rec = _f(r.recovery.iloc[0]) if len(r) else None
        kind = r.model.iloc[0] if len(r) else "REP"
        q = E3[(E3.scope == sc) & (E3["class"] == "same") & (E3.family == "dft·SH") & (E3.model == kind)
               & (E3.target == "oi") & (E3.input == m)]
        mae = _f(q.mae.iloc[0]) if len(q) else None
        c = E5[(E5.method == m) & (E5["class"] == "same")]
        ts, sk, ch = ((_f(c[x].iloc[0]) if len(c) else None) for x in ("ts_success", "shape_kept_rate",
                                                                        "ts_core_h_median"))
        c1 = rec is not None and rec >= CRIT["rec_oi"] and mae is not None and mae <= CRIT["mae_oi"]
        c2 = ts is not None and ts >= CRIT["ts_success"] and sk is not None and sk >= CRIT["shape_kept"]
        c3 = ch is not None and ch <= CRIT["core_h_median"]
        v = "go" if (c1 and c2 and c3) else ("partial" if c1 else "impossible")
        out[m] = dict(scope=sc, c1=dict(ok=c1, recovery_oi=rec, mae_oi_dftSH=mae, model=kind,
                                        n_pairs=int(q.n_pairs.iloc[0]) if len(q) else 0,
                                        thresholds=dict(recovery=CRIT["rec_oi"], mae=CRIT["mae_oi"])),
                      c2=dict(ok=c2, ts_success=ts, shape_kept=sk, thresholds=dict(ts_success=CRIT["ts_success"],
                                                                                  shape_kept=CRIT["shape_kept"])),
                      c3=dict(ok=c3, ts_core_h_median=ch, ts_core_h_mean=_f(c.ts_core_h_mean.iloc[0]) if len(c) else None,
                              threshold=CRIT["core_h_median"], label_ts_core_h=LABEL_TS_CORE_H),
                      verdict=v, verdict_ko=VERDICT_KO[v])
    gos = [m for m in C.METHODS if out[m]["verdict"] == "go"]
    if len(gos) >= 2:
        best = sorted(gos, key=lambda m: (out[m]["c3"]["ts_core_h_median"], out[m]["c3"]["ts_core_h_mean"] or 0.0,
                                          C.METHODS.index(m)))[0]
        oth = [m for m in gos if m != best][0]
        txt = (f"두 방법 모두 go이고 TS 하나당 코어시간 중앙값이 더 적은 {best}({out[best]['c3']['ts_core_h_median']:.3f} "
               f"대 {oth} {out[oth]['c3']['ts_core_h_median']:.3f} 코어시간)로 본격 진행을 권고한다.")
        rec_m = best
    elif len(gos) == 1:
        rec_m = gos[0]
        oth = [m for m in C.METHODS if m != rec_m][0]
        txt = f"{rec_m}로 본격 진행을 권고한다({oth} 판정은 '{out[oth]['verdict_ko']}')."
    else:
        rec_m = None
        part = [m for m in C.METHODS if out[m]["verdict"] == "partial"]
        if part:                                           # one sentence (spec 10: the third of the first three)
            txt = (f"본격 진행을 권고하는 방법은 없다({', '.join(part)}는 정확도 기준 ①만 만족: 정확도는 되지만 비용이나 "
                   f"안정성이 문제다).")
        else:
            txt = "본격 진행을 권고하는 방법은 없다(두 방법 모두 ① 불만족)."
    return dict(methods=out, recommended=rec_m, recommendation_ko=txt, scope={m: f"own_{m}" for m in C.METHODS},
                cls="same", rule=DECISIONS["EV-11"])


# ---------------------------------------------------------------- figures
def _save(plt, fig, name, df):
    paths = []
    for ext in ("png", "pdf"):
        p = C.write_atomic(FIGDIR / f"{name}.{ext}", lambda t, e=ext: fig.savefig(t, format=e, bbox_inches="tight"))
        paths.append(str(p))
    paths.append(str(write_csv(FIGDIR / f"{name}.csv", df)))
    plt.close(fig)
    return paths


def fig_noise(plt, DF, N):
    d = N[(N.scope == "common") & (N.level != "dft")].copy()
    lv = [x for x in LEVELS if x != "dft"]
    fig, axs = plt.subplots(1, 2, figsize=(DP.FIG_W, DP.FIG_W * 0.40), sharey=True, constrained_layout=True)
    x = np.arange(len(T9))
    w = 0.18
    for ax, cls in zip(axs, CLASSES):
        q = d[d["class"] == cls]
        n = int(q.n_pairs.max()) if len(q) else 0
        for k, l in enumerate(lv):
            v = np.array([q[(q.level == l) & (q.channel == t)].noise_median.mean() for t in T9], float)
            ok = np.isfinite(v) & (v > 0)
            ax.plot(x[ok] + (k - (len(lv) - 1) / 2) * w, v[ok], linestyle="none", marker=LEVEL_MARK[l], ms=6,
                    color=DF.SLOT[LEVEL_SLOT[l]], label=LEVEL_LABEL[l])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([DF.TLAB.get(t, t) for t in T9], rotation=45, ha="right")
        ax.set_title(f"{cls}-TS pairs (n = {n})")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("median |ΔΔ_HF(input) − ΔΔ_HF(DFT TS)| (kcal/mol)")
    axs[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(plt, fig, FIGS[0], d)


def fig_endtoend(plt, DF, E3):
    from matplotlib.lines import Line2D
    series = (("dft·SH", "o", True, -0.22), ("dft·S", "s", False, 0.0), ("LINE·dft", "X", True, 0.22))
    rows = []
    fig, axs = plt.subplots(2, 3, figsize=(DP.FIG_W, DP.FIG_W * 0.62), constrained_layout=True)
    for i, cls in enumerate(CLASSES):
        for j, t in enumerate(FIG2_TARGETS):
            ax = axs[i, j]
            n = 0
            for name, mk, filled, off in series:
                for xpos, X in enumerate(FIG2_INPUTS):
                    q = E3[(E3.scope == "common") & (E3["class"] == cls) & (E3.family == name) & (E3.target == t)
                           & (E3.input == X)]
                    q = q[q.model.isin(["REP", "LINE"]) & q.mae.notna()]
                    sub = False
                    if not len(q):
                        q = E3[(E3.scope == "common") & (E3["class"] == cls) & (E3.family == name) & (E3.target == t)
                               & (E3.input == X) & (E3.model == "REP_atom") & E3.mae.notna()]
                        sub = bool(len(q))
                    if not len(q):
                        continue
                    r = q.iloc[0]
                    n = max(n, int(r.n_pairs))
                    c = DF.SLOT[LEVEL_SLOT[X]]
                    yerr = [[max(0.0, r.mae - r.mae_ci_lo)], [max(0.0, r.mae_ci_hi - r.mae)]]
                    ax.errorbar(xpos + off, r.mae, yerr=yerr, fmt=mk, ms=6,
                                color=c, mfc=c if filled else "white", mec=c, elinewidth=1.0, capsize=0)
                    if sub:
                        ax.annotate("†", (xpos + off, r.mae), textcoords="offset points", xytext=(4, 2),
                                    fontsize=DP.FONT_PT, color=DF.INK2)
                    rows.append(dict(**{"class": cls}, target=t, family=name, input=X, model=r.model, source=r.source,
                                     n_pairs=int(r.n_pairs), mae=r.mae, mae_ci_lo=r.mae_ci_lo, mae_ci_hi=r.mae_ci_hi,
                                     substituted=sub))
            if t == "oi":
                ax.axhline(CRIT["mae_oi"], color=DF.MUTED, lw=0.8, ls="--")
                ax.annotate("0.20 (spec 9, criterion 1)", (len(FIG2_INPUTS) - 0.5, CRIT["mae_oi"]),
                            textcoords="offset points",
                            xytext=(0, 3), ha="right", fontsize=DP.FONT_PT - 1, color=DF.INK2)
            ax.set_xticks(range(len(FIG2_INPUTS)))
            ax.set_xticklabels(list(FIG2_INPUTS))
            ax.set_xlim(-0.6, len(FIG2_INPUTS) - 0.4)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{DF.TLAB.get(t, t)} · {cls}-TS pairs (n = {n})")
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel("para MAE (kcal/mol)")
    prox = [Line2D([0], [0], marker=mk, linestyle="none", color=DF.INK2, mfc=DF.INK2 if f else "white", mec=DF.INK2,
                   ms=6, label=f"{nm} {'REP' if nm != 'LINE·dft' else '(β of the dft SH unit)'}")
            for nm, mk, f, _ in series]
    axs[0, -1].legend(handles=prox, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(plt, fig, FIGS[1], pd.DataFrame(rows))


def fig_partb(plt, DF, Bt):
    res = Bt[Bt.resolved.astype(bool)]
    conv = int((res.converged_to_label == True).sum())                                  # noqa: E712
    sp = res[res.converged_to_label != True]                                            # noqa: E712
    d = sp.dE_kcal.astype(float).to_numpy()
    d = d[np.isfinite(d)]
    fig, ax = plt.subplots(figsize=(DP.FIG_W * 0.6, DP.FIG_W * 0.36), constrained_layout=True)
    rows = []
    cat_col = {"lt_m05": DF.SLOT[0], "abs_le_05": DF.MUTED, "gt_05": DF.SLOT[1]}
    cat_lab = {"lt_m05": "ΔE‡ < −0.5 (G2-form TS lower)", "abs_le_05": "|ΔE‡| ≤ 0.5",
               "gt_05": "ΔE‡ > 0.5 (label TS lower)"}
    clip = 20.0
    if len(d):
        lo = max(-clip, math.floor(min(d.min(), -1.0) * 2) / 2)
        hi = min(clip, math.ceil(max(d.max(), 1.0) * 2) / 2)
        edges = np.arange(int(round(lo * 2)), int(round(hi * 2)) + 1) / 2.0
        n_clip = int(((d < edges[0]) | (d > edges[-1])).sum())
        counts, _ = np.histogram(np.clip(d, edges[0], edges[-1]), edges)
        for l_, r_, c in zip(edges[:-1], edges[1:], counts):
            cat = "lt_m05" if r_ <= -PARTB_CUT else ("gt_05" if l_ >= PARTB_CUT else "abs_le_05")
            rows.append(dict(bin_lo=float(l_), bin_hi=float(r_), count=int(c), category=cat))
        ax.axvspan(-PARTB_CUT, PARTB_CUT, color=DF.GRID, alpha=0.6, zorder=0, lw=0)
        for cat in ("lt_m05", "abs_le_05", "gt_05"):
            sel = [x for x in rows if x["category"] == cat]
            ax.bar([x["bin_lo"] for x in sel], [x["count"] for x in sel], width=0.5, align="edge", color=cat_col[cat],
                   edgecolor="white", linewidth=1.0, label=cat_lab[cat], zorder=2)
        note = f"converged to the label TS: {conv} (ΔE‡ = 0, not drawn)"
        if n_clip:
            note += f"; {n_clip} values beyond ±{clip:.0f} kcal/mol in the end bins"
        ax.text(0.01, 0.98, note, transform=ax.transAxes, va="top", ha="left", fontsize=DP.FONT_PT - 1, color=DF.INK2)
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, f"no ΔE‡ values (converged to the label TS: {conv})", transform=ax.transAxes, ha="center",
                va="center", color=DF.INK2)
    ax.set_xlabel("ΔE‡ = E(G2-form TS) − E(label TS), label SP level (kcal/mol)")
    ax.set_ylabel("reactions")
    ax.grid(axis="x", visible=False)
    a = partb_summary(Bt)["all"]
    ax.set_title(f"Part B: {a['n_lt_m05']} / {a['n_resolved']} resolved reactions with ΔE‡ < −0.5")
    return _save(plt, fig, FIGS[2], pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "count", "category"]))


def figures(N, E3, Bt):
    sys.path.insert(0, str(C.P_DIR))
    import d1_figures as DF
    if Path(DF.__file__).resolve() != (C.P_DIR / "d1_figures.py").resolve():
        C.stop(f"d1_figures imported from {DF.__file__}")
    plt = DF.style()
    return {FIGS[0]: fig_noise(plt, DF, N), FIGS[1]: fig_endtoend(plt, DF, E3), FIGS[2]: fig_partb(plt, DF, Bt)}


# ---------------------------------------------------------------- run
def cmd_run(workers, with_figures=True):
    t0 = time.time()
    for k, v in DECISIONS.items():                             # spec 11: recorded before they are acted on
        record_decision(k, v)
    rep = dict(phase="E", spec="docs/specs/D1G3.md 6, 9, 12 E", code=CODE_VERSION, started=now(), env=env_rec(),
               code_sha256=C.sha256(Path(__file__).resolve()), g3_common_sha256=C.sha256(HERE / "g3_common.py"))
    rep["modules"] = check_modules()
    rep["spec_check"] = check_spec()
    rep["start"], f_start = load_start()
    rep["phase_t"] = check_t()
    SP, SR, A, B, cls_of, s_info, f_s = load_s()
    OS, OR, o_rep, o_est, f_o = load_o(A, B)
    Tg, ug, f_rep, f_f = load_f()
    Tl, ul, p012, Y, E2, f_l = load_ladder()
    T, usable = {**Tl, **Tg}, {**ul, **ug}
    rep["inputs_sha256"] = {**f_start, **f_s, **f_o, **f_f, **f_l, str(C_COMPARE): C.sha256(C_COMPARE)}
    rep["sample"] = s_info
    miss = sorted(set(SP.pair_id) - p012)
    notu = {r: sorted(set(A) - usable[r]) for r in LAD_RUNGS}
    if miss or any(notu.values()):
        nu = [(r, v[:3]) for r, v in notu.items() if v]
        C.stop(f"sample pairs outside P_012 {miss[:3]} / Part A reactions not usable in the D1LADDER rungs {nu}")
    reps = rep_choices(E2)
    betas = line_betas(E2)
    rep["representatives"] = {f"{r}|{a}|{t}": v for (r, a, t), v in reps.items()}
    print(f"[E {time.time() - t0:5.0f}s] inputs: {len(SP)} pairs, {len(A)} Part A, {len(B)} Part B", flush=True)
    # E-1
    lab = {int(d["rxn_id"]): d for d in C.targets()}
    G = run_geometry(geom_tasks(A, SR, OS, lab), workers)
    rep["e1_cross_check_c_compare"] = cc_check(G)
    E1, glists, detail = e1_tables(G, SP, cls_of)
    rep["e1_availability"] = {lv: sum(1 for g in G.values() if lv in g["levels"]) for lv in LEVELS}
    rep["e1_lists"] = glists
    rep["e1_detail"] = detail
    print(f"[E {time.time() - t0:5.0f}s] E-1 done: {rep['e1_availability']}", flush=True)
    # pair lists for E-2 .. E-4
    ML = model_lists(SP, usable)
    rep["pair_lists"] = {f"{s}|{c}": dict(n=int(len(P)), pair_ids=P.pair_id.tolist(),
                                          sha256=DP.sha256_text("\n".join(P.pair_id)))
                         for (s, c), P in ML.items()}
    rep["usable"] = {lv: dict(n_part_a_usable=len(set(A) & usable[lv])) for lv in LEVELS}
    N = e2_table(T, ML)
    pred = Pred(SP, Y)
    rep["predictions"] = pred.info
    ar = check_apply_report()
    mine = sorted(set(ML[("common", "same")].pair_id) | set(ML[("common", "different")].pair_id))
    ids_ar = ar.pop("common_ids")
    ar["common_vs_this_file"] = dict(n_apply=len(ids_ar), n_here=len(mine), only_apply=sorted(set(ids_ar) - set(mine)),
                                     only_here=sorted(set(mine) - set(ids_ar)),
                                     note="recorded; a pair of this file's list without predictions is a STOP in E-3")
    rep["apply_report"] = ar
    rep["check_diagonal"] = check_diag(pred, SP, reps)
    rep["check_base"] = check_base(pred, T, betas, SP, usable)
    rep["check_rep_rows"] = check_rep_rows(pred, reps)
    E3, ERR = e3_table(ML, pred, reps, T, betas, Y)
    E4, bchk = e4_table(ML, ERR, E3, reps)
    rep["check_boot"] = bchk
    rep["identical_lists"] = dict(rule=SPEC_TEXT["12-E"], ok=True,
                                  note="every scope x class: one pair list for all inputs (E-1 by construction; E-2 / "
                                       "E-3 dH and predictions on exactly that list, a missing pair = STOP)")
    print(f"[E {time.time() - t0:5.0f}s] E-2 / E-3 / E-4 done", flush=True)
    Bt = b_table(OS, SR, B)
    E5 = e5_table(OS, OR, cls_of, Bt)
    rep["partB"] = partb_summary(Bt)
    rep["spec9"] = spec9(E3, E4, E5)
    rep["phase_reports"] = dict(O=dict(complete=o_rep.get("complete"), stop=o_rep.get("stop"),
                                       measured_core_h=o_rep.get("measured_core_h"),
                                       estimate_total_core_h=(o_est or {}).get("total_core_h")),
                                F=dict(complete=f_rep.get("complete"), stop=f_rep.get("stop"),
                                       pairs_usable_all_methods=f_rep.get("pairs_usable_all_methods")))
    outs = {}
    for k, df in (("geometry", E1), ("noise", N), ("endtoend", E3), ("recovery", E4), ("cost", E5), ("partB", Bt)):
        write_csv(OUT[k], df)
        outs[str(OUT[k])] = C.sha256(OUT[k])
    if with_figures:
        figs = figures(N, E3, Bt)
        rep["figures"] = figs
        for ps in figs.values():
            for p in ps:
                outs[p] = C.sha256(p)
    else:
        rep["figures"] = None
    rep["decisions"] = DECISIONS
    rep.update(outputs=outs, stop=False, finished=now(), wall_s=round(time.time() - t0, 1))
    write_json(E_REPORT, rep)
    s9 = rep["spec9"]
    print(json.dumps(EV._clean(dict(verdicts={m: v["verdict"] for m, v in s9["methods"].items()},
                                    recommended=s9["recommended"], partB=rep["partB"]["interpretation"])),
                     indent=1, ensure_ascii=False), flush=True)
    print(f"[E] done in {time.time() - t0:.0f} s -> {E_REPORT}", flush=True)
    return 0


# ---------------------------------------------------------------- Phase R: SUMMARY_G3.md
def fm(x, nd=3):
    v = _f(x)
    if v is None:
        try:
            if x is not None and math.isinf(float(x)):
                return "∞"
        except (TypeError, ValueError):
            pass
        return "—"
    return f"{v:.{nd}f}"


def fi(x):
    v = _f(x)
    return "—" if v is None else f"{int(round(v)):,}"


def _rj(p):
    try:
        return C.read_json(p)
    except (OSError, ValueError):
        return None


def _row(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.iloc[0] if len(q) else None


def _ci(r, col="mae"):
    if r is None or _f(r[col]) is None:
        return "—"
    return f"{fm(r[col])} [{fm(r[col + '_ci_lo'])}, {fm(r[col + '_ci_hi'])}]"


def cmd_summary():
    need(E_REPORT, *OUT.values())
    E = C.read_json(E_REPORT)
    if E.get("stop"):
        C.stop(f"{E_REPORT}: stop")
    stale = [p for p, s in (E.get("outputs") or {}).items() if not Path(p).is_file() or C.sha256(p) != s]
    if stale:
        not_ready(f"E outputs changed since E_report.json (rerun 'g3_eval.py run'): {stale[:3]}")
    G = pd.read_csv(OUT["geometry"])
    N = pd.read_csv(OUT["noise"])
    E3 = pd.read_csv(OUT["endtoend"], keep_default_na=True)
    E4 = pd.read_csv(OUT["recovery"])
    E5 = pd.read_csv(OUT["cost"])
    st, srep = _rj(START) or {}, _rj(S_REPORT) or {}
    Tj = {k: _rj(p) for k, p in T_JSON.items()}
    orep, oest, frep = _rj(O_REPORT) or {}, _rj(O_ESTIMATE) or {}, _rj(F_REPORT) or {}
    s9, pb = E["spec9"], E["partB"]
    M = s9["methods"]
    hm, rm_ = C.METHODS
    n_own = {m: ((E.get("pair_lists") or {}).get(f"own_{m}|same") or {}).get("n") for m in C.METHODS}   # EV-11
    lab_ts = next((v for v in (_f(x) for x in E5.get("label_ts_core_h", pd.Series(dtype=float))) if v is not None),
                  None)                                    # EV-14: from E_cost.csv
    L_ = []
    w = L_.append
    w("# D1-G3 결과 요약 — G2 TS를 값싼 방법으로 다시 최적화하는 시범(HF/6-31G\\* 대 r2SCAN-3c)과 Part B TS 형태 확인")
    w("")
    # first three sentences (spec 10)
    s1 = (f"판정(명령서 9절, `same` 분류, 방법별 짝 목록 g3h {fi(n_own[hm])}개 / g3r {fi(n_own[rm_])}개): "
          f"{METHOD_KO[hm]} 판정은 '{M[hm]['verdict_ko']}', "
          f"{METHOD_KO[rm_]} 판정은 '{M[rm_]['verdict_ko']}'이며, 근거 수치(g3h / g3r)는 oi 회복률 "
          f"{fm(M[hm]['c1']['recovery_oi'])} / {fm(M[rm_]['c1']['recovery_oi'])}, dft·SH 대표 oi para MAE "
          f"{fm(M[hm]['c1']['mae_oi_dftSH'])} / {fm(M[rm_]['c1']['mae_oi_dftSH'])} kcal/mol, TS 성공률 "
          f"{fm(M[hm]['c2']['ts_success'])} / {fm(M[rm_]['c2']['ts_success'])}, 형태 유지율 "
          f"{fm(M[hm]['c2']['shape_kept'])} / {fm(M[rm_]['c2']['shape_kept'])}, TS 하나당 코어시간 중앙값 "
          f"{fm(M[hm]['c3']['ts_core_h_median'])} / {fm(M[rm_]['c3']['ts_core_h_median'])}이다.")
    a = pb["all"]
    if a.get("n_resolved"):
        s2 = (f"Part B에서 라벨 수준 ΔE‡ < −0.5 kcal/mol(G2 형태 TS가 더 낮음)인 반응은 결과가 있는 {fi(a['n_resolved'])}"
              f"반응 중 {fi(a['n_lt_m05'])}개(비율 {fm(a['frac_lt_m05'])})이고, 그중 라벨 TS로 수렴한 반응"
              f"(converged_to_label, ΔE‡ = 0으로 셈)은 {fi(a['n_converged_to_label'])}개이다.")
    else:
        s2 = f"Part B는 결과가 있는 반응이 없어(전체 {fi(a.get('n'))}반응) ΔE‡ 비율을 낼 수 없다."
    s3 = f"권고: {s9['recommendation_ko']}"
    w(f"{s1} {s2} {s3}")
    w("")
    w(f"(수치 출처: results/E_report.json, E_geometry.csv, E_noise.csv, E_endtoend.csv, E_recovery.csv, E_cost.csv, "
      f"B_energies.csv와 단계 보고 파일. E 작성 {E.get('finished')}, git {(E.get('env') or {}).get('git_head')}; "
      f"요약 작성 {now()}.)")
    w("")
    # 시작 조건
    w("## 시작 조건")
    w("")
    w(f"- results/start_report.json: stop = {st.get('stop')}; 0-1 = {(st.get('0-1') or {}).get('ok')}, "
      f"0-2 = {(st.get('0-2') or {}).get('ok')}, 0-3 = {(st.get('0-3') or {}).get('ok')}.")
    for d in st.get("deviations") or []:                   # EV-14: e.g. the strict reading of 0-3
        w(f"- 이탈: 시작 조건 {d.get('condition')}는 엄격하게 읽으면 STOP 조건이다({d.get('why')}); 결정 "
          f"{d.get('overridden_by_decision')}(추천안)으로 진행했다.")
    xg = sorted(k for k, v in (E.get("representatives") or {}).items() if v.get("rep") == XGB)   # E families x 9 targets
    w(f"- E에서 쓰는 모델 계열(dft·SH, dft·S, g2·SH, g2·S × 9 타깃) 중 XGB 대표 {len(xg)}칸(모델 파일 없음 → E에서 뺌, "
      f"T-4): {', '.join(xg) if xg else '없음'}.")
    w("")
    # S
    w("## S — 표본")
    w("")
    pop, pa, pbb = srep.get("population") or {}, srep.get("part_a") or {}, srep.get("part_b") or {}
    pbc = pop.get("pairs_by_class") or {}
    w(f"- 모집단(P_012의 P_main 기준 para 짝): {fi(pop.get('n_pairs'))}짝(same {fi(pbc.get('same'))}, mixed "
      f"{fi(pbc.get('mixed'))}, different {fi(pbc.get('different'))}), 반응 {fi(pop.get('n_reactions'))}개.")
    pac, req = pa.get("pairs_by_class") or {}, pa.get("requested") or {}
    w(f"- Part A: {fi(pa.get('n_pairs'))}짝(same {fi(pac.get('same'))} / 요청 {fi(req.get('same'))}, different "
      f"{fi(pac.get('different'))} / 요청 {fi(req.get('different'))}), 고유 반응 {fi(pa.get('n_unique_reactions'))}개.")
    w(f"- Part B: {fi(pbb.get('n_reactions'))}반응(요청 {fi(pbb.get('requested'))}), Part A와 겹침 "
      f"{fi(pbb.get('overlap_with_part_a'))}반응.")
    sf = srep.get("shortfalls") or []
    sft = "; ".join("Part {} {} {}: {}/{}".format(x.get("part"), x.get("stratum"), x.get("dipole"), x.get("drawn"),
                                                    x.get("need")) for x in sf)
    w(f"- 모자람(뽑은 수/요청 수): {sft if sf else '없음'}.")
    w("")
    # T
    w("## T — 회귀 검사")
    w("")
    names = {1: "방법 키워드(OptTS + Freq)", 2: "feature 경로(182열·토큰)", 3: "HF EDA 경로", 4: "모델 적용 경로",
             5: "Part B SP 수준"}
    for k in (1, 2, 3, 4, 5):
        tj = Tj.get(k)
        if tj is None:
            w(f"- T-{k} {names[k]}: results/T_{k}.json 없음.")
            continue
        extra = ""
        if k == 4:
            extra = (f"; unit {fi(tj.get('n_units_ok'))}/{fi(tj.get('n_units'))}, 최대 |yhat − L_predictions| "
                     f"{fm(tj.get('max_abs_yhat'), 6)} kcal/mol(허용 1e-3); XGB: {(tj.get('xgb') or {}).get('statement')}")
        if k == 1:
            extra = "; " + ", ".join(f"{x.get('method')} 허수 {x.get('n_imag')}개" for x in tj.get("methods") or [])
        if k == 2 and tj.get("deviation"):                 # EV-14: 1e-8 exceeded, passed on the F-0 precedent
            extra = (f"; 이탈: 182열 최대 차 {fm((tj.get('summary') or {}).get('max_abs_diff_182'), 9)} > 1e-8, "
                     f"scan_curv_* 선례(D1-G2 T-1)로 통과")
        if k == 5:
            diffs = [abs(_f(x.get("diff_kcal")) or 0) for x in (tj.get("sp") or []) + (tj.get("eda") or [])
                     if _f(x.get("diff_kcal")) is not None]
            extra = f"; 방식 {tj.get('mode')}, 최대 |E − 라벨| {fm(max(diffs) if diffs else None, 4)} kcal/mol"
        passed = tj.get("pass") if "pass" in tj else tj.get("all_pass")
        verd = f", verdict {tj.get('verdict')}" if tj.get("verdict") else ""
        w(f"- T-{k} {names[k]}: pass = {passed}{verd}{extra}.")
    w("")
    # O
    w("## O — 최적화")
    w("")
    for m in C.METHODS:
        o = (orep.get("methods") or {}).get(m) or {}
        fr = o.get("fail_reasons") or {}
        w(f"- {m}: TS {fi(o.get('n_ts_ok'))}/{fi(o.get('n_ts'))} 성공({fm(o.get('ts_success'))}), 형태 유지 "
          f"{fm(o.get('shape_kept'))}, 최종 .done {fi(o.get('final_done'))}, 재시도 사용 {fi(o.get('retry_used'))}; "
          f"기준 구조 단위 {fi(o.get('n_refs_ok'))}/{fi(o.get('n_refs'))} 성공; TS 실패 이유 "
          f"{', '.join(f'{k} {v}' for k, v in fr.items()) if fr else '없음'}.")
    ost = orep.get("o_stats") or {}
    w(f"- 파이프라인 오류 {fi(ost.get('n_error'))}/{fi(ost.get('n_finished'))} 단위; 예상 합계 "
      f"{fm(oest.get('total_core_h'), 1)} 코어시간(한도 3,000, results/O_estimate.json), 측정 합계 "
      f"{fm(orep.get('measured_core_h'), 1)} 코어시간(results/O_report.json).")
    ob = orep.get("partB") or {}
    w(f"- Part B: {fi(ob.get('n_ok'))}/{fi(ob.get('n'))} 성공, converged_to_label {fi(ob.get('converged_to_label'))}, "
      f"ΔE‡ 계산 {fi(ob.get('n_dE'))}; T-5 방식 {((orep.get('t5') or {}).get('mode'))}.")
    dv = orep.get("deviations") or {}                      # EV-14: opt.py O-7 / O-9 / O-14 / O-19
    if dv:
        o7, o9, o19 = dv.get("O7") or {}, dv.get("O9") or {}, dv.get("O19") or {}
        w(f"- 이탈(명령서와 다른 추천안): constrained 분자 키 {fi(o7.get('n_keys'))}개를 공유하지 않고 반응마다 계산"
          f"(방법마다 {fi(o7.get('n_units_per_method'))}단위; 4-2); Part B autodE식 재최적화 {fi(o9.get('n_reopt'))}회"
          f"(성공 {fi(o9.get('n_reopt_ok'))}, 허수 모드 사전 검사 없음); 첫 20 STOP 뒤에도 계속 돈 방법 이외 계산: "
          f"멈춘 방법 {', '.join(dv.get('O14', {}).get('stopped_methods') or []) or '없음'}; 벽시간 두 번 잘린 단위 "
          f"{fi(o19.get('n_wall_clipped'))}개(fail).")
    w("")
    # F
    w("## F — 새 구조 위 입력")
    w("")
    for m in C.METHODS:
        fm_ = (frep.get("methods") or {}).get(m) or {}
        pr = fm_.get("pairs") or {}
        fg = ((fm_.get("F") or {}).get("gate") or {})
        hg = ((fm_.get("H") or {}).get("gate_failure") or {})
        w(f"- {m}: 사용 가능 반응 {fi(fm_.get('n_usable'))}/{fi((fm_.get('O') or {}).get('n_part_a'))}; 짝 same "
          f"{fi((pr.get('same') or {}).get('n_usable'))}/{fi((pr.get('same') or {}).get('n_pairs'))}, different "
          f"{fi((pr.get('different') or {}).get('n_usable'))}/{fi((pr.get('different') or {}).get('n_pairs'))}; "
          f"feature gate 실패 {fi(fg.get('n_fail'))}/{fi(fg.get('n_base'))}({fm(fg.get('rate'))}), HF gate 실패 "
          f"{fi(hg.get('n'))}/{fi(hg.get('n_finished'))}({fm(hg.get('rate'))}).")
    pl = E.get("pair_lists") or {}
    w(f"- 공통 짝 목록(다섯 입력 모두 사용 가능; E-2–E-4 나란히 놓는 표·그림): same "
      f"{fi((pl.get('common|same') or {}).get('n'))}, different {fi((pl.get('common|different') or {}).get('n'))}. "
      f"방법별 목록(9절 판정, EV-11): "
      + ", ".join(f"{m} same {fi((pl.get(f'own_{m}|same') or {}).get('n'))} / different "
                  f"{fi((pl.get(f'own_{m}|different') or {}).get('n'))}" for m in C.METHODS) + ".")
    w("")
    # E-1
    w("## E-1 구조 — 라벨 DFT TS와의 차이 (공통 목록)")
    w("")
    for cls in CLASSES:
        q = G[(G.scope == "common") & (G["class"] == cls)]
        n0 = q.n_rxn.max() if len(q) else 0
        np0 = q.n_pairs.max() if len(q) else 0
        w(f"**{cls}** (반응 {fi(n0)}, 짝 {fi(np0)}; Å)")
        w("")
        w("| 입력 | \\|Δd\\| 중앙값 | \\|Δd\\| 90 % | 짝 \\|Δdd\\| 중앙값 | 짝 \\|Δdd\\| 90 % | RMSD 중앙값 | RMSD 90 % | "
          "라벨과 같은 TS |")
        w("|---|---|---|---|---|---|---|---|")
        for lv in LEVELS:
            r = _row(q, level=lv)
            if r is None:
                continue
            w(f"| {lv} | {fm(r.dform_abs_median)} | {fm(r.dform_abs_p90)} | {fm(r.ddd_abs_median)} | {fm(r.ddd_abs_p90)} | "
              f"{fm(r.rmsd_heavy_median)} | {fm(r.rmsd_heavy_p90)} | {fi(r.n_same_ts)}/{fi(r.n_rxn)} "
              f"({fm(r.same_ts_frac)}) |")
        w("")
    w("Δd = 형성 결합 길이(입력) − 라벨 DFT TS(라벨 TS에서 짧은 쪽·긴 쪽 결합, 둘을 모음); Δdd = 짝 단위 차이"
      "(a − b); RMSD = D1 검증 rmsd.py 무거운 원자 매핑 RMSD; 같은 TS = 0.10 Å / 0.02 Å 규칙. dft 행은 정의상 0.")
    w("")
    # E-2
    w("## E-2 짝 구조 잡음 — |ΔΔ_HF(입력) − ΔΔ_HF(DFT 구조)| 중앙값 (kcal/mol, para, 공통 목록)")
    w("")
    lv2 = [x for x in LEVELS if x != "dft"]
    for cls in CLASSES:
        q = N[(N.scope == "common") & (N["class"] == cls)]
        w(f"**{cls}** (짝 {fi(q.n_pairs.max() if len(q) else 0)}; 괄호 = 평균)")
        w("")
        w("| 채널 | " + " | ".join(lv2) + " |")
        w("|---|" + "---|" * len(lv2))
        for t in T9:
            cells = []
            for lv in lv2:
                r = _row(q, level=lv, channel=t)
                cells.append("—" if r is None else f"{fm(r.noise_median)} ({fm(r.noise_mean)})")
            w(f"| {t} | " + " | ".join(cells) + " |")
        w("")
    # E-3
    w("## E-3 처음부터 끝까지 — 학습된 모델에 새 입력 (para MAE [95 % CI], kcal/mol, 공통 목록)")
    w("")
    ins = [x for x in LEVELS if x in set(E3.input)]
    for cls in CLASSES:
        q = E3[(E3.scope == "common") & (E3["class"] == cls)]
        w(f"**{cls}** (짝 {fi(q.n_pairs.max() if len(q) else 0)})")
        w("")
        w("| 모델 | 타깃 | " + " | ".join(ins) + " |")
        w("|---|---|" + "---|" * len(ins))
        plan = [("dft·SH", T9), ("dft·S", T9), ("LINE·dft", REC_TARGETS), ("g2·SH", REC_TARGETS),
                ("g2·S", REC_TARGETS), ("LINE·g2", REC_TARGETS)]
        for fam, tg in plan:
            for t in tg:
                cells, src = [], ""
                for X in ins:
                    r = _row(q[q.model.isin(["REP", "LINE"])], family=fam, target=t, input=X)
                    if r is not None and _f(r.mae) is None and str(r.source) == XGB:
                        ra = _row(q[q.model == "REP_atom"], family=fam, target=t, input=X)
                        cells.append(f"{_ci(ra)} †" if ra is not None else "XGB 대표, 제외")
                        src = f"XGB → {ra.source}†" if ra is not None else "XGB"
                        continue
                    cells.append(_ci(r))
                    if r is not None:
                        src = "LINE" if fam.startswith("LINE") else str(r.source)
                w(f"| {fam} ({src}) | {t} | " + " | ".join(cells) + " |")
        w("")
    w("REP = D1LADDER E2 대표 모델(괄호 = unit 모델); † = 대표가 XGB라 모델 파일이 없어 같은 칸 검증 MAE가 가장 낮은 원자 "
      "모델로 대신한 설명용 값; LINE = 그 fold SH unit의 β × ΔΔ_HF(입력). dft·SH, dft·S가 주 모델, g2 모델은 참고.")
    w("")
    # E-4
    w("## E-4 회복률 — (MAE_g2 − MAE_g3) ÷ (MAE_g2 − MAE_dft), dft·SH 대표")
    w("")
    w("**공통 목록**(다섯 입력 모두 쓸 수 있는 짝, 나란히 비교)")
    w("")
    w("| 분류 | 방법 | 타깃 | 회복률 [95 % CI] | MAE g2 | MAE g3 | MAE dft |")
    w("|---|---|---|---|---|---|---|")
    for cls in CLASSES:
        for m in C.METHODS:
            for t in REC_TARGETS:
                r = _row(E4, scope="common", **{"class": cls}, method=m, target=t)
                if r is None:
                    continue
                w(f"| {cls} | {m} | {t} | {fm(r.recovery)} [{fm(r.recovery_ci_lo)}, {fm(r.recovery_ci_hi)}] | "
                  f"{fm(r.mae_g2)} | {fm(r.mae_g3)} | {fm(r.mae_dft)} |")
    w("")
    w("**방법별 목록**(판정에 쓰는 목록, EV-11: dft, g1, g2와 그 방법이 모두 쓸 수 있는 짝)")
    w("")
    w("| 분류 | 방법 | 타깃 | 짝 | 회복률 [95 % CI] | MAE g2 | MAE g3 | MAE dft |")
    w("|---|---|---|---|---|---|---|---|")
    for cls in CLASSES:
        for m in C.METHODS:
            for t in REC_TARGETS:
                r = _row(E4, scope=f"own_{m}", **{"class": cls}, method=m, target=t)
                if r is None:
                    continue
                w(f"| {cls} | {m} | {t} | {fi(r.n_pairs)} | {fm(r.recovery)} [{fm(r.recovery_ci_lo)}, "
                  f"{fm(r.recovery_ci_hi)}] | {fm(r.mae_g2)} | {fm(r.mae_g3)} | {fm(r.mae_dft)} |")
    w("")
    w("different 분류는 설명용이다(TS 형태가 달라 회복을 기대하지 않음).")
    w("")
    # E-5
    w("## E-5 비용·성공")
    w("")
    w("| 방법 | 분류 | TS 성공률 (ok/(ok+fail)) | 형태 유지율 | TS 코어시간 중앙값 / 평균 | 성공 TS당 | 기준 구조 코어시간"
      f"(반응당 중앙값) | TS 중앙값 ÷ {fm(lab_ts, 1)} |")
    w("|---|---|---|---|---|---|---|---|")
    for m in C.METHODS:
        for cls in CLASSES + ("all",):
            r = _row(E5, method=m, **{"class": cls})
            if r is None:
                continue
            w(f"| {m} | {cls} | {fm(r.ts_success)} ({fi(r.n_ts_ok)}/{fi((_f(r.n_ts_ok) or 0) + (_f(r.n_ts_fail) or 0))}"
              f"; error {fi(r.n_ts_error)}) | {fm(r.shape_kept_rate)} ({fi(r.n_shape_kept)}/{fi(r.n_ts_ok)}) | "
              f"{fm(r.ts_core_h_median, 2)} / {fm(r.ts_core_h_mean, 2)} | {fm(r.ts_core_h_per_success, 2)} | "
              f"{fm(r.ref_core_h_rxn_median, 2)} | {fm(r.ts_core_h_median_over_label)} |")
    w("")
    w(f"라벨 TS 단계 = 파일럿 평균 {fm(lab_ts, 1)} 코어시간/반응(명령서 6 E-5; E_cost.csv label_ts_core_h).")
    w("")
    # E-6
    w("## E-6 Part B — 라벨 수준 ΔE‡ = E(G2 형태 TS) − E(라벨 TS)")
    w("")
    w("| dipole | 반응 | 결과 있음 | converged_to_label | ΔE‡ < −0.5 | \\|ΔE‡\\| ≤ 0.5 | ΔE‡ > 0.5 | ΔE‡ < −0.5 비율 |")
    w("|---|---|---|---|---|---|---|---|")
    for name, g in list((pb.get("by_dipole") or {}).items()) + [("전체", pb["all"])]:
        w(f"| {name} | {fi(g['n'])} | {fi(g['n_resolved'])} | {fi(g['n_converged_to_label'])} | {fi(g['n_lt_m05'])} | "
          f"{fi(g['n_abs_le_05'])} | {fi(g['n_gt_05'])} | {fm(g['frac_lt_m05'])} |")
    w("")
    spo = pb["all"].get("sp_only") or {}
    w(f"SP를 계산한 반응만(converged 제외): {fi(spo.get('n'))}개, ΔE‡ < −0.5 {fi(spo.get('n_lt_m05'))}개, 중앙값 "
      f"{fm(spo.get('dE_median'), 2)}, 범위 {fm(spo.get('dE_min'), 2)}–{fm(spo.get('dE_max'), 2)} kcal/mol. "
      f"결과 없음 {fi(pb['all'].get('n_unresolved'))}반응(fail {fi(pb['all'].get('n_fail'))}, error "
      f"{fi(pb['all'].get('n_error'))}). converged_to_label은 ΔE‡ = 0으로 \\|ΔE‡\\| ≤ 0.5에 넣었다.")
    w("")
    # 판정
    w("## 판정 (명령서 9절, `same` 분류, 방법별 목록)")
    w("")
    w("| 방법 | ① 회복률_oi ≥ 0.8 · oi MAE ≤ 0.20 | ② TS 성공률 ≥ 0.9 · 형태 유지율 ≥ 0.9 | ③ TS 코어시간 중앙값 ≤ 2.0 | 판정 |")
    w("|---|---|---|---|---|")
    ok = {True: "만족", False: "불만족"}
    for m in C.METHODS:
        v = M[m]
        w(f"| {m} | {fm(v['c1']['recovery_oi'])} · {fm(v['c1']['mae_oi_dftSH'])} → {ok[bool(v['c1']['ok'])]} | "
          f"{fm(v['c2']['ts_success'])} · {fm(v['c2']['shape_kept'])} → {ok[bool(v['c2']['ok'])]} | "
          f"{fm(v['c3']['ts_core_h_median'])} → {ok[bool(v['c3']['ok'])]} | {v['verdict_ko']} |")
    w("")
    w(f"- 권고: {s9['recommendation_ko']}")
    w("- `different` 분류의 결과는 설명용이다. TS 형태가 달라 회복을 기대하지 않는다.")
    itp = pb.get("interpretation") or {}
    if itp.get("sentence"):
        w(f"- Part B 해석(설명용; ΔE‡ < −0.5 비율 {fm(itp.get('frac_lt_m05'))}, 기준 0.2): {itp['sentence']}")
    else:
        w("- Part B 해석: 결과가 있는 반응이 없어 9절 문장을 고를 수 없다.")
    w("")
    # 비용
    w("## 비용")
    w("")
    w(f"- Phase O 예상 합계 {fm(oest.get('total_core_h'), 1)} 코어시간(spec 7 한도 3,000), 측정 합계 "
      f"{fm(orep.get('measured_core_h'), 1)} 코어시간.")
    for m in C.METHODS:
        r = _row(E5, method=m, **{"class": "all"})
        fc = ((frep.get("methods") or {}).get(m) or {}).get("core_h") or {}
        g_ = fc.get("g_step") if isinstance(fc.get("g_step"), dict) else {}
        h_ = fc.get("hf") if isinstance(fc.get("hf"), dict) else {}
        if r is not None:
            w(f"- {m}: TS 합 {fm(r.ts_core_h_sum, 1)}, 기준 구조 분자 단위 합 {fm(r.ref_unit_core_h_sum, 1)} 코어시간"
              f"(단위 {fi(r.n_ref_units)}개, 중앙값 {fm(r.ref_unit_core_h_median, 2)}); TS 하나당 중앙값 "
              f"{fm(r.ts_core_h_median, 2)} = 라벨 TS 단계 {fm(lab_ts, 1)}의 {fm(r.ts_core_h_median_over_label)}배; "
              f"Phase F xTB Freq {fm(g_.get('sum'), 2)}, HF EDA {fm(h_.get('sum'), 1)} 코어시간.")
    rb = _row(E5, method="partB")
    if rb is not None:
        w(f"- Part B: OptTS 합 {fm(rb.ts_core_h_sum, 1)}(중앙값 {fm(rb.ts_core_h_median, 2)}), SP 합 "
          f"{fm(rb.ref_core_h_rxn_sum, 1)} 코어시간.")
    w("")
    # 한계
    w("## 한계")
    w("")
    for k in ("10-lim1", "10-lim2", "10-lim3", "10-lim4"):
        w(f"- {SPEC_TEXT[k]}")
    w(f"- XGB 대표 칸은 모델 파일이 없어 E에서 뺐다(T-4); 표의 † 값은 설명용 대체 모델이다.")
    w("- 판정은 방법마다 그 방법이 쓸 수 있는 짝(방법별 목록, E_*.csv scope own_<방법>)에서 했다. 두 방법의 짝 목록이 "
      "달라 두 방법의 판정 수치는 같은 짝 위의 비교가 아니다(같은 짝 비교는 공통 목록 표·그림).")
    w(f"- E-1의 g3 구조는 Phase O .done 폴더만 썼다(TS는 성공했지만 기준 구조가 실패한 반응은 빠짐). Part B SP는 T-5 방식 "
      f"'{((orep.get('t5') or {}).get('mode'))}'으로 계산했고 라벨 TS 에너지는 라벨 e_ab_eh다.")
    w("- 회복률 CI는 같은 core 재표집의 비율 백분위로 설명용이다(분모가 0에 가까우면 불안정).")
    w("")
    w("결정 기록: results/DECISIONS.md (EV-0 – EV-14, R-1), 판정 규칙은 결과를 보기 전에 고정된 명령서 9절 그대로.")
    C.write_atomic(SUMMARY, "\n".join(L_) + "\n")
    record_decision("R-1", "SUMMARY_G3.md: 맨 앞 세 문장 = (1) 두 방법 판정과 근거 수치(same, 방법별 목록), (2) Part B "
                             "ΔE‡ < −0.5 비율(결과 있는 반응 기준, converged는 0), (3) 권고; 절 순서 = 명령서 10절; 수치는 "
                             "E_report.json·E_*.csv·B_energies.csv·단계 보고 파일에서만(반올림 표시).")
    print(f"[R] {SUMMARY} written ({len(L_)} lines)", flush=True)
    return 0


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK") or 1))
    ap.add_argument("--no-figures", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        if a.cmd == "run":
            return cmd_run(max(1, a.workers), not a.no_figures)
        if a.cmd == "summary":
            return cmd_summary()
    except NotReady as e:
        print(f"NOT READY: {e}", file=sys.stderr, flush=True)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
