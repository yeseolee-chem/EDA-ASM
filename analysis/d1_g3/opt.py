#!/usr/bin/env python3
r"""D1-G3 Phase O (docs/specs/D1G3.md 4, 7, 12 O) and the Phase T tests T-1 / T-5 (spec 3). sbatch only (python, ORCA,
xtb never on the login node). Nothing in this file was run when it was written (login node): every path is untested
until it runs inside a job.

What is computed
  Part A TS (spec 4-1), per reaction x method, ORCA 6.1.1 from $G2/g2_final/<rxn>/ts.xyz (label atom order), charge =
       label charge1 + charge2, multiplicity 1:
         g3h  ! HF D3BJ 6-31G* TightSCF OptTS Freq CPCM(water)
         g3r  ! r2SCAN-3c TightSCF OptTS Freq CPCM(water)
       + '%geom Calc_Hess true Recalc_Hess 5 MaxIter 200 end' (the G1 / G2 block) + the d1_run config_run.yaml
       sp_stage.frag_solv_block ('%cpcm smd true smdsolvent "water" end') + '%maxcore 3000' + '%pal nprocs N end'.
       TS gate (O-4) -> on failure ONE more run from the same G2 TS with '%geom Calc_Hess true Recalc_Hess 1 MaxIter 300
       end' -> still failing = fail with the reason. Shape kept = same TS as the starting G2 TS (O-6).
  Part A references (spec 4-2): one calculation per molecule unit (Phase Q key rule, O-7), same method '! ... Opt Freq
       CPCM(water)' + the %cpcm block, started from the G2 rel1.xyz / rel2.xyz of the unit's start reaction, no
       imaginary frequency (O-8).
  Part B (spec 4-3): the D1 autodE OptTS input (O-9) from the G2 TS (+ autodE's one '_reopt' from the last geometry
       when OptTS did not converge and the last energy change <= 0.1 kcal/mol), the D1 TS gate (O-10), converged_to_label = same
       TS as the label TS ($D1_JOBS/<T>/ts.xyz); otherwise the label-level SP of the Part B TS in the T-5 mode and
       dE = (E_SP(G2-form TS) - e_ab_eh(label TS)) x 627.5094740631 kcal/mol (O-11).
  T-1  the smallest Part A reaction, g3h and g3r TS units (the same units Part A uses): normal termination, OptTS
       converged, exactly 1 imaginary frequency, Epsilon 78.3550 + the SMD CDS line, the method / basis words on the
       keyword line and the method / basis ORCA reports outside the input echo (O-21: g3h Hartree-Fock(GTOs), 6-31G*,
       4 D3 parameter lines; g3r DFT(GTOs), def2-mTZVPP, r2SCAN), ORCA 6.1.1 (GIT 487d211c) -> results/T_1.json.
  T-5  3 label TSs (the 3 smallest Part B reactions) recomputed at the 4-3 SP level (mode 'sp', O-11): |E - label
       e_ab_eh| <= 0.01 kcal/mol for all 3; else the label eda.inp with only the coordinates replaced (mode 'eda') is
       checked the same way -> results/T_5.json (mode used by Part B). Both fail = STOP.
  cost (spec 7): pilot = T-1 + 5 Part A reactions per method (+ their references) + 2 Part B reactions; estimate of
       all of section 7 (O-13); <= 3,000 core-h -> the main array runs (dependency chain), else STOP.
       O-23 (user 2026-10-24): the estimate and the 3,000 core-h limit are dropped; `estimate` is a no-op and main runs
       once check + T-1 .. T-5 have passed and there is no O STOP file (it may run alongside the pilot array).

CLI (exit 0 = done / up to date, 2 = incomplete (resubmit) or usage, 3 = STOP; submit from analysis/d1_g3)
  plan                  G3ENV=reactot (pyarrow for Q_keys.parquet): $G3/O/plan.json + results/O_plan.csv (units,
                        molecule units, T-1 / pilot / T-5 choices); refused (STOP) when the inputs changed after units ran
  check                 G3ENV=d1: D1 config / thresholds / autodE keywords / Part B text proof / MPI env / inputs ->
                        $G3/O/check.json
  tests                 G3ENV=d1: T-1 + T-5 (needs check ok) -> results/T_1.json, results/T_5.json
  work <pilot|main>     G3ENV=d1, array: claims units (pilot: T-1 + pilot units; main: every unit) and runs
                        $SLURM_CPUS_PER_TASK // 8 ORCA calculations at a time (O-23: main needs no estimate)
  work sel --only <f>   G3ENV=d1, U-1 (user 2026-10-25): only the plan units listed in <f> (one uid per line, '#' lines
                        are comments), e.g. results/O_sel_partB.txt, results/O_sel_refs.txt
  estimate              O-23: no-op (exit 0, writes nothing); before O-23: results/O_estimate.json (spec 7)
  select                G3ENV=d1, U-1a (after the ts.g3h / ts.g3r work is cancelled): results/U1_selection.json = the
                        S_pairs.csv pairs whose ts.g3r.<a_rxn> and ts.g3r.<b_rxn> both finished 'ok', their reactions
                        and the ref.g3r units they need, + results/O_sel_refs.txt (the needed ref.g3r units without
                        unit.json now, for 'work sel --only'); exit 2 = a g3r TS unit is still held by a live job, 3 = no
                        complete pair. A rerun keeps U1_selection.json (and its sha256) while the selection is the same
  assemble [--force]    G3ENV=d1: $G3/g3h_final/<rxn>/, $G3/g3r_final/<rxn>/, $G3/partB/<rxn>/, results/O_status.csv,
                        results/O_refs.csv, results/O_report.json (exit 3 = O STOP, 2 = units missing). U-1b: reads
                        results/U1_selection.json (missing / stale = exit 2); (method, rxn) outside it -> marker
                        .fail_stopped_user only, O_status status 'stopped_user'; only the selected g3r reactions and
                        Part B must be complete
                        U-1 order: select ::: work sel --only results/O_sel_refs.txt (one 32-CPU job), then assemble
                        after it and after the Part B 'work sel' job
  status                counts per phase / kind / status and the O STOP statistics (writes nothing)
  one <uid>             run one unit (debugging; the same claim / lock / idempotency rules)
  options: --nprocs N   MPI ranks of the Part A ORCA runs (default 8; slots per task = cpus // 8 because the Part B
                        and label SP inputs keep the D1 %pal 8)

Outputs
  $G3/O/units/<uid>/unit.json      one per calculation unit, written last (a unit with unit.json is never rerun);
        uid = ts.<m>.<rxn> | ref.<m>.<mid> | B.<rxn> | T5.<job>.<sp|eda>; per ORCA step <uid>/<step>/: <name>.inp,
        <name>.out.gz, <name>.err.gz (if not empty), <name>.xyz (final geometry), <name>.hess.gz, step.json (written last:
        a finished step is reused by a rerun, e.g. attempt 1 after a wall-clipped attempt 2). No .gbw / .densities /
        .tmp is ever copied back (node-local $TMPDIR work folders, quota).
  $G3/O/locks/<uid>.lock           O_EXCL claim with the SLURM job id (a lock of a job no longer in squeue is taken over)
  $G3/O/STOP, $G3/O/STOP_<m>.json  O STOP (all units / the units of method m are no longer claimed)
  $G3/<m>_final/<rxn>/             G1 folder format: ts.xyz, rel1.xyz (dipole), rel2.xyz (dipolarophile), result.json,
        .done | .fail_<reason>; label atom order (TS = G2 order = label order; references in the reaction's own G2 rel
        order). result.json: rxn_id, job_id, method, level, ts_header, charge1/2, formed_pairs_label, d_form_<m>,
        d_form_g2, e_ts_eh, e_rel1_eh, e_rel2_eh, barrier_<m>_kcal, opt_imag_freqs_cm, opt_n_imag,
        opt_imag_mode_forming_share, gate, shape_kept, rmsd_g2_A, dform_g2_A, ref1 / ref2, core_h, core_h_total, status,
        fail_reasons, units. imag_freqs_cm / imag_mode_forming_share / work/ts.hess are NOT written here (O-16): Phase F
        adds them from its XTB2 ALPB(water) Freq at ts.xyz (g0f0 G0 / g2_autode.finalize rule) before the features run.
  $G3/partB/<rxn>/                 ts.xyz (Part B TS, label order, when one exists), result.json, .done | .fail_<reason>
  results/O_plan.csv, O_estimate.json, O_status.csv (rxn_id, job_id, method, status, reason, imag_freqs_cm, n_imag,
        shape_kept, core_h, ... ; method partB rows carry converged_to_label, dE_kcal), O_refs.csv, O_report.json,
        T_1.json, T_5.json; decisions in results/DECISIONS.md (g3_common.record_decision) and O_report.json.

Reused, imported read only (never copied; module identity checked at start):
  D1 run code copy $D1_REPO = /gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM: run_ts.analyse / imag_gate /
  forming_share (the D1 TS gate, as D1-G2 used it), orca_direct (level_header / write_inp / parse_hess / imag_and_mode /
  level_checks / terminated / opt_converged / last_fspe / run_hours / orca_bin), pilot_common (read_xyz, write_xyz,
  import_repo_modules, configure_autode, hmethod_keywords_summary), fragmenter (D1 copy, Si 1.11 A), D1 validation
  analysis/rmsd.py same_ts (the c_compare.py rule); D1LADDER refs_q.load_keys (Q keys), refs_v.coord_only_check /
  hf_prep / hf_run / _job_alive; D1-HF prep.replace_coords, run.scf_good, hf_common (parse_inp, regexes, read_text);
  D1-G2 g2_common (PILOT, MANIFEST, MANIFEST_SHA), g2_autode.remap with g2_refinalize._graph_fixed (the known
  g2_autode._graph bond_matrix bug, fixed in that process only). Small helpers that exist only inline in other files are
  marked with their source (g1_geom.py hess $atoms / last-frequency-block check).

sbatch (login node: one-shot commands only; before each batch `squeue -u $USER -h -r | wc -l` must leave room for its
tasks (MaxSubmit 20, array elements count); -p cpu1,cpu2 lets SLURM take whichever starts first). JS = the Phase S job
(sample.py), JT2F / JT3 / JT4 = the feat.py t2_f / feat.py t3 / apply.py t4 jobs (Phase T, O-20):
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_g3; L=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs
  JP=$(sbatch --parsable -p cpu1,cpu2 -J g3o_plan --time=48:00:00 --cpus-per-task=1 --mem=8G \
       --export=ALL,G3ENV=reactot --dependency=afterok:$JS --kill-on-invalid-dep=yes jobs/g3.slurm opt.py plan)
  JT=$(sbatch --parsable -p cpu1,cpu2 -J g3o_test --time=48:00:00 --cpus-per-task=40 --mem=160G \
       --export=ALL,G3ENV=d1 --dependency=afterok:$JP --kill-on-invalid-dep=yes jobs/g3.slurm opt.py check ::: opt.py tests)
  JW=$(sbatch --parsable -p cpu1,cpu2 -J g3o_pilot --time=48:00:00 --cpus-per-task=32 --mem=128G --array=0-3 \
       --output=$L/%x_%A_%a.out --export=ALL,G3ENV=d1 --dependency=afterok:$JT:$JT2F:$JT3:$JT4 \
       --kill-on-invalid-dep=yes jobs/g3.slurm opt.py work pilot)
  (O-23: no JE estimate job; JM waits only for JT and may run alongside JW; JA waits for both work arrays)
  JM=$(sbatch --parsable -p cpu1,cpu2 -J g3o_main --time=48:00:00 --cpus-per-task=32 --mem=128G --array=0-9 \
       --output=$L/%x_%A_%a.out --export=ALL,G3ENV=d1 --dependency=afterok:$JT --kill-on-invalid-dep=yes \
       jobs/g3.slurm opt.py work main)
  JA=$(sbatch --parsable -p cpu1,cpu2 -J g3o_asm --time=48:00:00 --cpus-per-task=2 --mem=16G \
       --export=ALL,G3ENV=d1 --dependency=afterany:$JM:$JW jobs/g3.slurm opt.py assemble)
  Resubmission (all idempotent): a work array that hit the wall or exited 2 -> the same JM line without the
  dependency (finished units are skipped, dead jobs' locks are taken over; a step clipped twice closes its unit as
  'wall_clipped', O-19), then JA again. work exit 3 = STOP (an O STOP file, or a failed T). assemble exit 3 = O STOP
  (results/O_report.json stop_reasons) or a failed T, exit 2 = units missing.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # nothing compiled into the read-only code trees (D1 copy, d1_ladder, d1_hf, d1_g2)

import argparse  # noqa: E402
import gzip  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g3_common as C  # noqa: E402                     (paths, METHODS, write_atomic / write_json, record_decision)

DP, L = C.DP, C.L
sys.path.append(str(C.REPO / "analysis" / "d1_g2"))
import g2_common as G2C  # noqa: E402                   (PILOT, D1_REPO, MANIFEST, MANIFEST_SHA; constants only)

CODE_VERSION = "d1-g3-o1"
O = C.G3 / "O"
UNITS, LOCKS = O / "units", O / "locks"
PLAN, CHECK, STOP_FILE = O / "plan.json", O / "check.json", O / "STOP"
FINAL = {m: C.G3 / f"{m}_final" for m in C.METHODS}
PARTB = C.G3 / "partB"
S_PAIRS, S_RXNS = C.RES / "S_pairs.csv", C.RES / "S_rxns.csv"
RES_PLAN, T1_JSON, T5_JSON = C.RES / "O_plan.csv", C.RES / "T_1.json", C.RES / "T_5.json"
EST_RETRO = C.RES / "O_estimate_retro.json"                       # U-3a
EST_JSON, STATUS_CSV, REFS_CSV, REPORT = (C.RES / "O_estimate.json", C.RES / "O_status.csv", C.RES / "O_refs.csv",
                                          C.RES / "O_report.json")
G2_FINAL = C.G2 / "g2_final"
D1_JOBS = DP.D1_JOBS
PILOT = G2C.PILOT
VAL = G2C.D1_REPO / "D1_build" / "validation"
FRAG_D1 = G2C.D1_REPO / "analysis" / "espley_xtb_repro" / "fragmenter.py"
CFG_RUN = DP.D1_RUN / "config_run.yaml"
CFG_VAL = VAL / "config_val.yaml"
KW_RECORD = DP.UBAI_JOBS / "T01355" / "ts_result.json"         # a D1 run ts_result.json ('keywords' of autodE)
TEXT_SOURCES = (                                                 # D1 autodE OptTS inputs for the Part B text proof
    dict(name="D1 run T01612 (autodE hl_ad OptTS input, ts.tar checkpoint of the D1 run)", d1_run=True,
         tar=DP.UBAI_JOBS / "T01612" / "ts.tar",
         member="ts/T01612/transition_states/TS_tAq-ov_hl_ad_8-14_9-10_optts_orca.inp"),
    dict(name="pilot J10 (config_val.yaml val.pilot_optts_inp = the source of orca_direct.l0_header)", d1_run=False,
         path="val.pilot_optts_inp"),
)
G0_REPORT, F0_ESTIMATE = L.RES / "G0_report.json", L.RES / "F0_estimate.json"
HF_PMETRICS = C.HF_DIR / "results" / "P_metrics.json"

# ---------------------------------------------------------------- method (spec 4)
METHOD_KW = {"g3h": "HF D3BJ 6-31G* TightSCF", "g3r": "r2SCAN-3c TightSCF"}
METHOD_LEVEL = {"g3h": "HF-D3BJ/6-31G* SMD(water)", "g3r": "r2SCAN-3c SMD(water)"}
METHOD_MARK = {"g3h": ("HF", "6-31G*"), "g3r": ("r2SCAN-3c",)}
SOLV_WORD = "CPCM(water)"                                       # O-2
GEOM_BLOCK = {1: "%geom Calc_Hess true Recalc_Hess 5 MaxIter 200 end",
              2: "%geom Calc_Hess true Recalc_Hess 1 MaxIter 300 end"}
D1_OPTTS_BLOCK = "\n%geom\nCalc_Hess true\nRecalc_Hess 30\nTrust -0.1\nMaxIter 100\nend"
SOLV_BLOCK = '%cpcm\n  smd true\n  smdsolvent "water"\nend\n\n'   # = config_run.yaml sp_stage.frag_solv_block (checked)
MAXCORE = 3000                                                  # config_run.yaml maxcore_mb (checked)
SLOT_CORES = 8                                                  # D1 %pal of the Part B / label SP inputs
PARTB_NPROCS = 8
T5_TOL = 0.01                                                   # kcal/mol (spec T-5)
EPS_SMD_WATER = "78.3550"
CDS_MARK = "SMD CDS free energy correction energy"
SMD_MARK = "utilizes the SMD solvation module"
CORE_H_MAX = 3000.0                                             # spec 7
ERR_MAX = 0.02                                                  # spec 12 O
FIRST_N, FIRST_MIN = 20, 0.5                                    # spec 12 O
DISP_FACTOR = 1.0                                               # O-8 (refs_v V10)
BOHR = 0.529177210903
PILOT_Q = (0.1, 0.3, 0.5, 0.7, 0.9)
PILOT_B_Q = (1 / 3, 2 / 3)
PART_B_COLS = ("part_b", "in_part_b", "partB", "in_partB", "in_B", "is_B", "sample_b", "B")
PART_B_TEXT_COLS = ("roles", "role", "sample", "samples", "part", "parts")
RE_OPTCYC = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+\d+")
RE_OUTFREQ = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", re.M)
T0 = float(os.environ.get("D1G3_JOB_T0", time.time()))
STOP_AFTER_H = float(os.environ.get("D1G3_STOP_AFTER_H", "44"))
BUDGET_GB = float(os.environ.get("D1G3_O_BUDGET_GB", "3"))
T_JSON = {1: T1_JSON, 2: C.RES / "T_2.json", 3: C.RES / "T_3.json", 4: C.RES / "T_4.json", 5: T5_JSON}   # O-20
REOPT_DE_KCAL = 0.1                                             # O-9: autodE TransitionState._reoptimise
MAX_STARTS = 2                                                  # O-19: unfinished starts of one ORCA step
RE_ECHO = re.compile(r"^\|\s*\d+>.*$", re.M)                    # O-21: the input echo lines of an ORCA output
LEVEL_WANT = {"g3h": dict(method="Hartree-Fock(GTOs)", basis=["6-31G*"], d3bj=True, functional=None),
              "g3r": dict(method="DFT(GTOs)", basis=["def2-mTZVPP"], d3bj=False, functional="r2scan")}

DECISIONS = {
    "O-1": "환경: plan = G3ENV=reactot (Q_keys.parquet에 pyarrow 필요; D1 venv에는 없음), 나머지(check, tests, work, estimate, "
           "assemble) = G3ENV=d1 (D1 run env: autodE 1.4.5, ORCA 6.1.1, OpenMPI 4.1.8f, env.sh의 MPI 변수).",
    "O-2": "SMD 적용: 명령서 4-1의 '%cpcm smd true smdsolvent \"water\" end'에 더해 '!' 줄에 CPCM(water)를 둔다(라벨 단독 SP의 "
           "규약 = d1_run config_run.yaml sp_stage frag_level + frag_solv_block, autodE의 CPCM(Water) + %cpcm smd). %cpcm 블록만으로 "
           "용매 모델이 켜지는지에 기대지 않기 위함; T-1이 Epsilon 78.3550과 SMD CDS 줄로 확인한다.",
    "O-3": "Part A의 %maxcore 3000 (config_run.yaml maxcore_mb, D1 TS 단계 값), %pal nprocs = --nprocs (기본 8); 명령서에 값이 없다.",
    "O-4": "Part A TS gate = run_ts.analyse (D1 코드 사본; D1·G2가 쓴 gate: 허수 진동수 규칙(min -40, 추가 모드 > -50 cm-1), 형성 "
           "결합 범위(not_product_like, forming_bond_present), foreign bond, 허수 모드의 형성 결합 비중 >= 0.5, partition) + G1 규칙 "
           "(g1_geom.process): partition = 라벨 A_idx, .hess가 최종 구조의 것. 반응물 그래프 = G2 rel1/rel2. gate 기준값 = "
           "config_run.yaml gates.",
    "O-5": "재시도: 1차가 어떤 이유로든 실패(ORCA 비정상, 미수렴, gate)하면 같은 G2 TS에서 Recalc_Hess 1 MaxIter 300으로 1회. "
           "인프라(첫 SCF 반복 전에 ORCA가 죽음: 빈 출력, 시그널, orca_util error termination)는 같은 입력을 1회 더(D1 "
           "retry_infra_once); 두 번째도 인프라면 파이프라인 오류.",
    "O-6": "형태 유지 = rmsd.same_ts(G3 TS, G2 TS), D1 run 기준 run.same_ts 0.10 Å / 0.02 Å (= config_val prereg = same_ts 기본값, "
           "check에서 확인), 형성 거리는 라벨 formed_pairs_ts에서. 실패한 TS도 구조가 있으면 기록.",
    "O-7": "기준 구조 분자 단위: Q_keys.parquet의 라벨 행 key. constrained 쌍극자 key(Q constrained true)는 공유하지 않고 반응마다 "
           "따로 계산(D1LADDER P20 (c): DFT에서 온 '|fixed:' 상태가 G3 값을 정하지 않게). 공유 key의 시작 구조 = job_id가 가장 "
           "작은 구성원의 G2 rel (Q 동률 규칙); 결과는 각 구성원의 G2 rel 원자 순서로 g2_autode.remap(가장 낮은 RMSD 그래프 "
           "동형)으로 옮긴다(시작 구성원은 ORCA 순서 그대로). constrained key를 반응마다 따로 계산하는 것은 명령서 4-2 문구"
           "('같은 분자 키는 한 번만')에서 의도적으로 벗어난 것이다(Phase F F-7b와 같은 규칙); 그 단위 수와 코어시간은 "
           "O_report.json deviations.O7에 적는다.",
    "O-8": "기준 구조 허수: D1LADDER refs_v V10 규칙 — 진동수 < 0이 하나라도 있으면 가장 음수인 모드를 따라 1.0 Å 변위 후 Opt Freq "
           "1회 더, 그래도 < 0이면 fail 'imag'; Opt 미수렴·ORCA 오류 = fail; 무거운 원자 그래프는 시작 구조와 같아야 한다(G1 "
           "refN_graph_same).",
    "O-9": "Part B 입력 = orca_direct.level_header(config_val, 'L0', 라벨 형성 결합) (= D1 autodE OptTS 머리, prereg "
           "l0_expected_header로 검증됨) + autodE 좌표 줄; check에서 D1 run의 실제 autodE OptTS 입력(T01612 ts.tar)과 pilot J10 "
           "입력에 대해 바이트 단위 동일함을 증명하고, configure_autode가 만드는 opt_ts·hess 키워드 문자열 = D1 ts_result.json "
           "기록(T01355)을 확인한다. ORCA를 직접(OptTS Freq; D1 검증 V2b D0_reopt_L0와 같은 방식) 돌린다. D1 run의 autodE 1.4.5 "
           "TransitionState.optimise 후속 규칙 중 효과가 있는 것은 _reoptimise 하나다: OptTS가 수렴하지 않았고 마지막 두 "
           "FINAL SINGLE POINT ENERGY 차가 <= 0.1 kcal/mol이면 마지막 구조에서 같은 입력으로 OptTS를 1회 더('_reopt'). "
           "여기서도 그 규칙을 그대로 1회 쓴다(optts_reopt). autodE의 사전 검사 could_have_correct_imag_mode(별도 Hessian, "
           "-40 cm-1, 활성 결합 변위)는 하지 않고 재최적화 결과에 D1 TS gate를 그대로 적용한다(이탈, 보고: O_report "
           "deviations.O9). autodE의 '_dis'/'_dis2'(둘째 허수 모드 변위)는 1.4.5에서 변위 TS가 아니라 원래 TS의 허수 진동수를 "
           "검사하므로 결과 TS를 바꾸지 못한다(transition_state.py optimise) -> 재현하지 않음(D1 gate는 그 TS를 extra_imag_small로 "
           "떨어뜨림). 인프라는 1회 재실행.",
    "O-10": "Part B gate = run_ts.analyse (D1 gates), 반응물 = 라벨 기준 구조; partition의 라벨 일치는 기록만. converged_to_label = "
            "rmsd.same_ts(Part B TS, 라벨 TS).",
    "O-11": "SP 수준(T-5): 1차 = 라벨 eda.inp에서 EDA 단어와 %eda 블록만 빼고 좌표를 prep.replace_coords로 바꾼 일반 SP(조각 "
            "표지 유지, 'sp'); 라벨 TS 3개 중 하나라도 e_ab_eh와 > 0.01 kcal/mol이면 라벨 eda.inp에서 좌표만 바꾼 입력('eda')으로 "
            "재검사; Part B는 통과한 방식을 쓴다. E(라벨 TS) = 라벨 e_ab_eh (T-5가 같은 SP로 재현됨을 보임); dE = (E_SP(Part B TS) "
            "- e_ab_eh) x 627.5094740631. SP는 재시도 없음(라벨 입력 그대로), 인프라만 1회.",
    "O-12": "T-1 = 가장 작은 Part A 반응(n_atoms, 다음 rxn_id); pilot = 나머지 Part A 반응의 크기 순서 10/30/50/70/90 % 위치 5개, "
            "두 방법 모두(+ 그 반응들의 기준 구조); Part B pilot = Part B 크기 순서 1/3, 2/3 위치 2개; T-5 = 가장 작은 Part B 3개.",
    "O-13": "비용 예상(명령서 7) = D1LADDER V15 규칙: 계열마다 max(pilot 평균 코어시간 x N, pilot의 n_atoms^3 비례), 계열 = 방법별 "
            "TS (T-1 + pilot), 방법별 기준 구조, Part B OptTS, Part B SP(40개 모두 필요하다고 가정), Phase F HF(D1-HF P_metrics "
            "적합식 x 2 방법) + xTB Freq·feature(G0/F0 반응당 코어시간), + T-5 사용분. 코어시간 = ORCA 실행마다 wall x nprocs. "
            "> 3,000 = STOP.",
    "O-14": "O STOP: 파이프라인 오류(인프라 2회, 예상 밖 예외, 입력 불일치) > 2 % (작업 중에는 계획된 전체 단위의 2 %를 넘는 순간 "
            "새 단위를 받지 않음, 최종 판정은 assemble); 한 방법의 첫 20 TS 단위(끝난 순서) 성공률 < 0.5 -> STOP_<방법>: 그 방법의 "
            "단위만 더 받지 않고(다른 방법과 Part B는 계속) assemble이 exit 3으로 보고. 다른 방법·Part B를 계속 도는 것은 명령서 "
            "12절 '보고 후 대기'에서 벗어난 추천안이다(그 방법의 문제가 다른 방법의 계산을 바꾸지 않고, 대기 중에도 사용자 "
            "결정에 필요한 결과가 쌓인다); assemble이 STOP을 보고하고 Phase F(feat.py)는 O_report stop이면 시작하지 않는다.",
    "O-19": "벽시간 상한: ORCA 단계마다 시작 기록(starts.json)을 남긴다. 끝나지 않은 채 2번 시작된 단계(48 h 벽에 두 번 "
            "잘림)는 다시 돌리지 않고 그 단위를 fail 'wall_clipped'로 닫는다(CLAUDE.md: 잘린 칸 하나를 잃는 것은 허용; 파이프라인 "
            "오류가 아니라 방법 비용 실패로 센다).",
    "O-20": "Phase T 순서(명령서 3절 '계산 전', 12 T): work pilot / main은 T-1·T-5(이 파일)와 T-2·T-3(feat.py, pass)·T-4"
            "(apply.py, all_pass)가 모두 통과했을 때만 돈다(없음·실행 중 = exit 2, 실패 = exit 3). estimate·assemble도 T가 "
            "통과하지 않았으면 아무것도 쓰지 않고 끝난다(멈춘 O의 결과 파일이 생기지 않게). Part B도 같은 문을 쓴다(T-5가 "
            "Part B SP 검사이고 12 T STOP은 전체에 걸림).",
    "O-21": "T-1 방법·기저 확인(명령서 T-1 '출력에 방법·기저가 찍힘'): 입력 반향 줄(|  n>)이 아닌 출력에서 g3h = Hamiltonian "
            "Method 'Hartree-Fock(GTOs)', 'Your calculation utilizes the basis: 6-31G*', D3 매개변수 4줄(s6, a1, s8, a2; "
            "orca_direct.level_checks d3 = D1 L0 규칙); g3r = Method 'DFT(GTOs)', 기저 'def2-mTZVPP'(r2SCAN-3c 복합 기저), "
            "범함수 이름 r2SCAN이 출력 본문에 있음.",
    "O-22": "T-1 기저 확인 완화(2026-10-24, T-1이 g3r에서 이것 하나로 실패한 뒤; 고치기 전에 사용자에게 보고했고, 사용자 상시 지시('모든 분기마다 너의 추천대로 진행')에 따라 답을 기다리지 않고 추천안으로 고침): 출력의 기저 줄 목록이 [\"def2-mTZVPP\"]와 정확히 같아야 했으나 r2SCAN-3c는 보조 기저 def2-mTZVPP/J도 찍는다(계산 자체는 정상 종료, 허수 1개 -393 cm-1, SMD water eps 78.355, DFT(GTOs) + r2SCAN, TS gate ok). 추천안: 첫 기저 줄(궤도 기저)이 기대 기저와 같고 기대 기저가 모두 찍혀 있으면 통과, 보조 기저 줄은 허용. 다른 T-1 기준은 그대로.",
    "O-23": "비용 예상과 3,000 코어시간 한도 생략(사용자 지시 2026-10-24: \"비용추정도하지말고 3000core-h도 필요없음 그냥 본계산 10개 넣어줘\"): estimate를 돌리지 않고(job 1044263 취소), work main은 check + T-1~T-5 통과 + STOP 파일 없음만 확인한다(O_estimate.json go 조건 삭제). 이미 돌던 pilot 4 task(1044262)는 main 단위의 일부라 그대로 둔다(결과는 main이 다시 계산하지 않음); assemble(1044265)은 pilot과 main이 모두 끝난 뒤(afterany 1044262, 1044264). estimate 명령은 아무것도 쓰지 않는 no-op(O STOP을 만들지 않음). O-13의 \"> 3,000 = STOP\"은 적용하지 않는다.",
    "O-15": "작업 배치: 단위별 O_EXCL lock(SLURM job id; squeue에 없는 job의 lock은 인수), 32 CPU task에서 8 rank ORCA 4개 동시, "
            "OMPI_MCA_hwloc_base_binding_policy=none (여러 ORCA가 한 노드를 나눌 때의 d1_pair g_env.sh / r4_env.sh 설정; env.sh의 "
            "OpenMPI 4.1.8f는 XPMEM 없음), 44 h 뒤에는 새 단위를 받지 않음; pilot 4 task, main 10 task; node-local $TMPDIR; "
            "ORCA 실행마다 .inp, .out.gz, .err.gz, 최종 .xyz, .hess.gz만 보관(예산 D1G3_O_BUDGET_GB 기본 3 GB, 넘으면 STOP).",
    "O-16": "최종 폴더 result.json: 방법 수준 진동수는 opt_imag_freqs_cm / opt_n_imag / opt_imag_mode_forming_share로 둔다. "
            "imag_freqs_cm / imag_mode_forming_share / work/ts.hess는 Phase F가 ts.xyz에서 XTB2 ALPB(water) Freq로 만든다(g0f0 G0 / "
            "g2_autode.finalize 규칙) — ext_features.block_b6가 xTB Hessian 옆에서 방법 수준 값을 읽지 않게.",
    "O-17": "S 입력: Part A 반응 = S_pairs a_rxn ∪ b_rxn (S_rxns part_a와 일치 확인); Part B = S_rxns.csv의 part_b 열(없으면 다른 "
            "이름 / roles 열의 partB 표지), 없으면 STOP.",
    "O-18": "Part A 최종 폴더: TS ok이고 두 기준 구조 ok일 때만 .done; 아니면 첫 이유로 .fail_<ts_|ref1_|ref2_|pipeline_...>. xyz는 "
            ".done 폴더에만 쓴다(G2 finalize와 같음; 실패 구조는 units/에 남음).",
    "U-1a": "U-1 선택(사용자 결정 2026-10-25 1·5: ts.g3h·ts.g3r 신규 제출 중단, 완료분 보존, g3r 완료 반응으로 완성되는 P_012 para "
            "짝만 F·E 진행(설명용)): 'opt.py select'(G3ENV=d1)가 results/S_pairs.csv(P_012 para 짝의 Part A 표본) 중 "
            "ts.g3r.<a_rxn>와 ts.g3r.<b_rxn>의 unit.json status가 둘 다 'ok'인 짝 = 완성 짝을 pair_id 순으로 고른다(분류 same·"
            "different 모두; 한쪽 TS가 fail·error·미실행이면 완성 짝이 아님). 선택 반응 = 완성 짝의 a·b 반응, 방법은 g3r만(g3h는 "
            "선택 없음). 필요한 기준 구조 = 선택 반응마다 plan rxns의 mid1·mid2의 ref.g3r 단위(assemble_a가 읽는 연결 그대로); 그중 "
            "unit.json이 없는 것 = missing_refs -> results/O_sel_refs.txt('#' 머리 + 한 줄에 uid 하나)를 'opt.py work sel --only "
            "results/O_sel_refs.txt'가 돈다. 출력 results/U1_selection.json(complete_pairs, n_by_class, selected_rxns, needed_refs, "
            "missing_refs, plan 지문, S_pairs sha256)은 선택을 정하는 키(method, plan 지문, S_pairs sha256, complete_pairs, "
            "selected_rxns, needed_refs)가 같으면 다시 쓰지 않는다(sha256 고정: 기준 구조가 돈 뒤 select를 다시 돌려도 assemble·F·E가 "
            "기록한 sha256이 그대로; 파일의 missing_refs·ts_units_finished는 'written' 시점 값). O_sel_refs.txt는 매번 지금 없는 "
            "기준 구조로 쓴다(바뀔 때만). unit.json이 없는 g3r TS 단위를 "
            "살아 있는 job이 잡고 있으면(lock 주인이 squeue에 있음) 고르지 않는다(exit 2: scancel 먼저; 끝나면 완성 짝이 바뀔 수 "
            "있음). 완성 짝이 0이면 STOP(exit 3).",
    "U-1b": "U-1 조립: 'opt.py assemble'은 선택을 다시 정하지 않고 results/U1_selection.json을 읽는다(없음 = exit 2). 선택 파일의 "
            "plan 지문·S_pairs sha256이 다르거나, 지금 끝난 unit.json으로 본 완성 짝 목록이 파일과 다르면 exit 2('opt.py select' "
            "다시); selected_rxns·needed_refs가 파일의 complete_pairs와 plan에서 따라 나오지 않거나 완성 짝이 0이면 exit 3. 선택 밖 "
            "(방법, 반응)(g3h 전부, 완성 짝 밖의 g3r) = $G3/<m>_final/<rxn>/에 .fail_stopped_user 표지와 result.json만(xyz 없음), "
            "O_status status 'stopped_user', reason 'U-1'(TS 단위가 끝났으면 ts_status·core_h 등을 그대로 기록, core_h_refs는 비움). "
            "선택된 g3r 반응은 O-18 그대로(TS ok + 두 기준 구조 ok -> .done; 필요한 기준 구조의 unit.json 없음 = incomplete -> "
            "exit 2). Part B 행은 그대로. unit.json이 없는 단위(미실행, U-1 scancel)는 파이프라인 오류도 미완도 아니다: 어떤 선택 "
            "반응도 쓰지 않는 미실행 기준 구조는 O_refs.csv status 'stopped_user'(u1_needed 열). 표지가 선택과 맞지 않는 폴더(선택"
            "됐는데 .fail_stopped_user, 선택 밖인데 다른 표지)는 --force 없이도 다시 조립한다(이전 폴더는 <m>_final_prev/로); 그 "
            "밖의 --force 의미는 그대로. O STOP 통계(오류율, 첫 20, STOP 파일)는 끝난 단위만 센다: 끝난 TS가 20개 미만인 방법(g3h "
            "19개)은 첫 20 규칙을 판정하지 않는다(O_report u1.first20에 보고). O_report.json에 u1 블록(결정, 선택 파일 sha256, 수), "
            "O_status.csv에 u1_selected 열. unit.json 없는 TS(n_ts_not_run)는 killed(단위 폴더 있음 = ORCA 단계가 시작됐고 "
            "scancel로 죽음)와 never_started로 나눠 보고하고, killed 중 1차 시도가 gate를 통과하지 못한 뒤 죽은 단위(a2/ 있음 또는 "
            "a1/step.json gate 실패)를 ts_killed_in_retry(uid 목록)로 적는다; ts_success_lower_bound = TS ok / (끝난 TS + 재시도 "
            "중 죽은 TS)(그것들을 실패로 셈; 설명용, 판정 ②의 ts_success는 끝난 단위 기준 그대로). killed 단위의 끝난 ORCA 단계 "
            "core_h 합(ts_killed_core_h_finished_steps; 죽을 때 돌던 단계는 기록 없음 = 하한)도 보고.",
}


class StopAll(Exception):
    """a STOP found inside a worker (ORCA version, input inconsistency that affects every unit)."""


class Fail(Exception):
    pass


class WallClipped(Exception):
    """O-19: an ORCA step started MAX_STARTS times without finishing (48 h wall)."""


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


def _now():
    return time.strftime("%F %T")


def _cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def _jobtag():
    return os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"


def _san(s):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")[:90] or "unknown"


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


# ================================================================ D1 modules (imported in the order that makes
# 'fragmenter' the D1 copy with Si; checked)
_D = {}


def d1():
    if _D:
        return _D
    import yaml
    sys.path.insert(0, str(VAL / "analysis"))
    import rmsd as rm  # noqa: E402          rmsd.py puts EDA-ASM/analysis/espley_xtb_repro first -> D1 fragmenter
    sys.path.insert(0, str(PILOT))
    import pilot_common as pc  # noqa: E402
    import run_ts as RT  # noqa: E402
    import orca_direct as OD  # noqa: E402
    cfg_run = yaml.safe_load(CFG_RUN.read_text())
    cfg_val = yaml.safe_load(CFG_VAL.read_text())
    fr, s3 = pc.import_repo_modules(cfg_run)
    import refs_v as RV  # noqa: E402        (D1LADDER; d1_ladder is on sys.path through g3_common)
    H = RV.Q.hf_common()
    HP, HR = RV.hf_prep(), RV.hf_run()
    bad = []
    if fr is not rm.fr:
        bad.append(f"fragmenter of rmsd.py {rm.fr.__file__} != pilot_common's {fr.__file__}")
    if Path(fr.__file__).resolve() != FRAG_D1.resolve() or fr.COV.get("Si") != 1.11:
        bad.append(f"fragmenter {fr.__file__} is not the D1 copy with Si ({FRAG_D1})")
    if Path(RT.__file__).resolve().parent != PILOT.resolve() or Path(OD.__file__).resolve().parent != PILOT.resolve():
        bad.append(f"run_ts / orca_direct not from {PILOT}")
    if Path(rm.__file__).resolve().parent != (VAL / "analysis").resolve():
        bad.append(f"rmsd.py not from {VAL / 'analysis'}")
    if bad:
        C.stop("D1 modules: " + "; ".join(bad))
    env = dict(os.environ, OMPI_MCA_hwloc_base_binding_policy="none", OMP_NUM_THREADS="1")
    _D.update(rm=rm, pc=pc, RT=RT, OD=OD, fr=fr, s3=s3, RV=RV, H=H, HP=HP, HR=HR, cfg_run=cfg_run, cfg_val=cfg_val,
              orca=OD.orca_bin(cfg_run), env=env, man={})
    return _D


def same_thr(D):
    t = D["cfg_run"]["run"]["same_ts"]
    return float(t["rmsd_A"]), float(t["dform_A"])


# ================================================================ plan (G3ENV=reactot)
def load_plan():
    if not PLAN.is_file():
        C.stop(f"{PLAN} missing: run 'opt.py plan' first")
    return C.read_json(PLAN)


def plan_fingerprint(Q):
    return dict(code=CODE_VERSION, S_pairs=C.sha256(S_PAIRS), S_rxns=C.sha256(S_RXNS), Q_keys=C.sha256(Q.LAD_KEYS),
                labels=DP.D1_LABELS_SHA, manifest=G2C.MANIFEST_SHA)


def part_b_ids(R):
    """(sorted rxn_ids, column) of the Part B reactions in S_rxns.csv (O-17)."""
    idc = "rxn_id" if "rxn_id" in R.columns else None
    if idc is None:
        C.stop(f"{S_RXNS}: no rxn_id column ({list(R.columns)})")
    for c in PART_B_COLS:
        if c in R.columns:
            hit = R[c].map(lambda x: str(x).strip().lower() in ("1", "1.0", "true", "yes", "y", "b"))
            return sorted(int(r) for r in R.loc[hit, idc]), c
    for c in PART_B_TEXT_COLS:
        if c in R.columns:
            tok = R[c].map(lambda x: {t.lower() for t in re.split(r"[^A-Za-z0-9_]+", str(x)) if t})
            hit = tok.map(lambda s: bool(s & {"b", "partb", "part_b"}))
            return sorted(int(r) for r in R.loc[hit, idc]), c
    C.stop(f"{S_RXNS}: no Part B column (looked for {PART_B_COLS} and {PART_B_TEXT_COLS}; have {list(R.columns)})")


def _pairs(d):
    return sorted(tuple(sorted(int(a) for a in p.split("-"))) for p in str(d["formed_pairs_ts"]).split())


def rxn_record(d, man, H):
    """plan record of one label row + the problems found (G2 files, atom orders, eda.inp tags, manifest)."""
    T, rid = d["job_id"], int(d["rxn_id"])
    prob = []
    pairs = _pairs(d)
    if len(pairs) != 2:
        prob.append(f"formed_pairs_ts {d['formed_pairs_ts']!r}")
    jd, g2 = D1_JOBS / T, G2_FINAL / str(rid)
    rec = dict(rxn_id=rid, job_id=T, n_atoms=int(d["n_atoms"]), n_f1=int(d["n_f1"]), n_f2=int(d["n_f2"]),
               charge1=int(d.get("charge1") or 0), charge2=int(d.get("charge2") or 0), pairs=[list(p) for p in pairs],
               e_ab_eh=float(d["e_ab_eh"]), dipole=d.get("dipole"), skeleton=d.get("skeleton"), core=d.get("core"))
    if (d.get("role1"), d.get("role2")) != ("dipole", "dipolarophile"):
        prob.append(f"roles {d.get('role1')} / {d.get('role2')}")
    try:
        e = H.parse_inp((jd / "sp" / "eda.inp").read_text())
        tags = [a[1] for a in e["atoms"]]
        rec["A_idx"] = [i for i, t in enumerate(tags) if t == 1]
        if tags.count(1) != rec["n_f1"] or tags.count(2) != rec["n_f2"]:
            prob.append(f"eda.inp tags {tags.count(1)} / {tags.count(2)} != n_f1 / n_f2")
        lts = [a[0] for a in H.read_xyz(jd / d["ts_file"])]
        if [a[0] for a in e["atoms"]] != lts:
            prob.append("eda.inp elements != label ts.xyz")
        if not (g2 / ".done").is_file() or not all((g2 / f).is_file() for f in ("ts.xyz", "rel1.xyz", "rel2.xyz")):
            prob.append("G2 final folder not .done or files missing")
        else:
            if [a[0] for a in H.read_xyz(g2 / "ts.xyz")] != lts:
                prob.append("G2 ts.xyz elements != label ts.xyz")
            for k in (1, 2):
                lref = [a[0] for a in H.read_xyz(jd / d[f"rel{k}_file"])]
                if [a[0] for a in H.read_xyz(g2 / f"rel{k}.xyz")] != lref:
                    prob.append(f"G2 rel{k}.xyz elements != label {d[f'rel{k}_file']}")
    except Exception as ex:                                        # noqa: BLE001
        prob.append(f"read error {type(ex).__name__}: {ex}")
    if T not in man.index or not str(man.loc[T, "mapped_smiles"]).strip():
        prob.append("no mapped_smiles in d1_manifest.csv")
    else:
        rec["mapped_smiles"] = str(man.loc[T, "mapped_smiles"])
    return rec, prob


def cmd_plan(force=False):
    for k, v in DECISIONS.items():                                 # spec 11: recorded before they are acted on
        record_decision(k, v)
    import refs_q as Q
    H = Q.hf_common()
    for p in (S_PAIRS, S_RXNS, Q.LAD_KEYS, G2C.MANIFEST):
        if not p.is_file():
            C.stop(f"plan input {p} missing")
    fp = plan_fingerprint(Q)
    if PLAN.is_file():
        old = C.read_json(PLAN)
        if old.get("fingerprint") == fp and not force:
            print("plan up to date", flush=True)
            return 0
        if UNITS.is_dir() and any(UNITS.glob("*/unit.json")):
            C.stop(f"plan inputs changed ({old.get('fingerprint')} -> {fp}) after units ran; not re-planned")
    if C.sha256(G2C.MANIFEST) != G2C.MANIFEST_SHA:
        C.stop(f"{G2C.MANIFEST} sha256 != g2_common.MANIFEST_SHA")
    lab = {int(d["rxn_id"]): d for d in C.targets()}
    S = pd.read_csv(S_PAIRS)
    for c in ("pair_id", "a_rxn", "b_rxn"):
        if c not in S.columns:
            C.stop(f"{S_PAIRS}: column {c} missing")
    A = sorted(set(int(x) for x in S.a_rxn) | set(int(x) for x in S.b_rxn))
    cls = {}
    for r in S.to_dict("records"):                                 # (itertuples would rename the column 'class')
        for x in (int(r["a_rxn"]), int(r["b_rxn"])):
            cls.setdefault(x, set()).add(str(r.get("class", "")))
    R = pd.read_csv(S_RXNS)
    B, bcol = part_b_ids(R)
    if "part_a" in R.columns:
        ra = sorted(int(r) for r in R.loc[R.part_a.map(lambda x: str(x).strip().lower() in ("1", "true")), "rxn_id"])
        if ra != A:
            C.stop(f"S_rxns.csv part_a ({len(ra)}) != S_pairs a_rxn | b_rxn ({len(A)})")
    if not A or not B:
        C.stop(f"empty sample: Part A {len(A)}, Part B {len(B)}")
    man = pd.read_csv(G2C.MANIFEST, dtype=str, keep_default_na=False).set_index("job_id")
    rxns, probs = {}, {}
    for rid in sorted(set(A) | set(B)):
        if rid not in lab:
            probs[rid] = ["not a target label row"]
            continue
        rec, p = rxn_record(lab[rid], man, H)
        rec.update(part_a=rid in A, part_b=rid in B, classes=sorted(cls.get(rid, ())))
        rxns[rid] = rec
        if p:
            probs[rid] = p
    if probs:
        C.write_json(O / "plan_problems.json", {str(k): v for k, v in probs.items()})
        C.stop(f"{len(probs)} sample reactions fail the plan checks ({O / 'plan_problems.json'}): "
               + "; ".join(f"{k}: {v[0]}" for k, v in list(probs.items())[:5]))
    # molecule units (O-7)
    K = Q.load_keys()
    for c in ("job_id", "frag", "key", "constrained", "n_atoms", "charge"):
        if c not in K.columns:
            C.stop(f"Q_keys.parquet column {c} missing")
    jobsA = {rxns[r]["job_id"] for r in A}
    Kx = K[K.job_id.isin(jobsA)]
    if Kx.duplicated(["job_id", "frag"]).any():
        C.stop("Q_keys.parquet has duplicate (job_id, frag) rows")
    Kx = Kx.set_index(["job_id", "frag"])
    groups, kinfo = {}, {}
    for rid in A:
        rec = rxns[rid]
        for k in (1, 2):
            try:
                row = Kx.loc[(rec["job_id"], k)]
            except KeyError:
                C.stop(f"Q_keys.parquet has no row for {rec['job_id']} frag {k}")
            if int(row["n_atoms"]) != rec[f"n_f{k}"] or int(row["charge"]) != rec[f"charge{k}"]:
                C.stop(f"Q key row {rec['job_id']}/{k}: n_atoms / charge != label")
            cons = bool(row["constrained"])
            uk = f"{row['key']}#{rec['job_id']}" if cons else str(row["key"])
            groups.setdefault(uk, []).append((rec["job_id"], rid, k))
            kinfo[uk] = dict(key=str(row["key"]), constrained=cons, frag=k, kind="dipole" if k == 1 else "dipolarophile",
                             n_atoms=int(row["n_atoms"]), charge=int(row["charge"]))
    mols = {}
    for i, uk in enumerate(sorted(groups), 1):
        mid = f"M{i:04d}"
        mem = sorted(groups[uk])
        st = mem[0]
        if any(f != st[2] for _, _, f in mem):
            C.stop(f"molecule unit {uk} spans both fragment roles")
        mols[mid] = dict(mid=mid, unit_key=uk, **kinfo[uk], start_job=st[0], start_rxn=st[1],
                         members=[[r, f] for _, r, f in mem])
        for _, r, f in mem:
            rxns[r][f"mid{f}"] = mid
    # T-1 / pilot / T-5 (O-12)
    order = sorted(A, key=lambda r: (rxns[r]["n_atoms"], r))
    t1, rest = order[0], order[1:]
    pilot = sorted({rest[int(round(q * (len(rest) - 1)))] for q in PILOT_Q}) if rest else []
    orderB = sorted(B, key=lambda r: (rxns[r]["n_atoms"], r))
    pilot_b = sorted({orderB[int(round(q * (len(orderB) - 1)))] for q in PILOT_B_Q})
    t5 = [dict(job_id=rxns[r]["job_id"], rxn_id=r, n_atoms=rxns[r]["n_atoms"]) for r in orderB[:3]]
    early = {t1} | set(pilot)
    units = []
    for m in C.METHODS:
        for rid in A:
            n = rxns[rid]["n_atoms"]
            units.append(dict(uid=f"ts.{m}.{rid}", kind="ts", method=m, rxn_id=rid, job_id=rxns[rid]["job_id"], mid=None,
                              n_atoms=n, phase="t1" if rid == t1 else "pilot" if rid in pilot else "main",
                              weight=float(n) ** 3))
        for mid, M in mols.items():
            ph = "pilot" if any(r in early for r, _ in M["members"]) else "main"
            units.append(dict(uid=f"ref.{m}.{mid}", kind="ref", method=m, rxn_id=None, job_id=M["start_job"], mid=mid,
                              n_atoms=M["n_atoms"], phase=ph, weight=0.3 * float(M["n_atoms"]) ** 3))
    for rid in B:
        n = rxns[rid]["n_atoms"]
        units.append(dict(uid=f"B.{rid}", kind="B", method="partB", rxn_id=rid, job_id=rxns[rid]["job_id"], mid=None,
                          n_atoms=n, phase="pilot" if rid in pilot_b else "main", weight=4.0 * float(n) ** 3))
    plan = dict(code=CODE_VERSION, written=_now(), fingerprint=fp, part_b_column=bcol, part_a=A, part_b=B,
                t1_rxn=t1, pilot_rxns=pilot, pilot_b=pilot_b, t5=t5, rxns={str(k): v for k, v in rxns.items()},
                mols=mols, units=units, n_units=len(units),
                counts=dict(part_a=len(A), part_b=len(B), overlap=len(set(A) & set(B)), mols=len(mols),
                            mols_constrained=sum(M["constrained"] for M in mols.values()),
                            shared_mols=sum(len(M["members"]) > 1 for M in mols.values()),
                            units={k: sum(u["kind"] == k for u in units) for k in ("ts", "ref", "B")},
                            phases={p: sum(u["phase"] == p for u in units) for p in ("t1", "pilot", "main")}),
                decisions=DECISIONS)
    C.write_json(PLAN, plan)
    C.write_atomic(RES_PLAN, lambda t: pd.DataFrame(units).to_csv(t, index=False))
    print(json.dumps(dict(counts=plan["counts"], t1=t1, pilot=pilot, pilot_b=pilot_b, t5=t5), indent=1), flush=True)
    return 0


# ================================================================ inputs
def ts_header(m, att, nprocs):
    return (f"! {METHOD_KW[m]} OptTS Freq {SOLV_WORD}\n{GEOM_BLOCK[att]}\n{SOLV_BLOCK}"
            f"%maxcore {MAXCORE}\n%pal nprocs {int(nprocs)} end\n\n")


def ref_header(m, nprocs):
    return f"! {METHOD_KW[m]} Opt Freq {SOLV_WORD}\n{SOLV_BLOCK}%maxcore {MAXCORE}\n%pal nprocs {int(nprocs)} end\n\n"


def inp_text(D, header, syms, xyz, q, mult, local):
    """orca_direct.write_inp text (the G1 / G2 coordinate format)."""
    local.mkdir(parents=True, exist_ok=True)
    p = local / f"_inp_{threading.get_ident()}.inp"
    D["OD"].write_inp(p, header, list(syms), np.asarray(xyz, float), charge=int(q), mult=int(mult))
    t = p.read_text()
    p.unlink()
    return t


def partb_text(D, syms, xyz, q, mult, pairs, nprocs):
    """O-9: the D1 autodE OptTS input = orca_direct L0 header (pilot header checked against prereg, modify_internal
    blocks of the forming bonds, %pal / %maxcore set) + autodE's print_coordinates lines."""
    h = D["OD"].level_header(D["cfg_val"], "L0", [tuple(int(i) for i in p) for p in pairs], n_cores=int(nprocs))
    body = "".join(f"{s:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}\n" for s, (x, y, z) in zip(syms, np.asarray(xyz, float)))
    return f"{h}*xyz {int(q)} {int(mult)}\n{body}*\n"


def parse_autode_optts(text):
    """(syms, xyz, charge, mult, pairs, nprocs) of an autodE ORCA OptTS input (as refs_v.parse_autode_inp)."""
    lines = text.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("*xyz"))
    _, q, mlt = lines[i].split()
    syms, xyz = [], []
    for ln in lines[i + 1:]:
        if ln.strip() == "*":
            break
        t = ln.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\{ B (\d+) (\d+) A \}", text)]
    pal = re.search(r"^%pal nprocs (\d+)$", text, re.M)
    return syms, np.array(xyz, float), int(q), int(mlt), pairs, int(pal.group(1)) if pal else 1


def eda_text(T):
    return (D1_JOBS / T / "sp" / "eda.inp").read_text()


def sp_template(D, eda):
    """mode 'sp' template: the label eda.inp without the EDA word on the '!' line and without the %eda block."""
    H = D["H"]
    out, skip, nb = [], False, 0
    for ln in eda.splitlines(keepends=True):
        s = ln.strip()
        if skip:
            if s.lower() == "end":
                skip = False
            continue
        if s.lower().startswith("%eda"):
            skip = True
            continue
        if s.startswith("!"):
            toks = s[1:].split()
            if toks.count("EDA") != 1:
                raise ValueError("label eda.inp '!' line without exactly one EDA word")
            ln = "! " + " ".join(t for t in toks if t != "EDA") + "\n"
            nb += 1
        out.append(ln)
    tpl = "".join(out)
    pe, pt = H.parse_inp(eda), H.parse_inp(tpl)
    if (nb != 1 or pt["bang"] != [t for t in pe["bang"] if t != "EDA"] or set(pt["blocks"]) != set(pe["blocks"]) - {"eda"}
            or pt["atoms"] != pe["atoms"] or (pt["charge"], pt["mult"]) != (pe["charge"], pe["mult"]) or pt["frag"]):
        raise ValueError("sp template differs from the label eda.inp by more than the EDA word / %eda block")
    return tpl


def sp_text(D, mode, eda, syms, xyz):
    atoms = [(s, float(x), float(y), float(z)) for s, (x, y, z) in zip(syms, np.asarray(xyz, float))]
    tpl = eda if mode == "eda" else sp_template(D, eda)
    new = D["HP"].replace_coords(tpl, atoms)
    why = D["RV"].coord_only_check(tpl, new)
    if why:
        raise ValueError(f"SP input ({mode}) is not the template with new coordinates only: {why}")
    return new


# ================================================================ ORCA step
def _infra(txt, rc):
    """died before its first SCF iteration (MPI start-up and the like; d1_g1.cmd_infra signature) or no output."""
    D = d1()
    if D["H"].terminated(txt):
        return False
    if rc is not None and rc < 0:
        return True
    if not txt.strip():
        return True
    return ("error termination in orca_util" in txt.lower() and "SCF ITERATIONS" not in txt
            and "GEOMETRY OPTIMIZATION CYCLE" not in txt)


def _version(D, txt, where):
    H = D["H"]
    v, g = H.RE_VER.search(txt), H.RE_GIT.search(txt)
    if v and (v.group(1) != H.ORCA_VERSION or (g is not None and g.group(1) != H.ORCA_GIT)):
        raise StopAll(f"{where}: ORCA {v.group(1)} (GIT {g.group(1) if g else '?'}) != {H.ORCA_VERSION} ({H.ORCA_GIT})")
    return v.group(1) if v else None


def orca_step(D, dest, name, text, local, post=None):
    """one ORCA input in a node-local folder -> dest/step.json (cached: a finished step is never rerun). post(work, txt,
    step) adds the analysis that needs the work files (gate, displacement) before they are deleted."""
    sj = dest / "step.json"
    old = _rj(sj)
    if old is not None:
        return old
    sp = dest / "starts.json"                                      # O-19: unfinished starts (48 h wall clips)
    starts = _rj(sp) or []
    if len(starts) >= MAX_STARTS:
        raise WallClipped(f"{dest.relative_to(UNITS)}: started {len(starts)} times without finishing ({starts})")
    C.write_json(sp, starts + [dict(job=_jobtag(), host=socket.gethostname(), started=_now())])
    H, OD = D["H"], D["OD"]
    work = local / dest.relative_to(UNITS)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    inp = work / f"{name}.inp"
    m = re.search(r"%pal\s+nprocs\s+(\d+)", text)
    npr = int(m.group(1)) if m else 1
    runs, txt, nbytes = [], "", 0
    for k in (1, 2):
        for p in work.iterdir():
            if p.is_file():
                p.unlink()
        inp.write_text(text)
        t = time.time()
        with open(work / f"{name}.out", "w") as fo, open(work / f"{name}.err", "w") as fe:
            pr = subprocess.run([D["orca"], inp.name], cwd=work, stdout=fo, stderr=fe, env=D["env"])
        wall = time.time() - t
        txt = (work / f"{name}.out").read_text(errors="replace")
        infra = _infra(txt, pr.returncode)
        runs.append(dict(run=k, rc=pr.returncode, wall_s=wall, core_h=wall * npr / 3600, infra=infra,
                         orca_s=H.run_seconds(txt), host=socket.gethostname()))
        _version(D, txt, f"{dest}/{name}")
        if not infra:
            break
        data = gzip.compress((work / f"{name}.out").read_bytes(), 6)
        C.write_atomic(dest / f"{name}.infra{k}.out.gz", data)
        nbytes += len(data)
    lc = OD.level_checks(txt)
    step = dict(name=name, runs=runs, infra=runs[-1]["infra"], terminated=H.terminated(txt),
                converged=OD.opt_converged(txt), e_last_eh=OD.last_fspe(txt), n_fspe=len(H.RE_FSPE.findall(txt)),
                n_opt_cycles=len(RE_OPTCYC.findall(txt)), wall_s=sum(r["wall_s"] for r in runs),
                core_h=sum(r["core_h"] for r in runs), nprocs=npr, orca_hours=OD.run_hours(txt),
                orca_version=_version(D, txt, f"{dest}/{name}"), keyword_line=lc.get("keyword_line"),
                eps=sorted(set(H.RE_EPS.findall(txt))), smd_cds=CDS_MARK in txt, smd_module=SMD_MARK in txt,
                basis=sorted(set(H.RE_BASIS.findall(txt)))[:4], method=sorted(set(H.RE_METHOD.findall(txt)))[:4],
                has_xyz=(work / f"{name}.xyz").is_file(), has_hess=(work / f"{name}.hess").is_file(),
                finished=_now())
    for fn, gz in ((f"{name}.inp", False), (f"{name}.out", True), (f"{name}.err", True), (f"{name}.xyz", False),
                   (f"{name}.hess", True)):                          # kept before post: evidence even if post raises
        p = work / fn
        if not p.is_file() or (fn.endswith(".err") and p.stat().st_size == 0):
            continue
        data = gzip.compress(p.read_bytes(), 6) if gz else p.read_bytes()
        C.write_atomic(dest / (fn + (".gz" if gz else "")), data)
        nbytes += len(data)
    try:                                                           # the work folder goes even if post raises
        if post is not None and not step["infra"]:
            step.update(post(work, txt, step))
        step["bytes"] = nbytes
        C.write_json(sj, step)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return step


# ================================================================ gates
def hess_atoms(path):
    """(syms, xyz A) of the $atoms block of an ORCA .hess file (= g1_geom.hess_atoms, espley_xtb_repro/g1_geom.py:146)."""
    lines = Path(path).read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "$atoms")
    n = int(lines[i + 1].split()[0])
    rows = [lines[i + 2 + k].split() for k in range(n)]
    return [r[0] for r in rows], np.array([[float(v) for v in r[2:5]] for r in rows]) * BOHR


def _dists(xyz, pairs):
    X = np.asarray(xyz, float)
    return [float(np.linalg.norm(X[i] - X[j])) for i, j in pairs]


def shape(D, ts, ref, pairs):
    s, r, dd = D["rm"].same_ts(ts, ref, _dists(ts[1], pairs), _dists(ref[1], pairs), *same_thr(D))
    return dict(same_ts=bool(s), rmsd_A=float(r), dform_A=float(dd))


def gate_ts(D, R, work, name, txt, step, reacs, refs, partition_hard):
    """O-4 / O-10: the D1 TS gate (run_ts.analyse) + the G1 checks on an OptTS + Freq run. refs = {tag: (syms, xyz)}
    structures to compare with (same TS rule)."""
    pc, RT, OD = D["pc"], D["RT"], D["OD"]
    pairs = [tuple(p) for p in R["pairs"]]
    g = dict(ok=False, reason="")
    xp, hp = work / f"{name}.xyz", work / f"{name}.hess"
    ts1 = pc.read_xyz(xp) if xp.is_file() else None
    syms0 = refs["start"][0]
    if ts1 is not None:
        if list(ts1[0]) != list(syms0):
            raise RuntimeError("ORCA final geometry has a different atom order")
        g["d_form"] = _dists(ts1[1], pairs)
        for tag, ref in refs.items():
            g[tag] = shape(D, ts1, ref, pairs)
    if not step["terminated"]:
        g["reason"] = "optts_not_converged" if not step["converged"] else "orca_not_terminated"
        if not step["converged"] and step["n_opt_cycles"] == 0:
            g["reason"] = "orca_not_terminated"
        return g
    if not step["converged"]:
        g["reason"] = "optts_not_converged"
        return g
    if ts1 is None or not hp.is_file():
        g["reason"] = "orca_missing_xyz_or_hess"
        return g
    hs, hx = hess_atoms(hp)                                        # G1 rule (g1_geom.py:227-240)
    shift = hx - ts1[1]
    last = txt.rsplit("VIBRATIONAL FREQUENCIES", 1)[-1]
    f_out = [float(v) for v in RE_OUTFREQ.findall(last)]
    f_hess = OD.parse_hess(hp)[0]
    g["hess_geom_max_dev_A"] = float(np.abs(shift - shift.mean(0)).max())
    g["hess_freq_max_dev_cm"] = float(np.abs(np.asarray(f_out) - f_hess).max()) if len(f_out) == len(f_hess) else None
    if (list(hs) != list(ts1[0]) or g["hess_geom_max_dev_A"] > 1e-4 or g["hess_freq_max_dev_cm"] is None
            or g["hess_freq_max_dev_cm"] > 0.01):
        g["reason"] = "hess_not_at_final_geometry"
        return g
    imag, mode = OD.imag_and_mode(hp)
    g.update(imag_freqs_cm=[float(v) for v in imag], n_imag=len(imag))
    res = {}
    jd = work / "gate"
    jd.mkdir(exist_ok=True)
    got = dict(ts=(list(ts1[0]), np.asarray(ts1[1], float)), reacs=reacs, prod=None, imag=list(imag), mode=mode)
    job = dict(job_id=R["job_id"], mapped_smiles=R["mapped_smiles"])
    hard = None
    try:
        RT.analyse(D["cfg_run"], job, jd, res, got, D["fr"])
    except RT.StageFail as e:
        hard = str(e)
    pm = (sorted(res["A_idx"]) == sorted(R["A_idx"])) if "A_idx" in res else None
    if hard is None and partition_hard and pm is False:
        hard = "partition_mismatch"
    fp = sorted(tuple(sorted(p)) for p in res.get("formed_pairs_ts") or [])
    g.update(ok=hard is None, reason=hard or "", imag_mode_forming_share=res.get("imag_mode_forming_share"),
             checks=res.get("checks"), partition=res.get("partition"), A_idx=res.get("A_idx"), partition_match=pm,
             formed_pairs_ts=[list(p) for p in fp], formed_pairs_eq_label=(fp == sorted(pairs)) if fp else None,
             formed_d=res.get("formed_d"), foreign=res.get("foreign"), ts_dipole_config=res.get("ts_dipole_config"))
    return g


def post_ref(D, work, txt, step, start):
    pc, OD, fr = D["pc"], D["OD"], D["fr"]
    import networkx as nx
    r = dict(reason="", neg_cm=[])
    xp, hp = work / "ref.xyz", work / "ref.hess"
    if not step["terminated"]:
        r["reason"] = "opt_not_converged" if step["n_opt_cycles"] and not step["converged"] else "orca_not_terminated"
        return r
    if not step["converged"]:
        r["reason"] = "opt_not_converged"
        return r
    if not xp.is_file() or not hp.is_file():
        r["reason"] = "orca_missing_xyz_or_hess"
        return r
    fin = pc.read_xyz(xp)
    if list(fin[0]) != list(start[0]):
        raise RuntimeError("ORCA final reference geometry has a different atom order")
    Ga = fr.heavy_graph(list(start[0]), np.asarray(start[1], float))[0]
    Gb = fr.heavy_graph(list(fin[0]), np.asarray(fin[1], float))[0]
    r["graph_same"] = bool(nx.is_isomorphic(Ga, Gb, node_match=fr._nm))
    freqs, modes = OD.parse_hess(hp)
    neg = sorted(float(v) for v in freqs if v < 0)
    tol = float(D["cfg_run"]["gates"]["extra_imag_tol_cm"])
    r.update(neg_cm=neg, n_below_tol=sum(v < tol for v in neg), imag_tol_cm=tol, e_eh=step["e_last_eh"],
             lowest_real_cm=min([float(v) for v in freqs if v > 1e-6], default=None))
    if not r["graph_same"]:
        r["reason"] = "graph_changed"
    if neg:
        mode = modes[:, int(np.argmin(freqs))].reshape(-1, 3)
        disp = np.asarray(fin[1], float) + DISP_FACTOR * mode / np.linalg.norm(mode)
        r["disp_xyz"] = disp.tolist()
    return r


def post_sp(D, txt, step):
    H, HR = D["H"], D["HR"]
    e = H.RE_FSPE.findall(txt)
    eps = H.RE_EPS.findall(txt)
    reason = ""
    if not H.terminated(txt):
        reason = "orca_not_terminated"
    elif not HR.scf_good(txt):
        reason = "scf"
    elif len(e) != 1:
        reason = f"fspe_count_{len(e)}"
    elif not eps or any(x != EPS_SMD_WATER for x in eps):
        reason = "no_smd_water"
    return dict(sp=dict(ok=not reason, reason=reason, e_eh=float(e[-1]) if e else None, eps=sorted(set(eps))))


# ================================================================ units
def unit_ts(u, D, plan, nprocs, local):
    R = plan["rxns"][str(u["rxn_id"])]
    m, pc = u["method"], D["pc"]
    g2d = G2_FINAL / str(R["rxn_id"])
    g2ts = pc.read_xyz(g2d / "ts.xyz")
    reacs = [pc.read_xyz(g2d / "rel1.xyz"), pc.read_xyz(g2d / "rel2.xyz")]
    q = R["charge1"] + R["charge2"]
    atts, core = [], 0.0
    for att in (1, 2):
        text = inp_text(D, ts_header(m, att, nprocs), *g2ts, q, 1, local)
        st = orca_step(D, UNITS / u["uid"] / f"a{att}", "ts", text, local,
                       lambda w, t, s: {"gate": gate_ts(D, R, w, "ts", t, s, reacs, {"start": g2ts}, True)})
        core += st["core_h"]
        atts.append(st)
        if st["infra"]:
            return dict(status="error", reason="pipeline_infra", attempts=[_att(a) for a in atts], core_h=core)
        if st["gate"]["ok"]:
            break
    gt = atts[-1]["gate"]
    sh = gt.get("start") or {}
    return dict(status="ok" if gt["ok"] else "fail", reason=gt["reason"], attempt_used=len(atts),
                final_dir=f"a{len(atts)}", header=ts_header(m, len(atts), nprocs), attempts=[_att(a) for a in atts],
                gate=gt, shape_kept=sh.get("same_ts"), rmsd_g2_A=sh.get("rmsd_A"), dform_g2_A=sh.get("dform_A"),
                e_ts_eh=atts[-1]["e_last_eh"], ts_run_hours=atts[-1]["orca_hours"], core_h=core,
                orca_version=atts[-1]["orca_version"])


def _att(st):
    g = st.get("gate") or st.get("post") or st.get("sp") or {}
    ok = None if st["infra"] else g.get("ok", not g.get("reason") and not g.get("neg_cm"))
    return dict(ok=ok, reason="infra" if st["infra"] else g.get("reason"), core_h=st["core_h"], wall_s=st["wall_s"],
                n_opt_cycles=st["n_opt_cycles"], terminated=st["terminated"], converged=st["converged"],
                infra_runs=sum(r["infra"] for r in st["runs"]), n_imag=g.get("n_imag"), neg_cm=g.get("neg_cm"))


def unit_ref(u, D, plan, nprocs, local):
    M = plan["mols"][u["mid"]]
    m, pc = u["method"], D["pc"]
    start = pc.read_xyz(G2_FINAL / str(M["start_rxn"]) / f"rel{M['frag']}.xyz")
    coords, atts, core = start[1], [], 0.0
    for att in (1, 2):
        text = inp_text(D, ref_header(m, nprocs), start[0], coords, M["charge"], 1, local)
        st = orca_step(D, UNITS / u["uid"] / f"a{att}", "ref", text, local,
                       lambda w, t, s: {"post": post_ref(D, w, t, s, start)})
        core += st["core_h"]
        atts.append(st)
        if st["infra"]:
            return dict(status="error", reason="pipeline_infra", attempts=[_att(a) for a in atts], core_h=core)
        p = st["post"]
        if p["reason"] or not p["neg_cm"]:
            break
        if att == 1:
            coords = np.asarray(p["disp_xyz"], float)
    p = atts[-1]["post"]
    ok = not p["reason"] and not p["neg_cm"]
    return dict(status="ok" if ok else "fail", reason=p["reason"] or ("" if ok else "imag"), attempt_used=len(atts),
                final_dir=f"a{len(atts)}", attempts=[_att(a) for a in atts], neg_cm=p.get("neg_cm"),
                n_below_tol=p.get("n_below_tol"), e_eh=p.get("e_eh"), graph_same=p.get("graph_same"), core_h=core,
                header=ref_header(m, nprocs), key=M["key"], unit_key=M["unit_key"], start_rxn=M["start_rxn"],
                orca_version=atts[-1]["orca_version"])


def t5_mode():
    t = _rj(T5_JSON)
    return t.get("mode") if t and t.get("pass") else None


def unit_b(u, D, plan, nprocs, local):
    R = plan["rxns"][str(u["rxn_id"])]
    T, pc = R["job_id"], D["pc"]
    g2ts = pc.read_xyz(G2_FINAL / str(R["rxn_id"]) / "ts.xyz")
    lts = pc.read_xyz(D1_JOBS / T / "ts.xyz")
    reacs = [pc.read_xyz(D1_JOBS / T / "ref_dipole.xyz"), pc.read_xyz(D1_JOBS / T / "ref_dipolarophile.xyz")]
    q = R["charge1"] + R["charge2"]
    ud = UNITS / u["uid"]

    def post(w, t, s):
        e = D["H"].RE_FSPE.findall(t)                              # autodE ORCAOptimiser.last_energy_change
        return {"gate": gate_ts(D, R, w, "ts", t, s, reacs, {"start": g2ts, "label": lts}, False),
                "de_last_kcal": (float(e[-1]) - float(e[-2])) * C.KCAL if len(e) >= 2 else None}
    text = partb_text(D, g2ts[0], g2ts[1], q, 1, R["pairs"], PARTB_NPROCS)
    st = orca_step(D, ud / "optts", "ts", text, local, post)
    steps, final = [st], "optts"
    if st["infra"]:
        return dict(attempts=[_att(st)], core_h=st["core_h"], core_h_optts=st["core_h"], core_h_sp=0.0,
                    orca_version=st["orca_version"], status="error", reason="pipeline_infra", attempt_used=1)
    reopt = bool(not st["converged"] and st["n_opt_cycles"] >= 2 and st.get("de_last_kcal") is not None
                 and st["de_last_kcal"] <= REOPT_DE_KCAL and (ud / "optts" / "ts.xyz").is_file())
    if reopt:                                                      # O-9: autodE _reoptimise, once, from the last geometry
        x = pc.read_xyz(ud / "optts" / "ts.xyz")
        st2 = orca_step(D, ud / "optts_reopt", "ts", partb_text(D, x[0], x[1], q, 1, R["pairs"], PARTB_NPROCS), local,
                        post)
        steps.append(st2)
        final = "optts_reopt"
    st = steps[-1]
    core = sum(s["core_h"] for s in steps)
    out = dict(attempts=[_att(s) for s in steps], core_h=core, core_h_optts=core, core_h_sp=0.0,
               orca_version=st["orca_version"], reopt=reopt, attempt_used=len(steps),
               de_last_kcal_first=steps[0].get("de_last_kcal"))
    if st["infra"]:
        return dict(out, status="error", reason="pipeline_infra")
    gt = st["gate"]
    lb, sh = gt.get("label") or {}, gt.get("start") or {}
    out.update(gate=gt, converged_to_label=lb.get("same_ts"), rmsd_label_A=lb.get("rmsd_A"),
               dform_label_A=lb.get("dform_A"), same_as_g2_start=sh.get("same_ts"), rmsd_g2_A=sh.get("rmsd_A"),
               e_optts_eh=st["e_last_eh"], e_label_eh=R["e_ab_eh"], final_dir=final, dE_kcal=None, e_sp_eh=None)
    if not gt["ok"]:
        return dict(out, status="fail", reason=gt["reason"])
    if lb.get("same_ts"):
        return dict(out, status="ok", reason="converged_to_label")
    mode = t5_mode()
    if mode is None:
        raise StopAll("Part B SP needs a passing T-5 (results/T_5.json)")
    bts = pc.read_xyz(ud / final / "ts.xyz")
    spt = sp_text(D, mode, eda_text(T), *bts)
    sp = orca_step(D, ud / "sp", "sp", spt, local, lambda w, t, s: post_sp(D, t, s))
    out["core_h"] += sp["core_h"]
    out["core_h_sp"] = sp["core_h"]
    out["attempts"].append(_att(sp))
    out["sp_mode"] = mode
    if sp["infra"]:
        return dict(out, status="error", reason="pipeline_infra_sp")
    if not sp["sp"]["ok"]:
        return dict(out, status="fail", reason="sp_" + sp["sp"]["reason"])
    out.update(e_sp_eh=sp["sp"]["e_eh"], dE_kcal=(sp["sp"]["e_eh"] - R["e_ab_eh"]) * C.KCAL)
    return dict(out, status="ok", reason="")


def t5_units(plan, mode):
    return [dict(uid=f"T5.{x['job_id']}.{mode}", kind="T5", method=None, rxn_id=x["rxn_id"], job_id=x["job_id"], mid=None,
                 n_atoms=x["n_atoms"], phase="t5", mode=mode, weight=float(x["n_atoms"]) ** 3) for x in plan["t5"]]


def unit_t5(u, D, plan, nprocs, local):
    R = plan["rxns"][str(u["rxn_id"])]
    lts = D["pc"].read_xyz(D1_JOBS / u["job_id"] / "ts.xyz")
    text = sp_text(D, u["mode"], eda_text(u["job_id"]), *lts)
    st = orca_step(D, UNITS / u["uid"] / "sp", "sp", text, local, lambda w, t, s: post_sp(D, t, s))
    out = dict(mode=u["mode"], core_h=st["core_h"], e_label_eh=R["e_ab_eh"], attempts=[_att(st)],
               orca_version=st["orca_version"])
    if st["infra"]:
        return dict(out, status="error", reason="pipeline_infra")
    sp = st["sp"]
    if not sp["ok"]:
        return dict(out, status="fail", reason="sp_" + sp["reason"], e_sp_eh=sp["e_eh"])
    d = (sp["e_eh"] - R["e_ab_eh"]) * C.KCAL
    ok = abs(d) <= T5_TOL
    return dict(out, status="ok" if ok else "fail", reason="" if ok else "differs_from_label", e_sp_eh=sp["e_eh"],
                diff_kcal=d, eps=sp["eps"])


UNIT_FN = {"ts": unit_ts, "ref": unit_ref, "B": unit_b, "T5": unit_t5}


def run_unit(u, D, plan, nprocs, local):
    ud = UNITS / u["uid"]
    t0, started = time.time(), _now()
    try:
        res = UNIT_FN[u["kind"]](u, D, plan, nprocs, local / u["uid"].replace(".", "_"))
    except StopAll:
        raise
    except WallClipped as e:                                       # O-19: a method cost failure, not a pipeline error
        spent = [_rj(p) or {} for p in ud.glob("*/step.json")] if ud.is_dir() else []
        res = dict(status="fail", reason="wall_clipped", error=str(e)[:600],
                   core_h=sum(float(s.get("core_h") or 0) for s in spent))
    except Exception as e:                                         # noqa: BLE001
        spent = [_rj(p) or {} for p in ud.glob("*/step.json")] if ud.is_dir() else []
        res = dict(status="error", reason=f"pipeline_{type(e).__name__}", error=str(e)[:600],
                   traceback=traceback.format_exc()[-5000:], core_h=sum(float(s.get("core_h") or 0) for s in spent))
    res.update(uid=u["uid"], kind=u["kind"], method=u.get("method"), rxn_id=u.get("rxn_id"), mid=u.get("mid"),
               job_id=u.get("job_id"), n_atoms=u.get("n_atoms"), phase=u.get("phase"), started=started,
               finished=_now(), finished_ts=time.time(), wall_s=time.time() - t0, slurm_job=_jobtag(),
               host=socket.gethostname(), code=CODE_VERSION)
    res.setdefault("core_h", 0.0)
    res["bytes"] = sum(p.stat().st_size for p in ud.rglob("*") if p.is_file()) if ud.is_dir() else 0
    C.write_json(ud / "unit.json", res)
    print(f"  [{_now()}] {u['uid']}: {res['status']} {res.get('reason') or ''} core_h={res['core_h']:.2f}", flush=True)
    return res


# ================================================================ claims, statistics, workers
def claim(D, uid):
    p = C.assert_path(LOCKS / f"{uid}.lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    me = _jobtag()
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(f"{me} {_now()} {socket.gethostname()} {os.getpid()}\n")
        return True
    except FileExistsError:
        pass
    try:
        owner = p.read_text().split()[0]
    except (OSError, IndexError):
        owner = ""
    if owner == me:
        return False
    if not owner:                                                  # empty lock (writer died after O_EXCL): 1 h grace
        try:
            mt = p.stat().st_mtime
            if time.time() - mt < 3600:
                return False
        except OSError:
            return False
        tag = f"empty{int(mt)}"
    elif D["RV"]._job_alive(owner):
        return False
    else:
        tag = _san(owner)
    # take-over: only the job that creates <uid>.lock.takeover.<dead owner> (O_EXCL) may replace the lock, so two jobs
    # that both saw the same dead owner never both run the unit
    try:
        fd = os.open(p.with_name(f"{p.name}.takeover.{tag}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(f"{me} {_now()} {socket.gethostname()} {os.getpid()}\n")
    except FileExistsError:
        return False
    C.write_atomic(p, f"{me} {_now()} {socket.gethostname()} {os.getpid()} (took over from {owner or tag})\n")
    return True


def release(uid):
    p = LOCKS / f"{uid}.lock"
    try:
        if p.read_text().split()[0] == _jobtag():
            p.unlink()
    except (OSError, IndexError):
        pass


def load_results(plan):
    out = {}
    for u in plan["units"] + t5_units(plan, "sp") + t5_units(plan, "eda"):
        r = _rj(UNITS / u["uid"] / "unit.json")
        if r is not None:
            out[u["uid"]] = r
    return out


def o_stats(plan, res):
    units = plan["units"]
    fin = [res[u["uid"]] for u in units if u["uid"] in res]
    n_err = sum(r["status"] == "error" for r in fin)
    first = {}
    for m in C.METHODS:
        ts = sorted((r for r in fin if r["kind"] == "ts" and r["method"] == m), key=lambda r: r.get("finished_ts", 0))
        head = ts[:FIRST_N]
        rate = sum(r["status"] == "ok" for r in head) / len(head) if head else None
        first[m] = dict(n=len(head), n_ok=sum(r["status"] == "ok" for r in head), rate=rate,
                        stop=len(head) == FIRST_N and rate < FIRST_MIN, uids=[r["uid"] for r in head])
    nbytes = sum(int(r.get("bytes") or 0) for r in res.values())
    return dict(n_planned=len(units), n_finished=len(fin), n_error=n_err,
                error_rate_finished=n_err / len(fin) if fin else 0.0, error_rate_planned=n_err / len(units),
                early_error_stop=n_err > ERR_MAX * len(units), first20=first, bytes=nbytes,
                budget_stop=nbytes > BUDGET_GB * 1e9)


def stopped_methods():
    return {m for m in C.METHODS if (O / f"STOP_{m}.json").is_file()}


def write_stop(msg):
    C.write_atomic(STOP_FILE, f"{_now()} {msg}\n")
    print(f"STOP: {msg}", file=sys.stderr, flush=True)


class Worker:
    def __init__(self, D, plan, units, nprocs, local):
        self.D, self.plan, self.nprocs, self.local = D, plan, nprocs, local
        self.queue = sorted(units, key=lambda u: (-u["weight"], u["uid"]))
        self.i, self.mu, self.t_stats, self.stopped, self.halt = 0, threading.Lock(), 0.0, set(), ""
        self.n_locked = 0

    def refresh(self):
        st = o_stats(self.plan, load_results(self.plan))
        for m, f in st["first20"].items():
            if f["stop"] and not (O / f"STOP_{m}.json").is_file():
                C.write_json(O / f"STOP_{m}.json", dict(written=_now(), method=m, rule="spec 12 O first 20 < 0.5", **f))
                print(f"STOP {m}: TS success {f['n_ok']}/{f['n']} in its first {FIRST_N} TS units", file=sys.stderr,
                      flush=True)
        self.stopped = stopped_methods()
        if st["early_error_stop"]:
            self.halt = f"pipeline errors {st['n_error']} > {ERR_MAX:.0%} of {st['n_planned']} planned units"
        elif st["budget_stop"]:
            self.halt = f"stored bytes {st['bytes'] / 1e9:.2f} GB > D1G3_O_BUDGET_GB {BUDGET_GB}"
        if self.halt and not STOP_FILE.is_file():
            write_stop(self.halt)
        self.t_stats = time.time()

    def next_unit(self):
        with self.mu:
            while self.i < len(self.queue):
                if time.time() - T0 > STOP_AFTER_H * 3600:
                    return None
                if STOP_FILE.is_file():
                    return None
                if time.time() - self.t_stats > 300:
                    self.refresh()
                if self.halt:
                    return None
                u = self.queue[self.i]
                self.i += 1
                if u.get("method") in self.stopped or (UNITS / u["uid"] / "unit.json").is_file():
                    continue
                if not claim(self.D, u["uid"]):
                    self.n_locked += 1
                    continue
                if (UNITS / u["uid"] / "unit.json").is_file():
                    release(u["uid"])
                    continue
                return u
            return None

    def loop(self):
        while True:
            u = self.next_unit()
            if u is None:
                return
            try:
                run_unit(u, self.D, self.plan, self.nprocs, self.local)
            except StopAll as e:                                   # the other threads see it before their next claim
                with self.mu:
                    self.halt = str(e)
                    if not STOP_FILE.is_file():
                        write_stop(str(e))
                raise
            finally:
                release(u["uid"])


def _local():
    return Path(os.environ.get("TMPDIR", "/tmp")) / f"d1g3o_{_jobtag()}_{os.getpid()}"


def run_pool(D, plan, units, nprocs, local):
    """claim + run units, cpus // 8 at a time (tests, one)."""
    W = Worker(D, plan, units, nprocs, local)
    W.queue = list(units)
    W.t_stats = time.time() + 1e9                                  # no O statistics here
    k = max(1, min(len(units), _cpus() // SLOT_CORES))
    with ThreadPoolExecutor(k) as ex:
        for f in [ex.submit(W.loop) for _ in range(k)]:
            f.result()


def t_gate():
    """O-20: (0, '') when T-1 .. T-5 all passed (T_4.json: all_pass; the others: pass), (2, why) when one is missing /
    still running, (3, why) when one failed (spec 12 T STOP)."""
    st = {}
    for k, p in T_JSON.items():
        r = _rj(p)
        if r is None:
            st[k] = "missing"
        elif k == 4:
            st[k] = "pass" if r.get("all_pass") else "fail"
        elif r.get("pass"):
            st[k] = "pass"
        elif r.get("verdict") in ("running", "not_ready"):
            st[k] = str(r["verdict"])
        else:
            st[k] = "fail"
    bad = [f"T-{k}" for k, v in st.items() if v == "fail"]
    if bad:
        return 3, f"{bad} failed (spec 12 T STOP; results/T_<k>.json)"
    open_ = [f"T-{k} {v}" for k, v in st.items() if v != "pass"]
    if open_:
        return 2, f"{open_} not passed yet (spec 3: every T test before the O computation)"
    return 0, ""


def ready(plan, phase):
    """(0, '') if the phase may run, else (2 = not yet | 3 = STOP, the reason)."""
    ck = _rj(CHECK)
    if not (ck and ck.get("ok") and ck.get("plan_fingerprint") == plan["fingerprint"]):
        return 3, "check.json missing / not ok / for another plan"
    code, why = t_gate()
    if code:
        return code, why
    if STOP_FILE.is_file():
        return 3, f"{STOP_FILE} exists: {STOP_FILE.read_text().strip()}"
    # O-23 (user 2026-10-24): main no longer waits for the cost estimate / 3,000 core-h limit (estimate not run)
    return 0, ""


def cmd_work(phase, nprocs, only=None):
    if phase not in ("pilot", "main", "sel"):
        raise SystemExit(__doc__)
    D, plan = d1(), load_plan()
    code, why = ready(plan, phase)
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: work {phase}: {why}", file=sys.stderr, flush=True)
        return code
    if phase == "sel":                                             # U-1 (user 2026-10-25): only the listed plan units
        want = {ln.split()[0] for ln in Path(only).read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        units = [u for u in plan["units"] if u["uid"] in want]
        if len(units) != len(want):
            print(f"STOP: work sel: {sorted(want - {u['uid'] for u in units})} are not plan units", file=sys.stderr)
            return 3
    else:
        units = [u for u in plan["units"] if phase == "main" or u["phase"] in ("t1", "pilot")]
    local = _local()
    W = Worker(D, plan, units, nprocs, local)
    k = max(1, _cpus() // SLOT_CORES)
    print(f"[work {phase}] {len(units)} units, {k} slots x {SLOT_CORES} cores, job {_jobtag()} on {socket.gethostname()}",
          flush=True)
    rc = 0
    try:
        with ThreadPoolExecutor(k) as ex:
            for f in [ex.submit(W.loop) for _ in range(k)]:
                try:
                    f.result()
                except StopAll as e:
                    write_stop(str(e))
                    rc = 3
    finally:
        shutil.rmtree(local, ignore_errors=True)
    if rc or STOP_FILE.is_file():
        return 3
    left = [u["uid"] for u in units if u.get("method") not in stopped_methods()
            and not (UNITS / u["uid"] / "unit.json").is_file()]
    print(f"[work {phase}] remaining {len(left)} (locked by other jobs this pass {W.n_locked}); "
          f"stopped methods {sorted(stopped_methods())}", flush=True)
    if stopped_methods():
        return 3
    return 2 if left else 0


def cmd_one(uid, nprocs):
    D, plan = d1(), load_plan()
    units = {u["uid"]: u for u in plan["units"] + t5_units(plan, "sp") + t5_units(plan, "eda")}
    if uid not in units:
        print(f"unknown unit {uid}", file=sys.stderr)
        return 2
    local = _local()
    try:
        run_pool(D, plan, [units[uid]], nprocs, local)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    print(json.dumps(_rj(UNITS / uid / "unit.json"), indent=1, default=str)[:4000])
    return 0


# ================================================================ check (G3ENV=d1)
def cmd_check():
    D, plan = d1(), load_plan()
    rm, pc, OD = D["rm"], D["pc"], D["OD"]
    cr, cv = D["cfg_run"], D["cfg_val"]
    rep = dict(code=CODE_VERSION, started=_now(), host=socket.gethostname(), plan_fingerprint=plan["fingerprint"])
    bad = []
    # D1 config: the frozen blocks of config_run.yaml = config_val.yaml; values the inputs here rely on
    t = cr["ts_stage"]
    want = dict(engine="orca", functional="B3LYP", dispersion="D3BJ", opt_basis="def2-SVP", sp_basis="def2-TZVP",
                solvent="water", solvation="smd", orca_optts_block=D1_OPTTS_BLOCK)
    for k, v in want.items():
        if t.get(k) != v:
            bad.append(f"config_run ts_stage.{k} = {t.get(k)!r} != {v!r}")
    if cr["ts_stage"] != cv["ts_stage"] or cr["gates"] != cv["gates"]:
        bad.append("config_run ts_stage / gates != config_val (frozen copy)")
    if int(cr["maxcore_mb"]) != MAXCORE or int(cv["maxcore_mb"]) != MAXCORE or int(cr["cpus_per_task"]) != SLOT_CORES:
        bad.append("maxcore_mb / cpus_per_task != 3000 / 8")
    if cr["sp_stage"]["frag_solv_block"] != SOLV_BLOCK:
        bad.append("config_run sp_stage.frag_solv_block != the %cpcm block used here")
    sig = inspect.signature(rm.same_ts).parameters
    thr = dict(run=cr["run"]["same_ts"], prereg=cv["prereg"]["same_ts"],
               defaults=dict(rmsd_A=sig["rmsd_A"].default, dform_A=sig["dform_A"].default))
    if not all(float(x["rmsd_A"]) == 0.10 and float(x["dform_A"]) == 0.02 for x in thr.values()):
        bad.append(f"same-TS thresholds differ: {thr}")
    rep["config"] = dict(config_run=str(CFG_RUN), config_run_sha256=C.sha256(CFG_RUN), config_val=str(CFG_VAL),
                         config_val_sha256=C.sha256(CFG_VAL), same_ts=thr, gates=cr["gates"])
    if C.sha256(G2C.MANIFEST) != G2C.MANIFEST_SHA:
        bad.append("d1_manifest.csv sha256 changed")
    # autodE keyword strings (configure_autode) = the D1 run record
    try:
        ade = pc.configure_autode(cr, cores=SLOT_CORES)
        kw = pc.hmethod_keywords_summary()
        rec = json.loads(KW_RECORD.read_text())["keywords"]
        rep["autode"] = dict(version=ade.__version__, opt_ts=kw["opt_ts"], hess=kw["hess"], record=str(KW_RECORD),
                             opt_ts_equal=kw["opt_ts"] == rec["opt_ts"], hess_equal=kw["hess"] == rec["hess"])
        if not (ade.__version__ == "1.4.5" and rep["autode"]["opt_ts_equal"] and rep["autode"]["hess_equal"]):
            bad.append("autodE opt_ts / hess keywords (configure_autode) != the D1 ts_result.json record, or not 1.4.5")
    except Exception as e:                                         # noqa: BLE001
        bad.append(f"autodE keyword check: {type(e).__name__}: {e}")
    # Part B text proof (O-9)
    rep["text_sources"], n_d1 = [], 0
    for src in TEXT_SOURCES:
        r = dict(name=src["name"])
        try:
            if "tar" in src:
                with tarfile.open(src["tar"], "r:*") as T:
                    txt = T.extractfile(T.getmember(src["member"])).read().decode()
                r["file"] = f"{src['tar']}:{src['member']}"
            else:
                p = Path(cv["val"]["pilot_optts_inp"])
                txt = p.read_text()
                r["file"] = str(p)
            s, x, q, mlt, pairs, npr = parse_autode_optts(txt)
            regen = partb_text(D, s, x, q, mlt, pairs, npr)
            r.update(n_atoms=len(s), pairs=pairs, nprocs=npr, identical=regen == txt, sha256=C.DP.sha256_text(txt))
            if not r["identical"]:
                a, b = txt.splitlines(), regen.splitlines()
                k = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
                r["first_diff"] = dict(line=k, archive=a[k] if k < len(a) else None, regen=b[k] if k < len(b) else None)
                bad.append(f"Part B text != {r['file']}")
            elif src["d1_run"]:
                n_d1 += 1
        except Exception as e:                                     # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {e}"
            bad.append(f"Part B text source {src['name']}: {r['error']}")
        rep["text_sources"].append(r)
    if n_d1 < 1:
        bad.append("no D1 run OptTS input reproduced byte for byte")
    # environment (ORCA 6.1.1 binary, OpenMPI 4.1.8f of env.sh, MPI variables)
    env = {k: os.environ.get(k) for k in ("ORCA_BIN", "MPI_ROOT", "OMPI_MCA_pml", "OMPI_MCA_btl", "UCX_TLS",
                                          "OMPI_MCA_coll_hcoll_enable", "OMPI_MCA_rmaps_base_oversubscribe")}
    mr = shutil.which("mpirun")
    env.update(orca=D["orca"], orca_executable=os.access(D["orca"], os.X_OK), mpirun=mr,
               mpirun_under_MPI_ROOT=bool(mr and env["MPI_ROOT"]
                                          and Path(env["MPI_ROOT"]).resolve() in Path(mr).resolve().parents),
               binding_policy_for_orca=D["env"].get("OMPI_MCA_hwloc_base_binding_policy"))
    want_env = dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", UCX_TLS="tcp,self,sm",
                    OMPI_MCA_coll_hcoll_enable="0")
    if any(env[k] != v for k, v in want_env.items()) or not env["orca_executable"] or not env["mpirun_under_MPI_ROOT"]:
        bad.append(f"environment: {env}")
    if Path(D["orca"]).resolve() != Path(cr["orca_bin"]).resolve():
        bad.append(f"ORCA binary {D['orca']} != config_run orca_bin {cr['orca_bin']}")
    rep["env"] = env
    rep["modules"] = dict(fragmenter=D["fr"].__file__, rmsd=D["rm"].__file__, run_ts=D["RT"].__file__,
                          orca_direct=OD.__file__, pilot_common=pc.__file__, refs_v=D["RV"].__file__,
                          prep=D["HP"].__file__, run=D["HR"].__file__)
    # inputs: Part A texts (one per method / attempt), Part B / T-5 SP texts in both modes
    inp_bad = []
    local = _local()
    try:
        for rid in sorted(set(plan["part_a"]) | set(plan["part_b"])):
            R = plan["rxns"][str(rid)]
            try:
                g2ts = pc.read_xyz(G2_FINAL / str(rid) / "ts.xyz")
                if R["part_a"] and rid == plan["t1_rxn"]:
                    for m in C.METHODS:
                        for att in (1, 2):
                            inp_text(D, ts_header(m, att, SLOT_CORES), *g2ts, R["charge1"] + R["charge2"], 1, local)
                if R["part_b"]:
                    partb_text(D, g2ts[0], g2ts[1], R["charge1"] + R["charge2"], 1, R["pairs"], PARTB_NPROCS)
                    for mode in ("sp", "eda"):
                        sp_text(D, mode, eda_text(R["job_id"]), *g2ts)
            except Exception as e:                                 # noqa: BLE001
                inp_bad.append(f"{rid}: {type(e).__name__}: {e}")
        for x in plan["t5"]:
            lts = pc.read_xyz(D1_JOBS / x["job_id"] / "ts.xyz")
            for mode in ("sp", "eda"):
                try:
                    sp_text(D, mode, eda_text(x["job_id"]), *lts)
                except Exception as e:                             # noqa: BLE001
                    inp_bad.append(f"T5 {x['job_id']} {mode}: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(local, ignore_errors=True)
    if inp_bad:
        bad.append(f"{len(inp_bad)} input constructions failed: {inp_bad[:5]}")
    rep.update(input_problems=inp_bad, bad=bad, ok=not bad, finished=_now())
    C.write_json(CHECK, rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("text_sources",)}, indent=1, default=str)[:6000],
          flush=True)
    return 3 if bad else 0


# ================================================================ tests T-1 / T-5 (G3ENV=d1)
def t1_record(D, res, uid, m):
    r = res.get(uid)
    out = dict(uid=uid, method=m, unit_status=None if r is None else r["status"])
    if r is None:
        return dict(out, pass_=False, why="unit missing")
    a = r.get("final_dir") or "a1"
    sj = _rj(UNITS / uid / a / "step.json") or {}
    op = UNITS / uid / a / "ts.out.gz"
    txt = D["H"].read_text(op) if op.is_file() else ""
    g = sj.get("gate") or {}
    toks = (sj.get("keyword_line") or "").replace("!", " ").split()
    marks = {w: w in toks for w in METHOD_MARK[m]}                 # the input keyword line
    want = LEVEL_WANT[m]                                           # O-21: what the output (not the echo) reports
    body = RE_ECHO.sub("", txt)
    d3 = (D["OD"].level_checks(txt).get("d3") or {}) if txt else {}
    method_ok = sj.get("method") == [want["method"]]
    # O-22 (2026-10-24, after T-1 failed on g3r only for this): the orbital basis (first basis line) must equal the
    # wanted one and every wanted line must be printed; extra auxiliary lines (r2SCAN-3c prints def2-mTZVPP/J) are fine
    bl = sj.get("basis") or []
    basis_ok = bool(bl) and bl[0] == want["basis"][0] and all(b in bl for b in want["basis"])
    d3_ok = (len(d3) == 4) if want["d3bj"] else True               # = orca_direct.level_ok d3_present (D1 L0)
    func_ok = want["functional"] is None or re.search(want["functional"], body, re.I) is not None
    out.update(attempt_used=r.get("attempt_used"), terminated=sj.get("terminated"), converged=sj.get("converged"),
               n_imag=g.get("n_imag"), imag_freqs_cm=g.get("imag_freqs_cm"), one_imag=g.get("n_imag") == 1,
               eps=sj.get("eps"), eps_ok=bool(sj.get("eps")) and all(e == EPS_SMD_WATER for e in sj.get("eps")),
               smd_cds=sj.get("smd_cds"), smd_module=sj.get("smd_module"), keyword_line=sj.get("keyword_line"),
               method_lines=sj.get("method"), basis_lines=sj.get("basis"), marks=marks, want_level=want,
               method_ok=method_ok, basis_ok=basis_ok, d3=d3, d3_ok=d3_ok, functional_in_output=func_ok,
               marks_in_output=method_ok and basis_ok and d3_ok and func_ok, orca_version=sj.get("orca_version"),
               gate_reason=g.get("reason"), core_h=r.get("core_h"))
    # spec T-1 criteria (the TS gate result is recorded, it is a Part A data point and not a T-1 criterion)
    out["pass_"] = bool(r["status"] != "error" and out["terminated"] and out["converged"] and out["one_imag"]
                        and out["eps_ok"] and out["smd_cds"] and all(marks.values()) and out["marks_in_output"]
                        and out["orca_version"] == D["H"].ORCA_VERSION)
    out["ts_gate_ok"] = r["status"] == "ok"
    return out


def cmd_tests(nprocs):
    D, plan = d1(), load_plan()
    ck = _rj(CHECK)
    if not (ck and ck.get("ok") and ck.get("plan_fingerprint") == plan["fingerprint"]):
        print("STOP: tests need a passing check for this plan ($G3/O/check.json)", file=sys.stderr)
        return 3
    t1 = plan["t1_rxn"]
    tsu = [u for u in plan["units"] if u["kind"] == "ts" and u["rxn_id"] == t1]
    local = _local()
    try:
        run_pool(D, plan, tsu + t5_units(plan, "sp"), nprocs, local)
        res = load_results(plan)
        sp_rows = [res.get(u["uid"]) for u in t5_units(plan, "sp")]
        mode = "sp" if all(r is not None and r["status"] == "ok" for r in sp_rows) else None
        eda_rows = []
        if mode is None and all(r is not None and r["status"] != "error" for r in sp_rows):
            run_pool(D, plan, t5_units(plan, "eda"), nprocs, local)
            res = load_results(plan)
            eda_rows = [res.get(u["uid"]) for u in t5_units(plan, "eda")]
            mode = "eda" if all(r is not None and r["status"] == "ok" for r in eda_rows) else None
    except StopAll as e:
        write_stop(str(e))
        return 3
    finally:
        shutil.rmtree(local, ignore_errors=True)
    keep = ("uid", "status", "reason", "mode", "e_sp_eh", "e_label_eh", "diff_kcal", "eps", "core_h")
    t5 = dict(spec="D1G3.md 3 T-5", tol_kcal=T5_TOL, written=_now(), mode=mode, pass_=mode is not None,
              energy_key="labels_d1.json e_ab_eh (= stage3_parse FSPE of the label eda.out)",
              sp=[{k: (r or {}).get(k) for k in keep} for r in sp_rows],
              eda=[{k: (r or {}).get(k) for k in keep} for r in eda_rows], jobs=plan["t5"])
    t5["pass"] = t5.pop("pass_")
    C.write_json(T5_JSON, t5)
    recs = [t1_record(D, res, u["uid"], u["method"]) for u in tsu]
    for r in recs:
        r["pass"] = r.pop("pass_")
    t1j = dict(spec="D1G3.md 3 T-1", rxn_id=t1, job_id=plan["rxns"][str(t1)]["job_id"], written=_now(), methods=recs,
               pass_=all(r["pass"] for r in recs) and len(recs) == len(C.METHODS))
    t1j["pass"] = t1j.pop("pass_")
    C.write_json(T1_JSON, t1j)
    if mode:
        record_decision("O-T5", f"T-5 통과 방식 = '{mode}' (Part B SP가 이 방식을 쓴다; results/T_5.json)")
    print(json.dumps(dict(T1=t1j["pass"], T5=t5["pass"], mode=mode), indent=1), flush=True)
    return 0 if (t1j["pass"] and t5["pass"]) else 3


# ================================================================ estimate (spec 7)
def _fam(rows, all_n):
    """D1LADDER V15 rule: max(mean pilot core-h x N, pilot core-h x sum(n^3) / sum over the pilot (n^3))."""
    rows = [(float(n), float(c)) for n, c in rows if c is not None and c > 0]
    if not rows or not all_n:
        return None
    mean = sum(c for _, c in rows) / len(rows)
    scale = sum(c for _, c in rows) / sum(n ** 3 for n, _ in rows)
    e_mean, e_n3 = mean * len(all_n), scale * sum(float(n) ** 3 for n in all_n)
    return dict(n_pilot=len(rows), n_all=len(all_n), mean_core_h=mean, est_mean=e_mean, est_n3=e_n3,
                est=max(e_mean, e_n3))


def cmd_estimate(retro=False):
    # O-23 (user 2026-10-24): no cost estimate and no 3,000 core-h limit; writes nothing and never writes an O STOP
    if not retro:
        print("estimate: skipped (decision O-23); main does not wait for it", flush=True)
        return 0
    # U-3a (user 2026-10-25): 'estimate retro' = the spec 7 / O-13 estimate the cancelled job would have written, from
    # the pilot units, report only -> results/O_estimate_retro.json (never an O STOP; the work is already stopped)
    plan = load_plan()
    code, why = t_gate()                                           # O-20: nothing is written while T has not passed
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: estimate: {why}", file=sys.stderr, flush=True)
        return code
    res = load_results(plan)
    units = plan["units"]
    early = [u for u in units if u["phase"] in ("t1", "pilot")]
    missing = [u["uid"] for u in early if u["uid"] not in res and u.get("method") not in stopped_methods()]
    rep = dict(spec="D1G3.md 7", written=_now(), plan_fingerprint=plan["fingerprint"], limit_core_h=CORE_H_MAX,
               missing=missing, go=False, stop=False, stop_reasons=[], families={})
    if missing:
        rep["complete"] = False
        C.write_json(EST_RETRO, rep)
        print(f"estimate: {len(missing)} pilot units missing: {missing[:10]}", file=sys.stderr)
        return 2
    rep["complete"] = True
    fam = rep["families"]
    for m in C.METHODS:
        for kind in ("ts", "ref"):
            us = [u for u in units if u["kind"] == kind and u["method"] == m]
            fam[f"{kind}_{m}"] = _fam([(u["n_atoms"], res[u["uid"]].get("core_h")) for u in us if u["uid"] in res
                                       and u["phase"] in ("t1", "pilot")], [u["n_atoms"] for u in us])
    ub = [u for u in units if u["kind"] == "B"]
    fam["B_optts"] = _fam([(u["n_atoms"], res[u["uid"]].get("core_h_optts")) for u in ub if u["uid"] in res
                           and u["phase"] == "pilot"], [u["n_atoms"] for u in ub])
    mode = t5_mode()
    sp_rows = [(u["n_atoms"], res[u["uid"]].get("core_h")) for u in t5_units(plan, mode or "sp") if u["uid"] in res]
    sp_rows += [(u["n_atoms"], res[u["uid"]].get("core_h_sp")) for u in ub if u["uid"] in res and u["phase"] == "pilot"
                and res[u["uid"]].get("core_h_sp")]
    fam["B_sp"] = _fam(sp_rows, [u["n_atoms"] for u in ub])
    nA = [plan["rxns"][str(r)]["n_atoms"] for r in plan["part_a"]]
    try:
        fit = json.loads(HF_PMETRICS.read_text())["cost"]["fit"]
        hf = sum(float(fit["a"]) * float(n) ** float(fit["b"]) for n in nA) * len(C.METHODS)
        fam["F_hf"] = dict(est=hf, source=str(HF_PMETRICS), fit=fit, n_rxn=len(nA), methods=len(C.METHODS))
    except (OSError, KeyError, ValueError) as e:
        rep["stop_reasons"].append(f"D1-HF cost fit unreadable: {type(e).__name__}: {e}")
    try:
        g0 = json.loads(G0_REPORT.read_text())["core_h"]
        f0 = json.loads(F0_ESTIMATE.read_text())["parts"]
        per = float(g0["per_rxn_sum"]) / float(g0["per_rxn"]["n"]) + sum(float(p["core_h_per_rxn"]) for p in f0.values())
        fam["F_xtb_features"] = dict(est=per * len(nA) * len(C.METHODS), per_rxn=per, sources=[str(G0_REPORT),
                                                                                                str(F0_ESTIMATE)])
    except (OSError, KeyError, ValueError, ZeroDivisionError) as e:
        rep["stop_reasons"].append(f"G0 / F0 cost records unreadable: {type(e).__name__}: {e}")
    spent_t5 = sum(float(r.get("core_h") or 0) for uid, r in res.items() if uid.startswith("T5."))
    fam["T5_spent"] = dict(est=spent_t5)
    nodata = [k for k, v in fam.items() if v is None]
    if nodata:
        rep["stop_reasons"].append(f"no pilot measurement for {nodata}")
    total = sum(v["est"] for v in fam.values() if v)
    st = o_stats(plan, res)
    rep.update(total_core_h=total, measured_core_h=sum(float(r.get("core_h") or 0) for r in res.values()), o_stats=st)
    if total > CORE_H_MAX:
        rep["stop_reasons"].append(f"expected {total:.0f} core-h > {CORE_H_MAX:.0f} (spec 7)")
    if st["early_error_stop"]:
        rep["stop_reasons"].append(f"pipeline errors {st['n_error']} > {ERR_MAX:.0%} of {st['n_planned']} units")
    if stopped_methods() or STOP_FILE.is_file():
        rep["stop_reasons"].append(f"O STOP files present: {sorted(stopped_methods())} {STOP_FILE.is_file()}")
    rep["stop"] = bool(rep["stop_reasons"])
    rep["go"] = not rep["stop"]
    rep["retro"] = ("U-3a: computed after the fact (2026-10-25); the estimate job 1044263 was cancelled before it ran "
                    "(O-23); measured_core_h counts every finished unit at this time, not only the pilot")
    C.write_json(EST_RETRO, rep)
    print(json.dumps({k: rep[k] for k in ("total_core_h", "go", "stop_reasons")}, indent=1, default=str), flush=True)
    for k, v in fam.items():
        print(f"  {k}: {None if v is None else round(v['est'], 1)}", flush=True)
    return 0


# ================================================================ U-1 selection (user 2026-10-25; G3ENV=d1)
U1_METHOD = "g3r"                                               # U-1a: only g3r reactions go on to Phase F / E
U1_JSON, SEL_REFS = C.RES / "U1_selection.json", C.RES / "O_sel_refs.txt"
STOPPED = "stopped_user"                                        # U-1b: O_status / O_refs status; marker .fail_<this>
STOPPED_MARK = ".fail_" + STOPPED
U1_KEYS = ("method", "plan_fingerprint", "S_pairs_sha256", "complete_pairs", "selected_rxns", "needed_refs")  # U-1a


def _counts(vals):
    out = {}
    for v in vals:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def u1_pairs(plan, res):
    """U-1a: ([complete pair records], number of S_pairs.csv pairs): the pairs whose ts.g3r.<a_rxn> and ts.g3r.<b_rxn>
    unit.json both have status 'ok' (pair_id order)."""
    S = pd.read_csv(S_PAIRS)
    for c in ("pair_id", "a_rxn", "b_rxn", "fold", "class", "dipole"):
        if c not in S.columns:
            C.stop(f"{S_PAIRS}: column {c} missing")
    pa = {int(r) for r in plan["part_a"]}
    out = []
    for r in sorted(S.to_dict("records"), key=lambda x: str(x["pair_id"])):   # (itertuples would rename 'class')
        a, b = int(r["a_rxn"]), int(r["b_rxn"])
        if a not in pa or b not in pa:
            C.stop(f"{S_PAIRS} {r['pair_id']}: {a} / {b} are not Part A reactions of the plan")
        if all((res.get(f"ts.{U1_METHOD}.{x}") or {}).get("status") == "ok" for x in (a, b)):
            out.append({"pair_id": str(r["pair_id"]), "a_rxn": a, "b_rxn": b, "fold": int(r["fold"]),
                        "class": str(r["class"]), "dipole": str(r["dipole"])})
    return out, len(S)


def u1_needed_refs(plan, rxns):
    """U-1a: the ref.g3r unit ids the reactions need (plan rxns mid1 / mid2 = the link assemble_a reads), sorted."""
    return sorted({f"ref.{U1_METHOD}.{plan['rxns'][str(r)][f'mid{k}']}" for r in rxns for k in (1, 2)})


def _live_ts_locks(D, res):
    """{method: [[uid, job]]}: TS units without unit.json whose lock owner is a SLURM job still in squeue."""
    out = {}
    for p in (sorted(LOCKS.glob("ts.*.lock")) if LOCKS.is_dir() else []):
        uid = p.name[:-len(".lock")]
        if uid in res:
            continue
        try:
            owner = p.read_text().split()[0]
        except (OSError, IndexError):
            continue
        if owner.isdigit() and D["RV"]._job_alive(owner):
            out.setdefault(uid.split(".")[1], []).append([uid, owner])
    return out


def cmd_select():
    """U-1a: results/U1_selection.json + results/O_sel_refs.txt (decision U-1a). U1_selection.json is rewritten only
    when a key that defines the selection (U1_KEYS) changes, so a rerun - also after the needed refs have run - keeps
    the selection sha256 that assemble and the later steps record; its missing_refs / ts_units_finished are the values
    at 'written'. O_sel_refs.txt always lists the needed refs missing now (rewritten when that list changes)."""
    D, plan = d1(), load_plan()
    code, why = t_gate()                                           # O-20: nothing is written while T has not passed
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: select: {why}", file=sys.stderr, flush=True)
        return code
    res = load_results(plan)
    live = _live_ts_locks(D, res)
    if live.get(U1_METHOD):
        jobs = sorted({j for _, j in live[U1_METHOD]})
        print(f"NOT READY: select: {len(live[U1_METHOD])} {U1_METHOD} TS units are still held by live jobs {jobs} "
              f"(U-1 item 1: scancel them first)", file=sys.stderr, flush=True)
        return 2
    for m, v in live.items():                                      # other methods: reported, never selected
        print(f"warning: {len(v)} {m} TS units still held by live jobs {sorted({j for _, j in v})} (U-1 item 1)",
              file=sys.stderr, flush=True)
    record_decision("U-1a", DECISIONS["U-1a"])
    pairs, n_all = u1_pairs(plan, res)
    rxns = sorted({p[k] for p in pairs for k in ("a_rxn", "b_rxn")})
    need = u1_needed_refs(plan, rxns)
    known = {u["uid"] for u in plan["units"]}
    if any(u not in known for u in need):
        C.stop(f"needed references {sorted(set(need) - known)} are not plan units")
    missing = [u for u in need if u not in res]
    ts_fin = {m: _counts(r["status"] for r in res.values() if r.get("kind") == "ts" and r.get("method") == m)
              for m in C.METHODS}
    sel = dict(spec="D1G3 U-1 (user 2026-10-25, items 1 and 5)", decision="U-1a", method=U1_METHOD,
               rule=f"S_pairs.csv pairs whose ts.{U1_METHOD}.<a_rxn> and ts.{U1_METHOD}.<b_rxn> unit.json status are "
                    f"both 'ok'", plan_fingerprint=plan["fingerprint"], S_pairs_sha256=C.sha256(S_PAIRS),
               n_pairs_sample=n_all, n_complete_pairs=len(pairs), n_by_class=_counts(p["class"] for p in pairs),
               n_by_dipole=_counts(p["dipole"] for p in pairs), complete_pairs=pairs, n_selected_rxns=len(rxns),
               selected_rxns=rxns, needed_refs=need, missing_refs=missing, ts_units_finished=ts_fin,
               note=f"missing_refs / ts_units_finished = at 'written'; the file is kept while {list(U1_KEYS)} are "
                    f"unchanged (U-1a); the refs missing now are in the refs_file",
               refs_file=str(SEL_REFS.relative_to(HERE)))
    new = json.loads(json.dumps(sel, default=C._jsonable))
    old = _rj(U1_JSON)
    if old is None or any(old.get(k) != new[k] for k in U1_KEYS):  # U-1a: same selection -> same file / sha256
        C.write_json(U1_JSON, dict(new, written=_now()))
    txt = ("# U-1a (opt.py select): ref.g3r units the selected reactions of results/U1_selection.json need that have "
           "no unit.json now\n# run: opt.py work sel --only results/O_sel_refs.txt\n"
           + "".join(f"{u}\n" for u in missing))
    if not SEL_REFS.is_file() or SEL_REFS.read_text() != txt:
        C.write_atomic(SEL_REFS, txt)
    print(json.dumps(dict(n_complete_pairs=len(pairs), n_by_class=sel["n_by_class"], n_selected_rxns=len(rxns),
                          n_needed_refs=len(need), missing_refs=missing, ts_units_finished=ts_fin,
                          selection_sha256=C.sha256(U1_JSON)), indent=1), flush=True)
    if not pairs:
        print(f"STOP: select: no complete {U1_METHOD} pair (U-1a); nothing to carry to Phase F / E", file=sys.stderr,
              flush=True)
        return 3
    return 0


def u1_selection(plan, res):
    """U-1b: results/U1_selection.json as 'opt.py select' wrote it -> (selection + rxns / needed / sha256, 0, '') or
    (None, 2 | 3, why). The selection is read, not recomputed: the check only confirms that the file still describes
    the finished units (the same pairs are complete now), so select and assemble cannot disagree."""
    s = _rj(U1_JSON)
    if s is None:
        return None, 2, f"{U1_JSON} missing: run 'opt.py select' first (U-1)"
    if (s.get("method") != U1_METHOD or s.get("plan_fingerprint") != plan["fingerprint"]
            or s.get("S_pairs_sha256") != C.sha256(S_PAIRS)):
        return None, 2, f"{U1_JSON} was written for another plan / S_pairs.csv / method: rerun 'opt.py select'"
    cp = s.get("complete_pairs") or []
    got, now = [p["pair_id"] for p in cp], [p["pair_id"] for p in u1_pairs(plan, res)[0]]
    if now != got:
        return None, 2, (f"complete {U1_METHOD} pairs changed since 'opt.py select' (file {len(got)}, now {len(now)}: "
                         f"{sorted(set(got) ^ set(now))[:10]}): rerun 'opt.py select'")
    rxns = sorted(int(r) for r in s.get("selected_rxns") or [])
    need = sorted(s.get("needed_refs") or [])
    if (not rxns or rxns != sorted({int(p[k]) for p in cp for k in ("a_rxn", "b_rxn")})
            or not set(rxns) <= set(plan["part_a"]) or need != u1_needed_refs(plan, rxns)):
        return None, 3, (f"{U1_JSON}: no complete pair (U-1a STOP), or selected_rxns / needed_refs do not follow from "
                         f"its complete_pairs and the plan")
    return dict(s, rxns=set(rxns), needed=set(need), sha256=C.sha256(U1_JSON)), 0, ""


# ================================================================ assemble (G3ENV=d1)
def g2a_module(D):
    """g2_autode.remap with g2_refinalize's fixed _graph (import-time patches of g2_autode undone afterwards)."""
    import make_reference as MR  # noqa: E402                     (D1 pilot copy; cached by name)
    keep = (D["pc"].configure_autode, MR.dft_refine)
    import g2_refinalize as GR  # noqa: E402                      (imports g2_autode and sets GA._graph = _graph_fixed)
    D["pc"].configure_autode, MR.dft_refine = keep
    GA = GR.GA
    if GA._graph is not GR._graph_fixed:
        C.stop("g2_autode._graph is not the g2_refinalize fixed function")
    return GA


def _reason_a(prefix, r):
    if r["status"] == "error":
        rs = str(r.get("reason") or "error")
        return "pipeline_" + (prefix + "_" if prefix != "ts" else "") + (rs[len("pipeline_"):] if rs.startswith(
            "pipeline_") else rs)
    return f"{prefix}_{r.get('reason') or 'fail'}"


def _write_folder(dest, files, result, reasons):
    """dest.tmp -> dest (atomic per folder; marker written last inside the tmp folder)."""
    tmp = C.assert_path(dest.with_name(dest.name + ".tmp"))
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, text in files.items():
        (tmp / name).write_text(text)
    (tmp / "result.json").write_text(json.dumps(result, indent=1, default=C._jsonable))
    marker = ".done" if not reasons else ".fail_" + _san(reasons[0])
    (tmp / marker).write_text(_now() + "\n" + "\n".join(reasons) + "\n")
    if dest.exists():
        prev = C.assert_path(dest.parent.with_name(dest.parent.name + "_prev") / f"{dest.name}.{time.strftime('%Y%m%d_%H%M%S')}")
        prev.parent.mkdir(parents=True, exist_ok=True)
        os.replace(dest, prev)
    os.replace(tmp, dest)


def _xyz_text(syms, xyz, title):
    body = "\n".join(f"{s:<2} {x:15.8f} {y:15.8f} {z:15.8f}" for s, (x, y, z) in zip(syms, np.asarray(xyz, float)))
    return f"{len(syms)}\n{title}\n{body}\n"                       # = pilot_common.write_xyz format


def _marker(d):
    if (d / ".done").is_file():
        return ".done"
    f = sorted(p.name for p in d.glob(".fail_*")) if d.is_dir() else []
    return f[0] if f else None


def assemble_a(D, GA, plan, res, m, rid, force, selected=True):
    R = plan["rxns"][str(rid)]
    dest = FINAL[m] / str(rid)
    uts = res.get(f"ts.{m}.{rid}")
    mk = _marker(dest)
    if not selected:                                               # U-1b: outside U1_selection.json -> marker only
        if mk == STOPPED_MARK and not force:
            return dict(status=STOPPED, reason="U-1", result=_rj(dest / "result.json") or {}, kept=True)
        u = uts or {}
        core = dict(ts=float(u.get("core_h") or 0)) if uts else {}
        res_json = dict(rxn_id=rid, job_id=R["job_id"], method=m, level=METHOD_LEVEL[m], status=STOPPED, reason="U-1",
                        u1=f"user decision 2026-10-25 (U-1b): not a reaction of a complete {U1_METHOD} pair of "
                           f"{U1_JSON.relative_to(HERE)}; no geometry goes on to Phase F / E",
                        ts_unit=f"ts.{m}.{rid}", ts_finished=uts is not None, ts_status=u.get("status"),
                        ts_reason=u.get("reason"), attempt_used=u.get("attempt_used"), shape_kept=u.get("shape_kept"),
                        rmsd_g2_A=u.get("rmsd_g2_A"), dform_g2_A=u.get("dform_g2_A"), e_ts_eh=u.get("e_ts_eh"),
                        core_h=core, core_h_total=sum(core.values()), fail_reasons=[STOPPED], assembled=_now(),
                        code=CODE_VERSION)
        _write_folder(dest, {}, res_json, [STOPPED])               # a folder with another marker moves to _prev
        return dict(status=STOPPED, reason="U-1", result=res_json, kept=False)
    urs = [res.get(f"ref.{m}.{R['mid1']}"), res.get(f"ref.{m}.{R['mid2']}")]
    if uts is None or any(r is None for r in urs):
        return dict(status="incomplete", reason="units missing")
    if mk and mk != STOPPED_MARK and not force:                    # U-1b: a stopped marker is rebuilt once selected
        r = _rj(dest / "result.json") or {}
        return dict(status=r.get("status", "ok" if mk == ".done" else "fail"),
                    reason=(r.get("fail_reasons") or [""])[0], result=r, kept=True)
    pc, fr = D["pc"], D["fr"]
    reasons = []
    if uts["status"] != "ok":
        reasons.append(_reason_a("ts", uts))
    for k, r in enumerate(urs, 1):
        if r["status"] != "ok":
            reasons.append(_reason_a(f"ref{k}", r))
    gt = uts.get("gate") or {}
    pairs = R["pairs"]
    res_json = dict(rxn_id=rid, job_id=R["job_id"], method=m, level=METHOD_LEVEL[m], ts_header=uts.get("header"),
                    charge1=R["charge1"], charge2=R["charge2"], formed_pairs_label=pairs, **{f"d_form_{m}": gt.get("d_form")},
                    d_form_g2=_dists(pc.read_xyz(G2_FINAL / str(rid) / "ts.xyz")[1], [tuple(p) for p in pairs]),
                    e_ts_eh=uts.get("e_ts_eh"), ts_run_hours=uts.get("ts_run_hours"), attempt_used=uts.get("attempt_used"),
                    attempts=uts.get("attempts"), opt_imag_freqs_cm=gt.get("imag_freqs_cm"), opt_n_imag=gt.get("n_imag"),
                    opt_imag_mode_forming_share=gt.get("imag_mode_forming_share"), gate=gt,
                    shape_kept=uts.get("shape_kept"), rmsd_g2_A=uts.get("rmsd_g2_A"), dform_g2_A=uts.get("dform_g2_A"),
                    orca_bin=D["orca"], orca_version=uts.get("orca_version"),
                    units=[uts["uid"]] + [r["uid"] for r in urs],
                    note_phase_f="imag_freqs_cm / imag_mode_forming_share / work/ts.hess: Phase F (XTB2 ALPB(water) Freq "
                                 "at ts.xyz, g0f0 G0 / g2_autode.finalize rule); opt_* = the method-level Freq")
    files = {}
    core = {"ts": float(uts.get("core_h") or 0)}
    for k, r in enumerate(urs, 1):
        M = plan["mols"][R[f"mid{k}"]]
        core[f"ref{k}"] = float(r.get("core_h") or 0) / len(M["members"])
        res_json[f"ref{k}"] = dict(mid=M["mid"], key=M["key"], unit_key=M["unit_key"], constrained=M["constrained"],
                                   start_rxn=M["start_rxn"], n_members=len(M["members"]), status=r["status"],
                                   reason=r.get("reason"), e_eh=r.get("e_eh"), neg_cm=r.get("neg_cm"),
                                   attempt_used=r.get("attempt_used"), core_h=r.get("core_h"))
        res_json[f"e_rel{k}_eh"] = r.get("e_eh")
    if not reasons:
        try:
            ts = pc.read_xyz(UNITS / uts["uid"] / uts["final_dir"] / "ts.xyz")
            files["ts.xyz"] = _xyz_text(ts[0], ts[1], f"G3 {m} TS rxn {rid}")
            for k, r in enumerate(urs, 1):
                M = plan["mols"][R[f"mid{k}"]]
                geo = pc.read_xyz(UNITS / r["uid"] / r["final_dir"] / "ref.xyz")
                target = pc.read_xyz(G2_FINAL / str(rid) / f"rel{k}.xyz")
                if int(M["start_rxn"]) == rid:
                    if list(geo[0]) != list(target[0]):
                        raise Fail(f"ref{k}_order")
                    out, rmsd, remapped = geo, 0.0, False
                else:
                    p, rmsd = GA.remap(fr, target, geo)
                    if p is None:
                        raise Fail(f"ref{k}_order")
                    out, remapped = (list(target[0]), np.asarray(geo[1], float)[p]), True
                res_json[f"ref{k}"].update(remapped=remapped, remap_rmsd_A=float(rmsd))
                files[f"rel{k}.xyz"] = _xyz_text(out[0], out[1], f"G3 {m} reference {k} rxn {rid} ({M['mid']})")
            e1, e2 = res_json.get("e_rel1_eh"), res_json.get("e_rel2_eh")
            if None not in (uts.get("e_ts_eh"), e1, e2):
                res_json[f"barrier_{m}_kcal"] = (uts["e_ts_eh"] - e1 - e2) * C.KCAL
        except Fail as e:
            reasons.append(str(e))
            files = {}
        except Exception as e:                                     # noqa: BLE001
            reasons.append(f"pipeline_assemble_{type(e).__name__}")
            res_json["traceback"] = traceback.format_exc()[-3000:]
            files = {}
    res_json.update(core_h=core, core_h_total=sum(core.values()), fail_reasons=reasons,
                    status="ok" if not reasons else "fail:" + reasons[0], assembled=_now(), code=CODE_VERSION)
    _write_folder(dest, files, res_json, reasons)
    return dict(status=res_json["status"], reason=reasons[0] if reasons else "", result=res_json, kept=False)


def assemble_b(D, plan, res, rid, force):
    R = plan["rxns"][str(rid)]
    dest = PARTB / str(rid)
    ub = res.get(f"B.{rid}")
    if ub is None:
        return dict(status="incomplete", reason="unit missing")
    mk = _marker(dest)
    if mk and not force:
        r = _rj(dest / "result.json") or {}
        return dict(status=r.get("status"), reason=(r.get("fail_reasons") or [""])[0], result=r, kept=True)
    reasons = [] if ub["status"] == "ok" else [_reason_a("partB", ub)]
    r = {k: v for k, v in ub.items() if k not in ("traceback",)}
    r.update(rxn_id=rid, job_id=R["job_id"], level="D1 autodE OptTS (B3LYP RIJCOSX D3BJ def2-SVP SMD(water)), "
                                                    "label-level SP", fail_reasons=reasons,
             status="ok" if not reasons else "fail:" + reasons[0], assembled=_now())
    files = {}
    p = UNITS / ub["uid"] / (ub.get("final_dir") or "optts") / "ts.xyz"
    if p.is_file():
        ts = D["pc"].read_xyz(p)
        files["ts.xyz"] = _xyz_text(ts[0], ts[1], f"G3 Part B TS rxn {rid}")
    _write_folder(dest, files, r, reasons)
    return dict(status=r["status"], reason=reasons[0] if reasons else "", result=r, kept=False)


def _q(v):
    v = [float(x) for x in v if x is not None and np.isfinite(float(x))]
    if not v:
        return None
    return dict(n=len(v), median=float(np.median(v)), mean=float(np.mean(v)), p90=float(np.quantile(v, 0.9)))


def _killed_ts(uids):
    """U-1b: TS units without unit.json -> (killed, killed_in_retry, core-h): killed = the unit folder exists (an ORCA
    step started; no unit.json = the U-1 scancel), in retry = attempt 1 finished without passing its gate (a2/ exists or
    a1/step.json gate not ok); core-h = their finished ORCA steps only (the step running at the kill left no record)."""
    killed, retry, core = [], [], 0.0
    for x in uids:
        if not (UNITS / x).is_dir():
            continue
        killed.append(x)
        s1 = _rj(UNITS / x / "a1" / "step.json")
        if (UNITS / x / "a2").is_dir() or (s1 is not None and not (s1.get("gate") or {}).get("ok")):
            retry.append(x)
        core += sum(float((_rj(p) or {}).get("core_h") or 0) for p in (UNITS / x).glob("a*/step.json"))
    return killed, retry, core


def cmd_assemble(force=False):
    D, plan = d1(), load_plan()
    code, why = t_gate()                                           # O-20: no O results from a stopped / unstarted O
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: assemble: {why}", file=sys.stderr, flush=True)
        return code
    res = load_results(plan)
    u1, code, why = u1_selection(plan, res)                        # U-1b: read (never recomputed) before any write
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: assemble: {why}", file=sys.stderr, flush=True)
        return code
    record_decision("U-1b", DECISIONS["U-1b"])
    GA = g2a_module(D)
    rows, refrows, incomplete = [], [], []
    for m in C.METHODS:
        for rid in plan["part_a"]:
            R = plan["rxns"][str(rid)]
            sel = m == U1_METHOD and rid in u1["rxns"]
            a = assemble_a(D, GA, plan, res, m, rid, force, sel)
            r, uts = a.get("result") or {}, res.get(f"ts.{m}.{rid}") or {}
            gt = uts.get("gate") or {}
            if a["status"] == "incomplete":
                incomplete.append(f"{m}:{rid}")
            rows.append(dict(rxn_id=rid, job_id=R["job_id"], method=m, status=a["status"], reason=a["reason"],
                             ts_status=uts.get("status"), ts_reason=uts.get("reason"),
                             imag_freqs_cm=json.dumps(gt.get("imag_freqs_cm")), n_imag=gt.get("n_imag"),
                             shape_kept=uts.get("shape_kept"), rmsd_start_A=uts.get("rmsd_g2_A"),
                             dform_start_A=uts.get("dform_g2_A"), core_h=uts.get("core_h"),
                             core_h_refs=(r.get("core_h") or {}).get("ref1", 0) + (r.get("core_h") or {}).get("ref2", 0)
                             if r and a["status"] != STOPPED else None, attempt_used=uts.get("attempt_used"),
                             n_atoms=R["n_atoms"], dipole=R["dipole"], classes=";".join(R.get("classes") or []),
                             phase="t1" if rid == plan["t1_rxn"] else "pilot" if rid in plan["pilot_rxns"] else "main",
                             part="A", converged_to_label=None, dE_kcal=None, e_sp_eh=None, e_label_eh=None,
                             sp_mode=None, u1_selected=sel))
        for mid, M in plan["mols"].items():
            uid = f"ref.{m}.{mid}"
            r, need = res.get(uid), uid in u1["needed"]
            refrows.append(dict(mid=mid, method=m, key=M["key"], unit_key=M["unit_key"], constrained=M["constrained"],
                                frag=M["frag"], n_atoms=M["n_atoms"], n_members=len(M["members"]),
                                start_rxn=M["start_rxn"],
                                status=(None if need else STOPPED) if r is None else r["status"],
                                reason=(None if need else "U-1") if r is None else r.get("reason"),
                                neg_cm=None if r is None else json.dumps(r.get("neg_cm")),
                                attempt_used=None if r is None else r.get("attempt_used"),
                                core_h=None if r is None else r.get("core_h"), e_eh=None if r is None else r.get("e_eh"),
                                u1_needed=need))
            if r is None and need:                                 # U-1b: an unneeded ref never run is not missing
                incomplete.append(uid)
    for rid in plan["part_b"]:
        R = plan["rxns"][str(rid)]
        b = assemble_b(D, plan, res, rid, force)
        ub = res.get(f"B.{rid}") or {}
        gt = ub.get("gate") or {}
        if b["status"] == "incomplete":
            incomplete.append(f"B:{rid}")
        rows.append(dict(rxn_id=rid, job_id=R["job_id"], method="partB", status=b["status"], reason=b["reason"],
                         ts_status=ub.get("status"), ts_reason=ub.get("reason"),
                         imag_freqs_cm=json.dumps(gt.get("imag_freqs_cm")), n_imag=gt.get("n_imag"),
                         shape_kept=ub.get("same_as_g2_start"), rmsd_start_A=ub.get("rmsd_g2_A"), dform_start_A=None,
                         core_h=ub.get("core_h"), core_h_refs=None, attempt_used=ub.get("attempt_used"), n_atoms=R["n_atoms"],
                         dipole=R["dipole"], classes=";".join(R.get("classes") or []),
                         phase="pilot" if rid in plan["pilot_b"] else "main", part="B",
                         converged_to_label=ub.get("converged_to_label"), dE_kcal=ub.get("dE_kcal"),
                         e_sp_eh=ub.get("e_sp_eh"), e_label_eh=ub.get("e_label_eh"), sp_mode=ub.get("sp_mode"),
                         u1_selected=None))
    S = pd.DataFrame(rows)
    C.write_atomic(STATUS_CSV, lambda t: S.to_csv(t, index=False))
    C.write_atomic(REFS_CSV, lambda t: pd.DataFrame(refrows).to_csv(t, index=False))
    st = o_stats(plan, res)
    by = {}
    for m in C.METHODS:
        tsr = [res[u["uid"]] for u in plan["units"] if u["kind"] == "ts" and u["method"] == m and u["uid"] in res]
        rfr = [res[u["uid"]] for u in plan["units"] if u["kind"] == "ref" and u["method"] == m and u["uid"] in res]
        ok = [r for r in tsr if r["status"] == "ok"]
        sub = S[(S.method == m)]
        nr = [u["uid"] for u in plan["units"] if u["kind"] == "ts" and u["method"] == m and u["uid"] not in res]
        kl, kr, kc = _killed_ts(nr)                                # U-1b: n_ts_not_run = killed + never started
        by[m] = dict(n_ts=len(tsr), n_ts_ok=len(ok), ts_success=len(ok) / len(tsr) if tsr else None,
                     ts_success_lower_bound=len(ok) / (len(tsr) + len(kr)) if tsr or kr else None,
                     shape_kept=(sum(bool(r.get("shape_kept")) for r in ok) / len(ok)) if ok else None,
                     ts_core_h=_q([r.get("core_h") for r in tsr]), ref_core_h=_q([r.get("core_h") for r in rfr]),
                     n_refs=len(rfr), n_refs_ok=sum(r["status"] == "ok" for r in rfr),
                     final_done=int((sub.status == "ok").sum()), final_status=sub.status.value_counts().to_dict(),
                     retry_used=sum((r.get("attempt_used") or 1) == 2 for r in tsr),
                     fail_reasons=pd.Series([r.get("reason") for r in tsr if r["status"] != "ok"],
                                            dtype=object).value_counts().to_dict(),
                     n_selected=int(sub.u1_selected.astype(bool).sum()),                    # U-1b
                     n_stopped_user=int((sub.status == STOPPED).sum()),
                     n_stopped_user_ts_finished=int(((sub.status == STOPPED) & sub.ts_status.notna()).sum()),
                     n_ts_not_run=len(nr), n_ts_killed=len(kl), n_ts_never_started=len(nr) - len(kl),
                     n_ts_killed_in_retry=len(kr), ts_killed_in_retry=kr, ts_killed_core_h_finished_steps=kc,
                     n_refs_stopped_user=sum(x["method"] == m and x["status"] == STOPPED for x in refrows))
    br = [res[f"B.{rid}"] for rid in plan["part_b"] if f"B.{rid}" in res]
    dE = [r.get("dE_kcal") for r in br if r.get("dE_kcal") is not None]
    partb = dict(n=len(br), n_ok=sum(r["status"] == "ok" for r in br),
                 converged_to_label=sum(bool(r.get("converged_to_label")) for r in br if r["status"] == "ok"),
                 n_dE=len(dE), dE_lt_m05=sum(v < -0.5 for v in dE), dE_abs_le_05=sum(abs(v) <= 0.5 for v in dE),
                 dE_gt_05=sum(v > 0.5 for v in dE), dE=_q(dE), core_h=_q([r.get("core_h") for r in br]))
    reasons = []
    if st["error_rate_finished"] > ERR_MAX:
        reasons.append(f"pipeline errors {st['n_error']} / {st['n_finished']} = {st['error_rate_finished']:.3f} > {ERR_MAX}")
    for m, f in st["first20"].items():
        if f["stop"]:
            reasons.append(f"{m}: TS success {f['n_ok']}/{f['n']} in its first {FIRST_N} TS units < {FIRST_MIN}")
    if stopped_methods():
        reasons.append(f"STOP files of methods {sorted(stopped_methods())}")
    if STOP_FILE.is_file():
        reasons.append(f"{STOP_FILE}: {STOP_FILE.read_text().strip()}")
    est = _rj(EST_JSON) or {}
    cons = sorted(mid for mid, M in plan["mols"].items() if M["constrained"])
    dev = dict(
        O7=dict(rule="constrained Q keys computed per reaction, not shared (spec 4-2 wording; decision O-7)",
                n_units_per_method=len(cons), n_keys=len({plan["mols"][x]["key"] for x in cons}),
                core_h={m: sum(float((res.get(f"ref.{m}.{x}") or {}).get("core_h") or 0) for x in cons)
                        for m in C.METHODS}),
        O9=dict(rule="Part B: autodE _reoptimise once without its imaginary-mode precheck (decision O-9)",
                n_reopt=sum(bool(r.get("reopt")) for r in br), n_reopt_ok=sum(bool(r.get("reopt")) and r["status"] == "ok"
                                                                              for r in br)),
        O14=dict(rule="first-20 STOP stops only that method; the rest continued (decision O-14)",
                 stopped_methods=sorted(stopped_methods())),
        O19=dict(rule="steps started twice without finishing -> fail wall_clipped (decision O-19)",
                 n_wall_clipped=sum(r.get("reason") == "wall_clipped" for r in res.values())))
    done = set(S.rxn_id[(S.method == U1_METHOD) & (S.status == "ok")].astype(int))   # U-1b: selected and .done
    pd_ = [p for p in u1["complete_pairs"] if int(p["a_rxn"]) in done and int(p["b_rxn"]) in done]
    u1_rep = dict(decision="U-1a / U-1b (user 2026-10-25, items 1 and 5)", text=DECISIONS["U-1b"], method=U1_METHOD,
                  selection=str(U1_JSON), selection_sha256=u1["sha256"], selection_written=u1.get("written"),
                  n_complete_pairs=len(u1["complete_pairs"]), n_by_class=u1.get("n_by_class"),
                  n_selected_rxns=len(u1["rxns"]), n_selected_done=len(done & u1["rxns"]),
                  n_pairs_both_done=len(pd_), pairs_both_done_by_class=_counts(p["class"] for p in pd_),
                  n_needed_refs=len(u1["needed"]), needed_refs_missing=sorted(x for x in u1["needed"] if x not in res),
                  stopped_user={m: dict(rxns=by[m]["n_stopped_user"], ts_finished=by[m]["n_stopped_user_ts_finished"],
                                        ts_not_run=by[m]["n_ts_not_run"], ts_killed=by[m]["n_ts_killed"],
                                        ts_never_started=by[m]["n_ts_never_started"],
                                        ts_killed_in_retry=by[m]["ts_killed_in_retry"],
                                        ts_killed_core_h_finished_steps=by[m]["ts_killed_core_h_finished_steps"],
                                        ts_success=by[m]["ts_success"],
                                        ts_success_lower_bound=by[m]["ts_success_lower_bound"],
                                        refs_not_run=by[m]["n_refs_stopped_user"])
                                for m in C.METHODS},
                  first20={m: dict(n=f["n"], n_ok=f["n_ok"], judged=f["n"] == FIRST_N, stop=f["stop"])
                           for m, f in st["first20"].items()},
                  note="units without unit.json (never run, or killed by the U-1 scancel) are neither pipeline errors "
                       "nor missing; o_stats / the O STOP rules count finished units only; ts_success / shape_kept in "
                       "'methods' are over every finished TS unit of the method, not the selected subset (which is "
                       "TS-ok by construction); ts_not_run = ts_killed (unit folder, no unit.json: killed by the "
                       "scancel) + ts_never_started; ts_killed_in_retry = killed after attempt 1 failed its gate; "
                       "ts_success_lower_bound = n_ts_ok / (n_ts + n killed in retry) (those counted as fails); "
                       "ts_killed_core_h_finished_steps = finished ORCA steps of killed units only (lower bound)")
    retro = _rj(EST_RETRO) or {}                                   # U-3a (report only)
    rep = dict(spec="D1G3.md 4, 12 O", written=_now(), code=CODE_VERSION, plan_fingerprint=plan["fingerprint"],
               complete=not incomplete, n_incomplete=len(incomplete), incomplete=incomplete[:50], methods=by,
               partB=partb, o_stats=st, estimate=dict(total_core_h=est.get("total_core_h"), go=est.get("go")),
               estimate_retro=dict(file=str(EST_RETRO), total_core_h=retro.get("total_core_h"),
                                   stop_reasons=retro.get("stop_reasons")) if retro else None,
               t1=(_rj(T1_JSON) or {}).get("pass"), t5=dict(passed=(_rj(T5_JSON) or {}).get("pass"), mode=t5_mode()),
               measured_core_h=sum(float(r.get("core_h") or 0) for r in res.values()), deviations=dev, u1=u1_rep,
               stop=bool(reasons), stop_reasons=reasons, decisions=DECISIONS,
               outputs={str(p): C.sha256(p) for p in (STATUS_CSV, REFS_CSV)})
    C.write_json(REPORT, rep)
    print(json.dumps({k: rep[k] for k in ("complete", "n_incomplete", "stop", "stop_reasons")}, indent=1), flush=True)
    for m, v in by.items():
        print(f"  {m}: TS {v['n_ts_ok']}/{v['n_ts']} ok (lower bound {v['ts_success_lower_bound']}; killed "
              f"{v['n_ts_killed']}, {v['n_ts_killed_in_retry']} in retry), shape kept {v['shape_kept']}, final .done "
              f"{v['final_done']}, selected {v['n_selected']}, stopped_user {v['n_stopped_user']}", flush=True)
    print(f"  U-1: {u1_rep['n_complete_pairs']} complete pairs {u1_rep['n_by_class']}, both ends .done "
          f"{u1_rep['n_pairs_both_done']} {u1_rep['pairs_both_done_by_class']}, needed refs missing "
          f"{len(u1_rep['needed_refs_missing'])}", flush=True)
    print(f"  partB: {partb['n_ok']}/{partb['n']} ok, converged_to_label {partb['converged_to_label']}, dE n {partb['n_dE']}",
          flush=True)
    if reasons:
        return 3
    return 2 if incomplete else 0


def cmd_status():
    plan = load_plan()
    res = load_results(plan)
    cnt = {}
    for u in plan["units"]:
        r = res.get(u["uid"])
        k = (u["phase"], u["kind"], u.get("method"), "missing" if r is None else r["status"])
        cnt[k] = cnt.get(k, 0) + 1
    for k in sorted(cnt, key=str):
        print(" ".join(str(x) for x in k), cnt[k])
    st = o_stats(plan, res)
    print(json.dumps({k: v for k, v in st.items() if k != "first20"}, indent=1))
    for m, f in st["first20"].items():
        print(f"  first {FIRST_N} {m}: {f['n_ok']}/{f['n']} stop={f['stop']}")
    print(f"STOP={STOP_FILE.is_file()} stopped_methods={sorted(stopped_methods())}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--nprocs", type=int, default=SLOT_CORES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)                        # work sel: file of plan unit ids (U-1)
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        print(__doc__)
        return 2
    if a.nprocs < 1 or a.nprocs > SLOT_CORES:
        print(f"--nprocs must be 1..{SLOT_CORES}", file=sys.stderr)
        return 2
    if a.cmd == "plan":
        return cmd_plan(a.force)
    if a.cmd == "check":
        return cmd_check()
    if a.cmd == "tests":
        return cmd_tests(a.nprocs)
    if a.cmd == "work" and a.arg in ("pilot", "main"):
        return cmd_work(a.arg, a.nprocs)
    if a.cmd == "work" and a.arg == "sel" and a.only:
        return cmd_work("sel", a.nprocs, a.only)
    if a.cmd == "estimate":
        return cmd_estimate(a.arg == "retro")
    if a.cmd == "select":
        return cmd_select()
    if a.cmd == "assemble":
        return cmd_assemble(a.force)
    if a.cmd == "status":
        return cmd_status()
    if a.cmd == "one" and a.arg:
        return cmd_one(a.arg, a.nprocs)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
