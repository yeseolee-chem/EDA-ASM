#!/usr/bin/env python3
r"""feat.py — D1-G3 Phase F (docs/specs/D1G3.md 5; STOP rule 12 F) and the Phase T checks T-2 / T-3 (spec 3).
sbatch only; nothing in this file was run when it was written (login node), so every path is untested until it runs
inside a job. Reused code is imported: D1LADDER g0f0 (G0 Freq helpers, F0 child environments, child / env_f /
inputs_args / constants) and inputs (Phase P functions), D1-G2 test_t (run, env_f, env_hf, T-1 / T-3 constants) and
phase_fh (H status classes), D1PAIR d1_features.py runroot / mergeroot and fu2/atoms.py root run / gate / tokens, D1-HF
prep.py geominputs, run.py run, build.py allgeom / parse_rxn. Exception (decision F-2): the G work loop g_one / run_pool /
order_g is a reviewed adaptation of g0f0.g0_one / cmd_g0's pool / order_checks (those are bound to $LAD/dft_final and
the D1-HF dft inputs); it calls the same g0f0 / orca_direct / run_ts functions and keeps the G0-6 canary, abort and
byte-budget rules (D1G3_G_BUDGET_GB). Phase Q readiness is inputs._q_ready's rule with the fingerprint compared by
repo-relative path (decision F-13; the recorded keys are d1ladder-worktree absolute paths).

Per method m in (g3h, g3r) and per Part A reaction (results/S_rxns.csv part_a) whose Phase O folder
$G3/<m>_final/<rxn_id>/ has .done (= the base set of m):
  G  $G3/<m>_feat/<rxn_id>/ in the G1 folder format the feature and FU2 code read: ts.xyz, rel1.xyz, rel2.xyz = byte
     copies of the Phase O files; work/ts.hess = ORCA 6.1.1 '! XTB2 ALPB(water) Freq' at that fixed TS (no optimisation;
     the g0f0 G0 path: g0_modules, FREQ_HEADER nprocs 1, %maxcore config_g2 maxcore_mb, orca_direct write_inp / run /
     parse_hess / imag_and_mode, g0f0.hess_atoms, run_ts.forming_share on the label pairs), work/ts.inp, work/ts.out.gz;
     result.json with imag_freqs_cm (every negative frequency, ascending), n_imag, imag_mode_forming_share (most negative
     mode) = what ext_features.block_b6 recomputes from work/ts.hess and ts.xyz; .done | .fail_<reason> written last
     (folder built as <rxn_id>.tmp, then renamed). $G3/<m>_final is never written (decision F-1).
  F  d1_features.py runroot $G3/<m>_feat $G3/F_<m>/out $G3/F_<m>/ids_all.csv 0 1 --labels/--meta/--prof (the verbatim G2
     inputs $G2/G/labels_d1_g2.json, $G2/G/input_meta_d1_g2.csv, $G2/prof, read only; D1G2_SCRATCH=$G3 in the child env
     only), mergeroot -> $G3/X182_d1_<m>.parquet; fu2/atoms.py root run 0 1 / gate / tokens with D1PAIR_PHASE=FROOT,
     G1_ROOT=$G3/<m>_feat, FU2_SCR=$G3/fu2_<m>, FU2_RES=analysis/d1_g3/results/fu2_<m>, FU2_X182=$G3/X182_d1_<m>.parquet,
     FU2_ATOM_VERSION=fu2-atoms-3-<m>, FU2_TOKEN_VERSION=fu2-tokens-3-<m>, ESPLEY_OUT / R5_SCRATCH under $G3/F_<m>.
  H  D1-HF runner on the 'g2' geometry slot with D1HF_SCRATCH=$G3/hf_<m>, D1HF_RES=analysis/d1_g3/results/hf_<m>,
     D1HF_TEMPLATES=/gpfs/tmp_cpu2/yeseo1ee/d1_hf/inputs (read only), D1HF_G2_ROOT=$G3/<m>_final: prep.py geominputs g2
     (eda + frag*_dist from the new TS, frag*_rel from the new rel1 / rel2.xyz), run.py run g2 hf, build.py allgeom
     <list> g2 -> results/hf_<m>/labels_d1_hf_g2.parquet, H_status.csv, g2_inputs.csv; $G3/hf_<m>/labels_d1_hf_g2.json.
  P  D1LADDER Phase P rules with inputs.RUNGS[<m>] registered at run time (inputs.py itself unchanged): Phase Q ready
     (the _q_ready rule by repo-relative fingerprint, F-13),
     _read_h, refs_min (per Q-2 key the lowest HF reference energy within the rung; P6b graph check), constrained keys
     keep their own reference (P20 option (c), as g2), assemble_rung (gate P-a + re-assembly checks), token_info (layout
     = g1), make_table, table_summary -> $G3/inputs/<m>.parquet + results/P_build_<m>.json (rung_cfg recorded for E).
T-2  $G3/t2/root/<rxn_id> = byte copies of ts.xyz / rel1.xyz / rel2.xyz of 3 $LAD/dft_final folders, then the G code
     above and the F path with versions fu2-atoms-3-dft / fu2-tokens-3-dft: 182 columns vs $LAD/X182_d1_dft.parquet
     (<= 1e-8; D1-G2 T-1 precedent) and token sha256 vs analysis/d1_ladder/results/F0_tokens_manifest.csv -> results/T_2.json
T-3  the D1-HF g2 path with D1HF_G2_ROOT = $LAD/dft_final (read only) for T00003, T00019, T00614: parse_rxn(T, 'g2') vs
     analysis/d1_hf/results/labels_d1_hf.parquet geom=dft (<= 1e-6 kcal/mol) -> results/T_3.json

CLI (exit 0 = done, 2 = not ready / incomplete (resubmit later), 3 = STOP (spec 12) or hard error; every command except
the T checks writes results/F_steps/<step>.json; a rerun overwrites its own record):
  feat.py t2_g [--rerun]   jobs/g3.slurm G3ENV=d1 (submit from analysis/d1_g3): T-2 root + Freq (--rerun moves $G3/t2 aside)
  feat.py t2_f             analysis/d1_pair jobs/g.slurm, PYENV=venv5, D1PAIR_PHASE=F: T-2 features + tokens -> T_2.json
  feat.py t3 [--rerun]     analysis/d1_hf jobs/hf.slurm: T-3 -> results/T_3.json (--rerun moves $G3/t3 aside)
  feat.py g <m|A|0|1|all> [retry]   G3ENV=d1: G step ('retry' first moves the methods' .fail_freq_* / .fail_error_*
                           folders to $G3/<m>_feat_prev/<rxn_id>.<time>; explicit only)
  feat.py f <m|A|0|1>      d1_pair jobs/g.slurm venv5 F: features + FU2 atoms / gate / tokens of m
  feat.py h <m|A|0|1>      d1_hf jobs/hf.slurm: HF EDA (5 calculations) of m
  feat.py p <m|A|0|1>      G3ENV=reactot: rung input table of m
  feat.py report           G3ENV=reactot: results/F_report.json (+ results/F_G_status.csv, decisions -> DECISIONS.md)
  internal: _hfparse <list.csv> <out.json>   (child of t3: parse_rxn(T, 'g2') under the D1HF_* overrides)
  m: g3h | g3r; A = g3_common.METHODS[$SLURM_ARRAY_TASK_ID] (g.slurm turns '@I' into the digit).
Preconditions checked by g / f / h / p: results/T_2.json and T_3.json passed with the current code (sha256 of CODE_REL),
results/S_rxns.csv = S_report.json outputs, results/O_report.json complete and not stop with O_status.csv / O_refs.csv
unchanged and its ok rows = the .done markers (decision F-14: missing / incomplete = exit 2, stop = exit 3), every Part A
reaction has a Phase O marker in $G3/<m>_final (else exit 2). The F step records the feature root it ran on
($G3/F_<m>/root_fp.json = sha256 of every base G folder's result.json); a changed root moves the F outputs of m aside and
recomputes them (decision F-15). The T-2 / T-3 work folders carry the code sha256 they were made with; another code
moves them aside (decision F-16).
Consistency with Phase O (opt.py assemble --force replaces <m>_final folders): a G folder whose recorded source sha256
differs from the current <m>_final ts.xyz / rel1.xyz / rel2.xyz is 'stale' (STOP; 'g <m> retry' moves it aside), and an
HF run whose input coordinates differ from the current geometry inputs is stale (STOP; move the run folder aside).

sbatch (login node, one-shot commands only; first N = `squeue -u $USER -h -r | wc -l` (MaxSubmit 20 incl. array
elements; MaxJobs 10) and `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`; logs in $G3/logs; g.slurm / hf.slurm need --output
because their default log folders are the read-only d1_pair / d1_hf scratch):
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_g3; L=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs
  # Phase T (after the Phase S job JS; parallel to opt.py plan / check / tests and apply.py t4): 3 tasks
  JT2G=$(sbatch --parsable -p cpu1,cpu2 -J g3_t2g --time=48:00:00 --cpus-per-task=3 --mem=8G --output=$L/%x_%j.out \
       --export=ALL,G3ENV=d1 --dependency=afterok:${JS:?} --kill-on-invalid-dep=yes jobs/g3.slurm feat.py t2_g)
  JT2F=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J g3_t2f --time=48:00:00 --cpus-per-task=8 --mem=32G \
       --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JT2G:?} \
       --kill-on-invalid-dep=yes jobs/g.slurm ../d1_g3/feat.py t2_f)
  JT3=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J g3_t3 --time=48:00:00 --cpus-per-task=3 --mem=16G \
       --output=$L/%x_%j.out --dependency=afterok:${JS:?} --kill-on-invalid-dep=yes jobs/hf.slurm ../d1_g3/feat.py t3)
  # Phase F (after opt.py assemble JA exited 0 = O complete, no STOP): 6 tasks. G and H start together; F waits for
  # G; P + report wait for F and H.
  JG=$(sbatch --parsable -p cpu1,cpu2 -J g3_G --time=48:00:00 --cpus-per-task=16 --mem=32G --output=$L/%x_%j.out \
       --export=ALL,G3ENV=d1 --dependency=afterok:${JA:?} --kill-on-invalid-dep=yes jobs/g3.slurm feat.py g all)
  JH=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J g3_H --time=48:00:00 --cpus-per-task=48 --mem=180G \
       --array=0-1 --output=$L/%x_%A_%a.out --dependency=afterok:${JA:?} --kill-on-invalid-dep=yes \
       jobs/hf.slurm ../d1_g3/feat.py h A)
  JF=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J g3_F --time=48:00:00 --cpus-per-task=32 --mem=64G \
       --array=0-1 --output=$L/%x_%A_%a.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JG:?} \
       --kill-on-invalid-dep=yes jobs/g.slurm ../d1_g3/feat.py f @I)
  JP=$(sbatch --parsable -p cpu1,cpu2 -J g3_P --time=48:00:00 --cpus-per-task=2 --mem=16G --output=$L/%x_%j.out \
       --export=ALL,G3ENV=reactot --dependency=afterok:${JF:?}:${JH:?} --kill-on-invalid-dep=yes \
       jobs/g3.slurm feat.py p g3h ';;;' feat.py p g3r ';;;' feat.py report)
Sizes: G ~2 x 200 serial xTB Freq (G0: 0.5 s - 1 min each); F per method 1 runroot slice + atoms on 32 cores (D1PAIR:
8,433 reactions in 8 x 32 cores x 2-4 min); H per method ~200 x 5 serial HF single points (D1-HF pilot median 0.106
core-h per reaction, %maxcore 3500: 48 workers = the D1-G2 Phase H job size 48 cores / 180 GB); P 2 cores.
Resubmission (every step idempotent, writes atomic): a step that exited 2 or died -> the same line without
--dependency (G: finished folders skipped; F: slice / merge / atoms cached; H: run.py skips .done reactions and good
outputs; P / report recompute). Exit 3 = STOP: read results/F_steps/<step>.json (fail_stage, fail) and report.
A G folder failed by Freq / error is kept until 'g <m> retry' (explicit). T: t2_g --rerun / t3 --rerun start fresh;
a code change does that automatically (F-16).

Decisions (user: every judgment = the recommended option): DECISIONS below; written to results/DECISIONS.md when any
command starts, under the shared fcntl lock $G3/locks/DECISIONS.lock (F-11), and copied into F_report.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_ladder, d1_pair, d1_hf, d1_g2 trees

import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import g3_common as C  # noqa: E402   (puts analysis/d1_ladder on sys.path: L = lad_common, DP = d1pair_common)
import g0f0 as G0F  # noqa: E402      D1LADDER G0 / F0 helpers (read only, imported)

L, DP = C.L, C.DP
Stop, Fail = G0F.Stop, G0F.Fail
if Path(G0F.__file__).resolve() != (C.LAD_DIR / "g0f0.py").resolve():
    C.stop(f"g0f0 imported from {G0F.__file__}, not {C.LAD_DIR / 'g0f0.py'}")

# ---------------------------------------------------------------- constants
T0 = float(os.environ.get("D1G3_JOB_T0") or os.environ.get("D1HF_JOB_T0") or time.time())
STOP_AFTER_H = 44.0                                     # no new G reaction after this (48 h wall)
RATE_MAX = 0.05                                         # spec 12 F: existing gate failure rate
S_RXNS, S_PAIRS, S_REPORT = C.RES / "S_rxns.csv", C.RES / "S_pairs.csv", C.RES / "S_report.json"
STEPS = C.RES / "F_steps"
F_REPORT = C.RES / "F_report.json"
G_STATUS = C.RES / "F_G_status.csv"
T2_JSON, T3_JSON = C.RES / "T_2.json", C.RES / "T_3.json"
T2_DIR, T3_DIR = C.G3 / "t2", C.G3 / "t3"
O_REPORT, O_STATUS, O_REFS = C.RES / "O_report.json", C.RES / "O_status.csv", C.RES / "O_refs.csv"   # opt.py (F-14)
G_BUDGET_GB = float(os.environ.get("D1G3_G_BUDGET_GB", "2"))   # G-step bytes kept per run (G0-6 rule, F-2)
CODE_MARK = "code_sha256.txt"                                   # T work folder code stamp (F-16)
T2_RXN = [1000003, 1000019, 1000614]                    # = D1-G2 T3_JOBS (Br, Si, g1_rec)
T3_JOBS = ["T00003", "T00019", "T00614"]
T2_VERSIONS = dict(FU2_ATOM_VERSION="fu2-atoms-3-dft", FU2_TOKEN_VERSION="fu2-tokens-3-dft")   # = g0f0.VERSIONS
HF_TEMPLATES = C.HF_SCR / "inputs"                      # <T>/hf/*.inp (read only)
HF_KEYS = ("D1HF_SCRATCH", "D1HF_RES", "D1HF_TEMPLATES", "D1HF_G2_ROOT")
ORCA_WANT = Path("/home1/yeseo1ee/orca_6_1_1_avx2/orca")
XYZ_FILES = ("ts.xyz", "rel1.xyz", "rel2.xyz")
LEVEL = ("{tag}: TS and references = byte copies of {src} (ts.xyz, rel1.xyz, rel2.xyz); work/ts.hess = ORCA 6.1.1 "
         "XTB2 ALPB(water) Freq at that fixed TS (no optimisation; D1LADDER G0 path)")
CODE_REL = ["analysis/d1_g3/feat.py", "analysis/d1_g3/g3_common.py", "analysis/d1_g3/jobs/g3.slurm",
            "analysis/d1_ladder/g0f0.py", "analysis/d1_ladder/lad_common.py",
            "analysis/d1_g2/g2_autode.py", "analysis/d1_g2/g2_common.py", "analysis/d1_g2/test_t.py",
            "analysis/d1_g2/config_g2.yaml",
            "analysis/d1_pair/d1_features.py", "analysis/d1_pair/d1pair_common.py", "analysis/d1_pair/d1_g1.py",
            "analysis/d1_pair/fu2/atoms.py", "analysis/d1_pair/fu2/tokens.py", "analysis/d1_pair/fu2/fu2_common.py",
            "analysis/d1_pair/jobs/env.sh", "analysis/d1_pair/jobs/g_env.sh", "analysis/d1_pair/jobs/g.slurm",
            "analysis/d1_hf/prep.py", "analysis/d1_hf/run.py", "analysis/d1_hf/build.py", "analysis/d1_hf/hf_common.py",
            "analysis/d1_hf/jobs/hf.slurm", "analysis/d1_hf/jobs/hf_env.sh"]

DECISIONS = {
    "F-1 feature root": "feature 루트 = $G3/<m>_feat/<rxn_id> (G1 폴더 형식): $G3/<m>_final의 ts.xyz·rel1.xyz·rel2.xyz를 "
                        "바이트 그대로 복사하고 result.json·work/ts.hess(XTB2 ALPB(water) Freq)를 새로 쓴다. "
                        "ext_features.block_b6가 result.json의 imag_freqs_cm[0]·imag_mode_forming_share를 work/ts.hess에서 "
                        "다시 계산한 값과 1e-6 안에서 비교한다. opt.py(O-16)는 이 세 항목을 Phase F에 남겼는데, 그것을 "
                        "$G3/<m>_final 안에 덧쓰지 않고 G1 형식 사본에 쓴다: Phase O 폴더는 O가 조립한 그대로 남고, "
                        "'opt.py assemble --force'가 폴더를 바꿔 끼워도 F 추가분이 조용히 사라지거나 섞이지 않는다(사본의 "
                        "sources sha256이 지금 <m>_final 파일과 다르면 stale = STOP, 'g <m> retry'가 치운다). G0가 dft_final을 "
                        "따로 만든 것과 같은 방식. D1-HF는 명령서대로 $G3/<m>_final을 읽는다(D1HF_G2_ROOT). 두 루트의 좌표 "
                        "파일은 바이트 단위로 같다. feature 루트를 <m>_final이 아닌 <m>_feat로 두는 것은 명령서 5절 '루트 = "
                        "$G3/g3h_final, $G3/g3r_final' 문구에서 벗어난 것이다(좌표는 같은 바이트, 이탈로 기록).",
    "F-2 G step": "work/ts.hess = g0f0 G0 경로(g0_modules: g2_autode.FREQ_HEADER nprocs 1, %maxcore = config_g2 maxcore_mb; "
                  "orca_direct.write_inp / run / parse_hess / imag_and_mode; g0f0.hess_atoms; run_ts.forming_share(라벨 "
                  "형성 결합); TOL_HESS_GEOM_A 1e-4; 노드 로컬 작업 폴더; canary·abort·바이트 예산 규칙 G0-6, 예산 "
                  "D1G3_G_BUDGET_GB 기본 2 GB). g0_one은 $LAD/dft_final과 D1-HF dft 입력에 묶여 있어(루트 인자 없음) 그 "
                  "작업 루프(g_one / run_pool / order_g)를 검토해 옮겨 쓴다 = g0f0 코드의 수정 사본(import가 아님); T-2가 이 "
                  "사본을 DFT TS 3개에서 G0 결과와 비교한다. 원자 순서 gate: "
                  "TS 원소 순서 = D1 job ts.xyz, 조각 크기 = input_meta A_idx, 역할, rel1/rel2 원소 순서 = "
                  "ref_dipole/ref_dipolarophile, rel 원소 다중집합 = TS 조각(새 구조라 좌표 일치 검사는 없다). n_imag 0 / >= 2는 "
                  ".done으로 기록하고(G0-4) B6 실패는 F에서 센다.",
    "F-3 base set": "방법별 기준 집합 = S_rxns.csv part_a 반응 중 $G3/<m>_final/<rxn>/.done. 모든 Part A 반응에 Phase O "
                    "표지(.done / .fail_*)가 있어야 G·H를 시작한다(아니면 exit 2). Phase O 실패는 O의 수이고 F gate 실패가 아니다.",
    "F-4 gate": "명령서 12절 F(기존 gate 실패율 > 5 % = STOP). feature: 기준 반응 중 토큰 npz가 없는 반응(G 단계 order/Freq "
                "실패 + mergeroot not-ok(ts_fragments, B1..B6, 비유한) + Q use false + T-a bad) / 기준 반응 수. HF: H_status "
                "gate_fail + eda_table_unexplained + parse_fail / 끝난 반응(D1-G2 H 정의); HF 입력 검사(geominputs status != "
                "ok) / 기준 반응 수는 같은 5 % 규칙으로 따로. 'fail'(SlowConv 재시도 뒤 SCF 실패, ORCA 오류)은 run_fail로 보고만 "
                "한다(D1-G2 규칙). 그대로 두는 판정: mergeroot stop, Q_report stop(Q-a > 42 / Q-b / Q-c), T_report stop, D1PAIR "
                "g1 T_report와 다른 토큰 배치 -> STOP. G 단계는 order+Freq 실패율 > 5 %면 그 자리에서 STOP. G0/F0의 "
                "0.5 %·1 % 문턱은 보고만 한다(이 명령서의 문턱은 5 %).",
    "F-5 F layout": "F: 방법마다 한 job(32코어, 배열 0-1): runroot 1 slice(0 of 1) -> mergeroot -> atoms.py root run 0 1 -> "
                    "(err.json이 남으면 한 번 더) -> root gate -> root tokens. 버전 fu2-atoms-3-<m> / fu2-tokens-3-<m>; "
                    "D1G2_SCRATCH=$G3는 d1_features 자식 환경에만(F0-2); 입력 = G2 입력 그대로(--labels/--meta/--prof).",
    "F-6 HF": "D1-HF runner의 g2 자리를 방법마다 덮어쓴다: D1HF_SCRATCH=$G3/hf_<m>, D1HF_RES=results/hf_<m>, "
              "D1HF_TEMPLATES=d1_hf/inputs(읽기만), D1HF_G2_ROOT=$G3/<m>_final. prep.py geominputs g2 <기준 목록> -> run.py "
              "run(입력 ok 반응, 큰 반응 먼저, $SLURM_CPUS_PER_TASK 개 직렬 ORCA) -> build.py allgeom <기준 목록> g2. 결과의 "
              "geom 열은 슬롯 이름 'g2'(= 방법 <m>)다. SlowConv 재시도 1회는 run.py 그대로.",
    "F-7 P tables": "입력표 = inputs.py 함수를 그대로 부른다(RUNGS[<m>]를 실행 중에 등록: geom g2, hf_runs $G3/hf_<m>/runs, "
                    "hf_inputs $G3/hf_<m>/inputs, h_parquet results/hf_<m>/labels_d1_hf_g2.parquet, tokens / fu2_res / x182 = "
                    "F 출력, 버전 -<m>, refs min): Phase Q 준비(_q_ready 규칙, 지문은 repo 상대 경로: F-13), _read_h, "
                    "refs_min(rung 안 Q-2 키별 최저 HF 기준 에너지, P6b), "
                    "assemble_rung(P-a + 재조립 검사), token_info(g1 배치), make_table(9,266행, usable 표시), table_summary. "
                    "cmd_build는 $LAD에 쓰므로 부르지 않는다. 표 = $G3/inputs/<m>.parquet, 보고 = results/P_build_<m>.json"
                    "(rung_cfg 포함: E가 같은 RUNGS 항목을 등록한다).",
    "F-7b constrained keys": "제약 dipole 키(Q_keys constrained; '|fixed:' 접미사는 D1 DFT 기준 구조에서 나옴): g2 rung의 "
                             "P20 결정 (c)와 같이 교체하지 않고 자기 기준을 쓴다. refs_min은 rung 이름이 'g2'일 때만 그 방식을 "
                             "쓰므로(inputs.py 826행), refs_min 결과에서 constrained 행의 'min' 교체를 되돌리는 후처리로 같은 "
                             "결과를 만든다(되돌린 수는 P_build의 references_g3_cons_own).",
    "F-8 T-2": "T-2 반응 = 1000003(Br), 1000019(Si), 1000614(g1_rec) = D1-G2 T-3 묶음: $LAD/dft_final의 ts.xyz·rel1·rel2 "
               "바이트 복사 -> G 단계 코드 그대로 xTB Freq -> runroot/mergeroot -> 182열 vs X182_d1_dft(1e-8; 넘으면 D1-G2 "
               "T-1 선례(scan_curv_*만, <= 2.5e-7) = pass_f0_precedent, 편차로 기록). 토큰은 D1-G2 T-2처럼 FU2_X182 = "
               "X182_d1_dft 세 행의 정확한 사본, 버전 -dft로 만들어 sha256을 F0_tokens_manifest.csv와 비교한다(토큰 glob이 "
               "X182 행 그대로라 182열 비교와 토큰 비교를 나눈다). 기준 파일 sha256(X182_d1_dft, F0_tokens_manifest)은 "
               "F0_report와 맞춰 본다; 세 반응은 dft Q_manifest에서 use·ext_ok여야 한다.",
    "F-9 T-3": "T-3 = D1-HF g2 경로, D1HF_G2_ROOT = $LAD/dft_final(읽기만), T00003·T00019·T00614: geominputs -> 생성 입력을 "
               "d1_hf/inputs/<T>/hf와 비교(보고만) -> run -> allgeom -> parse_rxn(T, 'g2') vs labels_d1_hf.parquet geom=dft의 "
               "d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier, bond, eint_spe(<= 1e-6 kcal/mol, 둘 다 ok) + "
               "labels_d1_hf_g2.json 행 일치(D1-G2 t3 규칙).",
    "F-10 T currency": "g / f / h / p는 results/T_2.json·T_3.json이 지금 코드(CODE_REL sha256 묶음)로 통과했을 때만 돈다"
                       "(없음·실행 중·코드 바뀜 = exit 2, 실패 = exit 3).",
    "F-11 decision log": "모든 명령(t2_g, t2_f, t3, g, f, h, p, report)이 시작할 때 이 결정을 모두 results/DECISIONS.md에 "
                         "적는다(명령서 11 '적은 뒤 진행'). g3_common.record_decision은 잠금 없는 읽기-쓰기라 "
                         "$G3/locks/DECISIONS.lock의 fcntl 잠금 안에서 부른다(d1_g3의 다섯 스크립트가 같은 잠금을 쓴다).",
    "F-13 Q readiness": "Phase Q 준비 = inputs._q_ready(refs_v.q_ready)와 같은 규칙(Q_report all_pass, outputs sha256 그대로, "
                        "입력·코드 지문 그대로)이되 지문은 repo 상대 경로로 비교한다: 기록된 키가 d1ladder 작업 트리 절대 경로라 "
                        "이 작업 트리에서 refs_q.inputs_fingerprint()와 dict로 비교하면 항상 다르다(기존 코드 문제). 작업 트리 "
                        "밖 경로(/gpfs/tmp_cpu2/...)는 그대로 비교. sha256이 하나라도 다르면 STOP.",
    "F-14 O precondition": "g / f / h / p는 results/O_report.json이 complete이고 stop이 아니며 그 outputs(O_status.csv, "
                           "O_refs.csv) sha256이 그대로이고 O_status의 방법별 Part A ok 행 = <m>_final .done 표지일 때만 "
                           "돈다(없음·미완 = exit 2, O STOP = exit 3: 명령서 12 O '보고 후 대기'). opt.py assemble은 STOP을 "
                           "판정하기 전에 폴더를 모두 쓰므로 표지만으로는 O STOP을 알 수 없다.",
    "F-15 feature root currency": "F 단계는 실행한 feature 루트를 $G3/F_<m>/root_fp.json(기준 반응마다 G 폴더 result.json "
                                  "sha256)에 남긴다. 다음 f에서 다르면(예: assemble --force 뒤 'g <m> retry') $G3/F_<m>, "
                                  "$G3/fu2_<m>, results/fu2_<m>, $G3/X182_d1_<m>.parquet를 .prev_<시각>으로 치우고 다시 "
                                  "계산한다(runroot slice, mergeroot, atoms npz 캐시가 옛 구조를 쓰지 않게). p / report는 "
                                  "root_fp가 지금과 다르면 F 미완(exit 2).",
    "F-16 T currency of work folders": "T-2·T-3 작업 폴더($G3/t2, $G3/t3)에는 만든 코드의 sha256(code_sha256.txt)을 둔다. "
                                       "지금 코드와 다르면 t2_g / t3가 폴더를 .prev_<시각>으로 치우고 새로 계산하고, t2_f는 "
                                       "t2_g가 지금 코드로 만든 폴더에서만 돈다(아니면 exit 2: t2_g 다시) — 옛 코드의 캐시로 "
                                       "'지금 코드로 통과'가 기록되지 않게.",
    "F-12 quota": "G 폴더는 work/ts.inp, ts.out.gz, ts.hess만 남긴다(.gbw 등은 노드 로컬 작업 폴더와 함께 지움); HF는 run.py "
                  "규칙(.inp, .out.gz, .err.gz; .gbw·.tmp·.densities는 복사하지 않음).",
}


# ================================================================ small helpers
def now():
    return G0F.now()


def resolve_method(s):
    """g3h | g3r | A / @I (= METHODS[$SLURM_ARRAY_TASK_ID]) | 0 | 1."""
    s = str(s)
    if s in C.METHODS:
        return s
    if s.upper() in ("A", "@I"):
        v = os.environ.get("SLURM_ARRAY_TASK_ID")
        if v is None:
            raise SystemExit(f"{s}: no SLURM_ARRAY_TASK_ID (submit with --array or name the method)")
        s = v
    if s.isdigit() and int(s) < len(C.METHODS):
        return C.METHODS[int(s)]
    raise SystemExit(f"unknown method {s!r} (one of {C.METHODS}, A, 0 .. {len(C.METHODS) - 1})")


def paths(m):
    """every Phase F path of method m."""
    F = C.G3 / f"F_{m}"
    return SimpleNamespace(
        final=C.G3 / f"{m}_final", feat=C.G3 / f"{m}_feat", feat_prev=C.G3 / f"{m}_feat_prev", F=F, out=F / "out",
        slices=F / "out" / "xtb" / "f_slices", ids=F / "ids_all.csv", mrep=F / "mergeroot_report.json",
        g1p=F / "out" / "xtb" / "xtb_features_g1_d1.parquet", extp=F / "out" / "xtb" / "xtb_features_ext_g1_d1.parquet",
        x182=C.G3 / f"X182_d1_{m}.parquet", fu2_scr=C.G3 / f"fu2_{m}", fu2_res=C.RES / f"fu2_{m}", fu2_xtb=F / "fu2_xtb",
        fu2_r5=F / "fu2_r5", versions=dict(FU2_ATOM_VERSION=f"fu2-atoms-3-{m}", FU2_TOKEN_VERSION=f"fu2-tokens-3-{m}"),
        hf_scr=C.G3 / f"hf_{m}", hf_res=C.RES / f"hf_{m}", table=C.G3 / "inputs" / f"{m}.parquet",
        p_report=C.RES / f"P_build_{m}.json")


_LAB = {}


def labels():
    """{rxn_id: label row} of lad_common.targets() (labels_d1.json checks inside)."""
    if not _LAB:
        _LAB.update({int(d["rxn_id"]): d for d in L.targets()})
    return _LAB


def write_json(path, obj):
    return C.write_json(path, G0F.clean(obj))


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


def record_all():
    """F-11: every decision of this module, before the command acts on it (idempotent per key)."""
    for k, v in DECISIONS.items():
        record_decision(k, v)


def step(name, body):
    """body(rec) -> exit code (0 / 2 / 3); results/F_steps/<name>.json (core_h_alloc = wall x cpus)."""
    record_all()
    out = STEPS / f"{name}.json"
    rec = dict(step=name, argv=sys.argv[1:], started=now(), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_job=os.environ.get("SLURM_ARRAY_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), cpus=G0F.cpus(), git=G0F.git_head(),
               python=sys.executable)
    t0 = time.time()
    try:
        rc = int(body(rec) or 0)
    except Stop as e:
        rec.update(fail_stage=e.stage, fail=str(e.msg)[:4000])
        rc = e.rc
    except SystemExit as e:                                # C.stop / L.stop / load_d1_labels inside a body
        code = e.code
        rc = 0 if code in (None, 0) else (code if isinstance(code, int) else 3)
        rec.update(fail_stage="exit", fail=f"SystemExit({code!r}) inside the step (message on stderr)")
    except Exception as e:                                 # noqa: BLE001 — a crash is a failed step, recorded
        rec.update(fail_stage="error", fail=f"{type(e).__name__}: {e}"[:4000], traceback=traceback.format_exc()[-4000:])
        rc = 3
    rec.update(rc=rc, finished=now(), wall_s=round(time.time() - t0, 1))
    rec["core_h_alloc"] = rec["wall_s"] * G0F.cpus() / 3600.0
    write_json(out, rec)
    print(json.dumps(G0F.clean({k: v for k, v in rec.items() if k != "traceback"}), default=str)[:8000], flush=True)
    if rec.get("traceback"):
        print(rec["traceback"], file=sys.stderr, flush=True)
    print(f"[{name}] exit {rc} -> {out}", flush=True)
    return rc


def move_aside(p):
    """an existing $G3 path -> <name>.prev_<time> (never deleted)."""
    if p.exists() or p.is_symlink():
        q = C.assert_path(p.with_name(f"{p.name}.prev_{time.strftime('%Y%m%d_%H%M%S')}"))
        os.replace(C.assert_path(p), q)
        return str(q)
    return None


# ================================================================ code currency, T state
def code_files():
    f = [C.REPO / r for r in CODE_REL] + sorted((C.REPO / "analysis" / "espley_xtb_repro").glob("*.py")) \
        + sorted((C.P_DIR / "site_si").glob("*.py"))
    miss = [str(p) for p in f if not p.is_file()]
    if miss:
        raise Stop("code", f"code files missing: {miss[:5]}", 3)
    return sorted(set(f))


def code_sha():
    h = hashlib.sha256()
    for f in code_files():
        h.update(f"{f.relative_to(C.REPO)} {C.sha256(f)}\n".encode())
    return h.hexdigest()


def t_state():
    """{'2': {...}, '3': {...}}: verdict / pass of results/T_<k>.json and whether its code sha256 is the current one."""
    cur = code_sha()
    out = {}
    for k, f in (("2", T2_JSON), ("3", T3_JSON)):
        r = G0F.read_json(f) if f.is_file() else {}
        out[k] = dict(file=str(f), present=f.is_file(), verdict=r.get("verdict"), passed=bool(r.get("pass")),
                      stale=bool(r) and r.get("code_sha256") != cur, finished=r.get("finished"),
                      summary=r.get("summary"))
    return out


def need_t(rec):
    st = t_state()
    rec["T"] = st
    bad = [f"T-{k}" for k, v in st.items() if v["verdict"] == "fail"]
    if bad:
        raise Stop("T", f"{bad} failed (spec 12 T STOP): read results/T_<k>.json", 3)
    open_ = [f"T-{k} ({'missing' if not v['present'] else v['verdict']}{', stale code' if v['stale'] else ''})"
             for k, v in st.items() if not v["passed"] or v["stale"]]
    if open_:
        raise Stop("T", f"{open_} not passed with the current code (run / rerun t2_g + t2_f, t3)", 2)


# ================================================================ S / Phase O inputs
def part_a(rec=None):
    """[(rxn_id, job_id)] of the Part A reactions (results/S_rxns.csv part_a True; sha256 = S_report.json outputs)."""
    if not (S_RXNS.is_file() and S_REPORT.is_file()):
        raise Stop("S", f"{S_RXNS} / {S_REPORT} missing (Phase S)", 2)
    rep = C.read_json(S_REPORT)
    sha = C.sha256(S_RXNS)
    if (rep.get("outputs") or {}).get(str(S_RXNS)) != sha:
        raise Stop("S", f"{S_RXNS} sha256 {sha[:16]} != S_report.json outputs", 3)
    with open(S_RXNS, newline="") as f:
        rows = [r for r in csv.DictReader(f) if str(r.get("part_a")).strip() == "True"]
    lab = labels()
    out = []
    for r in rows:
        rid, T = int(r["rxn_id"]), str(r["job_id"])
        if rid not in lab or lab[rid]["job_id"] != T:
            raise Stop("S", f"S_rxns.csv row {rid} / {T} is not a target with that job_id", 3)
        out.append((rid, T))
    if not out:
        raise Stop("S", "no Part A reaction in S_rxns.csv", 3)
    out.sort()
    if rec is not None:
        rec["part_a"] = dict(n=len(out), s_rxns_sha256=sha)
    return out


def base_of(m, pa, rec=None, need_complete=True):
    """(base = [(rxn_id, job_id)] with $G3/<m>_final/<rxn_id>/.done, O info); need_complete: exit 2 while a Part A
    reaction has no Phase O marker."""
    P = paths(m)
    mk = {rid: G0F.marker(P.final / str(rid)) for rid, _ in pa}
    open_ = sorted(r for r, v in mk.items() if v is None)
    base = [(r, T) for r, T in pa if mk[r] == ".done"]
    info = dict(root=str(P.final), n_part_a=len(pa), n_done=len(base), n_open=len(open_),
                n_fail=sum(1 for v in mk.values() if v and v != ".done"),
                fail_markers=dict(Counter(v for v in mk.values() if v and v != ".done").most_common(30)), open=open_[:50])
    if rec is not None:
        rec.setdefault("O", {})[m] = info
    if need_complete and open_:
        raise Stop("O", f"{m}: {len(open_)} Part A reactions without a Phase O marker in {P.final} (O not finished)", 2)
    if need_complete and not base:
        raise Stop("O", f"{m}: no Part A reaction with .done in {P.final}", 3)
    return base, info


def o_state(pa):
    """decision F-14: results/O_report.json (opt.py assemble) state and the O_status ok rows vs the .done markers."""
    if not O_REPORT.is_file():
        return dict(present=False, complete=False, stop=False, ok=False, why=f"{O_REPORT} missing (opt.py assemble)")
    r = G0F.read_json(O_REPORT)
    outs = r.get("outputs") or {}
    changed = [str(p) for p in (O_STATUS, O_REFS) if outs.get(str(p)) != (C.sha256(p) if p.is_file() else None)]
    st = dict(present=True, file=str(O_REPORT), sha256=C.sha256(O_REPORT), written=r.get("written"),
              complete=bool(r.get("complete")), stop=bool(r.get("stop")), stop_reasons=r.get("stop_reasons"),
              changed=changed, marker_mismatch={})
    if not changed and O_STATUS.is_file():
        with open(O_STATUS, newline="") as f:
            rows = [x for x in csv.DictReader(f) if x.get("part") == "A"]
        for m in C.METHODS:
            ok = {int(x["rxn_id"]) for x in rows if x.get("method") == m and x.get("status") == "ok"}
            done = {rid for rid, _ in pa if G0F.marker(paths(m).final / str(rid)) == ".done"}
            if ok != done:
                st["marker_mismatch"][m] = sorted(ok ^ done)[:20]
    st["ok"] = bool(st["complete"] and not st["stop"] and not changed and not st["marker_mismatch"])
    return st


def need_o(rec, pa):
    """decision F-14: exit 2 while Phase O is unfinished, exit 3 when it ended in a STOP or its folders changed."""
    st = o_state(pa)
    rec["O_report"] = st
    if not st["present"]:
        raise Stop("O", st["why"], 2)
    if st["stop"]:
        raise Stop("O", f"Phase O ended in a STOP ({O_REPORT} stop_reasons {st['stop_reasons']}; spec 12 O: report "
                        f"and wait)", 3)
    if not st["complete"] or st["changed"]:
        raise Stop("O", f"Phase O not complete / O_status changed after O_report (complete {st['complete']}, changed "
                        f"{st['changed']}): rerun opt.py assemble", 2)
    if st["marker_mismatch"]:
        raise Stop("O", f"$G3/<m>_final .done markers != O_status.csv ok rows {st['marker_mismatch']} (folders "
                        f"changed after assemble)", 3)
    return st


# ================================================================ environments
def env_d1(rec):
    """G / t2_g: the D1 run environment of jobs/g3.slurm with G3ENV=d1 (ORCA 6.1.1, D1 code copy)."""
    bad = []
    d1py = os.environ.get("D1_PY")
    if not d1py or Path(sys.executable).resolve() != Path(d1py).resolve():
        bad.append(f"python {sys.executable} is not $D1_PY {d1py!r} (jobs/g3.slurm with G3ENV=d1)")
    orca = os.environ.get("ORCA_BIN", "")
    if not (orca and os.access(orca, os.X_OK)) or Path(orca).resolve() != ORCA_WANT.resolve():
        bad.append(f"ORCA_BIN={orca!r} (need executable {ORCA_WANT})")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        bad.append(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r} (d1 env.sh sets 1)")
    if os.environ.get("D1G3_SCRATCH") != str(C.G3) or os.environ.get("G3ENV", "d1") != "d1":
        bad.append(f"D1G3_SCRATCH={os.environ.get('D1G3_SCRATCH')!r}, G3ENV={os.environ.get('G3ENV')!r} "
                   f"(jobs/g3.slurm with G3ENV=d1)")
    rec["env"] = {k: os.environ.get(k) for k in ("D1_PY", "ORCA_BIN", "OMP_NUM_THREADS", "D1G3_SCRATCH", "G3ENV",
                                                 "TMPDIR", "SLURM_CPUS_PER_TASK")}
    if bad:
        raise Stop("environment", "; ".join(bad), 3)


def env_hf(rec):
    """h / t3: the D1-HF job environment (analysis/d1_hf jobs/hf.slurm) without D1HF_* overrides (test_t.env_hf)."""
    TT = G0F.tt()
    try:
        TT.env_hf(rec)
    except TT.Fail as e:
        raise Stop("environment", str(e.args[1] if len(e.args) > 1 else e), 3)


def env_reactot(rec):
    """p / report: jobs/g3.slurm with G3ENV=reactot; D1HF_SCRATCH unset (inputs._hf_modules needs the D1-HF default)."""
    bad = []
    if os.environ.get("G3ENV") != "reactot" or os.environ.get("D1G3_SCRATCH") != str(C.G3):
        bad.append(f"G3ENV={os.environ.get('G3ENV')!r}, D1G3_SCRATCH={os.environ.get('D1G3_SCRATCH')!r} "
                   f"(jobs/g3.slurm with G3ENV=reactot)")
    over = [k for k in HF_KEYS if os.environ.get(k)]
    if over:
        bad.append(f"{over} set in the job environment")
    rec["env"] = {k: os.environ.get(k) for k in ("G3ENV", "D1G3_SCRATCH", "CONDA_DEFAULT_ENV", "SLURM_CPUS_PER_TASK")}
    if bad:
        raise Stop("environment", "; ".join(bad), 3)


def feat_env():
    """d1_features.py children: D1G2_SCRATCH=$G3 so that its ROOT_WRITE_OK guard accepts $G3 (= g0f0.feat_env)."""
    return dict(os.environ, D1G2_SCRATCH=str(C.G3), PYTHONDONTWRITEBYTECODE="1")


def atoms_env(root, scr, res, x182, xtb, r5, versions):
    """fu2/atoms.py root children (= g0f0.atoms_env / phase_fh.atoms_env with the given roots)."""
    g2i = G0F.G2_INPUTS
    return dict(os.environ, D1PAIR_PHASE="FROOT", G1_ROOT=str(root), ESPLEY_OUT=str(xtb), R5_SCRATCH=str(r5),
                ESPLEY_LABELS=str(g2i["labels"]), ESPLEY_META=str(g2i["meta"]), ESPLEY_PROF=str(g2i["prof"]),
                FU2_SCR=str(scr), FU2_RES=str(res), FU2_X182=str(x182), PYTHONDONTWRITEBYTECODE="1", **versions)


def atoms_env_m(m):
    P = paths(m)
    return atoms_env(P.feat, P.fu2_scr, P.fu2_res, P.x182, P.fu2_xtb, P.fu2_r5, P.versions)


def env_h(m):
    """D1-HF children of method m: the g2 geometry slot on $G3/<m>_final (decision F-6)."""
    P = paths(m)
    return dict(os.environ, D1HF_SCRATCH=str(P.hf_scr), D1HF_RES=str(P.hf_res), D1HF_TEMPLATES=str(HF_TEMPLATES),
                D1HF_G2_ROOT=str(P.final), PYTHONDONTWRITEBYTECODE="1")


# ================================================================ G step (G3ENV=d1)
def order_g(lab, T, rid, syms, X, d, meta, M):
    """the atom-order gate of a new structure (decision F-2): raises Fail('order_*'); returns the checks."""
    n = int(lab["n_atoms"])
    lts = M.pc.read_xyz(DP.D1_JOBS / T / "ts.xyz")
    if list(lts[0]) != list(syms) or len(syms) != n:
        raise Fail("order_ts_elements_vs_label")
    if X.shape != (n, 3) or not np.isfinite(X).all():
        raise Fail("order_ts_coordinates")
    A = meta.get(rid)
    if A is None:
        raise Fail("order_no_label_A_idx")
    if len(A) != int(lab["n_f1"]) or n - len(A) != int(lab["n_f2"]):
        raise Fail("order_fragment_sizes")
    if (lab.get("role1"), lab.get("role2")) != ("dipole", "dipolarophile") or \
            (lab.get("rel1_file"), lab.get("rel2_file")) != ("ref_dipole.xyz", "ref_dipolarophile.xyz"):
        raise Fail("order_roles")
    sA = set(A)
    tags = [1 if j in sA else 2 for j in range(n)]
    chk = dict(n_f1=len(A), n_f2=n - len(A))
    for k, ref in ((1, "ref_dipole.xyz"), (2, "ref_dipolarophile.xyz")):
        s, x = M.pc.read_xyz(d / f"rel{k}.xyz")
        x = np.asarray(x, dtype=float)
        lr = M.pc.read_xyz(DP.D1_JOBS / T / ref)
        if list(lr[0]) != list(s):
            raise Fail(f"order_rel{k}_elements_vs_label")
        if x.shape != (len(s), 3) or not np.isfinite(x).all():
            raise Fail(f"order_rel{k}_coordinates")
        if Counter(s) != Counter(syms[j] for j in range(n) if tags[j] == k):
            raise Fail(f"order_rel{k}_elements_vs_ts")
    chk.update(ts_elements_eq_label=True, rel_elements_eq_label=True, rel_elements_eq_ts=True)
    return chk


def g_one(job, meta, ctx):
    """one G folder -> (job, status, category); never raises. Category: skip | ok | order | freq | error |
    not_started | write_error. job = dict(rid, T, lab, src, out, tag)."""
    rid, T, lab, src, out, tag = job["rid"], job["T"], job["lab"], job["src"], job["out"], job["tag"]
    if G0F.marker(out) is not None:
        return job, "skip", "skip"
    if ctx.abort.is_set():
        return job, "not_started_abort", "not_started"
    if time.time() - T0 > STOP_AFTER_H * 3600:
        return job, "not_started_time", "not_started"
    with ctx.lock:                                          # G0-6 byte budget (F-2)
        if ctx.bytes > ctx.budget:
            return job, "not_started_budget", "not_started"
    M = ctx.mods
    t0 = time.time()
    q1, q2 = int(lab["charge1"]), int(lab["charge2"])
    res = dict(rxn_id=rid, job_id=T, geom=tag, level=LEVEL.format(tag=tag, src=src), charge1=q1, charge2=q2,
               label_files={k: lab.get(k) for k in ("ts_file", "rel1_file", "rel2_file")},
               roles=[lab.get("role1"), lab.get("role2")], source=str(src), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    reasons = []
    tmp, lw = out.parent / f"{rid}.tmp", ctx.local / f"{tag}_{rid}"
    try:
        C.assert_path(tmp)
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "work").mkdir(parents=True)
        if lw.exists():
            shutil.rmtree(lw)
        lw.mkdir(parents=True)
        res["source_marker"] = G0F.marker(src)
        if res["source_marker"] != ".done":
            raise Fail("order_source_not_done")
        if any(not (src / f).is_file() for f in XYZ_FILES):
            raise Fail("order_source_files_missing")
        res["sources"] = {f: dict(path=str(src / f), sha256=C.sha256(src / f))
                          for f in XYZ_FILES + ("result.json",) if (src / f).is_file()}
        for f in XYZ_FILES:
            shutil.copyfile(src / f, tmp / f)
        if any(C.sha256(tmp / f) != res["sources"][f]["sha256"] for f in XYZ_FILES):
            raise Fail("order_source_changed_during_copy")
        syms, X = M.pc.read_xyz(tmp / "ts.xyz")                   # = what xtb_slice / ext_features read
        syms, X = list(syms), np.asarray(X, dtype=float)
        res["checks"] = order_g(lab, T, rid, syms, X, tmp, meta, M)
        # ts.hess: XTB2 ALPB(water) Freq at the fixed TS (g2_autode.finalize / g0f0 G0 path, nprocs 1)
        header = M.GA.FREQ_HEADER.format(n=1, m=M.cfg["maxcore_mb"])
        inp = lw / "ts.inp"
        M.OD.write_inp(inp, header, syms, X, charge=q1 + q2, mult=1)
        t1 = time.time()
        outp = M.OD.run(M.cfg, inp)
        res["freq_wall_s"] = time.time() - t1
        txt = outp.read_text(errors="replace") if outp.is_file() else ""
        shutil.copyfile(inp, tmp / "work" / "ts.inp")
        if outp.is_file():
            with open(outp, "rb") as fi, gzip.open(tmp / "work" / "ts.out.gz", "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
        err = lw / "ts.err"
        if err.is_file() and err.stat().st_size:
            res["orca_stderr_tail"] = err.read_text(errors="replace")[-2000:]
        ver = M.H.RE_VER.search(txt)
        res.update(ts_header=header, orca_bin=M.OD.orca_bin(M.cfg), orca_version=ver.group(1) if ver else None,
                   ts_terminated=M.OD.terminated(txt), e_ts_eh=M.OD.last_fspe(txt), ts_run_hours=M.OD.run_hours(txt))
        hess = lw / "ts.hess"
        if not (M.OD.terminated(txt) and hess.is_file()):
            raise Fail("freq_not_terminated")
        try:
            freqs, modes = M.OD.parse_hess(hess)
            imag, mode = M.OD.imag_and_mode(hess)
            hs, hx = G0F.hess_atoms(hess)
        except Exception as e:                             # noqa: BLE001
            res["hess_parse_error"] = f"{type(e).__name__}: {e}"[:300]
            raise Fail("freq_hess_unreadable")
        n = len(syms)
        if freqs.size != 3 * n or modes.shape != (3 * n, 3 * n):
            raise Fail("freq_hess_size")
        if list(hs) != syms:
            raise Fail("freq_hess_atoms")
        shift = hx - X
        dev = float(np.abs(shift - shift.mean(axis=0)).max())
        res.update(hess_geom_max_dev_A=dev, hess_shift_A=shift.mean(axis=0).tolist())
        if not dev <= G0F.TOL_HESS_GEOM_A:
            raise Fail("freq_hess_geometry")
        lp = M.GA.label_pairs(lab)
        heavy = [k for k, s in enumerate(syms) if s != "H"]
        share = None if mode is None else float(M.RT.forming_share(X, mode, lp, heavy))
        nz = freqs[freqs != 0.0]
        res.update(imag_freqs_cm=[float(v) for v in imag], n_imag=len(imag), imag_mode_forming_share=share,
                   lowest_freq_cm=float(nz.min()) if nz.size else None, n_zero_freq=int((freqs == 0.0).sum()),
                   n_freq=int(freqs.size), formed_pairs_label=[list(p) for p in lp],
                   d_form_A=[float(np.linalg.norm(X[a] - X[b])) for a, b in lp],
                   imag_note=None if len(imag) == 1 else f"n_imag = {len(imag)} at the {tag} TS (recorded, G0-4)")
        shutil.copyfile(hess, tmp / "work" / "ts.hess")
    except Fail as e:
        reasons.append(str(e))
    except Exception as e:                                 # noqa: BLE001 — recorded, the folder gets .fail_error_*
        res["traceback"] = traceback.format_exc()[-4000:]
        reasons.append(f"error_{type(e).__name__}")
    finally:
        shutil.rmtree(lw, ignore_errors=True)
    cat = "ok" if not reasons else reasons[0].split("_")[0]
    try:
        if reasons:                                         # a failed folder carries no structure files
            for f in XYZ_FILES + ("work/ts.hess",):
                (tmp / f).unlink(missing_ok=True)
        res.update(status="ok" if not reasons else "fail:" + reasons[0], fail_reasons=reasons,
                   wall_s=time.time() - t0, core_h=(time.time() - t0) / 3600.0, written=now())
        (tmp / "result.json").write_text(json.dumps(G0F.clean(res), indent=1))
        mk = ".done" if not reasons else ".fail_" + re.sub(r"[^A-Za-z0-9_]+", "_", reasons[0])
        (tmp / mk).write_text(now() + "\n" + "\n".join(reasons) + "\n")
        if out.exists() and G0F.marker(out) is None:      # an unfinished folder of this step (never a marked one)
            shutil.rmtree(C.assert_path(out))
        os.replace(tmp, out)
    except Exception as e:                                 # noqa: BLE001 — disk / quota: stop this task
        ctx.abort.set()
        print(f"  {tag} {rid}: WRITE ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return job, f"write_error:{type(e).__name__}", "write_error"
    nb = tree_bytes(out)
    with ctx.lock:
        ctx.bytes += nb
        ctx.n_fin += 1
        if cat in ("freq", "error"):
            ctx.n_bad += 1
            if ctx.n_bad >= G0F.ABORT_MIN and ctx.n_bad > G0F.ABORT_FRAC * ctx.n_fin:
                ctx.abort.set()
    return job, res["status"], cat


def tree_bytes(d):
    """bytes of the files under d (0 when d is absent)."""
    try:
        return sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())
    except OSError:
        return 0


def run_pool(jobs, meta, mods, rec, label):
    """the G0-6 worker loop (canary, thread pool of $SLURM_CPUS_PER_TASK serial ORCA, abort and byte-budget rules)."""
    W = G0F.cpus()
    local = Path(os.environ.get("TMPDIR") or "/tmp") / f"d1g3_{label}_{os.getpid()}"
    b0 = sum(tree_bytes(j["out"]) for j in jobs)            # what earlier runs of these folders already keep
    ctx = SimpleNamespace(lock=threading.Lock(), abort=threading.Event(), n_fin=0, n_bad=0, local=local, mods=mods,
                          bytes=b0, budget=G_BUDGET_GB * 1e9)
    todo = [j for j in jobs if G0F.marker(j["out"]) is None]
    cnt, st_cnt = Counter(), Counter()
    rec[f"{label}_pool"] = dict(n_jobs=len(jobs), n_todo=len(todo), workers=W, local=str(local), bytes_before=b0,
                                budget_gb=G_BUDGET_GB)
    print(f"[{label}] {len(jobs)} folders, {len(todo)} to do, {W} workers", flush=True)
    local.mkdir(parents=True, exist_ok=True)
    try:
        if todo:
            canary = []
            for j in todo[:2]:
                _, st, cat = g_one(j, meta, ctx)
                cnt[cat] += 1
                st_cnt[st if cat not in ("ok", "skip") else cat] += 1
                canary.append(dict(tag=j["tag"], rxn_id=j["rid"], status=st, category=cat))
                print(f"  canary {j['tag']} {j['rid']}: {st}", flush=True)
                if cat not in ("freq", "error"):
                    break
            rec[f"{label}_canary"] = canary
            bad_c = [c for c in canary if c["category"] in ("freq", "error", "write_error")]
            if canary[-1]["category"] == "write_error" or (len(canary) == 2 and len(bad_c) == 2):
                rec[f"{label}_counts"] = dict(cnt)
                raise Stop("canary", f"canary folders failed: {canary} (environment, disk?); fix the cause, then rerun "
                                     f"with 'retry' (moves .fail_freq_* / .fail_error_* folders)", 3)
            with ThreadPoolExecutor(max_workers=W) as ex:
                futs = [ex.submit(g_one, j, meta, ctx) for j in todo[len(canary):]]
                for k, fu in enumerate(as_completed(futs), 1):
                    try:
                        j, st, cat = fu.result()
                        name = f"{j['tag']} {j['rid']}"
                    except Exception as e:                # noqa: BLE001 — a bug in g_one: stop scheduling
                        ctx.abort.set()
                        name, st, cat = "?", f"crash:{type(e).__name__}:{e}", "error"
                    cnt[cat] += 1
                    st_cnt[st if cat not in ("ok", "skip") else cat] += 1
                    if cat not in ("ok", "skip", "not_started") or k % 50 == 0:
                        print(f"  {name}: {st}  [{k}/{len(futs)}] {dict(cnt)}", flush=True)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    rec[f"{label}_counts"], rec[f"{label}_statuses"] = dict(cnt), dict(st_cnt.most_common(40))
    rec[f"{label}_bytes"] = dict(total=ctx.bytes, written=ctx.bytes - b0, budget=ctx.budget)
    if ctx.abort.is_set():
        raise Stop("abort", f"stopped scheduling ({dict(cnt)}): Freq / error failures >= {G0F.ABORT_MIN} and > "
                            f"{G0F.ABORT_FRAC:.0%}, or a write error; read the .fail_* result.json files", 3)
    if st_cnt.get("not_started_budget"):
        raise Stop("budget", f"G byte budget {ctx.budget / 1e9:.2f} GB reached ({ctx.bytes / 1e9:.3f} GB incl. earlier "
                             f"runs; {st_cnt['not_started_budget']} folders not started): check the folder sizes and the "
                             f"quota, then resubmit with a larger D1G3_G_BUDGET_GB (explicit)", 3)
    return cnt


def g_scan(m, base):
    """(summary, rows) of $G3/<m>_feat over the base reactions (live scan; result.json values)."""
    P = paths(m)
    rows = []
    for rid, T in base:
        d = P.feat / str(rid)
        mk = G0F.marker(d)
        r = G0F.read_json(d / "result.json") if mk else {}
        if mk == ".done":
            src = r.get("sources") or {}
            cur = {f: (C.sha256(P.final / str(rid) / f) if (P.final / str(rid) / f).is_file() else None)
                   for f in XYZ_FILES}
            if any((src.get(f) or {}).get("sha256") != cur[f] for f in XYZ_FILES):
                cat, reason = "stale", "phase_O_files_changed_since_the_G_step"   # e.g. opt.py assemble --force
            else:
                cat, reason = "ok", ""
        elif mk is None:
            cat, reason = "open", "no_marker"
        else:
            reason = (r.get("fail_reasons") or [mk[len(".fail_"):]])[0]
            cat = reason.split("_")[0] if reason.split("_")[0] in ("order", "freq", "error") else "error"
        imag = r.get("imag_freqs_cm") or [None]
        rows.append(dict(method=m, rxn_id=rid, job_id=T, marker=mk, category=cat, reason=reason, n_imag=r.get("n_imag"),
                         imag1_cm=imag[0], forming_share=r.get("imag_mode_forming_share"),
                         lowest_freq_cm=r.get("lowest_freq_cm"), hess_geom_max_dev_A=r.get("hess_geom_max_dev_A"),
                         freq_wall_s=r.get("freq_wall_s"), core_h=r.get("core_h")))
    n = len(rows)
    done = [x for x in rows if x["category"] == "ok"]
    fail = [x for x in rows if x["category"] in ("order", "freq", "error")]
    n_open = sum(1 for x in rows if x["category"] == "open")
    stale = sorted(x["rxn_id"] for x in rows if x["category"] == "stale")
    ni = Counter(int(x["n_imag"]) for x in done if x["n_imag"] is not None)
    rate = len(fail) / n if n else float("nan")
    why = ([f"{m}: G-step failures {len(fail)}/{n} = {rate:.4f} > {RATE_MAX} (spec 12 F)"] if n and rate > RATE_MAX
           else []) + ([f"{m}: {len(stale)} G folders are stale copies of changed Phase O files {stale[:10]} "
                        f"(feat.py g {m} retry moves them aside)"] if stale else [])
    s = dict(root=str(P.feat), n_base=n, n_done=len(done), n_open=n_open, n_fail=len(fail), n_stale=len(stale),
             stale=stale[:100], by_category=dict(Counter(x["category"] for x in rows)),
             fail_reasons=dict(Counter(x["reason"] for x in fail).most_common(30)),
             fail_rxn={str(x["rxn_id"]): x["reason"] for x in fail},
             rate=rate, threshold=RATE_MAX, stop=bool(why), stop_reasons=why, complete=n_open == 0 and not stale,
             n_imag={"0": ni.get(0, 0), "1": ni.get(1, 0), "2": ni.get(2, 0), ">=3": sum(v for k, v in ni.items() if k >= 3)},
             imag1_cm=G0F.qstats(x["imag1_cm"] for x in done), forming_share=G0F.qstats(x["forming_share"] for x in done),
             n_share_below_0p5=sum(1 for x in done if isinstance(x["forming_share"], (int, float))
                                   and x["forming_share"] < 0.5),
             core_h=dict(sum=float(sum(x["core_h"] or 0.0 for x in rows)), per_rxn=G0F.qstats(x["core_h"] for x in rows)),
             g0_thresholds_info=dict(order=G0F.ORDER_MAX, freq=G0F.FREQ_MAX,
                                     order_rate=sum(1 for x in fail if x["category"] == "order") / n if n else None,
                                     freq_rate=sum(1 for x in fail if x["category"] != "order") / n if n else None,
                                     note="D1LADDER G0 thresholds, reported only (decision F-4)"))
    return s, rows


def cmd_g(spec, retry=False):
    ms = list(C.METHODS) if spec == "all" else [resolve_method(spec)]
    name = "g_all" if spec == "all" else f"g_{ms[0]}"

    def body(rec):
        env_d1(rec)
        need_t(rec)
        pa = part_a(rec)
        need_o(rec, pa)
        G0F.g2_inputs_check(rec)
        meta = G0F.read_meta()
        mods = G0F.g0_modules(rec)
        lab = labels()
        jobs, bases, not_ready, moved = [], {}, {}, []
        for m in ms:
            try:
                base, _ = base_of(m, pa, rec)
            except Stop as e:
                if e.rc != 2:
                    raise
                not_ready[m] = e.msg
                continue
            P = paths(m)
            C.assert_path(P.feat).mkdir(parents=True, exist_ok=True)
            if retry:                                      # explicit only: Freq / error failures and stale copies
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for x in g_scan(m, base)[1]:
                    mk = x["marker"]
                    if x["category"] == "stale" or (mk and mk.startswith((".fail_freq", ".fail_error"))):
                        dst = C.assert_path(P.feat_prev / f"{x['rxn_id']}.{stamp}")
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(C.assert_path(P.feat / str(x["rxn_id"])), dst)
                        moved.append(f"{m}/{x['rxn_id']}")
            bases[m] = base
            jobs += [dict(rid=r, T=T, lab=lab[r], src=P.final / str(r), out=P.feat / str(r), tag=m) for r, T in base]
        rec.update(not_ready=not_ready, retry_moved=moved, methods=ms)
        if jobs:
            run_pool(jobs, meta, mods, rec, "g")
        summ, stop = {}, []
        for m, base in bases.items():
            s, _ = g_scan(m, base)
            summ[m] = {k: v for k, v in s.items() if k != "fail_rxn"}
            if s["stop"]:
                stop += s["stop_reasons"] + [f"{m} fail reasons {s['fail_reasons']}"]
        rec["summary"] = summ
        if stop:
            raise Stop("gate", "; ".join(stop), 3)
        return 2 if (not_ready or any(not summ[m]["complete"] for m in summ)) else 0
    return step(name, body)


# ================================================================ F step (d1_pair g.slurm, venv5, D1PAIR_PHASE=F)
def merged(m):
    """the mergeroot report of m if it belongs to the current X182_d1_<m>.parquet and ids_all.csv, else None."""
    P = paths(m)
    if not (P.mrep.is_file() and P.x182.is_file()):
        return None
    r = G0F.read_json(P.mrep)
    if (r.get("x182") or {}).get("sha256") != C.sha256(P.x182) or r.get("ids_csv") != str(P.ids):
        return None
    return r


def root_fp(m, base):
    """F-15: {rxn_id: sha256 of $G3/<m>_feat/<rxn_id>/result.json} over the base reactions (None = no result.json)."""
    P = paths(m)
    return {str(r): (C.sha256(P.feat / str(r) / "result.json") if (P.feat / str(r) / "result.json").is_file() else None)
            for r, _ in sorted(base)}


def f_summary(m, base, gs=None, grows=None):
    """Phase F gate of m from the files (decision F-4): features (mergeroot), B6, Q, T, tokens, combined rate."""
    import pandas as pd
    P = paths(m)
    if grows is None:
        gs, grows = g_scan(m, base)
    ids = sorted(r for r, _ in base)
    g_done = {x["rxn_id"] for x in grows if x["category"] == "ok"}
    stop, missing = [], []
    fpf = P.F / "root_fp.json"                              # F-15
    if not fpf.is_file() or G0F.read_json(fpf) != root_fp(m, base):
        missing.append(f"{fpf}: absent or the feature root changed since the F step (rerun feat.py f {m})")
    out = dict(n_base=len(ids), n_g_done=len(g_done), root=str(P.feat), versions=P.versions)
    feat_fail, b6_fail, x_ids, q_excl, qa_fail, t_bad, tok_ok = {}, {}, None, None, None, None, None
    mr = merged(m)
    if mr is None:
        missing.append(f"{P.mrep} for the current {P.x182}")
    else:
        not_ok = {int(r): w for r, w in (mr.get("not_ok") or {}).items()}
        root_miss = {r: w for r, w in not_ok.items() if str(w).startswith("root:")}
        feat_fail = {r: w for r, w in not_ok.items() if not str(w).startswith("root:")}
        X1 = pd.read_parquet(P.x182)
        x_ids = set(int(r) for r in X1.rxn_id)
        X0_cols = list(pd.read_parquet(DP.X182_D1).columns)
        cols_equal = list(X1.columns) == X0_cols and len(X0_cols) == 183
        E = pd.read_parquet(P.extp, columns=["rxn_id", "ext_status"] + [f"status_B{k}" for k in range(1, 7)])
        b6_fail = {int(r): str(s) for r, s in zip(E.rxn_id, E.status_B6) if str(s) not in ("ok", "skipped")}
        out["features"] = dict(
            mergeroot_report=str(P.mrep), x182=dict(path=str(P.x182), sha256=C.sha256(P.x182), n=len(X1)),
            n_requested=mr.get("n_requested"), n_done=mr.get("n_done"), n_feat=mr.get("n_feat"),
            F1_all_finite=mr.get("F1_all_finite"), columns_equal_X182_d1=cols_equal, n_feature_fail=len(feat_fail),
            feature_fail_classes=dict(Counter(G0F.reason_class(w) for w in feat_fail.values()).most_common(40)),
            feature_fail_rxn={str(r): w for r, w in sorted(feat_fail.items())},
            n_B6_fail=len(b6_fail), B6_reasons=dict(Counter(b6_fail.values()).most_common(20)),
            n_B1_engine_gate=sum(1 for w in feat_fail.values() if "B1:engine_gate" in str(w)))
        if mr.get("stop") or not cols_equal or not mr.get("F1_all_finite"):
            stop.append(f"F-1 (mergeroot): stop {mr.get('stop')}, columns_equal_X182_d1 {cols_equal}, "
                        f"F1_all_finite {mr.get('F1_all_finite')}")
        if (mr.get("n_done") != len(g_done) or mr.get("n_requested") != len(ids) or set(root_miss) != set(ids) - g_done
                or x_ids != g_done - set(feat_fail)):
            stop.append(f"inconsistent feature counts: mergeroot n_done {mr.get('n_done')} / n_requested "
                        f"{mr.get('n_requested')} vs G .done {len(g_done)} / base {len(ids)}, root not done "
                        f"{len(root_miss)}, X182 rows {len(x_ids)} vs {len(g_done) - len(feat_fail)}")
    qf, qm = P.fu2_res / "Q_report.json", P.fu2_res / "Q_manifest.csv"
    if not (qf.is_file() and qm.is_file()):
        missing.append(f"{qf} / {qm}")
    else:
        q = G0F.read_json(qf)
        Q = pd.read_csv(qm)
        q_excl = sorted(int(r) for r in Q.loc[Q.use.astype(str) != "True", "rxn_id"])
        qa_fail = sorted(int(r) for r in Q.loc[Q.qa_ok.astype(str) != "True", "rxn_id"])
        ov = q.get("overrides") or {}
        out["Q"] = dict({k: q.get(k) for k in G0F.Q_KEYS}, n_excluded=len(q_excl), excluded=q_excl,
                        n_qa_fail_manifest=len(qa_fail), overrides=ov)
        if q.get("stop"):
            stop.append(f"Q gate STOP (atoms.py gate, unchanged): {q.get('stop_reason')}")
        want = dict(FU2_X182=str(P.x182), FU2_ATOM_VERSION=P.versions["FU2_ATOM_VERSION"], G1_ROOT=str(P.feat),
                    FU2_SCR=str(P.fu2_scr), FU2_RES=str(P.fu2_res))
        if any(ov.get(k) != v for k, v in want.items()) or q.get("manifest_sha256") != C.sha256(qm):
            stop.append(f"Q_report is not from this run: overrides {ov}")
        if x_ids is not None and set(int(r) for r in Q.rxn_id) != x_ids:
            stop.append(f"Q_manifest rows ({len(Q)}) != X182_d1_{m} rows ({len(x_ids)})")
    tf, tm = P.fu2_res / "T_report.json", P.fu2_res / "T_manifest.csv"
    if not (tf.is_file() and tm.is_file()):
        missing.append(f"{tf} / {tm}")
    else:
        t = G0F.read_json(tf)
        Tm = pd.read_csv(tm)
        t_bad = sorted(int(b["rxn_id"]) for b in (t.get("bad") or []))
        no_file = [int(r) for r, s in zip(Tm.rxn_id, Tm.sha256)
                   if not (P.fu2_scr / "tokens" / f"{int(r)}.npz").is_file()
                   or C.sha256(P.fu2_scr / "tokens" / f"{int(r)}.npz") != s]
        tok_ok = set(int(r) for r in Tm.rxn_id) - set(t_bad) - set(no_file)
        tref_p = DP.RES / "fu2" / "T_report.json"
        tref = G0F.read_json(tref_p)
        lay_bad = [k for k in G0F.T_LAYOUT_KEYS if t.get(k) != tref.get(k)]
        out["T"] = dict({k: t.get(k) for k in G0F.T_KEYS}, n_bad=len(t_bad), bad=t.get("bad"),
                        n_token_files_missing_or_changed=len(no_file), layout_vs_d1pair_g1=dict(ref=str(tref_p),
                                                                                                 differ=lay_bad),
                        manifest=dict(path=str(tm), sha256=C.sha256(tm)), tokens_dir=str(P.fu2_scr / "tokens"))
        if t.get("stop"):
            stop.append(f"T gate STOP (tokens.py, unchanged): gate_Ta {t.get('gate_Ta')}")
        if no_file:
            stop.append(f"{len(no_file)} token npz missing or not equal to T_manifest.csv")
        if t.get("token_version") != P.versions["FU2_TOKEN_VERSION"] or t.get("x182_sha256") != (
                C.sha256(P.x182) if P.x182.is_file() else None) or t.get("q_report_sha256") != (
                C.sha256(qf) if qf.is_file() else None):
            stop.append("T_report is not from this run (token version / X182 / Q_report sha256)")
        if not tref or lay_bad:
            stop.append(f"token layout differs from the D1PAIR g1 rung ({tref_p}) in {lay_bad or 'all (missing)'}")
    out["tokens_ok"] = sorted(tok_ok & set(ids)) if tok_ok is not None else None
    out["n_tokens_ok"] = len(out["tokens_ok"]) if tok_ok is not None else None
    if not missing:
        n_fail = len(ids) - len(tok_ok & set(ids))
        rate = n_fail / len(ids) if ids else float("nan")
        comp = dict(g_step=len(ids) - len(g_done), features=len(feat_fail), b6=len(b6_fail), q_excluded=len(q_excl),
                    q_a=len(qa_fail), t_bad=len(t_bad))
        out["gate"] = dict(n_fail=n_fail, n_base=len(ids), rate=rate, threshold=RATE_MAX, components=comp,
                           component_rates={k: (v / len(ids) if ids else None) for k, v in comp.items()},
                           definition=DECISIONS["F-4 gate"])
        if not rate <= RATE_MAX:
            stop.append(f"{m}: feature gate failures {n_fail}/{len(ids)} = {rate:.4f} > {RATE_MAX} (spec 12 F; "
                        f"components {comp})")
    out.update(missing=missing, complete=not missing, stop=bool(stop), stop_reasons=stop)
    return out


def cmd_f(m):
    P = paths(m)

    def body(rec):
        G0F.env_f(rec)
        need_t(rec)
        pa = part_a(rec)
        need_o(rec, pa)
        base, _ = base_of(m, pa, rec)
        G0F.g2_inputs_check(rec)
        gs, grows = g_scan(m, base)
        rec["G"] = {k: gs[k] for k in ("n_base", "n_done", "n_open", "n_fail", "n_stale", "rate", "stop", "complete")}
        if gs["stop"]:
            raise Stop("G", f"G-step STOP: {gs['stop_reasons']}", 3)
        if not gs["complete"]:
            raise Stop("G", f"{m}: {gs['n_open']} base reactions without a G marker in {P.feat} (run feat.py g)", 2)
        if not gs["n_done"]:
            raise Stop("G", f"{m}: no G-step .done folder", 3)
        # F-15: the F outputs belong to the current feature root, else they are moved aside and recomputed
        fp, fpf = root_fp(m, base), P.F / "root_fp.json"
        if fpf.is_file() and G0F.read_json(fpf) != fp:
            rec["invalidated_stale_F"] = {k: move_aside(p) for k, p in (("F", P.F), ("fu2_scr", P.fu2_scr),
                                                                         ("fu2_res", P.fu2_res), ("x182", P.x182))}
        if not fpf.is_file():
            write_json(fpf, fp)
        txt = "rxn_id\n" + "".join(f"{r}\n" for r, _ in sorted(base))
        if P.ids.is_file():
            if P.ids.read_text() != txt:
                raise Stop("ids", f"{P.ids} differs from the current base list (move {P.F} aside first)", 3)
        else:
            C.write_atomic(P.ids, txt)
        # 1. features: runroot, one slice (0 of 1); skipped when its 4 slice files exist
        tag = "slice00of01"
        have = [f for f in G0F.SLICE_FILES if (P.slices / f"{tag}_{f}").is_file()]
        other = sorted({p.name for p in P.slices.glob("slice*_root.json")} - {f"{tag}_root.json"}) \
            if P.slices.is_dir() else []
        if other:
            raise Stop("slices", f"{P.slices} has other slices {other}", 3)
        if len(have) == len(G0F.SLICE_FILES):
            rec["runroot"] = "skipped (slice files exist)"
        else:
            G0F.child(rec, "runroot", [C.P_DIR / "d1_features.py", "runroot", P.feat, P.out, P.ids, 0, 1]
                      + G0F.inputs_args(), C.P_DIR, feat_env())
        # 2. mergeroot -> $G3/X182_d1_<m>.parquet
        mr = merged(m)
        if mr is None:
            G0F.child(rec, "mergeroot", [C.P_DIR / "d1_features.py", "mergeroot", P.out, P.x182, P.mrep], C.P_DIR,
                      feat_env(), ok=(0, 3))
            mr = merged(m)
            if mr is None:
                raise Stop("mergeroot", f"{P.mrep} does not match {P.x182} after mergeroot", 3)
        else:
            rec["mergeroot_skipped"] = "report matches X182"
        rec["mergeroot"] = {k: mr.get(k) for k in ("n_slices", "n_requested", "n_done", "n_feat", "F_fail_rate", "x182",
                                                   "F1_all_finite", "stop", "reasons")}
        if mr.get("stop"):
            raise Stop("mergeroot", "mergeroot report stop (columns != X182_d1 or non-finite values: F-1)", 3)
        if mr.get("n_done") != gs["n_done"] or mr.get("n_requested") != len(base):
            raise Stop("mergeroot", f"n_done {mr.get('n_done')} / n_requested {mr.get('n_requested')} != G .done "
                                    f"{gs['n_done']} / base {len(base)}", 3)
        if not mr.get("n_feat"):
            raise Stop("mergeroot", f"{m}: no reaction with 182 features", 3)
        # 3. FU2 atoms / gate / tokens (atoms.py root, unchanged gates)
        env = atoms_env_m(m)
        rec["child_env_atoms"] = {k: env[k] for k in G0F.ATOMS_ENV_KEYS}
        atoms_py = C.P_DIR / "fu2" / "atoms.py"
        G0F.child(rec, "atoms_run", [atoms_py, "root", "run", 0, 1], C.P_DIR, env)
        err = sorted((P.fu2_scr / "atoms").glob("*.err.json"))
        if err:                                             # retryable infrastructure failures: one more pass
            rec["n_err_before_retry"] = len(err)
            G0F.child(rec, "atoms_retry", [atoms_py, "root", "run", 0, 1], C.P_DIR, env)
            rec["n_err_after_retry"] = len(list((P.fu2_scr / "atoms").glob("*.err.json")))
        rc = G0F.child(rec, "atoms_gate", [atoms_py, "root", "gate"], C.P_DIR, env, ok=(0, 2, 3))
        if rc == 2:
            raise Stop("Q", "atoms incomplete (atoms.py root gate exit 2): resubmit", 2)
        q = G0F.read_json(P.fu2_res / "Q_report.json")
        rec["Q_report"] = {k: q.get(k) for k in G0F.Q_KEYS}
        if rc == 3 or q.get("stop"):
            raise Stop("Q", f"Q gate STOP (atoms.py gate, unchanged): {q.get('stop_reason')}", 3)
        rc = G0F.child(rec, "tokens", [atoms_py, "root", "tokens"], C.P_DIR, env, ok=(0, 3))
        t = G0F.read_json(P.fu2_res / "T_report.json")
        rec["T_report"] = {k: t.get(k) for k in G0F.T_KEYS}
        if rc == 3:
            raise Stop("T", f"tokens exit 3 (T-a gate STOP or refused): gate_Ta {t.get('gate_Ta')}", 3)
        # 4. gate (decision F-4)
        s = f_summary(m, base, gs, grows)
        rec["summary"] = {k: v for k, v in s.items() if k != "tokens_ok"}
        if s["stop"]:
            raise Stop("gate", "; ".join(s["stop_reasons"]), 3)
        return 0 if s["complete"] else 2
    return step(f"f_{m}", body)


# ================================================================ H step (d1_hf hf.slurm)
def hf_common_mod():
    """analysis/d1_hf/hf_common.py as module 'hf_common' without putting analysis/d1_hf (run.py, build.py, report.py)
    on sys.path (the refs_q._load pattern; refs_q / inputs reuse the same module object). Parsing only here."""
    path = C.HF_DIR / "hf_common.py"
    m = sys.modules.get("hf_common")
    if m is not None:
        if Path(m.__file__).resolve() != path.resolve():
            raise Stop("code", f"module hf_common already imported from {m.__file__}", 3)
        return m
    import importlib.util
    spec = importlib.util.spec_from_file_location("hf_common", str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules["hf_common"] = m
    spec.loader.exec_module(m)
    return m


def hf_stale(P, jobs):
    """job_ids whose run folder inputs (runs/g2/<T>/<calc>.inp) carry other atoms / coordinates than the current
    geometry inputs (inputs/<T>/g2/<calc>.inp, rewritten by every geominputs call from $G3/<m>_final): a Phase O folder
    changed after the HF run (run.py skips .done reactions). Coordinate lines only, so the SlowConv retry text is
    not a difference."""
    H = hf_common_mod()

    def coords(p):
        return [ln.strip() for ln in p.read_text().splitlines() if H.XYZ_LINE.match(ln)]
    out = []
    for T in jobs:
        for c in H.CALCS:
            a, b = P.hf_scr / "runs" / "g2" / T / f"{c}.inp", P.hf_scr / "inputs" / T / "g2" / f"{c}.inp"
            if not (a.is_file() and b.is_file()) or coords(a) != coords(b):
                out.append(T)
                break
    return out


def coreh_jobs(runs, jobs):
    """{job_id: core-h} = sum of timing.tsv wall_seconds (serial ORCA: wall = core time; = phase_fh.coreh_by_job)."""
    out = {}
    for T in jobs:
        f = runs / T / "timing.tsv"
        if not f.is_file():
            continue
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            s = 0.0
            for r in rd:
                try:
                    s += float(r.get("wall_seconds") or 0.0)
                except ValueError:
                    pass
        out[T] = s / 3600.0
    return out


def h_summary(m, base):
    """Phase F HF gate of m (decision F-4; status classes = D1-G2 phase_fh H_FIN / H_GATE / H_OPEN)."""
    import pandas as pd
    if str(L.G2_DIR) not in sys.path:
        sys.path.insert(0, str(L.G2_DIR))
    import phase_fh as PF
    P = paths(m)
    jobs = sorted(T for _, T in base)
    gi_f, hs_f = P.hf_res / "g2_inputs.csv", P.hf_res / "H_status.csv"
    pq, lj = P.hf_res / "labels_d1_hf_g2.parquet", P.hf_scr / "labels_d1_hf_g2.json"
    out = dict(n_base=len(jobs), root=str(P.final), runs=str(P.hf_scr / "runs" / "g2"),
               outputs={str(p): (C.sha256(p) if p.is_file() else None) for p in (gi_f, hs_f, pq, lj)})
    missing = [str(p) for p in (gi_f, hs_f, pq, lj) if not p.is_file()]
    if missing:
        out.update(missing=missing, complete=False, stop=False, stop_reasons=[])
        return out
    gi = pd.read_csv(gi_f, dtype=str, keep_default_na=False)
    gst = dict(zip(gi.job_id, gi.status))
    S = pd.read_csv(hs_f, dtype=str, keep_default_na=False)
    S = S[S.job_id.isin(jobs)]
    by = S.status.value_counts().to_dict()
    fin = S[S.status.isin(PF.H_FIN)]
    gate = S[S.status.isin(PF.H_GATE)]
    inok = {T for T in jobs if gst.get(T) == "ok"}
    open_ = S[S.status.isin(PF.H_OPEN) & S.job_id.isin(inok)]
    absent = sorted(set(jobs) - set(S.job_id))
    input_fail = {T: gst.get(T, "missing") for T in jobs if gst.get(T) != "ok"}
    rate = len(gate) / len(fin) if len(fin) else float("nan")
    in_rate = len(input_fail) / len(jobs) if jobs else float("nan")
    lab = [r for r in C.read_json(lj) if r.get("job_id") in set(jobs)]
    ch = coreh_jobs(P.hf_scr / "runs" / "g2", jobs)
    stale = hf_stale(P, sorted(S.job_id[S.status == "ok"]))
    stop = []
    if stale:
        stop.append(f"{m}: {len(stale)} HF runs do not use the current $G3/{m}_final geometry {stale[:10]} (Phase O "
                    f"changed after H): move $G3/hf_{m}/runs/g2/<T> aside and rerun feat.py h {m}")
    if len(fin) and rate > RATE_MAX:
        stop.append(f"{m}: HF gate failures {len(gate)}/{len(fin)} = {rate:.4f} > {RATE_MAX} (spec 12 F)")
    if jobs and in_rate > RATE_MAX:
        stop.append(f"{m}: HF input check failed for {len(input_fail)}/{len(jobs)} = {in_rate:.4f} > {RATE_MAX}")
    if len(lab) != by.get("ok", 0):
        stop.append(f"{m}: labels_d1_hf_g2.json rows {len(lab)} != H_status ok {by.get('ok', 0)}")
    out.update(by_status=by, n_inputs_ok=len(inok), n_finished=len(fin), n_ok=by.get("ok", 0),
               ok_jobs=sorted(S.job_id[S.status == "ok"]),
               gate_failure=dict(n=len(gate), n_finished=len(fin), rate=rate, threshold=RATE_MAX,
                                 by_status=gate.status.value_counts().to_dict(),
                                 reasons=dict(Counter(gate.reason).most_common(40)), jobs=gate.job_id.tolist()),
               run_fail=dict(n=int((S.status == "fail").sum()),
                             reasons=dict(Counter(S.reason[S.status == "fail"]).most_common(20)),
                             note="SCF not converged after the one SlowConv retry / ORCA error: reported, not a gate"),
               input_fail=dict(n=len(input_fail), rate=in_rate, threshold=RATE_MAX, jobs=input_fail),
               open=dict(n=len(open_), by_status=open_.status.value_counts().to_dict(), jobs=open_.job_id.tolist()[:100]),
               absent=absent[:100], n_absent=len(absent), n_stale_runs=len(stale), stale_runs=stale[:100],
               core_h=dict(sum=float(sum(ch.values())), per_rxn=G0F.qstats(ch.values()),
                           ok_per_rxn=G0F.qstats(v for T, v in ch.items() if T in set(S.job_id[S.status == "ok"])),
                           reference="D1-HF pilot median 0.106 core-h per reaction (5 calculations)"),
               missing=[], complete=not len(open_) and not absent and len(fin) > 0, stop=bool(stop), stop_reasons=stop)
    return out


def cmd_h(m):
    P = paths(m)

    def body(rec):
        import pandas as pd
        env_hf(rec)
        need_t(rec)
        pa = part_a(rec)
        need_o(rec, pa)
        base, _ = base_of(m, pa, rec)
        lab = labels()
        rid = {T: r for r, T in base}
        jobs = sorted(rid)
        lst = C.write_atomic(P.hf_scr / "H_list.csv", "job_id\n" + "".join(f"{T}\n" for T in jobs))
        env = env_h(m)
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        G0F.child(rec, "geominputs", [C.HF_DIR / "prep.py", "geominputs", "g2", lst], C.HF_DIR, env)
        gi = pd.read_csv(P.hf_res / "g2_inputs.csv", dtype=str, keep_default_na=False)
        if sorted(gi.job_id) != jobs:
            raise Stop("geominputs", f"{P.hf_res / 'g2_inputs.csv'}: {len(gi)} rows != the {len(jobs)} base jobs", 3)
        st = dict(zip(gi.job_id, gi.status))
        rec["inputs"] = dict(Counter(st.values()))
        ok = sorted((T for T in jobs if st[T] == "ok"), key=lambda T: (-int(lab[rid[T]]["n_atoms"]), T))
        if not ok:
            raise Stop("geominputs", f"{m}: no reaction with HF inputs ({rec['inputs']})", 3)
        run_csv = C.write_atomic(P.hf_scr / "H_run.csv", "job_id,n_atoms\n"
                                 + "".join(f"{T},{int(lab[rid[T]]['n_atoms'])}\n" for T in ok))
        nw = max(1, G0F.cpus())
        rc_run = G0F.child(rec, "run", [C.HF_DIR / "run.py", "run", run_csv, "g2", "hf", nw], C.HF_DIR, env, ok=(0, 2))
        G0F.child(rec, "allgeom", [C.HF_DIR / "build.py", "allgeom", lst, "g2"], C.HF_DIR, env)
        s = h_summary(m, base)
        rec["summary"] = {k: v for k, v in s.items() if k != "ok_jobs"}
        if s["stop"]:
            raise Stop("gate", "; ".join(s["stop_reasons"]), 3)
        return 2 if (rc_run == 2 or not s["complete"]) else 0
    return step(f"h_{m}", body)


# ================================================================ P step (G3ENV=reactot)
def rung_cfg(m):
    """the inputs.RUNGS entry of method m (decision F-7; the 'g2' geometry slot of the D1-HF runner)."""
    P = paths(m)
    return dict(geom="g2", fu2_res=P.fu2_res, tokens=P.fu2_scr / "tokens", x182=P.x182,
                h_parquet=P.hf_res / "labels_d1_hf_g2.parquet", hf_runs=P.hf_scr / "runs", hf_inputs=P.hf_scr / "inputs",
                atom_version=P.versions["FU2_ATOM_VERSION"], token_version=P.versions["FU2_TOKEN_VERSION"], refs="min")


def cons_own(IN, ov, prov):
    """P20 option (c) for a g3 rung (decision F-7b): rows of constrained Q-2 keys keep their own reference (the result
    refs_min gives for rung 'g2' with g2_q_keys='own'); refs_min's 'min' replacements of those rows are undone."""
    K = IN._Q().load_keys()
    cons = {(str(j), int(k)) for j, k, c in zip(K.job_id, K.frag, K.constrained.astype(bool)) if bool(c)}
    kept = []
    for (j, k), p in sorted(prov.items()):
        if p.get("src") == "min" and (str(j), int(k)) in cons:
            kept.append(dict(row=f"{j}/frag{k}", key=p.get("key"), dev_kcal=(p["e_orig"] - p["e_new"]) * C.KCAL))
            prov[(j, k)] = dict(src="own", src_job=j, src_frag=int(k), key=p.get("key"), e_orig=p.get("e_orig"),
                                e_new=p.get("e_orig"))
            o = ov.get(j)
            if o is not None:
                o.pop(k, None)
                if not o:
                    ov.pop(j)
    rep = dict(rule=DECISIONS["F-7b constrained keys"],
               n_rows_constrained=sum(1 for (j, k) in prov if (str(j), int(k)) in cons),
               n_constrained_kept_own=len(kept), kept=kept[:50],
               dev_kcal_kept_own=G0F.qstats(x["dev_kcal"] for x in kept),
               n_jobs_replaced=len(ov), n_fragments_replaced=int(sum(len(v) for v in ov.values())))
    return ov, prov, rep


def q_ready_rel(IN, rep):
    """decision F-13: refs_v.q_ready (= inputs._q_ready) with the input fingerprint compared by repo-relative path ->
    (code, why). Q_report.json keys its fingerprint by the absolute paths of the worktree Phase Q ran in
    (eda-asm-prediction-d1ladder); refs_q.inputs_fingerprint() run here keys the same files by this worktree's paths."""
    Q = IN._Q()
    f = Path(Q.RES_REPORT)
    if not f.is_file():
        return 3, f"{f} missing (Phase Q has not run)"
    qr = json.loads(f.read_text())
    rep["q_report_sha256"] = C.sha256(f)
    if not qr.get("all_pass"):
        return 3, "Phase Q did not pass (Q_report.json all_pass false)"
    bad = [p for p, s in (qr.get("outputs") or {}).items() if not Path(p).is_file() or C.sha256(p) != s]
    if bad:
        return 3, f"Phase Q outputs changed since Q_report.json: {bad[:5]}"
    anchor = "/analysis/d1_ladder/refs_q.py"

    def rel(fp):
        roots = [k[:-len(anchor)] for k in fp if k.endswith(anchor)]
        if len(roots) != 1:
            return None, None
        r = roots[0]
        return r, {("repo", k[len(r) + 1:]) if k.startswith(r + "/") else ("abs", k): v for k, v in fp.items()}
    r0, a = rel(qr.get("fingerprint") or {})
    r1, b = rel(Q.inputs_fingerprint())
    diff = None if a is None or b is None else sorted(f"{t}:{k}" for (t, k) in set(a) | set(b)
                                                      if a.get((t, k)) != b.get((t, k)))
    rep["q_fingerprint_relative"] = dict(recorded_root=r0, current_root=r1, n_recorded=len(a or {}),
                                         n_current=len(b or {}), differ=diff, decision="F-13")
    if diff is None:
        return 3, f"Phase Q fingerprint without exactly one ...{anchor} key (recorded {r0}, current {r1})"
    if diff:
        return 3, f"Phase Q inputs or code changed since Q_report.json (repo-relative fingerprint): {diff[:5]}"
    return 0, ""


def p_inputs(m, cfg, IN):
    files = [cfg["h_parquet"], cfg["x182"], Path(cfg["fu2_res"]) / "T_report.json", Path(cfg["fu2_res"]) / "Q_report.json",
             Path(cfg["fu2_res"]) / "T_manifest.csv", Path(cfg["fu2_res"]) / "S_splits.json", IN.FU2_S_SPLITS, IN.Q_KEYS,
             L.RES / "Q_report.json", THIS, C.LAD_DIR / "inputs.py", C.LAD_DIR / "refs_q.py", C.LAD_DIR / "refs_v.py",
             C.LAD_DIR / "lad_common.py", C.HF_DIR / "build.py", C.HF_DIR / "hf_common.py"]
    return {str(p): (C.sha256(p) if Path(p).is_file() else None) for p in files}


def cmd_p(m):
    P = paths(m)

    def body(rec):
        env_reactot(rec)
        need_t(rec)
        pa = part_a(rec)
        need_o(rec, pa)
        base, _ = base_of(m, pa, rec)
        gs, grows = g_scan(m, base)
        fs, hs = f_summary(m, base, gs, grows), h_summary(m, base)
        rec["F"] = {k: fs.get(k) for k in ("complete", "stop", "stop_reasons", "missing", "n_tokens_ok", "gate")}
        rec["H"] = {k: hs.get(k) for k in ("complete", "stop", "stop_reasons", "missing", "n_ok", "gate_failure")}
        for name, s in (("F", fs), ("H", hs)):
            if s["stop"]:
                raise Stop(name, f"{m}: Phase F {name} STOP: {s['stop_reasons']}", 3)
            if not s["complete"]:
                raise Stop(name, f"{m}: Phase F {name} not complete ({s.get('missing') or s.get('open')})", 2)
        import inputs as IN
        if Path(IN.__file__).resolve() != (C.LAD_DIR / "inputs.py").resolve():
            raise Stop("code", f"inputs imported from {IN.__file__}", 3)
        cfg = rung_cfg(m)
        if m in IN.RUNGS and IN.RUNGS[m] != cfg:
            raise Stop("code", f"inputs.RUNGS already has {m}", 3)
        IN.RUNGS[m] = cfg
        rep = dict(phase="F (D1LADDER Phase P rules)", spec="docs/specs/D1G3.md 5", method=m, started=now(),
                   git_head=G0F.git_head(), rung_cfg={k: str(v) for k, v in cfg.items()},
                   decisions={k: DECISIONS[k] for k in ("F-7 P tables", "F-7b constrained keys")},
                   inputs_decisions=IN.DECISIONS, base=dict(n=len(base)), stop=False, all_pass=False)
        try:
            code, why = q_ready_rel(IN, rep)               # F-13 (inputs._q_ready cannot pass from this worktree)
            if code:
                raise Stop("Q", f"D1LADDER Phase Q not ready: {why}", code)
            t0 = time.time()
            tj = {d["job_id"]: int(d["rxn_id"]) for d in L.targets()}
            Hq = IN._read_h(m)
            bj = {T for _, T in base}
            extra = sorted(set(Hq.job_id) - bj)
            if extra:
                raise Stop("H", f"{len(extra)} rows of {cfg['h_parquet']} are not base reactions: {extra[:5]}", 3)
            rep["h_source"] = dict(path=str(cfg["h_parquet"]), sha256=C.sha256(cfg["h_parquet"]), n_rows=int(len(Hq)))
            ov, prov, rrep = IN.refs_min(m, Hq, rep)
            rep["references"] = rrep
            ov, prov, rep["references_g3_cons_own"] = cons_own(IN, ov, prov)
            pa_, rr, Hn = IN.assemble_rung(m, Hq, ov, tj, t0)
            rep["P-a"], rep["reassembly"] = pa_, rr
            if not pa_["ok"]:
                raise Stop("P-a", f"gate P-a failed for {m}: see {P.p_report} 'P-a'", 3)
            if not rr["ok"]:
                raise Stop("reassembly", f"re-assembly checks failed for {m}: see {P.p_report} 'reassembly'", 3)
            tg1 = json.loads((Path(IN.RUNGS["g1"]["fu2_res"]) / "T_report.json").read_text())
            tok, X, xcols, tinfo = IN.token_info(m, tg1)
            rep["tokens"] = tinfo
            T = IN.make_table(m, Hq, Hn, prov, tok, X, xcols)
            s = IN.table_summary(T, prov)
            rep["table"] = s
            if s["n_tok_without_x182"]:
                raise Stop("table", f"{s['n_tok_without_x182']} token reactions without an all-finite X182 row", 3)
            if not all(s["channels_unchanged"].values()):
                raise Stop("table", f"H channels changed by the re-referencing: {s['channels_unchanged']}", 3)
            usable = set(int(r) for r in T.rxn_id[T.usable])
            br = {r for r, _ in base}
            if not usable:
                raise Stop("table", f"{m}: no usable reaction", 3)
            if usable - br or set(tok) - br:
                raise Stop("table", f"usable / token reactions outside the base set: {sorted((usable | set(tok)) - br)[:5]}",
                           3)
            if set(tok) != set(fs["tokens_ok"]):
                raise Stop("table", f"token reactions {len(tok)} != F summary tokens_ok {fs['n_tokens_ok']}", 3)
            n_h_ok = len(set(Hq.job_id))
            rep["self_check"] = dict(usable_subset_of_base=True, tokens_eq_F_summary=True, n_h_rows=n_h_ok,
                                     h_rows_eq_H_status_ok=sorted(Hq.job_id) == sorted(hs["ok_jobs"]))
            if not rep["self_check"]["h_rows_eq_H_status_ok"]:
                raise Stop("table", f"{cfg['h_parquet']} rows != H_status ok jobs", 3)
            C.write_atomic(P.table, lambda t: T.to_parquet(t, index=False))
        except (Stop, IN._Stop, SystemExit, Exception) as e:
            if isinstance(e, Stop):
                msg, rc = f"{e.stage}: {e.msg}", e.rc
            elif isinstance(e, IN._Stop):
                msg, rc = f"inputs: {e}", 3
            elif isinstance(e, SystemExit):
                msg, rc = f"exit {e.code} inside an inputs / refs helper (message on stderr)", 3
            else:
                msg, rc = f"unexpected error {type(e).__name__}: {e}", 3
                rep["traceback"] = traceback.format_exc()[-4000:]
            stale = None
            if rc == 3 and P.table.is_file():
                stale = str(P.table.with_name(P.table.name + ".stale"))
                os.replace(C.assert_path(P.table), C.assert_path(stale))
            rep.update(stop=rc == 3, all_pass=False, stop_reason=msg, finished=now(), invalidated=stale)
            write_json(P.p_report, rep)
            raise Stop("P", msg, rc)
        rep.update(outputs={str(P.table): C.sha256(P.table)}, inputs=p_inputs(m, cfg, IN), all_pass=True, stop=False,
                   finished=now(), wall_s=round(time.time() - t0, 1), usable=sorted(usable))
        write_json(P.p_report, rep)
        rec["table"] = {k: s.get(k) for k in ("n_targets", "n_tok", "n_x182", "n_h", "n_usable", "h_changed")}
        rec["references"] = {k: rrep.get(k) for k in ("n_fragments_replaced", "n_jobs_replaced", "n_keys",
                                                      "n_fragments_dev_gt_0_05")}
        rec["cons_own"] = {k: v for k, v in rep["references_g3_cons_own"].items() if k != "kept"}
        return 0
    return step(f"p_{m}", body)


def p_summary(m):
    """results/P_build_<m>.json state: all_pass, table unchanged, inputs unchanged, usable reactions."""
    P = paths(m)
    if not P.p_report.is_file():
        return dict(present=False, all_pass=False, stop=False, table_ok=False, stale=None, n_usable=0, usable=[])
    r = G0F.read_json(P.p_report)
    ok = bool(r.get("all_pass")) and P.table.is_file() and (r.get("outputs") or {}).get(str(P.table)) == C.sha256(P.table)
    stale = None
    if ok:
        stale = sorted(p for p, s in (r.get("inputs") or {}).items() if (C.sha256(p) if Path(p).is_file() else None) != s)
    return dict(present=True, all_pass=bool(r.get("all_pass")), stop=bool(r.get("stop")), stop_reason=r.get("stop_reason"),
                table=str(P.table), table_ok=ok, stale=stale, n_usable=len(r.get("usable") or []),
                usable=r.get("usable") or [], table_summary=r.get("table"),
                references={k: (r.get("references") or {}).get(k) for k in ("n_fragments_replaced", "n_jobs_replaced",
                                                                           "n_keys", "n_fragments_dev_gt_0_05")},
                cons_own={k: v for k, v in (r.get("references_g3_cons_own") or {}).items() if k != "kept"},
                P_a_ok=(r.get("P-a") or {}).get("ok"), reassembly_ok=(r.get("reassembly") or {}).get("ok"))


# ================================================================ T-2
def t_check(k, out, body):
    """T-<k>: results/T_<k>.json marked 'running', body(rec), then the verdict (pass / not_ready / fail)."""
    record_all()
    rec = dict(test=f"T-{k}", spec="docs/specs/D1G3.md 3", started=now(), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), python=sys.executable, git_head=G0F.git_head())
    write_json(out, dict(rec, verdict="running", **{"pass": False}))
    rc = 0
    try:
        rec["code_sha256"] = code_sha()                    # the code this check ran with (need_t compares it)
        body(rec)
    except Stop as e:
        rc = 2 if e.rc == 2 else 3
        rec.update(verdict="not_ready" if rc == 2 else "fail", fail_stage=e.stage, fail=str(e.msg)[:4000],
                   **{"pass": False})
    except SystemExit as e:
        rc = 3
        rec.update(verdict="fail", fail_stage="exit", fail=f"SystemExit({e.code!r}) (message on stderr)", **{"pass": False})
    except Exception as e:                                 # noqa: BLE001 — a crash is a failed check, recorded
        rc = 3
        rec.update(verdict="fail", fail_stage="error", fail=f"{type(e).__name__}: {str(e)[:2000]}",
                   traceback=traceback.format_exc()[-4000:], **{"pass": False})
    rec.setdefault("pass", False)
    rec.setdefault("verdict", "pass" if rec["pass"] else "fail")
    if not rec["pass"] and rc == 0:
        rc = 3
    rec.update(finished=now(), stop=rec["verdict"] == "fail")
    write_json(out, rec)
    print(json.dumps(G0F.clean({x: v for x, v in rec.items() if x != "traceback"}), indent=1, default=str)[:20000],
          flush=True)
    if rec.get("traceback"):
        print(rec["traceback"], file=sys.stderr, flush=True)
    print(f"[T-{k}] {rec['verdict']} -> {out}", flush=True)
    return rc


def fresh_tdir(d, cur, rec):
    """F-16: the T work folder d carries the code sha256 it was made with; another code -> moved aside first."""
    f = d / CODE_MARK
    if d.exists() and (not f.is_file() or f.read_text().strip() != cur):
        rec["stale_moved_to"] = move_aside(d)
    C.assert_path(d).mkdir(parents=True, exist_ok=True)
    if not f.is_file():
        C.write_atomic(f, cur + "\n")


def tdir_current(d, cur):
    f = d / CODE_MARK
    return f.is_file() and f.read_text().strip() == cur


def cmd_t2_g(rerun=False):
    def body(rec):
        env_d1(rec)
        write_json(T2_JSON, dict(test="T-2", verdict="running", started=now(), note="t2_g started; t2_f writes the "
                                 "verdict", **{"pass": False}))
        if rerun:
            rec["previous_moved_to"] = move_aside(T2_DIR)
        rec["code_sha256"] = code_sha()
        fresh_tdir(T2_DIR, rec["code_sha256"], rec)
        lab = labels()
        G0F.g2_inputs_check(rec)
        meta = G0F.read_meta()
        mods = G0F.g0_modules(rec)
        root = C.assert_path(T2_DIR / "root")
        root.mkdir(parents=True, exist_ok=True)
        jobs = []
        for r in T2_RXN:
            src = G0F.DFT_FINAL / str(r)
            if G0F.marker(src) != ".done" or r not in lab:
                raise Stop("t2", f"{src}: no .done (or {r} not a target)", 3)
            jobs.append(dict(rid=r, T=lab[r]["job_id"], lab=lab[r], src=src, out=root / str(r), tag="t2_dft"))
        run_pool(jobs, meta, mods, rec, "t2")
        rec["folders"] = {str(r): G0F.marker(root / str(r)) for r in T2_RXN}
        bad = {r: v for r, v in rec["folders"].items() if v != ".done"}
        if bad:
            raise Stop("t2", f"T-2 G folders not .done: {bad} (read their result.json)", 3)
        return 0
    rc = step("t2_g", body)
    if rc == 3:                                            # T-2 cannot pass: record the failure (t2_f will not run)
        r = G0F.read_json(STEPS / "t2_g.json")
        write_json(T2_JSON, dict(test="T-2", verdict="fail", fail_stage=f"t2_g:{r.get('fail_stage')}",
                                 fail=r.get("fail"), finished=now(), stop=True, **{"pass": False}))
    return rc


def t2_body(rec):
    import pandas as pd
    TT = G0F.tt()
    G0F.env_f(rec)
    if not tdir_current(T2_DIR, rec["code_sha256"]):              # F-16: never pass on another code's folders
        raise Stop("t2_g", f"{T2_DIR} was not made by t2_g with the current code ({CODE_MARK}): run t2_g first", 2)
    root = T2_DIR / "root"
    mk = {r: G0F.marker(root / str(r)) for r in T2_RXN}
    rec["folders"] = {str(r): v for r, v in mk.items()}
    if any(v is None for v in mk.values()):
        raise Stop("t2_g", f"{root}: folders without marker {mk} (run t2_g first)", 2)
    if any(v != ".done" for v in mk.values()):
        raise Stop("t2_g", f"{root}: {mk}", 3)
    # references: X182_d1_dft, F0_tokens_manifest.csv (sha256 = F0_report), dft Q_manifest rows (use, ext_ok)
    f0 = G0F.read_json(G0F.F0_REPORT)
    want_x = ((f0.get("features") or {}).get("x182") or {}).get("sha256")
    want_t = ((f0.get("T") or {}).get("F0_tokens_manifest") or {}).get("sha256")
    got_x, got_t = C.sha256(G0F.X182_DFT), C.sha256(G0F.F0_TOK_MAN)
    rec["references"] = dict(x182=dict(path=str(G0F.X182_DFT), sha256=got_x, F0_report=want_x),
                             tokens_manifest=dict(path=str(G0F.F0_TOK_MAN), sha256=got_t, F0_report=want_t))
    if got_x != want_x or got_t != want_t:
        raise Stop("reference", f"X182_d1_dft / F0_tokens_manifest sha256 != F0_report.json: {rec['references']}", 3)
    Qd = pd.read_csv(G0F.FU2_RES / "Q_manifest.csv")
    Qd = Qd[Qd.rxn_id.astype(int).isin(T2_RXN)]
    okq = sorted(int(r) for r, u, e in zip(Qd.rxn_id, Qd.use.astype(str), Qd.ext_ok.astype(str))
                 if u == "True" and e == "True")
    if okq != T2_RXN:
        raise Stop("selection", f"T-2 reactions not all use / ext_ok in {G0F.FU2_RES / 'Q_manifest.csv'}: {okq}", 3)
    X0 = pd.read_parquet(G0F.X182_DFT)
    # 1. features (runroot / mergeroot, the Phase F form)
    ids = C.write_atomic(T2_DIR / "T2_ids.csv", "rxn_id\n" + "".join(f"{r}\n" for r in T2_RXN))
    out, x182, mrep = T2_DIR / "F" / "out", T2_DIR / "X182_T2.parquet", T2_DIR / "F" / "mergeroot_report.json"
    G0F.child(rec, "runroot", [C.P_DIR / "d1_features.py", "runroot", root, out, ids, 0, 1] + G0F.inputs_args(),
              C.P_DIR, feat_env())
    G0F.child(rec, "mergeroot", [C.P_DIR / "d1_features.py", "mergeroot", out, x182, mrep], C.P_DIR, feat_env(),
              ok=(0, 3))
    m = C.read_json(mrep)
    rec["mergeroot"] = {k: m.get(k) for k in ("n_requested", "n_done", "n_feat", "F_fail_rate", "reasons", "not_ok",
                                              "x182", "F1_all_finite", "versions", "stop")}
    X1 = pd.read_parquet(x182)
    cols = [c for c in X0.columns if c != "rxn_id"]
    if list(X1.columns) != list(X0.columns) or len(cols) != 182:
        raise Stop("columns", f"X182_T2 columns ({len(X1.columns)}) != X182_d1_dft ({len(X0.columns)})", 3)
    a, b = X1.set_index("rxn_id"), X0.set_index("rxn_id")
    a.index, b.index = a.index.astype(int), b.index.astype(int)
    if sorted(a.index) != T2_RXN or set(T2_RXN) - set(b.index):
        raise Stop("rows", f"X182_T2 rows {sorted(a.index)} (not_ok {m.get('not_ok')}); X182_d1_dft missing "
                           f"{sorted(set(T2_RXN) - set(b.index))}", 3)
    A, B = a.loc[T2_RXN, cols].to_numpy(float), b.loc[T2_RXN, cols].to_numpy(float)
    D = np.abs(A - B)
    D[np.isnan(A) & np.isnan(B)] = 0.0
    D[np.isnan(D)] = np.inf
    blk = {c: k for k, cs in DP.R5.BLOCKS.items() for c in cs}
    over = sorted((dict(rxn_id=T2_RXN[i], col=cols[j], block=blk.get(cols[j], "g1feat"), diff=float(D[i, j]),
                        new=float(A[i, j]), ref=float(B[i, j])) for i, j in zip(*np.nonzero(D > TT.TOL_T1))),
                  key=lambda o: -o["diff"])
    mx = float(D.max())
    strict = mx <= TT.TOL_T1
    within = (not strict) and mx <= TT.PRECEDENT_MAX and all(o["col"].startswith(TT.PRECEDENT_FAMILY) for o in over)
    v182 = "pass" if strict else ("pass_f0_precedent" if within else "fail")
    if v182 == "pass_f0_precedent":                        # recorded when it happens (spec 3 T-2 says 1e-8)
        record_decision("F-8 T-2 deviation", f"T-2 182열 최대 |차| {mx:.3g} > 1e-8(명령서 3절 T-2 기준)이고 넘은 열이 모두 "
                                             f"{TT.PRECEDENT_FAMILY}* (<= {TT.PRECEDENT_MAX:g})라 D1-G2 T-1 / D1PAIR F-0 "
                                             f"선례(사용자 결정)로 통과 처리했다: 명령서 기준에서 벗어남(T_2.json "
                                             f"deviation = true).")
    rec["x182_compare"] = dict(n_rows=len(T2_RXN), n_columns=len(cols), max_abs_diff=mx, tol=TT.TOL_T1,
                               bitwise_equal=bool(np.array_equal(A, B, equal_nan=True)), n_over_tol=len(over),
                               over_tolerance=over[:50],
                               max_abs_diff_by_block={k: float(max([D[:, j].max() for j, c in enumerate(cols)
                                                                    if blk.get(c, "g1feat") == k] or [0.0]))
                                                      for k in sorted(set(blk.values()) | {"g1feat"})},
                               precedent=dict(max_abs=TT.PRECEDENT_MAX, family=TT.PRECEDENT_FAMILY + "*",
                                              decision=DP.F0_DECISION, within=bool(within)),
                               verdict=v182)
    # 2. tokens: FU2_X182 = exact copy of the 3 X182_d1_dft rows, versions -dft (decision F-8)
    xref = T2_DIR / "X182_T2ref.parquet"
    x = X0[X0.rxn_id.astype(int).isin(T2_RXN)].sort_values("rxn_id").reset_index(drop=True)
    C.write_atomic(xref, lambda t: x.to_parquet(t, index=False))
    x2 = pd.read_parquet(xref)
    exact = bool(x2.equals(x) and list(x2.columns) == list(X0.columns) and (x2.dtypes == X0.dtypes).all()
                 and sorted(x2.rxn_id.astype(int)) == T2_RXN)
    rec["x182_copy"] = dict(path=str(xref), sha256=C.sha256(xref), exact_copy=exact)
    if not exact:
        raise Stop("x182_copy", f"{xref} is not an exact copy of the 3 X182_d1_dft rows", 3)
    env = atoms_env(root, T2_DIR / "fu2", T2_DIR / "fu2_res", xref, T2_DIR / "fu2_xtb", T2_DIR / "fu2_r5", T2_VERSIONS)
    rec["child_env_atoms"] = {k: env[k] for k in G0F.ATOMS_ENV_KEYS}
    atoms_py = C.P_DIR / "fu2" / "atoms.py"
    G0F.child(rec, "atoms_run", [atoms_py, "root", "run", 0, 1], C.P_DIR, env)
    G0F.child(rec, "atoms_gate", [atoms_py, "root", "gate"], C.P_DIR, env)
    G0F.child(rec, "tokens", [atoms_py, "root", "tokens"], C.P_DIR, env)
    q, t = (G0F.read_json(T2_DIR / "fu2_res" / f) for f in ("Q_report.json", "T_report.json"))
    rec["Q_report"] = {k: q.get(k) for k in G0F.Q_KEYS}
    rec["T_report"] = {k: t.get(k) for k in G0F.T_KEYS}
    Mt = pd.read_csv(G0F.F0_TOK_MAN)
    ref = dict(zip(Mt.rxn_id.astype(int), Mt.sha256))
    rows = []
    for r in T2_RXN:
        p = T2_DIR / "fu2" / "tokens" / f"{r}.npz"
        s = C.sha256(p) if p.is_file() else None
        row = dict(rxn_id=r, sha256=s, ref_sha256=ref.get(r), equal=bool(s is not None and s == ref.get(r)))
        pa_, pb_ = T2_DIR / "fu2" / "atoms" / f"{r}.npz", G0F.FU2_SCR / "atoms" / f"{r}.npz"
        row["atoms_npz_equal_dft"] = bool(pa_.is_file() and pb_.is_file() and C.sha256(pa_) == C.sha256(pb_))
        d1p = G0F.FU2_SCR / "tokens" / f"{r}.npz"
        if s and not row["equal"] and d1p.is_file():
            row["diff_vs_dft_npz"] = TT.npz_diff(p, d1p)
        rows.append(row)
    n_eq = sum(x_["equal"] for x_ in rows)
    rec["tokens"] = dict(rows=rows, n_equal=n_eq, n=len(T2_RXN), reference=str(G0F.F0_TOK_MAN))
    # 3. information: the G step vs the D1LADDER G0 folders
    info = {}
    for r in T2_RXN:
        n_, o_ = G0F.read_json(root / str(r) / "result.json"), G0F.read_json(G0F.DFT_FINAL / str(r) / "result.json")
        hn, ho = root / str(r) / "work" / "ts.hess", G0F.DFT_FINAL / str(r) / "work" / "ts.hess"
        info[str(r)] = dict(xyz_bytes_equal={f: C.sha256(root / str(r) / f) == C.sha256(G0F.DFT_FINAL / str(r) / f)
                                             for f in XYZ_FILES},
                            hess_bytes_equal=bool(hn.is_file() and ho.is_file() and C.sha256(hn) == C.sha256(ho)),
                            imag_freqs_cm=(n_.get("imag_freqs_cm"), o_.get("imag_freqs_cm")),
                            forming_share=(n_.get("imag_mode_forming_share"), o_.get("imag_mode_forming_share")))
    rec["g0_compare_info"] = info
    good = v182 in ("pass", "pass_f0_precedent") and n_eq == len(T2_RXN)
    rec.update(verdict="pass" if good else "fail", deviation=v182 == "pass_f0_precedent", **{"pass": bool(good)})
    rec["summary"] = dict(verdict=rec["verdict"], x182=v182, max_abs_diff_182=mx, tokens_equal=f"{n_eq}/{len(T2_RXN)}",
                          deviation=rec["deviation"])


def cmd_t2_f():
    return t_check("2", T2_JSON, t2_body)


# ================================================================ T-3
def inputs_vs_templates(root, jobs):
    """the g2-slot inputs of T-3 vs the D1-HF hf templates: bytes, elements, header, coordinate max |diff| (report)."""
    H = hf_common_mod()                                    # parse_inp only (this process keeps the D1-HF defaults)
    out = {}
    for T in jobs:
        res = {}
        for c in H.CALCS:
            fa, fb = root / T / "g2" / f"{c}.inp", HF_TEMPLATES / T / "hf" / f"{c}.inp"
            ta, tb = fa.read_bytes(), fb.read_bytes()
            r = dict(identical=ta == tb)
            if not r["identical"]:
                pa_, pb_ = H.parse_inp(ta.decode()), H.parse_inp(tb.decode())
                xa = np.array([z[2:] for z in pa_["atoms"]], float)
                xb = np.array([z[2:] for z in pb_["atoms"]], float)
                r.update(same_elements=[z[:2] for z in pa_["atoms"]] == [z[:2] for z in pb_["atoms"]],
                         same_header=all(pa_[k] == pb_[k] for k in ("bang", "blocks", "frag", "charge", "mult")),
                         max_coord_diff_A=float(np.abs(xa - xb).max()) if xa.shape == xb.shape and xa.size else None)
            res[c] = r
        out[T] = res
    return out


def t3_body(rec, rerun):
    import pandas as pd
    TT = G0F.tt()
    env_hf(rec)
    if rerun:
        rec["previous_moved_to"] = move_aside(T3_DIR)
    fresh_tdir(T3_DIR, rec["code_sha256"], rec)                    # F-16
    base = C.assert_path(T3_DIR)
    base.mkdir(parents=True, exist_ok=True)
    rid_of = {d["job_id"]: r for r, d in labels().items()}
    Href = pd.read_parquet(L.D1HF_LABELS)
    Href = Href[(Href.geom == "dft") & Href.job_id.isin(T3_JOBS)]
    if sorted(Href.job_id) != T3_JOBS or (Href.status != "ok").any():
        raise Stop("reference", f"{L.D1HF_LABELS}: geom=dft rows for {T3_JOBS}: {sorted(Href.job_id)}", 3)
    for T in T3_JOBS:
        if G0F.marker(G0F.DFT_FINAL / str(rid_of[T])) != ".done":
            raise Stop("reference", f"{G0F.DFT_FINAL / str(rid_of[T])} has no .done", 3)
    rec["reference"] = dict(path=str(L.D1HF_LABELS), sha256=C.sha256(L.D1HF_LABELS), root=str(G0F.DFT_FINAL))
    lst = C.write_atomic(base / "t3_list.csv", "job_id\n" + "".join(f"{T}\n" for T in T3_JOBS))
    env = dict(os.environ, D1HF_SCRATCH=str(base), D1HF_RES=str(base / "results"), D1HF_TEMPLATES=str(HF_TEMPLATES),
               D1HF_G2_ROOT=str(G0F.DFT_FINAL), PYTHONDONTWRITEBYTECODE="1")
    rec["child_env"] = {k: env[k] for k in HF_KEYS}
    G0F.child(rec, "geominputs", [C.HF_DIR / "prep.py", "geominputs", "g2", lst], C.HF_DIR, env)
    st = pd.read_csv(base / "results" / "g2_inputs.csv", dtype=str, keep_default_na=False)
    rec["g2_inputs"] = dict(zip(st.job_id, st.status))
    if sorted(st.job_id) != T3_JOBS or (st.status != "ok").any():
        raise Stop("geominputs", f"results/g2_inputs.csv statuses {rec['g2_inputs']}", 3)
    rec["inputs_vs_hf_templates"] = inputs_vs_templates(base / "inputs", T3_JOBS)
    nw = max(1, min(len(T3_JOBS), G0F.cpus()))
    G0F.child(rec, "run", [C.HF_DIR / "run.py", "run", lst, "g2", "hf", nw], C.HF_DIR, env)
    G0F.child(rec, "allgeom", [C.HF_DIR / "build.py", "allgeom", lst, "g2"], C.HF_DIR, env)
    pj = base / "parse_g2.json"
    G0F.child(rec, "parse_g2", [THIS, "_hfparse", lst, pj], C.HF_DIR, env)
    g2 = C.read_json(pj)
    if g2.get("env") != rec["child_env"]:
        raise Stop("parse_g2", f"_hfparse ran with {g2.get('env')}", 3)
    labj = {r["job_id"]: r for r in C.read_json(base / "labels_d1_hf_g2.json")}
    rows = {}
    for T in T3_JOBS:
        a = g2["records"][T]
        b = Href[Href.job_id == T].iloc[0].to_dict()
        d = {}
        for k in TT.T3_KEYS:
            try:
                d[k] = abs(float(a[k]) - float(b[k]))
            except (KeyError, TypeError, ValueError):
                d[k] = float("inf")
        mx = float(np.max(np.array(list(d.values()), dtype=float)))       # NaN propagates (= not <= tol)
        ok = a.get("status") == "ok" and b.get("status") == "ok" and mx <= TT.TOL_T3
        lab_eq = bool(T in labj and all(labj[T].get(k) == a.get(k) for k in TT.T3_KEYS))
        rows[T] = dict(ok=bool(ok), max_abs_diff_kcal=mx, diff=d, status_g2=a.get("status"), reason_g2=a.get("reason"),
                       status_dft=b.get("status"), g2={k: a.get(k) for k in TT.T3_KEYS},
                       dft={k: b.get(k) for k in TT.T3_KEYS}, gates_g2=a.get("gates"), labels_json_row_equal=lab_eq)
    mx = float(np.max(np.array([r["max_abs_diff_kcal"] for r in rows.values()], dtype=float)))
    good = all(r["ok"] and r["labels_json_row_equal"] for r in rows.values())
    rec.update(compare=rows, keys=TT.T3_KEYS, tol_kcal=TT.TOL_T3, max_abs_diff_kcal=mx,
               verdict="pass" if good else "fail", **{"pass": bool(good)})
    rec["summary"] = dict(verdict=rec["verdict"], max_abs_diff_kcal=mx,
                          inputs_identical={T: all(v["identical"] for v in r.values())
                                            for T, r in rec["inputs_vs_hf_templates"].items()},
                          status={T: (r["status_g2"], r["status_dft"]) for T, r in rows.items()})


def cmd_t3(rerun=False):
    return t_check("3", T3_JSON, lambda rec: t3_body(rec, rerun))


def _hfparse(lst, out):
    """child of t3 (D1HF_* overrides in its environment): parse_rxn(T, 'g2') of the listed reactions -> <out>."""
    import pandas as pd
    if str(C.HF_DIR) not in sys.path:
        sys.path.insert(0, str(C.HF_DIR))
    import build as HB
    H = HB.H
    got = dict(D1HF_SCRATCH=H.HF, D1HF_RES=H.RES, D1HF_TEMPLATES=H.TEMPLATES, D1HF_G2_ROOT=H.GEOM_ROOT["g2"])
    bad = {k: (str(v), os.environ.get(k)) for k, v in got.items()
           if not os.environ.get(k) or Path(v) != Path(os.environ[k])}
    if bad or Path(H.HF) == C.HF_SCR:
        C.stop(f"_hfparse: hf_common is not on the D1HF_* overrides: {bad}")
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()
    write_json(out, dict(env={k: os.environ[k] for k in got}, records={T: HB.parse_rxn(T, "g2") for T in ids}))
    return 0


# ================================================================ report (G3ENV=reactot)
def cmd_report():
    def body(rec):
        import pandas as pd
        env_reactot(rec)
        pa = part_a(rec)
        ts = t_state()
        stop, incomplete = [], []
        for k, v in ts.items():
            if v["verdict"] == "fail":
                stop.append(f"T-{k} failed ({v['summary']})")
            elif not v["passed"] or v["stale"]:
                incomplete.append(f"T-{k}: {v['verdict']}{' (stale code)' if v['stale'] else ''}")
        ost = o_state(pa)                                  # F-14
        if ost.get("stop") or ost.get("marker_mismatch"):
            stop.append(f"Phase O: stop {ost.get('stop')} {ost.get('stop_reasons')}, marker mismatch "
                        f"{ost.get('marker_mismatch')}")
        elif not ost.get("ok"):
            incomplete.append(f"Phase O: {ost.get('why') or 'O_report not complete / outputs changed'}")
        SP = pd.read_csv(S_PAIRS)
        rep = dict(spec="docs/specs/D1G3.md 5 (STOP: 12 F)", written=now(), git_head=G0F.git_head(),
                   decisions=DECISIONS, threshold=RATE_MAX, T=ts, O=ost, n_part_a=len(pa), methods={})
        usable_by = {}
        grows_all = []
        for m in C.METHODS:
            base, oinfo = base_of(m, pa, None, need_complete=False)
            if oinfo["n_open"]:
                incomplete.append(f"{m}: Phase O open for {oinfo['n_open']} Part A reactions")
            gs, grows = g_scan(m, base)
            grows_all += grows
            fs = f_summary(m, base, gs, grows)
            hs = h_summary(m, base)
            ps = p_summary(m)
            usable = set(int(x) for x in ps["usable"]) if ps["table_ok"] else set()
            usable_by[m] = usable
            pr = {}
            for cls, g in SP.groupby("class"):
                ok = [(int(a_) in usable and int(b_) in usable) for a_, b_ in zip(g.a_rxn, g.b_rxn)]
                pr[str(cls)] = dict(n_pairs=int(len(g)), n_usable=int(sum(ok)))
            mrep = dict(O=oinfo, G={k: v for k, v in gs.items() if k != "fail_rxn"}, G_fail_rxn=gs["fail_rxn"],
                        F={k: v for k, v in fs.items() if k != "tokens_ok"}, H={k: v for k, v in hs.items()
                                                                                 if k != "ok_jobs"},
                        P={k: v for k, v in ps.items()}, pairs=pr, n_usable=len(usable),
                        core_h=dict(g_step=gs["core_h"], hf=hs.get("core_h"),
                                    f_alloc=G0F.read_json(STEPS / f"f_{m}.json").get("core_h_alloc"),
                                    h_alloc=G0F.read_json(STEPS / f"h_{m}.json").get("core_h_alloc")))
            for name, s in (("G", gs), ("F", fs), ("H", hs)):
                if s.get("stop"):
                    stop += [f"{m} {name}: {x}" for x in (s.get("stop_reasons") or [f"rate {s.get('rate')}"])]
            if not gs["complete"]:
                incomplete.append(f"{m} G: {gs['n_open']} open")
            if not fs["complete"]:
                incomplete.append(f"{m} F: missing {fs['missing']}")
            if not hs["complete"]:
                incomplete.append(f"{m} H: missing {hs.get('missing')} open {(hs.get('open') or {}).get('n')} absent "
                                  f"{hs.get('n_absent')}")
            if ps["stop"]:
                stop.append(f"{m} P: {ps['stop_reason']}")
            elif not ps["table_ok"]:
                incomplete.append(f"{m} P: no passing P_build_{m}.json / table")
            elif ps["stale"]:
                incomplete.append(f"{m} P: inputs changed since the build {ps['stale'][:3]} (rerun p {m})")
            rep["methods"][m] = mrep
        both = usable_by[C.METHODS[0]] & usable_by[C.METHODS[1]]
        rep["pairs_usable_all_methods"] = {str(cls): int(sum(1 for a_, b_ in zip(g.a_rxn, g.b_rxn)
                                                            if int(a_) in both and int(b_) in both))
                                           for cls, g in SP.groupby("class")}
        G = pd.DataFrame(grows_all)
        C.write_atomic(G_STATUS, lambda t: G.to_csv(t, index=False))
        rep.update(g_status_csv=str(G_STATUS), g_status_sha256=C.sha256(G_STATUS), stop=bool(stop), stop_reasons=stop,
                   complete=not incomplete, incomplete=incomplete)
        write_json(F_REPORT, rep)
        rec["report"] = {k: rep[k] for k in ("stop", "stop_reasons", "complete", "incomplete")}
        rec["report"]["n_usable"] = {m: rep["methods"][m]["n_usable"] for m in C.METHODS}
        return 3 if stop else (2 if incomplete else 0)
    return step("report", body)


# ================================================================ CLI
def main(a):
    cmd, n = (a[0] if a else ""), len(a)
    if cmd == "t2_g" and a[1:] in ([], ["--rerun"]):
        return cmd_t2_g(rerun=n == 2)
    if cmd == "t2_f" and n == 1:
        return cmd_t2_f()
    if cmd == "t3" and a[1:] in ([], ["--rerun"]):
        return cmd_t3(rerun=n == 2)
    if cmd == "g" and n in (2, 3) and (n == 2 or a[2] == "retry"):
        return cmd_g(a[1], retry=n == 3)
    if cmd in ("f", "h", "p") and n == 2:
        return dict(f=cmd_f, h=cmd_h, p=cmd_p)[cmd](resolve_method(a[1]))
    if cmd == "report" and n == 1:
        return cmd_report()
    if cmd == "_hfparse" and n == 3:
        return _hfparse(a[1], a[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
