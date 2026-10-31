#!/usr/bin/env python3
r"""feat_geo.py — GEO2b Phase G / F / H / P of the geo rung over D1 + D0 (docs/specs/GEO2.md 6, STOP rules 6-2, read
with GEO1B.md 4-3 and the user decision D0D1 of 2026-10-31: analysis/geo1b/results/D0D1_design_investigation.json
design steps 6-8), the pair-set report of spec 7 (p_report, with D1 and D0 counted apart) and the Phase T checks
T-2 / T-3 (spec 5) plus the D0 smoke checks T-D0F / T-D0H. A port of the GEO2 copy (branch geo2, 0a912e78) of
analysis/d1_g3/feat.py (D1-G3, reviewed and run for g3r); analysis/geo2/g3c.py is the copy of g3_common.py and
analysis/geo2/d0d1.py builds the D0 HF templates (Phase X) and the combined D1 + D0 input root this file reads.
sbatch only; nothing in this file runs on the login node (the T checks and the C tests are its checks).
Reused code is imported unchanged: D1LADDER g0f0 (G0 Freq helpers, child / env_f / constants) and inputs (Phase P
functions), D1-G2 test_t (run, env_f, env_hf, T-1 / T-3 constants) and phase_fh (H status classes, batch rule), D1PAIR
d1_features.py runroot / mergeroot and fu2/atoms.py root run / gate / tokens, D1-HF prep.py geominputs (through the
_geominputs wrapper), run.py run, build.py check / allgeom (through _allgeom) / parse_rxn.

Base (decision GEO-1) = every row of the combined labels ($GEO2B/comb/labels_d0d1.json: the 9,266 D1 targets + the
5,255 D0 rows, job_id order) whose Phase Z folder $GEO2B/geo_final/<rxn_id>/ has .done (build_geo.py; results/
Z_report.json stop false). D1 rxn_id >= 1,000,001, D0 rxn_id 0 .. 5973 (job_id D0G2_<rid:04d>); dataset = rxn_id range.
  G  $GEO2B/geo_feat/<rxn_id>/ in the G1 folder format: ts.xyz, rel1.xyz, rel2.xyz = byte copies of the Z files;
     work/ts.hess = ORCA 6.1.1 '! XTB2 ALPB(water) Freq' at that fixed TS (g0f0 G0 path), result.json (imag_freqs_cm,
     n_imag, imag_mode_forming_share), .done | .fail_<reason>. Slices 'g <i|A> <n>' (GEO-2). Element-order references
     = $GEO2B/comb/jobs/<job_id>/{ts, ref_dipole, ref_dipolarophile}.xyz (D1 job folder | D0 labjobs copy, GEO-12).
     g_report (GEO-11): the spec 6-2 G gate per dataset with ts_fragments + ext_features.block_b6 on every .done folder.
  F  f_slice <i> <n>: d1_features.py runroot $GEO2B/geo_feat $GEO2B/F_geo/out ids i n --labels/--meta/--prof = the
     combined inputs ($GEO2B/comb, GEO-12); f_merge -> $GEO2B/X182_d1_geo.parquet (D1 + D0 rows; name = the trainer
     interface); f_atoms <i> <n>; f_gate (fu2/atoms.py root gate through _qgate: QA_MAX_FAIL = 0.5 % of the reactions,
     GEO-13); f_tokens; f_report (F gate per dataset, GEO-16); f_rest = merge, atoms 0 1, gate, tokens, report.
  H  D1-HF runner on the 'g2' slot: D1HF_SCRATCH=$GEO2B/hf_geo, D1HF_RES=analysis/geo2/results/hf_geo,
     D1HF_TEMPLATES=$GEO2B/comb/hf_templates (D1 -> d1_hf/inputs/<T>, D0 -> the Phase X templates), D1HF_G2_ROOT=
     $GEO2B/geo_final. h_inputs <W> (geominputs via _geominputs, batch plan, the stratified 25 D1 + 25 D0 pilot, GEO-4),
     h_pilot (core-h projection per dataset and n_atoms quartile, STOP > 4,000 core-h; rel gate), h_batch <k> <W>,
     h_build (build.py allgeom via _allgeom), h_report (per dataset run / input failure rates; the rel-energy gate only on
     D1 rows whose G2c choice is the original run, GEO-5).
  P  D1LADDER Phase P rules with inputs.RUNGS['geo'] (fixed in inputs.py; equal to rung_cfg('geo'), GEO-8): targets =
     the combined labels, refs 'min' per Q-2 key over $LAD/Q_keys.parquet + $GEO1B/d0/Q_keys_d0.parquet (D0 keys never
     equal D1 keys), constrained keys keep their own reference (F-7b; D0 'constrained' = alt), P6b per key ->
     $GEO2B/inputs/geo.parquet + results/P_build_geo.json.
  p_report  spec 7 for D1 (P_geo, P_23, P_023 over S_folds.csv; STOP |P_23| < 0.98 |P_g2|) and the D0 pairs of
     $GEO1B/d0/D0_pairs.csv (usable pairs per fold and edge kind, pairs lost because an end is unusable) ->
     results/P_geo_sets.json, results/P_geo_report.json.
T-2  $GEO2B/t2/root = byte copies of 3 $LAD/dft_final folders -> G + F with the combined inputs: 182 columns vs
     X182_d1_dft (<= 1e-8), token sha256 vs F0_tokens_manifest.csv -> results/T_2.json
T-3  the H path of this file (_geominputs, run.py, _allgeom, parse_rxn) with D1HF_G2_ROOT = $LAD/dft_final for T00003,
     T00019, T00614 vs labels_d1_hf.parquet geom=dft (<= 1e-6 kcal/mol) -> results/T_3.json
T-D0 20 D0 reactions (n_atoms quartiles, charged ones included) on the Coley TS / references (d0_g2 labjobs): td0_g (G),
     td0_f (F, Q, tokens; results/T_d0f.json), td0_h (H inputs = the Phase X templates, HF runs; results/T_d0h.json).

CLI (exit 0 = done, 2 = not ready / incomplete (resubmit later), 3 = STOP or hard error; every command except the T
checks writes results/F_steps/<step>.json). Job files and the exact sbatch lines: analysis/geo2/RUNBOOK_feat.md.
  t2_g [--rerun] | td0_g [--rerun]       jobs/geo2.slurm GEO2ENV=d1 (submit from analysis/geo2)
  t2_f | td0_f                           analysis/d1_pair jobs/g.slurm, PYENV=venv5, D1PAIR_PHASE=F
  t3 [--rerun] | td0_h [--rerun]         analysis/d1_hf jobs/hf.slurm
  g <i|A|@I> <n> [retry]                 GEO2ENV=d1
  g_report [--accept=B6]                 g.slurm venv5 F
  f_slice | f_atoms <i|A|@I> <n>, f_merge, f_gate, f_tokens, f_report, f_rest   g.slurm venv5 F
  h_inputs <W> | h_pilot | h_batch <k|A|@I> <W> | h_build | h_report             hf.slurm
  p [--accept=F-1[,B6]] | p_report | report [--accept=...]                         GEO2ENV=reactot
  internal: _hfparse <list.csv> <out.json>, _gcheck <ids.csv> <out.json>, _geominputs <list.csv>, _allgeom <list.csv>,
            _qgate
Preconditions: g / g_report / f_* / h_* / p need T-2, T-3, T-D0F, T-D0H passed with the current code (F-10) and the
combined inputs current (results/X_d0d1_report.json, GEO-12); Z_report.json stop false (GEO-1). Exit 3 = STOP: read
results/F_steps/<step>.json (fail_stage, fail) and the gate report, report and wait.
Decisions (user: every judgment = the recommended option): DECISIONS below, written when any command starts to
results/DECISIONS_feat_runtime.md (keys 'feat_geo/<key>', fcntl lock $GEO2B/locks/DECISIONS.lock); the judgment calls of
this port are in results/DECISIONS_feat.md.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_ladder, d1_pair, d1_hf, d1_g2 trees

import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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
import g3c as C  # noqa: E402   (puts analysis/d1_ladder on sys.path: L = lad_common, DP = d1pair_common)
import d0d1 as XD  # noqa: E402  (combined D1 + D0 root, Phase X)
import g0f0 as G0F  # noqa: E402      D1LADDER G0 / F0 helpers (read only, imported)

L, DP = C.L, C.DP
Stop, Fail = G0F.Stop, G0F.Fail
if Path(G0F.__file__).resolve() != (C.LAD_DIR / "g0f0.py").resolve():
    C.stop(f"g0f0 imported from {G0F.__file__}, not {C.LAD_DIR / 'g0f0.py'}")

# ---------------------------------------------------------------- constants
GEO = C.METHODS[0]                                        # "geo" (spec 6-1)
DATASETS = ("D1", "D0")
T0 = float(os.environ.get("GEO2_JOB_T0") or os.environ.get("D1HF_JOB_T0") or time.time())
STOP_AFTER_H = 44.0                                     # no new G reaction after this (48 h wall)
# spec 6-2 STOP rules (= D1LADDER G0 / F0; D1-G2 H), applied per dataset (GEO-16)
ORDER_MAX, FREQ_MAX, B6_MAX = G0F.ORDER_MAX, G0F.FREQ_MAX, G0F.B6_MAX          # G: 0.5 %, 1 %; B6 1 %
F_RATE_MAX = G0F.F0_RATE_MAX                            # F: F-1, B6, Q-a, T-a 1 %
H_RUN_MAX, H_INPUT_MAX, H_REL_MAX = 0.01, 0.01, 0.005   # H: run failures, input check, rel-energy gate
H_REL_TOL_EH = 1e-6                                     # spec 6-1 rel-energy gate (per reaction)
QA_MAX_RATE = 0.005                                     # GEO-13: atoms.py Q-a limit = ceil(0.5 % of the reactions)
PILOT_QUOTA = (6, 6, 6, 7)                              # GEO-4: 25 per dataset over the 4 n_atoms quartiles
H_COREH_MAX = 4000.0                                    # GEO-4 (D0D1 decision; GEO2.md 6-1 said 2,500 for D1)
D1G2_H_REF = dict(n=8616, core_h=1611.0, per_rxn=0.187, source="spec 6-1 (D1-G2 Phase H)")
ACCEPTABLE = ("F-1", "B6")                              # GEO-7: rate rules a user decision may accept (--accept=)
P_MIN_RATIO = 0.98                                      # spec 7: |P_23| >= 0.98 |P_g2| (D1)
SET_NAMES = ("P_geo", "P_23", "P_023", "P_g2", "P_dft")
STEPS = C.RES / "F_steps"
F_REPORT = C.RES / "F_report.json"
G_STATUS = C.RES / "F_G_status.csv"
H_REPORT, H_PILOT = C.RES / "H_report.json", C.RES / "H_pilot.json"
P_SETS, P_SETS_REPORT = C.RES / "P_geo_sets.json", C.RES / "P_geo_report.json"
Z_REPORT, Z_STATUS = C.RES / "Z_report.json", C.RES / "Z_status.csv"          # build_geo.py (Phase Z)
T2_JSON, T3_JSON = C.RES / "T_2.json", C.RES / "T_3.json"
TD0F_JSON, TD0H_JSON = C.RES / "T_d0f.json", C.RES / "T_d0h.json"
T_FILES = {"2": T2_JSON, "3": T3_JSON, "D0F": TD0F_JSON, "D0H": TD0H_JSON}
G_REPORT = C.RES / "G_report.json"                      # g_report (GEO-11): the spec 6-2 G gate incl. B6
G_CHECK = C.G3 / f"G_{GEO}"                             # $GEO2B/G_geo: gate_ids.csv, gate.json, gate_xtb, gate_r5
T2_DIR, T3_DIR, TD0_DIR, TD0H_DIR = C.G3 / "t2", C.G3 / "t3", C.G3 / "td0", C.G3 / "td0h"
G2_HF_LABELS = C.G2_RES / "labels_d1_hf_g2.parquet"   # D1-G2 HF labels (read only; rel-energy gate, calibration)
D1_G2_FINAL = C.G2 / "g2_final"                          # single-run D1-G2 folders (read only; rel gate byte check)
G2C_SELECT = Path(os.environ.get("G2C_RESULTS", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-g2c/analysis/g2c/"
                                 "results")) / "C_select.csv"   # = geo1b geo_common.G2C_SELECT (src, rxn_id, chosen)
Q_KEYS_D1 = C.LAD / "Q_keys.parquet"                     # = inputs.Q_KEYS
GEO1B = Path("/gpfs/tmp_cpu2/yeseo1ee/geo1b")
Q_KEYS_D0 = GEO1B / "d0" / "Q_keys_d0.parquet"           # geo1b d0_labels.py (keys within D0)
Q_D0_REPORT = C.REPO / "analysis" / "geo1b" / "results" / "Q_d0_report.json"
S_FOLDS = C.P_DIR / "results" / "S_folds.csv"
C_COMPARE = C.G2_RES / "C_compare.csv"
RETRY_REC = STEPS / "f_atoms_retry.json"
G_BUDGET_GB = float(os.environ.get("GEO2_G_BUDGET_GB", "2"))   # G-step bytes kept per slice run (G0-6 rule, F-2)
CODE_MARK = "code_sha256.txt"                                   # T work folder code stamp (F-16)
T2_RXN = [1000003, 1000019, 1000614]                    # = D1-G2 T3_JOBS (Br, Si, g1_rec)
T3_JOBS = ["T00003", "T00019", "T00614"]
TD0_N = 20                                              # GEO-14: 4 n_atoms quartiles x 5
T2_VERSIONS = dict(FU2_ATOM_VERSION="fu2-atoms-3-dft", FU2_TOKEN_VERSION="fu2-tokens-3-dft")   # = g0f0.VERSIONS
TD0_VERSIONS = dict(FU2_ATOM_VERSION="fu2-atoms-3-td0", FU2_TOKEN_VERSION="fu2-tokens-3-td0")
HF_TEMPLATES = XD.COMB_TEMPLATES                        # <T>/hf/*.inp (D1 -> d1_hf/inputs, D0 -> Phase X; GEO-12)
HF_KEYS = ("D1HF_SCRATCH", "D1HF_RES", "D1HF_TEMPLATES", "D1HF_G2_ROOT")
COMB_INPUTS = dict(labels=XD.COMB_LABELS, meta=XD.COMB_META, prof=XD.COMB_PROF)
ORCA_WANT = Path("/home1/yeseo1ee/orca_6_1_1_avx2/orca")
XYZ_FILES = ("ts.xyz", "rel1.xyz", "rel2.xyz")
LEVEL = ("{tag}: TS and references = byte copies of {src} (ts.xyz, rel1.xyz, rel2.xyz); work/ts.hess = ORCA 6.1.1 "
         "XTB2 ALPB(water) Freq at that fixed TS (no optimisation; D1LADDER G0 path)")
CODE_REL = ["analysis/geo2/feat_geo.py", "analysis/geo2/g3c.py", "analysis/geo2/d0d1.py", "analysis/geo2/jobs/geo2.slurm",
            "analysis/d1_ladder/g0f0.py", "analysis/d1_ladder/lad_common.py",
            "analysis/d1_g2/g2_autode.py", "analysis/d1_g2/g2_common.py", "analysis/d1_g2/test_t.py",
            "analysis/d1_g2/phase_fh.py", "analysis/d1_g2/config_g2.yaml",
            "analysis/d1_pair/d1_features.py", "analysis/d1_pair/d1pair_common.py", "analysis/d1_pair/d1_g1.py",
            "analysis/d1_pair/fu2/atoms.py", "analysis/d1_pair/fu2/tokens.py", "analysis/d1_pair/fu2/fu2_common.py",
            "analysis/d1_pair/jobs/env.sh", "analysis/d1_pair/jobs/g_env.sh", "analysis/d1_pair/jobs/g.slurm",
            "analysis/d1_hf/prep.py", "analysis/d1_hf/run.py", "analysis/d1_hf/build.py", "analysis/d1_hf/hf_common.py",
            "analysis/d1_hf/jobs/hf.slurm", "analysis/d1_hf/jobs/hf_env.sh"]
DKEY = "feat_geo/"                                      # prefix of this module's keys in the runtime decision log
DEC_LOG = C.RES / "DECISIONS_feat_runtime.md"

DECISIONS = {
    "GEO-1 base": "기준 집합 = 결합 라벨($GEO2B/comb/labels_d0d1.json: D1 9,266 + D0 5,255) 중 $GEO2B/geo_final/<rxn_id>/"
                  ".done인 반응 전부(job_id 순서). 전제: results/Z_report.json stop false(없음 = exit 2, stop = exit 3)이고 "
                  "Z_status.csv ok/kept 행 = Z_report n_folders(정수 또는 dataset별 정수의 합) = .done 표지. 대체(fallback) "
                  "폴더도 .done이므로 들어간다(명령서 0-2). D1 rxn_id >= 1,000,001, D0 0..5973(5,255행)이라 rxn_id 범위로 dataset을 정한다.",
    "GEO-2 G slices": "G = 'g <i|A> <n>'(LADDER g0f0 g0 방식): 기준 반응(job_id 순서)의 i::n 조각에 feat.py의 g_one / run_pool "
                      "/ order_g를 쓴다(조각마다 canary·abort·바이트 예산 2 GB). 조각은 stale만 STOP, 남은 반응이 있으면 "
                      "exit 2. 명령서 6-2 G 비율은 g_report가 dataset마다 전체 기준 집합에서 계산하고 f_* / p / report가 "
                      "다시 확인한다. 'retry'는 그 조각의 .fail_freq_* / .fail_error_* / stale 폴더만 옮긴다(명시적일 때만).",
    "F-1 feature root": "feature 루트 = $GEO2B/geo_feat/<rxn_id>(G1 폴더 형식): geo_final의 ts.xyz·rel1.xyz·rel2.xyz 바이트 "
                        "복사 + result.json·work/ts.hess(XTB2 ALPB(water) Freq). geo_final은 Z가 만든 그대로 두고 사본에 쓴다"
                        "(D1-G3 F-1). 사본의 sources sha256이 지금 geo_final 파일과 다르면 stale = STOP. D1-HF는 geo_final을 "
                        "읽는다(D1HF_G2_ROOT).",
    "F-2 G step": "work/ts.hess = g0f0 G0 경로(FREQ_HEADER nprocs 1, %maxcore = config_g2 maxcore_mb; orca_direct; "
                  "g0f0.hess_atoms; run_ts.forming_share(라벨 형성 결합); TOL_HESS_GEOM_A 1e-4; canary·abort·바이트 예산). "
                  "전하 = 라벨 charge1 + charge2(D0의 전하 반응 260개 포함), mult 1. 원자 순서 gate의 기준 파일 = "
                  "$GEO2B/comb/jobs/<job_id>/{ts, ref_dipole, ref_dipolarophile}.xyz(D1 = d1_jobs, D0 = d0_g2 labjobs = "
                  "Coley 파일의 바이트 사본, Phase X X-2가 확인). 역할 dipole/dipolarophile은 두 dataset 모두, rel 파일 "
                  "이름(ref_dipole / ref_dipolarophile.xyz) 검사는 D1만(D0 라벨의 rel 파일 이름은 Coley 이름). n_imag 0 / >= 2는 "
                  ".done(G0-4, 명령서 6-2).",
    "F-4 gate": "명령서 6-2 STOP 기준을 dataset(D1, D0)마다 따로 적용한다(GEO-16). G(g_report): 원자 순서(순서 gate + "
                "ts_fragments) > 기준 반응의 0.5 %, Freq / 오류 > 1 %, B6 계산 불가 > .done의 1 %, stale. F(G .done 수 기준): "
                "F-1(유한한 182열 행 없음, B6만 실패 포함 = g0f0 f0_report) > 1 %, B6 > 1 %, Q-a > 1 %, T-a > 1 %, FU2 gate 자체의 "
                "STOP, mergeroot stop, 개수 불일치, 토큰 배치 != D1PAIR g1, G의 B6 예측 != F status_B6. H: 실행 실패 / 끝난 반응 "
                "> 1 %(GEO-6), 입력 검사 실패 > 1 %, rel 에너지 gate(GEO-5) > 0.5 %. n_imag 0 / >= 2는 실패가 아니다.",
    "F-5 F layout": "F = g0f0 F0 구조: f_slice <i> <n> -> f_atoms <i> <n>(선택, 배열) -> f_rest(f_merge, f_atoms 0 1(캐시), "
                    "f_gate(err.json 재시도 1회 + _qgate), f_tokens, f_report). 버전 fu2-atoms-3-geo / fu2-tokens-3-geo; "
                    "D1G2_SCRATCH=$GEO2B는 d1_features 자식 환경에만(F0-2); 입력 = 결합 입력(--labels/--meta/--prof = "
                    "$GEO2B/comb, GEO-12). root_fp.json·ids_all.csv는 f_slice가 $GEO2B/locks/F_geo.lock 안에서 쓴다.",
    "F-6 HF": "D1-HF runner의 g2 자리를 geo로: D1HF_SCRATCH=$GEO2B/hf_geo, D1HF_RES=results/hf_geo, D1HF_TEMPLATES="
              "$GEO2B/comb/hf_templates(D1 -> d1_hf/inputs/<T>, D0 -> Phase X 템플릿; 읽기만), D1HF_G2_ROOT=$GEO2B/geo_final. "
              "D1-G2 phase_fh H 구조: h_inputs <W>(_geominputs 한 번 -> 입력 ok 반응을 (n_atoms, job_id) round robin, nb = W x "
              "ceil(n / (1000 W)), worker k = j % W; pilot 목록) -> h_pilot -> h_batch <k> <W>(run.py run, 큰 반응 먼저, "
              "$SLURM_CPUS_PER_TASK개 직렬 ORCA; 배치마다 build.py check) -> h_build(_allgeom) -> h_report. SlowConv 재시도 "
              "1회는 run.py 그대로.",
    "GEO-4 pilot": "pilot = dataset마다 25반응(D1 25 + D0 25 = 50): 입력 ok 기준 반응을 dataset마다 (n_atoms, job_id) 순으로 "
                   "4등분(np.array_split)하고 사분위마다 6, 6, 6, 7개를 default_rng(20261024)로 뽑는다; D0 사분위에 전하 "
                   "반응이 있으면 그중 1개는 전하 반응. 예상 = Σ_(dataset, 사분위) (pilot 평균 반응당 코어시간 x 그 칸의 입력 "
                   "ok 반응 수)(층화 추정). 상한 4,000코어시간(D0D1 결정; 명령서 6-1의 2,500은 D1만이었다). 초과·유한하지 "
                   "않음 = STOP. 반응당 코어시간 = timing.tsv wall_seconds 합(직렬 ORCA). D1-G2 보정 예상은 정보. rel gate를 "
                   "pilot에도 적용. h_batch는 지금 plan의 통과한 H_pilot.json이 있어야 돈다.",
    "GEO-5 rel gate": "명령서 6-1 HF 추가 gate를 A-G2c에 맞게 좁힌다: 비교 대상 = D1 반응 중 G2c C_select.csv chosen == 'orig'"
                      "(G2c 폴더 = 단일 run D1-G2 폴더의 사본)이고 geo HF ok이며 analysis/d1_g2/results/labels_d1_hf_g2.parquet에 "
                      "ok 행이 있는 반응. geo_final rel1/rel2.xyz sha256이 d1_g2/g2_final의 것과 다르거나 e_frag{k}_rel_eh 차이 "
                      "> 1e-6 Eh면 불일치. 불일치 / 비교 반응 > 0.5 % = STOP. r1/r2가 뽑힌 D1과 D0에는 기준이 없다(보고만). "
                      "C_select.csv에 G2c 선택(orig/r1/r2)이 없는 D1 geo 반응이 있으면 STOP(검토 R-3: 비교 대상이 비어 "
                      "gate가 그냥 통과하지 않게).",
    "GEO-6 H run failure": "명령서 6-2 H '실행 실패 > 1 %' = (H_status fail + gate_fail + eda_table_unexplained + parse_fail) / "
                           "끝난 반응, dataset마다. h_batch의 배치 누적 검사(M_checks)는 두 dataset을 합친 비율로 미리 "
                           "멈춘다(가중 평균 > 1 %면 어느 한 dataset이 > 1 %이므로 잘못된 STOP이 아니다).",
    "GEO-7 accept": "F rate STOP(F-1, B6)을 사용자가 받아들이면 'p --accept=F-1[,B6]' / 'report --accept=...' / 'g_report "
                    "--accept=B6'가 그 rate 규칙만 accepted_stops로 옮긴다. 기본값은 STOP, 다른 STOP은 받아들일 수 없다.",
    "F-7 P tables": "입력표 = inputs.py 함수 그대로(inputs.RUNGS['geo'] 고정값 = rung_cfg('geo'), GEO-8): Phase Q 준비(F-13) + "
                    "D0 Q 준비(Q_d0_report.json all_pass·stop false·Q_keys_d0 sha256), _read_h, refs_min(Q-2 키 = D1 "
                    "Q_keys.parquet + D0 Q_keys_d0.parquet 이어 붙이기; 키별 최저 HF 기준; P6b는 각 행의 키 formula·WL hash), "
                    "assemble_rung(P-a + 재조립), token_info(g1 배치), make_table(결합 대상 14,521행), table_summary -> "
                    "$GEO2B/inputs/geo.parquet, results/P_build_geo.json. 전제: G·G report·F·H gate가 STOP 없이 complete.",
    "F-7b constrained keys": "제약 키(Q_keys constrained: D1 = '|fixed:' 회전 막힌 dipole, D0 = alt 기준 구조 '|cfg:')는 g2 "
                             "rung의 P20 (c)와 같이 교체하지 않고 자기 기준을 쓴다(DFT 기준 구조에서 나온 상태가 geo H 값에 "
                             "들어가지 않게). refs_min 결과에서 constrained 행의 'min' 교체를 되돌린다(references_g3_cons_own).",
    "GEO-8 P registration": "inputs.py RUNGS['geo']는 명령서 8-1대로 고정이고, cmd_p는 rung_cfg('geo')와 문자열 값으로 같은지 "
                            "확인한다(다르면 STOP). 키 targets(결합 라벨)·keys(D1 + D0 Q 키 파일)는 geo에만 있다.",
    "GEO-9 pair sets": "명령서 7(D1): 짝 universe = D1PAIR S_folds.csv, P_<rung> = a·b 모두 usable. P_dft·P_g2는 LADDER 표에서 "
                       "다시 계산해 LADDER 집합 파일과 같아야 한다. P_23 = P_g2 ∩ P_geo, P_023 = P_dft ∩ P_g2 ∩ P_geo. "
                       "|P_23| < 0.98 |P_g2|(전체 또는 P_main 기준 짝)이면 STOP: D0D1 결정으로 geo-g2 판정은 없지만, geo가 "
                       "g2 위에 만든 것이라 반응을 잃지 않았는지 보는 파이프라인 검사로 남긴다. D0: $GEO1B/d0/D0_pairs.csv "
                       "(D0_pairs_report.json stop false, sha256, g2_source g2c; 예비판이면 exit 2), usable 짝 = 두 끝 usable "
                       "+ ok, fold(d0_folds.csv)·edge_kind별 수, 끝이 usable이 아니라 잃은 짝과 이유. dataset·fold별 usable 반응 수.",
    "F-8 T-2": "T-2 = 1000003, 1000019, 1000614: $LAD/dft_final 바이트 복사 -> G 단계 코드 -> runroot/mergeroot(결합 입력) -> "
               "182열 vs X182_d1_dft(1e-8; D1-G2 T-1 선례 scan_curv_* <= 2.5e-7 = pass_f0_precedent) + 토큰 sha256 vs "
               "F0_tokens_manifest.csv(FU2_X182 = X182_d1_dft 세 행 사본, 버전 -dft). 결합 입력의 D1 행이 G2 입력과 같은 "
               "결과를 내는지도 이것으로 확인한다. 작업 폴더 $GEO2B/t2.",
    "F-9 T-3": "T-3 = 이 파일의 H 경로(_geominputs -> run.py -> _allgeom -> parse_rxn), D1HF_G2_ROOT = $LAD/dft_final, "
               "D1HF_TEMPLATES = 결합 템플릿, T00003·T00019·T00614 vs labels_d1_hf.parquet geom=dft(<= 1e-6 kcal/mol) + "
               "labels_d1_hf_g2.json 행 일치. 작업 폴더 $GEO2B/t3.",
    "F-10 T currency": "g / g_report / f_* / h_* / p는 T_2·T_3·T_d0f·T_d0h.json이 지금 코드(CODE_REL sha256 묶음)로 통과했을 "
                       "때만 돈다(없음·실행 중·코드 바뀜 = exit 2, 실패 = exit 3).",
    "F-11 decision log": "모든 명령이 시작할 때 이 결정을 results/DECISIONS_feat_runtime.md에 'feat_geo/<키>'로 적는다(fcntl 잠금 "
                         "$GEO2B/locks/DECISIONS.lock). results/DECISIONS.md는 다른 역할도 쓰므로 이 파일은 쓰지 않는다.",
    "F-12 quota": "G 폴더는 work/ts.inp, ts.out.gz, ts.hess만 남긴다; HF는 run.py 규칙(.inp, .out.gz, .err.gz).",
    "F-13 Q readiness": "Phase Q 준비 = inputs._q_ready 규칙을 repo 상대 경로로 비교(기록 키가 d1ladder 작업 트리 절대 경로).",
    "F-15 feature root currency": "f_slice가 feature 루트(기준 반응마다 G result.json sha256)를 $GEO2B/F_geo/root_fp.json에 "
                                  "남긴다. 다르면 F 출력을 .prev_<시각>으로 치우고 다시 계산한다.",
    "F-16 T currency of work folders": "T 작업 폴더($GEO2B/t2, t3, td0, td0h)에는 만든 코드의 sha256을 둔다. 다르면 치우고 "
                                       "새로 계산하고, t2_f / td0_f는 t2_g / td0_g가 지금 코드로 만든 폴더에서만 돈다.",
    "GEO-10 removed": "D1-G3의 Part A·Phase O·U-1 논리와 결정(F-3, F-14, U-2f-*)은 해당하지 않아 지웠다.",
    "GEO-11 G report": "명령서 6-2 G 'B6 계산 불가 > 1 %'를 G 단계에서 본다(LADDER g0_report 방식): g_report가 전체 기준 "
                       "집합을 g_scan하고 자식 _gcheck(g0f0._g0c_init / _g0c_one 그대로, 결합 입력)를 모든 .done 폴더에 돌려 "
                       "results/G_report.json을 쓴다. 비율은 dataset마다. f_* 는 지금 기준·G 폴더에 맞는 통과한 G_report가 "
                       "있어야 돈다. F gate는 G의 B6 예측과 F status_B6가 같은지 본다.",
    "GEO-12 combined inputs": "설계 F/H-1: 모든 G·F·H·P와 T가 $GEO2B/comb(analysis/geo2/d0d1.py x)를 읽는다: labels_d0d1.json "
                              "(--labels / ESPLEY_LABELS), input_meta_d0d1.csv(A_idx), prof/, jobs/(원자 순서 기준), "
                              "hf_templates/(D1HF_TEMPLATES), x_manifest_d0d1.csv(X ok 목록). g0f0.g2_inputs_check 대신 "
                              "inputs_check가 results/X_d0d1_report.json(stop false, complete)의 sha256·symlink 목록 digest를 "
                              "지금 파일과 비교하고, 결합 라벨의 D1 행 (job_id, rxn_id) = lad_common.targets()인지 본다.",
    "GEO-13 QA rate": "fu2_common.QA_MAX_FAIL = 42(8,337의 0.5 %, 절대 수)는 결합 반응 수에 맞지 않는다. f_gate는 atoms.py root "
                      "gate를 자식 _qgate로 부른다: root_bind() 뒤 QA_MAX_FAIL = ceil(0.005 x n_target)으로 바꾸고 cmd_gate "
                      "그대로 실행(Q_report qa_max_fail에 그 값이 남고 f_summary가 같은 식으로 확인). 코드 파일은 바꾸지 않는다.",
    "GEO-14 T-D0": "D0 smoke 20반응: X ok D0를 (n_atoms, rxn_id)로 4등분하고 사분위마다 5개(default_rng(20261024)), 사분위에 "
                   "전하 반응이 있으면 그중 1개는 전하 반응. 구조 = Coley TS / 기준(d0_g2 labjobs). td0_g = G(원자 순서·Freq), "
                   "td0_f = F(features, Q, tokens; 배치 = g1), td0_h = H(_geominputs 입력이 Phase X 템플릿과 좌표 <= 1e-6 Å, "
                   "같은 header; HF 20개 status ok). 판정: 경로 오류(원자 순서, 입력 검사, parse/table/setting, Q-a/Q-b, 배치) 0, "
                   "화학적 실패(Freq, B6, ext, SCF) 2개 이하. D0 기준값이 없으므로(X182는 G1 구조, 삭제) 경로 검사다.",
    "GEO-15 HF wrappers": "prep.py geominputs는 X_manifest(D1만)와 d1_jobs/<T>/ts.xyz에, build.py allgeom은 hf_common.targets()"
                          "(labels_d1.json)에 묶여 있어 D0 행이 not_x_ok / rxn_id None이 된다. 파일을 고치지 않고 자식 "
                          "_geominputs가 prep.ok_manifest를 결합 X manifest(x_fail False)로, prep.DP.D1_JOBS를 $GEO2B/comb/jobs로 "
                          "바꾼 뒤 prep.cmd_geominputs('g2', list)를 부르고, 자식 _allgeom이 hf_common.targets를 결합 목록으로 "
                          "바꾼 뒤 build.cmd_all(list, ('g2',))를 부른다. D1 행은 같은 값(같은 링크 대상)이며 T-3이 이 경로를 검사한다.",
    "GEO-16 dataset gates": "실패 비율 gate는 dataset(D1, D0)마다 계산한다(D0D1: D1과 D0을 따로 보고). 합친 비율은 정보.",
    "GEO-17 S-1 gate": "D0 HF 템플릿의 근거인 원래 B3LYP 입력이 라벨을 재현하는지(D0 S-1, d0d1.py s1, results/X_S1.json)를 H "
                       "코어시간 전에 확인한다: h_inputs / h_pilot / h_batch는 X_S1.json pass일 때만 돈다(없음·미완 = exit 2, "
                       "실패 = exit 3). T 검사와 G / F는 S-1을 기다리지 않는다.",
}


# ================================================================ small helpers
def now():
    return G0F.now()


def paths(m):
    """every Phase G / F / H / P path of method m (m = 'geo': final = $GEO2B/geo_final, feat = $GEO2B/geo_feat)."""
    F = C.G3 / f"F_{m}"
    return SimpleNamespace(
        final=C.G3 / f"{m}_final", feat=C.G3 / f"{m}_feat", feat_prev=C.G3 / f"{m}_feat_prev", F=F, out=F / "out",
        slices=F / "out" / "xtb" / "f_slices", ids=F / "ids_all.csv", mrep=F / "mergeroot_report.json",
        g1p=F / "out" / "xtb" / "xtb_features_g1_d1.parquet", extp=F / "out" / "xtb" / "xtb_features_ext_g1_d1.parquet",
        x182=C.G3 / f"X182_d1_{m}.parquet", fu2_scr=C.G3 / f"fu2_{m}", fu2_res=C.RES / f"fu2_{m}", fu2_xtb=F / "fu2_xtb",
        fu2_r5=F / "fu2_r5", versions=dict(FU2_ATOM_VERSION=f"fu2-atoms-3-{m}", FU2_TOKEN_VERSION=f"fu2-tokens-3-{m}"),
        hf_scr=C.G3 / f"hf_{m}", hf_res=C.RES / f"hf_{m}", table=C.G3 / "inputs" / f"{m}.parquet",
        p_report=C.RES / f"P_build_{m}.json")


H_BATCHES = paths(GEO).hf_res / "H_batches"               # plan.json, H_bNN.csv, H_pilot.csv, H_w<k>of<W>.csv
H_PLAN = H_BATCHES / "plan.json"
M_CHECKS = paths(GEO).hf_res / "M_checks"                 # = build.py check output with D1HF_RES = results/hf_geo
H_LIST = paths(GEO).hf_scr / "H_list.csv"                 # the base job_ids (geominputs, allgeom)
PILOT_FILE = "H_pilot.csv"

_LAB = {}


def labels():
    """{rxn_id: label row} of the combined labels ($GEO2B/comb/labels_d0d1.json; GEO-12)."""
    if not _LAB:
        rows = XD.load_comb_labels()
        d = {int(r["rxn_id"]): r for r in rows}
        if len(d) != len(rows):
            C.stop(f"{XD.COMB_LABELS}: duplicate rxn_id")
        _LAB.update(d)
    return _LAB


def ds_of(rid):
    return XD.dataset_of(rid)


def write_json(path, obj):
    return C.write_json(path, G0F.clean(obj))


def record_decision(key, text):
    """append 'feat_geo/<key>' to results/DECISIONS_feat_runtime.md (idempotent per key) under the fcntl lock
    $GEO2B/locks/DECISIONS.lock (F-11)."""
    import fcntl
    lk = C.assert_path(C.G3 / "locks" / "DECISIONS.lock")
    lk.parent.mkdir(parents=True, exist_ok=True)
    with open(lk, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            old = DEC_LOG.read_text() if DEC_LOG.is_file() else \
                "# GEO2b feat_geo 실행 결정 기록 (feat_geo.py DECISIONS; 사람이 쓴 판단 기록은 DECISIONS_feat.md)\n\n"
            line = f"- **{DKEY}{key}**: {text}\n"
            if f"- **{DKEY}{key}**:" not in old:
                C.write_atomic(DEC_LOG, old + line)
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
    """an existing $GEO2B / results path -> <name>.prev_<time> (never deleted)."""
    if p.exists() or p.is_symlink():
        q = C.assert_path(p.with_name(f"{p.name}.prev_{time.strftime('%Y%m%d_%H%M%S')}"))
        os.replace(C.assert_path(p), q)
        return str(q)
    return None


def list_sha(ids):
    """spec 7 interface: sha256 of '\\n'.join(sorted ids) (no trailing newline)."""
    return hashlib.sha256("\n".join(sorted(str(x) for x in ids)).encode()).hexdigest()


def base_sha(base):
    return hashlib.sha256("".join(f"{r},{T}\n" for r, T in sorted(base)).encode()).hexdigest()


def parse_accept(s):
    """'--accept=F-1[,B6]' -> ('F-1', 'B6') (GEO-7)."""
    v = tuple(x for x in s[len("--accept="):].split(",") if x)
    if not v or set(v) - set(ACCEPTABLE):
        raise SystemExit(f"{s}: accept only {ACCEPTABLE} (decision GEO-7)")
    return v


def by_ds(ids):
    """{dataset: set of rxn_ids} of an iterable of rxn_ids."""
    out = {d: set() for d in DATASETS}
    for r in ids:
        out[ds_of(r)].add(int(r))
    return out


def rate_gate(name, fails, base, thr, unit="reactions"):
    """GEO-16: per dataset k / n of the failure set within the base set -> (info, stop reasons)."""
    f, b = by_ds(fails), by_ds(base)
    info, why = {}, []
    for d in DATASETS:
        n, k = len(b[d]), len(f[d] & b[d])
        rate = k / n if n else float("nan")
        info[d] = dict(n=n, n_fail=k, rate=rate, threshold=thr)
        if n and rate > thr:
            why.append(f"{d}: {name} {k}/{n} {unit} = {rate:.4f} > {thr} (spec 6-2; per dataset, GEO-16)")
    nb = sum(len(v) for v in b.values())
    nf = len({int(x) for x in fails} & {int(x) for x in base})
    info["combined_info"] = dict(n=nb, n_fail=nf, rate=(nf / nb) if nb else None)
    return info, why


def qa_max(n):
    """GEO-13: the atoms.py Q-a limit for n reactions = ceil(0.5 % of n) (D1PAIR: 42 = ceil(0.005 x 8,337))."""
    return int(math.ceil(QA_MAX_RATE * int(n)))


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
    """{'2' | '3' | 'D0F' | 'D0H': {...}}: verdict / pass of each T record and whether its code sha256 is current."""
    cur = code_sha()
    out = {}
    for k, f in T_FILES.items():
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
        raise Stop("T", f"{bad} failed (spec 5 / 14 T STOP): read results/T_<k>.json", 3)
    open_ = [f"T-{k} ({'missing' if not v['present'] else v['verdict']}{', stale code' if v['stale'] else ''})"
             for k, v in st.items() if not v["passed"] or v["stale"]]
    if open_:
        raise Stop("T", f"{open_} not passed with the current code (run / rerun t2_g + t2_f, t3, td0_g + td0_f, td0_h)",
                   2)


# ================================================================ combined inputs (GEO-12)
_INP = {}


def inputs_check(rec):
    """GEO-12 (replaces g0f0.g2_inputs_check): results/X_d0d1_report.json complete / stop false, the combined files and
    link folders unchanged since (sha256, listing digests), and the D1 rows of the combined labels = lad_common.targets()
    (job_id, rxn_id). Exit 2 while Phase X has not run, 3 on any difference."""
    if _INP:
        rec["comb_inputs"] = _INP["rec"]
        return _INP["rec"]
    st, pr = XD.comb_state()
    info = dict(report=str(XD.X_REPORT), report_sha256=C.sha256(XD.X_REPORT) if XD.X_REPORT.is_file() else None,
                outputs=st.get("outputs"), link_digests=st.get("link_digests"),
                problems=[w for _, w in pr])
    rec["comb_inputs"] = info
    if pr:
        raise Stop("inputs", "; ".join(w for _, w in pr) + " (d0d1.py x; GEO-12)", max(c for c, _ in pr))
    lab = labels()
    tg = {(str(d["job_id"]), int(d["rxn_id"])) for d in L.targets()}
    d1 = {(str(r["job_id"]), rid) for rid, r in lab.items() if ds_of(rid) == "D1"}
    d0bad = [rid for rid, r in lab.items() if ds_of(rid) == "D0" and r.get("job_id") != XD.job_id(rid)]
    if tg != d1 or d0bad:
        raise Stop("inputs", f"combined labels: D1 rows != lad_common.targets() ({len(d1 ^ tg)} differ) or D0 job_id "
                             f"rule broken {d0bad[:5]}", 3)
    info.update(n_d1=len(d1), n_d0=sum(1 for r in lab if ds_of(r) == "D0"), d1_rows_eq_targets=True)
    _INP["rec"] = info
    return info


def read_meta():
    """{rxn_id: sorted A_idx} of the combined meta csv (= g0f0.read_meta on $GEO2B/comb/input_meta_d0d1.csv)."""
    return XD.read_comb_meta()


def inputs_args():
    """--labels / --meta / --prof of the d1_features.py children (= g0f0.inputs_args on the combined inputs)."""
    return [a for k in ("labels", "meta", "prof") for a in (f"--{k}", COMB_INPUTS[k])]


# ================================================================ the geo base (Phase Z folders; decision GEO-1)
def geo_state():
    """(base [(rxn_id, job_id)] in job_id order, info with 'problems' [(rc, why)]): the combined targets with
    $GEO2B/geo_final/<rxn_id>/.done; Z_report.json (build_geo.py) present with stop false, and its Z_status.csv ok / kept
    rows and n_folders equal to these markers."""
    info = dict(z_report=str(Z_REPORT), present=Z_REPORT.is_file(), problems=[])
    probs = info["problems"]
    if not Z_REPORT.is_file():
        probs.append((2, f"{Z_REPORT} missing (Phase Z: build_geo.py build)"))
        return [], info
    z = G0F.read_json(Z_REPORT)
    info.update(sha256=C.sha256(Z_REPORT), stop=z.get("stop"), stop_reasons=z.get("stop_reasons"),
                n_folders=z.get("n_folders"), n_fallback=z.get("n_fallback"), by_status=z.get("by_status"),
                by_source=z.get("by_source"), fallback_reasons=z.get("fallback_reasons"), e_rxn_check=z.get("e_rxn_check"))
    if z.get("stop") is not False:
        probs.append((3, f"Z_report.json stop = {z.get('stop')!r}: {z.get('stop_reasons')} (spec 4-3 / 14 Z: report and "
                         f"wait)"))
    final = paths(GEO).final
    tg = sorted(labels().values(), key=lambda d: d["job_id"])
    base = [(int(d["rxn_id"]), str(d["job_id"])) for d in tg if G0F.marker(final / str(d["rxn_id"])) == ".done"]
    ids = {r for r, _ in base}
    zs = None
    if Z_STATUS.is_file():
        with open(Z_STATUS, newline="") as f:
            zs = {int(r["rxn_id"]) for r in csv.DictReader(f) if r.get("status") in ("ok", "kept")}
        info["z_status_sha256"] = C.sha256(Z_STATUS)
    nf = z.get("n_folders")
    nf_tot = nf if isinstance(nf, int) else (sum(v for v in nf.values() if isinstance(v, int))
                                             if isinstance(nf, dict) else None)
    if zs is None:
        probs.append((3, f"{Z_STATUS} missing although Z_report.json exists"))
    elif zs != ids or nf_tot != len(ids):
        probs.append((3, f"$GEO2B/geo_final .done markers ({len(ids)}) != Z_status.csv ok/kept rows ({len(zs)}) / "
                         f"Z_report n_folders ({nf}): only markers {sorted(ids - zs)[:10]}, only Z_status "
                         f"{sorted(zs - ids)[:10]} (folders changed after Phase Z)"))
    extra = sorted(r for r in (zs or set()) if r not in labels())
    if extra:
        probs.append((3, f"Z_status.csv rows outside the combined targets: {extra[:10]}"))
    if not base and not probs:
        probs.append((3, f"no target with .done in {final}"))
    info.update(root=str(final), n_targets=len(tg), n_base=len(base), base_sha256=base_sha(base),
                n_base_by_dataset={d: len(v) for d, v in by_ds(ids).items()})
    return base, info


def geo_base(rec):
    """the geo base (GEO-1) or Stop (2 = Z not finished, 3 = Z STOP / folders changed)."""
    base, info = geo_state()
    rec["Z"] = {k: v for k, v in info.items() if k != "problems"}
    pr = info["problems"]
    if pr:
        rc = max(c for c, _ in pr)
        raise Stop("Z", "; ".join(w for _, w in pr), rc)
    return base


# ================================================================ environments
def env_d1(rec):
    """G / t2_g / td0_g: the D1 run environment of jobs/geo2.slurm with GEO2ENV=d1 (ORCA 6.1.1, D1 code copy)."""
    bad = []
    d1py = os.environ.get("D1_PY")
    if not d1py or Path(sys.executable).resolve() != Path(d1py).resolve():
        bad.append(f"python {sys.executable} is not $D1_PY {d1py!r} (jobs/geo2.slurm with GEO2ENV=d1)")
    orca = os.environ.get("ORCA_BIN", "")
    if not (orca and os.access(orca, os.X_OK)) or Path(orca).resolve() != ORCA_WANT.resolve():
        bad.append(f"ORCA_BIN={orca!r} (need executable {ORCA_WANT})")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        bad.append(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r} (d1 env.sh sets 1)")
    if os.environ.get("GEO2_SCRATCH") != str(C.G3) or os.environ.get("GEO2ENV", "d1") != "d1":
        bad.append(f"GEO2_SCRATCH={os.environ.get('GEO2_SCRATCH')!r}, GEO2ENV={os.environ.get('GEO2ENV')!r} "
                   f"(jobs/geo2.slurm with GEO2ENV=d1)")
    rec["env"] = {k: os.environ.get(k) for k in ("D1_PY", "ORCA_BIN", "OMP_NUM_THREADS", "GEO2_SCRATCH", "GEO2ENV",
                                                 "TMPDIR", "SLURM_CPUS_PER_TASK")}
    if bad:
        raise Stop("environment", "; ".join(bad), 3)


def env_hf(rec):
    """h_* / t3 / td0_h: the D1-HF job environment (analysis/d1_hf jobs/hf.slurm) without D1HF_* overrides."""
    TT = G0F.tt()
    try:
        TT.env_hf(rec)
    except TT.Fail as e:
        raise Stop("environment", str(e.args[1] if len(e.args) > 1 else e), 3)


def env_reactot(rec):
    """p / p_report / report: jobs/geo2.slurm with GEO2ENV=reactot; D1HF_* unset (inputs._hf_modules needs the D1-HF
    default)."""
    bad = []
    if os.environ.get("GEO2ENV") != "reactot" or os.environ.get("GEO2_SCRATCH") != str(C.G3):
        bad.append(f"GEO2ENV={os.environ.get('GEO2ENV')!r}, GEO2_SCRATCH={os.environ.get('GEO2_SCRATCH')!r} "
                   f"(jobs/geo2.slurm with GEO2ENV=reactot)")
    over = [k for k in HF_KEYS if os.environ.get(k)]
    if over:
        bad.append(f"{over} set in the job environment")
    rec["env"] = {k: os.environ.get(k) for k in ("GEO2ENV", "GEO2_SCRATCH", "CONDA_DEFAULT_ENV", "SLURM_CPUS_PER_TASK")}
    if bad:
        raise Stop("environment", "; ".join(bad), 3)


def feat_env():
    """d1_features.py children: D1G2_SCRATCH=$GEO2B so that its ROOT_WRITE_OK guard accepts $GEO2B (= g0f0.feat_env)."""
    return dict(os.environ, D1G2_SCRATCH=str(C.G3), PYTHONDONTWRITEBYTECODE="1")


def atoms_env(root, scr, res, x182, xtb, r5, versions):
    """fu2/atoms.py root children (= g0f0.atoms_env / phase_fh.atoms_env) with the combined inputs (GEO-12)."""
    return dict(os.environ, D1PAIR_PHASE="FROOT", G1_ROOT=str(root), ESPLEY_OUT=str(xtb), R5_SCRATCH=str(r5),
                ESPLEY_LABELS=str(COMB_INPUTS["labels"]), ESPLEY_META=str(COMB_INPUTS["meta"]),
                ESPLEY_PROF=str(COMB_INPUTS["prof"]), FU2_SCR=str(scr), FU2_RES=str(res), FU2_X182=str(x182),
                PYTHONDONTWRITEBYTECODE="1", **versions)


def atoms_env_m(m):
    P = paths(m)
    return atoms_env(P.feat, P.fu2_scr, P.fu2_res, P.x182, P.fu2_xtb, P.fu2_r5, P.versions)


def env_h_roots(scr, res, g2_root):
    """D1-HF children: the g2 geometry slot on g2_root with the combined templates (decision F-6)."""
    return dict(os.environ, D1HF_SCRATCH=str(scr), D1HF_RES=str(res), D1HF_TEMPLATES=str(HF_TEMPLATES),
                D1HF_G2_ROOT=str(g2_root), PYTHONDONTWRITEBYTECODE="1")


def env_h(m):
    P = paths(m)
    return env_h_roots(P.hf_scr, P.hf_res, P.final)


# ================================================================ G step (GEO2ENV=d1)
def order_g(lab, T, rid, syms, X, d, meta, M):
    """the atom-order gate of a new structure (decisions F-2, GEO-12): raises Fail('order_*'); returns the checks."""
    n = int(lab["n_atoms"])
    ref = XD.COMB_JOBS / T                                  # D1: d1_jobs/<T>; D0: d0_g2 labjobs/<T> (Coley copies)
    lts = M.pc.read_xyz(ref / "ts.xyz")
    if list(lts[0]) != list(syms) or len(syms) != n:
        raise Fail("order_ts_elements_vs_label")
    if X.shape != (n, 3) or not np.isfinite(X).all():
        raise Fail("order_ts_coordinates")
    A = meta.get(rid)
    if A is None:
        raise Fail("order_no_label_A_idx")
    if len(A) != int(lab["n_f1"]) or n - len(A) != int(lab["n_f2"]):
        raise Fail("order_fragment_sizes")
    if (lab.get("role1"), lab.get("role2")) != ("dipole", "dipolarophile") or (
            ds_of(rid) == "D1" and (lab.get("rel1_file"), lab.get("rel2_file")) != ("ref_dipole.xyz",
                                                                                   "ref_dipolarophile.xyz")):
        raise Fail("order_roles")
    sA = set(A)
    tags = [1 if j in sA else 2 for j in range(n)]
    chk = dict(n_f1=len(A), n_f2=n - len(A), reference=str(ref))
    for k, rname in ((1, "ref_dipole.xyz"), (2, "ref_dipolarophile.xyz")):
        s, x = M.pc.read_xyz(d / f"rel{k}.xyz")
        x = np.asarray(x, dtype=float)
        lr = M.pc.read_xyz(ref / rname)
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
    res = dict(rxn_id=rid, job_id=T, dataset=ds_of(rid), geom=tag, level=LEVEL.format(tag=tag, src=src), charge1=q1,
               charge2=q2, label_files={k: lab.get(k) for k in ("ts_file", "rel1_file", "rel2_file")},
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
    local = Path(os.environ.get("TMPDIR") or "/tmp") / f"geo2b_{label}_{os.getpid()}"
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
                             f"quota, then resubmit with a larger GEO2_G_BUDGET_GB (explicit)", 3)
    return cnt


def g_scan(m, base, feat=None, final=None):
    """(summary, rows) of the G folders over the base reactions (live scan; result.json values). Gate (spec 6-2 G =
    D1LADDER G0 thresholds, per dataset, GEO-16): atom-order failures > 0.5 %, Freq / error failures > 1 % (G0-5:
    unexpected errors count as Freq failures), or stale copies of changed source files."""
    P = paths(m)
    feat, final = Path(feat or P.feat), Path(final or P.final)
    rows = []
    for rid, T in base:
        d = feat / str(rid)
        mk = G0F.marker(d)
        r = G0F.read_json(d / "result.json") if mk else {}
        if mk == ".done":
            src = r.get("sources") or {}
            cur = {f: (C.sha256(final / str(rid) / f) if (final / str(rid) / f).is_file() else None) for f in XYZ_FILES}
            if any((src.get(f) or {}).get("sha256") != cur[f] for f in XYZ_FILES):
                cat, reason = "stale", "source_files_changed_since_the_G_step"
            else:
                cat, reason = "ok", ""
        elif mk is None:
            cat, reason = "open", "no_marker"
        else:
            reason = (r.get("fail_reasons") or [mk[len(".fail_"):]])[0]
            cat = reason.split("_")[0] if reason.split("_")[0] in ("order", "freq", "error") else "error"
        imag = r.get("imag_freqs_cm") or [None]
        rows.append(dict(method=m, dataset=ds_of(rid), rxn_id=rid, job_id=T, marker=mk, category=cat, reason=reason,
                         n_imag=r.get("n_imag"), imag1_cm=imag[0], forming_share=r.get("imag_mode_forming_share"),
                         lowest_freq_cm=r.get("lowest_freq_cm"), hess_geom_max_dev_A=r.get("hess_geom_max_dev_A"),
                         freq_wall_s=r.get("freq_wall_s"), core_h=r.get("core_h")))
    ids = [x["rxn_id"] for x in rows]
    done = [x for x in rows if x["category"] == "ok"]
    fail = [x for x in rows if x["category"] in ("order", "freq", "error")]
    n_open = sum(1 for x in rows if x["category"] == "open")
    stale = sorted(x["rxn_id"] for x in rows if x["category"] == "stale")
    ni = Counter(int(x["n_imag"]) for x in done if x["n_imag"] is not None)
    g_order, why_o = rate_gate("atom-order failures", [x["rxn_id"] for x in fail if x["category"] == "order"], ids,
                               ORDER_MAX)
    g_freq, why_f = rate_gate("Freq / error failures", [x["rxn_id"] for x in fail if x["category"] != "order"], ids,
                              FREQ_MAX)
    why = why_o + why_f
    if stale:
        why.append(f"{m}: {len(stale)} G folders are stale copies of changed source files {stale[:10]} "
                   f"(feat_geo.py g <i> <n> retry moves them aside)")
    s = dict(root=str(feat), n_base=len(rows), n_done=len(done), n_open=n_open, n_fail=len(fail),
             n_order_fail=sum(1 for x in fail if x["category"] == "order"),
             n_freq_fail=sum(1 for x in fail if x["category"] != "order"), n_stale=len(stale), stale=stale[:100],
             by_category=dict(Counter(x["category"] for x in rows)),
             by_dataset={d: dict(Counter(x["category"] for x in rows if x["dataset"] == d)) for d in DATASETS},
             fail_reasons=dict(Counter(x["reason"] for x in fail).most_common(30)),
             fail_rxn={str(x["rxn_id"]): x["reason"] for x in fail},
             rates=dict(order=g_order, freq=g_freq), thresholds=dict(order=ORDER_MAX, freq=FREQ_MAX),
             stop=bool(why), stop_reasons=why, complete=n_open == 0 and not stale,
             n_imag={"0": ni.get(0, 0), "1": ni.get(1, 0), "2": ni.get(2, 0), ">=3": sum(v for k, v in ni.items() if k >= 3)},
             n_imag_note="n_imag 0 / >= 2 is recorded, not a failure (spec 6-2); B6 needs an imaginary mode: gated by "
                         "g_report (block_b6 on the folders, B6 > 1 % per dataset; GEO-11) and again at F",
             imag1_cm=G0F.qstats(x["imag1_cm"] for x in done), forming_share=G0F.qstats(x["forming_share"] for x in done),
             n_share_below_0p5=sum(1 for x in done if isinstance(x["forming_share"], (int, float))
                                   and x["forming_share"] < 0.5),
             core_h=dict(sum=float(sum(x["core_h"] or 0.0 for x in rows)), per_rxn=G0F.qstats(x["core_h"] for x in rows),
                         by_dataset={d: float(sum(x["core_h"] or 0.0 for x in rows if x["dataset"] == d))
                                     for d in DATASETS}))
    return s, rows


G_KEYS = ("n_base", "n_done", "n_open", "n_fail", "n_order_fail", "n_freq_fail", "n_stale", "rates", "stop",
          "stop_reasons", "complete", "n_imag", "by_dataset")


def cmd_g(i, n, retry=False):
    P = paths(GEO)

    def body(rec):
        env_d1(rec)
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        meta = read_meta()
        mods = G0F.g0_modules(rec)
        lab = labels()
        mine = base[i::n]                                  # job_id order (GEO-2 = g0f0 g0 slicing)
        C.assert_path(P.feat).mkdir(parents=True, exist_ok=True)
        moved = []
        if retry:                                          # explicit only: this slice's Freq / error failures, stale copies
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for x in g_scan(GEO, mine)[1]:
                mk = x["marker"]
                if x["category"] == "stale" or (mk and mk.startswith((".fail_freq", ".fail_error"))):
                    dst = C.assert_path(P.feat_prev / f"{x['rxn_id']}.{stamp}")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(C.assert_path(P.feat / str(x["rxn_id"])), dst)
                    moved.append(x["rxn_id"])
        jobs = [dict(rid=r, T=T, lab=lab[r], src=P.final / str(r), out=P.feat / str(r), tag=GEO) for r, T in mine]
        rec.update(slice=dict(i=i, n=n, n_slice=len(mine), n_base=len(base), budget_gb=G_BUDGET_GB,
                              by_dataset={d: len(v) for d, v in by_ds(r for r, _ in mine).items()}), retry_moved=moved)
        if jobs:
            run_pool(jobs, meta, mods, rec, "g")
        s, _ = g_scan(GEO, mine)
        rec["summary_slice"] = {k: v for k, v in s.items() if k not in ("fail_rxn",)}
        if s["n_stale"]:
            raise Stop("stale", f"{s['n_stale']} stale G folders in this slice {s['stale'][:10]} (rerun with 'retry')", 3)
        return 2 if s["n_open"] else 0
    return step(f"g_{i:02d}of{n:02d}", body)


# ================================================================ G report (d1_pair g.slurm, venv5, D1PAIR_PHASE=F; GEO-11)
GR_KEYS = ("present", "current", "complete", "stop", "stop_reasons", "accepted_stops", "counts", "rates", "written",
           "missing")


def fp_sha(fp):
    """sha256 of a root_fp dict (F-15 record of the G folders), key order free."""
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def gcheck_env(root=None, scratch=None):
    """environment of the _gcheck child (= g0f0.g0check_env with the given feature root and the combined inputs)."""
    root, scratch = Path(root or paths(GEO).feat), Path(scratch or G_CHECK)
    return dict(os.environ, G1_ROOT=str(root), ESPLEY_LABELS=str(COMB_INPUTS["labels"]),
                ESPLEY_META=str(COMB_INPUTS["meta"]), ESPLEY_PROF=str(COMB_INPUTS["prof"]),
                ESPLEY_OUT=str(scratch / "gate_xtb"), R5_SCRATCH=str(scratch / "gate_r5"), PYTHONDONTWRITEBYTECODE="1")


def _gcheck(ids_csv, out_json):
    """child of g_report / td0_f (gcheck_env; GEO-11) = g0f0._g0check on $G1_ROOT: exit 3 unless the rev 4-6 modules
    picked up that root and the combined inputs; then g0f0._g0c_one (ts_fragments + ext_features.block_b6, unchanged)
    for the listed rxn_ids in a spawn pool -> <out_json>."""
    import multiprocessing as mp
    import ext_features as ef  # noqa: F401
    import fragmenter as fr
    import r6_partition as RP
    import rev5_common as R5
    import xtb_slice as xs
    want = dict(G1_ROOT=Path(os.environ["G1_ROOT"]), ESPLEY_LABELS=COMB_INPUTS["labels"],
                ESPLEY_META=COMB_INPUTS["meta"], ESPLEY_PROF=COMB_INPUTS["prof"])
    got = dict(xs_g1=(xs.GEOM_ROOT["g1"], "G1_ROOT"), R5_G1_ROOT=(R5.G1_ROOT, "G1_ROOT"),
               xs_LABELS=(xs.LABELS_PATH, "ESPLEY_LABELS"), R5_LABELS=(R5.LABELS, "ESPLEY_LABELS"),
               xs_META=(xs.META, "ESPLEY_META"), R5_META=(R5.META, "ESPLEY_META"), RP_PROF=(RP.PROF, "ESPLEY_PROF"),
               R5_PROF=(R5.PROF, "ESPLEY_PROF"))
    bad = {k: (str(p), str(want[v])) for k, (p, v) in got.items() if Path(p) != Path(want[v])}
    si = (os.environ.get("D1PAIR_SI_PATCH"), fr.COV.get("Si"), fr.Z.get("Si"), R5.BONDI.get("Si"), R5.GFN2_VALENCE.get("Si"))
    if bad or xs.PARTITION_MODE != "label" or si != ("1", 1.11, 14, 2.10, 4):
        print(f"STOP: _gcheck modules not on the geo root environment: {bad}, partition {xs.PARTITION_MODE}, Si {si}",
              file=sys.stderr)
        return 3
    with open(ids_csv, newline="") as f:
        ids = [int(r["rxn_id"]) for r in csv.DictReader(f)]
    W = max(1, G0F.cpus())
    print(f"[_gcheck] {len(ids)} folders, {W} workers, root {xs.GEOM_ROOT['g1']}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(W, initializer=G0F._g0c_init) as pool:
        rows = pool.map(G0F._g0c_one, ids, chunksize=16)
    write_json(out_json, dict(ids_csv=str(ids_csv), ids_sha256=C.sha256(ids_csv), n=len(rows),
                              env={k: os.environ.get(k) for k in want}, partition=xs.PARTITION_MODE,
                              wall_s=time.time() - t0, rows=rows))
    print(f"[_gcheck] tsf {Counter(r['tsf'] for r in rows).most_common(8)} b6 "
          f"{Counter(r['b6'] for r in rows).most_common(8)} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def g_report_state(base, fp=None):
    """GEO-11: results/G_report.json for the current base and G folders -> dict (GR_KEYS + b6_pred = the .done rxn_ids
    with ts_fragments ok whose B6 block_b6 cannot compute, gate_json = the _gcheck output). current = same base list,
    same root_fp (result.json sha256 of every base folder) and, for a complete report, gate.json unchanged; stop counts
    only for a current report."""
    s = dict(path=str(G_REPORT), present=G_REPORT.is_file(), current=False, complete=False, stop=False, stop_reasons=[],
             accepted_stops=[], counts=None, rates=None, written=None, missing=None, b6_pred=[], gate_json=None)
    if not s["present"]:
        s["missing"] = f"{G_REPORT} missing (run feat_geo.py g_report)"
        return s
    r = G0F.read_json(G_REPORT)
    fp = root_fp(GEO, base) if fp is None else fp
    gj = Path(r.get("gate_json") or "/nonexistent")
    cur = (r.get("base_sha256") == base_sha(base) and r.get("root_fp_sha256") == fp_sha(fp)
           and (not r.get("complete") or (gj.is_file() and r.get("gate_json_sha256") == C.sha256(gj))))
    s.update(current=bool(cur), complete=bool(cur and r.get("complete")), stop=bool(cur and r.get("stop")),
             stop_reasons=r.get("stop_reasons") or [], accepted_stops=r.get("accepted_stops") or [],
             counts=r.get("counts"), rates=r.get("rates"), written=r.get("written"),
             b6_pred=[int(x) for x in ((r.get("b6_not_computable") or {}).get("rxn") or [])],
             gate_json=str(gj) if r.get("gate_json") else None)
    if not cur:
        s["missing"] = f"{G_REPORT} is not for the current base / G folders (rerun feat_geo.py g_report)"
    elif not r.get("complete"):
        s["missing"] = f"{G_REPORT}: G was not complete when it was written (rerun feat_geo.py g_report)"
    return s


def g_gate_from_check(grows, gate, base_ids, accept=()):
    """the GEO-11 rates from a g_scan and a _gcheck output -> (counts, rates, tsf_bad, b6_bad, stop, accepted)."""
    done = sorted(x["rxn_id"] for x in grows if x["category"] == "ok")
    gr = {int(x["rxn_id"]): x for x in gate["rows"]}
    tsf_bad = {r: str((gr.get(r) or {}).get("tsf")) for r in done if (gr.get(r) or {}).get("tsf") != "ok"}
    b6_bad = {r: str(gr[r].get("b6")) for r in done if r not in tsf_bad and gr[r].get("b6") != "ok"}
    order_fail = [x["rxn_id"] for x in grows if x["category"] == "order"] + list(tsf_bad)
    freq_fail = [x["rxn_id"] for x in grows if x["category"] in ("freq", "error")]
    r_o, w_o = rate_gate("atom-order failures (order gate + ts_fragments)", order_fail, base_ids, ORDER_MAX)
    r_f, w_f = rate_gate("Freq / error failures", freq_fail, base_ids, FREQ_MAX)
    r_b, w_b = rate_gate("B6 not computable (ext_features.block_b6)", list(b6_bad), done, B6_MAX, unit=".done folders")
    stop, accepted = w_o + w_f, []
    (accepted if "B6" in accept else stop).extend(w_b)
    bo, bf, bb, bd, bbase = by_ds(order_fail), by_ds(freq_fail), by_ds(b6_bad), by_ds(done), by_ds(base_ids)
    counts = dict(base=len(base_ids), done=len(done), order_gate_fail=sum(1 for x in grows if x["category"] == "order"),
                  ts_fragments_fail=len(tsf_bad), order_fail=len(order_fail), freq_error_fail=len(freq_fail),
                  b6_not_computable=len(b6_bad),
                  by_dataset={d: dict(base=len(bbase[d]), done=len(bd[d]), order_fail=len(bo[d]),
                                      freq_error_fail=len(bf[d]), b6_not_computable=len(bb[d])) for d in DATASETS})
    return counts, dict(order=r_o, freq=r_f, b6=r_b), tsf_bad, b6_bad, stop, accepted


def cmd_g_report(accept=()):
    """GEO-11 (= D1LADDER g0_report): the spec 6-2 G gate per dataset over the whole base with the F code's own checks.
    STOP (exit 3): atom-order failures > 0.5 %, Freq / error > 1 %, B6 not computable > 1 % of the .done folders (only
    this one may be accepted, --accept=B6, GEO-7), stale copies; exit 2 = G incomplete."""
    P = paths(GEO)

    def body(rec):
        G0F.env_f(rec)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        gs, grows = g_scan(GEO, base)
        fp = root_fp(GEO, base)
        rec["G"] = {k: gs[k] for k in G_KEYS}
        rep = dict(spec="docs/specs/GEO2.md 6-2 G (= D1LADDER g0_report), per dataset (GEO-16)", written=now(),
                   git_head=G0F.git_head(), rule=DECISIONS["GEO-11 G report"], root=str(P.feat), n_base=len(base),
                   base_sha256=base_sha(base), root_fp_sha256=fp_sha(fp),
                   thresholds=dict(order=ORDER_MAX, freq=FREQ_MAX, b6=B6_MAX), accept=list(accept),
                   scan={k: gs[k] for k in G_KEYS}, accepted_stops=[])
        if gs["n_stale"]:
            rep.update(complete=False, stop=True, stop_reasons=[x for x in gs["stop_reasons"] if "stale" in x])
            write_json(G_REPORT, rep)
            raise Stop("stale", f"{gs['n_stale']} stale G folders {gs['stale'][:10]} (feat_geo.py g <i> <n> retry)", 3)
        if gs["n_open"]:
            rep.update(complete=False, stop=False, stop_reasons=[])
            write_json(G_REPORT, rep)
            raise Stop("incomplete", f"{gs['n_open']} base reactions without a G marker in {P.feat} (run feat_geo.py g)", 2)
        done = sorted(x["rxn_id"] for x in grows if x["category"] == "ok")
        if not done:
            rep.update(complete=False, stop=True, stop_reasons=[f"{GEO}: no G-step .done folder"])
            write_json(G_REPORT, rep)
            raise Stop("G", f"{GEO}: no G-step .done folder", 3)
        ids_csv = C.write_atomic(G_CHECK / "gate_ids.csv", G0F.ids_text(done))
        gate_json = G_CHECK / "gate.json"
        G0F.child(rec, "gcheck", [THIS, "_gcheck", ids_csv, gate_json], HERE, gcheck_env())
        gate = G0F.read_json(gate_json)
        if gate.get("ids_sha256") != C.sha256(ids_csv) or gate.get("n") != len(done):
            raise Stop("gcheck", f"{gate_json} is not from this run", 3)
        counts, rates, tsf_bad, b6_bad, stop, accepted = g_gate_from_check(grows, gate, [r for r, _ in base], accept)
        ok_rows = [x for x in gate["rows"] if x.get("b6") == "ok"]
        rep.update(complete=True, gate_json=str(gate_json), gate_json_sha256=C.sha256(gate_json), ids_csv=str(ids_csv),
                   gcheck_wall_s=gate.get("wall_s"), counts=counts, rates=rates,
                   ts_fragments_fail=dict(n=len(tsf_bad), reasons=dict(Counter(tsf_bad.values()).most_common(20)),
                                          rxn={str(r): w for r, w in sorted(tsf_bad.items())}),
                   b6_not_computable=dict(n=len(b6_bad), threshold=B6_MAX,
                                          reasons=dict(Counter(b6_bad.values()).most_common(20)), rxn=sorted(b6_bad)),
                   b6_qc_max=dict(nu_diff_cm=max([abs(float((x.get("qc") or {}).get("qc_b6_nu_diff_cm", 0.0)))
                                                  for x in ok_rows] or [0.0]),
                                  share_diff=max([abs(float((x.get("qc") or {}).get("qc_b6_mode_share_diff", 0.0)))
                                                  for x in ok_rows] or [0.0])),
                   n_imag=gs["n_imag"], stop=bool(stop), stop_reasons=stop, accepted_stops=accepted)
        write_json(G_REPORT, rep)
        rec["report"] = {k: rep[k] for k in ("counts", "rates", "stop", "stop_reasons", "accepted_stops")}
        return 3 if stop else 0
    return step("g_report", body)


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
    """F-15: {rxn_id: sha256 of $GEO2B/<m>_feat/<rxn_id>/result.json} over the base reactions (None = no result.json)."""
    P = paths(m)
    return {str(r): (C.sha256(P.feat / str(r) / "result.json") if (P.feat / str(r) / "result.json").is_file() else None)
            for r, _ in sorted(base)}


def q_rule(q, out):
    """GEO-13: the Q_report Q-a limit must be ceil(0.5 % of its n_target) (written by _qgate) -> problems."""
    n = q.get("n_target")
    want = qa_max(n) if isinstance(n, int) else None
    out["qa_rule"] = dict(n_target=n, qa_max_fail=q.get("qa_max_fail"), want=want, rate=QA_MAX_RATE,
                          rule=DECISIONS["GEO-13 QA rate"])
    return [] if q.get("qa_max_fail") == want else [f"Q_report qa_max_fail {q.get('qa_max_fail')} != ceil({QA_MAX_RATE} x "
                                                     f"{n}) = {want} (not written by _qgate; GEO-13)"]


def f_summary(m, base, gs=None, grows=None, accept=()):
    """Phase F gate of m from the files (decision F-4 = spec 6-2 F per dataset, GEO-16; rates over the G .done
    reactions of each dataset as D1LADDER F0-3): F-1 (no finite 182-feature row, B6-only rows included), B6, Q-a, T-a
    each > 1 % = STOP, plus the FU2 verdicts (Q_report with the GEO-13 limit, T_report), mergeroot stop, inconsistent
    counts, the token layout and the G B6 prediction = F status_B6 (GEO-11). accept (GEO-7): 'F-1', 'B6'."""
    import pandas as pd
    P = paths(m)
    if grows is None:
        gs, grows = g_scan(m, base)
    ids = sorted(r for r, _ in base)
    g_done = {x["rxn_id"] for x in grows if x["category"] == "ok"}
    n_done = len(g_done)
    stop, missing, accepted = [], [], []
    fpf = P.F / "root_fp.json"                              # F-15
    fp_now = root_fp(m, base)
    if not fpf.is_file() or G0F.read_json(fpf) != fp_now:
        missing.append(f"{fpf}: absent or the feature root changed since the F step (rerun feat_geo.py f_slice)")
    grs = g_report_state(base, fp_now)                     # GEO-11
    if not grs["complete"]:
        missing.append(grs["missing"])
    out = dict(n_base=len(ids), n_g_done=n_done, n_g_done_by_dataset={d: len(v) for d, v in by_ds(g_done).items()},
               root=str(P.feat), versions=P.versions, G_report={k: grs.get(k) for k in GR_KEYS})
    feat_fail, f1_fail, b6_fail, b6_only = {}, {}, {}, set()
    x_ids, q_excl, qa_fail, t_bad, tok_ok = None, None, None, None, None
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
        E["rxn_id"] = E.rxn_id.astype(int)
        st = E.set_index("rxn_id")
        b6_fail = {int(r): str(s) for r, s in st.status_B6.items() if str(s) not in ("ok", "skipped")}
        b6_only = {r for r in b6_fail if all(str(st.at[r, f"status_B{k}"]) == "ok" for k in range(1, 6))}
        f1_fail = dict(feat_fail)                          # F-1 = g0f0 f0_report (B6-only rows included; R-1)
        out["features"] = dict(
            mergeroot_report=str(P.mrep), x182=dict(path=str(P.x182), sha256=C.sha256(P.x182), n=len(X1)),
            n_requested=mr.get("n_requested"), n_done=mr.get("n_done"), n_feat=mr.get("n_feat"),
            F1_all_finite=mr.get("F1_all_finite"), columns_equal_X182_d1=cols_equal, n_feature_fail=len(feat_fail),
            n_F1=len(f1_fail), n_B6_only=len(b6_only), n_F1_without_B6_only=len(set(f1_fail) - b6_only),
            F1_rule="every G .done reaction without a finite 182-feature row (mergeroot not-ok), B6-only rows included "
                    "(= g0f0 f0_report); n_F1_without_B6_only = information only (GEO-7)",
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
        if grs["complete"]:
            g_b6 = set(grs["b6_pred"])
            f_ran = set(int(r) for r, s in st.status_B6.items() if str(s) != "skipped" and not str(s).startswith("error:"))
            b6_agree = (g_b6 & f_ran) == (set(b6_fail) & f_ran)
            out["features"]["B6_vs_G_report"] = dict(
                g_report=str(G_REPORT), n_g_predicted=len(g_b6), n_f_ran=len(f_ran), equal=bool(b6_agree),
                only_G=sorted((g_b6 - set(b6_fail)) & f_ran)[:100], only_F=sorted((set(b6_fail) - g_b6) & f_ran)[:100])
            if not b6_agree:
                stop.append(f"B6 status differs between G ({G_REPORT.name}: {len(g_b6 & f_ran)}) and F (status_B6: "
                            f"{len(set(b6_fail) & f_ran)}) on the {len(f_ran)} rows whose B6 block ran (GEO-11)")
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
        stop += q_rule(q, out["Q"])
        if q.get("stop"):
            stop.append(f"Q gate STOP (atoms.py gate, unchanged; limit GEO-13): {q.get('stop_reason')}")
        if q.get("ext_used") is False:
            stop.append(f"FU2 Q ext_used false: ext pass rate {q.get('ext_pass_rate')} < {q.get('ext_min_pass')} over "
                        f"{q.get('n_target')} reactions; the tokens then lack mu_abs / f_plus / f_minus and differ from "
                        f"the g1 layout of the trained models")
        want = dict(FU2_X182=str(P.x182), FU2_ATOM_VERSION=P.versions["FU2_ATOM_VERSION"], G1_ROOT=str(P.feat),
                    FU2_SCR=str(P.fu2_scr), FU2_RES=str(P.fu2_res), ESPLEY_LABELS=str(COMB_INPUTS["labels"]),
                    ESPLEY_META=str(COMB_INPUTS["meta"]), ESPLEY_PROF=str(COMB_INPUTS["prof"]))
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
                        elements=t.get("elements"), elements_g1=tref.get("elements"),
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
        comp = {"F-1": set(f1_fail), "B6": set(b6_fail), "Q-a": set(qa_fail), "T-a": set(t_bad)}
        gates = {}
        for k, fails in comp.items():
            info, why = rate_gate(f"{k} failures", fails, g_done, F_RATE_MAX, unit="G .done reactions")
            gates[k] = info
            (accepted if k in accept else stop).extend(why)
        n_fail = len(ids) - len(tok_ok & set(ids))
        out["gate"] = dict(
            per_dataset=gates, threshold=F_RATE_MAX, accept=list(accept),
            F1_without_B6_only={d: len(by_ds(set(f1_fail) - b6_only)[d]) for d in DATASETS},
            combined_info=dict(n=n_fail, n_base=len(ids), rate=n_fail / len(ids) if ids else None,
                               components=dict(g_step=len(ids) - n_done, features=len(feat_fail), b6=len(b6_fail),
                                               q_excluded=len(q_excl), q_a=len(qa_fail), t_bad=len(t_bad)),
                               rule="base reactions without a token npz (D1-G3 combined rule; information only)"),
            definition=DECISIONS["F-4 gate"])
    out.update(missing=missing, complete=not missing, stop=bool(stop), stop_reasons=stop, accepted_stops=accepted)
    return out


_FP = {}


def f_prior(rec, root_current=True):
    """F steps: the F job environment (test_t.env_f), the combined inputs (GEO-12), the geo base (GEO-1), T (F-10), the
    G gate over the whole base (complete, no STOP; GEO-2), a current passing results/G_report.json (GEO-11) and, with
    root_current, the F-15 record = the current feature root. Computed once per process (f_rest runs several steps)."""
    G0F.env_f(rec)
    if not _FP:
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        gs, grows = g_scan(GEO, base)
        _FP.update(base=base, gs=gs, grows=grows, Z=rec.get("Z"), T=rec.get("T"), comb=rec.get("comb_inputs"),
                   GR=g_report_state(base))
    else:
        rec.update(Z=_FP["Z"], T=_FP["T"], comb_inputs=_FP["comb"])
    base, gs, grows, grs = _FP["base"], _FP["gs"], _FP["grows"], _FP["GR"]
    rec["G"] = {k: gs[k] for k in G_KEYS}
    rec["G_report"] = {k: grs.get(k) for k in GR_KEYS}
    if gs["stop"]:
        raise Stop("G", f"G-step STOP: {gs['stop_reasons']}", 3)
    if not gs["complete"]:
        raise Stop("G", f"{GEO}: {gs['n_open']} base reactions without a G marker in {paths(GEO).feat} (run feat_geo.py g)", 2)
    if not gs["n_done"]:
        raise Stop("G", f"{GEO}: no G-step .done folder", 3)
    if grs["stop"]:
        raise Stop("G", f"G report STOP ({G_REPORT}): {grs['stop_reasons']}", 3)
    if not grs["complete"]:
        raise Stop("G", f"{grs['missing']}", 2)
    if root_current:
        fpf = paths(GEO).F / "root_fp.json"
        if not fpf.is_file() or G0F.read_json(fpf) != root_fp(GEO, base):
            raise Stop("root", f"{fpf} absent or not the current feature root (run f_slice first; F-15)", 2)
    return base, gs, grows


def f_root(rec, base):
    """F-15 + ids_all.csv of f_slice under the fcntl lock $GEO2B/locks/F_<m>.lock (the f_slice array tasks start
    together): the first task after a feature-root change moves the F outputs aside and writes root_fp.json /
    ids_all.csv, later tasks find them current. A different ids_all.csv for an unchanged root = exit 3."""
    import fcntl
    P = paths(GEO)
    lk = C.assert_path(C.G3 / "locks" / f"F_{GEO}.lock")
    lk.parent.mkdir(parents=True, exist_ok=True)
    with open(lk, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            fp, fpf = root_fp(GEO, base), P.F / "root_fp.json"
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
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def need_merged(rec):
    mr = merged(GEO)
    if mr is None:
        raise Stop("f_merge", f"no mergeroot report for the current {paths(GEO).x182} (run f_merge)", 2)
    if mr.get("stop"):
        raise Stop("f_merge", "mergeroot report stop (columns != X182_d1 or non-finite values: F-1)", 3)
    rec["x182"] = dict(path=str(paths(GEO).x182), sha256=mr["x182"]["sha256"], n=mr["x182"]["n"])
    return mr


def atoms_counts(scr=None):
    a = Path(scr or paths(GEO).fu2_scr) / "atoms"
    return dict(npz=len(list(a.glob("*.npz"))), fail=len(list(a.glob("*.fail.json"))), err=len(list(a.glob("*.err.json"))))


def cmd_f_slice(i, n):
    P = paths(GEO)

    def body(rec):
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        base, gs, _ = f_prior(rec, root_current=False)
        f_root(rec, base)
        other = sorted({p.name.split("of")[1][:2] for p in P.slices.glob("slice*_root.json")} - {f"{n:02d}"}) \
            if P.slices.is_dir() else []
        if other:
            raise Stop("slices", f"{P.slices} has slices of n = {other}: mergeroot needs all slices of one n", 3)
        tag = f"slice{i:02d}of{n:02d}"
        have = [f for f in G0F.SLICE_FILES if (P.slices / f"{tag}_{f}").is_file()]
        rec["slice"] = dict(tag=tag, dir=str(P.slices), present=have, n_base=len(base), n_g_done=gs["n_done"])
        if len(have) == len(G0F.SLICE_FILES):
            rec["skipped"] = "slice outputs exist"
            return 0
        G0F.child(rec, "runroot", [C.P_DIR / "d1_features.py", "runroot", P.feat, P.out, P.ids, i, n]
                  + inputs_args(), C.P_DIR, feat_env())
        return 0
    return step(f"f_slice_{i:02d}of{n:02d}", body)


def cmd_f_merge():
    P = paths(GEO)

    def body(rec):
        base, gs, _ = f_prior(rec)
        mr = merged(GEO)
        if mr is None:
            G0F.child(rec, "mergeroot", [C.P_DIR / "d1_features.py", "mergeroot", P.out, P.x182, P.mrep], C.P_DIR,
                      feat_env(), ok=(0, 3))
            mr = merged(GEO)
            if mr is None:
                raise Stop("mergeroot", f"{P.mrep} does not match {P.x182} after mergeroot", 3)
        else:
            rec["skipped"] = "mergeroot report matches X182"
        rec["mergeroot"] = {k: mr.get(k) for k in ("n_slices", "n_requested", "n_done", "n_feat", "F_fail_rate", "x182",
                                                   "F1_all_finite", "stop", "reasons")}
        if mr.get("stop"):
            raise Stop("mergeroot", "mergeroot report stop (columns != X182_d1 or non-finite values: F-1)", 3)
        if mr.get("n_done") != gs["n_done"] or mr.get("n_requested") != len(base):
            raise Stop("mergeroot", f"n_done {mr.get('n_done')} / n_requested {mr.get('n_requested')} != G .done "
                                    f"{gs['n_done']} / base {len(base)}", 3)
        if not mr.get("n_feat"):
            raise Stop("mergeroot", f"{GEO}: no reaction with 182 features", 3)
        return 0
    return step("f_merge", body)


def cmd_f_atoms(i, n):
    def body(rec):
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        f_prior(rec)
        need_merged(rec)
        env = atoms_env_m(GEO)
        rec["child_env_atoms"] = {k: env[k] for k in G0F.ATOMS_ENV_KEYS}
        G0F.child(rec, "atoms_run", [C.P_DIR / "fu2" / "atoms.py", "root", "run", i, n], C.P_DIR, env)
        rec["atoms"] = atoms_counts()
        return 0
    return step(f"f_atoms_{i:02d}of{n:02d}", body)


def _qgate():
    """child of f_gate / td0_f (atoms env): fu2/atoms.py root gate with QA_MAX_FAIL = ceil(0.5 % of the reactions)
    (GEO-13). root_bind() and cmd_gate() are atoms.py's own; only the fu2_common constant is rebound in this process."""
    sys.path.insert(0, str(C.P_DIR / "fu2"))
    import atoms as A
    if Path(A.__file__).resolve() != (C.P_DIR / "fu2" / "atoms.py").resolve():
        C.stop(f"atoms imported from {A.__file__}")
    A.root_bind()
    n = len(A.rxn_ids())
    A.F.QA_MAX_FAIL = qa_max(n)
    print(f"[_qgate] {n} reactions: QA_MAX_FAIL = {A.F.QA_MAX_FAIL} (ceil({QA_MAX_RATE} x n); GEO-13)", flush=True)
    return A.cmd_gate()


def cmd_f_gate():
    P = paths(GEO)
    fpf = P.F / "root_fp.json"
    err = sorted((P.fu2_scr / "atoms").glob("*.err.json"))
    r0 = G0F.read_json(RETRY_REC)
    cur = C.sha256(fpf) if fpf.is_file() else None
    if err and (r0.get("rc") != 0 or r0.get("root_fp_sha256") != cur):   # no retry pass exited 0 for this root yet
        def retry(rec):
            f_prior(rec)
            need_merged(rec)
            rec.update(err_before=[p.name for p in err][:500], n_err_before=len(err), root_fp_sha256=cur)
            G0F.child(rec, "atoms_retry", [C.P_DIR / "fu2" / "atoms.py", "root", "run", 0, 1], C.P_DIR, atoms_env_m(GEO))
            rec["atoms"] = atoms_counts()
            return 0
        rc = step("f_atoms_retry", retry)
        if rc:
            return rc

    def body(rec):
        f_prior(rec)
        need_merged(rec)
        rec["atoms"] = atoms_counts()
        rc = G0F.child(rec, "atoms_gate", [THIS, "_qgate"], C.P_DIR, atoms_env_m(GEO), ok=(0, 2, 3))
        q = P.fu2_res / "Q_report.json"
        if rc == 2:
            rec["note"] = "atoms incomplete (atoms.py root gate exit 2): run f_atoms"
        elif q.is_file():
            r = G0F.read_json(q)
            rec["Q_report"] = {k: r.get(k) for k in G0F.Q_KEYS}
        return rc
    return step("f_gate", body)


def cmd_f_tokens():
    P = paths(GEO)

    def body(rec):
        f_prior(rec)
        need_merged(rec)
        q = P.fu2_res / "Q_report.json"
        if not q.is_file():
            raise Stop("Q", f"no {q} (run f_gate)", 2)
        r = G0F.read_json(q)
        if r.get("stop"):
            raise Stop("Q", f"Q gate STOP (atoms.py gate; GEO-13 limit): {r.get('stop_reason')}", 3)
        rc = G0F.child(rec, "tokens", [C.P_DIR / "fu2" / "atoms.py", "root", "tokens"], C.P_DIR, atoms_env_m(GEO),
                       ok=(0, 3))
        t = P.fu2_res / "T_report.json"
        if t.is_file():
            tr = G0F.read_json(t)
            rec["T_report"] = {k: tr.get(k) for k in G0F.T_KEYS}
        return rc
    return step("f_tokens", body)


def cmd_f_report():
    def body(rec):
        base, gs, grows = f_prior(rec)
        s = f_summary(GEO, base, gs, grows)
        rec["summary"] = {k: v for k, v in s.items() if k != "tokens_ok"}
        steps = {p.stem: G0F.read_json(p) for p in sorted(STEPS.glob("f_*.json")) if p.stem != "f_report"}
        rec["steps"] = {k: dict(rc=v.get("rc"), finished=v.get("finished"), slurm_job=v.get("slurm_job"),
                                slurm_array_task=v.get("slurm_array_task"), cpus=v.get("cpus"), wall_s=v.get("wall_s"),
                                core_h_alloc=v.get("core_h_alloc"), exit=v.get("exit"), fail=v.get("fail"),
                                skipped=v.get("skipped")) for k, v in steps.items()}
        if s["stop"]:
            raise Stop("gate", "; ".join(s["stop_reasons"]), 3)
        return 0 if s["complete"] else 2
    return step("f_report", body)


def cmd_f_rest():
    """g0f0 f0_rest: f_merge, f_atoms 0 1, f_gate, f_tokens (stops at the first non-zero), then always f_report."""
    seq = [("f_merge", cmd_f_merge), ("f_atoms 0 1", lambda: cmd_f_atoms(0, 1)), ("f_gate", cmd_f_gate),
           ("f_tokens", cmd_f_tokens)]
    rc = 0
    for name, fn in seq:
        r = fn()
        print(f"[f_rest] {name}: exit {r}  ({now()})", flush=True)
        if r:
            rc = r
            break
    r = cmd_f_report()
    print(f"[f_rest] f_report: exit {r}", flush=True)
    return max(rc, r)


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


def pf():
    """analysis/d1_g2/phase_fh.py (H status classes H_FIN / H_GATE / H_OPEN, BATCH_MAX), imported on first use."""
    if str(L.G2_DIR) not in sys.path:
        sys.path.insert(0, str(L.G2_DIR))
    import phase_fh as PF
    return PF


def h_bad(PF):
    """GEO-6: the 'run failure' statuses of spec 6-2 H = fail + the gate statuses."""
    return ("fail",) + tuple(PF.H_GATE)


def hf_stale(scr, jobs):
    """job_ids whose run folder inputs (runs/g2/<T>/<calc>.inp) carry other atoms / coordinates than the current
    geometry inputs (inputs/<T>/g2/<calc>.inp, written by geominputs from the geometry root): a geometry folder changed
    after the HF run (run.py skips .done reactions). Coordinate lines only (the SlowConv retry text is no difference)."""
    H = hf_common_mod()

    def coords(p):
        return [ln.strip() for ln in p.read_text().splitlines() if H.XYZ_LINE.match(ln)]
    out = []
    for T in jobs:
        for c in H.CALCS:
            a, b = Path(scr) / "runs" / "g2" / T / f"{c}.inp", Path(scr) / "inputs" / T / "g2" / f"{c}.inp"
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


def c_select():
    """{D1 rxn_id: chosen run} of the G2c selection (g2c worktree analysis/g2c/results/C_select.csv; read only)."""
    if not G2C_SELECT.is_file():
        raise Stop("g2c", f"{G2C_SELECT} missing (G2c build not finished)", 2)
    with open(G2C_SELECT, newline="") as f:
        return {int(r["rxn_id"]): str(r.get("chosen") or "") for r in csv.DictReader(f) if r.get("src") == "d1"}


def rel_gate(E, jobs):
    """GEO-5 (spec 6-1 HF gate narrowed for A-G2c): E = {job_id: (e_frag1_rel_eh, e_frag2_rel_eh)} of the geo ok
    reactions. Comparable = D1 rows whose G2c choice is 'orig' with a D1-G2 HF ok row; a row whose geo_final
    rel{k}.xyz bytes differ from d1_g2/g2_final or whose energies differ by > 1e-6 Eh is a mismatch; STOP if > 0.5 % of
    the comparable rows mismatch, or none is comparable while such D1 orig rows have geo HF values, or (review R-3) a D1
    job has no G2c choice in C_select.csv (every geo_final D1 folder is a G2c folder: the gate must not pass empty)."""
    import pandas as pd
    G = pd.read_parquet(G2_HF_LABELS)
    G = G[(G.geom == "g2") & (G.status == "ok")]
    if G.job_id.duplicated().any():
        raise Stop("rel_gate", f"{G2_HF_LABELS}: duplicate job_id rows", 3)
    ref = {str(j): (float(a), float(b)) for j, a, b in zip(G.job_id, G.e_frag1_rel_eh, G.e_frag2_rel_eh)}
    sel = c_select()
    rid_of = {str(d["job_id"]): int(r) for r, d in labels().items()}
    want = [str(j) for j in jobs]
    d1 = [j for j in want if ds_of(rid_of[j]) == "D1"]
    no_choice = sorted(j for j in d1 if sel.get(rid_of[j]) not in ("orig", "r1", "r2"))
    orig = [j for j in d1 if sel.get(rid_of[j]) == "orig"]
    have = {j: E[j] for j in orig if j in E}
    comp = sorted(set(have) & set(ref))
    final = paths(GEO).final
    d, byte_bad = {}, []
    for j in comp:
        rid = rid_of[j]
        same = all((final / str(rid) / f).is_file() and (D1_G2_FINAL / str(rid) / f).is_file()
                   and C.sha256(final / str(rid) / f) == C.sha256(D1_G2_FINAL / str(rid) / f)
                   for f in ("rel1.xyz", "rel2.xyz"))
        if not same:
            byte_bad.append(j)
        try:
            v = max(abs(float(have[j][0]) - ref[j][0]), abs(float(have[j][1]) - ref[j][1]))
        except (TypeError, ValueError):
            v = float("inf")
        d[j] = v if (math.isfinite(v) and same) else float("inf")
    mis = sorted(j for j, v in d.items() if not v <= H_REL_TOL_EH)
    rate = len(mis) / len(comp) if comp else float("nan")
    why = []
    if no_choice:
        why.append(f"rel-energy gate: {len(no_choice)} D1 geo reactions have no G2c choice in {G2C_SELECT} (another "
                   f"C_select.csv than Phase Z used?): {no_choice[:5]}")
    if have and not comp:
        why.append(f"rel-energy gate: none of the {len(have)} D1 orig-run geo ok reactions has a D1-G2 ok row")
    elif comp and rate > H_REL_MAX:
        why.append(f"rel-energy gate: {len(mis)}/{len(comp)} = {rate:.4f} D1 orig-run reactions differ from D1-G2 "
                   f"(> {H_REL_TOL_EH:g} Eh or rel bytes) (> {H_REL_MAX}; spec 6-1 / 6-2 H, GEO-5)")
    fin = [v for v in d.values() if math.isfinite(v)]
    return dict(rule=DECISIONS["GEO-5 rel gate"], reference=str(G2_HF_LABELS), reference_sha256=C.sha256(G2_HF_LABELS),
                c_select=str(G2C_SELECT), c_select_sha256=C.sha256(G2C_SELECT), tol_eh=H_REL_TOL_EH, threshold=H_REL_MAX,
                n_jobs=len(want), n_d1=len(d1), n_d1_orig=len(orig), n_d1_not_orig=len(d1) - len(orig),
                n_d1_no_g2c_choice=len(no_choice), d1_no_g2c_choice=no_choice[:50],
                n_d0_no_reference=len(want) - len(d1), n_geo=len(have), n_comparable=len(comp),
                n_not_in_reference=len(set(have) - set(ref)), not_in_reference=sorted(set(have) - set(ref))[:50],
                n_rel_bytes_differ=len(byte_bad), rel_bytes_differ=byte_bad[:50], n_mismatch=len(mis), rate=rate,
                max_abs_diff_eh=max(fin) if fin else None, mismatch={j: d[j] for j in mis[:100]}, stop=bool(why),
                stop_reasons=why)


def h_summary(m, base):
    """Phase H gate of m per dataset (decisions F-4 / GEO-5 / GEO-6 / GEO-16; status classes = D1-G2 phase_fh)."""
    import pandas as pd
    PF = pf()
    BAD = h_bad(PF)
    P = paths(m)
    jobs = sorted(T for _, T in base)
    rid_of = {T: r for r, T in base}
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
    bad = S[S.status.isin(BAD)]
    gate = S[S.status.isin(PF.H_GATE)]
    inok = {T for T in jobs if gst.get(T) == "ok"}
    open_ = S[S.status.isin(PF.H_OPEN) & S.job_id.isin(inok)]
    absent = sorted(set(jobs) - set(S.job_id))
    input_fail = {T: gst.get(T, "missing") for T in jobs if gst.get(T) != "ok"}
    run_g, why_r = rate_gate("HF run failures (GEO-6)", [rid_of[T] for T in bad.job_id],
                             [rid_of[T] for T in fin.job_id], H_RUN_MAX, unit="finished reactions")
    in_g, why_i = rate_gate("HF input check failures", [rid_of[T] for T in input_fail], list(rid_of.values()),
                            H_INPUT_MAX)
    jset = set(jobs)
    lab = [r for r in C.read_json(lj) if r.get("job_id") in jset]
    ch = coreh_jobs(P.hf_scr / "runs" / "g2", jobs)
    ok_jobs = sorted(S.job_id[S.status == "ok"])
    okset = set(ok_jobs)
    stale = hf_stale(P.hf_scr, ok_jobs)
    X = pd.read_parquet(pq)
    E = ({str(j): (a, b) for j, a, b in zip(X.job_id, X.e_frag1_rel_eh, X.e_frag2_rel_eh)}
         if len(X) and {"job_id", "e_frag1_rel_eh", "e_frag2_rel_eh"} <= set(X.columns) else {})
    rg = rel_gate(E, jobs)
    stop = []
    if stale:
        stop.append(f"{m}: {len(stale)} HF runs do not use the current $GEO2B/{m}_final geometry {stale[:10]} "
                    f"(geo_final changed after H): move $GEO2B/hf_{m}/runs/g2/<T> aside and rerun the H workers")
    stop += why_r + why_i
    if len(lab) != by.get("ok", 0):
        stop.append(f"{m}: labels_d1_hf_g2.json rows {len(lab)} != H_status ok {by.get('ok', 0)}")
    rid_x = {str(j): r for j, r in zip(X.job_id, X.rxn_id)} if len(X) and "rxn_id" in X.columns else {}
    bad_rid = [j for j, r in rid_x.items() if j in rid_of and (pd.isna(r) or int(r) != rid_of[j])]
    if bad_rid:
        stop.append(f"{m}: labels_d1_hf_g2.parquet rxn_id != the combined label rxn_id for {bad_rid[:10]} (_allgeom)")
    stop += rg["stop_reasons"]
    chd = {d: [v for T, v in ch.items() if ds_of(rid_of[T]) == d] for d in DATASETS}
    out.update(by_status=by, by_status_dataset={d: dict(Counter(s for T, s in zip(S.job_id, S.status)
                                                                if ds_of(rid_of[T]) == d)) for d in DATASETS},
               n_inputs_ok=len(inok), n_finished=len(fin), n_ok=by.get("ok", 0), ok_jobs=ok_jobs,
               run_failure=dict(per_dataset=run_g, definition=DECISIONS["GEO-6 H run failure"],
                                by_status=bad.status.value_counts().to_dict(),
                                reasons=dict(Counter(bad.reason).most_common(40)), jobs=bad.job_id.tolist()[:300]),
               gate_failure=dict(n=len(gate), by_status=gate.status.value_counts().to_dict(),
                                 note="subset of run_failure (D1-G2 gate statuses)"),
               input_fail=dict(per_dataset=in_g, jobs=dict(list(input_fail.items())[:300])),
               rel_gate=rg,
               open=dict(n=len(open_), by_status=open_.status.value_counts().to_dict(), jobs=open_.job_id.tolist()[:100]),
               absent=absent[:100], n_absent=len(absent), n_stale_runs=len(stale), stale_runs=stale[:100],
               core_h=dict(sum=float(sum(ch.values())), n_run_folders=len(ch), per_rxn=G0F.qstats(ch.values()),
                           by_dataset={d: dict(sum=float(sum(v)), per_rxn=G0F.qstats(v)) for d, v in chd.items()},
                           ok_per_rxn=G0F.qstats(v for T, v in ch.items() if T in okset), reference=D1G2_H_REF),
               missing=[], complete=not len(open_) and not absent and len(fin) > 0, stop=bool(stop), stop_reasons=stop)
    return out


def load_plan():
    return G0F.read_json(H_PLAN) if H_PLAN.is_file() else None


def plan_problems(plan, base):
    """why the H plan no longer belongs to the current inputs (geominputs table, base list, batch / pilot files)."""
    bad = []
    gi = paths(GEO).hf_res / "g2_inputs.csv"
    if not gi.is_file() or C.sha256(gi) != plan.get("g2_inputs_sha256"):
        bad.append("results/hf_geo/g2_inputs.csv differs from the plan")
    if not H_LIST.is_file() or C.sha256(H_LIST) != plan.get("h_list_sha256"):
        bad.append(f"{H_LIST} differs from the plan")
    if plan.get("base_sha256") != base_sha(base):
        bad.append("the geo base changed since the plan (Phase Z rerun?)")
    pil = plan.get("pilot") or {}
    for name, sha in list((plan.get("files") or {}).items()) + [(pil.get("file"), pil.get("sha256"))]:
        f = H_BATCHES / str(name)
        if not name or not f.is_file() or C.sha256(f) != sha:
            bad.append(f"{name} differs from the plan")
    return bad


def need_plan(rec, base):
    plan = load_plan()
    if plan is None:
        raise Stop("plan", f"no {H_PLAN} (run h_inputs)", 2)
    bad = plan_problems(plan, base)
    rec["plan"] = {k: plan.get(k) for k in ("W", "nb", "n_base", "n_ok", "sizes", "max_size", "written")}
    if bad:
        raise Stop("plan", f"{H_PLAN}: {bad} (move {H_BATCHES} away and rerun h_inputs)", 3)
    return plan


def h_checks(plan):
    """build.py check records (results/hf_geo/M_checks/H_bNN.json) of the current plan's batches; a file older than
    plan.json is from an earlier plan and is not read -> (records, missing, stale, cumulative GEO-6 rate over them)."""
    PF = pf()
    BAD = h_bad(PF)
    t_plan = H_PLAN.stat().st_mtime
    recs, missing, stale, fin, nbad = {}, [], [], 0, 0
    for name in sorted(plan["files"]):
        f = M_CHECKS / f"{Path(name).stem}.json"
        if not f.is_file():
            missing.append(name)
            continue
        if f.stat().st_mtime < t_plan:
            stale.append(name)
            continue
        c = G0F.read_json(f)
        bs = {str(s): int(v) for s, v in (c.get("by_status") or {}).items()}
        nf, nb_ = sum(bs.get(s, 0) for s in PF.H_FIN), sum(bs.get(s, 0) for s in BAD)
        recs[name] = dict(n=c.get("n"), by_status=bs, n_finished=nf, n_ok=c.get("n_ok"), n_run_failure=nb_,
                          d1hf_checkpoint_stop=c.get("stop"), sha256=C.sha256(f))
        fin, nbad = fin + nf, nbad + nb_
    rate = nbad / fin if fin else None
    cum = dict(n_checks=len(recs), n_finished=fin, n_run_failure=nbad, rate=rate, threshold=H_RUN_MAX,
               stop=bool(fin and rate > H_RUN_MAX), rule="combined over the batches (GEO-6: a combined rate > 1 % "
                                                         "implies a dataset > 1 %)")
    return recs, missing, stale, cum


def pilot_state():
    """results/H_pilot.json: present, made for the current plan.json, complete, pass / stop, the projection."""
    r = G0F.read_json(H_PILOT)
    cur = C.sha256(H_PLAN) if H_PLAN.is_file() else None
    return dict(present=bool(r), file=str(H_PILOT), current_plan=bool(r) and r.get("plan_sha256") == cur,
                complete=r.get("complete"), passed=r.get("pass"), stop=r.get("stop"), stop_reasons=r.get("stop_reasons"),
                projection_core_h=r.get("projection_core_h"))


def need_pilot(rec):
    """GEO-4: h_batch runs only after a passing pilot of the current plan (missing / incomplete = 2, STOP = 3)."""
    s = pilot_state()
    rec["pilot"] = s
    if not s["present"] or not s["current_plan"]:
        raise Stop("pilot", f"{H_PILOT} missing or made for another plan (run h_pilot)", 2)
    if s["stop"]:
        raise Stop("pilot", f"pilot STOP: {s['stop_reasons']}", 3)
    if not s["passed"]:
        raise Stop("pilot", "pilot not complete (rerun h_pilot)", 2)


def need_s1(rec):
    """GEO-17: the H steps run only after the D0 S-1 check passed (results/X_S1.json, d0d1.py s1): missing / not
    finished = exit 2, a failed reproduction = exit 3."""
    r = G0F.read_json(XD.S1_JSON)
    rec["s1"] = {k: r.get(k) for k in ("pass", "complete", "stop", "max_abs_diff_kcal", "jobs", "finished", "problems")}
    if not r:
        raise Stop("S-1", f"{XD.S1_JSON} missing (run d0d1.py s1; GEO-17)", 2)
    if r.get("stop"):
        raise Stop("S-1", f"D0 S-1 failed: {r.get('problems')} (GEO-17)", 3)
    if not r.get("pass"):
        raise Stop("S-1", f"D0 S-1 not passed / complete ({r.get('open')}; GEO-17)", 2)


def quartiles(rows):
    """rows [(id, n_atoms, charge)] -> 4 lists by rank of (n_atoms, id) (np.array_split of the sorted rows)."""
    srt = sorted(rows, key=lambda x: (int(x[1]), str(x[0])))
    idx = np.array_split(np.arange(len(srt)), 4)
    return [[srt[i] for i in part] for part in idx]


def strat_pick(rows, quota, rng, charged_one=False):
    """GEO-4 / GEO-14: per n_atoms quartile quota[q] picks (rng without replacement over the quartile rows in id
    order); with charged_one, 1 of the picks of a quartile that has charged rows is a charged row ->
    ([(id, quartile)], quartiles)."""
    qs = quartiles(rows)
    out = []
    for q, (rq, k) in enumerate(zip(qs, quota)):
        rq = sorted(rq, key=lambda x: str(x[0]))
        pick = []
        if charged_one and k:
            chg = [x for x in rq if int(x[2]) != 0]
            if chg:
                pick.append(chg[int(rng.integers(len(chg)))])
        pool = [x for x in rq if x not in pick and (not charged_one or int(x[2]) == 0)]
        need = min(k - len(pick), len(pool))
        if need > 0:
            pick += [pool[i] for i in sorted(int(v) for v in rng.choice(len(pool), size=need, replace=False))]
        out += [(str(x[0]), q) for x in pick]
    return out, qs


def pilot_select(ok_jobs, lab_by_job):
    """GEO-4: 25 D1 + 25 D0 inputs-ok base reactions, n_atoms-quartile stratified (6, 6, 6, 7), D0 with a charged row in
    each quartile that has one; rng = default_rng(C.SEED_S) -> (picks [(job_id, dataset, quartile)], strata
    {(dataset, q): [job_ids]})."""
    rng = np.random.default_rng(C.SEED_S)
    picks, strata = [], {}
    for d in DATASETS:
        rows = [(T, int(lab_by_job[T]["n_atoms"]), int(lab_by_job[T].get("charge1", 0) or 0)
                 + int(lab_by_job[T].get("charge2", 0) or 0)) for T in ok_jobs if ds_of(lab_by_job[T]["rxn_id"]) == d]
        p, qs = strat_pick(rows, PILOT_QUOTA, rng, charged_one=(d == "D0"))
        picks += [(T, d, q) for T, q in p]
        for q, rq in enumerate(qs):
            strata[(d, q)] = [str(x[0]) for x in rq]
    return picks, strata


def _geominputs(lst):
    """child of h_inputs / t3 / td0_h (D1HF_* overrides; GEO-15): prep.cmd_geominputs('g2', lst) with prep.ok_manifest
    = the combined X manifest (x_fail False) and prep.DP.D1_JOBS = $GEO2B/comb/jobs; prep.py itself is unchanged."""
    import pandas as pd
    if str(C.HF_DIR) not in sys.path:
        sys.path.insert(0, str(C.HF_DIR))
    import prep as PR
    H = PR.H
    got = dict(D1HF_SCRATCH=H.HF, D1HF_RES=H.RES, D1HF_TEMPLATES=H.TEMPLATES, D1HF_G2_ROOT=H.GEOM_ROOT["g2"])
    bad = {k: (str(v), os.environ.get(k)) for k, v in got.items()
           if not os.environ.get(k) or Path(v) != Path(os.environ[k])}
    if bad or Path(H.HF) == C.HF_SCR or Path(H.TEMPLATES) != HF_TEMPLATES:
        C.stop(f"_geominputs: hf_common is not on the D1HF_* overrides / combined templates: {bad}")
    man = pd.DataFrame(XD.read_comb_manifest())
    M = man[man.x_fail == "False"][["job_id", "rxn_id", "n_atoms", "dataset"]].copy()
    M["rxn_id"], M["n_atoms"] = M.rxn_id.astype(int), M.n_atoms.astype(int)
    PR.ok_manifest = lambda: M.copy()
    PR.DP.D1_JOBS = XD.COMB_JOBS
    print(f"[_geominputs] X-ok manifest {len(M)} rows ({dict(Counter(M.dataset))}), element-order root "
          f"{PR.DP.D1_JOBS}, templates {H.TEMPLATES}, geometry root {H.GEOM_ROOT['g2']}", flush=True)
    return PR.cmd_geominputs("g2", lst)


def _allgeom(lst):
    """child of h_build / t3 / td0_h (D1HF_* overrides; GEO-15): build.cmd_all(lst, ('g2',)) with hf_common.targets =
    the combined (job_id, rxn_id) rows, so D0 rows get their rxn_id; build.py itself is unchanged."""
    if str(C.HF_DIR) not in sys.path:
        sys.path.insert(0, str(C.HF_DIR))
    import build as HB
    H = HB.H
    got = dict(D1HF_SCRATCH=H.HF, D1HF_RES=H.RES, D1HF_TEMPLATES=H.TEMPLATES, D1HF_G2_ROOT=H.GEOM_ROOT["g2"])
    bad = {k: (str(v), os.environ.get(k)) for k, v in got.items()
           if not os.environ.get(k) or Path(v) != Path(os.environ[k])}
    if bad or Path(H.HF) == C.HF_SCR:
        C.stop(f"_allgeom: hf_common is not on the D1HF_* overrides: {bad}")
    rows = sorted((dict(job_id=r["job_id"], rxn_id=int(r["rxn_id"])) for r in XD.read_comb_manifest()),
                  key=lambda d: d["job_id"])
    H.targets = lambda: rows
    return HB.cmd_all(lst, ("g2",))


def cmd_h_inputs(W):
    P = paths(GEO)

    def body(rec):
        import pandas as pd
        env_hf(rec)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        need_s1(rec)                                       # GEO-17
        if W < 1:
            raise Stop("args", f"W = {W}?", 3)
        plan = load_plan()
        if plan is not None:
            bad = plan_problems(plan, base)
            if plan.get("W") != W or bad:
                raise Stop("plan", f"{H_PLAN} exists for W = {plan.get('W')} ({bad}); W = {W} asked: move {H_BATCHES} "
                                   f"away before planning again", 3)
            rec.update(skipped="plan exists for this W, inputs unchanged",
                       plan={k: plan.get(k) for k in ("W", "nb", "n_base", "n_ok", "sizes", "max_size")})
            return 0
        PF = pf()
        lab = labels()
        rid = {T: r for r, T in base}
        lab_by_job = {T: lab[r] for T, r in rid.items()}
        jobs = sorted(rid)
        lst = C.write_atomic(H_LIST, "job_id\n" + "".join(f"{T}\n" for T in jobs))
        env = env_h(GEO)
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        G0F.child(rec, "geominputs", [THIS, "_geominputs", lst], C.HF_DIR, env)
        gi_f = P.hf_res / "g2_inputs.csv"
        gi = pd.read_csv(gi_f, dtype=str, keep_default_na=False)
        if sorted(gi.job_id) != jobs:
            raise Stop("geominputs", f"{gi_f}: {len(gi)} rows != the {len(jobs)} base jobs", 3)
        st = dict(zip(gi.job_id, gi.status))
        ok = sorted(T for T in jobs if st[T] == "ok")
        input_fail = {T: st[T] for T in jobs if st[T] != "ok"}
        ig, why = rate_gate("HF input check failures", [rid[T] for T in input_fail], list(rid.values()), H_INPUT_MAX)
        rec["inputs"] = dict(Counter(st.values()))
        rec["input_fail"] = dict(per_dataset=ig, jobs=dict(list(input_fail.items())[:200]))
        if why:
            raise Stop("geominputs", "; ".join(why) + f" ({dict(Counter(input_fail.values()))})", 3)
        if not ok:
            raise Stop("geominputs", f"{GEO}: no reaction with HF inputs ({rec['inputs']})", 3)
        nb = W * math.ceil(len(ok) / (PF.BATCH_MAX * W))
        D = pd.DataFrame(dict(job_id=ok, n_atoms=[int(lab_by_job[T]["n_atoms"]) for T in ok]))
        D = D.sort_values(["n_atoms", "job_id"]).reset_index(drop=True)
        if H_BATCHES.is_dir() and any(H_BATCHES.glob("H_*.csv")):     # an unfinished earlier planning: kept, moved
            rec["previous_batches_moved_to"] = move_aside(H_BATCHES)
        if M_CHECKS.is_dir() and any(M_CHECKS.glob("*.json")):         # checks of an earlier plan: kept, moved
            rec["previous_checks_moved_to"] = move_aside(M_CHECKS)
        files, sizes = {}, []
        for j in range(nb):
            b = D.iloc[j::nb].sort_values("job_id")
            name = f"H_b{j:02d}.csv"
            C.write_atomic(H_BATCHES / name, lambda t, b=b: b.to_csv(t, index=False))
            files[name] = C.sha256(H_BATCHES / name)
            sizes.append(len(b))
        picks, strata = pilot_select(ok, lab_by_job)       # GEO-4: 25 D1 + 25 D0, n_atoms quartiles
        pq = {T: (d, q) for T, d, q in picks}
        pil = D[D.job_id.isin(set(pq))].copy()
        pil["dataset"] = [pq[T][0] for T in pil.job_id]
        pil["quartile"] = [pq[T][1] for T in pil.job_id]
        pil = pil.sort_values("job_id")
        C.write_atomic(H_BATCHES / PILOT_FILE, lambda t: pil.to_csv(t, index=False))
        plan = dict(W=W, nb=nb, n_base=len(jobs), n_ok=len(ok), sizes=sizes, max_size=max(sizes), files=files,
                    assignment={str(k): [f"H_b{j:02d}.csv" for j in range(nb) if j % W == k] for k in range(W)},
                    pilot=dict(file=PILOT_FILE, sha256=C.sha256(H_BATCHES / PILOT_FILE), n=int(len(pil)),
                               jobs=pil.job_id.tolist(), rule=DECISIONS["GEO-4 pilot"],
                               by_dataset=dict(Counter(pil.dataset))),
                    strata={f"{d}|{q}": v for (d, q), v in strata.items()},
                    n_ok_by_dataset={d: sum(1 for T in ok if ds_of(rid[T]) == d) for d in DATASETS},
                    ok_jobs=ok, h_list_sha256=C.sha256(lst), g2_inputs_sha256=C.sha256(gi_f),
                    g2_inputs_status=dict(Counter(gi.status).most_common()), base_sha256=base_sha(base),
                    n_input_fail=len(input_fail), input_fail=input_fail, written=now(),
                    rule=f"inputs-ok job_ids sorted by (n_atoms, job_id), dealt round robin into nb = W * ceil(n / "
                         f"({PF.BATCH_MAX} W)) batches (D1-G2 phase_fh / D1HF prep.batches rule), each file sorted by "
                         f"job_id; worker k of W owns the batches j with j % W == k (decision F-6)")
        write_json(H_PLAN, plan)                             # last: the plan exists only with all its files
        rec["plan"] = {k: plan[k] for k in ("W", "nb", "n_base", "n_ok", "sizes", "max_size", "g2_inputs_status",
                                            "n_input_fail", "n_ok_by_dataset")}
        rec["plan"]["pilot_n"] = plan["pilot"]["n"]
        return 0
    return step("h_inputs", body)


def pilot_projection(plan, per, qof):
    """GEO-4: the stratified projection -> (total, by dataset, cells). per = {job_id: core-h} of the finished pilot
    reactions, qof = {job_id: (dataset, quartile)} of the pilot list."""
    cells, proj = {}, 0.0
    for key, members in (plan.get("strata") or {}).items():
        d, q = key.split("|")
        vals = [v for T, v in per.items() if qof.get(T) == (d, int(q))]
        mean = float(np.mean(vals)) if vals else float("nan")
        cells[key] = dict(n_cell=len(members), n_pilot=len(vals), mean_core_h=mean,
                          projection=mean * len(members) if members else 0.0)
        proj += cells[key]["projection"]
    by_d = {d: float(sum(c["projection"] for k, c in cells.items() if k.startswith(d + "|"))) for d in DATASETS}
    return proj, by_d, cells


def cmd_h_pilot():
    P = paths(GEO)

    def body(rec):
        import pandas as pd
        PF = pf()
        env_hf(rec)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        need_s1(rec)                                       # GEO-17
        plan = need_plan(rec, base)
        pil = pd.read_csv(H_BATCHES / plan["pilot"]["file"], dtype={"job_id": str})
        jobs = sorted(pil.job_id)
        U = pil.sort_values(["n_atoms", "job_id"], ascending=[False, True])[["job_id", "n_atoms"]]
        run_csv = C.write_atomic(P.hf_scr / "H_pilot_run.csv", lambda t: U.to_csv(t, index=False))
        lst = C.write_atomic(P.hf_scr / "H_pilot_list.csv", "job_id\n" + "".join(f"{T}\n" for T in jobs))
        env = env_h(GEO)
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        nw = max(1, min(len(jobs), G0F.cpus()))
        t1 = time.time()
        rc_run = G0F.child(rec, "run", [C.HF_DIR / "run.py", "run", run_csv, "g2", "hf", nw], C.HF_DIR, env, ok=(0, 2))
        wall = time.time() - t1
        pj = P.hf_scr / "H_pilot_parse.json"
        G0F.child(rec, "parse", [THIS, "_hfparse", lst, pj], C.HF_DIR, env)
        g = C.read_json(pj)
        if g.get("env") != rec["child_env"]:
            raise Stop("parse", f"_hfparse ran with {g.get('env')}", 3)
        R = g["records"]
        st = {T: (R.get(T) or {}).get("status") for T in jobs}
        fin = [T for T in jobs if st[T] in PF.H_FIN]
        bad = [T for T in fin if st[T] in h_bad(PF)]
        open_ = [T for T in jobs if st[T] not in PF.H_FIN]
        ch = coreh_jobs(P.hf_scr / "runs" / "g2", jobs)
        per = {T: ch[T] for T in fin if T in ch}
        E = {T: (R[T].get("e_frag1_rel_eh"), R[T].get("e_frag2_rel_eh")) for T in jobs if st[T] == "ok"}
        rg = rel_gate(E, jobs)
        out = dict(spec="docs/specs/GEO2.md 6-1 pilot (GEO-4: 25 D1 + 25 D0, stratified)", written=now(),
                   plan_sha256=C.sha256(H_PLAN), rule=DECISIONS["GEO-4 pilot"], n_pilot=len(jobs), jobs=jobs,
                   by_status=dict(Counter(st.values())), n_finished=len(fin), n_open=len(open_), open=open_,
                   run_failure=dict(n=len(bad), jobs=bad, note="reported only at the pilot (GEO-4)"), run_exit=rc_run,
                   rel_gate=rg, core_h=dict(per_rxn=G0F.qstats(per.values()), sum=float(sum(per.values())), n=len(per),
                                            wall_s=wall, cpus=G0F.cpus(), alloc_core_h=wall * G0F.cpus() / 3600.0))
        if open_:
            out.update(complete=False, stop=False, stop_reasons=[], **{"pass": False})
            write_json(H_PILOT, out)
            raise Stop("pilot", f"{len(open_)} pilot reactions not finished ({open_[:5]}; run exit {rc_run}): rerun h_pilot", 2)
        qof = {str(T): (str(d), int(q)) for T, d, q in zip(pil.job_id, pil.dataset, pil.quartile)}
        proj, by_d, cells = pilot_projection(plan, per, qof)
        naive = float(np.mean(list(per.values()))) * int(plan["n_ok"]) if per else float("nan")
        G = pd.read_parquet(G2_HF_LABELS)
        G = G[(G.geom == "g2") & (G.status == "ok")]
        g2c = {str(j): float(c) for j, c in zip(G.job_id, G.coreh_total) if np.isfinite(float(c))} \
            if "coreh_total" in G.columns else {}
        both = [T for T in per if T in g2c]
        den = sum(g2c[T] for T in both)
        ratio = sum(per[T] for T in both) / den if both and den > 0 else float("nan")
        out["projection"] = dict(
            stratified=proj, by_dataset=by_d, cells=cells, naive_mean_x_n=naive, n_total=int(plan["n_ok"]),
            n_ok_by_dataset=plan.get("n_ok_by_dataset"), d1g2_ratio_geo_over_g2=ratio, n_pilot_with_g2=len(both),
            used=proj, limit=H_COREH_MAX, reference=D1G2_H_REF,
            rule="sum over (dataset, n_atoms quartile) of the pilot mean core-h x the inputs-ok reactions of the cell; > "
                 f"{H_COREH_MAX:.0f} core-h or not finite (a cell without a finished pilot reaction) = STOP (GEO-4)")
        out["projection_core_h"] = proj
        stop = []
        if not math.isfinite(proj) or proj > H_COREH_MAX:
            stop.append(f"H core-h projection {proj:.0f} > {H_COREH_MAX:.0f} or not finite (GEO-4: stratified pilot; "
                        f"by dataset {by_d}; naive {naive:.0f})")
        stop += rg["stop_reasons"]
        out.update(complete=True, stop=bool(stop), stop_reasons=stop, **{"pass": not stop})
        write_json(H_PILOT, out)
        rec["pilot"] = {k: out[k] for k in ("by_status", "projection_core_h", "stop", "stop_reasons")}
        rec["pilot"]["rel_gate"] = {k: rg[k] for k in ("n_comparable", "n_mismatch", "max_abs_diff_eh")}
        return 3 if stop else 0
    return step("h_pilot", body)


def cmd_h_batch(k, W):
    def body(rec):
        import pandas as pd
        env_hf(rec)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        need_s1(rec)                                       # GEO-17
        plan = need_plan(rec, base)
        if plan.get("W") != W or not 0 <= k < W:
            raise Stop("plan", f"worker {k} of {W} vs plan W = {plan.get('W')}", 3)
        need_pilot(rec)
        mine = plan["assignment"][str(k)]
        rec["batches"] = mine
        if not mine:
            rec["note"] = "no batch for this worker"
            return 0
        B = pd.concat([pd.read_csv(H_BATCHES / b, dtype={"job_id": str}) for b in mine], ignore_index=True)
        U = B.sort_values(["n_atoms", "job_id"], ascending=[False, True]).reset_index(drop=True)
        ulist = C.write_atomic(H_BATCHES / f"H_w{k}of{W}.csv", lambda t: U.to_csv(t, index=False))
        rec["n_reactions"] = len(U)
        env = env_h(GEO)
        rec["child_env"] = {k2: env[k2] for k2 in HF_KEYS}
        cum0 = h_checks(plan)[3]                           # GEO-6: no new run after the 1 % rule already failed
        rec["h_gate_before"] = cum0
        if cum0["stop"]:
            raise Stop("h_gate", f"spec 6-2 H STOP: cumulative run-failure rate {cum0['n_run_failure']}/"
                                 f"{cum0['n_finished']} > {H_RUN_MAX} over {cum0['n_checks']} batch checks: no new run", 3)
        nw = max(1, G0F.cpus())
        rc_run = G0F.child(rec, "run", [C.HF_DIR / "run.py", "run", ulist, "g2", "hf", nw], C.HF_DIR, env, ok=(0, 2, 3))
        checks = {}
        for b in mine:                                    # D1HF M checkpoint: exit 3 = its own rule, recorded only
            stem = Path(b).stem
            r = G0F.child(rec, f"check_{stem}", [C.HF_DIR / "build.py", "check", H_BATCHES / b, "g2"], C.HF_DIR, env,
                          ok=(0, 3))
            c = G0F.read_json(M_CHECKS / f"{stem}.json")
            checks[b] = dict(exit=r, n=c.get("n"), by_status=c.get("by_status"), n_finished=c.get("n_finished"),
                             n_ok=c.get("n_ok"), d1hf_checkpoint_stop=c.get("stop"))
        rec["checks"] = checks
        cum = h_checks(plan)[3]
        rec["h_gate_after"] = cum
        if cum["stop"]:
            rec["h_gate"] = (f"spec 6-2 H STOP: cumulative run-failure rate {cum['n_run_failure']}/{cum['n_finished']} "
                             f"> {H_RUN_MAX} over {cum['n_checks']} batch checks")
            return 3
        return rc_run
    return step(f"h_batch_w{k}of{W}", body)


def cmd_h_build():
    P = paths(GEO)

    def body(rec):
        env_hf(rec)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        need_plan(rec, base)
        env = env_h(GEO)
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        G0F.child(rec, "allgeom", [THIS, "_allgeom", H_LIST], C.HF_DIR, env)
        outs = (P.hf_scr / "labels_d1_hf_g2.json", P.hf_res / "labels_d1_hf_g2.parquet", P.hf_res / "H_status.csv")
        rec["outputs"] = {str(p): C.sha256(p) for p in outs}
        return 0
    return step("h_build", body)


def cmd_h_report():
    P = paths(GEO)

    def body(rec):
        env_hf(rec)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        plan = need_plan(rec, base)
        b = G0F.read_json(STEPS / "h_build.json")
        hs_f = P.hf_res / "H_status.csv"
        if b.get("rc") != 0 or not hs_f.is_file() or (b.get("outputs") or {}).get(str(hs_f)) != C.sha256(hs_f):
            raise Stop("h_build", "run h_build first (h_build not ok, or H_status.csv changed since)", 2)
        s = h_summary(GEO, base)
        checks, ck_missing, ck_stale, cum = h_checks(plan)
        workers = {}
        for k in range(int(plan["W"])):
            w = G0F.read_json(STEPS / f"h_batch_w{k}of{plan['W']}.json")
            workers[str(k)] = dict(rc=w.get("rc"), slurm_job=w.get("slurm_job"), host=w.get("host"),
                                   started=w.get("started"), finished=w.get("finished"), wall_s=w.get("wall_s"),
                                   core_h_alloc=w.get("core_h_alloc"), exit=w.get("exit"), fail=w.get("fail"),
                                   batches=w.get("batches"), h_gate=w.get("h_gate"))
        stop = list(s["stop_reasons"]) + ([f"cumulative batch-check run-failure rate {cum['n_run_failure']}/"
                                           f"{cum['n_finished']} > {H_RUN_MAX}"] if cum["stop"] else [])
        complete = bool(s["complete"] and not ck_missing and not ck_stale)
        rep = dict(spec="docs/specs/GEO2.md 6 H (STOP: 6-2 H per dataset, 6-1 rel gate / pilot; GEO-4/5/6/16)",
                   written=now(), thresholds=dict(run_failure=H_RUN_MAX, input_check=H_INPUT_MAX, rel_gate=H_REL_MAX,
                                                  rel_tol_eh=H_REL_TOL_EH, core_h_cap=H_COREH_MAX),
                   summary={k: v for k, v in s.items() if k != "ok_jobs"},
                   checkpoints=dict(n_batches=len(plan["files"]), missing=ck_missing, stale=ck_stale, cumulative=cum,
                                    batches=checks),
                   workers=workers, plan={k: plan.get(k) for k in ("W", "nb", "n_base", "n_ok", "sizes", "max_size",
                                                                   "n_ok_by_dataset")},
                   pilot=pilot_state(), child_env={k: env_h(GEO)[k] for k in HF_KEYS},
                   core_h=dict(timing=s.get("core_h"), alloc_steps=float(sum(
                       (G0F.read_json(p).get("core_h_alloc") or 0.0) for p in STEPS.glob("h_*.json"))),
                               reference=D1G2_H_REF, cap=H_COREH_MAX),
                   complete=complete, stop=bool(stop), stop_reasons=stop)
        write_json(H_REPORT, rep)
        rec["report"] = {k: rep[k] for k in ("complete", "stop", "stop_reasons")}
        rec["report"].update(n_ok=s.get("n_ok"), rel_gate={k: s["rel_gate"][k] for k in ("n_comparable", "n_mismatch")}
                             if "rel_gate" in s else None)
        return 3 if stop else (0 if complete else 2)
    return step("h_report", body)


# ================================================================ P step (GEO2ENV=reactot)
def rung_cfg(m):
    """the inputs.RUNGS entry of method m (decisions F-7, GEO-8; the 'g2' geometry slot of the D1-HF runner; targets =
    the combined labels, keys = the D1 + D0 Q-2 key tables)."""
    P = paths(m)
    return dict(geom="g2", fu2_res=P.fu2_res, tokens=P.fu2_scr / "tokens", x182=P.x182,
                h_parquet=P.hf_res / "labels_d1_hf_g2.parquet", hf_runs=P.hf_scr / "runs", hf_inputs=P.hf_scr / "inputs",
                atom_version=P.versions["FU2_ATOM_VERSION"], token_version=P.versions["FU2_TOKEN_VERSION"], refs="min",
                targets=XD.COMB_LABELS, keys=(Q_KEYS_D1, Q_KEYS_D0))


def cfg_str(cfg):
    """GEO-8: a RUNGS entry with every value as str (Path / str spellings of one path compare equal)."""
    return {str(k): (str(tuple(str(x) for x in v)) if isinstance(v, (tuple, list)) else str(v)) for k, v in cfg.items()}


def cons_own(IN, m, ov, prov):
    """P20 option (c) for the geo rung (decision F-7b): rows of constrained Q-2 keys (D1 '|fixed:', D0 alt '|cfg:') keep
    their own reference; refs_min's 'min' replacements of those rows are undone."""
    K = IN._load_keys(m)
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
               n_rows_constrained_by_dataset=dict(Counter("D0" if str(j).startswith("D0G2_") else "D1"
                                                          for (j, k) in prov if (str(j), int(k)) in cons)),
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


def q_d0_ready(rep):
    """the D0 Q keys (geo1b d0_labels.py): Q_d0_report.json all_pass true, stop false and the sha256 of
    Q_keys_d0.parquet recorded under 'outputs' (keyed by absolute path, matched by name) = the file (DECISIONS_LAB)."""
    if not Q_D0_REPORT.is_file():
        return 3, f"{Q_D0_REPORT} missing"
    r = json.loads(Q_D0_REPORT.read_text())
    outs = [v for k, v in (r.get("outputs") or {}).items() if Path(k).name == Q_KEYS_D0.name]
    sha = C.sha256(Q_KEYS_D0) if Q_KEYS_D0.is_file() else None
    rep["q_d0"] = dict(report=str(Q_D0_REPORT), report_sha256=C.sha256(Q_D0_REPORT), keys=str(Q_KEYS_D0), sha256=sha,
                       all_pass=r.get("all_pass"), stop=r.get("stop"))
    if r.get("all_pass") is not True or r.get("stop") is not False or len(outs) != 1 or outs[0] != sha:
        return 3, (f"D0 Q keys not ready: Q_d0_report all_pass {r.get('all_pass')!r}, stop {r.get('stop')!r}, "
                   f"Q_keys_d0 sha256 {sha} vs recorded {outs}")
    return 0, ""


def p_inputs(m, cfg, IN):
    files = [cfg["h_parquet"], cfg["x182"], Path(cfg["fu2_res"]) / "T_report.json", Path(cfg["fu2_res"]) / "Q_report.json",
             Path(cfg["fu2_res"]) / "T_manifest.csv", Path(cfg["fu2_res"]) / "S_splits.json", IN.FU2_S_SPLITS,
             Q_KEYS_D1, Q_KEYS_D0, Q_D0_REPORT, XD.COMB_LABELS, XD.X_REPORT,
             L.RES / "Q_report.json", THIS, HERE / "d0d1.py", C.LAD_DIR / "inputs.py", C.LAD_DIR / "refs_q.py",
             C.LAD_DIR / "refs_v.py", C.LAD_DIR / "lad_common.py", C.HF_DIR / "build.py", C.HF_DIR / "hf_common.py"]
    return {str(p): (C.sha256(p) if Path(p).is_file() else None) for p in files}


def import_inputs():
    import inputs as IN
    if Path(IN.__file__).resolve() != (C.LAD_DIR / "inputs.py").resolve():
        raise Stop("code", f"inputs imported from {IN.__file__}", 3)
    return IN


def cmd_p(accept=()):
    m = GEO
    P = paths(m)

    def body(rec):
        env_reactot(rec)
        inputs_check(rec)
        base = geo_base(rec)
        need_t(rec)
        gs, grows = g_scan(m, base)
        grs = g_report_state(base)                         # GEO-11
        fs, hs = f_summary(m, base, gs, grows, accept=accept), h_summary(m, base)
        rec["G"] = {k: gs[k] for k in G_KEYS}
        rec["G_report"] = {k: grs.get(k) for k in GR_KEYS}
        rec["F"] = {k: fs.get(k) for k in ("complete", "stop", "stop_reasons", "accepted_stops", "missing", "n_tokens_ok",
                                           "gate")}
        rec["H"] = {k: hs.get(k) for k in ("complete", "stop", "stop_reasons", "missing", "n_ok", "run_failure",
                                           "rel_gate")}
        for name, s in (("G", gs), ("G report", grs), ("F", fs), ("H", hs)):
            if s["stop"]:
                raise Stop(name, f"{m}: Phase {name} STOP: {s['stop_reasons']}", 3)
            if not s["complete"]:
                raise Stop(name, f"{m}: Phase {name} not complete ({s.get('missing') or s.get('open') or s.get('n_open')})",
                           2)
        IN = import_inputs()
        cfg = rung_cfg(m)
        if m not in IN.RUNGS or cfg_str(IN.RUNGS[m]) != cfg_str(cfg):   # GEO-8: fixed in inputs.py (spec 8-1)
            raise Stop("code", f"inputs.RUNGS[{m!r}] {cfg_str(IN.RUNGS.get(m) or {})} != rung_cfg {cfg_str(cfg)} (GEO-8)", 3)
        rep = dict(phase="P (D1LADDER Phase P rules)", spec="docs/specs/GEO2.md 6-1 P (STOP 6-2 P), D0D1 design step 8",
                   method=m, started=now(), git_head=G0F.git_head(), rung_cfg=cfg_str(cfg),
                   decisions={k: DECISIONS[k] for k in ("F-7 P tables", "F-7b constrained keys", "GEO-8 P registration",
                                                        "GEO-7 accept")},
                   inputs_decisions=IN.DECISIONS, base=dict(n=len(base), z=rec.get("Z")),
                   accept=list(accept), accepted_stops=fs.get("accepted_stops"),
                   g_report=dict(path=str(G_REPORT), sha256=C.sha256(G_REPORT), accepted_stops=grs.get("accepted_stops")),
                   stop=False, all_pass=False)
        t0 = time.time()
        try:
            for code, why in (q_ready_rel(IN, rep), q_d0_ready(rep)):     # F-13 + D0 keys
                if code:
                    raise Stop("Q", f"Phase Q not ready: {why}", code)
            tj = {str(d["job_id"]): int(r) for r, d in labels().items()}
            if [d["job_id"] for d in IN.targets_of(m)] != sorted(tj):
                raise Stop("targets", "inputs.targets_of('geo') != the combined labels (job_id order)", 3)
            Hq = IN._read_h(m)
            bj = {T for _, T in base}
            extra = sorted(set(Hq.job_id) - bj)
            if extra:
                raise Stop("H", f"{len(extra)} rows of {cfg['h_parquet']} are not base reactions: {extra[:5]}", 3)
            rep["h_source"] = dict(path=str(cfg["h_parquet"]), sha256=C.sha256(cfg["h_parquet"]), n_rows=int(len(Hq)),
                                   by_dataset=dict(Counter(ds_of(tj[j]) for j in Hq.job_id)))
            ov, prov, rrep = IN.refs_min(m, Hq, rep)
            rep["references"] = rrep
            ov, prov, rep["references_g3_cons_own"] = cons_own(IN, m, ov, prov)
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
            if len(T) != len(tj) or set(int(x) for x in T.rxn_id) != set(tj.values()):
                raise Stop("table", f"table rows {len(T)} != the {len(tj)} combined targets", 3)
            n_h_ok = len(set(Hq.job_id))
            rep["self_check"] = dict(usable_subset_of_base=True, tokens_eq_F_summary=True, n_h_rows=n_h_ok,
                                     h_rows_eq_H_status_ok=sorted(Hq.job_id) == sorted(hs["ok_jobs"]),
                                     table_rows_eq_targets=True)
            if not rep["self_check"]["h_rows_eq_H_status_ok"]:
                raise Stop("table", f"{cfg['h_parquet']} rows != H_status ok jobs", 3)
            rep["usable_by_dataset"] = {d: len(v) for d, v in by_ds(usable).items()}
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
                P_a_ok=(r.get("P-a") or {}).get("ok"), reassembly_ok=(r.get("reassembly") or {}).get("ok"),
                accepted_stops=r.get("accepted_stops"))


# ================================================================ p_report (spec 7 + D0 pairs; GEO2ENV=reactot)
def table_reason(row):
    """= inputs._reason: which of token / X182 / H a target lacks in a rung table."""
    miss = [x for x, ok in (("no_tokens", row.has_tok), ("no_x182", row.has_x182), ("no_h", row.has_h)) if not ok]
    return "+".join(miss) if miss else "ok"


def set_summary(S, m):
    """counts of one pair set (bool mask m over S_folds rows): reference / anchor / para, P_main reference pairs (the
    eval_geo2 universe), per T1 fold."""
    sub = S[m]
    para = (sub.pos == "p").to_numpy()
    main = ((sub.ref_kind == "reference") & sub.in_P_main.astype(bool)).to_numpy()

    def vc(s):
        return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}
    d = dict(n=int(m.sum()), by_ref_kind=vc(sub.ref_kind), para=int(para.sum()), para_by_ref_kind=vc(sub.ref_kind[para]),
             by_pos=vc(sub.pos), in_P_main=int(sub.in_P_main.astype(bool).sum()), P_main_reference=int(main.sum()),
             P_main_reference_para=int((main & para).sum()))
    fo = {}
    for k in sorted(set(int(x) for x in S.fold_T1)):
        fk = (sub.fold_T1.astype(int) == k).to_numpy()
        fo[str(k)] = dict(n=int(fk.sum()), para=int((fk & para).sum()), by_ref_kind=vc(sub.ref_kind[fk]),
                          P_main_reference=int((fk & main).sum()), P_main_reference_para=int((fk & main & para).sum()))
    d["folds_T1"] = fo
    return d


def d1_fold_map():
    """{D1 rxn_id: T1 fold} = geo1b geo_common.d1_core_table rule: the S_folds.csv fold_T1 of the reaction's core
    (C_compare dipole/skeleton); -9 for a core that has no fold."""
    import pandas as pd
    S = pd.read_csv(S_FOLDS, usecols=["core", "fold_T1"]).drop_duplicates()
    if S.groupby("core").fold_T1.nunique().max() != 1:
        raise Stop("folds", "a core has two T1 folds in S_folds.csv", 3)
    fm = dict(zip(S.core, S.fold_T1.astype(int)))
    Cc = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton"])
    return {int(r.rxn_id): int(fm.get(f"{r.dipole}/{r.skeleton}", -9)) for r in Cc.itertuples()}


def cmd_p_report():
    def body(rec):
        import pandas as pd
        env_reactot(rec)
        inputs_check(rec)
        IN = import_inputs()
        base, zinfo = geo_state()
        rec["Z"] = {k: v for k, v in zinfo.items() if k != "problems"}
        if zinfo["problems"]:
            raise Stop("Z", "; ".join(w for _, w in zinfo["problems"]), max(c for c, _ in zinfo["problems"]))
        ps = p_summary(GEO)
        rec["P_geo_build"] = {k: ps.get(k) for k in ("present", "all_pass", "stop", "table_ok", "stale", "n_usable")}
        if ps["stop"]:
            raise Stop("P", f"P_build_{GEO}.json STOP: {ps.get('stop_reason')}", 3)
        if not ps["table_ok"] or ps["stale"]:
            raise Stop("P", f"no passing, current P_build_{GEO}.json / {ps.get('table')} (run feat_geo.py p)", 2)
        d0rep, d0pr = XD.d0_pairs_state()
        if d0pr:
            raise Stop("d0_pairs", "; ".join(w for _, w in d0pr), max(c for c, _ in d0pr))
        lp = G0F.read_json(IN.P_REPORT)
        if not lp.get("all_pass") or lp.get("running") or lp.get("stop"):
            raise Stop("ladder", f"{IN.P_REPORT}: all_pass {lp.get('all_pass')}, stop {lp.get('stop')}", 3)
        S = pd.read_csv(IN.S_FOLDS)
        if S.pair_id.duplicated().any():
            raise Stop("universe", f"{IN.S_FOLDS}: pair_id not unique", 3)
        S["pair_id"] = S.pair_id.astype(str)
        inputs = {str(IN.S_FOLDS): C.sha256(IN.S_FOLDS), str(IN.P_REPORT): C.sha256(IN.P_REPORT),
                  str(XD.D0_PAIRS): C.sha256(XD.D0_PAIRS), str(XD.D0_FOLDS): C.sha256(XD.D0_FOLDS),
                  str(XD.D0_PAIRS_REPORT): C.sha256(XD.D0_PAIRS_REPORT)}
        U, why, tabs = {}, {}, {}
        for r in ("dft", "g2"):                            # GEO-9: LADDER tables, verified against their build / P_report
            tp, bp = IN.table_path(r), IN.build_report_path(r)
            br = G0F.read_json(bp)
            sha = C.sha256(tp) if tp.is_file() else None
            want = ((lp.get("rungs") or {}).get(r) or {}).get("table_sha256")
            if not br.get("all_pass") or (br.get("outputs") or {}).get(str(tp)) != sha or want != sha:
                raise Stop("ladder", f"{tp} sha256 {sha} != P_build_{r}.json outputs / P_report.json rungs (LADDER inputs "
                                     f"changed)", 3)
            tabs[r] = (tp, sha)
        tabs["geo"] = (paths(GEO).table, C.sha256(paths(GEO).table))
        for r, (tp, sha) in tabs.items():
            T = pd.read_parquet(tp, columns=["rxn_id", "has_tok", "has_x182", "has_h", "usable"])
            U[r] = set(int(x) for x in T.rxn_id[T.usable])
            why[r] = {int(x.rxn_id): table_reason(x) for x in T.itertuples(index=False)}
            inputs[str(tp)] = sha
        inset = {r: (S.a_rxn.astype(int).isin(U[r]) & S.b_rxn.astype(int).isin(U[r])).to_numpy() for r in U}
        sets = {"P_geo": inset["geo"], "P_23": inset["g2"] & inset["geo"], "P_023": inset["dft"] & inset["g2"] & inset["geo"],
                "P_g2": inset["g2"], "P_dft": inset["dft"]}
        lad_check = {}
        for name in ("P_dft", "P_g2"):                     # recomputed = LADDER's own set files
            f = IN.set_path(name)
            want = ((lp.get("sets") or {}).get(name) or {}).get("sha256")
            sha = C.sha256(f) if f.is_file() else None
            got = set(pd.read_csv(f, usecols=["pair_id"]).pair_id.astype(str)) if f.is_file() else None
            mine = set(S.pair_id[sets[name]])
            lad_check[name] = dict(file=str(f), sha256=sha, P_report_sha256=want, equal_ids=got == mine,
                                   n_file=len(got) if got is not None else None, n_recomputed=len(mine))
            inputs[str(f)] = sha
            if sha != want or got != mine:
                raise Stop("ladder", f"{name}: recomputed set ({len(mine)}) != LADDER {f} ({lad_check[name]})", 3)
        lists = {n: sorted(S.pair_id[sets[n]]) for n in SET_NAMES}
        zst = {}
        if Z_STATUS.is_file():
            with open(Z_STATUS, newline="") as fh:
                zst = {int(r["rxn_id"]): str(r.get("status")) for r in csv.DictReader(fh)}
        gs, grows = g_scan(GEO, base)
        gcat = {int(x["rxn_id"]): (x["category"], x["reason"]) for x in grows}
        base_ids = {r for r, _ in base}

        def geo_reason(rid):
            rid = int(rid)
            if rid not in why["geo"]:
                return "not_target"
            if rid not in base_ids:
                return f"no_geo_folder:{zst.get(rid, 'no_Z_status_row')}"
            c, w = gcat.get(rid, ("open", "no_marker"))
            if c != "ok":
                return f"G:{w or c}"
            return why["geo"][rid]
        miss_m = sets["P_g2"] & ~sets["P_geo"]
        sub = S[miss_m]
        rows = []
        for p, a, b, rk, po, mn, fo in zip(sub.pair_id, sub.a_rxn, sub.b_rxn, sub.ref_kind, sub.pos,
                                           sub.in_P_main.astype(bool), sub.fold_T1):
            ra, rb = geo_reason(a), geo_reason(b)
            rows.append(dict(pair_id=p, ref_kind=rk, pos=po, in_P_main=bool(mn), fold_T1=int(fo),
                             reason=f"a:{ra};b:{rb}", a_reason=ra, b_reason=rb))
        cls = Counter(x for r in rows for x in {r["a_reason"], r["b_reason"]} if x != "ok")
        n = {k: int(v.sum()) for k, v in sets.items()}
        main = ((S.ref_kind == "reference") & S.in_P_main.astype(bool)).to_numpy()
        nm = {k: int((v & main).sum()) for k, v in sets.items()}
        ratio = n["P_23"] / n["P_g2"] if n["P_g2"] else float("nan")
        ratio_m = nm["P_23"] / nm["P_g2"] if nm["P_g2"] else float("nan")
        stop = []
        if not ratio >= P_MIN_RATIO:
            stop.append(f"D1 |P_23| {n['P_23']} < {P_MIN_RATIO} x |P_g2| {n['P_g2']} (ratio {ratio:.4f}; spec 7)")
        if not ratio_m >= P_MIN_RATIO:
            stop.append(f"D1 P_main reference pairs: |P_23| {nm['P_23']} < {P_MIN_RATIO} x |P_g2| {nm['P_g2']} (ratio "
                        f"{ratio_m:.4f}; spec 7, eval universe)")
        f1, f0 = d1_fold_map(), XD.d0_fold_map()
        fold_of = {**f1, **f0}
        usable_tab = {}
        for d in DATASETS:
            ids_d = [r for r in labels() if ds_of(r) == d]
            cnt = Counter(fold_of.get(r, -9) for r in ids_d)
            ucnt = Counter(fold_of.get(r, -9) for r in ids_d if r in U["geo"])
            bcnt = Counter(fold_of.get(r, -9) for r in ids_d if r in base_ids)
            usable_tab[d] = dict(n_targets=len(ids_d), n_base=sum(bcnt.values()), n_usable=sum(ucnt.values()),
                                 by_fold={str(k): dict(targets=cnt[k], base=bcnt.get(k, 0), usable=ucnt.get(k, 0))
                                          for k in sorted(cnt)},
                                 unusable_reasons=dict(Counter(geo_reason(r) for r in ids_d if r not in U["geo"])
                                                       .most_common(40)))
        P0 = XD.read_d0_pairs()
        d0rows, lost = [], []
        for p in P0:
            a, b = int(p["a_rxn"]), int(p["b_rxn"])
            fa, fb = f0.get(a, -9), f0.get(b, -9)
            okl = str(p.get("ok")) == "True"
            ua, ub = a in U["geo"], b in U["geo"]
            d0rows.append(dict(pair_id=p["pair_id"], fold=fa, fold_b=fb, edge_kind=p.get("edge_kind"), ok=okl,
                               usable=bool(ua and ub), use=bool(ua and ub and okl)))
            if okl and not (ua and ub):
                lost.append(dict(pair_id=p["pair_id"], fold=fa, edge_kind=p.get("edge_kind"),
                                 a_reason=geo_reason(a) if not ua else "ok", b_reason=geo_reason(b) if not ub else "ok"))
        D0 = pd.DataFrame(d0rows, columns=["pair_id", "fold", "fold_b", "edge_kind", "ok", "usable", "use"])
        if len(D0) and (D0.fold != D0.fold_b).any():
            stop.append(f"D0_pairs.csv: {int((D0.fold != D0.fold_b).sum())} pairs whose ends sit in different folds")

        def d0c(mask):
            sub0 = D0[mask]
            return dict(n=int(len(sub0)), by_fold={str(k): int(v) for k, v in sub0.fold.value_counts().sort_index().items()},
                        by_edge_kind={str(k): int(v) for k, v in sub0.edge_kind.value_counts().sort_index().items()})
        d0sum = dict(source=str(XD.D0_PAIRS), report=str(XD.D0_PAIRS_REPORT), g2_source=d0rep.get("g2_source"),
                     all=d0c(np.ones(len(D0), bool)), ok_labels=d0c(D0.ok.to_numpy(bool)),
                     usable_geo=d0c(D0.use.to_numpy(bool)),
                     lost_because_end_unusable=dict(n=len(lost), by_reason=dict(Counter(
                         f"a:{x['a_reason']};b:{x['b_reason']}" for x in lost).most_common(40)),
                         by_fold=dict(Counter(str(x["fold"]) for x in lost)), pairs=lost[:2000]),
                     sha256_usable=list_sha(D0.pair_id[D0.use.to_numpy(bool)]))
        files = {}
        sets_dir = paths(GEO).table.parent / "sets"
        for name in ("P_geo", "P_23", "P_023"):            # LADDER set-file format (inputs.SET_COLS) for the loaders
            df = S.loc[sets[name], IN.SET_COLS]
            f = C.write_atomic(sets_dir / f"{name}.csv", lambda t, df=df: df.to_csv(t, index=False))
            files[str(f)] = C.sha256(f)
        d0u = D0[D0.use.to_numpy(bool)][["pair_id", "fold", "edge_kind"]]
        f = C.write_atomic(sets_dir / "P_geo_d0.csv", lambda t: d0u.to_csv(t, index=False))
        files[str(f)] = C.sha256(f)
        sj = C.write_json(P_SETS, dict({k: lists[k] for k in SET_NAMES}, P_geo_d0=sorted(d0u.pair_id.astype(str))))
        rep = dict(spec="docs/specs/GEO2.md 7 (D1) + D0D1 design step 8 (D0)", written=now(), git_head=G0F.git_head(),
                   rule=DECISIONS["GEO-9 pair sets"], universe=dict(path=str(IN.S_FOLDS), n=int(len(S))),
                   inputs=inputs, ladder_check=lad_check, n=n, n_P_main_reference=nm,
                   sha256={k: list_sha(lists[k]) for k in SET_NAMES},
                   sha256_rule="sha256 of '\\n'.join(sorted pair ids), no trailing newline",
                   sets={k: set_summary(S, sets[k]) for k in SET_NAMES},
                   check=dict(rule=f"D1 |P_23| >= {P_MIN_RATIO} |P_g2| (all pairs and P_main reference pairs)",
                              ratio=ratio, ratio_P_main_reference=ratio_m),
                   missing_from_geo=dict(n=len(rows), n_P_main_reference=sum(1 for r in rows
                                                                            if r["in_P_main"] and r["ref_kind"] == "reference"),
                                         by_reason=dict(Counter(r["reason"] for r in rows).most_common(60)),
                                         end_reason_classes=dict(cls.most_common(60)), pairs=rows[:3000]),
                   usable=dict(geo=len(U["geo"]), g2=len(U["g2"]), dft=len(U["dft"])),
                   reactions_by_dataset=usable_tab, d0_pairs=d0sum,
                   outputs=dict({str(sj): C.sha256(sj)}, **files), stop=bool(stop), stop_reasons=stop)
        write_json(P_SETS_REPORT, rep)
        rec["report"] = dict(n=n, n_P_main_reference=nm, ratio=ratio, ratio_P_main_reference=ratio_m,
                             d0_pairs_usable=d0sum["usable_geo"].get("n"), stop=bool(stop), stop_reasons=stop)
        return 3 if stop else 0
    return step("p_report", body)


# ================================================================ T checks
def t_check(k, out, body):
    """T-<k>: results/T_<k>.json marked 'running', body(rec), then the verdict (pass / not_ready / fail)."""
    record_all()
    rec = dict(test=f"T-{k}", spec="docs/specs/GEO2.md 5 (+ GEO-14 T-D0)", started=now(), host=socket.gethostname(),
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
        inputs_check(rec)
        lab = labels()
        meta = read_meta()
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
    inputs_check(rec)
    root = T2_DIR / "root"
    mk = {r: G0F.marker(root / str(r)) for r in T2_RXN}
    rec["folders"] = {str(r): v for r, v in mk.items()}
    if any(v is None for v in mk.values()):
        raise Stop("t2_g", f"{root}: folders without marker {mk} (run t2_g first)", 2)
    if any(v != ".done" for v in mk.values()):
        raise Stop("t2_g", f"{root}: {mk}", 3)
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
    ids = C.write_atomic(T2_DIR / "T2_ids.csv", "rxn_id\n" + "".join(f"{r}\n" for r in T2_RXN))
    out, x182, mrep = T2_DIR / "F" / "out", T2_DIR / "X182_T2.parquet", T2_DIR / "F" / "mergeroot_report.json"
    G0F.child(rec, "runroot", [C.P_DIR / "d1_features.py", "runroot", root, out, ids, 0, 1] + inputs_args(),
              C.P_DIR, feat_env())
    G0F.child(rec, "mergeroot", [C.P_DIR / "d1_features.py", "mergeroot", out, x182, mrep], C.P_DIR, feat_env(),
              ok=(0, 3))
    m = C.read_json(mrep)
    rec["mergeroot"] = {k: m.get(k) for k in ("n_requested", "n_done", "n_feat", "F_fail_rate", "reasons", "not_ok",
                                              "x182", "F1_all_finite", "versions", "stop", "labels_sha256",
                                              "meta_sha256")}
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
    if v182 == "pass_f0_precedent":                        # recorded when it happens (spec 5 T-2 says 1e-8)
        record_decision("F-8 T-2 deviation", f"T-2 182열 최대 |차| {mx:.3g} > 1e-8(명령서 5절 T-2 기준)이고 넘은 열이 모두 "
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
                          deviation=rec["deviation"], inputs="combined (GEO-12)")


def cmd_t2_f():
    return t_check("2", T2_JSON, t2_body)


def inputs_vs_templates(root, jobs):
    """the g2-slot inputs vs the combined HF templates: bytes, elements, header, coordinate max |diff| (report)."""
    H = hf_common_mod()
    out = {}
    for T in jobs:
        res = {}
        for c in H.CALCS:
            fa, fb = Path(root) / T / "g2" / f"{c}.inp", HF_TEMPLATES / T / "hf" / f"{c}.inp"
            ta, tb = fa.read_bytes(), fb.read_bytes()
            r = dict(identical=ta == tb)
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
    inputs_check(rec)
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
    env = env_h_roots(base, base / "results", G0F.DFT_FINAL)
    rec["child_env"] = {k: env[k] for k in HF_KEYS}
    G0F.child(rec, "geominputs", [THIS, "_geominputs", lst], C.HF_DIR, env)
    st = pd.read_csv(base / "results" / "g2_inputs.csv", dtype=str, keep_default_na=False)
    rec["g2_inputs"] = dict(zip(st.job_id, st.status))
    if sorted(st.job_id) != T3_JOBS or (st.status != "ok").any():
        raise Stop("geominputs", f"results/g2_inputs.csv statuses {rec['g2_inputs']}", 3)
    rec["inputs_vs_hf_templates"] = inputs_vs_templates(base / "inputs", T3_JOBS)
    nw = max(1, min(len(T3_JOBS), G0F.cpus()))
    G0F.child(rec, "run", [C.HF_DIR / "run.py", "run", lst, "g2", "hf", nw], C.HF_DIR, env)
    G0F.child(rec, "allgeom", [THIS, "_allgeom", lst], C.HF_DIR, env)
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
        lab_eq = bool(T in labj and all(labj[T].get(k) == a.get(k) for k in TT.T3_KEYS)
                      and labj[T].get("rxn_id") == rid_of[T])
        rows[T] = dict(ok=bool(ok), max_abs_diff_kcal=mx, diff=d, status_g2=a.get("status"), reason_g2=a.get("reason"),
                       status_dft=b.get("status"), g2={k: a.get(k) for k in TT.T3_KEYS},
                       dft={k: b.get(k) for k in TT.T3_KEYS}, gates_g2=a.get("gates"), labels_json_row_equal=lab_eq)
    mx = float(np.max(np.array([r["max_abs_diff_kcal"] for r in rows.values()], dtype=float)))
    good = all(r["ok"] and r["labels_json_row_equal"] for r in rows.values())
    rec.update(compare=rows, keys=TT.T3_KEYS, tol_kcal=TT.TOL_T3, max_abs_diff_kcal=mx,
               verdict="pass" if good else "fail", **{"pass": bool(good)})
    rec["summary"] = dict(verdict=rec["verdict"], max_abs_diff_kcal=mx, path="_geominputs / run.py / _allgeom (GEO-15)",
                          inputs_identical={T: all(v["identical"] for v in r.values())
                                            for T, r in rec["inputs_vs_hf_templates"].items()},
                          status={T: (r["status_g2"], r["status_dft"]) for T, r in rows.items()})


def cmd_t3(rerun=False):
    return t_check("3", T3_JSON, lambda rec: t3_body(rec, rerun))


def _hfparse(lst, out):
    """child of t3 / h_pilot / td0_h (D1HF_* overrides in its environment): parse_rxn(T, 'g2') -> <out>."""
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


# ---------------------------------------------------------------- T-D0 (GEO-14)
TD0_SRC, TD0_ROOT = TD0_DIR / "src", TD0_DIR / "root"


def td0_select():
    """GEO-14: the 20 D0 smoke reactions (X-ok D0 rows of the combined manifest, n_atoms quartiles x 5, one charged row
    in each quartile that has one; default_rng(C.SEED_S)) -> sorted rxn_ids."""
    man = [r for r in XD.read_comb_manifest() if r["dataset"] == "D0" and r["x_fail"] == "False"]
    rows = [(r["rxn_id"], int(r["n_atoms"]), int(r["charge"])) for r in man]
    p, _ = strat_pick(rows, (TD0_N // 4,) * 4, np.random.default_rng(C.SEED_S), charged_one=True)
    return sorted(int(x) for x, _ in p)


def td0_src(ids, rec, src_root=None):
    """<src_root>/<rid>/{ts, rel1, rel2}.xyz (default $GEO2B/td0/src) = byte copies of the d0_g2 labjobs files (Coley TS /
    references) + .done; an existing folder with other bytes = STOP."""
    lab = labels()
    src_root = Path(src_root or TD0_SRC)
    for r in ids:
        T = lab[r]["job_id"]
        d = src_root / str(r)
        srcs = {"ts.xyz": XD.COMB_JOBS / T / "ts.xyz", "rel1.xyz": XD.COMB_JOBS / T / "ref_dipole.xyz",
                "rel2.xyz": XD.COMB_JOBS / T / "ref_dipolarophile.xyz"}
        if (d / ".done").is_file():
            if any(C.sha256(d / f) != C.sha256(s) for f, s in srcs.items()):
                raise Stop("td0", f"{d} differs from its labjobs sources", 3)
            continue
        for f, s in srcs.items():
            C.write_atomic(d / f, s.read_bytes())
        C.write_atomic(d / ".done", now() + "\n")
    rec["td0_src"] = dict(root=str(src_root), n=len(ids))


def cmd_td0_g(rerun=False):
    def body(rec):
        env_d1(rec)
        write_json(TD0F_JSON, dict(test="T-D0F", verdict="running", started=now(), note="td0_g started; td0_f writes "
                                   "the verdict", **{"pass": False}))
        if rerun:
            rec["previous_moved_to"] = move_aside(TD0_DIR)
        rec["code_sha256"] = code_sha()
        fresh_tdir(TD0_DIR, rec["code_sha256"], rec)
        inputs_check(rec)
        lab = labels()
        meta = read_meta()
        mods = G0F.g0_modules(rec)
        ids = td0_select()
        rec["td0_ids"] = ids
        td0_src(ids, rec)
        jobs = [dict(rid=r, T=lab[r]["job_id"], lab=lab[r], src=TD0_SRC / str(r), out=TD0_ROOT / str(r), tag="td0")
                for r in ids]
        C.assert_path(TD0_ROOT).mkdir(parents=True, exist_ok=True)
        run_pool(jobs, meta, mods, rec, "td0")
        rec["folders"] = {str(r): G0F.marker(TD0_ROOT / str(r)) for r in ids}
        return 0
    rc = step("td0_g", body)
    if rc == 3:
        r = G0F.read_json(STEPS / "td0_g.json")
        write_json(TD0F_JSON, dict(test="T-D0F", verdict="fail", fail_stage=f"td0_g:{r.get('fail_stage')}",
                                   fail=r.get("fail"), finished=now(), stop=True, **{"pass": False}))
    return rc


def td0f_body(rec):
    """T-D0F (GEO-14): G gate, F features, Q, tokens on the 20 D0 smoke folders."""
    import pandas as pd
    G0F.env_f(rec)
    if not tdir_current(TD0_DIR, rec["code_sha256"]):
        raise Stop("td0_g", f"{TD0_DIR} was not made by td0_g with the current code: run td0_g first", 2)
    inputs_check(rec)
    ids = td0_select()
    lab = labels()
    base = [(r, lab[r]["job_id"]) for r in ids]
    gs, grows = g_scan(GEO, base, feat=TD0_ROOT, final=TD0_SRC)
    rec["G"] = {k: gs[k] for k in G_KEYS}
    if gs["n_open"]:
        raise Stop("td0_g", f"{gs['n_open']} smoke folders without marker (run td0_g)", 2)
    done = sorted(x["rxn_id"] for x in grows if x["category"] == "ok")
    path_err, chem = [], []
    path_err += [f"G:{x['rxn_id']}:{x['reason']}" for x in grows if x["category"] in ("order", "error", "stale")]
    chem += [f"G:{x['rxn_id']}:{x['reason']}" for x in grows if x["category"] == "freq"]
    if done:
        ids_csv = C.write_atomic(TD0_DIR / "gate_ids.csv", G0F.ids_text(done))
        gj = TD0_DIR / "gate.json"
        G0F.child(rec, "gcheck", [THIS, "_gcheck", ids_csv, gj], HERE, gcheck_env(TD0_ROOT, TD0_DIR / "G"))
        gate = C.read_json(gj)
        for x in gate["rows"]:
            if x.get("tsf") != "ok":
                path_err.append(f"ts_fragments:{x['rxn_id']}:{x.get('tsf')}")
            elif x.get("b6") != "ok":
                chem.append(f"B6:{x['rxn_id']}:{x.get('b6')}")
    fids = C.write_atomic(TD0_DIR / "F_ids.csv", G0F.ids_text(ids))
    out, x182, mrep = TD0_DIR / "F" / "out", TD0_DIR / "X182_td0.parquet", TD0_DIR / "F" / "mergeroot_report.json"
    G0F.child(rec, "runroot", [C.P_DIR / "d1_features.py", "runroot", TD0_ROOT, out, fids, 0, 1] + inputs_args(),
              C.P_DIR, feat_env())
    G0F.child(rec, "mergeroot", [C.P_DIR / "d1_features.py", "mergeroot", out, x182, mrep], C.P_DIR, feat_env(),
              ok=(0, 3))
    m = C.read_json(mrep)
    rec["mergeroot"] = {k: m.get(k) for k in ("n_requested", "n_done", "n_feat", "reasons", "not_ok", "F1_all_finite",
                                              "stop", "x182")}
    if m.get("stop"):
        path_err.append(f"mergeroot stop (columns / non-finite): {m.get('reasons')}")
    for r, w in (m.get("not_ok") or {}).items():
        if int(r) in set(done) and not str(w).startswith("root:"):
            (chem if "B6" in str(w) or str(w).startswith("ext:") else path_err).append(f"F:{r}:{w}")
    X1 = pd.read_parquet(x182)
    if not len(X1):
        raise Stop("td0_f", "no smoke reaction with 182 features", 3)
    env = atoms_env(TD0_ROOT, TD0_DIR / "fu2", TD0_DIR / "fu2_res", x182, TD0_DIR / "fu2_xtb", TD0_DIR / "fu2_r5",
                    TD0_VERSIONS)
    rec["child_env_atoms"] = {k: env[k] for k in G0F.ATOMS_ENV_KEYS}
    G0F.child(rec, "atoms_run", [C.P_DIR / "fu2" / "atoms.py", "root", "run", 0, 1], C.P_DIR, env)
    rc_g = G0F.child(rec, "atoms_gate", [THIS, "_qgate"], C.P_DIR, env, ok=(0, 3))
    q = G0F.read_json(TD0_DIR / "fu2_res" / "Q_report.json")
    rec["Q_report"] = {k: q.get(k) for k in G0F.Q_KEYS}
    Q = pd.read_csv(TD0_DIR / "fu2_res" / "Q_manifest.csv")
    for r, st, qa, qb, ext in zip(Q.rxn_id, Q.status.astype(str), Q.qa_ok.astype(str), Q.qb_ok.astype(str),
                                  Q.ext_ok.astype(str)):
        if st != "ok":
            (chem if "h_map" in st else path_err).append(f"Q:{r}:{st}")     # h_map = owned-H count (structure)
        elif qa != "True" or qb != "True":
            path_err.append(f"Q:{r}:qa {qa} qb {qb}")
        elif ext != "True":
            chem.append(f"Q:{r}:ext")
    if rc_g == 0:
        G0F.child(rec, "tokens", [C.P_DIR / "fu2" / "atoms.py", "root", "tokens"], C.P_DIR, env, ok=(0, 3))
        t = G0F.read_json(TD0_DIR / "fu2_res" / "T_report.json")
        rec["T_report"] = {k: t.get(k) for k in G0F.T_KEYS}
        tref = G0F.read_json(DP.RES / "fu2" / "T_report.json")
        lay = [k for k in G0F.T_LAYOUT_KEYS if t.get(k) != tref.get(k)]
        if lay:
            path_err.append(f"token layout differs from the g1 rung in {lay}")
        if t.get("stop"):
            path_err.append(f"T_report stop: {t.get('gate_Ta')}")
    else:
        path_err.append(f"Q gate exit {rc_g}: {q.get('stop_reason')}")
    good = not path_err and len(chem) <= 2
    rec.update(ids=ids, n=len(ids), n_charged=sum(1 for r in ids if int(lab[r]["charge1"]) + int(lab[r]["charge2"])),
               n_atoms={str(r): int(lab[r]["n_atoms"]) for r in ids}, path_errors=path_err, chemistry_failures=chem,
               rule=DECISIONS["GEO-14 T-D0"], verdict="pass" if good else "fail", **{"pass": bool(good)})
    rec["summary"] = dict(verdict=rec["verdict"], n_path_errors=len(path_err), n_chemistry_failures=len(chem),
                          n_x182=len(X1), n_imag=gs["n_imag"])


def cmd_td0_f():
    return t_check("D0F", TD0F_JSON, td0f_body)


def td0h_body(rec, rerun):
    """T-D0H (GEO-14): the H path on the 20 D0 smoke reactions at the Coley TS / references (D1HF_G2_ROOT =
    $GEO2B/td0h/src, its own work folder with the F-16 code stamp, independent of td0_g / td0_f): inputs = the
    Phase X templates (coordinates <= 1e-6 A, same header), HF status ok."""
    env_hf(rec)
    if rerun:
        rec["previous_moved_to"] = move_aside(TD0H_DIR)
    fresh_tdir(TD0H_DIR, rec["code_sha256"], rec)                  # F-16
    hd, src = TD0H_DIR / "hf", TD0H_DIR / "src"
    inputs_check(rec)
    ids = td0_select()
    lab = labels()
    td0_src(ids, rec, src)
    jobs = sorted(lab[r]["job_id"] for r in ids)
    rid_of = {lab[r]["job_id"]: r for r in ids}
    C.assert_path(hd).mkdir(parents=True, exist_ok=True)
    lst = C.write_atomic(hd / "list.csv", "job_id\n" + "".join(f"{T}\n" for T in jobs))
    env = env_h_roots(hd, hd / "results", src)
    rec["child_env"] = {k: env[k] for k in HF_KEYS}
    G0F.child(rec, "geominputs", [THIS, "_geominputs", lst], C.HF_DIR, env)
    with open(hd / "results" / "g2_inputs.csv", newline="") as f:
        st = {r["job_id"]: r["status"] for r in csv.DictReader(f)}
    path_err, chem = [f"inputs:{T}:{s}" for T, s in st.items() if s != "ok"], []
    okj = sorted(T for T, s in st.items() if s == "ok")
    vt = inputs_vs_templates(hd / "inputs", okj)
    for T, r in vt.items():
        for c, v in r.items():
            if not (v["same_elements"] and v["same_header"] and v["max_coord_diff_A"] is not None
                    and v["max_coord_diff_A"] <= 1e-6):
                path_err.append(f"template:{T}:{c}:{v}")
    rec["inputs_vs_templates"] = vt
    U = sorted(okj, key=lambda T: (-int(lab[rid_of[T]]["n_atoms"]), T))
    run_csv = C.write_atomic(hd / "run.csv", "job_id,n_atoms\n" + "".join(f"{T},{int(lab[rid_of[T]]['n_atoms'])}\n"
                                                                          for T in U))
    rc_run = G0F.child(rec, "run", [C.HF_DIR / "run.py", "run", run_csv, "g2", "hf", max(1, min(len(U), G0F.cpus()))],
                       C.HF_DIR, env, ok=(0, 2))
    if rc_run == 2:
        raise Stop("run", "run.py left reactions (rerun td0_h)", 2)
    G0F.child(rec, "allgeom", [THIS, "_allgeom", lst], C.HF_DIR, env)
    pj = hd / "parse.json"
    G0F.child(rec, "parse", [THIS, "_hfparse", lst, pj], C.HF_DIR, env)
    g = C.read_json(pj)
    if g.get("env") != rec["child_env"]:
        raise Stop("parse", f"_hfparse ran with {g.get('env')}", 3)
    R = g["records"]
    for T in jobs:
        s = (R.get(T) or {}).get("status")
        if s == "fail":
            chem.append(f"HF:{T}:{(R.get(T) or {}).get('reason')}")
        elif s != "ok":
            path_err.append(f"HF:{T}:{s}:{(R.get(T) or {}).get('reason')}")
    labj = {r["job_id"]: r for r in C.read_json(hd / "labels_d1_hf_g2.json")}
    bad_rid = [T for T in okj if T in labj and labj[T].get("rxn_id") != rid_of[T]]
    if bad_rid:
        path_err.append(f"_allgeom rxn_id wrong for {bad_rid}")
    ch = coreh_jobs(hd / "runs" / "g2", jobs)
    good = not path_err and len(chem) <= 2
    rec.update(ids=ids, jobs=jobs, path_errors=path_err, chemistry_failures=chem, core_h=dict(sum=float(sum(ch.values())),
               per_rxn=G0F.qstats(ch.values())), records={T: {k: (R.get(T) or {}).get(k) for k in (
                   "status", "reason", "d1", "d2", "barrier", "elst", "pauli", "oi", "disp", "cpcm", "cds")}
                   for T in jobs},
               rule=DECISIONS["GEO-14 T-D0"], verdict="pass" if good else "fail", **{"pass": bool(good)})
    rec["summary"] = dict(verdict=rec["verdict"], n_path_errors=len(path_err), n_chemistry_failures=len(chem),
                          core_h=float(sum(ch.values())))


def cmd_td0_h(rerun=False):
    return t_check("D0H", TD0H_JSON, lambda rec: td0h_body(rec, rerun))


# ================================================================ report (GEO2ENV=reactot)
def cmd_report(accept=()):
    def body(rec):
        import pandas as pd
        env_reactot(rec)
        ts = t_state()
        stop, incomplete = [], []
        for k, v in ts.items():
            if v["verdict"] == "fail":
                stop.append(f"T-{k} failed ({v['summary']})")
            elif not v["passed"] or v["stale"]:
                incomplete.append(f"T-{k}: {v['verdict']}{' (stale code)' if v['stale'] else ''}")
        try:
            inputs_check(rec)
        except Stop as e:
            (stop if e.rc == 3 else incomplete).append(f"combined inputs: {e.msg}")
        base, zinfo = geo_state()                          # GEO-1
        for c, w in zinfo["problems"]:
            (stop if c == 3 else incomplete).append(f"Phase Z: {w}")
        rep = dict(spec="docs/specs/GEO2.md 6 (STOP: 6-2 per dataset), 7; D0D1 design steps 6-8", written=now(),
                   git_head=G0F.git_head(), decisions=DECISIONS,
                   thresholds=dict(G_order=ORDER_MAX, G_freq=FREQ_MAX, F_rate=F_RATE_MAX, B6=B6_MAX, H_run=H_RUN_MAX,
                                   H_input=H_INPUT_MAX, H_rel=H_REL_MAX, H_rel_tol_eh=H_REL_TOL_EH,
                                   H_core_h_cap=H_COREH_MAX, QA_rate=QA_MAX_RATE, P_ratio=P_MIN_RATIO),
                   accept=list(accept), T=ts, comb_inputs=rec.get("comb_inputs"),
                   Z={k: v for k, v in zinfo.items() if k != "problems"}, n_base=len(base), methods={})
        grows = []
        if base:
            gs, grows = g_scan(GEO, base)
            grs = g_report_state(base)
            if grs["complete"] and grs.get("gate_json"):
                gv = {int(x["rxn_id"]): x for x in (G0F.read_json(grs["gate_json"]).get("rows") or [])}
                for x in grows:
                    v = gv.get(int(x["rxn_id"])) or {}
                    x.update(ts_fragments=v.get("tsf"), b6_check=v.get("b6"))
            fs = f_summary(GEO, base, gs, grows, accept=accept)
            hs = h_summary(GEO, base)
            ps = p_summary(GEO)
            pr = G0F.read_json(P_SETS_REPORT)
            usable = set(int(x) for x in ps["usable"]) if ps["table_ok"] else set()

            def alloc(prefix):
                return float(sum((G0F.read_json(p).get("core_h_alloc") or 0.0) for p in STEPS.glob(f"{prefix}*.json")))
            mrep = dict(G={k: v for k, v in gs.items() if k != "fail_rxn"}, G_fail_rxn=gs["fail_rxn"],
                        G_report={k: grs.get(k) for k in GR_KEYS},
                        F={k: v for k, v in fs.items() if k != "tokens_ok"}, H={k: v for k, v in hs.items() if k != "ok_jobs"},
                        H_pilot=pilot_state(), H_report=dict(present=H_REPORT.is_file(),
                                                             **{k: G0F.read_json(H_REPORT).get(k)
                                                                for k in ("complete", "stop", "stop_reasons")}),
                        P={k: v for k, v in ps.items() if k != "usable"}, n_usable=len(usable),
                        n_usable_by_dataset={d: len(v) for d, v in by_ds(usable).items()},
                        pairs=dict(present=bool(pr), stop=pr.get("stop"), stop_reasons=pr.get("stop_reasons"),
                                   n=pr.get("n"), n_P_main_reference=pr.get("n_P_main_reference"),
                                   sha256=pr.get("sha256"), check=pr.get("check"),
                                   d0_usable=((pr.get("d0_pairs") or {}).get("usable_geo") or {}).get("n")),
                        core_h=dict(g_step=gs["core_h"], hf=hs.get("core_h"), g_alloc=alloc("g_"), f_alloc=alloc("f_"),
                                    h_alloc=alloc("h_"), p_alloc=alloc("p_"), t_alloc=alloc("t")))
            for name, s in (("G", gs), ("G report", grs), ("F", fs), ("H", hs)):
                if s.get("stop"):
                    stop += [f"{GEO} {name}: {x}" for x in (s.get("stop_reasons") or ["stop"])]
            if not gs["complete"]:
                incomplete.append(f"{GEO} G: {gs['n_open']} open")
            elif not grs["complete"]:
                incomplete.append(f"{GEO} G report: {grs['missing']}")
            if not fs["complete"]:
                incomplete.append(f"{GEO} F: missing {fs['missing']}")
            if not hs["complete"]:
                incomplete.append(f"{GEO} H: missing {hs.get('missing')} open {(hs.get('open') or {}).get('n')} absent "
                                  f"{hs.get('n_absent')}")
            if ps["stop"]:
                stop.append(f"{GEO} P: {ps['stop_reason']}")
            elif not ps["table_ok"]:
                incomplete.append(f"{GEO} P: no passing P_build_{GEO}.json / table")
            elif ps["stale"]:
                incomplete.append(f"{GEO} P: inputs changed since the build {ps['stale'][:3]} (rerun p)")
            if pr.get("stop"):
                stop.append(f"{GEO} pairs (spec 7): {pr.get('stop_reasons')}")
            elif not pr:
                incomplete.append(f"{GEO} pairs: no {P_SETS_REPORT.name} (run p_report)")
            rep["methods"][GEO] = mrep
        G = pd.DataFrame(grows)
        C.write_atomic(G_STATUS, lambda t: G.to_csv(t, index=False))
        rep.update(g_status_csv=str(G_STATUS), g_status_sha256=C.sha256(G_STATUS), stop=bool(stop), stop_reasons=stop,
                   complete=not incomplete, incomplete=incomplete)
        write_json(F_REPORT, rep)
        rec["report"] = {k: rep[k] for k in ("stop", "stop_reasons", "complete", "incomplete")}
        rec["report"]["n_base"] = len(base)
        rec["report"]["n_usable"] = (rep["methods"].get(GEO) or {}).get("n_usable")
        return 3 if stop else (2 if incomplete else 0)
    return step("report", body)


# ================================================================ CLI
def main(a):
    cmd, n = (a[0] if a else ""), len(a)
    if cmd in ("t2_g", "t3", "td0_g", "td0_h") and a[1:] in ([], ["--rerun"]):
        return dict(t2_g=cmd_t2_g, t3=cmd_t3, td0_g=cmd_td0_g, td0_h=cmd_td0_h)[cmd](rerun=n == 2)
    if cmd in ("t2_f", "td0_f") and n == 1:
        return dict(t2_f=cmd_t2_f, td0_f=cmd_td0_f)[cmd]()
    if cmd == "g" and n in (3, 4) and (n == 3 or a[3] == "retry"):
        return cmd_g(G0F.array_k(a[1]), int(a[2]), retry=n == 4)
    if cmd == "g_report" and (n == 1 or (n == 2 and a[1].startswith("--accept="))):
        acc = parse_accept(a[1]) if n == 2 else ()
        if set(acc) - {"B6"}:
            raise SystemExit(f"{a[1]}: g_report accepts only --accept=B6 (decisions GEO-7, GEO-11)")
        return cmd_g_report(acc)
    if cmd == "_gcheck" and n == 3:
        return _gcheck(a[1], a[2])
    if cmd in ("f_slice", "f_atoms") and n == 3:
        return dict(f_slice=cmd_f_slice, f_atoms=cmd_f_atoms)[cmd](G0F.array_k(a[1]), int(a[2]))
    if cmd == "h_inputs" and n == 2:
        return cmd_h_inputs(int(a[1]))
    if cmd == "h_batch" and n == 3:
        return cmd_h_batch(G0F.array_k(a[1]), int(a[2]))
    if cmd in ("p", "report") and (n == 1 or (n == 2 and a[1].startswith("--accept="))):
        acc = parse_accept(a[1]) if n == 2 else ()
        return (cmd_p if cmd == "p" else cmd_report)(acc)
    plain = {"f_merge": cmd_f_merge, "f_gate": cmd_f_gate, "f_tokens": cmd_f_tokens, "f_report": cmd_f_report,
             "f_rest": cmd_f_rest, "h_pilot": cmd_h_pilot, "h_build": cmd_h_build, "h_report": cmd_h_report,
             "p_report": cmd_p_report, "_qgate": _qgate}
    if cmd in plain and n == 1:
        return plain[cmd]()
    if cmd == "_hfparse" and n == 3:
        return _hfparse(a[1], a[2])
    if cmd in ("_geominputs", "_allgeom") and n == 2:
        return dict(_geominputs=_geominputs, _allgeom=_allgeom)[cmd](a[1])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
