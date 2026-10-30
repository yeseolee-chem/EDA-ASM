#!/usr/bin/env python3
r"""feat_geo.py — GEO2 Phase G / F / H / P of the geo rung (docs/specs/GEO2.md 6; STOP rules 6-2), the pair-set report of
spec 7 (p_report) and the Phase T checks T-2 / T-3 (spec 5). A copy of analysis/d1_g3/feat.py (D1-G3, reviewed and run
for g3r) with the spec 6-1 changes; analysis/geo2/g3c.py is the matching copy of g3_common.py. sbatch only; nothing in
this file was run when it was written (login node), so every path is untested until it runs inside a job (T-2 / T-3 and
the C tests are its checks). Reused code is imported unchanged: D1LADDER g0f0 (G0 Freq helpers, F0 child environments,
child / env_f / inputs_args / constants) and inputs (Phase P functions), D1-G2 test_t (run, env_f, env_hf, T-1 / T-3
constants) and phase_fh (H status classes, batch rule), D1PAIR d1_features.py runroot / mergeroot and fu2/atoms.py root
run / gate / tokens, D1-HF prep.py geominputs, run.py run, build.py check / allgeom / parse_rxn.

The geo base (decision GEO-1) = every target (lad_common.targets(), job_id order) whose Phase Z folder
$GEO2/geo_final/<rxn_id>/ has .done (analysis/geo2/build_geo.py; results/Z_report.json must say stop false):
  G  $GEO2/geo_feat/<rxn_id>/ in the G1 folder format the feature and FU2 code read: ts.xyz, rel1.xyz, rel2.xyz = byte
     copies of the Z files; work/ts.hess = ORCA 6.1.1 '! XTB2 ALPB(water) Freq' at that fixed TS (no optimisation;
     the g0f0 G0 path), work/ts.inp, work/ts.out.gz; result.json with imag_freqs_cm, n_imag, imag_mode_forming_share =
     what ext_features.block_b6 recomputes; .done | .fail_<reason> written last (folder built as <rxn_id>.tmp, then
     renamed). Slices 'g <i|A> <n>' as LADDER g0 (decision GEO-2). $GEO2/geo_final is never written (F-1).
     g_report (GEO-11 = LADDER g0_report): the spec 6-2 G gate over the whole base, with ts_fragments +
     ext_features.block_b6 run on every .done folder (child _gcheck) -> results/G_report.json (B6 > 1 % = STOP).
  F  f_slice <i> <n>: d1_features.py runroot $GEO2/geo_feat $GEO2/F_geo/out $GEO2/F_geo/ids_all.csv i n --labels/--meta/
     --prof (the verbatim G2 inputs, read only; D1G2_SCRATCH=$GEO2 in the child env only); f_rest: mergeroot ->
     $GEO2/X182_d1_geo.parquet, fu2/atoms.py root run 0 1 / gate / tokens with D1PAIR_PHASE=FROOT,
     G1_ROOT=$GEO2/geo_feat, FU2_SCR=$GEO2/fu2_geo, FU2_RES=analysis/geo2/results/fu2_geo,
     FU2_X182=$GEO2/X182_d1_geo.parquet, FU2_ATOM_VERSION=fu2-atoms-3-geo, FU2_TOKEN_VERSION=fu2-tokens-3-geo,
     ESPLEY_OUT / R5_SCRATCH under $GEO2/F_geo; then the F gate (f_report).
  H  D1-HF runner on the 'g2' geometry slot with D1HF_SCRATCH=$GEO2/hf_geo, D1HF_RES=analysis/geo2/results/hf_geo,
     D1HF_TEMPLATES=/gpfs/tmp_cpu2/yeseo1ee/d1_hf/inputs (read only), D1HF_G2_ROOT=$GEO2/geo_final, split as D1-G2
     phase_fh: h_inputs <W> (prep.py geominputs g2 <base list> once, batch plan + the 50-reaction pilot list),
     h_pilot (spec 6-1 pilot: core-h projection, STOP > 2,500 core-h; rel-energy gate on the pilot), h_batch <k> <W>
     (run.py run + build.py check per batch), h_build (build.py allgeom <base list> g2 ->
     results/hf_geo/labels_d1_hf_g2.parquet, H_status.csv, $GEO2/hf_geo/labels_d1_hf_g2.json), h_report.
  P  D1LADDER Phase P rules with inputs.RUNGS['geo'] = rung_cfg('geo') registered at run time (inputs.py itself
     unchanged; if inputs.py already holds RUNGS['geo'] (spec 8-1) the values must be equal): Phase Q ready (_q_ready
     rule by repo-relative fingerprint, F-13), _read_h, refs_min, constrained keys keep their own reference (F-7b),
     assemble_rung, token_info, make_table, table_summary -> $GEO2/inputs/geo.parquet + results/P_build_geo.json.
  p_report  spec 7: P_geo, P_g2, P_dft (LADDER usable-pair rule), P_23 = P_g2 & P_geo, P_023 = P_dft & P_g2 & P_geo ->
     results/P_geo_sets.json, results/P_geo_report.json (STOP |P_23| < 0.98 |P_g2|), $GEO2/inputs/sets/P_<set>.csv.
T-2  $GEO2/t2/root/<rxn_id> = byte copies of ts.xyz / rel1.xyz / rel2.xyz of 3 $LAD/dft_final folders, then the G code
     above and the F path with versions fu2-atoms-3-dft / fu2-tokens-3-dft: 182 columns vs $LAD/X182_d1_dft.parquet
     (<= 1e-8) and token sha256 vs analysis/d1_ladder/results/F0_tokens_manifest.csv -> results/T_2.json
T-3  the D1-HF g2 path with D1HF_G2_ROOT = $LAD/dft_final (read only) for T00003, T00019, T00614: parse_rxn(T, 'g2') vs
     analysis/d1_hf/results/labels_d1_hf.parquet geom=dft (<= 1e-6 kcal/mol) -> results/T_3.json

CLI (exit 0 = done, 2 = not ready / incomplete (resubmit later), 3 = STOP (spec 6-2 / 7) or hard error; every command
except the T checks writes results/F_steps/<step>.json; a rerun overwrites its own record). Job files and the exact
sbatch lines: analysis/geo2/RUNBOOK_feat.md.
  feat_geo.py t2_g [--rerun]        jobs/geo2.slurm GEO2ENV=d1 (submit from analysis/geo2)
  feat_geo.py t2_f                  analysis/d1_pair jobs/g.slurm, PYENV=venv5, D1PAIR_PHASE=F
  feat_geo.py t3 [--rerun]          analysis/d1_hf jobs/hf.slurm
  feat_geo.py g <i|A|@I> <n> [retry]  GEO2ENV=d1: G slice i of n ('retry' first moves this slice's .fail_freq_* /
                                    .fail_error_* / stale folders to $GEO2/geo_feat_prev/<rxn_id>.<time>; explicit only)
  feat_geo.py g_report [--accept=B6]  d1_pair g.slurm venv5 F: results/G_report.json (G gate incl. B6, GEO-11; --accept
                                    only after a user decision, GEO-7)
  feat_geo.py f_slice <i|A|@I> <n>  d1_pair g.slurm venv5 F: runroot slice i of n (g.slurm turns '@I' into the digit)
  feat_geo.py f_atoms <i|A|@I> <n>  d1_pair g.slurm venv5 F: fu2/atoms.py root run i n (per-reaction npz only)
  feat_geo.py f_rest                d1_pair g.slurm venv5 F: f_merge, f_atoms 0 1, f_gate, f_tokens (stops at the first
                                    non-zero), then always f_report (each also a command of its own)
  feat_geo.py h_inputs <W>          d1_hf hf.slurm: geominputs + batch plan for W workers + pilot list
  feat_geo.py h_pilot               d1_hf hf.slurm: the 50 pilot reactions -> results/H_pilot.json
  feat_geo.py h_batch <k|A|@I> <W>  d1_hf hf.slurm: worker k of W (needs a passing pilot)
  feat_geo.py h_build | h_report    d1_hf hf.slurm: labels / H_status, then results/H_report.json
  feat_geo.py p [--accept=F-1[,B6]] GEO2ENV=reactot: rung input table (--accept only after a user decision, GEO-7)
  feat_geo.py p_report              GEO2ENV=reactot: results/P_geo_sets.json + P_geo_report.json (spec 7)
  feat_geo.py report [--accept=...] GEO2ENV=reactot: results/F_report.json + results/F_G_status.csv
  internal: _hfparse <list.csv> <out.json>   (child of t3 / h_pilot: parse_rxn(T, 'g2') under the D1HF_* overrides)
            _gcheck <ids.csv> <out.json>     (child of g_report: g0f0._g0c_one on $GEO2/geo_feat)
Preconditions: g / g_report / f_* / h_* / p need results/T_2.json and T_3.json passed with the current code (sha256 of
CODE_REL; F-10), results/Z_report.json stop false with Z_status.csv = the .done markers (GEO-1); f_* need the G gate
over the whole base (complete, no STOP) and a current passing results/G_report.json (GEO-11); P needs G, the G report,
F, H complete without STOP. Exit 3 = STOP: read results/F_steps/<step>.json
(fail_stage, fail) and the gate report, report and wait. Resubmission is always the same line (idempotent: finished
G folders, F slices / merge / atoms npz, H .done reactions and good outputs are kept; reports recompute).

Decisions (user: every judgment = the recommended option): DECISIONS below; written to results/DECISIONS.md (keys
'feat_geo/<key>') when any command starts, under the shared fcntl lock $GEO2/locks/DECISIONS.lock (F-11), and copied
into F_report.json. The judgment calls of this copy are also listed in results/DECISIONS_feat.md.
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
import g0f0 as G0F  # noqa: E402      D1LADDER G0 / F0 helpers (read only, imported)

L, DP = C.L, C.DP
Stop, Fail = G0F.Stop, G0F.Fail
if Path(G0F.__file__).resolve() != (C.LAD_DIR / "g0f0.py").resolve():
    C.stop(f"g0f0 imported from {G0F.__file__}, not {C.LAD_DIR / 'g0f0.py'}")

# ---------------------------------------------------------------- constants
GEO = C.METHODS[0]                                        # "geo" (spec 6-1)
T0 = float(os.environ.get("GEO2_JOB_T0") or os.environ.get("D1HF_JOB_T0") or time.time())
STOP_AFTER_H = 44.0                                     # no new G reaction after this (48 h wall)
# spec 6-2 STOP rules (= D1LADDER G0 / F0; D1-G2 H)
ORDER_MAX, FREQ_MAX, B6_MAX = G0F.ORDER_MAX, G0F.FREQ_MAX, G0F.B6_MAX          # G: 0.5 %, 1 %; B6 1 % (G report + F)
F_RATE_MAX = G0F.F0_RATE_MAX                            # F: F-1, Q-a, T-a 1 %
H_RUN_MAX, H_INPUT_MAX, H_REL_MAX = 0.01, 0.01, 0.005   # H: run failures, input check (D1-G2 rule), rel-energy gate
H_REL_TOL_EH = 1e-6                                     # spec 6-1 rel-energy gate (per reaction)
PILOT_N, PILOT_COREH_MAX = 50, 2500.0                   # spec 6-1 pilot
D1G2_H_REF = dict(n=8616, core_h=1611.0, per_rxn=0.187, source="spec 6-1 (D1-G2 Phase H)")
ACCEPTABLE = ("F-1", "B6")                              # GEO-7: rate rules a user decision may accept (--accept=)
P_MIN_RATIO = 0.98                                      # spec 7: |P_23| >= 0.98 |P_g2|
SET_NAMES = ("P_geo", "P_23", "P_023", "P_g2", "P_dft")
STEPS = C.RES / "F_steps"
F_REPORT = C.RES / "F_report.json"
G_STATUS = C.RES / "F_G_status.csv"
H_REPORT, H_PILOT = C.RES / "H_report.json", C.RES / "H_pilot.json"
P_SETS, P_SETS_REPORT = C.RES / "P_geo_sets.json", C.RES / "P_geo_report.json"
Z_REPORT, Z_STATUS = C.RES / "Z_report.json", C.RES / "Z_status.csv"          # build_geo.py (Phase Z)
T2_JSON, T3_JSON = C.RES / "T_2.json", C.RES / "T_3.json"
G_REPORT = C.RES / "G_report.json"                      # g_report (GEO-11): the spec 6-2 G gate incl. B6
G_CHECK = C.G3 / f"G_{GEO}"                             # $GEO2/G_geo: gate_ids.csv, gate.json, gate_xtb, gate_r5
T2_DIR, T3_DIR = C.G3 / "t2", C.G3 / "t3"
G2_HF_LABELS = C.G2_RES / "labels_d1_hf_g2.parquet"   # D1-G2 HF labels (read only; rel-energy gate, pilot calibration)
RETRY_REC = STEPS / "f_atoms_retry.json"
G_BUDGET_GB = float(os.environ.get("GEO2_G_BUDGET_GB", "2"))   # G-step bytes kept per slice run (G0-6 rule, F-2)
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
CODE_REL = ["analysis/geo2/feat_geo.py", "analysis/geo2/g3c.py", "analysis/geo2/jobs/geo2.slurm",
            "analysis/d1_ladder/g0f0.py", "analysis/d1_ladder/lad_common.py",
            "analysis/d1_g2/g2_autode.py", "analysis/d1_g2/g2_common.py", "analysis/d1_g2/test_t.py",
            "analysis/d1_g2/config_g2.yaml",
            "analysis/d1_pair/d1_features.py", "analysis/d1_pair/d1pair_common.py", "analysis/d1_pair/d1_g1.py",
            "analysis/d1_pair/fu2/atoms.py", "analysis/d1_pair/fu2/tokens.py", "analysis/d1_pair/fu2/fu2_common.py",
            "analysis/d1_pair/jobs/env.sh", "analysis/d1_pair/jobs/g_env.sh", "analysis/d1_pair/jobs/g.slurm",
            "analysis/d1_hf/prep.py", "analysis/d1_hf/run.py", "analysis/d1_hf/build.py", "analysis/d1_hf/hf_common.py",
            "analysis/d1_hf/jobs/hf.slurm", "analysis/d1_hf/jobs/hf_env.sh"]
DKEY = "feat_geo/"                                      # prefix of this module's keys in results/DECISIONS.md

DECISIONS = {
    "GEO-1 base": "기준 집합 = lad_common.targets() 중 $GEO2/geo_final/<rxn_id>/.done인 반응 전부(job_id 순서, 명령서 "
                  "6-1). 전제: results/Z_report.json stop false(없음 = exit 2, stop = exit 3)이고 Z_status.csv의 ok/kept "
                  "행 = Z_report n_folders = .done 표지(다르면 exit 3: Z 뒤 폴더가 바뀜). 대체(fallback) 폴더(G2 TS)도 "
                  ".done이므로 기준 집합에 들어간다(명령서 0-2).",
    "GEO-2 G slices": "G = 'g <i|A> <n>'(LADDER g0f0 g0와 같은 방식): 기준 반응(job_id 순서)의 i::n 조각에 feat.py의 "
                      "g_one / run_pool / order_g를 그대로 쓴다(조각마다 canary·abort·바이트 예산). 조각은 자기 반응 "
                      "폴더와 F_steps/g_<i>of<n>.json만 쓴다. 조각은 stale만 STOP(exit 3)으로 보고 끝나지 않은 반응이 "
                      "있으면 exit 2다. 명령서 6-2 G 비율(원자 순서 > 0.5 %, Freq > 1 %, B6 > 1 %)은 조각이 아니라 전체 "
                      "기준 집합에서 g_report(GEO-11)가 계산하고 f_slice / f_* / p / report가 다시 확인한다. 'retry'는 그 조각의 .fail_freq_* / .fail_error_* / "
                      "stale 폴더만 $GEO2/geo_feat_prev로 옮긴다(명시적일 때만).",
    "F-1 feature root": "feature 루트 = $GEO2/geo_feat/<rxn_id>(G1 폴더 형식): $GEO2/geo_final(Phase Z)의 ts.xyz·rel1.xyz·"
                        "rel2.xyz를 바이트 그대로 복사하고 result.json·work/ts.hess(XTB2 ALPB(water) Freq)를 새로 쓴다. "
                        "ext_features.block_b6가 result.json의 imag_freqs_cm[0]·imag_mode_forming_share를 work/ts.hess에서 "
                        "다시 계산한 값과 비교한다. geo_final은 Z가 만든 그대로 두고 G1 형식 사본에 쓴다(D1-G3 F-1, "
                        "LADDER G0의 dft_final과 같은 방식). 사본의 sources sha256이 지금 geo_final 파일과 다르면 stale = "
                        "STOP이고 'g <i> <n> retry'가 치운다. D1-HF는 명령서 6-1대로 $GEO2/geo_final을 읽는다"
                        "(D1HF_G2_ROOT). 두 루트의 좌표 파일은 바이트 단위로 같다.",
    "F-2 G step": "work/ts.hess = g0f0 G0 경로(g0_modules: g2_autode.FREQ_HEADER nprocs 1, %maxcore = config_g2 maxcore_mb; "
                  "orca_direct.write_inp / run / parse_hess / imag_and_mode; g0f0.hess_atoms; run_ts.forming_share(라벨 "
                  "형성 결합); TOL_HESS_GEOM_A 1e-4; 노드 로컬 작업 폴더; canary·abort·바이트 예산 규칙 G0-6, 예산 "
                  "GEO2_G_BUDGET_GB 기본 2 GB = 조각 하나의 run_pool 호출마다). g0_one은 $LAD/dft_final과 D1-HF dft 입력에 "
                  "묶여 있어 그 작업 루프를 옮긴 D1-G3 feat.py의 g_one / run_pool / order_g를 그대로 쓴다; T-2가 이 "
                  "코드를 DFT TS 3개에서 G0 결과와 비교한다. 원자 순서 gate: TS 원소 순서 = D1 job ts.xyz, 조각 크기 = "
                  "input_meta A_idx, 역할, rel1/rel2 원소 순서 = ref_dipole/ref_dipolarophile, rel 원소 다중집합 = TS 조각"
                  "(새 구조라 좌표 일치 검사는 없다). n_imag 0 / >= 2는 .done으로 기록하고(G0-4, 명령서 6-2) B6 계산 "
                  "불가는 G 단계의 g_report(GEO-11)가 ext_features.block_b6로 세고 F가 다시 센다.",
    "F-4 gate": "명령서 6-2 STOP 기준(= D1LADDER G0·F0, D1-G2 H). G(GEO-11 g_report = LADDER g0_report): 원자 순서 실패"
                "(순서 gate + ts_fragments) > 기준 반응의 0.5 %, Freq(+예상 밖 오류, G0-5) 실패 > 1 %, B6 계산 불가"
                "(ext_features.block_b6를 G 폴더에서) > .done 폴더의 1 %, stale 사본. F(G .done 반응 수 기준, F0-3): "
                "F-1(유한한 182열 행이 없는 G .done 반응 전부, B6만 실패한 반응 포함 = g0f0 f0_report, D1PAIR F-1) > 1 %, "
                "B6(status_B6 not ok/skipped = ext_features.block_b6 판정) > 1 %, Q-a(Q_manifest qa_ok false) > 1 %, "
                "T-a(T_report bad) > 1 %, FU2 gate 자체의 STOP(Q_report·T_report stop), mergeroot stop, 개수 불일치, "
                "토큰 배치 != D1PAIR g1, G가 예측한 B6 상태 != F status_B6(B6 블록이 실제로 돈 행; g0f0 f0_report). "
                "B6만 실패한 행을 F-1에서 빼는 LADDER F0_DECISION.md 해석(그쪽 F0 STOP 뒤 사용자 결정)은 미리 적용하지 "
                "않는다: 수는 정보(n_F1_without_B6_only)로만 내고, 적용은 사용자 결정 뒤 --accept=F-1로만(GEO-7). H: 실행 실패 / 끝난 반응 > 1 %(GEO-6), HF 입력 검사(geominputs status != ok) / "
                "기준 반응 > 1 %(D1-G2 규칙), rel 에너지 gate 불일치 > 0.5 %(GEO-5). n_imag 0 / >= 2는 실패가 아니다"
                "(기록만). 토큰이 없는 기준 반응 비율(D1-G3 합산 규칙)은 정보로만 보고한다.",
    "F-5 F layout": "F = g0f0 F0 구조: f_slice <i> <n>(d1_features.py runroot slice i of n, 배열) -> f_rest(mergeroot -> "
                    "atoms.py root run 0 1 -> err.json이 남고 지금 루트에서 exit 0인 재시도 기록이 없으면 한 번 더(f0_gate "
                    "규칙) -> root gate -> root tokens -> f_report). f_atoms <i> <n>은 따로 쓸 수 있다(반응별 npz만 씀). "
                    "버전 fu2-atoms-3-geo / fu2-tokens-3-geo; D1G2_SCRATCH=$GEO2는 d1_features 자식 환경에만(F0-2); "
                    "입력 = G2 입력 그대로(--labels/--meta/--prof). root_fp.json·ids_all.csv(F-15)는 f_slice가 "
                    "$GEO2/locks/F_geo.lock fcntl 잠금 안에서 쓴다(배열 task가 같은 파일을 동시에 쓰지 않는다).",
    "F-6 HF": "D1-HF runner의 g2 자리를 geo로 덮어쓴다: D1HF_SCRATCH=$GEO2/hf_geo, D1HF_RES=results/hf_geo, "
              "D1HF_TEMPLATES=d1_hf/inputs(읽기만), D1HF_G2_ROOT=$GEO2/geo_final. D1-G2 phase_fh H 구조: h_inputs <W>"
              "(prep.py geominputs g2 <기준 목록> 한 번 -> 입력 ok 반응을 (n_atoms, job_id) 순서로 round robin 배치, nb = "
              "W x ceil(n / (1000 W)), worker k = j % W == k; pilot 목록) -> h_pilot -> h_batch <k> <W>(run.py run 한 번, "
              "큰 반응 먼저, $SLURM_CPUS_PER_TASK개 직렬 ORCA; 배치마다 build.py check -> results/hf_geo/M_checks) -> "
              "h_build(build.py allgeom <기준 목록> g2) -> h_report. 결과의 geom 열은 슬롯 이름 'g2'(= geo)다. "
              "SlowConv 재시도 1회는 run.py 그대로. H는 G·F를 기다리지 않는다(독립 결과).",
    "GEO-4 pilot": "명령서 6-1 pilot: 입력 ok 기준 반응 중 job_id 순서 첫 50개를 h_pilot이 먼저 돌린다(run.py, 같은 run 폴더라 "
                   "h_batch에서는 .done으로 건너뜀). 반응당 코어시간 = timing.tsv wall_seconds 합(직렬 ORCA: wall = core "
                   "time; D1-G2 1,611코어시간과 같은 측정). 예상 = (pilot 평균) x 입력 ok 반응 수(명령서 6-1 규칙 그대로); "
                   "2,500 초과 = STOP(exit 3; 끝난 pilot 반응이 없어 예상이 유한하지 않아도 STOP). D1-G2 보정 예상(같은 "
                   "반응의 geo / G2 코어시간 비 x D1-G2 반응별 코어시간 합)은 정보로만 H_pilot.json에 적고, 그것만 2,500을 "
                   "넘으면 STOP이 아니라 경고로 H_pilot.json warnings와 results/DECISIONS.md('GEO-4 pilot warning')에 "
                   "남긴다(검토 2026-10-30: 판정 기준을 바꾸지 않는다, 명령서 11행). rel 에너지 gate를 pilot에도 같은 0.5 % 규칙으로 적용한다(50반응이면 불일치 1개 = STOP; 같은 "
                   "입력이므로 불일치는 경로 오류). pilot의 실행 실패는 보고만 하고, 1 % 규칙은 h_batch 누적 검사와 "
                   "h_report가 적용한다. h_batch는 results/H_pilot.json이 지금 plan.json으로 pass일 때만 돈다(없음·미완 = "
                   "exit 2, STOP = exit 3): '2,500 이하면 자동 진행'은 SLURM afterok 의존과 이 검사로 구현한다.",
    "GEO-5 rel gate": "명령서 6-1 HF 추가 gate: geo rel1/rel2.xyz = G2 바이트 복사이므로 e_frag1_rel_eh·e_frag2_rel_eh가 "
                      "analysis/d1_g2/results/labels_d1_hf_g2.parquet와 반응마다 <= 1e-6 Eh. 비교 대상 = geo ok 행 중 D1-G2 "
                      "ok 행이 있는 반응(없으면 n_not_in_reference로 보고하고 불일치로 세지 않는다). 불일치 / 비교 반응 > "
                      "0.5 % = STOP. geo ok 행이 있는데 비교할 반응이 0이면 STOP.",
    "GEO-6 H run failure": "명령서 6-2 H '실행 실패 > 1 %' = (H_status fail + gate_fail + eda_table_unexplained + "
                           "parse_fail) / 끝난 반응(phase_fh H_FIN). D1-G2는 gate 실패만 1 % 규칙에 넣고 fail(SCF / ORCA "
                           "실패)은 보고만 했으나 '실행 실패' 문구에 맞춰 둘 다 센다(D1-G2 H: 8,616반응 중 둘 다 0). "
                           "h_batch는 phase_fh처럼 실행 전·후에 이 plan의 배치 검사(M_checks)로 누적 비율을 본다.",
    "GEO-7 accept": "사용자가 F rate STOP을 받아들이는 결정을 하면(LADDER --accept-f0-stop, D1-G2 F_DECISION.md 선례) "
                    "'p --accept=F-1[,B6]' / 'report --accept=...'가 그 rate 규칙만 stop_reasons에서 accepted_stops로 "
                    "옮겨 기록한다(P_build_geo.json, F_report.json). G 단계의 B6 비율 STOP(GEO-11)은 'g_report "
                    "--accept=B6'로 같은 방식(G_report.json accepted_stops). 기본값은 STOP이고 다른 STOP(Q-a, T-a, FU2 gate, "
                    "mergeroot, 배치, 개수, G, H)은 받아들일 수 없다. 코드를 고치면 CODE_REL sha256이 바뀌어 T-2·T-3을 "
                    "다시 돌려야 하므로, 결정만 기록하고 진행할 수 있게 둔 장치다.",
    "F-7 P tables": "입력표 = inputs.py 함수를 그대로 부른다(RUNGS['geo']를 실행 중에 등록: geom g2, hf_runs $GEO2/hf_geo/runs, "
                    "hf_inputs $GEO2/hf_geo/inputs, h_parquet results/hf_geo/labels_d1_hf_g2.parquet, tokens / fu2_res / "
                    "x182 = F 출력, 버전 -geo, refs min): Phase Q 준비(_q_ready 규칙, 지문은 repo 상대 경로: F-13), "
                    "_read_h, refs_min(rung 안 Q-2 키별 최저 HF 기준 에너지, P6b), assemble_rung(P-a + 재조립 검사), "
                    "token_info(g1 배치), make_table(9,266행, usable 표시), table_summary. cmd_build는 $LAD에 쓰므로 "
                    "부르지 않는다. 표 = $GEO2/inputs/geo.parquet, 보고 = results/P_build_geo.json(rung_cfg 포함). "
                    "전제: G·F·H gate가 STOP 없이 complete.",
    "F-7b constrained keys": "제약 dipole 키(Q_keys constrained; '|fixed:' 접미사는 D1 DFT 기준 구조에서 나옴): g2 rung의 "
                             "P20 결정 (c)와 같이 교체하지 않고 자기 기준을 쓴다. refs_min은 rung 이름이 'g2'일 때만 그 방식을 "
                             "쓰므로(inputs.py refs_min), refs_min 결과에서 constrained 행의 'min' 교체를 되돌리는 후처리로 "
                             "같은 결과를 만든다(되돌린 수는 P_build의 references_g3_cons_own).",
    "GEO-8 P registration": "cmd_p는 inputs.RUNGS['geo'] = rung_cfg('geo')를 실행 중에 등록한다(명령서 6-1). 명령서 8-1에 "
                            "따라 inputs.py가 RUNGS['geo']를 고정해 두었으면 그 값과 rung_cfg를 문자열 값으로 비교한다"
                            "(Path 대 str 표기 차이로 같은 경로가 다르다고 판정되지 않게); 다르면 STOP(exit 3).",
    "GEO-9 pair sets": "명령서 7: 짝 universe = D1PAIR results/S_folds.csv(LADDER P8), P_<rung> = a·b 모두 rung 입력표 usable"
                       "(LADDER inputs.py _pairs와 같은 규칙). P_dft·P_g2는 $LAD/inputs/{dft,g2}.parquet에서 다시 계산하고 "
                       "LADDER P_report.json(표 sha256)·sets/P_dft.csv·P_g2.csv(sha256, pair_id 집합)와 같아야 한다(다르면 "
                       "STOP). P_geo = $GEO2/inputs/geo.parquet(P_build_geo.json 통과, sha256 일치), P_23 = P_g2 ∩ P_geo, "
                       "P_023 = P_dft ∩ P_g2 ∩ P_geo. 출력: results/P_geo_sets.json(정렬된 pair_id 목록), results/"
                       "P_geo_report.json(집합별 기준/anchor/para, P_main 기준 짝, fold별 수, sha256 = sha256('\\n'.join("
                       "정렬 id)), P_g2에 있고 P_geo에 없는 짝과 끝별 이유), $GEO2/inputs/sets/P_geo.csv·P_23.csv·P_023.csv"
                       "(LADDER SET_COLS). STOP: |P_23| < 0.98 |P_g2|(전체 집합 또는 P_main 기준 짝 부분집합 중 하나라도).",
    "F-8 T-2": "T-2 반응 = 1000003(Br), 1000019(Si), 1000614(g1_rec) = D1-G2 T-3 묶음: $LAD/dft_final의 ts.xyz·rel1·rel2 "
               "바이트 복사 -> G 단계 코드 그대로 xTB Freq -> runroot/mergeroot -> 182열 vs X182_d1_dft(1e-8; 넘으면 D1-G2 "
               "T-1 선례(scan_curv_*만, <= 2.5e-7) = pass_f0_precedent, 편차로 기록). 토큰은 D1-G2 T-2처럼 FU2_X182 = "
               "X182_d1_dft 세 행의 정확한 사본, 버전 -dft로 만들어 sha256을 F0_tokens_manifest.csv와 비교한다(토큰 glob이 "
               "X182 행 그대로라 182열 비교와 토큰 비교를 나눈다). 기준 파일 sha256(X182_d1_dft, F0_tokens_manifest)은 "
               "F0_report와 맞춰 본다; 세 반응은 dft Q_manifest에서 use·ext_ok여야 한다. 작업 폴더 $GEO2/t2.",
    "F-9 T-3": "T-3 = D1-HF g2 경로, D1HF_G2_ROOT = $LAD/dft_final(읽기만), T00003·T00019·T00614: geominputs -> 생성 입력을 "
               "d1_hf/inputs/<T>/hf와 비교(보고만) -> run -> allgeom -> parse_rxn(T, 'g2') vs labels_d1_hf.parquet geom=dft의 "
               "d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier, bond, eint_spe(<= 1e-6 kcal/mol, 둘 다 ok) + "
               "labels_d1_hf_g2.json 행 일치(D1-G2 t3 규칙). 작업 폴더 $GEO2/t3.",
    "F-10 T currency": "g / g_report / f_* / h_* / p는 results/T_2.json·T_3.json이 지금 코드(CODE_REL sha256 묶음)로 통과했을 때만 돈다"
                       "(없음·실행 중·코드 바뀜 = exit 2, 실패 = exit 3).",
    "F-11 decision log": "모든 명령(t2_g, t2_f, t3, g, g_report, f_*, h_*, p, p_report, report)이 시작할 때 이 결정을 모두 "
                         "results/DECISIONS.md에 'feat_geo/<키>'로 적는다(명령서 14 '적은 뒤 진행'). g3c.record_decision은 "
                         "잠금 없는 읽기-쓰기라 $GEO2/locks/DECISIONS.lock의 fcntl 잠금 안에서 부른다.",
    "F-12 quota": "G 폴더는 work/ts.inp, ts.out.gz, ts.hess만 남긴다(.gbw 등은 노드 로컬 작업 폴더와 함께 지움); HF는 run.py "
                  "규칙(.inp, .out.gz, .err.gz; .gbw·.tmp·.densities는 복사하지 않음).",
    "F-13 Q readiness": "Phase Q 준비 = inputs._q_ready(refs_v.q_ready)와 같은 규칙(Q_report all_pass, outputs sha256 그대로, "
                        "입력·코드 지문 그대로)이되 지문은 repo 상대 경로로 비교한다: 기록된 키가 d1ladder 작업 트리 절대 경로라 "
                        "이 작업 트리에서 refs_q.inputs_fingerprint()와 dict로 비교하면 항상 다르다(기존 코드 문제). 작업 트리 "
                        "밖 경로(/gpfs/tmp_cpu2/...)는 그대로 비교. sha256이 하나라도 다르면 STOP.",
    "F-15 feature root currency": "f_slice는 실행한 feature 루트를 $GEO2/F_geo/root_fp.json(기준 반응마다 G 폴더 result.json "
                                  "sha256)에 남긴다. 다음 f_slice에서 다르면(예: 'g ... retry' 뒤) $GEO2/F_geo, $GEO2/fu2_geo, "
                                  "results/fu2_geo, $GEO2/X182_d1_geo.parquet를 .prev_<시각>으로 치우고 다시 계산한다(runroot "
                                  "slice, mergeroot, atoms npz 캐시가 옛 구조를 쓰지 않게). f_merge·f_atoms·f_gate·f_tokens·"
                                  "f_report·p·report는 root_fp가 지금과 다르면 F 미완(exit 2).",
    "F-16 T currency of work folders": "T-2·T-3 작업 폴더($GEO2/t2, $GEO2/t3)에는 만든 코드의 sha256(code_sha256.txt)을 둔다. "
                                       "지금 코드와 다르면 t2_g / t3가 폴더를 .prev_<시각>으로 치우고 새로 계산하고, t2_f는 "
                                       "t2_g가 지금 코드로 만든 폴더에서만 돈다(아니면 exit 2: t2_g 다시) — 옛 코드의 캐시로 "
                                       "'지금 코드로 통과'가 기록되지 않게.",
    "GEO-10 removed": "D1-G3의 Part A(S_rxns.csv)·Phase O(O_report.json, opt.py)·U-1(U1_selection.json, stopped_user) "
                      "논리와 그 결정(F-3, F-14, U-2f-*)은 GEO2에 해당하지 않아 지웠다(기준 집합 = GEO-1). report의 "
                      "S_pairs.csv 짝 수는 p_report(GEO-9)의 집합 수로 바꿨다.",
    "GEO-11 G report": "명령서 6-2 G 'B6 계산 불가 > 1 % (ext_features.block_b6로 확인)'를 G 단계에서 본다(검토 2026-10-30; "
                       "LADDER g0_report와 같은 방식): g 조각이 모두 끝난 뒤 g_report(d1_pair g.slurm venv5 F)가 전체 기준 "
                       "집합을 g_scan하고, 자식 _gcheck(= g0f0 _g0check를 geo feature 루트 $GEO2/geo_feat에; 반응별 "
                       "검사는 g0f0._g0c_init / _g0c_one 그대로: ts_fragments + ext_features.block_b6)를 모든 .done "
                       "폴더에 돌려 results/G_report.json을 쓴다. STOP: 원자 순서(순서 gate + ts_fragments 실패) > 기준 "
                       "반응의 0.5 %, Freq / 오류 > 1 %, B6 계산 불가(ts_fragments ok인데 block_b6 실패) > .done의 1 %, "
                       "stale 사본. B6 비율 STOP만 사용자 결정 뒤 'g_report --accept=B6'로 accepted_stops에 옮길 수 있다"
                       "(GEO-7). f_slice / f_*는 지금 기준 집합·G 폴더(root_fp sha256)·gate.json에 맞는 통과한 G_report가 "
                       "있어야 돈다(없음·옛것 = exit 2, STOP = exit 3). F gate(f_summary)는 G가 예측한 B6 상태와 F의 "
                       "status_B6가 B6 블록이 실제로 돈 행에서 같은지 본다(다르면 STOP; g0f0 f0_report와 같음).",
}


# ================================================================ small helpers
def now():
    return G0F.now()


def paths(m):
    """every Phase G / F / H / P path of method m (m = 'geo': final = $GEO2/geo_final, feat = $GEO2/geo_feat)."""
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
    """{rxn_id: label row} of lad_common.targets() (labels_d1.json checks inside)."""
    if not _LAB:
        _LAB.update({int(d["rxn_id"]): d for d in L.targets()})
    return _LAB


def write_json(path, obj):
    return C.write_json(path, G0F.clean(obj))


def record_decision(key, text):
    """g3c.record_decision under an fcntl lock ($GEO2/locks/DECISIONS.lock, shared by every geo2 job): concurrent jobs
    never lose each other's DECISIONS.md lines (g3c.py itself has no lock). Keys carry the prefix 'feat_geo/'."""
    import fcntl
    lk = C.assert_path(C.G3 / "locks" / "DECISIONS.lock")
    lk.parent.mkdir(parents=True, exist_ok=True)
    with open(lk, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            C.record_decision(DKEY + key, text)
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
    """an existing $GEO2 / results path -> <name>.prev_<time> (never deleted)."""
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
        raise Stop("T", f"{bad} failed (spec 5 / 14 T STOP): read results/T_<k>.json", 3)
    open_ = [f"T-{k} ({'missing' if not v['present'] else v['verdict']}{', stale code' if v['stale'] else ''})"
             for k, v in st.items() if not v["passed"] or v["stale"]]
    if open_:
        raise Stop("T", f"{open_} not passed with the current code (run / rerun t2_g + t2_f, t3)", 2)


# ================================================================ the geo base (Phase Z folders; decision GEO-1)
def geo_state():
    """(base [(rxn_id, job_id)] in job_id order, info with 'problems' [(rc, why)]): the targets with
    $GEO2/geo_final/<rxn_id>/.done; Z_report.json (build_geo.py) present with stop false, and its Z_status.csv ok / kept
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
    if zs is None:
        probs.append((3, f"{Z_STATUS} missing although Z_report.json exists"))
    elif zs != ids or z.get("n_folders") != len(ids):
        probs.append((3, f"$GEO2/geo_final .done markers ({len(ids)}) != Z_status.csv ok/kept rows ({len(zs)}) / Z_report "
                         f"n_folders ({z.get('n_folders')}): only markers {sorted(ids - zs)[:10]}, only Z_status "
                         f"{sorted(zs - ids)[:10]} (folders changed after Phase Z)"))
    if not base and not probs:
        probs.append((3, f"no target with .done in {final}"))
    info.update(root=str(final), n_targets=len(tg), n_base=len(base), base_sha256=base_sha(base))
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
    """G / t2_g: the D1 run environment of jobs/geo2.slurm with GEO2ENV=d1 (ORCA 6.1.1, D1 code copy)."""
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
    """h_* / t3: the D1-HF job environment (analysis/d1_hf jobs/hf.slurm) without D1HF_* overrides (test_t.env_hf)."""
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
    """d1_features.py children: D1G2_SCRATCH=$GEO2 so that its ROOT_WRITE_OK guard accepts $GEO2 (= g0f0.feat_env)."""
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
    """D1-HF children of method m: the g2 geometry slot on $GEO2/<m>_final (decision F-6)."""
    P = paths(m)
    return dict(os.environ, D1HF_SCRATCH=str(P.hf_scr), D1HF_RES=str(P.hf_res), D1HF_TEMPLATES=str(HF_TEMPLATES),
                D1HF_G2_ROOT=str(P.final), PYTHONDONTWRITEBYTECODE="1")


# ================================================================ G step (GEO2ENV=d1)
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
                             f"quota, then resubmit with a larger GEO2_G_BUDGET_GB (explicit)", 3)
    return cnt


def g_scan(m, base):
    """(summary, rows) of $GEO2/<m>_feat over the base reactions (live scan; result.json values). Gate (spec 6-2 G =
    D1LADDER G0 thresholds, over the reactions given): atom-order failures > 0.5 %, Freq / error failures > 1 % (G0-5:
    unexpected errors count as Freq failures), or stale copies of changed geo_final files."""
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
                cat, reason = "stale", "geo_final_files_changed_since_the_G_step"
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
    n_order = sum(1 for x in fail if x["category"] == "order")
    n_freq = len(fail) - n_order
    n_open = sum(1 for x in rows if x["category"] == "open")
    stale = sorted(x["rxn_id"] for x in rows if x["category"] == "stale")
    ni = Counter(int(x["n_imag"]) for x in done if x["n_imag"] is not None)
    r_order = n_order / n if n else float("nan")
    r_freq = n_freq / n if n else float("nan")
    why = []
    if n and r_order > ORDER_MAX:
        why.append(f"{m}: atom-order failures {n_order}/{n} = {r_order:.4f} > {ORDER_MAX} (spec 6-2 G)")
    if n and r_freq > FREQ_MAX:
        why.append(f"{m}: Freq / error failures {n_freq}/{n} = {r_freq:.4f} > {FREQ_MAX} (spec 6-2 G)")
    if stale:
        why.append(f"{m}: {len(stale)} G folders are stale copies of changed geo_final files {stale[:10]} "
                   f"(feat_geo.py g <i> <n> retry moves them aside)")
    s = dict(root=str(P.feat), n_base=n, n_done=len(done), n_open=n_open, n_fail=len(fail), n_order_fail=n_order,
             n_freq_fail=n_freq, n_stale=len(stale), stale=stale[:100], by_category=dict(Counter(x["category"] for x in rows)),
             fail_reasons=dict(Counter(x["reason"] for x in fail).most_common(30)),
             fail_rxn={str(x["rxn_id"]): x["reason"] for x in fail},
             rates=dict(order=r_order, freq=r_freq), thresholds=dict(order=ORDER_MAX, freq=FREQ_MAX),
             stop=bool(why), stop_reasons=why, complete=n_open == 0 and not stale,
             n_imag={"0": ni.get(0, 0), "1": ni.get(1, 0), "2": ni.get(2, 0), ">=3": sum(v for k, v in ni.items() if k >= 3)},
             n_imag_note="n_imag 0 / >= 2 is recorded, not a failure (spec 6-2); B6 needs an imaginary mode: gated by "
                         "g_report (block_b6 on the folders, B6 > 1 %; GEO-11) and again at F",
             imag1_cm=G0F.qstats(x["imag1_cm"] for x in done), forming_share=G0F.qstats(x["forming_share"] for x in done),
             n_share_below_0p5=sum(1 for x in done if isinstance(x["forming_share"], (int, float))
                                   and x["forming_share"] < 0.5),
             core_h=dict(sum=float(sum(x["core_h"] or 0.0 for x in rows)), per_rxn=G0F.qstats(x["core_h"] for x in rows)))
    return s, rows


G_KEYS = ("n_base", "n_done", "n_open", "n_fail", "n_order_fail", "n_freq_fail", "n_stale", "rates", "stop",
          "stop_reasons", "complete", "n_imag")


def cmd_g(i, n, retry=False):
    P = paths(GEO)

    def body(rec):
        env_d1(rec)
        if not 0 <= i < n:
            raise Stop("args", f"slice {i} of {n}?", 3)
        base = geo_base(rec)
        need_t(rec)
        G0F.g2_inputs_check(rec)
        meta = G0F.read_meta()
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
        rec.update(slice=dict(i=i, n=n, n_slice=len(mine), n_base=len(base), budget_gb=G_BUDGET_GB), retry_moved=moved)
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


def gcheck_env():
    """environment of the _gcheck child (= g0f0.g0check_env with the geo feature root and $GEO2 scratch)."""
    g2i = G0F.G2_INPUTS
    return dict(os.environ, G1_ROOT=str(paths(GEO).feat), ESPLEY_LABELS=str(g2i["labels"]), ESPLEY_META=str(g2i["meta"]),
                ESPLEY_PROF=str(g2i["prof"]), ESPLEY_OUT=str(G_CHECK / "gate_xtb"), R5_SCRATCH=str(G_CHECK / "gate_r5"),
                PYTHONDONTWRITEBYTECODE="1")


def _gcheck(ids_csv, out_json):
    """child of g_report (gcheck_env; GEO-11) = g0f0._g0check on $GEO2/geo_feat: exit 3 unless the rev 4-6 modules picked
    up that root; then g0f0._g0c_one (ts_fragments + ext_features.block_b6, unchanged) for the listed rxn_ids in a spawn
    pool -> <out_json>."""
    import multiprocessing as mp
    import ext_features as ef  # noqa: F401
    import fragmenter as fr
    import r6_partition as RP
    import rev5_common as R5
    import xtb_slice as xs
    g2i = G0F.G2_INPUTS
    want = dict(G1_ROOT=paths(GEO).feat, ESPLEY_LABELS=g2i["labels"], ESPLEY_META=g2i["meta"], ESPLEY_PROF=g2i["prof"])
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


def cmd_g_report(accept=()):
    """GEO-11 (= D1LADDER g0_report): the spec 6-2 G gate over the whole base with the F code's own checks. Live scan
    (g_scan) + the _gcheck child on every G .done folder -> results/G_report.json. STOP (exit 3): atom-order failures
    (order gate + ts_fragments) > 0.5 % of the base, Freq / error > 1 % of the base, B6 not computable > 1 % of the .done
    folders (only this one may be accepted, with --accept=B6 after a user decision, GEO-7), stale copies; exit 2 = G
    incomplete. Recomputed on every run (idempotent; writes only $GEO2/G_geo and results/G_report.json)."""
    P = paths(GEO)

    def body(rec):
        G0F.env_f(rec)
        base = geo_base(rec)
        need_t(rec)
        G0F.g2_inputs_check(rec)
        gs, grows = g_scan(GEO, base)
        fp = root_fp(GEO, base)
        rec["G"] = {k: gs[k] for k in G_KEYS}
        rep = dict(spec="docs/specs/GEO2.md 6-2 G (= D1LADDER g0_report)", written=now(), git_head=G0F.git_head(),
                   rule=DECISIONS["GEO-11 G report"], root=str(P.feat), n_base=len(base), base_sha256=base_sha(base),
                   root_fp_sha256=fp_sha(fp), thresholds=dict(order=ORDER_MAX, freq=FREQ_MAX, b6=B6_MAX),
                   accept=list(accept), scan={k: gs[k] for k in G_KEYS}, accepted_stops=[])
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
        gr = {int(x["rxn_id"]): x for x in gate["rows"]}
        tsf_bad = {r: str((gr.get(r) or {}).get("tsf")) for r in done if (gr.get(r) or {}).get("tsf") != "ok"}
        b6_bad = {r: str(gr[r].get("b6")) for r in done if r not in tsf_bad and gr[r].get("b6") != "ok"}
        n_t, n_done = len(base), len(done)
        n_order = gs["n_order_fail"] + len(tsf_bad)
        n_freq = gs["n_freq_fail"]
        rates = dict(order=n_order / n_t, freq=n_freq / n_t, b6=len(b6_bad) / n_done)
        stop, accepted = [], []
        if rates["order"] > ORDER_MAX:
            stop.append(f"{GEO}: atom-order failures (order gate {gs['n_order_fail']} + ts_fragments {len(tsf_bad)}) "
                        f"{n_order}/{n_t} = {rates['order']:.4f} > {ORDER_MAX} (spec 6-2 G)")
        if rates["freq"] > FREQ_MAX:
            stop.append(f"{GEO}: Freq / error failures {n_freq}/{n_t} = {rates['freq']:.4f} > {FREQ_MAX} (spec 6-2 G)")
        if rates["b6"] > B6_MAX:
            msg = (f"{GEO}: B6 not computable (ext_features.block_b6) {len(b6_bad)}/{n_done} = {rates['b6']:.4f} > "
                   f"{B6_MAX} (spec 6-2 G)")
            (accepted if "B6" in accept else stop).append(msg)
        ok_rows = [x for x in gate["rows"] if x.get("b6") == "ok"]
        rep.update(complete=True, gate_json=str(gate_json), gate_json_sha256=C.sha256(gate_json), ids_csv=str(ids_csv),
                   gcheck_wall_s=gate.get("wall_s"),
                   counts=dict(base=n_t, done=n_done, order_gate_fail=gs["n_order_fail"], ts_fragments_fail=len(tsf_bad),
                               order_fail=n_order, freq_error_fail=n_freq, b6_not_computable=len(b6_bad)),
                   rates=rates,
                   ts_fragments_fail=dict(n=len(tsf_bad), reasons=dict(Counter(tsf_bad.values()).most_common(20)),
                                          rxn={str(r): w for r, w in sorted(tsf_bad.items())}),
                   b6_not_computable=dict(n=len(b6_bad), rate=rates["b6"], threshold=B6_MAX,
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
    """F-15: {rxn_id: sha256 of $GEO2/<m>_feat/<rxn_id>/result.json} over the base reactions (None = no result.json)."""
    P = paths(m)
    return {str(r): (C.sha256(P.feat / str(r) / "result.json") if (P.feat / str(r) / "result.json").is_file() else None)
            for r, _ in sorted(base)}


def f_summary(m, base, gs=None, grows=None, accept=()):
    """Phase F gate of m from the files (decision F-4 = spec 6-2 F; rates over the G .done reactions as D1LADDER F0-3):
    F-1 (no finite 182-feature row, B6-only rows included = g0f0 f0_report), B6 (status_B6 not ok / skipped), Q-a
    (Q_manifest qa_ok false), T-a (T_report bad) each > 1 % = STOP, plus the unchanged FU2 verdicts, mergeroot stop,
    inconsistent counts, the token layout and (GEO-11 = g0f0 f0_report) the B6 status predicted at G (results/
    G_report.json, block_b6 on the folders) equal to F's status_B6 on the rows whose B6 block ran. accept (GEO-7): the
    rate rules ('F-1', 'B6') a user decision accepted; they go to accepted_stops instead of stop_reasons."""
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
    grs = g_report_state(base, fp_now)                     # GEO-11: the G prediction of B6 (block_b6 on the folders)
    if not grs["complete"]:
        missing.append(grs["missing"])
    out = dict(n_base=len(ids), n_g_done=n_done, root=str(P.feat), versions=P.versions,
               G_report={k: grs.get(k) for k in GR_KEYS})
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
        # F-1 = g0f0 f0_report (D1PAIR REV9: all 182 columns finite) = every G .done reaction missing from X182_d1_geo,
        # B6-only failures included; B6 keeps its own 1 % gate on top (review 2026-10-30; LADDER's B6-only exclusion
        # was a user decision after its F0 STOP, here only through --accept=F-1, GEO-7)
        f1_fail = dict(feat_fail)
        out["features"] = dict(
            mergeroot_report=str(P.mrep), x182=dict(path=str(P.x182), sha256=C.sha256(P.x182), n=len(X1)),
            n_requested=mr.get("n_requested"), n_done=mr.get("n_done"), n_feat=mr.get("n_feat"),
            F1_all_finite=mr.get("F1_all_finite"), columns_equal_X182_d1=cols_equal, n_feature_fail=len(feat_fail),
            n_F1=len(f1_fail), n_B6_only=len(b6_only), n_F1_without_B6_only=len(set(f1_fail) - b6_only),
            F1_rule="every G .done reaction without a finite 182-feature row in X182_d1_geo (mergeroot not-ok: "
                    "xtb_status, ext geom / B1..B6, non-finite), B6-only rows included (= g0f0 f0_report, D1PAIR F-1); "
                    "n_F1_without_B6_only = information only (the LADDER F0_DECISION.md reading, accepted there "
                    "after the STOP; here only through --accept=F-1 after a user decision, GEO-7)",
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
            # GEO-11 = g0f0 f0_report: G prediction (block_b6 on the folders, G_report.json) vs F status_B6, on the rows
            # whose B6 block really ran at F (not geometry-skipped, not a lost record 'error:...'; those still count as
            # B6 failures above)
            g_b6 = set(grs["b6_pred"])
            f_ran = set(int(r) for r, s in st.status_B6.items() if str(s) != "skipped" and not str(s).startswith("error:"))
            b6_agree = (g_b6 & f_ran) == (set(b6_fail) & f_ran)
            out["features"]["B6_vs_G_report"] = dict(
                g_report=str(G_REPORT), n_g_predicted=len(g_b6), n_f_ran=len(f_ran), equal=bool(b6_agree),
                only_G=sorted((g_b6 - set(b6_fail)) & f_ran)[:100], only_F=sorted((set(b6_fail) - g_b6) & f_ran)[:100])
            if not b6_agree:
                stop.append(f"B6 status differs between G (block_b6 on the folders, {G_REPORT.name}: "
                            f"{len(g_b6 & f_ran)}) and F (status_B6: {len(set(b6_fail) & f_ran)}) on the {len(f_ran)} "
                            f"rows whose B6 block ran (GEO-11 = g0f0 f0_report)")
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
        if q.get("ext_used") is False:                     # the reason of the token layout STOP below
            stop.append(f"FU2 Q ext_used false: ext pass rate {q.get('ext_pass_rate')} < {q.get('ext_min_pass')} over "
                        f"{q.get('n_target')} reactions; the tokens then lack mu_abs / f_plus / f_minus and differ from "
                        f"the g1 layout of the trained models")
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
        comp = {"F-1": len(f1_fail), "B6": len(b6_fail), "Q-a": len(qa_fail), "T-a": len(t_bad)}
        rate = {k: (v / n_done if n_done else float("nan")) for k, v in comp.items()}
        over = [k for k, v in rate.items() if not v <= F_RATE_MAX]
        for k in over:
            msg = f"{m}: {k} failures {comp[k]}/{n_done} = {rate[k]:.4f} > {F_RATE_MAX} (spec 6-2 F)"
            (accepted if k in accept else stop).append(msg)
        n_fail = len(ids) - len(tok_ok & set(ids))
        out["gate"] = dict(
            n=comp, n_g_done=n_done, rate=rate, threshold=F_RATE_MAX, over=over, accept=list(accept),
            F1_without_B6_only=dict(n=len(set(f1_fail) - b6_only),
                                    rate=len(set(f1_fail) - b6_only) / n_done if n_done else None,
                                    rule="information only (LADDER F0_DECISION.md reading; not the gate, GEO-7)"),
            combined_info=dict(n=n_fail, n_base=len(ids), rate=n_fail / len(ids) if ids else None,
                               components=dict(g_step=len(ids) - n_done, features=len(feat_fail), b6=len(b6_fail),
                                               q_excluded=len(q_excl), q_a=len(qa_fail), t_bad=len(t_bad)),
                               rule="base reactions without a token npz (D1-G3 combined rule; information only)"),
            definition=DECISIONS["F-4 gate"])
    out.update(missing=missing, complete=not missing, stop=bool(stop), stop_reasons=stop, accepted_stops=accepted)
    return out


_FP = {}


def f_prior(rec, root_current=True):
    """F steps: the F job environment (test_t.env_f), the geo base (GEO-1), T (F-10), the G2 inputs, the G gate over the
    whole base (complete, no STOP; GEO-2), a current passing results/G_report.json (GEO-11: else exit 2 = run g_report,
    its STOP = exit 3) and, with root_current, the F-15 record = the current feature root (else exit 2: run f_slice).
    The scan is done once per process (f_rest runs several steps)."""
    G0F.env_f(rec)
    if not _FP:
        base = geo_base(rec)
        need_t(rec)
        G0F.g2_inputs_check(rec)
        gs, grows = g_scan(GEO, base)
        _FP.update(base=base, gs=gs, grows=grows, Z=rec.get("Z"), T=rec.get("T"), g2_inputs=rec.get("g2_inputs"),
                   GR=g_report_state(base))
    else:
        rec.update(Z=_FP["Z"], T=_FP["T"], g2_inputs=_FP["g2_inputs"])
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
    """F-15 + ids_all.csv of f_slice under the fcntl lock $GEO2/locks/F_<m>.lock (the f_slice array tasks start together):
    the first task after a feature-root change moves the F outputs aside and writes root_fp.json / ids_all.csv, later
    tasks find them current. A different ids_all.csv for an unchanged root = exit 3."""
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


def atoms_counts():
    a = paths(GEO).fu2_scr / "atoms"
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
                  + G0F.inputs_args(), C.P_DIR, feat_env())
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
        rc = G0F.child(rec, "atoms_gate", [C.P_DIR / "fu2" / "atoms.py", "root", "gate"], C.P_DIR, atoms_env_m(GEO),
                       ok=(0, 2, 3))
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
            raise Stop("Q", f"Q gate STOP (atoms.py gate, unchanged): {r.get('stop_reason')}", 3)
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


def hf_stale(P, jobs):
    """job_ids whose run folder inputs (runs/g2/<T>/<calc>.inp) carry other atoms / coordinates than the current
    geometry inputs (inputs/<T>/g2/<calc>.inp, written by geominputs from $GEO2/<m>_final): a geo_final folder changed
    after the HF run (run.py skips .done reactions). Coordinate lines only, so the SlowConv retry text is not a
    difference."""
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


def rel_gate(E, jobs):
    """GEO-5 (spec 6-1): E = {job_id: (e_frag1_rel_eh, e_frag2_rel_eh)} of geo ok reactions vs the D1-G2 HF labels
    (same rel1 / rel2 coordinates): |difference| <= 1e-6 Eh per reaction; STOP if > 0.5 % of the comparable reactions
    differ, or none is comparable while geo has ok rows."""
    import pandas as pd
    G = pd.read_parquet(G2_HF_LABELS)
    G = G[(G.geom == "g2") & (G.status == "ok")]
    if G.job_id.duplicated().any():
        raise Stop("rel_gate", f"{G2_HF_LABELS}: duplicate job_id rows", 3)
    ref = {str(j): (float(a), float(b)) for j, a, b in zip(G.job_id, G.e_frag1_rel_eh, G.e_frag2_rel_eh)}
    want = set(jobs)
    have = {str(j): e for j, e in E.items() if str(j) in want}
    comp = sorted(set(have) & set(ref))
    d = {}
    for j in comp:
        try:
            v = max(abs(float(have[j][0]) - ref[j][0]), abs(float(have[j][1]) - ref[j][1]))
        except (TypeError, ValueError):
            v = float("inf")
        d[j] = v if math.isfinite(v) else float("inf")
    mis = sorted(j for j, v in d.items() if not v <= H_REL_TOL_EH)
    rate = len(mis) / len(comp) if comp else float("nan")
    why = []
    if have and not comp:
        why.append(f"rel-energy gate: none of the {len(have)} geo ok reactions has a D1-G2 ok row")
    elif comp and rate > H_REL_MAX:
        why.append(f"rel-energy gate: {len(mis)}/{len(comp)} = {rate:.4f} reactions differ from D1-G2 by > "
                   f"{H_REL_TOL_EH:g} Eh (> {H_REL_MAX}; spec 6-1 / 6-2 H)")
    fin = [v for v in d.values() if math.isfinite(v)]
    return dict(rule=DECISIONS["GEO-5 rel gate"], reference=str(G2_HF_LABELS), reference_sha256=C.sha256(G2_HF_LABELS),
                tol_eh=H_REL_TOL_EH, threshold=H_REL_MAX, n_geo=len(have), n_comparable=len(comp),
                n_not_in_reference=len(set(have) - set(ref)), not_in_reference=sorted(set(have) - set(ref))[:50],
                n_mismatch=len(mis), rate=rate, max_abs_diff_eh=max(fin) if fin else None,
                mismatch={j: d[j] for j in mis[:100]}, stop=bool(why), stop_reasons=why)


def h_summary(m, base):
    """Phase H gate of m (decision F-4 / GEO-5 / GEO-6; status classes = D1-G2 phase_fh H_FIN / H_GATE / H_OPEN)."""
    import pandas as pd
    PF = pf()
    BAD = h_bad(PF)
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
    bad = S[S.status.isin(BAD)]
    gate = S[S.status.isin(PF.H_GATE)]
    inok = {T for T in jobs if gst.get(T) == "ok"}
    open_ = S[S.status.isin(PF.H_OPEN) & S.job_id.isin(inok)]
    absent = sorted(set(jobs) - set(S.job_id))
    input_fail = {T: gst.get(T, "missing") for T in jobs if gst.get(T) != "ok"}
    rate = len(bad) / len(fin) if len(fin) else float("nan")
    in_rate = len(input_fail) / len(jobs) if jobs else float("nan")
    jset = set(jobs)                                        # built once (8,600 reactions; feat.py rebuilt it per row)
    lab = [r for r in C.read_json(lj) if r.get("job_id") in jset]
    ch = coreh_jobs(P.hf_scr / "runs" / "g2", jobs)
    ok_jobs = sorted(S.job_id[S.status == "ok"])
    okset = set(ok_jobs)
    stale = hf_stale(P, ok_jobs)
    X = pd.read_parquet(pq)
    E = ({str(j): (a, b) for j, a, b in zip(X.job_id, X.e_frag1_rel_eh, X.e_frag2_rel_eh)}
         if len(X) and {"job_id", "e_frag1_rel_eh", "e_frag2_rel_eh"} <= set(X.columns) else {})
    rg = rel_gate(E, jobs)
    stop = []
    if stale:
        stop.append(f"{m}: {len(stale)} HF runs do not use the current $GEO2/{m}_final geometry {stale[:10]} "
                    f"(geo_final changed after H): move $GEO2/hf_{m}/runs/g2/<T> aside and rerun the H workers")
    if len(fin) and rate > H_RUN_MAX:
        stop.append(f"{m}: HF run failures {len(bad)}/{len(fin)} = {rate:.4f} > {H_RUN_MAX} (spec 6-2 H; GEO-6)")
    if jobs and in_rate > H_INPUT_MAX:
        stop.append(f"{m}: HF input check failed for {len(input_fail)}/{len(jobs)} = {in_rate:.4f} > {H_INPUT_MAX}")
    if len(lab) != by.get("ok", 0):
        stop.append(f"{m}: labels_d1_hf_g2.json rows {len(lab)} != H_status ok {by.get('ok', 0)}")
    stop += rg["stop_reasons"]
    out.update(by_status=by, n_inputs_ok=len(inok), n_finished=len(fin), n_ok=by.get("ok", 0), ok_jobs=ok_jobs,
               run_failure=dict(n=len(bad), n_finished=len(fin), rate=rate, threshold=H_RUN_MAX,
                                definition=DECISIONS["GEO-6 H run failure"],
                                by_status=bad.status.value_counts().to_dict(),
                                reasons=dict(Counter(bad.reason).most_common(40)), jobs=bad.job_id.tolist()[:300]),
               gate_failure=dict(n=len(gate), by_status=gate.status.value_counts().to_dict(),
                                 note="subset of run_failure (D1-G2 gate statuses)"),
               input_fail=dict(n=len(input_fail), rate=in_rate, threshold=H_INPUT_MAX, jobs=dict(list(input_fail.items())
                                                                                                [:300])),
               rel_gate=rg,
               open=dict(n=len(open_), by_status=open_.status.value_counts().to_dict(), jobs=open_.job_id.tolist()[:100]),
               absent=absent[:100], n_absent=len(absent), n_stale_runs=len(stale), stale_runs=stale[:100],
               core_h=dict(sum=float(sum(ch.values())), n_run_folders=len(ch), per_rxn=G0F.qstats(ch.values()),
                           ok_per_rxn=G0F.qstats(v for T, v in ch.items() if T in okset),
                           reference=D1G2_H_REF),
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
               stop=bool(fin and rate > H_RUN_MAX))
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


def cmd_h_inputs(W):
    P = paths(GEO)

    def body(rec):
        import pandas as pd
        env_hf(rec)
        base = geo_base(rec)
        need_t(rec)
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
        jobs = sorted(rid)
        lst = C.write_atomic(H_LIST, "job_id\n" + "".join(f"{T}\n" for T in jobs))
        env = env_h(GEO)
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        G0F.child(rec, "geominputs", [C.HF_DIR / "prep.py", "geominputs", "g2", lst], C.HF_DIR, env)
        gi_f = P.hf_res / "g2_inputs.csv"
        gi = pd.read_csv(gi_f, dtype=str, keep_default_na=False)
        if sorted(gi.job_id) != jobs:
            raise Stop("geominputs", f"{gi_f}: {len(gi)} rows != the {len(jobs)} base jobs", 3)
        st = dict(zip(gi.job_id, gi.status))
        ok = sorted(T for T in jobs if st[T] == "ok")
        input_fail = {T: st[T] for T in jobs if st[T] != "ok"}
        in_rate = len(input_fail) / len(jobs)
        rec["inputs"] = dict(Counter(st.values()))
        rec["input_fail"] = dict(n=len(input_fail), rate=in_rate, threshold=H_INPUT_MAX,
                                 jobs=dict(list(input_fail.items())[:200]))
        if in_rate > H_INPUT_MAX:
            raise Stop("geominputs", f"HF input check failed for {len(input_fail)}/{len(jobs)} = {in_rate:.4f} > "
                                     f"{H_INPUT_MAX} ({dict(Counter(input_fail.values()))})", 3)
        if not ok:
            raise Stop("geominputs", f"{GEO}: no reaction with HF inputs ({rec['inputs']})", 3)
        nb = W * math.ceil(len(ok) / (PF.BATCH_MAX * W))
        D = pd.DataFrame(dict(job_id=ok, n_atoms=[int(lab[rid[T]]["n_atoms"]) for T in ok]))
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
        pil = D[D.job_id.isin(ok[:PILOT_N])].sort_values("job_id")          # spec 6-1: first 50 by job_id
        C.write_atomic(H_BATCHES / PILOT_FILE, lambda t: pil.to_csv(t, index=False))
        plan = dict(W=W, nb=nb, n_base=len(jobs), n_ok=len(ok), sizes=sizes, max_size=max(sizes), files=files,
                    assignment={str(k): [f"H_b{j:02d}.csv" for j in range(nb) if j % W == k] for k in range(W)},
                    pilot=dict(file=PILOT_FILE, sha256=C.sha256(H_BATCHES / PILOT_FILE), n=int(len(pil)),
                               jobs=pil.job_id.tolist(), rule="the first 50 inputs-ok base reactions by job_id (spec 6-1)"),
                    ok_jobs=ok, h_list_sha256=C.sha256(lst), g2_inputs_sha256=C.sha256(gi_f),
                    g2_inputs_status=dict(Counter(gi.status).most_common()), base_sha256=base_sha(base),
                    n_input_fail=len(input_fail), input_fail=input_fail, written=now(),
                    rule=f"inputs-ok job_ids sorted by (n_atoms, job_id), dealt round robin into nb = W * ceil(n / "
                         f"({PF.BATCH_MAX} W)) batches (D1-G2 phase_fh / D1HF prep.batches rule), each file sorted by "
                         f"job_id; worker k of W owns the batches j with j % W == k (decision F-6)")
        write_json(H_PLAN, plan)                             # last: the plan exists only with all its files
        rec["plan"] = {k: plan[k] for k in ("W", "nb", "n_base", "n_ok", "sizes", "max_size", "g2_inputs_status",
                                            "n_input_fail")}
        rec["plan"]["pilot_n"] = plan["pilot"]["n"]
        return 0
    return step("h_inputs", body)


def cmd_h_pilot():
    P = paths(GEO)

    def body(rec):
        import pandas as pd
        PF = pf()
        env_hf(rec)
        base = geo_base(rec)
        need_t(rec)
        plan = need_plan(rec, base)
        pil = pd.read_csv(H_BATCHES / plan["pilot"]["file"], dtype={"job_id": str})
        jobs = sorted(pil.job_id)
        U = pil.sort_values(["n_atoms", "job_id"], ascending=[False, True])
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
        out = dict(spec="docs/specs/GEO2.md 6-1 pilot", written=now(), plan_sha256=C.sha256(H_PLAN),
                   rule=DECISIONS["GEO-4 pilot"], n_pilot=len(jobs), jobs=jobs, by_status=dict(Counter(st.values())),
                   n_finished=len(fin), n_open=len(open_), open=open_, run_failure=dict(n=len(bad), jobs=bad,
                   note="reported only at the pilot (GEO-4)"), run_exit=rc_run, rel_gate=rg,
                   core_h=dict(per_rxn=G0F.qstats(per.values()), sum=float(sum(per.values())), n=len(per),
                               wall_s=wall, cpus=G0F.cpus(), alloc_core_h=wall * G0F.cpus() / 3600.0))
        if open_:
            out.update(complete=False, stop=False, stop_reasons=[], **{"pass": False})
            write_json(H_PILOT, out)
            raise Stop("pilot", f"{len(open_)} pilot reactions not finished ({open_[:5]}; run exit {rc_run}): rerun h_pilot", 2)
        n_total = int(plan["n_ok"])
        mean = float(np.mean(list(per.values()))) if per else float("nan")
        proj = mean * n_total
        G = pd.read_parquet(G2_HF_LABELS)
        G = G[(G.geom == "g2") & (G.status == "ok")]
        g2c = {str(j): float(c) for j, c in zip(G.job_id, G.coreh_total) if np.isfinite(float(c))} \
            if "coreh_total" in G.columns else {}
        both = [T for T in per if T in g2c]
        den = sum(g2c[T] for T in both)
        ratio = sum(per[T] for T in both) / den if both and den > 0 else float("nan")
        g2_mean = float(np.mean(list(g2c.values()))) if g2c else float("nan")
        g2_total = float(sum(g2c.get(T, g2_mean) for T in plan["ok_jobs"]))
        proj_cal = ratio * g2_total
        used = proj                                        # spec 6-1: pilot mean core-h per reaction x the full set
        warn = []
        if math.isfinite(used) and used <= PILOT_COREH_MAX and math.isfinite(proj_cal) and proj_cal > PILOT_COREH_MAX:
            warn.append(f"D1-G2-calibrated projection {proj_cal:.0f} core-h > {PILOT_COREH_MAX:.0f} while the spec "
                        f"projection {proj:.0f} <= {PILOT_COREH_MAX:.0f}: information only, not a STOP (GEO-4; spec "
                        f"6-1 rule kept); H proceeds")
        out["projection"] = dict(
            n_total=n_total, mean_core_h_per_rxn=mean, spec_rule=proj,
            d1g2_calibrated=dict(value=proj_cal, ratio_geo_over_g2=ratio, n_pilot_with_g2=len(both),
                                 g2_core_h_of_the_ok_jobs=g2_total, g2_mean_fill=g2_mean,
                                 note="information only (GEO-4); a value > 2,500 alone is a warning, not a STOP"),
            alloc_info=(wall * G0F.cpus() / 3600.0) / len(jobs) * n_total if jobs else None,
            used=used, limit=PILOT_COREH_MAX, reference=D1G2_H_REF,
            rule="pilot mean core-h per reaction x n_ok (spec 6-1) > 2,500 core-h = STOP (GEO-4; no finite projection = "
                 "STOP: the pilot measured no finished reaction); d1g2_calibrated and alloc_info (allocated core-h per "
                 "pilot reaction x n_ok) are information")
        out["projection_core_h"] = used
        out["warnings"] = warn
        stop = []
        if not math.isfinite(used) or used > PILOT_COREH_MAX:
            stop.append(f"H core-h projection {used:.0f} > {PILOT_COREH_MAX:.0f} (spec 6-1 / 12 H: pilot mean "
                        f"{mean:.4f} core-h x {n_total}; D1-G2 calibrated {proj_cal:.0f}, information)")
        stop += rg["stop_reasons"]
        out.update(complete=True, stop=bool(stop), stop_reasons=stop, **{"pass": not stop})
        write_json(H_PILOT, out)
        for w in warn:                                     # review 2026-10-30: recorded in results/DECISIONS.md
            record_decision(f"GEO-4 pilot warning (plan {out['plan_sha256'][:12]})",
                            f"{w}. results/H_pilot.json projection: spec {proj:.1f}, calibrated {proj_cal:.1f}, "
                            f"limit {PILOT_COREH_MAX:.0f} ({now()}).")
        rec["pilot"] = {k: out[k] for k in ("by_status", "projection_core_h", "warnings", "stop", "stop_reasons")}
        rec["pilot"]["rel_gate"] = {k: rg[k] for k in ("n_comparable", "n_mismatch", "max_abs_diff_eh")}
        return 3 if stop else 0
    return step("h_pilot", body)


def cmd_h_batch(k, W):
    def body(rec):
        import pandas as pd
        env_hf(rec)
        base = geo_base(rec)
        need_t(rec)
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
        base = geo_base(rec)
        need_t(rec)
        need_plan(rec, base)
        env = env_h(GEO)
        rec["child_env"] = {k: env[k] for k in HF_KEYS}
        G0F.child(rec, "allgeom", [C.HF_DIR / "build.py", "allgeom", H_LIST, "g2"], C.HF_DIR, env)
        outs = (P.hf_scr / "labels_d1_hf_g2.json", P.hf_res / "labels_d1_hf_g2.parquet", P.hf_res / "H_status.csv")
        rec["outputs"] = {str(p): C.sha256(p) for p in outs}
        return 0
    return step("h_build", body)


def cmd_h_report():
    P = paths(GEO)

    def body(rec):
        env_hf(rec)
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
        rep = dict(spec="docs/specs/GEO2.md 6 H (STOP: 6-2 H, 6-1 rel gate / pilot)", written=now(),
                   thresholds=dict(run_failure=H_RUN_MAX, input_check=H_INPUT_MAX, rel_gate=H_REL_MAX,
                                   rel_tol_eh=H_REL_TOL_EH),
                   summary={k: v for k, v in s.items() if k != "ok_jobs"},
                   checkpoints=dict(n_batches=len(plan["files"]), missing=ck_missing, stale=ck_stale, cumulative=cum,
                                    batches=checks),
                   workers=workers, plan={k: plan.get(k) for k in ("W", "nb", "n_base", "n_ok", "sizes", "max_size")},
                   pilot=pilot_state(), child_env={k: env_h(GEO)[k] for k in HF_KEYS},
                   core_h=dict(timing=s.get("core_h"), alloc_steps=float(sum(
                       (G0F.read_json(p).get("core_h_alloc") or 0.0) for p in STEPS.glob("h_*.json"))),
                               reference=D1G2_H_REF),
                   complete=complete, stop=bool(stop), stop_reasons=stop)
        write_json(H_REPORT, rep)
        rec["report"] = {k: rep[k] for k in ("complete", "stop", "stop_reasons")}
        rec["report"].update(n_ok=s.get("n_ok"), rel_gate={k: s["rel_gate"][k] for k in ("n_comparable", "n_mismatch")}
                             if "rel_gate" in s else None)
        return 3 if stop else (0 if complete else 2)
    return step("h_report", body)


# ================================================================ P step (GEO2ENV=reactot)
def rung_cfg(m):
    """the inputs.RUNGS entry of method m (decision F-7; the 'g2' geometry slot of the D1-HF runner)."""
    P = paths(m)
    return dict(geom="g2", fu2_res=P.fu2_res, tokens=P.fu2_scr / "tokens", x182=P.x182,
                h_parquet=P.hf_res / "labels_d1_hf_g2.parquet", hf_runs=P.hf_scr / "runs", hf_inputs=P.hf_scr / "inputs",
                atom_version=P.versions["FU2_ATOM_VERSION"], token_version=P.versions["FU2_TOKEN_VERSION"], refs="min")


def cfg_str(cfg):
    """GEO-8: a RUNGS entry with every value as str (Path / str spellings of one path compare equal)."""
    return {str(k): str(v) for k, v in cfg.items()}


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
        fixed = m in IN.RUNGS                              # GEO-8: inputs.py may hold RUNGS['geo'] (spec 8-1)
        if fixed and cfg_str(IN.RUNGS[m]) != cfg_str(cfg):
            raise Stop("code", f"inputs.RUNGS[{m!r}] {cfg_str(IN.RUNGS[m])} != rung_cfg {cfg_str(cfg)} (GEO-8)", 3)
        IN.RUNGS[m] = cfg
        rep = dict(phase="P (D1LADDER Phase P rules)", spec="docs/specs/GEO2.md 6-1 P (STOP 6-2 P)", method=m,
                   started=now(), git_head=G0F.git_head(), rung_cfg={k: str(v) for k, v in cfg.items()},
                   rung_fixed_in_inputs_py=fixed,
                   decisions={k: DECISIONS[k] for k in ("F-7 P tables", "F-7b constrained keys", "GEO-8 P registration",
                                                        "GEO-7 accept")},
                   inputs_decisions=IN.DECISIONS, base=dict(n=len(base), z=rec.get("Z")),
                   accept=list(accept), accepted_stops=fs.get("accepted_stops"),
                   g_report=dict(path=str(G_REPORT), sha256=C.sha256(G_REPORT), accepted_stops=grs.get("accepted_stops")),
                   stop=False, all_pass=False)
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
                P_a_ok=(r.get("P-a") or {}).get("ok"), reassembly_ok=(r.get("reassembly") or {}).get("ok"),
                accepted_stops=r.get("accepted_stops"))


# ================================================================ p_report (spec 7; GEO2ENV=reactot)
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


def cmd_p_report():
    def body(rec):
        import pandas as pd
        env_reactot(rec)
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
        lp = G0F.read_json(IN.P_REPORT)
        if not lp.get("all_pass") or lp.get("running") or lp.get("stop"):
            raise Stop("ladder", f"{IN.P_REPORT}: all_pass {lp.get('all_pass')}, stop {lp.get('stop')}", 3)
        S = pd.read_csv(IN.S_FOLDS)
        if S.pair_id.duplicated().any():
            raise Stop("universe", f"{IN.S_FOLDS}: pair_id not unique", 3)
        S["pair_id"] = S.pair_id.astype(str)
        inputs = {str(IN.S_FOLDS): C.sha256(IN.S_FOLDS), str(IN.P_REPORT): C.sha256(IN.P_REPORT)}
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
        # end reasons for the g2 pairs without geo
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
            stop.append(f"|P_23| {n['P_23']} < {P_MIN_RATIO} x |P_g2| {n['P_g2']} (ratio {ratio:.4f}; spec 7)")
        if not ratio_m >= P_MIN_RATIO:
            stop.append(f"P_main reference pairs: |P_23| {nm['P_23']} < {P_MIN_RATIO} x |P_g2| {nm['P_g2']} (ratio "
                        f"{ratio_m:.4f}; spec 7, eval universe)")
        files = {}
        sets_dir = paths(GEO).table.parent / "sets"
        for name in ("P_geo", "P_23", "P_023"):            # LADDER set-file format (inputs.SET_COLS) for the loaders
            df = S.loc[sets[name], IN.SET_COLS]
            f = C.write_atomic(sets_dir / f"{name}.csv", lambda t, df=df: df.to_csv(t, index=False))
            files[str(f)] = C.sha256(f)
        sj = C.write_json(P_SETS, {k: lists[k] for k in SET_NAMES})
        rep = dict(spec="docs/specs/GEO2.md 7", written=now(), git_head=G0F.git_head(), rule=DECISIONS["GEO-9 pair sets"],
                   universe=dict(path=str(IN.S_FOLDS), n=int(len(S))), inputs=inputs, ladder_check=lad_check,
                   n=n, n_P_main_reference=nm,
                   sha256={k: list_sha(lists[k]) for k in SET_NAMES},
                   sha256_rule="sha256 of '\\n'.join(sorted pair ids), no trailing newline",
                   sets={k: set_summary(S, sets[k]) for k in SET_NAMES},
                   check=dict(rule=f"|P_23| >= {P_MIN_RATIO} |P_g2| (all pairs and P_main reference pairs)",
                              ratio=ratio, ratio_P_main_reference=ratio_m),
                   missing_from_geo=dict(n=len(rows), n_P_main_reference=sum(1 for r in rows
                                                                            if r["in_P_main"] and r["ref_kind"] == "reference"),
                                         by_reason=dict(Counter(r["reason"] for r in rows).most_common(60)),
                                         end_reason_classes=dict(cls.most_common(60)), pairs=rows[:3000]),
                   usable=dict(geo=len(U["geo"]), g2=len(U["g2"]), dft=len(U["dft"])),
                   outputs=dict({str(sj): C.sha256(sj)}, **files), stop=bool(stop), stop_reasons=stop)
        write_json(P_SETS_REPORT, rep)
        rec["report"] = dict(n=n, n_P_main_reference=nm, ratio=ratio, ratio_P_main_reference=ratio_m, stop=bool(stop),
                             stop_reasons=stop)
        return 3 if stop else 0
    return step("p_report", body)


# ================================================================ T-2
def t_check(k, out, body):
    """T-<k>: results/T_<k>.json marked 'running', body(rec), then the verdict (pass / not_ready / fail)."""
    record_all()
    rec = dict(test=f"T-{k}", spec="docs/specs/GEO2.md 5", started=now(), host=socket.gethostname(),
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
    """child of t3 / h_pilot (D1HF_* overrides in its environment): parse_rxn(T, 'g2') of the listed reactions -> <out>."""
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
        base, zinfo = geo_state()                          # GEO-1
        for c, w in zinfo["problems"]:
            (stop if c == 3 else incomplete).append(f"Phase Z: {w}")
        rep = dict(spec="docs/specs/GEO2.md 6 (STOP: 6-2), 7", written=now(), git_head=G0F.git_head(),
                   decisions=DECISIONS, thresholds=dict(G_order=ORDER_MAX, G_freq=FREQ_MAX, F_rate=F_RATE_MAX,
                                                       B6=B6_MAX, H_run=H_RUN_MAX, H_input=H_INPUT_MAX, H_rel=H_REL_MAX,
                                                       H_rel_tol_eh=H_REL_TOL_EH, pilot_core_h=PILOT_COREH_MAX,
                                                       P_ratio=P_MIN_RATIO),
                   accept=list(accept), T=ts, Z={k: v for k, v in zinfo.items() if k != "problems"},
                   n_base=len(base), methods={})
        grows = []
        if base:
            gs, grows = g_scan(GEO, base)
            grs = g_report_state(base)                     # GEO-11
            if grs["complete"] and grs.get("gate_json"):   # F_G_status.csv: the g_report verdicts of each .done folder
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
                        pairs=dict(present=bool(pr), stop=pr.get("stop"), stop_reasons=pr.get("stop_reasons"),
                                   n=pr.get("n"), n_P_main_reference=pr.get("n_P_main_reference"),
                                   sha256=pr.get("sha256"), check=pr.get("check")),
                        core_h=dict(g_step=gs["core_h"], hf=hs.get("core_h"), g_alloc=alloc("g_"), f_alloc=alloc("f_"),
                                    h_alloc=alloc("h_"), p_alloc=alloc("p_")))
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
    if cmd == "t2_g" and a[1:] in ([], ["--rerun"]):
        return cmd_t2_g(rerun=n == 2)
    if cmd == "t2_f" and n == 1:
        return cmd_t2_f()
    if cmd == "t3" and a[1:] in ([], ["--rerun"]):
        return cmd_t3(rerun=n == 2)
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
             "p_report": cmd_p_report}
    if cmd in plain and n == 1:
        return plain[cmd]()
    if cmd == "_hfparse" and n == 3:
        return _hfparse(a[1], a[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
