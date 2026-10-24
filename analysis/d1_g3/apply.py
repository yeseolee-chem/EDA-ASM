#!/usr/bin/env python3
r"""apply.py — D1-G3 Phase T test T-4 and the Phase E E-3 support predictions (docs/specs/D1G3.md 3 T-4, 6 E-3, 8, 12):
the trained D1LADDER atom-token unit models (AW / GNN / TF; $LAD/units/L__<rung>__<arm>__<model>__f<k>__s0..s4.pt)
applied to new inputs. No training; no model, label or D1LADDER file is changed (spec 0, 8). sbatch only; nothing in
this file was run when it was written (login node), so every path is untested until it runs inside a job.

CLI (env_d1fu2 = analysis/d1_ladder/jobs/gpu.slurm on a CPU partition: no GPU is needed or used, every tensor is a
float32 CPU tensor; submit from analysis/d1_g3; exit 0 = done / up to date, 2 = not ready / usage, 3 = STOP)
  python apply.py t4 [--force]
        T-4 (spec 3): model families dft·S, dft·SH, g2·S, g2·SH x AW, GNN, TF x T1 folds 0..4 (60 units): each unit's 5
        final models (architecture = the unit json chosen cell = the .pt cell, variant A+G, its own saved scaling
        statistics; SH: + the unit beta and residual SD) predict 20 test pairs of that fold from the ORIGINAL inputs
        ($LAD/inputs/<rung>.parquet + the rung tokens, the D1LADDER label file); seed mean = the final prediction ->
        must equal results/L_predictions.parquet yhat within 1e-3 kcal/mol (y 1e-12, base 1e-9, beta 1e-12 relative),
        the unit fingerprint (all keys but 'gpu', = train_atoms collect) must equal the current inputs / code, and the
        SH units' beta must equal E2_report line_betas (1e-9). XGB units have no model file: excluded from E and
        reported (not a STOP). -> results/T_4.json (always written; exit 3 = T-4 failed = spec 12 T STOP). Up to date
        (same fingerprint, all_pass) -> exit 0 without recomputing unless --force.
  python apply.py e3 [--force]
        E-3 support (spec 6): every Part A pair of results/S_pairs.csv predicted by its own T1 fold's units (fold =
        S_pairs fold = the T1 fold whose test list holds the pair, checked) of the model families dft·SH, dft·S
        (primary), g2·SH, g2·S (reference), fed the inputs of every structure level dft, g1, g2 ($LAD/inputs/<rung>.parquet
        + the D1LADDER rung tokens) and g3h, g3r ($G3/inputs/<m>.parquet + the feat.py tokens, below) = 'model trained
        on rung R, fed level X'. Rows: AW, GNN, TF (every atom unit), REP (per model rung x arm x target the E2_report
        representative; an XGB representative is replaced by the best atom model of the same rule, A-4), LINE (beta of
        the fold's SH unit x dH of the level, under both arms, A-6). Needs results/T_4.json all_pass with the same
        code and .pt sha256, results/S_pairs.csv (sha256 = S_report.json outputs) and the g3h / g3r inputs (exit 2
        until they exist). Self-check: input level = model rung reproduces L_predictions (1e-3) on the Part A pairs,
        LINE = the SH units' baseline (1e-9); else STOP. -> $G3/apply/predictions.parquet, $G3/apply/apply_report.json
        (up to date -> exit 0 unless --force).
        Needs results/O_report.json complete (exit 2) and not stop (exit 3). U-1 (user 2026-10-25; decisions U-2e-1,
        U-2e-2): when O_report.json has a 'u1' block, the pair list is the complete_pairs of results/U1_selection.json
        (ts.g3r ok at both ends; rows = S_pairs.csv, else STOP) and the input levels are dft, g1, g2, g3r: g3h is
        'stopped by user (U-1)' (no table / tokens / P_build read, no rows; apply_report levels_stopped), never an
        error. The selection sha256 must equal O_report u1.selection_sha256 and P_build_g3r.json u1.selection.sha256
        (else exit 2: rerun select / assemble / feat.py p); apply_report.json gets a 'u1' block (explanatory). Without
        a 'u1' block: the spec as written (every S_pairs pair, five levels).

Inputs of the levels g3h / g3r (feat.py contract, A-7): table $G3/inputs/<m>.parquet (inputs.py make_table layout: rxn_id,
  tok_path, tok_sha256, usable, the 182 X182 columns, H_<t>, ...), token reports $D1G3_FU2_RES (default
  analysis/d1_g3/results/fu2_<m>: T_report.json, T_manifest.csv, Q_report.json, S_splits.json = FU2 copy) and tokens
  $D1G3_FU2_TOKENS (default $G3/fu2_<m>/tokens/<rxn_id>.npz); both env values may contain '{m}'. Versions
  fu2-tokens-3-<m> / fu2-atoms-3-<m> (spec 5). train_atoms.load_tokens / load_inputs check them exactly as for the
  D1LADDER rungs (T_report / Q_report stop, versions, sha256 links, npz sha256 = T_manifest, 182 columns = token glob,
  tok_sha256, usable); a table tok_path outside the token root is a STOP (set the two env values). The defaults are
  feat.py's paths() (FU2_SCR = $G3/fu2_<m>, FU2_RES = results/fu2_<m>); e3 also needs feat.py's results/P_build_<m>.json
  all_pass (else exit 2) with outputs sha256 = the table and rung_cfg fu2_res / tokens = these roots (else STOP).

Outputs
  results/T_4.json                 T-4 record: per unit max |dyhat|, |dy|, |dbase|, beta, fingerprint keys that differ,
                                   .pt checks; the 20-pair draws; XGB statement and the XGB representatives with their
                                   replacements; LINE betas; pt_sha256 (E-3 uses exactly these files); fingerprint.
  $G3/apply/predictions.parquet    pair_id, fold, class, target, model_rung, arm, model, input_level, y, yhat, base,
                                   beta (+ rep_source, yhat_seed_sd, dipole, core); y = dd_<target> of the D1LADDER label
                                   file ($LAD/F_pairs_cr2.csv); S: base = beta = 0; SH: base = beta * dH(level);
                                   LINE: yhat = base = beta * dH(level), yhat_seed_sd NaN; 10 targets (TARGETS order).
  $G3/apply/apply_report.json      E-3 record: per level usable / missing pairs (with the reason), common pairs, layout
                                   checks, representatives, LINE betas, self-checks, fingerprint, outputs sha256;
                                   levels (used), levels_stopped (U-1: g3h 'stopped by user (U-1)'), u1 (U-2e-1).
  results/DECISIONS.md             the A-* decisions (g3_common.record_decision under the shared fcntl lock
                                   $G3/locks/DECISIONS.lock; written when t4 / e3 start).

sbatch (login node: one-shot commands only; first `squeue -u $USER -h -r | wc -l` (N + 1 <= 20) and
`sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`; logs go to $G3/logs, never to the D1LADDER log folder):
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_g3
  (Phase T, after the Phase S job JS; parallel to the other T jobs)
  JT4=$(sbatch --parsable -p cpu1,cpu2 -J g3_T4 --time=48:00:00 --cpus-per-task=4 --mem=16G \
      --output=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs/%x_%j.out --dependency=afterok:$JS --kill-on-invalid-dep=yes \
      ../d1_ladder/jobs/gpu.slurm apply.py t4)
  (after the feat.py 'p g3h ;;; p g3r ;;; report' job JPF = g3h / g3r tables + tokens; under U-1 the feat.py
   'p g3r ;;; report' job: the same line. t4 runs first in the same job (U-2e-2): A-12 hashes apply.py AND
   g3_common.py, so T-4 is re-checked against the code e3 runs with (up to date -> exit 0 at once, else ~70 s), and
   there is no afterok on an earlier T-4 job that slurmctld may have purged)
  JE3=$(sbatch --parsable -p cpu1,cpu2 -J g3_E3 --time=48:00:00 --cpus-per-task=4 --mem=24G \
      --output=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs/%x_%j.out --dependency=afterok:$JPF --kill-on-invalid-dep=yes \
      ../d1_ladder/jobs/gpu.slurm apply.py t4 ::: apply.py e3)
  (feat.py roots elsewhere: add --export=ALL,D1G3_FU2_RES=<dir with {m}>,D1G3_FU2_TOKENS=<dir with {m}>)

Decisions (user: every judgment = the recommended option): DECISIONS below, also in results/DECISIONS.md and the reports.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_ladder, d1_pair trees (spec 8)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g3_common as C  # noqa: E402   (puts analysis/d1_ladder on sys.path: lad_common, train_atoms)
import train_atoms as TA  # noqa: E402   D1LADDER M1 / M2: rung_base / LadData / final_sh / fingerprint (torch + FU2 code)

if Path(TA.__file__).resolve() != (C.LAD_DIR / "train_atoms.py").resolve():
    C.stop(f"train_atoms imported from {TA.__file__}, not {C.LAD_DIR / 'train_atoms.py'}")
FT, F, Mo, torch = TA.FT, TA.F, TA.Mo, TA.torch          # fu2 train.py, fu2_common, models.py, torch

# ---------------------------------------------------------------- constants
CODE_VERSION = "d1-g3-apply-1"
MODEL_RUNGS = ("dft", "g2")                               # spec 3 T-4 / 6 E-3 model families
ARMS = tuple(TA.ARMS)                                     # S, SH
FAMILIES = (("dft", "SH"), ("dft", "S"), ("g2", "SH"), ("g2", "S"))   # E-3: dft primary, g2 reference
ATOM = ("AW", "GNN", "TF")                                # models with model files (.pt)
XGB = "XGB"
REP_ORDER = ("AW", "GNN", "TF", "XGB")                    # spec 13-1 tie order (E2_report rep_rule)
LEVELS = ("dft", "g1", "g2") + tuple(C.METHODS)           # E-3 input structure levels
TARGETS = list(TA.TARGETS)                                # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
NF = TA.NF
N_SEEDS = len(F.FINAL_SEEDS)
N_T4 = 20                                                 # spec 3 T-4: 20 test pairs of the fold
TOL = dict(yhat=1e-3, y=1e-12, base=1e-9, beta_rel=1e-12, line_rel=1e-9, mean=1e-12, std=1e-12)
UNITS = TA.UNITS
L_PRED = C.LAD_RES / "L_predictions.parquet"
E2_REPORT = C.LAD_RES / "E2_report.json"
LABELS = TA.DEFAULT_LABELS                                # $LAD/F_pairs_cr2.csv (the y of L_predictions)
S_PAIRS, S_REPORT = C.RES / "S_pairs.csv", C.RES / "S_report.json"
T4_JSON = C.RES / "T_4.json"
OUT_DIR = C.G3 / "apply"
PRED_OUT = OUT_DIR / "predictions.parquet"
APPLY_REPORT = OUT_DIR / "apply_report.json"
G3_INPUTS = C.G3 / "inputs"
O_REPORT, U1_SEL = C.RES / "O_report.json", C.RES / "U1_selection.json"   # opt.py assemble / select (U-2e-1)
U1_METHOD = "g3r"                                         # U-1: the selection is made from the g3r TS units only
U1_TEXT = "stopped by user (U-1)"
G3_FU2_RES = os.environ.get("D1G3_FU2_RES") or str(C.RES / "fu2_{m}")
G3_TOKENS = os.environ.get("D1G3_FU2_TOKENS") or str(C.G3 / "fu2_{m}" / "tokens")
OUT_COLS = ["pair_id", "fold", "class", "target", "model_rung", "arm", "model", "input_level", "y", "yhat", "base",
            "beta", "rep_source", "yhat_seed_sd", "dipole", "core"]
KEY_COLS = ["pair_id", "model_rung", "arm", "model", "input_level", "target"]
CPU = torch.device("cpu")

DECISIONS = {
    "A-1": "T-4 fold·짝: 명령서 'fold k'를 5개 fold 모두(k = 0..4)로 읽는다(E-3가 다섯 fold의 모델 파일을 모두 쓰므로 모두 "
           "검사). rung·fold마다 L_predictions의 그 fold 시험 짝(= LadData.split(k) test, 확인) pair_id 정렬 목록에서 "
           "numpy.random.default_rng(20261024) 하나를 rung(dft, g2) × fold 순서로 써 rng.choice(n, 20, replace=False)로 "
           "20짝을 뽑는다. 같은 20짝을 S·SH × AW·GNN·TF에 쓴다.",
    "A-2": "장치: env_d1fu2(torch 2.2.1+cu118, sys.prefix 확인)를 analysis/d1_ladder/jobs/gpu.slurm으로 CPU 분할(cpu1, cpu2)"
           "에서 돈다. GPU는 쓰지 않는다. 텐서는 학습과 같은 float32, CPU, torch 스레드 수 = SLURM_CPUS_PER_TASK. "
           "GPU 학습 대 CPU 재계산 차이는 명령서 T-4 허용 오차 1e-3 kcal/mol로 판정한다.",
    "A-3": "T-4 통과 = 모든 unit에서 |yhat - L yhat| <= 1e-3, |y - L y| <= 1e-12, |base - L base| <= 1e-9, beta 상대 "
           "<= 1e-12, L 행 누락 0, unit fingerprint(gpu 제외, train_atoms collect 규칙) = 현재 입력·코드, .pt 5개의 unit·"
           "cell·variant·std(sy, SH: sr, beta)가 unit json과 같음, state_dict가 json 구조에 strict로 맞음, SH unit beta = "
           "E2_report line_betas(1e-9). 하나라도 어긋나면 T-4 실패(STOP, T_4.json은 기록).",
    "A-4": "XGB: unit에 json·parquet만 있고 모델 파일이 없다 -> 새 입력에 적용할 수 없어 E에서 빼고 T_4.json에 보고한다"
           "(명령서 T-4, STOP 아님). 대표가 XGB인 rung·arm·타깃은 같은 규칙(명령서 13-1: unit json val_kcal_mae의 5 fold "
           "평균 최저, 동률 AW -> GNN -> TF)으로 AW·GNN·TF 중 최선을 쓰고(REP 행의 rep_source), 목록을 기록한다. "
           "대표와 평균은 unit json에서 다시 계산해 E2_report representatives와 같아야 한다(1e-12, 아니면 STOP).",
    "A-5": "모델 복원: 구조 = unit json chosen.cell(= .pt cell), variant A+G, d_in = 모델 rung의 원소·bin·num 열 수, "
           "d_glob = 182(S) / 191(SH), dropout 0.1, models.build + load_state_dict(strict); 표준화는 .pt에 저장된 그 unit의 "
           "mu, sd, gmu, gsd(SH: + H 9열)를 그대로 쓴다(다시 맞추지 않음); 최종 예측 = 5 seed 평균(S: predict x sy, SH: "
           "train_atoms.final_sh = predict x sr + beta x dH), yhat_seed_sd = seed SD(ddof 0) = train_atoms run_unit.",
    "A-6": "LINE: beta = 그 모델 rung × fold의 SH unit json 'beta'(AW·GNN·TF SH unit에서 같음 확인, = E2_report "
           "line_betas 1e-9 = evaluate.line_beta의 outer-train T_main 값), 예측 = beta x dH(입력 수준) (= evaluate."
           "line_predict). LINE은 arm과 무관해 S·SH 두 arm 아래 같은 값으로 둔다(D1LADDER E1-4 선례).",
    "A-7": "입력 수준 등록: 이 프로세스 안에서만 train_atoms.RUNGS에 g3h·g3r(fu2_res, tokens, tag = 방법)를 더하고 "
           "train_atoms.inputs_path를 g3h·g3r에 대해 $G3/inputs/<m>.parquet로 돌려준다(D1LADDER 파일은 그대로). 토큰 루트 "
           "기본값 = analysis/d1_g3/results/fu2_<m>(T_report 등), $G3/fu2_<m>/tokens(npz); feat.py가 다른 곳에 쓰면 env "
           "D1G3_FU2_RES / D1G3_FU2_TOKENS('{m}' 포함)로 준다(= feat.py paths(): FU2_SCR $G3/fu2_<m>, FU2_RES "
           "results/fu2_<m>). E-3는 feat.py results/P_build_<m>.json all_pass(없거나 미통과 = exit 2), 그 outputs의 표 "
           "sha256 = 현재 표, rung_cfg fu2_res·tokens = 이 루트일 때만 돈다(아니면 STOP). 표의 tok_path가 토큰 루트 밖이면 "
           "STOP.",
    "A-8": "분포 이동 적용: 수준 X의 입력(토큰 원자 열, 182 전역, SH의 H 9)을 모델 rung R unit의 저장된 표준화 값으로 바꾼다"
           "(= 'R에서 학습한 모델에 X 입력'). 원소 one-hot은 R의 원소 순서로 다시 놓는다(R에 없는 원소의 원자 -> STOP). "
           "num_cols, bin_cols, glob_cols, ext_used가 R과 다르면 적용할 수 없으므로 STOP.",
    "A-9": "빠진 짝: 수준 X에서 a 또는 b가 usable이 아니면(토큰, 표 행, 유한 H 9·182; train_atoms MA-5) 그 짝·수준의 행을 "
           "만들지 않고 apply_report.json에 이유와 함께 적는다. 표마다 같은 짝 목록을 쓰는 판단은 E 단계(명령서 12 E)가 한다; "
           "다섯 수준 공통 짝 목록도 적는다.",
    "A-10": "출력 행: 모델 열 = AW, GNN, TF(모든 atom unit), REP(대표, rep_source = 원래 모델), LINE. 명령서 열 + rep_source, "
            "yhat_seed_sd, dipole, core(E-3 core 단위 bootstrap용, S_pairs.csv에서). 10 타깃(c_ghost 포함; E는 9 타깃).",
    "A-11": "E-3 자기 검사: 입력 수준 = 모델 rung인 atom 행은 L_predictions(같은 unit, 같은 짝)와 1e-3 kcal/mol, y 1e-12, "
            "base 1e-9 안에서 같아야 하고, LINE은 SH unit base와 1e-9 안에서 같아야 한다; 아니면 STOP. 각 짝은 그 fold의 "
            "모델 rung 시험 목록(LadData.split(fold).test)에 있어야 한다(누출 없음; 아니면 STOP).",
    "A-12": "E-3 실행 조건: results/T_4.json all_pass, 그 fingerprint의 코드 sha256(apply.py, g3_common.py, train_atoms "
            "CODE_SHA) = 현재(다르면 exit 2: t4 다시), .pt sha256 = T_4.json pt_sha256, 대표 선택 = T_4.json. 보고 파일 "
            "$G3/apply/apply_report.json을 predictions.parquet 옆에 둔다(누락 짝·검사 기록; 명령서에 없는 부속 출력).",
    "U-2e-1": "U-1 범위(사용자 결정 2026-10-25 1·5: ts.g3h·ts.g3r 중단, g3r 완료 반응으로 완성되는 P_012 para 짝만 F·E, 설명용; "
              "apply.py e3): results/O_report.json이 있어야 한다(없음·미완 = exit 2, stop = exit 3). 그 'u1' 블록이 있으면 "
              "U-1이다: 짝 목록 = results/U1_selection.json complete_pairs(행의 pair_id, a_rxn, b_rxn, fold, class가 "
              "S_pairs.csv와 같아야 하고 중복·빈 목록이면 STOP), 입력 수준 = dft, g1, g2와 선택 파일의 method(= g3r만; "
              "다르면 STOP). g3h는 표·토큰·P_build를 읽지 않고 예측 행을 만들지 않으며 apply_report.json levels_stopped에 "
              "'stopped by user (U-1)'로 적는다(오류·미완이 아님). 선택 파일 sha256 = O_report u1.selection_sha256 = "
              "P_build_g3r.json u1.selection.sha256이어야 한다(다르거나 P_build에 U-1 기록이 없으면 exit 2: select·assemble·"
              "feat.py p 다시; 'u1' 블록이 있는데 선택 파일이 없으면 STOP, 블록 없이 선택 파일만 있으면 exit 2). 지문에 선택 "
              "sha256과 범위를 넣어 선택이 바뀌면 다시 계산한다. apply_report.json에 u1 블록(결정, 선택 sha256, 짝 수·분류별, "
              "범위 안·밖 방법, explanatory = true)을 둔다. 'u1' 블록이 없으면 명령서 그대로(S_pairs 전체, 다섯 수준).",
    "U-2e-2": "작은 짝 목록(U-1 완성 짝 약 10개, fold 몇 개, core 몇 개): 짝이 없는 fold는 건너뛴다(그 fold의 unit 모델은 "
              "읽지 않음, A-12의 .pt sha256 검사는 그대로 전체), 수준 X에서 usable 짝이 0인 fold·수준은 행 없이 apply_report "
              "units n_pairs 0으로 남긴다. 범위 안 g3 수준에서 usable 짝이 하나도 없어도 STOP하지 않고 levels_info n_usable 0과 "
              "이유를 적는다(E 단계가 그 수준의 비교 목록을 0짝으로 보고한다). 자기 검사(A-11)는 대각선 행(dft·g2 입력 = "
              "표본 전부 usable)이 1개 이상이면 된다. bootstrap·CI는 여기서 계산하지 않는다(g3_eval.py U-2e-8). 실행 순서: "
              "U-1 코드 수정으로 code_sha(apply.py, g3_common.py)가 바뀌었으므로 같은 작업에서 't4 ::: e3'로 돌린다(T-4가 e3와 "
              "같은 코드로 다시 확인됨, 최신이면 t4는 바로 exit 0; 이미 끝나 slurmctld에서 지워진 T-4 작업에 afterok를 걸지 "
              "않음).",
}


# ---------------------------------------------------------------- small helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def uname(r, a, m, k):
    return f"L__{r}__{a}__{m}__f{k}"


def maxabs(a, b):
    """max |a - b| (inf on a shape / NaN-pattern difference; train_atoms._maxdiff)."""
    return TA._maxdiff(a, b)


def maxrel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float((np.abs(a - b) / np.maximum(1.0, np.abs(b))).max()) if a.size else 0.0


def norm(o):
    return TA.norm(o)


def code_sha():
    return dict(apply_py=C.sha256(Path(__file__).resolve()), g3_common_py=C.sha256(Path(C.__file__).resolve()),
                train_atoms=TA.CODE_SHA)


def env_check():
    """A-2: env_d1fu2 (pinned torch, prefix), CPU only; torch threads = SLURM_CPUS_PER_TASK."""
    bad = []
    if torch.__version__ != FT.TORCH_PIN:
        bad.append(f"torch {torch.__version__} != {FT.TORCH_PIN}")
    if Path(sys.prefix).resolve() != FT.ENV_PREFIX.resolve():
        bad.append(f"python prefix {sys.prefix} is not env_d1fu2 {FT.ENV_PREFIX}")
    if bad:
        C.stop("environment (run through analysis/d1_ladder/jobs/gpu.slurm): " + "; ".join(bad))
    n = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    torch.set_num_threads(max(1, int(n) if n.isdigit() else 1))
    return dict(host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), python=platform.python_version(),
                prefix=sys.prefix, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, device="cpu",
                dtype="float32", threads=torch.get_num_threads(), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_available=bool(torch.cuda.is_available()), versions_fingerprint=FT.versions())


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


def record_decisions(keys):
    for k in keys:
        record_decision(k, DECISIONS[k])


# ---------------------------------------------------------------- input levels g3h / g3r (A-7)
_ORIG_INPUTS_PATH = TA.inputs_path


def g3_paths(m):
    return dict(fu2_res=Path(G3_FU2_RES.format(m=m)), tokens=Path(G3_TOKENS.format(m=m)),
                table=G3_INPUTS / f"{m}.parquet")


def _inputs_path(rung):
    return g3_paths(rung)["table"] if rung in C.METHODS else _ORIG_INPUTS_PATH(rung)


def register_g3():
    """A-7: g3h / g3r as train_atoms rungs in this process only (versions fu2-*-3-<m> via the tag)."""
    for m in C.METHODS:
        p = g3_paths(m)
        TA.RUNGS[m] = dict(fu2_res=p["fu2_res"], tokens=p["tokens"], tag=m)
    TA.inputs_path = _inputs_path
    return {m: {k: str(v) for k, v in g3_paths(m).items()} for m in C.METHODS}


def level_files(X):
    res = Path(TA.RUNGS[X]["fu2_res"])
    return {"table": TA.inputs_path(X), **{n: res / n for n in ("T_report.json", "T_manifest.csv", "Q_report.json",
                                                                 "S_splits.json")}}


def g3_missing(methods=C.METHODS):
    miss = []
    for m in methods:
        p = g3_paths(m)
        miss += [str(f) for f in level_files(m).values() if not Path(f).is_file()]
        if not p["tokens"].is_dir():
            miss.append(str(p["tokens"]))
    return miss


def g3_pbuild(m, u1=None):
    """A-7: feat.py's results/P_build_<m>.json passed for the current table, with the same token roots (rung_cfg);
    U-2e-1: under U-1 its 'u1' block names the current U1_selection.json (sha256)."""
    p, pb = g3_paths(m), C.RES / f"P_build_{m}.json"
    if not pb.is_file():
        not_ready(f"{pb} missing (feat.py Phase F table of {m})")
    r = C.read_json(pb)
    if not r.get("all_pass") or r.get("stop"):
        not_ready(f"{pb}: all_pass {r.get('all_pass')}, stop {r.get('stop')} ({r.get('stop_reason')})")
    if u1 and u1["active"]:
        pu = r.get("u1") or {}
        got = (pu.get("selection") or {}).get("sha256")
        if not pu.get("active") or got != u1["selection"]["sha256"]:
            not_ready(f"{pb.name}: built without the current U-1 selection (u1 active {pu.get('active')}, selection "
                      f"sha256 {str(got)[:16]} vs {u1['selection']['sha256'][:16]}): rerun feat.py p {m} (U-2e-1)")
    if (r.get("outputs") or {}).get(str(p["table"])) != C.sha256(p["table"]):
        C.stop(f"{p['table']} sha256 != {pb.name} outputs (table changed after its build)")
    cfg = r.get("rung_cfg") or {}
    diff = {k: (cfg.get(k), str(p[k])) for k in ("fu2_res", "tokens")
            if os.path.abspath(str(cfg.get(k))) != os.path.abspath(str(p[k]))}
    if diff:
        C.stop(f"{pb.name} rung_cfg roots differ from apply.py's {diff} (set D1G3_FU2_RES / D1G3_FU2_TOKENS, A-7)")
    return dict(file=str(pb), sha256=C.sha256(pb), n_usable=len(r.get("usable") or []), rung_cfg=cfg, u1=r.get("u1"))


def u1_state(SP):
    """U-2e-1: the U-1 scope from results/O_report.json ('u1' block, opt.py assemble) and results/U1_selection.json
    (opt.py select) -> dict(active, methods, levels, stopped, pairs, selection, fingerprint, record). Without a 'u1'
    block the spec as written (every S_pairs pair; both methods)."""
    if not O_REPORT.is_file():
        not_ready(f"{O_REPORT} missing (opt.py assemble)")
    o = C.read_json(O_REPORT)
    if o.get("stop"):
        C.stop(f"{O_REPORT}: Phase O ended in a STOP ({o.get('stop_reasons')}; spec 12 O)")
    if not o.get("complete"):
        not_ready(f"{O_REPORT}: Phase O not complete ({o.get('n_incomplete')} open; opt.py assemble)")
    u = o.get("u1")
    if not u:
        if U1_SEL.is_file():
            not_ready(f"{U1_SEL} exists but {O_REPORT.name} has no 'u1' block (rerun opt.py assemble, U-1)")
        return dict(active=False, methods=list(C.METHODS), levels=list(LEVELS), stopped={}, pairs=None, selection=None,
                    fingerprint=dict(active=False), record=dict(active=False, rule="spec as written (no U-1 block in "
                                                                                    "O_report.json)"))
    if not U1_SEL.is_file():
        C.stop(f"{O_REPORT.name} has a 'u1' block but {U1_SEL} is missing (U-1 contract)")
    sha = C.sha256(U1_SEL)
    if u.get("selection_sha256") != sha:
        not_ready(f"{U1_SEL.name} sha256 {sha[:16]} != O_report u1.selection_sha256 "
                  f"{str(u.get('selection_sha256'))[:16]} (selection changed after assemble: rerun opt.py select / "
                  f"assemble, feat.py, U-2e-1)")
    s = C.read_json(U1_SEL)
    if s.get("method") != U1_METHOD:
        C.stop(f"{U1_SEL.name}: method {s.get('method')!r} != {U1_METHOD} (U-1: g3r only)")
    cp = s.get("complete_pairs") or []
    ids = [str(p.get("pair_id")) for p in cp]
    if not cp or len(set(ids)) != len(ids):
        C.stop(f"{U1_SEL.name}: {len(cp)} complete pairs, {len(ids) - len(set(ids))} duplicated pair ids (U-1a)")
    bad = []
    for p in cp:
        pid = str(p["pair_id"])
        if pid not in SP.index:
            bad.append(f"{pid}: not in S_pairs.csv")
            continue
        r = SP.loc[pid]
        mine = (int(r.a_rxn), int(r.b_rxn), int(r.fold), str(r["class"]))
        theirs = (int(p["a_rxn"]), int(p["b_rxn"]), int(p["fold"]), str(p["class"]))
        if mine != theirs:
            bad.append(f"{pid}: S_pairs {mine} vs selection {theirs}")
    if bad:
        C.stop(f"{U1_SEL.name}: complete_pairs differ from {S_PAIRS} (pair_id, a_rxn, b_rxn, fold, class): {bad[:4]}")
    ms = [U1_METHOD]
    stopped = {m: U1_TEXT for m in C.METHODS if m not in ms}
    sel = dict(path=str(U1_SEL), sha256=sha, written=s.get("written"), n_complete_pairs=len(cp),
               n_by_class=s.get("n_by_class"), n_selected_rxns=len(s.get("selected_rxns") or []))
    rec = dict(active=True, decision=DECISIONS["U-2e-1"], explanatory=True,
               label="설명용 (U-1: g3r 완성 짝 부분집합, 사용자 결정 2026-10-25 5)", selection=sel,
               methods_in_scope=ms, methods_stopped=stopped, levels=list(LEVELS[:3]) + ms,
               o_report_u1={k: u.get(k) for k in ("decision", "selection_sha256", "n_complete_pairs", "n_by_class",
                                                  "n_selected_rxns", "n_selected_done", "n_pairs_both_done",
                                                  "pairs_both_done_by_class", "stopped_user")})
    return dict(active=True, methods=ms, levels=rec["levels"], stopped=stopped, pairs=sorted(ids), selection=sel,
                fingerprint=dict(active=True, selection_sha256=sha, methods=ms), record=rec)


def tok_path_check(m):
    """A-7: the g3 table's tok_path (inputs.make_table: str(tokens / '<rxn_id>.npz')) lies under the token root."""
    p = g3_paths(m)
    T = pd.read_parquet(p["table"])
    if "tok_path" not in T.columns or "rxn_id" not in T.columns:
        return "tok_path / rxn_id column absent: not checked"
    has = T.has_tok.astype(bool) if "has_tok" in T.columns else T.tok_path.fillna("").astype(str).str.len() > 0
    want = [os.path.abspath(p["tokens"] / f"{int(r)}.npz") for r in T.rxn_id[has]]
    got = [os.path.abspath(str(x)) for x in T.tok_path[has]]
    bad = [(w, g) for w, g in zip(want, got) if w != g]
    if bad:
        C.stop(f"{p['table']}: tok_path of {len(bad)} rows is not <tokens root>/<rxn_id>.npz, e.g. {bad[0]} (feat.py "
               f"roots differ: set D1G3_FU2_RES / D1G3_FU2_TOKENS, A-7)")
    return f"{len(want)} token rows: tok_path = {p['tokens']}/<rxn_id>.npz"


# ---------------------------------------------------------------- unit records
_UNIT: dict = {}


def read_unit(name):
    """(json record, sha256) of $LAD/units/<name>.json (cached)."""
    if name not in _UNIT:
        p = UNITS / f"{name}.json"
        if not p.is_file():
            C.stop(f"{p} missing (D1LADDER unit cache, spec 1 0-1)")
        _UNIT[name] = (json.loads(p.read_text()), C.sha256(p))
    return _UNIT[name]


def check_units_vs_e2(E2, names):
    eu = E2.get("units") or {}
    bad = [x for x in names if (eu.get(x) or {}).get("json_sha256") != read_unit(x)[1]]
    if bad:
        C.stop(f"unit json sha256 != E2_report.json units for {len(bad)} units, e.g. {bad[:4]}")


def pt_names(rungs=MODEL_RUNGS):
    return [f"{uname(r, a, m, k)}__s{s - F.SEED}.pt" for r in rungs for a in ARMS for m in ATOM for k in range(NF)
            for s in F.FINAL_SEEDS]


def pt_sha():
    miss = [f for f in pt_names() if not (UNITS / f).is_file()]
    if miss:
        C.stop(f"{len(miss)} unit model files missing under {UNITS}, e.g. {miss[:3]}")
    return {f: C.sha256(UNITS / f) for f in pt_names()}


_SPLIT: dict = {}


def split_of(D, k):
    key = (D.rung, D.arm, int(k))
    if key not in _SPLIT:
        _SPLIT[key] = D.split(int(k))
    return _SPLIT[key]


def unit_gate(rec, D, u):
    """unit json state (= train_atoms collect / evaluate val_table rules) and fingerprint vs the current files."""
    probs = []
    ju = rec.get("unit") or {}
    if dict(rung=ju.get("rung"), arm=ju.get("arm"), model=ju.get("model"), fold=ju.get("fold")) != u:
        probs.append(f"json unit {ju} != {u}")
    if rec.get("smoke"):
        probs.append("smoke unit")
    if rec.get("variant") != TA.VARIANT:
        probs.append(f"variant {rec.get('variant')} != {TA.VARIANT}")
    if not (all(t["gate_M1"]["ok"] for t in rec.get("tune") or []) and all(f["gate_M1"]["ok"] for f in rec.get("final")
                                                                             or [])):
        probs.append("gate M-1' not ok in the unit json")
    if [int(f.get("seed")) for f in rec.get("final") or []] != list(F.FINAL_SEEDS):
        probs.append(f"final seeds {[f.get('seed') for f in rec.get('final') or []]} != {F.FINAL_SEEDS}")
    if u["arm"] == "SH" and len(rec.get("beta") or []) != len(TARGETS):
        probs.append("SH unit without a 10-target beta")
    now_fp = TA.fingerprint(D, u, split_of(D, u["fold"]))
    old = rec.get("fingerprint") or {}
    diff = sorted(k for k in set(old) | set(now_fp) if k != "gpu" and old.get(k) != now_fp.get(k))
    return probs, diff


class UnitStd(FT.Std):
    """fu2 train.Std carrying a saved unit's own scaling statistics (A-5: no refit); tensors() is inherited."""

    def __init__(self, st, arm):                                        # noqa: super().__init__ would refit
        self.mu, self.sd, self.gmu, self.gsd, self.sy = (np.asarray(st[k], dtype=np.float64)
                                                         for k in ("mu", "sd", "gmu", "gsd", "sy"))
        self.sr = np.asarray(st["sr"], dtype=np.float64) if arm == "SH" else None
        self.beta = np.asarray(st["beta"], dtype=np.float64) if arm == "SH" else None


def load_models(name, rec, arm, DR):
    """A-5: the 5 final models of an atom unit on CPU, each with its saved std; problems vs the unit json listed."""
    u = rec["unit"]
    cell = (rec.get("chosen") or {}).get("cell")
    d_in = len(DR.elements) + len(DR.bin_cols) + len(DR.num_cols)
    d_glob = int(DR.d_glob)
    beta = np.asarray(rec["beta"], dtype=float) if arm == "SH" else None
    out, probs = [], []
    for j, s in enumerate(F.FINAL_SEEDS):
        pt = UNITS / f"{name}__s{s - F.SEED}.pt"
        if not pt.is_file():
            C.stop(f"{pt} missing")
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        ou = obj.get("unit") or {}
        if {k: ou.get(k) for k in u} != u:
            probs.append(f"{pt.name}: unit {ou} != json {u}")
        for k, want in (("cell", cell), ("rung", u["rung"]), ("arm", u["arm"]), ("variant", rec.get("variant"))):
            if obj.get(k) != want:
                probs.append(f"{pt.name}: {k} {obj.get(k)} != json {want}")
        st = obj.get("std") or {}
        need = ("mu", "sd", "gmu", "gsd", "sy") + (("sr", "beta") if arm == "SH" else ())
        if any(k not in st for k in need):
            probs.append(f"{pt.name}: std lacks {[k for k in need if k not in st]}")
            continue
        std = UnitStd(st, arm)
        fin = rec["final"][j]
        if std.mu.shape != (len(DR.num_cols),) or std.gmu.shape != (d_glob,) or std.sy.shape != (len(TARGETS),):
            probs.append(f"{pt.name}: std shapes mu {std.mu.shape} gmu {std.gmu.shape} sy {std.sy.shape} vs num "
                         f"{len(DR.num_cols)} glob {d_glob}")
        if maxrel(std.sy, fin["sy"]) > TOL["std"]:
            probs.append(f"{pt.name}: sy != json final sy")
        if arm == "SH":
            if maxrel(std.sr, fin["sr"]) > TOL["std"]:
                probs.append(f"{pt.name}: sr != json final sr")
            if maxrel(std.beta, beta) > TOL["beta_rel"] or maxrel(fin["beta"], beta) > TOL["beta_rel"]:
                probs.append(f"{pt.name}: beta != json beta")
        try:
            model = Mo.build(u["model"], obj.get("cell"), d_in, len(TARGETS), d_glob, obj.get("variant"), F.DROPOUT)
            model.load_state_dict(obj["state"])                         # strict: every parameter name and shape
        except (RuntimeError, KeyError, TypeError) as e:
            probs.append(f"{pt.name}: state_dict does not fit the json architecture: {str(e)[:300]}")
            continue
        out.append(dict(seed=int(s), file=pt.name, model=model.to(CPU).eval(), std=std))
    if len(out) != N_SEEDS:
        probs.append(f"{name}: {len(out)} usable model files of {N_SEEDS}")
    return out, probs


# ---------------------------------------------------------------- views, prediction
def remap_onehot(oh, mask, src, dst):
    """A-8: element one-hot from the level's element order `src` to the model rung's `dst`."""
    out = np.zeros(oh.shape[:2] + (len(dst),), dtype=oh.dtype)
    unk = []
    for j, z in enumerate(src):
        if z in dst:
            out[..., dst.index(z)] = oh[..., j]
        elif oh[..., j].any():
            unk.append(int(z))
    if unk:
        C.stop(f"elements {unk} of the input level are unknown to the model rung (elements {dst}): cannot apply (A-8)")
    if not np.array_equal(out.sum(-1), mask.astype(out.dtype)):
        C.stop("one-hot remap: not exactly one element per real atom")
    return out


class View(FT.Data):
    """the fu2 train.Data arrays of selected token rows of a LadData (any level), one-hot in the model rung's element
    order; static() is inherited (fu2 train.Data.__init__ is not called)."""

    def __init__(self, D, rows, elements):
        rows = np.asarray(rows, dtype=int)
        self.src_rows = rows
        self.elements = [int(z) for z in elements]
        self.onehot = remap_onehot(np.asarray(D.onehot)[rows], np.asarray(D.mask)[rows], [int(z) for z in D.elements],
                                   self.elements)
        for k in ("bin", "num", "pos", "mask", "etype", "frag", "n", "glob", "H10"):
            setattr(self, k, np.asarray(getattr(D, k))[rows])
        self.num_cols, self.bin_cols = list(D.num_cols), list(D.bin_cols)
        self.d_in = len(self.elements) + len(self.bin_cols) + len(self.num_cols)
        self.d_glob = int(self.glob.shape[1])


def layout_check(DR, DX):
    """A-8: token layout of level X = the model rung's (else STOP); elements reported."""
    bad = [k for k in ("num_cols", "bin_cols") if list(getattr(DX, k)) != list(getattr(DR, k))]
    if list(DX.trep.get("glob_cols") or []) != list(DR.trep.get("glob_cols") or []):
        bad.append("glob_cols")
    if bool(DX.trep.get("ext_used")) != bool(DR.trep.get("ext_used")):
        bad.append("ext_used")
    if int(DX.d_glob) != int(DR.d_glob):
        bad.append(f"d_glob {DX.d_glob} != {DR.d_glob}")
    if bad:
        C.stop(f"level {DX.rung} token layout differs from model rung {DR.rung} ({DX.arm}): {bad} (cannot apply, A-8)")
    return dict(elements_model=[int(z) for z in DR.elements], elements_level=[int(z) for z in DX.elements],
                extra_elements=sorted(set(int(z) for z in DX.elements) - set(int(z) for z in DR.elements)),
                same_elements=[int(z) for z in DX.elements] == [int(z) for z in DR.elements])


def why_not(D, r):
    out = []
    for end, rr, rid in (("a", D.ra[r], D.P.a_rxn.iat[r]), ("b", D.rb[r], D.P.b_rxn.iat[r])):
        if rr < 0:
            out.append(f"{end} {int(rid)}: no token")
        elif not D.usable_rxn[rr]:
            out.append(f"{end} {int(rid)}: not usable (no table row / non-finite H or 182)")
    return "; ".join(out) or "not usable"


def pair_index(D, pids):
    """(usable pair ids, their pair-table rows, {pair id: reason} of the others) at the level of D (A-9)."""
    ok, prow, miss = [], [], {}
    for p in pids:
        r = D.pid_row.get(p)
        if r is None:
            C.stop(f"pair {p} is not in the label file {D.labels}")
        if D.usable_pair[r]:
            ok.append(p)
            prow.append(r)
        else:
            miss[p] = why_not(D, r)
    return ok, np.asarray(prow, dtype=int), miss


@torch.no_grad()
def predict_view(arm, models, V, ia, ib):
    """A-5: seed mean / SD of the final kcal prediction (S: predict x sy; SH: train_atoms.final_sh)."""
    Y = []
    for s in models:
        T = s["std"].tensors(V, dtype=torch.float32, device=CPU)
        if arm == "S":
            Y.append(FT.predict(s["model"], T, ia, ib) * s["std"].sy)
        else:
            Y.append(TA.final_sh(s["model"], T, V, s["std"], ia, ib))
        del T
    Y = np.stack(Y)
    return Y.mean(0), Y.std(0)


def apply_unit(arm, models, beta, DX, prow, elements):
    """one unit's 5 models on pair-table rows prow of level DX -> yhat, sd, base, beta, y, dH [n, 10]."""
    ra, rb = DX.ra[prow], DX.rb[prow]
    U, inv = np.unique(np.r_[ra, rb], return_inverse=True)
    V = View(DX, U, elements)
    ia, ib = inv[:len(prow)], inv[len(prow):]
    yhat, sd = predict_view(arm, models, V, ia, ib)
    dH = V.H10[ia] - V.H10[ib]
    if not np.array_equal(dH, DX.dH[prow]):
        C.stop("internal: view dH differs from the data dH")
    b = np.asarray(beta, dtype=float) if arm == "SH" else np.zeros(len(TARGETS))
    base = b[None, :] * dH if arm == "SH" else np.zeros_like(yhat)
    if not (np.isfinite(yhat).all() and np.isfinite(base).all()):
        C.stop(f"non-finite prediction ({DX.rung} level, arm {arm})")
    return dict(yhat=yhat, sd=sd, base=base, beta=b, y=DX.y[prow], dH=dH)


# ---------------------------------------------------------------- representatives, XGB, LINE betas
def rep_choice(E2):
    """A-4: spec 13-1 representatives recomputed from the unit json val_kcal_mae (= E2_report, else STOP); XGB ->
    the best atom model by the same rule."""
    reps = E2.get("representatives") or {}
    if "AW -> GNN -> TF -> XGB" not in str(E2.get("rep_rule") or ""):
        C.stop(f"E2_report rep_rule {E2.get('rep_rule')!r} does not state the tie order AW -> GNN -> TF -> XGB")
    out, bad = {}, []
    for R in MODEL_RUNGS:
        for a in ARMS:
            for t in TARGETS:
                mm = {m: float(np.mean(np.asarray([float(read_unit(uname(R, a, m, k))[0]["val_kcal_mae"][t])
                                                   for k in range(NF)], dtype=float))) for m in REP_ORDER}
                mine = sorted(REP_ORDER, key=lambda q: (mm[q], REP_ORDER.index(q)))[0]
                e = ((reps.get(R) or {}).get(a) or {}).get(t) or {}
                em = e.get("mean_val_kcal_mae") or {}
                dmax = max(abs(mm[q] - float(em[q])) if q in em and em[q] is not None else float("inf") for q in REP_ORDER)
                if e.get("model") != mine or not dmax <= TOL["mean"]:
                    bad.append(f"{R}|{a}|{t}: E2 {e.get('model')} vs recomputed {mine} (max |mean diff| {dmax})")
                use = mine if mine != XGB else sorted(ATOM, key=lambda q: (mm[q], ATOM.index(q)))[0]
                out[f"{R}|{a}|{t}"] = dict(e2_model=e.get("model"), model=use, xgb_replaced=mine == XGB,
                                           mean_val_kcal_mae=mm)
    if bad:
        C.stop("E2_report representatives differ from the unit json validation MAE: " + "; ".join(bad[:5]))
    return out


def xgb_statement(reps):
    listing = sorted(os.listdir(UNITS))
    files = {}
    for R in MODEL_RUNGS:
        for a in ARMS:
            for k in range(NF):
                x = uname(R, a, XGB, k)
                files[x] = [f for f in listing if f.startswith(x + "__") or
                            (f.startswith(x + ".") and f not in (x + ".json", x + ".parquet"))]
    has = any(files.values())
    return dict(model_files=files, has_model_files=has, excluded_from_E=True,
                statement=("XGB units: <unit>.json + <unit>.parquet only (train_xgb.py saves no booster): XGB cannot be "
                           "applied to new inputs; excluded from E and reported (spec T-4; not a STOP)") if not has else
                ("XGB files besides json / parquet found (listed); apply.py has no XGB loader: XGB stays excluded from "
                 "E (spec T-4), reported"),
                representatives_xgb={key: dict(replaced_by=v["model"], mean_val_kcal_mae=v["mean_val_kcal_mae"])
                                     for key, v in reps.items() if v["xgb_replaced"]})


def line_betas(E2):
    """A-6: per model rung x fold the SH unit beta (equal for AW / GNN / TF) and its E2_report line_betas check."""
    lb, out, bad = E2.get("line_betas") or {}, {}, []
    for R in MODEL_RUNGS:
        for k in range(NF):
            bs = {m: np.asarray(read_unit(uname(R, "SH", m, k))[0]["beta"], dtype=float) for m in ATOM}
            e2 = (lb.get(R) or {}).get(str(k)) or {}
            e2v = np.asarray([float(e2[t]) if t in e2 else np.nan for t in TARGETS], dtype=float)
            d_models = max(maxrel(bs[m], bs["AW"]) for m in ATOM)
            d_e2 = maxrel(bs["AW"], e2v)
            if d_models > TOL["beta_rel"] or d_e2 > TOL["line_rel"]:
                bad.append(f"{R} f{k}: SH betas between models {d_models}, vs E2 line_betas {d_e2}")
            out[f"{R}|{k}"] = dict(beta=dict(zip(TARGETS, bs["AW"].tolist())), max_rel_between_models=d_models,
                                   max_rel_vs_e2_line_betas=d_e2)
    return out, bad


# ---------------------------------------------------------------- T-4
def compare_L(Lu, pids, res):
    """our unit prediction vs its L_predictions rows (same pairs, 10 targets)."""
    idx = pd.MultiIndex.from_product([list(pids), TARGETS], names=["pair_id", "target"])
    q = Lu.set_index(["pair_id", "target"]).reindex(idx)
    shp = (len(pids), len(TARGETS))

    def g(c):
        return q[c].to_numpy(dtype=float).reshape(shp)
    dy = np.abs(res["yhat"] - g("yhat"))
    out = dict(n_pairs=len(pids), n_L_rows_missing=int(q.yhat.isna().sum()),
               yhat_max_abs=maxabs(res["yhat"], g("yhat")), y_max_abs=maxabs(res["y"], g("y")),
               base_max_abs=maxabs(res["base"], g("base")),
               beta_max_rel=maxrel(np.broadcast_to(res["beta"][None, :], shp), g("beta")),
               seed_sd_max_abs=maxabs(res["sd"], g("yhat_seed_sd")),
               yhat_max_abs_by_target={t: (float(dy[:, j].max()) if np.isfinite(dy[:, j]).all() else float("inf"))
                                       for j, t in enumerate(TARGETS)})
    out["ok"] = bool(out["n_L_rows_missing"] == 0 and out["yhat_max_abs"] <= TOL["yhat"] and out["y_max_abs"] <= TOL["y"]
                     and out["base_max_abs"] <= TOL["base"] and out["beta_max_rel"] <= TOL["beta_rel"])
    return out


def read_lpred(pids=None):
    LP = pd.read_parquet(L_PRED, columns=TA.PRED_COLS)
    LP = LP[LP.rung.isin(list(MODEL_RUNGS)) & LP.model.isin(list(ATOM))].copy()
    LP["pair_id"] = LP.pair_id.astype(str)
    LP["fold"] = LP.fold.astype(int)
    if pids is not None:
        LP = LP[LP.pair_id.isin(set(pids))]
    return LP.reset_index(drop=True)


def base_fingerprint(E2_sha, lp_sha, levels):
    names = [uname(r, a, m, k) for r in MODEL_RUNGS for a in ARMS for m in REP_ORDER for k in range(NF)]
    return dict(code=CODE_VERSION, code_sha256=code_sha(), e2_report=E2_sha, l_predictions=lp_sha,
                labels=C.sha256(LABELS), unit_json={x: read_unit(x)[1] for x in names}, pt=pt_sha(),
                levels={X: {k: C.sha256(p) for k, p in level_files(X).items()} for X in levels})


def cmd_t4(force=False):
    t0 = time.time()
    record_decisions(("A-1", "A-2", "A-3", "A-4", "A-5", "A-6"))  # spec 11: recorded before they are acted on
    env = env_check()
    for p in (L_PRED, E2_REPORT, LABELS):
        if not p.is_file():
            C.stop(f"{p} missing (spec 1 0-1)")
    E2 = C.read_json(E2_REPORT)
    lp_sha = C.sha256(L_PRED)
    if lp_sha != (E2.get("l_report") or {}).get("predictions_sha256"):
        C.stop(f"{L_PRED} sha256 != E2_report l_report.predictions_sha256")
    check_units_vs_e2(E2, [uname(r, a, m, k) for r in MODEL_RUNGS for a in ARMS for m in REP_ORDER for k in range(NF)])
    fp = norm(base_fingerprint(C.sha256(E2_REPORT), lp_sha, MODEL_RUNGS))
    if not force and T4_JSON.is_file():
        old = C.read_json(T4_JSON)
        if old.get("all_pass") and old.get("fingerprint") == fp:
            print(f"[T-4] up to date ({T4_JSON}, all_pass)", flush=True)
            return 0
    reps = rep_choice(E2)
    xgb = xgb_statement(reps)
    lines, line_bad = line_betas(E2)
    LP = read_lpred()
    rng = np.random.default_rng(C.SEED_S)
    units, picks = {}, {}
    for R in MODEL_RUNGS:
        D = {a: TA.get_data(R, a, LABELS) for a in ARMS}
        for k in range(NF):
            g = LP[(LP.rung == R) & (LP.fold == k)]
            te = sorted(set(g.pair_id))
            te_split = sorted(TA._ids(D["S"], split_of(D["S"], k)["test"]))
            if te != te_split:
                C.stop(f"{R} fold {k}: L_predictions test pairs ({len(te)}) != LadData.split test ({len(te_split)})")
            pos = [int(x) for x in rng.choice(len(te), size=min(N_T4, len(te)), replace=False)]
            pick = sorted(te[j] for j in pos)
            picks[f"{R}|f{k}"] = dict(n_test=len(te), positions=pos, pairs=pick)
            for a in ARMS:
                Da = D[a]
                ok, prow, miss = pair_index(Da, pick)
                if miss:
                    C.stop(f"internal: {R} {a} fold {k} test pairs not usable: {list(miss.items())[:3]}")
                for m in ATOM:
                    name, u = uname(R, a, m, k), dict(rung=R, arm=a, model=m, fold=k)
                    t1 = time.time()
                    rec, jsha = read_unit(name)
                    probs, fdiff = unit_gate(rec, Da, u)
                    models, mprobs = load_models(name, rec, a, Da)
                    probs += mprobs
                    e = dict(json_sha256=jsha, chosen=rec.get("chosen"), problems=probs, fingerprint_diff_keys=fdiff,
                             pt=[s["file"] for s in models])
                    if len(models) == N_SEEDS:
                        res = apply_unit(a, models, rec.get("beta"), Da, prow, Da.elements)
                        Lu = LP[(LP.rung == R) & (LP.arm == a) & (LP.model == m) & (LP.fold == k)
                                & LP.pair_id.isin(set(ok))]
                        e.update(compare_L(Lu, ok, res))
                    else:
                        e["ok"] = False
                    e["ok"] = bool(e["ok"] and not probs and not fdiff)
                    e["seconds"] = time.time() - t1
                    units[name] = e
                    print(f"[T-4] {name}: ok {e['ok']} max|dyhat| {e.get('yhat_max_abs')} "
                          f"fp-diff {fdiff} problems {probs[:2]} ({e['seconds']:.1f} s)", flush=True)
                    del models
        del D
        TA._DATA.clear()
        TA._BASE.clear()
    worst = max(units, key=lambda x: (units[x].get("yhat_max_abs") if np.isfinite(units[x].get("yhat_max_abs", np.inf))
                                      else np.inf))
    all_pass = bool(all(e["ok"] for e in units.values()) and not line_bad and len(units) == len(MODEL_RUNGS) * len(ARMS)
                    * len(ATOM) * NF)
    rep = dict(
        test="T-4", spec="docs/specs/D1G3.md 3 T-4", created=now(), env=env, all_pass=all_pass, stop=not all_pass,
        rule=("dft·S, dft·SH, g2·S, g2·SH x AW, GNN, TF x T1 folds 0..4: the unit's 5 final .pt models (json chosen cell, "
              "variant A+G, the .pt std; SH: + unit beta / sr) predict 20 test pairs of the fold from the original "
              "inputs (LadData of the rung, labels F_pairs_cr2.csv); seed mean vs L_predictions.parquet yhat"),
        tol=TOL, n_units=len(units), n_units_ok=int(sum(e["ok"] for e in units.values())),
        max_abs_yhat=float(units[worst].get("yhat_max_abs", np.inf)), worst_unit=worst,
        failed_units=sorted(x for x, e in units.items() if not e["ok"]), units=units, picks=picks,
        line_betas=lines, line_betas_problems=line_bad, xgb=xgb, representatives=reps,
        pt_sha256=fp["pt"], fingerprint=fp,
        inputs=dict(l_predictions=str(L_PRED), e2_report=str(E2_REPORT), labels=str(LABELS), units=str(UNITS),
                    tables={R: str(TA.inputs_path(R)) for R in MODEL_RUNGS}),
        decisions={k: DECISIONS[k] for k in ("A-1", "A-2", "A-3", "A-4", "A-5", "A-6")}, wall_s=time.time() - t0)
    C.write_json(T4_JSON, rep)
    print(f"[T-4] all_pass {all_pass}: {rep['n_units_ok']} / {rep['n_units']} units, max |dyhat| "
          f"{rep['max_abs_yhat']:.3g} kcal/mol ({worst}); XGB: {xgb['statement']}; XGB representatives replaced: "
          f"{ {k: v['replaced_by'] for k, v in xgb['representatives_xgb'].items()} }", flush=True)
    if not all_pass:
        print(f"STOP: T-4 failed (spec 12 T): units {rep['failed_units'][:6]}, LINE {line_bad[:3]}", file=sys.stderr,
              flush=True)
        return 3
    return 0


# ---------------------------------------------------------------- E-3
def read_s_pairs():
    if not (S_PAIRS.is_file() and S_REPORT.is_file()):
        not_ready(f"{S_PAIRS} / {S_REPORT} missing (Phase S: sample.py sample)")
    srep = C.read_json(S_REPORT)
    if (srep.get("outputs") or {}).get(str(S_PAIRS)) != C.sha256(S_PAIRS):
        C.stop(f"{S_PAIRS} sha256 != S_report.json outputs")
    SP = pd.read_csv(S_PAIRS, dtype={"pair_id": str, "class": str, "dipole": str, "core": str})
    need = ["pair_id", "a_rxn", "b_rxn", "fold", "class"]
    if any(c not in SP.columns for c in need):
        C.stop(f"{S_PAIRS}: columns {[c for c in need if c not in SP.columns]} missing")
    for c in ("dipole", "core"):
        if c not in SP.columns:
            SP[c] = ""
    SP = SP.assign(a_rxn=SP.a_rxn.astype(int), b_rxn=SP.b_rxn.astype(int), fold=SP.fold.astype(int))
    if SP.pair_id.duplicated().any() or not SP.fold.between(0, NF - 1).all():
        C.stop(f"{S_PAIRS}: duplicated pair_id or fold outside 0..{NF - 1}")
    return SP.set_index("pair_id", drop=False).rename_axis(None), srep


def block(pids, SP, k, R, a, X, model, res, rep_source):
    n = len(pids)
    meta = SP.loc[list(pids)]
    parts = []
    for j, t in enumerate(TARGETS):
        parts.append(pd.DataFrame({
            "pair_id": list(pids), "fold": np.full(n, int(k)), "class": meta["class"].to_numpy(), "target": t,
            "model_rung": R, "arm": a, "model": model, "input_level": X, "y": res["y"][:, j], "yhat": res["yhat"][:, j],
            "base": res["base"][:, j], "beta": float(res["beta"][j]), "rep_source": rep_source,
            "yhat_seed_sd": res["sd"][:, j], "dipole": meta["dipole"].to_numpy(), "core": meta["core"].to_numpy()}))
    return pd.concat(parts, ignore_index=True)[OUT_COLS]


def self_check(A, LP):
    """A-11: input level = model rung -> L_predictions (atom rows) and the SH baseline (LINE rows)."""
    key = ["pair_id", "model_rung", "arm", "model", "target"]
    Lk = LP.rename(columns={"rung": "model_rung"}).set_index(key)[["y", "yhat", "base", "beta"]]
    S = A[A.model.isin(list(ATOM)) & (A.input_level == A.model_rung)].set_index(key)
    j = S.join(Lk, rsuffix="_L", how="left")
    atoms = dict(n_rows=int(len(j)), n_L_missing=int(j.yhat_L.isna().sum()),
                 yhat_max_abs=maxabs(j.yhat, j.yhat_L), y_max_abs=maxabs(j.y, j.y_L), base_max_abs=maxabs(j.base, j.base_L),
                 beta_max_rel=maxrel(j.beta, j.beta_L))
    atoms["ok"] = bool(atoms["n_rows"] > 0 and atoms["n_L_missing"] == 0 and atoms["yhat_max_abs"] <= TOL["yhat"]
                       and atoms["y_max_abs"] <= TOL["y"] and atoms["base_max_abs"] <= TOL["base"]
                       and atoms["beta_max_rel"] <= TOL["beta_rel"])
    Ls = LP[(LP.arm == "SH") & (LP.model == "AW")].rename(columns={"rung": "model_rung"}).set_index(
        ["pair_id", "model_rung", "target"])[["base", "beta"]]
    Q = A[(A.model == "LINE") & (A.arm == "SH") & (A.input_level == A.model_rung)].set_index(
        ["pair_id", "model_rung", "target"]).join(Ls, rsuffix="_L", how="left")
    line = dict(n_rows=int(len(Q)), n_L_missing=int(Q.base_L.isna().sum()), yhat_vs_sh_base_max_abs=maxabs(Q.yhat, Q.base_L),
                beta_max_rel=maxrel(Q.beta, Q.beta_L))
    line["ok"] = bool(line["n_rows"] > 0 and line["n_L_missing"] == 0 and line["yhat_vs_sh_base_max_abs"] <= TOL["base"]
                      and line["beta_max_rel"] <= TOL["line_rel"])
    return dict(atoms=atoms, line=line, ok=bool(atoms["ok"] and line["ok"]))


def cmd_e3(force=False):
    t0 = time.time()
    record_decisions(tuple(DECISIONS))                             # spec 11: recorded before they are acted on
    env = env_check()
    roots = register_g3()
    if not T4_JSON.is_file():
        not_ready(f"{T4_JSON} missing (apply.py t4 first)")
    t4 = C.read_json(T4_JSON)
    if not t4.get("all_pass"):
        C.stop(f"{T4_JSON}: T-4 failed (spec 12 T)")
    if (t4.get("fingerprint") or {}).get("code_sha256") != norm(code_sha()):
        not_ready("apply.py / g3_common.py / trainer code changed since T-4: rerun `apply.py t4` (A-12)")
    SP, srep = read_s_pairs()
    u1 = u1_state(SP)                                               # U-2e-1: U-1 pair list and levels
    if u1["active"]:
        SP = SP[SP.pair_id.isin(set(u1["pairs"]))]
    MS, LV = u1["methods"], u1["levels"]
    miss = g3_missing(MS)
    if miss:
        not_ready(f"g3 inputs missing (Phase F / feat.py; roots A-7): {miss[:6]}")
    pbuild = {m: g3_pbuild(m, u1) for m in MS}
    tokchk = {m: tok_path_check(m) for m in MS}
    E2 = C.read_json(E2_REPORT)
    lp_sha = C.sha256(L_PRED)
    if lp_sha != (E2.get("l_report") or {}).get("predictions_sha256"):
        C.stop(f"{L_PRED} sha256 != E2_report l_report.predictions_sha256")
    fp = base_fingerprint(C.sha256(E2_REPORT), lp_sha, LV)
    t4fp = t4.get("fingerprint") or {}
    if fp["pt"] != t4.get("pt_sha256") or norm(fp["unit_json"]) != t4fp.get("unit_json") or \
            lp_sha != t4fp.get("l_predictions") or fp["e2_report"] != t4fp.get("e2_report"):
        C.stop("unit .pt / json files, L_predictions or E2_report differ from the ones T-4 tested (results/T_4.json)")
    fp = norm(dict(fp, t_4=C.sha256(T4_JSON), s_pairs=C.sha256(S_PAIRS), roots=roots,
                   p_build={m: v["sha256"] for m, v in pbuild.items()}, u1=u1["fingerprint"]))
    if not force and APPLY_REPORT.is_file() and PRED_OUT.is_file():
        old = C.read_json(APPLY_REPORT)
        if old.get("fingerprint") == fp and (old.get("outputs") or {}).get("predictions_sha256") == C.sha256(PRED_OUT):
            print(f"[E-3] up to date ({PRED_OUT})", flush=True)
            return 0
    reps = rep_choice(E2)
    if {k: v["model"] for k, v in reps.items()} != {k: v["model"] for k, v in (t4.get("representatives") or {}).items()}:
        C.stop("representatives differ from results/T_4.json")
    lines, line_bad = line_betas(E2)
    if line_bad:
        C.stop(f"LINE betas: {line_bad[:3]}")
    pids = SP.pair_id.tolist()
    DL = {(X, a): TA.get_data(X, a, LABELS) for X in LV for a in ARMS}
    P = DL[("dft", "S")].P
    absent = [p for p in pids if p not in P.index]
    if absent:
        C.stop(f"{len(absent)} Part A pairs are not in the label file {LABELS}: {absent[:4]}")
    ends = P.loc[pids, ["a_rxn", "b_rxn"]]
    if not (np.array_equal(ends.a_rxn.to_numpy(int), SP.a_rxn.to_numpy(int))
            and np.array_equal(ends.b_rxn.to_numpy(int), SP.b_rxn.to_numpy(int))):
        C.stop(f"{S_PAIRS}: a_rxn / b_rxn differ from the label file {LABELS}")
    leak = []                                                       # A-11: pair in its fold's test list of every model rung
    for R in MODEL_RUNGS:
        for a in ARMS:
            for k in range(NF):
                te = set(TA._ids(DL[(R, a)], split_of(DL[(R, a)], k)["test"]))
                leak += [f"{R} {a} f{k} {p}" for p in SP.pair_id[SP.fold == k] if p not in te]
    if leak:
        C.stop(f"{len(leak)} Part A pairs are not in their fold's test list of the model rung: {leak[:4]}")
    layouts = {f"{R}|{X}|{a}": layout_check(DL[(R, a)], DL[(X, a)]) for R in MODEL_RUNGS for X in LV for a in ARMS}
    lev = {}
    for X in LV:
        ok, _, mi = pair_index(DL[(X, "S")], pids)
        lev[X] = dict(n_usable=len(ok), n_missing=len(mi), missing=mi, table=str(TA.inputs_path(X)),
                      fu2_res=str(TA.RUNGS[X]["fu2_res"]), tokens=str(TA.RUNGS[X]["tokens"]),
                      hinfo={k: v for k, v in DL[(X, "S")].hinfo.items() if k != "h_cols"},
                      by_class={c: int(sum(1 for p in ok if SP.at[p, "class"] == c)) for c in sorted(set(SP["class"]))})
    common = [p for p in pids if all(p not in lev[X]["missing"] for X in LV)]
    parts, urec = [], {}
    for R, a in FAMILIES:
        Dm = DL[(R, a)]
        for k in range(NF):
            pk = SP.pair_id[SP.fold == k].tolist()
            if not pk:
                continue
            for m in ATOM:
                name, u = uname(R, a, m, k), dict(rung=R, arm=a, model=m, fold=k)
                rec, jsha = read_unit(name)
                probs, fdiff = unit_gate(rec, Dm, u)
                models, mprobs = load_models(name, rec, a, Dm)
                if probs or fdiff or mprobs:
                    C.stop(f"{name}: {probs + mprobs} fingerprint keys {fdiff}")
                urec[name] = dict(json_sha256=jsha, n_pairs={})
                for X in LV:
                    DX = DL[(X, a)]
                    ok, prow, _ = pair_index(DX, pk)
                    urec[name]["n_pairs"][X] = len(ok)
                    if not ok:
                        continue
                    res = apply_unit(a, models, rec.get("beta"), DX, prow, Dm.elements)
                    parts.append(block(ok, SP, k, R, a, X, m, res, m))
                print(f"[E-3] {name}: {urec[name]['n_pairs']}", flush=True)
                del models
    for R in MODEL_RUNGS:                                           # A-6 LINE (arm-independent: both arms)
        for k in range(NF):
            pk = SP.pair_id[SP.fold == k].tolist()
            if not pk:
                continue
            b = np.asarray([lines[f"{R}|{k}"]["beta"][t] for t in TARGETS], dtype=float)
            for X in LV:
                DX = DL[(X, "SH")]
                ok, prow, _ = pair_index(DX, pk)
                if not ok:
                    continue
                dH = DX.dH[prow]
                yl = dH * b[None, :]
                res = dict(yhat=yl, base=yl, beta=b, y=DX.y[prow], sd=np.full_like(yl, np.nan))
                for a in ARMS:
                    parts.append(block(ok, SP, k, R, a, X, "LINE", res, "LINE"))
    if not parts:
        C.stop("no predictions (no usable Part A pair at any level)")
    A = pd.concat(parts, ignore_index=True)
    rparts = []
    for key, v in reps.items():                                     # A-4 REP rows
        R, a, t = key.split("|")
        s = A[(A.model_rung == R) & (A.arm == a) & (A.target == t) & (A.model == v["model"])]
        rparts.append(s.assign(model="REP", rep_source=v["model"]))
    A = pd.concat([A] + rparts, ignore_index=True)[OUT_COLS]
    if A.duplicated(KEY_COLS).any():
        C.stop("duplicated prediction rows")
    if not np.isfinite(A.yhat.to_numpy(float)).all():
        C.stop("non-finite yhat")
    chk = self_check(A, read_lpred(pids))
    if not chk["ok"]:
        C.stop(f"E-3 self-check failed (A-11): {chk}")
    C.write_atomic(PRED_OUT, lambda t: A.to_parquet(t, index=False))
    counts = A.groupby(["model_rung", "arm", "model", "input_level"]).pair_id.nunique()
    rep = dict(
        phase="E-3 support", spec="docs/specs/D1G3.md 6 E-3", created=now(), env=env, fingerprint=fp,
        rule=(("U-1 complete pairs (results/U1_selection.json, U-2e-1)" if u1["active"] else "Part A pairs (S_pairs.csv)")
              + " x their own T1 fold's units of dft·SH, dft·S, g2·SH, g2·S x input levels " + ", ".join(LV)
              + "; AW / GNN / TF rows, REP (E2_report representatives, XGB -> best atom model), "
              "LINE (fold SH unit beta x dH of the level, both arms)"),
        families=[f"{R}·{a}" for R, a in FAMILIES], levels=list(LV), levels_stopped=u1["stopped"], u1=u1["record"],
        n_pairs=len(pids), folds_with_pairs=sorted(int(k) for k in set(SP.fold)),
        pairs_by_class=SP["class"].value_counts().to_dict(), levels_info=lev,
        common_pairs=dict(n=len(common), by_class={c: int(sum(1 for p in common if SP.at[p, "class"] == c))
                                                   for c in sorted(set(SP["class"]))}, ids=common),
        layouts=layouts, tok_path_check=tokchk, p_build=pbuild, roots=roots, representatives=reps,
        xgb_replaced={k: v["model"] for k, v in reps.items() if v["xgb_replaced"]}, line_betas=lines, units=urec,
        self_check=chk, t_4=dict(file=str(T4_JSON), sha256=C.sha256(T4_JSON)),
        s_pairs=dict(file=str(S_PAIRS), sha256=C.sha256(S_PAIRS), s_report_sha256=C.sha256(S_REPORT)),
        rows=dict(n=int(len(A)), pairs_by_family_model_level={"|".join(map(str, i)): int(v) for i, v in counts.items()}),
        outputs=dict(predictions=str(PRED_OUT), predictions_sha256=C.sha256(PRED_OUT), columns=OUT_COLS),
        decisions={k: DECISIONS[k] for k in DECISIONS}, wall_s=time.time() - t0)
    C.write_json(APPLY_REPORT, rep)
    print(f"[E-3] {len(A)} rows -> {PRED_OUT}; usable pairs per level "
          f"{ {X: lev[X]['n_usable'] for X in LV} } of {len(pids)}, common {len(common)}; self-check ok "
          f"(max |dyhat| {chk['atoms']['yhat_max_abs']:.3g})"
          + (f"; U-1: {', '.join(f'{m} {v}' for m, v in u1['stopped'].items())}" if u1["active"] else ""), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("t4", "e3"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return cmd_t4(a.force) if a.cmd == "t4" else cmd_e3(a.force)


if __name__ == "__main__":
    sys.exit(main())
