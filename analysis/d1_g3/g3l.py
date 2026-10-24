#!/usr/bin/env python3
r"""g3l.py — D1-G3 G3L pilot (user decision 2026-10-25 item 7, new, user-approved; explanatory / 설명용): the Part B
label-level OptTS (opt.partb_text: the D1 autodE L0 OptTS + Freq, B3LYP RIJCOSX D3BJ def2-SVP SMD(water)) started from
the G2 TS with the G2 xTB Hessian as the initial Hessian ('InHess Read') and without Recalc_Hess, on 20 reactions of the
'same' class; 200 core-h cap for the whole pilot. sbatch only (jobs/g3.slurm G3ENV=d1); nothing in this file was run
when it was written (login node), so every path is untested until it runs inside a job.

What is computed (decisions U-7-1 .. U-7-9)
  sample   S_pairs.csv class 'same': the unique a_rxn / b_rxn (sorted) whose G2 xTB Hessian passes the check (U-7-3) ->
           numpy.random.default_rng(20261025).choice(<those rxn_ids>, size=20, replace=False); the pick order is the
           claim order (U-7-1, U-7-7).
  input    opt.partb_text(G2 TS, label charge, label forming bonds, %pal 8) with, in its first %geom block, the line
           'Calc_Hess true' replaced by 'InHess Read' + 'InHessName "g2.hess"' and the line 'Recalc_Hess 30' removed;
           nothing else changes (checked, U-7-2). g2.hess = byte copy of $G2/g2_final/<rxn>/work/ts.hess (the
           g2_autode.finalize XTB2 ALPB(water) Freq at the G2 TS; $atoms = ts.xyz order and coordinates, U-7-3).
  run      opt.orca_step($G3/O/units/L.<rxn>/optts, 'ts') through the ORCA launcher $G3/L/bin/orca_g2hess.sh (copies the
           node-local g2.hess into the ORCA work folder, which orca_step empties before every run, U-7-4); unit.json by
           opt.run_unit (O-19 wall_clipped, starts.json, step.json reuse, infra rerun once all as opt.py).
  gate     opt.gate_ts exactly as opt.unit_b: refs {'start': G2 TS, 'label': label TS ($D1_JOBS/<T>/ts.xyz)}, reactants
           = the D1 reference structures, partition not hard; converged_to_label = rmsd.same_ts(G3L TS, label TS). No
           autodE _reoptimise, no SP (U-7-5). The output must show that g2.hess was read (else pipeline error + STOP).
  cap      200 core-h: before each claim (under $G3/locks/G3L_cap.lock) committed + 8 for the next unit must stay
           <= 200, committed = finished L.* core_h + runs that never finished (killed / wall: start .. last heartbeat
           $G3/L/hb/<uid>.<job> + 120 s, x 8 ranks) + per unit held by a live job max(8, 8 ranks x elapsed since its
           start); a refused claim ends that slot (the unit stays first in line for the job's other slots); events ->
           $G3/L/cap_events.jsonl (U-7-6). Running units are never stopped, so the total can still pass 200.

CLI (exit 0 = done / up to date, 2 = incomplete (resubmit) or usage, 3 = STOP; submit from analysis/d1_g3, G3ENV=d1)
  plan [--force]   $G3/L/plan.json + results/G3L_plan.csv (sample, Hessian checks, input-text check for every pick);
                   refused (STOP) when the sample rule inputs changed after L units ran
  work             claims L units (opt.claim; pick order) and runs $SLURM_CPUS_PER_TASK // 8 ORCA calculations (8 ranks)
                   at a time; exit 0 = every unit finished or the cap reached, 2 = units left (time / held elsewhere)
  report           results/G3L_status.csv, results/G3L_report.json (exit 2 = units neither finished nor capped)

Outputs
  $G3/O/units/L.<rxn>/unit.json (opt.run_unit; + g3l fields), L.<rxn>/g2.hess.gz (the Hessian read), L.<rxn>/optts/
  (ts.inp, ts.out.gz, ts.err.gz, ts.xyz, ts.hess.gz, starts.json, step.json); $G3/O/locks/L.<rxn>.lock; $G3/L/plan.json,
  $G3/L/cap_events.jsonl, $G3/L/hb/<uid>.<job>, $G3/L/STOP (pipeline STOP of the pilot), $G3/L/bin/orca_g2hess.sh;
  results/G3L_plan.csv, G3L_status.csv, G3L_report.json; decisions in results/DECISIONS.md (opt.record_decision).
  NOTE: the L.* units share $G3/O/units with Phase O (orca_step needs dest under opt.UNITS); done_json.py globs
  units/*/unit.json and skips only T5.*, so a Phase O export made after this pilot must also skip L.* (U-7-5).

Reused, imported read only (never copied): opt.py d1 (D1 modules / env), load_plan, partb_text, orca_step, gate_ts,
_dists, hess_atoms, claim / release, run_unit (UNIT_FN['L'] = unit_l registered in this process only), _att, _rj, _f,
_jobtag, _cpus, _now, record_decision, T0 / STOP_AFTER_H, CHECK, LOCKS, PARTB_NPROCS; the D1 code copy (pilot_common
read_xyz, orca_direct parse_hess, run_ts.analyse inside gate_ts, rmsd.same_ts); refs_v._job_alive (in claim, the cap).

sbatch (login node: one-shot commands only; first `squeue -u $USER -h -r | wc -l` (MaxSubmit 20 incl. array elements,
MaxJobs 10) and `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`). One task, 5 slots x 8 ranks (%maxcore 3000 MB per rank):
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_g3; L=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs
  JL=$(sbatch --parsable -p cpu1,cpu2 -J g3l --time=48:00:00 --cpus-per-task=40 --mem=160G --output=$L/%x_%j.out \
       --export=ALL,G3ENV=d1 jobs/g3.slurm g3l.py plan ::: g3l.py work ';;;' g3l.py report)
  (array alternative, more slots: JLP = the same line with only 'g3l.py plan' and --cpus-per-task=2 --mem=8G, then
   --array=0-1 --cpus-per-task=32 --mem=128G --output=$L/%x_%A_%a.out --dependency=afterok:$JLP ... g3l.py work, then
   --cpus-per-task=2 --mem=8G --dependency=afterany:$JLW ... g3l.py report; the cap holds across tasks)
  Resubmission (idempotent): the same line (plan up to date; finished units skipped; dead jobs' locks taken over; a
  step clipped twice by the wall closes its unit as wall_clipped, O-19; the cap counts finished, killed (heartbeat)
  and running units). exit 3 = $G3/L/STOP or $G3/O/check.json not ok: read the job log and report.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # nothing compiled into the read-only code trees (D1 copy, d1_ladder, d1_g2)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g3_common as C  # noqa: E402   (paths, write_atomic / write_json)
import opt as O  # noqa: E402         (Phase O machinery; read only)

CODE_VERSION = "d1-g3-l1"
PLAN_RULE = "g3l-sample-1"                             # the sample rule version in the plan fingerprint (U-7-1)
LD = C.G3 / "L"
PLAN_L, STOP_L, CAP_LOG = LD / "plan.json", LD / "STOP", LD / "cap_events.jsonl"
WRAPPER = LD / "bin" / "orca_g2hess.sh"
CAP_LOCK = C.G3 / "locks" / "G3L_cap.lock"
RES_PLAN, STATUS_CSV, REPORT = C.RES / "G3L_plan.csv", C.RES / "G3L_status.csv", C.RES / "G3L_report.json"
SEED_L, N_L, CLASS_L = 20261025, 20, "same"            # user decision 7 / U-7-1
CAP_CORE_H, RESERVE_CORE_H = 200.0, 8.0                # U-7-6
HB_DIR, HB_EVERY_S = LD / "hb", 120.0                  # U-7-6: heartbeats of running L units (wall of a killed run)
EMPTY_LOCK_S = 3600                                    # opt.claim: an empty lock is taken over after 1 h
HESS_NAME = "g2.hess"
INHESS_LINES = ("InHess Read", f'InHessName "{HESS_NAME}"')
HESS_GEOM_TOL = 1e-4                                   # A (the opt.gate_ts hess-at-geometry tolerance), no shift removal
INHESS_FAIL = "pipeline_inhess_not_read"
LABEL_TS_CORE_H = 8.2                                  # spec E-5: pilot mean of the label TS step (context only)
RE_INHESS = re.compile(r"InHessName:\s*(\S+)\s+The file <([^>]+)> is opened as a \.hess file")
RE_RELAX = re.compile(r"ORCA GEOMETRY RELAXATION STEP")
RE_SCF_HESS = re.compile(r"^SCF HESSIAN\s*$", re.M)
RE_INHESS_SET = re.compile(r"^Initial Hessian\s+InHess\s+\.+\s*(.*?)\s*$", re.M)
WRAPPER_TEXT = """#!/bin/bash
# written by analysis/d1_g3/g3l.py (decision U-7-4): ORCA launcher of the G3L pilot. opt.orca_step empties the ORCA work
# folder before every run, so g3l.py puts the G2 xTB Hessian next to it (../{hess}); it is copied in here as {hess}
# (InHessName "{hess}") and ORCA 6.1.1 is exec'd by its full path (ORCA finds its MPI programs from that path).
set -eu
cp -f ../{hess} ./{hess}
exec {orca} "$@"
"""

DECISIONS = {
    "U-7-1": "G3L 표본(사용자 결정 2026-10-25 7, 신규·사용자 승인, 설명용): results/S_pairs.csv class == 'same' 짝의 "
             "a_rxn ∪ b_rxn(고유, 정렬) 중 G2 xTB Hessian 검사(U-7-3)를 통과한 반응 = 후보(정렬 유지) -> "
             "numpy.random.default_rng(20261025).choice(후보 rxn_id 배열, size=20, replace=False). 검사 실패 반응은 뽑기 "
             "전에 빼고 plan.json excluded에 이유와 함께 적는다(결정 7 'G2 xTB Hessian이 있는 반응'); 후보가 20보다 적으면 "
             "있는 만큼(명령서 2절과 같은 처리). 반응 정보(형성 결합, 전하, A_idx, mapped_smiles) = opt plan rxns(Part A 반응).",
    "U-7-2": "G3L 입력: opt.partb_text(D, G2 TS, 라벨 charge1 + charge2, 다중도 1, 라벨 형성 결합, %pal 8) = Part B 입력(D1 "
             "autodE L0 OptTS Freq 머리, modify_internal 블록, %maxcore 3000; opt check가 D1 run 입력과 바이트 단위로 같음을 "
             "증명한 문자열)에서 첫 %geom 블록의 'Calc_Hess true' 한 줄을 'InHess Read' + 'InHessName \"g2.hess\"' 두 줄로 "
             "바꾸고 'Recalc_Hess 30' 줄을 지운다(Trust -0.1, MaxIter 100 그대로). 그 밖의 줄은 그대로임을 만들 때마다 "
             "확인한다(plan에서 뽑힌 모든 반응, 단위 실행 때 다시). 시작 구조 = $G2/g2_final/<rxn>/ts.xyz(라벨 원자 순서).",
    "U-7-3": "G2 Hessian 검사: $G2/g2_final/<rxn>/work/ts.hess(g2_autode.finalize의 XTB2 ALPB(water) Freq, G2 TS에서): G2 "
             ".done, $atoms 원소 순서 = ts.xyz, 좌표 최대 차 <= 1e-4 Å(Bohr -> Å; Cartesian Hessian을 그대로 쓰므로 평행 "
             "이동 보정 없음), $hessian 차원 = 3N, work/ts.inp '!' 줄에 XTB2. sha256을 plan에 적고 단위 실행 때 다시 "
             "확인(다르면 pipeline 오류). 읽힌 파일의 사본 = 단위 폴더 g2.hess.gz.",
    "U-7-4": "ORCA launcher: opt.orca_step은 ORCA를 부르기 전마다 작업 폴더의 파일을 지운다(인프라 재실행 포함) -> g2.hess를 "
             "미리 넣어 둘 수 없다. opt.py를 고치지 않고 D['orca']만 $G3/L/bin/orca_g2hess.sh로 바꾼다: g3l.py가 노드 로컬 "
             "작업 폴더 바로 위(<local>/L.<rxn>/g2.hess)에 둔 파일을 작업 폴더로 복사하고 config_run orca_bin(= D1 ORCA "
             "6.1.1)을 전체 경로로 exec한다(MPI·환경은 그대로). 입력의 InHessName은 경로 없는 \"g2.hess\"(ORCA가 파일 "
             "형식을 이름의 확장자로 고르므로 경로의 점에 기대지 않음).",
    "U-7-5": "G3L 실행·gate: opt.orca_step(UNITS/L.<rxn>/optts, 이름 'ts'; starts.json·O-19 wall_clipped·step.json 재사용·"
             "인프라 1회 재실행 그대로) + opt.gate_ts(refs start = G2 TS, label = 라벨 TS, 반응물 = D1 기준 구조, partition "
             "hard 아님) = opt.unit_b의 gate 그대로; unit.json은 opt.run_unit(UNIT_FN['L'] = unit_l, 이 프로세스에서만 "
             "등록). autodE _reoptimise(O-9)는 하지 않는다(두 번째 실행의 초기 Hessian이 결정 7에 없고 G2 Hessian은 다른 "
             "구조의 것; Part B에서도 reopt 3건이 모두 실패). SP 없음. 구조 이완 단계까지 갔는데 출력에 'InHessName: "
             "g2.hess ... opened as a .hess file'이 없으면 pipeline 오류 pipeline_inhess_not_read: 모든 단위에 같은 문제이므로 "
             "$G3/L/STOP을 쓰고 새 단위를 받지 않는다. 'SCF HESSIAN' 수(기대 1 = 마지막 Freq; Recalc 없음)를 기록한다. "
             "L.* 단위는 $G3/O/units를 Phase O와 같이 쓰므로(orca_step이 dest를 opt.UNITS 아래로 요구) done_json.py "
             "내보내기는 T5.*처럼 L.*도 빼야 한다(main session).",
    "U-7-6": "200 코어시간 상한: 코어시간 = ORCA 실행 wall x 8(opt.orca_step 규칙). 단위를 claim하기 전에 "
             "$G3/locks/G3L_cap.lock(fcntl, 모든 job 공유) 안에서 committed + 8(이번 단위)을 계산해 200을 넘으면 "
             "claim하지 않는다. committed = 끝난 L 단위 unit.json core_h 합 + 끝나지 않은 시작(starts.json 중 step.json에 "
             "이르지 못한 것: kill·48 h 벽) 각각 8 x (그 job의 마지막 heartbeat $G3/L/hb/<uid>.<job> mtime + 120 s - "
             "시작 시각; heartbeat가 없으면 8; 상한 추정) + unit.json 없이 살아 있는 job이 lock을 쥔 단위(opt.claim과 "
             "같이 1 h 안의 빈 lock 포함; 더 오래된 빈 lock은 claim이 인수) 각각 max(8, 8 x 지금 실행의 경과 시간). "
             "heartbeat = work가 도는 단위마다 120 s마다 touch. 거절된 슬롯은 끝나고 그 단위는 같은 job의 다른 슬롯이 "
             "자기 단위를 끝낸 뒤 다시 본다(짧게 끝난 단위로 여유가 생기면 받음). 사건 = $G3/L/cap_events.jsonl(시각, "
             "job, 거절된 단위, 합계). 도는 단위는 멈추지 않으므로 실제 합계가 200을 넘을 수 있다(report "
             "over_cap; Part B OptTS 40개 core_h 중앙값 6.35, 평균 10.7, 최대 66.2). report는 cap 사건이 있고 지금도 "
             "다음 단위가 200을 넘을 때 남은 단위를 not_run_cap으로 적는다.",
    "U-7-7": "G3L 배치: claim 순서 = 뽑힌 순서(rng 순서; 상한에 걸려도 끝난 반응이 크기에 치우치지 않게 — opt.Worker의 큰 것 "
             "먼저와 다름). opt.claim / release($G3/O/locks/L.<rxn>.lock; squeue에 없는 job의 lock은 인수), task마다 "
             "$SLURM_CPUS_PER_TASK // 8 슬롯(8 rank ORCA), 44 h 뒤 새 단위 없음(opt.T0, STOP_AFTER_H). opt.Worker는 쓰지 "
             "않는다: refresh / loop가 $G3/O의 STOP 파일을 쓰고 그것에 막히므로(G3L은 Phase O를 멈추거나 그것에 멈추지 "
             "않는다).",
    "U-7-8": "G3L 준비 조건: work는 $G3/O/check.json이 지금 opt plan 지문에 대해 ok일 때만 돈다(Part B 입력 문자열 증명, D1 "
             "환경·MPI 검사); T-2/T-3 재실행, $G3/O/STOP, assemble 상태는 보지 않는다(G3L은 그 결과를 쓰지 않음). "
             "$G3/L/STOP이 있으면 exit 3.",
    "U-7-9": "G3L 보고: G3L_status.csv(반응별 상태, 이유, core_h, n_opt_cycles, n_imag, 허수 진동수, converged_to_label, "
             "rmsd_label_A, dform_label_A, 형성 결합별 d(G3L)·d(라벨)·차이; 결합 순서 = 라벨 formed_pairs_ts) / "
             "G3L_report.json: 코어시간 중앙값·평균·합(끝난 단위), TS gate 통과율 = ok / (ok + fail)(pipeline 오류는 빼고 "
             "따로 셈; 끝난 단위 전체 기준 값도), converged_to_label 비율(ok 중, 끝난 것 중), |형성 결합 차이|(결합별 모음과 "
             "반응별 최대의 중앙값·최대), 무거운 원자 RMSD(rmsd.same_ts) 중앙값·최대, cap. same-TS 규칙(0.1 Å / 0.02 Å)은 "
             "여기서 같은 수준(라벨 수준 OptTS 대 라벨 TS) 비교라 U-3b(수준 간 비교 부적합)에 해당하지 않는다. 참고값: Part B "
             "OptTS 코어시간(O 단위 B.*; 반응 집합이 다름), 라벨 TS 단계 파일럿 평균 8.2(명령서 E-5). 결과는 설명용.",
}


# ================================================================ helpers
def record_decision(key, text):
    O.record_decision(key, text)                     # g3_common.record_decision under the shared fcntl lock


def record_all():
    """spec 11: every decision of this file, before the command acts on it (idempotent per key)."""
    for k, v in DECISIONS.items():
        record_decision(k, v)


@contextlib.contextmanager
def cap_lock():
    """U-7-6: the cross-job lock of the cap check + claim ($G3/locks/G3L_cap.lock, fcntl like opt.record_decision)."""
    import fcntl
    lk = C.assert_path(CAP_LOCK)
    lk.parent.mkdir(parents=True, exist_ok=True)
    with open(lk, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def write_stop_l(msg):
    C.write_atomic(STOP_L, f"{O._now()} {msg}\n")
    print(f"STOP: G3L: {msg}", file=sys.stderr, flush=True)


def _stats(v):
    a = [float(x) for x in v if x is not None and np.isfinite(float(x))]
    n_bad = sum(1 for x in v if x is not None and not np.isfinite(float(x))) if isinstance(v, list) else None
    if not a:
        return dict(n=0, n_nonfinite=n_bad)
    return dict(n=len(a), median=float(np.median(a)), mean=float(np.mean(a)), min=float(np.min(a)),
                max=float(np.max(a)), sum=float(np.sum(a)), n_nonfinite=n_bad)


def hess_dim(path):
    lines = Path(path).read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "$hessian")
    return int(lines[i + 1].split()[0])


def hess_check(D, rid):
    """U-7-3: the G2 xTB Hessian of reaction rid -> dict(ok, reason, path, sha256, ...)."""
    g2d = O.G2_FINAL / str(rid)
    hp, xp, ip = g2d / "work" / "ts.hess", g2d / "ts.xyz", g2d / "work" / "ts.inp"
    out = dict(path=str(hp), ok=False, reason="")
    if not ((g2d / ".done").is_file() and xp.is_file()):
        return dict(out, reason="no_g2_ts")
    if not hp.is_file():
        return dict(out, reason="no_hess")
    try:
        ts = D["pc"].read_xyz(xp)
        hs, hx = O.hess_atoms(hp)
        same = list(hs) == list(ts[0])
        dev = float(np.abs(np.asarray(hx, float) - np.asarray(ts[1], float)).max()) if same else None
        kw = next((ln.strip() for ln in ip.read_text().splitlines() if ln.strip().startswith("!")), None) \
            if ip.is_file() else None
        freqs = [float(v) for v in D["OD"].parse_hess(hp)[0]]
        out.update(sha256=C.sha256(hp), n_atoms=len(hs), syms_equal=same, max_dev_A=dev, dim=hess_dim(hp),
                   keyword_line=kw, imag_cm=sorted(v for v in freqs if v < 0))
    except Exception as e:                                         # noqa: BLE001
        return dict(out, reason=f"hess_unreadable: {type(e).__name__}: {e}"[:300])
    if not same:
        return dict(out, reason="hess_atom_order")
    if dev > HESS_GEOM_TOL:
        return dict(out, reason="hess_geometry")
    if out["dim"] != 3 * len(hs):
        return dict(out, reason="hess_dimension")
    if not kw or "XTB2" not in kw.replace("!", " ").split():
        return dict(out, reason="hess_not_xtb2")
    return dict(out, ok=True)


def g3l_text(D, syms, xyz, q, pairs):
    """U-7-2: (G3L input, the Part B input it was made from)."""
    base = O.partb_text(D, syms, xyz, q, 1, pairs, O.PARTB_NPROCS)
    lines = base.split("\n")
    ic = [k for k, ln in enumerate(lines) if ln.strip().lower() == "calc_hess true"]
    ir = [k for k, ln in enumerate(lines) if ln.strip().lower().startswith("recalc_hess")]
    if len(ic) != 1 or len(ir) != 1 or ir[0] != ic[0] + 1 or lines[ic[0] - 1].strip().lower() != "%geom":
        raise ValueError("Part B input: no single '%geom / Calc_Hess true / Recalc_Hess' block")
    new = lines[:ic[0]] + list(INHESS_LINES) + lines[ir[0] + 1:]
    kept = [ln for k, ln in enumerate(lines) if k not in (ic[0], ir[0])]
    if (new[:ic[0]] + new[ic[0] + len(INHESS_LINES):] != kept or any("calc_hess" in ln.lower() for ln in new)
            or sum(ln.strip() == INHESS_LINES[0] for ln in new) != 1):
        raise ValueError("G3L input differs from the Part B input by more than the Hessian lines")
    return "\n".join(new), base


def inhess_evidence(txt):
    """U-7-5: was g2.hess read as the initial Hessian (and how many analytic Hessians were computed)?"""
    names = [m.group(2) for m in RE_INHESS.finditer(txt)]
    relax = len(RE_RELAX.findall(txt))
    read = bool(names) and names[0] == HESS_NAME
    return dict(read=read, names=names[:6], n_reads=len(names), n_relax_steps=relax,
                n_scf_hessian=len(RE_SCF_HESS.findall(txt)), initial_hessian_setting=RE_INHESS_SET.findall(txt)[:1],
                not_read=relax >= 1 and not read)


def write_wrapper(orca):
    """U-7-4: $G3/L/bin/orca_g2hess.sh (written with its exec bit before the rename: no job sees it half made)."""
    txt = WRAPPER_TEXT.format(hess=HESS_NAME, orca=shlex.quote(str(orca)))
    if WRAPPER.is_file() and WRAPPER.read_text() == txt and os.access(WRAPPER, os.X_OK):
        return WRAPPER

    def _w(t):
        t.write_text(txt)
        os.chmod(t, 0o755)
    C.write_atomic(WRAPPER, _w)
    return WRAPPER


def load_plan_l(plan):
    """(G3L plan, 0, '') or (None, 2 | 3, why)."""
    P = O._rj(PLAN_L)
    if P is None:
        return None, 2, f"{PLAN_L} missing: run 'g3l.py plan' first"
    if P.get("fingerprint", {}).get("opt_plan") != plan["fingerprint"]:
        return None, 3, f"{PLAN_L} was made for another opt plan"
    return P, 0, ""


# ================================================================ plan
def fingerprint(plan):
    return dict(rule=PLAN_RULE, S_pairs=C.sha256(O.S_PAIRS), opt_plan=plan["fingerprint"], seed=SEED_L, n=N_L,
                cls=CLASS_L)


def cmd_plan(force=False):
    record_all()
    D, plan = O.d1(), O.load_plan()
    fp = fingerprint(plan)
    if PLAN_L.is_file():
        old = C.read_json(PLAN_L)
        if old.get("fingerprint") == fp and not force:
            print("G3L plan up to date", flush=True)
            return 0
        if any((O.UNITS / u["uid"] / "unit.json").is_file() for u in old.get("units", [])):
            C.stop(f"G3L plan inputs changed ({old.get('fingerprint')} -> {fp}) after L units ran; not re-planned")
    S = pd.read_csv(O.S_PAIRS)
    for c in ("pair_id", "a_rxn", "b_rxn", "class"):
        if c not in S.columns:
            C.stop(f"{O.S_PAIRS}: column {c} missing")
    same = S[S["class"].astype(str) == CLASS_L]
    pool = sorted({int(x) for x in same.a_rxn} | {int(x) for x in same.b_rxn})
    if not pool:
        C.stop(f"{O.S_PAIRS}: no '{CLASS_L}' pair")
    checks = {rid: hess_check(D, rid) for rid in pool}
    cand = [rid for rid in pool if checks[rid]["ok"]]
    if not cand:
        C.stop(f"no '{CLASS_L}' reaction with a usable G2 xTB Hessian ({dict(Counter(c['reason'] for c in checks.values()))})")
    rng = np.random.default_rng(SEED_L)
    pick = [int(x) for x in rng.choice(np.asarray(cand, dtype=np.int64), size=min(N_L, len(cand)), replace=False)]
    units, texts = [], {}
    pc = D["pc"]
    for k, rid in enumerate(pick, 1):
        R = plan["rxns"].get(str(rid))
        if R is None or not R.get("part_a"):
            C.stop(f"G3L pick {rid} is not a Part A reaction of the opt plan")
        g2ts = pc.read_xyz(O.G2_FINAL / str(rid) / "ts.xyz")
        try:
            txt, _ = g3l_text(D, g2ts[0], g2ts[1], R["charge1"] + R["charge2"], R["pairs"])
        except ValueError as e:
            C.stop(f"G3L input of {rid}: {e}")
        texts[rid] = txt
        units.append(dict(uid=f"L.{rid}", kind="L", method="g3l", rxn_id=rid, job_id=R["job_id"], mid=None,
                          n_atoms=int(R["n_atoms"]), phase="g3l", pick=k, weight=float(R["n_atoms"]) ** 3,
                          dipole=R.get("dipole"), classes=R.get("classes"), hess_sha256=checks[rid]["sha256"],
                          g2_imag_cm=checks[rid].get("imag_cm")))
    first = texts[pick[0]]
    rep = dict(code=CODE_VERSION, written=O._now(), fingerprint=fp, decisions=DECISIONS,
               rule=f"S_pairs.csv class == '{CLASS_L}': unique a_rxn | b_rxn sorted, G2 xTB Hessian check (U-7-3) -> "
                    f"numpy.random.default_rng({SEED_L}).choice(<candidates>, size={N_L}, replace=False); claim order = "
                    f"pick order",
               n_pool=len(pool), n_candidates=len(cand), n_picked=len(pick), short=len(pick) < N_L,
               excluded={str(r): c["reason"] for r, c in checks.items() if not c["ok"]}, picks=pick,
               picks_sorted=sorted(pick), hess_checks={str(r): checks[r] for r in pick},
               inhess_lines=list(INHESS_LINES), header_example=first.split("*xyz")[0], header_example_rxn=pick[0],
               cap_core_h=CAP_CORE_H, reserve_core_h=RESERVE_CORE_H, units=units, n_units=len(units))
    C.write_json(PLAN_L, rep)
    cols = ["uid", "rxn_id", "job_id", "n_atoms", "pick", "dipole", "hess_sha256"]
    C.write_atomic(RES_PLAN, lambda t: pd.DataFrame(units)[cols].to_csv(t, index=False))
    print(json.dumps(dict(n_pool=len(pool), n_candidates=len(cand), picks=pick, excluded=rep["excluded"]), indent=1),
          flush=True)
    return 0


# ================================================================ unit (registered as opt.UNIT_FN['L'])
def unit_l(u, D, plan, nprocs, local):
    """one G3L unit (U-7-2 .. U-7-5): the InHess-read Part B OptTS + Freq from the G2 TS, gated as opt.unit_b."""
    if os.path.abspath(D["orca"]) != os.path.abspath(WRAPPER):
        raise RuntimeError(f"G3L units run only through 'g3l.py work' (ORCA launcher {WRAPPER}, decision U-7-4)")
    R = plan["rxns"][str(u["rxn_id"])]
    T, pc = R["job_id"], D["pc"]
    g2d = O.G2_FINAL / str(R["rxn_id"])
    g2ts = pc.read_xyz(g2d / "ts.xyz")
    lts = pc.read_xyz(O.D1_JOBS / T / "ts.xyz")
    reacs = [pc.read_xyz(O.D1_JOBS / T / "ref_dipole.xyz"), pc.read_xyz(O.D1_JOBS / T / "ref_dipolarophile.xyz")]
    q = R["charge1"] + R["charge2"]
    pairs = [tuple(int(i) for i in p) for p in R["pairs"]]
    ud = O.UNITS / u["uid"]
    hc = hess_check(D, R["rxn_id"])
    if not hc["ok"] or hc.get("sha256") != u["hess_sha256"]:
        raise RuntimeError(f"G2 Hessian {hc['path']}: {hc['reason'] or 'sha256 differs from the G3L plan'}")
    hb = Path(hc["path"]).read_bytes()
    if not (ud / f"{HESS_NAME}.gz").is_file():
        C.write_atomic(ud / f"{HESS_NAME}.gz", gzip.compress(hb, 6))
    hl = local / u["uid"] / HESS_NAME                      # = <orca_step work folder>/.. (work = local / L.<rxn> / optts)
    hl.parent.mkdir(parents=True, exist_ok=True)
    hl.write_bytes(hb)
    text, _ = g3l_text(D, g2ts[0], g2ts[1], q, R["pairs"])

    def post(w, t, s):
        e = D["H"].RE_FSPE.findall(t)                              # autodE ORCAOptimiser.last_energy_change (recorded)
        return {"gate": O.gate_ts(D, R, w, "ts", t, s, reacs, {"start": g2ts, "label": lts}, False),
                "de_last_kcal": (float(e[-1]) - float(e[-2])) * C.KCAL if len(e) >= 2 else None,
                "inhess": inhess_evidence(t)}
    st = O.orca_step(D, ud / "optts", "ts", text, local, post)
    out = dict(attempts=[O._att(st)], core_h=st["core_h"], orca_version=st["orca_version"],
               n_opt_cycles=st["n_opt_cycles"], reopt=False, attempt_used=1, final_dir="optts",
               header=text.split("*xyz")[0], hess_source=hc["path"], hess_sha256=hc["sha256"],
               g2_imag_cm=hc.get("imag_cm"), inhess=st.get("inhess"), de_last_kcal=st.get("de_last_kcal"),
               pairs=[list(p) for p in pairs], d_form_label=O._dists(lts[1], pairs), g3l_code=CODE_VERSION)
    if st["infra"]:
        return dict(out, status="error", reason="pipeline_infra")
    if (st.get("inhess") or {}).get("not_read"):
        return dict(out, status="error", reason=INHESS_FAIL, gate=st.get("gate"))
    gt = st["gate"]
    lb, sh = gt.get("label") or {}, gt.get("start") or {}
    dn = gt.get("d_form")
    out.update(gate=gt, converged_to_label=lb.get("same_ts"), rmsd_label_A=lb.get("rmsd_A"),
               dform_label_A=lb.get("dform_A"), same_as_g2_start=sh.get("same_ts"), rmsd_g2_A=sh.get("rmsd_A"),
               dform_g2_A=sh.get("dform_A"), d_form_g3l=dn,
               dd_form_label=[a - b for a, b in zip(dn, out["d_form_label"])] if dn else None,
               n_imag=gt.get("n_imag"), imag_freqs_cm=gt.get("imag_freqs_cm"), e_optts_eh=st["e_last_eh"])
    if not gt["ok"]:
        return dict(out, status="fail", reason=gt["reason"])
    return dict(out, status="ok", reason="converged_to_label" if lb.get("same_ts") else "")


O.UNIT_FN["L"] = unit_l                                    # opt.run_unit dispatch (this process only; opt.py unchanged)


# ================================================================ cap, worker
def _epoch(s):
    """opt._now() text ('%Y-%m-%d %H:%M:%S', node local time) -> epoch s; None if unreadable."""
    try:
        return time.mktime(time.strptime(str(s), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        return None


def beat(uid):
    """U-7-6: touch $G3/L/hb/<uid>.<job> (this job's run of uid is alive now)."""
    try:
        p = C.assert_path(HB_DIR / f"{uid}.{O._jobtag()}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def heartbeat(W, ev):
    """U-7-6: beat() every unit this job runs, every HB_EVERY_S, until ev is set (daemon thread of cmd_work)."""
    while not ev.wait(HB_EVERY_S):
        ids = sorted(W.running.copy())     # set.copy is atomic (GIL); not under W.mu, which a cap check (squeue) holds
        for uid in ids:
            beat(uid)


def lock_live(D, uid):
    """(lock owner, held): held = owner is this job or a live job, or an empty lock younger than 1 h (the opt.claim
    rule; an older empty lock is not held, so opt.claim takes it over)."""
    p = O.LOCKS / f"{uid}.lock"
    try:
        owner = p.read_text().split()[0]
    except (OSError, IndexError):
        try:
            return "", time.time() - p.stat().st_mtime < EMPTY_LOCK_S
        except OSError:                                # no lock
            return None, False
    return owner, owner == O._jobtag() or D["RV"]._job_alive(owner)


def unit_cost(D, uid, now):
    """U-7-6: core-h of L unit uid: finished (unit.json core_h), killed (starts that never reached step.json, each
    8 ranks x (last heartbeat of that job + HB_EVERY_S - start), RESERVE_CORE_H without heartbeat / start time),
    inflight (unit held by a live job without unit.json: max(RESERVE_CORE_H, 8 ranks x elapsed of the run now))."""
    ud = O.UNITS / uid
    r = O._rj(ud / "unit.json")
    starts = O._rj(ud / "optts" / "starts.json") or []
    open_ = list(starts[:-1] if (ud / "optts" / "step.json").is_file() else starts)
    owner, held = (None, False) if r is not None else lock_live(D, uid)
    cur = open_.pop() if held and open_ and open_[-1].get("job") == owner else None   # the run going on now
    killed = 0.0
    for s in open_:
        t0 = _epoch(s.get("started"))
        try:
            hb = (HB_DIR / f"{uid}.{s.get('job')}").stat().st_mtime
        except OSError:
            hb = None
        killed += (O.PARTB_NPROCS * max(0.0, hb + HB_EVERY_S - t0) / 3600 if hb is not None and t0 is not None
                   else RESERVE_CORE_H)
    inflight = 0.0
    if held:
        t0 = _epoch(cur.get("started")) if cur else None
        inflight = max(RESERVE_CORE_H, O.PARTB_NPROCS * (now - t0) / 3600 if t0 is not None else 0.0)
    return dict(finished=float((r or {}).get("core_h") or 0.0), killed=killed, n_killed=len(open_),
                inflight=inflight, held=held, owner=owner)


def cap_state(D, units):
    """U-7-6: dict(finished, killed, inflight, committed = their sum, held = [uids], per = {uid: unit_cost})."""
    now = time.time()
    per = {u["uid"]: unit_cost(D, u["uid"], now) for u in units}
    cs = {k: sum(c[k] for c in per.values()) for k in ("finished", "killed", "inflight")}
    cs.update(committed=cs["finished"] + cs["killed"] + cs["inflight"], held=[k for k, c in per.items() if c["held"]],
              per=per)
    return cs


def cap_event(uid, cs, proj):
    ev = dict(time=O._now(), job=O._jobtag(), host=socket.gethostname(), refused=uid, finished_core_h=cs["finished"],
              killed_core_h=cs["killed"], inflight_core_h=cs["inflight"], committed_core_h=cs["committed"],
              n_held=len(cs["held"]), held=cs["held"], reserve_core_h=RESERVE_CORE_H, projected_core_h=proj,
              cap_core_h=CAP_CORE_H)
    p = C.assert_path(CAP_LOG)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:                                # appended under cap_lock (one writer at a time)
        f.write(json.dumps(ev) + "\n")
    print(f"[g3l] cap: {uid} not claimed ({cs['finished']:.1f} finished + {cs['killed']:.1f} killed + "
          f"{cs['inflight']:.1f} in flight ({len(cs['held'])} held) + {RESERVE_CORE_H:g} = {proj:.1f} > "
          f"{CAP_CORE_H:g} core-h)", flush=True)


def cap_events():
    if not CAP_LOG.is_file():
        return []
    return [json.loads(ln) for ln in CAP_LOG.read_text().splitlines() if ln.strip()]


class LWorker:
    """U-7-7: opt.Worker's claim loop for the G3L units: pick order, opt.claim / release, opt.run_unit; the cap checked
    under cap_lock before every claim (a refusal ends that slot only, U-7-6); no Phase O statistics or STOP files."""

    def __init__(self, D, plan, units, local):
        self.D, self.plan, self.local = D, plan, local
        self.queue = list(units)
        self.i, self.mu, self.halt, self.n_locked, self.n_cap = 0, threading.Lock(), "", 0, 0
        self.running = set()                               # uids this job runs now (heartbeat)

    def next_unit(self):
        with self.mu:
            while self.i < len(self.queue):
                if time.time() - O.T0 > O.STOP_AFTER_H * 3600:
                    return None
                if STOP_L.is_file() and not self.halt:
                    self.halt = f"{STOP_L} exists"
                if self.halt:
                    return None
                u = self.queue[self.i]
                self.i += 1
                if (O.UNITS / u["uid"] / "unit.json").is_file():
                    continue
                with cap_lock():
                    cs = cap_state(self.D, self.queue)
                    if u["uid"] in cs["held"]:             # a live job runs it (or an empty lock < 1 h old)
                        self.n_locked += 1
                        continue
                    proj = cs["committed"] + RESERVE_CORE_H
                    if proj > CAP_CORE_H:
                        cap_event(u["uid"], cs, proj)
                        self.i -= 1                        # stays first in line: this job's other slots look again
                        self.n_cap += 1                    # after their unit (a short unit may free the room)
                        return None
                    if not O.claim(self.D, u["uid"]):
                        self.n_locked += 1
                        continue
                if (O.UNITS / u["uid"] / "unit.json").is_file():
                    O.release(u["uid"])
                    continue
                self.running.add(u["uid"])
                beat(u["uid"])
                return u
            return None

    def loop(self):
        while True:
            u = self.next_unit()
            if u is None:
                return
            try:
                r = O.run_unit(u, self.D, self.plan, O.PARTB_NPROCS, self.local)
                if r.get("reason") == INHESS_FAIL:         # U-7-5: every unit would fail the same way
                    with self.mu:
                        self.halt = f"{u['uid']}: {INHESS_FAIL}"
                        if not STOP_L.is_file():
                            write_stop_l(self.halt)
            except O.StopAll as e:
                with self.mu:
                    self.halt = str(e)
                    if not STOP_L.is_file():
                        write_stop_l(str(e))
                raise
            finally:
                O.release(u["uid"])
                with self.mu:
                    self.running.discard(u["uid"])


def ready_l(plan):
    """U-7-8: (0, '') or (3, why)."""
    ck = O._rj(O.CHECK)
    if not (ck and ck.get("ok") and ck.get("plan_fingerprint") == plan["fingerprint"]):
        return 3, f"{O.CHECK} missing / not ok / for another opt plan (Part B input proof)"
    if STOP_L.is_file():
        return 3, f"{STOP_L} exists: {STOP_L.read_text().strip()}"
    return 0, ""


def cmd_work():
    record_all()
    D, plan = O.d1(), O.load_plan()
    P, code, why = load_plan_l(plan)
    if not code:
        code, why = ready_l(plan)
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: g3l work: {why}", file=sys.stderr, flush=True)
        return code
    D2 = dict(D, orca=str(write_wrapper(D["orca"])))       # U-7-4: only the ORCA launcher differs from opt's D
    units = P["units"]
    local = Path(os.environ.get("TMPDIR", "/tmp")) / f"d1g3l_{O._jobtag()}_{os.getpid()}"
    W = LWorker(D2, plan, units, local)
    left = [u for u in units if not (O.UNITS / u["uid"] / "unit.json").is_file()]
    k = max(1, min(len(left), O._cpus() // O.SLOT_CORES))
    print(f"[g3l work] {len(units)} units ({len(left)} left), {k} slots x {O.SLOT_CORES} cores, cap {CAP_CORE_H:g} "
          f"core-h, job {O._jobtag()} on {socket.gethostname()}", flush=True)
    rc, hb_stop = 0, threading.Event()
    threading.Thread(target=heartbeat, args=(W, hb_stop), daemon=True).start()        # U-7-6
    try:
        with ThreadPoolExecutor(k) as ex:
            for f in [ex.submit(W.loop) for _ in range(k)]:
                try:
                    f.result()
                except O.StopAll as e:
                    if not STOP_L.is_file():
                        write_stop_l(str(e))
                    rc = 3
    finally:
        hb_stop.set()
        shutil.rmtree(local, ignore_errors=True)
    if rc or STOP_L.is_file():
        return 3
    left = [u["uid"] for u in units if not (O.UNITS / u["uid"] / "unit.json").is_file()]
    with cap_lock():
        cs = cap_state(D2, units)
    capped = bool(left) and cs["committed"] + RESERVE_CORE_H > CAP_CORE_H
    print(f"[g3l work] remaining {len(left)} (held elsewhere this pass {W.n_locked}); committed {cs['committed']:.1f} "
          f"core-h (finished {cs['finished']:.1f}, killed {cs['killed']:.1f}, in flight {cs['inflight']:.1f}), held "
          f"{len(cs['held'])}, cap reached {capped} (this job: cap refusals {W.n_cap}, halt {W.halt or '-'})", flush=True)
    return 0 if (not left or (capped and not cs["held"])) else 2


# ================================================================ report
def cmd_report():
    record_all()
    D, plan = O.d1(), O.load_plan()
    P, code, why = load_plan_l(plan)
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: g3l report: {why}", file=sys.stderr, flush=True)
        return code
    units = P["units"]
    res = {u["uid"]: O._rj(O.UNITS / u["uid"] / "unit.json") for u in units}
    with cap_lock():
        cs = cap_state(D, units)
    held = cs["held"]
    ev = cap_events()
    capped_now = cs["committed"] + RESERVE_CORE_H > CAP_CORE_H
    rows = []
    for u in units:
        r, pc = res[u["uid"]], cs["per"][u["uid"]]
        row = dict(rxn_id=u["rxn_id"], job_id=u["job_id"], n_atoms=u["n_atoms"], pick=u["pick"], dipole=u.get("dipole"),
                   core_h_killed_runs=pc["killed"], n_killed_runs=pc["n_killed"])
        if r is None:
            st = "running" if u["uid"] in held else ("not_run_cap" if ev and capped_now else "missing")
            rows.append(dict(row, status=st, reason="U-7-6 cap" if st == "not_run_cap" else ""))
            continue
        gt, ih = r.get("gate") or {}, r.get("inhess") or {}
        dn, dl, dd = r.get("d_form_g3l") or [], r.get("d_form_label") or [], r.get("dd_form_label") or []
        row.update(status=r["status"], reason=r.get("reason"), core_h=O._f(r.get("core_h")),
                   n_opt_cycles=r.get("n_opt_cycles"), n_imag=gt.get("n_imag"),
                   imag_freqs_cm=json.dumps(gt.get("imag_freqs_cm")), converged_to_label=r.get("converged_to_label"),
                   rmsd_label_A=O._f(r.get("rmsd_label_A")), dform_label_A=O._f(r.get("dform_label_A")))
        for k in (0, 1):
            pr = (r.get("pairs") or [[None, None]] * 2)[k]
            row.update({f"pair_{k + 1}": f"{pr[0]}-{pr[1]}", f"d_form_{k + 1}_g3l": dn[k] if len(dn) > k else None,
                        f"d_form_{k + 1}_label": dl[k] if len(dl) > k else None,
                        f"dd_form_{k + 1}": dd[k] if len(dd) > k else None})
        row.update(max_abs_dd_form=max((abs(x) for x in dd), default=None) if dd else None,
                   same_as_g2_start=r.get("same_as_g2_start"), rmsd_g2_A=O._f(r.get("rmsd_g2_A")),
                   dform_g2_A=O._f(r.get("dform_g2_A")), partition_match=gt.get("partition_match"),
                   formed_pairs_eq_label=gt.get("formed_pairs_eq_label"),
                   imag_mode_forming_share=gt.get("imag_mode_forming_share"), inhess_read=ih.get("read"),
                   n_scf_hessian=ih.get("n_scf_hessian"), de_last_kcal=O._f(r.get("de_last_kcal")),
                   e_optts_eh=r.get("e_optts_eh"), wall_s=r.get("wall_s"), slurm_job=r.get("slurm_job"),
                   finished=r.get("finished"))
        rows.append(row)
    S = pd.DataFrame(rows)
    C.write_atomic(STATUS_CSV, lambda t: S.to_csv(t, index=False))
    fin = [x for x in rows if x["status"] in ("ok", "fail", "error")]
    ok = [x for x in fin if x["status"] == "ok"]
    n_ok, n_fail, n_err = len(ok), sum(x["status"] == "fail" for x in fin), sum(x["status"] == "error" for x in fin)
    conv = sum(bool(x.get("converged_to_label")) for x in ok)
    dd_all = [abs(x[f"dd_form_{k}"]) for x in ok for k in (1, 2) if x.get(f"dd_form_{k}") is not None]
    br = [O._rj(O.UNITS / f"B.{rid}" / "unit.json") for rid in plan["part_b"]]
    br = [x for x in br if x is not None]
    by = dict(Counter(x["status"] for x in rows))
    rep = dict(
        spec="D1G3 U-7 (user decision 2026-10-25 item 7, new, user-approved): G3L pilot", explanatory=True,
        label="설명용", written=O._now(), code=CODE_VERSION, plan=str(PLAN_L), plan_fingerprint=P["fingerprint"],
        rule=P["rule"], n_planned=len(units), n_pool=P.get("n_pool"), n_candidates=P.get("n_candidates"),
        by_status=by, n_finished=len(fin),
        core_h=dict(finished=_stats([x.get("core_h") for x in fin]), ok=_stats([x.get("core_h") for x in ok])),
        ts_gate=dict(n_ok=n_ok, n_fail=n_fail, n_error=n_err, pass_rate=n_ok / (n_ok + n_fail) if n_ok + n_fail else None,
                     pass_rate_finished=n_ok / len(fin) if fin else None,
                     fail_reasons=dict(Counter(x["reason"] for x in fin if x["status"] != "ok"))),
        converged_to_label=dict(n=conv, rate_ok=conv / n_ok if n_ok else None,
                                rate_finished=conv / len(fin) if fin else None,
                                rule="rmsd.same_ts(G3L TS, label TS), 0.10 A / 0.02 A (same level: not affected by U-3b)"),
        forming_bond_abs_diff_A=dict(bonds=_stats(dd_all), per_rxn_max=_stats([x.get("max_abs_dd_form") for x in ok]),
                                     dform_label_A=_stats([x.get("dform_label_A") for x in ok]),
                                     note="d(G3L) - d(label) per label forming bond; dform_label_A = rmsd.same_ts "
                                          "(sorted forming distances)"),
        rmsd_label_A=_stats([x.get("rmsd_label_A") for x in ok]),
        n_opt_cycles=_stats([x.get("n_opt_cycles") for x in fin]),
        inhess=dict(n_read=sum(bool(x.get("inhess_read")) for x in fin),
                    n_scf_hessian=dict(Counter(str(x.get("n_scf_hessian")) for x in fin))),
        cap=dict(cap_core_h=CAP_CORE_H, reserve_core_h=RESERVE_CORE_H, spent_core_h=cs["finished"],
                 killed_core_h_est=cs["killed"], n_killed_runs=sum(c["n_killed"] for c in cs["per"].values()),
                 inflight_core_h=cs["inflight"], committed_core_h=cs["committed"], n_held=len(held),
                 rule="committed = finished unit.json core_h + killed runs (start .. last heartbeat + 120 s, x 8; upper "
                      "estimate) + per held unit max(8, 8 x elapsed); a claim needs committed + 8 <= cap (U-7-6)",
                 hit=bool(ev), n_events=len(ev), events=ev, reached_now=capped_now,
                 over_cap=cs["committed"] > CAP_CORE_H, n_not_run_cap=by.get("not_run_cap", 0)),
        context=dict(note="other reactions (Part B = G2 TS != G1 TS class; G3L = same class): reference values only",
                     label_ts_step_core_h_pilot_mean=LABEL_TS_CORE_H,
                     partB_optts_core_h=_stats([x.get("core_h_optts") for x in br]),
                     partB_n_opt_cycles=_stats([(x.get("attempts") or [{}])[0].get("n_opt_cycles") for x in br]),
                     partB_ts_ok=sum(x.get("status") == "ok" for x in br), partB_n=len(br)),
        complete=not (by.get("missing") or by.get("running")), stop=STOP_L.is_file(),
        stop_reason=STOP_L.read_text().strip() if STOP_L.is_file() else None, decisions=DECISIONS,
        outputs={str(STATUS_CSV): C.sha256(STATUS_CSV)})
    C.write_json(REPORT, rep)
    print(json.dumps({k: rep[k] for k in ("by_status", "ts_gate", "converged_to_label", "complete", "stop")}, indent=1,
                     default=str), flush=True)
    print(f"  core-h finished {rep['core_h']['finished']}; cap committed {cs['committed']:.1f} (killed "
          f"{cs['killed']:.1f}) hit {rep['cap']['hit']}", flush=True)
    if rep["stop"]:
        return 3
    return 0 if rep["complete"] else 2


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd")
    ap.add_argument("--force", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        print(__doc__)
        return 2
    if a.cmd == "plan":
        return cmd_plan(a.force)
    if a.cmd == "work":
        return cmd_work()
    if a.cmd == "report":
        return cmd_report()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
