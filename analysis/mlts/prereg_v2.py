#!/usr/bin/env python3
r"""prereg_v2.py — DEPLOY-V2 (MLTS) Phase R pre-registration (docs/specs/DEPLOY_V2_MLTS.md 9), written BEFORE any
channel result exists and committed by the launcher before K starts. Compute nodes only (jobs/mlt.slurm / jobs/sbx.sh).

  python prereg_v2.py write [--force] [--allow-partial] [--from-results DIR]
        -> results/PREREG_V2.md + results/prereg_v2.json (json md_sha256 = sha256 of the md)
        exit 0 written (or already written and unchanged: never rewritten without --force), 2 not ready (C_tests.json
        missing / incomplete, unless --allow-partial), 3 REFUSED: a channel result file exists (list printed; nothing
        written, also with --force)
  python prereg_v2.py check
        exit 0 iff md + json exist, json md_sha256 = sha256(md) and both are tracked and unchanged in git HEAD, else 3

What is registered (all read or quoted at write time; the md quotes the spec verbatim in '>' blocks)
  1 grade criteria (spec 0-4, verbatim) + the operational reading (sets, point estimates, the model judged)
  2 primary metrics (spec 11 / 15), 3 comparators (dft rung upper bound; lf = the DEPLOY_MODEL deployed model),
  4 the representative / primary-configuration rule (inner validation para MAE oi + cpcm; spec 10), 5 the
  re-training noise band delta_H (DEPLOY_MODEL_W_AMEND1.md is not in the repository: defined from DEPLOY_MODEL_W.md 7
  with a note), 6 the C1 reference rule (min primary, own secondary), 7 disp = D3_int (H-0), 8 'seen so far' =
  structural results only (I, G, Z, FT, A, H counts / metrics + the C-1..C-6 pass flags; zero channel results), 9 the
  STOP rule after K / E, 10 input sha256.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)

CODE_VERSION = "mlts-prereg-1"
REPO = M.REPO
SPEC = REPO / "docs" / "specs" / "DEPLOY_V2_MLTS.md"
SPEC_W = REPO / "docs" / "specs" / "DEPLOY_MODEL_W.md"
AMEND1_NAME = "DEPLOY_MODEL_W_AMEND1.md"
MD = M.RES / "PREREG_V2.md"
JS = M.RES / "prereg_v2.json"
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
DEP_RES = REPO / "analysis" / "deploy" / "results"
PRIMARY9 = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")
GOAL_MAE = ("oi", "cpcm")
GOAL_BEND = ("d1", "elst", "pauli", "oi")
GOAL = dict(mae=0.13, sens=0.7, fpr=0.1)
PARTIAL_FACTOR = 0.5
N_BOOT, BOOT_SEED = 2000, 20261019          # = LADDER evaluate.py N_BOOT, SEED (spec 11: LADDER E1 functions)

# channel result files: their existence means K / E already ran -> refuse (task R)
CHANNEL_GLOBS = (("RES", ("C1_report.json", "C1_*_report.json", "K_report.json", "K_val*.csv", "K_*predictions*.parquet",
                          "E_*", "C2K_report.json")),
                 ("MLT/preds", ("C1_predictions.parquet", "C[123]*_predictions.parquet", "K_*.parquet",
                                "parts*/atoms_mlts.*", "parts*/xgb_mlts.*")),
                 ("MLT/units", ("**/*.parquet", "**/*__k.json", "**/L__*.json")))       # train_k: units/<config>/...

# 'seen so far': phase -> result files (relative to the results dir); explicit selectors where the format is known
PHASES = (("I", ("I_inventory.json",)),
          ("G", ("G_plan.json", "G0_estimate.json", "G_report.json", "G0_estimate_d0.json", "G_report_d0.json",
                 "G4_plan.json", "G4_report.json")),
          ("Z", ("Z_report.json", "Z_DECISION.md")),
          ("FT", ("FT_report.json", "FT_geom.csv", "ft/*_ft1*.json")),
          ("A", ("A2_smoke.json", "A3_report.json", "A4_report.json")),
          ("H", ("H0_report.json", "H1_report.json", "H2_report.json", "H_build_mlts.json", "H_build_min_mlts.json")),
          ("X", ("X_report.json",)),
          ("C", ("C_tests.json", "C1_tests.json", "C6_report.json")))
SELECT = {
    "I_inventory.json": ("summary", "I1"),
    "G_plan.json": ("counts", "n_targets", "hessian_source"),
    "G0_estimate.json": ("predicted_core_h", "per_kind_core_h", "n_planned_engrads", "n_planned_g2", "stop"),
    "G0_estimate_d0.json": ("predicted_core_h", "per_kind_core_h", "stop"),
    "G_report.json": ("complete", "core_h_measured", "counts", "n_structures", "stop"),
    "G_report_d0.json": ("complete", "core_h_measured", "counts", "stop"),
    "G4_plan.json": ("complete", "predicted_core_h"),
    "G4_report.json": ("complete", "core_h_measured", "n_added"),
    "A2_smoke.json": ("pass",),
    "A3_report.json": ("complete", "engine", "arms.*.same_ts_frac", "arms.*.core_h_median", "arms.*.gpu_min_median",
                       "arms.*.n", "arms.*.disk_gb_per_rxn", "disk_gb_per_rxn"),
    "A4_report.json": ("complete", "folds.*.complete", "n_ok", "n_fail", "fail_reasons"),
    "H0_report.json": ("n", "median_abs_diff", "max_abs_diff", "decision"),
    "H1_report.json": ("pass", "n", "gates", "tol"),
    "H2_report.json": ("n_with_ts", "n_fail", "rate", "stop", "by_stage"),
    "H_build_mlts.json": ("refs", "n_targets", "n_usable", "n_tok", "n_x182", "n_h", "n_d3", "fail_stage"),
    "H_build_min_mlts.json": ("refs", "n_usable", "n_h_changed", "references.n_fragments_replaced"),
    "C_tests.json": ("complete", "all_pass", "C1.pass", "C2.pass", "C3.pass", "C4.pass", "C5.pass", "C6.pass",
                     "C7.pass", "stand_in"),
    "C1_tests.json": ("models.*.pass_C1", "models.*.pass_C4"),
    "C6_report.json": ("passed", "max_abs_yhat", "tol_kcal", "n_units"),
}
SELECT_PAT = ((re.compile(r".+_ft1(_.+)?\.json"), ("run", "which", "sets.*.n", "sets.*.force_mae_kcal_A.ts_near",
                                                   "sets.*.rel_e.ts_near.mae_kcal", "sets.*.meets")),)
SKIP_FILES = re.compile(r"BENCH_.*")                # finetune timing / smoke runs, never used downstream
SKIP_KEYS = {"rows", "units", "trace", "traceback", "per_engrad", "problems", "not_ok", "fmax_top20", "rxn", "sample",
             "pairs", "overlaps", "decisions", "outputs", "inputs", "versions", "code", "files", "detail", "rule",
             "definition", "note", "notes", "summary_text", "C7"}
CHANNEL_WORDS = ("yhat", "para_mae", "mae_para", "ppv", "auc", "bend", "predictions", "y_hat")
MAX_GENERIC = 40


# ================================================================ helpers
def now():
    return M.now()


def rj(p):
    try:
        return M.read_json(p)
    except Exception:                                               # noqa: BLE001
        return None


def git(*a):
    try:
        r = subprocess.run(["git", "-C", str(REPO)] + list(a), capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip()
    except Exception as e:                                          # noqa: BLE001
        return 99, str(e)


def fmt(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        return f"{v:.4g}"
    if isinstance(v, (int, str)):
        s = str(v)
        return s if len(s) <= 120 else s[:117] + "..."
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= 160 else s[:157] + "..."


def get_path(J, path):
    """'a.*.b' -> {'a.<k>.b': value} (wildcard over dict keys)."""
    out = {}

    def rec(o, parts, pre):
        if not parts:
            out[pre] = o
            return
        p = parts[0]
        if p == "*" and isinstance(o, dict):
            for k, v in o.items():
                rec(v, parts[1:], f"{pre}.{k}" if pre else str(k))
        elif isinstance(o, dict) and p in o:
            rec(o[p], parts[1:], f"{pre}.{p}" if pre else p)
    rec(J, path.split("."), "")
    return out


def channel_key(k):
    kl = k.lower()
    return any(w in kl for w in CHANNEL_WORDS)


def flatten(o, pre="", depth=0, out=None):
    """generic structural flatten: scalars / short scalar lists, depth <= 3, skip SKIP_KEYS and channel-like keys."""
    out = {} if out is None else out
    if len(out) >= MAX_GENERIC:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            if str(k) in SKIP_KEYS or channel_key(str(k)):
                continue
            key = f"{pre}.{k}" if pre else str(k)
            if isinstance(v, dict) and depth < 3:
                flatten(v, key, depth + 1, out)
            elif isinstance(v, (int, float, bool, str)) or v is None:
                out[key] = v
            elif isinstance(v, list) and len(v) <= 8 and all(isinstance(x, (int, float, bool, str)) for x in v):
                out[key] = v
            if len(out) >= MAX_GENERIC:
                break
    return out


def file_summary(path):
    name = path.name
    rec = dict(path=str(path), sha256=M.sha256(path))
    if path.suffix == ".json":
        J = rj(path)
        if J is None:
            rec["values"] = {"error": "unreadable json"}
            return rec
        sel = SELECT.get(name) or next((s for rx, s in SELECT_PAT if rx.fullmatch(name)), None)
        vals = {}
        if sel:
            for p in sel:
                vals.update(get_path(J, p))
        else:
            vals = flatten(J)
        rec["values"] = {k: v for k, v in vals.items() if not (sel is None and channel_key(k))}
    elif path.suffix == ".csv":
        with open(path) as fh:
            head = fh.readline().strip()
            n = sum(1 for _ in fh)
        rec["values"] = dict(n_rows=n, columns=head[:200])
    else:
        txt = path.read_text(errors="replace").splitlines()
        rec["values"] = dict(first_line=(txt[0] if txt else "")[:160], n_lines=len(txt))
    return rec


def spec_lines(text, start, stop):
    lines = text.splitlines()
    i = next((k for k, ln in enumerate(lines) if re.match(start, ln)), None)
    if i is None:
        raise SystemExit(f"spec block {start!r} not found")
    j = next((k for k in range(i + 1, len(lines)) if re.match(stop, lines[k])), len(lines))
    while j > i and not lines[j - 1].strip():
        j -= 1
    return lines[i:j]


def quote(lines):
    return "\n".join(("> " + ln) if ln.strip() else ">" for ln in lines)


# ================================================================ guard
def channel_files(src=None):
    """channel result files under results/ (and the --from-results folder), $MLT/preds, $MLT/units."""
    roots = dict(RES=[M.RES] + ([Path(src)] if src else []), **{"MLT/preds": [M.MLT / "preds"],
                                                                 "MLT/units": [M.MLT / "units"]})
    found = []
    for r, pats in CHANNEL_GLOBS:
        for d in roots[r]:
            if not d.is_dir():
                continue
            for pat in pats:
                found += [str(p) for p in sorted(d.glob(pat)) if p.is_file() and p.name != "C1_representatives.json"]
    return sorted(set(found)), [f"{r}/{p}" for r, pats in CHANNEL_GLOBS for p in pats]


# ================================================================ content
def seen_so_far(src):
    out = []
    for ph, pats in PHASES:
        files = []
        for pat in pats:
            files += sorted(p for p in src.glob(pat) if p.is_file() and not SKIP_FILES.match(p.name))
        out.append(dict(phase=ph, files=[file_summary(p) for p in files],
                        missing=[pat for pat in pats if not any(src.glob(pat))]))
    return out


def build(o, src):
    spec_txt = SPEC.read_text()
    w_txt = SPEC_W.read_text()
    q04 = spec_lines(spec_txt, r"^### 0-4\.", r"^(### |## |---)")
    q09 = spec_lines(spec_txt, r"^## 9\.", r"^## ")
    q10 = spec_lines(spec_txt, r"^## 10\.", r"^## ")
    q11 = spec_lines(spec_txt, r"^## 11\.", r"^## ")
    q14 = spec_lines(spec_txt, r"^## 14\.", r"^## ")
    q15 = spec_lines(spec_txt, r"^## 15\.", r"^(## |---)")
    qH = spec_lines(w_txt, r"^\*\*1차 지표 H", r"^\*\*채택 규칙")
    qWfam = spec_lines(w_txt, r"^\*\*평가 대상 계열", r"^\*\*1차 지표 H")
    amend = [str(p) for p in (REPO / "docs").rglob(AMEND1_NAME)] + [str(p) for p in (REPO / "analysis").rglob(AMEND1_NAME)]
    for wt in sorted(REPO.parent.glob(REPO.name.split("-mlts")[0] + "*")):            # sibling worktrees: docs/ only
        if wt.resolve() != REPO.resolve() and (wt / "docs").is_dir():
            amend += [str(p) for p in (wt / "docs").rglob(AMEND1_NAME)]
    rc_g, out_g = git("log", "--all", "--format=%h", "--", f"*{AMEND1_NAME}")
    amend_git = sorted(set(out_g.split())) if rc_g == 0 else [f"git log failed rc {rc_g}"]
    h0 = rj(src / "H0_report.json") or {}
    c1reps = M.MLT / "preds" / "C1_representatives.json"
    reps = rj(c1reps) or {}
    rep8 = (reps.get("REP8") or {}).get("dft") or {}
    seen = seen_so_far(src)
    guard_found, guard_globs = channel_files(src)
    head = git("rev-parse", "HEAD")[1]
    inputs = {}
    for p in [SPEC, SPEC_W, src / "DECISIONS.md", HERE / "PIPELINE.md", M.S_FOLDS, LAD_RES / "E1_report.json",
              LAD_RES / "L_predictions.parquet", LAD_RES / "E1_goal.csv", DEP_RES / "K_predictions.parquet",
              DEP_RES / "w" / "D_decision.json", src / "H0_report.json", c1reps,
              REPO / "analysis" / "d1_g3" / "results" / "E_noise.csv"]:
        if Path(p).is_file():
            inputs[str(p)] = M.sha256(p)
    for s in seen:
        for f in s["files"]:
            inputs[f["path"]] = f["sha256"]
    crit = dict(
        grades=[dict(grade="목표 달성 (goal)", rule=f"para MAE <= {GOAL['mae']} for {list(GOAL_MAE)} AND bend sensitivity "
                                                  f">= {GOAL['sens']} and false-positive rate <= {GOAL['fpr']} for "
                                                  f"{list(GOAL_BEND)}"),
                dict(grade="부분 개선 (partial)", rule=f"para MAE_MLTS <= {PARTIAL_FACTOR} x MAE_lf for all 9 targets "
                                                       f"{list(PRIMARY9)} (same pairs)"),
                dict(grade="실패 (fail)", rule="otherwise")],
        grade_order="goal checked first, then partial, else fail",
        estimate="point estimates (LADDER 13-2 convention); 'surely' flag when the core-bootstrap 95 % upper (MAE) / "
                 "lower (sensitivity) / upper (FPR) bound also meets the threshold",
        set="D1 T1 test pairs, P_main, pos para (LADDER pair definition), common to MLTS primary, dft rung and lf",
        judged_model="MLTS primary configuration (rule 4), REP8 per target; disp = D3_int (H-0)",
        bends="LADDER E1 functions (evaluate.bends_e1 -> d1_eval bend_rates, families 'all'); hidden bends = deploy "
              "eval_tf.hidden / hidden_rates on the same bends_e1 families (P_main reference para families >= 12 "
              "points incl. H; DFT truth: barrier straight and >= 1 of the 8 channels bent; prediction: the same rule "
              "on the predicted calls), dft / lf / MLTS on the same families (DEPLOY's own lf number used its 89 "
              "P_sens families: not the comparator)",
        primary_metrics=["grade", "oi para MAE", "cpcm para MAE", "hidden-bend PPV"],
        secondary_metrics=["9-target para MAE + core bootstrap 95 % CI (2,000, seed 20261019)",
                           "per-channel bend sensitivity / FPR", "hidden-bend sensitivity / PPV", "continuous G AUC",
                           "breakdowns: same-TS vs not (A-3/A-4), dipole, g1 shape class"],
        comparators=dict(dft="LADDER dft rung (label TS inputs) REP8, analysis/d1_ladder/results/L_predictions.parquet "
                             "rung dft (upper bound)",
                         lf="DEPLOY_MODEL deployed model = lf (S / TF, direct readout; DEPLOY_MODEL_W D_decision "
                            "adopted lf), 5-fold predictions analysis/deploy/results/K_predictions.parquet",
                         descriptive=["C1", "C2", "C3"]),
        representative=dict(within_config="C1 = LADDER REP / REP8 unchanged (C1_representatives.json); C2 / C3 = "
                                          "LADDER 13-1 on their own 5-fold mean validation kcal MAE (ties AW -> GNN -> "
                                          "TF -> XGB), REP8 over both arms (ties SH before S, then model order)",
                            primary_config="min over {C1, C2, C3} of inner-validation para MAE(oi) + MAE(cpcm), REP8 "
                                           "models, pooled over the 5 folds' inner_val pairs in P_main para; ties C1 "
                                           "-> C2 -> C3; test pairs never used",
                            c1_inner_val="C1 = the fold-k LADDER dft unit's chosen tuning fit re-run (its cell, seed, "
                                         "inner_train, early stopping on the dft inner_val) and applied to the MLTS "
                                         "inputs of those inner_val pairs (the saved final models were trained on "
                                         "the inner_val pairs: in-sample, reported only); GPU training is not "
                                         "bit-reproducible across devices (recorded, train_k K-5)"),
        delta_H=dict(amend1_found=amend, amend1_git_commits=amend_git, definition=(
            "H = mean_t mean_f [r_ft / s_t] exactly as DEPLOY_MODEL_W.md 7 (para families of hammett_eval."
            "family_series with >= 12 points incl. H; r_ft = RMS residual of the least-squares line of e = yhat - y "
            "on sigma through the points + (0, 0); s_t = SD of the DFT y over the evaluation families); C2 trained 3 "
            "times (seed set s = 0, 1, 2: s = 0 the C2 unit itself, s = 1, 2 with tuning and final seeds + 1000 s); "
            "delta_H = max_s H_s - min_s H_s (absolute), delta_H_rel = delta_H / mean_s H_s; an H difference (or "
            "|R - 1| of an H ratio) within delta_H (delta_H_rel) is 'within re-training noise' (descriptive, not a "
            "grade criterion); per target delta_MAE_t = max_s - min_s of the para MAE, reported; XGB units are seed "
            "free (subsample = colsample = 1): a target whose REP8 is XGB contributes 0 to delta_H by construction")),
        refs=dict(primary="min", secondary="own",
                  implementation="train_k.py c1pred (predict_c1 on mlts_min.parquet -> C1_predictions; on "
                                 "mlts.parquet -> C1_own_predictions, descriptive); C2 / C3 / C1v use the K_plan refs "
                                 "= this primary"),
        disp=dict(prediction="D3_int", h0_median_abs_diff_kcal=h0.get("median_abs_diff"), h0_decision=h0.get("decision"),
                  model_disp="reported (secondary)"),
        stop="K / E results suggest the registration should change -> STOP (spec 14), registration not edited")
    # ---------------------------------------------------------------- markdown
    L = []
    L.append(f"# PREREG V2 — DEPLOY-V2 (MLTS) 채널 모델 사전 등록 ({now()[:10]})")
    L.append("")
    L.append(f"이 문서는 DEPLOY-V2 (MLTS)의 사전 등록이다(`docs/specs/DEPLOY_V2_MLTS.md` 9절, sha256 `{M.sha256(SPEC)}`). "
             f"`analysis/mlts/prereg_v2.py`({CODE_VERSION})가 결과 파일만 읽어 만들었다(git HEAD `{head}`, 작성 "
             f"{now()}). 인용 블록 = 명령서 원문. **K(K-C1 · K-C2C3)는 이 md와 `prereg_v2.json`이 commit되고 바뀌지 않았을 "
             f"때만 시작한다**(launcher commit_gate, `prereg_v2.py check`). 채널 결과 파일 검사: "
             f"{'없음' if not guard_found else guard_found} (검사한 경로: {', '.join(guard_globs)}).")
    L.append("")
    L.append("## 1. 등급 기준 (명령서 0-4, 원문)")
    L.append(quote(q04))
    L.append("")
    L.append("판정 세부(위 원문을 계산 규칙으로 옮긴 것, 원문이 우선):")
    L.append(f"- 집합: D1 T1 시험 짝 중 P_main · para(pos p). 세 대상(MLTS 1차 · dft rung · lf) 모두 예측이 있는 공통 짝에서 "
             f"계산한다. P_main para 시험 짝 중 MLTS 예측이 있는 비율(coverage)을 함께 적는다.")
    L.append("- 판정 모델: 4절 규칙으로 고른 MLTS 1차 설정의 REP8(타깃마다 두 arm 중 대표). disp 예측값 = D3_int (7절, H-0).")
    L.append("- 숨은 꺾임(1차 지표 PPV): 아래 꺾임과 같은 bends_e1 계열(P_main reference para, H 포함 12점 이상)에서 "
             "deploy `eval_tf.hidden` 규칙(DFT 기준 barrier 곧음 · 8채널 중 하나 이상 꺾임; 예측은 예측 판정에 같은 규칙)으로 "
             "센다. dft · lf · MLTS를 같은 계열에서 비교한다(DEPLOY 보고의 lf 값은 P_sens 89계열이라 비교 대상이 아님).")
    L.append(f"- 점 추정으로 판정한다(LADDER 13-2 규약). core 부트스트랩 95 % CI({N_BOOT}회, seed {BOOT_SEED})의 바깥쪽 끝"
             f"까지 기준을 만족하면 '확실히'로 따로 적는다.")
    L.append(f"- **목표 달성**: oi · cpcm para MAE ≤ {GOAL['mae']} kcal/mol **그리고** d1 · elst · pauli · oi 각각 꺾임 민감도 "
             f"≥ {GOAL['sens']}, 위양성률 ≤ {GOAL['fpr']}. 꺾임 = LADDER E1 함수(`analysis/d1_ladder/evaluate.py` "
             f"bends_e1 → bend_rates, 'all' 계열). LADDER 13-2는 꺾임 기준을 oi · cpcm에 걸었지만 이 명령서 0-4 원문은 "
             f"d1 · elst · pauli · oi를 적었으므로 원문을 따른다(cpcm 꺾임은 보고만).")
    L.append(f"- **부분 개선**: 9개 타깃({', '.join(PRIMARY9)}) 모두 같은 짝에서 MAE_MLTS ≤ {PARTIAL_FACTOR} × MAE_lf "
             f"(점 추정).")
    L.append("- **실패**: 그 외. 판정 순서: 목표 달성 → 부분 개선 → 실패(처음 맞는 것).")
    L.append("")
    L.append("## 2. 1차 지표 (명령서 11 · 15절, 원문)")
    L.append(quote(q11))
    L.append(">")
    L.append(quote(q15))
    L.append("")
    L.append("- 1차 지표 = 등급, oi · cpcm para MAE, 숨은 꺾임 PPV(15절 맨 위 한 줄). 나머지(9개 타깃 MAE와 CI, 채널별 "
             "꺾임 민감도 · 위양성률, 숨은 꺾임 민감도, 연속 G AUC, 분해 (i)–(iii))는 2차 · 설명용이다.")
    L.append("")
    L.append("## 3. 비교 대상")
    L.append("- **dft rung (상한)**: LADDER dft 단위를 라벨 TS 입력에 쓴 예측(`analysis/d1_ladder/results/L_predictions.parquet` "
             "rung dft), REP8. 같은 짝.")
    L.append("- **lf (현재 배포 모델)**: DEPLOY_MODEL의 lf(S · TF, direct 읽기; DEPLOY_MODEL_W D 판정에서 lfW가 채택되지 "
             "않아 배포 모델 = lf). 5-fold 예측 `analysis/deploy/results/K_predictions.parquet`. 같은 짝.")
    L.append("- C1 · C2 · C3 전부는 설명용으로 같은 표에 적는다(명령서 11절).")
    L.append("")
    L.append("## 4. 대표 모델 고르는 규칙 (안쪽 검증; 명령서 10절 원문)")
    L.append(quote(q10))
    L.append("")
    L.append("- 설정 안: C1 = LADDER가 고른 REP / REP8 그대로(`$MLT/preds/C1_representatives.json`, sha256은 10절). "
             "C2 · C3 = LADDER 13-1 규칙(바깥 fold 5개 평균 검증 kcal MAE 최저, 동점 AW → GNN → TF → XGB), REP8 = 두 arm 중"
             "(동점 SH → S, 그다음 모델 순서; eval_geo2 규칙).")
    L.append("- 1차 설정: C1 · C2 · C3 중 **안쪽 검증 para MAE(oi) + MAE(cpcm)**(각 설정의 REP8 모델, 5 fold의 inner_val 짝 중 "
             "P_main para를 모두 모은 값)가 가장 작은 것. 동점이면 C1 → C2 → C3. C1의 안쪽 검증 = fold k dft 단위가 고른 "
             "튜닝 fit(그 셀 · seed · inner_train, dft inner_val 조기 종료)을 다시 학습해 같은 inner_val 짝의 MLTS 입력에 "
             "넣은 예측이다(저장된 최종 모델은 inner_val 짝으로 학습되어 표본 안이므로 보고만 한다; GPU 학습은 장치가 "
             "다르면 비트 단위로 재현되지 않아 그 차이를 함께 적는다). 시험 짝은 쓰지 않는다. 결과는 "
             "`results/K_report.json` {primary, inner_val}에 적힌다.")
    if rep8:
        L.append("- 참고(구조 정보, 채널 결과 아님): LADDER dft REP8 = " +
                 ", ".join(f"{t} {'/'.join(rep8.get(t, []))}" for t in PRIMARY9 if t in rep8) + ".")
    L.append("")
    L.append("## 5. 재학습 잡음 대역 δ_H")
    L.append(f"- 명령서 10절은 δ_H를 `{AMEND1_NAME}`의 정의로 계산하라고 한다. **이 파일은 저장소에 없다**(이 worktree의 "
             f"docs/ · analysis/ 전체, 같은 저장소 다른 worktree의 docs/, 모든 git ref의 이력을 {now()[:10]}에 찾음; 찾은 "
             f"파일: {amend or '없음'}, commit: {amend_git or '없음'}). 그래서 아래 정의를 DEPLOY_MODEL_W.md "
             f"7절(H)에서 만들어 등록한다. 원래 문서가 나오면 다르더라도 이 등록을 바꾸지 않고 차이를 보고한다.")
    L.append("- DEPLOY_MODEL_W.md 7절 원문(평가 계열과 H):")
    L.append(quote(qWfam + [""] + qH))
    L.append("- 정의: C2를 seed 묶음 3개로 학습한다(s = 0은 C2 단위 그 자체, s = 1, 2는 튜닝 seed와 최종 seed에 1000 × s를 "
             "더함 — DEPLOY null 검사와 같은 규약). 각 s의 시험 예측으로 위 H를 같은 계열 · 같은 점 · 같은 s_t로 계산해 H_s를 "
             "얻는다. **δ_H = max_s H_s − min_s H_s**, 상대값 δ_H,rel = δ_H / 평균 H_s.")
    L.append("- 쓰임: 두 모델의 H 차이(또는 H 비 R의 |R − 1|)가 δ_H(δ_H,rel) 안이면 '재학습 잡음 안'으로 적는다(설명용, 등급 판정에 "
             "쓰지 않음). 타깃별로 para MAE의 max − min(δ_MAE,t)도 보고한다.")
    L.append("- XGB 단위는 seed가 결과를 바꾸지 않으므로(subsample = colsample = 1) 한 번만 학습해 s1 · s2에 복사한다. REP8이 "
             "XGB인 타깃은 정의상 δ_H · δ_MAE에 0을 더한다.")
    L.append("")
    L.append("## 6. C1 참조 반응물 (refs)")
    L.append("- **1차 = min**, 2차 = own(둘 다 보고). min = LADDER g1 · g2 rung 규칙(P6 + P20 (c)): Q-2 키마다 이 입력 집합의 "
             "HF 기준 반응물 에너지 최저값, 구속 dipole은 자기 것(`inputs_mlts.py build mlts --refs min` → "
             "`$MLT/inputs/mlts_min.parquet`).")
    L.append("- 이유: C1의 LADDER dft 단위는 refs_cr2로 다시 잡은 H 입력(own 15,233 · Q 3,176 · V 123 조각, 3,131반응의 H 변경)으로 "
             "학습됐다. 라벨 수준 DFT가 필요한 cr2는 배포에서 쓸 수 없고, 그중 가장 큰 교체인 Q(같은 분자의 최저 기준)를 라벨 "
             "없이 흉내 내는 것이 min이다. 그래서 min이 학습 입력 분포에 가장 가깝고(SH arm의 H · d1 · d2 · barrier 입력의 "
             "학습/배포 차이가 작음), LADDER g1 · g2 rung과도 같은 규칙이라 비교할 수 있다. own은 반응 하나만으로 정해져 "
             "배포 묶음 구성에 의존하지 않으므로 2차로 함께 보고한다.")
    L.append("- 실행: `train_k.py c1pred`가 predict_c1을 min 표(`mlts_min.parquet`)로 돌려 `C1_predictions.parquet` · "
             "`results/C1_report.json`(refs 기록)을, own 표로 돌려 `C1_own_*`(설명용)을 만든다. E는 C1 보고의 refs가 이 "
             "등록과 다르면 STOP한다.")
    L.append("- C2 · C3는 학습과 시험에 같은 refs 모드를 쓰고, 그 모드는 시험 결과를 보기 전에 `results/K_plan.json`에 "
             "적는다(추천 = min: C1 1차와 같은 입력 표라 설정끼리 같은 입력으로 비교된다). 시험 결과를 보고 모드를 바꾸면 "
             "등록 변경 → STOP.")
    L.append("")
    L.append("## 7. disp = D3_int (H-0)")
    L.append(f"- H-0(라벨 TS {h0.get('n')}반응): |D3_int − 라벨 disp| 중앙값 {fmt(h0.get('median_abs_diff'))} kcal/mol, "
             f"최대 {fmt(h0.get('max_abs_diff'))} ≤ 0.01 → 결정 `{h0.get('decision')}`. disp 예측값은 모델 대신 "
             "D3_int(짝 차이 D3_int(a) − D3_int(b))로 하고, 모델의 disp도 E에서 함께 보고한다.")
    L.append("")
    L.append("## 8. 지금까지 본 것 (구조 지표만, 채널 결과 0개)")
    L.append(f"읽은 결과 폴더: `{src}`. 채널 결과 파일(K · E 예측과 지표)은 없다(위 검사). C-7(입력 민감도)은 라벨 없이 예측 "
             "차이만 담고 오차를 계산하지 않으므로 채널 결과가 아니며, 여기에는 통과 여부만 적는다.")
    L.append("")
    L.append("| 단계 | 파일 | sha256 | 값 |")
    L.append("|---|---|---|---|")
    for s in seen:
        if not s["files"]:
            L.append(f"| {s['phase']} | (없음: {', '.join(s['missing'])}) | | |")
        for f in s["files"]:
            vals = "; ".join(f"{k} = {fmt(v)}" for k, v in (f.get("values") or {}).items()) or "—"
            L.append(f"| {s['phase']} | `{Path(f['path']).name}` | `{f['sha256'][:12]}` | {vals.replace('|', '/')} |")
    L.append("")
    L.append("## 9. 사전 등록 절차와 STOP (명령서 9 · 14절 원문)")
    L.append(quote(q09))
    L.append(">")
    L.append(quote(q14))
    L.append("")
    L.append("- K · E 결과를 본 뒤 이 등록을 바꿔야 할 것 같으면 바꾸지 않고 STOP한다(`E_report.json` prereg_change_needed).")
    L.append("")
    L.append("## 10. 입력 sha256")
    for k, v in sorted(inputs.items()):
        L.append(f"- `{k}`: `{v}`")
    md = "\n".join(L) + "\n"
    J = dict(version=CODE_VERSION, written=now(), git_head=head, spec=str(SPEC.relative_to(REPO)), spec_sha256=M.sha256(SPEC),
             spec_w=str(SPEC_W.relative_to(REPO)), spec_w_sha256=M.sha256(SPEC_W), md=str(MD), results_source=str(src),
             criteria=crit, seen_so_far=seen, channel_guard=dict(globs=guard_globs, found=guard_found),
             inputs_sha256=inputs, quoted=dict(spec_0_4=q04, spec_10=q10, spec_11=q11, spec_14=q14, spec_w_H=qH))
    return md, J


# ================================================================ commands
def cmd_write(o):
    src = Path(o.from_results) if o.from_results else M.RES
    if MD.is_file() and not o.force:
        J = rj(JS) or {}
        if J.get("md_sha256") == M.sha256(MD):
            print(f"[prereg] {MD} exists and matches {JS.name} (not rewritten; --force to rewrite)")
            return 0
        print(f"[prereg] {MD} exists but {JS.name} is missing or its md_sha256 differs: refusing to overwrite without "
              "--force", file=sys.stderr)
        return 3
    found, globs = channel_files(src)
    if found:
        print("REFUSED: channel result files exist (pre-registration must precede K / E):\n  " + "\n  ".join(found),
              file=sys.stderr)
        return 3
    C = rj(src / "C_tests.json")
    if not o.allow_partial and not (C and C.get("complete")):
        M.not_ready(f"{src / 'C_tests.json'} missing or incomplete (Phase C first; --allow-partial for a test)")
    md, J = build(o, src)
    found2, _ = channel_files(src)                                   # nothing appeared while building
    if found2:
        print("REFUSED: channel result files appeared:\n  " + "\n  ".join(found2), file=sys.stderr)
        return 3
    M.write_text(MD, md)
    J["md_sha256"] = M.sha256(MD)
    M.write_json(JS, J)
    print(f"[prereg] wrote {MD} ({len(md.splitlines())} lines, sha256 {J['md_sha256'][:12]}) + {JS}")
    return 0


def cmd_check(o):
    out = dict(md=str(MD), json=str(JS))
    J = rj(JS)
    out["exists"] = MD.is_file() and J is not None
    out["md_sha_ok"] = bool(out["exists"] and J.get("md_sha256") == M.sha256(MD))
    git_ok = True
    for p in (MD, JS):
        rc1, _ = git("ls-files", "--error-unmatch", str(p))
        rc2, _ = git("diff", "--quiet", "HEAD", "--", str(p))
        out[f"git_{p.name}"] = dict(tracked=rc1 == 0, clean=rc2 == 0)
        git_ok &= rc1 == 0 and rc2 == 0
    out["git_ok"] = git_ok
    ok = out["exists"] and out["md_sha_ok"] and git_ok
    print(json.dumps(out, indent=1))
    return 0 if ok else 3


def main(argv=None):
    ap = argparse.ArgumentParser(prog="prereg_v2.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("write")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--from-results", default=None, help="read the structural results from DIR (test; default results/)")
    sub.add_parser("check")
    o = ap.parse_args(argv)
    if o.cmd == "write":
        return cmd_write(o)
    if o.cmd == "check":
        return cmd_check(o)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
