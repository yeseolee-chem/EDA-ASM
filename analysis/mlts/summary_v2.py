#!/usr/bin/env python3
r"""summary_v2.py — DEPLOY-V2 (MLTS) Phase S (docs/specs/DEPLOY_V2_MLTS.md 15): results/SUMMARY_V2.md (Korean, the
style of the earlier SUMMARY files) from the phase reports, then results/STOP_S.md ('명령서 S: 멈춤').
reactot env (pandas); reads only.

  python summary_v2.py write [--stop-file auto|yes|no]
        needs results/E_report.json, W_report.json, P_report.json, X_report.json complete (else exit 2). Optional:
        FT_report.json, A3_report.json, A4_report.json, PREREG_V2.md, analysis/d1_g2/results/C_compare.csv (xTB row),
        analysis/d1_g3/results/G3L_report.json (G3L row). --stop-file auto (default): STOP_S.md is written unless the
        command runs as the launcher's S job (SLURM_JOB_NAME mltp_*): pipeline.py decide_s writes it after its commit,
        and a STOP_S.md present earlier makes the next launcher pass exit before that commit (jobs/pipeline.slurm step 1).

Sections: top line (grade + oi / cpcm para MAE + hidden-bend PPV), 1 structure table (xTB(g2) / MLTS / G3L / label:
|dd|, RMSD, same-TS fraction, cost per reaction), 2 channel table (dft upper bound / lf / MLTS: 9 targets para MAE,
bend sensitivity / FPR), 3 configurations (C1 / C2 / C3, primary rule), 4 breakdowns, 5 W / P, 6 limits (shape
mismatch, LODO, deployment cost), 7 inputs.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402

CODE_VERSION = "mlts-s-1"
REPO = M.REPO
SUMMARY = M.RES / "SUMMARY_V2.md"
STOP_S = M.RES / "STOP_S.md"
G3L = REPO / "analysis" / "d1_g3" / "results" / "G3L_report.json"
G2_STATUS = REPO / "analysis" / "d1_g2" / "results" / "G_status.csv"
PRIMARY9 = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")
TLAB = dict(d1="d1", d2="d2", elst="elst", pauli="Pauli", oi="oi", disp="disp", cpcm="cpcm", cds="cds",
            barrier="barrier")
GRADE_KO = {"goal": "목표 달성", "partial": "부분 개선", "fail": "실패"}
LABEL_TS_CORE_H = 8.2                       # spec 0-2: label-level DFT TS per reaction


def rj(p):
    try:
        return M.read_json(p)
    except Exception:                                               # noqa: BLE001
        return None


def f(x, nd=3):
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "—"
    if v != 0 and abs(v) < 10 ** (-nd):
        return f"{v:.1e}"
    return f"{v:.{nd}f}"


def pct(x):
    return "—" if x is None else f"{100 * float(x):.1f} %"


def need(name):
    J = rj(M.RES / name)
    if not J or J.get("complete") is not True:
        M.not_ready(f"results/{name} missing or incomplete")
    return J


def xtb_row():
    """G2 (autodE + XTB2 ALPB water) vs the label TS from C_compare.csv (same_ts_dft, form_diff_dft_max,
    rmsd_heavy_dft) + G_status core-h."""
    import pandas as pd
    out = dict(n=None)
    if M.C_COMPARE.is_file():
        C = pd.read_csv(M.C_COMPARE, usecols=["rxn_id", "g2_status", "same_ts_dft", "form_diff_dft_max",
                                             "rmsd_heavy_dft"])
        ok = C[C.g2_status.astype(str) == "ok"]
        same = ok.same_ts_dft.astype(str).str.lower() == "true"
        out = dict(n=int(len(C)), n_ok=int(len(ok)), dd_median=float(pd.to_numeric(ok.form_diff_dft_max,
                                                                                   errors="coerce").median()),
                   rmsd_median=float(pd.to_numeric(ok.rmsd_heavy_dft, errors="coerce").median()),
                   same_frac=float(same.sum()) / 9266)
    if G2_STATUS.is_file():
        S = pd.read_csv(G2_STATUS, usecols=["status", "core_h_total"])
        out["core_h"] = float(S[S.status == "ok"].core_h_total.median())
    return out


def g3l_row():
    J = rj(G3L) or {}
    fb = (J.get("forming_bond_abs_diff_A") or {}).get("per_rxn_max") or {}
    return dict(n=fb.get("n"), dd_median=fb.get("median"), rmsd_median=(J.get("rmsd_label_A") or {}).get("median"),
                same=(J.get("converged_to_label") or {}).get("rate_ok"),
                core_h=((J.get("core_h") or {}).get("ok") or {}).get("median"))


def rates_of(R, key):
    r = ((R.get("rates") or {}).get(key) or {}).get("all") or {}
    return {t: (r.get(t) or {}) for t in PRIMARY9}


def build():
    E = need("E_report.json")
    W = need("W_report.json")
    P = need("P_report.json")
    X = need("X_report.json")
    BR = rj(M.RES / "E_bend_rates.json") or {}
    FT = rj(M.RES / "FT_report.json") or {}
    A3 = rj(M.RES / "A3_report.json") or {}
    A4 = rj(M.RES / "A4_report.json") or {}
    pk = E["primary_key"]
    prim = E["primary"]
    pm = E.get("para_mae") or {}
    hd = E.get("headline") or {}
    gd = E.get("grade_detail") or {}
    geo = (E.get("geometry") or {}).get("main") or {}
    L = []
    g = E["grade"]
    L.append(f"# DEPLOY-V2 (MLTS) 요약 — 라벨 수준 TS를 미세조정 MLIP로 얻어 8채널 예측 ({M.now()[:10]})")
    L.append("")
    L.append(f"**등급: {GRADE_KO.get(g, g)} ({g})** · oi para MAE **{f(hd.get('oi_para_mae'))}** "
             f"(CI {f((hd.get('oi_ci') or [None])[0])}–{f((hd.get('oi_ci') or [None, None])[1])}) · cpcm para MAE "
             f"**{f(hd.get('cpcm_para_mae'))}** (CI {f((hd.get('cpcm_ci') or [None])[0])}–"
             f"{f((hd.get('cpcm_ci') or [None, None])[1])}) kcal/mol · 숨은 꺾임 PPV **{f(hd.get('hidden_bend_ppv'), 2)}**"
             f" (민감도 {f(hd.get('hidden_bend_sens'), 2)}, para {hd.get('hidden_n_fam')}계열)")
    L.append("")
    L.append(f"- 명령서 `docs/specs/DEPLOY_V2_MLTS.md`, 결정 기록 `results/DECISIONS.md`, 사전 등록 `results/PREREG_V2.md`. "
             f"평가 `results/E_report.json`(1차 설정 {prim}, 판정 키 `{pk}`: REP8 + disp = D3_int).")
    if E.get("prereg_change_needed"):
        L.append(f"- **사전 등록 변경 필요(STOP 대상)**: {E.get('prereg_change_reason')}")
    n = E.get("n") or {}
    L.append(f"- 평가 짝: D1 T1 시험 P_main para 중 dft · lf · MLTS 1차가 모두 예측한 공통 짝 **{n.get('common_para')}**개 "
             f"(LADDER dft para 시험 짝 {n.get('ladder_dft_para')}개 중; MLTS 1차 coverage "
             f"{pct(n.get('coverage_mlts_para'))}, 공통 {pct(n.get('coverage_common_para'))}).")
    if E.get("flags"):
        L.append("- 표시(판정 아님): " + "; ".join(str(x) for x in E["flags"]))
    L.append("")
    # ---------------------------------------------------------------- 1 structures
    L.append("## 1. 구조 — 라벨 TS 대비")
    L.append("")
    L.append("| 입력 구조 | 형성 결합 \\|Δd\\| 중앙값 (Å) | 무거운 원자 RMSD 중앙값 (Å) | 라벨과 같은 TS 비율 | 반응당 비용 | 출처 |")
    L.append("|---|---|---|---|---|---|")
    xr = xtb_row()
    L.append(f"| xTB (g2: autodE + XTB2 ALPB water) | {f(xr.get('dd_median'), 4)} | {f(xr.get('rmsd_median'), 4)} | "
             f"{pct(xr.get('same_frac'))} | {f(xr.get('core_h'), 3)} core-h | `analysis/d1_g2/results/C_compare.csv` |")
    eng = (E.get("arms") or {}).get("engine") or "A"
    L.append(f"| **MLTS** (엔진 {eng}, fold별 FT_k) | {f(geo.get('dd_median_ok'), 4)} (같은 TS만 "
             f"{f(geo.get('dd_median_same'), 4)}) | {f(geo.get('rmsd_median_ok'), 4)} | "
             f"{pct(geo.get('same_ts_frac_of_targets'))} (ok 중 {pct(geo.get('same_ts_frac_of_ok'))}) | "
             f"{f(geo.get('core_h_median'), 3)} core-h + {f(geo.get('gpu_min_median'), 2)} GPU-분 | "
             f"`$MLT/eval/E_geom_main.csv` (A-4 {geo.get('n_ok')}/{geo.get('n_targets')} ok) |")
    gr = g3l_row()
    L.append(f"| G3L (G2 TS → 라벨 수준 OptTS, {gr.get('n')}반응) | {f(gr.get('dd_median'), 4)} | "
             f"{f(gr.get('rmsd_median'), 4)} | {pct(gr.get('same'))} | {f(gr.get('core_h'), 2)} core-h | "
             f"`analysis/d1_g3/results/G3L_report.json` |")
    L.append(f"| 라벨 (D1 DFT TS) | 0 | 0 | 100 % | 약 {LABEL_TS_CORE_H} core-h | 명령서 0-2 |")
    L.append("")
    arms = A3.get("arms") or {}
    if arms:
        L.append("A-3 시범(fold마다 60반응): " + "; ".join(
            f"{a} 같은 TS {pct((v or {}).get('same_ts_frac'))}, |Δd| {f((v or {}).get('dd_median'), 4)} Å, "
            f"{f((v or {}).get('core_h_median'), 2)} core-h + {f((v or {}).get('gpu_min_median'), 2)} GPU-분"
            for a, v in arms.items()) + ".")
    g2r = (FT.get("gate_round2") or {})                    # pipeline.write_ft_outputs: pooled round-2 FT-2 numbers
    pooled = g2r.get("pooled") or {}
    ft2 = pooled.get("label") or FT.get("ft2_a") or {}
    if ft2:
        gs = pooled.get("g2") or {}
        L.append(f"FT-2 (a) 라벨 TS 출발(2차, 5 fold 합침, {ft2.get('n_attempted')}반응): |Δd| 중앙값 "
                 f"{f(ft2.get('dd_median_A'), 4)} Å, 90 % {f(ft2.get('dd_p90_A'), 4)} Å, RMSD {f(ft2.get('rmsd_median_A'), 4)} "
                 f"Å, 같은 TS {pct(ft2.get('same_ts_frac'))}, 허수 1개 {pct(ft2.get('nimag1_frac'))} → "
                 f"{'통과' if g2r.get('pass_all') else '미달'}; (b) G2 TS 출발: |Δd| 중앙값 {f(gs.get('dd_median_A'), 4)} Å, "
                 f"같은 TS {pct(gs.get('same_ts_frac'))}.")
    L.append("")
    # ---------------------------------------------------------------- 2 channels
    L.append("## 2. 채널 — 9개 타깃 para MAE (kcal/mol)와 꺾임 판정 (공통 짝)")
    L.append("")
    cases = [("dft#model", "dft 상한 (라벨 TS 입력, REP8)"), ("lf#model", "lf (현재 배포, xTB 입력)"),
             (pk, f"**MLTS** ({prim}, disp = D3_int)")]
    L.append("| 모델 | " + " | ".join(TLAB[t] for t in PRIMARY9) + " |")
    L.append("|---|" + "---|" * len(PRIMARY9))
    for k, lab in cases:
        r = pm.get(k) or {}
        L.append(f"| {lab} | " + " | ".join(f((r.get(t) or {}).get("mae")) for t in PRIMARY9) + " |")
    md = (E.get("disp_both") or {}).get(prim) or {}
    L.append("")
    L.append(f"- disp 두 방식: 모델 {f((md.get('model') or {}).get('mae'), 4)}, D3_int {f((md.get('D3_int') or {}).get('mae'), 4)} "
             "kcal/mol(H-0 결정: D3_int를 예측값으로 씀).")
    part = (gd.get("partial") or {}).get("per_target") or {}
    L.append("- MLTS / lf 비(부분 개선 기준 ≤ 0.5): " + ", ".join(f"{TLAB[t]} {f((part.get(t) or {}).get('ratio'), 2)}"
                                                       for t in PRIMARY9) + ".")
    L.append("")
    L.append("꺾임 민감도 / 위양성률 (LADDER E1 bends_e1, P_main para 계열):")
    L.append("")
    L.append("| 모델 | " + " | ".join(TLAB[t] for t in PRIMARY9) + " |")
    L.append("|---|" + "---|" * len(PRIMARY9))
    for k, lab in cases:
        rr = rates_of(BR, k)
        L.append(f"| {lab} | " + " | ".join(f"{f(rr[t].get('sens'), 2)} / {f(rr[t].get('fpr'), 2)}" for t in PRIMARY9)
                 + " |")
    L.append("")
    gm, gb = (gd.get("goal") or {}).get("mae") or {}, (gd.get("goal") or {}).get("bend") or {}
    L.append("등급 판정(사전 등록 1절, 점 추정): 목표 = oi · cpcm para MAE ≤ 0.13 그리고 d1 · elst · pauli · oi 민감도 ≥ 0.7, "
             "위양성률 ≤ 0.1; 부분 개선 = 9개 타깃 모두 MAE ≤ 0.5 × lf.")
    L.append("- MAE: " + ", ".join(f"{t} {f(v.get('para_mae'))} ({'통과' if v.get('ok') else '미달'})" for t, v in gm.items())
             + "; 꺾임: " + ", ".join(f"{t} {f(v.get('sens'), 2)} / {f(v.get('fpr'), 2)} ({'통과' if v.get('ok') else '미달'})"
                                     for t, v in gb.items()) + ".")
    L.append(f"- 목표 달성 {gd.get('goal', {}).get('passed')}, 부분 개선 {gd.get('partial', {}).get('passed')} → "
             f"**{GRADE_KO.get(g, g)}**. disp를 모델 값으로 두면: {(E.get('grade_model_disp') or {}).get('grade')}.")
    auc = (E.get("auc_g") or {}).get(pk) or {}
    if auc:
        L.append("- 연속 g AUC (MLTS): " + ", ".join(f"{TLAB.get(c, c)} {f(v, 3)}" for c, v in auc.items()) + ".")
    L.append("")
    # ---------------------------------------------------------------- 3 configs
    L.append("## 3. 채널 모델 설정 (K)")
    L.append("")
    K = rj(M.RES / "K_report.json") or {}
    L.append(f"- 1차 설정 = **{prim}** (안쪽 검증 para MAE oi + cpcm 최소, 시험 결과 미사용): inner_val "
             + ", ".join(f"{k} {f(v)}" for k, v in (K.get("inner_val") or {}).items()) + ".")
    for k in sorted(pm):
        if k.endswith("#D3_int") and k != pk:
            r = pm[k]
            L.append(f"- 설명용 {k.split('#')[0]}: oi {f((r.get('oi') or {}).get('mae'))}, cpcm {f((r.get('cpcm') or {}).get('mae'))}, "
                     f"pauli {f((r.get('pauli') or {}).get('mae'))} (n {(r.get('oi') or {}).get('n')}).")
    if K.get("delta_H"):
        L.append(f"- 재학습 잡음 δ_H: {K.get('delta_H')}")
    L.append("")
    # ---------------------------------------------------------------- 4 breakdowns
    L.append("## 4. 분해 (MLTS 1차, oi / cpcm para MAE)")
    L.append("")
    bc = E.get("by_class_oi_cpcm") or {}
    for sp, name in (("same_ts", "A의 TS가 라벨과 같은지(짝: 둘 다 / 한쪽 / 아님)"), ("g1", "g1 형태 class")):
        z = bc.get(sp) or {}
        if z:
            L.append(f"- {name}: " + "; ".join(f"{k.replace('|', ' ')} {f(v.get('mae'))} (n {v.get('n')})"
                                               for k, v in sorted(z.items())))
    L.append("- dipole별 표와 CI: `results/E_metrics.csv` (split = dipole / same_ts / g1).")
    L.append("")
    # ---------------------------------------------------------------- 5 W / P
    L.append("## 5. 최종 모델 (W) · 추론기 (P)")
    L.append("")
    ch, cm, co = W.get("channel") or {}, W.get("committee") or {}, W.get("cost") or {}
    L.append(f"- W-1 채널 모델: {ch.get('config')} 설정을 rung {ch.get('trained_rung')}에서 전체 D1로 학습 "
             f"(`$MLT/w/channel/MODEL_CARD.json`).")
    L.append(f"- W-2 위원회(FT_all_s0–2) δ_com: 중앙값 {f((cm.get('delta_com_quantiles_A') or {}).get('50'), 4)} Å, "
             f"경고 기준(95 백분위) {f(cm.get('warn_threshold_A'), 4)} Å ({cm.get('n_with_delta')}/{cm.get('n_reactions')}반응).")
    rel = W.get("relation") or {}
    if rel:
        L.append("- δ_com 대 채널 오차 Spearman ρ: " + ", ".join(f"{TLAB.get(t, t)} {f(v.get('spearman_rho'), 2)}"
                                                       for t, v in rel.items()) + ".")
    L.append(f"- P-2 자체 검사: **{'통과' if P.get('selfcheck_pass') else '실패'}** "
             f"(R1 fold 재현 {((P.get('selfcheck') or {}).get('R1_fold_replay') or {}).get('pass_')}, "
             f"R2 최종 모델 재현 {((P.get('selfcheck') or {}).get('R2_final_replay') or {}).get('pass_')}, "
             f"SMILES부터 {((P.get('e2e') or {}).get('pass_'))}).")
    ce = P.get("cert") or {}
    if ce:
        fq = ce.get("fmax_atom_quantiles") or ce.get("fmax_comp_quantiles") or {}
        L.append(f"- P-3 DFT 증명서: {ce.get('n_ok')}/{ce.get('n_reactions')}반응, 라벨 수준 원자 최대 힘 중앙값 "
                 f"{f(fq.get('50'), 5)} Eh/bohr, > {ce.get('threshold_eh_bohr')} 비율 "
                 f"{pct(ce.get('frac_above_threshold'))}.")
    L.append("")
    # ---------------------------------------------------------------- 6 limits
    L.append("## 6. 한계")
    L.append("")
    same = geo.get("same_ts_frac_of_targets")
    L.append(f"- 형태 불일치: MLTS TS가 라벨과 같은 TS가 아닌 반응 {pct(None if same is None else 1 - same)} "
             f"(D1 9,266반응 기준; 실패 포함).")
    pdp = X.get("per_dipole") or {}
    if pdp:
        bad = [d for d, v in pdp.items() if not (v or {}).get("pass_a")]
        L.append(f"- LODO(X): FT-2 (a)를 못 넘은 dipole {len(bad)}개" + (f" ({', '.join(bad)}) → 새 dipole 계열에는 그 계열 DFT "
                                                                     "데이터를 조금 넣어 미세조정해야 함" if bad else "") + ".")
    D0e = rj(M.RES / "D0_eval_report.json") or {}                # optional (D0-eval stage, report only)
    if D0e.get("complete"):
        pl = D0e.get("pooled") or {}
        a, b = pl.get("label") or {}, pl.get("g2") or {}
        L.append(f"- D0 보조 기하 평가(평가 전용 D0 {D0e.get('n_units')}반응, 기준 = Coley Gaussian TS, FT_all 3 seed): "
                 f"기준 TS 출발 |Δd| 중앙값 {f(a.get('dd_median_A'), 4)} Å · 같은 TS {pct(a.get('same_ts_frac'))}, xTB TS "
                 f"출발 |Δd| 중앙값 {f(b.get('dd_median_A'), 4)} Å · 같은 TS {pct(b.get('same_ts_frac'))} (바닥값 Gaussian–"
                 f"ORCA {D0e.get('floor_rmsd_A')} Å RMSD; 등급 판정에는 쓰지 않음).")
    L.append(f"- 배포 비용 실측(반응당 중앙값): {f(co.get('total_core_h'), 3)} core-h + {f(co.get('total_gpu_min'), 2)} GPU-분 "
             f"(목표 ≤ 1 core-h + ≤ 3 GPU-분: {'충족' if co.get('within_target') else '미충족'}).")
    L.append("- 위원회 δ_com은 FT_all이 학습한 D1 반응에서 잰 값이라 새 반응에서는 더 클 수 있다(W 결정 W-com-set).")
    if A4:
        L.append(f"- A-4: ok {A4.get('n_ok')}, 실패 {A4.get('n_fail')}.")
    L.append("")
    L.append("## 7. 입력")
    L.append("")
    for name in ("PREREG_V2.md", "E_report.json", "E_metrics.csv", "E_bend_rates.json", "W_report.json", "P_report.json",
                 "X_report.json", "FT_report.json", "A3_report.json", "A4_report.json", "D0_eval_report.json"):
        p = M.RES / name
        if p.is_file():
            L.append(f"- `results/{name}` sha256 `{M.sha256(p)[:12]}`")
    L.append("")
    L.append(f"(작성: `analysis/mlts/summary_v2.py` {CODE_VERSION}, {M.now()}. `$MLT` 아래 산출물은 commit하지 않는다.)")
    return "\n".join(L) + "\n"


def cmd_write(stop_file="auto"):
    import inputs_mlts as IM
    IM.ensure_env("reactot")
    md = build()
    M.write_text(SUMMARY, md)
    print(f"[S] wrote {SUMMARY} ({len(md.splitlines())} lines)", flush=True)
    launcher = str(os.environ.get("SLURM_JOB_NAME", "")).startswith("mltp_")
    if stop_file == "yes" or (stop_file == "auto" and not launcher):
        M.write_text(STOP_S, f"# 명령서 S: 멈춤\n\n{M.now()} summary_v2.py: SUMMARY_V2.md를 썼다. 명령서 순서의 끝(S 요약 후 "
                             "멈춤). launcher는 이 파일이 있으면 더 이상 자신을 다시 내지 않는다.\n")
        print(f"[S] wrote {STOP_S}", flush=True)
    else:
        print("[S] STOP_S.md left to the launcher (decide_s writes it after its commit)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="summary_v2.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("write")
    p.add_argument("--stop-file", default="auto", choices=("auto", "yes", "no"))
    o = ap.parse_args(argv)
    if o.cmd == "write":
        return cmd_write(o.stop_file)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
