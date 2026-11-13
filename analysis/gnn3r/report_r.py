#!/usr/bin/env python3
"""GNN3R Phase R (docs/specs/GNN3R.md 10-1): results/RUN_REPORT_GNN3R.md (Korean) from the result files.

    python analysis/gnn3r/report_r.py [--pushed <sha>]      env_d1fu2 (gnn3/jobs/run.slurm), CPU; after the manifest

Reads only results/ (S0, A, B, S, D, K, E, W reports, gnn3r.json) and gnn3/results/eval/metrics.csv (the earlier GNN3,
for reference). --pushed records the HEAD that was pushed (spec 10-3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

TARGETS = ("strain", "inter", "solv", "barrier")
DIPOLES = ("nitrone", "azide", "nitrile oxide", "azomethine ylide", "carbonyl ylide")
EXPECT_DIPOLE = {"nitrone": (266, 1986), "azide": (260, 1536), "nitrile oxide": (44, 1663),
                 "azomethine ylide": (488, 966), "carbonyl ylide": (320, 826)}      # (anchor, reference), spec 0-3(d)
EXPECT_POS = {"p": (328, 1839), "m": (325, 1879), "o": (405, 1705), "direct": (320, 1554)}
EXPECT_FOLD = {"-1": (170, 0), "0": (269, 1417), "1": (305, 1401), "2": (171, 1243), "3": (36, 1308),
               "4": (427, 1608)}
EXPECT_UNPAIRED = {"T08439": "p4", "T09972": "p4", "T10679": "p4", "T10704": "p4", "T01507": "no_family",
                   "T02374": "no_family", "T02510": "no_family", "T02573": "no_family", "T02833": "no_family",
                   "T11185": "no_family", "T11220": "no_family", "T01073": "singleton", "T09850": "singleton"}
OLD_METRICS = C.GNN3_PACKAGE / "results" / "eval" / "metrics.csv"


def n(v) -> str:
    return f"{int(v):,}"


def f3(v) -> str:
    return "" if v is None or pd.isna(v) else f"{float(v):.3f}"


def kind_table(lines, title, counts, keys, expect):
    lines += [f"**{title}** (실제 / 로컬 추정)", "", "| | anchor | 기준 | 합 |", "|---|---|---|---|"]
    ta = tr = ea = er = 0
    for k in keys:
        c = counts.get(str(k), {})
        a, r = int(c.get("anchor", 0)), int(c.get("reference", 0))
        e = expect.get(str(k), (0, 0))
        ta, tr, ea, er = ta + a, tr + r, ea + e[0], er + e[1]
        lines.append(f"| {k} | {n(a)} / {n(e[0])} | {n(r)} / {n(e[1])} | {n(a + r)} / {n(sum(e))} |")
    lines += [f"| **합** | {n(ta)} / {n(ea)} | {n(tr)} / {n(er)} | {n(ta + tr)} / {n(ea + er)} |", ""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pushed", default=None)
    args = ap.parse_args()
    R = C.RES
    s0, a, b = C.read_json(R / "S0_start.json"), C.read_json(R / "A_report.json"), C.read_json(R / "B_report.json")
    s, d = C.read_json(C.S_SPLITS_R), C.read_json(R / "D_check.json")
    k, e, w = C.read_json(R / "cv" / "K_check.json"), C.read_json(R / "eval" / "E_check.json"), \
        C.read_json(R / "final" / "W_check.json")
    fr = C.read_json(R / "final" / "final_report.json")
    man = C.read_json(R / "gnn3r.json")["header"]
    g2 = C.read_json(C.G2R_MANIFEST)["header"]["counts"]
    rows = pd.read_csv(R / "B_row_status.csv", dtype=str, keep_default_na=False)
    cb = b["counts"]
    L = ["# GNN3R 실행 보고 — g2r 입력과 다시 만든 짝으로 GNN3 다시 학습", "",
         f"- 지시서: `docs/specs/GNN3R.md` · 브랜치 `gnn3r` · 결과 commit `{man['results_commit'][:8]}`"
         + (f" · push한 HEAD `{args.pushed}`" if args.pushed else ""),
         f"- scratch `{C.GR}` (cv/, final/ = .pt), 결과 `analysis/gnn3r/results/`, manifest `results/gnn3r.json`",
         f"- 데이터셋 고정 파일: `results/F_pairs_r.csv` sha256 `{man['files']['F_pairs_r.csv']['sha256']}` "
         "(`B_label_report.json`에 기록, gnn3 로더가 이 값으로 확인)", "",
         "## 1. 단계별 판정", "", "| 단계 | 판정 | 비고 |", "|---|---|---|",
         f"| 1 시작 조건 | {'통과' if s0['all_pass'] else 'STOP'} | 1-1 … 1-11, RDKit "
         f"{s0['checks']['1-10']['reactot']['rdkit']}, 여유 {s0['checks']['1-11']['free_gb']:.0f} GB |",
         f"| A 행 속성 | {'통과' if a['all_pass'] else 'STOP'} | A-1 … A-6; B′ need {a['gates']['A-3']['n_bp']}행 "
         f"{a['gates']['A-3']['counts']} (빈칸 {a['gates']['A-3'].get('n_bp_h_reference', 0)}행 = X = H 기준 행, "
         f"DEC-1); F 안 Ph 중복 {a['gates']['A-6']['ref_aryl_duplicated_in_F_before']} → "
         f"{a['gates']['A-6']['after']} |",
         f"| B 짝 | {'통과' if b.get('all_pass') else 'STOP'} | B-0 … B-7, B-R |",
         f"| S 분할 | {'통과' if s['all_pass'] else 'STOP'} | S-1 … S-4, sklearn {s['sklearn']} |",
         f"| D 데이터 | {'통과' if d['all_pass'] else 'STOP'} | D-1 … D-4 |",
         f"| K 교차검증 | {'통과' if k['all_pass'] else 'STOP'} | K-1 … K-3 (RTX 3090) |",
         f"| E 평가 | {'통과' if e['all_pass'] else 'STOP'} | E-1: {n(e['gates']['E-1']['n_evaluated'])}짝 평가 |",
         f"| W 최종 모델 | {'통과' if w['all_pass'] else 'STOP'} | W-1: {n(w['gates']['W-1']['n_train_pairs'])}짝 학습 |", ""]
    # funnel
    use = pd.read_csv(C.USABLE)
    ms = use.missing_stage.fillna("").value_counts().to_dict()
    unp = rows[(rows.usable == "True") & (rows.status != "in_dataset")]
    L += ["## 2. 깔때기 (반응)", "", "| 단계 | 빠짐 (기대 / 실제) | 남음 (기대 / 실제) |", "|---|---|---|",
          "| D1 비D″ 라벨 | — | 9,266 / 9,266 |",
          f"| short_forming 9 + regio_ambiguous 5 | 14 / 14 | 9,252 / {n(len(use))} |",
          f"| g2 TS 없음·autodE 오류 등 (X, TS 단계) | 599 / {n(g2['groups'].get('excluded_ts', 0))} | 8,653 / "
          f"{n(len(use) - g2['groups'].get('excluded_ts', 0))} |",
          f"| g2 TS 검사·조각 분할 (X, 검사) | 19 / {n(g2['groups'].get('excluded_check', 0))} | 8,634 / "
          f"{n(len(use) - ms.get('X', 0))} |",
          f"| g2 기준 구조 복구 못 함 (R) | 17 / {n(ms.get('R', 0))} | 8,617 / {n(int(use.usable.sum()))} |",
          f"| 짝 없음 | 13 / {n(len(unp))} | **8,604 / {n(cb['reactions'])}** |", "",
          f"사용 가능하지만 짝이 없는 반응 {len(unp)}개 (로컬 추정 13개와 비교):", "",
          "| job_id | 서버 status | 로컬 추정 | fam |", "|---|---|---|---|"]
    for r in unp.itertuples():
        L.append(f"| {r.job_id} | {r.status} | {EXPECT_UNPAIRED.get(r.job_id, '(짝 있음)')} | `{r.fam}` |")
    for j, why in EXPECT_UNPAIRED.items():
        if j not in set(unp.job_id):
            st = rows.set_index("job_id").status.get(j, "not_target")
            L.append(f"| {j} | {st} | {why} | (서버에서는 데이터셋 안) |")
    L += ["", f"9,252행 status: " + ", ".join(f"{kk} {n(v)}" for kk, v in b["row_status"]["counts"].items())
          + f" (in_dataset 중 F+Xb+pos 중복 직접 행 {b['row_status']['in_dataset_also_dup_key']})", ""]
    # data set
    L += ["## 3. 데이터셋", "",
          f"- 반응 **{n(cb['reactions'])}** (기대 8,604, 허용 8,432–8,776), 짝 **{n(cb['pairs'])}** (기대 8,355, 허용 "
          f"8,188–8,522): 기준 짝 {n(cb['by_ref_kind'].get('reference', 0))}, anchor 짝 "
          f"{n(cb['by_ref_kind'].get('anchor', 0))}",
          f"- 계열: fam 문자열 {n(cb['distinct_fam'])}개, 짝이 있는 계열 묶음 "
          f"{sum(g['with_pairs'] for g in cb['groups'].values())}개 ("
          + ", ".join(f"{k} {g['with_pairs']}" for k, g in cb['groups'].items())
          + f"). 지시서의 로컬 추정 '계열 323개'와는 세는 단위가 다를 수 있다. core {cb['cores']}개, 기준 반응(b) "
          f"{cb['reference_b']}개, anchor b {cb['anchor_b']}개",
          f"- anchor 짝의 출처: {cb['anchor_from']} (anchor_family = D1PAIR anchor 계열, reference_not_usable = 기준이 "
          "g2r에서 쓸 수 없어 F로 다시 묶은 계열)", ""]
    kind_table(L, "fold_T1별", cb["by_fold_T1"], ["-1", "0", "1", "2", "3", "4"], EXPECT_FOLD)
    kind_table(L, "쌍극자별", cb["by_dipole"], DIPOLES, EXPECT_DIPOLE)
    kind_table(L, "위치별", cb["by_pos"], ["p", "m", "o", "direct"], EXPECT_POS)
    L += ["**참조 목록(`docs/specs/gnn3r_ref/`)과의 차이** (`results/B_diff_vs_est.csv`, 이유별 수)", "",
          "| 종류 | 추정에 있음 | 서버에 있음 | 이유 | 수 |", "|---|---|---|---|---|"]
    for kk, v in sorted(b["diff_vs_est"]["by_reason"].items(), key=lambda t: -t[1]):
        L.append(f"| {kk} | {n(v)} |")                  # kk = 'kind | in_est | in_server | reason'
    L += ["", f"test 크기 (실제 / 로컬 추정): " + ", ".join(
        f"f{kk} {n(s['test_sizes'][kk])} / {n(s['expected_test_sizes'][kk])}" for kk in ["0", "1", "2", "3", "4"])
          + f", 늘 학습 {n(s['test_sizes']['-1'])} / {n(s['expected_test_sizes']['-1'])}", ""]
    # P-4
    p4 = b["p4"]
    L += ["## 4. P-4 입체 검사", "",
          f"- 검사한 짝 {n(p4['n_pairs_checked'])} (반응 {n(p4['n_jobs'])}, 고리 구성 실패 반응 {p4['n_ring_fail_jobs']})",
          f"- v2: " + ", ".join(f"{kk} {n(v)}" for kk, v in p4["v2"].items()),
          f"- mismatch로 버린 짝 {p4['n_dropped']} (a는 다시 짝짓지 않음): "
          + ", ".join(f"{x['a_job']}→{x['b_job']} ({x['ref_kind']})" for x in p4["dropped"]),
          f"- D1PAIR와 같은 짝 {n(b['gates']['B-3']['n_shared'])}개의 v2 = `P4_stereo.csv` (B-3), "
          f"cr2 라벨 차이 최대 {max(b['gates']['B-4']['max_abs'].values()):.1e} kcal/mol ({n(b['gates']['B-4']['n_shared'])}짝, B-4)",
          ""]
    # training
    L += ["## 5. 학습", "", "| fold | 고른 칸 (층×폭) | epoch | 검증 MAE strain / inter / solv / barrier | 학습 / test 짝 | "
          "시간 (h) | GPU |", "|---|---|---|---|---|---|---|"]
    for f in range(5):
        rec = C.read_json(R / "cv" / f"gnn3_fold{f}.json")
        cell, mae = rec["chosen"]["cell"], rec["val_kcal_mae"]
        L.append(f"| {f} | {cell['n_layers']}×{cell['width']} | {rec['chosen']['best_epoch']} | "
                 + " / ".join(f3(mae[t]) for t in TARGETS) + f" | {n(rec['n_train'])} / {n(rec['n_test'])} | "
                 f"{rec['wall_seconds'] / 3600:.2f} | {rec['environment']['gpu']} |")
    st = w["setting"]
    L += ["", f"- 최종 setting: {st['cell']['n_layers']}층 × {st['cell']['width']}폭, epoch {st['n_epochs']} "
          f"(fold best epoch {st['best_epochs']}, 칸 선택 수 {st['cell_counts']})",
          f"- 최종 모델: seed 5개, {n(fr['n_train_pairs'])}짝, {fr['wall_seconds'] / 3600:.2f} h, "
          f"{fr['environment']['gpu']}; `.pt` = `{C.FINAL}/gnn3_final__seed{{0..4}}.pt`", ""]
    # performance
    m = pd.read_csv(R / "eval" / "metrics.csv")
    old = pd.read_csv(OLD_METRICS) if OLD_METRICS.is_file() else None
    L += ["## 6. 성능 (교차검증 test 예측, kcal/mol)", "",
          "| 집합 | 위치 | target | n | MAE | r | 예전 GNN3 MAE (n) |", "|---|---|---|---|---|---|---|"]
    for es in ("P_main", "test_all"):
        for pos in ("all", "p"):
            for t in TARGETS:
                q = m[(m.eval_set == es) & (m.pos == pos) & (m.target == t)]
                if q.empty:
                    continue
                q = q.iloc[0]
                o = old[(old.eval_set == es) & (old.pos == pos) & (old.target == t)] if old is not None else None
                ot = f"{f3(o.iloc[0].mae)} ({n(o.iloc[0].n)})" if o is not None and not o.empty else ""
                ci = f" [{f3(q.mae_ci_lo)}, {f3(q.mae_ci_hi)}]" if not pd.isna(q.get("mae_ci_lo")) else ""
                L.append(f"| {es} | {pos} | {t} | {n(q.n)} | {f3(q.mae)}{ci} | {f3(q.r)} | {ot} |")
    L += ["", "- 예전 GNN3(`gnn3/results/eval/metrics.csv`)는 다른 짝 목록(7,970짝, g2 입력)이라 같은 짝 비교가 아니다. "
          "참고용이다.", "- P_main = 기준 짝(ref_kind reference) 전부(D-10). [ ] = core bootstrap 95 % 구간(para).", ""]
    C.write_atomic(R / "RUN_REPORT_GNN3R.md", "\n".join(L) + "\n")
    print(f"wrote {R / 'RUN_REPORT_GNN3R.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
