#!/usr/bin/env python3
"""GNN3R2 Phase R (docs/specs/GNN3R2.md 10-1): results/RUN_REPORT_GNN3R2.md (Korean) from the result files.
Rewritten from the GNN3R report script (analysis/gnn3r/report_r.py) for the GNN3R2 sections.

    python analysis/gnn3r2/report_r.py [--pushed <sha>]     env_d1fu2 (gnn3/jobs/run.slurm), CPU; after the manifest

Reads only results/ (S0, COPY, A, B, G, S, S-5, D, K, E, W reports, gnn3r2.json, A_c3_core.csv, F_pairs_r.csv,
S_folds_r.csv) and ref_gnn3r/ (GNN3R F_pairs_r.csv, S_folds_r.csv, metrics.csv) for the GNN3R columns.
--pushed records the HEAD that was pushed (spec 10-3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

TARGETS = ("strain", "inter", "solv", "barrier")
DIPOLES = ("nitrone", "azide", "nitrile oxide", "azomethine ylide", "carbonyl ylide")
GNN3R = dict(reactions=8589, pairs=8339, reference=6978, anchor=1361, fold1_test=1700, p4_dropped_rows=16)   # 0-4
NEW_PAIRS = [("T10104", "T07195"), ("T10116", "T07195"), ("T10047", "T10056"), ("T10067", "T10056"),
             ("T10092", "T10056"), ("T10111", "T10056")]


def n(v) -> str:
    return f"{int(v):,}"


def f3(v) -> str:
    return "" if v is None or pd.isna(v) else f"{float(v):.3f}"


def ok(rep: dict) -> str:
    return "통과" if rep.get("all_pass", rep.get("all_equal")) else "STOP"


def by_kind(P: pd.DataFrame, col: str) -> dict:
    t = P.groupby(col).ref_kind.value_counts().unstack(fill_value=0)
    return {str(k): (int(r.get("anchor", 0)), int(r.get("reference", 0))) for k, r in t.iterrows()}


def kind_table(L, title, old: dict, new: dict, keys):
    L += [f"**{title}** (GNN3R → GNN3R2)", "", "| | anchor | 기준 | 합 |", "|---|---|---|---|"]
    for k in keys:
        a0, r0 = old.get(str(k), (0, 0))
        a1, r1 = new.get(str(k), (0, 0))
        L.append(f"| {k} | {n(a0)} → {n(a1)} | {n(r0)} → {n(r1)} | {n(a0 + r0)} → {n(a1 + r1)} |")
    L.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pushed", default=None)
    args = ap.parse_args()
    R = C.RES
    s0, cp, a, b = (C.read_json(R / n_) for n_ in ("S0_start.json", "COPY.json", "A_report.json", "B_report.json"))
    g, s, s5, d = (C.read_json(R / n_) for n_ in ("G_report.json", "S_splits_r.json", "S5_report.json", "D_check.json"))
    k, e, w = C.read_json(R / "cv" / "K_check.json"), C.read_json(R / "eval" / "E_check.json"), \
        C.read_json(R / "final" / "W_check.json")
    fr = C.read_json(R / "final" / "final_report.json")
    man = C.read_json(R / "gnn3r2.json")["header"]
    cb, a3, a5 = b["counts"], a["gates"]["A-3"], a["gates"]["A-5"]
    P, P0 = pd.read_csv(C.PAIRS_R, dtype=str, keep_default_na=False), \
        pd.read_csv(C.REF_GNN3R / "F_pairs_r.csv", dtype=str, keep_default_na=False)
    S, S0 = pd.read_csv(C.S_FOLDS_R, dtype=str, keep_default_na=False), \
        pd.read_csv(C.REF_GNN3R / "S_folds_r.csv", dtype=str, keep_default_na=False)
    rs = b["row_status"]["counts"]
    L = ["# GNN3R2 실행 보고 — c3 판정 오류 2건을 고친 짝으로 GNN3 다시 학습", "",
         f"- 지시서: `docs/specs/GNN3R2.md` · 브랜치 `gnn3r2` · 결과 commit `{man['results_commit'][:8]}`"
         + (f" · push한 HEAD `{args.pushed}`" if args.pushed else ""),
         f"- scratch `{C.GR}` (cv/, final/ = .pt), 결과 `analysis/gnn3r2/results/`, manifest `results/gnn3r2.json`, "
         "GNN3R 대조 사본 `analysis/gnn3r2/ref_gnn3r/`",
         f"- 데이터셋 고정 파일: `results/F_pairs_r.csv` sha256 `{man['files']['F_pairs_r.csv']['sha256']}` "
         "(`B_label_report.json`에 기록, gnn3 로더가 이 값으로 확인)",
         "- 결정·편차: `results/DECISIONS.md`", "",
         "## 1. 단계별 판정", "", "| 단계 | 판정 | 비고 |", "|---|---|---|",
         f"| 1 시작 조건 | {ok(s0)} | 1-1 … 1-5 (GNN3R 1-2 … 1-10 포함), RDKit {s0['checks']['1-4']['rdkit']}, 여유 "
         f"{s0['checks']['1-5']['free_gb']:.0f} GB |",
         f"| 2 복사 | {ok(cp)} | 코드·참조 결과·scratch 파일 sha256 = 원본 (`COPY.json`) |",
         f"| A c3 다시 판정 | {ok(a)} | A-1 … A-6; c3가 바뀐 행 {len(a5['c3_changed'])}개 |",
         f"| B 짝 | {ok(b)} | B-0 … B-7, B-R |",
         f"| G GNN3R 대조 | {ok(g)} | G-1 … G-4 |",
         f"| S 분할 | {ok(s)} / S-5 {ok(s5)} | S-1 … S-5, sklearn {s['sklearn']} |",
         f"| D 데이터 | {ok(d)} | D-1 … D-4 |",
         f"| K 교차검증 | {ok(k)} | K-1 … K-3 (RTX 3090) |",
         f"| E 평가 | {ok(e)} | E-1: {n(e['gates']['E-1']['n_evaluated'])}짝 평가 |",
         f"| W 최종 모델 | {ok(w)} | W-1: {n(w['gates']['W-1']['n_train_pairs'])}짝 학습 |", ""]
    # 2. c3 fix
    core = pd.read_csv(R / "A_c3_core.csv", dtype=str, keep_default_na=False)
    ch = core[core.c3_gnn3r != core.c3_core]
    L += ["## 2. c3 수정 (D-12)", "",
          "- 규칙 `c3_core`: C3·X 첫 원자(xs)·C4·C4 기준 치환기는 예전과 같이 고른다(B′는 메틸 규칙). X 가지를 xs 하나로 줄이고 "
          "xs를 CH₃로 바꾼 뼈대(core, 다른 입체 표기 유지)를 정규 원자 순서로 놓고, seed 7·11·23·42·97로 각각 "
          "EmbedMolecule + MMFF 배치한다. 고리 5원자 법선(SVD)에서 u·v > 0이면 cis. 5개가 모두 같아야 한다. X = H 행은 빈칸.",
          f"- `need` {n(a['counts']['need'])}행: C3 치환기 있음 {n(a3['n_substituted'])} (cis/trans "
          f"{a3['counts']}), X = H {a3['n_h']}행(모두 `ref_direct`: {a3['h_rows_all_ref_direct']}).",
          f"- 서로 다른 core {a3['n_cores']}개(지시서 로컬 72), seed 5개가 모두 같은 행 {n(a3['n_seed_unanimous'])} / "
          f"{n(a3['n_substituted'])}, 서로 거울상인 core 쌍 {a3['n_mirror_core_pairs']}개(c3 충돌 "
          f"{len(a3['mirror_conflicts'])}), 같은 core 안 c3 충돌 {len(a3['same_core_conflicts'])}.",
          f"- GNN3R c3와 다른 행 {len(ch)}개 (A-5 기대: {', '.join(f'{j} {v[0]}→{v[1]}' for j, v in C.EXPECT_C3_CHANGED.items())}):",
          "", "| job_id | 위치 · X | c3 GNN3R → GNN3R2 | seed 7 · 11 · 23 · 42 · 97 | core SMILES |", "|---|---|---|---|---|"]
    for r in ch.itertuples():
        seeds = " · ".join(getattr(r, f"seed_{s_}") for s_ in C.CORE_SEEDS)
        L.append(f"| {r.job_id} | {r.pos} · {r.Xb} | {r.c3_gnn3r} → {r.c3_core} | {seeds} | `{r.core_smiles}` |")
    L.append("")
    # 3. data set
    gg = g["gates"]
    fold1 = s["units"]["T1__f1.json"]["n_test"]
    L += ["## 3. 데이터셋", "",
          "| 항목 | GNN3R | GNN3R2 기대 (0-4) | GNN3R2 실제 |", "|---|---|---|---|",
          f"| 반응 | {n(GNN3R['reactions'])} | {n(C.EXPECT_REACTIONS)} | {n(cb['reactions'])} |",
          f"| 짝 | {n(GNN3R['pairs'])} | {n(C.EXPECT_PAIRS)} | {n(cb['pairs'])} |",
          f"| 기준 짝 / anchor 짝 | {n(GNN3R['reference'])} / {n(GNN3R['anchor'])} | 6,978 / 1,366 | "
          f"{n(cb['by_ref_kind'].get('reference', 0))} / {n(cb['by_ref_kind'].get('anchor', 0))} |",
          f"| fold 1 test 짝 | {n(GNN3R['fold1_test'])} | {n(C.EXPECT_TEST[1])} | {n(fold1)} |",
          f"| `p4_dropped` 행 | {GNN3R['p4_dropped_rows']} | 11 | {rs.get('p4_dropped', 0)} |", "",
          "**새 짝 6개** (0-4절)", "", "| a → b | 서버 | P-4 v2 |", "|---|---|---|"]
    dropped = {(x["a_job"], x["b_job"]) for x in b["p4"]["dropped"]}
    keyed = P.set_index(["a_job", "b_job"])
    for ab in NEW_PAIRS:
        if ab in keyed.index:
            L.append(f"| {ab[0]} → {ab[1]} | 데이터셋 안 (`{keyed.loc[ab, 'pair_id']}`) | {keyed.loc[ab, 'p4_v2']} |")
        else:
            L.append(f"| {ab[0]} → {ab[1]} | {'P-4로 버림' if ab in dropped else '없음'} | "
                     f"{'mismatch' if ab in dropped else ''} |")
    L += ["", "**GNN3R 대조** (`results/G_report.json`, `G_diff_gnn3r.csv`)", "", "| # | 판정 | 내용 |", "|---|---|---|",
          f"| G-1 | {'통과' if gg['G-1']['ok'] else 'STOP'} | 계열 T 밖 짝 {n(gg['G-1']['n_gnn3r2'])}개: GNN3R과 (a, b)·값 모두 같음 "
          f"(GNN3R에만 {gg['G-1']['only_gnn3r']}, GNN3R2에만 {gg['G-1']['only_gnn3r2']}, 값 차이 {gg['G-1']['value_diffs']}) |",
          f"| G-2 | {'통과' if gg['G-2']['ok'] else 'STOP'} | 계열 T 안: 추가 {len(gg['G-2']['added'])}, 빠짐 "
          f"{gg['G-2']['removed']}, P-4로 빠진 기대 짝 {gg['G-2']['missing_dropped_by_p4']}, 값 차이 "
          f"{gg['G-2']['value_diffs_in_T']} (DEC-1 기준) |",
          f"| G-3 | {'통과' if gg['G-3']['ok'] else 'STOP'} | `B_row_status`가 다른 행: {', '.join(gg['G-3']['jobs'])} |",
          f"| G-4 | {'통과' if gg['G-4']['ok'] else 'STOP'} | 짝 {n(gg['G-4']['n_pairs'])} (기대 {n(gg['G-4']['want_pairs'])}), "
          f"반응 {n(gg['G-4']['n_reactions'])} (기대 {n(gg['G-4']['want_reactions'])}), 기준 짝 {n(gg['G-4']['n_reference'])} |",
          f"| S-5 | {ok(s5)} | fold 0·2·3·4 test 목록 = GNN3R |", ""]
    f_old = dict(zip(S0.pair_id, S0.fold_T1))
    f_new = dict(zip(S.pair_id, S.fold_T1))
    kind_table(L, "fold_T1별", by_kind(P0.assign(fold=P0.pair_id.map(f_old)), "fold"),
               by_kind(P.assign(fold=P.pair_id.map(f_new)), "fold"), ["-1", "0", "1", "2", "3", "4"])
    kind_table(L, "쌍극자별", by_kind(P0, "dipole"), by_kind(P, "dipole"), DIPOLES)
    kind_table(L, "위치별", by_kind(P0, "pos"), by_kind(P, "pos"), ["p", "m", "o", "direct"])
    L += [f"9,252행 status: " + ", ".join(f"{kk} {n(v)}" for kk, v in rs.items()), ""]
    # 4. P-4
    p4 = b["p4"]
    L += ["## 4. P-4 입체 검사", "",
          f"- 검사한 짝 {n(p4['n_pairs_checked'])} (반응 {n(p4['n_jobs'])}, 이번에 새로 계산 {p4['n_computed_now']}, "
          f"고리 구성 실패 {p4['n_ring_fail_jobs']}; 링 정보는 GNN3R 캐시 사본)",
          "- v2: " + ", ".join(f"{kk} {n(v)}" for kk, v in p4["v2"].items()),
          f"- mismatch로 버린 짝 {p4['n_dropped']}: " + ", ".join(f"{x['a_job']}→{x['b_job']} ({x['ref_kind']})"
                                                         for x in p4["dropped"]), ""]
    # 5. training
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
    # 6. performance
    m = pd.read_csv(R / "eval" / "metrics.csv")
    old = pd.read_csv(C.REF_GNN3R / "metrics.csv")
    L += ["## 6. 성능 (교차검증 test 예측, kcal/mol)", "",
          "| 집합 | 위치 | target | n | MAE | r | GNN3R MAE (n) | GNN3R r |", "|---|---|---|---|---|---|---|---|"]
    for es in ("P_main", "test_all"):
        for pos in ("all", "p"):
            for t in TARGETS:
                q = m[(m.eval_set == es) & (m.pos == pos) & (m.target == t)]
                o = old[(old.eval_set == es) & (old.pos == pos) & (old.target == t)]
                if q.empty:
                    continue
                q = q.iloc[0]
                ci = f" [{f3(q.mae_ci_lo)}, {f3(q.mae_ci_hi)}]" if not pd.isna(q.get("mae_ci_lo")) else ""
                ot = (f"{f3(o.iloc[0].mae)} ({n(o.iloc[0].n)})", f3(o.iloc[0].r)) if not o.empty else ("", "")
                L.append(f"| {es} | {pos} | {t} | {n(q.n)} | {f3(q.mae)}{ci} | {f3(q.r)} | {ot[0]} | {ot[1]} |")
    L += ["", "- GNN3R 값(`ref_gnn3r/metrics.csv`)은 참고용이다. 짝 목록이 6짝 다르고(fold 1) 모든 fold를 다시 학습했다.",
          "- P_main = 기준 짝(ref_kind reference) 전부(D-10). [ ] = core bootstrap 95 % 구간(para).", ""]
    C.write_atomic(R / "RUN_REPORT_GNN3R2.md", "\n".join(L) + "\n")
    print(f"wrote {R / 'RUN_REPORT_GNN3R2.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
