"""missing_cause.py — SPEC_D1_ALL B2 "꼭 답할 것": 빈 칸이 no_ts 인지, 그중 고개가 없는(경로 에너지가 계속 내려가는) 경우인지.

python missing_cause.py STATUS_CSV MANIFEST ARCHIVE_SCAN_CSV RESULTS_DIR
  STATUS_CSV        확정 d1_status.csv (분류 열 class)
  MANIFEST          d1_manifest.csv (dipole, skeleton, series, R1, R2, attach)
  ARCHIVE_SCAN_CSV  d1_archive_scan.py (SCAN_VERSION v4) 출력 (no_ts 행의 path_verdict: barrierless / no_peak_short_scan / peak / no_path / unparsed / unread)
출력 (RESULTS_DIR):
  missing_cause.csv        no_ts 행 하나씩: 칸 정보 + path_verdict + 반응물 쪽 끝 형성 결합 거리 + 경로 고개 높이 + autodE 로그 표식
  missing_cause_cells.csv  SPEC 가 지목한 칸: (1) azomethine ylide × 골격 A · C × dipolarophile 쪽(R2) 전자 끄는/큰 X
                           (NO2, CN, CHO, Ac, CO2Me, CF3, Ts, SiMe3, Ph, SMe), 부착 direct · 아릴 따로;
                           (2) carbonyl ylide × R2 계열 × 골격 A · C · D. 칸마다 분류 건수, no_ts 의 판정 건수
  missing_cause_by_dipole.csv  dipole × 계열 × path_verdict 건수 (전체 그림)
  missing_cause.json       요약: no_ts 판정 분포, 지목 칸의 barrierless (주 판정) · no_peak_short_scan · 극대 없음 전체 (민감도) 비율 vs 나머지, 미계산 건수
판정 규칙(사전 고정, 2026-10-14 사용자; d1_archive_scan.py v4 = v3 규칙 + 이름 · 열): hl 경로가 있으면 hl, 없으면 ll 경로. 경로마다 엄격한 내부 극대(E[i-1] < E[i] > E[i+1])가
있으면 peak; 없고 반응물 쪽 끝 frame 의 형성 결합 두 개 중 큰 거리 ≥ 3.0 Å 이면 barrierless (파일럿 T02894: 3.04 Å); 없는데 그 거리 < 3.0 Å 이면
no_peak_short_scan (barrierless 로 세지 않는다). 하나라도 peak 면 peak, 하나라도 짧으면 no_peak_short_scan. autodE 는 고개 없는 경로를 늘 2 × (생성물 결합) 에서
끝내므로 (ratio 2.000), d_react < 3.0 Å 는 생성물 결합 < 1.5 Å 을 뜻한다 → 민감도: 극대 없음 전체 (barrierless + no_peak_short_scan) 비율도 함께 낸다 (사용자 결정 2026-10-14).
held · pending · running (예: parked 된 D'' 행) 은 실패가 아니라 미계산이다 — ok 비율의 분모에서 빼고 따로 센다 (n_not_computed).
unread = 아카이브를 읽지 못한 no_ts 행 (d1_archive_scan.py 가 rc 3 으로 알린다) — 고개 판정에 넣지 않고 n_unread 로 보고.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ST, MAN, SCAN, RD = sys.argv[1:5]
EWG = ["NO2", "CN", "CHO", "Ac", "CO2Me", "CF3", "Ts", "SiMe3", "Ph", "SMe"]
S = pd.read_csv(ST, dtype=str, keep_default_na=False)
M = pd.read_csv(MAN, dtype=str, keep_default_na=False)
A = pd.read_csv(SCAN, dtype={"a_idx_meta": str, "a_idx_tag": str, "hl_frames": str, "path_detail": str})
if "dform_ratio_min" not in A.columns:
    sys.exit(f"{SCAN}: made by d1_archive_scan.py before v4 (no d_start / ratio columns) — rescan first")
cols = ["job_id", "dipole", "skeleton", "series", "R1", "R2", "attach", "cell_id", "core"]
D = S[["job_id", "class"]].merge(M[cols], on="job_id", how="left", validate="1:1")
D["R"] = np.where(D.series == "R1", D.R1, D.R2)
D["X"] = D.R.str.replace(r"^[pmo]-", "", regex=True)
D = D.merge(A[A["class"] == "no_ts"][["job_id", "path_verdict", "hl_peak_max_kcal", "ll_peak_max_kcal", "n_hl_paths",
                                       "n_ll_paths", "dform_react_min_A", "dform_start_min_A", "dform_ratio_min", "path_detail", "log_no_peak", "log_barrierless",
                                       "parse_error"]],
            on="job_id", how="left")
nt = D[D["class"] == "no_ts"].copy()
nt["path_verdict"] = nt.path_verdict.fillna("not_scanned")
nt.to_csv(os.path.join(RD, "missing_cause.csv"), index=False)


def cell_rows(sub, label):
    r = dict(cell=label, n=len(sub))
    for c, k in sub["class"].value_counts().items():
        r[f"class_{c}"] = int(k)
    v = sub.loc[sub["class"] == "no_ts", "path_verdict"].fillna("not_scanned").value_counts()
    for c, k in v.items():
        r[f"no_ts_{c}"] = int(k)
    n_nt = int((sub["class"] == "no_ts").sum())
    r["no_ts_barrierless_frac"] = float(v.get("barrierless", 0) / n_nt) if n_nt else np.nan
    r["no_ts_short_scan_frac"] = float(v.get("no_peak_short_scan", 0) / n_nt) if n_nt else np.nan
    r["no_ts_no_peak_any_frac"] = float((v.get("barrierless", 0) + v.get("no_peak_short_scan", 0)) / n_nt) if n_nt else np.nan   # 민감도
    return r


rows = []
ay = D[(D.dipole == "azomethine ylide") & (D.series == "R2") & D.skeleton.isin(["A", "C"]) & D.X.isin(EWG)]
for (sk, att), g in ay.groupby(["skeleton", np.where(ay.attach == "direct", "direct", "aryl")]):
    rows.append(cell_rows(g, f"azomethine ylide | {sk} | R2 EWG/큰 X | {att}"))
for sk, g in ay.groupby("skeleton"):
    for x, gx in g.groupby("X"):
        rows.append(cell_rows(gx, f"azomethine ylide | {sk} | R2 | X={x} (direct+aryl)"))
cy = D[(D.dipole == "carbonyl ylide") & (D.series == "R2") & D.skeleton.isin(["A", "C", "D"])]
for sk, g in cy.groupby("skeleton"):
    rows.append(cell_rows(g, f"carbonyl ylide | {sk} | R2 계열 전체"))
C = pd.DataFrame(rows)
cnt = [c for c in C.columns if c.startswith(("class_", "no_ts_")) and not c.endswith("_frac")]
C[cnt] = C[cnt].fillna(0).astype(int)                    # counts 0; a fraction with no no_ts row stays empty
C.to_csv(os.path.join(RD, "missing_cause_cells.csv"), index=False)

B = (nt.groupby(["dipole", "series", "path_verdict"]).size().rename("n").reset_index())
B.to_csv(os.path.join(RD, "missing_cause_by_dipole.csv"), index=False)

target = set(ay.job_id) | set(cy.job_id)
inT = nt.job_id.isin(target)
rd = nt.path_verdict != "unread"                         # unread rows carry no verdict: out of the fractions
frac = lambda s, v="barrierless": float((s == v).mean()) if len(s) else np.nan
OPEN = ("held", "pending", "running")
Dc = D[~D["class"].isin(OPEN)]
pkn = nt[nt.path_verdict == "peak"]
pk = pkn.hl_peak_max_kcal.where(pkn.n_hl_paths.fillna(0) > 0, pkn.ll_peak_max_kcal).dropna()   # the level the verdict used
summ = dict(n_rows=len(D), class_counts=D["class"].value_counts().to_dict(),
            n_held=int((D["class"] == "held").sum()), n_no_ts=len(nt),
            no_ts_verdict=nt.path_verdict.value_counts().to_dict(),
            barrierless_frac_in_named_cells=frac(nt.loc[inT & rd, "path_verdict"]), n_no_ts_in_named_cells=int(inT.sum()),
            barrierless_frac_elsewhere=frac(nt.loc[~inT & rd, "path_verdict"]), n_no_ts_elsewhere=int((~inT).sum()),
            short_scan_frac_in_named_cells=frac(nt.loc[inT & rd, "path_verdict"], "no_peak_short_scan"),
            no_peak_any_frac_in_named_cells=frac(nt.loc[inT & rd, "path_verdict"].replace("no_peak_short_scan", "barrierless")),   # 민감도
            short_scan_frac_elsewhere=frac(nt.loc[~inT & rd, "path_verdict"], "no_peak_short_scan"),
            no_peak_any_frac_elsewhere=frac(nt.loc[~inT & rd, "path_verdict"].replace("no_peak_short_scan", "barrierless")),   # 민감도
            dform_ratio_quantiles_no_peak=nt.loc[nt.path_verdict.isin(["barrierless", "no_peak_short_scan"]), "dform_ratio_min"].quantile([0, 0.5, 1]).round(4).to_dict(),
            dform_react_min_A_quantiles_no_peak=nt.loc[nt.path_verdict.isin(["barrierless", "no_peak_short_scan"]), "dform_react_min_A"]
            .quantile([0, 0.1, 0.5, 0.9, 1]).round(3).to_dict(),
            # ok rates over COMPUTED rows only: held · pending · running (e.g. parked D'' rows) are not failures
            n_not_computed=int(D["class"].isin(OPEN).sum()), not_computed_by_class=D.loc[D["class"].isin(OPEN), "class"].value_counts().to_dict(),
            named_cells_ok_rate=float((Dc[Dc.job_id.isin(target)]["class"] == "ok").mean()) if target else np.nan,
            other_ok_rate=float((Dc[~Dc.job_id.isin(target)]["class"] == "ok").mean()),
            n_unread=int((nt.path_verdict == "unread").sum()),
            # peak size (the rule counts any strict interior maximum): how many peak verdicts are tiny bumps or spikes
            peak_kcal_quantiles=pk.quantile([0, 0.1, 0.5, 0.9, 1]).round(3).to_dict(),
            n_peak_below_0p5_kcal=int((pk < 0.5).sum()), n_peak_above_30_kcal=int((pk > 30).sum()),
            rule="hl 경로(없으면 ll) 마다: 엄격한 내부 극대 → peak; 없고 반응물 쪽 끝 형성 결합 큰 거리 ≥ 3.0 Å → barrierless; 없고 < 3.0 Å → no_peak_short_scan (barrierless 로 세지 않음; autodE 끝점 = 2 × 생성물 결합이므로 생성물 결합 < 1.5 Å 이면 늘 이쪽) — 민감도 no_peak_any = 둘의 합",
            verdict_x_log_no_peak={str(k): int(v) for k, v in nt.groupby(["path_verdict", np.where(nt.log_no_peak.fillna(0) > 0, "log_no_peak", "no_log_no_peak")]).size().items()},
            note="barrierless 가 지목 칸에 몰리면 '가장 반응성 큰 구석이 라벨로 존재하지 않음' = MNAR (Methods 에 실패율과 함께)")
json.dump(summ, open(os.path.join(RD, "missing_cause.json"), "w"), indent=1, ensure_ascii=False)
print(C.to_string(index=False)); print(json.dumps(summ, indent=1, ensure_ascii=False))
