"""missing_cause.py — SPEC_D1_ALL B2 "꼭 답할 것": 빈 칸이 no_ts 인지, 그중 고개가 없는(경로 에너지가 계속 내려가는) 경우인지.

python missing_cause.py STATUS_CSV MANIFEST ARCHIVE_SCAN_CSV RESULTS_DIR
  STATUS_CSV        확정 d1_status.csv (분류 열 class)
  MANIFEST          d1_manifest.csv (dipole, skeleton, series, R1, R2, attach)
  ARCHIVE_SCAN_CSV  d1_archive_scan.py 출력 (no_ts 행의 path_verdict: barrierless / peak / no_path / unparsed)
출력 (RESULTS_DIR):
  missing_cause.csv        no_ts 행 하나씩: 칸 정보 + path_verdict + 경로 고개 높이 + autodE 로그 표식
  missing_cause_cells.csv  SPEC 가 지목한 칸: (1) azomethine ylide × 골격 A · C × dipolarophile 쪽(R2) 전자 끄는/큰 X
                           (NO2, CN, CHO, Ac, CO2Me, CF3, Ts, SiMe3, Ph, SMe), 부착 direct · 아릴 따로;
                           (2) carbonyl ylide × R2 계열 × 골격 A · C · D. 칸마다 분류 건수, no_ts 의 판정 건수
  missing_cause_by_dipole.csv  dipole × 계열 × path_verdict 건수 (전체 그림)
  missing_cause.json       요약: no_ts 판정 분포, 지목 칸의 고개 없음 비율 vs 나머지, held(미계산) 건수
판정 규칙(사전 고정, d1_archive_scan.py): hl 경로가 있으면 hl, 없으면 ll 경로. 어느 경로에도 엄격한 내부 극대(E[i-1] < E[i] > E[i+1])가
없으면 barrierless (파일럿 T02894 와 같은 판정: 생성물 쪽에서 반응물 쪽으로 에너지가 계속 오른다). 크기와 무관하게 극대가 하나라도 있으면 peak.
held 는 실패가 아니라 미계산이다 — 따로 센다.
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
A = pd.read_csv(SCAN, dtype={"a_idx_meta": str, "a_idx_tag": str, "hl_frames": str})
cols = ["job_id", "dipole", "skeleton", "series", "R1", "R2", "attach", "cell_id", "core"]
D = S[["job_id", "class"]].merge(M[cols], on="job_id", how="left", validate="1:1")
D["R"] = np.where(D.series == "R1", D.R1, D.R2)
D["X"] = D.R.str.replace(r"^[pmo]-", "", regex=True)
D = D.merge(A[A["class"] == "no_ts"][["job_id", "path_verdict", "hl_peak_max_kcal", "ll_peak_max_kcal", "n_hl_paths",
                                       "n_ll_paths", "log_no_peak", "log_barrierless", "parse_error"]],
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
C = pd.DataFrame(rows).fillna(0)
C.to_csv(os.path.join(RD, "missing_cause_cells.csv"), index=False)

B = (nt.groupby(["dipole", "series", "path_verdict"]).size().rename("n").reset_index())
B.to_csv(os.path.join(RD, "missing_cause_by_dipole.csv"), index=False)

target = set(ay.job_id) | set(cy.job_id)
inT = nt.job_id.isin(target)
frac = lambda s: float((s == "barrierless").mean()) if len(s) else np.nan
summ = dict(n_rows=len(D), class_counts=D["class"].value_counts().to_dict(),
            n_held=int((D["class"] == "held").sum()), n_no_ts=len(nt),
            no_ts_verdict=nt.path_verdict.value_counts().to_dict(),
            barrierless_frac_in_named_cells=frac(nt.loc[inT, "path_verdict"]), n_no_ts_in_named_cells=int(inT.sum()),
            barrierless_frac_elsewhere=frac(nt.loc[~inT, "path_verdict"]), n_no_ts_elsewhere=int((~inT).sum()),
            named_cells_ok_rate=float((D[D.job_id.isin(target)]["class"] == "ok").mean()) if target else np.nan,
            other_ok_rate=float((D[~D.job_id.isin(target)]["class"] == "ok").mean()),
            rule="hl 경로(없으면 ll) 어디에도 엄격한 내부 극대가 없으면 barrierless; 하나라도 있으면 peak (autodE path.is_saddle 과 같은 검사)",
            verdict_x_log_no_peak=pd.crosstab(nt.path_verdict, nt.log_no_peak.fillna(-1).astype(int) > 0).to_dict(),
            note="barrierless 가 지목 칸에 몰리면 '가장 반응성 큰 구석이 라벨로 존재하지 않음' = MNAR (Methods 에 실패율과 함께)")
json.dump(summ, open(os.path.join(RD, "missing_cause.json"), "w"), indent=1, ensure_ascii=False)
print(C.to_string(index=False)); print(json.dumps(summ, indent=1, ensure_ascii=False))
