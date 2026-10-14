"""b2_spotcheck.py — Phase B 전 점검: d1_archive_scan.py (v3) 의 no_ts 판정을 표본 작업에 그대로 돌린다 (기존 출력만 읽음, 새 QM 없음).

python b2_spotcheck.py SAMPLE_TSV NURION_TAR_DIR UBAI_JOBS_DIR OUT_DIR
  SAMPLE_TSV  job_id <TAB> bundle <TAB> group   (bundle 이 비면 UBAI 작업 폴더 <UBAI_JOBS_DIR>/<T>/)
출력 (OUT_DIR):
  b2_v3_spotcheck.csv        작업 하나에 한 줄 = d1_archive_scan.summarize() 결과 그대로 (Phase B 가 쓰는 것과 같은 함수) + group
  b2_v3_spotcheck_paths.csv  경로 파일 하나에 한 줄 = d1_archive_scan.path_info() 결과 + 첫 · 끝 frame 의 형성 결합 거리 · 에너지
                             (반응물 쪽 끝이 에너지가 높은 끝인지 · 3.0 Å 근처인지 사람이 보도록)
  b2_v3_spotcheck.json       판정 분포, 파일럿 T02894 기대값(barrierless, d_react 3.037 Å) 대조, 읽기 실패
"""
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_archive_scan as S  # noqa: E402

SAMPLE, TARS, UBAI, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
smp = pd.read_csv(SAMPLE, sep="\t", dtype=str, keep_default_na=False)


def ends(name, text):
    """(E_first, E_last, d_first, d_last): energies and the longer forming bond of the two end frames (plain re-parse)."""
    L = [l for l in text.splitlines()]
    fr, i = [], 0
    while i < len(L):
        if not L[i].strip():
            i += 1; continue
        n = int(L[i].split()[0]); m = S.E_RE.search(L[i + 1])
        fr.append((float(m.group(1)) if m else np.nan, np.array([[float(v) for v in l.split()[1:4]] for l in L[i + 2:i + 2 + n]])))
        i += n + 2
    b = S.path_bonds(name)
    d = [max(float(np.linalg.norm(x[p] - x[q])) for p, q in b) if b else np.nan for _, x in (fr[0], fr[-1])]
    return fr[0][0], fr[-1][0], d[0], d[1]


rows, paths, errs = [], [], []
for r in smp.itertuples():
    got = {}
    try:
        if r.bundle:
            with tarfile.open(TARS / f"{r.bundle}.tar", mode="r:") as T:
                m = next(x for x in T if x.name.lstrip("./") == f"jobs/{r.job_id}/archive.tgz")
                S.read_archive(T.extractfile(m), "no_ts", got)
            src = r.bundle
        else:
            d = UBAI / r.job_id
            if (d / "archive.tgz").is_file():
                with open(d / "archive.tgz", "rb") as f:
                    S.read_archive(f, "no_ts", got)
                src = "ubai:archive"
            else:
                S.read_dir(d, "no_ts", got); src = "ubai:dir"
    except Exception as e:                                               # noqa: BLE001
        errs.append(dict(job_id=r.job_id, error=f"{type(e).__name__}: {e}"))
        continue
    s = S.summarize(r.job_id, "no_ts", src, got)
    s["group"] = r.group
    rows.append(s)
    for nm, txt in got.get("ts", []):
        if S.PATH_RE.search(nm):
            p = dict(job_id=r.job_id, name=nm.split("/")[-1], level=S.PATH_RE.search(nm).group(1))
            try:
                p.update(S.path_info(nm, txt))
                ef, el, df, dl = ends(nm, txt)
                p.update(E_first=ef, E_last=el, d_first=df, d_last=dl,
                         react_end_is_higher_E=bool((p["end"] == "last") == (el > ef)) if p["end"] else None,
                         near_threshold=bool(np.isfinite(p["d_react"]) and abs(p["d_react"] - S.DFORM_MIN_A) < 0.05))
            except Exception as e:                                       # noqa: BLE001
                p.update(verdict="error", error=f"{type(e).__name__}: {e}")
            paths.append(p)
J = pd.DataFrame(rows); P = pd.DataFrame(paths)
J.to_csv(OUT / "b2_v3_spotcheck.csv", index=False)
P.to_csv(OUT / "b2_v3_spotcheck_paths.csv", index=False)
pilot = J[J.job_id == "T02894"]
rep = dict(scan_version=S.SCAN_VERSION, dform_min_A=S.DFORM_MIN_A, n_sample=len(smp), n_read=len(J), read_errors=errs,
           parse_errors={k: v for k, v in zip(J.job_id, J.parse_error) if isinstance(v, str) and v} if len(J) else {},
           job_verdicts=J.path_verdict.value_counts().to_dict() if len(J) else {},
           job_verdicts_by_group={g: x.path_verdict.value_counts().to_dict() for g, x in J.groupby("group")} if len(J) else {},
           path_verdicts=P.verdict.value_counts().to_dict() if len(P) else {},
           paths_react_end_not_higher_E=int((P.react_end_is_higher_E == False).sum()) if len(P) else 0,   # noqa: E712
           paths_near_3A=P.loc[P.near_threshold == True, ["job_id", "name", "d_react"]].to_dict("records") if len(P) else [],  # noqa: E712
           pilot_T02894=dict(verdict=pilot.path_verdict.iloc[0] if len(pilot) else None,
                             dform_react_min_A=float(pilot.dform_react_min_A.iloc[0]) if len(pilot) else None,
                             expected="barrierless, 3.037"),
           pilot_ok=bool(len(pilot) and pilot.path_verdict.iloc[0] == "barrierless" and abs(pilot.dform_react_min_A.iloc[0] - 3.037) < 0.002))
with open(OUT / "b2_v3_spotcheck.json", "w") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
cols = [c for c in ["job_id", "group", "path_verdict", "dform_react_min_A", "n_hl_paths", "n_ll_paths", "hl_peak_max_kcal", "log_no_peak", "parse_error"] if c in J]
print(J[cols].to_string(index=False))
sys.exit(0 if rep["pilot_ok"] and not errs else 1)
