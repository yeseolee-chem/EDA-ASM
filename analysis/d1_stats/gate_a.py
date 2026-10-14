"""gate_a.py — 게이트 A (SPEC_D1_ALL Phase A): 확정 데이터의 QC.

python gate_a.py RESULTS_FINAL_DIR FINAL_LABELS INTERIM_LABELS
  1. qc.json: G1–G3 탈락 0 (n_gates_failed), closure_max ≤ 0.11, orient_counts 의 값이 a · b · "" 뿐
     (noring · tie · err:* 등 그 밖의 값이 하나라도 있으면 실패)
  2. 확정 파일의 열 구성이 중간 파일과 같다 (모든 행의 키 합집합 비교; 다르면 §9 STOP)
  3. 제외 건수(q_badref · q_stepwise)를 중간 파일 기준(10 · 218)과 나란히 적는다 (판정 아님)
  4. inspect_large_ddbarrier.csv 에 부모·자식의 alt_used 를 붙여 inspect_large_ddbarrier_alt.csv 로 쓴다
     (사람이 하나씩 열어 판단만 적는다 — 제외하지 않는다)
결과: RESULTS_FINAL_DIR/gate_a.json · 종료 코드 0 = 통과, 1 = 실패(STOP)
"""
import csv, json, sys
from pathlib import Path

RD, FIN, INT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
qc = json.load(open(RD / "qc.json"))
fails = []
if qc["n_gates_failed"] != 0:
    fails.append(f"G1–G3 탈락 {qc['n_gates_failed']}행")
if not qc["closure_max"] <= 0.11:
    fails.append(f"closure_max {qc['closure_max']:.4f} > 0.11")
odd = {k: v for k, v in qc["orient_counts"].items() if k not in ("a", "b", "")}
if odd:
    fails.append(f"orient_counts 에 a · b · '' 밖의 값: {odd}")


def keys(path):
    ks = set()
    for r in json.load(open(path)):
        ks |= set(r.keys())
    return ks


kf, ki = keys(FIN), keys(INT)
if kf != ki:
    fails.append(f"열 구성이 중간 파일과 다르다: 확정에만 {sorted(kf - ki)} · 중간에만 {sorted(ki - kf)}")
# 4. large ΔΔE‡ pairs: attach alt_used of child and parent
alt = {r["job_id"]: r.get("alt_used") for r in json.load(open(FIN))}
src = RD / "inspect_large_ddbarrier.csv"
n_big = n_diff = 0
if src.is_file():
    rows = list(csv.DictReader(open(src)))
    for r in rows:
        r["alt_child"], r["alt_parent"] = alt.get(r["job_id"]), alt.get(r["parent_job"])
        r["alt_differs"] = r["alt_child"] != r["alt_parent"]
        r["judgement"] = ""
        n_big += 1; n_diff += bool(r["alt_differs"])
    if rows:
        with open(RD / "inspect_large_ddbarrier_alt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
out = dict(passed=not fails, fails=fails, n_rows=qc["n_rows"],
           exclusions=dict(q_badref=qc["q_badref"], q_stepwise=qc["q_stepwise"], q_bad=qc["q_bad"],
                           interim_reference=dict(q_badref=10, q_stepwise=218)),
           closure_max=qc["closure_max"], orient_counts=qc["orient_counts"],
           large_ddbarrier_pairs=n_big, large_pairs_alt_used_differs=n_diff)
json.dump(out, open(RD / "gate_a.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
sys.exit(0 if not fails else 1)
