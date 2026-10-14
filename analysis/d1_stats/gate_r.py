"""gate_r.py — 게이트 R (SPEC_D1_ALL Phase R): results_interim/headline.json 이 reference_interim/headline.json 과
모든 키에서 같은지 본다. 정수·문자열은 그대로, 실수는 소수 셋째 자리(round 3)에서 비교한다.

python gate_r.py RESULT_HEADLINE REFERENCE_HEADLINE [OUT_MD]
종료 코드 0 = 통과, 1 = 불일치(차이 표를 출력·OUT_MD 에 쓴다 — STOP 하고 보고). OUT_MD 옆에 gate_r.json (passed) 을 쓴다.
"""
import json, sys


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{pre}[{i}]."))
    else:
        out[pre[:-1]] = d
    return out


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    try:
        return round(float(a), 3) == round(float(b), 3)
    except (TypeError, ValueError):
        return a == b


got, ref = (flat(json.load(open(p))) for p in sys.argv[1:3])
rows = []
for k in sorted(set(got) | set(ref)):
    g, r = got.get(k, "<없음>"), ref.get(k, "<없음>")
    ok = k in got and k in ref and same(g, r)
    rows.append((k, r, g, ok))
bad = [x for x in rows if not x[3]]
lines = ["| 키 | 기준값 | 재현값 | 일치 |", "|---|---|---|---|"] + [f"| {k} | {r} | {g} | {'예' if ok else '**아니오**'} |" for k, r, g, ok in rows]
summary = f"게이트 R: 키 {len(rows)}개 중 불일치 {len(bad)}개 → {'PASS' if not bad else 'FAIL (STOP)'}"
print(summary)
for k, r, g, _ in bad:
    print(f"  {k}: 기준 {r} · 재현 {g}")
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write("# 게이트 R 일치표\n\n" + summary + "\n\n" + "\n".join(lines) + "\n")
    # gate_r.json next to the table: phase_a.slurm refuses to start unless passed is true
    with open(sys.argv[3].rsplit("/", 1)[0] + "/gate_r.json" if "/" in sys.argv[3] else "gate_r.json", "w") as f:
        json.dump(dict(passed=not bad, n_keys=len(rows), n_mismatch=len(bad), mismatches=[k for k, *_ in bad]),
                  f, indent=1, ensure_ascii=False)
sys.exit(0 if not bad else 1)
