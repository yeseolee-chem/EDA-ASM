#!/usr/bin/env python3
r"""prereg_abs.py — ABS3 Phase R (docs/specs/ABS3_MODELS.md 6; same rules as analysis/ch3m/prereg_m.py).
Exit 0 ok, 2 not ready, 3 STOP.

  python prereg_abs.py write   -> results/PREREG_ABS3.md + prereg_abs3.json (once: refuses when the md is tracked). Needs
                                  results/LS_report.json all_pass, C_tests.json all_pass, EA_selftest.json from the current
                                  eval_abs.py, and no K result ($ABS3/units/A__*, results/KA_*). The md quotes spec
                                  section 0 (questions, H2, cells, differences, design) and section 8 (metrics,
                                  interpretation rules), the eval_abs DEFINITIONS, the selftest and the sha256 of every input.
  python prereg_abs.py check   the gate of train_abs.py / xgb_abs.py run: both files tracked and unchanged since HEAD,
                                  sha256(md) = json md_sha256, every inputs_sha256 file unchanged.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402

MD = C.RES / "PREREG_ABS3.md"
JSON = C.RES / "prereg_abs3.json"
SPEC = C.REPO / "docs" / "specs" / "ABS3_MODELS.md"
VERSION = "abs3-prereg-1"


def git(*a):
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)


def tracked(p):
    return git("ls-files", "--error-unmatch", str(p)).returncode == 0


def unchanged(p):
    return git("diff", "--quiet", "HEAD", "--", str(p)).returncode == 0


def inputs():
    fu2 = C.CH3_RES / "lf_inputs" / "fu2_lf"
    return [C.CH3_LF / "X182_lf.parquet", fu2 / "T_manifest.csv", fu2 / "T_report.json", fu2 / "Q_report.json", C.LABELS,
            C.LABELS_JSON, C.LABELS_PQ, C.V_REPORT, C.S_SPLITS, C.S_FOLDS, *C.SPLIT_FILES, C.K3_PRED, C.KM_PRED,
            C.EM_NOISE, C.EM_METRICS, C.EM_RANK, C.PACK_NPZ, C.PACK_JSON, C.RES / "LS_report.json",
            C.RES / "C_tests.json", C.RES / "S0_start.json"]


def spec_section(text, start, end):
    lines = text.splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith(start)), None)
    if i is None:
        C.stop(f"{SPEC.name}: heading '{start}' not found")
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith(end)), len(lines))
    return "\n".join(lines[i:j]).rstrip()


def quote(s):
    return "\n".join("> " + ln if ln else ">" for ln in s.splitlines())


def seen_k():
    units = sorted(p.name for p in C.UNITS.glob("A__*")) if C.UNITS.is_dir() else []
    res = sorted(p.name for p in C.RES.glob("KA_*"))
    return dict(units=units, results=res)


def code_files():
    return sorted(HERE.glob("*.py")) + sorted((HERE / "jobs").glob("*.slurm"))


def cmd_write():
    if MD.is_file() and tracked(MD):
        # written once (spec 6): a resubmitted R job finds it committed -> no-op when unchanged (check STOPs otherwise)
        r = check()
        print(f"[prereg] {MD.name} already committed and unchanged (md {r['md_sha256'][:12]}): no-op", flush=True)
        return 0
    s0 = C.read_json(C.RES / "S0_start.json") if (C.RES / "S0_start.json").is_file() else {}
    ls = C.read_json(C.RES / "LS_report.json") if (C.RES / "LS_report.json").is_file() else {}
    ct = C.read_json(C.RES / "C_tests.json") if (C.RES / "C_tests.json").is_file() else {}
    if not s0.get("all_pass"):
        C.stop(f"S0_start.json all_pass {s0.get('all_pass')} (spec 12 stage 1)")
    if not ls.get("all_pass") or not ct.get("all_pass"):
        C.not_ready(f"LS all_pass {ls.get('all_pass')} / C_tests all_pass {ct.get('all_pass')}")
    loose = [str(p.relative_to(C.REPO)) for p in code_files() if not (tracked(p) and unchanged(p))]
    if loose:
        C.stop(f"analysis/abs3 code not committed / modified since HEAD (spec 10): {loose}")
    import eval_abs as EA
    st = C.read_json(C.RES / "EA_selftest.json") if (C.RES / "EA_selftest.json").is_file() else {}
    if (st.get("code_sha256") or {}).get("analysis/abs3/eval_abs.py") != C.sha256(HERE / "eval_abs.py"):
        C.not_ready("results/EA_selftest.json missing or made by another eval_abs.py: rerun eval_abs.py selftest")
    s = seen_k()
    if any(s.values()):
        C.stop(f"K results exist before the pre-registration: {s}")
    miss = [str(p) for p in inputs() if not p.is_file()]
    if miss:
        C.not_ready(f"inputs missing: {miss}")
    ins = {str(p): C.sha256(p) for p in inputs()}
    spec = SPEC.read_text()
    q = dict(zero=spec_section(spec, "## 0.", "## 1."), eval=spec_section(spec, "## 8.", "## 9."))
    head = git("rev-parse", "HEAD").stdout.strip()
    now = C.now()
    md = [f"# PREREG ABS3 — 반응 하나의 3채널 절대값 모델(TF · GNN · XGB) 대 짝(ΔΔ) 모델 사전 등록 ({now[:10]})", "",
          f"`docs/specs/ABS3_MODELS.md` 6절의 사전 등록이다(명령서 sha256 `{C.sha256(SPEC)}`). `analysis/abs3/prereg_abs.py`"
          f"({VERSION})가 결과 파일만 읽어 만들었다(git HEAD `{head}`, 작성 {now}). 인용 블록 = 명령서 원문. "
          "**K(`train_abs.py run`, `xgb_abs.py run`)는 이 md와 `prereg_abs3.json`이 commit되고 바뀌지 않았을 때만 "
          "돈다**(`prereg_abs.py check`).", "",
          "## 1. 질문, 가설 H2, 실험 칸, 짝 모델과 다른 점, 공통 설계 (0절 전체)", quote(q["zero"]), "",
          "## 2. 평가: 지표 정의와 해석 규칙 (8절)", quote(q["eval"]), "",
          "### 2-1. 구현 정의 (`eval_abs.py` DEFINITIONS, K 결과를 보기 전에 고정)", ""]
    md += [f"- **{k}**: {v}" for k, v in EA.DEFINITIONS.items()]
    md += ["", "### 2-2. selftest (`results/EA_selftest.json`, 8-6)", ""]
    md += [f"- {k}: {'통과' if v else '불일치(정의 차이 보고)'}" for k, v in (st.get("checks") or {}).items()]
    md += ["", "## 3. 지금까지 본 것",
           f"- K 학습 결과: **0개** (`$ABS3/units/A__*`, `results/KA_*` 없음; {now})",
           "- 본 것: 반응별 라벨 · 분할 검사(LS), 짝 모델 예측(TF3 · GNN3 · XGB3, CH3M에서 이미 공개), C 검사(과적합 시험 "
           "64반응, smoke 2 epoch / 2칸)",
           f"- LS: all_pass {ls.get('all_pass')}, 반응 {((ls.get('LS4') or {}).get('n_reactions'))}개; C 검사: all_pass "
           f"{ct.get('all_pass')}", "",
           "## 4. 입력 · 기준 파일 sha256", "", "| 파일 | sha256 |", "|---|---|"]
    md += [f"| `{p}` | `{h}` |" for p, h in ins.items()]
    md += ["", "## 5. 코드 sha256 (작성 시)", "", "| 파일 | sha256 |", "|---|---|"]
    code = {str(p.relative_to(C.REPO)): C.sha256(p) for p in code_files()}
    md += [f"| `{p}` | `{h}` |" for p, h in code.items()]
    C.write_text(MD, "\n".join(md) + "\n")
    rec = dict(version=VERSION, written=now, git_head=head, spec=str(SPEC.relative_to(C.REPO)), spec_sha256=C.sha256(SPEC),
               md=str(MD.relative_to(C.REPO)), md_sha256=C.sha256(MD), inputs_sha256=ins, definitions=EA.DEFINITIONS,
               rules=dict(noise_floor=EA.NOISE_FLOOR, marks=list(EA.MARKS), noise_source="CH3M EM_noise.csv |r(TF10 vs dep)|",
                          best_rule="4-target P_main para MAE mean, ties A-TF -> A-GNN -> A-XGB", best_ref=EA.BEST_REF,
                          n_boot=C.N_BOOT, seed=C.SEED),
               seen=dict(k_results=s, ea_selftest_sha256=C.sha256(C.RES / "EA_selftest.json")), code_sha256=code)
    C.write_json(JSON, rec)
    print(f"[prereg] wrote {MD.name} ({rec['md_sha256'][:12]}) and {JSON.name}; commit both before K", flush=True)
    return 0


def check():
    bad = []
    for p in (MD, JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif not tracked(p):
            bad.append(f"{p.name} not committed")
        elif not unchanged(p):
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = C.read_json(JSON)
        if C.sha256(MD) != rec.get("md_sha256"):
            bad.append(f"{MD.name} sha256 differs from {JSON.name} md_sha256")
        for path, s in (rec.get("inputs_sha256") or {}).items():
            if not Path(path).is_file() or C.sha256(path) != s:
                bad.append(f"input {path} changed or missing")
    if bad:
        print("PREREG_ABS3 gate (spec 6: commit PREREG_ABS3.md + prereg_abs3.json before K):\n  " + "\n  ".join(bad),
              file=sys.stderr, flush=True)
        raise C.Stop(3)
    return dict(rec, md_sha256=C.sha256(MD), git_head=git("rev-parse", "HEAD").stdout.strip())


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["write"]:
        sys.exit(cmd_write())
    if a == ["check"]:
        r = check()
        print(json.dumps(dict(ok=True, md_sha256=r["md_sha256"], git_head=r["git_head"]), indent=1))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
