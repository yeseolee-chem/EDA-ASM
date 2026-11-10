#!/usr/bin/env python3
r"""prereg_m.py — CH3M Phase R (docs/specs/CH3M_MODELS.md 6; same rules as analysis/ch3/prereg_ch3.py).
Exit 0 ok, 2 not ready, 3 STOP.

  python prereg_m.py write   -> results/PREREG_CH3M.md + prereg_ch3m.json (once: refuses when the md is tracked). Needs
                                results/XP_report.json ok, C_tests.json all_pass, EM_selftest.json from the current
                                eval_m.py, and no K result (no $CH3M/units/D__*, results/KM_*). The md quotes the spec
                                (0-2, 0-3 with H1, 0-4, section 8 incl. 8-2 / 8-3), the eval_m DEFINITIONS, the selftest,
                                and the sha256 of the comparator files, labels, splits, inputs and XGB packs.
  python prereg_m.py check   the gate of train_m.py / xgb_m.py run and final: both files tracked and unchanged since HEAD,
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
import ch3m_common as C  # noqa: E402

MD = C.RES / "PREREG_CH3M.md"
JSON = C.RES / "prereg_ch3m.json"
SPEC = C.REPO / "docs" / "specs" / "CH3M_MODELS.md"
VERSION = "ch3m-prereg-1"


def git(*a):
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)


def tracked(p):
    return git("ls-files", "--error-unmatch", str(p)).returncode == 0


def unchanged(p):
    return git("diff", "--quiet", "HEAD", "--", str(p)).returncode == 0


def inputs():
    fu2 = C.CH3_RES / "lf_inputs" / "fu2_lf"
    files = [C.CH3_LF / "X182_lf.parquet", fu2 / "T_manifest.csv", fu2 / "T_report.json", fu2 / "Q_report.json", C.LABELS,
             C.REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json",
             C.REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv",
             C.DEPLOY_RES / "K_predictions.parquet", C.CH3_RES / "K3_predictions.parquet",
             C.LAD_RES / "L_predictions.parquet", C.XPACK / "lf.npz", C.XPACK / "lf.json", C.XPACK / "g2.npz",
             C.XPACK / "g2.json", C.RES / "XP_report.json", C.RES / "C_tests.json", C.RES / "S0_start.json"]
    files += [C.D1PAIR / "fu2" / "splits" / f"T1__f{k}.json" for k in range(5)]
    if (C.RES / "C_deviation.json").is_file():
        files.append(C.RES / "C_deviation.json")
    return files


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
    units = sorted(p.name for p in C.UNITS.glob("D__*")) if C.UNITS.is_dir() else []
    res = sorted(p.name for p in C.RES.glob("KM_*"))
    return dict(units=units, results=res)


def cmd_write():
    if MD.is_file() and tracked(MD):
        C.stop(f"{MD.name} is already tracked by git: the pre-registration is written once (spec 6)")
    xp = C.read_json(C.RES / "XP_report.json") if (C.RES / "XP_report.json").is_file() else {}
    ct = C.read_json(C.RES / "C_tests.json") if (C.RES / "C_tests.json").is_file() else {}
    if not xp.get("ok") or not ct.get("all_pass"):
        C.not_ready(f"XP ok {xp.get('ok')} / C_tests all_pass {ct.get('all_pass')}")
    import eval_m as EM
    st = C.read_json(C.RES / "EM_selftest.json") if (C.RES / "EM_selftest.json").is_file() else {}
    if (st.get("code_sha256") or {}).get("analysis/ch3m/eval_m.py") != C.sha256(HERE / "eval_m.py"):
        C.not_ready("results/EM_selftest.json missing or made by another eval_m.py: rerun eval_m.py selftest")
    s = seen_k()
    if any(s.values()):
        C.stop(f"K results exist before the pre-registration: {s}")
    miss = [str(p) for p in inputs() if not p.is_file()]
    if miss:
        C.not_ready(f"inputs missing: {miss}")
    ins = {str(p): C.sha256(p) for p in inputs()}
    spec = SPEC.read_text()
    q = dict(cells=spec_section(spec, "### 0-2.", "### 0-3."), known=spec_section(spec, "### 0-3.", "### 0-4."),
             design=spec_section(spec, "### 0-4.", "---"), eval=spec_section(spec, "## 8.", "## 9."),
             final=spec_section(spec, "## 9.", "## 10."))
    head = git("rev-parse", "HEAD").stdout.strip()
    now = C.now()
    md = [f"# PREREG CH3M — 3채널 직접 대 10타깃 합 (TF · GNN · XGB) 사전 등록 ({now[:10]})", "",
          f"`docs/specs/CH3M_MODELS.md` 6절의 사전 등록이다(명령서 sha256 `{C.sha256(SPEC)}`). `analysis/ch3m/prereg_m.py`"
          f"({VERSION})가 결과 파일만 읽어 만들었다(git HEAD `{head}`, 작성 {now}). 인용 블록 = 명령서 원문. "
          "**K(`train_m.py run`, `xgb_m.py run`)와 W(`train_m.py final`)는 이 md와 `prereg_ch3m.json`이 commit되고 바뀌지 "
          "않았을 때만 돈다**(`prereg_m.py check`).", "",
          "## 1. 실험 칸 (0-2)", quote(q["cells"]), "",
          "## 2. 이미 아는 것과 가설 H1 (0-3)", quote(q["known"]), "",
          "## 3. 공통 설계 (0-4)", quote(q["design"]), "",
          "## 4. 평가 정의 전부, 해석 규칙(8-2), 교체 규칙(8-3) (8절)", quote(q["eval"]), "",
          "## 5. 9절 (최종 모델 · 추론기)", quote(q["final"]), "",
          "### 5-1. 구현 정의 (`eval_m.py` DEFINITIONS, K 결과를 보기 전에 고정)", ""]
    md += [f"- **{k}**: {v}" for k, v in EM.DEFINITIONS.items()]
    md += ["", "### 5-2. selftest (`results/EM_selftest.json`)", ""]
    md += [f"- {k}: {'통과' if v else '불일치(정의 차이 보고)'}" for k, v in (st.get("checks") or {}).items()]
    tab = (st.get("detail") or {}).get("ii_table") or []
    if tab:
        md += ["", "| 행 | 명령서 0-3 (strain, inter, solv, barrier) | 재현 | 일치 |", "|---|---|---|---|"]
        md += [f"| {r['row']} | {r['spec']} | {r['value']} | {'예' if r['match'] else '아니오'} |" for r in tab]
    md += ["", "## 6. 지금까지 본 것",
           f"- K 학습 결과: **0개** (`$CH3M/units/D__*`, `results/KM_*` 없음; {now})",
           "- 본 것: 기준 dep · ch3(TF3) 예측과 LADDER g2 예측(0-3 표, selftest), C 검사(g2 fold 0 회귀: TF · GNN · XGB, "
           "smoke 2 epoch / 2칸)",
           f"- C 검사: all_pass {ct.get('all_pass')}"
           + (" (C-RG 예외 규칙 적용: `results/C_deviation.json`)" if (C.RES / "C_deviation.json").is_file() else ""), "",
           "## 7. 입력 · 기준 파일 sha256", "", "| 파일 | sha256 |", "|---|---|"]
    md += [f"| `{p}` | `{h}` |" for p, h in ins.items()]
    md += ["", "## 8. 코드 sha256 (작성 시)", "", "| 파일 | sha256 |", "|---|---|"]
    code = {f"analysis/ch3m/{p.name}": C.sha256(p) for p in sorted(HERE.glob("*.py"))}
    md += [f"| `{p}` | `{h}` |" for p, h in code.items()]
    C.write_text(MD, "\n".join(md) + "\n")
    rec = dict(version=VERSION, written=now, git_head=head, spec=str(SPEC.relative_to(C.REPO)), spec_sha256=C.sha256(SPEC),
               md=str(MD.relative_to(C.REPO)), md_sha256=C.sha256(MD), inputs_sha256=ins, definitions=EM.DEFINITIONS,
               rules=dict(noise_floor=EM.NOISE_FLOOR, replace_a_upper_max=EM.REL_A, replace_b_rel_max=EM.REL_B,
                          replace_b_min_targets=2, tie_order=EM.TIE_ORDER, n_boot=C.N_BOOT, seed=C.SEED),
               seen=dict(k_results=s, em_selftest_sha256=C.sha256(C.RES / "EM_selftest.json")), code_sha256=code)
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
        print("PREREG_CH3M gate (spec 6: commit PREREG_CH3M.md + prereg_ch3m.json before K):\n  " + "\n  ".join(bad),
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
