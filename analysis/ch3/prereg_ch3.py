#!/usr/bin/env python3
r"""prereg_ch3.py — CH3-MODEL Phase R: pre-registration before any ch3 cross-validation training
(docs/specs/CH3_MODEL.md 7; same rules as analysis/deploy/prereg_deploy.py). Any env with python >= 3.9 + git
(jobs/cpu_ch3.slurm CH3ENV=reactot). Exit 0 ok, 2 not ready, 3 STOP.

  python prereg_ch3.py write    -> results/PREREG_CH3.md + results/prereg_ch3.json (written once: refuses when the md is
                                   already tracked by git). Needs results/V_inputs.json all_pass, results/C_tests.json
                                   all_pass, results/E3_selftest.json from the current eval_ch3.py, and no ch3 training
                                   result (no $CH3/units/D__lf3__*, results/K3_*, $CH3/model/D__lf3__*). The md quotes the
                                   spec (0-2 design, section 9 metric definitions incl. the 9-2 adoption rule, 8 training
                                   rules), records the implementation definitions of eval_ch3.py (DEFINITIONS) and its
                                   selftest on lf_sum, the comparator file (deploy K_predictions.parquet) and the sha256
                                   of the labels / splits / inputs.
  python prereg_ch3.py check    the gate of train_ch3.py run / final: both files tracked by git and unchanged since HEAD,
                                   sha256(md) = json md_sha256, every json inputs_sha256 file unchanged.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402

MD = C3.RES / "PREREG_CH3.md"
JSON = C3.RES / "prereg_ch3.json"
SPEC = C3.REPO / "docs" / "specs" / "CH3_MODEL.md"
VERSION = "ch3-prereg-1"
REL_CI_HI_MAX = 0.05


def git(*a):
    return subprocess.run(["git", "-C", str(C3.REPO), *a], capture_output=True, text=True)


def tracked(p):
    return git("ls-files", "--error-unmatch", str(p)).returncode == 0


def unchanged(p):
    return git("diff", "--quiet", "HEAD", "--", str(p)).returncode == 0


def inputs():
    """the files whose sha256 the pre-registration fixes (absolute paths)."""
    fu2 = C3.LF_RES / "fu2_lf"
    files = [C3.LF / "labels_lf.json", C3.LF / "input_meta_lf.csv", C3.LF / "X182_lf.parquet",
             fu2 / "T_manifest.csv", fu2 / "T_report.json", fu2 / "Q_report.json", fu2 / "Q_manifest.csv",
             C3.LABELS, C3.REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json",
             C3.REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv",
             C3.DEPLOY_RES / "K_predictions.parquet", C3.DEPLOY_RES / "K_report.json",
             C3.RES / "V_inputs.json"]
    files += [C3.D1PAIR / "fu2" / "splits" / f"T1__f{k}.json" for k in range(5)]
    return files


def spec_section(text, start, end):
    lines = text.splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith(start)), None)
    if i is None:
        C3.stop(f"{SPEC.name}: heading '{start}' not found")
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith(end)), len(lines))
    return "\n".join(lines[i:j]).rstrip()


def quote(s):
    return "\n".join("> " + ln if ln else ">" for ln in s.splitlines())


def seen_ch3():
    units = sorted(p.name for p in C3.UNITS.glob("D__lf3__*")) if C3.UNITS.is_dir() else []
    res = sorted(p.name for p in C3.RES.glob("K3_*"))
    model = sorted(p.name for p in C3.MODEL_DIR.glob("D__lf3__*")) if C3.MODEL_DIR.is_dir() else []
    return dict(units=units, results=res, model=model)


def _need(path, key="all_pass"):
    p = C3.RES / path
    r = C3.read_json(p) if p.is_file() else {}
    if not r.get(key):
        C3.not_ready(f"{p}: {key} {r.get(key)}")
    return r


def cmd_write():
    if MD.is_file() and tracked(MD):
        C3.stop(f"{MD.name} is already tracked by git: the pre-registration is written once (spec 7)")
    v = _need("V_inputs.json")
    c = _need("C_tests.json")
    sys.path.insert(0, str(HERE))
    import eval_ch3 as EV                                                   # definitions (needs pandas: reactot / espley)
    st = C3.read_json(C3.RES / "E3_selftest.json") if (C3.RES / "E3_selftest.json").is_file() else {}
    if (st.get("code_sha256") or {}).get("analysis/ch3/eval_ch3.py") != C3.sha256(HERE / "eval_ch3.py"):
        C3.not_ready("results/E3_selftest.json missing or made by another eval_ch3.py: rerun eval_ch3.py selftest")
    s = seen_ch3()
    if any(s.values()):
        C3.stop(f"ch3 training results exist before the pre-registration: {s}")
    miss = [str(p) for p in inputs() if not p.is_file()]
    if miss:
        C3.not_ready(f"inputs missing: {miss}")
    ins = {str(p): C3.sha256(p) for p in inputs()}
    spec = SPEC.read_text()
    q = dict(design=spec_section(spec, "### 0-2.", "### 0-3."), train=spec_section(spec, "## 8.", "## 9."),
             eval=spec_section(spec, "## 9.", "## 10."), final=spec_section(spec, "## 10.", "## 11."))
    head = git("rev-parse", "HEAD").stdout.strip()
    now = C3.now()
    c3r = C3.read_json(C3.RES / "C3r_regress.json") if (C3.RES / "C3r_regress.json").is_file() else {}
    sm = C3.read_json(C3.RES / "C8_smoke.json") if (C3.RES / "C8_smoke.json").is_file() else {}
    ref_rows = st.get("reference_table") or []
    md = [f"# PREREG CH3 — 3채널(변형 · 상호작용 · 용매) + 장벽 모델 사전 등록 ({now[:10]})", "",
          f"`docs/specs/CH3_MODEL.md` 7절의 사전 등록이다(명령서 sha256 `{C3.sha256(SPEC)}`). "
          f"`analysis/ch3/prereg_ch3.py`({VERSION})가 결과 파일만 읽어 만들었다(git HEAD `{head}`, 작성 {now}). "
          "규칙은 명령서를 그대로 인용한다(인용 블록 = 원문). **K 학습(`train_ch3.py run`)과 W(`train_ch3.py final`)는 "
          "이 md와 `prereg_ch3.json`이 commit되고 바뀌지 않았을 때만 돈다**(`prereg_ch3.py check`).", "",
          "## 1. 설계 (0-2절)", quote(q["design"]), "",
          "## 2. 학습 (8절)과 최종 모델 (10절)", quote(q["train"]), "", quote(q["final"]), "",
          "## 3. 평가 지표 정의 전부와 채택 규칙 (9절, 9-2 포함)", quote(q["eval"]), "",
          f"- 채택 기준값: strain · inter · solv 상대 차 점추정 ≤ 0 그리고 95 % 상한 ≤ +{REL_CI_HI_MAX * 100:.0f} %, "
          f"barrier 95 % 상한 ≤ +{REL_CI_HI_MAX * 100:.0f} % (P_main · para 공통 짝, core 부트스트랩 2,000회, "
          "seed 20261019, `eval_geo2.boot_diff`와 같은 재표본으로 상대 차 CI). 못 넘으면 STOP 없이 나란히 보고.", "",
          "### 3-1. 구현 정의 (`eval_ch3.py` DEFINITIONS, ch3 결과를 보기 전에 고정)", ""]
    md += [f"- **{k}**: {v}" for k, v in EV.DEFINITIONS.items()]
    md += ["", "### 3-2. selftest (lf 합으로 9-3 참고값 재현, `results/E3_selftest.json`)", "",
           "| 지표 | 참고값 | 재현 | 소수 둘째 자리 일치 |", "|---|---|---|---|"]
    md += [f"| {r['metric']} | {r['ref']} | {r['value']} | {'예' if r['match'] else '아니오'} |" for r in ref_rows]
    if st.get("notes"):
        md += ["", "정의 차이 메모: " + " ".join(st["notes"])]
    md += ["", "## 4. 지금까지 본 것",
           f"- ch3로 학습한 교차검증 결과: **0개** (`$CH3/units/D__lf3__*`, `results/K3_*`, `$CH3/model/D__lf3__*` 없음; {now})",
           "- 본 것: lf 합(비교 기준, deploy `K_predictions.parquet`)의 9-3 지표(selftest), C-3′(10타깃 deploy 재현, "
           f"fold 0; 결과 `C3r_regress.json` ok {c3r.get('ok')}), C-8 smoke(ch3 모드 fold 0, 2 epoch; 성능 판단에 쓰지 않음, "
           f"ok {sm.get('ok')})",
           f"- 입력 재생성 확인 V: `results/V_inputs.json` all_pass {v.get('all_pass')}", "",
           "## 5. 비교 기준 파일과 입력 sha256", "", "| 파일 | sha256 |", "|---|---|"]
    md += [f"| `{p}` | `{h}` |" for p, h in ins.items()]
    md += ["", f"- C 검사: `results/C_tests.json` sha256 `{C3.sha256(C3.RES / 'C_tests.json')}` (all_pass)",
           f"- eval selftest: `results/E3_selftest.json` sha256 `{C3.sha256(C3.RES / 'E3_selftest.json')}`", "",
           "## 6. 코드 sha256 (작성 시)", "", "| 파일 | sha256 |", "|---|---|"]
    code = {f"analysis/ch3/{p.name}": C3.sha256(p) for p in sorted(HERE.glob("*.py"))}
    md += [f"| `{p}` | `{h}` |" for p, h in code.items()]
    text = "\n".join(md) + "\n"
    C3.write_text(MD, text)
    rec = dict(version=VERSION, written=now, git_head=head, spec=str(SPEC.relative_to(C3.REPO)),
               spec_sha256=C3.sha256(SPEC), md=str(MD.relative_to(C3.REPO)), md_sha256=C3.sha256(MD),
               inputs_sha256=ins, comparator=str(C3.DEPLOY_RES / "K_predictions.parquet"),
               adoption=dict(rel_ci_hi_max=REL_CI_HI_MAX, channels=["strain", "inter", "solv"], barrier="barrier",
                             rule="channels: rel point <= 0 and 95 % upper <= +5 %; barrier: 95 % upper <= +5 %",
                             pairs="common P_main para pairs", n_boot=C3.N_BOOT, seed=C3.SEED),
               definitions=EV.DEFINITIONS, seen=dict(ch3_results=s, e3_selftest_sha256=C3.sha256(C3.RES / "E3_selftest.json")),
               c_tests_sha256=C3.sha256(C3.RES / "C_tests.json"), code_sha256=code)
    C3.write_json(JSON, rec)
    print(f"[prereg] wrote {MD.name} ({rec['md_sha256'][:12]}) and {JSON.name}; commit both before K", flush=True)
    return 0


def check():
    """spec 7 gate: exit 3 unless both files are tracked, unchanged since HEAD, md sha256 = json md_sha256 and every
    listed input unchanged. Returns the json record + git head."""
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
        rec = C3.read_json(JSON)
        if C3.sha256(MD) != rec.get("md_sha256"):
            bad.append(f"{MD.name} sha256 differs from {JSON.name} md_sha256")
        for path, s in (rec.get("inputs_sha256") or {}).items():
            if not Path(path).is_file() or C3.sha256(path) != s:
                bad.append(f"input {path} changed or missing")
    if bad:
        print("PREREG_CH3 gate (spec 7: commit PREREG_CH3.md + prereg_ch3.json before K):\n  " + "\n  ".join(bad),
              file=sys.stderr, flush=True)
        raise C3.Stop(3)
    return dict(rec, md_sha256=C3.sha256(MD), git_head=git("rev-parse", "HEAD").stdout.strip())


def rel_threshold():
    if JSON.is_file():
        v = (C3.read_json(JSON).get("adoption") or {}).get("rel_ci_hi_max")
        if v is not None:
            return float(v)
    return REL_CI_HI_MAX


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
