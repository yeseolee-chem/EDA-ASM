#!/usr/bin/env python3
r"""prereg_deploy.py — DEPLOY-MODEL Phase R: pre-registration before any lf training (docs/specs/DEPLOY_MODEL.md 10).
Any env with python >= 3.9 + git (jobs/cpu_py.slurm DEPENV=reactot). Exit 0 ok, 2 not ready, 3 STOP.

  python prereg_deploy.py write    -> results/PREREG_DEPLOY.md + results/prereg_deploy.json (written once: refuses when
                                      the md is already tracked by git). Needs results/C_tests.json all_pass (C before
                                      R), the G-A record (results/Q_parity.json 'G_A', see ga_record) passing, and no
                                      lf training result (no $DEP/units/D__lf__*, no results/K_*). The md quotes the
                                      spec rules verbatim (0-2, 8, 11-2, 12, 13, G-A of 7, G-C of 10), lists what has
                                      been seen (LADDER g2 . S . TF table recomputed by eval_tf.py selftest =
                                      results/E_selftest.json, GEO2b SUMMARY_GEO2B.md / SUMMARY_DEPLOY.md section 2
                                      tables, the Q parity summary) and the sha256 of every input.
  python prereg_deploy.py check    the gate of train_tf.py run / final: both files tracked by git and unchanged since
                                      HEAD, sha256(md) = json md_sha256, every json inputs_sha256 file unchanged.
After `write`: the user reviews (G-C threshold may be changed in the md + json before the commit), then commits both.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402

MD = C.RES / "PREREG_DEPLOY.md"
JSON = C.RES / "prereg_deploy.json"
SPEC = C.REPO / "docs" / "specs" / "DEPLOY_MODEL.md"
GEO2_RES = C.REPO / "analysis" / "geo2" / "results"
SUMMARIES = (GEO2_RES / "SUMMARY_GEO2B.md", GEO2_RES / "SUMMARY_DEPLOY.md")
LAD_RES = C.REPO / "analysis" / "d1_ladder" / "results"
D1PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
GC_REL_MAX = 0.05
GA_MAX = 0.02
VERSION = "deploy-prereg-1"


def git(*a):
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)


def _rel(p):
    """repo-relative path string when inside the repository, else the absolute path."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


def tracked(p):
    return git("ls-files", "--error-unmatch", str(p)).returncode == 0


def unchanged(p):
    return git("diff", "--quiet", "HEAD", "--", str(p)).returncode == 0


def inputs():
    """the files whose sha256 the pre-registration fixes (absolute paths)."""
    fu2 = C.RES / "fu2_lf"
    files = [C.LABELS_LF, C.META_LF, C.X182_LF, fu2 / "T_manifest.csv", fu2 / "T_report.json", fu2 / "Q_report.json",
             fu2 / "Q_manifest.csv", C.LAD / "F_pairs_cr2.csv",
             C.REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json",
             C.REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv",
             LAD_RES / "L_predictions.parquet", LAD_RES / "V_report.json"]
    files += [D1PAIR / "fu2" / "splits" / f"T1__f{k}.json" for k in range(5)]
    return files


# ---------------------------------------------------------------- spec quotes
def spec_section(text, start, end):
    """the spec lines from the heading that starts with `start` up to (excluding) the heading `end`."""
    lines = text.splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith(start)), None)
    if i is None:
        C.stop(f"{SPEC.name}: heading '{start}' not found")
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith(end)), len(lines))
    return "\n".join(lines[i:j]).rstrip()


def quote(s):
    return "\n".join("> " + ln if ln else ">" for ln in s.splitlines())


def first_tables(path, heading_prefix="## 2."):
    """the markdown table lines of the section starting with heading_prefix (verbatim)."""
    if not path.is_file():
        return None
    lines = path.read_text().splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith(heading_prefix)), None)
    if i is None:
        return None
    out = [lines[i]]
    for ln in lines[i + 1:]:
        if ln.startswith("## "):
            break
        if ln.startswith("|") or ln.startswith("- "):
            out.append(ln)
    return "\n".join(out)


# ---------------------------------------------------------------- G-A
def ga_record():
    """G-A (spec 7): redetection failures <= 2 % of the Phase L .done reactions, from results/Q_parity.json (parity.py:
    'redetection' {n, n_L_done, rate} and 'G_A' {threshold, rate, pass_}); recomputed here and required to agree;
    Q_parity stop = STOP; missing = not ready."""
    p = C.RES / "Q_parity.json"
    if not p.is_file():
        C.not_ready(f"{p} missing (parity.py, Phase Q)")
    r = C.read_json(p)
    if r.get("stop"):
        C.stop(f"{p.name} stop: {r.get('stop_reasons')}")
    g, red = r.get("G_A") or {}, r.get("redetection") or {}
    nf, nd = red.get("n"), red.get("n_L_done")
    rate = g.get("rate")
    if rate is None and nf is not None and nd:
        rate = float(nf) / float(nd)
    if rate is None:
        C.not_ready(f"{p.name}: no G-A rate (G_A.rate or redetection.n / n_L_done)")
    if nf is not None and nd and abs(float(nf) / float(nd) - float(rate)) > 1e-12:
        C.stop(f"{p.name}: G_A.rate {rate} != redetection n / n_L_done {nf} / {nd}")
    mine = bool(float(rate) <= GA_MAX)
    theirs = g.get("pass_", g.get("pass"))
    if theirs is not None and bool(theirs) != mine:
        C.stop(f"{p.name}: G_A pass {theirs} disagrees with rate {rate} <= {GA_MAX}")
    return dict(source=str(p), sha256=C.sha256(p), n_redetect_fail=nf, n_done=nd, rate=float(rate), threshold=GA_MAX,
                pass_=mine, record=g)


def seen_lf():
    """lf training results that exist now (must be none at write time)."""
    units = sorted(p.name for p in C.UNITS.glob("D__lf__*")) if C.UNITS.is_dir() else []
    res = sorted(p.name for p in C.RES.glob("K_*"))
    model = sorted(p.name for p in C.MODEL_DIR.glob("D__lf__*")) if C.MODEL_DIR.is_dir() else []
    return dict(units=units, results=res, model=model)


# ---------------------------------------------------------------- write
def cmd_write():
    if MD.is_file() and tracked(MD):
        C.stop(f"{MD.name} is already tracked by git: the pre-registration is written once (spec 10)")
    ct = C.RES / "C_tests.json"
    c = C.read_json(ct) if ct.is_file() else {}
    if not c.get("all_pass"):
        C.not_ready(f"{ct}: all_pass {c.get('all_pass')} (Phase C before R)")
    ga = ga_record()
    if not ga["pass_"]:
        C.stop(f"G-A failed: {ga['n_redetect_fail']} / {ga['n_done']} = {ga['rate']:.4f} > {GA_MAX} (spec 7)")
    s = seen_lf()
    if any(s.values()):
        C.stop(f"lf training results exist before the pre-registration: {s}")
    est = C.RES / "E_selftest.json"
    e = C.read_json(est) if est.is_file() else {}
    if not e.get("all_pass"):
        C.not_ready(f"{est}: eval_tf.py selftest all_pass {e.get('all_pass')} (the evaluation code must be validated "
                    "and the LADDER g2 numbers recomputed before the pre-registration)")
    if (e.get("code_sha256") or {}).get("analysis/deploy/eval_tf.py") != C.sha256(HERE / "eval_tf.py"):
        C.not_ready(f"{est} was made by another eval_tf.py than the current one: rerun eval_tf.py selftest")
    miss = [str(p) for p in inputs() if not p.is_file()]
    if miss:
        C.not_ready(f"inputs missing: {miss}")
    ins = {str(p): C.sha256(p) for p in inputs()}
    spec = SPEC.read_text()
    q = dict(design=spec_section(spec, "### 0-2.", "### 0-3."), usable=spec_section(spec, "## 8.", "## 9."),
             train=spec_section(spec, "### 11-2.", "### 11-3."), eval=spec_section(spec, "## 12.", "## 13."),
             final=spec_section(spec, "## 13.", "## 14."), gate_a=spec_section(spec, "**G-A", "## 8."),
             gate_c=spec_section(spec, "**G-C", "## 11."))
    qp = C.RES / "Q_parity.json"
    qsum = {k: v for k, v in C.read_json(qp).items() if not isinstance(v, (list,)) or len(v) <= 20} if qp.is_file() else None
    code = {f"analysis/deploy/{p.name}": C.sha256(p) for p in sorted(HERE.glob("*.py"))}
    head = git("rev-parse", "HEAD").stdout.strip()
    det = e.get("detail") or {}
    pm, au = det.get("para_mae") or {}, det.get("auc_g") or {}
    order = ["oi", "cpcm", "pauli", "elst", "d1", "d2", "disp", "cds", "barrier"]
    row_m = " | ".join(f"{pm[t]['value']:.3f}" if t in pm else "—" for t in order)
    row_a = " | ".join(f"{au[t]['value']:.3f}" if t in au and au[t]["value"] is not None else "—" for t in order)
    hd = det.get("hidden") or {}
    summ = {str(p.relative_to(C.REPO)): dict(sha256=C.sha256(p), tables=first_tables(p)) for p in SUMMARIES
            if p.is_file()}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    md = [f"# PREREG DEPLOY — 라벨 없는 g2 입력(lf) · S · TF 배포 모델 사전 등록 ({now[:10]})", "",
          f"이 문서는 DEPLOY-MODEL의 사전 등록이다(`docs/specs/DEPLOY_MODEL.md` 10절, sha256 `{C.sha256(SPEC)}`). "
          f"`analysis/deploy/prereg_deploy.py`({VERSION})가 결과 파일만 읽어 만들었다(git HEAD `{head}`, 작성 {now}). "
          "규칙은 명령서를 그대로 인용한다(인용 블록 = 원문). **K 학습(`train_tf.py run`)과 W(`train_tf.py final`)는 "
          "이 md와 `prereg_deploy.json`이 commit되고 바뀌지 않았을 때만 돈다**(`prereg_deploy.py check`).", "",
          "## 1. 설계 (0-2절)와 사용 가능 규칙 (8절)", quote(q["design"]), "", quote(q["usable"]), "",
          "## 2. 학습 규칙 (11-2절), 최종 모델 (13절), 평가 (12절)", quote(q["train"]), "", quote(q["final"]), "",
          quote(q["eval"]), "",
          "구현 메모(코드, 결과를 보기 전): 꺾임 계열 = 시험 예측의 reference 짝(σ 있음, P_sens 전체) 위 "
          "`hammett_eval.family_series`(점 ≥ 12, H 포함) — 0-4절 89계열과 같은 규칙; para MAE = P_main reference "
          "para 짝(0-4절 1,658짝). 상대 차이 CI = `eval_geo2.boot_diff`와 같은 core 재표본에서 (MAE_lf − MAE_g2)/MAE_g2. "
          "E_ad 문턱 = lf 시험 짝 `yhat_seed_sd`의 타깃별 95 백분위(numpy linear).", "",
          "## 3. 관문", "### G-A (7절)", quote(q["gate_a"]), "",
          f"- 판정: 재검출 실패 {ga['n_redetect_fail']} / L `.done` {ga['n_done']} = {ga['rate'] * 100:.2f} % "
          f"(기준 ≤ {GA_MAX * 100:.0f} %) → **{'통과' if ga['pass_'] else '실패'}** (`{Path(ga['source']).name}`)", "",
          "### G-C (10절)", quote(q["gate_c"]), "",
          f"- 기준값: 9개 주요 타깃마다 상대 차이 95 % 상한 ≤ **+{GC_REL_MAX * 100:.0f} %** "
          "(사용자가 commit 전에 바꿀 수 있다: md와 json `gates.G_C.rel_ci_hi_max`를 함께 바꾼다)", "",
          "## 4. 지금까지 본 것", f"- lf로 학습한 결과: **0개** (`$DEP/units/D__lf__*`, `results/K_*`, `$DEP/model` 없음; {now})",
          f"- LADDER g2 · S · TF (D1 T1 시험; `eval_tf.py selftest`가 `L_predictions.parquet`에서 다시 계산, "
          f"`E_selftest.json` sha256 `{C.sha256(est)}`):", "",
          "| 타깃 | " + " | ".join(order) + " |", "|---|" + "---|" * len(order),
          f"| para MAE (kcal/mol, {(pm.get('oi') or {}).get('n', '?')}짝) | {row_m} |",
          f"| 연속 g AUC (para {det.get('n_para_families', '?')}계열) | {row_a} |", "",
          f"- 숨은 꺾임 계열(같은 89계열): 민감도 {hd.get('sens', {}).get('value', float('nan')):.3f}, 위양성률 "
          f"{hd.get('fpr', {}).get('value', float('nan')):.3f}, 정밀도 {hd.get('ppv', {}).get('value', float('nan')):.3f}, "
          f"기저율 {hd.get('base', {}).get('value', float('nan')):.3f}", ""]
    for rel, v in summ.items():
        md += [f"- GEO2b `{rel}` (sha256 `{v['sha256']}`), 2절 그대로:", "", quote(v["tables"] or "(2절 표 없음)"), ""]
    md += ["- Q 재현 보고 (`results/Q_parity.json`" + (f", sha256 `{C.sha256(qp)}`" if qp.is_file() else ", 없음") + "):",
           "", "```json", json.dumps(qsum, indent=1, ensure_ascii=False, default=str)[:20000] if qsum else "null", "```",
           "", "## 5. 입력 sha256", "", "| 파일 | sha256 |", "|---|---|"]
    md += [f"| `{p}` | `{s}` |" for p, s in ins.items()]
    md += ["", f"- C 검사: `results/C_tests.json` sha256 `{C.sha256(ct)}` (all_pass)", "",
           "## 6. 코드 sha256 (작성 시)", "", "| 파일 | sha256 |", "|---|---|"]
    md += [f"| `{p}` | `{s}` |" for p, s in code.items()]
    text = "\n".join(md) + "\n"
    C.write_text(MD, text)
    ga_rec = {("pass" if k == "pass_" else k): v for k, v in ga.items()}
    rec = dict(version=VERSION, written=now, git_head=head, spec=str(SPEC.relative_to(C.REPO)), spec_sha256=C.sha256(SPEC),
               md=_rel(MD), md_sha256=C.sha256(MD), inputs_sha256=ins,
               gates=dict(G_A=ga_rec,
                          G_C=dict(rel_ci_hi_max=GC_REL_MAX, targets=list(C.PRIMARY), n_boot=C.N_BOOT, seed=C.SEED,
                                   pairs="common para P_main pairs (lf and LADDER g2 . S . TF both predicted)",
                                   comparator="analysis/d1_ladder/results/L_predictions.parquet rung g2 arm S model TF")),
               seen=dict(lf_results=seen_lf(), e_selftest=dict(path=str(est), sha256=C.sha256(est)),
                         summaries={k: v["sha256"] for k, v in summ.items()},
                         q_parity=dict(path=str(qp), sha256=C.sha256(qp)) if qp.is_file() else None),
               c_tests_sha256=C.sha256(ct), code_sha256=code)
    C.write_json(JSON, rec)
    print(f"[prereg] wrote {MD.name} ({rec['md_sha256'][:12]}) and {JSON.name}; review, then commit both", flush=True)
    return 0


# ---------------------------------------------------------------- check (gate)
def check():
    """spec 10 gate: exit 3 (STOP) unless both files are tracked, unchanged since HEAD, md sha256 = json md_sha256 and
    every listed input unchanged. Returns the json record + git head."""
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
        g = (rec.get("gates") or {})
        if not (g.get("G_A") or {}).get("pass"):
            bad.append("G-A not passed in prereg_deploy.json")
        if (g.get("G_C") or {}).get("rel_ci_hi_max") is None:
            bad.append("G-C threshold missing in prereg_deploy.json")
    if bad:
        print("PREREG_DEPLOY gate (spec 10: commit PREREG_DEPLOY.md + prereg_deploy.json before K):\n  "
              + "\n  ".join(bad), file=sys.stderr, flush=True)
        raise C.Stop(3)
    return dict(rec, md_sha256=C.sha256(MD), git_head=git("rev-parse", "HEAD").stdout.strip())


def gc_threshold():
    """the committed G-C threshold (eval_tf uses it when the pre-registration exists)."""
    if JSON.is_file():
        v = ((C.read_json(JSON).get("gates") or {}).get("G_C") or {}).get("rel_ci_hi_max")
        if v is not None:
            return float(v)
    return GC_REL_MAX


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
