#!/usr/bin/env python3
r"""prereg_w.py — DEPLOY-MODEL-W start conditions (spec 1) and Phase R pre-registration (spec 7).
Any env with python >= 3.9 + git (+ numpy / pandas for `start`): jobs/cpu_py.slurm DEPENV=reactot or d1fu2.
Exit 0 ok, 2 not ready, 3 STOP.

  python prereg_w.py start    spec 1 (0-1 .. 0-5) -> results/w/S0_start_w.json; 0-5 reads DEPW_FREE_GB (the headroom
                              read with mmlsquota on the login node; compute nodes cannot read the quota)
  python prereg_w.py write    -> results/w/PREREG_W.md + results/w/prereg_w.json (written once: refuses when the md is
                              tracked by git). Needs the start conditions (recomputed), results/w/C_tests_w.json
                              all_pass, the Phase B outputs and no lfV / lfW result. The md quotes the spec (0, 3-5, 7)
                              verbatim, lists what has been seen (spec 0-1 table, DEPLOY_MODEL lf K / E), the
                              implementation decisions and the sha256 of every input.
  python prereg_w.py check    gate of train_w.py run / final: both files tracked and unchanged since HEAD, sha256(md) =
                              json md_sha256, every json input unchanged.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w_common as W  # noqa: E402

C = W.C
import prereg_deploy as PR  # noqa: E402     (git helpers, spec quoting)

MD = W.RESW / "PREREG_W.md"
JSON = W.RESW / "prereg_w.json"
VERSION = "deploy-w-prereg-1"
DECISIONS = [
    "B: group key (b_rxn, fam, pos) from S_folds.csv over every pair (ref_kind reference and anchor); S_folds itself has "
    "610 groups (579 with >= 2 pairs), median size 17, 67,636 within pairs - not the spec 0-2 estimate (353 / 19 / "
    "79,600), which no key on S_folds reproduces; recorded in results/w/B_report.json, no STOP rule applies.",
    "W-2 sampling: 'w = 2/(n_g-1), without replacement' is realised as inclusion probability pi_i = min(1, c w_i) with "
    "sum pi = n_w (systematic sampling on a random order): every uncapped group gets exactly c n_g expected pairs per "
    "epoch; groups whose pool is exhausted are drawn whole ('all when the pool is small'). numpy's weighted "
    "choice(replace=False) misses the n_g-proportionality by 19-26 % at these sampling fractions (C-W6).",
    "C-W1: labels_d1.json fails the within-group identity for d1 / d2 / barrier where the LADDER cr2 correction changed "
    "an a-side reaction; with labels_d1_cr2.json (V_report all_pass) every group passes (<= 1e-6) - the spec fallback.",
    "C-W3: fu2 fit computes in float32 (train.py 155, 226-227), so 'CPU float64' is not possible without editing it; "
    "both fits run on CPU in float32 with 1 thread (multi-threaded CPU reductions are not bitwise reproducible) and must "
    "agree bit for bit (history, best epoch, validation, weights).",
    "V_nl / H points: pairs with a finite sigma only; H point (0, 0) appended; RMS over n + 1 points; sigma = S_folds "
    "sigma (sigma_p for p, sigma_m for m). V_nl groups: inner_val pairs, pos p / m, >= 6 such pairs.",
    "H: s_t = SD (ddof 0) of the DFT y over every evaluation point (para families, H points excluded); core bootstrap = "
    "eval_geo2.boot_diff resampling (default_rng(20261019), 2,000), family means weighted by core counts.",
    "grade: A-1 fails -> 'no improvement' (also when the point R <= 0.70); else R <= 0.70 -> 'goal reached', else "
    "'partial improvement'. Secondary graph readout for the adopted model only (lf and lfW both reported).",
    "W2 epoch = lower median of the best epochs of the folds whose lfW r is the mode (spec 12); C-W8 for K = 200 random "
    "test pairs + 200 random within pairs of the fit (r > 0); W2 = 200 training + 200 within pairs.",
    "lfV = lfW selection (same cell, r = 0) -> one set of final fits; the lfV .pt files are hard links of the lfW ones.",
    "E: the Phase P (a, ref) predictions must equal the K predictions (<= 1e-3 kcal/mol, else STOP: wrong model files).",
]


def inputs():
    """files fixed by the pre-registration (absolute paths): the DEPLOY_MODEL pre-registered inputs, S_folds, the Phase B
    outputs, the lf K predictions / E report and the lf units (used by P and C-W4)."""
    pj = C.RES / "prereg_deploy.json"
    files = [Path(p) for p in ((W.read_json(pj).get("inputs_sha256") or {}) if pj.is_file() else {})]
    files += [pj, C.RES / "PREREG_DEPLOY.md", W.S_FOLDS, C.RES / "K_predictions.parquet", C.RES / "E_report.json",
              C.RES / "K_report.json"]
    if W.WITHIN.is_dir():
        files += sorted(W.WITHIN.glob("*.npz")) + sorted(W.WITHIN.glob("test_groups__f*.parquet")) + [W.WITHIN / "meta.json"]
    for k in range(W.NF):
        nm = f"D__lf__S__TF__f{k}"
        files += [C.UNITS / f"{nm}.json", C.UNITS / f"{nm}.parquet"] + [C.UNITS / f"{nm}__s{j}.pt" for j in range(5)]
    out, seen = [], set()
    for p in files:
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def seen_w():
    units = sorted(p.name for p in W.UNITS_W.glob("D__lf[WV]__*")) if W.UNITS_W.is_dir() else []
    res = sorted(p.name for p in W.RESW.glob("K_*_w*")) + sorted(p.name for p in W.RESW.glob("E_*_w*")) + \
        sorted(p.name for p in W.RESW.glob("E_H.csv")) + sorted(p.name for p in W.RESW.glob("E_ratio.csv"))
    pred = sorted(p.name for p in W.PRED_W.glob("*.parquet")) if W.PRED_W.is_dir() else []
    model = sorted(p.name for p in W.MODEL_W.glob("D__lfW__*")) if W.MODEL_W.is_dir() else []
    return dict(units=units, results=res, within_pred=pred, model=model)


def cmd_start():
    fg = os.environ.get("DEPW_FREE_GB")
    rec = W.check_start(float(fg) if fg else None)
    rec.update(time=time.strftime("%Y-%m-%dT%H:%M:%S"), host=os.uname().nodename)
    W.write_json(W.RESW / "S0_start_w.json", rec)
    print(json.dumps(W._jsonable(dict(all_pass=rec["all_pass"], checks=rec["checks"])), indent=1))
    if fg is None:
        print("NOT READY: 0-5 needs DEPW_FREE_GB (login-node mmlsquota headroom in GB)", file=sys.stderr)
        return 2
    return 0 if rec["all_pass"] else 3


def _lf_seen():
    """DEPLOY_MODEL lf results already seen (K / E) for the md."""
    out = {}
    kr, er = C.RES / "K_report.json", C.RES / "E_report.json"
    if kr.is_file():
        k = W.read_json(kr)
        out["K"] = {u: dict(chosen=v.get("chosen"), n_test=v.get("n_test")) for u, v in (k.get("units") or {}).items()}
        out["K_sha256"] = W.sha256(kr)
    if er.is_file():
        e = W.read_json(er)
        s = e.get("summary") or {}
        out["E"] = dict(G_C=(e.get("G_C") or {}).get("pass"), para_mae_own=(s.get("para_mae_own") or {}).get("lf"),
                        auc_g_own=(s.get("auc_g_own") or {}).get("lf"), hidden_own=(s.get("hidden_own") or {}).get("lf"),
                        n_para_families=(s.get("n_para_families_own") or {}).get("lf"))
        out["E_sha256"] = W.sha256(er)
    return out


def cmd_write():
    if MD.is_file() and PR.tracked(MD):
        W.stop(f"{MD.name} is already tracked by git: the pre-registration is written once (spec 7)")
    fg = os.environ.get("DEPW_FREE_GB")
    st = W.check_start(float(fg) if fg else None)
    if not st["all_pass"]:
        W.stop(f"start conditions (spec 1) not met: {st['checks']} ({json.dumps(W._jsonable(st['detail']))[:600]})")
    ct = W.RESW / "C_tests_w.json"
    c = W.read_json(ct) if ct.is_file() else {}
    if not c.get("all_pass"):
        W.not_ready(f"{ct}: all_pass {c.get('all_pass')} (Phase C before R)")
    if not (W.WITHIN / "meta.json").is_file() or not (W.RESW / "B_report.json").is_file():
        W.not_ready("Phase B outputs missing (within.py build)")
    s = seen_w()
    if any(s.values()):
        W.stop(f"lfV / lfW results exist before the pre-registration: {s}")
    miss = [str(p) for p in inputs() if not p.is_file()]
    if miss:
        W.not_ready(f"inputs missing: {miss[:5]}")
    ins = {str(p): W.sha256(p) for p in inputs()}
    spec = W.SPEC_W.read_text()
    q = dict(design=PR.spec_section(spec, "## 0.", "## 1."), defs=PR.spec_section(spec, "## 3.", "## 6."),
             rules=PR.spec_section(spec, "## 7.", "## 8."))
    lf = _lf_seen()
    head = PR.git("rev-parse", "HEAD").stdout.strip()
    code = {**W.w_code_sha(), **{f"analysis/deploy/{p.name}": W.sha256(p) for p in sorted(W.DEPLOY_DIR.glob("*.py"))}}
    brep = W.read_json(W.RESW / "B_report.json")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    md = [f"# PREREG W — 계열 안 짝 학습(lfW)과 대조(lfV) 사전 등록 ({now[:10]})", "",
          f"이 문서는 DEPLOY-MODEL-W의 사전 등록이다(`docs/specs/DEPLOY_MODEL_W.md` 7절, sha256 `{W.sha256(W.SPEC_W)}`). "
          f"`analysis/deploy/w/prereg_w.py`({VERSION})가 결과 파일만 읽어 만들었다(git HEAD `{head}`, 작성 {now}). "
          "인용 블록 = 명령서 원문. **K(`train_w.py run`)와 W2(`train_w.py final`)는 이 md와 `prereg_w.json`이 commit되고 "
          "바뀌지 않았을 때만 돈다**(`prereg_w.py check`).", "",
          "## 1. 설계 (0절)", PR.quote(q["design"]), "", "## 2. 정의 (3–5절)", PR.quote(q["defs"]), "",
          "## 3. 사전 등록 규칙과 판정 (7절)", PR.quote(q["rules"]), "",
          "## 4. 구현 결정 (코드 작성 시, 결과를 보기 전; 사용자 지시: 모든 분기 = 추천안)", ""]
    md += [f"{i + 1}. {d}" for i, d in enumerate(DECISIONS)]
    md += ["", "## 5. 지금까지 본 것",
           "- 0-1절 표(LADDER g2 · S · TF 분석; 위 1절 인용 그대로)",
           f"- DEPLOY_MODEL lf 결과 (K · E): ```{json.dumps(W._jsonable(lf), ensure_ascii=False)[:6000]}```",
           f"- lfV · lfW 결과: **0개** ({json.dumps(seen_w())}; {now})",
           f"- Phase B 규모: {json.dumps(W._jsonable({k: v for k, v in brep.get('folds', {}).items()}), ensure_ascii=False)[:4000]}",
           f"- 시작 조건(1절): {json.dumps(st['checks'])}", "",
           "## 6. 입력 sha256", "", "| 파일 | sha256 |", "|---|---|"]
    md += [f"| `{p}` | `{x}` |" for p, x in ins.items()]
    md += ["", f"- C 검사: `results/w/C_tests_w.json` sha256 `{W.sha256(ct)}` (all_pass)", "",
           "## 7. 코드 sha256 (작성 시)", "", "| 파일 | sha256 |", "|---|---|"]
    md += [f"| `{p}` | `{x}` |" for p, x in code.items() if x]
    W.write_text(MD, "\n".join(md) + "\n")
    rec = dict(version=VERSION, written=now, git_head=head, spec=str(W.SPEC_W.relative_to(C.REPO)),
               spec_sha256=W.sha256(W.SPEC_W), md=PR._rel(MD), md_sha256=W.sha256(MD), inputs_sha256=ins,
               rules=dict(A1="95 % upper bound of R = H_lfW / H_lf < 1.00",
                          A2="each primary target: 95 % upper bound of (MAE_lfW - MAE_lf) / MAE_lf <= +5 % (para P_main)",
                          rel_max=0.05, goal_R=0.70, mae_guard=W.MAE_GUARD, r_grid=list(W.R_GRID),
                          min_group_val=W.MIN_GROUP_VAL, min_fam_eval=W.MIN_FAM_EVAL, primary=list(W.PRIMARY),
                          n_boot=W.N_BOOT, seed=W.SEED,
                          secondary="R_g = H_graph / H_direct upper bound < 1.00 and the A-2 non-inferiority graph vs direct"),
               decisions=DECISIONS, start=st, seen=dict(lf=lf, w=seen_w()), c_tests_w_sha256=W.sha256(ct), code_sha256=code)
    W.write_json(JSON, rec)
    print(f"[prereg_w] wrote {MD.name} ({rec['md_sha256'][:12]}) and {JSON.name}; review, then commit both", flush=True)
    return 0


def check():
    bad = []
    for p in (MD, JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif not PR.tracked(p):
            bad.append(f"{p.name} not committed")
        elif not PR.unchanged(p):
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = W.read_json(JSON)
        if W.sha256(MD) != rec.get("md_sha256"):
            bad.append(f"{MD.name} sha256 differs from {JSON.name} md_sha256")
        for path, s in (rec.get("inputs_sha256") or {}).items():
            if not Path(path).is_file() or W.sha256(path) != s:
                bad.append(f"input {path} changed or missing")
    if bad:
        print("PREREG_W gate (spec 7: commit PREREG_W.md + prereg_w.json before K):\n  " + "\n  ".join(bad),
              file=sys.stderr, flush=True)
        raise C.Stop(3)
    return dict(rec, md_sha256=W.sha256(MD), git_head=PR.git("rev-parse", "HEAD").stdout.strip())


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["start"]:
        sys.exit(cmd_start())
    if a == ["write"]:
        sys.exit(cmd_write())
    if a == ["check"]:
        r = check()
        print(json.dumps(dict(ok=True, md_sha256=r["md_sha256"], git_head=r["git_head"]), indent=1))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
