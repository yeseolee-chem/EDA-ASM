#!/usr/bin/env python3
"""D1-G2 Phase S (docs/specs/D1G2.md 5): smoke of the 10 reactions of results/S_list.csv (prep_g.py slist). sbatch only.

  (G2)  jobs/g2.slurm (G2ENV=d1)  g2_autode.py pool results/S_list.csv 10 4      -> $G2/g2_final/<rxn_id>
  python s_smoke.py f       # d1_pair jobs/g.slurm (PYENV=venv5, D1PAIR_PHASE=F): the T-1 / T-2 paths on the S ok rxns
                            #   (runroot $G2/g2_final with the G2 inputs -> $G2/S/X182_S.parquet; atoms.py root run /
                            #   gate / tokens with fu2-atoms-3-g2 / fu2-tokens-3-g2 -> $G2/S/fu2) -> results/S_F.json
  python s_smoke.py h       # d1_hf jobs/hf.slurm: the T-3 path (D1HF_SCRATCH = $G2/S/hf, templates = d1_hf/inputs,
                            #   D1HF_G2_ROOT = $G2/g2_final): geominputs g2, run.py run g2 hf, build.py allgeom / check
                            #   g2 -> results/S_H.json
  python s_smoke.py report  # jobs/g2.slurm: S-1 .. S-5 -> results/S_report.json, results/S_REPORT.md (exit 3 = STOP)

S-1 every ORCA input autodE / the wrapper wrote ($G2/jobs/<T>/inputs.tar.gz): the '!' lines contain XTB2 and
    ALPB(water), the whole file has no B3LYP / def2 / CPCM / SMD (case-insensitive); one input pasted in the report.
S-2 every reaction that reached the reference stage: ref_result n_orca_outputs = 0 and >= 1 xtb --opt and >= 1 xtb
    --ohess call (result.json); at least one reaction reached it.
S-3 all 10 in the g2_final format (.done | .fail_<reason>; .done: ts.xyz, ts.hess, rel1.xyz, rel2.xyz, result.json,
    work/ts.hess); every ok reaction has its 182 features (F_ok), its token npz and HF status ok.
S-4 ok >= 7 / 10 and 0 pipeline errors (no g2_final folder, .fail_error_*, an infra failure after the retry, or a
    feature / token / HF step that did not finish).
S-5 G2 core-hours per reaction (result.json core_h_total, all 10) fitted as log(core_h) = a + b log(n_atoms); the
    expected total = sum over the 9,266 targets + HF (S_H core-h fit the same way) <= 50,000 core-h.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_common as G  # noqa: E402
import test_t as TT  # noqa: E402                      (run / step / Fail / env_f / env_hf / clean: the T paths)

DP = G.DP
S_DIR = G.G2 / "S"
S_LIST = G.RES / "S_list.csv"
INPUTS_G2 = dict(labels=G.G2 / "G" / "labels_d1_g2.json", meta=G.G2 / "G" / "input_meta_d1_g2.csv", prof=G.G2 / "prof")
VERSIONS = dict(FU2_ATOM_VERSION="fu2-atoms-3-g2", FU2_TOKEN_VERSION="fu2-tokens-3-g2")      # spec 8
FINAL_FILES = ("ts.xyz", "ts.hess", "rel1.xyz", "rel2.xyz", "result.json", "work/ts.hess")
BAD_WORDS = re.compile(r"B3LYP|def2|CPCM|SMD", re.I)
N_OK_MIN, COREH_MAX = 7, 50000.0


def s_list():
    return pd.read_csv(S_LIST, dtype={"job_id": str})


def ok_rows(S):
    return S[[(G.G2_FINAL / str(r) / ".done").is_file() for r in S.rxn_id]]


def fresh(d):
    """$G2/S/<name>: an existing folder is moved to <name>.prev_<time> (never deleted)."""
    if d.exists():
        os.replace(G.assert_path(d), G.assert_path(d.with_name(f"{d.name}.prev_{time.strftime('%Y%m%d_%H%M%S')}")))
    G.assert_path(d).mkdir(parents=True)
    return d


def finish(out, rec, body):
    rec.update(started=TT.now(), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), git=TT.git_head())
    try:
        body(rec)
        rec["pass"] = True
    except TT.Fail as e:
        rec.update(fail_stage=e.args[0], fail=str(e.args[1])[:4000], **{"pass": False})
    except Exception as e:                                 # noqa: BLE001
        import traceback
        rec.update(fail_stage="error", fail=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:],
                   **{"pass": False})
    rec["finished"] = TT.now()
    G.write_json(out, TT.clean(rec))
    print(json.dumps(TT.clean({k: v for k, v in rec.items() if k != "traceback"}), indent=1, default=str), flush=True)
    return 0 if rec["pass"] else 3


# ================================================================ S-3: features + tokens (T-1 / T-2 paths)
def cmd_f():
    def body(rec):
        TT.env_f(rec)
        S = ok_rows(s_list())
        rec["n_ok_g2"] = len(S)
        if not len(S):
            raise TT.Fail("ok", "no S reaction with .done")
        base = fresh(S_DIR / "F")
        ids = G.write_atomic(base / "S_ok_rxn.csv", "rxn_id\n" + "".join(f"{r}\n" for r in S.rxn_id))
        x182, rep = base / "X182_S.parquet", base / "mergeroot_report.json"
        inp = [a for k in ("labels", "meta", "prof") for a in (f"--{k}", INPUTS_G2[k])]
        TT.step(rec, "runroot", [TT.P_DIR / "d1_features.py", "runroot", G.G2_FINAL, base / "out", ids] + inp, TT.P_DIR)
        TT.step(rec, "mergeroot", [TT.P_DIR / "d1_features.py", "mergeroot", base / "out", x182, rep], TT.P_DIR)
        m = json.loads(rep.read_text())
        rec["mergeroot"] = {k: m.get(k) for k in ("n_requested", "n_done", "n_feat", "F_fail_rate", "reasons", "not_ok",
                                                  "x182", "F1_all_finite", "env")}
        env = dict(os.environ, D1PAIR_PHASE="FROOT", G1_ROOT=str(G.G2_FINAL), ESPLEY_OUT=str(base / "fu2_xtb"),
                   R5_SCRATCH=str(base / "fu2_r5"), ESPLEY_LABELS=str(INPUTS_G2["labels"]),
                   ESPLEY_META=str(INPUTS_G2["meta"]), ESPLEY_PROF=str(INPUTS_G2["prof"]), FU2_SCR=str(base / "fu2"),
                   FU2_RES=str(base / "fu2_res"), FU2_X182=str(x182), PYTHONDONTWRITEBYTECODE="1", **VERSIONS)
        rec["child_env"] = {k: env[k] for k in ("G1_ROOT", "ESPLEY_LABELS", "ESPLEY_META", "ESPLEY_PROF", "FU2_SCR",
                                                "FU2_RES", "FU2_X182", *VERSIONS)}
        TT.step(rec, "atoms_run", [TT.P_DIR / "fu2" / "atoms.py", "root", "run", "0", "1"], TT.P_DIR, env)
        TT.step(rec, "atoms_gate", [TT.P_DIR / "fu2" / "atoms.py", "root", "gate"], TT.P_DIR, env, ok=(0, 3))
        TT.step(rec, "tokens_build", [TT.P_DIR / "fu2" / "atoms.py", "root", "tokens"], TT.P_DIR, env, ok=(0, 3))
        q = json.loads((base / "fu2_res" / "Q_report.json").read_text())
        t = json.loads((base / "fu2_res" / "T_report.json").read_text()) if (base / "fu2_res" / "T_report.json").is_file() \
            else {}
        rec["Q_report"] = {k: q.get(k) for k in ("n_target", "n_ok", "n_use", "excluded", "ext_used", "ext_pass_rate",
                                                 "qa_maxdiff_max", "qb_maxdiff_max", "atom_version", "stop", "stop_reason")}
        rec["T_report"] = {k: t.get(k) for k in ("token_version", "n_reactions", "num_cols", "ext_used", "gate_Ta", "stop")}
        X = pd.read_parquet(x182)
        tok = {str(r): (base / "fu2" / "tokens" / f"{r}.npz").is_file() for r in S.rxn_id}
        rec["per_rxn"] = {str(r): dict(features=bool(int(r) in set(X.rxn_id.astype(int))), tokens=tok[str(r)],
                                       why=m.get("not_ok", {}).get(str(r), "ok")) for r in S.rxn_id}
    return finish(G.RES / "S_F.json", dict(step="S-3 features + tokens", inputs={k: str(v) for k, v in INPUTS_G2.items()}),
                  body)


# ================================================================ S-3: HF (T-3 path)
def cmd_h():
    def body(rec):
        TT.env_hf(rec)
        S = ok_rows(s_list())
        rec["n_ok_g2"] = len(S)
        if not len(S):
            raise TT.Fail("ok", "no S reaction with .done")
        base = fresh(S_DIR / "hf")
        lst = G.write_atomic(base / "S_ok_job.csv", "job_id\n" + "".join(f"{T}\n" for T in S.job_id))
        env = dict(os.environ, D1HF_SCRATCH=str(base), D1HF_RES=str(base / "results"),
                   D1HF_TEMPLATES=str(TT.HF_SCR / "inputs"), D1HF_G2_ROOT=str(G.G2_FINAL), PYTHONDONTWRITEBYTECODE="1")
        rec["child_env"] = {k: env[k] for k in ("D1HF_SCRATCH", "D1HF_RES", "D1HF_TEMPLATES", "D1HF_G2_ROOT")}
        TT.step(rec, "geominputs", [TT.H_DIR / "prep.py", "geominputs", "g2", lst], TT.H_DIR, env)
        nw = max(1, min(len(S), int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
        TT.step(rec, "run", [TT.H_DIR / "run.py", "run", lst, "g2", "hf", nw], TT.H_DIR, env, ok=(0, 3))
        TT.step(rec, "build_all", [TT.H_DIR / "build.py", "allgeom", lst, "g2"], TT.H_DIR, env)
        TT.step(rec, "build_check", [TT.H_DIR / "build.py", "check", lst, "g2"], TT.H_DIR, env, ok=(0, 3))
        gi = pd.read_csv(base / "results" / "g2_inputs.csv", dtype=str, keep_default_na=False)
        hs = pd.read_csv(base / "results" / "H_status.csv", dtype=str, keep_default_na=False)
        lab = {r["job_id"]: r for r in json.loads((base / "labels_d1_hf_g2.json").read_text())}
        rec["per_rxn"] = {T: dict(inputs=dict(zip(gi.job_id, gi.status)).get(T), status=dict(zip(hs.job_id, hs.status)).get(T),
                                  reason=dict(zip(hs.job_id, hs.reason)).get(T),
                                  coreh_total=lab.get(T, {}).get("coreh_total"),
                                  values={k: lab[T].get(k) for k in TT.T3_KEYS} if T in lab else None)
                          for T in S.job_id}
    return finish(G.RES / "S_H.json", dict(step="S-3 HF"), body)


# ================================================================ report
def audit_inputs(T):
    """S-1 for one reaction: (n inputs, [problems], {name: text} of the OptTS inputs)."""
    f = G.G2_JOBS / T / "inputs.tar.gz"
    if not f.is_file():
        return 0, [f"{T}: no inputs.tar.gz"], {}
    n, bad, ex = 0, [], {}
    with tarfile.open(f, "r:gz") as t:
        for m in t:
            if not (m.isfile() and m.name.endswith(".inp")):
                continue
            n += 1
            txt = t.extractfile(m).read().decode(errors="replace")
            bang = " ".join(ln for ln in txt.splitlines() if ln.lstrip().startswith("!"))
            if "XTB2" not in bang or "ALPB(water)" not in bang:
                bad.append(f"{T}/{m.name}: '!' lines {bang!r}")
            w = sorted(set(x.upper() for x in BAD_WORDS.findall(txt)))
            if w:
                bad.append(f"{T}/{m.name}: {w}")
            if "OptTS" in bang:
                ex[m.name] = txt
    if not n:
        bad.append(f"{T}: inputs.tar.gz has no .inp")
    return n, bad, ex


def fit_power(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = (x > 0) & (y > 0)
    b, a = np.polyfit(np.log(x[k]), np.log(y[k]), 1)
    return float(a), float(b)


def cmd_report():
    S = s_list()
    X = pd.read_csv(G.RES / "X_targets.csv", dtype={"job_id": str})
    rows, s1_bad, s1_n, example = [], [], 0, None
    s2_bad, s2_reached, s3_bad, pipe = [], 0, [], []
    for r in S.itertuples():
        d = G.G2_FINAL / str(r.rxn_id)
        marks = sorted(p.name for p in d.glob(".*")) if d.is_dir() else []
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else {}
        st = "ok" if ".done" in marks else (marks[0][len(".fail_"):] if marks else "missing")
        rows.append(dict(job_id=r.job_id, rxn_id=r.rxn_id, dipole=r.dipole, n_atoms=r.n_atoms, elements=r.elements,
                         status=st, core_h=res.get("core_h_total"), remapped=res.get("remapped"),
                         imag=res.get("imag_freqs_cm"), share=res.get("imag_mode_forming_share"),
                         stereo=(res.get("stereo_result") or {}).get("status"), fail=res.get("fail_reasons")))
        # S-1
        n, bad, ex = audit_inputs(r.job_id)
        s1_n += n
        s1_bad += bad
        if example is None and ex and st == "ok":
            k = sorted(ex)[0]
            example = dict(job_id=r.job_id, name=k, text=ex[k])
        # S-2
        ref = res.get("ref_result")
        if ref:
            s2_reached += 1
            calls = [" ".join(c.get("args", [])) for c in ref.get("xtb_calls") or []]
            if ref.get("n_orca_outputs") != 0 or not any("--opt" in c for c in calls) or not any("--ohess" in c for c in calls):
                s2_bad.append(f"{r.job_id}: n_orca_outputs {ref.get('n_orca_outputs')}, xtb calls {calls}")
        # S-3 format / S-4 pipeline errors
        if len(marks) != 1:
            s3_bad.append(f"{r.job_id}: markers {marks}")
        if st == "ok":
            miss = [f for f in FINAL_FILES if not (d / f).is_file()]
            if miss or res.get("status") != "ok":
                s3_bad.append(f"{r.job_id}: missing {miss}, result status {res.get('status')}")
        if st == "missing" or st.startswith("error") or "infra" in st:
            pipe.append(f"{r.job_id}: {st}")
    D = pd.DataFrame(rows)
    sf = json.loads((G.RES / "S_F.json").read_text()) if (G.RES / "S_F.json").is_file() else {}
    sh = json.loads((G.RES / "S_H.json").read_text()) if (G.RES / "S_H.json").is_file() else {}
    for k, s in (("S_F", sf), ("S_H", sh)):
        if not s.get("pass"):
            pipe.append(f"{k}: {s.get('fail_stage')} {s.get('fail')}" if s else f"{k}: not run")
    for r in D[D.status == "ok"].itertuples():
        f = (sf.get("per_rxn") or {}).get(str(r.rxn_id), {})
        h = (sh.get("per_rxn") or {}).get(r.job_id, {})
        if not (f.get("features") and f.get("tokens")):
            s3_bad.append(f"{r.job_id}: features {f.get('features')} tokens {f.get('tokens')} ({f.get('why')})")
        if h.get("status") != "ok":
            s3_bad.append(f"{r.job_id}: HF {h.get('status')} {h.get('reason')}")
    for k, v in (("Q", sf.get("Q_report") or {}), ("T", sf.get("T_report") or {})):
        if v.get("stop") is not False:
            s3_bad.append(f"FU2 {k} gate: stop {v.get('stop')} {v.get('stop_reason', '')}")
    D["hf_status"] = [((sh.get("per_rxn") or {}).get(j) or {}).get("status") for j in D.job_id]
    D["hf_coreh"] = [((sh.get("per_rxn") or {}).get(j) or {}).get("coreh_total") for j in D.job_id]
    D["features"] = [((sf.get("per_rxn") or {}).get(str(r)) or {}).get("features") for r in D.rxn_id]
    D["tokens"] = [((sf.get("per_rxn") or {}).get(str(r)) or {}).get("tokens") for r in D.rxn_id]
    n_ok = int((D.status == "ok").sum())
    # S-5
    a, b = fit_power(D.n_atoms, D.core_h)
    N = X.n_atoms.to_numpy(float)
    g2_tot = float(np.exp(a) * (N ** b).sum())
    hk = D.hf_coreh.notna()
    ha, hb = fit_power(D.n_atoms[hk], D.hf_coreh[hk]) if hk.sum() >= 3 else (np.nan, np.nan)
    hf_tot = float(np.exp(ha) * (N ** hb).sum()) if hk.sum() >= 3 else float("nan")
    tot = g2_tot + (hf_tot if np.isfinite(hf_tot) else 0.0)
    s5 = dict(fit_g2=dict(a=a, b=b, form="core_h = exp(a) * n_atoms^b", n=int(D.core_h.notna().sum())),
              g2_total_core_h=g2_tot, g2_mean_x_N=float(D.core_h.mean() * len(X)),
              g2_median_x_N=float(D.core_h.median() * len(X)),
              fit_hf=dict(a=ha, b=hb, n=int(hk.sum())), hf_total_core_h=hf_tot, total_core_h=tot, limit=COREH_MAX,
              n_targets=len(X), sum_n_atoms=float(N.sum()))
    chk = {"S-1": dict(n_inputs=s1_n, problems=s1_bad[:30], pass_=not s1_bad and s1_n > 0),
           "S-2": dict(n_reached=s2_reached, problems=s2_bad, pass_=not s2_bad and s2_reached > 0),
           "S-3": dict(problems=s3_bad, pass_=not s3_bad),
           "S-4": dict(n_ok=n_ok, n=len(D), min_ok=N_OK_MIN, pipeline_errors=pipe, pass_=n_ok >= N_OK_MIN and not pipe),
           "S-5": dict(**s5, pass_=bool(tot <= COREH_MAX))}
    rep = dict(spec="docs/specs/D1G2.md 5", rows=D.to_dict("records"), checks=chk,
               all_pass=all(c["pass_"] for c in chk.values()), example_input=example,
               cores_per_rxn=4, n_parallel=10, S_F=sf.get("Q_report"), S_T=sf.get("T_report"))
    G.write_json(G.RES / "S_report.json", TT.clean(rep))
    md = ["# D1-G2 Phase S — smoke 10반응", "",
          f"S-1–S-5 판정: **{'모두 통과' if rep['all_pass'] else 'STOP'}** "
          f"(ok {n_ok}/{len(D)}, 전체 예상 {tot:,.0f} 코어시간).", "",
          "| 확인 | 결과 | 통과 |", "|---|---|---|",
          f"| S-1 ORCA 입력 | {s1_n}개 입력, 문제 {len(s1_bad)} | {'✓' if chk['S-1']['pass_'] else '✗'} |",
          f"| S-2 기준 구조 | 기준 단계 도달 {s2_reached}반응, DFT 0회·xtb opt/ohess 확인, 문제 {len(s2_bad)} | "
          f"{'✓' if chk['S-2']['pass_'] else '✗'} |",
          f"| S-3 출력 형식·끝까지 계산 | 문제 {len(s3_bad)} | {'✓' if chk['S-3']['pass_'] else '✗'} |",
          f"| S-4 성공 | ok {n_ok}/{len(D)} (≥ {N_OK_MIN}), 파이프라인 오류 {len(pipe)} | "
          f"{'✓' if chk['S-4']['pass_'] else '✗'} |",
          f"| S-5 비용 | G2 {g2_tot:,.0f} + HF {hf_tot:,.0f} = {tot:,.0f} 코어시간 (≤ {COREH_MAX:,.0f}) | "
          f"{'✓' if chk['S-5']['pass_'] else '✗'} |", "",
          "## 반응별", "", "| job | rxn | dipole | 원자 | 원소 | G2 | 코어시간 | 허수 (cm⁻¹) | feature | 토큰 | HF |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        im = ", ".join(f"{v:.0f}" for v in (r.imag or [])) if isinstance(r.imag, list) else ""
        md.append(f"| {r.job_id} | {r.rxn_id} | {r.dipole} | {r.n_atoms} | {r.elements} | {r.status} | "
                  f"{r.core_h:.3f} | {im} | {r.features} | {r.tokens} | {r.hf_status} |" if r.core_h == r.core_h else
                  f"| {r.job_id} | {r.rxn_id} | {r.dipole} | {r.n_atoms} | {r.elements} | {r.status} | – | {im} | "
                  f"{r.features} | {r.tokens} | {r.hf_status} |")
    md += ["", f"S-5 맞춤: G2 코어시간 = exp({a:.3f}) × 원자수^{b:.3f} (10반응); 평균 × 9,266 = "
               f"{s5['g2_mean_x_N']:,.0f}, 중앙값 × 9,266 = {s5['g2_median_x_N']:,.0f}. HF: exp({ha:.3f}) × 원자수^{hb:.3f}.",
           "", "문제 목록:", ""] + [f"- {p}" for c in ("S-1", "S-2", "S-3") for p in chk[c]["problems"]] + \
          [f"- {p}" for p in pipe] + ["", "## S-1 예시 입력 (그대로)", ""]
    if example:
        md += [f"`{example['job_id']}/{example['name']}`", "", "```", example["text"].rstrip(), "```"]
    G.write_atomic(G.RES / "S_REPORT.md", "\n".join(md) + "\n")
    print(json.dumps(TT.clean({k: {x: y for x, y in v.items() if x != "problems"} for k, v in chk.items()}), indent=1))
    print("all_pass", rep["all_pass"], flush=True)
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    cmds = {"f": cmd_f, "h": cmd_h, "report": cmd_report}
    if sys.argv[1:2] and sys.argv[1] in cmds:
        sys.exit(cmds[sys.argv[1]]())
    raise SystemExit(__doc__)
