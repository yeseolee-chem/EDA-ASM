#!/usr/bin/env python3
"""D1-G2 Phase G checks (docs/specs/D1G2.md 6, STOP 14 G). sbatch only (jobs/g2.slurm, G2ENV=d1); reads $G2/g2_final,
writes only results/. The pool code (g2_autode.py) is not touched while Phase G runs.

  python g2_check.py interim   # every completed block of 1,000 finished reactions without a report ->
                               #   results/G_interim_<k>.md / .json; exit 3 if a block fails a STOP rule
  python g2_check.py status    # results/G_status.csv, results/G_report.json (+ the interim blocks); exit 0 = all
                               #   targets finished and no STOP, 2 = unfinished, 3 = STOP

Finished = $G2/g2_final/<rxn_id>/ (all-digit name, an X_targets rxn) with .done or .fail_<reason>; finish time = the
marker mtime (ties by rxn_id). Block k = the k-th 1,000 finished in that order (the last, partial block is judged by
'status' once every target is finished). STOP (spec 14 G), judged per block, cumulative reported: ok < 70 %, infra +
pipeline > 2 %, .fail_order > 1 %.
Categories (g2_autode.finalize reasons = <stage>_<first part of the stage failure text> | order | freq_not_terminated |
hess_no_imaginary_frequency | error_<Exception>; stage texts from d1_autode_pilot run_ts / make_reference / stereo):
  ok        .done
  order     order (no label-order isomorphism)
  infra     ts autodE exception after the D1 retry_infra_once retry (ts_result attempts = 2); freq_not_terminated
  pipeline  error_* (wrapper exception); <stage>_not_run (no stage marker); ref_<ExceptionName> (non-StageFail);
            ref xtb rotatability scan not parsed / energies.csv missing; no result.json
  method    everything else: autodE found no TS, autodE exception without infra evidence, TS check failed, partition /
            forming-bond map, OptTS not converged / not terminated, stereo mismatch, reference rules (ring count,
            dipolarophile configuration, H assignment, partition), hess_no_imaginary_frequency
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_common as G  # noqa: E402

BLOCK = 1000
OK_MIN, INFRA_PIPE_MAX, ORDER_MAX = 0.70, 0.02, 0.01
PIPE_RE = re.compile(r"^(error_|\w+_not_run$|ref_[A-Z]\w*(Error|Exception)$|ref_xtb_rotatability|ref_\S+ not in energies)")


def category(marker_reason, res):
    if marker_reason == "done":
        return "ok"
    r = (res.get("fail_reasons") or [marker_reason])[0] if res else None
    if r is None:
        return "pipeline"
    if r == "order":
        return "order"
    if r == "freq_not_terminated":
        return "infra"
    if r.startswith("ts_autodE exception") and len((res.get("ts_result") or {}).get("attempts") or []) >= 2:
        return "infra"
    if PIPE_RE.match(r) or r.startswith("ref_xtb rotatability") or " not in energies.csv" in r:
        return "pipeline"
    return "method"


def scan():
    X = pd.read_csv(G.RES / "X_targets.csv", dtype={"job_id": str})
    O = pd.read_csv(G.RES / "G_order.csv", dtype={"job_id": str})[["job_id", "group_name", "order"]]
    X = X.merge(O, on="job_id", how="left")
    rows = []
    for t in X.itertuples():
        d = G.G2_FINAL / str(t.rxn_id)
        marks = sorted(p for p in d.glob(".*") if p.name == ".done" or p.name.startswith(".fail_")) if d.is_dir() else []
        row = dict(rxn_id=int(t.rxn_id), job_id=t.job_id, group_name=t.group_name, order=t.order, n_atoms=int(t.n_atoms),
                   dipole=t.dipole, skeleton=t.skeleton)
        if not marks:
            rows.append(dict(row, status="not_finished"))
            continue
        m = marks[0]
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else {}
        reason = "done" if m.name == ".done" else m.name[len(".fail_"):]
        ch = res.get("core_h") or {}
        rows.append(dict(row, status=category(reason, res), fail_reason="" if reason == "done" else
                         ";".join(res.get("fail_reasons") or [reason]), marker=m.name, finish_time=m.stat().st_mtime,
                         remapped=res.get("remapped"), core_h_total=res.get("core_h_total"), cores=res.get("cores"),
                         **{f"core_h_{k}": v for k, v in ch.items()},
                         imag_freqs_cm=json.dumps(res.get("imag_freqs_cm")) if res.get("imag_freqs_cm") is not None else "",
                         n_imag=res.get("n_imag"), stereo=(res.get("stereo_result") or {}).get("status")))
    return pd.DataFrame(rows)


def stats(D):
    n = len(D)
    c = D.status.value_counts().to_dict()
    f = lambda k: c.get(k, 0) / n if n else float("nan")      # noqa: E731
    ok = D[D.status == "ok"]
    return dict(n=n, by_status=c, ok_rate=f("ok"), infra_rate=f("infra"), pipeline_rate=f("pipeline"),
                infra_pipeline_rate=f("infra") + f("pipeline"), order_rate=f("order"),
                remapped_rate_ok=float(ok.remapped.fillna(False).astype(bool).mean()) if len(ok) else float("nan"),
                reasons=D[D.status != "ok"].fail_reason.value_counts().head(40).to_dict(),
                core_h=float(pd.to_numeric(D.core_h_total, errors="coerce").sum()),
                core_h_per_rxn_median=float(pd.to_numeric(D.core_h_total, errors="coerce").median()) if n else None)


def judge(s):
    bad = []
    if s["ok_rate"] < OK_MIN:
        bad.append(f"ok {s['ok_rate']:.3f} < {OK_MIN}")
    if s["infra_pipeline_rate"] > INFRA_PIPE_MAX:
        bad.append(f"infra + pipeline {s['infra_pipeline_rate']:.3f} > {INFRA_PIPE_MAX}")
    if s["order_rate"] > ORDER_MAX:
        bad.append(f".fail_order {s['order_rate']:.3f} > {ORDER_MAX}")
    return bad


def md_block(k, b, cum, bad):
    p = lambda x: f"{100 * x:.1f} %"                         # noqa: E731
    L = [f"# D1-G2 Phase G 중간 점검 {k} (끝난 반응 {k * BLOCK if b['n'] == BLOCK else cum['n']:,}개)", "",
         f"판정: **{'STOP — ' + '; '.join(bad) if bad else '통과'}** (기준은 이 블록 {b['n']}개: ok ≥ 70 %, "
         f"인프라+파이프라인 ≤ 2 %, .fail_order ≤ 1 %)", "",
         "| | 이 블록 | 누적 |", "|---|---|---|"]
    for lab, key in (("반응 수", "n"), ("ok 비율", "ok_rate"), ("인프라 실패", "infra_rate"),
                     ("파이프라인 실패", "pipeline_rate"), (".fail_order", "order_rate"),
                     ("원자 순서 다시 매김 (ok 중)", "remapped_rate_ok"), ("코어시간 합", "core_h")):
        fmt = (lambda x: f"{x:,}") if key == "n" else ((lambda x: f"{x:,.1f}") if key == "core_h" else p)
        L.append(f"| {lab} | {fmt(b[key])} | {fmt(cum[key])} |")
    L += ["", "실패 이유 (이 블록):", ""] + [f"- {r}: {n}" for r, n in b["reasons"].items()]
    return "\n".join(L) + "\n"


def blocks(D, final=False):
    """(list of (k, block stats, cumulative stats, STOP reasons)) for every full block (+ the last partial one if final)."""
    F = D[D.status != "not_finished"].sort_values(["finish_time", "rxn_id"]).reset_index(drop=True)
    out = []
    nb = len(F) // BLOCK + (1 if final and len(F) % BLOCK else 0)
    for k in range(1, nb + 1):
        B, C = F.iloc[(k - 1) * BLOCK:k * BLOCK], F.iloc[:k * BLOCK]
        b, c = stats(B), stats(C)
        out.append((k, b, c, judge(b)))
    return out


def write_blocks(D, final=False):
    stop = []
    for k, b, c, bad in blocks(D, final):
        f = G.RES / f"G_interim_{k}.json"
        if not f.is_file() or json.loads(f.read_text()).get("block", {}).get("n") != b["n"]:
            G.write_json(f, dict(k=k, block=b, cumulative=c, stop=bad, written=time.strftime("%F %T"),
                                 rule=dict(ok_min=OK_MIN, infra_pipeline_max=INFRA_PIPE_MAX, order_max=ORDER_MAX,
                                           judged_on="block of 1,000 finished (finish-time order)")))
            G.write_atomic(G.RES / f"G_interim_{k}.md", md_block(k, b, c, bad))
            print(f"[interim {k}] n {b['n']} ok {b['ok_rate']:.3f} infra+pipe {b['infra_pipeline_rate']:.3f} order "
                  f"{b['order_rate']:.3f} -> {'STOP ' + '; '.join(bad) if bad else 'pass'}", flush=True)
        if bad:
            stop.append(dict(k=k, reasons=bad))
    return stop


def cmd_interim():
    D = scan()
    stop = write_blocks(D)
    n_fin = int((D.status != "not_finished").sum())
    print(f"[interim] finished {n_fin}/{len(D)}, cumulative {json.dumps(stats(D[D.status != 'not_finished']), default=str)}",
          flush=True)
    if stop:
        print(f"STOP: {stop}", flush=True)
    return 3 if stop else 0


def cmd_status():
    D = scan()
    done = bool((D.status != "not_finished").all())
    stop = write_blocks(D, final=done)
    F = D[D.status != "not_finished"]
    G.write_atomic(G.RES / "G_status.csv", lambda t: D.drop(columns=["order"]).to_csv(t, index=False))
    s = json.loads((G.RES / "S_report.json").read_text()) if (G.RES / "S_report.json").is_file() else {}
    rep = dict(spec="docs/specs/D1G2.md 6", n_targets=len(D), n_finished=len(F), all_finished=done,
               overall=stats(F), overall_stop=judge(stats(F)) if len(F) else [],
               by_group={g: stats(x) for g, x in F.groupby("group_name")},
               by_dipole={g: dict(n=len(x), ok_rate=float((x.status == "ok").mean())) for g, x in F.groupby("dipole")},
               blocks=[dict(k=k, n=b["n"], ok_rate=b["ok_rate"], infra_pipeline_rate=b["infra_pipeline_rate"],
                            order_rate=b["order_rate"], stop=bad) for k, b, c, bad in blocks(D, final=done)],
               stop=stop, cores_per_rxn=4, n_parallel_per_job=12, cpus_per_job=48, n_pool_jobs=9,
               cores_basis="S: 10 rxns x 4 cores, wall 75-521 s per rxn (results/S_report.json rows)",
               s_rows=[{k: r.get(k) for k in ("job_id", "n_atoms", "status", "core_h")} for r in s.get("rows", [])],
               status_csv_sha256=G.sha256(G.RES / "G_status.csv"))
    G.write_json(G.RES / "G_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_group", "s_rows")}, indent=1, default=str), flush=True)
    if stop:
        return 3
    return 0 if done else 2


if __name__ == "__main__":
    cmds = {"interim": cmd_interim, "status": cmd_status}
    if sys.argv[1:2] and sys.argv[1] in cmds:
        sys.exit(cmds[sys.argv[1]]())
    raise SystemExit(__doc__)
