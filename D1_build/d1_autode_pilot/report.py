#!/usr/bin/env python3
"""report.py — aggregate the pilot: status, labels, TS/reference diagnostics, controls, cost.

  python report.py [--config config.yaml] [--manifest pilot_manifest.csv] [--out DIR]
Writes <out>/pilot_report.md, labels_d1_pilot.json, pilot_summary.csv (default out =
<scratch>/results; run_report.sh copies them into the repo).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_common as pc  # noqa: E402

TARGETS = ["barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
N_D1_TOTAL = 2846            # backbone TS rows of D1_설계_v8 (sample_pilot.py pool)


GEOMETRIES = ("ts.xyz", "ref_dipole.xyz", "ref_dipolarophile.xyz", "product.xyz")


def jload(p: Path):
    return json.loads(p.read_text()) if p.is_file() else {}


def ref_corrected(dE, dG, alt_used, port):
    """F2: autodE's ΔE‡/ΔG‡ are relative to autodE's own reactant dipole. When the Coley-port alt replaced
    it (alt_used == 1), refer them to the alt, as the label and Coley's G_act are:
    x_ref = x - port.<x>_alt_minus_orig. Returns (dE_ref, dG_ref); unchanged when no alt was used."""
    if not alt_used or not port:
        return dE, dG
    dEs, dGs = port.get("dEsp_alt_minus_orig_kcal"), port.get("dG_alt_minus_orig_kcal")
    return (None if dE is None or dEs is None else dE - dEs,
            None if dG is None or dGs is None else dG - dGs)


def copy_geometries(jd: Path, dest: Path):
    """F5: the small xyz files that define a label, kept with the committed results."""
    import shutil
    got = [f for f in GEOMETRIES if (jd / f).is_file()]
    if got:
        dest.mkdir(parents=True, exist_ok=True)
        for f in got:
            shutil.copy(jd / f, dest / f)


def stage_state(jd: Path):
    last, fail = None, None
    for s in pc.STAGES:
        if pc.done(jd, s):
            last = s
        elif pc.failed(jd, s):
            fail = (s, pc.failed(jd, s)); break
        else:
            break
    return last, fail


def timing(jd: Path):
    t = {}
    f = jd / "timing.tsv"
    if f.is_file():
        for line in f.read_text().splitlines():
            p = line.split("\t")
            if len(p) >= 5 and p[3] == "0":
                t[p[0]] = t.get(p[0], 0) + (int(p[2]) - int(p[1])) * int(p[4]) / 3600.0   # core-hours
    return t


def fmt(x, n=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--manifest"); ap.add_argument("--out")
    a = ap.parse_args()
    cfg = pc.load_config(a.config)
    man = pd.read_csv(a.manifest or HERE / "pilot_manifest.csv", keep_default_na=False)
    out = Path(a.out or Path(cfg["scratch"]) / "results"); out.mkdir(parents=True, exist_ok=True)
    g = cfg["gates"]
    ropt = cfg.get("report", {})
    use_ref = bool(ropt.get("ref_corrected_autode", False))       # F2; False reproduces the pilot report
    rows, labels = [], []
    for _, j in man.iterrows():
        jd = Path(cfg["scratch"]) / "jobs" / j.job_id
        last, fail = stage_state(jd)
        ts, ref, lab = jload(jd / "ts_result.json"), jload(jd / "ref_result.json"), jload(jd / "label.json")
        port = ref.get("port") or ref.get("port_diagnostic") or {}
        tm = timing(jd)
        if ropt.get("copy_geometries", True):
            copy_geometries(jd, out / "geometries" / j.job_id)
        dE_ref, dG_ref = ref_corrected(ts.get("dE_act_sp_kcal"), ts.get("dG_act_kcal"), ref.get("alt_used"), port)
        r = dict(job_id=j.job_id, kind=j.kind, ts_id=j.ts_id, core=j.core, panel=j.panel,
                 n_atoms=(lab.get("n_atoms") or (len(ts.get("A_idx", [])) + len(ts.get("B_idx", [])) or None)),
                 last_stage=last, fail_stage=fail[0] if fail else None, fail_reason=fail[1] if fail else None,
                 imag_cm=(ts.get("imag_freqs_cm") or [None])[0], d_form_1=(ts.get("formed_d") or [None])[0],
                 d_form_2=(ts.get("formed_d") or [None, None])[-1], mode_share=ts.get("imag_mode_forming_share"),
                 ts_cfg=ts.get("ts_dipole_config"), reac_cfg=ts.get("reactant_dipole_config"),
                 frozen=sum(1 for s in port.get("scans", []) if s.get("rotatable") is False),
                 to_run=port.get("to_run"), alt_used=ref.get("alt_used"), port_reason=port.get("reason"),
                 strict_would_use_alt=port.get("strict_rule_would_use_alt"),
                 port_agrees_coley=port.get("agrees_with_coley"),
                 autode_dE=ts.get("dE_act_sp_kcal"), autode_dG=ts.get("dG_act_kcal"),
                 autode_dE_ref=dE_ref, autode_dG_ref=dG_ref,
                 status=lab.get("status"), **{k: lab.get(k) for k in TARGETS},
                 gates_failed=",".join(k for k, v in (lab.get("gates") or {}).items() if not v),
                 **{f"coreh_{s}": tm.get(s) for s in ("ts", "ref", "inputs", "sp", "label")})
        r["coreh_total"] = sum(v for k, v in r.items() if k.startswith("coreh_") and v)
        if lab and lab.get("status") == "ok" and last == "label":
            labels.append(lab)
        if j.kind.startswith("D0"):
            for k in TARGETS:
                d0 = j.get(f"d0_{k}")
                r[f"delta_{k}"] = (lab[k] - float(d0)) if (lab.get(k) is not None and d0 != "") else None
            r["delta_dform_max"] = (max(abs(a_ - float(b_)) for a_, b_ in zip(sorted(ts["formed_d"]),
                                    (j.d0_formed_d1, j.d0_formed_d2))) if ts.get("formed_d") else None)
            dG_cmp = dG_ref if use_ref else ts.get("dG_act_kcal")
            r["delta_dG_vs_coley"] = (dG_cmp - float(j.d0_G_act_kcal)) if dG_cmp is not None else None
            r["d0_alt_used"] = j.d0_alt_used
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pilot_summary.csv", index=False)
    pc.write_json(out / "labels_d1_pilot.json", labels)

    d1 = df[df.kind == "D1"]
    ok_d1 = d1[(d1.last_stage == "label") & (d1.status == "ok")]
    rep = df[df.kind == "D0_replay"]
    d0c = df[df.kind == "D0_control"]
    L = []
    L.append("# D1 autodE pilot — report\n")
    L.append(f"engine: `{cfg['ts_stage']['engine']}`; jobs: {len(df)} (D1 {len(d1)}, D0_control {len(d0c)}, D0_replay {len(rep)})\n")
    # ---- acceptance
    acc = []
    acc.append(("A1 D1 TS rows with an ok label >= 8/10", f"{len(ok_d1)}/{len(d1)}", len(ok_d1) >= 8))
    gates_bad = df[(df.last_stage == "label") & (df.gates_failed != "")]
    acc.append(("A2 every produced label passes all gates", f"{len(gates_bad)} with failed gates", len(gates_bad) == 0))
    rep_ok = rep[rep.status == "ok"]
    rep_dev = max((abs(rep_ok[f"delta_{k}"]).max() for k in TARGETS), default=np.nan) if len(rep_ok) else np.nan
    acc.append(("A3 D0_replay reproduces labels_all (9 targets, |Δ| <= 0.1 kcal/mol)",
                f"max |Δ| = {fmt(rep_dev, 4)} over {len(rep_ok)}/{len(rep)}", len(rep_ok) == len(rep) and rep_dev <= 0.1))
    agree = rep.port_agrees_coley.tolist()
    acc.append(("A4 Coley-port decision == Coley's alt choice on the replay controls", str(agree),
                all(x is True for x in agree) and len(agree) > 0))
    c_ok = d0c[d0c.status == "ok"]
    cdev = abs(c_ok["delta_barrier_kcal"]).max() if len(c_ok) else np.nan
    acc.append((f"A5 D0_control |Δbarrier| <= {g['control_barrier_tol']} (diagnose if not; not a hard stop)",
                f"max {fmt(cdev)} over {len(c_ok)}/{len(d0c)}", len(c_ok) == len(d0c) and cdev <= g["control_barrier_tol"]))
    L.append("## Acceptance\n\n| criterion | value | pass |\n|---|---|---|")
    L += [f"| {c} | {v} | {'PASS' if p else 'FAIL'} |" for c, v, p in acc]
    # ---- per-job status
    L.append("\n## Jobs\n")
    cols = ["job_id", "kind", "ts_id", "core", "n_atoms", "last_stage", "fail_stage", "imag_cm", "d_form_1", "d_form_2",
            "ts_cfg", "reac_cfg", "frozen", "alt_used", "status", "coreh_total"]
    L.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for _, r in df.iterrows():
        L.append("| " + " | ".join(fmt(r[c]) if isinstance(r[c], float) else str(r[c] if r[c] is not None else "") for c in cols) + " |")
    fails = df[df.fail_stage.notna()]
    if len(fails):
        L.append("\n### Failures\n")
        L += [f"- **{r.job_id}** ({r.ts_id}, {r.core}) at `{r.fail_stage}`: {r.fail_reason}" for _, r in fails.iterrows()]
    # ---- labels
    L.append("\n## Labels (kcal/mol, method ①)\n")
    L.append("| job | ts_id | " + " | ".join(t.replace("_kcal", "").replace("_dft", "") for t in TARGETS) + " | autodE ΔE‡(sp) | autodE ΔG‡ |")
    L.append("|" + "---|" * (len(TARGETS) + 4))
    for _, r in df[df.status == "ok"].iterrows():
        L.append(f"| {r.job_id} | {r.ts_id} | " + " | ".join(fmt(r[t]) for t in TARGETS) + f" | {fmt(r.autode_dE)} | {fmt(r.autode_dG)} |")
    # ---- controls
    if len(d0c) + len(rep):
        L.append("\n## D0 controls (label − labels_all)\n")
        L.append("| job | kind | rxn | " + " | ".join(f"Δ{t.replace('_kcal', '').replace('_dft', '')}" for t in TARGETS)
                 + " | Δd_form max (Å) | ΔG‡ − Coley G_act | alt (Coley / ours) |")
        L.append("|" + "---|" * (len(TARGETS) + 6))
        for _, r in pd.concat([d0c, rep]).iterrows():
            L.append(f"| {r.job_id} | {r.kind} | {r.ts_id} | " + " | ".join(fmt(r.get(f'delta_{t}'), 3) for t in TARGETS)
                     + f" | {fmt(r.get('delta_dform_max'), 3)} | {fmt(r.get('delta_dG_vs_coley'))} | {r.get('d0_alt_used')} / {r.alt_used} |")
    # ---- stereo / reference
    L.append("\n## Dipole configuration and reference choice\n")
    L.append("| job | core | TS cfg | reactant cfg | frozen dihedrals | to_run | alt used | Coley-port reason | lowest-compatible rule would use alt |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        if isinstance(r.ts_cfg, str) and r.ts_cfg:
            L.append(f"| {r.job_id} | {r.core} | {r.ts_cfg} | {r.reac_cfg} | {r.frozen} | {r.to_run} | {r.alt_used} | "
                     f"{r.port_reason} | {r.strict_would_use_alt} |")
    # ---- cost
    done = df[(df.kind == "D1") & (df.last_stage == "label")]
    if len(done):
        ch = done.coreh_total
        per = dict(mean=ch.mean(), median=ch.median(), max=ch.max())
        cpus = int(cfg["cpus_per_task"])
        wall_days = ch.mean() * N_D1_TOTAL / (10 * cpus) / 24
        L.append("\n## Cost\n")
        L.append(f"core-hours per D1 TS row: mean {per['mean']:.1f}, median {per['median']:.1f}, max {per['max']:.1f} "
                 f"(stages: " + ", ".join(f"{s} {done[f'coreh_{s}'].mean():.1f}" for s in ('ts', 'ref', 'sp')) + ")")
        L.append(f"\nextrapolation to {N_D1_TOTAL} backbone TS rows: {ch.mean() * N_D1_TOTAL:,.0f} core-hours; "
                 f"at MaxJobs=10 × {cpus} cores ≈ {wall_days:.0f} days wall (mean-based; the max job sets the tail).")
    (out / "pilot_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
