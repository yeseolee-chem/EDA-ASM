#!/usr/bin/env python3
"""assemble.py — stage 'label': method-① labels + the D0 audit gates for one job.

Label record : label_true/scripts/stage3_parse.parse_one_rxn  (the parser that built labels_all)
Audit gates  : label_true/scripts/smd_relabel_audit.check     (the audit that passed D0),
               pointed at this job's sp/ directory through a small symlink view; plus the
               uniformity checks of smd_relabel_uniformity.sh (ORCA 6.1.1 / SMD / water /
               eps 78.3550 / CDS present / terminated) for all 7 outputs.
Promotion    : eda_sum_mismatch -> ok only if |sum6ch - Bond| <= gates.closure_tol (0.11);
               D0 promoted every such record (max residual there 0.108 kcal/mol).
Writes label.json and .done_label (or .fail_label with the list of failed gates).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_common as pc  # noqa: E402

OUTS = ("eda", "eda_frag1", "eda_frag2", "frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel")
STANDALONE = ("eda", "frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel")     # the 5 processes run_sp starts
RUNTIME_RE = re.compile(r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec")


def sp_coreh(spd: Path) -> dict:
    """F4: core-hours per output = TOTAL RUN TIME x nprocs (eda: %pal of eda.inp; fragments run serially).
    eda_frag{1,2} are ORCA's own fragment SCFs inside the eda run, so the total counts the 5 processes only."""
    pal = re.search(r"%pal nprocs (\d+) end", (spd / "eda.inp").read_text()) if (spd / "eda.inp").is_file() else None
    np_eda = int(pal.group(1)) if pal else 1
    out = {}
    for k in OUTS:
        p = spd / f"{k}.out"
        m = RUNTIME_RE.findall(p.read_text(errors="replace")) if p.is_file() else []
        if not m:
            out[k] = None
            continue
        d, h, mi, s, ms = (int(x) for x in m[-1])
        out[k] = (d * 24 + h + mi / 60 + (s + ms / 1000) / 3600) * (np_eda if k.startswith("eda") else 1)
    out["total"] = sum(out[k] for k in STANDALONE if out.get(k))
    return out


def uniformity(spd: Path) -> dict:
    """Port of smd_relabel_uniformity.sh::outinfo for the 7 label-feeding outputs."""
    res, bad = {}, []
    for k in OUTS:
        p = spd / f"{k}.out"
        t = p.read_text(errors="replace") if p.is_file() else ""
        ver = re.search(r"Program Version\s+(\S+)", t)
        solv = re.findall(r"^Solvent:\s*(.*)$", t, re.M)
        eps = re.findall(r"^\s*Epsilon\s+\.\.\.\s*(\S+)", t, re.M)
        r = dict(version=ver.group(1) if ver else None,
                 solvent=solv[0].split()[-1] if solv else None,
                 eps=eps[0] if eps else None,
                 smd_module=t.count("utilizes the SMD solvation module"),
                 cds=t.count("SMD CDS free energy correction energy"),
                 terminated=t.count("ORCA TERMINATED NORMALLY"))
        ok = (r["version"] == "6.1.1" and (r["solvent"] or "").upper() == "WATER" and r["eps"] == "78.3550"
              and r["smd_module"] >= 1 and r["cds"] >= 1 and r["terminated"] >= 1)
        res[k] = r
        if not ok:
            bad.append(k)
    return dict(per_output=res, failed=bad)


def d0_audit(cfg, jd: Path, rid_name: str):
    """Run the D0 audit's check() on this job (NEW = sp/, OLD = view with the CPCM-form eda.inp)."""
    import smd_relabel_audit as au
    import build_sp_inputs as bsi
    spd = jd / "sp"
    view = jd / "audit_view"
    new, old = view / "new", view / "old" / rid_name
    old.mkdir(parents=True, exist_ok=True); new.mkdir(parents=True, exist_ok=True)
    link = new / rid_name
    if not link.exists():
        link.symlink_to(spd.resolve(), target_is_directory=True)
    # the pre-SMD form of this eda.inp (repo eda_header, same %pal) so check()'s rewrite test applies
    big, _ = bsi.repo_builders(cfg)
    eda_new = (spd / "eda.inp").read_text()
    pal = re.search(r"%pal nprocs \d+ end", eda_new).group(0)
    q1 = int(re.search(r"FRAG1_C\s+(-?\d+)", eda_new).group(1)); q2 = int(re.search(r"FRAG2_C\s+(-?\d+)", eda_new).group(1))
    body = eda_new.split(f"* xyz {q1 + q2} 1\n", 1)[1]
    old_form = re.sub(r"%pal nprocs \d+ end", pal, big.eda_header(q1, q2)) + body
    (old / "eda.inp").write_text(old_form)
    for s in ("frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel"):
        for ext in (".inp", ".out"):
            dst = old / f"{s}{ext}"
            if not dst.exists() and (spd / f"{s}{ext}").exists():
                dst.symlink_to((spd / f"{s}{ext}").resolve())
    au.NEW, au.OLD = view / "new", view / "old"
    r = au.check(rid_name)
    r["problems"] = [p for p in r["problems"] if p != "missing:old/eda.out"]   # D1 has no pre-SMD run
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--config"); ap.add_argument("--manifest")
    a = ap.parse_args()
    cfg = pc.load_config(a.config); job = pc.load_job(a.job, a.manifest); jd = pc.job_dir(cfg, a.job)
    if pc.done(jd, "label"):
        print(f"{a.job}: label already done"); return
    if not pc.done(jd, "sp"):
        raise SystemExit(f"{a.job}: sp stage not done")
    _, s3 = pc.import_repo_modules(cfg)
    g = cfg["gates"]
    spd = jd / "sp"
    meta = json.loads((spd / "input_meta.json").read_text())
    rec = s3.parse_one_rxn(spd, meta)
    promoted = False
    if rec.get("status") == "eda_sum_mismatch" and abs(rec["eda_sum_minus_bond_kcal"]) <= g["closure_tol"]:
        # sum_mismatch_promoted goes into the output once, below (it was also set in rec: duplicate keyword ->
        # TypeError on every promoted label; found by validation task V6b_J09_r2, 2026-09-30)
        rec["status"] = "ok"; promoted = True

    rid_name = f"rxn_{meta['rxn_id']}"
    try:
        audit = d0_audit(cfg, jd, rid_name)
    except Exception as e:                                        # noqa: BLE001
        audit = dict(problems=[f"audit_error:{type(e).__name__}: {e}"], gates_computable=False)
    uni = uniformity(spd)
    ts_res = json.loads((jd / "ts_result.json").read_text())
    ref_res = json.loads((jd / "ref_result.json").read_text())

    gates = {}
    if rec.get("status") in ("ok",):
        gates["g1_identity"] = abs(rec["identity_residual_kcal"]) <= g["identity_tol"]
        gates["g_bond_reference"] = abs(rec["bond_reference_residual_kcal"]) <= g["bond_ref_tol"]
        gates["g2_sum6_minus_bond"] = abs(rec["eda_sum_minus_bond_kcal"]) <= g["closure_tol"]
        gates["g3_channel_closure"] = abs(rec["channel_closure_residual_kcal"]) <= g["closure_tol"]
    gates["g_audit_no_problems"] = not audit.get("problems")
    gates["g_uniform_orca611_smd_water"] = not uni["failed"]
    flags = dict(
        d1_below_min=rec.get("d1_kcal", 0) < g["d_min_kcal"], d2_below_min=rec.get("d2_kcal", 0) < g["d_min_kcal"],
        pauli_not_positive=rec.get("pauli_dft", 1) <= 0, oi_not_negative=rec.get("oi_dft", -1) >= 0,
        flag_async=meta["flag_async"], audit_flags=audit.get("flags", []),
    )
    out = dict(job_id=a.job, kind=job["kind"], ts_id=job["ts_id"], cell_id=job.get("cell_id"), core=job["core"],
               panel=job["panel"], regio=job.get("regio"), R1=job.get("R1"), R2=job.get("R2"),
               rxn_smiles=job["rxn_smiles"], engine=cfg["ts_stage"]["engine"],
               **{k: v for k, v in rec.items() if not k.startswith("_")},
               sum_mismatch_promoted=promoted, gates=gates, flags=flags,
               audit_problems=audit.get("problems"), audit_stage3_status=audit.get("stage3_status"),
               uniformity_failed=uni["failed"], sp_coreh=sp_coreh(spd),
               ts_imag_cm=(ts_res.get("imag_freqs_cm") or [None])[0], ts_dipole_config=ts_res.get("ts_dipole_config"),
               reactant_dipole_config=ts_res.get("reactant_dipole_config"),
               autode_dE_act_sp_kcal=ts_res.get("dE_act_sp_kcal"), autode_dG_act_kcal=ts_res.get("dG_act_kcal"),
               ref_source=ref_res.get("source"), ref_port=ref_res.get("port") or ref_res.get("port_diagnostic"))
    if job["kind"].startswith("D0"):
        out["d0"] = {k[3:]: job[k] for k in job if k.startswith("d0_")}
    pc.write_json(jd / "label.json", out)
    failed = [k for k, v in gates.items() if not v]
    if rec.get("status") != "ok" or failed:
        pc.mark_fail(jd, "label", f"status={rec.get('status')} failed_gates={failed} "
                                  f"audit={audit.get('problems')} uniformity={uni['failed']}")
        return
    pc.mark_done(jd, "label")
    print(f"{a.job}: label ok  barrier={rec['barrier_kcal']:.2f} d1={rec['d1_kcal']:.2f} d2={rec['d2_kcal']:.2f} "
          f"elst={rec['elst_dft']:.2f} pauli={rec['pauli_dft']:.2f} oi={rec['oi_dft']:.2f} disp={rec['disp_dft']:.2f} "
          f"cpcm={rec['cpcm_dft']:.2f} cds={rec['cds_dft']:.2f}")


if __name__ == "__main__":
    main()
