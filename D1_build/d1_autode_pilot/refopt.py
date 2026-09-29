#!/usr/bin/env python3
"""refopt.py — stage 'refopt' (validation V2c, D0_reopt_L0 jobs only; VALIDATION_SPEC §6.3).

Coley's two strain references (labels_all rel1_file / rel2_file) are re-optimised at the L0 Opt level
(autodE's own Opt header from the pilot, orca_direct.opt_header), then frag{1,2}_rel.inp are rebuilt with
the repo builder text (build_inputs_graph.frag_header + write_block) and run. d1, d2 and the barrier are
recomputed against this job's eda / frag*_dist outputs; the 6 EDA channels do not depend on the
references, so they are not recomputed.

  python refopt.py --job V2L0_0020
Writes refopt/{rel1_opt,rel2_opt,frag1_rel,frag2_rel}.*, label_refopt.json, .done_refopt | .fail_refopt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_common as pc  # noqa: E402
import orca_direct as od  # noqa: E402
import build_sp_inputs as bsi  # noqa: E402


def heavy_rmsd_same_order(a, b):
    """Kabsch heavy-atom RMSD for two geometries with the same atom order (an optimisation of `a`)."""
    idx = [i for i, s in enumerate(a[0]) if s != "H"]
    P, Q = np.asarray(a[1])[idx], np.asarray(b[1])[idx]
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--config"); ap.add_argument("--manifest")
    a = ap.parse_args()
    cfg = pc.load_config(a.config); job = pc.load_job(a.job, a.manifest); jd = pc.job_dir(cfg, a.job)
    if pc.done(jd, "refopt"):
        print(f"{a.job}: refopt already done"); return
    if not pc.done(jd, "label"):
        raise SystemExit(f"{a.job}: label stage not done")
    _, s3 = pc.import_repo_modules(cfg)
    big, _ = bsi.repo_builders(cfg)
    meta = json.loads((jd / "sp" / "input_meta.json").read_text())
    lab = json.loads((jd / "label.json").read_text())
    q = (int(meta["charge1"]), int(meta["charge2"]))
    rel = [pc.read_xyz(jd / f"d0_{job['d0_rel1_file']}"), pc.read_xyz(jd / f"d0_{job['d0_rel2_file']}")]
    wd = jd / "refopt"; wd.mkdir(exist_ok=True)
    header = od.opt_header(cfg, n_cores=pc.n_cores())
    res = dict(job_id=a.job, opt_header=header, rel_files=[job["d0_rel1_file"], job["d0_rel2_file"]])
    opt = []
    for k, (r, qk) in enumerate(zip(rel, q), start=1):
        inp = wd / f"rel{k}_opt.inp"
        if not inp.is_file():
            od.write_inp(inp, header, *r, charge=qk, mult=1)
        text = od.run(cfg, inp).read_text(errors="replace")
        res[f"rel{k}_opt_hours"] = od.run_hours(text)
        if not (od.terminated(text) and od.opt_converged(text)):
            pc.write_json(jd / "label_refopt.json", res)
            pc.mark_fail(jd, "refopt", f"L0 Opt of reference {k} did not converge / terminate"); return
        o = pc.read_xyz(wd / f"rel{k}_opt.xyz")
        res[f"rel{k}_heavy_rmsd_vs_coley"] = heavy_rmsd_same_order(r, o)
        opt.append(o)
    E = {}
    for k, (o, qk) in enumerate(zip(opt, q), start=1):
        inp = wd / f"frag{k}_rel.inp"
        if not inp.is_file():
            inp.write_text(big.write_block(big.frag_header(qk), o[0], np.asarray(o[1])))
        text = od.run(cfg, inp).read_text(errors="replace")
        if not od.terminated(text):
            pc.write_json(jd / "label_refopt.json", res)
            pc.mark_fail(jd, "refopt", f"frag{k}_rel SP did not terminate normally"); return
        E[f"rel{k}"] = s3.read_fspe(inp.with_suffix(".out"), 1)
    EH = pc.EH_TO_KCAL
    e_ab, e_d1, e_d2 = lab["e_ab_eh"], lab["e_frag1_dist_eh"], lab["e_frag2_dist_eh"]
    new = dict(d1_kcal=(e_d1 - E["rel1"]) * EH, d2_kcal=(e_d2 - E["rel2"]) * EH,
               barrier_kcal=(e_ab - E["rel1"] - E["rel2"]) * EH)
    res.update(e_frag1_rel_refopt_eh=E["rel1"], e_frag2_rel_refopt_eh=E["rel2"], refopt=new,
               label={k: lab[k] for k in new},
               delta_refopt_minus_label={k: new[k] - lab[k] for k in new},
               delta_refopt_minus_d0={k: new[k] - float(job[f"d0_{k}"]) for k in new if job.get(f"d0_{k}") not in (None, "")})
    pc.write_json(jd / "label_refopt.json", res)
    pc.mark_done(jd, "refopt")
    print(f"{a.job}: refopt ok  " + "  ".join(f"{k} {v:+.3f}" for k, v in res["delta_refopt_minus_label"].items()))


if __name__ == "__main__":
    main()
