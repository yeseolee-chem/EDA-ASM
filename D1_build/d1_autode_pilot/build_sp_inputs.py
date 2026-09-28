#!/usr/bin/env python3
"""build_sp_inputs.py — stage 'inputs': the 5 ORCA single-point inputs of one job.

The text comes from the repo's own D0 builder, so D1 and D0 inputs are identical by
construction:
  frag{1,2}_dist.inp, frag{1,2}_rel.inp : build_inputs_graph.frag_header + write_block
                                          (B3LYP D3BJ def2-TZVP CPCM(water) + %cpcm smd block)
  eda.inp                               : smd_relabel_build_inputs.rewrite(eda_header(...))
                                          (SMD(water) in the header and both FRAG strings)
Only the `%pal nprocs` line of eda.inp is set from config (energies do not depend on it).
frag1 = dipole, frag2 = dipolarophile (graph partition of the TS with the two references).
Writes sp/{eda,frag1_dist,frag2_dist,frag1_rel,frag2_rel}.inp, sp/input_meta.json, .done_inputs
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

SP_FILES = ("eda.inp", "frag1_dist.inp", "frag2_dist.inp", "frag1_rel.inp", "frag2_rel.inp")


def repo_builders(cfg):
    for sub in ("analysis/b3lyp_full/build_strategy/graph", "label_true/scripts"):
        p = str(pc.repo_path(cfg, sub))
        if p not in sys.path:
            sys.path.insert(0, p)
    import build_inputs_graph as big
    import smd_relabel_build_inputs as smd
    return big, smd


def render(cfg, syms, xyz, A, B, rel, charges):
    """{filename: text} — same calls as build_inputs_graph.emit, then the SMD rewrite of eda.inp."""
    big, smd = repo_builders(cfg)
    q1, q2 = charges
    lab = [1 if i in A else 2 for i in range(len(syms))]
    out = {"eda.inp": smd.rewrite(big.write_block(big.eda_header(q1, q2), syms, xyz, lab))}
    out["eda.inp"] = re.sub(r"%pal nprocs \d+ end", f"%pal nprocs {int(cfg['sp_stage']['eda_nprocs'])} end",
                            out["eda.inp"])
    for k, (idx, q) in enumerate([(sorted(A), q1), (sorted(B), q2)], start=1):
        out[f"frag{k}_dist.inp"] = big.write_block(big.frag_header(q), [syms[i] for i in idx], xyz[idx])
    for k, ((rs, rx), q) in enumerate(zip(rel, (q1, q2)), start=1):
        out[f"frag{k}_rel.inp"] = big.write_block(big.frag_header(q), rs, rx)
    return out


def check_method_lines(cfg, texts):
    """g_setting: the level of theory the labels are defined at (fails loudly otherwise)."""
    sp = cfg["sp_stage"]
    eda = texts["eda.inp"]
    errs = []
    head = eda.splitlines()[0]
    if head != "! " + sp["eda_level"].replace("TightSCF", "EDA TightSCF"):
        errs.append(f"eda header: {head}")
    if eda.count("SMD(water)") != 3 or "%cpcm" in eda or "CPCM(water)" in eda:
        errs.append("eda.inp must carry SMD(water) 3x (header + FRAG1 + FRAG2) and no %cpcm block")
    if eda.count(f'"{sp["eda_level"]}"') != 2:
        errs.append("FRAG strings differ from eda_level")
    for f in SP_FILES[1:]:
        t = texts[f]
        if t.splitlines()[0] != "! " + sp["frag_level"] or 'smdsolvent "water"' not in t or "smd true" not in t:
            errs.append(f"{f}: method line / smd block")
    if f"%maxcore {sp['maxcore_mb']}" not in eda:
        errs.append("maxcore")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--config"); ap.add_argument("--manifest")
    a = ap.parse_args()
    cfg = pc.load_config(a.config); job = pc.load_job(a.job, a.manifest); jd = pc.job_dir(cfg, a.job)
    if pc.done(jd, "inputs"):
        print(f"{a.job}: inputs already done"); return
    if not pc.done(jd, "ref"):
        raise SystemExit(f"{a.job}: ref stage not done")
    fr, _ = pc.import_repo_modules(cfg)
    ref_res = json.loads((jd / "ref_result.json").read_text())
    ts = pc.read_xyz(jd / "ts.xyz")
    rd, rp = pc.read_xyz(jd / "ref_dipole.xyz"), pc.read_xyz(jd / "ref_dipolarophile.xyz")
    P = fr.partition(ts, rd, rp)                     # frag1 = dipole by construction (r0 = dipole)
    if P.status != "ok":
        pc.mark_fail(jd, "inputs", f"partition with references: {P.status}"); return
    rmols, formed, _ = fr.smiles_reactants_and_formed(job["mapped_smiles"])
    assign = fr.match_smiles_to_reactants(rmols, rd, rp)
    if assign is None:
        pc.mark_fail(jd, "inputs", "reference geometries do not match the mapped SMILES"); return
    q1, q2 = fr.formal_charges(rmols, assign)
    for k, (idx, q) in enumerate([(P.A, q1), (P.B, q2)], start=1):
        if fr.n_electrons([ts[0][i] for i in idx], q) % 2:
            pc.mark_fail(jd, "inputs", f"odd_electrons_frag{k}"); return
    audit = fr.audit_forming_bonds(P, ts, rd, rp, job["mapped_smiles"])
    texts = render(cfg, ts[0], ts[1], P.A, P.B, [rd, rp], (q1, q2))
    errs = check_method_lines(cfg, texts)
    if errs:
        pc.mark_fail(jd, "inputs", "g_setting: " + "; ".join(errs)); return
    spd = jd / "sp"; spd.mkdir(exist_ok=True)
    for f, t in texts.items():
        (spd / f).write_text(t)
    d = sorted(audit.d_formed)
    meta = dict(
        rxn_id=(999000 if a.job.startswith("S") else 900000) + int(a.job.lstrip("JS") or 0), job_id=a.job, kind=job["kind"], ts_id=job["ts_id"],
        ts_file="ts.xyz", n_atoms=len(ts[0]), n_f1=len(P.A), n_f2=len(P.B),
        rel1_file="ref_dipole.xyz", rel2_file="ref_dipolarophile.xyz", role1="dipole", role2="dipolarophile",
        charge1=q1, charge2=q2, alt_used=int(ref_res.get("alt_used", 0)),
        flag_foreign_bond=len(audit.foreign) > 0, flag_async=d[-1] > cfg["gates"]["forming_max_A"],
        formed_d1=d[0], formed_d2=d[-1], formed_pairs_ts=" ".join(f"{i}-{j}" for i, j in sorted(audit.formed_ts)),
        recovery_method="graph", A_idx=" ".join(map(str, sorted(P.A))),
    )
    pc.write_json(spd / "input_meta.json", meta)
    pc.mark_done(jd, "inputs")
    print(f"{a.job}: 5 inputs written (q1={q1}, q2={q2}, n_f1={len(P.A)}, n_f2={len(P.B)}, "
          f"d_form={d[0]:.3f}/{d[-1]:.3f}, alt_used={meta['alt_used']})")


if __name__ == "__main__":
    main()
