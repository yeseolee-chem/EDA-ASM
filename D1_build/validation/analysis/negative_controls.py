#!/usr/bin/env python3
"""negative_controls.py — V8 (VALIDATION_SPEC §6.7, A7): does each gate catch a planted defect?

  python analysis/negative_controls.py --task V8
Every defect is planted in a COPY (pilot J12 = D0 rxn 20 replay, copied from the read-only pilot scratch,
or D0 profile files copied by run_ts.load_replay); the gate function the pipeline uses is then called.
  N1 eda.inp SMD(water) -> CPCM(water)        build_sp_inputs.check_method_lines, D0 audit
  N2 eda.inp FRAG1_C 0 -> 1                   fragment electron parity, D0 audit (assemble's audit_error path)
  N3 D0 rxn 3090 analysed as a replay         run_ts.analyse no_foreign_bond
  N4 product geometry + random mode as a TS   run_ts.analyse forming_in_range / imag_mode_on_forming_bonds
  N5 'ORCA TERMINATED NORMALLY' removed        run_sp criterion, assemble.uniformity, stage3 status
  N6 SMD CDS line removed                      assemble.uniformity cds, D0 audit no_smd_cds
  N7 plain dipole instead of _alt (rxn 20)     port diagnostic use_alt disagrees (A4); d1 shift from one SP
  N8 dipolarophile reference C=C flipped       make_reference.dipolarophile_check
Writes <scratch>/v8/result.json; exit 0 when all 8 were evaluated (detection itself is judged in the report).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

VAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VAL.parent / "d1_autode_pilot"))
import pilot_common as pc  # noqa: E402
import orca_direct as od  # noqa: E402

TERM = "****ORCA TERMINATED NORMALLY****"
CDS = "SMD CDS free energy correction energy"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="V8")
    a = ap.parse_args()
    cfg = pc.load_config()
    base = Path(cfg["scratch"]) / "v8"; base.mkdir(parents=True, exist_ok=True)
    if (base / "result.json").is_file():
        print("V8: done"); return
    fr, s3 = pc.import_repo_modules(cfg)
    import run_ts, make_reference as mr, assemble as asm, build_sp_inputs as bsi
    import stereo_utils as su
    from rdkit.Chem import rdMolTransforms

    man = pd.read_csv(cfg["val"]["pilot_manifest"], keep_default_na=False).set_index("job_id", drop=False)
    job12 = man.loc["J12"].to_dict()
    coley = pd.read_csv(cfg["d0_csv"])
    smi = dict(zip(coley.rxn_id.astype(int), coley.rxn_smiles))
    labels = {int(r["rxn_id"]): r for r in json.load(open(pc.repo_path(cfg, cfg["labels_all"])))}

    # ---- the replay job to plant defects in (copy, never the pilot scratch itself)
    src = Path(cfg["pilot_scratch"]) / "jobs" / "J12"
    ref12 = base / "J12"
    if not ref12.is_dir():
        (ref12 / "sp").mkdir(parents=True)
        for pat in ("*.json", "*.xyz", "energies.csv"):
            for f in src.glob(pat):
                shutil.copy(f, ref12 / f.name)
        for pat in ("*.inp", "*.out", "input_meta.json"):
            for f in (src / "sp").glob(pat):
                shutil.copy(f, ref12 / "sp" / f.name)
    meta = json.loads((ref12 / "sp" / "input_meta.json").read_text())
    rid_name = f"rxn_{meta['rxn_id']}"

    def sp_copy(name):
        jd = base / name
        if jd.exists():
            shutil.rmtree(jd)
        shutil.copytree(ref12 / "sp", jd / "sp")
        for f in ("ts_result.json", "ref_result.json"):
            shutil.copy(ref12 / f, jd / f)
        return jd

    def audit(jd):
        try:
            return list(asm.d0_audit(cfg, jd, rid_name).get("problems") or [])
        except Exception as e:                                        # noqa: BLE001  (assemble does the same)
            return [f"audit_error:{type(e).__name__}: {e}"]

    base_problems = set(audit(sp_copy("baseline")))
    out = dict(baseline_audit_problems=sorted(base_problems), controls={})

    def record(nid, defect, gate, detected, **detail):
        out["controls"][nid] = dict(defect=defect, gate=gate, detected=bool(detected), **detail)
        print(f"{nid}: detected={bool(detected)}  {defect}")

    def guarded(nid, defect, gate, fn):
        try:
            fn()
        except Exception as e:                                        # noqa: BLE001
            record(nid, defect, gate, False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())

    def n1():
        jd = sp_copy("N1"); spd = jd / "sp"
        (spd / "eda.inp").write_text((spd / "eda.inp").read_text().replace("SMD(water)", "CPCM(water)"))
        errs = bsi.check_method_lines(cfg, {f: (spd / f).read_text() for f in bsi.SP_FILES})
        new = sorted(set(audit(jd)) - base_problems)
        record("N1", "eda.inp SMD(water) -> CPCM(water)", "check_method_lines / audit eda_inp_smd_count",
               bool(errs) or bool(new), check_method_lines=errs, new_audit_problems=new)

    def n2():
        jd = sp_copy("N2"); spd = jd / "sp"
        t = (spd / "eda.inp").read_text()
        t2 = re.sub(r"FRAG1_C\s+0\b", "FRAG1_C 1", t)
        (spd / "eda.inp").write_text(t2)
        syms = re.findall(r"^\s*([A-Z][a-z]?)\(1\)", t2, re.M)
        odd = fr.n_electrons(syms, 1) % 2 == 1
        new = sorted(set(audit(jd)) - base_problems)
        record("N2", "eda.inp FRAG1_C 0 -> 1", "electron parity / audit charge-mult",
               t2 != t and (odd or bool(new)), planted=t2 != t, odd_electrons=odd, new_audit_problems=new)

    def n3():
        rid = 3090
        job = dict(job_id="N3_3090", kind="D0_replay", mapped_smiles=smi[rid], d0_rxn_id=rid)
        jd = base / "N3"; jd.mkdir(exist_ok=True); res = {}
        got = run_ts.load_replay(cfg, job, jd, res)
        try:
            run_ts.analyse(cfg, job, jd, res, got, fr); outcome = "passed all TS gates"
        except run_ts.StageFail as e:
            outcome = str(e)
        record("N3", "D0 rxn 3090 (foreign bond) as a replay", "no_foreign_bond",
               "no_foreign_bond" in outcome, outcome=outcome, foreign=res.get("foreign"))

    def n4():
        pdir = Path(cfg["d0_profiles"]) / "20"
        prod = pc.read_xyz(sorted(pdir.glob("p*.xyz"))[0])
        reacs = [pc.read_xyz(p) for p in sorted(pdir.glob("r*.xyz")) if "_alt" not in p.name]
        mode = np.random.default_rng(int(cfg["val"]["seed"])).normal(size=(len(prod[0]), 3))
        got = dict(ts=prod, reacs=reacs, prod=None, imag=[-400.0], mode=mode)
        jd = base / "N4"; jd.mkdir(exist_ok=True); res = {}
        try:
            run_ts.analyse(cfg, dict(job12, job_id="N4"), jd, res, got, fr); outcome = "passed all TS gates"
        except run_ts.StageFail as e:
            outcome = str(e)
        record("N4", "product geometry + random mode as the TS", "forming_in_range / imag_mode_on_forming_bonds",
               ("forming_in_range" in outcome) or ("imag_mode_on_forming_bonds" in outcome),
               outcome=outcome, formed_d=res.get("formed_d"), share=res.get("imag_mode_forming_share"))

    def n5():
        jd = sp_copy("N5"); spd = jd / "sp"; f = spd / "frag2_rel.out"
        t = f.read_text(errors="replace"); f.write_text(t.replace(TERM, ""))
        runsp = [s for s in asm.OUTS if "ORCA TERMINATED NORMALLY" not in (spd / f"{s}.out").read_text(errors="replace")]
        uni = asm.uniformity(spd)["failed"]
        stat = s3.parse_one_rxn(spd, meta).get("status")
        record("N5", "'ORCA TERMINATED NORMALLY' removed from frag2_rel.out", "run_sp / uniformity",
               TERM in t and bool(runsp) and "frag2_rel" in uni, run_sp_missing=runsp, uniformity_failed=uni,
               stage3_status=stat)

    def n6():
        jd = sp_copy("N6"); spd = jd / "sp"; f = spd / "frag1_dist.out"
        t = f.read_text(errors="replace")
        f.write_text("\n".join(l for l in t.splitlines() if CDS not in l) + "\n")
        uni = asm.uniformity(spd)["failed"]
        new = sorted(set(audit(jd)) - base_problems)
        record("N6", "SMD CDS line removed from frag1_dist.out", "uniformity cds / audit no_smd_cds",
               CDS in t and ("frag1_dist" in uni or any("no_smd_cds" in p for p in new)),
               uniformity_failed=uni, new_audit_problems=new)

    def n7():
        port = (json.loads((ref12 / "ref_result.json").read_text()).get("port_diagnostic") or {})
        rel1 = job12["d0_rel1_file"]
        plain = rel1.replace("_alt", "")
        jd = base / "N7"; jd.mkdir(exist_ok=True)
        big, _ = bsi.repo_builders(cfg)
        syms, xyz = pc.read_xyz(ref12 / f"d0_{plain}")
        inp = jd / "frag1_rel_plain.inp"
        if not inp.is_file():
            inp.write_text(big.write_block(big.frag_header(int(meta["charge1"])), syms, np.asarray(xyz)))
        od.run(cfg, inp)
        lab = json.loads((ref12 / "label.json").read_text())
        e_plain = s3.read_fspe(inp.with_suffix(".out"), 1)
        d1_plain = (lab["e_frag1_dist_eh"] - e_plain) * pc.EH_TO_KCAL
        record("N7", f"plain {plain} instead of {rel1} (rxn 20)", "port diagnostic use_alt (A4)",
               port.get("use_alt") is True and "_alt" in rel1, port_use_alt=port.get("use_alt"),
               d1_alt=lab["d1_kcal"], d1_plain=d1_plain, d1_shift=d1_plain - lab["d1_kcal"])

    def n8():
        rng = random.Random(int(cfg["val"]["seed"]))
        cand = sorted(r for r, L in labels.items() if L["status"] == "ok" and L["n_atoms"] <= 35)
        rng.shuffle(cand)
        tried = []
        for rid in cand[:300]:
            job = dict(job_id=f"N8_{rid}", kind="D0_replay", mapped_smiles=smi[rid], d0_rxn_id=rid)
            jd = base / "N8" / str(rid); jd.mkdir(parents=True, exist_ok=True); res = {}
            try:
                got = run_ts.load_replay(cfg, job, jd, res)
                run_ts.analyse(cfg, job, jd, res, got, fr)
                ref = pc.read_xyz(jd / f"d0_{labels[rid]['rel2_file']}")
                c0 = mr.dipolarophile_check(fr, jd, res, job, ref)
                tried.append((rid, c0.get("checked"), c0.get("why")))
                if not (c0.get("checked") and c0.get("ok")):
                    continue
                flipped = flip_reacting_bond(fr, su, rdMolTransforms, job["mapped_smiles"], ref,
                                             int(res["charge_dipolarophile"]))
                c1 = mr.dipolarophile_check(fr, jd, res, job, flipped)
            except Exception:                                         # noqa: BLE001  (ring C=C, clash, ...)
                continue
            record("N8", f"dipolarophile reference C=C flipped (rxn {rid})", "dipolarophile_check",
                   c1.get("ok") is False, rxn_id=rid, original=c0, flipped=c1, n_candidates_tried=len(tried))
            return
        record("N8", "dipolarophile reference C=C flipped", "dipolarophile_check", False,
               error="no candidate with checked=True that could be flipped", tried=tried[:20])

    for nid, fn, d, g in (("N1", n1, "SMD->CPCM", "check_method_lines"), ("N2", n2, "FRAG1_C", "parity/audit"),
                          ("N3", n3, "rxn 3090", "no_foreign_bond"), ("N4", n4, "product as TS", "forming gates"),
                          ("N5", n5, "no TERMINATED", "run_sp/uniformity"), ("N6", n6, "no CDS", "uniformity/audit"),
                          ("N7", n7, "plain ref", "port use_alt"), ("N8", n8, "flipped C=C", "dipolarophile_check")):
        guarded(nid, d, g, fn)
    out["n_detected"] = sum(c["detected"] for c in out["controls"].values())
    out["n_controls"] = len(out["controls"])
    (base / "result.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"V8: {out['n_detected']}/{out['n_controls']} detected")


def flip_reacting_bond(fr, su, tf, mapped, ref, q):
    """Rotate the dipolarophile reference 180 deg about its reacting C=C (no re-optimisation)."""
    rmols, formed, _ = fr.smiles_reactants_and_formed(mapped)
    dmol = rmols[1 - fr.dipole_smiles_index(mapped)[0]]
    maps = {a.GetAtomMapNum() for a in dmol.GetAtoms()}
    react = sorted({m for pair in formed for m in pair if m in maps})
    smi = su.Chem.MolToSmiles(dmol)
    m, t2g, tmpl = su.template_match(su.xyz_block_from_atoms(*ref), smi, q)
    idx = {a.GetAtomMapNum(): a.GetIdx() for a in tmpl.GetAtoms() if a.GetAtomMapNum()}
    u, v = idx[react[0]], idx[react[1]]
    nu = [n.GetIdx() for n in tmpl.GetAtomWithIdx(u).GetNeighbors() if n.GetIdx() != v][0]
    nv = [n.GetIdx() for n in tmpl.GetAtomWithIdx(v).GetNeighbors() if n.GetIdx() != u][0]
    conf = m.GetConformer()
    g = [t2g[i] for i in (nu, u, v, nv)]
    tf.SetDihedralDeg(conf, *g, tf.GetDihedralDeg(conf, *g) + 180.0)
    return ref[0], conf.GetPositions()


if __name__ == "__main__":
    main()
