#!/usr/bin/env python3
"""D1-G2R Phase R (docs/specs/D1G2R.md 5): rerun only the reference stage of the 20 repair reactions with two read-only
wrappers around the D1 code (results/R_diagnosis.md). D1 run env (python 3.9, autodE 1.4.5, rdkit 2024.3.2) via
jobs/cpu_g2r.slurm (D1G2_SCRATCH=$G2R, so g2_autode.g2_configure makes its ts_templates folder in $G2R, not in $G2).

    python analysis/g2r/ref_repair.py run <n_parallel> <cores>    # every repair job_id ($G2R/x/repair_ids.csv), one
                                                                  # subprocess each, then `report`
    python analysis/g2r/ref_repair.py one <job_id> <cores>        # one reaction (what `run` starts)
    python analysis/g2r/ref_repair.py report                      # results/R_status.csv, R_report.json,
                                                                  # $G2R/x/repaired_ok_ids.csv, $G2R/x/lf_ids.csv

Wrappers (spec 5-2: only on the exception of the original, else the original's result unchanged):
  * stereo_utils.template_match: on rdkit AtomValenceException only, the same steps with
    AllChem.AssignBondOrdersFromTemplate (rdkit AllChem.py:394-433) redone verbatim except the final SanitizeMol, which
    runs without its valence check (UpdatePropertyCache(strict=False), then SanitizeMol(SANITIZE_ALL ^
    SANITIZE_PROPERTIES)); the extra perceived bonds are logged. Same return value (mol in geometry order with the xyz
    conformer, {template index: geometry index}, template).
  * make_reference.coley_port: on KeyError('dipole_mapped_smiles' | 'dipole_backbone_maps') only (raised at
    make_reference.py:158 before any xtb call or chdir), the two values are computed exactly as run_ts.py:238 does,
    `_, bb, dip_smi = su.dipole_backbone(job['mapped_smiles'])` (manifest row), and the original is called once more
    with them added to an in-memory copy of ts_result (ts_result.json is not rewritten).
The Coley rule, xtb commands and levels, conformer count and every threshold stay as in the D1 / G2 code.

Per reaction (spec 5-4), in $G2R/jobs/<job_id>/: .done_ts and .done_stereo required; the G2 failure is kept as
ref_result.orig.json (renamed ref_result.json) and $G2R/x/R/<job_id>.pre.json (.fail_ref text, sha256s); .fail_ref is
removed; then g2_autode.stage_ref(cfg, job, jd) with cfg = analysis/d1_g2/config_g2.yaml (scratch -> $G2R; stage_ref
does not use it). One finished attempt per reaction (recorded decision; autodE conformer sampling is unseeded): a
reaction with ref_result.orig.json and a new .done_ref or .fail_ref is never rerun; only a crashed attempt (neither
marker) is rerun, after removing its jd/ref work folder.
Exit: run/one 0 done, 2 not finished (resubmit), 3 STOP; report 0 all 20 finished, 2 not finished, 3 STOP.
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g2r_common as C  # noqa: E402

N_REPAIR = 20
KEY_FIELDS = ("dipole_mapped_smiles", "dipole_backbone_maps")
R_DIR = C.G2R_X / "R"
BYPASS_LOG: list = []


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def load_d1():
    """analysis/d1_g2/g2_autode (imports pilot_common, run_ts, make_reference, stereo_utils of the D1 run code)."""
    scr = os.environ.get("D1G2_SCRATCH")
    if not scr or Path(scr).resolve() != C.G2R.resolve():
        stop(f"D1G2_SCRATCH={scr!r} must be {C.G2R} (g2_configure mkdirs <D1G2_SCRATCH>/ts_templates)")
    sys.path.insert(0, str(C.REPO / "analysis" / "d1_g2"))
    import g2_autode as GA
    return GA


# ================================================================ wrappers
def template_match_bypass(su, xyz_block, template_smiles, charge=0):
    """stereo_utils.template_match (stereo_utils.py:56-64) with AssignBondOrdersFromTemplate (AllChem.py:394-433) redone
    verbatim up to its final SanitizeMol, which here skips only the valence check. -> (m, t2g, tmpl, info)."""
    from rdkit import Chem
    from rdkit.Chem import rdchem
    tmpl = su._clean_template(template_smiles)
    geo = su._mol_from_xyz(xyz_block, charge)
    refmol, mol = tmpl, geo
    refmol2 = rdchem.Mol(refmol)
    mol2 = rdchem.Mol(mol)
    info = dict(direct_match=False, sanitize="all_but_properties", extra_bonds=[])
    matching = mol2.GetSubstructMatch(refmol2)
    if matching:                                     # AllChem returns mol2 unsanitised here (it cannot have raised)
        info["direct_match"] = True
    else:
        for b in mol2.GetBonds():
            if b.GetBondType() != rdchem.BondType.SINGLE:
                b.SetBondType(rdchem.BondType.SINGLE)
                b.SetIsAromatic(False)
        for b in refmol2.GetBonds():
            b.SetBondType(rdchem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)
        matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
        if not matching:
            raise ValueError("No matching found")
        matching = matching[0]
        for b in refmol.GetBonds():
            atom1 = matching[b.GetBeginAtomIdx()]
            atom2 = matching[b.GetEndAtomIdx()]
            b2 = mol2.GetBondBetweenAtoms(atom1, atom2)
            b2.SetBondType(b.GetBondType())
            b2.SetIsAromatic(b.GetIsAromatic())
        for a in refmol.GetAtoms():
            a2 = mol2.GetAtomWithIdx(matching[a.GetIdx()])
            a2.SetHybridization(a.GetHybridization())
            a2.SetIsAromatic(a.GetIsAromatic())
            a2.SetNumExplicitHs(a.GetNumExplicitHs())
            a2.SetFormalCharge(a.GetFormalCharge())
        mol2.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(mol2, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception as e:                                                # noqa: BLE001
            info["sanitize"] = f"none ({type(e).__name__}: {e})"
            mol2 = rdchem.Mol(mol2)
            mol2.UpdatePropertyCache(strict=False)
        t_bonds = {tuple(sorted((matching[b.GetBeginAtomIdx()], matching[b.GetEndAtomIdx()])))
                   for b in refmol.GetBonds()}
        pos = mol2.GetConformer().GetPositions()
        for b in mol2.GetBonds():
            i, j = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            if (i, j) not in t_bonds:
                info["extra_bonds"].append(dict(
                    i=i, j=j, el=f"{mol2.GetAtomWithIdx(i).GetSymbol()}-{mol2.GetAtomWithIdx(j).GetSymbol()}",
                    map=[refmol.GetAtomWithIdx(matching.index(i)).GetAtomMapNum() if i in matching else None,
                         refmol.GetAtomWithIdx(matching.index(j)).GetAtomMapNum() if j in matching else None],
                    d_A=round(float(((pos[i] - pos[j]) ** 2).sum() ** 0.5), 4)))
    m = mol2
    match = m.GetSubstructMatch(tmpl)
    if not match:
        raise ValueError("geometry does not match the template graph")
    return m, dict(enumerate(match)), tmpl, info


def install(GA, log=BYPASS_LOG):
    """su.template_match and MR.coley_port -> the wrappers (module attributes, looked up at call time:
    stereo_utils.py:153 / make_reference.py:192, 244, 265, 320 and g2_autode.py:186). Returns the originals."""
    from rdkit.Chem.rdchem import AtomValenceException
    MR = GA.MR
    su = MR.su
    orig_tm, orig_cp = su.template_match, MR.coley_port

    def template_match(xyz_block, template_smiles, charge=0):
        try:
            return orig_tm(xyz_block, template_smiles, charge)
        except AtomValenceException as e:
            m, t2g, tmpl, info = template_match_bypass(su, xyz_block, template_smiles, charge)
            log.append(dict(wrapper="template_match", exception=f"{type(e).__name__}: {e}", **info))
            return m, t2g, tmpl

    def coley_port(cfg, jd, work, ts_res, *rest, **kw):
        try:
            return orig_cp(cfg, jd, work, ts_res, *rest, **kw)
        except KeyError as e:
            if not e.args or e.args[0] not in KEY_FIELDS or all(k in ts_res for k in KEY_FIELDS):
                raise
            job = GA.pc.load_job(Path(jd).name, GA.G.MANIFEST)
            _, bb, dip_smi = su.dipole_backbone(job["mapped_smiles"])
            log.append(dict(wrapper="coley_port", exception=f"KeyError: {e}", dipole_mapped_smiles=dip_smi,
                            dipole_backbone_maps=list(bb)))
            return orig_cp(cfg, jd, work, dict(ts_res, dipole_mapped_smiles=dip_smi, dipole_backbone_maps=list(bb)),
                           *rest, **kw)

    template_match.original, coley_port.original = orig_tm, orig_cp
    su.template_match, MR.coley_port = template_match, coley_port
    return orig_tm, orig_cp


# ================================================================ one reaction
def g_status():
    with open(C.G_STATUS, newline="") as fh:
        return {r["job_id"]: r for r in csv.DictReader(fh)}


def record(T, jd, pre, extra):
    rr = C.read_json(jd / "ref_result.json") if (jd / "ref_result.json").is_file() else {}
    done, fail = (jd / ".done_ref").is_file(), (jd / ".fail_ref").is_file()
    port = rr.get("port") or {}
    return dict(job_id=T, rxn_id=pre.get("rxn_id"), orig_marker=pre.get("orig_marker"),
                orig_reason=pre.get("orig_fail_ref_text"),
                new_status="ok" if done else ("fail" if fail else "not_finished"),
                new_reason=(jd / ".fail_ref").read_text() if fail else "", alt_used=rr.get("alt_used"),
                wall_s=rr.get("wall_s"), port_abort=port.get("abort"), port_cyclization=port.get("cyclization"),
                port_reason=port.get("reason"), n_xtb_calls=len(rr.get("xtb_calls") or []),
                ref_result_sha256=C.sha256(jd / "ref_result.json") if rr else None, **extra)


def cmd_one(T, cores):
    cores = int(cores)
    os.environ["SLURM_CPUS_PER_TASK"] = str(cores)                  # pilot_common.n_cores() -> autodE n_cores (cmd_one)
    os.environ["OMP_NUM_THREADS"] = str(cores)
    jd, out, pre_p = C.G2R_JOBS / T, R_DIR / f"{T}.json", R_DIR / f"{T}.pre.json"
    orig, fail, done = jd / "ref_result.orig.json", jd / ".fail_ref", jd / ".done_ref"
    if not jd.is_dir():
        stop(f"{jd} missing")
    if done.is_file() or (orig.is_file() and fail.is_file()):            # finished attempt: never rerun
        if not out.is_file():
            pre = C.read_json(pre_p) if pre_p.is_file() else {}
            C.write_json(out, record(T, jd, pre, dict(cores=cores, bypass=None, note="record rebuilt after the run")))
        print(f"{T}: finished ({'done' if done.is_file() else 'fail'}), skip", flush=True)
        return 0
    if not ((jd / ".done_ts").is_file() and (jd / ".done_stereo").is_file()):
        C.write_json(out, dict(job_id=T, new_status="stage_not_done", note="spec 5-4-2: .done_ts/.done_stereo missing"))
        stop(f"{T}: .done_ts / .done_stereo missing")
    st = g_status()[T]
    if not orig.is_file():
        if not pre_p.is_file():
            if not fail.is_file():
                stop(f"{T}: neither .done_ref, .fail_ref nor ref_result.orig.json")
            C.write_json(pre_p, dict(job_id=T, rxn_id=int(st["rxn_id"]), orig_marker=st["marker"],
                                     orig_fail_ref_text=fail.read_text(),
                                     orig_ref_result_sha256=C.sha256(jd / "ref_result.json"),
                                     orig_fail_ref_sha256=C.sha256(fail), written=time.strftime("%Y-%m-%d %H:%M:%S")))
        if fail.is_file():
            fail.unlink()
        os.replace(jd / "ref_result.json", orig)
        rerun = False
    else:
        rerun = True                                                     # crashed attempt: fresh work folder
        shutil.rmtree(jd / "ref", ignore_errors=True)
    pre = C.read_json(pre_p)
    GA = load_d1()
    install(GA)
    import yaml
    cfg = yaml.safe_load(GA.CONFIG.read_text())
    cfg["scratch"] = str(C.G2R)                                          # not used by stage_ref (jd is passed)
    job = GA.pc.load_job(T, GA.G.MANIFEST)
    t0 = time.time()
    try:
        ok = GA.stage_ref(cfg, job, jd)
    finally:
        os.chdir(HERE)
    wall = time.time() - t0
    rec = record(T, jd, pre, dict(cores=cores, attempt_wall_s=wall, core_h=wall * cores / 3600, rerun_after_crash=rerun,
                                  bypass=list(BYPASS_LOG), slurm_job=os.environ.get("SLURM_JOB_ID")))
    C.write_json(out, rec)
    print(f"{T}: {rec['new_status']} {rec['new_reason'][:120]!r} bypass={[b['wrapper'] for b in BYPASS_LOG]} "
          f"wall {wall:.0f} s", flush=True)
    return 0 if ok in (True, False) else 2


# ================================================================ all 20
def repair_ids():
    with open(C.G2R_X / "repair_ids.csv", newline="") as fh:
        ids = [r["job_id"] for r in csv.DictReader(fh)]
    if len(ids) != N_REPAIR:
        stop(f"repair_ids.csv has {len(ids)} rows, not {N_REPAIR}")
    return ids


def cmd_run(n_par, cores):
    ids = repair_ids()
    (C.G2R_LOGS / "R").mkdir(parents=True, exist_ok=True)
    print(f"[ref_repair run] {len(ids)} reactions, {n_par} at a time x {cores} cores", flush=True)

    def one(T):
        with open(C.G2R_LOGS / "R" / f"{T}.log", "a") as f:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", T, str(cores)],
                               stdout=f, stderr=subprocess.STDOUT)
        return T, r.returncode
    rcs = {}
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for f in as_completed([ex.submit(one, T) for T in ids]):
            T, rc = f.result()
            rcs[T] = rc
            print(f"  {T}: rc {rc}", flush=True)
    rc_rep = cmd_report()
    return 3 if any(rc == 3 for rc in rcs.values()) else (rc_rep if rc_rep else (2 if any(rcs.values()) else 0))


def ids_csv(path, ids):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["job_id"])
    w.writerows([j] for j in ids)
    C.write_atomic(path, buf.getvalue())
    return C.sha256(path)


COLS = ["job_id", "rxn_id", "orig_marker", "orig_reason", "new_status", "new_reason", "alt_used", "core_h", "wall_s",
        "cores", "valence_bypass_calls", "keyerror_bypass", "extra_bonds", "port_abort", "n_xtb_calls"]


def cmd_report():
    ids = repair_ids()
    rows, missing = [], []
    for T in ids:
        p = R_DIR / f"{T}.json"
        if not p.is_file():
            missing.append(T)
            continue
        r = C.read_json(p)
        bp = r.get("bypass") or []
        vb = [b for b in bp if b.get("wrapper") == "template_match"]
        eb = sorted({f"{e['el']} {e['map'][0]}-{e['map'][1]} {e['d_A']:.3f}" for b in vb for e in b.get("extra_bonds", [])})
        rows.append(dict(job_id=T, rxn_id=r.get("rxn_id"), orig_marker=r.get("orig_marker"),
                         orig_reason=(r.get("orig_reason") or "").strip(), new_status=r.get("new_status"),
                         new_reason=(r.get("new_reason") or "").strip(), alt_used=r.get("alt_used"),
                         core_h=r.get("core_h"), wall_s=r.get("wall_s"), cores=r.get("cores"),
                         valence_bypass_calls=len(vb) if r.get("bypass") is not None else "",
                         keyerror_bypass=any(b.get("wrapper") == "coley_port" for b in bp),
                         extra_bonds="; ".join(eb), port_abort=r.get("port_abort") or "",
                         n_xtb_calls=r.get("n_xtb_calls")))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
    C.write_atomic(C.RES / "R_status.csv", buf.getvalue())
    ok_ids = sorted(r["job_id"] for r in rows if r["new_status"] == "ok")
    finished = not missing and all(r["new_status"] in ("ok", "fail") for r in rows)
    with open(C.G2R_X / "reuse_ids.csv", newline="") as fh:
        reuse = [r["job_id"] for r in csv.DictReader(fh)]
    out = {}
    if finished:
        out["repaired_ok_ids.csv"] = ids_csv(C.G2R_X / "repaired_ok_ids.csv", ok_ids)
        out["lf_ids.csv"] = ids_csv(C.G2R_X / "lf_ids.csv", sorted(set(reuse) | set(ok_ids)))
    reasons = {}
    for r in rows:
        if r["new_status"] == "fail":
            k = r["new_reason"].split(":")[0]
            reasons[k] = reasons.get(k, 0) + 1
    rep = dict(spec="docs/specs/D1G2R.md 5-4", written=time.strftime("%Y-%m-%d %H:%M:%S"), n_repair=len(ids),
               n_records=len(rows), missing=missing, finished=finished, k_ok=len(ok_ids), ok_ids=ok_ids,
               n_fail=sum(r["new_status"] == "fail" for r in rows), fail_reasons=reasons,
               n_stage_not_done=sum(r["new_status"] == "stage_not_done" for r in rows),
               n_valence_bypass_reactions=sum(bool(r["valence_bypass_calls"]) for r in rows),
               n_keyerror_bypass=sum(bool(r["keyerror_bypass"]) for r in rows),
               core_h_total=sum(float(r["core_h"] or 0) for r in rows), n_reuse=len(reuse),
               n_lf_ids=len(set(reuse) | set(ok_ids)) if finished else None, outputs=out,
               all_pass=finished and not any(r["new_status"] == "stage_not_done" for r in rows))
    C.write_json(C.RES / "R_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("ok_ids",)}, indent=1, default=str), flush=True)
    if any(r["new_status"] == "stage_not_done" for r in rows):
        return 3
    return 0 if finished else 2


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["one"] and len(a) == 3:
        sys.exit(cmd_one(a[1], a[2]))
    if a[:1] == ["run"] and len(a) == 3:
        sys.exit(cmd_run(a[1], a[2]))
    if a == ["report"]:
        sys.exit(cmd_report())
    raise SystemExit(__doc__)
