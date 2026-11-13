#!/usr/bin/env python3
"""D1-G2R 5-3: wrapper tests R-T1 / R-T2 (no xtb, no ORCA) -> results/R_tests.json. D1 run env, jobs/cpu_g2r.slurm
(D1G2_SCRATCH=$G2R). Reads the copied job folders in $G2R/jobs only (after Phase C). Exit 0 all pass, 3 = STOP.

R-T1  no exception -> the original is called once and its result is returned unchanged:
      (a) mock originals for both wrappers (identity of the returned object, call count, arguments);
      (b) real code: stereo_utils.config_tags on stored TS cuts that do not raise and have tracked dihedrals (so
          template_match is reached: T09757, T11023 and the first 3 reuse reactions with a non-empty
          ts_dipole_dihedrals), wrapped vs original give equal tags / dihedrals / quadruples, no bypass is logged.
R-T2  one reaction per cause group, the bypass path at function level:
      A  (AtomValenceException in template_match, 19 + T10518's TS stage): the run_ts.py:240 call on the stored T10518
         TS cut (the only stored geometry that reproduces the error; the 19 alt conformers were not kept): original
         raises AtomValenceException, wrapped returns tags with the extra perceived bond logged;
         A2 group member T09757 (supplementary): its stored TS dipole cut with the ylide CH2 group (map 1 carbon and
         its two H, rigid) moved along the line to the ortho acetyl carbon (map 11) in 0.05 A steps until the original
         raises AtomValenceException (the contact the alt conformer made); then as A. Not judged if no step reproduces.
      B  (KeyError in coley_port, T10518): the unwrapped make_reference.coley_port with T10518's ts_result.json raises
         KeyError('dipole_mapped_smiles') before any xtb call or folder; the wrapper (original replaced by a recorder
         that raises like make_reference.py:158 when a key is missing) calls it twice, the second time with the values
         of su.dipole_backbone(manifest mapped_smiles) (= run_ts.py:238).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g2r_common as C  # noqa: E402
import ref_repair as RR  # noqa: E402


def ts_cut(GA, T):
    jd = C.G2R_JOBS / T
    res = json.loads((jd / "ts_result.json").read_text())
    syms, xyz = GA.pc.read_xyz(jd / "ts.xyz")
    A = sorted(res["A_idx"])
    job = GA.pc.load_job(T, GA.G.MANIFEST)
    _, bb, dip_smi = GA.MR.su.dipole_backbone(job["mapped_smiles"])
    return [syms[i] for i in A], np.asarray(xyz)[A], dip_smi, bb, int(res["charge_dipole"]), res, job


def same(a, b):
    ta, va, qa = a
    tb, vb, qb = b
    return ta == tb and qa == qb and len(va) == len(vb) and all(
        (np.isnan(x) and np.isnan(y)) or x == y for x, y in zip(va, vb))


def main():
    GA = RR.load_d1()
    su, MR = GA.MR.su, GA.MR
    from rdkit.Chem.rdchem import AtomValenceException
    orig_tm, orig_cp = su.template_match, MR.coley_port
    out = {}

    def unwrap():
        su.template_match, MR.coley_port = orig_tm, orig_cp

    def reinstall(log):
        unwrap()
        RR.install(GA, log)

    # ---------------- R-T1 (a) mocks
    calls = []
    sentinel_tm, sentinel_cp = object(), object()

    def mock_tm(*a):
        calls.append(("tm", a))
        return sentinel_tm

    def mock_cp(*a, **k):
        calls.append(("cp", a, k))
        return sentinel_cp
    fake = SimpleNamespace(MR=SimpleNamespace(su=SimpleNamespace(template_match=mock_tm), coley_port=mock_cp),
                           pc=GA.pc, G=GA.G)
    log = []
    RR.install(fake, log)
    r_tm = fake.MR.su.template_match("XYZ", "SMI", 0)
    ts_res = dict(dipole_mapped_smiles="x", dipole_backbone_maps=[1, 2, 3], charge_dipole=0)
    r_cp = fake.MR.coley_port("cfg", "/nonexistent/T00000", "work", ts_res, "fr", "orig", None, False)
    t1a = dict(template_match=dict(returned_same_object=r_tm is sentinel_tm, n_calls=sum(c[0] == "tm" for c in calls),
                                   args_passed=[c[1] for c in calls if c[0] == "tm"] == [("XYZ", "SMI", 0)]),
               coley_port=dict(returned_same_object=r_cp is sentinel_cp, n_calls=sum(c[0] == "cp" for c in calls),
                               ts_res_passed_unchanged=[c[1][3] for c in calls if c[0] == "cp"] == [ts_res]
                               and [c[1][3] for c in calls if c[0] == "cp"][0] is ts_res),
               bypass_logged=len(log))
    t1a["ok"] = (t1a["template_match"]["returned_same_object"] and t1a["template_match"]["n_calls"] == 1
                 and t1a["template_match"]["args_passed"] and t1a["coley_port"]["returned_same_object"]
                 and t1a["coley_port"]["n_calls"] == 1 and t1a["coley_port"]["ts_res_passed_unchanged"] and not log)

    # ---------------- R-T1 (b) real code, no exception
    real_log = []
    t1b = {}
    import csv
    with open(C.RES / "X_targets.csv", newline="") as fh:
        reuse = [r["job_id"] for r in csv.DictReader(fh) if r["group"] == "reuse"]
    picked = []
    for T in reuse:
        if json.loads((C.G2R_JOBS / T / "ts_result.json").read_text()).get("ts_dipole_dihedrals"):
            picked.append(T)
        if len(picked) == 3:
            break
    for T in ["T09757", "T11023"] + picked:
        try:
            syms, xyz, dip_smi, bb, q, _, _ = ts_cut(GA, T)
            blk = su.xyz_block_from_atoms(syms, xyz)
            n0 = len(real_log)
            unwrap()
            a = su.config_tags(blk, dip_smi, bb, q)
            reinstall(real_log)
            b = su.config_tags(blk, dip_smi, bb, q)
            t1b[T] = dict(tags=a[0], dihedrals=a[1], n_quads=len(a[2]), equal=same(a, b),
                          bypass_logged=len(real_log) - n0, ok=same(a, b) and len(real_log) == n0 and len(a[2]) > 0)
        except Exception as e:                                                    # noqa: BLE001
            t1b[T] = dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    out["R-T1"] = dict(mock=t1a, real_no_exception=t1b, ok=t1a["ok"] and all(v["ok"] for v in t1b.values()))

    # ---------------- R-T2 A: T10518 TS cut (run_ts.py:240)
    def cause_a(syms, xyz, dip_smi, bb, q):
        blk = su.xyz_block_from_atoms(syms, xyz)
        unwrap()
        try:
            su.config_tags(blk, dip_smi, bb, q)
            unwrapped = "no exception"
        except AtomValenceException as e:
            unwrapped = f"AtomValenceException: {e}"
        except Exception as e:                                                    # noqa: BLE001
            unwrapped = f"{type(e).__name__}: {e}"
        log_a = []
        reinstall(log_a)
        try:
            tag, dih, quads = su.config_tags(blk, dip_smi, bb, q)
            wrapped = dict(tags=tag, dihedrals=dih, quads=quads, error=None)
        except Exception as e:                                                    # noqa: BLE001
            wrapped = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        return dict(unwrapped=unwrapped, wrapped=wrapped, bypass=log_a,
                    ok=unwrapped.startswith("AtomValenceException") and wrapped["error"] is None and len(log_a) >= 1)
    try:
        syms, xyz, dip_smi, bb, q, _, _ = ts_cut(GA, "T10518")
        a1 = cause_a(syms, xyz, dip_smi, bb, q)
        a1.update(job_id="T10518", geometry="stored ts.xyz, A_idx cut (run_ts.py:240)", backbone=list(bb))
    except Exception as e:                                                        # noqa: BLE001
        a1 = dict(job_id="T10518", ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    try:
        syms, xyz, dip_smi, bb, q, _, _ = ts_cut(GA, "T09757")
        unwrap()
        _, t2g, tmpl = orig_tm(su.xyz_block_from_atoms(syms, xyz), dip_smi, q)
        tidx = {tmpl.GetAtomWithIdx(i).GetAtomMapNum(): i for i in t2g if tmpl.GetAtomWithIdx(i).GetAtomMapNum()}
        t1, t11 = tidx[1], tidx[11]
        i1, i11 = t2g[t1], t2g[t11]
        grp = [i1] + [t2g[n.GetIdx()] for n in tmpl.GetAtomWithIdx(t1).GetNeighbors() if n.GetAtomicNum() == 1]
        a11 = tmpl.GetAtomWithIdx(t11)
        d0 = float(np.linalg.norm(xyz[i1] - xyz[i11]))
        u = (np.asarray(xyz[i1], float) - np.asarray(xyz[i11], float)) / d0
        scan, x_hit, d_hit = [], None, None
        for d in np.round(np.arange(np.floor(d0 / 0.05) * 0.05, 1.399, -0.05), 2):
            x2 = np.array(xyz, dtype=float)
            x2[grp] += (x2[i11] + d * u) - x2[i1]                    # the CH2 group moved as one rigid unit
            unwrap()
            try:
                su.config_tags(su.xyz_block_from_atoms(syms, x2), dip_smi, bb, q)
                res_ = "no exception"
            except Exception as e:                                                # noqa: BLE001
                res_ = f"{type(e).__name__}: {e}"
            scan.append([float(d), res_[:100]])
            if res_.startswith("AtomValenceException"):
                x_hit, d_hit = x2, float(d)
                break
        geom = (f"stored TS cut; CH2 group (map 1 C, cut index {i1}, + its H {grp[1:]}) moved rigidly along the line to "
                f"map 11 (cut index {i11}, {a11.GetSymbol()}, =O {any(n.GetSymbol() == 'O' for n in a11.GetNeighbors())}) "
                f"from {d0:.3f} A in 0.05 A steps until the original raises AtomValenceException")
        if x_hit is None:
            a2 = dict(job_id="T09757", geometry=geom, scan=scan, reproduced=False, ok=None,
                      note="no scanned geometry reproduced the exception: supplementary case not judged")
        else:
            a2 = cause_a(syms, x_hit, dip_smi, bb, q)
            a2.update(job_id="T09757", geometry=geom, d_A=d_hit, scan=scan, reproduced=True)
    except Exception as e:                                                        # noqa: BLE001
        a2 = dict(job_id="T09757", ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())

    # ---------------- R-T2 B: T10518 coley_port KeyError
    try:
        import yaml
        cfg = yaml.safe_load(GA.CONFIG.read_text())
        cfg["scratch"] = str(C.G2R)
        syms, xyz, dip_smi, bb, q, ts_res, job = ts_cut(GA, "T10518")
        jd = C.G2R_JOBS / "T10518"
        fr, _ = GA.pc.import_repo_modules(cfg)
        orig = GA.pc.read_xyz(jd / "r_dipole_autode.xyz")
        work = Path(tempfile.mkdtemp(prefix="rt2b_", dir=str(C.G2R_X))) / "ref"
        cwd = os.getcwd()
        try:
            orig_cp(cfg, jd, work, ts_res, fr, orig, None, False)
            unwrapped = "no exception"
        except KeyError as e:
            unwrapped = f"KeyError: {e}"
        work_created, cwd_changed = work.exists(), os.getcwd() != cwd
        os.chdir(cwd)
        seen = []
        KEYS = RR.KEY_FIELDS

        def recorder(cfg_, jd_, work_, ts_res_, *rest, **kw):            # make_reference.py:157-158 behaviour
            seen.append({k: ts_res_.get(k) for k in KEYS})
            int(ts_res_["charge_dipole"])
            ts_res_["dipole_mapped_smiles"], tuple(ts_res_["dipole_backbone_maps"])
            return "recorder_result"
        fake_su = SimpleNamespace(template_match=orig_tm, dipole_backbone=su.dipole_backbone)
        fake_b = SimpleNamespace(MR=SimpleNamespace(su=fake_su, coley_port=recorder), pc=GA.pc, G=GA.G)
        log_b = []
        RR.install(fake_b, log_b)
        got = fake_b.MR.coley_port(cfg, jd, work, ts_res, fr, orig, None, False)
        _, bb2, dip2 = su.dipole_backbone(job["mapped_smiles"])
        b = dict(job_id="T10518", unwrapped=unwrapped, unwrapped_made_work_folder=work_created,
                 unwrapped_changed_cwd=cwd_changed, recorder_calls=seen, result=got, bypass=log_b,
                 backbone=list(bb2), ts_result_has_keys=[k in ts_res for k in KEYS],
                 ok=(unwrapped == "KeyError: 'dipole_mapped_smiles'" and not work_created and not cwd_changed
                     and got == "recorder_result" and len(seen) == 2 and seen[0] == {k: None for k in KEYS}
                     and seen[1] == dict(dipole_mapped_smiles=dip2, dipole_backbone_maps=list(bb2))
                     and len(log_b) == 1 and not any(k in ts_res for k in KEYS)))
        work.parent.rmdir() if work.parent.is_dir() and not any(work.parent.iterdir()) else None
    except Exception as e:                                                        # noqa: BLE001
        b = dict(job_id="T10518", ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    out["R-T2"] = dict(A=a1, A2=a2, B=b, ok=bool(a1.get("ok") and a2.get("ok") is not False and b.get("ok")),
                       rule="A (T10518 stored cut) and B must pass; A2 (constructed geometry) must pass when it "
                            "reproduces the exception")
    unwrap()
    rep = dict(spec="docs/specs/D1G2R.md 5-3", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), tests=out, all_pass=all(v["ok"] for v in out.values()))
    C.write_json(C.RES / "R_tests.json", rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], **{k: v["ok"] for k, v in out.items()},
                          A=a1.get("ok"), A2=a2.get("ok"), B=b.get("ok")), indent=1))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
