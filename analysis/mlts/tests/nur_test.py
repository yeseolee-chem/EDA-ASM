#!/usr/bin/env python3
"""tests/nur_test.py — DEPLOY-V2 (MLTS) inv_nurion.py checks (sandbox / compute node only; writes only $MLT/tmp/nurtest).

  unit                       path normalisation, directory classes, kinds, registry, tag rules on synthetic inputs
  io <scan root> <tar>       timing of the extxyz.gz write / read (mlt_common.read_extxyz) on that tar's output,
                             tag round trip vs report, finetune-style dedupe key, npz load
  idem <scan root> <tar>     a second scan of a done tar is a no-op (exit 0, outputs untouched: mtime + sha256)
Prints one JSON object; exit 0 = all checks passed, 3 = a check failed."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import inv_nurion as N  # noqa: E402
import mlt_common as C  # noqa: E402

TMP = C.MLT / "tmp" / "nurtest"


def check(res, name, ok, **kw):
    res[name] = dict(ok=bool(ok), **kw)
    return ok


def cmd_unit():
    res = {}
    rels = ["ts/T00001/transition_states/TS_ts_guess_o32HrQ_template_3-7_5-6_optts_orca.out",
            "ts/T00001/transition_states/o32HrQ_template_3-7_5-6_constrained_opt_orca.out",
            "ts/T00001/transition_states/initial_path/path_opt-1_orca.out",
            "ts/T00001/transition_states/neb/3/3_7_orca.out",
            "ts/T00001/transition_states/truncated/TS__vudpT_hl_ad_1-2_3-4_optts_orca.out",
            "ts/T00001/transition_states/TS_ypzsZ__hl_ad_1-2_3-4_backwards_opt_orca.out",
            "ts/T00001/reactants_and_products/r1_C3H3NO2_opt_orca.out",
            "ts/T00001/reactants_and_products/p0_C5H7NO2_opt_orca.out",
            "ts/T00001/thermal/r0_C2H4_hess_orca.out", "ts/T00001/single_points/r0_C2H4_sp_orca.out",
            "ref/dipole_alt_opt_orca.out", "ref/port/dipole_alt_opt_orca.out", "sp/eda.out", "ts/autode.log"]
    nm, roots = N._norm_map(rels)
    check(res, "norm_root", roots == ["T00001"] and nm[rels[0]].startswith("ts/transition_states/") and
          nm["ts/autode.log"] == "ts/autode.log", roots=roots)
    want = [("tsdir", "optts", "ts"), ("tsdir", "const_opt", "tsguess_opt"), ("path", "const_opt", "path"),
            ("neb", "engrad", "neb"), ("trunc", "optts", "truncated"), ("tsdir", "opt", "fwd_bwd"),
            ("rp", "opt", "react_opt"), ("rp", "opt", "prod_opt"), ("thermal", "freq", "hess_geom"),
            ("ts_sp", "sp", "other_sp"), ("ref", "opt", "ref_opt"), ("ref", "opt", "ref_opt"),
            ("sp_stage", "sp", "other_sp")]
    got = []
    for r, (c, jt, k) in zip(rels, want):
        n = nm[r]
        got.append((N.classify(n), N.base_kind(N.classify(n), n.rsplit("/", 1)[-1], jt)))
    check(res, "classes_kinds", got == [(c, k) for c, _, k in want], got=got)
    # two autodE roots -> nothing stripped (recorded)
    nm2, roots2 = N._norm_map(["ts/A/x/y.out", "ts/B/x/y.out"])
    check(res, "norm_two_roots", roots2 == ["A", "B"] and nm2["ts/A/x/y.out"] == "ts/A/x/y.out")
    check(res, "wanted", N._wanted("ts/T/transition_states/X_optts_orca.out") and
          not N._wanted("ts/T/conformers/r0_conf0_opt_xtb.out") and not N._wanted("ts/T/x/path_opt1_xtb.out") and
          N._wanted("ts/T/transition_states/X_optts_orca.002.hess") and not N._wanted("sp/eda.cpcm"))
    m = N.RE_UNIT.match("jobs/T00001/archive.tgz")
    m2 = N.RE_UNIT.match("d1_scratch/jobs_pilot_v1/C0020/ref/port/dipole_alt_opt_orca.out")
    check(res, "unit_re", m.groups() == ("jobs", "T00001", "archive.tgz") and
          m2.groups() == ("jobs_pilot_v1", "C0020", "ref/port/dipole_alt_opt_orca.out") and
          N.RE_UNIT.match("logs/T00001.log") is None and N.RE_UNIT.match("attempts/T00001") is None)
    check(res, "hess_idx", N.RE_HESS_IDX.match("ts/transition_states/X_optts_orca.002.hess").groups() ==
          ("ts/transition_states/X_optts_orca", "002") and N.RE_HESS_IDX.match("a/X_hess_orca.hess") is None)
    check(res, "short", N.short("d1_20261010_g01_002") == "g01_002" and
          N.short("d1_extra_02_jobs_unfinished") == "extra_02_jobs_unfinished")
    reg = N.registry()
    nres = sum(e["kind"] == "result" for e in reg.values())
    check(res, "registry", nres == 23 and all("sha256" in e for e in reg.values() if e["kind"] == "result")
          and all(reg[x]["kind"] == "extra" for x in N.EXTRA_DATA), n_result=nres, n=len(reg))
    # kabsch / dists / freqs
    P = np.random.default_rng(0).normal(size=(9, 3))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Q = P @ R.T + 3.0
    check(res, "kabsch", N.kabsch(P, Q)[1] < 1e-9 and N.raw_dev(P, Q) > 1.0)
    H = np.diag([0.5, 0.5, 0.5, -0.02, 0.5, 0.5])
    fr = N.freqs_cm(H, [1.0, 1.0])
    check(res, "freqs", int((fr < N.IMAG_TOL_CM).sum()) == 1, freqs=[round(float(x), 1) for x in fr])
    # tag rules on real targets
    ctx = N.Ctx()
    T = ctx.T
    r0 = int(T.index[0])
    ji = ctx.job_info(C.job_of(r0), {})
    check(res, "target_tags", ji["target"] and ji["tags"]["rxn_ids"] == str(r0) and
          ji["tags"]["folds"] == str(int(T.at[r0, "fold_T1"])), tags=ji["tags"])
    nt = next(j for j, r in ctx.d1m.items() if j not in ctx.job_rxn and r.get("kind") == "D1"
              and f"{r['dipole']}/{r['skeleton']}" in ctx.cf)
    jn = ctx.job_info(nt, {})
    core = f"{ctx.d1m[nt]['dipole']}/{ctx.d1m[nt]['skeleton']}"
    check(res, "nontarget_tags", (not jn["target"]) and str(ctx.cf[core]) in jn["tags"]["folds"].split(",") and
          ctx.d1m[nt]["dipole"] in jn["tags"]["dipoles"].split(",") and jn["tags"]["rxn_ids"] == "", job=nt,
          tags=jn["tags"])
    # non-target jobs over the whole D1 manifest: same-cell target folds outside the core fold (should be 0)
    for j, r in ctx.d1m.items():
        if j not in ctx.job_rxn and r.get("kind", "D1") == "D1":
            ctx.job_info(j, {})
    check(res, "nontarget_cell_folds_in_core", ctx.checks.get("nontarget_cell_folds_outside_core", 0) == 0,
          checks={k: v for k, v in ctx.checks.items() if k.startswith("nontarget")})
    # multi-member gzip extxyz round trip via mlt_common.read_extxyz (direct, on the .gz) and read_ext
    al = [C.ext_atoms(["C", "H"], [[0, 0, 0], [0, 0, 1.1 + 0.01 * i]], -38.0 - i, [[0, 0, 0.01], [0, 0, -0.01]],
                      struct_id=f"N:T{i}:x:{i}", kind="nur_ts_path", source="nurion", rxn_ids="" if i % 2 else "1000001",
                      folds="" if i % 2 else "0,3", dipoles="azide", level="label", job="T00001", file_sha="h0123")
          for i in range(5)]
    old = N.EXT_CHUNK
    N.EXT_CHUNK = 2
    p = TMP / "unit_mm.extxyz.gz"
    N.write_extxyz_gz(p, al, workers=2)
    N.EXT_CHUNK = old
    b1, b2 = C.read_extxyz(p), N.read_ext(p)
    check(res, "gz_multimember_roundtrip", len(b1) == len(b2) == 5 and all(
        x.info["rxn_ids"] == y.info["rxn_ids"] == z.info["rxn_ids"] and x.info["folds"] == z.info["folds"] and
        np.allclose(x.arrays["REF_forces"], z.arrays["REF_forces"]) for x, y, z in zip(b1, b2, al)),
          rxn=[x.info["rxn_ids"] for x in b1], folds=[x.info["folds"] for x in b1])
    p.unlink()
    # molecule union for the dipolarophile of the first target
    meta = dict(ts_result=dict(rxn_smiles=ctx.d1m[C.job_of(r0)]["rxn_smiles"]))
    smi = meta["ts_result"]["rxn_smiles"].split(">>")[0].split(".")[0]
    from rdkit import Chem
    sym = [a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(smi)).GetAtoms()]
    t, key = ctx.mol_tags(ji, meta, "react_opt", "ts/reactants_and_products/r0_X_opt_orca.out", sym)
    nrx = len([x for x in t["rxn_ids"].split(",") if x])
    check(res, "mol_union", key is not None and str(r0) in t["rxn_ids"].split(",") and
          nrx == len(ctx.tg.smi_rxns.get(key, set()) | {r0}), n_rxn=nrx, smiles=smi)
    tp, _ = ctx.mol_tags(ji, meta, "prod_opt", "ts/reactants_and_products/p0_X_opt_orca.out",
                         [a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(
                             meta["ts_result"]["rxn_smiles"].split(">>")[1])).GetAtoms()])
    check(res, "prod_union", str(r0) in tp["rxn_ids"].split(","), tags=tp)
    return res


def cmd_io(root, tar):
    res = {}
    od = Path(root) / tar
    rep = C.read_json(od / "report.json")
    t0 = time.time()
    al = N.read_ext(od / "nurion.extxyz.gz")
    t_read = time.time() - t0
    check(res, "read", len(al) == rep["summary"]["n_structures"], n=len(al), s=round(t_read, 2),
          per_1k_s=round(1000 * t_read / max(1, len(al)), 2))
    out = TMP / "io_roundtrip.extxyz.gz"
    t0 = time.time()
    N.write_extxyz_gz(out, al)
    t_w = time.time() - t0
    b = N.read_ext(out)
    same = all(x.info["struct_id"] == y.info["struct_id"] and x.info["rxn_ids"] == y.info["rxn_ids"] and
               x.info["folds"] == y.info["folds"] and x.info["dipoles"] == y.info["dipoles"] and
               x.info["kind"] == y.info["kind"] and np.abs(x.positions - y.positions).max() == 0 and
               np.abs(x.arrays["REF_forces"] - y.arrays["REF_forces"]).max() == 0 for x, y in zip(al, b))
    check(res, "roundtrip", same and len(b) == len(al), write_s=round(t_w, 2), per_1k_s=round(1000 * t_w / max(1, len(al)), 2),
          bytes=out.stat().st_size)
    plain = TMP / "io_plain.extxyz"
    t0 = time.time()
    C.write_extxyz(plain, al)
    check(res, "plain_write", True, s=round(time.time() - t0, 2), bytes=plain.stat().st_size,
          gz_ratio=round(plain.stat().st_size / max(1, (od / "nurion.extxyz.gz").stat().st_size), 2))
    out.unlink()
    plain.unlink()
    empty = [a.info["struct_id"] for a in al if any(isinstance(v, str) and v == "" for k, v in a.info.items()
                                                       if k not in C.EXT_TAGS)]
    check(res, "no_empty_info", not empty, first=empty[:3])
    num = [a.info["struct_id"] for a in al if not (isinstance(a.info.get("file_sha"), str) and
                                                  a.info["file_sha"].startswith("h") and len(a.info["file_sha"]) == 17)
           or ("mol" in a.info and not str(a.info["mol"]).startswith("m"))]
    check(res, "ids_stay_str", not num, n_bad=len(num), first=num[:3])
    tags_bad = [a.info["struct_id"] for a in al if a.info["kind"].startswith("nur_") and
                not a.info["kind"].startswith("nur_x_") and not a.info["rxn_ids"]]
    check(res, "target_rows_tagged", not tags_bad, n_bad=len(tags_bad))
    near = sum(1 for a in al if "dform_label_A" in a.info and float(a.info["dform_label_A"]) <= N.NEAR_A)
    check(res, "near_count", near == rep["summary"]["ts_near"]["n"], n=near)
    keys = {N.geom_key(a.get_chemical_symbols(), a.positions) for a in al}
    check(res, "no_dup_geom", len(keys) == len(al))
    with np.load(od / "hess_other.npz") as z:
        idx = C.read_json(od / "hess_other.json")["hessians"]
        okh = all(len(z[f"H{r['i']}"]) == (3 * len(r["symbols"])) * (3 * len(r["symbols"]) + 1) // 2 for r in idx)
    check(res, "hess_other_shapes", okh, n=len(idx))
    return res


def _snap(od):
    return {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(od.iterdir())}


def cmd_idem(root, tar):
    res = {}
    od = Path(root) / tar
    before = _snap(od)
    r = subprocess.run([sys.executable, str(HERE / "inv_nurion.py"), "scan", tar, "--root", str(root), "--workers", "1"],
                       capture_output=True, text=True)
    after = _snap(od)
    check(res, "second_scan_noop", r.returncode == 0 and before == after and "skipped" in r.stdout,
          rc=r.returncode, out=r.stdout[-200:])
    r2 = subprocess.run([sys.executable, str(HERE / "inv_nurion.py"), "scan", tar, "--jobs", "T00001"],
                        capture_output=True, text=True)
    check(res, "jobs_needs_test_root", r2.returncode == 2, rc=r2.returncode)
    return res


def cmd_phasei(root, tar):
    """Nurion rows whose file_sha equals a Phase I row's (same source file?) + the Phase I source paths."""
    import csv
    pi = {}
    for a in C.read_extxyz(N.PHASE_I_EXT):
        pi.setdefault(str(a.info.get("file_sha")), []).append(a.info["struct_id"])
    src = {}
    with open(N.I.RES_SRC) as f:
        for r in csv.DictReader(f):
            src[r["sha256"][:16]] = (r["path"], r["sha256"], r["kept"])
    hits = {}
    for a in N.read_ext(Path(root) / tar / "nurion.extxyz.gz"):
        fs = str(a.info.get("file_sha"))
        if fs in pi:
            h = hits.setdefault(fs, dict(nurion=[], phase_i=pi[fs][:2], source=src.get(fs)))
            h["nurion"].append(a.info["struct_id"])
    res = {}
    check(res, "phase_i_file_sha", True, n_files=len(hits), hits={k: dict(v, nurion=v["nurion"][:2], n=len(v["nurion"]))
                                                               for k, v in list(hits.items())[:10]})
    return res


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    fn = dict(io=cmd_io, idem=cmd_idem, phasei=cmd_phasei)
    res = cmd_unit() if a[0] == "unit" else fn[a[0]](a[1], a[2])
    print(json.dumps(res, indent=1, default=C._json_default))
    return 0 if all(v["ok"] for v in res.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
