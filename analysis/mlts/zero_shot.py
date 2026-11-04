#!/usr/bin/env python3
"""zero_shot.py — DEPLOY-V2 Phase Z: choose the base MLIP (docs/specs/DEPLOY_V2_MLTS.md 4).

  python zero_shot.py select                  results/Z_list.csv: 200 reactions = 40 per dipole from the training side
                                              of fold 0 (fold_T1 != 0), seeded (mlt_common.SEED), label TS present
  python zero_shot.py run <cand> [n]          one candidate (GPU job; xtb: CPU job), first n of Z_list (default all):
        (b) geometry: ts_geom.run_list from the label TS -> $MLT/z/<cand>/geom_label/ (records + summary.json)
        (a) forces / energies on the label-level G data present now ($MLT/dft/{existing,gen_G1,gen_G2,gen_G3}.extxyz,
            mlt_common.read_extxyz): structures tagged with a Z reaction (z_tagged) + a seeded sample of TS-region
            structures of any reaction (ts_region; zero-shot models never saw them). Force MAE (per component) and
            within-group relative energy MAE (energies minus the group mean, group = extxyz 'group': one reaction or
            one molecule). Reference only: the candidates' training levels differ from the label level.
        (c) speed: E+F per call and Hessian time on Z label TSs in this process (GPU)
        -> $MLT/z/<cand>/z.json; an unavailable candidate gets z.json with available=false and the reason
  python zero_shot.py speed [cand ...]        E+F / Hessian timing of dtype / cuEquivariance variants (lead 11-04):
                                              MACE-OMOL-0 and MACE-MH-1 in float64 / float32 / cuEq -> $MLT/z/speed.json
  python zero_shot.py report                  results/Z_report.json + results/Z_DECISION.md (ranking rule below)
Candidates: MACE-OMOL-0, AIMNet2, MACE-MH-1 (omol head; added by the lead 11-04, not in the spec list), MACE-OFF24
(no Si: Si-free reactions only, reference value), xtb = ORCA XTB2 ALPB(water) OptTS (G2 level, baseline, CPU).
UMA-S is not installed (no HuggingFace token; DECISIONS 0-7) and is reported as unavailable.
Ranking rule (pre-registered here, before any Z number): eligible = MLIPs that cover every D1 element (Si);
  sort by same-TS fraction from the label start (higher first); within 0.025 of each other -> median |Δd| (lower
  first); within 0.002 A -> faster median GPU E+F call. Top 2 -> FT pilot (spec 4). MACE-OFF24 (no Si) and xtb
  (not fine-tunable) are reported, not ranked.
Pre-registered expectation (spec 4): zero-shot |Δd| similar to g3r's 0.063 A, i.e. insufficient for FT-2 (a)."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as C  # noqa: E402
import mlts_bridge as B  # noqa: E402
import ts_geom as TG  # noqa: E402

ZDIR = C.MLT / "z"
ZLIST = C.RES / "Z_list.csv"
N_PER_DIPOLE = 40
CANDS = ("MACE-OMOL-0", "AIMNet2", "MACE-MH-1", "MACE-OFF24", "xtb")
UNAVAILABLE = {"UMA-S": "not installed: HuggingFace access token required (DECISIONS.md 0-7; spec 4 '없으면 뺀다')"}
G_FILES = ("existing", "gen_G1", "gen_G2", "gen_G3")
TS_KINDS = ("ts_path", "ts_final", "d0_ts_path", "d0_ts_final", "d0_ts_guess", "d0_ts_grad", "d0_irc",   # Phase I
            "ts_near", "ts_imag", "ts_label", "g2_path", "ts_final_g2")                                  # gen_dft G1 / G2
N_TS_REGION = int(os.environ.get("MLT_Z_NTS", "1500"))     # seeded sample of TS-region structures (other reactions)
N_Z_TAGGED = int(os.environ.get("MLT_Z_NZ", "2000"))       # seeded cap on Z-tagged structures (mostly reactant conformers)
N_SPEED = int(os.environ.get("MLT_Z_NSPEED", "20"))
N_HESS_SPEED = int(os.environ.get("MLT_Z_NHESS", "5"))
TIE_SAME = 0.025
TIE_DD = 0.002
G3R_DD = 0.063
KCAL_PER_EV = C.EH2KCAL / C.EH2EV


def cand_dir(c):
    return ZDIR / c


# ================================================================ select
def cmd_select():
    T = C.targets()
    T = T[T.fold_T1 != 0]
    T = T[[C.label_ts(r).is_file() for r in T.rxn_id]]
    rng = np.random.default_rng(C.SEED)
    rows = []
    for d in C.DIPOLES:
        g = T[T.dipole == d].sort_values("rxn_id")
        if len(g) < N_PER_DIPOLE:
            C.stop(f"only {len(g)} training-side reactions for {d}")
        pick = g.iloc[np.sort(rng.choice(len(g), N_PER_DIPOLE, replace=False))]
        rows.append(pick)
    import pandas as pd
    Z = pd.concat(rows)
    Z["has_si"] = [("Si" in C.read_xyz(C.label_ts(r))[0]) for r in Z.rxn_id]
    Z = Z[["rxn_id", "job_id", "dipole", "skeleton", "fold_T1", "n_atoms", "has_si", "formed_pairs_ts"]]
    C.write_csv(ZLIST, Z)
    print(f"[Z] {len(Z)} reactions -> {ZLIST}; per dipole {Z.groupby('dipole').size().to_dict()}; with Si {int(Z.has_si.sum())}; "
          f"n_atoms median {int(Z.n_atoms.median())} (max {int(Z.n_atoms.max())})")
    return 0


def zlist(n=None):
    """the Z list; with n, a dry-run subset taken round-robin over the dipoles (n = 5: one per dipole)."""
    import pandas as pd
    if not ZLIST.is_file():
        C.not_ready(f"{ZLIST} missing: run 'zero_shot.py select'")
    Z = pd.read_csv(ZLIST)
    if n is None:
        return Z
    Z = Z.assign(_k=Z.groupby("dipole", sort=False).cumcount()).sort_values(["_k", "dipole", "rxn_id"])
    return Z.drop(columns="_k").head(int(n)).reset_index(drop=True)


# ================================================================ (a) forces / energies on G data
def g_structures(zset):
    """(atoms list, availability info): structures tagged with a Z reaction (z_tagged) + a seeded sample of the
    TS-region structures of all other reactions (ts_region), over every G file present now."""
    avail, zt, pool = {}, [], []
    for f in G_FILES:
        p = C.MLT / "dft" / f"{f}.extxyz"
        if not p.is_file():
            avail[f] = dict(present=False)
            continue
        al = C.read_extxyz(p)
        nz = nt = 0
        for a in al:
            rx = {int(x) for x in str(a.info.get("rxn_ids", "")).split(",") if x.strip().isdigit()}
            a.info["_src"] = f
            if rx & zset:
                a.info["_set"] = "z_tagged"
                zt.append(a)
                nz += 1
            elif str(a.info.get("kind", "")) in TS_KINDS:
                pool.append(a)
                nt += 1
        avail[f] = dict(present=True, n=len(al), n_z_tagged=nz, n_ts_region_pool=nt, sha256=C.sha256(p)[:16])
    rng = np.random.default_rng(C.SEED)
    pick = sorted(rng.choice(len(pool), min(N_TS_REGION, len(pool)), replace=False)) if pool else []
    ts = [pool[i] for i in pick]
    for a in ts:
        a.info["_set"] = "ts_region"
    if len(zt) > N_Z_TAGGED:
        zt = [zt[i] for i in sorted(rng.choice(len(zt), N_Z_TAGGED, replace=False))]
    avail["caps"] = dict(n_ts_region=N_TS_REGION, n_z_tagged=N_Z_TAGGED, seed=C.SEED)
    return zt + ts, avail


def group_of(a):
    """relative-energy group: existing.extxyz 'group' (one reaction's OptTS / one molecule's conformers); gen_G*
    struct_id without its last '_' part (g1_<rxn>, g2_<rxn>, g3_<molecule>); TS fragments are singletons. The formula
    is part of the key, so only structures of one composition are compared."""
    from collections import Counter
    f = "".join(f"{e}{n}" for e, n in sorted(Counter(a.get_chemical_symbols()).items()))
    if a.info["_src"] == "existing" and a.info.get("group") not in (None, ""):
        return f"existing:{a.info['group']}:{f}"
    sid = str(a.info.get("struct_id", ""))
    if str(a.info.get("kind")) == "frag":
        return f"{a.info['_src']}:{sid}"
    return f"{a.info['_src']}:{sid.rsplit('_', 1)[0]}:{f}"


def predict_all(spec, atoms):
    """(E eV, F eV/A) per structure; xtb through ORCA EnGrad in a thread pool, MLIPs in this process."""
    xtb = B.is_xtb(spec)

    def one(a):
        s = a.get_chemical_symbols()
        if xtb:
            e, g = B.direct_engrad(spec, s, a.positions)
            return e * C.EH2EV, -np.asarray(g) * C.EHBOHR2EVA
        e, f, _ = B.direct_ev(spec, s, a.positions)
        return e, f
    if xtb:
        with ThreadPoolExecutor(TG.cpus()) as ex:
            return list(ex.map(one, atoms))
    return [one(a) for a in atoms]


def force_metrics(spec, zset):
    atoms, avail = g_structures(zset)
    atoms = [a for a in atoms if B.covers(spec, a.get_chemical_symbols())]
    t0 = time.time()
    pred = predict_all(spec, atoms)
    rows = []
    for a, (e, f) in zip(atoms, pred):
        fr = np.asarray(a.arrays["REF_forces"], float)
        rows.append(dict(set=a.info["_set"], kind=str(a.info.get("kind", "")), src=a.info["_src"], group=group_of(a),
                         n_atoms=len(a), has_si="Si" in a.get_chemical_symbols(), e_ref=float(a.info["REF_energy"]),
                         e_pred=float(e), f_abs_err_sum=float(np.abs(np.asarray(f) - fr).sum()), n_comp=int(fr.size),
                         f_sq_err_sum=float(((np.asarray(f) - fr) ** 2).sum())))
    import pandas as pd
    D = pd.DataFrame(rows)

    def m(d):
        if not len(d):
            return dict(n_struct=0)
        out = dict(n_struct=int(len(d)), n_groups=int(d.group.nunique()),
                   force_mae_eV_A=float(d.f_abs_err_sum.sum() / d.n_comp.sum()),
                   force_rmse_eV_A=float(np.sqrt(d.f_sq_err_sum.sum() / d.n_comp.sum())))
        out["force_mae_kcal_mol_A"] = out["force_mae_eV_A"] * KCAL_PER_EV
        g = d.groupby("group")
        dd = d.assign(r=d.e_ref - g.e_ref.transform("mean"), p=d.e_pred - g.e_pred.transform("mean"),
                      ng=g.e_ref.transform("size"))
        dd = dd[dd.ng >= 2]
        out["rel_energy_mae_kcal_mol"] = float((dd.p - dd.r).abs().mean() * KCAL_PER_EV) if len(dd) else None
        out["n_struct_rel_energy"] = int(len(dd))
        return out
    res = dict(available_files=avail, wall_s=time.time() - t0, sets={})
    for sname in ("z_tagged", "ts_region"):
        d = D[D.set == sname] if len(D) else D
        res["sets"][sname] = dict(all=m(d), no_si=m(d[~d.has_si]) if len(d) else dict(n_struct=0),
                                  by_kind={k: m(g) for k, g in d.groupby("kind")} if len(d) else {})
    return res


# ================================================================ (c) speed
def label_structs(Z, n):
    out = []
    for r in Z.rxn_id.tolist():
        s, p = C.read_xyz(C.label_ts(r))
        out.append((int(r), s, np.asarray(p, float)))
        if len(out) >= n:
            break
    return out


def bench(spec, structs, n_rep=3, n_hess=N_HESS_SPEED, ref=None, ref_nu=None, keep_nu=None):
    """E+F ms per call (median of n_rep after one warm-up per structure) and Hessian s, in this process.
    ref: {rxn: (E, F)} and ref_nu: {rxn: frequencies} of the reference variant (deviations reported); keep_nu: dict
    filled with this variant's frequencies."""
    t0 = time.time()
    B.evaluator(spec)
    load = time.time() - t0
    ef, hs, dev, dnu = [], [], [], []
    for k, (r, s, p) in enumerate(structs):
        if not B.covers(spec, s):
            continue
        e0, f0, _ = B.direct_ev(spec, s, p)
        ts = []
        for _ in range(n_rep):
            q = p + 1e-6 * (len(ts) + 1)                          # new positions: no calculator cache
            t = time.perf_counter()
            B.direct_ev(spec, s, q)
            ts.append(time.perf_counter() - t)
        ef.append(dict(rxn_id=r, n_atoms=len(s), ms=1e3 * float(np.median(ts))))
        if ref is not None and r in ref:
            dev.append(dict(rxn_id=r, dE_eV=abs(e0 - ref[r][0]), dF_max_eV_A=float(np.abs(f0 - ref[r][1]).max())))
        if len(hs) < n_hess:
            B.hessian(spec, s, p + 2e-6)                              # warm-up (first call per shape is slow)
            t = time.perf_counter()
            H = B.hessian(spec, s, p)
            hs.append(dict(rxn_id=r, n_atoms=len(s), s=time.perf_counter() - t))
            nu = B.freqs_cm(H, s, p)
            if keep_nu is not None:
                keep_nu[r] = nu
            if ref_nu is not None and r in ref_nu:
                dnu.append(float(np.abs(nu - ref_nu[r]).max()))
    out = dict(load_s=load, n=len(ef), ef_ms_median=float(np.median([x["ms"] for x in ef])) if ef else None,
               ef_ms_by_n_atoms=sorted((x["n_atoms"], round(x["ms"], 1)) for x in ef),
               hess_s_median=float(np.median([x["s"] for x in hs])) if hs else None,
               hess_by_n_atoms=sorted((x["n_atoms"], round(x["s"], 2)) for x in hs), **B.device_info())
    if dev:
        out["vs_reference"] = dict(max_dE_eV=max(x["dE_eV"] for x in dev), max_dF_eV_A=max(x["dF_max_eV_A"] for x in dev))
    if dnu:
        out.setdefault("vs_reference", {})["max_dnu_cm"] = max(dnu)
    return out


SPEED_VARIANTS = {
    "MACE-OMOL-0": [dict(dtype="float64"), dict(dtype="float32"), dict(dtype="float64", enable_cueq=True),
                    dict(dtype="float32", enable_cueq=True), dict(dtype="float64", hessian="fd")],
    "MACE-MH-1": [dict(dtype="float64"), dict(dtype="float32"), dict(dtype="float64", enable_cueq=True),
                  dict(dtype="float32", enable_cueq=True), dict(dtype="float64", hessian="fd")],
    "AIMNet2": [dict()],
    "MACE-OFF24": [dict(dtype="float64")],
}
SPEED_DEFAULT_TAG = {"MACE-OMOL-0": "MACE-OMOL-0|float64|", "MACE-MH-1": "MACE-MH-1|float64|", "AIMNet2": "AIMNet2||",
                     "MACE-OFF24": "MACE-OFF24|float64|"}


def cmd_speed(cands):
    Z = zlist()
    structs = label_structs(Z, N_SPEED)
    out = C.read_json(ZDIR / "speed.json") if (ZDIR / "speed.json").is_file() else {}
    for c in cands or list(SPEED_VARIANTS):
        ref, ref_nu = None, None
        for v in SPEED_VARIANTS.get(c, [dict()]):
            spec = B.resolve_spec(dict(B.resolve_spec(c), **v))
            tag = (f"{c}|{v.get('dtype', '')}|{'cueq' if v.get('enable_cueq') else ''}"
                   + ("|fdhess" if v.get("hessian") == "fd" else ""))
            try:
                keep = {} if ref_nu is None else None
                r = bench(spec, structs, ref=ref, ref_nu=ref_nu, keep_nu=keep)
                if ref is None:
                    ref = {rr: B.direct_ev(spec, s, p)[:2] for rr, s, p in structs if B.covers(spec, s)}
                    ref_nu = keep
            except Exception as exc:                                         # noqa: BLE001
                r = dict(error=f"{type(exc).__name__}: {exc}"[:400], trace=traceback.format_exc()[-1200:])
            r.update(spec={k: spec[k] for k in spec if k != "card"}, time=C.now(), host=os.uname().nodename)
            out[tag] = r
            print(f"[Z speed] {tag}: {r.get('ef_ms_median')} ms/E+F, Hessian {r.get('hess_s_median')} s "
                  f"{r.get('vs_reference', '')} {r.get('error', '')}", flush=True)
            B._EV.clear()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                                                # noqa: BLE001
                pass
            C.write_json(ZDIR / "speed.json", out)
    return 0


# ================================================================ run one candidate
def cmd_run(cand, n=None):
    """full run -> $MLT/z/<cand>/; dry run (n given) -> $MLT/z/dry<n>/<cand>/ (never read by report)."""
    Z = zlist(n)
    d = C.assert_write_ok(cand_dir(cand) if n is None else ZDIR / f"dry{int(n)}" / cand)
    d.mkdir(parents=True, exist_ok=True)
    zj = d / "z.json"
    J = C.read_json(zj) if zj.is_file() else {}
    J.update(candidate=cand, n_list=int(len(Z)), host=os.uname().nodename, job=B.jobtag(), **B.device_info())
    if cand in UNAVAILABLE:
        J.update(available=False, reason=UNAVAILABLE[cand])
        C.write_json(zj, J)
        return 0
    try:
        spec = B.resolve_spec(cand)
        J["spec"] = {k: v for k, v in spec.items() if k != "card"}
        if not B.is_xtb(spec):
            t0 = time.time()
            B.evaluator(spec)
            J["load_s"] = time.time() - t0
    except Exception as exc:                                                 # noqa: BLE001
        J.update(available=False, reason=f"{type(exc).__name__}: {exc}"[:500], trace=traceback.format_exc()[-1500:])
        C.write_json(zj, J)
        print(f"[Z] {cand} unavailable: {J['reason']}", flush=True)
        return 0
    J["available"] = True
    rx = [int(r) for r, si in zip(Z.rxn_id, Z.has_si) if not (si and spec.get("no_si"))]
    J["n_rxn"] = len(rx)
    J["n_skipped_si"] = int(len(Z) - len(rx))
    C.write_json(zj, J)
    # (b) geometry from the label TS
    gdir = d / "geom_label"
    info = TG.run_list(spec, rx, "label", gdir)
    C.write_json(gdir / f"run_{B.jobtag()}.json", info)
    S = TG.summarize(gdir)
    J["geometry"] = dict(all=S["all"], by_dipole=S["by_dipole"], run=info.get("server_end") or {}, workers=info.get("workers"))
    C.write_json(zj, J)
    # (a) forces / energies
    if "forces" not in J:
        try:
            J["forces"] = force_metrics(spec, set(int(r) for r in Z.rxn_id))
        except Exception as exc:                                             # noqa: BLE001
            J["forces"] = dict(error=f"{type(exc).__name__}: {exc}"[:500], trace=traceback.format_exc()[-1500:])
        C.write_json(zj, J)
    # (c) speed in this process
    if not B.is_xtb(spec) and "speed" not in J:
        try:
            J["speed"] = bench(spec, label_structs(Z, N_SPEED))
        except Exception as exc:                                             # noqa: BLE001
            J["speed"] = dict(error=f"{type(exc).__name__}: {exc}"[:500])
    J["finished"] = C.now()
    C.write_json(zj, J)
    print(json.dumps({k: J.get(k) for k in ("candidate", "available", "n_rxn")}, default=C._json_default), flush=True)
    print(json.dumps(J["geometry"]["all"], indent=1, default=C._json_default), flush=True)
    return 0


# ================================================================ report
def rank(cands):
    """eligible candidates sorted by the pre-registered rule (comparator, see module doc)."""
    from functools import cmp_to_key

    def key(c):
        g = c["geometry"]
        return (g.get("same_ts_frac") or 0.0, g.get("dd_median_A") if g.get("dd_median_A") is not None else 9.9,
                c.get("ef_ms") if c.get("ef_ms") is not None else 1e9)

    def cmp(a, b):
        sa, da, ta = key(a)
        sb, db, tb = key(b)
        if abs(sa - sb) > TIE_SAME:
            return -1 if sa > sb else 1
        if abs(da - db) > TIE_DD:
            return -1 if da < db else 1
        return -1 if ta < tb else (1 if ta > tb else 0)
    return sorted(cands, key=cmp_to_key(cmp))


def cmd_report(sub=None):
    """sub = 'dry<n>': read $MLT/z/dry<n>/<cand>/z.json and write the report there (a test; results/ untouched)."""
    base = ZDIR / sub if sub else ZDIR
    n_list = int(sub[3:]) if sub and sub.startswith("dry") else int(len(zlist()))
    rep = dict(spec_section="4 Phase Z", time=C.now(), z_list=str(ZLIST), n_list=n_list, source=str(base),
               rule=dict(eligible="MLIP covering every D1 element (Si); MACE-OFF24 (no Si) and xtb (baseline) not ranked",
                         order=["same_ts_frac (label start) higher", f"tie within {TIE_SAME} -> dd_median_A lower",
                                f"tie within {TIE_DD} A -> ef_ms (GPU E+F per call) lower"], top=2),
               expectation=dict(text="zero-shot |Δd| similar to g3r (0.063 A) = insufficient for FT-2 (a) "
                                     "(median <= 0.010 A, p90 <= 0.030 A, RMSD <= 0.02 A, same TS >= 0.90)",
                                g3r_dd_A=G3R_DD, ft2_a=TG.FT2),
               additions=dict(**{"MACE-MH-1": "added by the lead (11-04): omol head, 6.4M params, 2.9x faster than "
                                                "MACE-OMOL-0 in finetune.py"}),
               unavailable=dict(UNAVAILABLE), candidates={})
    elig = []
    for c in CANDS:
        zj = base / c / "z.json"
        if not zj.is_file():
            rep["candidates"][c] = dict(available=None, reason="not run")
            continue
        J = C.read_json(zj)
        e = dict(available=J.get("available"), reason=J.get("reason"), n_rxn=J.get("n_rxn"),
                 n_skipped_si=J.get("n_skipped_si"), spec=J.get("spec"))
        if J.get("available"):
            g = (J.get("geometry") or {}).get("all") or {}
            e["geometry"] = {k: g.get(k) for k in ("n_attempted", "n_converged", "conv_frac", "dd_median_A", "dd_p90_A",
                                                   "rmsd_median_A", "same_ts_frac", "nimag1_frac", "cycles_median",
                                                   "wall_median_s", "gpu_s_median", "ext_calls_median", "status_counts",
                                                   "ft2_a_pass")}
            e["geometry_by_dipole"] = {dp: {k: v.get(k) for k in ("n_attempted", "dd_median_A", "same_ts_frac",
                                                                  "nimag1_frac")}
                                       for dp, v in ((J.get("geometry") or {}).get("by_dipole") or {}).items()}
            fo = J.get("forces") or {}
            e["forces"] = {s: dict(all=v.get("all"), no_si=v.get("no_si")) for s, v in (fo.get("sets") or {}).items()}
            e["force_files"] = fo.get("available_files")
            sp = J.get("speed") or {}
            e["speed"] = {k: sp.get(k) for k in ("ef_ms_median", "hess_s_median", "gpu", "load_s", "ef_ms_by_n_atoms")}
            e["ef_ms"] = sp.get("ef_ms_median")
            spec = J.get("spec") or {}
            e["eligible"] = spec.get("backend") in ("mace", "aimnet2") and not spec.get("no_si")
            if e["eligible"] and e["geometry"].get("n_attempted"):
                elig.append(dict(name=c, geometry=e["geometry"], ef_ms=e["ef_ms"]))
        rep["candidates"][c] = e
    if (ZDIR / "speed.json").is_file():
        SP = C.read_json(ZDIR / "speed.json")
        rep["speed_variants"] = {k: {kk: v.get(kk) for kk in ("ef_ms_median", "hess_s_median", "vs_reference", "error",
                                                              "gpu")} for k, v in SP.items()}
        same_gpu = {c: SP[t].get("ef_ms_median") for c, t in SPEED_DEFAULT_TAG.items() if t in SP}
        if all(x["name"] in same_gpu and same_gpu[x["name"]] is not None for x in elig):
            for x in elig:                                     # one GPU for every candidate: fair tie-break
                x["ef_ms"] = same_gpu[x["name"]]
            rep["rule"]["speed_source"] = "speed.json (one job, one GPU)"
    rep["rule"].setdefault("speed_source", "per-candidate z.json (GPU type may differ between jobs)")
    order = rank(elig)
    rep["ranking"] = [x["name"] for x in order]
    rep["top2_for_ft_pilot"] = rep["ranking"][:2]
    dds = [x["geometry"].get("dd_median_A") for x in elig if x["geometry"].get("dd_median_A") is not None]
    rep["expectation_met"] = (bool(dds) and min(dds) > TG.FT2["dd_median_A"])
    out = base if sub else C.RES
    C.write_json(out / "Z_report.json", rep)
    C.write_text(out / "Z_DECISION.md", decision_md(rep))
    print(json.dumps(dict(ranking=rep["ranking"], top2=rep["top2_for_ft_pilot"]), indent=1))
    return 0


def _fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def decision_md(rep):
    L = ["# Phase Z 결정 (zero_shot.py report)", "",
         f"작성 {rep['time']}. 시험 반응 {rep['n_list']}개(`results/Z_list.csv`: fold 0 학습 쪽, dipole별 40, seed "
         f"{C.SEED}). 모든 기하 지표는 라벨 TS에서 출발한 ORCA `ExtOpt OptTS`(D1 블록, MLIP 초기 Hessian; xtb는 G2 블록)",
         "", "## 등록한 규칙 (Z 결과 전에 코드에 고정)", "",
         "- 대상: D1 원소 전부(Si 포함)를 다루는 MLIP만 순위에 넣는다. MACE-OFF24(Si 없음)는 Si 없는 반응의 참고값, "
         "xtb(GFN2-xTB ALPB(water) = G2 수준)는 기준선이다.",
         f"- 순서: 같은 TS 비율(높을수록) → 차이 {TIE_SAME} 이내면 |Δd| 중앙값(낮을수록) → 차이 {TIE_DD} Å 이내면 "
         "GPU E+F 호출 시간(빠를수록). 상위 2개로 fold 0 FT 시범(명령서 4절).",
         f"- 사전 예상(명령서 4절): 미세조정 전 |Δd|는 g3r({G3R_DD} Å)과 비슷하게 부족하다(FT-2 (a): 중앙값 ≤ 0.010 Å).",
         "- 추가 후보: MACE-MH-1(omol head) — 명령서 목록에 없고 lead 지시(11-04)로 더했다. UMA-S는 설치 불가(토큰 없음).",
         "", "## 결과", "",
         "| 후보 | 사용 | 반응 | 같은 TS | \\|Δd\\| 중앙값 Å | \\|Δd\\| 90% Å | RMSD 중앙값 Å | 허수 1개 | 수렴 | "
         "힘 MAE eV/Å (TS 영역) | 상대 E MAE kcal/mol | GPU ms/E+F | Hessian s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, e in rep["candidates"].items():
        if not e.get("available"):
            L.append(f"| {c} | 아니오: {str(e.get('reason'))[:80].replace('|', '/')} | | | | | | | | | | | |")
            continue
        g = e.get("geometry") or {}
        fo = ((e.get("forces") or {}).get("ts_region") or {}).get("all") or {}
        sp = e.get("speed") or {}
        L.append(f"| {c} | 예 | {g.get('n_attempted')} | {_fmt(g.get('same_ts_frac'), 2)} | {_fmt(g.get('dd_median_A'))} | "
                 f"{_fmt(g.get('dd_p90_A'))} | {_fmt(g.get('rmsd_median_A'))} | {_fmt(g.get('nimag1_frac'), 2)} | "
                 f"{_fmt(g.get('conv_frac'), 2)} | {_fmt(fo.get('force_mae_eV_A'))} | {_fmt(fo.get('rel_energy_mae_kcal_mol'), 2)} | "
                 f"{_fmt(sp.get('ef_ms_median'), 1)} | {_fmt(sp.get('hess_s_median'), 2)} |")
    L += ["", f"- 순위(대상만): {', '.join(rep['ranking']) or '—'}",
          f"- **FT 시범 상위 2개: {', '.join(rep['top2_for_ft_pilot']) or '—'}**",
          f"- 사전 예상(미세조정 전 부족) 확인: {rep['expectation_met']}",
          "- 힘·에너지 지표는 수준이 달라(후보는 ωB97M 계열·기체상, 라벨은 B3LYP-D3BJ/def2-SVP SMD water) 참고값이다.", ""]
    if rep.get("speed_variants"):
        L += ["## 속도 변형 (dtype / cuEquivariance)", "",
              "| 변형 | GPU | ms/E+F | Hessian s | f64 대비 max \\|ΔF\\| eV/Å | 오류 |", "|---|---|---|---|---|---|"]
        for k, v in rep["speed_variants"].items():
            tag = " ".join(t for t in k.split("|") if t) or k
            L.append(f"| {tag} | {v.get('gpu') or ''} | {_fmt(v.get('ef_ms_median'), 1)} | {_fmt(v.get('hess_s_median'), 2)} | "
                     f"{_fmt((v.get('vs_reference') or {}).get('max_dF_eV_A'), 5)} | {str(v.get('error') or '')[:60]} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        print(__doc__)
        return 2
    if a[0] == "select":
        return cmd_select()
    if a[0] == "run" and len(a) >= 2:
        return cmd_run(a[1], a[2] if len(a) > 2 else None)
    if a[0] == "speed":
        return cmd_speed(a[1:])
    if a[0] == "report":
        return cmd_report(a[1] if len(a) > 1 else None)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
