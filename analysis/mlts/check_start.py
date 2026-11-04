#!/usr/bin/env python3
"""check_start.py — DEPLOY-V2 (MLTS) start conditions 0-1 .. 0-8 (docs/specs/DEPLOY_V2_MLTS.md section 1).

  python check_start.py run [--free-gb X]     -> results/S0_start.json, results/0_elements.json

  0-1  git: HEAD (and origin/deploy, recorded) contain 2f9ffa35; analysis/{d1_g2,d1_g3,d1_hf,d1_ladder,d1_pair,deploy,geo2}
  0-2  $LAD/units/L__dft__{S,SH}__{TF,GNN,AW}__f{0..4}.{json,parquet} + __s{0..4}.pt (30 units x 7 files, non-empty)
  0-3  $G2/jobs/<job_id>/{ts.xyz, ts_result.json} for every G2 ok reaction (G_status.csv = g2_inputs.csv ok, 8,616)
  0-4  $D1RUN/config_run.yaml and analysis/d1_g2/config_g2.yaml load (yaml.safe_load) with a ts_stage block
  0-5  ORCA `! ExtOpt` works -> deferred to C-1 (recorded, not a failure here)
  0-6  element set over ALL label structures (TS + rel1 + rel2 of the 9,266 targets) == {H,C,N,O,F,Si,S,Cl,Br};
       charge1 = charge2 = 0 (labels) and an even electron count of TS, rel1, rel2 (multiplicity 1 is consistent);
       the D1 label inputs ($D1HF/inputs/<job>/orig/{eda,frag1_rel,frag2_rel}.inp) are read too: '* xyz 0 1',
       FRAG{1,2}_C 0, FRAG{1,2}_M 1. -> results/0_elements.json (element counts per dipole and per skeleton)
  0-7  results/0_env.json pass_0_7 (written by env_check.py); missing -> 'pending'
  0-8  free GB (argument = login-node mmlsquota value; compute nodes cannot run mmlsquota) >= 200; a failure is
       recorded only (the deviation is decided in results/DECISIONS.md '0-8')
  also: mlt_common.targets() fold sizes (fold_T1, LODO dipole), cores without a fold.

Exit 0 = everything except 0-5 (deferred), 0-7 (pending / env) and 0-8 (recorded deviation) passes;
exit 3 = a real STOP (any of 0-1 .. 0-4, 0-6 failed; S0_start.json is written first)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlt_common as C  # noqa: E402

BASE_COMMIT = "2f9ffa35"
ANALYSIS_DIRS = ("d1_g2", "d1_g3", "d1_hf", "d1_ladder", "d1_pair", "deploy", "geo2")
N_G2_OK = 8616
N_TARGETS = 9266
FREE_GB_MIN = 200
OUT = C.RES / "S0_start.json"
OUT_EL = C.RES / "0_elements.json"
G2_STATUS = C.ANALYSIS / "d1_g2" / "results" / "G_status.csv"
G2_INPUTS = C.ANALYSIS / "d1_g2" / "results" / "g2_inputs.csv"


def n_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _git(*args):
    r = subprocess.run(["git", "-C", str(C.REPO), *args], capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


# ---------------------------------------------------------------- 0-1
def check_0_1():
    rc, head, err = _git("rev-parse", "HEAD")
    rc_b, base, _ = _git("rev-parse", BASE_COMMIT)
    anc_head = _git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD")[0] == 0 if rc == 0 and rc_b == 0 else False
    rc_o, odep, _ = _git("rev-parse", "origin/deploy")
    anc_odep = (_git("merge-base", "--is-ancestor", BASE_COMMIT, "origin/deploy")[0] == 0) if rc_o == 0 else None
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")[1]
    dirs = {d: (C.ANALYSIS / d).is_dir() for d in ANALYSIS_DIRS}
    ok = bool(anc_head and all(dirs.values()))
    return dict(ok=ok, repo=str(C.REPO), head=head if rc == 0 else f"error: {err}", branch=branch,
                base_commit=base if rc_b == 0 else None, head_contains_base=anc_head,
                origin_deploy=odep if rc_o == 0 else None, origin_deploy_contains_base=anc_odep,
                analysis_dirs=dirs, missing_dirs=[d for d, v in dirs.items() if not v])


# ---------------------------------------------------------------- 0-2
def check_0_2():
    files, missing, empty = [], [], []
    for ref in ("S", "SH"):
        for arch in ("TF", "GNN", "AW"):
            for k in range(5):
                stem = f"L__dft__{ref}__{arch}__f{k}"
                names = [f"{stem}.json", f"{stem}.parquet"] + [f"{stem}__s{s}.pt" for s in range(5)]
                for n in names:
                    p = C.LAD / "units" / n
                    files.append(n)
                    if not p.exists():
                        missing.append(n)
                    elif p.stat().st_size == 0:
                        empty.append(n)
    return dict(ok=not missing and not empty and len(files) == 210, n_units=30, n_files_expected=210,
                n_files_present=len(files) - len(missing), missing=missing[:50], n_missing=len(missing),
                empty=empty[:50], units_dir=str(C.LAD / "units"))


# ---------------------------------------------------------------- 0-3
def _g2_files(job):
    d = C.G2_JOBS / job
    out = dict(job_id=job)
    for n in ("ts.xyz", "ts_result.json"):
        p = d / n
        out[n] = p.exists() and p.stat().st_size > 0
    if out["ts.xyz"]:
        try:
            out["n_atoms"] = int((d / "ts.xyz").open().readline().split()[0])
        except Exception:                                    # noqa: BLE001
            out["n_atoms"] = None
    return out


def check_0_3(pool, T):
    import pandas as pd
    S = pd.read_csv(G2_STATUS, usecols=["rxn_id", "job_id", "status"])
    I = pd.read_csv(G2_INPUTS, usecols=["job_id", "rxn_id", "status"])
    ok_s = set(S.loc[S.status == "ok", "job_id"])
    ok_i = set(I.loc[I.status == "ok", "job_id"])
    jobs = sorted(ok_s)
    res = pool.map(_g2_files, jobs, chunksize=64)
    miss_ts = [r["job_id"] for r in res if not r["ts.xyz"]]
    miss_js = [r["job_id"] for r in res if not r["ts_result.json"]]
    nat = dict(zip(T.job_id, T.n_atoms.astype(int)))
    bad_nat = [r["job_id"] for r in res if r.get("n_atoms") is not None and nat.get(r["job_id"]) != r["n_atoms"]]
    not_target = sorted(ok_s - set(T.job_id))
    ok = (len(jobs) == N_G2_OK and ok_s == ok_i and not miss_ts and not miss_js)
    return dict(ok=ok, n_ok_status=len(ok_s), n_ok_inputs_csv=len(ok_i), expected=N_G2_OK, sets_equal=ok_s == ok_i,
                status_counts=S.status.value_counts().to_dict(), missing_ts_xyz=miss_ts[:50], n_missing_ts_xyz=len(miss_ts),
                missing_ts_result=miss_js[:50], n_missing_ts_result=len(miss_js),
                n_atoms_ne_label=len(bad_nat), n_atoms_ne_label_ids=bad_nat[:20], ok_not_in_targets=not_target[:20],
                g2_jobs=str(C.G2_JOBS), status_csv=str(G2_STATUS))


# ---------------------------------------------------------------- 0-4
def check_0_4():
    import yaml
    out = {}
    for name, p in (("d1_run", C.D1_CONFIG), ("g2", C.G2_CONFIG)):
        rec = dict(path=str(p))
        try:
            cfg = yaml.safe_load(Path(p).read_text())
            t = cfg.get("ts_stage") or {}
            rec.update(ok=isinstance(cfg, dict) and bool(t), sha256=C.sha256(p),
                       ts_stage={k: t.get(k) for k in ("engine", "functional", "dispersion", "opt_basis", "sp_basis",
                                                      "solvent", "solvation", "hmethod_conformers", "orca_optts_block")},
                       maxcore_mb=cfg.get("maxcore_mb"), cpus_per_task=cfg.get("cpus_per_task"))
        except Exception as e:                                # noqa: BLE001
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        out[name] = rec
    return dict(ok=all(r["ok"] for r in out.values()), configs=out)


# ---------------------------------------------------------------- 0-6
def _symbols(p):
    lines = Path(p).read_text().splitlines()
    n = int(lines[0].split()[0])
    return [ln.split()[0] for ln in lines[2:2 + n]]


def _charge_mult(p):
    """'* xyz c m' / '*xyz c m' and FRAGk_C / FRAGk_M of an ORCA input."""
    out = {}
    for ln in Path(p).read_text().splitlines():
        t = ln.replace("*xyz", "* xyz").split()
        if len(t) >= 4 and t[0] == "*" and t[1].lower() in ("xyz", "xyzfile"):
            out["charge"], out["mult"] = int(t[2]), int(t[3])
        elif len(t) >= 2 and t[0].upper() in ("FRAG1_C", "FRAG1_M", "FRAG2_C", "FRAG2_M"):
            out[t[0].upper()] = int(t[1])
    return out


def _elements_one(row):
    from ase.data import atomic_numbers
    rxn, job = int(row["rxn_id"]), row["job_id"]
    d = C.DFT_FINAL / str(rxn)
    out = dict(rxn_id=rxn, job_id=job, dipole=row["dipole"], skeleton=row["skeleton"], err=[])
    syms = {}
    for k, n in (("ts", "ts.xyz"), ("rel1", "rel1.xyz"), ("rel2", "rel2.xyz")):
        try:
            syms[k] = _symbols(d / n)
        except Exception as e:                                # noqa: BLE001
            out["err"].append(f"{k}: {type(e).__name__}: {e}")
            syms[k] = None
    out["count"] = {k: dict(Counter(v)) if v is not None else None for k, v in syms.items()}
    out["n_el"] = {k: (sum(atomic_numbers[s] for s in v) if v is not None else None) for k, v in syms.items()}
    out["n_atoms"] = {k: (len(v) if v is not None else None) for k, v in syms.items()}
    inp = {}
    for k in ("eda", "frag1_rel", "frag2_rel"):
        p = C.D1HF / "inputs" / job / "orig" / f"{k}.inp"
        try:
            inp[k] = _charge_mult(p)
        except FileNotFoundError:
            inp[k] = None
        except Exception as e:                                # noqa: BLE001
            inp[k] = dict(error=f"{type(e).__name__}: {e}")
    out["inp"] = inp
    return out


def check_0_6(pool, T, labels):
    expected = set(C.ELEMENTS)
    lab = {int(r["rxn_id"]): r for r in labels}
    rows = T[["rxn_id", "job_id", "dipole", "skeleton"]].to_dict("records")
    res = pool.map(_elements_one, rows, chunksize=32)
    tot, per_kind = Counter(), {k: Counter() for k in ("ts", "rel1", "rel2")}
    per_dip, per_sk = {}, {}
    errors, odd, nat_bad, sum_bad = [], [], [], []
    inp_missing, inp_bad = [], []
    for r in res:
        if r["err"]:
            errors.append(f"{r['rxn_id']}: {'; '.join(r['err'])}")
            continue
        L = lab[r["rxn_id"]]
        for k in ("ts", "rel1", "rel2"):
            per_kind[k].update(r["count"][k])
            if r["n_el"][k] % 2:
                odd.append(f"{r['rxn_id']}:{k}")
        tot.update(r["count"]["ts"])
        if (r["n_atoms"]["ts"] != int(L["n_atoms"]) or r["n_atoms"]["rel1"] != int(L["n_f1"])
                or r["n_atoms"]["rel2"] != int(L["n_f2"])):
            nat_bad.append(r["rxn_id"])
        if Counter(r["count"]["rel1"]) + Counter(r["count"]["rel2"]) != Counter(r["count"]["ts"]):
            sum_bad.append(r["rxn_id"])
        for grp, key in ((per_dip, r["dipole"]), (per_sk, r["skeleton"])):
            g = grp.setdefault(key, dict(n_reactions=0, atoms=Counter(), reactions_with=Counter()))
            g["n_reactions"] += 1
            g["atoms"].update(r["count"]["ts"])
            g["reactions_with"].update(set(r["count"]["ts"]))
        inp = r["inp"]
        if any(v is None for v in inp.values()):
            inp_missing.append(r["rxn_id"])
        else:
            e, f1, f2 = inp["eda"], inp["frag1_rel"], inp["frag2_rel"]
            good = (e.get("charge") == 0 and e.get("mult") == 1 and e.get("FRAG1_C") == 0 and e.get("FRAG2_C") == 0
                    and e.get("FRAG1_M") == 1 and e.get("FRAG2_M") == 1 and f1.get("charge") == 0
                    and f1.get("mult") == 1 and f2.get("charge") == 0 and f2.get("mult") == 1)
            if not good:
                inp_bad.append(dict(rxn_id=r["rxn_id"], inp=inp))
    charges = Counter((int(lab[r]["charge1"]), int(lab[r]["charge2"])) for r in lab)
    nonzero = [r for r in lab if int(lab[r]["charge1"]) != 0 or int(lab[r]["charge2"]) != 0]
    found = set(tot) | set(per_kind["rel1"]) | set(per_kind["rel2"])
    el_ok = found == expected
    charge_ok = not nonzero
    mult_ok = not odd and not inp_bad
    ok = bool(el_ok and charge_ok and mult_ok and not errors and len(res) == N_TARGETS)

    def _c(c):
        return {e: int(c[e]) for e in sorted(c, key=lambda x: (C.ELEMENTS.index(x) if x in C.ELEMENTS else 99, x))}

    rep = dict(
        ok=ok, stop=not ok, n_targets=len(res), expected=sorted(expected), found=sorted(found),
        extra=sorted(found - expected), missing=sorted(expected - found), elements_ok=el_ok,
        atoms_total_ts=_c(tot), atoms_by_structure={k: _c(v) for k, v in per_kind.items()},
        per_dipole={k: dict(n_reactions=v["n_reactions"], atoms=_c(v["atoms"]), reactions_with=_c(v["reactions_with"]))
                    for k, v in sorted(per_dip.items())},
        per_skeleton={k: dict(n_reactions=v["n_reactions"], atoms=_c(v["atoms"]), reactions_with=_c(v["reactions_with"]))
                      for k, v in sorted(per_sk.items())},
        charge=dict(ok=charge_ok, source="labels_d1_cr2.json charge1 / charge2",
                    counts={f"{a},{b}": n for (a, b), n in charges.items()}, nonzero=nonzero[:50]),
        multiplicity=dict(ok=mult_ok, rule="even electron count (sum Z - charge) of TS, rel1, rel2 -> singlet consistent",
                          odd_electron=odd[:50], n_odd=len(odd),
                          label_inputs=dict(source="$D1HF/inputs/<job>/orig/{eda,frag1_rel,frag2_rel}.inp",
                                            n_missing=len(inp_missing), missing=inp_missing[:20],
                                            n_not_0_1=len(inp_bad), not_0_1=inp_bad[:20])),
        consistency=dict(n_atoms_ne_labels=len(nat_bad), ids=nat_bad[:20], rel1_plus_rel2_ne_ts=len(sum_bad),
                         ids_sum=sum_bad[:20]),
        read_errors=errors[:50], n_read_errors=len(errors), time=C.now())
    C.write_json(OUT_EL, rep)
    return dict(ok=ok, file=str(OUT_EL), found=rep["found"], extra=rep["extra"], missing=rep["missing"],
                charge_ok=charge_ok, multiplicity_ok=mult_ok, n_odd_electron=len(odd),
                n_label_inputs_missing=len(inp_missing), n_label_inputs_not_0_1=len(inp_bad),
                n_read_errors=len(errors), consistency=rep["consistency"],
                reactions_with_si=sum(v["reactions_with"].get("Si", 0) for v in rep["per_dipole"].values()))


# ---------------------------------------------------------------- 0-7, 0-8, folds
def check_0_7():
    p = C.RES / "0_env.json"
    if not p.exists():
        return dict(status="pending", ok=None, file=str(p), note="results/0_env.json not written yet (env_check.py)")
    e = C.read_json(p)
    return dict(status="pass" if e.get("pass_0_7") else "fail", ok=bool(e.get("pass_0_7")), file=str(p),
                mlip_ok=e.get("mlip_ok"), gpu=e.get("gpu"), cuda=e.get("cuda"), time=e.get("time"))


def check_0_8(free_gb):
    if free_gb is None:
        return dict(status="not_given", ok=None, min_gb=FREE_GB_MIN,
                    note="pass --free-gb <login mmlsquota free GB>; compute nodes cannot run mmlsquota")
    ok = float(free_gb) >= FREE_GB_MIN
    return dict(status="pass" if ok else "fail_recorded", ok=ok, free_gb=float(free_gb), min_gb=FREE_GB_MIN,
                source="login-node /usr/lpp/mmfs/bin/mmlsquota --block-size G (argument)",
                deviation=None if ok else "results/DECISIONS.md '0-8': proceed with small disk use (decided)")


def fold_sizes(T):
    return dict(n_targets=len(T), fold_T1={str(k): int(v) for k, v in T.fold_T1.value_counts().sort_index().items()},
                lodo={d: int((T.dipole == d).sum()) for d in C.DIPOLES},
                cores_without_fold=list(T.attrs.get("cores_without_fold", [])),
                n_cores=int(T.fold_core.nunique()))


# ---------------------------------------------------------------- main
def cmd_run(args):
    t0 = time.time()
    T = C.targets()
    labels = C.read_json(C.LABELS)
    checks = {}
    checks["0-1"] = check_0_1()
    checks["0-2"] = check_0_2()
    checks["0-4"] = check_0_4()
    with Pool(max(1, n_cpus())) as pool:
        checks["0-3"] = check_0_3(pool, T)
        checks["0-6"] = check_0_6(pool, T, labels)
    checks["0-5"] = dict(status="deferred", ok=None, note="ORCA 6.1.1 `! ExtOpt` is checked in C-1 (spec 1, 0-5)")
    checks["0-7"] = check_0_7()
    checks["0-8"] = check_0_8(args.free_gb)
    real = ("0-1", "0-2", "0-3", "0-4", "0-6")
    failed = [k for k in real if not checks[k]["ok"]]
    for k in real:
        checks[k]["status"] = "pass" if checks[k]["ok"] else "STOP"
    out = dict(spec="docs/specs/DEPLOY_V2_MLTS.md section 1", time=C.now(), host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"), n_cpus=n_cpus(), wall_s=round(time.time() - t0, 1),
               checks={k: checks[k] for k in sorted(checks)}, folds=fold_sizes(T),
               summary=dict(stop=bool(failed), failed=failed,
                            status={k: checks[k]["status"] for k in sorted(checks)},
                            rule="exit 0 iff 0-1..0-4 and 0-6 pass; 0-5 deferred, 0-7 pending/env, 0-8 recorded"))
    C.write_json(OUT, out)
    print({k: checks[k]["status"] for k in sorted(checks)})
    print("folds", out["folds"]["fold_T1"], "lodo", out["folds"]["lodo"])
    print("0-6 found", checks["0-6"]["found"], "extra", checks["0-6"]["extra"], "missing", checks["0-6"]["missing"])
    if failed:
        C.stop(f"start conditions failed: {failed} (see {OUT})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--free-gb", type=float, default=None, help="login-node mmlsquota free GB (limit - used)")
    args = ap.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
