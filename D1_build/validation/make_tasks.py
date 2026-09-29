#!/usr/bin/env python3
"""make_tasks.py — the validation sample, chain manifest and task list (VALIDATION_SPEC §4, §5, §6.2).

  python make_tasks.py            (sbatch via submit_make_tasks.sh; python never runs on the login node)

Writes (all committed with config_val.yaml as the pre-registration):
  val_d0_set.csv    rxn_id, stratum, subset8, n_atoms, alt_used, n_track, aromatic_backbone
  val_manifest.csv  chain jobs, in the pilot manifest format (+ exp, ade_name, meta_rxn_id)
  tasks.csv         task_id, exp, priority, type, job_id, stages (';'), arg — sorted by priority
Also copies the small pilot files the analysis needs (json, timing, xyz) to <scratch>/pilot_copy/
(the pilot scratch is read-only for the validation).

Sampling (§5): pool = labels_all ok, charge1 == charge2 == 0, not flag_async, n_atoms <= 35.
  S1 tracked & alt_used 1 | S2 tracked & alt_used 0 | S3 untracked, non-aromatic backbone | S4 untracked,
  aromatic backbone; tracked = n_track > 0 in evidence/cfg_validation_D0.csv; aromatic = any dipole
  backbone atom (stereo_utils.dipole_backbone) aromatic in the dipole SMILES.
  24 = 6 per stratum incl. the forced rxns (20 -> S1, 105 -> S4): random.Random(seed) over the sorted
  pool, strata in order S1..S4. subset8 = 2 per stratum incl. the forced rxns: a second
  random.Random(seed) over each stratum's 6, same order.
Priorities (§6.2 table). V8 and V9 are not in that table; they are QM-free and A7 is a STOP criterion,
so they run at priority 0 with V1 (executor choice, stated in EXEC_PLAN.md).
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "config_val.yaml"


def main():
    import yaml
    cfg = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
    v = cfg["val"]
    pilot = Path(v["pilot_dir"])
    sys.path.insert(0, str(pilot))
    import stereo_utils as su                          # noqa: E402
    import sample_pilot as sp                          # noqa: E402  (dipole_of, strip_maps, unspecified_db)
    from rdkit import Chem

    repo = Path(cfg["repo_root"])
    labels = json.load(open(repo / cfg["labels_all"]))
    lab = {int(r["rxn_id"]): r for r in labels}
    coley = pd.read_csv(cfg["d0_csv"])
    smi = dict(zip(coley.rxn_id.astype(int), coley.rxn_smiles))
    gact = dict(zip(coley.rxn_id.astype(int), coley.G_act))
    ntrack = pd.read_csv(v["cfg_validation_csv"]).set_index("rxn_id")["n_track"].fillna(0).astype(int).to_dict()

    # ---------------------------------------------------------------- §5 pool and strata
    P = v["pool"]
    pool = []
    for rid, r in sorted(lab.items()):
        if r["status"] != "ok" or r["n_atoms"] > P["max_atoms"]:
            continue
        if P.get("neutral", True) and (r["charge1"] != 0 or r["charge2"] != 0):
            continue
        if P.get("exclude_async", True) and r["flag_async"]:
            continue
        _, bb, dip_smi = su.dipole_backbone(smi[rid])
        m = Chem.MolFromSmiles(dip_smi)
        arom = any(a.GetIsAromatic() for a in m.GetAtoms() if a.GetAtomMapNum() in bb)
        nt = int(ntrack.get(rid, 0))
        st = ("S1" if r["alt_used"] else "S2") if nt > 0 else ("S4" if arom else "S3")
        pool.append(dict(rxn_id=rid, stratum=st, n_atoms=r["n_atoms"], alt_used=r["alt_used"],
                         n_track=nt, aromatic_backbone=arom))
    pool = pd.DataFrame(pool)
    print("pool per stratum:", pool.groupby("stratum").size().to_dict(), file=sys.stderr)
    force = {int(k): s for k, s in v["force"].items()}
    for rid, s in force.items():
        got = pool.loc[pool.rxn_id == rid, "stratum"].tolist()
        if got != [s]:
            raise SystemExit(f"STOP: forced rxn {rid} is in {got or 'no stratum (not in pool)'}, expected {s}")

    n, n_sub = int(v["n_per_stratum"]), int(v["n_subset_per_stratum"])
    rng, rng_sub = random.Random(int(v["seed"])), random.Random(int(v["seed"]))
    rows = []
    for s in ("S1", "S2", "S3", "S4"):
        ids = sorted(pool.rxn_id[pool.stratum == s].tolist())
        forced = [r for r, fs in force.items() if fs == s]
        chosen = forced + rng.sample([i for i in ids if i not in forced], n - len(forced))
        sub = forced + rng_sub.sample([i for i in chosen if i not in forced], n_sub - len(forced))
        for rid in chosen:
            rows.append(dict(pool[pool.rxn_id == rid].iloc[0]) | dict(subset8=rid in sub))
    dset = pd.DataFrame(rows)[["rxn_id", "stratum", "subset8", "n_atoms", "alt_used", "n_track", "aromatic_backbone"]]
    dset.to_csv(HERE / "val_d0_set.csv", index=False)
    print(dset.to_string(index=False), file=sys.stderr)

    # ---------------------------------------------------------------- manifest rows
    CH = ("barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft",
          "cds_dft", "formed_d1", "formed_d2", "alt_used", "rel1_file", "rel2_file")

    def d0_row(job_id, kind, rid, exp):
        L = lab[rid]
        dsmi, psmi = sp.dipole_of(smi[rid])
        return dict(job_id=job_id, kind=kind, ts_id=f"D0-{rid:04d}", cell_id="", core="D0", panel=kind, regio="",
                    R1="", R2="", charge=0, mult=1, product_smiles=sp.strip_maps(smi[rid]).split(">>")[1],
                    mapped_smiles=smi[rid], dipole_smiles=dsmi, dipolarophile_smiles=psmi,
                    n_unspec_dipole_db=sp.unspecified_db(dsmi), rxn_smiles=sp.strip_maps(smi[rid]),
                    d0_rxn_id=rid, d0_G_act_kcal=float(gact[rid]), **{f"d0_{k}": L[k] for k in CH},
                    exp=exp, ade_name="")

    pman = pd.read_csv(v["pilot_manifest"], keep_default_na=False)
    prow = {r["job_id"]: r for r in pman.to_dict("records")}

    def d1_row(job_id, src, exp, ade_name=""):
        r = dict(prow[src]); r.update(job_id=job_id, exp=exp, ade_name=ade_name)
        return r

    man, tasks = [], []
    CHAIN = "ts;ref;inputs;sp;label"

    def chain(task_id, exp, prio, row, stages=CHAIN, arg=""):
        man.append(row)
        tasks.append(dict(task_id=task_id, exp=exp, priority=prio, type="chain", job_id=row["job_id"],
                          stages=stages, arg=arg))

    def other(task_id, exp, prio, typ, arg=""):
        tasks.append(dict(task_id=task_id, exp=exp, priority=prio, type=typ, job_id="", stages="", arg=arg))

    d24 = dset.rxn_id.tolist()
    d8 = dset.rxn_id[dset.subset8].tolist()
    for j in v["v1_src_jobs"]:
        other(f"V1a_{j}", "V1a", 0, "v1a", j)
        other(f"V1b_{j}", "V1b", 0, "v1b", j)
    other("V8", "V8", 0, "v8")
    other("V9", "V9", 0, "v9")
    for rid in d24:
        other(f"V2a_{rid:04d}", "V2a", 1, "v2a", str(rid))
    for rid in d24:
        chain(f"V5_{rid:04d}", "V5", 2, d0_row(f"V5_{rid:04d}", "D0_replay", rid, "V5"))
    for rid in d24:
        chain(f"V2L0_{rid:04d}", "V2b-L0", 3, d0_row(f"V2L0_{rid:04d}", "D0_reopt_L0", rid, "V2b-L0"),
              stages=CHAIN + ";refopt")
    for rid in v["v1c_rxns"]:
        chain(f"V1c_{rid:04d}", "V1c", 3, d0_row(f"V1c_{rid:04d}", "D0_reopt_L0", int(rid), "V1c"), stages="ts")
    other(f"V7_{v['v7_job']}", "V7", 4, "v7", v["v7_job"])
    j7 = v["v6c_job"]
    chain(f"V6c_{j7}", "V6c", 4, d1_row(f"V6c_{j7}", j7, "V6c", ade_name=j7), arg=j7)
    for rid in d8:
        chain(f"V2L2_{rid:04d}", "V2b-L2", 5, d0_row(f"V2L2_{rid:04d}", "D0_reopt_L2", rid, "V2b-L2"))
    for rid in d8:
        for k in (1, 2, 3):
            chain(f"V3_{rid:04d}_r{k}", "V3", 6, d0_row(f"V3_{rid:04d}_r{k}", "D0_control", rid, "V3"))
    for rid in d24:
        if rid not in d8:
            chain(f"V4_{rid:04d}", "V4", 7, d0_row(f"V4_{rid:04d}", "D0_control", rid, "V4"))
    v6a = HERE / v["v6a_manifest"]
    if v6a.is_file():
        for r in pd.read_csv(v6a, keep_default_na=False).to_dict("records"):
            r.update(exp="V6a", ade_name="")
            chain(r["job_id"], "V6a", 8, r)
    else:
        print(f"WARNING: {v6a.name} not found — V6a (10 new D1 rows) left out; A1 is then judged on "
              f"the pilot rows only (needs D1_설계_v8.xlsx)", file=sys.stderr)
    for j in v["v6b_jobs"]:
        for k in range(2, 2 + int(v["v6b_extra_runs"])):
            chain(f"V6b_{j}_r{k}", "V6b", 9, d1_row(f"V6b_{j}_r{k}", j, "V6b"))

    man = pd.DataFrame(man)
    man.insert(len(man.columns), "meta_rxn_id", [800000 + i for i in range(len(man))])
    if man.job_id.duplicated().any():
        raise SystemExit(f"duplicate job ids: {man.job_id[man.job_id.duplicated()].tolist()}")
    man.to_csv(HERE / "val_manifest.csv", index=False)
    tk = pd.DataFrame(tasks).sort_values("priority", kind="stable")
    if tk.task_id.duplicated().any() or tk.task_id.str.contains(",").any():
        raise SystemExit("task ids must be unique and comma-free")
    tk.to_csv(HERE / "tasks.csv", index=False)
    print(tk.groupby(["priority", "exp"]).size().to_string(), file=sys.stderr)
    print(f"tasks: {len(tk)}  chain jobs: {len(man)}", file=sys.stderr)

    # ---------------------------------------------------------------- read-only pilot -> copy
    src, dst = Path(cfg["pilot_scratch"]) / "jobs", Path(cfg["scratch"]) / "pilot_copy" / "jobs"
    for jd in sorted(src.glob("J*")):
        out = dst / jd.name
        out.mkdir(parents=True, exist_ok=True)
        for pat in ("*.json", "timing.tsv", "*.xyz", ".done_*", ".fail_*", "sp/input_meta.json"):
            for f in jd.glob(pat):
                t = out / f.relative_to(jd)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, t)
    print(f"pilot small files copied to {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
