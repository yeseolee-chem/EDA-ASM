#!/usr/bin/env python3
"""Phase Z: D0-G2 = the D1-G2 protocol (analysis/d1_g2/g2_autode.py imported, compute unchanged) on the D0 reactions.
sbatch only, D1 run env (jobs/cpu.slurm).

  python d0g2.py manifest                        # $D0G2/d0_manifest.csv, $D0G2/G/labels_d0_g2.json, $D0G2/labjobs/
  python d0g2.py pilot_list                      # results/Z_pilot.csv
  python d0g2.py one <job_id> <cores>            # one reaction (g2_autode.cmd_one with the D0 overrides)
  python d0g2.py pool <list.csv> <n_par> <cores> [i n]
  python d0g2.py report                          # results/Z_status.csv, results/Z_report.json
"""
from __future__ import annotations

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
import geo_common as GC  # noqa: E402

os.environ["D1G2_SCRATCH"] = str(GC.D0G2)          # MUST precede the g2 imports (g2_common.G2 is read at import)
sys.path.insert(0, str(GC.REPO / "analysis" / "d1_g2"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import g2_common as G  # noqa: E402
import g2_autode as GA  # noqa: E402
import g2_refinalize as GR  # noqa: E402  (decision Z-graph: sets GA._graph = GR._graph_fixed, the D1-G2 Phase G repair)

assert GA._graph is GR._graph_fixed, "the D1-G2 _graph repair (g2_refinalize) is not active"
assert G.G2 == GC.D0G2 and str(G.G2) != "/gpfs/tmp_cpu2/yeseo1ee/d1_g2", "D0G2 scratch must not be the D1-G2 scratch"
MANIFEST = GC.D0G2 / "d0_manifest.csv"
LABJOBS = GC.D0G2 / "labjobs"                       # <job_id>/{ts.xyz, ref_dipole.xyz, ref_dipolarophile.xyz} = Coley
LABELS = GC.D0G2 / "G" / "labels_d0_g2.json"

# ---- overrides (this process only); every name below is looked up by g2_autode at call time
G.MANIFEST = MANIFEST                               # cmd_one (pc.load_job) and stage_stereo (--manifest)
G.G2_FINAL, G.G2_JOBS, G.LOGS = GC.D0G2 / "g2_final", GC.D0G2 / "jobs", GC.D0G2 / "logs"
G.DP.D1_JOBS = LABJOBS                              # finalize reads the label TS / references from here
G.targets = lambda: json.loads(LABELS.read_text())  # cmd_one: lab = {job_id: row}
_META = {}


def _lab_A_idx(job):                                # finalize -> lab_A_idx(job): label partition (frag1 = dipole)
    if not _META:
        _META.update(GC.load_meta(GC.D0_META))
    return _META[int(float(job["d0_rxn_id"]))]


GA.lab_A_idx = _lab_A_idx
SKIP_STEREO = os.environ.get("D0G2_SKIP_STEREO") == "1"   # decision Z-2 only (5-4); default off
_orig_stereo = GA.stage_stereo


def _stage_stereo(cfg_path, job, jd):
    if SKIP_STEREO:
        (jd / "stereo_skipped.txt").write_text("D0G2_SKIP_STEREO=1 (decision Z-2)\n")
        GA.pc.mark_done(jd, "stereo")
        return True
    return _orig_stereo(cfg_path, job, jd)


GA.stage_stereo = _stage_stereo


def _atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cmd_manifest():
    sys.path.insert(0, str(G.PILOT))
    import sample_pilot as sp                        # dipole_of, strip_maps, unspecified_db (as make_tasks.d0_row)
    hdr = list(pd.read_csv(G.D1_REPO / "D1_build" / "validation" / "val_manifest.csv", nrows=0).columns)
    labs, smi = GC.load_d0_labels(), GC.load_d0_smiles()
    rows, tg = [], []
    for L in labs:
        rid = int(L["rxn_id"])
        jid, s = f"D0G2_{rid:04d}", smi[rid]
        dsmi, psmi = sp.dipole_of(s)
        r = {k: "" for k in hdr}
        r.update(job_id=jid, kind="D0_control", ts_id=f"D0-{rid:04d}", core="D0", panel="D0_G2",
                 charge=int(L["charge1"]) + int(L["charge2"]), mult=1,     # make_tasks.d0_row hard-codes 0: not here
                 product_smiles=sp.strip_maps(s).split(">>")[1], mapped_smiles=s, dipole_smiles=dsmi,
                 dipolarophile_smiles=psmi, n_unspec_dipole_db=sp.unspecified_db(dsmi), rxn_smiles=sp.strip_maps(s),
                 d0_rxn_id=rid, exp="D0G2", ade_name="", meta_rxn_id=rid)
        for k in ("barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft",
                  "cds_dft", "formed_d1", "formed_d2", "alt_used", "rel1_file", "rel2_file"):
            if f"d0_{k}" in r and k in L:                                   # informational columns only
                r[f"d0_{k}"] = L[k]
        rows.append(r)
        jd = LABJOBS / jid
        jd.mkdir(parents=True, exist_ok=True)
        pdir = GC.D0_PROF / str(rid)
        shutil.copy(pdir / L["ts_file"], jd / "ts.xyz")
        rel = {L["role1"]: L["rel1_file"], L["role2"]: L["rel2_file"]}
        shutil.copy(pdir / rel["dipole"], jd / "ref_dipole.xyz")
        shutil.copy(pdir / rel["dipolarophile"], jd / "ref_dipolarophile.xyz")
        tg.append(dict(job_id=jid, rxn_id=rid, formed_pairs_ts=L["formed_pairs_ts"], charge1=int(L["charge1"]),
                       charge2=int(L["charge2"]), n_f1=int(L["n_f1"]), n_f2=int(L["n_f2"]), ts_file="ts.xyz",
                       dipole="D0", skeleton="D0"))
    _atomic_csv(pd.DataFrame(rows, columns=hdr), MANIFEST)
    GC.write_json(LABELS, tg)
    print(f"manifest {len(rows)} rows -> {MANIFEST}", flush=True)
    return 0


def cmd_pilot_list():
    labs = GC.load_d0_labels()
    df = pd.DataFrame(dict(rxn_id=[int(d["rxn_id"]) for d in labs], n_atoms=[int(d["n_atoms"]) for d in labs],
                           charged=[int(d["charge1"]) + int(d["charge2"]) != 0 for d in labs]))
    df["q"] = pd.qcut(df.n_atoms, 4, labels=False, duplicates="drop")
    rng = np.random.default_rng(20261026)
    pick = []
    for q in sorted(df.q.unique()):
        pool = sorted(df[(df.q == q) & ~df.charged].rxn_id)
        pick += sorted(int(v) for v in rng.choice(pool, 5, replace=False))
    pick += sorted(int(v) for v in rng.choice(sorted(df[df.charged].rxn_id), 2, replace=False))
    _atomic_csv(pd.DataFrame(dict(job_id=[f"D0G2_{r:04d}" for r in pick])), GC.RES / "Z_pilot.csv")
    return 0


def cmd_one(T, cores):
    return GA.cmd_one(T, cores)


def cmd_pool(lst, n_par, cores, i=0, m=1):
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()[int(i)::int(m)]
    lab = {d["job_id"]: d for d in G.targets()}
    t0, stop_h = time.time(), float(os.environ.get("D0G2_STOP_AFTER_H", "44"))

    def one(T):
        fin = G.G2_FINAL / str(lab[T]["rxn_id"])
        if (fin / ".done").is_file() or list(fin.glob(".fail_*")):
            return T, "skip"
        if time.time() - t0 > stop_h * 3600:
            return T, "not_started"
        log = G.LOGS / "rxn" / f"{T}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", T, str(cores)],
                               stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        fs = [p.name for p in fin.glob(".*")] if fin.is_dir() else []
        return T, (fs[0] if fs else f"error rc={r.returncode}")

    res = {}
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for fu in as_completed([ex.submit(one, T) for T in ids]):
            T, s = fu.result()
            k = ".fail" if s.startswith(".fail") else s
            res[k] = res.get(k, 0) + 1
            print(f"  {T}: {s}", flush=True)
    print(f"[d0g2 pool] {res}", flush=True)
    return 2 if res.get("not_started") or any(k.startswith("error") for k in res) else 0


def cmd_report():
    """status / core-h / forming distances (label atom order, no isomorphism search). The 'same TS' judgment of
    D0 G2 vs D0 G1 (rmsd.same_ts) is made in data.py, which imports rmsd before anything else (fragmenter path)."""
    rows = []
    for L in GC.load_d0_labels():
        rid = int(L["rxn_id"])
        fin = G.G2_FINAL / str(rid)
        r = dict(rxn_id=rid, status="missing", n_atoms=int(L["n_atoms"]))
        if (fin / "result.json").is_file():
            js = json.loads((fin / "result.json").read_text())
            r.update(status=js.get("status"), core_h=js.get("core_h_total"), remapped=js.get("remapped"),
                     fail=";".join(js.get("fail_reasons", [])))
        if GC.done(fin):
            pairs = GC.parse_pairs(L["formed_pairs_ts"])
            g2 = GC.read_xyz(fin / "ts.xyz")
            lab = GC.read_xyz(GC.D0_PROF / str(rid) / L["ts_file"])
            dg2 = np.sort(GC.form_d(g2[1], pairs))
            r.update(dform_label=float(np.abs(dg2 - np.sort(GC.form_d(lab[1], pairs))).max()))
            g1 = GC.read_if_done(GC.D0_G1_ROOT / str(rid))
            if g1 is not None and list(g1[0]) == list(g2[0]):
                r.update(dform_g1=float(np.abs(dg2 - np.sort(GC.form_d(g1[1], pairs))).max()))
        rows.append(r)
    D = pd.DataFrame(rows)
    D["core_h"] = pd.to_numeric(D.get("core_h"), errors="coerce")
    _atomic_csv(D, GC.RES / "Z_status.csv")
    ok = D[D.status == "ok"]
    med = lambda c: float(ok[c].median()) if c in ok and ok[c].notna().any() else None   # noqa: E731
    GC.write_json(GC.RES / "Z_report.json", dict(
        n=len(D), by_status={str(k): int(v) for k, v in D.status.value_counts().items()},
        ok_rate=float((D.status == "ok").mean()),
        core_h=dict(sum=float(D.core_h.sum()), mean=float(D.core_h.mean()), median=float(D.core_h.median()),
                    p90=float(D.core_h.quantile(.9))),
        dform_label_median=med("dform_label"), dform_g1_median=med("dform_g1"),
        share_dform_g1_le_002=float((ok.dform_g1 <= 0.02).mean()) if "dform_g1" in ok else None,
        reference_D1=dict(g2_ok_rate=0.930, same_g1_rate=0.561, core_h_mean=0.336)))   # C_compare, comparison only
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmds = dict(manifest=(cmd_manifest, 0), pilot_list=(cmd_pilot_list, 0), one=(cmd_one, 2), report=(cmd_report, 0))
    if a and a[0] == "pool" and len(a) in (4, 6):
        sys.exit(cmd_pool(*a[1:]))
    if a and a[0] in cmds and len(a) - 1 == cmds[a[0]][1]:
        sys.exit(cmds[a[0]][0](*a[1:]))
    raise SystemExit(__doc__)
