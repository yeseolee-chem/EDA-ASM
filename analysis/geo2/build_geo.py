#!/usr/bin/env python3
"""GEO2 Phase Z: the geo-rung structure folders $GEO2/geo_final/<rxn_id>/ from the GEO1 test predictions
(docs/specs/GEO2.md 4). GPU env on a CPU partition (jobs/gpu.slurm of analysis/geo; it reads the GEO1 pickles).

  python build_geo.py build     -> $GEO2/geo_final/<rxn_id>/{ts.xyz, rel1.xyz, rel2.xyz, result.json, .done}
                                   + results/Z_status.csv + results/Z_report.json   (exit 3 = STOP, spec 4-3)
Each D1 reaction gets exactly one TS: the 3-seed mean of the GEO1 'final' model of the T1 fold whose test set holds it
(folds 0-4: final_f<k>_s<seed>.pkl; fold -1: final_fm1_s<seed>.pkl of train_m1.py). rel1 / rel2 = byte copies of the
G2 reference structures (the g2 rung's). A structure that fails the sanity rules (4-2) is replaced by the G2 TS
(deployable fallback, flagged). Idempotent: an existing .done folder whose sources are unchanged is kept.
"""
from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo2_common as C  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
MIN_CONTACT = 0.60          # d_ij >= 0.60 (r_i + r_j) for every pair
BOND_IN, BOND_OUT = 1.10, 1.40   # G2 bond (d < 1.10 sum r) must stay < 1.40 sum r; G2 non-bond (> 1.40) stays > 1.10
FORM_RANGE = (1.4, 4.0)     # forming distances (A)
FALLBACK_MAX = 0.02         # spec 4-3
E_RXN_TOL = 1e-6            # A, forming-distance errors vs GEO1 E_rxn.csv


def load_preds():
    """uid -> (mean xyz, source tag, pkl sha256 list). Folds 0-4 and fold -1; seed sets must agree."""
    out = {}
    files = {f"f{k}": [C.GEO1 / "preds" / f"final_f{k}_s{s}.pkl" for s in C.SEEDS] for k in FOLDS}
    files["fm1"] = [C.GEO1 / "preds" / f"final_fm1_s{s}.pkl" for s in C.SEEDS]
    for tag, fs in files.items():
        miss = [str(f) for f in fs if not f.is_file()]
        if miss:
            C.stop(f"GEO1 prediction files missing: {miss}")
        parts = []
        for f in fs:
            with open(f, "rb") as fh:
                parts.append(pickle.load(fh))
        if any(set(p) != set(parts[0]) for p in parts):
            C.stop(f"{tag}: the seed files hold different reaction sets")
        shas = [C.sha256(f) for f in fs]
        for u in parts[0]:
            if u in out:
                C.stop(f"{u} predicted by two folds ({out[u][1]} and {tag})")
            out[u] = (np.mean([np.asarray(p[u], float) for p in parts], axis=0), tag, shas)
    return out


def sanity(syms, X, Xg2, pairs):
    """-> (ok, reason, info). X = geo TS, Xg2 = G2 TS (same atom order = label order)."""
    if X.shape != Xg2.shape or not np.isfinite(X).all():
        return False, "shape_or_nonfinite", {}
    r = np.array([C.COV[s] for s in syms])
    S = r[:, None] + r[None, :]
    D, Dg = C.dmat(X), C.dmat(Xg2)
    iu = np.triu_indices(len(syms), 1)
    ratio = (D / S)[iu]
    info = dict(min_contact_ratio=float(ratio.min()))
    if ratio.min() < MIN_CONTACT:
        return False, "close_contact", info
    fp = {tuple(p) for p in pairs}
    mask = np.array([(i, j) not in fp for i, j in zip(*iu)])
    rg, rx = (Dg / S)[iu][mask], (D / S)[iu][mask]
    broke = int(((rg < BOND_IN) & (rx > BOND_OUT)).sum())
    made = int(((rg > BOND_OUT) & (rx < BOND_IN)).sum())
    info.update(n_bonds_broken=broke, n_bonds_made=made)
    if broke or made:
        return False, "bond_graph_changed", info
    d = C.form_d(X, pairs)
    info["d_form"] = d.tolist()
    if d.min() < FORM_RANGE[0] or d.max() > FORM_RANGE[1]:
        return False, "forming_distance_range", info
    return True, "", info


def e_sorted(X, lab, pairs):
    dl = C.form_d(lab, pairs)
    o = np.argsort(dl)
    return (C.form_d(X, pairs) - dl)[o]


def build():
    t0 = time.time()
    labs = json.loads(C.D1_LABELS_G2.read_text())                       # 9,266 D1 targets (job_id, rxn_id, pairs)
    P = load_preds()
    erxn_path = C.GEO1_RES / "E_rxn.csv"
    erxn = {}
    if erxn_path.is_file():
        with open(erxn_path, newline="") as f:
            for row in csv.DictReader(f):
                erxn[int(row["rxn_id"])] = (float(row["e_short_model"]), float(row["e_long_model"]))
    else:
        C.stop(f"{erxn_path} missing (GEO1 Phase E must be finished)")
    rows, n_fb, e_dev, n_e = [], 0, 0.0, 0
    for d in labs:
        rid, T = int(d["rxn_id"]), d["job_id"]
        uid = f"D1_{rid}"
        g2 = C.G2_FINAL / str(rid)
        row = dict(rxn_id=rid, job_id=T, status="", source="", fallback=False, reason="")
        if C.marker(g2) != ".done":
            row.update(status="no_g2")
            rows.append(row)
            continue
        if uid not in P:
            row.update(status="no_geo1_prediction")
            rows.append(row)
            continue
        X, tag, shas = P[uid]
        syms_l, lab = C.read_xyz(C.D1_JOBS / T / "ts.xyz")
        syms_g, Xg2 = C.read_xyz(g2 / "ts.xyz")
        if list(syms_g) != list(syms_l):
            C.stop(f"{rid}: G2 ts.xyz element order != label")
        pairs = C.parse_pairs(d["formed_pairs_ts"])
        if rid in erxn:                                                  # consistency with the GEO1 evaluation
            dev = float(np.abs(e_sorted(X, lab, pairs) - np.array(erxn[rid])).max())
            e_dev, n_e = max(e_dev, dev), n_e + 1
        ok, reason, info = sanity(syms_l, X, Xg2, pairs)
        Xuse = X if ok else Xg2
        n_fb += 0 if ok else 1
        out = C.GEO_FINAL / str(rid)
        src = {"g2_rel1": C.sha256(g2 / "rel1.xyz"), "g2_rel2": C.sha256(g2 / "rel2.xyz"), "pkl": shas}
        if C.marker(out) == ".done":
            old = json.loads((out / "result.json").read_text())
            if old.get("sources") == src and old.get("fallback") == (not ok):
                row.update(status="kept", source=tag, fallback=not ok, reason=reason)
                rows.append(row)
                continue
            C.stop(f"{out} exists with other sources: move it aside explicitly (never automatic)")
        tmp = C.GEO_FINAL / f"{rid}.tmp"
        C.assert_path(tmp)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        (tmp / "ts.xyz").write_text(C.xyz_text(syms_l, Xuse, f"geo TS rxn {rid} ({'G2 fallback: ' + reason if not ok else 'GEO1 ' + tag + ' 3-seed mean'})"))
        for k in (1, 2):
            shutil.copyfile(g2 / f"rel{k}.xyz", tmp / f"rel{k}.xyz")
        heavy = [i for i, s in enumerate(syms_l) if s != "H"]
        res = dict(rxn_id=rid, job_id=T, geom="geo", source=f"GEO1 {tag} seeds {list(C.SEEDS)} mean", fold_tag=tag,
                   fallback=not ok, fallback_reason=reason, sanity=info, sources=src,
                   formed_pairs_label=[list(p) for p in pairs], d_form_geo=C.form_d(X, pairs).tolist(),
                   d_form_used=C.form_d(Xuse, pairs).tolist(), d_form_g2=C.form_d(Xg2, pairs).tolist(),
                   d_form_label=C.form_d(lab, pairs).tolist(),
                   rmsd_heavy_label_geo=C.kabsch_rmsd(X[heavy], lab[heavy]),
                   rmsd_heavy_label_g2=C.kabsch_rmsd(Xg2[heavy], lab[heavy]),
                   status="ok", written=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (tmp / "result.json").write_text(json.dumps(res, indent=1, default=C._jsonable))
        (tmp / ".done").write_text(res["written"] + "\n")
        if out.exists():
            shutil.rmtree(C.assert_path(out))
        tmp.replace(out)
        row.update(status="ok", source=tag, fallback=not ok, reason=reason)
        rows.append(row)
    n_ok = sum(r["status"] in ("ok", "kept") for r in rows)
    stops = []
    if n_e == 0 or e_dev > E_RXN_TOL:
        stops.append(f"forming-distance errors differ from GEO1 E_rxn.csv (max {e_dev:.2e} A over {n_e} rxns)")
    if n_ok and n_fb / n_ok > FALLBACK_MAX:
        stops.append(f"fallback rate {n_fb / n_ok:.3f} > {FALLBACK_MAX}")
    C.RES.mkdir(parents=True, exist_ok=True)
    with open(C.RES / "Z_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    rep = dict(n_targets=len(labs), n_folders=n_ok, by_status=dict(Counter(r["status"] for r in rows)),
               by_source=dict(Counter(r["source"] for r in rows if r["source"])), n_fallback=n_fb,
               fallback_reasons=dict(Counter(r["reason"] for r in rows if r["fallback"])),
               e_rxn_check=dict(n=n_e, max_abs_dev_A=e_dev, tol=E_RXN_TOL), stop=bool(stops), stop_reasons=stops,
               wall_s=time.time() - t0)
    C.write_json(C.RES / "Z_report.json", rep)
    print(json.dumps(rep, indent=1, default=str), flush=True)
    return 3 if stops else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        sys.exit(build())
    raise SystemExit(__doc__)
