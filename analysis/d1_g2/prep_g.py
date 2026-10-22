#!/usr/bin/env python3
"""D1-G2 inputs of the rev 4-6 feature code on G2 (docs/specs/D1G2.md 8) and the run lists (5, 6).
Run through sbatch only (jobs/g2.slurm, G2ENV=d1).

  python prep_g.py inputs   # $G2/G/labels_d1_g2.json, $G2/G/input_meta_d1_g2.csv, $G2/prof/<rxn_id> -> D1_JOBS/<job_id>
                            #   + results/G_inputs_report.json (exit 3 = STOP)
  python prep_g.py slist    # results/S_list.csv (spec 5)
  python prep_g.py order    # results/G_order.csv (spec 6)

inputs = the D1PAIR Phase G adapter files (d1_g1.py prep) for all 9,266 targets instead of the 8,659 D1PAIR G1 rows:
  labels_d1_g2.json   the target rows of labels_d1.json, verbatim (d1pair_common.load_d1_labels; no D'' skeleton)
  input_meta_d1_g2.csv  rxn_id, A_idx = the atoms tagged (1) in D1_JOBS/<job_id>/sp/eda.inp (frag1 = dipole), 0-based
  prof/<rxn_id>       symlink -> D1_JOBS/<job_id> (the label ts_file / rel1_file / rel2_file are relative to it)
  gates (as d1_g1 G-1): eda.inp elements = ts.xyz, coordinates within 1e-6 A, #(1) = n_f1, #(2) = n_f2; the rows of
  the 8,659 D1PAIR targets must equal labels_d1_g.json / input_meta_d1.csv / d1_pair/prof (same values, same link).
  ESPLEY_SMILES is read only by g1_geom (G1 construction, not run here) and stays the D1PAIR file.
slist: per dipole (sorted) numpy.random.default_rng(20261022).choice(its job_ids in job_id order, 2, replace=False);
  then for each of F, Si, S, Cl, Br (in that order) absent from the 10: the last selected reaction whose removal keeps
  every element already covered is swapped for rng.choice(the same dipole's unselected reactions containing it), the
  next one from the end if that dipole has none. Elements = D1_JOBS/<job_id>/ts.xyz.
order: D1PAIR reference-pair reactions (G_targets.csv in_ref_pairs), then anchor-pair reactions (in_anchor_pairs), then
  the rest; job_id order inside each group.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_common as G  # noqa: E402

DP = G.DP
import d1_phase_p as PP  # noqa: E402                  (parse_eda_inp: the D1PAIR eda.inp reader)

G_DIR, PROF = G.G2 / "G", G.G2 / "prof"
LABELS, META = G_DIR / "labels_d1_g2.json", G_DIR / "input_meta_d1_g2.csv"
S_SEED, S_PER_DIPOLE, S_ELEMENTS = 20261022, 2, ("F", "Si", "S", "Cl", "Br")


def read_xyz(p):
    L = Path(p).read_text().splitlines()
    n = int(L[0].split()[0])
    rows = [ln.split() for ln in L[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def cmd_inputs():
    tg = G.targets()
    X = pd.read_csv(G.RES / "X_targets.csv", dtype={"job_id": str})
    if [d["job_id"] for d in tg] != list(X.job_id):
        G.stop("targets != results/X_targets.csv")
    G_DIR.mkdir(parents=True, exist_ok=True)
    PROF.mkdir(parents=True, exist_ok=True)
    G.assert_path(PROF / "x")
    meta, bad, n_link_new = [], [], 0
    for d in tg:
        j, rid = d["job_id"], int(d["rxn_id"])
        jd = DP.D1_JOBS / j
        try:
            s1, x1 = read_xyz(jd / d["ts_file"])
            s2, tags, x2 = PP.parse_eda_inp(jd / "sp" / "eda.inp")
            n1, n2 = int((tags == 1).sum()), int((tags == 2).sum())
            dev = float(np.abs(x2 - x1).max()) if s1 == list(s2) else np.inf
            why = [k for k, ok in (("elements", s1 == list(s2)), ("coords", dev <= DP.G1_TOL_COORD_A),
                                   ("n_f1", n1 == int(d["n_f1"])), ("n_f2", n2 == int(d["n_f2"]))) if not ok]
        except Exception as e:                                       # noqa: BLE001
            why, tags = [f"{type(e).__name__}: {e}"], None
        if why:
            bad.append(f"{j}: {why}")
            continue
        meta.append(dict(rxn_id=rid, A_idx=" ".join(str(i) for i in np.where(tags == 1)[0])))
        link = PROF / str(rid)
        if link.is_symlink() or link.exists():
            if os.readlink(link) != str(jd):
                bad.append(f"{j}: {link} -> {os.readlink(link)} != {jd}")
        else:
            os.symlink(jd, link)
            n_link_new += 1
    extra = sorted(set(p.name for p in PROF.iterdir()) - {str(int(d["rxn_id"])) for d in tg})
    if extra:
        bad.append(f"prof has non-target entries: {extra[:10]}")
    M = pd.DataFrame(meta, columns=["rxn_id", "A_idx"])
    if not bad:
        G.write_atomic(META, lambda t: M.to_csv(t, index=False))
        G.write_atomic(LABELS, json.dumps(tg))
    # the D1PAIR rows must be identical
    lab0 = {int(d["rxn_id"]): d for d in json.loads(DP.D1G_LABELS.read_text())}
    meta0 = dict(pd.read_csv(DP.D1G_META, dtype={"A_idx": str}).astype({"rxn_id": int}).values.tolist())
    mine = {int(d["rxn_id"]): d for d in tg}
    m1 = dict(zip(M.rxn_id.astype(int), M.A_idx))
    lab_diff = [r for r, d in lab0.items() if mine.get(r) != d]
    meta_diff = [r for r, a in meta0.items() if m1.get(r) != a]
    link_diff = [r for r in lab0 if not (DP.PROF_D1 / str(r)).is_symlink()
                 or os.readlink(DP.PROF_D1 / str(r)) != os.readlink(PROF / str(r))] if not bad else []
    bad += [f"labels differ from {DP.D1G_LABELS}: {lab_diff[:10]} ({len(lab_diff)})"] if lab_diff else []
    bad += [f"A_idx differs from {DP.D1G_META}: {meta_diff[:10]} ({len(meta_diff)})"] if meta_diff else []
    bad += [f"prof links differ from {DP.PROF_D1}: {link_diff[:10]} ({len(link_diff)})"] if link_diff else []
    rep = dict(n_targets=len(tg), n_meta=len(M), n_prof_links=len(list(PROF.iterdir())), n_prof_new=n_link_new,
               n_d1pair_rows=len(lab0), n_d1pair_rows_same=len(lab0) - len(lab_diff), n_fail=len(bad), fail=bad[:50],
               outputs={str(p): G.sha256(p) for p in (LABELS, META) if p.is_file()} if not bad else {},
               sources={str(DP.D1_LABELS): DP.D1_LABELS_SHA, str(DP.D1G_LABELS): G.sha256(DP.D1G_LABELS),
                        str(DP.D1G_META): G.sha256(DP.D1G_META)},
               rule="d1_g1.py prep (G-1 a, b) for all targets; labels verbatim; ESPLEY_SMILES unchanged (g1_geom only)")
    rep["stop"] = bool(bad)
    G.write_json(G.RES / "G_inputs_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "sources"}, indent=1), flush=True)
    return 3 if bad else 0


def elements(job_id):
    return set(read_xyz(DP.D1_JOBS / job_id / "ts.xyz")[0])


def cmd_slist():
    X = pd.read_csv(G.RES / "X_targets.csv", dtype={"job_id": str}).sort_values("job_id")
    el = {j: elements(j) for j in X.job_id}
    rng = np.random.default_rng(S_SEED)
    sel, how = [], {}
    for dip in sorted(X.dipole.unique()):
        for j in rng.choice(sorted(X[X.dipole == dip].job_id), size=S_PER_DIPOLE, replace=False):
            sel.append(str(j))
            how[str(j)] = "rng"
    dip_of = dict(zip(X.job_id, X.dipole))
    swaps = []
    for e in S_ELEMENTS:
        if any(e in el[j] for j in sel):
            continue
        covered = {x for x in S_ELEMENTS if any(x in el[j] for j in sel)}
        done = False
        for k in range(len(sel) - 1, -1, -1):
            rest = sel[:k] + sel[k + 1:]
            if not covered <= {x for x in S_ELEMENTS if any(x in el[j] for j in rest)}:
                continue
            cand = sorted(j for j in X.job_id if dip_of[j] == dip_of[sel[k]] and e in el[j] and j not in sel)
            if not cand:
                continue
            new = str(rng.choice(cand))
            swaps.append(dict(element=e, out=sel[k], into=new, dipole=dip_of[new]))
            how[new] = f"swap_{e}:{sel[k]}"
            sel[k] = new
            done = True
            break
        if not done:
            G.stop(f"S list: no reaction with {e} can be swapped in")
    S = X.set_index("job_id").loc[sel].reset_index()[["job_id", "rxn_id", "dipole", "skeleton", "n_atoms", "charge"]]
    S["elements"] = [" ".join(sorted(el[j])) for j in sel]
    S["picked_by"] = [how[j] for j in sel]
    G.write_atomic(G.RES / "S_list.csv", lambda t: S.to_csv(t, index=False))
    print(S.to_string(index=False), "\nswaps:", swaps, flush=True)
    return 0


def cmd_order():
    X = pd.read_csv(G.RES / "X_targets.csv", dtype={"job_id": str})
    T = pd.read_csv(DP.G_TARGETS, dtype={"job_id": str}).set_index("job_id")
    ref, anc = set(T.index[T.in_ref_pairs.astype(bool)]), set(T.index[T.in_anchor_pairs.astype(bool)])
    X["group"] = [0 if j in ref else 1 if j in anc else 2 for j in X.job_id]
    X["group_name"] = X.group.map({0: "ref_pairs", 1: "anchor_pairs", 2: "rest"})
    O = X.sort_values(["group", "job_id"])[["job_id", "rxn_id", "group_name", "n_atoms", "dipole"]].reset_index(drop=True)
    O.insert(0, "order", range(len(O)))
    G.write_atomic(G.RES / "G_order.csv", lambda t: O.to_csv(t, index=False))
    print(O.group_name.value_counts().to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    cmds = {"inputs": cmd_inputs, "slist": cmd_slist, "order": cmd_order}
    if sys.argv[1:2] and sys.argv[1] in cmds:
        sys.exit(cmds[sys.argv[1]]())
    raise SystemExit(__doc__)
