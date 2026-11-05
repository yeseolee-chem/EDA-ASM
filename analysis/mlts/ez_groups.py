#!/usr/bin/env python3
"""ez_groups.py — DEPLOY-V2 (MLTS): E/Z (stereo) sibling groups of the fold-k FT-2 test reactions (user 11-05 request
(2); read by pipeline.decide_ft2p and ts_geom.summarize). Compute nodes only (sbatch jobs/mlt.slurm, MLTENV=mlts; CPU,
about a minute).

  python ez_groups.py run [--fold 0] [--out $MLT/pipeline/lists/f<k>_test_ez.csv]

Rule: a fold-k test reaction (mlt_common.targets fold_T1 == k = pipeline list f<k>_test.csv, every reaction FT-2
measures) is 'with sibling' (ez_sibling_in_train = True) when a training-side D1 core reaction — every D1 target with
fold_T1 != k: the other folds' cores, the always-training -1 anchor-only cores, and among them the fold-k validation
cores — has the same stereo-free reactant-pair identity but a different stereo-kept identity. Identities are the leak
check's (verify_data.rxn_idents on the labels_d1_cr2 rxn_smiles, manifest as fallback): pair_smi = RDKit canonical
SMILES of each reactant, atom maps and stereo removed, sorted pair; pair_smi_iso = isomeric canonical SMILES (stereo
kept). Example: nitrile oxide/AppE (fold 0 test) vs nitrile oxide/AppZ (fold 2, a fold-0 validation core). The InChI
pair (pair_inchi_ns / pair_inchi_iso) is computed as a cross-check (inchi_agrees).
Columns: rxn_id, role (a | b | a+b | none: the S_folds fold_T1 == k pair roles, a_rxn = evaluated reaction, b_rxn = the
pair's reference reaction, none = fold-k target in no pair), is_a, is_b, dipole, fold_core, ez_sibling_in_train,
matching_train_core (';'-joined fold_core of the matching training-side reactions), n_sibling_rxns, sibling_rxns
(first 50), sibling_folds, sibling_in_valid_core (a matching reaction is in a fold-k validation core: validation rows,
not trained on, they select the checkpoint), same_pair_iso_in_train (the same stereo-kept pair on the training side =
a cross-fold leak, expected 0), identity_ok, inchi_agrees.
Report <out>.json: counts with / without by role (a, b; a reaction in both roles counts in both), validation cores
(fold_core of the D1 rxn_ids of $MLT/ft/data/fold<k>/valid.xyz), matching cores, examples. Reads only; writes under
$MLT. gen_*.py, finetune.py, mlt_common.py, orca_io.py are not modified (verify_data's identity functions are imported)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import verify_data as V  # noqa: E402

CODE = "mlts-ez-1"
RULE = ("with sibling = a training-side D1 target (fold_T1 != k, incl. fold -1 anchor-only cores and the fold-k "
        "validation cores) has the same stereo-free reactant-pair identity (verify_data.rxn_idents pair_smi: canonical "
        "SMILES, maps + stereo removed, sorted) but a different stereo-kept identity (pair_smi_iso: isomeric canonical "
        "SMILES)")


def valid_cores(fold, core_of):
    """(sorted fold-k validation cores, info): fold_core of the D1 rxn_ids in $MLT/ft/data/fold<k>/valid.xyz (a row is
    validation iff all its rxn_ids are validation reactions, finetune FT-split); (None, info) when the split is absent."""
    p = M.MLT / "ft" / "data" / f"fold{fold}" / "valid.xyz"
    meta = V._rj(p.parent / "meta.json") or {}
    info = dict(path=str(p), n_cores_meta=(meta.get("valid") or {}).get("n_cores"),
                n_rxns_meta=(meta.get("valid") or {}).get("n_rxns"), fingerprint=meta.get("fingerprint"),
                code=(meta.get("params") or {}).get("code"))
    if not p.is_file():
        return None, info
    cores, rx, n = Counter(), set(), 0
    for a in M.read_extxyz(p):
        n += 1
        for x in str(a.info.get("rxn_ids") or "").split(","):
            x = x.strip()
            if x.isdigit() and int(x) in core_of:
                rx.add(int(x))
                cores[core_of[int(x)]] += 1
    info.update(n_rows=n, n_rxns_in_rows=len(rx), rows_by_core=dict(cores))
    return sorted(cores), info


def cmd_run(fold, out):
    import pandas as pd
    t0 = time.time()
    T = M.targets()
    Sf = pd.read_csv(M.S_FOLDS)
    Pk = Sf[Sf["fold_T1"] == fold]
    A, B = set(Pk["a_rxn"].astype(int)), set(Pk["b_rxn"].astype(int))
    man = pd.read_csv(V.D1_MANIFEST, usecols=["rxn_smiles", "meta_rxn_id"], keep_default_na=False, dtype=str)
    smi_mrx = {int(r): s for r, s in zip(man["meta_rxn_id"], man["rxn_smiles"]) if str(r).strip().isdigit()}
    lab = dict(zip(T["rxn_id"].astype(int), T["rxn_smiles"]))
    ident = {int(r): V.rxn_idents(lab.get(int(r)) or smi_mrx.get(int(r)) or "") for r in T["rxn_id"].astype(int)}
    print(f"[ez] identities of {len(ident)} D1 targets in {time.time() - t0:.0f} s "
          f"({sum(1 for v in ident.values() if not v)} without)", flush=True)
    core_of = dict(zip(T["rxn_id"].astype(int), T["fold_core"]))
    vc, vinfo = valid_cores(fold, core_of)
    vset = set(vc or [])
    test, train = T[T["fold_T1"] == fold], T[T["fold_T1"] != fold]
    by_ns, by_ins = defaultdict(list), defaultdict(list)
    for r, f, core in zip(train["rxn_id"].astype(int), train["fold_T1"].astype(int), train["fold_core"]):
        d = ident.get(int(r))
        if not d:
            continue
        if d.get("pair_smi") and d.get("pair_smi_iso"):
            by_ns[d["pair_smi"]].append((int(r), d["pair_smi_iso"], int(f), str(core)))
        if d.get("pair_inchi_ns") and d.get("pair_inchi_iso"):
            by_ins[d["pair_inchi_ns"]].append((int(r), d["pair_inchi_iso"]))
    rows = []
    for r, dip, core in zip(test["rxn_id"].astype(int), test["dipole"], test["fold_core"]):
        r = int(r)
        d = ident.get(r) or {}
        isa, isb = r in A, r in B
        row = dict(rxn_id=r, role="a+b" if isa and isb else "a" if isa else "b" if isb else "none", is_a=isa, is_b=isb,
                   dipole=dip, fold_core=core)
        if not (d.get("pair_smi") and d.get("pair_smi_iso")):
            row.update(ez_sibling_in_train=False, matching_train_core="", n_sibling_rxns=0, sibling_rxns="",
                       sibling_folds="", sibling_in_valid_core=False, same_pair_iso_in_train=False, identity_ok=False,
                       inchi_agrees="")
            rows.append(row)
            continue
        cand = by_ns.get(d["pair_smi"], [])
        sib = sorted(x for x in cand if x[1] != d["pair_smi_iso"])
        same = [x for x in cand if x[1] == d["pair_smi_iso"]]
        isib = None
        if d.get("pair_inchi_ns") and d.get("pair_inchi_iso"):
            isib = any(x[1] != d["pair_inchi_iso"] for x in by_ins.get(d["pair_inchi_ns"], []))
        row.update(ez_sibling_in_train=bool(sib), matching_train_core=";".join(sorted({x[3] for x in sib})),
                   n_sibling_rxns=len(sib), sibling_rxns=";".join(str(x[0]) for x in sib[:50]),
                   sibling_folds=";".join(str(f) for f in sorted({x[2] for x in sib})),
                   sibling_in_valid_core=any(x[3] in vset for x in sib), same_pair_iso_in_train=bool(same),
                   identity_ok=True, inchi_agrees="" if isib is None else bool(isib) == bool(sib))
        rows.append(row)
    D = pd.DataFrame(rows).sort_values("rxn_id").reset_index(drop=True)
    out = Path(out)
    M.write_csv(out, D)

    def wn(sel):
        s = D[sel]
        return dict(n=int(len(s)), with_sibling=int(s["ez_sibling_in_train"].sum()),
                    without_sibling=int((~s["ez_sibling_in_train"].astype(bool)).sum()))
    W = D[D["ez_sibling_in_train"].astype(bool)]
    summ = dict(code=CODE, written=M.now(), fold=fold, out=str(out), rule=RULE, n_test=int(len(D)),
                n_training_side=int(len(train)), n_pairs=int(len(Pk)), all=wn(D.index == D.index),
                by_role=dict(a=wn(D["is_a"].astype(bool)), b=wn(D["is_b"].astype(bool)),
                             none=wn(D["role"] == "none")),
                role_counts=dict(Counter(D["role"])), n_identity_missing=int((~D["identity_ok"].astype(bool)).sum()),
                n_same_pair_iso_in_train=int(D["same_pair_iso_in_train"].astype(bool).sum()),
                n_sibling_in_valid_core=int(D["sibling_in_valid_core"].astype(bool).sum()),
                inchi_agrees=dict(Counter(str(x) for x in D["inchi_agrees"])),
                with_by_dipole=dict(Counter(W["dipole"])), with_by_core=dict(Counter(W["fold_core"])),
                matching_cores=dict(Counter(c for s in W["matching_train_core"] for c in str(s).split(";") if c)),
                valid_cores=vc, valid_info=vinfo,
                examples=W.head(15)[["rxn_id", "role", "fold_core", "matching_train_core", "sibling_rxns"]]
                .to_dict("records"), seconds=round(time.time() - t0, 1))
    M.write_json(out.with_suffix(".json"), summ)
    print(f"[ez] fold {fold}: {summ['n_test']} test reactions, with sibling {summ['all']['with_sibling']} / without "
          f"{summ['all']['without_sibling']}; a {summ['by_role']['a']}; b {summ['by_role']['b']}; none "
          f"{summ['by_role']['none']}; identity missing {summ['n_identity_missing']}; same stereo-kept pair on the "
          f"training side {summ['n_same_pair_iso_in_train']}; sibling in a validation core "
          f"{summ['n_sibling_in_valid_core']}; InChI agrees {summ['inchi_agrees']}; validation cores {vc}; matching "
          f"cores {summ['matching_cores']} -> {out}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="E/Z sibling groups of the fold-k FT-2 test reactions (module docstring)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or (M.MLT / "pipeline" / "lists" / f"f{a.fold}_test_ez.csv")
    return cmd_run(a.fold, out)


if __name__ == "__main__":
    sys.exit(main())
