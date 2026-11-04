#!/usr/bin/env python3
"""tests/nurrev_tags.py — READ-ONLY review probe (DEPLOY-V2 Nurion extraction, lens B: leakage tags). Sandbox only:
  bash jobs/sbx.sh reactot tests/nurrev_tags.py [workers]
Independent of inv_nurion's tag code: expected job tags from mlt_common.targets() / core_folds() / d1_manifest.csv;
molecule identity from the geometry (covalent-radius connectivity graph -> canonical all-single-bond SMILES with explicit
H, charges / stereo dropped) vs the reactant / product SMILES of every D1 target (labels rxn_smiles). Full scan of every
per-tar nurion.extxyz.gz + report.json + hess_other.json/npz and every hess/*.npz. Writes only $MLT/tmp/nurrev/tags.json."""
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import mlt_common as C  # noqa: E402

OUT = C.MLT / "dft" / "nurion"
REV = C.MLT / "tmp" / "nurrev"
RAD = dict(H=0.31, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, S=1.05, Cl=1.02, Br=1.20)
MOLK = ("react_opt", "ref_opt", "prod_opt")
G = {}


def skel_graph(symbols, edges):
    from rdkit import Chem
    rw = Chem.RWMol()
    for s in symbols:
        a = Chem.Atom(str(s))
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    for i, j in edges:
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return Chem.MolToSmiles(m, canonical=True)


def skel_smiles(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = Chem.AddHs(m)
    return skel_graph([a.GetSymbol() for a in m.GetAtoms()],
                      [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()])


def skel_xyz(symbols, pos):
    P = np.asarray(pos, float)
    R = np.array([RAD[s] for s in symbols])
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    ii, jj = np.where(np.triu(D < 1.3 * (R[:, None] + R[None]), 1))
    return skel_graph(symbols, zip(ii, jj))


def mmhash(smi):
    import refs_q
    return "m" + hashlib.sha1(refs_q.mirror_min(smi).encode()).hexdigest()[:12]


def setup():
    import pandas as pd
    sys.path.insert(0, str(C.ANALYSIS / "d1_ladder"))
    T = C.targets()
    G["fold"] = dict(zip(T.rxn_id.astype(int), T.fold_T1.astype(int)))
    G["dip"] = dict(zip(T.rxn_id.astype(int), T.dipole.astype(str)))
    G["job2rxn"] = dict(zip(T.job_id.astype(str), T.rxn_id.astype(int)))
    G["cores_without_fold"] = list(T.attrs.get("cores_without_fold", []))
    G["tcore"] = dict(zip(T.rxn_id.astype(int), T.fold_core.astype(str)))
    G["cf"] = C.core_folds()
    M = pd.read_csv(C.D1RUN / "d1_manifest.csv", dtype=str).fillna("")
    G["d1m"] = {r["job_id"]: r for r in M.to_dict("records")}
    cell = defaultdict(set)
    for j, r in G["d1m"].items():
        if j in G["job2rxn"]:
            cell[r["cell_id"]].add(G["job2rxn"][j])
    G["cell"] = dict(cell)
    mapr, mapp, mmr, mmp, own_l, own_r, canon = defaultdict(set), defaultdict(set), {}, {}, {}, {}, {}
    from rdkit import Chem
    for rid, rs in zip(T.rxn_id.astype(int), T.rxn_smiles.astype(str)):
        lhs, rhs = rs.split(">>")
        ls = [skel_smiles(s) for s in lhs.split(".")]
        rr = [skel_smiles(s) for s in rhs.split(".")]
        for s in ls:
            mapr[s].add(rid)
        for s in rr:
            mapp[s].add(rid)
        own_l[rid], own_r[rid] = set(ls), set(rr)
        mmr[rid] = {mmhash(s) for s in lhs.split(".")}
        mmp[rid] = {mmhash(s) for s in rhs.split(".")}
        try:
            key = ".".join(sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in lhs.split("."))) + ">>" + \
                ".".join(sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in rhs.split(".")))
            canon.setdefault(key, set()).add(rid)
        except Exception:                                          # noqa: BLE001
            pass
    G.update(mapr=dict(mapr), mapp=dict(mapp), mmr=mmr, mmp=mmp, own_l=own_l, own_r=own_r, canon=canon)
    # non-target own reactant / product skeletons from the D1 manifest
    nl, nr = {}, {}
    for j, r in G["d1m"].items():
        if j in G["job2rxn"] or ">>" not in r["rxn_smiles"]:
            continue
        lhs, rhs = r["rxn_smiles"].split(">>")
        nl[j] = {skel_smiles(s) for s in lhs.split(".")}
        nr[j] = {skel_smiles(s) for s in rhs.split(".")}
    G["nt_l"], G["nt_r"] = nl, nr


def expect(job):
    if job in G["job2rxn"]:
        r = G["job2rxn"][job]
        return dict(target=True, rx={r}, folds={str(G["fold"][r])}, dips={G["dip"][r]}, core_fold=True, cellf=set())
    row = G["d1m"].get(job)
    if row is None:
        return dict(target=False, rx=set(), folds=set(), dips=set(), core_fold=False, cellf=set(), unknown=True)
    core = f"{row['dipole']}/{row['skeleton']}"
    f = {str(int(G["cf"][core]))} if core in G["cf"] else set()
    cellf = {str(G["fold"][r]) for r in G["cell"].get(row["cell_id"], set())}
    return dict(target=False, rx=set(), folds=f, dips={row["dipole"]} if row["dipole"] else set(), core_fold=core in G["cf"],
                cellf=cellf)


def tagsets(info):
    rx = {int(x) for x in str(info["rxn_ids"]).split(",") if x}
    fo = {x for x in str(info["folds"]).split(",") if x}
    dp = {x for x in str(info["dipoles"]).split(",") if x}
    return rx, fo, dp


def scan_tar(stem):
    import inv_nurion as N                                        # read_ext only (the documented reader of .gz)
    c, ex = Counter(), defaultdict(list)
    molsk = defaultdict(set)
    sids = []
    tshort = N.short(stem)
    p = OUT / stem / "nurion.extxyz.gz"
    al = N.read_ext(p)

    def note(k, v):
        c[k] += 1
        if len(ex[k]) < 4:
            ex[k].append(v)
    for a in al:
        inf = a.info
        kd, job = inf["kind"], inf["job"]
        sids.append(inf["struct_id"])
        e = expect(job)
        is_t = kd.startswith("nur_") and not kd.startswith("nur_x_")
        base = kd[len("nur_"):] if is_t else kd[len("nur_x_"):]
        c["rows"] += 1
        if e.get("unknown"):
            note("job_not_in_d1_manifest", job)
        if is_t != e["target"]:
            note("prefix_vs_target_mismatch", inf["struct_id"])
        rx, fo, dp = tagsets(inf)
        if any(r not in G["fold"] for r in rx):
            note("rxn_not_d1_target", inf["struct_id"])
        if not {str(G["fold"][r]) for r in rx if r in G["fold"]} <= fo:
            note("fold_of_listed_rxn_missing", inf["struct_id"])
        if not {G["dip"][r] for r in rx if r in G["dip"]} <= dp:
            note("dipole_of_listed_rxn_missing", inf["struct_id"])
        mol = base in MOLK
        if (not mol) or is_t or base == "prod_opt":
            if not e["rx"] <= rx:
                note("own_rxn_missing", inf["struct_id"])
            if not e["folds"] <= fo:
                note(f"job_fold_missing:{base}", inf["struct_id"])
            if not e["dips"] <= dp:
                note(f"job_dipole_missing:{base}", inf["struct_id"])
            if not e["cellf"] <= fo:
                note("cell_fold_missing", inf["struct_id"])
        else:                                                    # non-target reactant molecule: molecule union only
            c[f"nt_mol_rows:{base}"] += 1
            if not e["folds"] <= fo:
                c[f"nt_mol_without_job_core_fold:{base}"] += 1
            if not e["dips"] <= dp:
                c[f"nt_mol_without_job_dipole:{base}"] += 1
        if not mol:
            if rx != e["rx"]:
                note(f"nonmol_rxn_ids_ne_job:{base}", f"{inf['struct_id']} {sorted(rx)}")
            if not is_t and not e["core_fold"]:
                c[f"nt_nocorefold_rows:{base}"] += 1
                if not fo:
                    c[f"nt_rows_no_fold_at_all:{base}"] += 1
        if inf.get("tar") != tshort:
            note("tar_info_mismatch", inf["struct_id"])
        if inf.get("label_ts") and base != "ts_final":
            note("label_ts_not_final", inf["struct_id"])
        if inf.get("label_ts"):
            c["label_ts_rows"] += 1
        if inf.get("source") != "nurion" or inf.get("level") != "label":
            note("source_level", inf["struct_id"])
        if base == "truncated" and is_t:
            c["trunc_target_rows"] += 1
        if mol:
            sk = skel_xyz(a.get_chemical_symbols(), a.positions)
            if inf.get("mol"):
                molsk[inf["mol"]].add(sk)
            else:
                note(f"mol_row_without_mol_key:{base}", inf["struct_id"])
            mp = G["mapp"] if base == "prod_opt" else G["mapr"]
            exp = mp.get(sk, set())
            if is_t:
                own = G["own_r"][G["job2rxn"][job]] if base == "prod_opt" else G["own_l"][G["job2rxn"][job]]
            else:
                own = (G["nt_r"] if base == "prod_opt" else G["nt_l"]).get(job, set())
            if sk not in own:
                note(f"mol_geometry_not_own_{'t' if is_t else 'x'}:{base}", inf["struct_id"])
            if not exp:
                c[f"mol_not_in_any_target:{base}:{'t' if is_t else 'x'}"] += 1
            miss = exp - rx
            if miss:
                mm = G["mmp"] if base == "prod_opt" else G["mmr"]
                strict = {r for r in miss if inf.get("mol") in mm[r]}
                if strict:
                    note(f"mol_union_miss_same_key:{base}", f"{inf['struct_id']} miss {sorted(strict)[:5]}")
                    c[f"mol_union_miss_same_key_rxns:{base}"] += len(strict)
                    fm = {str(G["fold"][r]) for r in strict} - fo
                    if fm:
                        note(f"mol_union_miss_same_key_new_fold:{base}", f"{inf['struct_id']} folds {sorted(fm)}")
                if miss - strict:
                    note(f"mol_union_miss_stereo_variant:{base}", f"{inf['struct_id']} miss {sorted(miss - strict)[:5]}")
                    fm = {str(G["fold"][r]) for r in miss - strict} - fo
                    if fm:
                        note(f"mol_union_miss_stereo_variant_new_fold:{base}", f"{inf['struct_id']} folds {sorted(fm)}")
                    dm = {G["dip"][r] for r in miss - strict} - dp
                    if dm:
                        c[f"mol_union_miss_stereo_variant_new_dipole:{base}"] += 1
    return dict(stem=stem, counts=dict(c), ex=dict(ex), molsk={k: sorted(v) for k, v in molsk.items()}, sids=sids)


def scan_hess(stem):
    c, ex = Counter(), defaultdict(list)
    idx = C.read_json(OUT / stem / "hess_other.json")["hessians"]
    with np.load(OUT / stem / "hess_other.npz", allow_pickle=False) as z:
        nH = len([k for k in z.files if k.startswith("H")])
        if nH != len(idx):
            c["npz_n_ne_index"] += 1
        for r in idx:
            e = expect(r["job"])
            rx = {int(x) for x in str(r["rxn_ids"]).split(",") if x}
            fo = {x for x in str(r["folds"]).split(",") if x}
            dp = {x for x in str(r["dipoles"]).split(",") if x}
            c[f"kind:{r['kind']}"] += 1
            if rx != e["rx"] or not e["folds"] <= fo or not e["dips"] <= dp:
                c["job_tag_mismatch"] += 1
                if len(ex["job_tag_mismatch"]) < 4:
                    ex["job_tag_mismatch"].append(r["struct_id"])
            if r["kind"].endswith("mol_hess"):
                sk = skel_xyz(r["symbols"], z[f"P{r['i']}"])
                exp = G["mapr"].get(sk, set()) | G["mapp"].get(sk, set())
                miss = exp - rx
                c["mol_hess"] += 1
                if miss:
                    c["mol_hess_union_miss"] += 1
                    if {str(G["fold"][x]) for x in miss} - fo:
                        c["mol_hess_union_miss_new_fold"] += 1
            n3 = 3 * len(r["symbols"])
            if z[f"H{r['i']}"].shape[0] != n3 * (n3 + 1) // 2:
                c["triangle_size_bad"] += 1
    return dict(stem=stem, counts=dict(c), ex=dict(ex))


def scan_hl(paths):
    c, rows = Counter(), []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            rid = int(z["rxn_id"])
            ok = (Path(p).stem == str(rid) and rid in G["fold"] and int(z["fold"]) == G["fold"][rid]
                  and str(z["dipole"]) == G["dip"][rid] and str(z["code"]) == "mlts-nur-2"
                  and z["hessian"].shape == (3 * len(z["symbols"]),) * 2 and np.isfinite(z["hessian"]).all()
                  and float(z["kabsch_max_dev_A"]) <= 1e-4)
            c["ok" if ok else "bad"] += 1
            rows.append((rid, str(z["tar"]), str(z["job"]), C.job_of(rid) == str(z["job"])))
    return dict(counts=dict(c), rows=rows)


def main():
    t0 = time.time()
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    setup()
    t_setup = time.time() - t0
    stems = sorted(p.parent.name for p in OUT.glob("d1_*/.done"))
    hfiles = sorted(str(p) for p in (OUT / "hess").glob("*.npz"))
    chunks = [hfiles[i::nw] for i in range(nw)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        ft = [ex.submit(scan_tar, s) for s in sorted(stems, key=lambda s: -(OUT / s / "nurion.extxyz.gz").stat().st_size)]
        fh = [ex.submit(scan_hess, s) for s in stems]
        fl = [ex.submit(scan_hl, ch) for ch in chunks]
        rt = [f.result() for f in ft]
        rh = [f.result() for f in fh]
        rl = [f.result() for f in fl]
    tot, exs = Counter(), defaultdict(list)
    molsk = defaultdict(set)
    sid_tar = Counter()
    for r in rt:
        tot.update(r["counts"])
        for k, v in r["ex"].items():
            exs[k].extend(v[: max(0, 4 - len(exs[k]))])
        for k, v in r["molsk"].items():
            molsk[k].update(v)
        sid_tar.update(r["sids"])
    htot, hex_ = Counter(), defaultdict(list)
    for r in rh:
        htot.update(r["counts"])
        for k, v in r["ex"].items():
            hex_[k].extend(v[: max(0, 4 - len(hex_[k]))])
    ltot = Counter()
    lrows = []
    for r in rl:
        ltot.update(r["counts"])
        lrows.extend(r["rows"])
    # reports: full scan
    rep_c, rep_ex = Counter(), defaultdict(list)
    job_tars = defaultdict(list)
    hl_listed = set()
    nocore = Counter()
    same_rxn = Counter()
    from rdkit import Chem
    for s in stems:
        rp = C.read_json(OUT / s / "report.json")
        hl_listed |= set(rp["outputs"]["hess_label"]["rxn_ids"])
        for j in rp["jobs"]:
            job_tars[j["job"]].append(s)
            e = expect(j["job"])
            rep_c["jobs"] += 1
            if bool(j["target"]) != e["target"]:
                rep_c["target_flag_mismatch"] += 1
                rep_ex["target_flag_mismatch"].append(j["job"])
            if j["target"] and j["rxn"] != G["job2rxn"].get(j["job"]):
                rep_c["rxn_mismatch"] += 1
            want_rule = "target_rxn" if e["target"] else ("nontarget_core" if not e.get("unknown") else "?")
            if j["tag_rule"] != want_rule:
                rep_c[f"tag_rule_unexpected:{j['tag_rule']}"] += 1
            if not e["target"]:
                row = G["d1m"].get(j["job"], {})
                core = f"{row.get('dipole')}/{row.get('skeleton')}"
                nocore[(row.get("kind"), "core_fold" if e["core_fold"] else "no_core_fold", j["cls"],
                        "core_of_some_target" if core in set(G["tcore"].values()) else "core_not_target")] += 1
                rs = row.get("rxn_smiles", "")
                if ">>" in rs:
                    lhs, rhs = rs.split(">>")
                    try:
                        key = ".".join(sorted(Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in lhs.split("."))) + ">>" + \
                            ".".join(sorted(Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in rhs.split(".")))
                    except Exception:                              # noqa: BLE001
                        key = None
                    hit = G["canon"].get(key, set()) if key else set()
                    if hit:
                        same_rxn["nontarget_same_rxn_as_target"] += 1
                        if {str(G["fold"][r]) for r in hit} - (e["folds"] | e["cellf"]):
                            same_rxn["nontarget_same_rxn_other_fold"] += 1
                            rep_ex["nontarget_same_rxn_other_fold"].append([j["job"], sorted(hit)])
                    sk = tuple(sorted(skel_smiles(x) for x in lhs.split("."))) + tuple(sorted(skel_smiles(x) for x in rhs.split(".")))
                    same_rxn["nontarget_checked"] += 1
                    G.setdefault("_ntsk", []).append((j["job"], sk, e["folds"] | e["cellf"]))
    # stereo-agnostic same reaction skeleton as a target in another fold
    tsk = defaultdict(set)
    for rid in G["fold"]:
        tsk[tuple(sorted(G["own_l"][rid])) + tuple(sorted(G["own_r"][rid]))].add(rid)
    for job, sk, f in G.get("_ntsk", []):
        hit = tsk.get(sk, set())
        if hit:
            same_rxn["nontarget_same_skeleton_rxn_as_target"] += 1
            if {str(G["fold"][r]) for r in hit} - f:
                same_rxn["nontarget_same_skeleton_rxn_other_fold"] += 1
                if len(rep_ex["nontarget_same_skeleton_other_fold"]) < 6:
                    rep_ex["nontarget_same_skeleton_other_fold"].append([job, sorted(hit)[:4], sorted(f)])
    multi = {j: t for j, t in job_tars.items() if len(t) > 1}
    lset = {r[0] for r in lrows}
    out = dict(
        tars=stems, t_setup_s=round(t_setup, 1), wall_s=round(time.time() - t0, 1),
        cores_without_fold_targets=G["cores_without_fold"],
        extxyz=dict(counts=dict(sorted(tot.items())), examples=dict(exs),
                    struct_id_dups=sum(1 for v in sid_tar.values() if v > 1),
                    mol_key_multi_skeleton=sum(1 for v in molsk.values() if len(v) > 1), mol_keys=len(molsk)),
        hess_other=dict(counts=dict(sorted(htot.items())), examples=dict(hex_)),
        hess_label=dict(counts=dict(ltot), files=len(lrows), listed_in_reports=len(hl_listed),
                        files_eq_listed=lset == hl_listed, only_files=sorted(lset - hl_listed)[:10],
                        only_listed=sorted(hl_listed - lset)[:10],
                        job_eq_rxn=sum(r[3] for r in lrows), tars=dict(Counter(r[1] for r in lrows))),
        reports=dict(counts=dict(rep_c), examples={k: v[:6] for k, v in rep_ex.items()},
                     jobs_in_several_tars=len(multi), multi_examples=dict(list(multi.items())[:5]),
                     nontarget_breakdown=[[list(k), v] for k, v in sorted(nocore.items(), key=lambda x: -x[1])],
                     same_rxn=dict(same_rxn)))
    os.makedirs(REV, exist_ok=True)
    (REV / "tags.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
