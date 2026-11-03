#!/usr/bin/env python3
r"""compat_deploy.py — GEO2b DEPLOY input compatibility and the common pair set (analysis/geo2/results/DECISIONS_deploy.md
DP-1, DP-3, DP-4). CPU, env_espley (pandas, pyarrow; no torch). Reads only; nothing here was run when it was written.

  python compat_deploy.py run
        -> analysis/geo2/results/DEPLOY_compat.json, DEPLOY_input_shift.csv, $GEO2B/deploy/common_pairs.csv.
           Exit 0 done, 2 not ready, 3 STOP (layout difference, a deployed pair whose geo structure came from a GEO1b
           model that saw its fold, or label / pair mismatch).

What is checked (data through LADDER train_xgb.get_data = the same loaders / usable rule as the trainers)
  pairs     per T1 fold: test pairs of the FU2 split file, usable in dft, usable in geo, both (= the common / deployed
            set = deploy_common.deploy_rows on the dft unit test list), dft only, geo only; P_main reference para counts.
  leak      both ends of every common fold-k pair: analysis/geo2/results/Z_status.csv source = 'f<k>' (the GEO1b final
            model of fold k, which never saw fold k; fm1 = fold -1 reactions, never a test end) -> else STOP.
  tokens    T_report / Q_report of both rungs: layout keys (elements, bin_cols, num_cols, glob_cols, ext_used, bond_cap,
            h_map_rule, Q tolerances) must be equal (else STOP); versions = base + rung tag; provenance keys listed;
            T_manifest n_atoms of the common reactions.
  X182      T_report glob_cols = the X182 parquet columns (train_xgb.load_x182 enforces the order per rung); XGB feature
            columns of S and SH equal.
  H / SH    input tables ($LAD/inputs/dft.parquet, $GEO2B/inputs/geo.parquet): H_<t> (re-referenced, the trainers' H) and
            H0_<t> (labels_d1_hf values, own references); reference sources ref1_src / ref2_src cross-tab (dft: own / Q /
            V = refs_cr2; geo: own / min); HF input header lines (method, FRAG strings, charge / multiplicity) of the 5
            calculations of every common reaction, dft = $D1HF/inputs/<T>/hf, geo = $GEO2B/hf_geo/inputs/<T>/g2.
  labels    the dft label file (F_pairs_cr2.csv) and the geo one (F_pairs_d0d1.csv, D1 rows = F_pairs_cr2 bytes): sha256,
            provenance (train_xgb.load_labels), y equal on the common pairs.
  shift     DEPLOY_input_shift.csv: per X182 / H / H0 column on the common reactions: mean dft / geo, SD dft, mean and
            median |geo - dft|, |geo - dft| / SD, Pearson r (descriptive).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_common as DC  # noqa: E402

sys.path.insert(0, str(DC.LAD_DIR))
import train_xgb as TX  # noqa: E402

H_TARGETS = list(TX.H_TARGETS)
CALCS = ("eda", "frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel")              # = d1_hf hf_common.CALCS
HF_DIRS = {"dft": (TX.L.HF_SCR / "inputs", "hf"),                                    # inputs.RUNGS['dft'] hf_inputs
           "geo": (DC.GEO2 / "hf_geo" / "inputs", "g2")}                             # inputs.RUNGS['geo'] (geom g2)
T_LAYOUT = ("elements", "bin_cols", "num_cols", "glob_cols", "ext_used", "bond_cap", "cov_edges_equal_rel_bonds")
Q_LAYOUT = ("h_map_rule", "qa_tol", "qb_tol", "ext_min_pass", "ext_used")
T_PROV = ("n_reactions", "manifest_sha256", "q_report_sha256", "x182_sha256", "splits_sha256", "n_excluded_from_Q")


def pair_sets(Dd, Dg, S):
    """per fold: test-pair counts by usability + the common (deployed) pairs (dft unit test order)."""
    rows, common = [], []
    for k in range(TX.NF):
        ids = [str(q) for q in Dd.folds[k]["test"]]
        r_d = np.array([Dd.pid_row[q] for q in ids], dtype=int)
        r_g = np.array([Dg.pid_row[q] for q in ids], dtype=int)
        ud, ug = Dd.usable_pair[r_d], Dg.usable_pair[r_g]
        te = Dd.split(k)["test"]
        keep, rows_g = DC.deploy_rows(Dd, Dg, te)
        DC.check_same_pairs(Dd, Dg, te, keep, rows_g)
        cp = np.asarray(Dd.P.index)[te[keep]].astype(str)
        if sorted(cp.tolist()) != sorted(np.asarray(ids)[ud & ug].tolist()):
            DC.stop(f"fold {k}: deploy_rows pairs != split-file test pairs usable in both rungs")
        para = S.reindex(ids)
        pm = (para.in_P_main.astype(bool) & (para.ref_kind == "reference")).to_numpy()
        pp = pm & (para.pos == "p").to_numpy()
        rows.append(dict(fold=k, n_test=len(ids), usable_dft=int(ud.sum()), usable_geo=int(ug.sum()),
                         common=int((ud & ug).sum()), dft_only=int((ud & ~ug).sum()), geo_only=int((~ud & ug).sum()),
                         neither=int((~ud & ~ug).sum()), p_main=int(pm.sum()), p_main_common=int((pm & ud & ug).sum()),
                         p_main_para=int(pp.sum()), p_main_para_common=int((pp & ud & ug).sum()),
                         p_main_para_dft=int((pp & ud).sum()), p_main_para_geo=int((pp & ug).sum()),
                         anchors_common=int(((para.ref_kind == "anchor").to_numpy() & ud & ug).sum())))
        P = Dd.P.loc[cp]
        common.append(pd.DataFrame(dict(pair_id=cp, fold=k, ref_kind=P.ref_kind.to_numpy(), pos=P.pos.to_numpy(),
                                        a_rxn=P.a_rxn.to_numpy(int), b_rxn=P.b_rxn.to_numpy(int),
                                        in_P_main=S.reindex(cp).in_P_main.astype(bool).to_numpy(),
                                        core=S.reindex(cp).core.astype(str).to_numpy())))
    return pd.DataFrame(rows), pd.concat(common, ignore_index=True)


def leak_check(CP):
    """DP-1: Z_status source of both ends of a fold-k common pair must be f<k>."""
    Z = pd.read_csv(DC.Z_STATUS, usecols=["rxn_id", "src", "source", "status"])
    Z = Z[Z.src == "D1"].assign(rxn_id=lambda d: d.rxn_id.astype(int)).set_index("rxn_id")
    if Z.index.duplicated().any():
        DC.stop(f"{DC.Z_STATUS}: duplicated D1 rxn_id")
    sa = Z.source.reindex(CP.a_rxn).fillna("none").astype(str).to_numpy()
    sb = Z.source.reindex(CP.b_rxn).fillna("none").astype(str).to_numpy()
    want = np.array([f"f{k}" for k in CP.fold])
    bad = (sa != want) | (sb != want)
    cnt = Counter((int(k), a, b) for k, a, b in zip(CP.fold, sa, sb))
    rec = dict(rule="both ends of a fold-k common pair: Z_status source = f<k> (GEO1b final_f<k> never saw fold k)",
               counts={f"f{k}|{a}|{b}": int(n) for (k, a, b), n in sorted(cnt.items())}, n_bad=int(bad.sum()),
               bad=CP.pair_id[bad].head(10).tolist(), z_status_sha256=DC.sha256(DC.Z_STATUS))
    if bad.any():
        DC.stop(f"{int(bad.sum())} common pairs have an end whose geo structure is not from final_f<fold>: "
                f"{rec['bad']} ({DC.Z_STATUS})")
    return rec


def token_reports():
    """DP-3: token / atom report layout equality (STOP otherwise) + versions and provenance."""
    def rd(rung, name):
        return DC.read_json(Path(TX.RUNGS[rung]["fu2_res"]) / name)
    T = {r: rd(r, "T_report.json") for r in ("dft", "geo")}
    Q = {r: rd(r, "Q_report.json") for r in ("dft", "geo")}
    lay = {k: T["dft"].get(k) == T["geo"].get(k) for k in T_LAYOUT}
    qlay = {k: Q["dft"].get(k) == Q["geo"].get(k) for k in Q_LAYOUT}
    tags = {r: TX.RUNGS[r]["tag"] for r in ("dft", "geo")}
    base = {r: dict(token=str(T[r].get("token_version")), atom=str(Q[r].get("atom_version"))) for r in T}
    strip = {r: {k: v[:-len(tags[r]) - 1] if tags[r] and v.endswith("-" + tags[r]) else v for k, v in base[r].items()}
             for r in base}
    rec = dict(t_layout_equal=lay, q_layout_equal=qlay, versions=base, versions_without_rung_tag=strip,
               versions_base_equal=strip["dft"] == strip["geo"] and strip["dft"]["token"] == TX.FTok.TOKEN_VERSION
               and strip["dft"]["atom"] == TX.FA.ATOM_VERSION_D1PAIR,
               provenance={k: dict(dft=T["dft"].get(k), geo=T["geo"].get(k)) for k in T_PROV},
               n_atoms_stats={r: T[r].get("n_atoms") for r in T}, excluded_from_Q={r: T[r].get("excluded_from_Q")
                                                                                 for r in T},
               note="definitions (layout, version base, Q tolerances, H mapping rule) identical; provenance (structures, "
                    "manifest, X182 file) differs by construction")
    if not all(lay.values()) or not all(qlay.values()) or not rec["versions_base_equal"]:
        DC.stop(f"token / atom report layout or version base differs between rungs dft and geo: {lay} {qlay} {strip}")
    return rec


def manifest_atoms(rids):
    out = {}
    for r in ("dft", "geo"):
        M = pd.read_csv(Path(TX.RUNGS[r]["fu2_res"]) / "T_manifest.csv", usecols=["rxn_id", "n_atoms"])
        out[r] = M.assign(rxn_id=M.rxn_id.astype(int)).set_index("rxn_id").n_atoms.reindex(rids).to_numpy(float)
    eq = out["dft"] == out["geo"]
    return dict(n=len(rids), n_atoms_equal=int(eq.sum()),
                differ=[int(x) for x in np.asarray(rids)[~eq][:20]])


def read_table(rung):
    p = TX.inputs_path(rung)
    T = pd.read_parquet(p)
    T = T.assign(rxn_id=T.rxn_id.astype(int)).set_index("rxn_id")
    return T, dict(path=str(p), sha256=DC.sha256(p))


def h_part(Td, Tg, rids):
    """DP-3 H definition record on the common reactions: reference sources, re-referencing size per rung."""
    a, b = Td.reindex(rids), Tg.reindex(rids)
    rec = {}
    for k in (1, 2):
        c = f"ref{k}_src"
        if c in a.columns and c in b.columns:
            ct = Counter(zip(a[c].fillna("").astype(str), b[c].fillna("").astype(str)))
            rec[f"{c}_dft_geo"] = {f"{x}|{y}": int(n) for (x, y), n in sorted(ct.items())}
        else:
            rec[f"{c}_dft_geo"] = "column missing"
    rr = {}
    for t in ("d1", "d2", "barrier"):
        row = {}
        for nm, T in (("dft", a), ("geo", b)):
            if f"H_{t}" in T.columns and f"H0_{t}" in T.columns:
                d = (T[f"H_{t}"] - T[f"H0_{t}"]).to_numpy(float)
                row[nm] = dict(n_changed=int((np.abs(d) > 0).sum()), mean_abs=float(np.nanmean(np.abs(d))),
                               max_abs=float(np.nanmax(np.abs(d))))
        rr[t] = row
    rec["rereference_H_minus_H0"] = rr
    rec["rule"] = dict(dft="H = labels_d1_hf (HF/6-31G* at the DFT label TS) with the d1 / d2 / barrier reference energies "
                           "of refs_cr2.parquet (own | Q | V, LADDER Phase V)",
                       geo="H = labels_d1_hf_g2 of the geo structures (HF/6-31G* at the GEO1b TS + G2c rel structures) "
                           "with 'min' references (lowest HF per Q-2 key over the geo rows; constrained keys keep their "
                           "own; GEO2b F-7b)",
                       sh_arm="global token / XGB features [X182 | H 9 of the rung table]; baseline beta x (H(a) - H(b)) "
                              "with the dft unit's beta (fitted on dft T_main pairs, no refit); the 6 EDA channels have no "
                              "reference energy, d1 / d2 / barrier differ in the reference rule")
    return rec


def inp_header(p):
    """method / block lines of an ORCA input up to the coordinate block (resource lines %pal / %maxcore / nprocs
    dropped; whitespace normalised)."""
    try:
        lines = Path(p).read_text().splitlines()
    except OSError:
        return None
    out = []
    for ln in lines:
        s = " ".join(ln.split())
        if s.startswith("*"):
            break
        lo = s.lower()
        if not s or s.startswith("#") or lo.startswith(("%pal", "%maxcore")) or "nprocs" in lo:
            continue
        out.append(s)
    return tuple(out)


def hf_headers(jobs):
    rec = {}
    for calc in CALCS:
        n = miss_d = miss_g = diff = 0
        ex, kinds = [], Counter()
        for j in jobs:
            hd = inp_header(HF_DIRS["dft"][0] / j / HF_DIRS["dft"][1] / f"{calc}.inp")
            hg = inp_header(HF_DIRS["geo"][0] / j / HF_DIRS["geo"][1] / f"{calc}.inp")
            if hd is None:
                miss_d += 1
            if hg is None:
                miss_g += 1
            if hd is None or hg is None:
                continue
            n += 1
            kinds[hd[0] if hd else ""] += 1
            if hd != hg:
                diff += 1
                if len(ex) < 3:
                    ex.append(dict(job=j, dft=list(hd), geo=list(hg)))
        rec[calc] = dict(n_compared=n, missing_dft=miss_d, missing_geo=miss_g, n_differ=diff, examples=ex,
                         method_lines=dict(kinds))
    return rec


def shift_table(Dd, Dg, Td, Tg, rids):
    rd = {int(r): i for i, r in enumerate(Dd.rids)}
    rg = {int(r): i for i, r in enumerate(Dg.rids)}
    Xd = Dd.X[[rd[r] for r in rids]]
    Xg = Dg.X[[rg[r] for r in rids]]
    cols = [(c, Xd[:, i], Xg[:, i], "X182") for i, c in enumerate(Dd.glob_cols)]
    for pre, kind in (("H_", "H"), ("H0_", "H0")):
        for t in H_TARGETS:
            c = f"{pre}{t}"
            if c in Td.columns and c in Tg.columns:
                cols.append((c, Td[c].reindex(rids).to_numpy(float), Tg[c].reindex(rids).to_numpy(float), kind))
    out = []
    for c, a, b, kind in cols:
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        d = np.abs(b - a)
        sd = float(a.std()) if len(a) else float("nan")
        r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
        out.append(dict(column=c, kind=kind, n=int(m.sum()), mean_dft=float(a.mean()) if len(a) else np.nan,
                        mean_geo=float(b.mean()) if len(b) else np.nan, sd_dft=sd,
                        mean_abs_diff=float(d.mean()) if len(d) else np.nan,
                        median_abs_diff=float(np.median(d)) if len(d) else np.nan,
                        mad_over_sd_dft=float(d.mean() / sd) if len(d) and sd > 0 else np.nan, pearson_r=r))
    return pd.DataFrame(out)


def cmd_run():
    t0 = time.time()
    Dd, Dg = TX.get_data(DC.SRC_RUNG, "S"), TX.get_data(DC.IN_RUNG, "S")
    Dd_h, Dg_h = TX.get_data(DC.SRC_RUNG, "SH"), TX.get_data(DC.IN_RUNG, "SH")
    DC.log(t0, "data loaded (dft / geo, S / SH)")
    feat = dict(S=list(Dd.feat_cols) == list(Dg.feat_cols), SH=list(Dd_h.feat_cols) == list(Dg_h.feat_cols),
                glob_cols=list(Dd.glob_cols) == list(Dg.glob_cols), n_glob=len(Dd.glob_cols),
                sh_cols=list(Dd_h.feat_cols)[len(Dd.glob_cols):])
    if not (feat["S"] and feat["SH"] and feat["glob_cols"]):
        DC.stop(f"X182 / XGB feature columns differ between rungs dft and geo: {feat}")
    S = pd.read_csv(TX.S_FOLDS, dtype={"pair_id": str}).set_index("pair_id")
    S["ref_kind"] = S.ref_kind.fillna("").astype(str)
    S["pos"] = S.pos.fillna("").astype(str)
    counts, CP = pair_sets(Dd, Dg, S)
    DC.log(t0, f"common pairs {len(CP)}")
    leak = leak_check(CP)
    rids = sorted(set(CP.a_rxn.tolist()) | set(CP.b_rxn.tolist()))
    tok = token_reports()
    tok["manifest_n_atoms_common"] = manifest_atoms(rids)
    Td, td_info = read_table(DC.SRC_RUNG)
    Tg, tg_info = read_table(DC.IN_RUNG)
    for nm, T in (("dft", Td), ("geo", Tg)):
        if not T.reindex(rids).usable.fillna(False).astype(bool).all():
            DC.stop(f"a common reaction is not usable in the {nm} input table")
    hrec = h_part(Td, Tg, rids)
    jobs = sorted(set(Td.job_id.reindex(rids).astype(str)))
    if set(jobs) != set(Tg.job_id.reindex(rids).astype(str)):
        DC.stop("job ids of the common reactions differ between the dft and geo input tables")
    hrec["hf_input_headers"] = hf_headers(jobs)
    DC.log(t0, "HF input headers compared")
    dy = DC.maxdiff(Dd.y[[Dd.pid_row[p] for p in CP.pair_id]], Dg.y[[Dg.pid_row[p] for p in CP.pair_id]])
    labels = dict(dft=Dd.linfo, geo=Dg.linfo, y_common_max_abs_diff=dy,
                  rule="y = dd_<t> of the label file; the geo file's D1 rows are F_pairs_cr2.csv byte for byte (L-11')")
    if dy > DC.Y_TOL:
        DC.stop(f"labels differ on the common pairs (max {dy})")
    sh = shift_table(Dd, Dg, Td, Tg, rids)
    DC.write_csv(DC.COMMON_CSV, CP)
    DC.write_csv(DC.RES / "DEPLOY_input_shift.csv", sh)
    hsum = {r.column: dict(mean_abs_diff=r.mean_abs_diff, mad_over_sd_dft=r.mad_over_sd_dft, pearson_r=r.pearson_r)
            for r in sh.itertuples() if r.kind in ("H", "H0")}
    xs = sh[sh.kind == "X182"]
    rep = dict(phase="GEO2b DEPLOY compat (DP-1, DP-3, DP-4)", ok=True,
               pairs=dict(by_fold=counts.to_dict("records"), totals=counts.drop(columns="fold").sum().to_dict(),
                          common_csv=str(DC.COMMON_CSV), common_csv_sha256=DC.sha256(DC.COMMON_CSV),
                          rule="common = T1 test pairs whose both ends are usable in rung dft AND rung geo (the "
                               "trainers' usable rule) = the pairs deploy_rows keeps; D1 only"),
               leak=leak, tokens=tok, features=feat, tables=dict(dft=td_info, geo=tg_info), h=hrec,
               h_shift_common=hsum, x182_shift_summary=dict(
                   n_cols=int(len(xs)), median_mad_over_sd=float(xs.mad_over_sd_dft.median()),
                   n_cols_r_below_0_9=int((xs.pearson_r < 0.9).sum()),
                   worst=xs.sort_values("mad_over_sd_dft", ascending=False).head(10)[
                       ["column", "mad_over_sd_dft", "pearson_r"]].to_dict("records")),
               labels=labels, inputs=dict(dft=Dd.inputs, geo=Dg.inputs), env=DC.env_info(), code_sha256=DC.code_sha(),
               seconds=time.time() - t0)
    DC.write_json(DC.COMPAT_JSON, rep)
    DC.log(t0, f"-> {DC.COMPAT_JSON} ({len(CP)} common pairs)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        sys.exit(cmd_run())
    raise SystemExit(__doc__)
