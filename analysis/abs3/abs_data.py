#!/usr/bin/env python3
r"""abs_data.py — ABS3 reaction-level data (docs/specs/ABS3_MODELS.md 3, 4). env_d1fu2 (DeployData needs torch).

AbsData = the token arrays, usable rule and T1 split lists of DeployData('lf') (analysis/ch3/data_ch3.py binds
DEP_SCRATCH=$CH3/lf, DEP_RESULTS=analysis/ch3/results/lf_inputs, unchanged) with
  * the per-reaction labels of $LAD/labels_d1_cr2.json (key map = V_report.json pairs.code.d1_pairs, spec 4):
    y10 [token row, 10], y4 = y10 @ M (strain, inter, solv, barrier);
  * the reaction sets of every fold: for each list (train, test, inner_train, inner_val) the reactions a and b of its
    usable pairs (DeployData.split(k), the same pairs as the pair models), as token-row indices (sorted);
  * per-reaction attributes: dipole, skeleton, series, attach (analysis/d1_ladder/results/labels_d1_cr2.parquet), core
    (S_folds.csv core of the usable pairs the reaction belongs to; one per reaction, else STOP), role ('H' = the b reaction
    of a reference pair, else 'sub').
AbsStd = fu2 train.Std (fu2/train.py 140-160) for reaction-level fits: token numeric and global statistics from the fit's
training reactions only (the same statistics FT.Std computes from the reactions of the matching training pairs), plus
y mean / std (ddof 0) of the training reactions; .tensors() is FT.Std.tensors unchanged.

  python abs_data.py summary    Phase LS (spec 4): LS-1 .. LS-5 -> results/LS_report.json (exit 3 = STOP on LS-1, LS-3,
                                LS-5 or a missing label key; LS-2 is reported, not a STOP). LS-5 reads the XGB pack
                                ($ABS3/pack/lf_abs.npz, abs_pack.py).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abs_common as C  # noqa: E402
import data_ch3 as DC3  # noqa: E402            (analysis/ch3, on sys.path through abs_common; binds the lf paths)

DL, FT, Mo, F = DC3.DL, DC3.FT, DC3.Mo, DC3.F
T10, T3 = list(C.TARGETS10), list(C.TARGETS3)
NF = C.NF
LS2_MAX = 0.12                                   # spec LS-2 (label QC Σ6 − Bond residual max 0.11 + rounding)
LS1_TOL = 1e-6


# ================================================================ labels
def labels_json():
    """$LAD/labels_d1_cr2.json -> {rxn_id: record}; sha256 = V_report.json outputs, key map = pairs.code.d1_pairs."""
    p = C.LABELS_JSON
    if not p.is_file():
        C.not_ready(f"{p} missing")
    sha = C.sha256(p)
    V = C.read_json(C.V_REPORT)
    want = (V.get("outputs") or {}).get(str(p))
    if not V.get("all_pass") or want != sha:
        C.stop(f"{p}: sha256 {sha[:12]} != V_report outputs {str(want)[:12]} (or V_report not all_pass)")
    km = ((V.get("pairs") or {}).get("code") or {}).get("d1_pairs")
    if km != C.LABEL_KEYS:
        C.stop(f"label key map {C.LABEL_KEYS} != V_report pairs.code.d1_pairs {km}")
    by = {}
    for r in json.loads(p.read_text()):
        rid = int(r["rxn_id"])
        if rid in by:
            C.stop(f"{p}: duplicated rxn_id {rid}")
        by[rid] = r
    return by, sha


def label_matrix(by, rids):
    """y10 [n, 10] for the reaction ids (NaN where the reaction or a key is absent) and the absent (rxn_id, key) list."""
    y = np.full((len(rids), len(T10)), np.nan)
    miss = []
    for k, rid in enumerate(rids):
        r = by.get(int(rid))
        if r is None:
            miss.append((int(rid), "<reaction>"))
            continue
        for j, t in enumerate(T10):
            v = r.get(C.LABEL_KEYS[t])
            if v is None:
                miss.append((int(rid), C.LABEL_KEYS[t]))
                continue
            y[k, j] = float(v)
    return y, miss


# ================================================================ data
class AbsData:
    """reaction-level view of DeployData('lf') (spec 3 abs_data.py, 4)."""

    ARRAYS = ("rids", "row", "n", "d_in", "d_glob", "elements", "bin_cols", "num_cols", "trep", "onehot", "bin", "num",
              "pos", "mask", "etype", "frag", "glob", "nmax")

    def __init__(self):
        t0 = time.time()
        DC3.use_targets(T3)
        D = DC3.get_data("lf")
        self.D = D
        for k in self.ARRAYS:
            setattr(self, k, getattr(D, k))
        self.by, self.labels_json_sha = labels_json()
        self.y10, self.label_missing = label_matrix(self.by, D.rids)
        self.y4 = self.y10 @ C.M()
        # pairs: S_folds core per pair (pair-table order)
        S = pd.read_csv(C.S_FOLDS, usecols=["pair_id", "core", "ref_kind", "pos"])
        S = S.assign(pair_id=S.pair_id.astype(str)).set_index("pair_id")
        self.pair_core = S.core.astype(str).reindex(D.P.index).fillna("").to_numpy()
        # fold reaction sets
        self.sets = {k: self._sets(k) for k in range(NF)}
        rs = set()
        for k in range(NF):
            for key in C.FOLD_KEYS:
                rs |= set(self.sets[k][key].tolist())
        self.rxn_set = np.array(sorted(rs), dtype=int)
        in_set = np.zeros(len(D.rids), dtype=bool)
        in_set[self.rxn_set] = True
        self.in_set = in_set
        miss_set = [(r, k) for r, k in self.label_missing if in_set[self.row[r]]]
        if miss_set:
            C.stop(f"spec 4: label key missing for {len(miss_set)} reactions of the reaction set: {miss_set[:5]}")
        if not np.isfinite(self.y4[self.rxn_set]).all():
            C.stop("spec 4: non-finite y4 in the reaction set")
        # attributes
        Lq = pd.read_parquet(C.LABELS_PQ, columns=["rxn_id", "dipole", "skeleton", "series", "attach"])
        Lq = Lq.assign(rxn_id=Lq.rxn_id.astype(int))
        if Lq.rxn_id.duplicated().any():
            C.stop(f"{C.LABELS_PQ}: duplicated rxn_id")
        Lq = Lq.set_index("rxn_id").reindex([int(r) for r in D.rids])
        self.attr = {c: Lq[c].fillna("").astype(str).to_numpy() for c in ("dipole", "skeleton", "series", "attach")}
        self.attr_missing_in_set = int(sum(1 for r in self.rxn_set if self.attr["dipole"][r] == ""))
        self.core_rxn, self.core_conflicts = self._rxn_core()
        if self.core_conflicts:
            C.stop(f"{len(self.core_conflicts)} reactions belong to pairs of more than one core: {self.core_conflicts[:5]}")
        self.role = self._role()
        self.load_seconds = time.time() - t0
        print(f"[abs_data] {len(D.rids)} token reactions, reaction set {len(self.rxn_set)} "
              f"({int((self.role[self.rxn_set] == 'H').sum())} H references) ({self.load_seconds:.0f} s)", flush=True)

    # ---------------------------------------------------------------- sets
    def _sets(self, k):
        D = self.D
        sp = D.split(k)
        out = {key: np.unique(np.r_[D.ra[sp[key]], D.rb[sp[key]]]).astype(int) for key in C.FOLD_KEYS}
        out["pair_rows"] = {key: np.asarray(sp[key], dtype=int) for key in C.FOLD_KEYS}
        out["pairs_sha"] = sp["pairs_sha"]
        ids = {key: [str(int(D.rids[r])) for r in out[key]] for key in C.FOLD_KEYS}
        out["rxn_sha"] = DL.list_sha(ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"] + ["|"]
                                     + ids["inner_val"])
        out["test_sha"] = DL.list_sha(ids["test"])
        out["val_sha"] = DL.list_sha(ids["inner_val"])
        return out

    def split(self, k):
        """reaction sets of fold k: train / test / inner_train / inner_val token rows (sorted) + pair rows + sha256."""
        return self.sets[int(k)]

    def rxn_ids(self, rows):
        return np.asarray(self.rids, dtype=int)[np.asarray(rows, dtype=int)]

    def _rxn_core(self):
        D = self.D
        cores = {}
        up = np.zeros(len(D.P), dtype=bool)
        for k in range(NF):
            for key in C.FOLD_KEYS:
                up[self.sets[k]["pair_rows"][key]] = True
        for i in np.flatnonzero(up):
            for r in (int(D.ra[i]), int(D.rb[i])):
                cores.setdefault(r, set()).add(self.pair_core[i])
        core = np.full(len(D.rids), "", dtype=object)
        conflicts = []
        for r, s in cores.items():
            if len(s) != 1 or "" in s:
                conflicts.append((int(D.rids[r]), sorted(s)))
            core[r] = sorted(s)[0]
        return core.astype(str), conflicts

    def _role(self):
        D = self.D
        ref = (D.P.ref_kind.to_numpy() == "reference") & D.has_tok
        h = set(D.rb[ref].tolist())
        role = np.full(len(D.rids), "", dtype=object)
        for r in self.rxn_set:
            role[r] = "H" if int(r) in h else "sub"
        return role.astype(str)

    def sets_fingerprint(self):
        return {str(k): dict(rxn_sha=self.sets[k]["rxn_sha"], pairs_sha=self.sets[k]["pairs_sha"]) for k in range(NF)}

    def x182(self):
        """$CH3/lf/X182_lf.parquet values reindexed to the token rows (columns = T_report glob_cols; LADDER XData rule)."""
        cols = list(self.trep["glob_cols"])
        path = DL.SOURCES["lf"]["x182"]
        X = pd.read_parquet(path)
        X = X.assign(rxn_id=X.rxn_id.astype(int)).set_index("rxn_id")
        if list(X.columns) != cols:
            C.stop(f"{path}: columns differ from T_report glob_cols")
        return X.reindex([int(r) for r in self.rids]).to_numpy(dtype=float), str(path)

    def sha(self):
        return dict(self.D.sha, labels_json=self.labels_json_sha, labels_parquet=C.sha256(C.LABELS_PQ),
                    s_folds=C.sha256(C.S_FOLDS))


class AbsStd(FT.Std):
    """FT.Std for a reaction-level fit (spec 0-4): statistics from the training reactions only; adds ymu / sy of y4."""

    def __init__(self, A, rxn_rows):
        rows = np.asarray(rxn_rows, dtype=int)
        self.pair_rows = None
        self.rxn_rows = np.unique(rows)
        if len(self.rxn_rows) != len(rows):
            C.stop("AbsStd: duplicated training reactions")
        m = A.mask[self.rxn_rows]
        vals = A.num[self.rxn_rows][m]
        self.mu, self.sd = vals.mean(0), vals.std(0)
        self.sd[self.sd < FT.C.SCALE_EPS] = 1.0
        g = A.glob[self.rxn_rows]
        self.gmu, self.gsd = g.mean(0), g.std(0)
        self.gsd[self.gsd < FT.C.SCALE_EPS] = 1.0
        y = A.y4[self.rxn_rows]
        if not np.isfinite(y).all():
            C.stop("AbsStd: non-finite y4 among the training reactions")
        self.ymu, self.sy = y.mean(0), y.std(0)
        if not (np.isfinite(self.sy).all() and (self.sy > 0).all()):
            C.stop(f"AbsStd: y std not positive {self.sy}")

    def as_dict(self):
        return dict(mu=self.mu, sd=self.sd, gmu=self.gmu, gsd=self.gsd, ymu=self.ymu, sy=self.sy)


def std_sha(d):
    import hashlib
    h = hashlib.sha256()
    for k in ("mu", "sd", "gmu", "gsd", "ymu", "sy"):
        h.update(np.ascontiguousarray(np.asarray(d[k], dtype=np.float64)).tobytes())
    return h.hexdigest()


_DATA: dict = {}


def get_data():
    if "abs" not in _DATA:
        _DATA["abs"] = AbsData()
    return _DATA["abs"]


# ================================================================ Phase LS
def pack_arrays(A):
    """the arrays the XGB pack must hold (LS-5): reaction ids, X, y4 and the fold reaction sets (token rows)."""
    X, _ = A.x182()
    out = dict(rids=np.asarray(A.rids, dtype=np.int64), X=X, y4=A.y4)
    for k in range(NF):
        for key in C.FOLD_KEYS:
            out[f"f{k}_{key}"] = A.sets[k][key].astype(np.int64)
    return out


def ls_report(A):
    D = A.D
    rep = dict(spec="docs/specs/ABS3_MODELS.md 4", written=C.now())
    # LS-1: every pair of F_pairs_cr2.csv: dd_t = y_t(a) - y_t(b) (labels json, 10 targets)
    P = pd.read_csv(C.LABELS)
    ya, ma = label_matrix(A.by, P.a_rxn.astype(int).tolist())
    yb, mb = label_matrix(A.by, P.b_rxn.astype(int).tolist())
    dd = P[[f"dd_{t}" for t in T10]].to_numpy(float)
    diff = np.abs(dd - (ya - yb))
    per = {t: float(np.nanmax(diff[:, j])) if np.isfinite(diff[:, j]).any() else None for j, t in enumerate(T10)}
    ls1_ok = not ma and not mb and bool(np.isfinite(diff).all()) and max(per.values()) <= LS1_TOL
    rep["LS1"] = dict(n_pairs=int(len(P)), max_abs_diff=per, tol=LS1_TOL, missing_a=len(ma), missing_b=len(mb),
                      ok=bool(ls1_ok))
    # LS-2: per-reaction identity |barrier - (strain + inter + solv)|
    def dist(y4):
        r = np.abs(y4[:, 3] - y4[:, :3].sum(1))
        r = r[np.isfinite(r)]
        return dict(n=int(len(r)), max=float(r.max()), p99=float(np.percentile(r, 99)), p95=float(np.percentile(r, 95)),
                    mean=float(r.mean()), n_gt_tol=int((r > LS2_MAX).sum()))
    ally, _ = label_matrix(A.by, sorted(A.by))
    rep["LS2"] = dict(reaction_set=dist(A.y4[A.rxn_set]), all_labels=dist(ally @ C.M()), tol=LS2_MAX,
                      ok=bool(dist(A.y4[A.rxn_set])["max"] <= LS2_MAX), stop_rule="reported only (spec 12)")
    # LS-3 / LS-4: per fold
    folds = {}
    for k in range(NF):
        s = A.sets[k]
        st = {key: set(s[key].tolist()) for key in C.FOLD_KEYS}
        te_pairs = s["pair_rows"]["test"]
        c = dict(train_test_disjoint=not (st["train"] & st["test"]),
                 inner_disjoint=not (st["inner_train"] & st["inner_val"]),
                 test_pairs_in_test=set(D.ra[te_pairs].tolist()) | set(D.rb[te_pairs].tolist()) <= st["test"],
                 inner_union_is_train=(st["inner_train"] | st["inner_val"]) == st["train"])
        folds[str(k)] = dict(checks=c, n_rxn={key: int(len(s[key])) for key in C.FOLD_KEYS},
                             n_pairs={key: int(len(s["pair_rows"][key])) for key in C.FOLD_KEYS},
                             n_H={key: int((A.role[s[key]] == "H").sum()) for key in C.FOLD_KEYS},
                             rxn_sha=s["rxn_sha"])
    rep["LS3"] = dict(ok=bool(all(v["checks"]["train_test_disjoint"] and v["checks"]["inner_disjoint"]
                                  and v["checks"]["test_pairs_in_test"] for v in folds.values())),
                      rule="per fold: train ∩ test = ∅, inner_train ∩ inner_val = ∅, a and b of every test pair in the "
                           "test reactions (STOP); inner_train ∪ inner_val = train reported",
                      folds={k: v["checks"] for k, v in folds.items()})
    rs = A.rxn_set
    rep["LS4"] = dict(n_reactions=int(len(rs)), n_pairs_P=len(D.pair_list), n_token_reactions=int(len(D.rids)),
                      n_H=int((A.role[rs] == "H").sum()), n_sub=int((A.role[rs] == "sub").sum()),
                      folds={k: dict(n_rxn=v["n_rxn"], n_pairs=v["n_pairs"], n_H=v["n_H"]) for k, v in folds.items()},
                      attach_counts=pd.Series(A.attr["attach"][rs]).value_counts().to_dict(),
                      dipole_counts=pd.Series(A.attr["dipole"][rs]).value_counts().to_dict(),
                      attach_by_role={r: pd.Series(A.attr["attach"][rs][A.role[rs] == r]).value_counts().to_dict()
                                      for r in ("H", "sub")},
                      n_cores=int(len(set(A.core_rxn[rs].tolist()))), attr_missing_in_set=A.attr_missing_in_set,
                      test_reaction_folds=_test_fold_count(A), ok=True)
    # LS-5: the pack equals abs_data
    want = pack_arrays(A)
    if not (C.PACK_NPZ.is_file() and C.PACK_JSON.is_file()):
        C.not_ready(f"{C.PACK_NPZ} missing (abs_pack.py)")
    info = C.read_json(C.PACK_JSON)
    z = np.load(C.PACK_NPZ, allow_pickle=False)
    eq = {k: C.sha256_array(z[k]) == C.sha256_array(v) for k, v in want.items()}
    rep["LS5"] = dict(arrays_equal=eq, pack_json_ok=bool(info.get("ok")), npz_sha256=C.sha256(C.PACK_NPZ),
                      npz_sha256_in_json=info.get("npz_sha256"),
                      ok=bool(all(eq.values()) and info.get("ok") and C.sha256(C.PACK_NPZ) == info.get("npz_sha256")))
    rep["labels"] = dict(json=str(C.LABELS_JSON), json_sha256=A.labels_json_sha, parquet=str(C.LABELS_PQ),
                         parquet_sha256=C.sha256(C.LABELS_PQ), key_map=C.LABEL_KEYS,
                         n_json_reactions=len(A.by), n_token_reactions_without_label=int(np.isnan(A.y10).any(1).sum()))
    rep["inputs_sha256"] = A.sha()
    rep["all_pass"] = bool(rep["LS1"]["ok"] and rep["LS3"]["ok"] and rep["LS5"]["ok"])
    return rep


def _test_fold_count(A):
    c = {}
    for k in range(NF):
        for r in A.sets[k]["test"].tolist():
            c[r] = c.get(r, 0) + 1
    v = pd.Series(list(c.values())).value_counts().to_dict() if c else {}
    return {f"in_{k}_folds": int(n) for k, n in v.items()}


def cmd_summary():
    A = get_data()
    rep = ls_report(A)
    C.write_json(C.RES / "LS_report.json", rep)
    print(json.dumps(C.clean(dict(all_pass=rep["all_pass"], LS1=rep["LS1"]["ok"], LS2=rep["LS2"], LS3=rep["LS3"]["ok"],
                                  LS4={k: rep["LS4"][k] for k in ("n_reactions", "n_H", "n_sub", "n_cores")},
                                  LS5=rep["LS5"]["ok"])), indent=1, ensure_ascii=False))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["summary"]:
        sys.exit(cmd_summary())
    print(__doc__)
    sys.exit(2)
