#!/usr/bin/env python3
"""data_lf.py — DEPLOY-MODEL 8: usable reactions / pairs and DeployData(fu2 train.Data) (docs/specs/DEPLOY_MODEL.md 8).

DeployData builds the attributes that train_atoms.LadData builds for arm S (onehot, bin, num, pos, mask, etype, frag, n,
glob, y, ra, rb, P, d_in, d_glob; static() inherited from fu2 train.Data; split(k)) in the same way; only the usable
rule and the token / 182 paths differ (spec 8-2).

Sources (SOURCES)
  lf  (production)  tokens $DEP/fu2_lf/tokens + reports analysis/deploy/results/fu2_lf/{T_report.json, T_manifest.csv,
                    Q_report.json} (fu2-tokens-3-lf / fu2-atoms-3-lf), global table $DEP/X182_lf.parquet.
                    usable reaction = token (T_manifest row passing T-a: n_ok, finite, 2 formed edges) AND an X182_lf row
                    whose 182 columns are all finite; NO HF (spec 8-1).
  g2  (test C-2)    the LADDER g2 inputs: tokens $G2/fu2/tokens + analysis/d1_g2/results/fu2 reports (fu2-tokens-3-g2 /
                    fu2-atoms-3-g2), $G2/X182_d1_g2.parquet, and the LADDER usable rule incl. H: train_atoms.load_inputs
                    on $LAD/inputs/g2.parquet (token + table row + finite H 9 + finite 182 of the table). Built only to
                    compare with train_atoms.LadData("g2", "S") (tests.py C-2) and for train_tf.py c3 (C-3).
  usable pair = a and b usable; pairs not usable are removed from every split list (LADDER rule) -> P_lf.

Labels  $LAD/F_pairs_cr2.csv (dd_<t>, 10 targets; = LADDER), provenance = analysis/d1_ladder/results/V_report.json
        all_pass + outputs sha256 (train_atoms.label_provenance rule); parsed exactly as train_atoms.load_labels.
Splits  FU2 T1 files $D1PAIR_SCRATCH/fu2/splits/T1__f<k>.json, sha256 = analysis/d1_pair/results/fu2/S_splits.json
        (train_atoms.load_folds('T1')).
split(k)     train (outer T_main), test (P_sens u A_main), inner_train, inner_val (first inner GroupKFold(4) fold); leak
             checks of LadData.split: test pair in train, inner folds not partitioning train, test reaction in train
             -> STOP.
split("all") (Phase W only): every usable in_T_main pair (= union of the 5 fold train lists, fold -1 cores included;
             cross-checked against D1PAIR S_folds.csv in_T_main, whose sha256 FU2 S_splits.json records); no test /
             validation lists (empty arrays).

Reused code (imported, unchanged): fu2/train.py Data (46; static 110), Std (140); fu2/tokens.py BIN_COLS / NUM_BASE /
NUM_EXT (35-37); d1pair_common.list_sha; train_atoms.load_inputs (543, source g2 only). The token array loop is the
fu2 train.Data.__init__ loop (82-94) = train_atoms.load_tokens (522-534), parametrised by root (fill_token_arrays).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_pair / d1_ladder trees

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402

FU2_DIR = C.REPO / "analysis" / "d1_pair" / "fu2"
sys.path.insert(0, str(FU2_DIR))
import fu2_common as F  # noqa: E402          (constants, FU2 paths, d1pair_common as F.C)
import models as Mo  # noqa: E402             (TF, GRID, build)
import train as FT  # noqa: E402              (Data, Std, fit, gate_m1, predict, batch, epoch_batches, versions)
import tokens as FTok  # noqa: E402           (TOKEN_VERSION, BIN_COLS, NUM_BASE, NUM_EXT)

DP = F.C                                                        # d1pair_common
if tuple(F.TARGETS) != C.TARGETS:
    C.stop(f"fu2_common.TARGETS {F.TARGETS} != dep_common.TARGETS {C.TARGETS}")
TARGETS = list(F.TARGETS)
SPLIT, NF = "T1", 5
FOLD_KEYS = ("train", "test", "inner_train", "inner_val")
DEFAULT_LABELS = C.LAD / "F_pairs_cr2.csv"
V_REPORT = C.REPO / "analysis" / "d1_ladder" / "results" / "V_report.json"
FU2_S_SPLITS = F.FU2_RES / "S_splits.json"
S_FOLDS = DP.RES / "S_folds.csv"
NUM_COLS = list(FTok.NUM_BASE) + list(FTok.NUM_EXT)             # ext_used (spec 6: d_in = 29)
D_IN = len(C.ELEMENTS) + len(FTok.BIN_COLS) + len(NUM_COLS)
GLOB_REL_TOL = 1e-12                                             # token glob vs 182 table (train_atoms.load_inputs)

SOURCES = {
    "lf": dict(res=C.RES / "fu2_lf", tokens=C.TOK_LF, x182=C.X182_LF, token_version=C.TOKEN_VERSION,
               atom_version=C.ATOM_VERSION, usable="lf", table=None),
    "g2": dict(res=C.REPO / "analysis" / "d1_g2" / "results" / "fu2", tokens=C.G2 / "fu2" / "tokens",
               x182=C.G2 / "X182_d1_g2.parquet", token_version=FTok.TOKEN_VERSION + "-g2",
               atom_version="fu2-atoms-3-g2", usable="ladder", table=C.LAD / "inputs" / "g2.parquet"),
}
USABLE_RULES = {
    "lf": "token (T_manifest row, T-a ok) AND X182 row with 182 finite values; no HF (DEPLOY_MODEL 8-1)",
    "ladder": "token AND $LAD/inputs/<rung>.parquet row AND finite H 9 (+ finite 182 of the table) "
              "(train_atoms.load_inputs, MA-5); C-2 / C-3 only",
}


def _abs(p) -> Path:
    return Path(os.path.abspath(p))


def list_sha(ids) -> str:
    return DP.list_sha([(q,) for q in ids])


# ================================================================ tokens
def fill_token_arrays(paths, elements, n_bin, n_num, nmax=None, shas=None):
    """the fu2 train.Data.__init__ token loop (fu2/train.py 82-94) for the token files `paths` (row order kept):
    padded float64 arrays + edge types / fragment ids; `shas` (same order) -> each file's sha256 must match (STOP).
    Returns (dict of arrays, list of the files' sha256)."""
    zs = []
    if nmax is None:
        for p in paths:
            with np.load(p) as t:
                zs.append(len(t["Z"]))
        nmax = int(max(zs)) if zs else 0
    nr = len(paths)
    el = {z: k for k, z in enumerate(elements)}
    onehot = np.zeros((nr, nmax, len(elements)))
    binm = np.zeros((nr, nmax, n_bin))
    num = np.zeros((nr, nmax, n_num))
    pos = np.zeros((nr, nmax, 3))
    mask = np.zeros((nr, nmax), dtype=bool)
    etype = np.zeros((nr, nmax, nmax), dtype=np.int8)
    frag = np.zeros((nr, nmax), dtype=np.int8)
    glob = None
    got = []
    for k, p in enumerate(paths):
        s = C.sha256(p)
        got.append(s)
        if shas is not None and s != shas[k]:
            C.stop(f"{p} sha256 differs from its manifest")
        t = np.load(p)
        n = len(t["Z"])
        onehot[k, np.arange(n), [el[int(z)] for z in t["Z"]]] = 1.0
        binm[k, :n], num[k, :n], pos[k, :n] = t["bin"], t["num"], t["pos"]
        mask[k, :n] = True
        frag[k, :n] = t["bin"][:, 0]
        for (i, j), e in zip(t["edges"], t["etype"]):
            etype[k, i, j] = etype[k, j, i] = int(e)
        if glob is None:
            glob = np.zeros((nr, len(t["glob"])))
        glob[k] = t["glob"]
    if glob is None:
        glob = np.zeros((nr, 0))
    return dict(onehot=onehot, bin=binm, num=num, pos=pos, mask=mask, etype=etype, frag=frag, glob=glob,
                nmax=nmax), got


def load_tokens(cfg):
    """the source's token reactions as fu2 train.Data arrays (same checks as train_atoms.load_tokens 481-540, same
    T_manifest row order); per-reaction T-a flags from T_manifest (n_ok, finite, n_formed == 2)."""
    res, tdir = Path(cfg["res"]), Path(cfg["tokens"])
    tpath, mpath, qpath = (res / n for n in ("T_report.json", "T_manifest.csv", "Q_report.json"))
    miss = [str(p) for p in (tpath, mpath, qpath) if not p.is_file()]
    if miss or not tdir.is_dir():
        C.not_ready(f"token files missing {miss or [str(tdir)]}")
    trep, qrep = C.read_json(tpath), C.read_json(qpath)
    if trep.get("stop") or qrep.get("stop"):
        C.stop(f"{tpath} / {qpath}: stop")
    sha = dict(t_report=C.sha256(tpath), t_manifest=C.sha256(mpath), q_report=C.sha256(qpath))
    fu2_ss = C.sha256(FU2_S_SPLITS)
    rung_ss = res / "S_splits.json"
    ga = trep.get("gate_Ta") or {}
    checks = [("token_version", trep.get("token_version") == cfg["token_version"]),
              ("atom_version", qrep.get("atom_version") == cfg["atom_version"]),
              ("q_report_sha256", trep.get("q_report_sha256") == sha["q_report"]),
              ("manifest_sha256", trep.get("manifest_sha256") == sha["t_manifest"]),
              ("splits_sha256 = FU2 S_splits.json", trep.get("splits_sha256") in (None, fu2_ss)),
              ("rung S_splits.json = FU2 S_splits.json", (not rung_ss.is_file()) or C.sha256(rung_ss) == fu2_ss),
              ("bin_cols", list(trep.get("bin_cols") or []) == list(FTok.BIN_COLS)),
              ("num_cols", list(trep.get("num_cols") or []) == NUM_COLS and bool(trep.get("ext_used"))),
              ("glob_cols 182", len(trep.get("glob_cols") or []) == 182),
              ("elements", [int(z) for z in trep.get("elements") or []] == list(C.ELEMENTS)),
              ("gate_Ta", bool(ga) and all(bool(v) for v in ga.values()))]
    if trep.get("x182_sha256") is not None and Path(cfg["x182"]).is_file():
        checks.append(("x182_sha256", trep["x182_sha256"] == C.sha256(cfg["x182"])))
    bad = [k for k, ok in checks if not ok]
    if bad:
        C.stop(f"{tpath}: token report does not match the current inputs / expected layout: {bad}")
    man = pd.read_csv(mpath)
    rids = man.rxn_id.astype(int).to_numpy()
    if len(set(rids.tolist())) != len(rids):
        C.stop(f"{mpath}: duplicated rxn_id")
    elements = [int(z) for z in trep["elements"]]
    num_cols, bin_cols = list(trep["num_cols"]), list(trep["bin_cols"])
    paths = [tdir / f"{rid}.npz" for rid in rids]
    arr, _ = fill_token_arrays(paths, elements, len(bin_cols), len(num_cols), nmax=int(man.n_atoms.max()),
                               shas=man.sha256.astype(str).tolist())
    ta_ok = np.ones(len(rids), dtype=bool)
    for col, want in (("n_ok", True), ("finite", True)):
        if col in man.columns:
            ta_ok &= man[col].astype(bool).to_numpy() == want
    if "n_formed" in man.columns:
        ta_ok &= man.n_formed.to_numpy(int) == 2
    return dict(trep=trep, qrep=qrep, elements=elements, num_cols=num_cols, bin_cols=bin_cols, rids=rids,
                row={int(r): k for k, r in enumerate(rids)}, n=man.n_atoms.to_numpy(int), ta_ok=ta_ok,
                man_sha=dict(zip(rids.tolist(), man.sha256.astype(str))), sha=sha, glob182=arr.pop("glob"),
                d_in=len(elements) + len(bin_cols) + len(num_cols), **arr,
                inputs=dict(t_report=str(tpath), t_manifest=str(mpath), q_report=str(qpath), tokens_dir=str(tdir),
                            n_tokens=int(len(rids))))


def load_x182(path, tok):
    """182-column table -> per token row: has a row, all finite; names / order = T_report glob_cols (else STOP); the
    token glob must equal the table row (<= 1e-12 relative, train_atoms.load_inputs rule) where both are finite."""
    path = Path(path)
    if not path.is_file():
        C.not_ready(f"{path} missing")
    X = pd.read_parquet(path)
    if "rxn_id" not in X.columns:
        C.stop(f"{path}: no rxn_id column")
    cols = [c for c in X.columns if c != "rxn_id"]
    if cols != list(tok["trep"]["glob_cols"]):
        C.stop(f"{path}: column names / order differ from T_report glob_cols (F-1)")
    X = X.assign(rxn_id=X.rxn_id.astype(int))
    if X.rxn_id.duplicated().any():
        C.stop(f"{path}: duplicated rxn_id")
    X = X.set_index("rxn_id")
    rids, nr = tok["rids"], len(tok["rids"])
    common = [int(r) for r in rids if int(r) in X.index]
    rows = np.array([tok["row"][r] for r in common], dtype=int)
    has = np.zeros(nr, dtype=bool)
    fin = np.zeros(nr, dtype=bool)
    G = X.loc[common, cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float) if common else np.zeros((0, 182))
    has[rows] = True
    fin[rows] = np.isfinite(G).all(1)
    g0 = tok["glob182"][rows]
    dev = (np.abs(G - g0) > GLOB_REL_TOL * np.maximum(1.0, np.abs(g0))) & np.isfinite(G)
    nbad = int(dev.any(1).sum())
    if nbad:
        C.stop(f"{path}: {nbad} reactions whose 182 columns differ from their token glob (> 1e-12 relative)")
    info = dict(path=str(path), sha256=C.sha256(path), n_rows=int(len(X)), n_token_rows_in_table=len(common),
                n_token_rows_without_row=int(nr - len(common)), n_nonfinite=int((has & ~fin).sum()),
                glob_equal_token=True)
    return has & fin, info


def _ladder_usable(rung, tok):
    """the LADDER usable rule (train_atoms.load_inputs, H included) on the source's tokens (C-2 / C-3 only)."""
    sys.path.insert(0, str(C.REPO / "analysis" / "d1_ladder"))
    import train_atoms as TA  # noqa: E402  (lazy: production never imports the LADDER trainer)
    H9, ur, info = TA.load_inputs(rung, tok)
    return ur, dict(info, h9_finite_rows=int(np.isfinite(H9).all(1).sum()))


# ================================================================ labels / folds
def label_provenance(path, sha):
    """train_atoms.label_provenance (613-638) for $LAD/F_pairs_cr2.csv: V_report.json all_pass + outputs sha256."""
    if _abs(path) != _abs(DEFAULT_LABELS):
        return "unverified (not $LAD/F_pairs_cr2.csv; tests only)"
    rep = C.read_json(V_REPORT) if V_REPORT.is_file() else {}
    if not rep.get("all_pass"):
        C.stop(f"{path}: {V_REPORT} missing or not all_pass")
    rec = (rep.get("outputs") or {}).get(str(C.LAD / Path(path).name))
    if rec != sha:
        C.stop(f"{path}: sha256 differs from {V_REPORT.name} outputs ({str(rec)[:12]} vs {sha[:12]})")
    return f"{V_REPORT.name} all_pass, outputs sha256 equal"


def load_labels(path):
    """label file -> pair table (index pair_id) + info; = train_atoms.load_labels (641-663) parsing."""
    path = _abs(path)
    if not path.is_file():
        C.not_ready(f"label file {path} missing")
    sha = C.sha256(path)
    prov = label_provenance(path, sha)
    P = pd.read_csv(path)
    need = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos"] + [f"dd_{t}" for t in TARGETS]
    miss = [c for c in need if c not in P.columns]
    if miss:
        C.stop(f"{path}: columns missing {miss}")
    if P.pair_id.duplicated().any():
        C.stop(f"{path}: duplicated pair_id")
    P = P.assign(pair_id=P.pair_id.astype(str), ref_kind=P.ref_kind.fillna("").astype(str),
                 pos=P.pos.fillna("").astype(str), a_rxn=P.a_rxn.astype(int), b_rxn=P.b_rxn.astype(int))
    return P.set_index("pair_id"), dict(path=str(path), sha256=sha, provenance=prov, n_pairs=int(len(P)))


def load_folds():
    """FU2 T1 split files, sha256 = FU2 S_splits.json (= train_atoms.load_folds('T1') 697-715)."""
    srep = C.read_json(FU2_S_SPLITS)
    folds, files, shas = {}, {}, {}
    for k in range(NF):
        name = f"{SPLIT}__f{k}.json"
        p = F.SPLITS / name
        if not p.is_file() or C.sha256(p) != srep["units"][name]["sha256"]:
            C.stop(f"{p} missing or changed after {FU2_S_SPLITS}")
        folds[k] = C.read_json(p)
        files[f"split_T1_f{k}"] = str(p)
        shas[f"split_T1_f{k}"] = srep["units"][name]["sha256"]
    return srep, folds, files, shas


# ================================================================ data object
class DeployData(FT.Data):
    """fu2 train.Data for the deploy model (arm S, TF): the FU2 / LadData attributes from the source's tokens and the
    label file, the usable-pair filter (spec 8-1) and split(k) / split('all'). fu2 train.Data.__init__ (FU2 paths) is
    not called; static() is inherited."""

    def __init__(self, source="lf", labels=DEFAULT_LABELS):
        if source not in SOURCES:
            C.stop(f"unknown source {source}")
        t0 = time.time()
        cfg = SOURCES[source]
        tok = load_tokens(cfg)
        x_ok, xinfo = load_x182(cfg["x182"], tok)
        if cfg["usable"] == "lf":
            ur, hinfo = tok["ta_ok"] & x_ok, None
        else:
            ur, hinfo = _ladder_usable(source, tok)
        P, linfo = load_labels(labels)
        srep, folds, sfiles, sshas = load_folds()
        fold_ids = {q for f in folds.values() for k in FOLD_KEYS for q in f[k]}
        miss = sorted(fold_ids - set(P.index))
        if miss:
            C.stop(f"{len(miss)} split pair ids are not in {linfo['path']}: {miss[:5]}")
        for k in ("trep", "elements", "num_cols", "bin_cols", "rids", "row", "nmax", "n", "onehot", "bin", "num", "pos",
                  "mask", "etype", "frag", "d_in"):
            setattr(self, k, tok[k])
        self.rung, self.source, self.arm = source, source, "S"
        self.usable_rule = cfg["usable"]
        self.P, self.linfo, self.xinfo, self.hinfo = P, linfo, xinfo, hinfo
        self.srep, self.folds = srep, folds
        self.y = P[[f"dd_{t}" for t in TARGETS]].to_numpy(float)
        pid = P.index.to_numpy()
        self.pid_row = {p: k for k, p in enumerate(pid)}
        self.ra = np.array([self.row.get(int(r), -1) for r in P.a_rxn])
        self.rb = np.array([self.row.get(int(r), -1) for r in P.b_rxn])
        self.has_tok = (self.ra >= 0) & (self.rb >= 0)
        self.usable_rxn = ur
        self.usable_pair = self.has_tok & ur[np.maximum(self.ra, 0)] & ur[np.maximum(self.rb, 0)]
        in_folds = np.isin(pid, sorted({q for f in folds.values() for q in f["train"] + f["test"]}))
        self.pair_list = sorted(pid[self.usable_pair & in_folds].tolist())
        self.pair_list_sha = list_sha(self.pair_list)
        self.glob = tok["glob182"]
        self.d_glob = self.glob.shape[1]
        self.ta_ok = tok["ta_ok"]
        self.sha = dict(tok["sha"], x182=xinfo["sha256"], labels=linfo["sha256"], s_splits=C.sha256(FU2_S_SPLITS),
                        **sshas)
        if hinfo is not None:
            self.sha["inputs_table"] = hinfo["sha256"]
        self.s_splits_sha = self.sha["s_splits"]
        self.inputs = dict(tok["inputs"], x182=str(cfg["x182"]), labels=linfo["path"], s_splits=str(FU2_S_SPLITS),
                           **sfiles, **({"inputs_table": hinfo["path"]} if hinfo else {}))
        self.load_seconds = time.time() - t0
        if self.d_in != D_IN or self.d_glob != 182:
            C.stop(f"d_in {self.d_in} / d_glob {self.d_glob} != {D_IN} / 182 (spec 6 / 11-2)")
        print(f"[data] {source}: {len(self.rids)} token reactions, {int(ur.sum())} usable ({cfg['usable']} rule), "
              f"{len(self.pair_list)} pairs in P_{source} ({int(self.usable_pair.sum())} usable of {len(P)}) "
              f"({self.load_seconds:.0f} s)", flush=True)

    # ---------------------------------------------------------------- layout (stored in every .pt, MODEL_CARD)
    def layout(self):
        return dict(elements=[int(z) for z in self.elements], bin_cols=list(self.bin_cols),
                    num_cols=list(self.num_cols), glob_cols=list(self.trep["glob_cols"]),
                    token_version=self.trep.get("token_version"), d_in=int(self.d_in), d_glob=int(self.d_glob))

    def input_columns(self):
        """names of the model input tensor columns: x = element one-hot | bin | num, glob = the 182 columns (C-6)."""
        return dict(x=[f"Z={z}" for z in self.elements] + list(self.bin_cols) + list(self.num_cols),
                    glob=list(self.trep["glob_cols"]))

    def _ids(self, rows):
        return [str(q) for q in self.P.index.to_numpy()[rows]]

    def rxn_rows(self, rows):
        rows = np.asarray(rows, dtype=int)
        return np.unique(np.r_[self.ra[rows], self.rb[rows]])

    # ---------------------------------------------------------------- splits (= LadData.split 798-829)
    def split(self, a, b=None):
        """split(k) / split('T1', k) -> {train, test, inner_train, inner_val}: pair-table rows of the usable pairs +
        dropped counts + list sha256, LadData leak checks. split('all'): every usable in_T_main pair (Phase W)."""
        k = a if b is None else b
        if b is not None and a != SPLIT:
            C.stop(f"split {a}: DEPLOY uses T1 only")
        if str(k) == "all":
            return self._split_all()
        rec = self.folds[int(k)]
        out, dropped = {}, {}
        for key in FOLD_KEYS:
            rows = np.array([self.pid_row[q] for q in rec[key]], dtype=int)
            keep = self.usable_pair[rows]
            out[key] = rows[keep]
            dropped[key] = dict(no_tokens=int((~self.has_tok[rows]).sum()), not_usable=int((~keep).sum()))
        st = {key: set(out[key].tolist()) for key in FOLD_KEYS}
        if st["test"] & st["train"]:
            C.stop(f"{self.rung} fold {k}: test pair in train")
        if st["inner_train"] | st["inner_val"] != st["train"] or st["inner_train"] & st["inner_val"]:
            C.stop(f"{self.rung} fold {k}: inner folds do not partition train")
        rx = lambda r: set(self.ra[r].tolist()) | set(self.rb[r].tolist())          # noqa: E731
        if rx(out["test"]) & rx(out["train"]):
            C.stop(f"{self.rung} fold {k}: test reaction in train")
        ids = {key: self._ids(out[key]) for key in FOLD_KEYS}
        out.update(dropped=dropped,
                   pairs_sha=list_sha(ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"] + ["|"]
                                      + ids["inner_val"]),
                   val_sha=list_sha(ids["inner_val"]), test_sha=list_sha(ids["test"]))
        return out

    def _split_all(self):
        """spec 8-2 / 13: usable in_T_main pairs (union of the fold train lists), pair-table order."""
        ids = sorted({q for f in self.folds.values() for q in f["train"]})
        if S_FOLDS.is_file():
            want = (self.srep.get("inputs_sha256") or {}).get("S_folds")
            if want is not None and C.sha256(S_FOLDS) != want:
                C.stop(f"{S_FOLDS} sha256 differs from FU2 S_splits.json inputs_sha256.S_folds")
            S = pd.read_csv(S_FOLDS, usecols=["pair_id", "in_T_main"])
            tm = set(S.loc[S.in_T_main.astype(bool), "pair_id"].astype(str))
            if tm != set(ids):
                C.stop(f"union of the T1 fold train lists ({len(ids)}) != S_folds in_T_main ({len(tm)})")
        rows = np.array(sorted(self.pid_row[q] for q in ids), dtype=int)
        keep = self.usable_pair[rows]
        tr = rows[keep]
        e = np.zeros(0, dtype=int)
        tid = self._ids(tr)
        return dict(train=tr, test=e, inner_train=e, inner_val=e,
                    dropped=dict(train=dict(no_tokens=int((~self.has_tok[rows]).sum()), not_usable=int((~keep).sum()))),
                    pairs_sha=list_sha(tid), train_sha=list_sha(tid), val_sha=None, test_sha=None,
                    n_in_T_main=int(len(rows)))


_DATA: dict = {}


def get_data(source="lf", labels=DEFAULT_LABELS):
    key = (source, str(_abs(labels)))
    if key not in _DATA:
        _DATA[key] = DeployData(source, labels)
    return _DATA[key]


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["summary"] and len(a) <= 2:
        D = DeployData(a[1] if len(a) == 2 else "lf")
        out = dict(source=D.source, usable_rule=USABLE_RULES[D.usable_rule], n_tokens=int(len(D.rids)),
                   n_usable=int(D.usable_rxn.sum()), n_pairs_P=len(D.pair_list), pair_list_sha=D.pair_list_sha,
                   folds={k: {key: int(len(v)) for key, v in D.split(k).items() if key in FOLD_KEYS} for k in range(NF)},
                   all=int(len(D.split("all")["train"])), sha=D.sha)
        print(json.dumps(out, indent=1))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
