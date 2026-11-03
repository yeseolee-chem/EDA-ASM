"""DEPLOY-MODEL-W common paths, constants and helpers (docs/specs/DEPLOY_MODEL_W.md 2).

Imports dep_common (paths, sha256, atomic writers, STOP / not ready) and adds the W rules:
  writes only under $DEPW (= $DEP/w) and analysis/deploy/results/w (write guard, checked before the dep_common guard);
  group key = (b_rxn, fam, pos) from D1PAIR S_folds.csv; V_nl (spec 5), the line-residual RMS shared with H (spec 7);
  the within-group sampling design (spec 3-1 / W-2) and the graph readout (spec 0-4).
Light at import (stdlib + numpy / pandas); torch is never imported here."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent                      # analysis/deploy/w
DEPLOY_DIR = HERE.parent                                   # analysis/deploy
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))
import dep_common as C  # noqa: E402

REPO = C.REPO
DEPW = Path(os.environ.get("DEPW_SCRATCH", C.DEP / "w"))
RESW = Path(os.environ.get("DEPW_RESULTS", C.RES / "w"))
WITHIN = DEPW / "within"
UNITS_W = DEPW / "units"
PRED_W = DEPW / "within_pred"
MODEL_W = DEPW / "model"
SMOKE_W = DEPW / "smoke"
EVAL_W = DEPW / "eval"
I2_DIR = DEPW / "I2"
DEV = DEPW / "dev"
SPEC_W = REPO / "docs" / "specs" / "DEPLOY_MODEL_W.md"
S_FOLDS = REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"

# ---------------------------------------------------------------- constants (spec 2)
R_GRID = (0, 1, 4)
MAE_GUARD = 1.05
MIN_GROUP_VAL = 6                    # V_nl: minimum pairs (with sigma) of an inner-validation group
MIN_FAM_EVAL = 12                    # evaluation family: >= 12 points incl. H (= hammett_eval.family_series)
PRIMARY = tuple(C.PRIMARY)           # d1 d2 elst pauli oi disp cpcm cds barrier
TARGETS = tuple(C.TARGETS)
I_PRIM = [TARGETS.index(t) for t in PRIMARY]
SEED = 20261019
N_BOOT = 2000
RNG_W_OFFSET = 7919                  # W-3: rng_w = default_rng(seed + 7919)
SPLIT, NF = "T1", 5
GROUP_KEY = ("b_rxn", "fam", "pos")
VNL_POS = ("p", "m")
RUNGS = ("lfW", "lfV")
GRAPH_MAX = 80                       # 13-1: a group larger than this is read directly (predict_tf extension)
S_FOLDS_COLS = ("pair_id", "ref_kind", "a_rxn", "b_rxn", "core", "fam", "pos", "Xb", "sigma", "in_T_main",
                "in_P_main", "fold_T1")
assert SEED == C.SEED and N_BOOT == C.N_BOOT

WRITE_OK_W = (DEPW, RESW)
stop, not_ready, Stop = C.stop, C.not_ready, C.Stop
sha256, sha256_text, read_json = C.sha256, C.sha256_text, C.read_json


# ---------------------------------------------------------------- write guard + writers
def assert_write_ok(path) -> Path:
    p = Path(os.path.abspath(path))
    for root in WRITE_OK_W:
        r = Path(os.path.abspath(root))
        if p == r or r in p.parents:
            return C.assert_write_ok(p)
    stop(f"W write outside {[str(x) for x in WRITE_OK_W]}: {p}")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path, obj):
    return C.write_json(assert_write_ok(path), _jsonable(obj))


def write_text(path, text):
    return C.write_text(assert_write_ok(path), text)


def write_parquet(path, df):
    return C.write_parquet(assert_write_ok(path), df)


def write_csv(path, df):
    return C.write_csv(assert_write_ok(path), df)


def write_npz(path, **arrays):
    """atomic np.savez (no pickle: numeric and fixed-width unicode arrays only)."""
    p = assert_write_ok(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, p)
    return p


def code_sha(files):
    return {str(Path(f).resolve().relative_to(REPO)): sha256(f) for f in files}


W_CODE = ("w_common.py", "within.py", "train_w.py", "within_predict.py", "eval_w.py", "prereg_w.py", "tests_w.py")


def w_code_sha():
    return {f"analysis/deploy/w/{f}": (sha256(HERE / f) if (HERE / f).is_file() else None) for f in W_CODE}


# ---------------------------------------------------------------- S_folds (0-3) and groups
def load_s_folds(path=S_FOLDS):
    """S_folds.csv with the 0-3 columns (STOP when one is missing); pair_id / pos / ref_kind / fam as str."""
    if not Path(path).is_file():
        not_ready(f"{path} missing")
    S = pd.read_csv(path)
    miss = [c for c in S_FOLDS_COLS if c not in S.columns]
    if miss:
        stop(f"{path}: columns missing {miss} (start condition 0-3)")
    S = S.assign(pair_id=S.pair_id.astype(str), pos=S.pos.fillna("").astype(str), fam=S.fam.fillna("").astype(str),
                 ref_kind=S.ref_kind.fillna("").astype(str), a_rxn=S.a_rxn.astype(int), b_rxn=S.b_rxn.astype(int))
    if S.pair_id.duplicated().any():
        stop(f"{path}: duplicated pair_id")
    return S.set_index("pair_id")


def group_table(S):
    """global group ids: sorted unique (b_rxn, fam, pos) over every S_folds pair -> DataFrame gid, b_rxn, fam, pos and a
    per-pair Series gid (index pair_id). Missing key = STOP (spec 15 B)."""
    bad = S[(S.fam == "") | (S.pos == "") | S.b_rxn.isna()]
    if len(bad):
        stop(f"S_folds: {len(bad)} pairs without a group key (b_rxn, fam, pos): {list(bad.index[:5])}")
    G = S[list(GROUP_KEY)].drop_duplicates().sort_values(list(GROUP_KEY)).reset_index(drop=True)
    G.insert(0, "gid", np.arange(len(G), dtype=np.int64))
    key = pd.MultiIndex.from_frame(G[list(GROUP_KEY)])
    pos = key.get_indexer(pd.MultiIndex.from_frame(S[list(GROUP_KEY)]))
    if (pos < 0).any():
        stop("group table lookup failed")
    return G, pd.Series(G.gid.to_numpy()[pos], index=S.index, name="gid")


def check_pairs_vs_sfolds(S, P):
    """0-3 / spec 15 B: every pair of the pair table P (index pair_id, columns a_rxn, b_rxn) is in S_folds with the same
    a_rxn and b_rxn (STOP otherwise). Returns counts of fam / pos / sigma agreement (information)."""
    miss = sorted(set(P.index) - set(S.index))
    if miss:
        stop(f"{len(miss)} pairs of the label table are not in S_folds.csv: {miss[:5]}")
    s = S.reindex(P.index)
    da = int((s.a_rxn.to_numpy() != P.a_rxn.to_numpy().astype(int)).sum())
    db = int((s.b_rxn.to_numpy() != P.b_rxn.to_numpy().astype(int)).sum())
    if da or db:
        stop(f"S_folds.csv vs the label table: a_rxn differs for {da}, b_rxn for {db} pairs (start condition 0-3)")
    info = dict(n_pairs=int(len(P)), a_rxn_equal=True, b_rxn_equal=True)
    for c in ("fam", "pos", "Xb"):
        if c in P.columns:
            info[f"{c}_equal"] = int((s[c].astype(str).to_numpy() == P[c].fillna("").astype(str).to_numpy()).sum())
    if "sigma" in P.columns:
        a, b = s.sigma.to_numpy(float), P.sigma.to_numpy(float)
        info["sigma_equal"] = int(((a == b) | (np.isnan(a) & np.isnan(b))).sum())
    return info


# ---------------------------------------------------------------- line residual RMS, V_nl (spec 5), H (spec 7)
def line_rms(x, E):
    """RMS of the least-squares line residuals: x (n,), E (n,) or (n, m) -> float or (m,). The caller includes the H
    point (0, 0) in x / E."""
    x = np.asarray(x, dtype=float)
    E = np.asarray(E, dtype=float)
    one = E.ndim == 1
    Y = E[:, None] if one else E
    A = np.c_[np.ones(len(x)), x]
    coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    r = np.sqrt(((Y - A @ coef) ** 2).mean(0))
    return float(r[0]) if one else r


def with_h(x, E):
    """append the H reference point (sigma 0, error 0)."""
    x = np.asarray(x, dtype=float)
    E = np.asarray(E, dtype=float)
    return np.r_[x, 0.0], (np.r_[E, 0.0] if E.ndim == 1 else np.vstack([E, np.zeros((1, E.shape[1]))]))


def vnl(e_kcal, groups, sy, prim=I_PRIM):
    """V_nl (spec 5): e_kcal [n_val, 10] = yhat - y (kcal/mol) of the inner-validation pairs; groups = list of
    dict(idx = positions in the validation list, sigma) (p / m groups with >= MIN_GROUP_VAL pairs, see within.val_groups);
    sy [10] = the fit's std.sy. V_nl = mean_{t in PRIMARY} [ mean_g r_gt / sy_t ], r_gt = line_rms over the group's
    (sigma, e) + (0, 0). NaN without groups."""
    if not groups:
        return float("nan")
    e = np.asarray(e_kcal, dtype=float)[:, prim]
    R = np.stack([line_rms(*with_h(g["sigma"], e[g["idx"]])) for g in groups])        # [n_groups, 9]
    return float((R.mean(0) / np.asarray(sy, dtype=float)[prim]).mean())


# ---------------------------------------------------------------- sampling design (spec 3-1, W-2)
def inclusion(w, n):
    """inclusion probabilities of a without-replacement design with pi_i = min(1, c w_i) and sum(pi) = min(n, N):
    the expected count of a group is proportional to its weight sum (= n_g with w = 2 / (n_g - 1)) except where the
    group's pool is exhausted (pi = 1: every within pair of that group is drawn, 'all if the pool is small')."""
    w = np.asarray(w, dtype=float)
    N = len(w)
    if n >= N:
        return np.ones(N)
    if n <= 0:
        return np.zeros(N)
    cap = np.zeros(N, dtype=bool)
    while True:
        rest = ~cap
        c = (n - cap.sum()) / w[rest].sum()
        new = rest & (c * w >= 1.0)
        if not new.any():
            break
        cap |= new
    return np.where(cap, 1.0, c * w)


def draw(rng, pi):
    """one epoch's draw: systematic sampling over a random order (sorted indices; each index at most once; count =
    round(sum(pi)) up to floating rounding). Inclusion probability of item i = pi_i exactly."""
    N = len(pi)
    if N == 0:
        return np.zeros(0, dtype=np.int64)
    perm = rng.permutation(N)
    cs = np.cumsum(pi[perm])
    u = rng.random()
    hit = np.ceil(cs - u) - np.ceil(np.r_[0.0, cs[:-1]] - u)
    return np.sort(perm[hit > 0.5])


# ---------------------------------------------------------------- graph readout (spec 0-4)
def graph_values(u, v, g, node_group, m, ref_node):
    """graph readout for every node from unordered pair predictions g = g(u, v) (antisymmetric model: g(v, u) = -g):
    s_i = sum_j g(i, j) (j over the node's group, g(i, i) = 0); ghat_i = (s_i - s_ref) / m = (1/m) sum_j [g(i, j) -
    g(ref, j)]. u, v int node ids [n_pairs]; g [n_pairs] or [n_pairs, k]; node_group [n_nodes] group index; m [n_groups]
    group sizes; ref_node [n_groups] the reference node of each group -> ghat [n_nodes] (or [n_nodes, k])."""
    g = np.asarray(g, dtype=float)
    one = g.ndim == 1
    G = g[:, None] if one else g
    n_nodes = len(node_group)
    s = np.zeros((n_nodes, G.shape[1]))
    np.add.at(s, np.asarray(u), G)
    np.add.at(s, np.asarray(v), -G)
    node_group = np.asarray(node_group)
    out = (s - s[np.asarray(ref_node)[node_group]]) / np.asarray(m, dtype=float)[node_group][:, None]
    return out[:, 0] if one else out


def graph_direct_sum(M, ref):
    """reference implementation: full ordered matrix M[i, j] = g(i, j) of one group -> (1/m) sum_j [M[i, j] - M[ref, j]]."""
    M = np.asarray(M, dtype=float)
    return (M - M[ref][None, :]).sum(1) / M.shape[0]


# ---------------------------------------------------------------- start conditions (spec 1)
def check_start(free_gb=None):
    """spec 1, 0-1 .. 0-5 -> dict(all_pass, checks, detail). free_gb: disk headroom read on the login node with
    mmlsquota (compute nodes cannot read the quota); None = not checked (all_pass false)."""
    import prereg_deploy as PR
    chk, det = {}, {}
    ct = C.RES / "C_tests.json"
    c = read_json(ct) if ct.is_file() else {}
    try:
        PR.check()
        pre_ok = True
    except SystemExit as e:                               # noqa: PERF203
        pre_ok, det["prereg_deploy_check"] = False, f"exit {e.code}"
    er = C.RES / "E_report.json"
    e = read_json(er) if er.is_file() else {}
    chk["0-1"] = bool(c.get("all_pass")) and pre_ok and bool((e.get("G_C") or {}).get("pass") is True)
    det["0-1"] = dict(c_tests_all_pass=c.get("all_pass"), prereg_committed=pre_ok, G_C=(e.get("G_C") or {}).get("pass"))
    miss = []
    for k in range(NF):
        name = f"D__lf__S__TF__f{k}"
        for p in [C.UNITS / f"{name}.json", C.UNITS / f"{name}.parquet"] + [C.UNITS / f"{name}__s{j}.pt" for j in range(5)]:
            if not p.is_file():
                miss.append(p.name)
    kp = C.RES / "K_predictions.parquet"
    if not kp.is_file():
        miss.append(str(kp))
    chk["0-2"] = not miss
    det["0-2"] = dict(missing=miss)
    try:
        S = load_s_folds()
        import data_lf as DL
        P, _ = DL.load_labels(DL.DEFAULT_LABELS)
        det["0-3"] = check_pairs_vs_sfolds(S, P)
        chk["0-3"] = True
    except SystemExit as ex:
        chk["0-3"], det["0-3"] = False, f"exit {ex.code}"
    pj = C.RES / "prereg_deploy.json"
    rec = read_json(pj) if pj.is_file() else {}
    ins = rec.get("inputs_sha256") or {}
    need = [str(C.LABELS_LF), str(C.X182_LF), str(C.RES / "fu2_lf" / "T_manifest.csv"), str(C.LAD / "F_pairs_cr2.csv")]
    need += [k for k in ins if "/fu2/splits/T1__f" in k]
    diff = [p for p in need if p not in ins or not Path(p).is_file() or sha256(p) != ins[p]]
    chk["0-4"] = bool(ins) and not diff and len([k for k in ins if "/fu2/splits/T1__f" in k]) == NF
    det["0-4"] = dict(prereg_deploy_json=str(pj), differs_or_missing=diff)
    chk["0-5"] = free_gb is not None and float(free_gb) >= 20.0
    det["0-5"] = dict(free_gb=free_gb, need_gb=20, source="login-node mmlsquota (DEPW_FREE_GB)")
    return dict(all_pass=bool(all(chk.values())), checks=chk, detail=det)
