#!/usr/bin/env python3
r"""d0d1_splits.py — GEO2b D0 + D1 pair labels and split files for the LADDER stage-3 trainers (rung geo, split 'T1D0'),
and the reader rules both trainers import (user decision D0D1, 2026-10-31: analysis/geo1b/results/DECISIONS.md 'D0D1';
design analysis/geo1b/results/D0D1_design_investigation.json result.design step 9; docs/specs/GEO2.md 8-1).
Compute nodes only (sbatch / srun). Any env with numpy + pandas: conda reactot, env_espley, env_d1fu2 (python >= 3.9).

  python d0d1_splits.py build                  -> $GEO2B/F_pairs_d0d1.csv, $GEO2B/splits/T1D0__f<k>.json (k = 0..4),
                                                  analysis/geo2/results/S_splits_d0d1.json (written last)
  python d0d1_splits.py build --test-dir D [--allow-preliminary]
                                                  the same three outputs below D (a folder inside $GEO2B), never the
                                                  production paths; --allow-preliminary accepts the preliminary GEO1b
                                                  D0_pairs.csv (anchors from D0-G2 orig; the json then says preliminary
                                                  true and the trainers refuse it). Tests only.
  python d0d1_splits.py check [--test-dir D] [--allow-preliminary]
                                                  the trainers' reader check (load_s_splits) on the written files
  exit 0 = done / passed, 2 = not ready (an input missing; nothing written), 3 = STOP (nothing written).

Inputs (read only): $LAD/F_pairs_cr2.csv (LADDER y; results/V_report.json all_pass and outputs sha256),
  analysis/d1_pair/results/F_pairs.csv (header / D1 pair order), the FU2 T1 split files $D1PAIR_SCRATCH/fu2/splits/
  T1__f<k>.json (sha256 = analysis/d1_pair/results/fu2/S_splits.json units), the GEO1b D0 files $GEO1B/d0/
  {D0_pairs.csv, d0_folds.csv, d0_twins.csv} with analysis/geo1b/results/{D0_pairs_report.json, Q_d0_report.json}
  of THIS worktree (they must describe the files in $GEO1B/d0: bring the final GEO1b commit into geo2b first).

Rules (fixed before any geo result; DECISIONS_lad.md LD-*)
  labels   F_pairs_d0d1.csv = the bytes of F_pairs_cr2.csv (header + every D1 row byte-identical), then one row per
           D0_pairs.csv row, in its order, in the F_pairs.csv columns: every field copied as text from D0_pairs.csv
           (pair_id D#####, ref_kind 'd0', pos 'd0', fam = core = 'D0/<component>', D1-only columns empty); the two extra
           D0_pairs.csv columns edge_kind / ok are not copied (join D0_pairs.csv by pair_id). All D0 rows are in the
           label file; only energy-eligible ones (ok True) enter a split list.
  splits   T1D0__f<k>.json = {split 'T1D0', fold k, train, test, inner_train, inner_val}: each list = the FU2
           T1__f<k>.json list exactly (same ids, same order) followed by the D0 pairs (ok True, sorted pair_id):
           train and inner_train += D0 pairs of D0 fold != k; test += D0 pairs of fold k; inner_val += nothing (D1
           only: the REP / REP8 validation pairs keep their LADDER 13-1 meaning). D0 pairs never enter the GroupKFold
           call (arm P precedent): train_xgb.py adds them to every inner training side.
  D0 fold  of a pair = d0_folds.csv fold of a_rxn = fold of b_rxn (0..4; a pair across folds or with a D-1 overlap
           reaction = STOP). Folds are frozen in GEO1b (d0_pairs.py P-frozen); D0_pairs.csv is rebuilt after G2c.
  D0 inputs must be the final GEO1b build: D0_pairs_report.json stop false, g2_source 'g2c', preliminary false, status
           'ok', outputs sha256 = the files; Q_d0_report.json all_pass true, stop false, its sha256 = the D0_pairs_report
           inputs record and its labels_d0_q.json sha256 = the one D0_pairs_report read (design 'provenance via the
           Q_d0_report sha'). Twins: no D0 training reaction of fold k has a D1 twin bound to fold k (d0_twins.csv
           kind forced / iso_resolved; accepted conflicts are reported only).
Reader contract (train_atoms.py / train_xgb.py split 'T1D0', tests/run_tests.py geo): load_s_splits() = S_splits_d0d1.json
  all_pass true, stop false, preliminary false (ALLOW_PRELIMINARY is a test-only module switch), outputs sha256 = the
  label file and the 5 split files now, recorded data inputs (F_pairs_cr2.csv, F_pairs.csv, FU2 S_splits.json + T1
  files, D0_pairs.csv, d0_folds.csv, d0_twins.csv) unchanged, d0_provenance() passes now, and check_lists() holds for
  every fold with the d0_folds.csv folds and the ok set of D0_pairs.csv. l11_d0d1() = the label-file invariant L-11'.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent                                  # analysis/geo2
REPO = HERE.parents[1]
RES = HERE / "results"
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2b"))       # = geo2_common.GEO2 ($GEO2B)
LAD = Path(os.environ.get("D1LADDER_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder"))  # = lad_common.LAD
D1PAIR_SCR = Path(os.environ.get("D1PAIR_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair"))  # = d1pair_common.SCRATCH
CODE_VERSION = "geo2b-d0d1-1"
SPLIT = "T1D0"
NF = 5
FOLD_KEYS = ("train", "test", "inner_train", "inner_val")
CR2 = LAD / "F_pairs_cr2.csv"                                           # = train_atoms.DEFAULT_LABELS
ORIG = REPO / "analysis" / "d1_pair" / "results" / "F_pairs.csv"         # = train_atoms.ORIG_LABELS
V_REPORT = REPO / "analysis" / "d1_ladder" / "results" / "V_report.json"
FU2_S_SPLITS = REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json"
FU2_SPLITS = D1PAIR_SCR / "fu2" / "splits"                              # = fu2_common.SPLITS
D0_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0")                       # $GEO1B/d0 (GEO1b d0_pairs.py; read only)
D0_PAIRS = D0_DIR / "D0_pairs.csv"
D0_FOLDS = D0_DIR / "d0_folds.csv"
D0_TWINS = D0_DIR / "d0_twins.csv"
GEO1B_RES = REPO / "analysis" / "geo1b" / "results"
D0_REPORT = GEO1B_RES / "D0_pairs_report.json"
Q_D0_REPORT = GEO1B_RES / "Q_d0_report.json"
Q_REL = "analysis/geo1b/results/Q_d0_report.json"                       # repo-relative key (D1-G3 F-13 rule)
LABELS = GEO2 / "F_pairs_d0d1.csv"                                      # production outputs
SPLITS = GEO2 / "splits"
S_SPLITS = RES / "S_splits_d0d1.json"
ALLOW_PRELIMINARY = False                                               # tests only; never set in a training job
D0_EXTRA = ("edge_kind", "ok")                                          # D0_pairs.csv columns after the F_pairs ones
D0_FOLD_OVERLAP = -9                                                    # = d0_pairs.FOLD_OVERLAP
FOLD_COLS = ["rxn_id", "component", "fold", "twin_of", "overlap"]
TWIN_BINDING = ("forced", "iso_resolved")                               # d0_twins.csv kinds that set the D0 fold
RULES = dict(
    labels="F_pairs_cr2.csv bytes (D1 rows byte-identical) + every D0_pairs.csv row in the F_pairs.csv columns (text "
           "copy; edge_kind / ok not copied)",
    splits="T1D0__f<k> = FU2 T1__f<k> lists exactly + D0 pairs (ok True, sorted pair_id): train / inner_train += D0 "
           "fold != k, test += D0 fold k, inner_val += nothing; D0 never in the GroupKFold call",
    d0_fold="pair fold = d0_folds.csv fold of a_rxn = of b_rxn (0..4)",
    d0_inputs="D0_pairs_report.json stop false, g2_source g2c, preliminary false, status ok, outputs sha256 = files; "
              "Q_d0_report.json all_pass, stop false, sha256 = D0_pairs_report inputs record",
    twins="no D0 training reaction of fold k with a D1 twin bound to fold k (forced / iso_resolved)")


class Stop(Exception):
    """a rule failed (exit 3)."""


class NotReady(Exception):
    """an input is missing (exit 2)."""


# ================================================================ helpers
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def ab(p):
    return str(Path(os.path.abspath(str(p))))


def read_json(p, what):
    p = Path(p)
    if not p.is_file():
        raise NotReady(f"{p} missing ({what})")
    try:
        return json.loads(p.read_text())
    except ValueError as e:
        raise Stop(f"{p} unreadable ({e})") from None


def is_d0_id(q):
    """D0 pair ids are D##### (GEO1b d0_pairs.py); D1 ids are P##### / A#####."""
    return str(q).startswith("D")


def _assert_out(p):
    q = Path(os.path.abspath(str(p)))
    for root in (GEO2, RES):
        r = Path(os.path.abspath(str(root)))
        if q == r or r in q.parents:
            return q
    raise Stop(f"write path {q} outside $GEO2B ({GEO2}) / analysis/geo2/results")


def write_bytes_atomic(p, data):
    p = _assert_out(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return p


def split_name(k):
    return f"{SPLIT}__f{int(k)}.json"


def set_paths(labels=None, splits=None, s_splits=None):
    """point the outputs at other files (tests only; e.g. a --test-dir build); returns the previous values."""
    global LABELS, SPLITS, S_SPLITS
    old = (LABELS, SPLITS, S_SPLITS)
    LABELS = Path(labels) if labels is not None else LABELS
    SPLITS = Path(splits) if splits is not None else SPLITS
    S_SPLITS = Path(s_splits) if s_splits is not None else S_SPLITS
    return old


def test_paths(test_dir):
    d = _assert_out(Path(test_dir) / "probe").parent
    return d / "F_pairs_d0d1.csv", d / "splits", d / "S_splits_d0d1.json"


# ================================================================ inputs
def fu2_t1():
    """the FU2 T1 split files (sha256 = FU2 S_splits.json units) -> ({k: list dict}, {path: sha256})."""
    srep = read_json(FU2_S_SPLITS, "FU2 S_splits.json")
    out, shas = {}, {}
    for k in range(NF):
        name = f"T1__f{k}.json"
        p = FU2_SPLITS / name
        if not p.is_file():
            raise NotReady(f"{p} missing")
        s = sha256(p)
        if s != ((srep.get("units") or {}).get(name) or {}).get("sha256"):
            raise Stop(f"{p} sha256 differs from {FU2_S_SPLITS} units.{name}")
        out[k] = json.loads(p.read_text())
        shas[ab(p)] = s
    return out, shas


def d0_provenance(allow_preliminary=None):
    """the GEO1b D0 inputs are a passed, final build (module docstring 'D0 inputs'); -> record. Stop / NotReady."""
    allow = ALLOW_PRELIMINARY if allow_preliminary is None else bool(allow_preliminary)
    rep = read_json(D0_REPORT, "GEO1b d0_pairs.py build")
    bad = []
    if rep.get("stop") is not False:
        bad.append(f"{D0_REPORT.name} stop = {rep.get('stop')!r}")
    prelim = bool(rep.get("preliminary") is not False or rep.get("g2_source") != "g2c" or rep.get("status") != "ok")
    if prelim and not allow:
        raise Stop(f"{D0_REPORT} is preliminary (g2_source {rep.get('g2_source')!r}, preliminary "
                   f"{rep.get('preliminary')!r}, status {rep.get('status')!r}): rerun d0_pairs.py build after "
                   f"select_g2c.py build (GEO1b DECISIONS_PAIR P-g2-source) and bring that analysis/geo1b/results "
                   f"commit into this worktree")
    outs = rep.get("outputs") or {}

    def out_sha(p):
        hits = [v.get("sha256") for k, v in outs.items() if Path(k).name == p.name and isinstance(v, dict)]
        return hits[0] if len(hits) == 1 else None
    shas = {}
    for p in (D0_PAIRS, D0_FOLDS, D0_TWINS):
        if not p.is_file():
            raise NotReady(f"{p} missing (GEO1b d0_pairs.py build)")
        shas[p.name] = sha256(p)
        if out_sha(p) != shas[p.name]:
            bad.append(f"{p} sha256 {shas[p.name][:12]} != {D0_REPORT.name} outputs {str(out_sha(p))[:12]}")
    q = read_json(Q_D0_REPORT, "GEO1b d0_labels.py run")
    if q.get("all_pass") is not True or q.get("stop") is not False:
        bad.append(f"{Q_D0_REPORT.name}: all_pass {q.get('all_pass')!r}, stop {q.get('stop')!r}")
    qsha = sha256(Q_D0_REPORT)
    rec_q = [v for k, v in (rep.get("inputs_sha256") or {}).items()
             if str(k).replace(os.sep, "/").endswith(Q_REL)]
    if not rec_q or any(v != qsha for v in rec_q):
        bad.append(f"{Q_D0_REPORT} sha256 {qsha[:12]} != the {D0_REPORT.name} inputs record {[str(v)[:12] for v in rec_q]}")
    lab_q = [v for k, v in (q.get("outputs") or {}).items() if Path(k).name == "labels_d0_q.json"]
    lab_p = [v for k, v in (rep.get("inputs_sha256") or {}).items() if Path(k).name == "labels_d0_q.json"]
    qr = rep.get("q_report") or {}
    if not (len(lab_q) == 1 and lab_p == lab_q and qr.get("labels_sha256") == lab_q[0] and qr.get("all_pass") is True
            and qr.get("stop") is False):
        bad.append(f"labels_d0_q.json sha256: Q_d0_report outputs {lab_q}, D0_pairs_report inputs {lab_p}, q_report "
                   f"{qr.get('labels_sha256')} (all_pass {qr.get('all_pass')}, stop {qr.get('stop')})")
    if bad:
        raise Stop("GEO1b D0 inputs not a consistent passed build: " + "; ".join(bad))
    return dict(report=ab(D0_REPORT), report_sha256=sha256(D0_REPORT), report_code_version=rep.get("code_version"),
                q_report=ab(Q_D0_REPORT), q_report_sha256=qsha, labels_d0_q_sha256=lab_q[0],
                g2_source=rep.get("g2_source"), status=rep.get("status"), preliminary=prelim,
                d0_pairs_sha256=shas[D0_PAIRS.name], d0_folds_sha256=shas[D0_FOLDS.name],
                d0_twins_sha256=shas[D0_TWINS.name], conflict_policy=(rep.get("twins") or {}).get("conflict_policy"))


def read_d0_folds():
    """d0_folds.csv -> {rxn_id: (fold, overlap)} (GEO1b geo_common.load_d0_folds rules: overlap = fold -9)."""
    F = pd.read_csv(D0_FOLDS, dtype=str, keep_default_na=False)
    if list(F.columns) != FOLD_COLS:
        raise Stop(f"{D0_FOLDS}: columns {list(F.columns)} != {FOLD_COLS}")
    out = {}
    for r in F.itertuples(index=False):
        rid, fold, ov = int(r.rxn_id), int(r.fold), r.overlap
        if rid in out or ov not in ("True", "False"):
            raise Stop(f"{D0_FOLDS}: rxn {rid} duplicated or overlap {ov!r}")
        ov = ov == "True"
        if (ov and fold != D0_FOLD_OVERLAP) or (not ov and fold not in range(NF)):
            raise Stop(f"{D0_FOLDS}: rxn {rid} overlap {ov} fold {fold}")
        out[rid] = (fold, ov)
    return out


def read_twins():
    """d0_twins.csv -> {D0 rxn_id: (kind, {T1 folds of the binding twins})} (GEO1b P-twin-table)."""
    T = pd.read_csv(D0_TWINS, dtype=str, keep_default_na=False)
    out = {}
    for r in T.itertuples(index=False):
        folds = {int(x.split(":")[1]) for x in str(r.bind).split() if ":" in x}
        out[int(r.rxn_id)] = (str(r.kind), folds)
    return out


def cr2_columns():
    with open(CR2, newline="") as f:
        return next(csv.reader(f))


def read_d0_pairs(cols=None):
    """D0_pairs.csv as text (dtype str, no NA conversion); header = the F_pairs columns + edge_kind, ok (checked)."""
    cols = cols if cols is not None else cr2_columns()
    T = pd.read_csv(D0_PAIRS, dtype=str, keep_default_na=False)
    if list(T.columns) != list(cols) + list(D0_EXTRA):
        raise Stop(f"{D0_PAIRS}: columns {list(T.columns)} != F_pairs columns + {list(D0_EXTRA)}")
    return T


def pair_folds(T, folds):
    """D0 pair -> fold (a / b in the same fold 0..4, no overlap reaction); Stop otherwise."""
    out, bad = {}, []
    for pid, a, b in zip(T.pair_id, T.a_rxn, T.b_rxn):
        fa, fb = folds.get(int(a)), folds.get(int(b))
        if fa is None or fb is None or fa[1] or fb[1] or fa[0] != fb[0]:
            bad.append(f"{pid} ({a}: {fa}, {b}: {fb})")
            continue
        out[str(pid)] = fa[0]
    if bad:
        raise Stop(f"{len(bad)} D0 pairs without one common D0 fold 0..4: {bad[:5]}")
    return out


# ================================================================ rules (builder = readers)
def d0_rows_frame(label_bytes, cr2_bytes):
    """the D0 rows of a label file (bytes after the F_pairs_cr2.csv prefix) as a text frame with the cr2 header."""
    head = cr2_bytes.split(b"\n", 1)[0] + b"\n"
    return pd.read_csv(io.BytesIO(head + label_bytes[len(cr2_bytes):]), dtype=str, keep_default_na=False)


def l11_d0d1(label_bytes=None, path=None):
    """L-11' on the label file: (1) its first len(F_pairs_cr2.csv) bytes are that file (header + D1 rows byte-identical);
    (2) the rest, read as text with that header, equals the F_pairs columns of D0_pairs.csv (every field, same order);
    (3) the D0 ids are D#####, unique, ref_kind 'd0', none a D1 id. -> record; Stop otherwise."""
    b = label_bytes if label_bytes is not None else Path(path or LABELS).read_bytes()
    c = CR2.read_bytes()
    if not c.endswith(b"\n") or b[:len(c)] != c:
        raise Stop(f"label file: the first {len(c)} bytes are not {CR2} (D1 rows must be byte-identical)")
    cols = cr2_columns()
    D0 = d0_rows_frame(b, c)
    T = read_d0_pairs(cols)[cols].reset_index(drop=True)
    if list(D0.columns) != cols or len(D0) != len(T) or not D0.equals(T):
        raise Stop(f"label file D0 rows ({len(D0)}) != the F_pairs columns of {D0_PAIRS} ({len(T)} rows)")
    ids = D0.pair_id.tolist()
    d1 = set(pd.read_csv(CR2, usecols=["pair_id"], dtype=str).pair_id)
    bad = [q for q in ids if not (is_d0_id(q) and len(q) == 6 and q[1:].isdigit())]
    if bad or len(set(ids)) != len(ids) or set(ids) & d1 or (D0.ref_kind != "d0").any() \
            or any(is_d0_id(q) for q in d1):
        raise Stop(f"D0 ids: malformed {bad[:3]}, duplicated {len(ids) - len(set(ids))}, shared with D1 "
                   f"{sorted(set(ids) & d1)[:3]}, ref_kind not d0 {int((D0.ref_kind != 'd0').sum())}")
    return dict(d1_prefix_bytes=len(c), d1_prefix_equal_F_pairs_cr2=True, n_d1_rows=len(d1), n_d0_rows=len(D0),
                d0_rows_equal_D0_pairs_text=True, F_pairs_cr2_sha256=sha256_bytes(c), D0_pairs_sha256=sha256(D0_PAIRS))


def check_lists(k, f, t1, pfold=None, ok_ids=None):
    """rules of one T1D0__f<k>.json dict against the FU2 T1__f<k>.json dict (module docstring 'splits'); pfold = D0
    pair -> fold, ok_ids = the energy-eligible D0 pairs; -> counts. Stop otherwise."""
    k = int(k)
    if f.get("split") != SPLIT or int(f.get("fold", -1)) != k:
        raise Stop(f"{split_name(k)}: split {f.get('split')!r} fold {f.get('fold')!r}")
    d0 = {}
    for key in FOLD_KEYS:
        lst = [str(q) for q in f.get(key) or []]
        a = [q for q in lst if not is_d0_id(q)]
        b = [q for q in lst if is_d0_id(q)]
        if a != [str(q) for q in t1[key]]:
            raise Stop(f"{split_name(k)} {key}: the D1 list differs from FU2 T1__f{k}.json")
        if lst != a + b or b != sorted(set(b)):
            raise Stop(f"{split_name(k)} {key}: not 'FU2 list, then sorted unique D0 pairs'")
        d0[key] = b
    if d0["inner_val"]:
        raise Stop(f"{split_name(k)}: {len(d0['inner_val'])} D0 pairs in inner_val (validation is D1 only)")
    if d0["inner_train"] != d0["train"]:
        raise Stop(f"{split_name(k)}: D0 pairs of inner_train != those of train")
    if set(d0["train"]) & set(d0["test"]):
        raise Stop(f"{split_name(k)}: a D0 pair in train and test")
    if pfold is not None:
        bt = [q for q in d0["train"] if pfold.get(q) in (None, k)]
        be = [q for q in d0["test"] if pfold.get(q) != k]
        if bt or be:
            raise Stop(f"{split_name(k)}: D0 train pairs of fold {k} / unknown {bt[:3]}, D0 test pairs of another fold "
                       f"{be[:3]}")
    if pfold is not None and ok_ids is not None:
        want_tr = sorted(q for q in ok_ids if pfold[q] != k)
        want_te = sorted(q for q in ok_ids if pfold[q] == k)
        if d0["train"] != want_tr or d0["test"] != want_te:
            raise Stop(f"{split_name(k)}: D0 lists != the ok D0 pairs of folds != {k} / = {k}")
    return dict(n={key: len(f[key]) for key in FOLD_KEYS}, n_d0={key: len(d0[key]) for key in FOLD_KEYS})


def twins_check(lists, T, twins):
    """no D0 training reaction of fold k with a binding D1 twin of T1 fold k -> record; Stop otherwise."""
    rx = {str(p): (int(a), int(b)) for p, a, b in zip(T.pair_id, T.a_rxn, T.b_rxn)}
    bad, info = [], {}
    for k, f in lists.items():
        tr = {r for q in f["train"] if is_d0_id(q) for r in rx[q]}
        hit = sorted(r for r in tr if r in twins and twins[r][0] in TWIN_BINDING and k in twins[r][1])
        soft = sorted(r for r in tr if r in twins and twins[r][0] not in TWIN_BINDING and k in twins[r][1])
        if hit:
            bad.append(f"fold {k}: {hit[:5]}")
        info[str(k)] = dict(n_twin_rxn_in_train=sum(r in twins for r in tr), reported_only=soft)
    if bad:
        raise Stop(f"D0 training reactions with a D1 twin tested in the same fold: {bad}")
    return dict(n_twins=len(twins), by_fold=info, rule=RULES["twins"])


def make_lists(t1, okd, pfold):
    """{k: T1D0 dict} from the FU2 T1 dicts, the ok D0 pair ids and their folds."""
    out = {}
    for k in range(NF):
        tr = sorted(q for q in okd if pfold[q] != k)
        te = sorted(q for q in okd if pfold[q] == k)
        out[k] = dict(split=SPLIT, fold=k, train=list(t1[k]["train"]) + tr, test=list(t1[k]["test"]) + te,
                      inner_train=list(t1[k]["inner_train"]) + tr, inner_val=list(t1[k]["inner_val"]))
    return out


def split_bytes(f):
    return (json.dumps(f, indent=1) + "\n").encode()


# ================================================================ reader (trainers, tests)
_CACHE: dict = {}


def load_s_splits(allow_preliminary=None):
    """reader check (module docstring 'Reader contract') -> (rec, {k: list dict}, {role: path}). Cached per process for
    the same files (a changed file sha256 re-reads)."""
    allow = ALLOW_PRELIMINARY if allow_preliminary is None else bool(allow_preliminary)
    key = (ab(S_SPLITS), ab(LABELS), ab(SPLITS), allow)
    rec = read_json(S_SPLITS, "d0d1_splits.py build")
    rsha = sha256(S_SPLITS)
    if key in _CACHE and _CACHE[key][0] == rsha:
        return _CACHE[key][1]
    if rec.get("all_pass") is not True or rec.get("stop") is not False or rec.get("split") != SPLIT:
        raise Stop(f"{S_SPLITS}: all_pass {rec.get('all_pass')!r}, stop {rec.get('stop')!r}, split {rec.get('split')!r}")
    if rec.get("preliminary") is not False and not allow:
        raise Stop(f"{S_SPLITS} was built from the preliminary GEO1b D0_pairs.csv (preliminary "
                   f"{rec.get('preliminary')!r}): rebuild after the final d0_pairs.py build (RUNBOOK_lad S-1)")
    outs = rec.get("outputs") or {}
    lab = Path(LABELS)
    if not lab.is_file():
        raise NotReady(f"{lab} missing (d0d1_splits.py build)")
    if outs.get(ab(lab)) != sha256(lab):
        raise Stop(f"{lab} sha256 != {S_SPLITS.name} outputs")
    t1, t1sha = fu2_t1()
    ins = rec.get("inputs_sha256") or {}
    gated = [CR2, ORIG, FU2_S_SPLITS, D0_PAIRS, D0_FOLDS, D0_TWINS] + [Path(p) for p in t1sha]
    changed = [ab(p) for p in gated if not Path(p).is_file() or ins.get(ab(p)) != sha256(p)]
    if changed:
        raise Stop(f"inputs of {S_SPLITS.name} changed since its build (rerun d0d1_splits.py build): {changed[:6]}")
    prov = d0_provenance(allow)
    if prov["preliminary"] != bool(rec.get("preliminary")):
        raise Stop(f"{S_SPLITS.name} preliminary {rec.get('preliminary')!r} != the GEO1b report now ({prov['preliminary']})")
    cols = cr2_columns()
    T = read_d0_pairs(cols)
    pfold = pair_folds(T, read_d0_folds())
    okd = sorted(str(q) for q, o in zip(T.pair_id, T.ok) if o == "True")
    lists, files, seen_te = {}, {}, []
    for k in range(NF):
        p = Path(SPLITS) / split_name(k)
        u = (rec.get("units") or {}).get(split_name(k)) or {}
        if not p.is_file():
            raise NotReady(f"{p} missing")
        if u.get("sha256") != sha256(p) or outs.get(ab(p)) != u.get("sha256"):
            raise Stop(f"{p} sha256 != {S_SPLITS.name} units / outputs")
        f = json.loads(p.read_text())
        check_lists(k, f, t1[k], pfold, okd)
        lists[k] = f
        seen_te += [q for q in f["test"] if is_d0_id(q)]
        files[f"split_T1D0_f{k}"] = ab(p)
    if sorted(seen_te) != okd:
        raise Stop("the D0 test lists of the 5 folds are not each ok D0 pair exactly once")
    files.update({f"split_T1_fu2_f{k}": p for k, p in enumerate(sorted(t1sha))},
                 split_s_splits_d0d1=ab(S_SPLITS), split_s_splits_fu2=ab(FU2_S_SPLITS), d0_pairs=ab(D0_PAIRS),
                 d0_folds=ab(D0_FOLDS), d0_twins=ab(D0_TWINS), d0_report=ab(D0_REPORT), q_d0_report=ab(Q_D0_REPORT))
    out = (rec, lists, files)
    _CACHE[key] = (rsha, out)
    return out


# ================================================================ build
def build(test_dir=None, allow_preliminary=False):
    t0 = time.time()
    labels, splits, s_json = test_paths(test_dir) if test_dir else (LABELS, SPLITS, S_SPLITS)
    if allow_preliminary and not test_dir:
        raise Stop("--allow-preliminary is for --test-dir builds only")
    v = read_json(V_REPORT, "LADDER Phase V")
    if not CR2.is_file():
        raise NotReady(f"{CR2} missing")
    c = CR2.read_bytes()
    if v.get("all_pass") is not True or (v.get("outputs") or {}).get(str(CR2)) != sha256_bytes(c):
        raise Stop(f"{CR2}: results/V_report.json not all_pass or its outputs sha256 differs (LADDER label provenance)")
    cols = cr2_columns()
    O = pd.read_csv(ORIG, usecols=["pair_id"], dtype=str)
    with open(ORIG, newline="") as fh:
        ocols = next(csv.reader(fh))
    cr2_ids = pd.read_csv(CR2, usecols=["pair_id"], dtype=str).pair_id.tolist()
    if cols != ocols or cr2_ids != O.pair_id.tolist() or not c.endswith(b"\n"):
        raise Stop(f"{CR2}: header / pair order differ from {ORIG} or no final newline")
    prov = d0_provenance(allow_preliminary)
    T = read_d0_pairs(cols)
    ids = T.pair_id.tolist()
    if len(set(ids)) != len(ids) or not set(T.ok) <= {"True", "False"} or (T.ref_kind != "d0").any():
        raise Stop(f"{D0_PAIRS}: duplicated ids, ok values {sorted(set(T.ok))}, or ref_kind other than d0")
    R1 = pd.read_csv(CR2, usecols=["a_rxn", "b_rxn"])
    r1 = set(R1.a_rxn.astype(int)) | set(R1.b_rxn.astype(int))
    r0 = set(T.a_rxn.astype(int)) | set(T.b_rxn.astype(int))
    if r0 & r1:
        raise Stop(f"D0 and D1 reaction ids overlap (one token / table row per rxn_id): {sorted(r0 & r1)[:5]}")
    pfold = pair_folds(T, read_d0_folds())
    okd = sorted(str(q) for q, o in zip(T.pair_id, T.ok) if o == "True")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for row in T[cols].itertuples(index=False, name=None):
        w.writerow(list(row))
    lab_bytes = c + buf.getvalue().encode()
    l11 = l11_d0d1(label_bytes=lab_bytes)
    t1, t1sha = fu2_t1()
    lists = make_lists(t1, okd, pfold)
    counts = {k: check_lists(k, lists[k], t1[k], pfold, okd) for k in range(NF)}
    tw = twins_check(lists, T, read_twins())
    # ---- write (outputs first, the json last = the readers' anchor)
    write_bytes_atomic(labels, lab_bytes)
    units, outs = {}, {ab(labels): sha256_bytes(lab_bytes)}
    for k in range(NF):
        b = split_bytes(lists[k])
        p = write_bytes_atomic(Path(splits) / split_name(k), b)
        units[split_name(k)] = dict(path=ab(p), sha256=sha256_bytes(b), t1_sha256=t1sha[ab(FU2_SPLITS / f"T1__f{k}.json")],
                                    **counts[k])
        outs[ab(p)] = sha256_bytes(b)
    edge = dict(zip(T.pair_id, T.edge_kind))
    per_fold = {str(k): dict(n_d0_ok_train=counts[k]["n_d0"]["train"], n_d0_ok_test=counts[k]["n_d0"]["test"],
                             n_d0_all=sum(1 for q in ids if pfold[q] == k),
                             test_by_edge_kind={e: sum(1 for q in lists[k]["test"] if is_d0_id(q) and edge[q] == e)
                                                for e in sorted(set(T.edge_kind))}) for k in range(NF)}
    ins = {ab(p): sha256(p) for p in (CR2, ORIG, V_REPORT, FU2_S_SPLITS, D0_PAIRS, D0_FOLDS, D0_TWINS, D0_REPORT,
                                      Q_D0_REPORT)}
    ins.update(t1sha)
    rec = dict(code_version=CODE_VERSION, split=SPLIT, spec="docs/specs/GEO2.md 8-1; analysis/geo1b/results/"
               "D0D1_design_investigation.json design step 9; DECISIONS.md D0D1 (user 2026-10-31)",
               decisions="analysis/geo2/results/DECISIONS_lad.md LD-*", all_pass=True, stop=False,
               preliminary=bool(prov["preliminary"]), test_dir=ab(test_dir) if test_dir else None, rules=RULES,
               inputs_sha256=ins, d0=dict(prov, n_pairs=len(ids), n_ok=len(okd), n_not_ok=len(ids) - len(okd),
                                          per_fold=per_fold),
               labels=dict(path=ab(labels), sha256=outs[ab(labels)], bytes=len(lab_bytes), l11=l11),
               units=units, outputs=outs, twins=tw, started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
               finished=now(), wall_s=round(time.time() - t0, 2))
    stamp = ("started", "finished", "wall_s")
    old = {}
    if Path(s_json).is_file():
        try:
            old = json.loads(Path(s_json).read_text())
        except ValueError:
            old = {}
    if old and {k: v for k, v in old.items() if k not in stamp} == json.loads(json.dumps(
            {k: v for k, v in rec.items() if k not in stamp})):
        return dict(old, up_to_date=True)                   # LD-8: same content -> the index keeps its sha256
    write_bytes_atomic(s_json, (json.dumps(rec, indent=1) + "\n").encode())
    return rec


def _summary(rec):
    return dict(split=rec.get("split"), preliminary=rec.get("preliminary"), n_d0_pairs=(rec.get("d0") or {}).get("n_pairs"),
                n_d0_ok=(rec.get("d0") or {}).get("n_ok"), labels=(rec.get("labels") or {}).get("path"),
                labels_sha256=(rec.get("labels") or {}).get("sha256"),
                units={k: v.get("n") for k, v in (rec.get("units") or {}).items()})


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] not in ("build", "check"):
        print(__doc__)
        return 2
    td = None
    if "--test-dir" in a:
        i = a.index("--test-dir")
        if i + 1 >= len(a):
            print("--test-dir needs a folder", file=sys.stderr)
            return 2
        td = a[i + 1]
    allow = "--allow-preliminary" in a
    try:
        if a[0] == "build":
            rec = build(td, allow)
        else:
            global ALLOW_PRELIMINARY
            if td:
                set_paths(*test_paths(td))
            ALLOW_PRELIMINARY = allow and bool(td)
            rec, lists, files = load_s_splits()
            rec = dict(rec, reader_check="pass", files=files)
        print(json.dumps(_summary(rec), indent=1), flush=True)
        return 0
    except NotReady as e:
        print(f"NOT READY: {e}", file=sys.stderr, flush=True)
        return 2
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
