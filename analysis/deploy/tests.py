#!/usr/bin/env python3
r"""tests.py — DEPLOY-MODEL Phase C code checks (docs/specs/DEPLOY_MODEL.md 9). Any failure = STOP (exit 3).

The checks need four environments, so they run as parts; `merge` writes results/C_tests.json (all_pass):
  python tests.py cpu [--source lf|g2]   C-2 (DeployData on the LADDER g2 inputs with the LADDER usable rule vs
                                         train_atoms.LadData('g2', 'S')), C-5 (standardisation leak), C-6 (forbidden
                                         inputs); env_d1fu2 on CPU (jobs/cpu_py.slurm DEPENV=d1fu2). C-5 / C-6 run on
                                         the production lf data (--source g2: a dry run on the g2 data before the lf
                                         inputs exist, written to $DEP/dev_tf/, never merged)  -> results/C_part_cpu.json
  python tests.py gpu                    C-3 (train_tf.py c3) + C-8 (train_tf.py smoke) on GPU (RTX 3090 = the LADDER
                                         g2 TF unit type, gpu1), C-4 = gate M-1' of every fit of both
                                         -> results/C_part_gpu.json
  python tests.py c1                     C-1: c1_dft_free.run_c1() (F / T agent; feature env)  -> results/C_part_c1.json
  python tests.py c7                     C-7: finalize_lf.selftest_c7() (L / M agent; D1 run env) -> C_part_c7.json
  python tests.py merge                  every part present and ok, run on the current code -> results/C_tests.json
  python tests.py toy [--source g2|lf]   developer round trip (not a spec check): a tiny CPU fit through train_tf's
                                         code path (fold 0, cell 0, 2 epochs, 1 seed, 300 pairs per list) saved under
                                         $DEP/dev_tf/toy, reloaded, and predict_tf.py run --model cv on its test pairs
                                         (subprocess, DEP_UNITS = the toy folder) compared with the unit parquet.
Top level = stdlib + dep_common only (the c7 part runs in the D1 run env, python 3.9); numpy / torch are imported in
the parts that need them.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402

DEV = C.DEP / "dev_tf"
PART_FILES = {"cpu": "C_part_cpu.json", "gpu": "C_part_gpu.json", "c1": "C_part_c1.json", "c7": "C_part_c7.json"}
PART_CHECKS = {"cpu": ("C-2", "C-5", "C-6"), "gpu": ("C-3", "C-4", "C-8"), "c1": ("C-1",), "c7": ("C-7",)}
PART_CODE = {"cpu": ("tests.py", "data_lf.py", "train_tf.py", "dep_common.py", "predict_tf.py"),
             "gpu": ("tests.py", "data_lf.py", "train_tf.py", "dep_common.py"),
             "c1": ("tests.py", "c1_dft_free.py", "feat_lf.py", "tokens_lf.py", "dep_common.py"),
             "c7": ("tests.py", "finalize_lf.py", "dep_common.py")}
TRAINER_FILES = ("train_tf.py", "data_lf.py", "dep_common.py")
FORBIDDEN_NAMES = ("sigma", "fam", "family", "core", "Xb", "series", "skeleton", "dipole", "substituent",
                   "fold_T1", "fold_LODO", "fold_LOSO", "fold_T2")          # spec C-6: sigma, family / core, substituent
META_PERMUTE = FORBIDDEN_NAMES + ("cfg", "orient", "face", "c3", "dipolarophile")
FORBIDDEN_RE = re.compile(r"(^|_)(sigma|sig|fam|family|core|xb|subst|substituent|series|skeleton|fold|pair|pairid|"
                          r"rxn|rxnid|ref|kind|hammett|job)(_|$)", re.I)
# code places that may name pair metadata (never a tensor): C-3 para MAE set, the MODEL_CARD application domain
ALLOWED_META_FUNCS = {"train_tf.py": {"para_mae", "model_card"}}


def code_sha(names):
    return {n: (C.sha256(HERE / n) if (HERE / n).is_file() else None) for n in names}


def _ok(rec):
    return bool(isinstance(rec, dict) and (rec.get("ok") is True or rec.get("pass") is True or rec.get("all_pass") is True))


def _write_part(part, checks, out=None, extra=None):
    rec = dict(part=part, ok=bool(all(_ok(v) for v in checks.values())), checks=checks,
               code_sha256=code_sha(PART_CODE[part]), python=sys.version.split()[0], host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"), time=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    path = out or (C.RES / PART_FILES[part])
    C.write_json(path, rec)
    print(json.dumps(dict(part=part, ok=rec["ok"], checks={k: _ok(v) for k, v in checks.items()}, file=str(path)),
                     indent=1))
    return 0 if rec["ok"] else 3


def _guard(fn, *a, **k):
    """run one check; an exception / STOP becomes a failed check with the reason (the part still writes its file)."""
    try:
        return fn(*a, **k)
    except SystemExit as e:
        return dict(ok=False, error=f"SystemExit {e.code}", traceback=traceback.format_exc()[-3000:])
    except Exception as e:                                         # noqa: BLE001
        return dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])


# ================================================================ C-2
def c2():
    """DeployData('g2') (LADDER g2 tokens + X182_d1_g2 + LADDER usable rule incl. H) vs train_atoms.LadData('g2', 'S'):
    every array np.array_equal (and dtype), pair table equal, pair list and every split list (and their sha256) equal."""
    import numpy as np
    import data_lf as DL
    sys.path.insert(0, str(C.REPO / "analysis" / "d1_ladder"))
    import train_atoms as TA
    Dd = DL.DeployData("g2")
    Dl = TA.LadData("g2", "S")
    arrays = ("onehot", "bin", "num", "pos", "mask", "etype", "frag", "n", "glob", "y", "ra", "rb", "rids",
              "has_tok", "usable_pair", "usable_rxn")
    res = {}
    for a in arrays:
        x, y = getattr(Dd, a), getattr(Dl, a)
        res[a] = dict(equal=bool(np.array_equal(x, y)), dtype=[str(x.dtype), str(y.dtype)],
                      shape=list(np.shape(x)), same_dtype=x.dtype == y.dtype)
    scal = {k: getattr(Dd, k) == getattr(Dl, k) for k in ("d_in", "d_glob", "nmax", "elements", "num_cols", "bin_cols",
                                                          "row", "pair_list", "pair_list_sha")}
    P_eq = bool(Dd.P.equals(Dl.P) and list(Dd.P.columns) == list(Dl.P.columns))
    splits = {}
    for k in range(DL.NF):
        a, b = Dd.split(k), Dl.split(k)
        splits[k] = {key: bool(np.array_equal(a[key], b[key])) for key in DL.FOLD_KEYS}
        splits[k].update({s: a[s] == b[s] for s in ("pairs_sha", "val_sha", "test_sha")}, dropped=a["dropped"] == b["dropped"])
    import torch
    sp = Dd.split(0)
    stat = {}
    for dt in (torch.float64, torch.float32):
        Td, Tl = FT_std_tensors(Dd, sp["train"], dt), FT_std_tensors(Dl, sp["train"], dt)
        stat[str(dt)] = {k: bool(torch.equal(Td[k], Tl[k])) for k in Td}
    ok = all(v["equal"] and v["same_dtype"] for v in res.values()) and all(scal.values()) and P_eq and \
        all(all(v.values()) for v in splits.values()) and all(all(v.values()) for v in stat.values())
    return dict(ok=bool(ok), arrays=res, scalars={k: bool(v) for k, v in scal.items()}, pair_table_equal=P_eq,
                splits={str(k): v for k, v in splits.items()}, std_tensors_fold0_train=stat,
                n_tokens=int(len(Dd.rids)), n_usable=int(Dd.usable_rxn.sum()), n_pairs_P=len(Dd.pair_list),
                pair_list_sha=Dd.pair_list_sha, deploy_sha=Dd.sha, ladder_sha=Dl.sha,
                ladder_unit_pair_list=(C.read_json(C.LAD / "units" / "L__g2__S__TF__f0.json").get("fingerprint") or {})
                .get("pair_list"))


def FT_std_tensors(D, rows, dtype):
    import data_lf as DL
    import torch
    return DL.FT.Std(D, rows).tensors(D, dtype=dtype, device=torch.device("cpu"))


# ================================================================ C-5
def c5(source):
    """standardisation leak: the reactions behind the token / global statistics and the y scale of every fit are the
    fit's training reactions only (0 test-side reactions); recomputed independently; test-side values perturbed ->
    statistics bitwise unchanged; train_tf.leak_check passes on real fits and STOPs on a planted leak."""
    import numpy as np
    import data_lf as DL
    import train_tf as TT
    D = DL.get_data(source)
    FT = DL.FT
    eps = DL.DP.SCALE_EPS
    out, ok = {}, True
    for k in list(range(DL.NF)) + ["all"]:
        sp = D.split(k)
        fits = [("outer", sp["train"], (sp["test"],), sp["test"])]
        if k != "all":
            fits.append(("tuning", sp["inner_train"], (sp["inner_val"], sp["test"]), sp["test"]))
        for name, fr, others, te in fits:
            std = FT.Std(D, fr)
            rx = np.unique(np.r_[D.ra[fr], D.rb[fr]])
            te_rx = set(D.ra[te].tolist()) | set(D.rb[te].tolist())
            side = set(te_rx)
            for o in others:
                side |= set(D.ra[o].tolist()) | set(D.rb[o].tolist())
            n_test = len(te_rx & set(rx.tolist()))
            n_side = len(side & set(rx.tolist()))
            m = D.mask[rx]
            vals = D.num[rx][m]
            mu, sd = vals.mean(0), vals.std(0)
            sd[sd < eps] = 1.0
            g = D.glob[rx]
            gmu, gsd = g.mean(0), g.std(0)
            gsd[gsd < eps] = 1.0
            sy = D.y[fr].std(0)
            recomputed = dict(rxn_rows=bool(np.array_equal(std.rxn_rows, rx)), mu=bool(np.array_equal(std.mu, mu)),
                              sd=bool(np.array_equal(std.sd, sd)), gmu=bool(np.array_equal(std.gmu, gmu)),
                              gsd=bool(np.array_equal(std.gsd, gsd)), sy=bool(np.array_equal(std.sy, sy)))
            pert = None
            only = sorted(side - set(rx.tolist()))
            if only:                                     # perturb every test-side-only reaction / test-side pair
                pr = [int(x) for x in only]
                side_pairs = np.unique(np.concatenate([np.asarray(o) for o in others])) if others else np.zeros(0, int)
                keep = (D.num[pr].copy(), D.glob[pr].copy(), D.y[side_pairs].copy())
                try:
                    D.num[pr] += 1e3
                    D.glob[pr] += 1e3
                    D.y[side_pairs] += 1e3
                    s2 = FT.Std(D, fr)
                    pert = all(np.array_equal(getattr(std, a), getattr(s2, a)) for a in ("mu", "sd", "gmu", "gsd", "sy"))
                finally:
                    D.num[pr], D.glob[pr], D.y[side_pairs] = keep
            lk = _guard(TT.leak_check, D, std, fr, others, te)
            rec = dict(n_fit_pairs=int(len(fr)), n_scale_rxns=int(len(rx)), n_test_rxn_in_scaling=n_test,
                       n_test_side_rxn_in_scaling=n_side, recomputed=recomputed, perturbation_invariant=pert,
                       leak_check=lk)
            good = n_test == 0 and all(recomputed.values()) and pert in (True, None) and lk.get("ok") is True
            if name == "tuning":
                rec["note"] = "test side = inner_val + test pairs (inner GroupKFold by core)"
                good = good and n_side == 0
            rec["ok"] = bool(good)
            ok &= good
            out[f"{k}/{name}"] = rec
    # negative controls: a planted leak must STOP
    sp = D.split(0)
    std = FT.Std(D, np.r_[sp["train"], sp["test"][:1]])
    neg1 = _guard(TT.leak_check, D, std, sp["train"], (sp["test"],), sp["test"])
    std2 = FT.Std(D, sp["train"])
    neg2 = _guard(TT.leak_check, D, std2, sp["train"], (sp["train"][:5],), sp["test"])
    neg = dict(test_pair_in_scaling_stops=neg1.get("ok") is False, val_pair_in_scaling_stops=neg2.get("ok") is False)
    ok &= all(neg.values())
    return dict(ok=bool(ok), source=source, fits=out, negative_controls=neg)


# ================================================================ C-6
def _ast_names(path):
    """(function name or '<module>', name) for every string constant / attribute / subscript key in a file."""
    tree = ast.parse(Path(path).read_text())
    out = []

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(ch, ast.Constant) and isinstance(ch.value, str):
                out.append((f, ch.value))
            elif isinstance(ch, ast.Attribute):
                out.append((f, ch.attr))
            visit(ch, f)
    visit(tree, "<module>")
    return out


def c6(source):
    """forbidden inputs: sigma, family / core ids, substituent names never enter the model input tensors: (a) the tensor
    column lists, (b) the batch keys / widths, (c) code inspection of the tensor-building code (AST: no pair-metadata
    name outside the allowed non-tensor functions; FU2 Std.tensors / batch / fit / predict read no pair table), (d) the
    pair metadata columns permuted -> standardised tensors and a forward pass bitwise unchanged."""
    import inspect
    import numpy as np
    import torch
    import data_lf as DL
    FT, Mo = DL.FT, DL.Mo
    D = DL.get_data(source)
    cols = D.input_columns()
    bad_cols = {k: [c for c in v if FORBIDDEN_RE.search(c)] for k, v in cols.items()}
    sp = D.split(0)
    std = FT.Std(D, sp["train"])
    cpu = torch.device("cpu")
    T = std.tensors(D, dtype=torch.float64, device=cpu)
    rows = sp["test"][:64]
    R, ia, ib = FT.batch(T, D.ra[rows], D.rb[rows])
    keys_ok = set(R) == {"x", "pos", "mask", "etype", "frag", "glob"}
    width_ok = R["x"].shape[-1] == len(cols["x"]) == D.d_in and R["glob"].shape[-1] == len(cols["glob"]) == D.d_glob
    names = set(FORBIDDEN_NAMES)
    hits = {}
    for f in ("data_lf.py", "train_tf.py", "predict_tf.py"):
        allowed = ALLOWED_META_FUNCS.get(f, set())
        h = sorted({f"{fn}:{nm}" for fn, nm in _ast_names(HERE / f) if nm in names and fn not in allowed})
        hits[f] = h
    fu2_hits = {}
    for fn in ("Std.tensors", "Std.__init__", "batch", "fit", "predict", "epoch_batches", "orient"):
        obj = FT
        for part in fn.split("."):
            obj = getattr(obj, part)
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        h = set()
        for nd in ast.walk(tree):
            if isinstance(nd, ast.Constant) and isinstance(nd.value, str) and nd.value in names:
                h.add(nd.value)
            if isinstance(nd, ast.Attribute) and (nd.attr in names or nd.attr == "P"):
                h.add(nd.attr)                                      # D.P = the pair table
        fu2_hits[fn] = sorted(h)
    # (d) permutation of the metadata columns of the pair table
    torch.manual_seed(0)
    model = Mo.build("TF", Mo.GRID["TF"][0], D.d_in, len(DL.TARGETS), D.d_glob, "A+G").double().eval()
    with torch.no_grad():
        y0 = model(R, ia, ib)
    P0 = D.P
    meta = [c for c in P0.columns if c in META_PERMUTE or c.startswith("fold_")]
    rng = np.random.default_rng(C.SEED)
    P1 = P0.copy()
    for c in meta:
        P1[c] = P1[c].to_numpy()[rng.permutation(len(P1))]
    try:
        D.P = P1
        std1 = FT.Std(D, sp["train"])
        T1 = std1.tensors(D, dtype=torch.float64, device=cpu)
        R1, ia1, ib1 = FT.batch(T1, D.ra[rows], D.rb[rows])
        with torch.no_grad():
            y1 = model(R1, ia1, ib1)
        perm = bool(all(torch.equal(T[k], T1[k]) for k in T) and torch.equal(y0, y1))
    finally:
        D.P = P0
    ok = not any(bad_cols.values()) and keys_ok and width_ok and not any(hits.values()) and \
        not any(fu2_hits.values()) and perm
    return dict(ok=bool(ok), source=source, columns=cols, forbidden_columns=bad_cols, batch_keys=sorted(R),
                batch_keys_ok=keys_ok, widths_ok=bool(width_ok), code_hits=hits, fu2_code_hits=fu2_hits,
                allowed_meta_functions={k: sorted(v) for k, v in ALLOWED_META_FUNCS.items()},
                metadata_permuted=meta, permutation_invariant=perm, forbidden_names=list(FORBIDDEN_NAMES),
                forbidden_regex=FORBIDDEN_RE.pattern)


# ================================================================ parts
def part_cpu(source):
    checks = dict()
    checks["C-2"] = _guard(c2)
    checks["C-5"] = _guard(c5, source)
    checks["C-6"] = _guard(c6, source)
    out = None if source == "lf" else DEV / f"C_part_cpu_{source}.json"
    return _write_part("cpu", checks, out, extra=dict(source=source))


def part_gpu():
    import train_tf as TT
    rc3 = _guard(TT.cmd_c3)
    c3 = C.read_json(C.RES / "C3_regress.json") if (C.RES / "C3_regress.json").is_file() else {}
    rc8 = _guard(TT.cmd_smoke)
    c8 = C.read_json(C.RES / "C8_smoke.json") if (C.RES / "C8_smoke.json").is_file() else {}
    fresh3 = rc3 in (0, 3) and (c3.get("code_sha256") or {}).get("train_tf.py") == TT.CODE_SHA["train_tf.py"]
    fresh8 = rc8 in (0, 3) and (c8.get("code_sha256") or {}).get("train_tf.py") == TT.CODE_SHA["train_tf.py"]
    c3rec = dict(c3, ok=bool(rc3 == 0 and c3.get("ok") and fresh3), exit=rc3 if isinstance(rc3, int) else rc3)
    c8rec = dict(c8, ok=bool(rc8 == 0 and c8.get("ok") and fresh8), exit=rc8 if isinstance(rc8, int) else rc8)
    g4 = dict(ok=bool(c3.get("checks", {}).get("gates_C4") and c8.get("ok")),
              rule="|g(a,b)+g(b,a)| <= 1e-4 and |g(a,a)| <= 1e-4 kcal/mol (float64) on 200 random test pairs, every "
                   "tuning and final fit of C-3 and C-8 (train_tf.run_unit stops otherwise)",
              c8_tune=c8.get("gate_C4_tune"), c8_final=c8.get("gate_C4_final"))
    return _write_part("gpu", {"C-3": c3rec, "C-4": g4, "C-8": c8rec},
                       extra=dict(trainer_code_sha256=code_sha(TRAINER_FILES)))


def part_c1():
    try:
        import c1_dft_free as M
    except ImportError as e:
        C.not_ready(f"c1_dft_free.py not importable: {e}")
    return _write_part("c1", {"C-1": _guard(M.run_c1)})


def part_c7():
    try:
        import finalize_lf as M
    except ImportError as e:
        C.not_ready(f"finalize_lf.py not importable: {e}")
    return _write_part("c7", {"C-7": _guard(M.selftest_c7)})


def merge():
    parts, bad = {}, []
    for part, fn in PART_FILES.items():
        p = C.RES / fn
        if not p.is_file():
            bad.append(f"{fn} missing")
            continue
        r = C.read_json(p)
        now = code_sha(PART_CODE[part])
        stale = [f for f, s in (r.get("code_sha256") or {}).items() if s != now.get(f)]
        parts[part] = dict(file=fn, sha256=C.sha256(p), ok=bool(r.get("ok")), stale_code=stale,
                           checks={k: _ok(v) for k, v in (r.get("checks") or {}).items()})
        if part == "cpu" and r.get("source") != "lf":
            bad.append("cpu part did not run on the lf data")
        if stale:
            bad.append(f"{fn} ran on other code: {stale}")
        if not r.get("ok"):
            bad.append(f"{fn} not ok")
        miss = [c for c in PART_CHECKS[part] if c not in (r.get("checks") or {})]
        if miss:
            bad.append(f"{fn} lacks {miss}")
    checks = {c: bool(parts.get(p, {}).get("checks", {}).get(c)) for p, cs in PART_CHECKS.items() for c in cs}
    trainer = code_sha(TRAINER_FILES)
    gpu_tr = (C.read_json(C.RES / PART_FILES["gpu"]).get("trainer_code_sha256") if (C.RES / PART_FILES["gpu"]).is_file()
              else None)
    if gpu_tr != trainer:
        bad.append("C-3 / C-8 ran on other trainer code than the current files")
    rec = dict(all_pass=bool(not bad and all(checks.values())), checks=dict(sorted(checks.items())), parts=parts,
               problems=bad, trainer_code_sha256=trainer, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    C.write_json(C.RES / "C_tests.json", rec)
    print(json.dumps({k: rec[k] for k in ("all_pass", "checks", "problems")}, indent=1))
    return 0 if rec["all_pass"] else 3


# ================================================================ developer round trip (not a spec check)
def toy(source):
    import numpy as np
    import pandas as pd
    import torch
    import data_lf as DL
    import train_tf as TT
    FT, F = DL.FT, DL.F
    root = DEV / "toy"
    for p in root.glob("D__*"):
        C.assert_write_ok(p)
        p.unlink()
    D = DL.get_data(source)
    u = dict(rung=D.rung, arm="S", model="TF", fold=0)
    t0 = time.time()
    rec = TT.run_unit(D, u, root=root, n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=[0], log=1, smoke=True, limit=300)
    name = TT.uname(u)
    obj = torch.load(root / f"{name}__s0.pt", map_location="cpu")
    sp = TT._limit(D.split(0), 300)
    std = FT.Std(D, sp["train"])
    std_eq = {k: bool(np.array_equal(np.asarray(obj["std"][k]), np.asarray(getattr(std, k)))) for k in
              ("mu", "sd", "gmu", "gsd", "sy")}
    keys = sorted(obj)
    pq = pd.read_parquet(root / f"{name}.parquet")
    pairs = D.P.loc[pq.pair_id.unique(), ["a_rxn", "b_rxn"]].reset_index()
    pc = root / "toy_pairs.csv"
    C.write_csv(pc, pairs)
    cfg = DL.SOURCES[source]
    out = root / "toy_pred.parquet"
    env = dict(os.environ, DEP_UNITS=str(root))
    cmd = [sys.executable, str(HERE / "predict_tf.py"), "run", str(cfg["tokens"]), str(cfg["x182"]), str(pc), str(out),
           "--model", "cv", "--rung", D.rung, "--folds", "0", "--seeds", "0",
           "--t-report", str(Path(cfg["res"]) / "T_report.json")]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    res = dict(source=source, unit=name, unit_seconds=time.time() - t0, pt_keys=keys, std_equal=std_eq,
               predict_rc=r.returncode, predict_stdout=r.stdout[-2000:], predict_stderr=r.stderr[-3000:],
               gates=[t["gate_M1"] for t in rec["tune"]] + [f["gate_M1"] for f in rec["final"]])
    if r.returncode == 0:
        p = pd.read_parquet(out)
        m = pq.merge(p, on=["pair_id", "target"], suffixes=("_unit", "_pred"))
        res.update(n_rows_unit=int(len(pq)), n_rows_pred=int(len(p)), n_merged=int(len(m)),
                   max_abs_dyhat=float(np.abs(m.yhat_unit - m.yhat_pred).max()),
                   max_abs_dsd=float(np.abs(m.yhat_seed_sd - m.yhat_sd).max()),
                   out_of_range_null=bool(p.out_of_range.isna().all()), columns=list(p.columns))
    res["ok"] = bool(r.returncode == 0 and all(std_eq.values()) and res.get("n_merged") == res.get("n_rows_unit")
                     and res.get("max_abs_dyhat", 1.0) <= 1e-6 and all(g["ok"] for g in res["gates"]))
    C.write_json(DEV / f"toy_{source}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("predict_stdout",)}, indent=1, default=str))
    return 0 if res["ok"] else 3


def main(argv):
    a = argv[1:]
    src = "lf"
    if "--source" in a:
        i = a.index("--source")
        src = a[i + 1]
        a = a[:i] + a[i + 2:]
    if a == ["cpu"]:
        return part_cpu(src)
    if a == ["gpu"]:
        return part_gpu()
    if a == ["c1"]:
        return part_c1()
    if a == ["c7"]:
        return part_c7()
    if a == ["merge"]:
        return merge()
    if a == ["toy"]:
        return toy(src if "--source" in argv else "g2")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
