#!/usr/bin/env python3
"""c1_dft_free.py — DEPLOY-MODEL check C-1 (docs/specs/DEPLOY_MODEL.md 9): DFT values do not flow into the features or
the tokens. n = 20 D1 reactions of the lf root (default $DEP/g2lf_final, seeded choice), computed twice from scratch
through the Phase F / T code paths (feat_lf.run_slice + feat_lf.merge_slices = d1_features.run_ids / merge_rows;
tokens_lf children atoms / gate / tokens = fu2/atoms.py root run / gate / tokens with the lf root check) with
  (a) 'true': the labels_lf record + the 14 DFT keys filled with the true D1 values ($G2/G/labels_d1_g2.json)
  (b) 'nan' : the labels_lf record (DFT keys NaN, spec 4-1)
and everything else identical (structures, input_meta_lf.csv, prof_empty, code, engines; each variant has its own
slices, ext cache, atoms cache and tokens, so nothing is shared between the two). Pass = for every reaction the same
xtb_status / ext_status, the 182 columns bit-identical (float64 bytes) and every token array bit-identical
(np.array_equal per npz key, glob included). Diagnostics: the g1 / ext / atoms columns that differ (expected: only the
non-feature dft_* columns, dft_c_ghost_kcal and flag_negative_strain of the g1 parquet).

  python c1_dft_free.py run [n]       # -> $DEP/C1/<key>/{true,nan}/, results/C1_dft_free.json (exit 0 pass, 3 fail)
  from c1_dft_free import run_c1; rep = run_c1()      # for analysis/deploy/tests.py (returns the report dict)

Job environment = Phase F / T (analysis/d1_pair jobs/g.slurm, PYENV=venv5, D1PAIR_PHASE=F). <key> = sha256 of the
selection + inputs, so a rerun with the same inputs reuses its finished parts (idempotent) and another selection never
mixes with it.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import feat_lf as FL  # noqa: E402
import tokens_lf as TL  # noqa: E402

D = FL.D
N_C1 = 20
TRUTH = D.G2 / "G" / "labels_d1_g2.json"              # D1 labels with the DFT values (read only)
DFT_KEYS = ("d1_kcal", "d2_kcal", "e_bond_kcal", "e_ab_eh", "e_frag1_dist_eh", "e_frag2_dist_eh", "e_frag1_rel_eh",
            "e_frag2_rel_eh", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft")   # spec 4-1
OUT_JSON = D.RES / "C1_dft_free.json"
VARIANTS = ("true", "nan")


def _bits_equal(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _npz_diff(pa, pb):
    """keys whose arrays differ (np.array_equal; NaN-aware for floats)."""
    a, b = _npz(pa), _npz(pb)
    out = sorted(set(a) ^ set(b))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        same = x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" and y.dtype.kind == "f"
                                       else np.array_equal(x, y))
        if not same:
            out.append(k)
    return out


def _frame_diff(a, b, skip=()):
    """columns of two row-aligned frames (same rxn order) whose values differ (NaN == NaN)."""
    out = []
    for c in sorted(set(a.columns) | set(b.columns)):
        if c in skip:
            continue
        if c not in a or c not in b:
            out.append(c)
            continue
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if not ((x == y) | (x.isna() & y.isna())).all():
            out.append(c)
    return out


def pick(root, labels, n, seed):
    lab = {int(r["rxn_id"]): r for r in json.loads(Path(labels).read_text())}
    done, _ = FL.DF.root_ids(root)
    pool = sorted(set(done) & set(lab))
    if not pool:
        D.not_ready(f"no .done reaction with a labels_lf record in {root}")
    rng = np.random.default_rng(seed)
    ids = sorted(int(x) for x in rng.choice(pool, size=min(int(n), len(pool)), replace=False))
    return ids, lab, len(pool)


def run_c1(n=N_C1, root=None, labels=None, meta=None, prof=None, out=None, truth=TRUTH, seed=D.SEED, write=True):
    """C-1 -> report dict (key 'pass'); also results/C1_dft_free.json when write. Needs the Phase F job environment."""
    root, labels, meta, prof = (Path(p or q) for p, q in ((root, D.LF_FINAL), (labels, D.LABELS_LF),
                                                          (meta, D.META_LF), (prof, D.PROF_EMPTY)))
    rep = dict(check="C-1", spec="docs/specs/DEPLOY_MODEL.md 9 C-1", written=FL.now(), root=str(root), labels=str(labels),
               meta=str(meta), prof=str(prof), truth=str(truth), seed=int(seed), n_requested=int(n), dft_keys=list(DFT_KEYS))
    try:
        FL.parent_env()
        ids, lab, n_pool = pick(root, labels, n, seed)
        tru = {int(r["rxn_id"]): r for r in json.loads(Path(truth).read_text())}
        miss = [r for r in ids if r not in tru or not all(np.isfinite(float(tru[r][k])) for k in DFT_KEYS)]
        if miss:
            D.stop(f"no true DFT values in {truth} for {miss}")
        nan_ok = [r for r in ids if all(isinstance(lab[r].get(k), float) and math.isnan(lab[r][k]) for k in DFT_KEYS)]
        recs = {"nan": [lab[r] for r in ids], "true": [dict(lab[r], **{k: float(tru[r][k]) for k in DFT_KEYS}) for r in ids]}
        key = D.sha256_text(json.dumps(dict(ids=ids, root=str(root), labels=D.sha256(labels), meta=D.sha256(meta),
                                            truth=D.sha256(truth)), sort_keys=True))[:12]
        base = Path(out or D.DEP / "C1") / key
        rep.update(rxn_ids=ids, n=len(ids), n_pool=n_pool, out=str(base), labels_nan_all_nan=len(nan_ok) == len(ids),
                   labels_nan_not_nan=[r for r in ids if r not in nan_ok])
        res = {}
        for v in VARIANTS:
            vd = base / v
            lp = vd / f"labels_{v}.json"
            txt = json.dumps(recs[v])                          # NaN as the JSON NaN literal (= labels_lf.json)
            if not lp.is_file() or lp.read_text() != txt:
                D.write_text(lp, txt)
            FL.run_slice(root, lp, meta, prof, vd / "F", 0, 1, ids=ids)
            M = FL.merge_slices(vd / "F")
            x = M["m"].loc[M["m"].F_ok, ["rxn_id"] + M["cols"]].sort_values("rxn_id").reset_index(drop=True)
            xp = vd / "X182.parquet"
            D.write_parquet(xp, x)
            env = TL.fu2_env(vd / "fu2", vd / "fu2_res", xp, root, lp, meta, prof, vd / "F")
            rc = {c: TL.run_child(c, a, env) for c, a in (("atoms", [0, 1]), ("gate", []), ("tokens", []))}
            res[v] = dict(M=M, x=x, rc=rc, tok=vd / "fu2" / "tokens", atoms=vd / "fu2" / "atoms")
        a, b = res["true"], res["nan"]
        ma, mb = (r["M"]["m"].sort_values("rxn_id").reset_index(drop=True) for r in (a, b))
        cols = a["M"]["cols"]
        st_same = (list(ma.rxn_id) == list(mb.rxn_id) and list(ma.xtb_status) == list(mb.xtb_status)
                   and list(ma.ext_status.astype(str)) == list(mb.ext_status.astype(str)))
        rows, n_tok = [], 0
        for k, rid in enumerate(ma.rxn_id.astype(int)):
            rr = dict(rxn_id=rid, xtb_status=str(ma.xtb_status[k]), ext_status=str(ma.ext_status[k]),
                      F_ok=bool(ma.F_ok[k]), status_same=bool(ma.xtb_status[k] == mb.xtb_status[k]
                                                                and str(ma.ext_status[k]) == str(mb.ext_status[k])))
            rr["x182_bit_equal"] = _bits_equal(ma.loc[k, cols].to_numpy(float), mb.loc[k, cols].to_numpy(float))
            ta, tb = a["tok"] / f"{rid}.npz", b["tok"] / f"{rid}.npz"
            if ta.is_file() or tb.is_file():
                rr["token_files"] = [ta.is_file(), tb.is_file()]
                rr["token_keys_differ"] = _npz_diff(ta, tb) if ta.is_file() and tb.is_file() else ["<missing>"]
                n_tok += ta.is_file() and tb.is_file()
            aa, ab = a["atoms"] / f"{rid}.npz", b["atoms"] / f"{rid}.npz"
            if aa.is_file() and ab.is_file():
                rr["atoms_keys_differ"] = _npz_diff(aa, ab)
            rows.append(rr)
        g_diff = _frame_diff(a["M"]["g"].sort_values("rxn_id"), b["M"]["g"].sort_values("rxn_id"))
        e_diff = _frame_diff(a["M"]["e"].sort_values("rxn_id"), b["M"]["e"].sort_values("rxn_id"),
                             skip=[c for c in a["M"]["e"].columns if c.startswith("t_")])
        feat_in_gdiff = sorted(set(g_diff) & set(cols))
        tok_ok = all(not r.get("token_keys_differ") for r in rows if "token_files" in r)
        rep.update(rc={v: res[v]["rc"] for v in VARIANTS}, status_same=st_same,
                   n_F_ok=int(ma.F_ok.sum()), n_tokens_compared=int(n_tok),
                   x182_all_bit_equal=all(r["x182_bit_equal"] for r in rows),
                   tokens_all_bit_equal=tok_ok,
                   atoms_all_equal=all(not r.get("atoms_keys_differ") for r in rows if "atoms_keys_differ" in r),
                   g1_columns_differ=g_diff, g1_feature_columns_differ=feat_in_gdiff, ext_columns_differ=e_diff,
                   x182_sha256={v: D.sha256(base / v / "X182.parquet") for v in VARIANTS}, rows=rows)
        # the FU2 children (atoms.py root run / gate / tokens) have no element rule, so each must exit 0
        rep["pass"] = bool(st_same and rep["x182_all_bit_equal"] and tok_ok and not feat_in_gdiff
                           and n_tok == int(ma.F_ok.sum()) and n_tok > 0 and rep["labels_nan_all_nan"]
                           and all(c == 0 for v in VARIANTS for c in res[v]["rc"].values()))
    except SystemExit as e:                                # dep_common.Stop / d1_g1.check_env / check_engines
        rep.update(error=f"exit {e.code}", traceback=traceback.format_exc()[-3000:])
        rep["pass"] = False
    except Exception as e:                                 # noqa: BLE001 — a crash is a failed check, recorded
        rep.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        rep["pass"] = False
    if write:
        D.write_json(OUT_JSON, rep)
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"] and len(a) in (1, 2):
        r = run_c1(int(a[1]) if len(a) == 2 else N_C1)
        print(json.dumps({k: v for k, v in r.items() if k not in ("rows", "traceback")}, indent=1, default=str))
        if r.get("traceback"):
            print(r["traceback"], file=sys.stderr)
        sys.exit(0 if r["pass"] else 3)
    print(__doc__)
    sys.exit(2)
