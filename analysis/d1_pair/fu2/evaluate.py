#!/usr/bin/env python3
"""fu2/evaluate.py — D1PAIR FU2 Phase E (docs/specs/D1PAIR_FU2_E.md; the rules are results/PREREG_D1PAIR_ADD2.md = ADD2).

Commands (submit from analysis/d1_pair/ through fu2/jobs/fu2.slurm; exit 0 = ok, 3 = STOP (spec 17), 2 = usage):
  python fu2/evaluate.py check       E-0  input gate          -> results/fu2/E_check.json, E_gpu.csv
  python fu2/evaluate.py rep         E-1  representatives     -> E_representative.json, E_val_kcal.csv  (no test score read)
  python fu2/evaluate.py primary     E-2 + E-7                -> E_primary_tests.csv, E_primary_verdict.json, E_crosscheck.csv
                                     [--prov FILE]  full-precision provisional values (json, see crosscheck()) instead of
                                                    the 6-decimal table of spec 8
  python fu2/evaluate.py secondary   E-3                      -> E_secondary_tests.csv
  python fu2/evaluate.py hammett     E-4                      -> E_metrics.csv, E_bends.csv, E_bend_rates.json,
                                                                 E_stages_descriptive.csv, E_target_multiples.csv
  python fu2/evaluate.py describe    E-5                      -> E_epochs.csv, E_epochs_summary.json, E_seed_sd.csv, E_axis.json
  python fu2/evaluate.py figures     figF2_1 / 2 / 3 / 5      -> results/fu2/figures/*.{png,pdf,csv}, CAPTIONS_FU2.md
  python fu2/evaluate.py explain     E-6 (exploratory)        -> E_explain.csv, E_explain_zero.csv, E_explain.json, figF2_4_atoms
  python fu2/evaluate.py all         check -> rep -> primary -> secondary -> hammett -> describe -> figures
  (--prov FILE only with primary / all and FILE must exist; any other extra argument -> usage, exit 2)
Every output is written atomically (fu2_common.write_atomic / write_json, results/fu2 only); a rerun recomputes the same
values (fixed bootstrap seed, deterministic order), so resubmitting is safe. NaN cells of E_* CSV and JSON files are
audited (E_nan_audit.json, rebuilt per written file: only NaN that are NaN by definition, with the reason; any other NaN =
STOP). Every written file is recorded in E_provenance.json (D14: sha256, evaluate.py, E_check.json and
E_representative.json sha256, command, interpreter / package versions).

PUBLIC API (plain functions, importable without torch / matplotlib; D1LADDER spec 11 imports them):

  compare(x: pd.DataFrame, b: pd.DataFrame, pairs: pd.DataFrame, targets: list[str] | None = None,
          n_boot: int = 2000, seed: int = 20261019) -> pd.DataFrame
      x = comparator predictions, b = atom-side predictions; one row per target with d = MAE(x) - MAE(b) (d > 0: b better,
      ADD2 4 sign) from d1_eval.boot_diff(e_x, e_b, cores) unchanged (core cluster bootstrap, 95 % percentile CI, two-sided
      p) and d1_eval.fold_wins. x / b: long prediction frames with columns pair_id, target, y, yhat (+ fold: b's fold is
      used for the fold count). pairs: the evaluation pair set, one row per pair, column core (+ optional others), pair ids
      as the index or as a pair_id column. Every pair of `pairs` must occur exactly once per target in x and in b, with
      finite yhat and the same y (|dy| <= 1e-9), and, when both frames have a fold column, the same fold; otherwise STOP
      (SystemExit(3)). Rows outside `pairs` are ignored (also for the duplicate check). targets=None = every target
      present in x or in b (a target missing from one side = STOP).
      Returns columns: target, n_pairs, n_cores, mae_x, mae_b, d, ci_lo, ci_hi, p, folds_b_lower, n_folds.

  bends_tab(d: pd.DataFrame, **labels) -> pd.DataFrame
      the loop body of d1_eval.bends (rev 9, fixed to the arm B representative there) for ONE model's pair rows: keeps
      in_P_main & ref_kind == "reference" & pos == "p" & sigma known, per para family and per hammett_eval.CH target
      hammett_eval.family_series (MIN_X_FAM 12, H at the origin) + hammett_eval.calls on [DFT, prediction] unchanged.
      d: columns target, y, yhat, fam, pos, sigma, Xb, in_P_main, ref_kind, hidden_V_fam, hidden_bend_fam (= a prediction
      frame joined with S_folds.csv, see annotate()); at most one row per (pair_id, target) (else STOP).
      labels (e.g. split="T1", model="REP") become the leading constant columns. Returns columns: *labels, fam, target,
      n_points, bent_dft, bent_pred, V_dft, V_pred, gain_dft, gain_pred, psi_dft, psi_pred, left_dft, right_dft,
      left_pred, right_pred, hidden_V_fam, hidden_bend_fam  (bends_tab(d, split="T1") = the d1_eval.bends row layout).

  metrics_tab(L, by=("model",)) -> DataFrame   n / MAE / RMSE per set x position, R2_fam / rho error (d1_eval.family_eval)
  pair_sets_gate(frames, ref, S) -> dict      STOP unless every frame has exactly the (fold, pair_id) set ref per target
  pick_representatives(V) -> {target: config} ADD2 3 rule (lowest 5-fold mean, ties in CONFIG_NAMES order)
  annotate(frame, S), s_table(), eval_pairs(S, ref, para=False), sy_of(D, rows) (= train.Std(D, rows).sy; needs torch)
  The prediction frames M_predictions.parquet carries (pair_id, ref_kind, fold, split, model, variant, target, y, yhat,
  yhat_seed_sd) have no pos / fam / core columns: annotate() joins them from S_folds.csv (SCOL).

Reused unchanged (same function = same scoring): d1_eval.boot_diff (B 2,000, seed 20261019), d1_eval.clopper (via
d1_eval.bend_rates), d1_eval.family_eval / hammett_frame / fold_wins / mae / rmse, d1pair_common.holm, hammett_eval.calls /
family_series (MIN_X_FAM 12), fu2/train.Data / Data.split / Std (s_y), fu2/models.build / AW.atom_contrib / BiasedMHA
return_attn (E-6). fu2_common.py, tokens.py, models.py, train.py are imported, never modified.

DECISIONS (spec gaps; each = the recommended option, fixed before any FU2 test score was computed):
  D1  Review gate (spec 3 / 17 "E-0 before the user review = STOP"): every command except the tests requires
      results/fu2/E_tests.json written by fu2/tests/test_eval.py for THIS evaluate.py and test_eval.py (sha256) with
      T-2..T-8 and X-1..X-3 passed, both files committed and unchanged (git), and FU2E_REVIEWED=<sha256 prefix (>= 12 hex)
      of this evaluate.py> in the job environment (the user's review mark). T-1 (package imports) is reported, not gated:
      env_d1fu2 has no matplotlib / RDKit (results/fu2/env_pip_freeze.txt); spec says report, do not install. `figures`
      and `explain` therefore check their own imports and STOP if they are missing; the scoring commands need neither.
      T-1env (E-T ran in env_d1fu2) IS gated, and the scoring commands (check .. describe) must run in the interpreter
      E-T ran in (sys.prefix = E_tests.json prefix), so the T-2 / T-3 / X-3 equality certificates hold for the scoring
      stack; figures / explain may run elsewhere (their interpreter is recorded, D14). E_tests.json also pins the
      sha256 of the reused code (code_files(): d1_eval, hammett_eval, d1_figures, d1pair_common, rev5/rev6_common,
      fu2_common, train, models, tokens, atoms); a change after E-T = STOP.
  D2  T-5: S_splits.json counts are BEFORE the atom filter, unit json counts after it (fold 1-4 inner_train lose the one
      pair without atom data). Checked: json n_inner_* + dropped_no_atoms[inner_*] = S_splits.json, and json n_inner_* =
      len(Data.split()[inner_*]) recomputed.
  D3  T-2 "all columns equal": bool / int / str columns exactly, float columns |a - b| <= 1e-9 max(1, |b|) (CSV round
      trip, BLAS); exact equality is recorded as well.
  D4  E-7 with the 6-decimal table of spec 8: tolerance max(spec tolerance, 5e-7) = the official value must round to the
      table value (1e-9 cannot be met by 6-decimal numbers). With --prov FILE (full precision) the spec tolerances apply.
      FU2_RESULTS_JUDGMENT.md is not on disk; the numbers are spec 8 verbatim (PROV_* below).
  D5  E-3 sign: "P vs Q" -> b = P (first named: the atom side / the richer input), x = Q; d = MAE(Q) - MAE(P) > 0 means P
      is better (A+G vs A = what the 182 features add; A+G vs C-XGB = what the atoms add; AW vs GNN -> b = AW).
  D6  Extra E-0 gates (stricter than spec 4, all STOP): ADD2 inputs_sha256 (11 files) unchanged; S_folds.csv / F_pairs.csv
      = rev 9 results/E.json inputs_sha256; FU2 M_predictions sha256 = M_report, 472,440 rows, no NaN; unit parquet sha256
      = unit json; every gate_M1 recorded ok; y of every model = S dd (1e-9); split-file folds = S fold_T1; test-pair
      counts = ADD2 data; removed no-atom pairs = ADD2 data; M_predictions ref_kind = S; hammett_eval.py sha256 = the
      appendix pin (d1pair_common.APPENDIX_SHA). E_check.json inputs_sha256 (re-verified by every later command) also
      holds the reused code (code_files()), the 11 ADD2 inputs, the 150 final-model .pt files and smiles_d1.csv (E-6).
  D7  E-1 reads only unit json tuning records and training-side labels (train.Data): pandas.read_parquet is blocked for
      every prediction file (FU2 M_predictions, $FU2 units, $D1PAIR_SCRATCH/M) while rep runs (no_test_scores()).
  D8  E_representative.json is flat {target: config} (as rev 9); per-fold and 5-fold-mean kcal values in E_val_kcal.csv.
  D9  E-4: 11 models (6 configs, REP, C-XGB, C-rep, XGBW, null), 10 targets; every model restricted to the same test
      pairs (pairs without atom data removed); stages (descriptive) for REP and C-XGB on the 10 targets; target multiples
      for all 11 models on the 9 ADD2 7 targets.
  D10 E-5: E_seed_sd over all T1 test rows of each config (P_sens | A_main with atom data); E_axis: PCA of the centred
      covariance (kcal/mol, no scaling), loadings sign-fixed (largest |loading| > 0); the REP errors are reported both as
      the variance share along the truth PC1 and as their own PC1 share; Pearson correlations.
  D11 figF2_3 families: the first three of the rev 9 fig3_hammett.csv families (file order) with a P_main para test
      series (>= 12 points incl. H) for barrier, oi and elst.
  D12 E-6: one drawn pair per family = largest |sigma_p| (ties: Xb name); maps from the 5 final models of the fold in
      which the pair is test (AW-A+G contributions, TF-A+G cross-attention); H values added to their owner heavy atom for
      the drawing (owner = nearest heavy atom of the same fragment in the TS); zero-out test on every para P_main test pair
      of the six families with all 6 configs, seed-mean predictions, control = 10 draws of the same number of random
      non-reacting non-X atoms of a (rng 20261019 + pair-table row).
      X group of a pair (a = X-substituted, b = its reference; d1_pairs.py): the reactant heavy-atom graph of b (reactant
      side of $D1PAIR_SCRATCH/G/smiles_d1.csv, atom map = TS index + 1, maps and elements verified against the tokens) is
      matched into a's with RDKit GetSubstructMatches (uniquify False); the first match that maps b's five reacting atoms
      (dip_a, dip_mid, dip_b, dph_a, dph_b) onto a's same roles is used; a's unmatched heavy atoms are the X heavy atoms,
      plus every H whose owner is one of them. No such match -> the pair is skipped and listed (E_explain.json).
  D13 Figures: d1_figures style() / palette (rev 9); composite encoding (model hue, lighter = variant A); hue slots in
      fixed order AW 1, GNN 2, C-XGB 3 (rev 9 arm C), XGBW 4, C-rep 5, TF 6, REP 8; DFT in ink; diverging blue-gray-red,
      sequential gray-blue for the E-6 maps.
  D14 Provenance between commands (fixer review 2026-10-23): every results/fu2 file written by a command is recorded in
      E_provenance.json; a command reads an earlier E_* output only if its recorded sha256, evaluate.py sha256 and
      E_check.json sha256 equal the current ones (E_representative.json: read_rep; E-2 / E-7 outputs: require_e7;
      E_metrics / E_bend_rates / E_epochs: figures), else STOP. secondary, hammett, describe, figures and explain also
      require E-7 passed (spec 17): E_crosscheck.csv all ok, or the user's acceptance FU2E_E7_ACCEPTED=<sha256 prefix
      (>= 12 hex) of that E_crosscheck.csv> after an E-7 STOP. Interpreter and package versions per file (env).
  D15 E-6 integrity: each final-model .pt has unit = its name and std.sy = unit json final[s].sy (1e-12); the seed-mean
      CPU reproduction of M_predictions must be within RECON_TOL = 1e-3 kcal/mol for every config (float32 CPU single-pair
      batch vs GPU 256-pair batch differ ~1e-5; wrong weights / standardisation >= 1e-2), else STOP before writing.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                    # analysis/d1_pair/fu2
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import fu2_common as F  # noqa: E402
import d1_eval as E9  # noqa: E402
import hammett_eval as HE  # noqa: E402

C = F.C
THIS = Path(__file__).resolve()
RES = F.FU2_RES                                           # results/fu2
FIG = RES / "figures"
UNITS = F.UNITS                                           # $D1PAIR_SCRATCH/fu2/units
M9_T1 = C.SCRATCH / "M" / "units" / "T1"                  # rev 9 units (C-XGB, C representative)
XW_DIR = C.SCRATCH / "M" / "units_fu" / "XW"              # FU1 XGBW units
ADD2_MD = C.RES / "PREREG_D1PAIR_ADD2.md"
ADD2_JSON = C.RES / "prereg_d1pair_add2.json"
REV9_REP = C.RES / "E_representative.json"
REV9_E = C.RES / "E.json"
REV9_FIG3 = C.RES / "figures" / "fig3_hammett.csv"
TESTS_JSON = RES / "E_tests.json"
TEST_FILE = HERE / "tests" / "test_eval.py"
REVIEW_ENV = "FU2E_REVIEWED"
E7_ENV = "FU2E_E7_ACCEPTED"                               # D14: the user's acceptance of listed E-7 differences
PROV_FILE = RES / "E_provenance.json"                     # D14
SMILES = C.SCRATCH / "G" / "smiles_d1.csv"                # E-6 (pinned at E-0, D6)
RECON_TOL = 1e-3                                          # D15 (kcal/mol)
RATE_KEYS = {"sens", "fpr", "V_sens", "sens_ci", "fpr_ci", "V_sens_ci"}   # d1_eval.bend_rates: NaN when denominator 0

MODELS3 = ("AW", "GNN", "TF")
VARIANTS = ("A", "A+G")
CONFIGS = [(m, v) for m in MODELS3 for v in VARIANTS]     # ADD2 3 tie order: AW-A, AW-A+G, GNN-A, GNN-A+G, TF-A, TF-A+G
CONFIG_NAMES = [f"{m}-{v}" for m, v in CONFIGS]
TARGETS = list(F.TARGETS)                                 # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
PRIMARY = list(F.PRIMARY)                                 # 9 primary targets (8 channels + barrier)
CH9 = list(E9.CH)                                         # hammett_eval.CH (bend targets)
N_FOLDS = 5
N_SEEDS = len(F.FINAL_SEEDS)
N_M_ROWS = 472440
Y_TOL = 1e-9
SETS = ("P_main", "A_main", "P_sens", "P_main_no_d0_overlap")
SCOL = ["ref_kind", "in_P_main", "in_P_sens", "in_A_main", "pos", "core", "fam", "sigma", "Xb", "d0_overlap",
        "hidden_V_fam", "hidden_bend_fam"]
MODELS_E4 = CONFIG_NAMES + ["REP", "C-XGB", "C-rep", "XGBW", "null"]
TARGET_MAE_SPEC = {"barrier": 0.16, "d1": 0.19, "d2": 0.04, "elst": 0.21, "pauli": 0.41, "oi": 0.29, "cpcm": 0.22,
                   "cds": 0.01, "disp": 0.05}            # spec 10 = ADD2 7 (checked against the ADD2 json)
POST_HOC = "사후 정의 (post-hoc definition: median bend size x 0.28, FU1 L-3; ADD2 7)"
DESCRIPTIVE = "설명용 (descriptive only, rev 9 stage criteria, not a verdict)"
VERDICT = {"ge5": ("원자 입력이 182개 feature보다 낫다", "atom input better than the 182 features"),
           "2-4": ("일부 채널에서만", "only some channels"), "le1": ("이득 없음", "no gain")}

# ---------------------------------------------------------------- spec 8: reviewer provisional values (2026-10-21)
PROV_REP = {t: "GNN-A+G" for t in ("barrier", "d1", "d2", "disp", "cpcm", "cds")} | \
           {t: "AW-A+G" for t in ("elst", "pauli", "oi", "c_ghost")}
_PROV_COLS = ("n_pairs", "mae_CXGB", "mae_atom", "d", "ci_lo", "ci_hi", "p", "p_holm")
PROV_PRIMARY = {t: dict(zip(_PROV_COLS, v)) for t, v in {
    "d1": (6884, 0.456793, 0.414311, 0.042482, 0.017794, 0.064004, 0.001999, 0.009995),
    "d2": (6884, 0.339738, 0.328760, 0.010977, -0.005557, 0.029202, 0.190905, 0.381809),
    "elst": (6884, 1.127615, 1.018368, 0.109247, 0.060284, 0.175941, 0.001000, 0.008996),
    "pauli": (6884, 2.156280, 1.937573, 0.218707, 0.114549, 0.351784, 0.001000, 0.008996),
    "oi": (6884, 1.238167, 1.138605, 0.099562, 0.050517, 0.158294, 0.001000, 0.008996),
    "disp": (6884, 0.176034, 0.175996, 0.000038, -0.009202, 0.009100, 0.965517, 0.965517),
    "cpcm": (6884, 0.401902, 0.381013, 0.020889, 0.007183, 0.034424, 0.006997, 0.027986),
    "cds": (6884, 0.079739, 0.076301, 0.003438, 0.000099, 0.006998, 0.044978, 0.134933),
    "barrier": (6884, 0.554189, 0.465983, 0.088206, 0.064392, 0.111542, 0.001000, 0.008996)}.items()}
PROV_BETTER = ["d1", "elst", "pauli", "oi", "cpcm", "barrier"]
PROV_WORSE: list = []
TOL_SPEC = {"n_pairs": 1e-6, "mae_CXGB": 1e-6, "mae_atom": 1e-6, "d": 1e-6, "ci_lo": 1e-9, "ci_hi": 1e-9, "p": 1e-9,
            "p_holm": 1e-9}
ROUND6 = 5e-7 + 1e-12                                     # half a unit of the 6th decimal

# ---------------------------------------------------------------- E-6
ROLES = ("dip_a", "dip_mid", "dip_b", "dph_a", "dph_b")
ZERO_COLS = ("dq_dist", "dq_int", "dv_dist", "dv_int")
MAP_CHANNELS = ("oi", "elst", "pauli")
N_CTRL = 10
X_ATOM_METHOD = ("X group of a pair (a = X-substituted reaction, b = its reference, d1_pairs.py): b's reactant heavy-atom "
                 "graph (reactant side of smiles_d1.csv, atom map = TS index + 1, verified against the token elements) is "
                 "matched into a's (RDKit GetSubstructMatches, uniquify False); the first match mapping b's five reacting "
                 "atoms onto a's same roles is used; a's unmatched heavy atoms = X heavy atoms, plus every H owned by one "
                 "of them (owner = nearest heavy atom of the same fragment in the TS). No such match: pair skipped.")
NAN_LOG: dict = {}
WRITTEN: list = []                                        # results/fu2-relative paths written by this command (D14)


# ================================================================ small helpers
def stop(msg):
    """spec 17 STOP: message to stderr, SystemExit(3)."""
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(3)


def need(*paths):
    miss = [str(p) for p in paths if not Path(p).is_file()]
    if miss:
        stop(f"input missing: {miss[:10]}" + (f" (+{len(miss) - 10} more)" if len(miss) > 10 else ""))


def read_json(p):
    return json.loads(Path(p).read_text())


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True)


def git_state(p):
    """None if p is committed and unchanged since its commit, else the problem."""
    if git("ls-files", "--error-unmatch", str(p)).returncode != 0:
        return "not committed"
    if git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
        return "modified since its commit"
    return None


def unit_name(m, v, k):
    return f"T1__{m}__{v.replace('+', '')}__f{k}"               # = train.uname


def unit_json(m, v, k):
    p = UNITS / f"{unit_name(m, v, k)}.json"
    need(p)
    return read_json(p)


def rel(path):
    """results/fu2-relative name of an output (key of E_nan_audit.json / E_provenance.json)."""
    p = Path(os.path.abspath(path))
    try:
        return p.relative_to(Path(os.path.abspath(RES))).as_posix()
    except ValueError:
        return p.name


def write_csv(path, df, rules=None):
    audit(df, rel(path), rules)
    F.write_atomic(path, lambda t: df.to_csv(t, index=False))
    WRITTEN.append(rel(path))


def json_nans(o, path=""):
    """paths ('/key/0/...') of the NaN floats inside a JSON-able object."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from json_nans(v, f"{path}/{k}")
    elif isinstance(o, (list, tuple, np.ndarray)):
        for i, v in enumerate(list(o)):
            yield from json_nans(v, f"{path}/{i}")
    elif isinstance(o, (float, np.floating)) and np.isnan(o):
        yield path or "/"


def write_json(path, obj, nan_ok=None):
    """F.write_json + the NaN audit of spec 17 for JSON results: nan_ok = (fn(path str) -> allowed, reason)."""
    name = rel(path)
    found = list(json_nans(obj))
    if found:
        ok_fn, reason = nan_ok or ((lambda q: False), "")
        bad = [q for q in found if not ok_fn(q)]
        if bad:
            stop(f"{name}: {len(bad)} NaN values outside the defined case (e.g. {bad[:5]})")
        NAN_LOG.setdefault(name, {})["(json)"] = dict(n_nan=len(found), reason=reason, paths=found[:50])
    F.write_json(path, obj)
    WRITTEN.append(name)


def audit(df, name, rules=None):
    """spec 17 common: NaN only where NaN by definition. rules = {col: (fn(df) -> bool mask of rows allowed NaN, reason)}."""
    rules = rules or {}
    for col in df.columns:
        isn = df[col].isna()
        if not isn.any():
            continue
        if col not in rules:
            stop(f"{name}: {int(isn.sum())} NaN in column {col} (not NaN by definition)")
        ok_fn, reason = rules[col]
        bad = isn & ~ok_fn(df).astype(bool)
        if bad.any():
            stop(f"{name}: {int(bad.sum())} NaN in column {col} outside the defined case ({reason})")
        NAN_LOG.setdefault(name, {})[col] = dict(n_nan=int(isn.sum()), reason=reason)


def env_info():
    """interpreter of this command + versions of the packages it has loaded (D14; nothing is imported here)."""
    mods = {m: str(getattr(sys.modules[m], "__version__", "?")) for m in
            ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "torch", "matplotlib", "rdkit") if m in sys.modules}
    return dict(prefix=sys.prefix, python=sys.version.split()[0], host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), packages=mods)


def code_files():
    """D1 / D14: the reused code whose sha256 E-T pins (scoring functions, s_y path, models, figure style)."""
    mods = [E9, HE, C, C.R6, C.R5, F]
    return sorted({str(Path(m.__file__).resolve()) for m in mods}
                  | {str(HERE / f) for f in ("train.py", "models.py", "tokens.py", "atoms.py")}
                  | {str(HERE.parent / "d1_figures.py")})


def code_sha():
    return {p: F.sha256(p) for p in code_files()}


def flush_nan_log():
    """E_nan_audit.json: the entries of the files written by this command are replaced (stale ones dropped);
    E_provenance.json: one record per written file (D14)."""
    if not WRITTEN:
        NAN_LOG.clear()
        return
    names = list(dict.fromkeys(WRITTEN))
    p = RES / "E_nan_audit.json"
    old = read_json(p) if p.is_file() else {}
    for n in names:
        old.pop(n, None)
    old.update({n: v for n, v in NAN_LOG.items() if n in names})
    F.write_json(p, old)
    ec, rp = RES / "E_check.json", RES / "E_representative.json"
    common = dict(evaluate_sha256=F.sha256(THIS), e_check_sha256=F.sha256(ec) if ec.is_file() else None,
                  representative_sha256=F.sha256(rp) if rp.is_file() else None, command=" ".join(sys.argv[1:]),
                  env=env_info(), date=time.strftime("%Y-%m-%d %H:%M:%S"))
    pv = read_json(PROV_FILE) if PROV_FILE.is_file() else {}
    for n in names:
        pv[n] = dict(sha256=F.sha256(RES / n), **common)
    F.write_json(PROV_FILE, pv)
    NAN_LOG.clear()
    WRITTEN.clear()


def require_fresh(*names, rep=True):
    """D14: each results/fu2 file was written by THIS evaluate.py after the current E_check.json (and, with rep, from the
    current E_representative.json) and is unchanged since (E_provenance.json); STOP otherwise."""
    pv = read_json(PROV_FILE) if PROV_FILE.is_file() else {}
    me, ec = F.sha256(THIS), F.sha256(RES / "E_check.json")
    rp = RES / "E_representative.json"
    rs = F.sha256(rp) if rp.is_file() else None
    bad = []
    for n in names:
        p, r = RES / n, pv.get(n)
        if not p.is_file():
            bad.append(f"{n}: missing")
        elif r is None or r.get("sha256") != F.sha256(p):
            bad.append(f"{n}: not in E_provenance.json or changed after it was written")
        elif r.get("evaluate_sha256") != me or r.get("e_check_sha256") != ec:
            bad.append(f"{n}: written by another evaluate.py or before the current E_check.json (rerun the earlier steps)")
        elif rep and r.get("representative_sha256") != rs:
            bad.append(f"{n}: written from another E_representative.json (rerun primary)")
    if bad:
        stop("stale or unrecorded results (D14):\n  " + "\n  ".join(bad))


def require_e7():
    """spec 17 / D14: E-7 passed for the current E-0 / E-1 / E-2 (E_crosscheck.csv all ok), or the user accepted the
    listed differences (FU2E_E7_ACCEPTED = sha256 prefix (>= 12 hex) of that E_crosscheck.csv)."""
    require_fresh("E_primary_tests.csv", "E_crosscheck.csv")
    p = RES / "E_crosscheck.csv"
    X = pd.read_csv(p)
    nbad = int((X.ok.astype(str) != "True").sum())
    if nbad:
        tok, sh = os.environ.get(E7_ENV, "").strip().lower(), F.sha256(p)
        if len(tok) < 12 or not sh.startswith(tok):
            stop(f"E-7 did not pass ({nbad} rows of E_crosscheck.csv outside the tolerance): later steps need the user's "
                 f"decision ({E7_ENV}=<sha256 prefix of E_crosscheck.csv, {sh[:12]}...> to accept them)")
        print(f"E-7: {nbad} differences accepted by the user ({E7_ENV}={tok})", flush=True)


def always(df):
    return pd.Series(True, index=df.index)


# ================================================================ gates (D1)
def require_review(same_env=True):
    """D1: E-T passed for this code, code committed, user review mark (FU2E_REVIEWED) present; with same_env (the scoring
    commands) this interpreter = the one E-T ran in; exit 3 otherwise."""
    need(TESTS_JSON, TEST_FILE)
    t = read_json(TESTS_JSON)
    me, tf = F.sha256(THIS), F.sha256(TEST_FILE)
    bad = []
    if t.get("evaluate_sha256") != me:
        bad.append("E_tests.json was written for another evaluate.py (rerun fu2/tests/test_eval.py)")
    if t.get("test_sha256") != tf:
        bad.append("E_tests.json was written by another test_eval.py")
    if not t.get("gate_pass"):
        bad.append(f"E_tests.json: not all of T-1env, T-2..T-8 / X-1..X-3 passed ({t.get('failed')})")
    cs = code_sha()
    if t.get("code_sha256") != cs:
        old = t.get("code_sha256") or {}
        bad.append(f"reused code changed after E-T: {sorted(k for k in set(cs) | set(old) if cs.get(k) != old.get(k))}")
    if same_env and Path(sys.prefix).resolve() != Path(str(t.get("prefix"))).resolve():
        bad.append(f"scoring command runs in {sys.prefix}, E-T ran in {t.get('prefix')} (D1: submit with PYENV=fu2)")
    for p in (THIS, TEST_FILE):
        st = git_state(p)
        if st:
            bad.append(f"{p.name} {st} (spec 3: commit after E-T)")
    tok = os.environ.get(REVIEW_ENV, "").strip().lower()
    if len(tok) < 12 or not me.startswith(tok):
        bad.append(f"{REVIEW_ENV} is not the sha256 prefix (>= 12 hex) of this evaluate.py ({me[:12]}...): the user "
                   "review of the evaluation code (ADD2 9, spec 3) is not recorded")
    if bad:
        stop("E-T / review gate:\n  " + "\n  ".join(bad))
    return t


def require_check():
    """E-0 must have passed with this evaluate.py, and every input it hashed must be unchanged."""
    p = RES / "E_check.json"
    need(p)
    rec = read_json(p)
    if not rec.get("ok"):
        stop("E_check.json ok = false (E-0 STOP)")
    if rec.get("evaluate_sha256") != F.sha256(THIS):
        stop("E_check.json was written by another evaluate.py (rerun check)")
    changed = [k for k, s in rec["inputs_sha256"].items() if not Path(k).is_file() or F.sha256(k) != s]
    if changed:
        stop(f"inputs changed after E-0: {changed[:5]}")
    return rec


@contextmanager
def no_test_scores(strict=True):
    """D7: pandas.read_parquet refuses prediction files while active (FU2 M_predictions + FU2 unit parquets; with strict
    also every rev 9 / FU1 unit under $D1PAIR_SCRATCH/M). Raises STOP before anything is read."""
    orig = pd.read_parquet
    blocked = [UNITS.resolve()] + ([(C.SCRATCH / "M").resolve()] if strict else [])
    fu2_pred = (RES / "M_predictions.parquet").resolve()

    def guarded(path, *a, **kw):
        p = Path(str(path)).resolve()
        if p == fu2_pred or any(b == p.parent or b in p.parents for b in blocked):
            stop(f"test-score file read while forbidden (E-1 must not read test scores): {p}")
        return orig(path, *a, **kw)
    pd.read_parquet = guarded
    try:
        yield
    finally:
        pd.read_parquet = orig


# ================================================================ tables and frames (no torch)
def s_table():
    """S_folds.csv + F_pairs dd_<t> (= d1_eval.load) indexed by pair_id, + has_atoms (a and b in T_manifest, = train.Data)."""
    need(C.RES / "S_folds.csv", C.RES / "F_pairs.csv", RES / "T_manifest.csv")
    S = pd.read_csv(C.RES / "S_folds.csv")
    Fp = pd.read_csv(C.RES / "F_pairs.csv", usecols=["pair_id"] + [f"dd_{t}" for t in TARGETS])
    S = S.merge(Fp, on="pair_id", how="left", validate="1:1").set_index("pair_id")
    have = set(pd.read_csv(RES / "T_manifest.csv", usecols=["rxn_id"]).rxn_id.astype(int))
    S["has_atoms"] = S.a_rxn.astype(int).isin(have) & S.b_rxn.astype(int).isin(have)
    return S


def split_tests():
    """{fold: test pair ids} of T1 from the exported d1_train.masks files (sha256 = S_splits.json)."""
    need(RES / "S_splits.json")
    srep = read_json(RES / "S_splits.json")
    out = {}
    for k in range(N_FOLDS):
        name = f"T1__f{k}.json"
        p = F.SPLITS / name
        need(p)
        if F.sha256(p) != srep["units"][name]["sha256"]:
            stop(f"{p} changed after S_splits.json")
        out[k] = [str(x) for x in read_json(p)["test"]]
    return out


def ref_sets(S, tests):
    """(set of (fold, pair_id) with atom data, [(fold, pair_id) removed: no atom data])."""
    ref, removed = set(), []
    for k, ids in tests.items():
        for pid in ids:
            if bool(S.at[pid, "has_atoms"]):
                ref.add((int(k), pid))
            else:
                removed.append((int(k), pid))
    return ref, removed


def c_rep_models():
    """rev 9 arm C representative per target (results/E_representative.json T1|C|<t>; rev 9 rule over 6 models incl. SVR)."""
    need(REV9_REP)
    r = read_json(REV9_REP)
    return {t: r[f"T1|C|{t}"] for t in TARGETS}


def comparator_paths(rep9C):
    cx = {(t, k): M9_T1 / f"T1__C__XGB__{t}__f{k}.parquet" for t in TARGETS for k in range(N_FOLDS)}
    cr = {(t, k): M9_T1 / f"T1__C__{rep9C[t]}__{t}__f{k}.parquet" for t in TARGETS for k in range(N_FOLDS)}
    xw = {(t, k): XW_DIR / f"C__XGBW__{t}__f{k}.parquet" for t in TARGETS for k in range(N_FOLDS)}
    return cx, cr, xw


def _std(q, source):
    q = q.assign(source=source) if "source" not in q.columns else q
    cols = ["pair_id", "fold", "target", "y", "yhat", "source"] + (["yhat_seed_sd"] if "yhat_seed_sd" in q.columns else [])
    out = q[cols].copy()
    out["pair_id"] = out.pair_id.astype(str)
    out["fold"] = out.fold.astype(int)
    return out.reset_index(drop=True)


def read_unit_parquets(paths, source_by_t):
    parts = []
    for (t, k), p in paths.items():
        need(p)
        q = pd.read_parquet(p)
        if not ((q.target == t).all() and (q.fold.astype(int) == k).all()):
            stop(f"{p.name}: target / fold columns differ from the file name")
        parts.append(_std(q, source_by_t[t]))
    return pd.concat(parts, ignore_index=True)


def raw_frames(S, tests, rep9C):
    """{model: standard frame (pair_id, fold, target, y, yhat, source[, yhat_seed_sd])} before the atom filter."""
    need(RES / "M_predictions.parquet")
    M = pd.read_parquet(RES / "M_predictions.parquet")
    fr = {}
    for m, v in CONFIGS:
        fr[f"{m}-{v}"] = _std(M[(M.split == "T1") & (M.model == m) & (M.variant == v)], f"{m}-{v}")
    cx, cr, xw = comparator_paths(rep9C)
    fr["C-XGB"] = read_unit_parquets(cx, {t: "XGB" for t in TARGETS})
    fr["C-rep"] = read_unit_parquets(cr, rep9C)
    fr["XGBW"] = read_unit_parquets(xw, {t: "XGBW" for t in TARGETS})
    pid = [p for k in range(N_FOLDS) for p in tests[k]]
    fold = [k for k in range(N_FOLDS) for _ in tests[k]]
    fr["null"] = pd.concat([pd.DataFrame(dict(pair_id=pid, fold=fold, target=t, y=S.loc[pid, f"dd_{t}"].to_numpy(float),
                                              yhat=0.0, source="0")) for t in TARGETS], ignore_index=True)
    return fr, M


def filter_atoms(fr, S):
    """drop the pairs without atom data, the same rule for every model (ADD2 4); unknown pair ids = STOP."""
    out, removed = {}, {}
    for k, v in fr.items():
        unk = ~v.pair_id.isin(S.index)
        if unk.any():
            stop(f"{k}: {int(unk.sum())} rows with pair ids not in S_folds.csv")
        keep = v.pair_id.map(S.has_atoms).astype(bool).to_numpy()
        out[k], removed[k] = v[keep].reset_index(drop=True), int((~keep).sum())
    return out, removed


def pair_sets_diff(frames, ref, S, targets=TARGETS):
    """problems (list of str): per frame and target, (fold, pair_id) set != ref, duplicates, y != S dd, non-finite yhat."""
    probs = []
    for name, q0 in frames.items():
        extra = sorted(set(q0.target) - set(targets))
        if extra:
            probs.append(f"{name}: unexpected targets {extra}")
        for t in targets:
            q = q0[q0.target == t]
            if q.pair_id.duplicated().any():
                probs.append(f"{name} {t}: {int(q.pair_id.duplicated().sum())} duplicate pair rows")
                continue
            got = set(zip(q.fold.astype(int).tolist(), q.pair_id.astype(str).tolist()))
            if got != ref:
                probs.append(f"{name} {t}: {len(ref - got)} (fold, pair_id) missing, {len(got - ref)} unexpected "
                             f"(e.g. {sorted(ref - got)[:2]} / {sorted(got - ref)[:2]})")
            if len(q):
                dy = np.abs(q.y.to_numpy(float) - S.loc[q.pair_id, f"dd_{t}"].to_numpy(float))
                if not np.isfinite(dy).all() or dy.max() > Y_TOL:
                    probs.append(f"{name} {t}: y differs from S dd_{t} (max {np.nanmax(dy):.3g})")
                if not np.isfinite(q.yhat.to_numpy(float)).all():
                    probs.append(f"{name} {t}: non-finite yhat")
    return probs


def pair_sets_gate(frames, ref, S, targets=TARGETS):
    """STOP (exit 3) unless every frame has, per target, exactly the (fold, pair_id) set ref (after the atom filter), no
    duplicates, y = S dd and finite yhat (spec 4 / ADD2 9). Returns {model: {target: n}}."""
    probs = pair_sets_diff(frames, ref, S, targets)
    if probs:
        stop("test-pair lists differ between models (after removing the pairs without atom data):\n  "
             + "\n  ".join(probs[:20]))
    return {k: {t: int((v.target == t).sum()) for t in targets} for k, v in frames.items()}


def rep_frame(frames, rep):
    """REP = per target the rows of that target's representative config (source = the config)."""
    return pd.concat([frames[rep[t]][frames[rep[t]].target == t].assign(source=rep[t]) for t in TARGETS],
                     ignore_index=True)


def annotate(frame, S):
    """join the S_folds.csv columns SCOL (pos, fam, core, sigma, Xb, set flags, hidden flags, ref_kind) on pair_id."""
    return frame.join(S[SCOL], on="pair_id")


def eval_pairs(S, ref, para=False):
    """evaluation pairs (index pair_id, columns core, fold_T1): T1 test pairs with atom data in P_main reference
    (ADD2 4), para only with para=True. Sorted by pair_id."""
    ids = sorted({pid for _, pid in ref})
    P = S.loc[ids]
    m = P.in_P_main.astype(bool) & (P.ref_kind == "reference")
    if para:
        m &= P.pos == "p"
    return P.loc[m, ["core", "fold_T1"]]


def load_all(rep=None):
    """(S, {model: annotated frame}, ref): every model on the common test pairs, pair-list gate re-checked."""
    S = s_table()
    tests = split_tests()
    ref, _ = ref_sets(S, tests)
    raw, _ = raw_frames(S, tests, c_rep_models())
    fr, _ = filter_atoms(raw, S)
    pair_sets_gate(fr, ref, S)
    if rep is not None:
        fr["REP"] = rep_frame(fr, rep)
    return S, {k: annotate(v, S) for k, v in fr.items()}, ref


def read_rep():
    p = RES / "E_representative.json"
    need(p)
    require_fresh(p.name, rep=False)                      # D14: written by this evaluate.py after the current E-0
    rep = read_json(p)
    if set(rep) != set(TARGETS) or not set(rep.values()) <= set(CONFIG_NAMES):
        stop(f"{p.name} malformed: {rep}")
    return {t: rep[t] for t in TARGETS}


# ================================================================ public scoring functions
def _errors(fr, t, ids, name):
    q = fr[fr.target == t]
    q = q[q.pair_id.isin(ids)]                            # rows outside the evaluation pairs are ignored
    if q.pair_id.duplicated().any():
        stop(f"compare: {name} has {int(q.pair_id.duplicated().sum())} duplicate rows for target {t}")
    q = q.set_index("pair_id")
    miss = ids.difference(q.index)
    if len(miss):
        stop(f"compare: {len(miss)} pairs of the evaluation set missing from {name} for target {t} (e.g. {list(miss[:3])})")
    q = q.loc[ids]
    e = (q.yhat - q.y).to_numpy(float)
    if not np.isfinite(e).all():
        stop(f"compare: non-finite errors in {name} for target {t}")
    return e, q.y.to_numpy(float), (q.fold.to_numpy() if "fold" in q.columns else None)


def _same_folds(a, b):
    try:
        return bool(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
    except (TypeError, ValueError):
        return bool((np.asarray(a).astype(str) == np.asarray(b).astype(str)).all())


def compare(x, b, pairs, targets=None, n_boot=C.N_BOOT, seed=C.SEED):
    """d = MAE(x) - MAE(b) per target with d1_eval.boot_diff(e_x, e_b, cores) (x = comparator, b = atom side; d > 0 =
    b better). See the module docstring for the frame contract. Returns a DataFrame (one row per target)."""
    P = pairs.set_index("pair_id") if "pair_id" in pairs.columns else pairs
    if P.index.duplicated().any():
        stop("compare: duplicate pair ids in the evaluation pair set")
    if not len(P):
        stop("compare: empty evaluation pair set")
    if P.core.isna().any():
        stop("compare: pairs without core")
    ids = pd.Index(P.index.astype(str))
    cores = P.core.to_numpy()
    if targets is None:                                   # every target of either side; one side missing it = STOP
        have = set(x.target) | set(b.target)
        targets = [t for t in TARGETS if t in have] + sorted(str(t) for t in have - set(TARGETS))
    rows = []
    for t in targets:
        ex, yx, fx = _errors(x, t, ids, "x")
        eb, yb, fb = _errors(b, t, ids, "b")
        if not (np.abs(yx - yb) <= Y_TOL).all():
            stop(f"compare: x and b have different y for target {t} (max {np.nanmax(np.abs(yx - yb)):.3g})")
        if fx is not None and fb is not None and not _same_folds(fx, fb):
            stop(f"compare: x and b assign different folds to some evaluation pairs for target {t}")
        d, lo, hi, p = E9.boot_diff(ex, eb, cores, n_boot=n_boot, seed=seed)
        wins, nf = E9.fold_wins(ex, eb, fb) if fb is not None else (np.nan, np.nan)
        rows.append(dict(target=t, n_pairs=len(ex), n_cores=len(np.unique(cores)), mae_x=float(np.mean(np.abs(ex))),
                         mae_b=float(np.mean(np.abs(eb))), d=d, ci_lo=lo, ci_hi=hi, p=p, folds_b_lower=wins, n_folds=nf))
    return pd.DataFrame(rows, columns=["target", "n_pairs", "n_cores", "mae_x", "mae_b", "d", "ci_lo", "ci_hi", "p",
                                       "folds_b_lower", "n_folds"])


BEND_COLS = ["fam", "target", "n_points", "bent_dft", "bent_pred", "V_dft", "V_pred", "gain_dft", "gain_pred", "psi_dft",
             "psi_pred", "left_dft", "right_dft", "left_pred", "right_pred", "hidden_V_fam", "hidden_bend_fam"]


def bends_tab(d, **labels):
    """d1_eval.bends loop body for one model's pair rows (module docstring). Returns the bend / V calls per family x target."""
    if "pair_id" in d.columns and d[["pair_id", "target"]].duplicated().any():
        stop("bends_tab: more than one row per (pair_id, target): pass one model's rows")
    d = d[d.in_P_main & (d.ref_kind == "reference") & (d.pos == "p") & d.sigma.notna()]
    rows = []
    for f, g in d.groupby("fam"):
        for c in E9.CH:
            q = g[g.target == c]
            sy = HE.family_series(E9.hammett_frame(q, "y"), f, "y", C.MIN_X_FAM)
            if sy is None:
                continue
            sp_ = HE.family_series(E9.hammett_frame(q, "pred"), f, "pred", C.MIN_X_FAM)
            x, y, names = sy
            Y = np.c_[y, sp_[1]]
            ok, v, gain, psi, Lsl, Rsl = HE.calls(x, Y, names)
            rows.append(dict(**labels, fam=f, target=c, n_points=len(x), bent_dft=bool(ok[0]), bent_pred=bool(ok[1]),
                             V_dft=bool(v[0]), V_pred=bool(v[1]), gain_dft=float(gain[0]), gain_pred=float(gain[1]),
                             psi_dft=float(psi[0]), psi_pred=float(psi[1]), left_dft=float(Lsl[0]), right_dft=float(Rsl[0]),
                             left_pred=float(Lsl[1]), right_pred=float(Rsl[1]),
                             hidden_V_fam=bool(q.hidden_V_fam.iloc[0]), hidden_bend_fam=bool(q.hidden_bend_fam.iloc[0])))
    return pd.DataFrame(rows, columns=list(labels) + BEND_COLS)


def set_mask(d, s):
    """the four rev 9 evaluation sets (d1_eval.metrics)."""
    pm = d.in_P_main.astype(bool) & (d.ref_kind == "reference")
    if s == "P_main":
        return pm
    if s == "A_main":
        return d.in_A_main.astype(bool) & (d.ref_kind == "anchor")
    if s == "P_sens":
        return d.in_P_sens.astype(bool) & (d.ref_kind == "reference")
    if s == "P_main_no_d0_overlap":
        return pm & ~d.d0_overlap.astype(bool)
    raise KeyError(s)


def metrics_tab(L, by=("model",)):
    """rev 9 d1_eval.metrics body per (by..., target): n / MAE / RMSE for SETS x (all + positions); on P_main / all also
    n_fam, R2_fam median / quartiles, rho error median (d1_eval.family_eval)."""
    rows = []
    for key, d in L.groupby(list(by) + ["target"], sort=False):
        fe = E9.family_eval(d)
        for s in SETS:
            q0 = d[set_mask(d, s)]
            for pos in ["all"] + C.POSITIONS:
                q = q0 if pos == "all" else q0[q0.pos == pos]
                e = (q.yhat - q.y).to_numpy(float)
                r = dict(zip(by, key[:-1]), target=key[-1], set=s, pos=pos, n=len(q), mae=E9.mae(e), rmse=E9.rmse(e))
                if s == "P_main" and pos == "all" and len(fe):
                    r.update(n_fam=len(fe), r2_fam_median=float(fe.r2_fam.median()), r2_fam_q1=float(fe.r2_fam.quantile(0.25)),
                             r2_fam_q3=float(fe.r2_fam.quantile(0.75)), rho_err_median=float(fe.rho_err.median()))
                rows.append(r)
    return pd.DataFrame(rows, columns=list(by) + ["target", "set", "pos", "n", "mae", "rmse", "n_fam", "r2_fam_median",
                                                  "r2_fam_q1", "r2_fam_q3", "rho_err_median"])


def pick_representatives(V):
    """ADD2 3: per target the config with the lowest mean over the outer folds of val_kcal (float64, fold order);
    exact ties -> CONFIG_NAMES order. V: config, fold, target, val_kcal. NaN = STOP (spec 17 E-1)."""
    order = {c: i for i, c in enumerate(CONFIG_NAMES)}
    out = {}
    for t in [t for t in TARGETS if t in set(V.target)]:
        d = V[V.target == t]
        means = {}
        for c, g in d.groupby("config"):
            v = g.sort_values("fold").val_kcal.to_numpy(float)
            if len(v) != N_FOLDS or not np.isfinite(v).all():
                stop(f"E-1: validation MAE of {c} / {t} not finite or not {N_FOLDS} folds")
            means[c] = float(np.mean(v))
        out[t] = sorted(means, key=lambda c: (means[c], order[c]))[0]
    return out


def sy_of(D, rows):
    """s_y exactly as train.fit computes it: train.Std(D, rows).sy (std of y over the pairs rows, ddof 0, per target)."""
    import train as T
    return np.asarray(T.Std(D, rows).sy, dtype=float)


# ================================================================ E-0 check
def cmd_check():
    require_review()
    gates, inputs = [], {}

    def gate(name, ok, **detail):
        gates.append(dict(name=name, ok=bool(ok), **detail))
        if not ok:
            print(f"[E-0] FAIL {name}: {json.dumps(detail, default=str)[:800]}", file=sys.stderr, flush=True)

    rep9C = c_rep_models()
    cx, cr, xw = comparator_paths(rep9C)
    unit_files = [UNITS / f"{unit_name(m, v, k)}.{e}" for m, v in CONFIGS for k in range(N_FOLDS) for e in ("json", "parquet")]
    fixed = [ADD2_MD, ADD2_JSON, REV9_E, REV9_REP, REV9_FIG3, C.RES / "S_folds.csv", C.RES / "F_pairs.csv",
             RES / "S_splits.json", RES / "T_manifest.csv", RES / "T_report.json", RES / "M_predictions.parquet",
             RES / "M_report.json", F.SPLITS / "pairs.parquet"] + [F.SPLITS / f"T1__f{k}.json" for k in range(N_FOLDS)]
    fixed += [Path(p) for p in code_files()] + [SMILES, RES / "Q_manifest.csv"] + \
        [UNITS / f"{unit_name(m, v, k)}__s{s}.pt" for m, v in CONFIGS for k in range(N_FOLDS) for s in range(N_SEEDS)]
    comp = list(cx.values()) + list(cr.values()) + list(xw.values())
    comp_js = [p.with_suffix(".json") for p in comp]
    missing = [str(p) for p in fixed + unit_files + comp + comp_js if not p.is_file()]
    if missing:
        write_json(RES / "E_check.json", dict(ok=False, missing=missing, evaluate_sha256=F.sha256(THIS)))
        stop(f"E-0: input missing: {missing[:10]}")
    for p in fixed + unit_files + comp:
        inputs[str(p)] = F.sha256(p)

    # ---- ADD2
    add2 = read_json(ADD2_JSON)
    md = inputs[str(ADD2_MD)]
    gate("ADD2 md sha256 = json md_sha256", md == add2.get("md_sha256"), md=md, json=add2.get("md_sha256"))
    for p in (ADD2_MD, ADD2_JSON):
        st = git_state(p)
        gate(f"{p.name} committed and unchanged since", st is None, state=st)
    ch = sorted(k for k, s in add2["inputs_sha256"].items() if not Path(k).is_file() or F.sha256(k) != s)
    gate("ADD2 inputs_sha256 (11 files) unchanged (D6)", not ch and len(add2["inputs_sha256"]) == 11, changed=ch)
    inputs.update({k: F.sha256(k) for k in add2["inputs_sha256"] if Path(k).is_file()})   # re-verified later (D6)
    he = str(Path(HE.__file__).resolve())
    gate("hammett_eval.py sha256 = d1pair_common.APPENDIX_SHA (D6)", inputs.get(he) == C.APPENDIX_SHA["hammett_eval.py"],
         now=inputs.get(he))
    # ---- rev 9 pair tables
    e9 = read_json(REV9_E)["inputs_sha256"]
    for p in (C.RES / "S_folds.csv", C.RES / "F_pairs.csv"):
        gate(f"{p.name} sha256 = rev 9 E.json (D6)", inputs[str(p)] == e9.get(str(p)), now=inputs[str(p)], rev9=e9.get(str(p)))
    # ---- FU2 predictions
    mrep = read_json(RES / "M_report.json")
    gate("M_predictions.parquet sha256 = M_report.json", inputs[str(RES / "M_predictions.parquet")] == mrep.get("predictions_sha256"))
    M = pd.read_parquet(RES / "M_predictions.parquet")
    nan = {c: int(M[c].isna().sum()) for c in ("y", "yhat", "yhat_seed_sd")}
    gate("M_predictions rows = 472,440 and no NaN (spec 1)", len(M) == N_M_ROWS and not any(nan.values()), rows=len(M), nan=nan)
    # ---- units: fingerprints (ADD2 1 / 9)
    want = add2["code_sha256_fingerprint_keys"]
    gpu, bad_fp, bad_add2, bad_pq, bad_m1, bad_unit, bad_chosen, pts = [], [], [], [], [], [], [], {}
    for m, v in CONFIGS:
        for k in range(N_FOLDS):
            name = unit_name(m, v, k)
            r = read_json(UNITS / f"{name}.json")
            fp = r["fingerprint"]
            if fp.get("code_sha256") != want:
                bad_fp.append(name)
            if fp.get("add2") != md:
                bad_add2.append(name)
            if inputs[str(UNITS / f"{name}.parquet")] != r.get("parquet_sha256"):
                bad_pq.append(name)
            if not (all(t["gate_M1"]["ok"] for t in r["tune"]) and all(f["gate_M1"]["ok"] for f in r["final"])):
                bad_m1.append(name)
            if r.get("unit") != dict(split="T1", model=m, variant=v, fold=k) or len(r["tune"]) != 4 or len(r["final"]) != N_SEEDS:
                bad_unit.append(name)
            if mrep.get("units", {}).get(name, {}).get("chosen") != r.get("chosen"):
                bad_chosen.append(name)
            pts[name] = sum((UNITS / f"{name}__s{s}.pt").is_file() for s in range(N_SEEDS))
            gpu.append(dict(unit=name, model=m, variant=v, fold=k, gpu=fp.get("gpu"), torch=fp.get("torch"),
                            cuda=fp.get("cuda"), cudnn=fp.get("cudnn")))
    gate("unit fingerprint code_sha256 = ADD2 code_sha256_fingerprint_keys", not bad_fp, units=bad_fp)
    gate("unit fingerprint add2 = ADD2 md sha256", not bad_add2, units=bad_add2)
    gate("unit parquet sha256 = unit json (D6)", not bad_pq, units=bad_pq)
    gate("gate M-1' ok in every fit (D6)", not bad_m1, units=bad_m1)
    gate("unit json identity (unit, 4 tuning cells, 5 finals)", not bad_unit, units=bad_unit)
    gate("M_report chosen = unit json chosen", not bad_chosen, units=bad_chosen)
    # ---- comparators
    cs = add2["comparator_sha256"]
    gate("C-XGB 50 file names = ADD2 comparator_sha256 keys", set(cs) == {p.name for p in cx.values()} and len(cs) == 50)
    badc = sorted(n for n, s in cs.items() if inputs.get(str(M9_T1 / n)) != s)
    gate("C-XGB parquet sha256 = ADD2 comparator_sha256", not badc, files=badc)
    m1c = sorted(p.name for p in comp_js if read_json(p).get("gate_M1") is not True)
    gate("gate M-1 recorded ok in every comparator unit (C-XGB, C-rep, XGBW)", not m1c, files=m1c)
    # ---- test pairs
    S = s_table()
    tests = split_tests()
    srep = read_json(RES / "S_splits.json")
    fold_bad = [k for k, ids in tests.items() if not (S.loc[ids, "fold_T1"].astype(int) == k).all()]
    gate("split-file test folds = S_folds fold_T1 (D6)", not fold_bad, folds=fold_bad)
    n_test = [len(tests[k]) for k in range(N_FOLDS)]
    n_add2 = [r[1] for r in add2["data"]["t1_train_test_before_atom_filter"]]
    n_srep = [srep["units"][f"T1__f{k}.json"]["n_test"] for k in range(N_FOLDS)]
    gate("test pairs per fold = ADD2 data = S_splits.json (D6)", n_test == n_add2 == n_srep, now=n_test, add2=n_add2)
    ref, removed = ref_sets(S, tests)
    rem_n = [sum(1 for k, _ in removed if k == f) for f in range(N_FOLDS)]
    rem_add2 = [add2["data"]["t1_pairs_without_atoms"][f"f{f}"]["test"] for f in range(N_FOLDS)]
    gate("test pairs without atom data per fold = ADD2 data (D6)", rem_n == rem_add2, now=rem_n, add2=rem_add2, pairs=removed)
    raw, _ = raw_frames(S, tests, rep9C)
    flt, n_removed = filter_atoms(raw, S)
    probs = pair_sets_diff(flt, ref, S)
    gate("same (fold, pair_id) test set for the 6 configs, C-XGB, C-rep, XGBW and null per target, after the same "
         "no-atom removal; y = S dd; finite yhat", not probs, problems=probs[:30])
    rk = M.drop_duplicates("pair_id").set_index("pair_id").ref_kind
    gate("M_predictions ref_kind = S_folds ref_kind (D6)", bool((rk == S.loc[rk.index, "ref_kind"]).all()))
    Pm = eval_pairs(S, ref)
    G = pd.DataFrame(gpu)
    write_csv(RES / "E_gpu.csv", G)
    ok = all(g["ok"] for g in gates)
    rec = dict(ok=ok, n_gates=len(gates), n_fail=sum(not g["ok"] for g in gates), gates=gates,
               removed_no_atom_test_pairs=removed, rows_removed_per_model=n_removed,
               n_test_pairs_with_atoms=len(ref), n_P_main_test_pairs=len(Pm), n_P_main_cores=int(Pm.core.nunique()),
               c_rep_models=rep9C, c_rep_note="rev 9 arm C representative (rev 9 rule, chosen among all 6 models incl. SVR; "
                                               "rev 9 Phase E ran after SVR finished)",
               gpu_counts=G.gpu.value_counts().to_dict(), pt_files_per_unit=pts,
               fig3_families=fig3_families(), inputs_sha256=inputs, evaluate_sha256=F.sha256(THIS), env=env_info(),
               git_head=git("rev-parse", "HEAD").stdout.strip(), date=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(RES / "E_check.json", rec)
    flush_nan_log()
    print(json.dumps({k: v for k, v in rec.items() if k not in ("inputs_sha256", "gates", "pt_files_per_unit")},
                     indent=1, default=str))
    if not ok:
        stop(f"E-0: {rec['n_fail']} gate(s) failed: {[g['name'] for g in gates if not g['ok']]}")
    return 0


def fig3_families():
    need(REV9_FIG3)
    return list(dict.fromkeys(pd.read_csv(REV9_FIG3).fam))


# ================================================================ E-1 rep
def val_kcal_table(D):
    """E-1 steps 1-2: chosen cell's per-target inner-validation MAE (standardised, best epoch) x s_y of that fold's
    tuning-training pairs (first inner fold, pairs with atom data; train.Std path). Reads unit jsons only."""
    rows, cache = [], {}
    for m, v in CONFIGS:
        for k in range(N_FOLDS):
            r = unit_json(m, v, k)
            ci = int(r["chosen"]["index"])
            tc = r["tune"][ci]
            vpt = np.asarray(tc["val_per_target"], dtype=float)
            if len(vpt) != len(TARGETS):
                stop(f"E-1: {unit_name(m, v, k)} val_per_target has {len(vpt)} values")
            if abs(float(vpt[F.I_PRIM].mean()) - float(tc["val"])) > 1e-12 or tc["best_epoch"] != r["chosen"]["best_epoch"]:
                stop(f"E-1: {unit_name(m, v, k)} chosen tuning record inconsistent (val / best_epoch)")
            if k not in cache:
                sp = D.split("T1", k)
                cache[k] = (sy_of(D, sp["inner_train"]), len(sp["inner_train"]))
            sy, n_itr = cache[k]
            if n_itr != r["n_inner_train"]:
                stop(f"E-1: {unit_name(m, v, k)} n_inner_train {r['n_inner_train']} != recomputed {n_itr}")
            for j, t in enumerate(TARGETS):
                rows.append(dict(config=f"{m}-{v}", model=m, variant=v, fold=k, target=t, chosen_index=ci,
                                 best_epoch=int(tc["best_epoch"]), val_std=float(vpt[j]), s_y_inner_train=float(sy[j]),
                                 val_kcal=float(vpt[j] * sy[j])))
    V = pd.DataFrame(rows)
    if not np.isfinite(V[["val_std", "s_y_inner_train", "val_kcal"]].to_numpy(float)).all():
        stop("E-1: validation MAE NaN")
    return V


def cmd_rep():
    require_review()
    require_check()
    with no_test_scores(strict=True):
        import train as T
        D = T.Data()
        V = val_kcal_table(D)
        rep = pick_representatives(V)
    mean = {(c, t): float(np.mean(g.sort_values("fold").val_kcal.to_numpy(float))) for (c, t), g in V.groupby(["config", "target"])}
    V["mean_val_kcal_5fold"] = [mean[(c, t)] for c, t in zip(V.config, V.target)]
    V["representative"] = V.config.to_numpy() == V.target.map(rep).to_numpy()
    write_csv(RES / "E_val_kcal.csv", V)
    write_json(RES / "E_representative.json", rep)
    flush_nan_log()
    print(json.dumps(rep, indent=1))
    print(V.pivot_table(index="config", columns="target", values="mean_val_kcal_5fold").round(4).to_string())
    return 0


# ================================================================ E-2 primary + E-7
def cmd_primary(prov=None):
    require_review()
    require_check()
    rep = read_rep()
    S, L, ref = load_all(rep)
    Pm = eval_pairs(S, ref)
    rows = []
    for t in PRIMARY:
        r = compare(L["C-XGB"], L["REP"], Pm, targets=[t]).iloc[0]
        rows.append(dict(target=t, rep=rep[t], n_pairs=int(r.n_pairs), n_cores=int(r.n_cores), mae_CXGB=float(r.mae_x),
                         mae_atom=float(r.mae_b), d=float(r.d), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi), p=float(r.p),
                         folds_atom_lower=int(r.folds_b_lower)))
    Pt = pd.DataFrame(rows)
    Pt["p_holm"] = C.holm(Pt.p.to_numpy())
    Pt["better"] = (Pt.p_holm < C.ALPHA) & (Pt.ci_lo > 0)
    Pt["worse"] = (Pt.p_holm < C.ALPHA) & (Pt.ci_hi < 0)
    Pt = Pt[["target", "rep", "n_pairs", "n_cores", "mae_CXGB", "mae_atom", "d", "ci_lo", "ci_hi", "p", "p_holm", "better",
             "worse", "folds_atom_lower"]]
    write_csv(RES / "E_primary_tests.csv", Pt)
    nb, nw = int(Pt.better.sum()), int(Pt.worse.sum())
    key = "ge5" if nb >= 5 else ("2-4" if nb >= 2 else "le1")
    write_json(RES / "E_primary_verdict.json",
               dict(n_better=nb, better=Pt.target[Pt.better].tolist(), n_worse=nw, worse=Pt.target[Pt.worse].tolist(),
                    verdict_ko=VERDICT[key][0], verdict_en=VERDICT[key][1],
                    rule="ADD2 4: count of targets with Holm p < 0.05 (9 targets) and CI lower bound > 0; >= 5 / 2-4 / <= 1",
                    statistic="d = MAE(C-XGB) - MAE(atom representative), d1_eval.boot_diff 2,000 core resamples, seed 20261019",
                    n_pairs=int(Pt.n_pairs.iloc[0]), n_cores=int(Pt.n_cores.iloc[0])))
    flush_nan_log()
    print(Pt.to_string())
    print(f"verdict: {nb} better, {nw} worse -> {VERDICT[key][0]} ({VERDICT[key][1]})")
    return crosscheck(Pt, rep, prov)


def crosscheck(Pt, rep, prov=None):
    """E-7: official E-1 / E-2 vs the provisional values. prov = optional json path
    {"rep": {target: config}, "primary": {target: {n_pairs, mae_CXGB, mae_atom, d, ci_lo, ci_hi, p, p_holm}},
     "better": [...], "worse": [...]} at full precision (spec tolerances); default = spec 8 table (6 decimals, D4)."""
    if prov:
        P = read_json(prov)
        if not (isinstance(P.get("rep"), dict) and set(TARGETS) <= set(P["rep"]) and isinstance(P.get("primary"), dict)
                and all(t in P["primary"] and set(_PROV_COLS) <= set(P["primary"][t]) for t in PRIMARY)):
            stop(f"--prov {prov}: needs rep (10 targets) and primary (9 targets x {list(_PROV_COLS)})")
        prep, pprim, pb, pw, src, exact = P["rep"], P["primary"], P.get("better", PROV_BETTER), P.get("worse", PROV_WORSE), \
            str(prov), True
    else:
        prep, pprim, pb, pw, src, exact = PROV_REP, PROV_PRIMARY, PROV_BETTER, PROV_WORSE, \
            "spec 8 table (FU2_RESULTS_JUDGMENT.md, reviewer 2026-10-21; 6 decimals)", False
    rows = []
    for t in TARGETS:
        same = rep[t] == prep[t]
        rows.append(dict(target=t, quantity="rep", official=rep[t], provisional=prep[t], abs_diff=0.0 if same else 1.0,
                         tol_spec=0.0, tol_used=0.0, ok=same))
    P2 = Pt.set_index("target")
    for t in PRIMARY:
        for q in _PROV_COLS:
            off, pv = float(P2.at[t, q]), float(pprim[t][q])
            tol = TOL_SPEC[q] if exact else max(TOL_SPEC[q], ROUND6)
            diff = abs(off - pv)
            rows.append(dict(target=t, quantity=q, official=repr(off), provisional=repr(pv), abs_diff=diff,
                             tol_spec=TOL_SPEC[q], tol_used=tol, ok=bool(diff <= tol)))
        for q, lst in (("better", pb), ("worse", pw)):
            o, pv = bool(P2.at[t, q]), t in lst
            rows.append(dict(target=t, quantity=q, official=str(o), provisional=str(pv), abs_diff=float(o != pv),
                             tol_spec=0.0, tol_used=0.0, ok=o == pv))
    X = pd.DataFrame(rows).assign(source=src)
    write_csv(RES / "E_crosscheck.csv", X)
    flush_nan_log()
    bad = X[~X.ok]
    if len(bad):
        print(bad.to_string(), file=sys.stderr)
        stop(f"E-7: {len(bad)} official values differ from the provisional ones beyond the tolerance (code not changed; "
             "both values side by side in E_crosscheck.csv)")
    print(f"E-7: all {len(X)} cross-check rows within tolerance ({src})")
    return 0


# ================================================================ E-3 secondary
def secondary_list(rep, rep9C):
    """(comparison, b, x, set) in the spec 9 order (D5: b = first named)."""
    out = []
    for v in VARIANTS:
        for m1, m2 in (("AW", "GNN"), ("AW", "TF"), ("GNN", "TF")):
            out.append((f"model pair, variant {v}", f"{m1}-{v}", f"{m2}-{v}", "P_main"))
    for m in MODELS3:
        out.append(("variant A+G vs A (what the 182 features add)", f"{m}-A+G", f"{m}-A", "P_main"))
    for m in MODELS3:
        out.append(("A+G vs C-XGB (what the atoms add)", f"{m}-A+G", "C-XGB", "P_main"))
    out += [("para pairs: REP vs C-XGB", "REP", "C-XGB", "P_main_para"),
            ("REP vs C-rep (rev 9 arm C representative, 6 models incl. SVR)", "REP", "C-rep", "P_main"),
            ("REP vs XGBW (FU1)", "REP", "XGBW", "P_main"),
            ("REP vs null (dd = 0)", "REP", "null", "P_main")]
    return out


def cmd_secondary():
    require_review()
    require_check()
    require_e7()
    rep = read_rep()
    rep9C = c_rep_models()
    S, L, ref = load_all(rep)
    sets = {"P_main": eval_pairs(S, ref), "P_main_para": eval_pairs(S, ref, para=True)}

    def lab(n, t):
        return f"REP:{rep[t]}" if n == "REP" else (f"C-rep:{rep9C[t]}" if n == "C-rep" else n)
    rows = []
    for comp, b, x, s in secondary_list(rep, rep9C):
        R = compare(L[x], L[b], sets[s], targets=PRIMARY)
        for r in R.itertuples():
            rows.append(dict(comparison=comp, x=lab(x, r.target), b=lab(b, r.target), target=r.target, set=s,
                             n_pairs=int(r.n_pairs), mae_x=r.mae_x, mae_b=r.mae_b, d=r.d, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p))
    St = pd.DataFrame(rows)
    write_csv(RES / "E_secondary_tests.csv", St)
    flush_nan_log()
    print(St.to_string())
    return 0


# ================================================================ E-4 hammett
def long_frame(L, rep9C):
    return pd.concat([L[n].assign(model=n) for n in MODELS_E4], ignore_index=True)


def cmd_hammett():
    require_review()
    require_check()
    require_e7()
    rep = read_rep()
    rep9C = c_rep_models()
    S, L, ref = load_all(rep)
    add2 = read_json(ADD2_JSON)
    tmae = add2["hammett"]["target_mae_kcal"]
    if tmae != TARGET_MAE_SPEC:
        stop(f"ADD2 json target MAE {tmae} != spec 10 {TARGET_MAE_SPEC}")
    LL = long_frame(L, rep9C)
    Mt = metrics_tab(LL, by=("model", "source"))
    is_pm_all = lambda d: (d["set"] == "P_main") & (d.pos == "all") & d.n_fam.notna()      # noqa: E731
    r2rule = (lambda d: ~is_pm_all(d), "defined on the P_main / all row only (family_eval)")
    write_csv(RES / "E_metrics.csv", Mt, rules={"mae": (lambda d: d.n == 0, "empty set x position"),
                                                "rmse": (lambda d: d.n == 0, "empty set x position"),
                                                "n_fam": r2rule, "r2_fam_median": r2rule, "r2_fam_q1": r2rule,
                                                "r2_fam_q3": r2rule, "rho_err_median": r2rule})
    Bd = pd.concat([bends_tab(L[n], model=n, split="T1") for n in MODELS_E4], ignore_index=True)
    hinge = (always, "no admissible hinge position (hammett_eval.bent_gain needs >= 3 points per side)")
    write_csv(RES / "E_bends.csv", Bd, rules={c: hinge for c in ("psi_dft", "psi_pred", "left_dft", "right_dft",
                                                                  "left_pred", "right_pred")})
    hv = sorted(S.loc[S.hidden_V_fam.astype(bool), "fam"].unique())
    hb = sorted(S.loc[S.hidden_bend_fam.astype(bool), "fam"].unique())
    BR = {n: {"all": E9.bend_rates(Bd[Bd.model == n], "T1"), "hidden_V": E9.bend_rates(Bd[Bd.model == n], "T1", hv),
              "hidden_bend": E9.bend_rates(Bd[Bd.model == n], "T1", hb)} for n in MODELS_E4}
    write_json(RES / "E_bend_rates.json",
               dict(rates=BR, families=dict(hidden_V=hv, hidden_bend=hb, n_hidden_V=len(hv), n_hidden_bend=len(hb),
                                            expected=dict(hidden_V=len(C.T2_HIDDEN_V), hidden_bend=C.T2_N_HIDDEN_BEND)),
                    note="sens / fpr / V_sens = NaN and CI = [NaN, NaN] where the DFT denominator is 0 (by definition); "
                         "clopper = d1_eval.clopper; families = T1 P_main para test families with >= 12 points incl. H",
                    sources={n: (rep if n == "REP" else rep9C if n == "C-rep" else n) for n in MODELS_E4}),
               nan_ok=(lambda q: bool(set(q.split("/")) & RATE_KEYS), "DFT denominator 0 (d1_eval.bend_rates / clopper)"))
    rows = []
    for n in ("REP", "C-XGB"):
        for t in TARGETS:
            a = Mt[(Mt.model == n) & (Mt.target == t) & (Mt["set"] == "P_main") & (Mt.pos == "all")].iloc[0]
            pp = Mt[(Mt.model == n) & (Mt.target == t) & (Mt["set"] == "P_main") & (Mt.pos == "p")].iloc[0]
            br = BR[n]["all"].get(t, {})
            r2 = a.r2_fam_median
            s1 = bool(r2 >= C.STAGE1_R2) if np.isfinite(r2) else None
            s2 = None
            if t in C.STAGE2_TARGETS:
                s2 = bool(pp.mae <= C.STAGE2_MAE and (br.get("sens") or 0) >= C.STAGE2_SENS
                          and (br.get("fpr") if br.get("fpr") == br.get("fpr") else 1) <= C.STAGE2_FPR)
            rows.append(dict(model=n, source=a.source, target=t, r2_fam_median=r2, stage1_descriptive=s1, mae_para=pp.mae,
                             bend_sens=br.get("sens", np.nan), bend_fpr=br.get("fpr", np.nan), stage2_descriptive=s2,
                             note=DESCRIPTIVE))
    Sg = pd.DataFrame(rows)
    write_csv(RES / "E_stages_descriptive.csv", Sg,
              rules={"r2_fam_median": (always, "no para family with >= 12 points"),
                     "stage1_descriptive": (lambda d: d.r2_fam_median.isna(), "R2_fam undefined"),
                     "bend_sens": (always, "c_ghost is not a bend target, or no DFT-bent family (0 denominator)"),
                     "bend_fpr": (always, "c_ghost is not a bend target, or no DFT-straight family (0 denominator)"),
                     "stage2_descriptive": (lambda d: ~d.target.isin(C.STAGE2_TARGETS), "stage 2 defined for oi, cpcm only")})
    rows = []
    for n in MODELS_E4:
        for t, tm in TARGET_MAE_SPEC.items():
            pp = Mt[(Mt.model == n) & (Mt.target == t) & (Mt["set"] == "P_main") & (Mt.pos == "p")].iloc[0]
            rows.append(dict(model=n, source=pp.source, target=t, n_para=int(pp.n), mae_para=pp.mae, target_mae=tm,
                             multiple=pp.mae / tm, note=POST_HOC))
    write_csv(RES / "E_target_multiples.csv", pd.DataFrame(rows))
    flush_nan_log()
    print(Mt[(Mt["set"] == "P_main") & (Mt.pos.isin(["all", "p"]))].pivot_table(index=["model"], columns=["target", "pos"],
                                                                             values="mae").round(3).to_string())
    print(Sg.to_string())
    return 0


# ================================================================ E-5 describe
def pca(X):
    Xc = X - X.mean(0)
    cov = Xc.T @ Xc / (len(X) - 1)
    w, V = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    return w, V, cov


def cmd_describe():
    require_review()
    require_check()
    require_e7()
    rep = read_rep()
    rows = []
    for m, v in CONFIGS:
        for k in range(N_FOLDS):
            r = unit_json(m, v, k)
            for i, tc in enumerate(r["tune"]):
                rows.append(dict(config=f"{m}-{v}", model=m, variant=v, fold=k, cell=i, n_layers=tc["cell"]["n_layers"],
                                 width=tc["cell"]["width"], best_epoch=int(tc["best_epoch"]), epochs_run=int(tc["epochs_run"]),
                                 val=float(tc["val"]), chosen=i == int(r["chosen"]["index"])))
    Ep = pd.DataFrame(rows)
    write_csv(RES / "E_epochs.csv", Ep)
    write_json(RES / "E_epochs_summary.json",
               dict(n_cells=len(Ep), n_best_epoch_gt_270=int((Ep.best_epoch > 270).sum()),
                    n_epochs_run_eq_300=int((Ep.epochs_run == F.MAX_EPOCHS).sum()), max_best_epoch=int(Ep.best_epoch.max()),
                    max_best_epoch_chosen=int(Ep[Ep.chosen].best_epoch.max()), cap=F.MAX_EPOCHS, patience=F.PATIENCE,
                    note="E-5, outside ADD2, post-hoc, not used for any decision; reviewer reference: max 266, 0 cells > 270"))
    S, L, ref = load_all(rep)
    rows = []
    for c in CONFIG_NAMES:
        for t in TARGETS:
            sd = L[c][L[c].target == t].yhat_seed_sd.to_numpy(float)
            rows.append(dict(config=c, target=t, set="T1 test (P_sens | A_main, with atom data)", n=len(sd),
                             median=float(np.median(sd)), q90=float(np.quantile(sd, 0.9)), q99=float(np.quantile(sd, 0.99))))
    write_csv(RES / "E_seed_sd.csv", pd.DataFrame(rows))
    ch = ["elst", "pauli", "oi"]
    R = L["REP"]
    q = R[set_mask(R, "P_main") & R.target.isin(ch)]
    Y = q.pivot(index="pair_id", columns="target", values="y")[ch]
    Eh = q.assign(e=q.yhat - q.y).pivot(index="pair_id", columns="target", values="e")[ch]
    wy, Vy, _ = pca(Y.to_numpy(float))
    we, Ve, ce = pca(Eh.to_numpy(float))
    v1 = Vy[:, 0]
    Bd = bends_tab(R, model="REP", split="T1")
    bp = Bd.pivot(index="fam", columns="target", values="bent_dft")
    pb = bp[bp.pauli.fillna(False).astype(bool)]
    co = int((pb.elst.fillna(False).astype(bool) | pb.oi.fillna(False).astype(bool)).sum())
    ax = dict(set="T1 P_main test pairs with atom data", n_pairs=len(Y), channels=ch,
              rep_configs={c: rep[c] for c in ch},
              truth=dict(loadings=Vy.T.tolist(), variance_ratio=(wy / wy.sum()).tolist()),
              rep_error=dict(variance_share_on_truth_pc1=float(v1 @ ce @ v1 / np.trace(ce)), own_loadings=Ve.T.tolist(),
                             own_variance_ratio=(we / we.sum()).tolist(),
                             angle_own_pc1_vs_truth_pc1_deg=float(np.degrees(np.arccos(min(1.0, abs(float(Ve[:, 0] @ v1))))))),
              corr_pauli_vs_elst_plus_oi=dict(truth=float(np.corrcoef(Y.pauli, Y.elst + Y.oi)[0, 1]),
                                              rep_error=float(np.corrcoef(Eh.pauli, Eh.elst + Eh.oi)[0, 1])),
              dft_bends=dict(n_families=int(len(bp)), n_pauli_bent=int(len(pb)), n_pauli_bent_with_elst_or_oi_bent=co,
                             n_elst_bent=int(bp.elst.fillna(False).astype(bool).sum()),
                             n_oi_bent=int(bp.oi.fillna(False).astype(bool).sum())),
              definitions="PCA of the centred covariance (kcal/mol, unscaled), loadings rows = PCs, sign: largest |loading| "
                          "> 0; variance_share_on_truth_pc1 = v1' Cov(err) v1 / tr Cov(err); Pearson correlations; bends = "
                          "hammett_eval.calls on DFT values (bends_tab, T1 P_main para test families)",
              note="E-5, outside ADD2, post-hoc, not used for any decision; reviewer reference: PC1 share truth 0.979, "
                   "error 0.965; pauli bent 22 families, 21 with elst or oi bent")
    write_json(RES / "E_axis.json", ax)
    flush_nan_log()
    print(json.dumps(ax, indent=1)[:3000])
    return 0


# ================================================================ figures
COL_SLOT = {"AW": 0, "GNN": 1, "C-XGB": 2, "XGBW": 3, "C-rep": 4, "TF": 5, "REP": 7}
CAPTIONS = {
    "figF2_1_mae": "Relative test-pair ΔΔ MAE (MAE / MAE of null ΔΔ = 0; T1, P_main reference pairs with atom data, 5 outer "
                   "folds pooled): top all positions, bottom para pairs. C-XGB = rev 9 arm C XGB, XGBW = FU1 wide-grid XGB, "
                   "C-rep = rev 9 arm C representative (6 models incl. SVR), AW / GNN / TF = atom-token models (lighter = "
                   "variant A, atoms only; solid = A+G, atoms + the 182-feature global token). Gray line = null.",
    "figF2_2_hammett_level": "Hammett level, atom representative (REP: per target the config chosen on the inner-validation "
                             "MAE) vs C-XGB, T1 para test families (>= 12 points incl. H): left median R²_fam (drawn at −1 "
                             "when below), middle bend-call sensitivity, right false-positive rate, Clopper–Pearson 95 % "
                             "intervals; gray lines = rev 9 stage thresholds (0.5, 0.7, 0.1), descriptive only.",
    "figF2_3_examples": "Hammett plots (y = −ΔΔ/1.3642, log k units, H at the origin) of three of the six rev 9 fig3 "
                        "families (the first three in that order with a para test series for barrier, OI and elst): DFT "
                        "points, C-XGB and atom-representative predictions (T1 test pairs).",
    "figF2_4_atoms": "Exploratory (E-6). Rows: the rev 9 fig3 families, one pair each (largest |σp|). Columns 1–6: AW-A+G "
                     "per-atom contributions ε (kcal/mol; mean of the 5 final models of the fold in which the pair is test; "
                     "H contributions added to their heavy atom) for OI, elst, Pauli on a (X-substituted) and b (reference). "
                     "Columns 7–8: TF-A+G cross-attention received per atom (head and seed mean; on b from a's atoms, on a "
                     "from b's atoms). Ring = reacting atom, diamond = X-group atom, † = fragment whose heavy-atom RMSD "
                     "between relaxed reactant and TS fragment exceeds 1 Å (ADD2 'known'). Attention and contributions are "
                     "diagnostics, not explanations (attention·기여는 설명이 아니라 진단).",
    "figF2_5_epochs": "Best epoch of the 120 tuning cells (6 configs × 5 outer folds × 4 grid cells); filled = chosen cell; "
                      "solid line = epoch cap 300, dashed = 270 (E-5, descriptive).",
}


def color_of(DF, name):
    base = name.split("-")[0] if name in CONFIG_NAMES else name
    return DF.MUTED if base == "null" else DF.SLOT[COL_SLOT[base]]


def save_fig(fig, name, df):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        F.write_atomic(FIG / f"{name}.{ext}", lambda t, e=ext: fig.savefig(t, format=e, bbox_inches="tight"))
    F.write_atomic(FIG / f"{name}.csv", lambda t: df.to_csv(t, index=False))
    WRITTEN.extend(rel(FIG / f"{name}.{e}") for e in ("pdf", "png", "csv"))       # D14 (interpreter recorded)


def write_captions():
    md = ["# D1PAIR FU2 figures — captions", ""] + [f"- **{k}**: {v}" for k, v in CAPTIONS.items()
                                                    if (FIG / f"{k}.png").is_file()]
    F.write_atomic(FIG / "CAPTIONS_FU2.md", "\n".join(md) + "\n")
    WRITTEN.append(rel(FIG / "CAPTIONS_FU2.md"))


def plot_style():
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        stop(f"matplotlib is not importable in this environment ({e}; T-1): run figures / explain in an env that has it "
             "(user decision)")
    import d1_figures as DF
    return DF.style(), DF


def figF2_1(plt, DF, Mt):
    models = ["C-XGB", "XGBW", "C-rep"] + CONFIG_NAMES
    rows = []
    for pos in ("all", "p"):
        for t in TARGETS:
            nl = Mt[(Mt.model == "null") & (Mt["set"] == "P_main") & (Mt.pos == pos) & (Mt.target == t)].mae.iloc[0]
            for m in models:
                q = Mt[(Mt.model == m) & (Mt["set"] == "P_main") & (Mt.pos == pos) & (Mt.target == t)].iloc[0]
                rows.append(dict(pos=pos, target=t, model=m, source=q.source, n=int(q.n), mae=q.mae, mae_null=nl,
                                 rel_mae=q.mae / nl))
    d = pd.DataFrame(rows)
    fig, axs = plt.subplots(2, 1, figsize=(C.FIG_W, C.FIG_W * 0.55), constrained_layout=True)
    w = 0.84 / len(models)
    x = np.arange(len(TARGETS))
    for ax, pos in zip(axs, ("all", "p")):
        q = d[d.pos == pos]
        for i, m in enumerate(models):
            v = [q[(q.model == m) & (q.target == t)].rel_mae.iloc[0] for t in TARGETS]
            ax.bar(x + (i - (len(models) - 1) / 2) * w, v, width=w * 0.88, color=color_of(DF, m),
                   alpha=0.5 if m.endswith("-A") else 1.0, label=m, linewidth=0)
        ax.axhline(1.0, color=DF.MUTED, lw=0.6)
        ax.set_xticks(x, [DF.TLAB[t] for t in TARGETS])
        ax.set_ylabel("MAE / MAE(null)")
        ax.set_title("P_main, all positions" if pos == "all" else "P_main, para pairs")
        ax.grid(axis="x", visible=False)
    axs[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_fig(fig, "figF2_1_mae", d)
    plt.close(fig)


def figF2_2(plt, DF, Mt, BR):
    rows = []
    for m in ("REP", "C-XGB"):
        for t in CH9:
            q = Mt[(Mt.model == m) & (Mt["set"] == "P_main") & (Mt.pos == "all") & (Mt.target == t)].iloc[0]
            br = BR[m]["all"][t]
            rows.append(dict(model=m, target=t, source=q.source, r2_fam_median=q.r2_fam_median, sens=br["sens"],
                             sens_lo=br["sens_ci"][0], sens_hi=br["sens_ci"][1], n_bent=br["n_bent_dft"], fpr=br["fpr"],
                             fpr_lo=br["fpr_ci"][0], fpr_hi=br["fpr_ci"][1], n_straight=br["n_straight_dft"]))
    d = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 3, figsize=(C.FIG_W, C.FIG_W * 0.3), constrained_layout=True)
    x = np.arange(len(CH9))
    for k, m in enumerate(("REP", "C-XGB")):
        q = d[d.model == m].set_index("target").loc[CH9]
        off = (k - 0.5) * 0.36
        col = color_of(DF, m)
        axs[0].bar(x + off, q.r2_fam_median.clip(lower=-1), width=0.32, color=col, label=m, linewidth=0)
        for ax, key in ((axs[1], "sens"), (axs[2], "fpr")):
            v = q[key].to_numpy(float)
            err = np.vstack([v - q[f"{key}_lo"].to_numpy(float), q[f"{key}_hi"].to_numpy(float) - v])
            ax.bar(x + off, v, width=0.32, color=col, linewidth=0, label=m)
            ax.errorbar(x + off, v, yerr=err, fmt="none", ecolor=DF.INK2, elinewidth=0.6, capsize=1.5)
    for ax, lab, ref in ((axs[0], "median R²_fam", C.STAGE1_R2), (axs[1], "bend-call sensitivity", C.STAGE2_SENS),
                         (axs[2], "false-positive rate", C.STAGE2_FPR)):
        ax.axhline(ref, color=DF.MUTED, lw=0.6)
        ax.set_xticks(x, [DF.TLAB[t] for t in CH9], rotation=45, ha="right")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylim(-1.02, 1.02)
    axs[1].set_ylim(0, 1.05)
    axs[2].set_ylim(0, 1.05)
    axs[0].legend(loc="lower left")
    save_fig(fig, "figF2_2_hammett_level", d)
    plt.close(fig)


def hseries(q, col):
    return HE.family_series(E9.hammett_frame(q, col), q.fam.iloc[0], col, C.MIN_X_FAM)


def figF2_3(plt, DF, L, rep):
    tg = ["barrier", "oi", "elst"]

    def para(n, f, t):
        d = L[n]
        return d[set_mask(d, "P_main") & (d.pos == "p") & d.sigma.notna() & (d.fam == f) & (d.target == t)]
    fams = [f for f in fig3_families() if all(len(para("REP", f, t)) and hseries(para("REP", f, t), "y") is not None
                                               for t in tg)][:3]
    rows = []
    fig, axs = plt.subplots(len(tg), max(1, len(fams)), figsize=(C.FIG_W, C.FIG_W * 0.6), constrained_layout=True,
                            squeeze=False)
    for j, f in enumerate(fams):
        for i, t in enumerate(tg):
            ax = axs[i, j]
            x, y, nm = hseries(para("REP", f, t), "y")
            _, yr, nr = hseries(para("REP", f, t), "pred")
            sc = hseries(para("C-XGB", f, t), "pred")
            cmap_ = dict(zip(sc[2], sc[1])) if sc is not None else {}
            if nr != nm or set(cmap_) != set(nm):
                stop(f"figF2_3: points differ between REP and C-XGB for {f} / {t}")
            yc = np.array([cmap_[n] for n in nm])                    # C-XGB values in the REP point order
            o = np.argsort(x)
            ax.scatter(x, y, s=10, color=DF.INK, linewidths=0, label="DFT", zorder=3)
            ax.plot(x[o], yc[o], color=color_of(DF, "C-XGB"), lw=1.0, marker="o", ms=2.2, label="C-XGB", zorder=2)
            ax.plot(x[o], yr[o], color=color_of(DF, "REP"), lw=1.0, marker="o", ms=2.2, label="atom REP", zorder=2)
            rows += [dict(fam=f, target=t, rep_config=rep[t], X=n, sigma_p=a, y_dft=b, y_cxgb=c, y_rep=e)
                     for n, a, b, c, e in zip(nm, x, y, yc, yr)]
            if i == 0:
                ax.set_title(f, fontsize=C.FONT_PT - 1)
            if j == 0:
                ax.set_ylabel(f"{DF.TLAB[t]}: −ΔΔ/1.3642")
            if i == len(tg) - 1:
                ax.set_xlabel("σp")
            ax.text(0.02, 0.98, f"REP {rep[t]}", transform=ax.transAxes, ha="left", va="top", fontsize=C.FONT_PT - 2,
                    color=DF.INK2)
    axs[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_fig(fig, "figF2_3_examples", pd.DataFrame(rows))
    plt.close(fig)
    return fams


def figF2_5(plt, DF, Ep):
    fig, ax = plt.subplots(figsize=(C.FIG_W * 0.6, C.FIG_W * 0.3), constrained_layout=True)
    for i, c in enumerate(CONFIG_NAMES):
        q = Ep[Ep.config == c]
        xx = i + (q.fold.to_numpy() - 2) * 0.1 + (q.cell.to_numpy() - 1.5) * 0.02
        col = color_of(DF, c)
        ch = q.chosen.to_numpy(bool)
        ax.scatter(xx[~ch], q.best_epoch[~ch], s=14, facecolors="none", edgecolors=col, linewidths=0.6)
        ax.scatter(xx[ch], q.best_epoch[ch], s=14, color=col, linewidths=0)
    ax.axhline(F.MAX_EPOCHS, color=DF.MUTED, lw=0.6)
    ax.axhline(270, color=DF.MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(len(CONFIG_NAMES)), CONFIG_NAMES)
    ax.set_ylabel("best epoch")
    ax.set_ylim(0, F.MAX_EPOCHS + 10)
    ax.grid(axis="x", visible=False)
    save_fig(fig, "figF2_5_epochs", Ep)
    plt.close(fig)


def cmd_figures():
    require_review(same_env=False)
    require_check()
    require_e7()
    plt, DF = plot_style()
    rep = read_rep()
    for f in ("E_metrics.csv", "E_bend_rates.json", "E_epochs.csv"):
        need(RES / f)
    require_fresh("E_metrics.csv", "E_bend_rates.json", "E_epochs.csv")          # D14: from this run, not stale
    Mt = pd.read_csv(RES / "E_metrics.csv", keep_default_na=False, na_values=[""])   # model 'null' must stay a string
    BR = read_json(RES / "E_bend_rates.json")["rates"]
    Ep = pd.read_csv(RES / "E_epochs.csv")
    figF2_1(plt, DF, Mt)
    figF2_2(plt, DF, Mt, BR)
    S, L, ref = load_all(rep)
    fams = figF2_3(plt, DF, L, rep)
    figF2_5(plt, DF, Ep)
    write_captions()
    flush_nan_log()
    print(f"figures written to {FIG} (figF2_3 families: {fams})")
    return 0


# ================================================================ E-6 explain (exploratory)
def rxn_info(D, row):
    """per-atom TS data of one token reaction: n, Z, frag, roles {role: ts index}, owner (H -> nearest heavy atom of the
    same fragment; heavy -> itself)."""
    n = int(D.n[row])
    Z = np.asarray(D.elements)[D.onehot[row, :n].argmax(1)]
    frag = D.frag[row, :n].astype(int)
    pos = D.pos[row, :n]
    roles = {}
    for r in ROLES:
        idx = np.where(D.bin[row, :n, D.bin_cols.index(r)] == 1)[0]
        if len(idx) != 1:
            return None
        roles[r] = int(idx[0])
    heavy = np.where(Z > 1)[0]
    owner = np.arange(n)
    for i in np.where(Z == 1)[0]:
        cand = heavy[frag[heavy] == frag[i]]
        if not len(cand):
            return None
        owner[i] = int(cand[np.argmin(np.linalg.norm(pos[cand] - pos[i], axis=1))])
    return dict(n=n, Z=Z, frag=frag, roles=roles, owner=owner)


def react_mol(smiles, info):
    """RDKit reactant-side molecule (heavy atoms, atom map = TS index + 1) with 2D coordinates; (mol, ts index per mol
    atom) or (None, reason). Maps and elements are verified against the tokens."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    if not isinstance(smiles, str) or ">>" not in smiles:
        return None, "no reaction SMILES"
    mol = Chem.MolFromSmiles(smiles.split(">>")[0])
    if mol is None:
        return None, "SMILES parse failed"
    ts = [a.GetAtomMapNum() - 1 for a in mol.GetAtoms()]
    heavy = set(np.where(info["Z"] > 1)[0].tolist())
    if len(set(ts)) != len(ts) or set(ts) != heavy:
        return None, "atom maps != TS heavy atoms"
    if any(int(info["Z"][t]) != a.GetAtomicNum() for t, a in zip(ts, mol.GetAtoms())):
        return None, "element mismatch between SMILES map and tokens"
    rdDepictor.Compute2DCoords(mol)
    return mol, ts


def x_group(ma, ts_a, ia, mb, ts_b, ib):
    """D12 / X_ATOM_METHOD: X-group TS indices of reaction a (heavy + owned H) or None."""
    ida = {t: i for i, t in enumerate(ts_a)}
    idb = {t: i for i, t in enumerate(ts_b)}
    need_ = [(idb[ib["roles"][r]], ida[ia["roles"][r]]) for r in ROLES]
    for match in ma.GetSubstructMatches(mb, uniquify=False, useChirality=False, maxMatches=100000):
        if all(match[qb] == ta for qb, ta in need_):
            un = sorted(set(range(ma.GetNumAtoms())) - set(match))
            xh = sorted(ts_a[i] for i in un)
            if not xh:
                return None
            xs = set(xh)
            return dict(heavy=xh, atoms=sorted(i for i in range(ia["n"]) if int(ia["owner"][i]) in xs))
    return None


def load_final(T, Mo, D, m, v, k, s):
    """(model eval CPU, std (train.Std with the saved statistics), tensors) of final seed s of unit T1 m v fold k."""
    import torch
    pt = UNITS / f"{unit_name(m, v, k)}__s{s}.pt"
    need(pt)
    obj = torch.load(pt, map_location="cpu")
    if obj["unit"] != dict(split="T1", model=m, variant=v, fold=k):
        stop(f"{pt.name}: unit record differs")
    sy_js = np.asarray(unit_json(m, v, k)["final"][s]["sy"], dtype=float)
    if not np.allclose(np.asarray(obj["std"]["sy"], dtype=float), sy_js, rtol=0.0, atol=1e-12):
        stop(f"{pt.name}: std.sy differs from the unit json final[{s}].sy (D15)")
    model = Mo.build(m, obj["cell"], D.d_in, len(F.TARGETS), D.d_glob, v)
    model.load_state_dict(obj["state"])
    model.eval()
    std = T.Std.__new__(T.Std)
    for key in ("mu", "sd", "gmu", "gsd", "sy"):
        setattr(std, key, np.asarray(obj["std"][key], dtype=float))
    return model, std, std.tensors(D, dtype=torch.float32, device=torch.device("cpu"))


def cmd_explain():
    require_review(same_env=False)
    require_check()
    require_e7()
    plt, DF = plot_style()
    try:
        import rdkit  # noqa: F401
    except ImportError as e:
        stop(f"RDKit is not importable in this environment ({e}; T-1): explain needs it for the 2D drawings")
    import torch
    import train as T
    import models as Mo
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    rep = read_rep()
    D = T.Data()
    S = s_table()
    tests = split_tests()
    ref, _ = ref_sets(S, tests)
    fams = fig3_families()
    smi = pd.read_csv(SMILES).set_index("rxn_id").rxn_smiles
    Q = pd.read_csv(RES / "Q_manifest.csv").set_index("rxn_id")
    ids = sorted({pid for _, pid in ref})
    P = S.loc[ids]
    P = P[P.in_P_main.astype(bool) & (P.ref_kind == "reference") & (P.pos == "p") & P.sigma.notna() & P.fam.isin(fams)]
    off = len(D.elements) + len(D.bin_cols)
    jz = [D.num_cols.index(c) for c in ZERO_COLS]

    def rmsd_flag(rid):
        r = Q.loc[int(rid)]
        return dict(A=bool(r.qc_rmsd_A_v2 > 1.0), B=bool(r.qc_rmsd_B_v2 > 1.0))

    # ---- pairs, reactions, X groups
    pinfo, skipped = {}, []
    for pid, r in P.iterrows():
        row = D.pid_row[pid]
        ra, rb = int(D.ra[row]), int(D.rb[row])
        ia, ib = rxn_info(D, ra), rxn_info(D, rb)
        if ia is None or ib is None:
            skipped.append(dict(pair_id=pid, reason="reacting-atom flags / H owners not resolvable")); continue
        ma, tsa = react_mol(smi.get(int(r.a_rxn)), ia)
        mb, tsb = react_mol(smi.get(int(r.b_rxn)), ib)
        if ma is None or mb is None:
            skipped.append(dict(pair_id=pid, reason=f"a: {tsa if ma is None else 'ok'}; b: {tsb if mb is None else 'ok'}"))
            continue
        xg = x_group(ma, tsa, ia, mb, tsb, ib)
        if xg is None:
            skipped.append(dict(pair_id=pid, reason="no role-preserving substructure match of b in a")); continue
        pinfo[pid] = dict(row=row, ra=ra, rb=rb, ia=ia, ib=ib, ma=ma, mb=mb, tsa=tsa, tsb=tsb, xg=xg, fam=r.fam, Xb=r.Xb,
                          sigma=float(r.sigma), fold=int(r.fold_T1), a_rxn=int(r.a_rxn), b_rxn=int(r.b_rxn),
                          rmsd_a=rmsd_flag(r.a_rxn), rmsd_b=rmsd_flag(r.b_rxn))
    sel = {}
    for f in fams:
        c = [(-abs(v["sigma"]), v["Xb"], pid) for pid, v in pinfo.items() if v["fam"] == f]
        if c:
            sel[f] = sorted(c)[0][2]
    # ---- model passes
    Mp = pd.read_parquet(RES / "M_predictions.parquet")
    Mp = Mp[Mp.pair_id.isin(list(pinfo))]
    preds, maps, recon, decomp = {}, [], {}, []
    for m, v in CONFIGS:
        cfg = f"{m}-{v}"
        for k in sorted({p["fold"] for p in pinfo.values()}):
            pk = [pid for pid, p in pinfo.items() if p["fold"] == k]
            for s in range(N_SEEDS):
                model, std, Tt = load_final(T, Mo, D, m, v, k, s)
                z0 = torch.tensor([(0.0 - std.mu[j]) / std.sd[j] for j in jz], dtype=torch.float32)
                for pid in pk:
                    p = pinfo[pid]
                    R, ia_t, ib_t = T.batch(Tt, np.array([p["ra"]]), np.array([p["rb"]]))
                    ii, jj = int(ia_t[0]), int(ib_t[0])

                    def run(atoms):
                        R2 = dict(R)
                        if atoms:
                            x = R["x"].clone()
                            for a in atoms:
                                x[ii, a, [off + j for j in jz]] = z0
                            R2["x"] = x
                        with torch.no_grad():
                            return model(R2, ia_t, ib_t)[0].double().numpy() * std.sy
                    rng = np.random.default_rng(C.SEED + int(p["row"]))
                    xa = p["xg"]["atoms"]
                    elig = [i for i in range(p["ia"]["n"]) if i not in set(xa) and i not in set(p["ia"]["roles"].values())]
                    ctrl = [sorted(rng.choice(elig, size=len(xa), replace=False).tolist()) for _ in range(N_CTRL)] \
                        if len(elig) >= len(xa) else []
                    rec = preds.setdefault((cfg, pid), dict(base=[], xz=[], ctrl=[]))
                    base = run([])
                    rec["base"].append(base)
                    rec["xz"].append(run(xa))
                    rec["ctrl"].append([run(c) for c in ctrl])
                    if sel.get(p["fam"]) != pid:
                        continue
                    if cfg == "AW-A+G":
                        with torch.no_grad():
                            eps = model.atom_contrib(R).double().numpy() * std.sy
                            phi = model.phi(R["glob"]).double().numpy() * std.sy
                        na, nb = p["ia"]["n"], p["ib"]["n"]
                        decomp.append(float(np.abs(eps[ii, :na].sum(0) + phi[ii] - eps[jj, :nb].sum(0) - phi[jj] - base).max()))
                        for role, idx, n_ in (("a", ii, na), ("b", jj, nb)):
                            for a in range(n_):
                                for c, t in enumerate(TARGETS):
                                    maps.append(dict(analysis="aw_eps", config=cfg, pair_id=pid, role=role, atom=a, seed=s,
                                                     channel=t, value=float(eps[idx, a, c])))
                            for c, t in enumerate(TARGETS):
                                maps.append(dict(analysis="aw_phi", config=cfg, pair_id=pid, role=role, atom=-1, seed=s,
                                                 channel=t, value=float(phi[idx, c])))
                    if cfg == "TF-A+G":
                        store = []

                        def hook(mod, args, out):
                            with torch.no_grad():
                                store.append(mod.forward(*args, return_attn=True)[1].double().numpy())
                        h = model.cross.att.register_forward_hook(hook)
                        with torch.no_grad():
                            model(R, ia_t, ib_t)
                        h.remove()
                        if len(store) != 2:
                            stop(f"explain: expected 2 cross-attention calls, got {len(store)}")
                        na, nb = p["ia"]["n"], p["ib"]["n"]
                        nt = store[0].shape[-1]
                        for role, A, nq, nk in (("b", store[0][0].mean(0), na, nb), ("a", store[1][0].mean(0), nb, na)):
                            recv = A[:nq].mean(0)                                  # received by each key token
                            for a in range(nk):
                                maps.append(dict(analysis="tf_attn_recv", config=cfg, pair_id=pid, role=role, atom=a,
                                                 seed=s, channel="attention", value=float(recv[a])))
                            maps.append(dict(analysis="tf_attn_recv_global", config=cfg, pair_id=pid, role=role, atom=-1,
                                             seed=s, channel="attention", value=float(recv[nt - 1])))
            for pid in pk:
                b = np.mean(preds[(cfg, pid)]["base"], 0)
                mm = Mp[(Mp.model == m) & (Mp.variant == v) & (Mp.pair_id == pid)].set_index("target").yhat
                dv = np.abs(b - mm.loc[TARGETS].to_numpy(float))
                recon[cfg] = max(recon.get(cfg, 0.0), float(dv.max()) if np.isfinite(dv).all() else np.inf)
    bad_rc = {c: v for c, v in recon.items() if not v <= RECON_TOL}
    if bad_rc:                                                   # D15: wrong weights / standardisation
        stop(f"explain: CPU reproduction of M_predictions differs by more than {RECON_TOL} kcal/mol: {bad_rc}")
    # ---- zero-out table
    zrows = []
    for (cfg, pid), rec in preds.items():
        p = pinfo[pid]
        b, xz = np.mean(rec["base"], 0), np.mean(rec["xz"], 0)
        nct = len(rec["ctrl"][0])
        cm = [np.mean([rec["ctrl"][s][d] for s in range(N_SEEDS)], 0) for d in range(nct)]
        for c, t in enumerate(TARGETS):
            zrows.append(dict(config=cfg, is_rep=rep[t] == cfg, fam=p["fam"], pair_id=pid, Xb=p["Xb"], sigma=p["sigma"],
                              fold=p["fold"], channel=t, n_X_atoms=len(p["xg"]["atoms"]), n_X_heavy=len(p["xg"]["heavy"]),
                              abs_dy_X=float(abs(xz[c] - b[c])),
                              abs_dy_ctrl=float(np.mean([abs(q[c] - b[c]) for q in cm])) if nct else np.nan,
                              n_ctrl_draws=nct, a_rmsd_gt1=p["rmsd_a"]["A"] or p["rmsd_a"]["B"],
                              b_rmsd_gt1=p["rmsd_b"]["A"] or p["rmsd_b"]["B"]))
    Z = pd.DataFrame(zrows)
    write_csv(RES / "E_explain_zero.csv", Z, rules={"abs_dy_ctrl": (lambda d: d.n_ctrl_draws == 0,
                                                                    "fewer eligible control atoms than X atoms")})
    # ---- map table (seed mean, atom annotations)
    Mx = pd.DataFrame(maps)
    G = Mx.groupby(["analysis", "config", "pair_id", "role", "atom", "channel"], sort=False).value
    E = G.mean().reset_index().merge(G.std(ddof=0).rename("value_sd_seed").reset_index(),
                                     on=["analysis", "config", "pair_id", "role", "atom", "channel"])
    ext = []
    for r in E.itertuples():
        p = pinfo[r.pair_id]
        inf = p["ia"] if r.role == "a" else p["ib"]
        glob = r.atom < 0
        ext.append(dict(fam=p["fam"], Xb=p["Xb"], fold=p["fold"], rxn_id=p["a_rxn"] if r.role == "a" else p["b_rxn"],
                        Z=0 if glob else int(inf["Z"][r.atom]), owner=-1 if glob else int(inf["owner"][r.atom]),
                        frag="global" if glob else ("B" if inf["frag"][r.atom] else "A"),
                        is_react=(not glob) and r.atom in set(inf["roles"].values()),
                        is_X=(not glob) and r.role == "a" and r.atom in set(p["xg"]["atoms"]),
                        frag_rmsd_gt1=False if glob else (p["rmsd_a"] if r.role == "a" else p["rmsd_b"])[
                            "B" if inf["frag"][r.atom] else "A"]))
    E = pd.concat([E, pd.DataFrame(ext)], axis=1)
    write_csv(RES / "E_explain.csv", E)
    # ---- figure
    fig_rows = figF2_4(plt, DF, E, pinfo, sel, fams)
    zs = Z.groupby(["config", "channel"]).agg(n=("pair_id", "size"), mean_abs_dy_X=("abs_dy_X", "mean"),
                                              mean_abs_dy_ctrl=("abs_dy_ctrl", "mean")).reset_index()
    write_json(RES / "E_explain.json",
               dict(note="E-6 exploratory; attention and contributions are diagnostics, not explanations "
                         "(attention·기여는 설명이 아니라 진단)",
                    families=fams, families_match_d1pair_common=sorted(fams) == sorted(C.FIG3_FAMILIES),
                    selected_pairs=sel, n_pairs_candidate=len(P), n_pairs_used=len(pinfo), skipped=skipped,
                    x_atom_method=X_ATOM_METHOD, zero_cols=list(ZERO_COLS), n_ctrl_draws=N_CTRL,
                    x_groups={pid: dict(Xb=p["Xb"], heavy=p["xg"]["heavy"], atoms=p["xg"]["atoms"]) for pid, p in pinfo.items()},
                    reproduction_max_abs_diff_vs_M_predictions=recon,
                    aw_decomposition_max_abs_diff=max(decomp) if decomp else None,
                    rmsd_gt1_reactions={pid: dict(a=p["rmsd_a"], b=p["rmsd_b"]) for pid, p in pinfo.items()
                                        if any(p["rmsd_a"].values()) or any(p["rmsd_b"].values())},
                    zero_test_summary=zs.to_dict("records"), figure_panels=fig_rows, env=env_info()),
               nan_ok=(lambda q: q.endswith("/mean_abs_dy_ctrl"),
                       "every pair of the group has fewer eligible control atoms than X atoms"))
    write_captions()
    flush_nan_log()
    print(zs.to_string())
    return 0


def draw_mol(ax, DF, mol, ts, vals, cmap, norm, react, xs):
    conf = mol.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    for bd in mol.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=DF.AXIS, lw=0.5 + 0.35 * (bd.GetBondTypeAsDouble() - 1), zorder=1,
                solid_capstyle="round")
    ax.scatter(xy[:, 0], xy[:, 1], c=[vals.get(t, 0.0) for t in ts], cmap=cmap, norm=norm, s=16, linewidths=0, zorder=2)
    r = [i for i, t in enumerate(ts) if t in react]
    ax.scatter(xy[r, 0], xy[r, 1], s=36, facecolors="none", edgecolors=DF.INK, linewidths=0.5, zorder=3)
    x = [i for i, t in enumerate(ts) if t in xs]
    if x:
        ax.scatter(xy[x, 0], xy[x, 1], s=30, marker="D", facecolors="none", edgecolors=DF.INK2, linewidths=0.5, zorder=3)
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetSymbol() != "C":
            ax.text(xy[i, 0], xy[i, 1] - 0.5, a.GetSymbol(), fontsize=C.FONT_PT - 3, ha="center", va="top", color=DF.INK2)
    ax.set_aspect("equal")
    ax.axis("off")


def figF2_4(plt, DF, E, pinfo, sel, fams):
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    div = LinearSegmentedColormap.from_list("fu2_div", [DF.SLOT[0], "#f0efec", DF.SLOT[7]])
    seq = LinearSegmentedColormap.from_list("fu2_seq", ["#f0efec", DF.SLOT[0], "#0d366b"])
    cols = [("aw_eps", ch, role) for ch in MAP_CHANNELS for role in ("a", "b")] + [("tf_attn_recv", "attention", r)
                                                                                 for r in ("b", "a")]
    heads = [f"AW ε {DF.TLAB[ch]} ({role})" for ch in MAP_CHANNELS for role in ("a", "b")] + \
            ["TF attention on b (from a)", "TF attention on a (from b)"]
    rows_f = [f for f in fams if f in sel]

    def heavy_vals(an, ch, pid, role):
        p = pinfo[pid]
        inf = p["ia"] if role == "a" else p["ib"]
        q = E[(E.analysis == an) & (E.channel == ch) & (E.pair_id == pid) & (E.role == role) & (E.atom >= 0)]
        out = {}
        for a, v in zip(q.atom, q.value):
            o = int(inf["owner"][a])
            out[o] = out.get(o, 0.0) + float(v)
        return out
    vmax = {}
    for j, (an, ch, role) in enumerate(cols):
        vv = [abs(x) for f in rows_f for x in heavy_vals(an, ch, sel[f], role).values()]
        vmax[j] = max(vv) if vv else 1.0
    for an, ch, _ in cols:                                       # one scale per channel (a and b share it)
        js = [j for j, c in enumerate(cols) if c[0] == an and c[1] == ch]
        top = max(vmax[j] for j in js)
        for j in js:
            vmax[j] = top
    fig, axs = plt.subplots(max(1, len(rows_f)), len(cols), figsize=(C.FIG_W, C.FIG_W * 0.16 * max(1, len(rows_f)) + 0.8),
                            constrained_layout=True, squeeze=False)
    panel = []
    for i, f in enumerate(rows_f):
        pid = sel[f]
        p = pinfo[pid]
        for j, (an, ch, role) in enumerate(cols):
            ax = axs[i, j]
            mol, ts = (p["ma"], p["tsa"]) if role == "a" else (p["mb"], p["tsb"])
            inf = p["ia"] if role == "a" else p["ib"]
            vals = heavy_vals(an, ch, pid, role)
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax[j], vmax=vmax[j]) if an == "aw_eps" else Normalize(0.0, vmax[j])
            draw_mol(ax, DF, mol, ts, vals, div if an == "aw_eps" else seq, norm, set(inf["roles"].values()),
                     set(p["xg"]["heavy"]) if role == "a" else set())
            fl = p["rmsd_a"] if role == "a" else p["rmsd_b"]
            dag = "".join(f"†{k}" for k in ("A", "B") if fl[k])
            if j == 0:
                ax.set_title(f"{f}\nX = {p['Xb']} ({pid}) {dag}", fontsize=C.FONT_PT - 3, loc="left")
            elif dag:
                ax.set_title(dag, fontsize=C.FONT_PT - 3, loc="left")
            panel.append(dict(row=i, col=j, fam=f, pair_id=pid, analysis=an, channel=ch, role=role, vmax=vmax[j]))
    for j, (an, ch, role) in enumerate(cols):
        axs[0, j].text(0.5, 1.35 if len(rows_f) else 1.0, heads[j], transform=axs[0, j].transAxes, ha="center",
                       fontsize=C.FONT_PT - 2)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax[j], vmax=vmax[j]) if an == "aw_eps" else Normalize(0.0, vmax[j])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=div if an == "aw_eps" else seq), ax=axs[:, j],
                          orientation="horizontal", fraction=0.04, pad=0.01)
        cb.ax.tick_params(labelsize=C.FONT_PT - 3)
        cb.set_label("kcal/mol" if an == "aw_eps" else "attention", fontsize=C.FONT_PT - 3)
    save_fig(fig, "figF2_4_atoms", pd.DataFrame(panel))
    plt.close(fig)
    return panel


# ================================================================ main
def cmd_all(prov=None):
    for f in (cmd_check, cmd_rep, lambda: cmd_primary(prov), cmd_secondary, cmd_hammett, cmd_describe, cmd_figures):
        rc = f()
        if rc:
            return rc
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], list(argv[1:])
    prov = None
    if "--prov" in rest:
        i = rest.index("--prov")
        if i + 1 >= len(rest) or not Path(rest[i + 1]).is_file():
            print(__doc__)
            print("usage: --prov needs an existing json file", file=sys.stderr)
            return 2
        prov = rest[i + 1]
        del rest[i:i + 2]
    table = {"check": cmd_check, "rep": cmd_rep, "primary": lambda: cmd_primary(prov), "secondary": cmd_secondary,
             "hammett": cmd_hammett, "describe": cmd_describe, "figures": cmd_figures, "explain": cmd_explain,
             "all": lambda: cmd_all(prov)}
    if cmd not in table or rest or (prov is not None and cmd not in ("primary", "all")):
        print(__doc__)
        print(f"usage: unknown command or extra arguments ({cmd!r} {rest})", file=sys.stderr)
        return 2
    return table[cmd]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
