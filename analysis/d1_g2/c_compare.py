#!/usr/bin/env python3
r"""D1-G2 Phase C (docs/specs/D1G2.md 7): compare the G2 TS (xTB-only autodE from SMILES) with the DFT TS of the label
and with G1 (the DFT TS re-optimised with xTB), structures only, explanatory. Run through sbatch only (jobs/g2.slurm,
G2ENV=d1 = the D1 run venv: numpy, pandas, networkx, scipy, yaml). Reads only; writes only results/C_compare.csv and
results/C_summary.json (no markdown report: user decision 2026-10-22, same as Phase R).

CLI (submit from analysis/d1_g2):
  python c_compare.py selftest [--workers N]   rule checks + VALIDATION_SPEC 6.4 unit tests + ~16 sample rows printed
                                               (writes nothing)
  python c_compare.py run [--workers N] [--force]
                                               all 9,266 X_targets rows -> results/C_compare.csv, then the summary is
                                               built from the CSV as written -> results/C_summary.json. Skipped (exit
                                               0) when both files exist and the summary's outputs.csv_sha256 = the
                                               CSV; if only the summary is missing or stale, only the summary is
                                               rebuilt (= summary); --force recomputes. N = $SLURM_CPUS_PER_TASK.
  python c_compare.py summary                  rebuild results/C_summary.json from the existing results/C_compare.csv
Exit codes: 0 done; 2 outputs written but some rows have an 'error' (see the column); 3 STOP (rule / threshold / input
mismatch, unit test failure) -- nothing written.

Comparison rule = D1 validation (VALIDATION_SPEC.md 6.4), imported read only, not copied:
  $D1_REPO/D1_build/validation/analysis/rmsd.py (D1 runtime code copy; byte-identical to the main worktree d1-build copy,
  both sha256 recorded) -> mapped_heavy_rmsd (heavy-atom graph isomorphism, node label (element, nH), cap 5000
  isomorphisms, each also against the mirror image x -> -x, proper-rotation Kabsch) and same_ts(a, b, formed_a,
  formed_b, rmsd_A, dform_A) = mapped RMSD <= rmsd_A and max |sorted forming distances difference| <= dform_A.
  Thresholds 0.10 A / 0.02 A = config_val.yaml prereg.same_ts, checked equal to the same_ts defaults and to the spec
  sentence; any mismatch = STOP. No isomorphism -> RMSD inf (written 'inf'), same TS False (the rule as is).

Per row (results/C_compare.csv, one row per X_targets row; columns after 'ts_id' are extra diagnostics):
  rxn_id, job_id, g2_status (G_status.csv: ok / method / pipeline), same_ts_dft, rmsd_heavy_dft, form_diff_dft_max
  (G2 vs DFT TS; empty unless g2_status ok), same_ts_g1, rmsd_heavy_g1, form_diff_g1_max (G2 vs G1; empty unless G2 ok
  and G1 .done), rel1_rmsd_g1, rel2_rmsd_g1 (G2 rel{1,2}.xyz vs G1 rel{1,2}.xyz, mapped heavy RMSD), dipole, skeleton,
  position fields cell_id, core, panel, regio, R1, R2, series, attach, face, stereo_index, stage, ts_id;
  extras: g2_fail_reason, g2_remapped, g2_core_h_total, g2_marker, g1_status, n_atoms, n_heavy, elements, d_dft_1/2,
  d_g2_1/2, d_g1_1/2 (forming distances at the label formed_pairs_ts, pair 1 / 2 = formed_d1 / formed_d2), dform_dft_short
  / _long (G2 - DFT at the DFT-shorter / longer forming pair), dform_g1_short / _long (G2 - G1), rmsd_heavy_dft_order
  (label atom order, mirror-aware Kabsch, no isomorphism search), rel1/2_rmsd_dft (G2 rel vs the label reference
  ref_dipole / ref_dipolarophile), same_ts_g1_dft, rmsd_heavy_g1_dft, form_diff_g1_dft_max, dform_g1dft_short / _long
  (G1 vs DFT TS, same rule: the level-shift baseline), label_* (label fields used in the failure profile), QC columns
  (*_record_dev, *_syms_match, *_pairs_match), error.
  same_ts_dft: True / False for G2-ok rows, empty otherwise (the literal rule, G2 vs the DFT TS). The "same TS"
  judgment for the summary headline, Phase A, Phase R and D1LADDER E2 is same_ts_g1 (decision C-13): True / False where
  G2 is ok and G1 has .done, empty (= not classified) otherwise.

Decisions (user: every judgment = the recommended option; recorded here and in C_summary.json 'decisions'):
  C-1 rule source = the D1 runtime copy of rmsd.py: its fragmenter (EDA-ASM/analysis/espley_xtb_repro) has Si in COV / Z
      (the D1PAIR Si decision, 1.11 A); the main-worktree fragmenter has no Si (KeyError on the SiMe3 rows). rmsd.py itself
      is byte-identical in both places.
  C-2 thresholds from config_val.yaml prereg.same_ts (0.10 / 0.02), verified against rmsd.same_ts defaults and the spec.
  C-3 forming distances: computed from each structure's ts.xyz at the label formed_pairs_ts (0-based, label order;
      G2 / G1 ts.xyz are in label atom order), cross-checked against the records (label formed_d1/2, result.json
      d_form_g2 / d_form_g1: *_record_dev); same_ts compares them sorted (rmsd.same_ts). If a G2 element order differed
      from the label (never expected; g2_syms_match False) the result.json d_form_g2 is used instead (also when the DFT
      TS could not be read: *_syms_match empty, the row carries a 'dft:' error).
  C-4 argument order a = G2 (G1 in the calibration), b = the reference (DFT TS / G1), as the validation (run vs Coley).
  C-5 G2 compared only when G_status.csv status is ok (and .done present; else row error); G1 = g1_final/<rxn_id>/.done.
  C-6 level-shift baseline: G2 is an xTB TS and the DFT TS is DFT, so the 0.02 A forming criterion also measures the
      method shift (d1_pair results/G_g1_summary.csv, G1's own metrics for the SAME TS re-optimised with xTB, 8,367 ok:
      |dd_form| <= 0.02 A for both bonds 179, rmsd_ts_heavy_A <= 0.10 A 3,030, both 88; the shift is scatter, not an
      offset: dd_form_short signed median +0.003 A, |dd_form_short| median 0.07 A, |dd_form_long| median 0.09 A,
      same-order heavy RMSD median 0.13 A). same_ts_dft is kept exactly as the rule says (spec 7 / 13: no threshold
      change); G1 vs DFT with the same rule is reported next to it (extra columns / g1_vs_dft_baseline /
      g1_vs_dft_same_rows), and G2 = G1 is the level-matched (xTB vs xTB) comparison. Not recalibrated (option (c), new
      thresholds, would be a method change; not taken).
  C-7 the two rule components (RMSD <= 0.10, forming <= 0.02) are reported as separate pass fractions (decomposition
      of the same rule, not a new criterion).
  C-8 fractions with Wilson 95 % CI (validation stats.wilson, VALIDATION_SPEC 6.5); distributions: finite values
      (linear percentiles 10/25/50/75/90) + n_inf (no isomorphism) + median_all / p90_all with inf kept (inverted_cdf).
  C-9 groups: full blocks by dipole, skeleton, dipole/skeleton, core, regio, panel, series, attach, face,
      stereo_index, stage; compact counts by R1, R2 and cell_id (81 / 82 / 5,047 values).
  C-10 G2-failed profile: failure category = G_status status (method / pipeline) and the first fail reason; by the same
      fields + R1 / R2 (substituents) + elements (DFT ts.xyz symbols) + n_atoms bins (<=15, 16-20, 21-25, 26-30, 31-40,
      >40) + label flags; numeric medians ok vs failed; G1 status of the failed rows.
  C-11 inputs: labels = $G2/G/labels_d1_g2.json (prep_g.py, verbatim labels_d1.json rows), checked row-equal to
      g2_common.targets() and to X_targets / G_status (job_id, rxn_id, dipole, skeleton); mismatch = STOP.
  C-12 strict JSON summary (NaN -> null, inf -> "inf"); summary built from the CSV re-read (what E2 will read).
  C-13 the "same TS" judgment used downstream (summary headline, Phase A split of 10, Phase R sentence 2, D1LADDER 12
      E2 split) = same_ts_g1: G2 vs G1, both xTB, the same rule and thresholds. Option (b) of the builder's three
      (a: literal same_ts_dft, b: same_ts_g1, c: new thresholds), the recommended one (user: every judgment = the
      recommended option). Reason: same_ts_dft crosses the xTB / DFT level and calls the identical TS 'different' in
      ~99 % of rows (C-6), so it cannot separate "G2 found the label's TS" from "G2 found another TS"; G1 is the label
      TS carried to the xTB surface. This deviates from the literal spec 7 wording ("G2 = DFT TS"): recorded in
      C_summary.json same_ts_downstream and reported; same_ts_dft stays in the CSV unchanged. Coverage: same_ts_g1 is
      empty (not classified, never counted as same or different) where G2 is not ok or G1 has no .done (607 'rest'
      targets have no G1, 292 G1 runs failed).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
VAL = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation")      # D1 runtime code copy (read only)
VAL_MAIN = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/D1_build/validation")  # main worktree, branch d1-build
FRAG_EXPECTED = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/analysis/espley_xtb_repro/fragmenter.py")
# rmsd / stats first, so that 'fragmenter' resolves to the D1 copy (with Si) before any other path is added; no .pyc
# is written (the D1 code copy is read only: stats.py has no cpython-39 .pyc there)
sys.dont_write_bytecode = True
sys.path.insert(0, str(VAL / "analysis"))
import rmsd as rm  # noqa: E402
import stats as vstats  # noqa: E402

sys.path.insert(0, str(HERE))
import g2_common as G  # noqa: E402

D1_JOBS = G.DP.D1_JOBS                                  # /gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs (DFT TS, label refs)
G1_FINAL = G.DP.G1_ROOT_D1F                             # /gpfs/tmp_cpu2/yeseo1ee/d1_pair/g1_final
G2_FINAL = G.G2_FINAL
LABELS = G.G2 / "G" / "labels_d1_g2.json"
G1_SUMMARY = G.DP.RES / "G_g1_summary.csv"
OUT_CSV, OUT_JSON = G.RES / "C_compare.csv", G.RES / "C_summary.json"
SPEC_SENTENCE = "매핑 RMSD ≤ 0.10 Å이고, 형성 결합 두 길이 차이가 모두 ≤ 0.02 Å"
N_TARGETS = 9266
RMSD_A, DFORM_A = 0.10, 0.02                            # verified in verify_rule(); STOP if the sources disagree

POS = ["cell_id", "core", "panel", "regio", "R1", "R2", "series", "attach", "face", "stereo_index", "stage", "ts_id"]
REQUIRED = ["rxn_id", "job_id", "g2_status", "same_ts_dft", "rmsd_heavy_dft", "form_diff_dft_max", "same_ts_g1",
            "rmsd_heavy_g1", "form_diff_g1_max", "rel1_rmsd_g1", "rel2_rmsd_g1", "dipole", "skeleton"]
EXTRA = ["g2_fail_reason", "g2_remapped", "g2_core_h_total", "g2_marker", "g1_status", "n_atoms", "n_heavy", "elements",
         "d_dft_1", "d_dft_2", "d_g2_1", "d_g2_2", "d_g1_1", "d_g1_2", "dform_dft_short", "dform_dft_long",
         "dform_g1_short", "dform_g1_long", "rmsd_heavy_dft_order", "rel1_rmsd_dft", "rel2_rmsd_dft", "same_ts_g1_dft",
         "rmsd_heavy_g1_dft", "form_diff_g1_dft_max", "dform_g1dft_short", "dform_g1dft_long", "label_formed_pairs_ts",
         "label_ts_imag_cm", "label_barrier_kcal", "label_alt_used", "label_flag_async", "label_flag_short_forming",
         "label_flag_small_extra_imag", "label_flag_regio_ambiguous", "dft_record_dev", "g2_record_dev", "g1_record_dev",
         "g2_syms_match", "g1_syms_match", "g2_pairs_match", "g1_pairs_match", "error"]
COLS = REQUIRED + POS + EXTRA
BOOL_COLS = ["same_ts_dft", "same_ts_g1", "same_ts_g1_dft", "g2_remapped", "label_flag_async",
             "label_flag_short_forming", "label_flag_small_extra_imag", "label_flag_regio_ambiguous", "g2_syms_match",
             "g1_syms_match", "g2_pairs_match", "g1_pairs_match"]
STR_COLS = ["job_id", "g2_status", "g2_fail_reason", "g2_marker", "g1_status", "dipole", "skeleton", "elements",
            "label_formed_pairs_ts", "error"] + POS
FULL_GROUPS = ["dipole", "skeleton", "dipole_skeleton", "core", "regio", "panel", "series", "attach", "face",
               "stereo_index", "stage"]
LITE_GROUPS = ["R1", "R2", "cell_id"]
FAIL_FIELDS = ["dipole", "skeleton", "dipole_skeleton", "core", "regio", "panel", "series", "attach", "face",
               "stereo_index", "stage", "R1", "R2", "n_atoms_bin"]
NATOM_BINS, NATOM_LABELS = [0, 15, 20, 25, 30, 40, 10 ** 6], ["<=15", "16-20", "21-25", "26-30", "31-40", ">40"]
LABEL_FLAGS = ["label_flag_async", "label_flag_short_forming", "label_flag_small_extra_imag",
               "label_flag_regio_ambiguous"]
DECISIONS = [" ".join(x.split()) for x in
             re.split(r"\n\s+(?=C-\d+ )", "\n" + __doc__.split("Decisions (", 1)[1].split("\n", 1)[1]) if x.strip()]


# ================================================================ rule (imported, verified)
def verify_rule():
    """rmsd.py / fragmenter / thresholds as the validation defines them; STOP (exit 3) on any mismatch."""
    cfg = yaml.safe_load((VAL / "config_val.yaml").read_text())
    thr = cfg["prereg"]["same_ts"]
    sig = inspect.signature(rm.same_ts).parameters
    cap = inspect.signature(rm.mapped_heavy_rmsd).parameters["cap"].default
    spec = (VAL / "VALIDATION_SPEC.md").read_text()
    frag = Path(rm.fr.__file__).resolve()
    info = dict(
        section="VALIDATION_SPEC.md 6.4 (TS comparison: heavy-atom graph isomorphism + mirror-aware Kabsch RMSD; same TS)",
        rmsd_py=str(Path(rm.__file__).resolve()), rmsd_py_sha256=G.sha256(rm.__file__),
        rmsd_py_main=str(VAL_MAIN / "analysis" / "rmsd.py"),
        rmsd_py_main_sha256=G.sha256(VAL_MAIN / "analysis" / "rmsd.py") if (VAL_MAIN / "analysis" / "rmsd.py").is_file()
        else None,
        fragmenter_py=str(frag), fragmenter_sha256=G.sha256(frag), fragmenter_has_si="Si" in rm.fr.COV,
        fragmenter_factor=rm.fr.FACTOR, stats_py=str(Path(vstats.__file__).resolve()),
        stats_py_sha256=G.sha256(vstats.__file__), config_val=str(VAL / "config_val.yaml"),
        config_val_sha256=G.sha256(VAL / "config_val.yaml"), validation_spec=str(VAL / "VALIDATION_SPEC.md"),
        validation_spec_sha256=G.sha256(VAL / "VALIDATION_SPEC.md"), same_ts_config=thr,
        same_ts_defaults=dict(rmsd_A=sig["rmsd_A"].default, dform_A=sig["dform_A"].default), isomorphism_cap=cap,
        spec_sentence_found=SPEC_SENTENCE in spec, rmsd_A=RMSD_A, dform_A=DFORM_A)
    bad = []
    if not (float(thr["rmsd_A"]) == sig["rmsd_A"].default == RMSD_A):
        bad.append(f"rmsd_A config {thr['rmsd_A']} / default {sig['rmsd_A'].default} / {RMSD_A}")
    if not (float(thr["dform_A"]) == sig["dform_A"].default == DFORM_A):
        bad.append(f"dform_A config {thr['dform_A']} / default {sig['dform_A'].default} / {DFORM_A}")
    if not info["spec_sentence_found"] or "### 6.4 TS 비교" not in spec:
        bad.append("VALIDATION_SPEC 6.4 'same TS' sentence not found")
    if frag != FRAG_EXPECTED.resolve() or not info["fragmenter_has_si"]:
        bad.append(f"fragmenter {frag} (Si {info['fragmenter_has_si']}) != {FRAG_EXPECTED}")
    if info["rmsd_py_main_sha256"] not in (None, info["rmsd_py_sha256"]):
        bad.append("rmsd.py D1 copy != main worktree copy")
    if cap != 5000:
        bad.append(f"mapped_heavy_rmsd cap {cap} != 5000")
    if bad:
        G.stop("rule check: " + "; ".join(bad))
    return info


def selftest_rule(ts):
    """VALIDATION_SPEC 6.4 unit tests 1-3 (+ rotation, as tests/test_units.py) on one DFT TS and the threshold logic."""
    s, x = list(ts[0]), np.asarray(ts[1], dtype=float)
    perm = np.random.default_rng(0).permutation(len(s))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    v = dict(identity=rm.mapped_heavy_rmsd(ts, ts),
             shuffled=rm.mapped_heavy_rmsd(ts, ([s[i] for i in perm], x[perm])),
             mirror=rm.mapped_heavy_rmsd(ts, (s, x * np.array([-1.0, 1.0, 1.0]))),
             rot_trans=rm.mapped_heavy_rmsd(ts, (s, x @ R.T + 3.0)))
    h = next(i for i, e in enumerate(s) if e != "H")
    xp = x.copy()
    xp[h] += np.array([1.0, 0.0, 0.0])
    t = dict(same_identity=bool(rm.same_ts(ts, ts, [2.19, 2.61], [2.19, 2.61], RMSD_A, DFORM_A)[0]),
             same_dform_0015=bool(rm.same_ts(ts, ts, [2.19, 2.61], [2.205, 2.61], RMSD_A, DFORM_A)[0]),
             same_dform_0025=bool(rm.same_ts(ts, ts, [2.19, 2.61], [2.215, 2.61], RMSD_A, DFORM_A)[0]),
             same_heavy_atom_moved_1A=bool(rm.same_ts(ts, (s, xp), [2.19, 2.61], [2.19, 2.61], RMSD_A, DFORM_A)[0]))
    passed = (all(r < 1e-6 for r in v.values()) and t["same_identity"] and t["same_dform_0015"]
              and not t["same_dform_0025"] and not t["same_heavy_atom_moved_1A"])
    return dict(rmsd=v, same_ts=t, passed=bool(passed))


# ================================================================ inputs
def label_pairs(d):
    """formed_pairs_ts in label order ('3-7 5-6' -> [(3, 7), (5, 6)]); pair k <-> formed_d{k}."""
    return [tuple(int(a) for a in p.split("-")) for p in str(d["formed_pairs_ts"]).split()]


def load_inputs():
    X = pd.read_csv(G.RES / "X_targets.csv", dtype={"job_id": str})
    S = pd.read_csv(G.RES / "G_status.csv", dtype={"job_id": str, "fail_reason": str, "status": str})
    labs = json.loads(LABELS.read_text())
    L = {d["job_id"]: d for d in labs}
    bad = []
    if not (len(X) == len(L) == len(labs) == len(S) == N_TARGETS):
        bad.append(f"row counts X {len(X)} labels {len(labs)} ({len(L)} unique) G_status {len(S)} != {N_TARGETS}")
    if not (X.job_id.is_unique and S.job_id.is_unique):
        bad.append("duplicate job_id in X_targets / G_status")
    if set(X.job_id) != set(L) or set(X.job_id) != set(S.job_id):
        bad.append("job_id sets differ between X_targets / labels_d1_g2.json / G_status")
    tg = {d["job_id"]: d for d in G.targets()}                    # labels_d1.json (sha256-checked) minus D''
    canon = lambda d: json.dumps(d, sort_keys=True)               # noqa: E731  (NaN-safe row equality)
    n_not_verbatim = sum(1 for j, d in L.items() if j not in tg or canon(tg[j]) != canon(d))
    if set(tg) != set(L) or n_not_verbatim:
        bad.append(f"labels_d1_g2.json != g2_common.targets() ({n_not_verbatim} rows differ)")
    Si = S.set_index("job_id")
    rows = []
    for t in X.itertuples():
        d, s = L.get(t.job_id), Si.loc[t.job_id] if t.job_id in Si.index else None
        if d is None or s is None:
            continue
        if int(d["rxn_id"]) != int(t.rxn_id) or int(s.rxn_id) != int(t.rxn_id) or d["dipole"] != t.dipole \
                or d["skeleton"] != t.skeleton:
            bad.append(f"{t.job_id}: rxn_id / dipole / skeleton differ")
            continue
        p = label_pairs(d)
        if len(p) != 2:
            bad.append(f"{t.job_id}: formed_pairs_ts {d['formed_pairs_ts']!r} is not two pairs")
            continue
        fl = d.get("flags") or {}
        rows.append(dict(rxn_id=int(t.rxn_id), job_id=t.job_id, g2_status=str(s.status), dipole=d["dipole"],
                         skeleton=d["skeleton"], **{f: "" if d.get(f) is None else str(d.get(f)) for f in POS},
                         g2_fail_reason="" if pd.isna(s.fail_reason) else str(s.fail_reason),
                         g2_remapped=None if pd.isna(s.remapped) else str(s.remapped) == "True",
                         g2_core_h_total=s.core_h_total, label_formed_pairs_ts=d["formed_pairs_ts"],
                         label_ts_imag_cm=d.get("ts_imag_cm"), label_barrier_kcal=d.get("barrier_kcal"),
                         label_alt_used=d.get("alt_used"), label_flag_async=d.get("flag_async"),
                         label_flag_short_forming=fl.get("flag_short_forming"),
                         label_flag_small_extra_imag=fl.get("flag_small_extra_imag"),
                         label_flag_regio_ambiguous=fl.get("flag_regio_ambiguous"),
                         _pairs=p, _formed=[float(d["formed_d1"]), float(d["formed_d2"])],
                         _rel=[d["rel1_file"], d["rel2_file"]], _ts=d["ts_file"]))
    if bad:
        G.stop("inputs: " + "; ".join(bad[:10]) + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""))
    info = {k: dict(path=str(p), sha256=G.sha256(p)) for k, p in
            dict(x_targets=G.RES / "X_targets.csv", g_status=G.RES / "G_status.csv", g_report=G.RES / "G_report.json",
                 labels=LABELS, g1_summary=G1_SUMMARY).items()}
    g_rep = json.loads((G.RES / "G_report.json").read_text())
    if g_rep.get("status_csv_sha256") != info["g_status"]["sha256"] or not g_rep.get("all_finished"):
        G.stop("G_report.json: G_status.csv sha256 differs or Phase G not all finished")
    info.update(n_targets=len(rows), d1_jobs=str(D1_JOBS), g1_final=str(G1_FINAL), g2_final=str(G2_FINAL))
    return pd.DataFrame(rows), info


# ================================================================ per row (worker)
def _read(p):
    return rm.fr.read_xyz(p)


def _dist(x, p):
    return float(np.linalg.norm(np.asarray(x[1][p[0]]) - np.asarray(x[1][p[1]])))


def marker_status(d):
    """ok (.done) | fail_<reason> (.fail_<reason>) | unfinished (folder, no marker) | absent (no folder)."""
    if not d.is_dir():
        return "absent"
    names = os.listdir(d)
    if ".done" in names:
        return "ok"
    f = sorted(n for n in names if n.startswith(".fail_"))
    return "fail_" + f[0][len(".fail_"):] if f else "unfinished"


def order_rmsd(a, b):
    """label atom order, heavy atoms, proper-rotation Kabsch against b and its mirror image (no isomorphism search)."""
    if list(a[0]) != list(b[0]):
        return float("nan")
    h = [i for i, s in enumerate(a[0]) if s != "H"]
    P, Q = np.asarray(a[1], float)[h], np.asarray(b[1], float)[h]
    return min(rm.kabsch_rmsd(P, Q * np.array([m, 1.0, 1.0])) for m in (1.0, -1.0))


def record_map(res, key):
    """{sorted pair: distance} from a G1 / G2 result.json (formed_pairs_label + d_form_*)."""
    rp, rd = res.get("formed_pairs_label"), res.get(key)
    if not rp or not rd or len(rp) != len(rd):
        return {}
    return {tuple(sorted(int(i) for i in p)): float(v) for p, v in zip(rp, rd)}


def rule(a, b, fa, fb):
    s, r, dd = rm.same_ts(a, b, fa, fb, RMSD_A, DFORM_A)
    return bool(s), float(r), float(dd)


def one(t):
    """all comparisons of one target row; every part in its own try (errors -> 'error', the other parts still run)."""
    rid, job, pairs = t["rxn_id"], t["job_id"], t["_pairs"]
    keys = [tuple(sorted(p)) for p in pairs]
    o, err = dict(rxn_id=rid, job_id=job), []
    dft = g1 = g1r = None
    d_dft = d_g1 = None
    try:
        jd = D1_JOBS / job
        dft = _read(jd / t["_ts"])
        d_dft = [_dist(dft, p) for p in pairs]
        o.update(n_atoms=len(dft[0]), n_heavy=sum(s != "H" for s in dft[0]), elements=",".join(sorted(set(dft[0]))),
                 d_dft_1=d_dft[0], d_dft_2=d_dft[1],
                 dft_record_dev=max(abs(a - b) for a, b in zip(d_dft, t["_formed"])))
        dref = [_read(jd / f) for f in t["_rel"]]
    except Exception as e:                                       # noqa: BLE001
        err.append(f"dft: {type(e).__name__}: {e}")
        dft = None
    try:
        g1d = G1_FINAL / str(rid)
        o["g1_status"] = marker_status(g1d)
        if o["g1_status"] == "ok":
            g1 = _read(g1d / "ts.xyz")
            g1r = [_read(g1d / "rel1.xyz"), _read(g1d / "rel2.xyz")]
            o["g1_syms_match"] = (list(g1[0]) == list(dft[0])) if dft is not None else None   # None: DFT TS unread
            m = record_map(json.loads((g1d / "result.json").read_text()), "d_form_g1")
            o["g1_pairs_match"] = set(m) == set(keys)
            d_g1 = [_dist(g1, p) for p in pairs] if o["g1_syms_match"] else ([m[k] for k in keys]
                                                                             if o["g1_pairs_match"] else None)
            if d_g1 is not None:
                o.update(d_g1_1=d_g1[0], d_g1_2=d_g1[1])
                if o["g1_pairs_match"]:
                    o["g1_record_dev"] = max(abs(m[k] - v) for k, v in zip(keys, d_g1))
            if dft is not None and d_g1 is not None:
                s, r, dd = rule(g1, dft, d_g1, d_dft)
                o.update(same_ts_g1_dft=s, rmsd_heavy_g1_dft=r, form_diff_g1_dft_max=dd)
    except Exception as e:                                       # noqa: BLE001
        err.append(f"g1: {type(e).__name__}: {e}")
        g1 = None
    g2d = G2_FINAL / str(rid)
    o["g2_marker"] = marker_status(g2d)
    if t["g2_status"] == "ok":
        try:
            if o["g2_marker"] != "ok":
                raise RuntimeError(f"G_status ok but g2_final marker {o['g2_marker']}")
            g2 = _read(g2d / "ts.xyz")
            g2r = [_read(g2d / "rel1.xyz"), _read(g2d / "rel2.xyz")]
            m = record_map(json.loads((g2d / "result.json").read_text()), "d_form_g2")
            o["g2_pairs_match"] = set(m) == set(keys)
            o["g2_syms_match"] = (list(g2[0]) == list(dft[0])) if dft is not None else None   # None: DFT TS unread
            d_g2 = [_dist(g2, p) for p in pairs] if o["g2_syms_match"] else ([m[k] for k in keys]
                                                                             if o["g2_pairs_match"] else None)
            if d_g2 is None:
                raise RuntimeError("G2 element order differs from the label and result.json pairs differ")
            o.update(d_g2_1=d_g2[0], d_g2_2=d_g2[1])
            if o["g2_pairs_match"]:
                o["g2_record_dev"] = max(abs(m[k] - v) for k, v in zip(keys, d_g2))
            if dft is not None:
                s, r, dd = rule(g2, dft, d_g2, d_dft)
                o.update(same_ts_dft=s, rmsd_heavy_dft=r, form_diff_dft_max=dd,
                         rmsd_heavy_dft_order=order_rmsd(g2, dft) if o["g2_syms_match"] else float("nan"),
                         rel1_rmsd_dft=rm.mapped_heavy_rmsd(g2r[0], dref[0]),
                         rel2_rmsd_dft=rm.mapped_heavy_rmsd(g2r[1], dref[1]))
            if g1 is not None and d_g1 is not None:
                s, r, dd = rule(g2, g1, d_g2, d_g1)
                o.update(same_ts_g1=s, rmsd_heavy_g1=r, form_diff_g1_max=dd,
                         rel1_rmsd_g1=rm.mapped_heavy_rmsd(g2r[0], g1r[0]),
                         rel2_rmsd_g1=rm.mapped_heavy_rmsd(g2r[1], g1r[1]))
        except Exception as e:                                   # noqa: BLE001
            err.append(f"g2: {type(e).__name__}: {e}")
    o["error"] = " | ".join(err)
    return o


def assemble(base, rows):
    R = pd.DataFrame(rows)
    D = base.drop(columns=[c for c in base.columns if c.startswith("_")]).merge(R, on=["rxn_id", "job_id"], how="left",
                                                                               validate="one_to_one")
    for c in COLS:
        if c not in D.columns:
            D[c] = np.nan
    sh = D.d_dft_1 <= D.d_dft_2                                  # pair 1 is the DFT-shorter forming bond
    for name, a, b in (("dform_dft", "d_g2", "d_dft"), ("dform_g1", "d_g2", "d_g1"), ("dform_g1dft", "d_g1", "d_dft")):
        d1, d2 = D[f"{a}_1"] - D[f"{b}_1"], D[f"{a}_2"] - D[f"{b}_2"]
        D[f"{name}_short"] = np.where(sh, d1, d2)
        D[f"{name}_long"] = np.where(sh, d2, d1)
    for c in BOOL_COLS:
        D[c] = D[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)) \
            .astype("boolean")
    D["error"] = D["error"].fillna("")
    return D[COLS].sort_values("job_id").reset_index(drop=True)


def read_compare(path=OUT_CSV):
    """C_compare.csv as consumers read it (empty cell = missing; 'inf' = no isomorphism)."""
    D = pd.read_csv(path, dtype={c: str for c in STR_COLS}, keep_default_na=False, na_values=[""])
    for c in STR_COLS:
        D[c] = D[c].fillna("")
    tf = {"True": True, "False": False, True: True, False: False}
    for c in BOOL_COLS:
        D[c] = D[c].map(lambda v: tf.get(v) if not (isinstance(v, float) and np.isnan(v)) else None).astype("boolean")
    return D


# ================================================================ summary
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if np.isnan(f) else ("inf" if f == np.inf else "-inf" if f == -np.inf else f)
    if o is pd.NA or o is None:
        return None
    return o


def frac(k, n):
    k, n = int(k), int(n)
    lo, hi = vstats.wilson(k, n) if n else (float("nan"), float("nan"))
    return dict(k=k, n=n, frac=k / n if n else None, ci95=[lo, hi])


def q(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    fin = x[np.isfinite(x)]
    out = dict(n=int(len(x)), n_inf=int(len(x) - len(fin)))
    if len(fin):
        p = np.percentile(fin, [10, 25, 50, 75, 90])
        out.update(mean=float(fin.mean()), p10=p[0], p25=p[1], median=p[2], p75=p[3], p90=p[4], max=float(fin.max()))
    if len(x) and out["n_inf"]:
        out.update(median_all=float(np.quantile(x, 0.5, method="inverted_cdf")),
                   p90_all=float(np.quantile(x, 0.9, method="inverted_cdf")))
    return out


def nb(s):
    return int(s.astype(bool).sum())


def vc(s, k=None):
    v = s.value_counts()
    return {str(i): int(n) for i, n in (v.head(k) if k else v).items()}


def rule_parts(C, same, r, dd):
    C = C[C[same].notna()]
    return dict(same_ts=frac(nb(C[same]), len(C)), rmsd_pass=frac(int((C[r] <= RMSD_A).sum()), len(C)),
                dform_pass=frac(int((C[dd] <= DFORM_A).sum()), len(C)),
                heavy_graph_isomorphic=frac(int(np.isfinite(C[r].astype(float)).sum()), len(C)),
                rmsd_heavy=q(C[r]), form_diff_max=q(C[dd]))


def block(D, lite=False):
    ok = D[D.g2_status.eq("ok")]
    c = ok[ok.same_ts_dft.notna()]
    c1 = ok[ok.same_ts_g1.notna()]
    if lite:
        return dict(n_targets=len(D), n_g2_ok=len(ok), n_cmp_dft=len(c), n_same_dft=nb(c.same_ts_dft),
                    n_cmp_g1=len(c1), n_same_g1=nb(c1.same_ts_g1),
                    rmsd_dft_median=q(c.rmsd_heavy_dft).get("median"), rmsd_g1_median=q(c1.rmsd_heavy_g1).get("median"))
    b = dict(n_targets=len(D), g2_ok=frac(len(ok), len(D)), g2_eq_dft=frac(nb(c.same_ts_dft), len(c)),
             g2_eq_dft_of_targets=frac(nb(c.same_ts_dft), len(D)), g2_eq_g1=frac(nb(c1.same_ts_g1), len(c1)),
             g1_ok_of_g2_ok=frac(len(c1), len(ok)))
    b["g2_vs_dft"] = dict(rule_parts(c, "same_ts_dft", "rmsd_heavy_dft", "form_diff_dft_max"),
                          rmsd_heavy_label_order=q(c.rmsd_heavy_dft_order), dform_short=q(c.dform_dft_short),
                          dform_long=q(c.dform_dft_long), abs_dform_short=q(c.dform_dft_short.abs()),
                          abs_dform_long=q(c.dform_dft_long.abs()), rel1_rmsd_vs_label_ref=q(c.rel1_rmsd_dft),
                          rel2_rmsd_vs_label_ref=q(c.rel2_rmsd_dft))
    b["g2_vs_g1"] = dict(rule_parts(c1, "same_ts_g1", "rmsd_heavy_g1", "form_diff_g1_max"),
                         dform_short=q(c1.dform_g1_short), dform_long=q(c1.dform_g1_long),
                         abs_dform_short=q(c1.dform_g1_short.abs()), abs_dform_long=q(c1.dform_g1_long.abs()),
                         rel1_rmsd=q(c1.rel1_rmsd_g1), rel2_rmsd=q(c1.rel2_rmsd_g1),
                         rel1_graph_isomorphic=frac(int(np.isfinite(c1.rel1_rmsd_g1.astype(float)).sum()), len(c1)),
                         rel2_graph_isomorphic=frac(int(np.isfinite(c1.rel2_rmsd_g1.astype(float)).sum()), len(c1)))
    b["g1_vs_dft_same_rows"] = dict(rule_parts(c1, "same_ts_g1_dft", "rmsd_heavy_g1_dft", "form_diff_g1_dft_max"),
                                    dform_short=q(c1.dform_g1dft_short), dform_long=q(c1.dform_g1dft_long))
    x = c1[c1.same_ts_dft.notna() & c1.same_ts_g1.notna()]
    b["crosstab_same_dft_x_same_g1"] = {f"dft_{a}_g1_{g}": int(((x.same_ts_dft == a) & (x.same_ts_g1 == g)).sum())
                                        for a in (True, False) for g in (True, False)}
    return b


def fail_profile(D):
    D = D.assign(fail=~D.g2_status.eq("ok"), reason=D.g2_fail_reason.str.split(";").str[0].fillna(""))
    F = D[D.fail]
    out = dict(n_targets=len(D), failed=frac(len(F), len(D)), by_status=vc(F.g2_status), by_reason=vc(F.reason),
               by_status_reason={s: vc(x.reason) for s, x in F.groupby("g2_status")},
               pipeline_rxn_ids=sorted(int(r) for r in F[F.g2_status.eq("pipeline")].rxn_id))
    out["by_field"] = {}
    for f in FAIL_FIELDS:
        g = {}
        for v, x in D.groupby(f, sort=True, observed=True):
            y = x[x.fail]
            g[str(v)] = dict(n=len(x), n_fail=len(y), fail_rate=len(y) / len(x), by_status=vc(y.g2_status),
                             top_reasons=vc(y.reason, 5))
        out["by_field"][f] = g
    els = sorted({e for s in D.elements for e in s.split(",") if e})
    out["by_element"] = {}
    for e in els:
        has = D.elements.map(lambda s: e in s.split(","))
        w, wo = D[has], D[~has]
        out["by_element"][e] = dict(n_with=len(w), fail_rate_with=float(w.fail.mean()) if len(w) else None,
                                    n_without=len(wo), fail_rate_without=float(wo.fail.mean()) if len(wo) else None,
                                    top_reasons_with=vc(w[w.fail].reason, 5))
    out["by_label_flag"] = {}
    for fl in LABEL_FLAGS:
        t = D[fl].fillna(False).astype(bool)
        out["by_label_flag"][fl] = dict(n_true=int(t.sum()), fail_rate_true=float(D[t].fail.mean()) if t.any() else None,
                                        n_false=int((~t).sum()),
                                        fail_rate_false=float(D[~t].fail.mean()) if (~t).any() else None)
    num = dict(n_atoms=D.n_atoms, n_heavy=D.n_heavy, async_dft_A=(D.d_dft_1 - D.d_dft_2).abs(),
               d_dft_short_A=D[["d_dft_1", "d_dft_2"]].min(axis=1), d_dft_long_A=D[["d_dft_1", "d_dft_2"]].max(axis=1),
               label_ts_imag_cm=D.label_ts_imag_cm, label_barrier_kcal=D.label_barrier_kcal,
               g2_core_h_total=D.g2_core_h_total)
    out["numeric_ok_vs_failed"] = {k: dict(ok=q(v[~D.fail]), failed=q(v[D.fail])) for k, v in num.items()}
    out["g1"] = dict(g1_status_of_failed=vc(F.g1_status),
                     g1_ok_of_failed_with_g1=frac(int(F.g1_status.eq("ok").sum()), int((F.g1_status != "absent").sum())),
                     g1_ok_of_g2_ok_with_g1=frac(int(D[~D.fail].g1_status.eq("ok").sum()),
                                                 int((D[~D.fail].g1_status != "absent").sum())))
    return out


def qc(D):
    ok = D[D.g2_status.eq("ok")]
    e = D[D.error != ""]
    return dict(n_rows=len(D), n_error_rows=len(e), errors_first=[f"{r.job_id}: {r.error}" for r in e.head(20).itertuples()],
                max_dft_record_dev_A=pd.to_numeric(D.dft_record_dev).max(),
                max_g2_record_dev_A=pd.to_numeric(D.g2_record_dev).max(),
                max_g1_record_dev_A=pd.to_numeric(D.g1_record_dev).max(),
                n_g2_syms_mismatch=int(ok.g2_syms_match.eq(False).sum()),
                n_g2_pairs_mismatch=int(ok.g2_pairs_match.eq(False).sum()),
                n_g1_syms_mismatch=int(D.g1_syms_match.eq(False).sum()),
                n_g1_pairs_mismatch=int(D.g1_pairs_match.eq(False).sum()),
                n_g2_ok_marker_not_done=int((ok.g2_marker != "ok").sum()),
                n_g2_notok_marker_done=int((D[~D.g2_status.eq("ok")].g2_marker == "ok").sum()),
                n_g2_ok_not_compared=int(ok.same_ts_dft.isna().sum()),
                n_label_order_below_mapped=int((np.isfinite(pd.to_numeric(ok.rmsd_heavy_dft, errors="coerce"))
                                                & (ok.rmsd_heavy_dft_order < ok.rmsd_heavy_dft - 1e-9)).sum()),
                n_label_order_below_mapped_note="finite mapped RMSD only (inf = no isomorphism: n_inf / "
                                                "heavy_graph_isomorphic)",
                g1_status_counts=vc(D.g1_status))


def g1_summary_check(D):
    """g1_final markers vs d1_pair results/G_g1_summary.csv status (ok / fail:<reason>)."""
    S = pd.read_csv(G1_SUMMARY)[["rxn_id", "status"]]
    m = D[["rxn_id", "g1_status"]].merge(S, on="rxn_id", how="left")
    have = m.status.notna()
    agree = (m.status.eq("ok") == m.g1_status.eq("ok"))
    return dict(n_in_g1_summary=int(have.sum()), n_ok_mismatch=int((have & ~agree).sum()),
                n_marker_without_summary=int((~have & m.g1_status.ne("absent")).sum()))


def build_summary(D, rule_info, st, inputs, run):
    D = D.assign(dipole_skeleton=D.dipole + "/" + D.skeleton,
                 n_atoms_bin=pd.cut(pd.to_numeric(D.n_atoms), NATOM_BINS, labels=NATOM_LABELS).astype(str))
    ov = block(D)
    allg1 = D[D.same_ts_g1_dft.notna()]
    cal = dict(rule_parts(allg1, "same_ts_g1_dft", "rmsd_heavy_g1_dft", "form_diff_g1_dft_max"),
               dform_short=q(allg1.dform_g1dft_short), dform_long=q(allg1.dform_g1dft_long),
               rows="every G1-ok target (any G2 status)")
    head = dict(n_targets=len(D), n_g2_ok=ov["g2_ok"]["k"], g2_ok_rate=ov["g2_ok"]["frac"],
                frac_g2_eq_dft=ov["g2_eq_dft"]["frac"], frac_g2_eq_dft_of_targets=ov["g2_eq_dft_of_targets"]["frac"],
                frac_g2_eq_g1=ov["g2_eq_g1"]["frac"], n_g2_vs_g1=ov["g2_eq_g1"]["n"],
                rmsd_heavy_dft_median=ov["g2_vs_dft"]["rmsd_heavy"].get("median"),
                rmsd_heavy_dft_p90=ov["g2_vs_dft"]["rmsd_heavy"].get("p90"),
                rmsd_heavy_dft_n_inf=ov["g2_vs_dft"]["rmsd_heavy"].get("n_inf"),
                rmsd_heavy_g1_median=ov["g2_vs_g1"]["rmsd_heavy"].get("median"),
                rmsd_heavy_g1_p90=ov["g2_vs_g1"]["rmsd_heavy"].get("p90"),
                rmsd_heavy_g1_n_inf=ov["g2_vs_g1"]["rmsd_heavy"].get("n_inf"),
                frac_rmsd_pass_dft=ov["g2_vs_dft"]["rmsd_pass"]["frac"],
                frac_dform_pass_dft=ov["g2_vs_dft"]["dform_pass"]["frac"],
                frac_g1_eq_dft_baseline=cal["same_ts"]["frac"], frac_g1_dform_pass_dft_baseline=cal["dform_pass"]["frac"],
                rel1_rmsd_g1_median=ov["g2_vs_g1"]["rel1_rmsd"].get("median"),
                rel2_rmsd_g1_median=ov["g2_vs_g1"]["rel2_rmsd"].get("median"),
                same_ts_column="same_ts_g1 (C-13)", frac_same_ts=ov["g2_eq_g1"]["frac"])
    ok2 = D[D.g2_status.eq("ok")]
    downstream = dict(
        column="same_ts_g1", literal_column="same_ts_dft",
        decision="C-13: option (b), the recommended one (user rule); deviation from the literal spec 7 'G2 = DFT TS'",
        consumers=["summary headline", "D1G2 Phase A 10 (same / different TS split)", "D1G2 Phase R sentence 2",
                   "D1LADDER 12 E2 (same TS split)"],
        n_targets=len(D), n_g2_ok=len(ok2), classified=ov["g2_eq_g1"], same_of_targets=frac(ov["g2_eq_g1"]["k"], len(D)),
        not_classified_g2_ok_by_g1_status=vc(ok2[ok2.same_ts_g1.isna()].g1_status),
        literal_rule_for_reference=ov["g2_eq_dft"], level_shift_baseline_g1_vs_dft=cal["same_ts"])
    return dict(
        spec="docs/specs/D1G2.md 7 (Phase C, structures only, explanatory)", headline=head, rule=rule_info,
        rule_selftest=st, inputs=inputs, run=run, qc=qc(D), g1_marker_vs_g1_summary=g1_summary_check(D),
        same_ts_downstream=downstream, overall=ov, g1_vs_dft_baseline=cal,
        by={f: {str(v): block(x) for v, x in D.groupby(f, sort=True)} for f in FULL_GROUPS},
        by_lite={f: {str(v): block(x, lite=True) for v, x in D.groupby(f, sort=True)} for f in LITE_GROUPS},
        g2_failed=fail_profile(D),
        notes=["same_ts_dft = the VALIDATION_SPEC 6.4 rule as is, G2 (xTB) vs the label DFT TS: the 0.02 A forming "
               "criterion crosses levels of theory; g1_vs_dft_baseline = the same rule on the identical TS carried to xTB "
               "(G1), i.e. what the rule gives for 'same TS' across the level shift. same_ts_g1 is the level-matched "
               "comparison (decisions C-6, C-13).",
               "the 'same TS' judgment for the headline, Phase A, Phase R and D1LADDER E2 is same_ts_g1 (decision C-13, "
               "deviation from the literal spec 7 wording); empty = not classified (G2 not ok or no G1 .done). "
               "same_ts_dft = the literal rule (G2 vs DFT TS), kept for reference.",
               "rmsd 'inf' = no heavy-atom graph isomorphism under fragmenter.heavy_graph (factor 1.25)."],
        decisions=DECISIONS,
        outputs=dict(csv=str(OUT_CSV), csv_sha256=G.sha256(OUT_CSV) if OUT_CSV.is_file() else None, columns=COLS))


# ================================================================ commands
def tasks_of(base):
    return [dict(rxn_id=int(r["rxn_id"]), job_id=r["job_id"], g2_status=r["g2_status"], _pairs=r["_pairs"],
                 _formed=r["_formed"], _rel=r["_rel"], _ts=r["_ts"]) for r in base.to_dict("records")]


def prepare():
    rule_info = verify_rule()
    base, inputs = load_inputs()
    first = base.iloc[0]
    st = selftest_rule(_read(D1_JOBS / first["job_id"] / first["_ts"]))
    st["ts"] = f"{first['job_id']} DFT TS"
    if not st["passed"]:
        G.stop(f"rule self-test failed: {json.dumps(clean(st))}")
    return rule_info, base, inputs, st


def workers_of(a):
    return int(a.workers or os.environ.get("SLURM_CPUS_PER_TASK") or 1)


def run_rows(tasks, n):
    rows, t0 = [], time.time()
    if n <= 1:
        it = map(one, tasks)
        ex = None
    else:
        ex = ProcessPoolExecutor(max_workers=n)
        it = ex.map(one, tasks, chunksize=8)
    try:
        for k, r in enumerate(it, 1):
            rows.append(r)
            if k % 500 == 0 or k == len(tasks):
                print(f"[C] {k}/{len(tasks)} rows, {time.time() - t0:.0f} s", flush=True)
    finally:
        if ex is not None:
            ex.shutdown()
    return rows


def cmd_selftest(a):
    rule_info, base, inputs, st = prepare()
    print(json.dumps(clean(dict(rule=rule_info, selftest=st)), indent=1), flush=True)
    pick = list(base.index[:8])
    pick += list(base.index[base.g2_remapped.eq(True)][:3])
    pick += list(base.index[~base.g2_status.eq("ok")][:3])
    pick += list(base.index[~base.rxn_id.astype(str).isin(set(os.listdir(G1_FINAL)))][:2])
    T = tasks_of(base.loc[sorted(set(pick))])
    rows = run_rows(T, min(workers_of(a), len(T)))
    for r in rows:
        print(json.dumps(clean(r)), flush=True)
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"[selftest] {len(rows)} rows, {n_err} with error; nothing written", flush=True)
    return 2 if n_err else 0


def summary_matches_csv():
    """True if C_summary.json was built from the C_compare.csv now on disk (outputs.csv_sha256)."""
    try:
        return json.loads(OUT_JSON.read_text()).get("outputs", {}).get("csv_sha256") == G.sha256(OUT_CSV)
    except (OSError, ValueError, AttributeError):
        return False


def cmd_run(a):
    if OUT_CSV.is_file() and not a.force:                     # the CSV is written atomically after all rows
        if OUT_JSON.is_file() and summary_matches_csv():
            print(f"[C] {OUT_CSV.name} and {OUT_JSON.name} (built from it) exist; skipped (use --force to recompute)",
                  flush=True)
            return 0
        print(f"[C] {OUT_CSV.name} exists, {OUT_JSON.name} missing or built from another CSV: rebuilding the summary",
              flush=True)
        return cmd_summary(a)
    t0 = time.time()
    rule_info, base, inputs, st = prepare()
    n = workers_of(a)
    rows = run_rows(tasks_of(base), n)
    D = assemble(base, rows)
    G.write_atomic(OUT_CSV, lambda t: D.to_csv(t, index=False))
    run = dict(cmd="run", host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), workers=n,
               wall_s=time.time() - t0, finished=time.strftime("%F %T"), python=sys.version.split()[0],
               numpy=np.__version__, pandas=pd.__version__, networkx=__import__("networkx").__version__)
    return write_summary(rule_info, st, inputs, run)


def write_summary(rule_info, st, inputs, run):
    D = read_compare()
    S = build_summary(D, rule_info, st, inputs, run)
    G.write_json(OUT_JSON, clean(S))
    print(json.dumps(clean(dict(headline=S["headline"], qc=S["qc"])), indent=1), flush=True)
    return 2 if S["qc"]["n_error_rows"] else 0


def cmd_summary(a):
    if not OUT_CSV.is_file():
        G.stop(f"{OUT_CSV} missing (run first)")
    rule_info, base, inputs, st = prepare()
    run = dict(cmd="summary", host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               finished=time.strftime("%F %T"), csv_mtime=time.strftime("%F %T", time.localtime(OUT_CSV.stat().st_mtime)))
    return write_summary(rule_info, st, inputs, run)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("selftest", "run", "summary"):
        p = sp.add_parser(c)
        if c != "summary":
            p.add_argument("--workers", type=int, default=None)
        if c == "run":
            p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return dict(selftest=cmd_selftest, run=cmd_run, summary=cmd_summary)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
