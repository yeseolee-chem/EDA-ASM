#!/usr/bin/env python3
"""r6_recover.py — rev 6 Phase R: recover the rev 4 G1 failures (docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md §5).
Settings are fixed in results_rev6/R_plan.md (committed before any run); PLAN below is that table, and `summary`
STOPs if an attempt used a setting that is not in PLAN.

  python r6_recover.py inventory            # targets, initial ladders, R-1 DFT-partition diagnostic -> R_inventory.csv
  python r6_recover.py run <slice> <n>      # ladders for one interleaved slice of the targets (Espley rows first)
  python r6_recover.py summary              # R2_attempts.csv, R_rows.csv, R2_by_reason.csv, R2_report.json (+ STOPs)

Per rxn -> $G1_ROOT_R6/<rid>/ : ts.xyz rel1.xyz rel2.xyz result.json work/ts.hess attempts.json, then .done or
.fail_<reason> (written last; a rxn with a marker is skipped = idempotent; ORCA / xtb steps are idempotent too).
rev 4 inputs ($G1_ROOT_REV4, default espley_xtb_g1) are only read. Gates: g1_geom.GATES and the gate code of
g1_geom.process, with the rev 6 partition (r6_partition.label_partition: A = label A_idx).
Exit codes: 0 ok, 2 bad inputs, 3 STOP (R-5).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_geom as G  # noqa: E402  (fragmenter, D1 snapshot: orca_direct, rmsd, run_ts)
import r6_partition as RP  # noqa: E402
import rev6_common as R6  # noqa: E402

fr, od, run_ts = G.fr, G.od, G.run_ts
REV4_ROOT = Path(os.environ.get("G1_ROOT_REV4", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1"))
OUT_ROOT = R6.G1_ROOT_R6
REV4_SUMMARY = R6.R5.RES4 / "g1_geometry_summary.csv"
RES = R6.RES6
ORCA_TIMEOUT_S = 6 * 3600
DISP_A = 0.10                                   # max atomic displacement along a mode (R_plan §4)
NPROCS = G.ORCA_NPROCS                          # 2, as rev 4

# ---------------------------------------------------------------- R_plan §4 (the only allowed settings)
H_COMMON = "! XTB2 ALPB(water) OptTS Freq\n%geom Calc_Hess true Recalc_Hess 5 MaxIter 500 end\n" \
           f"%pal nprocs {NPROCS} end\n"
H_EXACT = "! XTB2 ALPB(water) OptTS Freq\n%geom Calc_Hess true Recalc_Hess 1 MaxIter 500 end\n" \
          f"%pal nprocs {NPROCS} end\n"
H_EXACT_TIGHT = "! XTB2 ALPB(water) OptTS Freq TightOpt\n%geom Calc_Hess true Recalc_Hess 1 MaxIter 500 end\n" \
                f"%pal nprocs {NPROCS} end\n"


def h_pre(pairs):
    cons = "".join(f"  {{B {i} {j} C}}\n" for i, j in pairs)
    return f"! XTB2 ALPB(water) Opt\n%geom\n Constraints\n{cons} end\n MaxIter 500\nend\n%pal nprocs {NPROCS} end\n"


XTB_CONSTRAIN_FC = 1.0
# ladder -> ordered steps (name, start, settings id)
LADDERS = {
    "A": [("optts_restart", "rev4_last", "H_COMMON"), ("exact_hess", "rev4_last", "H_EXACT"),
          ("constrained_pre", "dft_ts", "PRE+H_COMMON")],
    "B": [("exact_hess", "dft_ts", "H_EXACT_TIGHT"), ("displace_mode", "rev4_ts_lowest", "H_COMMON"),
          ("constrained_pre", "dft_ts", "PRE+H_COMMON")],
    "C": [("displace_mode", "rev4_ts_extra", "H_COMMON")],
    "D": [("constrained_pre", "dft_ts", "PRE+H_COMMON")],
}
REF_STEPS = [("alt_start", "XTB_OPT"), ("constrained_ref", "XTB_OPT+CONSTRAIN_CHANGED"),
             ("constrained_ref", "XTB_OPT+CONSTRAIN_ALL")]
PLAN_SETTINGS = {"H_COMMON": H_COMMON, "H_EXACT": H_EXACT, "H_EXACT_TIGHT": H_EXACT_TIGHT,
                 "PRE": "h_pre(label forming pairs)", "XTB_OPT": " ".join(G.XTB_OPT),
                 "CONSTRAIN": f"$constrain force constant={XTB_CONSTRAIN_FC} distance: i, j, auto",
                 "DISP_A": DISP_A, "ORCA_TIMEOUT_S": ORCA_TIMEOUT_S}
# rev 4 gate order of the TS reasons -> ladder
TS_REASON_LADDER = [("optts_not_converged", "A"), ("orca_not_terminated", "A"), ("orca_missing_xyz_or_hess", "A"),
                    ("hess_not_at_final_geometry", "A"), ("imag_below_min", "B"), ("extra_imag_small", "C"),
                    ("imag_mode_on_forming_bonds", "D"), ("not_product_like", "D"), ("forming_bond_present", "D"),
                    ("no_foreign_bond", "D"), ("forming_bond_map_failed", "D")]
GATE_KEYS = ["ts_converged", "hess_geom_max_dev_A", "hess_freq_max_dev_cm", "imag_freqs_cm", "imag_checks",
             "imag_mode_forming_share", "not_product_like", "forming_bond_present", "foreign", "ref1_graph_same",
             "ref2_graph_same", "partition_diag"]


class Fail(Exception):
    pass


# ---------------------------------------------------------------- inputs
def load_esp_ids():
    return set(int(x) for x in pd.read_pickle(R6.ESP_DS)["reaction_number"])


def rev4_result(rid):
    f = REV4_ROOT / str(rid) / "result.json"
    return json.loads(f.read_text()) if f.is_file() else None


def targets():
    """rev 4 G1 failures among the accepted labels: rxn_id -> rev 4 result.json."""
    out = {}
    for lab in G.load_labels():
        rid = int(lab["rxn_id"])
        r = rev4_result(rid)
        if r is None:
            raise SystemExit(f"rev 4 G1 result missing for rxn {rid} under {REV4_ROOT}")
        if r.get("status") != "ok":
            out[rid] = r
    return out


def alt_file(label, k):
    """The other Coley file of reference k (R_plan §4-E a) or None."""
    name = label[f"rel{k + 1}_file"]
    pdir = RP.PROF / str(int(label["rxn_id"]))
    other = name.replace("_alt.xyz", ".xyz") if name.endswith("_alt.xyz") else name[:-4] + "_alt.xyz"
    return pdir / other if (pdir / other).is_file() else None


def same_order(a, b):
    """Same element sequence and identical bond matrix index by index (atom order preserved)."""
    if list(a[0]) != list(b[0]):
        return False
    return bool(np.array_equal(fr.bond_matrix(*a)[0], fr.bond_matrix(*b)[0]))


# ---------------------------------------------------------------- engines
def orca_run(inp: Path):
    """ORCA on inp (own folder), idempotent on a normally terminated .out; timeout kills the process group."""
    out = inp.with_suffix(".out")
    if out.is_file() and od.terminated(out.read_text(errors="replace")):
        return out.read_text(errors="replace"), False
    with open(out, "w") as fo, open(inp.with_suffix(".err"), "w") as fe:
        p = subprocess.Popen([G.ORCA_BIN, inp.name], cwd=inp.parent, stdout=fo, stderr=fe, start_new_session=True)
        try:
            p.wait(timeout=ORCA_TIMEOUT_S)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            timed_out = True
    return out.read_text(errors="replace"), timed_out


def xtb_opt_args(syms, xyz, charge, wd: Path, constrain_pairs=None):
    """xtb --opt tight (rev 4 settings), optionally with $constrain distance: i, j, auto (1-based). Idempotent.
    Returns (syms, xyz, E) or None."""
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "opt.out"
    if not ((wd / "xtbopt.xyz").is_file() and out.is_file()
            and "GEOMETRY OPTIMIZATION CONVERGED" in out.read_text(errors="replace")):
        G.fr_write_xyz(wd / "start.xyz", syms, xyz)
        args = ["start.xyz", *G.XTB_OPT, "--chrg", str(int(charge))]
        if constrain_pairs:
            lines = ["$constrain", f"   force constant={XTB_CONSTRAIN_FC}"]
            lines += [f"   distance: {i + 1}, {j + 1}, auto" for i, j in constrain_pairs]
            (wd / "constrain.inp").write_text("\n".join(lines + ["$end", ""]))
            args += ["--input", "constrain.inp"]
        G.xtb_run(args, wd, "opt.out")
    text = out.read_text(errors="replace")
    if "GEOMETRY OPTIMIZATION CONVERGED" not in text or not (wd / "xtbopt.xyz").is_file():
        return None
    s, x = fr.read_xyz(wd / "xtbopt.xyz")
    e = G.TOTAL_E_RE.findall(text)
    return s, x, float(e[-1]) if e else None


def parse_ts(wd: Path, text=None):
    """g1_geom.process TS checks on <wd>/ts.{out,xyz,hess}: (res fields, ts1, imag, mode); raises Fail(reason)."""
    if text is None:
        text = (wd / "ts.out").read_text(errors="replace") if (wd / "ts.out").is_file() else ""
    res = dict(ts_terminated=od.terminated(text), ts_converged=od.opt_converged(text), e_ts_eh=od.last_fspe(text),
               n_opt_cycles=len(re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+\d+", text)))
    if not res["ts_terminated"]:
        raise Fail("orca_not_terminated")
    if not res["ts_converged"]:
        raise Fail("optts_not_converged")
    if not (wd / "ts.xyz").is_file() or not (wd / "ts.hess").is_file():
        raise Fail("orca_missing_xyz_or_hess")
    ts1 = fr.read_xyz(wd / "ts.xyz")
    hs, hx = G.hess_atoms(wd / "ts.hess")
    shift = hx - ts1[1]
    res["hess_geom_max_dev_A"] = float(np.abs(shift - shift.mean(0)).max())
    last = text.rsplit("VIBRATIONAL FREQUENCIES", 1)[-1]
    f_out = [float(v) for v in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", last, re.M)]
    f_hess = od.parse_hess(wd / "ts.hess")[0]
    res["hess_freq_max_dev_cm"] = (float(np.abs(np.asarray(f_out) - f_hess).max())
                                   if len(f_out) == len(f_hess) else None)
    if (hs != ts1[0] or res["hess_geom_max_dev_A"] > 1e-4 or res["hess_freq_max_dev_cm"] is None
            or res["hess_freq_max_dev_cm"] > 0.01):
        raise Fail("hess_not_at_final_geometry")
    imag, mode = od.imag_and_mode(wd / "ts.hess")
    res.update(imag_freqs_cm=imag, n_imag=len(imag))
    return res, ts1, imag, mode


def evaluate(label, ts0, ts1, imag, mode, refs, ref_same, a_idx, smiles):
    """The gate block of g1_geom.process (lines 260-303) with the rev 6 partition. Returns (res, reasons)."""
    rid = int(label["rxn_id"])
    pairs = [tuple(sorted(map(int, p.split("-")))) for p in str(label["formed_pairs_ts"]).split()]
    heavy = [i for i, s in enumerate(ts0[0]) if s != "H"]
    res, reasons = {}, []
    for k in (1, 2):
        res[f"ref{k}_graph_same"] = bool(ref_same[k - 1])
        if not ref_same[k - 1]:
            reasons.append(f"ref{k}_graph_changed")
    res["rmsd_ts_heavy_A"] = G.heavy_rmsd_same_order(ts0[1], ts1[1], heavy)
    D0, D1 = fr.dist_matrix(ts0[1]), fr.dist_matrix(ts1[1])
    order = sorted(pairs, key=lambda p: D0[p])
    res.update(formed_pairs_label=[list(p) for p in order], d_form_dft=[float(D0[p]) for p in order],
               d_form_g1=[float(D1[p]) for p in order])
    res["dd_form"] = [b - a for a, b in zip(res["d_form_dft"], res["d_form_g1"])]
    ichecks, ihard = run_ts.imag_gate(imag, G.GATES)
    res["imag_checks"] = ichecks
    reasons += ihard
    share = None if mode is None else run_ts.forming_share(ts1[1], mode, order, heavy)
    res["imag_mode_forming_share"] = share
    if share is None or share < G.GATES["mode_share_min"]:
        reasons.append("imag_mode_on_forming_bonds")
    dg = res["d_form_g1"]
    res.update(not_product_like=max(dg) >= G.GATES["forming_min_A"], forming_bond_present=min(dg) < G.GATES["no_bond_A"],
               flag_short_forming=min(dg) < G.GATES["forming_min_A"], flag_async=max(dg) > G.GATES["forming_max_A"])
    reasons += [k for k in ("not_product_like", "forming_bond_present") if not res[k]]
    P, diag = RP.label_partition(label, ts1, refs[0], refs[1], a_idx)
    res["partition_diag"] = diag
    res["A_idx_label"] = sorted(a_idx)
    res["partition_status"] = "label" if P is not None else "label_map_failed"
    res["foreign"] = None
    if P is None:
        reasons.append("partition_label_map_failed")            # R-5 STOP when it reaches the summary
    else:
        audit = fr.audit_forming_bonds(P, ts1, refs[0], refs[1], smiles[rid])
        if audit is None:
            reasons.append("forming_bond_map_failed")
        else:
            res.update(foreign=audit.foreign, formed_pairs_audit=sorted(list(p) for p in audit.formed_ts))
            res["formed_pairs_audit_eq_label"] = {tuple(p) for p in audit.formed_ts} == set(order)
            if audit.foreign:
                reasons.append("no_foreign_bond")
    return res, reasons


def ts_ladder_for(reasons):
    for r, lad in TS_REASON_LADDER:
        if r in reasons:
            return lad
    return None


# ---------------------------------------------------------------- one rxn
def mode_start(rev4_wd: Path, which):
    """(start xyz list ±, freq) from the rev 4 TS and its ts.hess (R_plan §4 B-b / C-a), or None."""
    if not (rev4_wd / "ts.hess").is_file() or not (rev4_wd / "ts.xyz").is_file():
        return None
    freqs, modes = od.parse_hess(rev4_wd / "ts.hess")
    if which == "lowest":
        cand = [k for k in range(len(freqs)) if abs(freqs[k]) > 1e-6]
        if not cand:
            return None
        k = min(cand, key=lambda k: freqs[k])
    else:
        neg = sorted((k for k in range(len(freqs)) if freqs[k] < 0), key=lambda k: freqs[k])
        if len(neg) < 2:
            return None
        k = neg[1]
    syms, x0 = fr.read_xyz(rev4_wd / "ts.xyz")
    v = modes[:, k].reshape(-1, 3)
    v = v / np.linalg.norm(v, axis=1).max()
    return syms, {"plus": x0 + DISP_A * v, "minus": x0 - DISP_A * v}, float(freqs[k])


def run_ts_attempt(wd: Path, header, syms, xyz, charge):
    wd.mkdir(parents=True, exist_ok=True)
    inp = wd / "ts.inp"
    if not inp.is_file():
        od.write_inp(inp, header, syms, xyz, charge=charge, mult=1)
    text, timed_out = orca_run(inp)
    if timed_out:
        raise Fail("orca_timeout")
    return parse_ts(wd, text)


def process(label, rev4, smiles, meta):
    rid = int(label["rxn_id"])
    d = OUT_ROOT / str(rid)
    if (d / ".done").is_file() or list(d.glob(".fail_*")):
        return rid, "skip"
    d.mkdir(parents=True, exist_ok=True)
    wd = d / "work"
    wd.mkdir(exist_ok=True)
    rev4_wd = REV4_ROOT / str(rid) / "work"
    q1, q2 = int(label.get("charge1") or 0), int(label.get("charge2") or 0)
    pdir = RP.PROF / str(rid)
    ts0 = fr.read_xyz(pdir / label["ts_file"])
    rel0 = [fr.read_xyz(pdir / label[k]) for k in ("rel1_file", "rel2_file")]
    a_idx = meta.get(rid)
    pairs = [tuple(sorted(map(int, p.split("-")))) for p in str(label["formed_pairs_ts"]).split()]
    attempts, t0 = [], time.time()
    res = dict(rxn_id=rid, rev4_status=rev4.get("status"), rev4_fail_reasons=rev4.get("fail_reasons"),
               gates_cfg=G.GATES, plan=PLAN_SETTINGS, host=os.uname().nodename)
    final_reasons, recovered_by = [], []
    try:
        if a_idx is None:
            raise Fail("no_label_A_idx")
        # ---- references: rev 4 result when it exists, else the rev 4 procedure now; ladder E if the graph changed
        refs, ref_same, e_ref = [], [], []
        for k in (0, 1):
            src = rev4_wd / f"ref{k + 1}"
            got = None
            src_ok = ((src / "xtbopt.xyz").is_file() and (src / "opt.out").is_file()
                      and "GEOMETRY OPTIMIZATION CONVERGED" in (src / "opt.out").read_text(errors="replace"))
            if src_ok:
                s, x = fr.read_xyz(src / "xtbopt.xyz")
                e = G.TOTAL_E_RE.findall((src / "opt.out").read_text(errors="replace"))
                got, res[f"ref{k + 1}_source"] = (s, x, float(e[-1]) if e else None), "rev4"
            else:
                got = xtb_opt_args(*rel0[k], (q1, q2)[k], wd / f"ref{k + 1}_base")
                res[f"ref{k + 1}_source"] = "rev4_procedure_now"
            if got is None:
                raise Fail(f"ref{k + 1}_opt_not_converged")
            same = G.heavy_graph_same(rel0[k], got[:2])
            if not same:
                base_adj = fr.bond_matrix(*got[:2])[0]
                for step, setting in REF_STEPS:
                    att = dict(rxn_id=rid, part=f"ref{k + 1}", ladder="E", step=step, setting=setting,
                               direction=None)
                    if step == "alt_start":
                        alt = alt_file(label, k)
                        if alt is None:
                            att.update(status="skipped", reason="no_alt_file"); attempts.append(att); continue
                        a = fr.read_xyz(alt)
                        if not same_order(a, rel0[k]):
                            att.update(status="skipped", reason="alt_atom_order_differs", start=alt.name)
                            attempts.append(att); continue
                        att["start"] = alt.name
                        cand = xtb_opt_args(*a, (q1, q2)[k], wd / f"ref{k + 1}_alt")
                    else:
                        adj0 = fr.bond_matrix(*rel0[k])[0]
                        n = len(rel0[k][0])
                        if setting.endswith("CHANGED"):
                            cp = [(i, j) for i in range(n) for j in range(i + 1, n) if adj0[i, j] != base_adj[i, j]]
                            sub = "changed"
                        else:
                            cp = [(i, j) for i in range(n) for j in range(i + 1, n) if adj0[i, j]]
                            sub = "all"
                        att.update(start=label[f"rel{k + 1}_file"], n_constrained=len(cp))
                        cand = xtb_opt_args(*rel0[k], (q1, q2)[k], wd / f"ref{k + 1}_con_{sub}", constrain_pairs=cp)
                    if cand is None:
                        att.update(status="fail", reason="opt_not_converged"); attempts.append(att); continue
                    ok = G.heavy_graph_same(rel0[k], cand[:2])
                    att.update(status="pass" if ok else "fail", reason=None if ok else "graph_changed",
                               e_eh=cand[2])
                    attempts.append(att)
                    if ok:
                        got, same = cand, True
                        recovered_by.append(step)
                        res[f"ref{k + 1}_source"] = f"{step}:{setting}"
                        break
            refs.append(got[:2]); ref_same.append(same); e_ref.append(got[2])
            res[f"e_rel{k + 1}_eh"] = got[2]
            hk = [i for i, s in enumerate(rel0[k][0]) if s != "H"]
            res[f"rmsd_rel{k + 1}_heavy_A"] = G.heavy_rmsd_same_order(rel0[k][1], got[1], hk)
        if not all(ref_same):
            raise Fail(f"ref{ref_same.index(False) + 1}_graph_changed")

        # ---- TS candidate = rev 4 TS; evaluate with the final references and the label partition
        best = None
        try:
            tsres, ts1, imag, mode = parse_ts(rev4_wd)
            ev, reasons = evaluate(label, ts0, ts1, imag, mode, refs, ref_same, a_idx, smiles)
            tsres.update(ev)
            cand_reasons = reasons
            attempts.append(dict(rxn_id=rid, part="ts", ladder="-", step="rev4_ts", setting="rev4", direction=None,
                                 status="pass" if not reasons else "fail", reason=";".join(reasons),
                                 e_eh=tsres.get("e_ts_eh")))
            if not reasons:
                best = (tsres, ts1, rev4_wd / "ts.hess", "label_aidx" if not recovered_by else None)
        except Fail as e:
            cand_reasons = [str(e)]
            attempts.append(dict(rxn_id=rid, part="ts", ladder="-", step="rev4_ts", setting="rev4", direction=None,
                                 status="fail", reason=str(e)))
        res["ts_initial_reasons"] = cand_reasons
        if best is None:
            lad = ts_ladder_for(cand_reasons)
            res["ts_ladder"] = lad
            if lad is None:
                raise Fail(cand_reasons[0] if cand_reasons else "unknown_ts_failure")
            last_reasons = cand_reasons
            for step, start, setting in LADDERS[lad]:
                runs = []                                   # (direction, wd, header, syms, xyz)
                if start == "rev4_last":
                    if not (rev4_wd / "ts.xyz").is_file():
                        attempts.append(dict(rxn_id=rid, part="ts", ladder=lad, step=step, setting=setting,
                                             status="skipped", reason="no_rev4_last_structure")); continue
                    s, x = fr.read_xyz(rev4_wd / "ts.xyz")
                    runs.append((None, wd / f"{lad}_{step}", {"H_COMMON": H_COMMON, "H_EXACT": H_EXACT}[setting], s, x))
                elif start == "dft_ts" and setting == "H_EXACT_TIGHT":
                    runs.append((None, wd / f"{lad}_{step}", H_EXACT_TIGHT, *ts0))
                elif start == "dft_ts":                     # PRE + H_COMMON
                    pre_wd = wd / f"{lad}_{step}_pre"
                    pre_wd.mkdir(parents=True, exist_ok=True)
                    pinp = pre_wd / "pre.inp"
                    if not pinp.is_file():
                        od.write_inp(pinp, h_pre(pairs), *ts0, charge=q1 + q2, mult=1)
                    ptext, pto = orca_run(pinp)
                    if pto or not od.terminated(ptext) or not od.opt_converged(ptext) or not (pre_wd / "pre.xyz").is_file():
                        attempts.append(dict(rxn_id=rid, part="ts", ladder=lad, step=step, setting=setting,
                                             status="fail", reason="preopt_timeout" if pto else "preopt_not_converged",
                                             header=h_pre(pairs)))
                        last_reasons = ["preopt_not_converged"]
                        continue
                    s, x = fr.read_xyz(pre_wd / "pre.xyz")
                    runs.append((None, wd / f"{lad}_{step}", H_COMMON, s, x))
                else:                                       # displaced starts
                    got = mode_start(rev4_wd, "lowest" if start.endswith("lowest") else "extra")
                    if got is None:
                        attempts.append(dict(rxn_id=rid, part="ts", ladder=lad, step=step, setting=setting,
                                             status="skipped", reason="no_mode")); continue
                    s, starts, nu = got
                    for dirn in ("plus", "minus"):
                        runs.append((dirn, wd / f"{lad}_{step}_{dirn}", H_COMMON, s, starts[dirn]))
                passing = []
                for dirn, rwd, header, s, x in runs:
                    att = dict(rxn_id=rid, part="ts", ladder=lad, step=step, setting=setting, direction=dirn,
                               header=header, start=start)
                    try:
                        tsres, ts1, imag, mode = run_ts_attempt(rwd, header, s, x, q1 + q2)
                        ev, reasons = evaluate(label, ts0, ts1, imag, mode, refs, ref_same, a_idx, smiles)
                        tsres.update(ev)
                        att.update(status="pass" if not reasons else "fail", reason=";".join(reasons),
                                   e_eh=tsres.get("e_ts_eh"), imag=tsres.get("imag_freqs_cm"),
                                   share=tsres.get("imag_mode_forming_share"), d_form=tsres.get("d_form_g1"),
                                   n_cycles=tsres.get("n_opt_cycles"), partition=tsres["partition_diag"])
                        if not reasons:
                            passing.append((tsres, ts1, rwd / "ts.hess", step))
                        last_reasons = reasons
                    except Fail as e:
                        att.update(status="fail", reason=str(e))
                        last_reasons = [str(e)]
                    attempts.append(att)
                if passing:
                    best = min(passing, key=lambda t: t[0]["e_ts_eh"])
                    break
            if best is None:
                raise Fail(last_reasons[0] if last_reasons else "ladder_exhausted")
            recovered_by.insert(0, best[3])
        tsres, ts1, hess_src, how = best
        if how == "label_aidx" and not recovered_by:
            recovered_by = ["label_aidx"]
        res.update(tsres)
        if all(res.get(f"e_rel{k}_eh") is not None for k in (1, 2)) and res.get("e_ts_eh") is not None:
            res["xtb_barrier_g1_kcal"] = (res["e_ts_eh"] - res["e_rel1_eh"] - res["e_rel2_eh"]) * G.EH2KCAL
        res["recovered_by"] = ";".join(recovered_by)
        (d / "work").mkdir(exist_ok=True)
        shutil.copyfile(hess_src, d / "work" / "ts.hess.tmp")
        os.replace(d / "work" / "ts.hess.tmp", d / "work" / "ts.hess")
        G.fr_write_xyz(d / "ts.xyz", *ts1, f"G1 rev6 TS rxn {rid}")
        for k in (0, 1):
            G.fr_write_xyz(d / f"rel{k + 1}.xyz", *refs[k], f"G1 rev6 reference {k + 1} rxn {rid}")
        missing = [k for k in GATE_KEYS if res.get(k) is None]
        res["gate_keys_missing"] = missing
    except Fail as e:
        final_reasons = [str(e)]
    except Exception as e:                                               # noqa: BLE001
        res["traceback"] = traceback.format_exc()
        final_reasons = [f"error_{type(e).__name__}"]
    res["fail_reasons"] = final_reasons
    res["status"] = "ok" if not final_reasons else "fail:" + final_reasons[0]
    res["wall_s"] = time.time() - t0
    G.od_write_json(d / "attempts.json", attempts)
    G.od_write_json(d / "result.json", res)
    for f in ("gbw", "densities", "engrad", "opt", "cis", "tmp", "xtbrestart", "charges", "wbo"):
        for p in wd.rglob(f"*.{f}"):
            p.unlink(missing_ok=True)
    for p in wd.rglob("ts.[0-9][0-9][0-9].hess"):          # per-step Hessians of Recalc_Hess 1 (the final ts.hess stays)
        p.unlink(missing_ok=True)
    marker = d / (".done" if not final_reasons else f".fail_{re.sub(r'[^A-Za-z0-9_]+', '_', final_reasons[0])}")
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n" + "\n".join(final_reasons) + "\n")
    return rid, res["status"]


# ---------------------------------------------------------------- commands
def ordered_targets():
    tg = targets()
    esp = load_esp_ids()
    return sorted(tg, key=lambda r: (r not in esp, r)), tg, esp


def cmd_inventory():
    order, tg, esp = ordered_targets()
    labels = {int(d["rxn_id"]): d for d in G.load_labels()}
    meta = G.load_meta()
    rows = []
    for rid in sorted(labels):
        lab = labels[rid]
        st, match = RP.dft_redetect_matches(lab, meta[rid]) if rid in meta else ("no_label_A_idx", False)
        r4 = tg.get(rid)
        reasons = (r4 or {}).get("fail_reasons") or []
        rows.append(dict(rxn_id=rid, in_esp=rid in esp, rev4_ok=r4 is None,
                         rev4_status=(r4 or {}).get("status", "ok"), rev4_reasons=";".join(reasons),
                         ts_ladder_initial=ts_ladder_for(reasons) if r4 else None,
                         ref1_changed="ref1_graph_changed" in reasons, ref2_changed="ref2_graph_changed" in reasons,
                         alt1=bool(alt_file(lab, 0)), alt2=bool(alt_file(lab, 1)),
                         dft_partition_status=st, dft_A_match=match, order=order.index(rid) if rid in tg else None))
    df = pd.DataFrame(rows)
    R6.write_atomic(RES / "R_inventory.csv", lambda t: df.to_csv(t, index=False))
    bad_t = df[(~df.rev4_ok) & (~df.dft_A_match)]
    rep = dict(n_labels=len(df), n_targets=int((~df.rev4_ok).sum()), n_targets_esp=int(((~df.rev4_ok) & df.in_esp).sum()),
               rev4_status_counts=df.loc[~df.rev4_ok, "rev4_status"].value_counts().to_dict(),
               ts_ladder_initial=df.loc[~df.rev4_ok, "ts_ladder_initial"].fillna("none").value_counts().to_dict(),
               dft_A_mismatch_targets=bad_t.rxn_id.tolist(),
               dft_A_mismatch_all=int((~df.dft_A_match).sum()),
               dft_partition_status_all=df.dft_partition_status.value_counts().to_dict(),
               stop_R1_dft_mismatch=bool(len(bad_t)))
    R6.write_json(RES / "R_inventory.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop_R1_dft_mismatch"] else 0


def cmd_run(slice_id, n_slices):
    order, tg, _ = ordered_targets()
    only = [int(x) for x in os.environ.get("R6_ONLY", "").split(",") if x.strip()]     # smoke: these rxns only
    if only:
        bad = [r for r in only if r not in tg]
        if bad:
            raise SystemExit(f"R6_ONLY: not rev 4 G1 failures: {bad}")
        order = [r for r in order if r in only]
    skip = {int(x) for x in os.environ.get("R6_SKIP", "").split(",") if x.strip()}   # rxns another job is running
    order = [r for r in order if r not in skip]
    mine = order[slice_id::n_slices]
    labels = {int(d["rxn_id"]): d for d in G.load_labels()}
    smiles, meta = G.load_smiles(), G.load_meta()
    workers = int(os.environ.get("R6_WORKERS", max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) // NPROCS)))
    print(f"[R slice {slice_id}/{n_slices}] {len(mine)} rxns, {workers} workers x ORCA nprocs {NPROCS}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, labels[r], tg[r], smiles, meta) for r in mine]
        for k, f in enumerate(futs):
            rid, st = f.result()
            print(f"[{k + 1}/{len(futs)}] rxn {rid}: {st}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


def cmd_summary():
    order, tg, esp = ordered_targets()
    rows, atts, unmarked = [], [], []
    for rid in order:
        d = OUT_ROOT / str(rid)
        if not ((d / ".done").is_file() or list(d.glob(".fail_*"))):
            unmarked.append(rid); continue
        r = json.loads((d / "result.json").read_text())
        a = json.loads((d / "attempts.json").read_text()) if (d / "attempts.json").is_file() else []
        atts += a
        r4 = tg[rid]
        dd = r.get("dd_form") or [np.nan, np.nan]
        rows.append(dict(rxn_id=rid, in_esp=rid in esp, rev4_status=r4.get("status"),
                         rev4_reasons=";".join(r4.get("fail_reasons") or []), status=r["status"],
                         fail_reasons=";".join(r.get("fail_reasons") or []), recovered_by=r.get("recovered_by"),
                         ts_ladder=r.get("ts_ladder"), imag1_cm=(r.get("imag_freqs_cm") or [np.nan])[0],
                         n_imag=r.get("n_imag"), forming_share=r.get("imag_mode_forming_share"),
                         rmsd_ts_heavy_A=r.get("rmsd_ts_heavy_A"), dd_form_short_A=dd[0], dd_form_long_A=dd[1],
                         map_source=(r.get("partition_diag") or {}).get("map_source"),
                         redetect_A_match=(r.get("partition_diag") or {}).get("redetect_A_match"),
                         gate_keys_missing=";".join(r.get("gate_keys_missing") or []), wall_s=r.get("wall_s")))
    if unmarked:
        print(f"GATE: {len(unmarked)} targets without a marker under {OUT_ROOT}: {unmarked[:10]}", file=sys.stderr)
        return 2
    df, at = pd.DataFrame(rows), pd.DataFrame(atts)
    R6.write_atomic(RES / "R_rows.csv", lambda t: df.to_csv(t, index=False))
    R6.write_atomic(RES / "R2_attempts.csv", lambda t: at.to_csv(t, index=False))
    df["category"] = df.rev4_status.str.replace("fail:", "", regex=False)
    df["first_method"] = df.recovered_by.fillna("").str.split(";").str[0]
    tab = (df.assign(recovered=df.status == "ok")
           .pivot_table(index="category", columns="first_method", values="rxn_id", aggfunc="count", fill_value=0))
    tab["n_rev4"] = df.groupby("category").size()
    tab["n_recovered"] = df[df.status == "ok"].groupby("category").size().reindex(tab.index).fillna(0).astype(int)
    tab["n_remaining"] = tab["n_rev4"] - tab["n_recovered"]
    tab = tab.drop(columns=[""], errors="ignore").reset_index()
    R6.write_atomic(RES / "R2_by_reason.csv", lambda t: tab.to_csv(t, index=False))
    ok = df.status == "ok"
    # STOP checks (R-5)
    allowed = {s for lad in LADDERS.values() for _, _, s in lad} | {s for _, s in REF_STEPS} | {"rev4"}
    bad_setting = sorted(set(at.get("setting", pd.Series(dtype=str)).dropna()) - allowed)
    hdr_ok = {H_COMMON, H_EXACT, H_EXACT_TIGHT}
    bad_hdr = [h for h in at.get("header", pd.Series(dtype=str)).dropna().unique()
               if h not in hdr_ok and not h.startswith("! XTB2 ALPB(water) Opt\n%geom\n Constraints\n")]
    missing = df.loc[ok & (df.gate_keys_missing.fillna("") != ""), "rxn_id"].tolist()
    label_map_failed = df.loc[df.fail_reasons.str.contains("partition_label_map_failed", na=False), "rxn_id"].tolist()
    rep = dict(n_targets=len(df), n_recovered=int(ok.sum()), n_remaining=int((~ok).sum()),
               n_recovered_esp=int((ok & df.in_esp).sum()), n_targets_esp=int(df.in_esp.sum()),
               recovered_by=df.loc[ok, "recovered_by"].value_counts().to_dict(),
               remaining_reasons=df.loc[~ok, "status"].value_counts().to_dict(),
               map_source_recovered=df.loc[ok, "map_source"].value_counts(dropna=False).to_dict(),
               n_attempts=len(at), settings_used=at.get("setting", pd.Series(dtype=str)).value_counts().to_dict(),
               stop_gate_record_missing=missing, stop_setting_not_in_plan=bad_setting + bad_hdr,
               stop_label_map_failed=label_map_failed)
    for c in ("rmsd_ts_heavy_A", "dd_form_short_A", "dd_form_long_A"):
        v = df.loc[ok, c].astype(float)
        a = v.abs() if c.startswith("dd") else v
        rep[c + "_recovered"] = dict(n=int(v.notna().sum()), median=float(v.median()) if len(v) else None,
                                     median_abs=float(a.median()) if len(v) else None,
                                     p90_abs=float(a.quantile(0.9)) if len(v) else None,
                                     max_abs=float(a.max()) if len(v) else None)
    rep["stop"] = bool(missing or bad_setting or bad_hdr or label_map_failed)
    R6.write_json(RES / "R2_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop"] else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "inventory":
        sys.exit(cmd_inventory())
    elif cmd == "run":
        sys.exit(cmd_run(int(sys.argv[2]), int(sys.argv[3])))
    elif cmd == "summary":
        sys.exit(cmd_summary())
    raise SystemExit(__doc__)
