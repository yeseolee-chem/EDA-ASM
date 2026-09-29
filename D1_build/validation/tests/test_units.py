#!/usr/bin/env python3
"""Phase F unit tests (VALIDATION_SPEC §3 completion conditions 2-3, §6.3 parser check, §6.4 RMSD tests)
plus the L0 header, MAE_ref vs SUMMARY.md, the new parsers, and the FIX_348bc3e tests (F-A distance
gates on D0 4295/3400/5783, F-B level-parser positive control). Plain asserts; run by run_tests.sh on a
compute node. Pilot scratch and D0 data are only read; files are written only to a temp dir and to the
V9 worker dir <scratch>/v9/tmp/. Exit 1 if any test fails.
"""
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

VAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VAL / "analysis"))
sys.path.insert(0, str(VAL.parent / "d1_autode_pilot"))
import pilot_common as pc  # noqa: E402

cfg = pc.load_config(VAL / "config_val.yaml")
PS = Path(cfg["pilot_scratch"]) / "jobs"
RESULTS = []


def test(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, True)); print(f"PASS  {fn.__name__}")
    except Exception as e:                                            # noqa: BLE001
        RESULTS.append((fn.__name__, False)); print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
        traceback.print_exc()
    return fn


def coley_ts(rid):
    pdir = Path(cfg["d0_profiles"]) / str(rid)
    return pc.read_xyz(sorted(p for p in pdir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz")[0])


@test
def f1_imag_gate():
    import run_ts
    g = dict(min_imag_cm=-40, extra_imag_tol_cm=-50)
    c, h = run_ts.imag_gate([-360.7, -14.9], g)
    assert h == [] and c["flag_small_extra_imag"] is True, (c, h)
    _, h = run_ts.imag_gate([-360.7, -60], g); assert "extra_imag_small" in h, h
    _, h = run_ts.imag_gate([-30], g); assert "imag_below_min" in h, h
    _, h = run_ts.imag_gate([], g); assert h, "no imaginary mode must fail"
    _, h = run_ts.imag_gate(None, g); assert h == [], h
    _, h = run_ts.imag_gate([-360.7, -14.9], dict(min_imag_cm=-40)); assert h == ["one_imag"], h   # pilot rule (J07)


@test
def f2_ref_corrected_j10():
    import report
    labs = {l["job_id"]: l for l in json.load(open(cfg["val"]["pilot_labels"]))}
    L = labs["J10"]
    _, dG = report.ref_corrected(L["autode_dE_act_sp_kcal"], L["autode_dG_act_kcal"], L["alt_used"], L["ref_port"])
    d = dG - float(L["d0"]["G_act_kcal"])
    print(f"      J10 dG_ref - G_act = {d:+.4f}")
    assert abs(d - 0.32) <= 0.01, d
    dE, dG2 = report.ref_corrected(1.0, 2.0, 0, L["ref_port"]); assert (dE, dG2) == (1.0, 2.0)


@test
def hess_parser_j10():
    import orca_direct as od, run_ts
    tsd = PS / "J10" / "ts" / "J10" / "transition_states"
    imag, mode = od.imag_and_mode(tsd / "TS_LXmTs6_hl_ad_1-10_4-5_optts_orca.hess")
    print(f"      J10 first imaginary {imag[0]:.4f} cm-1")
    assert abs(imag[0] - (-186.75)) < 0.01, imag
    tr = json.loads((PS / "J10" / "ts_result.json").read_text())
    syms, X = pc.read_xyz(PS / "J10" / "ts.xyz")
    share = run_ts.forming_share(X, mode, [tuple(p) for p in tr["formed_pairs_ts"]], [i for i, s in enumerate(syms) if s != "H"])
    print(f"      J10 mode share: ORCA $normal_modes {share:.3f} vs autodE {tr['imag_mode_forming_share']:.3f}")
    assert share >= 0.5


@test
def l0_header_reproduces_pilot():
    import orca_direct as od
    base = od.input_header(cfg["val"]["pilot_optts_inp"])
    h = od.l0_header(cfg, [(1, 10), (4, 5)], n_cores=8)
    assert h == base, "L0 header with J10's forming bonds must equal the pilot header byte for byte"
    h1 = od.level_header(cfg, "L1", [(1, 10), (4, 5)], 8).splitlines()[0]
    assert " B3LYP/G " in h1 and " B3LYP " not in h1, h1
    h2 = od.level_header(cfg, "L2", [(1, 10), (4, 5)], 8)
    assert h2.startswith("! OptTS Freq B3LYP/G NORI D3BJ def2-SVP DefGrid3 TightSCF CPCM(Water)\n"), h2[:90]
    assert od.normalise_header(h2.split("\n", 1)[1]) == od.normalise_header(h.split("\n", 1)[1])
    e = od.level_header(cfg, "L2", (), 8, engrad=True)
    assert e.splitlines()[0].startswith("! EnGrad B3LYP/G NORI") and "%geom" not in e, e
    assert od.opt_header(cfg, 8).startswith("! Opt B3LYP RIJCOSX D3BJ def2-SVP def2/J")
    blk = od.scan_block(1, 2, 3, 4, 1.6, 3.4, 10)
    assert "B 1 2 = 1.60, 3.40, 10" in blk and blk.endswith("end\nend\n")


@test
def rmsd_unit_tests():
    import rmsd as rm
    ts = coley_ts(20)
    s, x = ts[0], np.asarray(ts[1])
    assert rm.mapped_heavy_rmsd(ts, ts) < 1e-6
    perm = np.random.default_rng(0).permutation(len(s))
    assert rm.mapped_heavy_rmsd(ts, ([s[i] for i in perm], x[perm])) < 1e-6, "shuffled atom order"
    assert rm.mapped_heavy_rmsd(ts, (s, x * np.array([-1.0, 1.0, 1.0]))) < 1e-6, "mirror image"
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert rm.mapped_heavy_rmsd(ts, (s, x @ R.T + 3.0)) < 1e-6, "rotation + translation"
    r = rm.mapped_heavy_rmsd(ts, pc.read_xyz(PS / "J12" / "ts.xyz"))
    print(f"      rxn 20 TS vs pilot J12 TS: {r:.2e} A")
    assert r < 1e-4
    same, _, _ = rm.same_ts(ts, ts, [2.19, 2.61], [2.19, 2.61]); assert same


@test
def v9_mode_from_animation():
    import gate_audit_d0 as ga, run_ts
    mode = ga.mode_from_animation(Path(cfg["d0_profiles"]) / "20" / "TS_imag_mode.xyz")
    syms, X = coley_ts(20)
    tr = json.loads((PS / "J12" / "ts_result.json").read_text())
    share = run_ts.forming_share(X, mode, [tuple(p) for p in tr["formed_pairs_ts"]], [i for i, s in enumerate(syms) if s != "H"])
    print(f"      rxn 20 mode share from TS_imag_mode.xyz: {share:.3f}")
    assert mode.shape == X.shape and share >= 0.5


@test
def mae_ref_matches_summary():
    text = (Path(cfg["repo_root"]) / "analysis" / "espley_xtb_repro" / "results" / "SUMMARY.md").read_text()
    rows = {"ΔE‡ (barrier)": "barrier_kcal", "d1 (dipole strain)": "d1_kcal", "d2 (dipolarophile strain)": "d2_kcal",
            "elst": "elst_dft", "Pauli": "pauli_dft", "OI": "oi_dft", "CPCM": "cpcm_dft", "CDS": "cds_dft"}
    for label, key in rows.items():
        m = re.search(rf"^\| {re.escape(label)} \| \**([\d.]+)", text, re.M)
        assert m and float(m.group(1)) == float(cfg["prereg"]["MAE_ref"][key]), (label, m and m.group(1))
    assert "MAE 1.25" in text and float(cfg["prereg"]["MAE_ref"]["disp_dft"]) == 1.25


@test
def parsers_engrad_runtime_meta():
    import orca_direct as od, assemble as asm
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.engrad"
        p.write_text("#\n# Number of atoms\n#\n 2\n#\n# The current total energy in Eh\n#\n   -1.5\n#\n"
                     "# The current gradient in Eh/bohr\n#\n" + "".join(f"  {v}\n" for v in (0.1, 0, 0, -0.1, 0, 0)) +
                     "#\n# The atomic numbers and current coordinates in Bohr\n#\n   1  0.0 0.0 0.0\n   1  1.4 0.0 0.0\n")
        e, g = od.parse_engrad(p)
        assert e == -1.5 and g.shape == (2, 3) and g[0, 0] == 0.1 and g[1, 0] == -0.1
    c = asm.sp_coreh(PS / "J08" / "sp")
    print(f"      J08 sp core-h: {json.dumps({k: round(v, 4) if v else v for k, v in c.items()})}")
    assert abs(c["eda"] - (5 * 60 + 25.553) / 3600) < 1e-4 and c["total"] > c["eda"]
    assert pc.meta_rxn_id({"job_id": "J05"}) == 900005 and pc.meta_rxn_id({"job_id": "S0"}) == 999000
    assert pc.meta_rxn_id({"job_id": "V5_0020", "meta_rxn_id": "800003"}) == 800003
    assert pc.ade_name({"job_id": "V6c_J07", "ade_name": "J07"}) == "J07" and pc.ade_name({"job_id": "J01", "ade_name": ""}) == "J01"


@test
def fA_distance_gates_d0():
    """FIX F-A (from FIX_348bc3e.md): 4295 (1.537 A bond) passes, 3400 / 5783 (no bond < 3.3 A) fail."""
    import gate_audit_d0 as ga
    ga.init(None)
    for rid, want in ((4295, ""), (3400, "forming_bond_present"), (5783, "imag_mode_on_forming_bonds")):
        row = ga.one(dict(rxn_id=rid, status="x"))
        print(f"      rxn {rid}: hard_fail={row['hard_fail']!r}")
        assert (want == "" and row["hard_fail"] == "") or (want and want in row["hard_fail"]), (rid, row["hard_fail"])


@test
def fB_level_checks_positive_control():
    """FIX F-B: the level parser must see RIJCOSX, VWN-V and D3(BJ) in the pilot's own L0 OptTS output."""
    import orca_direct as od
    out = Path(cfg["val"]["pilot_optts_inp"]).with_suffix(".out")
    chk = od.level_checks(out.read_text(errors="replace"))
    print(f"      J10 L0: rijcosx={chk['rijcosx']!r} vwn={chk['vwn']!r} d3={chk['d3']} eps={chk['eps']!r} smd_cds={chk['smd_cds']}")
    assert chk["rijcosx"] == "on", chk["rijcosx"]
    assert od.VWN_V_RE.search(chk["vwn"] or ""), chk["vwn"]
    assert len(chk["d3"]) == 4, chk["d3"]
    lo = od.level_ok("L0", chk)
    assert lo["rijcosx_on"] is True and lo["vwn_ok"] is True, lo
    off = "RIJ-COSX (HFX calculated with COS-X)).... off"
    assert od.level_ok("L2", dict(chk, rijcosx="off", ri_lines=[off]))["no_rijcosx"] is True
    assert od.level_ok("L2", dict(chk))["no_rijcosx"] is False          # an L2 output that shows RIJCOSX on fails
    assert od.level_ok("L1", dict(chk, rijcosx=None, ri_lines=[]))["rijcosx_on"] is False


@test
def d1_engine_guard_rules():
    """D-1 guard: autodE 1.4.5 termination rules, synthetic lines."""
    import validation_report as vr
    assert vr._orca_ok(["x"] * 5 + ["                             ****ORCA TERMINATED NORMALLY****",
                                    "TOTAL RUN TIME: 0 days 0 hours 1 minutes 2 seconds 3 msec"])
    assert vr._orca_ok(["x"] * 5 + ["The optimization did not converge but reached the maximum number"])
    assert not vr._orca_ok(["x"] * 50), "killed mid-run must be abnormal"
    assert not vr._orca_ok([])
    assert vr._xtb_ok(["x"] * 30 + [" * finished run on 2026/09/29"])
    assert not vr._xtb_ok(["x"] * 30 + ["#ERROR! setup failed"])
    assert not vr._xtb_ok([])
    assert vr.ORCA_OUT.search("TS_abc_optts_orca.out") and vr.ORCA_OUT.search("r0_opt_orca1.out")
    assert vr.XTB_OUT.search("r0_conf0_opt_xtb.out") and not vr.ORCA_OUT.search("r0_conf0_opt_xtb.out")


@test
def d1_engine_guard_j03():
    """D-1 guard on the pilot J03 ts tree (read-only). Printed, not asserted: a non-empty ORCA list is a STOP."""
    import validation_report as vr
    e = vr.engine_scan(Path(cfg["pilot_scratch"]) / "jobs" / "J03" / "ts")
    print(f"      J03 ts tree: tree={e['tree']} ORCA outs {e['n_orca']} (abnormal: {e['orca_abnormal'] or 'none'}), "
          f"xtb outs {e['n_xtb']} (abnormal: {len(e['xtb_abnormal'])})")


n_fail = sum(not ok for _, ok in RESULTS)
print(f"\nUNIT TESTS {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'}: {len(RESULTS) - n_fail}/{len(RESULTS)}")
sys.exit(1 if n_fail else 0)
