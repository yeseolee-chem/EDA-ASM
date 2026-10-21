#!/usr/bin/env python3
"""D1-HF label assembly (docs/specs/D1HF.md section 4) and the S checks (section 5). Run through sbatch only.

  python build.py all [list.csv]     # every reaction of runs/dft and runs/g1 (or of the list) -> $HF/labels_d1_hf.json,
                                     #   results/labels_d1_hf.parquet, results/hf_status.csv
  python build.py s1                 # S-1: runs/dft_orig1 of the S-1 reactions vs their label.json (9 values,
                                     #   <= 0.01 kcal/mol) -> results/S1_compare.csv (exit 3 = STOP)
  python build.py s2                 # S-2: the S-2 reactions (runs/dft) -> results/S2_status.csv (exit 3 = STOP)
  python build.py s3 <job_id>        # S-3: the full EDA table of runs/dft/<job_id>/eda.out.gz -> results/S3_eda_table.txt

Formulas (Hartree column, x 627.5094740631; = label_true/scripts/stage3_parse.py, method 1):
  d1 = E(frag1_dist) - E(frag1_rel)   d2 = E(frag2_dist) - E(frag2_rel)
  eint_spe = E(eda) - E(frag1_dist) - E(frag2_dist)   barrier = E(eda) - E(frag1_rel) - E(frag2_rel)
  bond = Bond Energy   bsse_total = bond - eint_spe (spec sign)   bsse_shift_kcal = eint_spe - bond (label sign, = c_ghost)
  elst = Electrostatic Energy   pauli = Pauli Energy + the exchange / XC row (Delta E^0(XC) in the B3LYP table)
  oi = Orbital Energy   disp = Delta Dispersion   cpcm = Delta CPCM Dielectric   cds = Delta SMD CDS correction
A table row outside this list -> status eda_table_unexplained (spec 4: STOP at S).
Gates: g1_asm |barrier - (d1 + d2 + eint_spe)| < 1e-6; g2_sum |sum6 - bond| <= 0.11 and g3_seam |(d1 + d2 + sum6 -
barrier) - bsse_total| <= 0.11 (the label thresholds; the spec's 0.01 / 1e-3 are reported as g2_sum_spec / g3_seam_spec);
g_enuc = the label test (d1_run gates.py audit_gate): the real atoms of the ORCA-written eda_frag*.inp equal frag*_dist.inp
(elements in order, <= 1e-6 A; ORCA rounds eda_frag coordinates to 7 decimals, so ENuc itself differs by ~1e-6 Eh for the
same geometry and is reported only, enuc_diff_frag*_eh); g_bond_ref |E(eda) - E(eda_frag1) - E(eda_frag2) - bond|
<= 0.02 (label g_bond_reference); g_scf every SCF output has one 'SCF CONVERGED AFTER' and no non-convergence warning;
g_setting (inputs and all outputs: method, basis, SMD water); orca (every output ORCA 6.1.1, GIT 487d211c).
  python build.py check <list.csv>   # M checkpoint (spec 7): exit 3 if ok / finished < 90 %, any gate / table failure or
                                     #   any ORCA version / hash mismatch among the listed reactions
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_common as H  # noqa: E402

DP = H.DP
ROWS = {"Bond Energy": "bond", "Orbital Energy": "oi", "Electrostatic Energy": "elst", "Pauli Energy": "pauli_raw",
        "Delta Dispersion": "disp", "Delta CPCM Dielectric": "cpcm", "Delta SMD CDS correction": "cds"}
RE_XROW = re.compile(r"^Delta E\^0\((XC|X|Exchange)\)$")                  # the exchange / XC row (added to Pauli)
CH6 = ["elst", "pauli", "oi", "disp", "cpcm", "cds"]
LEVEL = {"hf": dict(method="Hartree-Fock(GTOs)", basis="6-31G*", words=("HF", "6-31G*")),
         "orig1": dict(method="DFT(GTOs)", basis="def2-TZVP", words=("B3LYP", "def2-TZVP"))}
S1_KEYS = dict(d1="d1_kcal", d2="d2_kcal", elst="elst_dft", pauli="pauli_dft", oi="oi_dft", disp="disp_dft",
               cpcm="cpcm_dft", cds="cds_dft", barrier="barrier_kcal")
S1_TOL = 0.01
RE_FRAG_ATOM = re.compile(r"^\s*([A-Z][a-z]?)\s*(:?)\s*\((\d)\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.M)
RE_SCF_BAD = re.compile(r"NOT FULLY CONVERGED|SCF NOT CONVERGED|did not converge", re.I)
BOND_REF_TOL = 0.02                                                       # label config_run.yaml bond_ref_tol
XYZ_TOL = 1e-6                                                            # A, = d1_run gates.py _same_geometry


def frag_real_atoms(inp_text, k):
    """real (non-ghost) atoms of an ORCA-written eda_frag<k>.inp ('F (1) x y z'; ghosts 'C :(2) x y z')."""
    return [(m.group(1), float(m.group(4)), float(m.group(5)), float(m.group(6)))
            for m in RE_FRAG_ATOM.finditer(inp_text) if not m.group(2) and int(m.group(3)) == k]


def same_geom(a, b, tol=XYZ_TOL):
    """elements in order and coordinates within tol (A)."""
    return len(a) == len(b) > 0 and all(x[0] == y[0] for x, y in zip(a, b)) and \
        float(np.abs(np.array([x[1:] for x in a]) - np.array([y[1:] for y in b])).max()) <= tol


def outputs(geom):
    return (list(H.CALCS) if geom == "dft" else ["eda"]) + ["eda_frag1", "eda_frag2"]


def setting_ok(inp_text, out_text, kind, level):
    """g_setting for one calculation: the input carries the level's method + basis words and SMD water (spec 2; a
    retry may add SlowConv / %scf MaxIter only) and the output reports that method, basis and SMD water."""
    lv = LEVEL[level]
    bad = []
    if inp_text is not None:
        p = H.parse_inp(inp_text, atoms=False)
        extra = {"SlowConv"}
        bang = [t for t in p["bang"] if t not in extra]
        if not all(w in bang for w in lv["words"]) or not H.smd_ok(p, kind):
            bad.append("input")
        if kind == "eda" and not all(all(w in p["frag"].get(f, "").split() for w in lv["words"]) for f in ("FRAG1", "FRAG2")):
            bad.append("frag_strings")
        other = set(LEVEL) - {level}
        if any(w in inp_text.split() or f'"{w}' in inp_text for o in other for w in LEVEL[o]["words"] if w not in lv["words"]):
            bad.append("other_level_words")
    m = H.RE_METHOD.search(out_text)
    if not m or m.group(1) != lv["method"]:
        bad.append(f"method:{m.group(1) if m else None}")
    b = H.RE_BASIS.search(out_text)
    if not b or b.group(1) != lv["basis"]:
        bad.append(f"basis:{b.group(1) if b else None}")
    eps, solv = H.RE_EPS.search(out_text), H.RE_SOLV.search(out_text)
    if ("utilizes the SMD solvation module" not in out_text or "SMD CDS free energy correction energy" not in out_text
            or not eps or abs(float(eps.group(1)) - H.EPS_WATER) > 1e-3 or not solv or solv.group(1).upper() != "WATER"):
        bad.append("smd_water")
    return bad


def parse_rxn(T, geom, level="hf"):
    """one reaction folder -> record (energies in kcal/mol, gates, status)."""
    run = geom if level == "hf" else f"{geom}_{level}"
    d = H.RUNS / run / T
    rec = dict(job_id=T, geom=geom, level=level)
    if (d / ".fail").is_file():
        return dict(rec, status="fail", reason=(d / ".fail").read_text().strip())
    if not (d / ".done").is_file():
        return dict(rec, status="not_done", reason="")
    txt = {c: H.read_text(d / f"{c}.out.gz") for c in outputs(geom)}
    inp = {c: (d / f"{c}.inp").read_text() if (d / f"{c}.inp").is_file() else None for c in outputs(geom)}
    E, enuc, ver, setting = {}, {}, set(), []
    for c, t in txt.items():
        f = H.RE_FSPE.findall(t)
        if len(f) != 1 or not H.terminated(t):
            return dict(rec, status="parse_fail", reason=f"{c}: {len(f)} FSPE / terminated {H.terminated(t)}")
        E[c] = float(f[0])
        en = H.RE_ENUC.findall(t)
        enuc[c] = float(en[0]) if en else np.nan
        ver.add((tuple(H.RE_VER.findall(t)[:1]), tuple(H.RE_GIT.findall(t)[:1])))
        setting += [f"{c}:{x}" for x in setting_ok(inp[c], t, "eda" if c == "eda" else "frag", level)]
    rows = H.eda_table(txt["eda"])
    if rows is None:
        return dict(rec, status="parse_fail", reason="EDA table not found exactly once")
    tab = dict(rows)
    unexplained = [r for r in tab if r not in ROWS and not RE_XROW.match(r)]
    missing = [r for r in ROWS if r not in tab]
    if unexplained or missing:
        return dict(rec, status="eda_table_unexplained" if unexplained else "parse_fail",
                    reason=f"rows not explained {unexplained}, missing {missing}", eda_rows=list(tab))
    xrows = [r for r in tab if RE_XROW.match(r)]
    v = {ROWS[r]: tab[r] * H.KCAL for r in ROWS}
    rec.update({f"eda_{k}_eh": tab[r] for r, k in ROWS.items()}, eda_xrow=";".join(xrows),
               eda_xrow_eh=sum(tab[r] for r in xrows) if xrows else 0.0)
    v["pauli"] = v.pop("pauli_raw") + sum(tab[r] for r in xrows) * H.KCAL
    rec.update({k: v[k] for k in ["bond"] + CH6})
    s6 = sum(v[k] for k in CH6)
    rec.update(sum6=s6, eda_sum_minus_bond=s6 - v["bond"],
               bond_reference_residual=(E["eda"] - E["eda_frag1"] - E["eda_frag2"]) * H.KCAL - v["bond"],
               e_eda_eh=E["eda"], e_eda_frag1_eh=E["eda_frag1"], e_eda_frag2_eh=E["eda_frag2"])
    g = dict(g2_sum=abs(s6 - v["bond"]) <= H.G2_TOL, g2_sum_spec=abs(s6 - v["bond"]) < H.G2_TOL_SPEC)
    if geom == "dft":
        d1 = (E["frag1_dist"] - E["frag1_rel"]) * H.KCAL
        d2 = (E["frag2_dist"] - E["frag2_rel"]) * H.KCAL
        eint = (E["eda"] - E["frag1_dist"] - E["frag2_dist"]) * H.KCAL
        bar = (E["eda"] - E["frag1_rel"] - E["frag2_rel"]) * H.KCAL
        bsse = v["bond"] - eint
        seam = (d1 + d2 + s6 - bar) - bsse
        rec.update(d1=d1, d2=d2, eint_spe=eint, barrier=bar, bsse_total=bsse, bsse_shift_kcal=eint - v["bond"],
                   identity_residual=d1 + d2 + eint - bar, seam_residual=seam,
                   **{f"e_{c}_eh": E[c] for c in H.CALCS[:4]})
        g.update(g1_asm=abs(bar - (d1 + d2 + eint)) < H.G1_TOL, g3_seam=abs(seam) <= H.G3_TOL,
                 g3_seam_spec=abs(seam) < H.G3_TOL_SPEC,
                 g_enuc=all(inp[f"eda_frag{k}"] is not None and inp[f"frag{k}_dist"] is not None and
                            same_geom(frag_real_atoms(inp[f"eda_frag{k}"], k),
                                      [(a[0],) + a[2:] for a in H.parse_inp(inp[f"frag{k}_dist"])["atoms"]])
                            for k in (1, 2)))
        rec.update({f"enuc_diff_frag{k}_eh": enuc[f"frag{k}_dist"] - enuc[f"eda_frag{k}"] for k in (1, 2)})
    g["g_bond_ref"] = abs(rec["bond_reference_residual"]) <= BOND_REF_TOL
    g["g_scf"] = all(t.count("SCF CONVERGED AFTER") == 1 and not RE_SCF_BAD.search(t) for t in txt.values())
    g["g_setting"] = not setting
    g["orca"] = ver == {((H.ORCA_VERSION,), (H.ORCA_GIT,))}
    rec.update(gates=g, setting_problems=";".join(setting), orca_versions=sorted(map(str, ver)))
    tim = pd.read_csv(d / "timing.tsv", sep="\t") if (d / "timing.tsv").is_file() else pd.DataFrame()
    if len(tim):
        sec = tim.groupby("calc").wall_seconds.sum()                     # serial: wall = core time
        rec.update({f"coreh_{c}": float(sec.get(c, np.nan)) / 3600 for c in (H.CALCS if geom == "dft" else ["eda"])})
        rec["coreh_total"] = float(sec.sum()) / 3600
        rec["n_attempts"] = int(len(tim))
    decide = [k for k in g if not k.endswith("_spec")]
    rec["status"] = "ok" if all(g[k] for k in decide) else "gate_fail"
    rec["reason"] = ";".join(k for k in decide if not g[k])
    return rec


def flat(r):
    out = {k: v for k, v in r.items() if k != "gates"}
    out.update({k: bool(v) for k, v in r.get("gates", {}).items()})
    return out


def cmd_all(lst=None):
    rows = []
    for geom in ("dft", "g1"):
        d = H.RUNS / geom
        have = set(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else set()
        if geom == "dft":                                  # every target (a target without a folder = not_started)
            ids = sorted(set(pd.read_csv(lst, dtype=str).job_id)) if lst else [t["job_id"] for t in H.targets()]
        else:                                              # g1: the folders of G1 runs
            ids = sorted(have & set(pd.read_csv(lst, dtype=str).job_id)) if lst else sorted(have)
        rows += [parse_rxn(T, geom) if T in have else dict(job_id=T, geom=geom, level="hf", status="not_started",
                                                         reason="") for T in ids]
    lab = {d["job_id"]: d for d in H.targets()}
    for r in rows:
        r["rxn_id"] = int(lab[r["job_id"]]["rxn_id"]) if r["job_id"] in lab else None
    ok = [r for r in rows if r["status"] == "ok"]
    H.write_json(H.LABELS_JSON, ok)
    df = pd.DataFrame([flat(r) for r in ok])
    H.write_atomic(H.RES / "labels_d1_hf.parquet", lambda t: df.to_parquet(t, index=False))
    st = pd.DataFrame([dict(job_id=r["job_id"], rxn_id=r["rxn_id"], geom=r["geom"], status=r["status"],
                            reason=r.get("reason", "")) for r in rows])
    H.write_atomic(H.RES / "hf_status.csv", lambda t: st.to_csv(t, index=False))
    summ = dict(n=len(rows), by_status=st.groupby(["geom", "status"]).size().reset_index().values.tolist(),
                labels_json_sha256=H.sha256(H.LABELS_JSON), parquet_sha256=H.sha256(H.RES / "labels_d1_hf.parquet"))
    print(json.dumps(summ, indent=1, default=str), flush=True)
    return 0


def cmd_check(lst):
    """spec 7 checkpoint over the listed reactions (dft): ok / finished >= 90 %, no gate_fail / table / parse failure,
    every finished output ORCA 6.1.1 + GIT 487d211c; exit 3 otherwise (report in results/M_checks/<list>.json)."""
    ids = sorted(set(pd.read_csv(lst, dtype=str).job_id))
    rows = [parse_rxn(T, "dft") for T in ids]
    st = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    fin = sum(v for k, v in st.items() if k not in ("not_done", "not_started"))
    n_ok = st.get("ok", 0)
    bad_gate = [r["job_id"] for r in rows if r["status"] in ("gate_fail", "eda_table_unexplained", "parse_fail")]
    bad_orca = [r["job_id"] for r in rows if r.get("gates") and not r["gates"].get("orca")]
    stop = []
    if fin and n_ok / fin < 0.90:
        stop.append(f"ok {n_ok}/{fin} < 90 %")
    if bad_gate:
        stop.append(f"gate / table / parse failures: {bad_gate[:10]}")
    if bad_orca:
        stop.append(f"ORCA version / hash mismatch: {bad_orca[:10]}")
    rep = dict(list=str(lst), n=len(ids), by_status=st, n_finished=fin, n_ok=n_ok, stop=stop)
    H.write_json(H.RES / "M_checks" / f"{Path(lst).stem}.json", rep)
    print(json.dumps(rep, indent=1), flush=True)
    return 3 if stop else 0


def sel(group):
    S = pd.read_csv(H.RES / "S_select.csv", dtype={"job_id": str})
    return S[S.group == group].job_id.tolist()


def cmd_s1():
    rows, bad = [], []
    for T in sel("S-1"):
        r = parse_rxn(T, "dft", "orig1")
        lj = json.loads((DP.D1_JOBS / T / "label.json").read_text())
        if r["status"] != "ok":
            bad.append(f"{T}: {r['status']} {r.get('reason')}")
        for k, lk in S1_KEYS.items():
            a, b = r.get(k, np.nan), float(lj[lk])
            rows.append(dict(job_id=T, value=k, run=a, label=b, diff=a - b, ok=bool(abs(a - b) <= S1_TOL)))
        rows.append(dict(job_id=T, value="status", run=r["status"], label="ok", diff=None, ok=r["status"] == "ok"))
    X = pd.DataFrame(rows)
    H.write_atomic(H.RES / "S1_compare.csv", lambda t: X.to_csv(t, index=False))
    print(X.to_string(index=False), flush=True)
    if bad or not X.ok.all():
        H.stop("S-1: the runner / build path does not reproduce the label within 0.01 kcal/mol:\n" + "\n".join(bad))
    return 0


def cmd_s2():
    rows = [flat(parse_rxn(T, "dft")) for T in sel("S-2")]
    X = pd.DataFrame(rows)
    H.write_atomic(H.RES / "S2_status.csv", lambda t: X.to_csv(t, index=False))
    print(X[[c for c in ("job_id", "status", "reason", "eda_xrow", "setting_problems", "coreh_total") if c in X]]
          .to_string(index=False), flush=True)
    return 0 if (X.status == "ok").all() else 3


def cmd_s3(T):
    t = H.read_text(H.RUNS / "dft" / T / "eda.out.gz")
    m = re.search(r"-+\n\s*Energy Decomposition Analysis\s*\n-+\n.*?\n-+\n(.*?)\n\s*\n-+", t, re.S)
    rows = H.eda_table(t)
    body = (m.group(0) if m else "(table block not found)") + "\n\nparsed rows (Hartree):\n" + \
        "\n".join(f"  {k:<32s}{v:>18.10f}  -> {ROWS.get(k, 'pauli (exchange / XC row)' if RE_XROW.match(k) else 'UNEXPLAINED')}"
                  for k, v in (rows or []))
    H.write_atomic(H.RES / "S3_eda_table.txt", f"# S-3: EDA table of runs/dft/{T}/eda.out.gz (HF / 6-31G*)\n\n{body}\n")
    print(body, flush=True)
    return 0 if rows and all(k in ROWS or RE_XROW.match(k) for k, _ in rows) else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["all"]:
        sys.exit(cmd_all(a[1] if len(a) > 1 else None))
    if a[:1] == ["check"] and len(a) == 2:
        sys.exit(cmd_check(a[1]))
    if a[:1] == ["s1"]:
        sys.exit(cmd_s1())
    if a[:1] == ["s2"]:
        sys.exit(cmd_s2())
    if a[:1] == ["s3"]:                                   # default: the first S-2 reaction (smallest with F)
        sys.exit(cmd_s3(a[1] if len(a) > 1 else sel("S-2")[0]))
    raise SystemExit(__doc__)
