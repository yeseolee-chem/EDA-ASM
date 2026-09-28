#!/usr/bin/env python3
"""Full consistency audit of the SMD relabel (all 5,265 rxns of label_true).

Per rxn that has a local done flag, checks:
  files   : eda.inp / eda.out / eda_frag{1,2}.{inp,out} in scratch, and
            frag{1,2}_rel.{inp,out} + old eda.out / frag{1,2}_dist.out in
            label_true; every .out terminated normally with an FSPE; no
            "NOT CONVERGED" message; exactly one EDA table
  input   : new eda.inp == rewrite(old eda.inp) (CPCM(water) -> SMD(water),
            %cpcm block dropped); ORCA's auto-generated eda_frag*.inp carry
            SMD(water); fragment atom / ghost counts match eda.inp; fragment
            charge / multiplicity and composition match frag*_rel.inp
  level   : SMD CDS term present in eda.out, eda_frag*.out, frag*_rel.out;
            new eda.out FSPE == old eda.out FSPE (supermolecule unchanged)
  gates   : (1) ASM   |d1 + d2 + eint_spe - barrier|   < 1e-4 kcal/mol
                      (an algebraic identity: a parse sanity check only)
            (2) BSSE  |eint_spe - e_bond|              < 0.01 (stage3: 0.02)
                      (proves eda_frag*.out are the references ORCA's Bond
                      Energy was built from)
            (3) 8ch   |d1 + d2 + sum(6ch) - barrier|   < 0.03 (stage3 6ch: 0.02)
                      (given (2), this is ORCA's own EDA table closure)
            aux       |sum(7 table rows) - Bond Energy| in hartree
  sanity  : ranges of barrier / d1 / d2 / eint_spe (flags only)

v2 (2026-09-28, cross-check review): exactly one FSPE per file (stage3 rule);
eda_frag*.inp charge/multiplicity and real-atom coordinates vs eda.inp; method
lines of eda_frag*.inp and frag*_rel.inp vs the FRAG strings; raw energies and
table values in the CSV; stage3-equivalent status; rates without the excluded
5; supermolecule difference is a warning (hard problem only > 1e-3 Eh); the
eda_frag vs standalone-fragment shift is split into its dielectric (shared AB
cavity), CDS and electronic (ghost basis) parts, and the cavity convention is
checked (eda_frag cavity == eda.out cavity).

Writes a per-rxn CSV and a summary to $ROOT/audit/.
"""
import csv
import re
import sys
from multiprocessing import Pool
from pathlib import Path

EH = 627.5094740631
ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel")
NEW = ROOT / "inputs"
DONE = ROOT / "state" / "done"
OLD = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/inputs")
OUT_DIR = ROOT / "audit"
EXCLUDED = {"rxn_3090", "rxn_3766", "rxn_4252", "rxn_3400", "rxn_5783"}

FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
EDA_TABLE_RE = re.compile(r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.DOTALL)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
NOTCONV_RE = re.compile(r"NOT CONVERGED|SCF CONVERGENCE FAILED", re.IGNORECASE)
CDS_RE = re.compile(r"SMD CDS free energy correction energy")
CPCM_BLOCK_RE = re.compile(r"^%cpcm\s*\n(?:.*?\n)*?^end\s*\n", re.MULTILINE)
XYZ_HDR_RE = re.compile(r"^\s*\*\s*xyz\s+(-?\d+)\s+(\d+)", re.MULTILINE | re.IGNORECASE)
ATOM_RE = re.compile(r"^\s*([A-Z][a-z]?)\s*(:)?\s*\((\d)\)\s+-?\d", re.MULTILINE)
ATOM_XYZ_RE = re.compile(r"^\s*([A-Z][a-z]?)\s*(:)?\s*\((\d)\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", re.MULTILINE)
METHOD_RE = re.compile(r"^\s*!\s*(.+?)\s*$", re.MULTILINE)
FRAG_STR_RE = {f: re.compile(rf'FRAG{f}\s+"([^"]+)"') for f in ("1", "2")}
DIEL_RE = re.compile(r"CPCM Dielectric\s*:\s*(-?\d+\.\d+)\s*Eh")
GCDS_RE = re.compile(r"SMD CDS \(Gcds\)\s*:\s*(-?\d+\.\d+)\s*Eh")
CAV_RE = re.compile(r"Cavity Volume\s*\.\.\.\s*(\d+\.\d+)")
PLAIN_ATOM_RE = re.compile(r"^\s*([A-Z][a-z]?)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*$", re.MULTILINE)
DIST_XYZ_RE = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
LABELS = {
    "Bond Energy": "bond", "Orbital Energy": "orb",
    "Electrostatic Energy": "elst", "Pauli Energy": "pauli",
    "Delta E^0(XC)": "xc", "Delta Dispersion": "disp",
    "Delta CPCM Dielectric": "cpcm", "Delta SMD CDS correction": "cds",
}

TOL_ASM, TOL_BSSE, TOL_8CH = 1e-4, 0.01, 0.03
# outputs that feed the labels (option ①: d1/d2/eint_spe from frag*_dist.out)
CORE = ("eda", "ef1", "ef2", "rel1", "rel2", "sd1", "sd2")
TOL_STAGE3 = 0.02
TOL_SUPER_EH = 1e-5           # warning: SCF-convergence-level difference
TOL_SUPER_HARD_EH = 1e-3      # problem: would indicate a changed model
TOL_XYZ = 1e-5


def rewrite(text):
    return CPCM_BLOCK_RE.sub("", text.replace("CPCM(water)", "SMD(water)"))


def lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def read(p):
    try:
        return Path(p).read_text(errors="replace")
    except OSError:
        return None


def out_info(text):
    """(terminated, last FSPE or None, FSPE count, not-converged hits, has SMD CDS)."""
    if text is None:
        return False, None, 0, 0, False
    m = FSPE_RE.findall(text)
    return ("ORCA TERMINATED NORMALLY" in text, float(m[-1]) if m else None, len(m),
            len(NOTCONV_RE.findall(text)), bool(CDS_RE.search(text)))


def last(rx, text):
    m = rx.findall(text or "")
    return float(m[-1]) if m else None


def tokens(s):
    return s.replace("CPCM(water)", "SMD(water)").split()


def check(rid):
    n, o = NEW / rid, OLD / rid
    r = {"rxn": rid, "excluded5": rid in EXCLUDED, "problems": []}
    P = r["problems"]

    # ---- files + termination
    t = {}
    for key, path in (("eda", n / "eda.out"), ("ef1", n / "eda_frag1.out"), ("ef2", n / "eda_frag2.out"),
                      ("rel1", o / "frag1_rel.out"), ("rel2", o / "frag2_rel.out"),
                      ("old_eda", o / "eda.out"), ("sd1", o / "frag1_dist.out"), ("sd2", o / "frag2_dist.out")):
        t[key] = read(path)
        term, e, nf, nc, cds = out_info(t[key])
        r[f"E_{key}"] = e
        r[f"diel_{key}"] = last(DIEL_RE, t[key])
        r[f"gcds_{key}"] = last(GCDS_RE, t[key])
        r[f"cav_{key}"] = last(CAV_RE, t[key])
        if t[key] is None:
            P.append(f"missing:{'new' if path.parent == n else 'old'}/{path.name}")
            continue
        if key in CORE and not term:
            P.append(f"not_terminated:{key}")
        if key in CORE and nf != 1:
            P.append(f"fspe_count={nf}:{key}")
        if e is None and (key in CORE or key == "old_eda"):
            P.append(f"no_fspe:{key}")
        if nc and key in CORE:
            P.append(f"not_converged:{key}")
        if key in CORE and not cds:
            P.append(f"no_smd_cds:{key}")

    # ---- input integrity
    inp_new, inp_old = read(n / "eda.inp"), read(o / "eda.inp")
    if inp_new is None or inp_old is None:
        P.append("missing:eda.inp(new or old)")
    else:
        r["inp_exact"] = inp_new == rewrite(inp_old)
        r["inp_semantic"] = lines(inp_new) == lines(rewrite(inp_old))
        if not r["inp_semantic"]:
            P.append("eda_inp_differs_from_rewrite")
        if "CPCM(water)" in inp_new or "%cpcm" in inp_new.lower():
            P.append("eda_inp_still_has_cpcm")
        if inp_new.count("SMD(water)") != 3:
            P.append(f"eda_inp_smd_count={inp_new.count('SMD(water)')}")
        cm = {k: re.search(rf"{k}\s+(-?\d+)", inp_new) for k in ("FRAG1_C", "FRAG1_M", "FRAG2_C", "FRAG2_M")}
        atoms = ATOM_RE.findall(inp_new)
        comp = {"1": sorted(a[0] for a in atoms if a[2] == "1"),
                "2": sorted(a[0] for a in atoms if a[2] == "2")}
        xyz_new = [(a[0], a[2], float(a[3]), float(a[4]), float(a[5])) for a in ATOM_XYZ_RE.findall(inp_new)]
        for f, g in (("1", "2"), ("2", "1")):
            fstr = FRAG_STR_RE[f].search(inp_new)
            c, m = cm[f"FRAG{f}_C"], cm[f"FRAG{f}_M"]
            fi = read(n / f"eda_frag{f}.inp")
            if fi is None:
                P.append(f"missing:eda_frag{f}.inp")
            else:
                if "SMD(water)" not in fi:
                    P.append(f"eda_frag{f}_inp_no_smd")
                fa = ATOM_RE.findall(fi)
                real = sorted(a[0] for a in fa if not a[1] and a[2] == f)
                ghost = sorted(a[0] for a in fa if a[1] and a[2] == g)
                if real != comp[f] or ghost != comp[g] or len(fa) != len(atoms):
                    P.append(f"eda_frag{f}_atoms_mismatch")
                h = XYZ_HDR_RE.search(fi)
                if not (h and c and m and h.group(1) == c.group(1) and h.group(2) == m.group(1)):
                    P.append(f"eda_frag{f}_inp_charge_mult_mismatch")
                mf = METHOD_RE.search(fi)
                if not (mf and fstr and tokens(mf.group(1)) == tokens(fstr.group(1))):
                    P.append(f"eda_frag{f}_inp_method_mismatch")
                # every atom (real and ghost) at the same position as in eda.inp
                fx = [(a[0], a[2], float(a[3]), float(a[4]), float(a[5])) for a in ATOM_XYZ_RE.findall(fi)]
                same = len(fx) == len(xyz_new) and all(
                    p[0] == q[0] and p[1] == q[1] and max(abs(p[i] - q[i]) for i in (2, 3, 4)) < TOL_XYZ
                    for p, q in zip(fx, xyz_new))
                if not same:
                    P.append(f"eda_frag{f}_coords_mismatch")
            # standalone distorted fragment: same level as the FRAG string, same
            # charge/multiplicity, and exactly the TS coordinates of fragment f
            di = read(o / f"frag{f}_dist.inp")
            if di is None:
                P.append(f"missing:frag{f}_dist.inp")
            else:
                md = METHOD_RE.search(di)
                if not (md and fstr and tokens(md.group(1)) == tokens(fstr.group(1))):
                    P.append(f"frag{f}_dist_method_mismatch")
                if "CPCM(water)" in di and not re.search(r"^%cpcm\s*\n(?:.*\n)*?\s*smd\s+true\s*\n(?:.*\n)*?\s*smdsolvent\s+\"water\"",
                                                         di, re.MULTILINE | re.IGNORECASE):
                    P.append(f"frag{f}_dist_no_smd_block")
                h = XYZ_HDR_RE.search(di)
                if not (h and c and m and h.group(1) == c.group(1) and h.group(2) == m.group(1)):
                    P.append(f"frag{f}_dist_charge_mult_mismatch")
                dx = sorted((a[0], round(float(a[1]), 5), round(float(a[2]), 5), round(float(a[3]), 5))
                            for a in DIST_XYZ_RE.findall(di))
                tx = sorted((a[0], round(a[2], 5), round(a[3], 5), round(a[4], 5)) for a in xyz_new if a[1] == f)
                if not dx or dx != tx:
                    P.append(f"frag{f}_dist_coords_not_TS_fragment")
            ri = read(o / f"frag{f}_rel.inp")
            if ri is None:
                P.append(f"missing:frag{f}_rel.inp")
                continue
            h = XYZ_HDR_RE.search(ri)
            if not (h and c and m and h.group(1) == c.group(1) and h.group(2) == m.group(1)):
                P.append(f"frag{f}_rel_charge_mult_mismatch")
            if sorted(PLAIN_ATOM_RE.findall(ri)) != comp[f]:
                P.append(f"frag{f}_rel_composition_mismatch")
            # rel level == fragment EDA level: same method line (CPCM(water) + %cpcm smd
            # block counts as SMD(water)), and the smd block really sets water
            mr = METHOD_RE.search(ri)
            smd_block = re.search(r"^%cpcm\s*\n(?:.*\n)*?\s*smd\s+true\s*\n(?:.*\n)*?\s*smdsolvent\s+\"water\"", ri,
                                  re.MULTILINE | re.IGNORECASE)
            if not (mr and fstr and tokens(mr.group(1)) == tokens(fstr.group(1))):
                P.append(f"frag{f}_rel_method_mismatch")
            if "CPCM(water)" in ri and not smd_block:
                P.append(f"frag{f}_rel_no_smd_block")

    # ---- EDA table
    tabs = EDA_TABLE_RE.findall(t["eda"] or "")
    r["n_eda_tables"] = len(tabs)
    if len(tabs) != 1:
        P.append(f"eda_tables={len(tabs)}")
    tab = {}
    if tabs:
        for row in ROW_RE.finditer(tabs[-1]):
            if row.group(1).strip() in LABELS:
                tab[LABELS[row.group(1).strip()]] = float(row.group(2))
    if set(tab) != set(LABELS.values()):
        P.append(f"eda_table_rows={sorted(tab)}")

    e = {k: r.get(f"E_{k}") for k in ("eda", "ef1", "ef2", "rel1", "rel2", "old_eda", "sd1", "sd2")}
    if None in (e["eda"], e["ef1"], e["ef2"], e["rel1"], e["rel2"]) or len(tab) != 8:
        r["gates_computable"] = False
        return r
    r["gates_computable"] = True

    # ---- labels + gates
    d1 = (e["ef1"] - e["rel1"]) * EH
    d2 = (e["ef2"] - e["rel2"]) * EH
    eint = (e["eda"] - e["ef1"] - e["ef2"]) * EH
    barrier = (e["eda"] - e["rel1"] - e["rel2"]) * EH
    rows_eh = tab["orb"] + tab["elst"] + tab["pauli"] + tab["xc"] + tab["disp"] + tab["cpcm"] + tab["cds"]
    six = rows_eh * EH
    r.update(d1=d1, d2=d2, eint_spe=eint, barrier=barrier, e_bond=tab["bond"] * EH,
             pauli_dft=(tab["pauli"] + tab["xc"]) * EH, elst=tab["elst"] * EH, oi=tab["orb"] * EH,
             disp=tab["disp"] * EH, cpcm=tab["cpcm"] * EH, cds=tab["cds"] * EH)
    # option ① (the labels written to labels_all.json): standalone distorted fragments
    if e["sd1"] is not None and e["sd2"] is not None:
        d1s, d2s = (e["sd1"] - e["rel1"]) * EH, (e["sd2"] - e["rel2"]) * EH
        eints = (e["eda"] - e["sd1"] - e["sd2"]) * EH
        bsse = eints - tab["bond"] * EH
        r.update(d1_std=d1s, d2_std=d2s, eint_spe_std=eints, bsse_shift=bsse,
                 asm_res_std=d1s + d2s + eints - barrier,
                 closure_std=d1s + d2s + six + bsse - barrier)
    r["asm_res"] = d1 + d2 + eint - barrier
    r["bsse_res"] = eint - tab["bond"] * EH
    r["eight_res"] = d1 + d2 + six - barrier
    r["rows_minus_bond_eh"] = rows_eh - tab["bond"]
    r["g1"] = abs(r["asm_res"]) < TOL_ASM
    r["g2"] = abs(r["bsse_res"]) < TOL_BSSE
    r["g3"] = abs(r["eight_res"]) < TOL_8CH
    r["g2_stage3"] = abs(r["bsse_res"]) < TOL_STAGE3
    r["g_rows_stage3"] = abs(r["rows_minus_bond_eh"] * EH) < TOL_STAGE3
    r["all3"] = r["g1"] and r["g2"] and r["g3"]

    # ---- level / sanity (flags)
    flags = []
    if e["old_eda"] is not None:
        r["super_diff_eh"] = e["eda"] - e["old_eda"]
        if abs(r["super_diff_eh"]) > TOL_SUPER_HARD_EH:
            P.append("supermolecule_energy_changed_hard")
        elif abs(r["super_diff_eh"]) > TOL_SUPER_EH:
            flags.append("supermolecule_scf_noise")
    # eda_frag (ghost basis, AB cavity) vs standalone fragment at the same geometry:
    # total = dielectric (cavity) + CDS + electronic (counterpoise) parts
    for f in ("1", "2"):
        ef, sd = f"ef{f}", f"sd{f}"
        if e[sd] is None:
            continue
        tot = (e[ef] - e[sd]) * EH
        r[f"ghost_ref_shift{f}"] = tot
        if r.get(f"diel_{ef}") is not None and r.get(f"diel_{sd}") is not None:
            diel = (r[f"diel_{ef}"] - r[f"diel_{sd}"]) * EH
            cds = ((r.get(f"gcds_{ef}") or 0.0) - (r.get(f"gcds_{sd}") or 0.0)) * EH
            r[f"shift{f}_dielectric"] = diel
            r[f"shift{f}_cds"] = cds
            r[f"shift{f}_electronic"] = tot - diel - cds
        # convention: ORCA builds the fragment cavity of eda_frag*.out from the AB atoms
        if r.get(f"cav_{ef}") is not None and r.get("cav_eda") is not None \
                and abs(r[f"cav_{ef}"] - r["cav_eda"]) > 1e-3:
            flags.append(f"eda_frag{f}_cavity_not_AB")
    if not -40 < barrier < 90: flags.append("barrier_range")
    s1, s2 = r.get("d1_std", d1), r.get("d2_std", d2)      # option ① deformation energies
    if s1 < -2 or s2 < -2: flags.append("negative_strain")
    if s1 > 150 or s2 > 150: flags.append("huge_strain")
    r["flags"] = flags
    # same order as stage3_parse.parse_one_rxn
    r["stage3_status"] = ("eda_sum_mismatch" if not r["g_rows_stage3"] else
                          "bsse_reference_mismatch" if not r["g2_stage3"] else "ok")
    for k, v in tab.items():
        r[f"tab_{k}_eh"] = v
    return r


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.2f}%)" if b else f"{a}/0"


def dist(vals):
    v = sorted(abs(x) for x in vals)
    if not v:
        return "n=0"
    return f"p50={v[len(v)//2]:.2e} p99={v[min(len(v)-1, int(len(v)*0.99))]:.2e} max={v[-1]:.2e}"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_ids = sorted(p.name for p in OLD.iterdir() if re.fullmatch(r"rxn_\d+", p.name))
    local = {p.name for p in NEW.iterdir() if re.fullmatch(r"rxn_\d+", p.name)}
    done = {p.name for p in DONE.iterdir()}
    extra_local = sorted(local - set(all_ids))
    done_not_local = sorted(done - local)
    todo = sorted(set(all_ids) & local & done)
    outstanding = sorted(set(all_ids) - local)
    local_not_done = sorted((set(all_ids) & local) - done)

    with Pool(8) as pool:
        res = pool.map(check, todo, chunksize=16)

    cols = ["rxn", "excluded5", "gates_computable", "all3", "g1", "g2", "g3", "g2_stage3", "g_rows_stage3",
            "stage3_status", "asm_res", "bsse_res", "eight_res", "rows_minus_bond_eh", "super_diff_eh",
            "d1", "d2", "eint_spe", "barrier", "e_bond", "pauli_dft", "elst", "oi", "disp", "cpcm", "cds",
            "d1_std", "d2_std", "eint_spe_std", "bsse_shift", "asm_res_std", "closure_std",
            "ghost_ref_shift1", "shift1_dielectric", "shift1_cds", "shift1_electronic",
            "ghost_ref_shift2", "shift2_dielectric", "shift2_cds", "shift2_electronic",
            "E_eda", "E_ef1", "E_ef2", "E_rel1", "E_rel2", "E_old_eda", "E_sd1", "E_sd2",
            "tab_bond_eh", "tab_orb_eh", "tab_elst_eh", "tab_pauli_eh", "tab_xc_eh", "tab_disp_eh",
            "tab_cpcm_eh", "tab_cds_eh", "cav_eda", "cav_ef1", "cav_ef2", "cav_rel1", "cav_rel2",
            "inp_exact", "inp_semantic", "n_eda_tables", "problems", "flags"]
    csv_path = OUT_DIR / "audit_full.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in res:
            row = dict(r)
            row["problems"] = ";".join(r["problems"])
            row["flags"] = ";".join(r.get("flags", []))
            w.writerow(row)

    g = [r for r in res if r.get("gates_computable")]
    n = len(g)
    probs = {}
    for r in res:
        for p in r["problems"]:
            probs.setdefault(p, []).append(r["rxn"])
    flags = {}
    for r in g:
        for f in r["flags"]:
            flags.setdefault(f, []).append(r["rxn"])

    L = []
    L.append(f"label_true rxns: {len(all_ids)}   local: {len(local & set(all_ids))}   done+local: {len(todo)}")
    L.append(f"outstanding (not local, still in bundles): {len(outstanding)}   local not done: {len(local_not_done)} {local_not_done[:10]}")
    L.append(f"extra local dirs not in label_true: {extra_local}   done flags without local dir: {done_not_local[:10]}")
    L.append("")
    L.append(f"gates computable: {pct(n, len(res))}")
    L.append(f"  (1) ASM  |d1+d2+eint_spe-barrier| < {TOL_ASM}:   {pct(sum(r['g1'] for r in g), n)}")
    L.append(f"  (2) BSSE |eint_spe-e_bond| < {TOL_BSSE}:          {pct(sum(r['g2'] for r in g), n)}   (stage3 0.02: {pct(sum(r['g2_stage3'] for r in g), n)})")
    L.append(f"  (3) 8ch  |d1+d2+6ch-barrier| < {TOL_8CH}:        {pct(sum(r['g3'] for r in g), n)}")
    L.append(f"  aux |7 rows - Bond| < 0.02 kcal (stage3):        {pct(sum(r['g_rows_stage3'] for r in g), n)}")
    L.append(f"  ALL THREE:                                       {pct(sum(r['all3'] for r in g), n)}")
    gx = [r for r in g if not r["excluded5"]]
    L.append(f"  ALL THREE without the excluded 5:                {pct(sum(r['all3'] for r in gx), len(gx))}")
    L.append(f"  excluded-5 rxns among done: {[ (r['rxn'], r.get('all3')) for r in res if r['excluded5']]}")
    L.append("  note: (1) is an algebraic identity (parse sanity); (2) shows eda_frag*.out are the references of")
    L.append("        ORCA's Bond Energy; given (2), (3) is ORCA's own EDA-table closure (rows vs Bond Energy)")
    st = {}
    for r in g:
        st[r["stage3_status"]] = st.get(r["stage3_status"], 0) + 1
    stx = {}
    for r in gx:
        stx[r["stage3_status"]] = stx.get(r["stage3_status"], 0) + 1
    L.append(f"  stage3-equivalent status (0.02 kcal): {st}   without excluded 5: {stx}")
    L.append("")
    L.append("residuals (abs):")
    L.append(f"  asm_res  kcal  {dist([r['asm_res'] for r in g])}")
    L.append(f"  bsse_res kcal  {dist([r['bsse_res'] for r in g])}")
    L.append(f"  8ch_res  kcal  {dist([r['eight_res'] for r in g])}")
    L.append(f"  rows-bond Eh   {dist([r['rows_minus_bond_eh'] for r in g])}")
    L.append(f"  option ①  |d1+d2+eint_spe-barrier| kcal         {dist([r['asm_res_std'] for r in g if 'asm_res_std' in r])}")
    L.append(f"  option ①  |d1+d2+6ch+bsse_shift-barrier| kcal  {dist([r['closure_std'] for r in g if 'closure_std' in r])}")
    sd = [r['super_diff_eh'] for r in g if r.get('super_diff_eh') is not None]
    L.append(f"  new-old supermolecule FSPE Eh  {dist(sd)}  (n={len(sd)})")
    L.append(f"  input exact rewrite match: {pct(sum(bool(r.get('inp_exact')) for r in res), len(res))}   semantic: {pct(sum(bool(r.get('inp_semantic')) for r in res), len(res))}")
    L.append("")
    L.append("value ranges (kcal/mol):")
    for k in ("barrier", "d1_std", "d2_std", "eint_spe_std", "bsse_shift", "d1", "d2", "eint_spe", "e_bond",
              "pauli_dft", "elst", "oi", "disp", "cpcm", "cds", "ghost_ref_shift1", "shift1_dielectric", "shift1_cds", "shift1_electronic",
              "ghost_ref_shift2", "shift2_dielectric", "shift2_cds", "shift2_electronic"):
        v = sorted(r[k] for r in g if r.get(k) is not None)
        if v:
            L.append(f"  {k:12s} min={v[0]:9.3f} p1={v[int(len(v)*0.01)]:9.3f} p50={v[len(v)//2]:9.3f} p99={v[min(len(v)-1,int(len(v)*0.99))]:9.3f} max={v[-1]:9.3f}")
    L.append("")
    L.append(f"problems ({sum(1 for r in res if r['problems'])} rxns with any):")
    for k, v in sorted(probs.items(), key=lambda kv: -len(kv[1])):
        L.append(f"  {k}: {len(v)}  {v[:12]}")
    L.append(f"sanity flags ({sum(1 for r in g if r['flags'])} rxns with any):")
    for k, v in sorted(flags.items(), key=lambda kv: -len(kv[1])):
        L.append(f"  {k}: {len(v)}  {v[:12]}")
    L.append("")
    L.append("gate failures (any of 1/2/3):")
    for r in sorted((r for r in g if not r["all3"]), key=lambda r: -abs(r["eight_res"]))[:60]:
        L.append(f"  {r['rxn']}  asm={r['asm_res']:+.2e} bsse={r['bsse_res']:+.4f} 8ch={r['eight_res']:+.4f} rows-bond={r['rows_minus_bond_eh']:+.1e}Eh")
    text = "\n".join(L)
    (OUT_DIR / "audit_summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())
