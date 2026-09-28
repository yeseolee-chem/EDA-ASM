#!/usr/bin/env python3
"""Verify three consistency conditions on all completed SMD-relabel outputs.

Uses new eda.out / eda_frag1.out / eda_frag2.out from scratch dir + existing
frag*_rel.out from label_true/work/inputs/ (rel geometries unchanged — those
already had SMD from the original run).

Checks per rxn:
  (1) ASM identity: |d1 + d2 + eint_spe - barrier| < 1e-4 kcal/mol (should be exact)
  (2) BSSE-SMD consistency: |eint_spe - e_bond| < 0.01 kcal/mol
  (3) 8-channel closure: |d1 + d2 + Σ(6ch) - barrier| < 0.03 kcal/mol
"""
import re
import sys
from pathlib import Path

EH_TO_KCAL = 627.5094740631
SCRATCH = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")
OLD_INPUTS = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/inputs")
STATE_DONE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/state/done")

FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
EDA_TABLE_RE = re.compile(r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.DOTALL)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
LABEL_TO_KEY = {
    "Bond Energy": "bond", "Orbital Energy": "orb",
    "Electrostatic Energy": "elst", "Pauli Energy": "pauli",
    "Delta E^0(XC)": "xc", "Delta Dispersion": "disp",
    "Delta CPCM Dielectric": "cpcm", "Delta SMD CDS correction": "smd_cds",
}


def fspe(p):
    m = FSPE_RE.findall(Path(p).read_text(errors="replace"))
    return float(m[-1]) if m else None


def eda_table(p):
    text = Path(p).read_text(errors="replace")
    m = EDA_TABLE_RE.search(text)
    if not m:
        return None
    out = {}
    for row in ROW_RE.finditer(m.group(1)):
        lab = row.group(1).strip()
        if lab in LABEL_TO_KEY:
            out[LABEL_TO_KEY[lab]] = float(row.group(2))
    return out


def check_rxn(rid_str):
    scratch_dir = SCRATCH / rid_str
    old_dir = OLD_INPUTS / rid_str
    try:
        e_ab = fspe(scratch_dir / "eda.out")
        e_f1_dist = fspe(scratch_dir / "eda_frag1.out")
        e_f2_dist = fspe(scratch_dir / "eda_frag2.out")
        e_f1_rel = fspe(old_dir / "frag1_rel.out")
        e_f2_rel = fspe(old_dir / "frag2_rel.out")
        tab = eda_table(scratch_dir / "eda.out")
        if None in (e_ab, e_f1_dist, e_f2_dist, e_f1_rel, e_f2_rel) or tab is None:
            return {"rxn": rid_str, "err": "missing FSPE or EDA table"}
    except FileNotFoundError as e:
        return {"rxn": rid_str, "err": f"file not found: {e}"}

    d1 = (e_f1_dist - e_f1_rel) * EH_TO_KCAL
    d2 = (e_f2_dist - e_f2_rel) * EH_TO_KCAL
    eint_spe = (e_ab - e_f1_dist - e_f2_dist) * EH_TO_KCAL
    barrier = (e_ab - e_f1_rel - e_f2_rel) * EH_TO_KCAL
    e_bond = tab["bond"] * EH_TO_KCAL
    six_sum = ((tab["elst"] + tab["pauli"] + tab["xc"] + tab["orb"]
                + tab["disp"] + tab["cpcm"] + tab["smd_cds"]) * EH_TO_KCAL)

    return {
        "rxn": rid_str,
        "asm_res": d1 + d2 + eint_spe - barrier,
        "bsse_res": eint_spe - e_bond,
        "eight_res": d1 + d2 + six_sum - barrier,
        "six_vs_bond": six_sum - e_bond,
        "d1": d1, "d2": d2, "eint_spe": eint_spe, "barrier": barrier, "e_bond": e_bond,
    }


def main():
    done_ids = sorted(p.name for p in STATE_DONE.iterdir())
    print(f"Verifying {len(done_ids)} completed rxns...")
    results, errors = [], []
    for rid in done_ids:
        r = check_rxn(rid)
        if "err" in r:
            errors.append(r)
        else:
            results.append(r)

    n = len(results)
    if n == 0:
        print("no successful reads"); return
    asm_ok  = sum(1 for r in results if abs(r["asm_res"])   < 1e-4)
    bsse_ok = sum(1 for r in results if abs(r["bsse_res"])  < 0.01)
    six_ok  = sum(1 for r in results if abs(r["six_vs_bond"]) < 0.02)
    eight_ok= sum(1 for r in results if abs(r["eight_res"]) < 0.03)
    all3    = sum(1 for r in results if abs(r["asm_res"])<1e-4 and abs(r["bsse_res"])<0.01 and abs(r["eight_res"])<0.03)

    def stat(key, label):
        vs = [abs(r[key]) for r in results]
        vs.sort()
        p50 = vs[n//2]; p99 = vs[int(n*0.99)]; mx = vs[-1]
        print(f"  |{label:20s}|  p50={p50:.2e}  p99={p99:.2e}  max={mx:.2e}")

    print(f"\nResults on {n} rxns (errors={len(errors)}):")
    print(f"  (1) ASM  d1+d2+eint_spe = barrier   pass: {asm_ok}/{n} ({100*asm_ok/n:.2f}%)")
    print(f"  (2) BSSE  eint_spe = e_bond          pass: {bsse_ok}/{n} ({100*bsse_ok/n:.2f}%)")
    print(f"  (3) 8ch  d1+d2+6ch = barrier         pass: {eight_ok}/{n} ({100*eight_ok/n:.2f}%)")
    print(f"       aux 6ch = e_bond                pass: {six_ok}/{n} ({100*six_ok/n:.2f}%)")
    print(f"  ALL THREE (strict tols) pass: {all3}/{n} ({100*all3/n:.2f}%)")
    print("\nresidual distribution (abs):")
    stat("asm_res", "d1+d2+eint-barrier")
    stat("bsse_res", "eint_spe - e_bond")
    stat("eight_res", "d1+d2+6ch-barrier")
    stat("six_vs_bond", "6ch - e_bond")

    # Top 10 worst offenders on 8ch closure
    print("\ntop-10 worst by |8ch closure residual|:")
    for r in sorted(results, key=lambda r: -abs(r["eight_res"]))[:10]:
        print(f"  {r['rxn']}  8ch_res={r['eight_res']:+.4f}  bsse_res={r['bsse_res']:+.2e}  asm_res={r['asm_res']:+.2e}")

    if errors:
        print(f"\n{len(errors)} errors (first 5):")
        for e in errors[:5]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
