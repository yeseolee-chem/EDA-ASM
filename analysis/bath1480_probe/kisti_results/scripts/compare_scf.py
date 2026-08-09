#!/usr/bin/env python3
"""Difference the seven EDA channels between TightSCF and VeryTightSCF runs.

Answers reviewer point 4: 72% of production runs converge on ORCA's ENERGY
criterion while density/DIIS/orbital-gradient are still loose. The total energy
is second order in the density error; the ETS channels are first order. In this
system Pauli (+237) cancels against the attractive terms to leave only -19.5
kcal/mol, so a channel-level error can be much larger than the error in the sum.

Pass criterion suggested by the reviewer: every channel drifts < 0.05 kcal/mol.

    python3 scripts/compare_scf.py [threshold_kcal]

No third-party dependencies (Nurion python has no pandas/numpy).
"""
import re
import sys
from pathlib import Path

LABELS = {
    "Bond Energy": "bond",
    "Orbital Energy": "orb",
    "Electrostatic Energy": "elst",
    "Pauli Energy": "pauli",
    "Delta E^0(XC)": "xc",
    "Delta Dispersion": "disp",
    "Delta CPCM Dielectric": "cpcm",
    "Delta SMD CDS correction": "smd",
}
ORDER = ["bond", "pauli", "elst", "orb", "xc", "disp", "cpcm", "smd"]
TABLE = re.compile(r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.DOTALL)
ROW = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
ECHECK = re.compile(r"Energy Check signals convergence")


def channels(path):
    if not path.exists():
        return None, None
    t = path.read_text(errors="replace")
    if "ORCA TERMINATED NORMALLY" not in t:
        return None, None
    m = TABLE.search(t)
    if not m:
        return None, None
    out = {}
    for r in ROW.finditer(m.group(1)):
        lab = r.group(1).strip()
        if lab in LABELS:
            # Use the Hartree column, not kcal/mol: the printed kcal values are
            # rounded to 0.01, which is the same order as the drift we are trying
            # to measure. 1 Eh = 627.5094740631 kcal/mol.
            out[LABELS[lab]] = float(r.group(2)) * 627.5094740631
    return out, bool(ECHECK.search(t))


def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    root = Path(".")
    ref = Path("data_unverified") if Path("data_unverified").is_dir() else Path("data")
    print(f"TightSCF reference dir: {ref}")
    pairs = []
    for vt in sorted((root / "data_vtscf").glob("rxn_*")):
        tight, ec_t = channels(ref / vt.name / "eda.out")
        very, ec_v = channels(vt / "eda.out")
        if tight and very:
            pairs.append((vt.name, tight, very, ec_t))

    if not pairs:
        print("no comparable pairs yet (run scripts/validate_scf.pbs first)")
        return

    print(f"pairs compared: {len(pairs)}   threshold: {thr} kcal/mol")
    print(f"  of which TightSCF exited on the energy check: "
          f"{sum(1 for p in pairs if p[3])}")
    print()
    hdr = f"{'channel':<8}" + "".join(f"{c:>12}" for c in ["max|d|", "mean|d|", "n>thr"])
    print(hdr)
    print("-" * len(hdr))
    worst = {}
    for c in ORDER:
        d = [abs(v[c] - t[c]) for _, t, v, _ in pairs if c in t and c in v]
        if not d:
            continue
        over = sum(1 for x in d if x > thr)
        worst[c] = max(d)
        flag = "  <-- FAILS" if over else ""
        print(f"{c:<8}{max(d):>12.4f}{sum(d)/len(d):>12.4f}{over:>12d}{flag}")
    print()
    print("per-reaction detail (channels exceeding threshold only):")
    any_bad = False
    for name, t, v, ec in pairs:
        bad = [(c, v[c] - t[c]) for c in ORDER if c in t and c in v and abs(v[c] - t[c]) > thr]
        if bad:
            any_bad = True
            ecs = "energy-check" if ec else "full-criteria"
            print(f"  {name} ({ecs}): " + ", ".join(f"{c}{d:+.3f}" for c, d in bad))
    if not any_bad:
        print("  none - every channel is within threshold")
    print()
    mx = max(worst.values()) if worst else 0.0
    verdict = "PASS" if mx <= thr else "FAIL"
    print(f"VERDICT: {verdict}  (largest channel drift {mx:.4f} kcal/mol)")
    if verdict == "FAIL":
        print("  -> TightSCF is not sufficient for the channels. Re-run production")
        print("     with VeryTightSCF, or report the drift as a systematic uncertainty.")


if __name__ == "__main__":
    main()
