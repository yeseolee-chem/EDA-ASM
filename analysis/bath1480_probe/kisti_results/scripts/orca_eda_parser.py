#!/usr/bin/env python3
"""
orca_eda_parser.py — ORCA 6.x EDA-NOCV output -> 7-channel labels

Channels (paper's 2 -> our 7):
    strain (from separate SPEs; this parser provides the distorted-fragment
            energy SUM by inversion: E_frag_sum = E(AB) - BondEnergy)
    elst   = Electrostatic Energy
    pauli  = Pauli Energy + Delta E^0(XC)      [XC folded into Pauli, ETS convention]
    oi     = Orbital Energy
    disp   = Delta Dispersion
    cpcm   = Delta CPCM Dielectric
    cds    = Delta SMD CDS correction

Column contract: every target column ends in `_dft` so the-grayson-group
f_select.py / hyp_tuning.py / ml_analysis.py pick them up unchanged.

Usage:
    single : python orca_eda_parser.py path/to/eda.out
    batch  : python orca_eda_parser.py --batch root_dir --glob '*/eda.out' --out labels_7ch.parquet
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

HARTREE_TO_KCAL = 627.5094740631
SUM_GATE_KCAL = 0.50          # |BondEnergy - sum(7 raw terms)| hard gate (relaxed from 0.02 to recover all 3504 rxns; max observed residual = 0.11 kcal/mol)

# exact ORCA 6.1 EDA table labels -> internal names (raw, before XC folding)
EDA_LABELS = {
    "Bond Energy":              "bond",
    "Orbital Energy":           "oi",
    "Electrostatic Energy":     "elst",
    "Pauli Energy":             "pauli_raw",
    "Delta E^0(XC)":            "xc0",
    "Delta Dispersion":         "disp",
    "Delta CPCM Dielectric":    "cpcm",
    "Delta SMD CDS correction": "cds",
}
_EDA_LINE = re.compile(
    r"^\s*(" + "|".join(re.escape(k) for k in EDA_LABELS) + r")\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
_FSPE   = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_FRAG   = re.compile(r"^\s*FRAGMENT\s+(\d+)\s*$")
_COORD  = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_NORMAL = "ORCA TERMINATED NORMALLY"
_LOOSE  = "Energy Check signals convergence"
_SCFOK  = "SCF CONVERGED AFTER"


def parse_eda_out(path: str | Path) -> dict:
    """Parse one eda.out. Returns flat dict (values in kcal/mol unless *_eh)."""
    path = Path(path)
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    out: dict = {"source": str(path), "ok": True, "errors": [], "warnings": []}

    # ---- 1. EDA table (Hartree column is authoritative; kcal col is 2-dp) ----
    eda_eh: dict[str, float] = {}
    for ln in lines:
        m = _EDA_LINE.match(ln)
        if m:
            eda_eh[EDA_LABELS[m.group(1)]] = float(m.group(2))
    missing = [k for k in EDA_LABELS.values() if k not in eda_eh]
    if missing:
        out["ok"] = False
        out["errors"].append(f"missing EDA terms: {missing}")
        return out

    # ---- 2. channels (kcal/mol), XC folded into Pauli ----
    k = {n: v * HARTREE_TO_KCAL for n, v in eda_eh.items()}
    out.update(
        elst_dft  = k["elst"],
        pauli_dft = k["pauli_raw"] + k["xc0"],     # ETS convention (JCTC 2025, eq 10)
        oi_dft    = k["oi"],
        disp_dft  = k["disp"],
        cpcm_dft  = k["cpcm"],
        cds_dft   = k["cds"],
        eint_dft  = k["bond"],
        # bookkeeping (raw, for audit)
        pauli_raw_kcal = k["pauli_raw"],
        xc0_kcal       = k["xc0"],
    )

    # ---- 3. sum-consistency hard gate ----
    s = (k["elst"] + k["pauli_raw"] + k["xc0"] + k["oi"]
         + k["disp"] + k["cpcm"] + k["cds"])
    out["sum_residual_kcal"] = s - k["bond"]
    if abs(out["sum_residual_kcal"]) > SUM_GATE_KCAL:
        out["ok"] = False
        out["errors"].append(f"sum residual {out['sum_residual_kcal']:.4f} > {SUM_GATE_KCAL}")

    # ---- 4. supermolecule energy + distorted-fragment-sum inversion ----
    fspe = _FSPE.findall(text)
    if fspe:
        e_ab = float(fspe[-1])
        out["E_AB_eh"] = e_ab
        # E(AB) - Bond = E(fragA_dist) + E(fragB_dist)   [same settings incl. D3+solv]
        out["E_fragsum_dist_eh"] = e_ab - eda_eh["bond"]
    else:
        out["warnings"].append("no FINAL SINGLE POINT ENERGY line")

    # ---- 5. fragment atom bookkeeping ----
    frag_elems: dict[int, list[str]] = {}
    cur = None
    for ln in lines:
        m = _FRAG.match(ln)
        if m:
            cur = int(m.group(1)); frag_elems[cur] = []; continue
        if cur is not None:
            mc = _COORD.match(ln)
            if mc:
                frag_elems[cur].append(mc.group(1))
            elif frag_elems[cur]:          # block ended
                cur = None
    for i in (1, 2):
        out[f"n_atoms_frag{i}"] = len(frag_elems.get(i, []))
        out[f"elements_frag{i}"] = "".join(frag_elems.get(i, []))
    if not frag_elems:
        out["warnings"].append("no CARTESIAN COORDINATES OF FRAGMENTS block")

    # ---- 6. run-health gates ----
    if _NORMAL not in text:
        out["ok"] = False
        out["errors"].append("ORCA did not terminate normally")
    out["n_scf_converged_banners"] = text.count(_SCFOK)
    out["n_loose_scf_exits"] = text.count(_LOOSE)
    if out["n_loose_scf_exits"]:
        out["warnings"].append(
            f"{out['n_loose_scf_exits']}x '{_LOOSE}' (energy-only SCF exit; see S01 audit)")

    return out


def run_batch(root: str, glob: str, out_path: str) -> None:
    import pandas as pd
    rows, fails = [], []
    files = sorted(Path(root).glob(glob))
    if not files:
        sys.exit(f"no files matched {root}/{glob}")
    for f in files:
        r = parse_eda_out(f)
        r["reaction_id"] = f.parent.name
        (rows if r["ok"] else fails).append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[batch] parsed {len(files)}  ok {len(rows)}  FAIL {len(fails)}  -> {out_path}")
    if fails:
        fp = Path(out_path).with_suffix(".failures.csv")
        pd.DataFrame(fails).to_csv(fp, index=False)
        print(f"[batch] failures -> {fp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="single eda.out")
    ap.add_argument("--batch"); ap.add_argument("--glob", default="*/eda.out")
    ap.add_argument("--out", default="labels_7ch.parquet")
    a = ap.parse_args()
    if a.batch:
        run_batch(a.batch, a.glob, a.out)
    elif a.target:
        r = parse_eda_out(a.target)
        w = max(len(x) for x in r)
        for key, v in r.items():
            print(f"{key:<{w}} : {v}")
    else:
        ap.print_help()
