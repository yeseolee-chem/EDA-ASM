#!/usr/bin/env python3
"""SPEC18 Step 8 — channel Δ error via ORCA EDA on GUESS-generated TS.

Mirrors spec17rev2 Step 10 (`10_channel_impact.py`) but uses
`generated/guess/*.xyz` as the TS source. Reuses the SAME 200 rxns
sampled in spec17rev2 (`otfm_train/channel_impact/sampled_ids.csv`)
so per-channel σ can be compared like-for-like against the RP-mode run.

Stages: sample | inputs | parse

  sample   – copy spec17rev2 sampled_ids.csv into this stage's channel_impact/
  inputs   – build 5 ORCA .inp per sampled rxn under channel_impact/inputs/
             using GUESS TS coords + spec17rev2 R geometry for strain refs
  parse    – parse .out files, compute δ_lin = √2·σ per channel vs Coley
             DFT reference (b3lyp_full labels_final.parquet)

GATE-8:
  all channels δ_lin < 5 kcal/mol  → PASS
  any channel      5..15           → WARN
  any channel     > 15             → FAIL
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"
GEN = BASE / "generated" / "guess"
OUT_ROOT = BASE / "channel_impact"
INP = OUT_ROOT / "inputs"

# Coley DFT reference labels
B3LYP_LABELS = REPO / "analysis" / "b3lyp_full" / "artifacts" / "labels_final.parquet"

# EDA channels emitted by b3lyp_full/03_parse.py
CHANNELS = ("elst", "orb", "pauli", "xc", "disp")

OUT_ROOT.mkdir(parents=True, exist_ok=True)


def read_xyz(path: Path):
    lines = path.read_text().split("\n")
    n = int(lines[0])
    syms, xyz = [], []
    for l in lines[2:2 + n]:
        t = l.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    return syms, np.array(xyz)


# ---------------------------------------------------------------------------
# stage: sample — reuse spec17rev2's 200 IDs for like-for-like comparison
# ---------------------------------------------------------------------------

def stage_sample() -> int:
    src = PREV / "channel_impact" / "sampled_ids.csv"
    if not src.exists():
        print(f"[FATAL] {src} not found. Run spec17rev2 Step 10 sample first.",
              file=sys.stderr)
        return 1
    dst = OUT_ROOT / "sampled_ids.csv"
    shutil.copy2(src, dst)
    n = len(pd.read_csv(dst))
    print(f"copied {n} sampled IDs -> {dst}")
    return 0


# ---------------------------------------------------------------------------
# stage: inputs — build ORCA inputs for eda + 4 strain SPEs per sampled rxn
# ---------------------------------------------------------------------------

EDA_HEADER = (
    "! B3LYP D3BJ def2-TZVP CPCM(water) NoSym EDA TightSCF\n"
    "%maxcore 3500\n\n"
    "%pal nprocs 5 end\n\n"
    "%cpcm\n  smd true\n  smdsolvent \"water\"\nend\n\n"
)
STRAIN_HEADER = (
    "! B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF\n"
    "%maxcore 3500\n\n"
    "%pal nprocs 5 end\n\n"
    "%cpcm\n  smd true\n  smdsolvent \"water\"\nend\n\n"
)


def load_frag_indices(rid: int):
    npz = np.load(PREV / "data" / "reactant_complex" / f"rxn_{rid:04d}.npz",
                  allow_pickle=True)
    return list(int(i) for i in npz["frag1"]), list(int(i) for i in npz["frag2"])


def write_orca(path: Path, header: str, coords_lines, charge: int, mult: int):
    body = header + f"* xyz {charge} {mult}\n" + "".join(coords_lines) + "*\n"
    tmp = path.with_suffix(".inp.tmp")
    tmp.write_text(body)
    tmp.replace(path)


def coord_lines(syms, xyz, indices=None):
    idx = list(range(len(syms))) if indices is None else list(indices)
    lines = []
    for i in idx:
        s = syms[i]
        x, y, z = xyz[i]
        lines.append(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
    return lines


def stage_inputs() -> int:
    sampled = pd.read_csv(OUT_ROOT / "sampled_ids.csv").rxn_id.tolist()
    INP.mkdir(parents=True, exist_ok=True)
    charge, mult = 0, 1     # Coley dipolar cycloaddition: neutral, closed shell

    built = 0
    missing = 0
    for rid in sampled:
        gen_xyz = GEN / f"rxn_{rid:04d}.xyz"
        if not gen_xyz.exists():
            missing += 1
            continue
        syms, xyz = read_xyz(gen_xyz)
        f1, f2 = load_frag_indices(rid)
        d = INP / f"rxn_{rid:04d}"
        d.mkdir(exist_ok=True)

        eda_body = EDA_HEADER + f"* xyz {charge} {mult}\n"
        for s, (x, y, z) in zip(syms, xyz):
            eda_body += f"{s} {x:.6f} {y:.6f} {z:.6f}\n"
        eda_body += "*\n"
        eda_body += "%eda\n"
        eda_body += (f"  fragments {{ 1 [{' '.join(str(i) for i in f1)}] }} "
                     f"{{ 2 [{' '.join(str(i) for i in f2)}] }}\n")
        eda_body += "end\n"
        tmp = d / "eda.inp.tmp"
        tmp.write_text(eda_body)
        tmp.replace(d / "eda.inp")

        rc = np.load(PREV / "data" / "reactant_complex" / f"rxn_{rid:04d}.npz",
                     allow_pickle=True)
        R_geom = rc["R"]
        for tag, indices, geom in (
            ("frag1_dist", f1, xyz),
            ("frag2_dist", f2, xyz),
            ("frag1_rel",  f1, R_geom),
            ("frag2_rel",  f2, R_geom),
        ):
            lines = coord_lines(syms, geom, indices=indices)
            write_orca(d / f"{tag}.inp", STRAIN_HEADER, lines, charge, mult)
        built += 1

    print(f"built inputs for {built}/{len(sampled)} rxns   missing={missing}")
    return 0


# ---------------------------------------------------------------------------
# stage: parse
# ---------------------------------------------------------------------------

EDA_TABLE_RE = re.compile(
    r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.DOTALL)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
HARTREE_TO_KCAL = 627.5094740631

LABEL_TO_KEY = {
    "Electrostatic Energy": "elst",
    "Orbital Energy": "orb",
    "Pauli Energy": "pauli",
    "Delta E^0(XC)": "xc",
    "Delta Dispersion": "disp",
}


def parse_eda(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text()
    if "ORCA TERMINATED NORMALLY" not in txt:
        return {}
    m = EDA_TABLE_RE.search(txt)
    if not m:
        return {}
    out = {}
    for row in ROW_RE.finditer(m.group(1)):
        label = row.group(1).strip()
        kcal = float(row.group(3))
        if label in LABEL_TO_KEY:
            out[LABEL_TO_KEY[label]] = kcal
    return out


def parse_fspe(path: Path):
    if not path.exists():
        return None
    txt = path.read_text()
    if "ORCA TERMINATED NORMALLY" not in txt:
        return None
    hits = FSPE_RE.findall(txt)
    return float(hits[-1]) * HARTREE_TO_KCAL if hits else None


def stage_parse() -> int:
    sampled = pd.read_csv(OUT_ROOT / "sampled_ids.csv").rxn_id.tolist()
    if not B3LYP_LABELS.exists():
        print(f"[WARN] Coley DFT reference not found: {B3LYP_LABELS}",
              file=sys.stderr)
        ref = None
    else:
        ref = pd.read_parquet(B3LYP_LABELS).set_index("rxn_id")

    rows = []
    for rid in sampled:
        d = INP / f"rxn_{rid:04d}"
        eda = parse_eda(d / "eda.out")
        if not eda:
            rows.append(dict(rxn_id=rid, ok=False, reason="eda parse fail"))
            continue
        f1_d = parse_fspe(d / "frag1_dist.out")
        f1_r = parse_fspe(d / "frag1_rel.out")
        f2_d = parse_fspe(d / "frag2_dist.out")
        f2_r = parse_fspe(d / "frag2_rel.out")
        strain1 = (f1_d - f1_r) if None not in (f1_d, f1_r) else None
        strain2 = (f2_d - f2_r) if None not in (f2_d, f2_r) else None
        rows.append(dict(rxn_id=rid, ok=True,
                          strain1=strain1, strain2=strain2, **eda))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "generated_channels_guess.csv", index=False)
    ok = df[df.ok]
    print(f"parsed {len(ok)}/{len(df)} sampled rxns")

    if ref is None:
        (BASE / "artifacts" / "GATE8_STATUS.txt").write_text(
            "PARTIAL\nreason=no b3lyp reference\n"
            f"parsed={len(ok)}\n"
        )
        return 0

    stats = {}
    for c in CHANNELS:
        if c not in ok.columns or c not in ref.columns:
            continue
        joined = ok[["rxn_id", c]].merge(
            ref[[c]].rename(columns={c: f"ref_{c}"}),
            left_on="rxn_id", right_index=True, how="inner",
        )
        joined[f"d_{c}"] = joined[c] - joined[f"ref_{c}"]
        s = joined[f"d_{c}"]
        stats[c] = dict(n=len(s), mean=float(s.mean()),
                         std=float(s.std()),
                         delta_lin=float(np.sqrt(2) * s.std()))

    ci = pd.DataFrame(stats).T
    ci.to_csv(OUT_ROOT / "channel_impact_guess.csv")
    print("\nchannel Δ statistics for GUESS-mode TS (kcal/mol):")
    print(ci.to_string())

    worst = max((v["delta_lin"] for v in stats.values()), default=float("nan"))
    gate = "PASS" if worst < 5 else ("WARN" if worst < 15 else "FAIL")
    (BASE / "artifacts" / "GATE8_STATUS.txt").write_text(
        f"{gate}\n"
        f"worst_delta_lin={worst:.4f}\n"
        f"channels={list(stats.keys())}\n"
    )
    print(f"\n=== GATE-8 {gate}  (worst channel δ_lin = {worst:.3f} kcal/mol) ===")
    return 0 if gate != "FAIL" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("sample", "inputs", "parse"))
    args = ap.parse_args()
    return {"sample": stage_sample,
            "inputs": stage_inputs,
            "parse":  stage_parse}[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
