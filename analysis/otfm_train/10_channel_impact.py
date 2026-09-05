#!/usr/bin/env python3
"""SPEC17rev2 Step 10 — channel Δ error via ORCA EDA on generated TS.

Motivation (SPEC §10): the OA barrier can be predicted to 1.06 kcal/mol
via error-cancellation while individual EDA channels drift much more.
Measuring per-channel σ tells us the true resolution ceiling of the
generated-TS geometry.

Stages (select via --stage):
  sample   -> stratified 200-rxn pick from generated/crossfit + Coley labels
  inputs   -> build 5 ORCA .inp files per sampled rxn in inputs/rxn_NNNN/
              (eda + 4 strain SPEs) using generated TS coords
  parse    -> parse .out files, compare Δ to Coley DFT channels, write
              channel_impact.csv with per-channel mean / σ / δ_lin (=√2·σ)

Actual ORCA execution happens under a launcher sbatch (sbatch/10_launcher.sh)
that chain-submits arrays sized to fit the 20-task SLURM submit cap.

GATE-9 (spec §10 table):
  all channels δ_lin  < 5 kcal/mol  -> PASS
  any channel        5..15          -> WARN
  any channel        > 15           -> FAIL
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis/otfm_train"
GEN = BASE / "generated/crossfit"
INP = BASE / "channel_impact" / "inputs"
OUT_ROOT = BASE / "channel_impact"
PROF = BASE / "coley_profiles/full_dataset_profiles"

# Coley DFT reference labels — produced by spec16rev b3lyp_full.
B3LYP_LABELS = REPO / "analysis/b3lyp_full/artifacts/labels_final.parquet"

# EDA channel keys emitted by b3lyp_full/03_parse.py.
CHANNELS = ("elst", "orb", "pauli", "xc", "disp")

N_SAMPLES = 200
SEED = 42


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
# STAGE: sample
# ---------------------------------------------------------------------------

def stage_sample() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ok_gen = {int(p.stem.split("_")[1]) for p in GEN.glob("rxn_*.xyz")}
    print(f"generated TS available: {len(ok_gen)}")

    comp = pd.read_csv(BASE / "artifacts" / "composition.csv")
    comp = comp[comp.status == "ok"].set_index("rxn_id")

    # stratify by (element set, n_atoms bin)
    def bin_atoms(n):
        if n <= 20:
            return "small"
        if n <= 30:
            return "medium"
        return "large"

    df = comp.loc[sorted(ok_gen & set(comp.index))].copy()
    df["size_bin"] = df.n_atoms.apply(bin_atoms)
    df["stratum"] = df.elements + "_" + df.size_bin

    rng = np.random.RandomState(SEED)
    n_strata = df.stratum.nunique()
    per_stratum = max(1, N_SAMPLES // max(1, n_strata))
    picks = []
    for _, g in df.groupby("stratum"):
        take = min(per_stratum, len(g))
        picks.extend(rng.choice(g.index.values, size=take, replace=False).tolist())
    # top up to N_SAMPLES if strata were sparse
    remain = sorted(set(df.index) - set(picks))
    if len(picks) < N_SAMPLES and remain:
        extra = rng.choice(remain, size=min(N_SAMPLES - len(picks), len(remain)),
                            replace=False)
        picks.extend(extra.tolist())
    picks = sorted(set(picks))[:N_SAMPLES]

    out = OUT_ROOT / "sampled_ids.csv"
    pd.DataFrame({"rxn_id": picks}).to_csv(out, index=False)
    print(f"sampled {len(picks)} rxns across {n_strata} strata -> {out}")
    return 0


# ---------------------------------------------------------------------------
# STAGE: inputs
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


def load_frag_indices(rid: int) -> tuple[list[int], list[int]]:
    """Get fragment index partition from Step 2's npz (matches TS filename)."""
    npz = np.load(BASE / "data" / "reactant_complex" / f"rxn_{rid:04d}.npz",
                  allow_pickle=True)
    return list(int(i) for i in npz["frag1"]), list(int(i) for i in npz["frag2"])


def total_charge_and_mult() -> tuple[int, int]:
    # Coley dipolar cycloaddition: neutral, closed shell.
    return 0, 1


def write_orca_input(path: Path, header: str, coords: list[str], charge: int, mult: int):
    body = header + f"* xyz {charge} {mult}\n" + "".join(coords) + "*\n"
    tmp = path.with_suffix(".inp.tmp")
    tmp.write_text(body)
    tmp.replace(path)


def coord_lines(syms, xyz, indices=None, ghost=None):
    """Return list of ORCA xyz lines. ghost=list of indices to emit as basis-only."""
    idx = list(range(len(syms))) if indices is None else list(indices)
    ghost = set() if ghost is None else set(ghost)
    lines = []
    for i in idx:
        s, (x, y, z) = syms[i], xyz[i]
        tag = f"{s}:" if i in ghost else s
        lines.append(f"{tag} {x:.6f} {y:.6f} {z:.6f}\n")
    return lines


def stage_inputs() -> int:
    sampled = pd.read_csv(OUT_ROOT / "sampled_ids.csv").rxn_id.tolist()
    INP.mkdir(parents=True, exist_ok=True)
    charge, mult = total_charge_and_mult()

    built = 0
    for rid in sampled:
        gen_xyz = GEN / f"rxn_{rid:04d}.xyz"
        if not gen_xyz.exists():
            print(f"  [skip] rxn {rid}: no generated TS")
            continue
        syms, xyz = read_xyz(gen_xyz)
        f1, f2 = load_frag_indices(rid)
        d = INP / f"rxn_{rid:04d}"
        d.mkdir(exist_ok=True)

        # eda: full system, fragmented via ORCA's %fragments block would need
        # index remap; using two `%pal`-friendly single files instead:
        #   eda.inp   : full complex (for EDA using the fragmentation info
        #               parsed from TS filename in b3lyp_full's builder).
        # Here we mirror b3lyp_full: fragments listed after coordinates.
        # (Simplification: the parser only needs Bond/Orbital/Pauli/xc/disp
        #  fields from EDA output; ORCA infers fragments from bond distance
        #  when %fragments is absent, but explicit is safer — we emit it.)
        eda_body = EDA_HEADER + f"* xyz {charge} {mult}\n"
        for i, (s, (x, y, z)) in enumerate(zip(syms, xyz)):
            eda_body += f"{s} {x:.6f} {y:.6f} {z:.6f}\n"
        eda_body += "*\n"
        # fragment assignment for EDA
        eda_body += "%eda\n"
        eda_body += f"  fragments {{ 1 [{' '.join(str(i) for i in f1)}] }} {{ 2 [{' '.join(str(i) for i in f2)}] }}\n"
        eda_body += "end\n"
        tmp = (d / "eda.inp.tmp")
        tmp.write_text(eda_body)
        tmp.replace(d / "eda.inp")

        # strain SPEs: each fragment alone at distorted (TS) geometry and at
        # relaxed (Coley reactant) geometry — δ_strain = E_dist − E_rel.
        rc = np.load(BASE / "data" / "reactant_complex" / f"rxn_{rid:04d}.npz",
                     allow_pickle=True)
        R_geom = rc["R"]
        for tag, indices, geom in (
            ("frag1_dist", f1, xyz),
            ("frag2_dist", f2, xyz),
            ("frag1_rel",  f1, R_geom),
            ("frag2_rel",  f2, R_geom),
        ):
            lines = coord_lines(syms, geom, indices=indices)
            write_orca_input(d / f"{tag}.inp", STRAIN_HEADER, lines, charge, mult)
        built += 1

    print(f"built inputs for {built} rxns under {INP}")
    return 0


# ---------------------------------------------------------------------------
# STAGE: parse
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


def parse_fspe(path: Path) -> float | None:
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
        print(f"[WARN] Coley DFT reference not found: {B3LYP_LABELS}", file=sys.stderr)
        print("       Waiting for spec16rev b3lyp_full/06_export.py output.", file=sys.stderr)
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
        strain1 = None
        strain2 = None
        f1_d = parse_fspe(d / "frag1_dist.out")
        f1_r = parse_fspe(d / "frag1_rel.out")
        f2_d = parse_fspe(d / "frag2_dist.out")
        f2_r = parse_fspe(d / "frag2_rel.out")
        if None not in (f1_d, f1_r):
            strain1 = f1_d - f1_r
        if None not in (f2_d, f2_r):
            strain2 = f2_d - f2_r
        row = dict(rxn_id=rid, ok=True,
                   strain1=strain1, strain2=strain2, **eda)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "generated_channels.csv", index=False)
    ok = df[df.ok]
    print(f"parsed {len(ok)} of {len(df)} sampled rxns")

    if ref is None:
        (BASE / "artifacts" / "GATE9_STATUS.txt").write_text(
            "PARTIAL\nreason=no b3lyp reference yet\n"
            f"parsed={len(ok)}\n"
        )
        return 0

    # Δ per channel
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
    ci.to_csv(OUT_ROOT / "channel_impact.csv")
    print("\nchannel Δ statistics (kcal/mol):")
    print(ci.to_string())

    worst = max((v["delta_lin"] for v in stats.values()), default=float("nan"))
    if worst < 5:
        gate = "PASS"
    elif worst < 15:
        gate = "WARN"
    else:
        gate = "FAIL"
    (BASE / "artifacts" / "GATE9_STATUS.txt").write_text(
        f"{gate}\n"
        f"worst_delta_lin={worst:.4f}\n"
        f"channels={list(stats.keys())}\n"
    )
    print(f"\n=== GATE-9 {gate}  (worst channel δ_lin = {worst:.3f} kcal/mol) ===")
    return 0 if gate != "FAIL" else 1


# ---------------------------------------------------------------------------

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
