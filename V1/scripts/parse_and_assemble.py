"""Phase 4 — assemble per-substrate ASR results into a single parquet/CSV.

Per V1 Claisen spec §6 Phase 4:
  Schema columns:
    id, R, smiles, F, R_res, sigma_p,
    dE_strain, dV_elst, dE_Pauli, dE_oi, dE_disp,
    dE_int, dE_frag_barrier_eda, dE_barrier_wb97x3c,
    n_imag_ts, imag_freq_ts_cm1, eda_level, geom_level, status

Reads:
  substrates.csv            — F, R_res, sigma_p per id
  runs/<id>/orca/reactant.out, ts.out  → wB97X-3c energies + imag freq
  runs/<id>/eda/asr_vector.json — ADF EDA 5-vector
Writes:
  outputs/v1_claisen_asr.parquet
  outputs/v1_claisen_asr.csv
  outputs/report.md (summary)
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"
SUBSTRATES_CSV = PROJECT / "substrates.csv"
OUTPUTS = PROJECT / "outputs"
HARTREE_TO_KCAL = 627.5094740631  # NIST

GEOM_LEVEL = "wB97X-3c (ORCA 6.1.1, selective AVX2)"
EDA_LEVEL = "ZORA-BLYP-D3(BJ)/TZ2P all-electron NOSYM Good (ADF 2026.103) // wB97X-3c"


def grep_final_E(orca_out: Path) -> float | None:
    if not orca_out.exists():
        return None
    last = None
    for line in orca_out.read_text(errors="ignore").splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            try:
                last = float(line.split()[-1])
            except ValueError:
                pass
    return last


def parse_imag(orca_out: Path) -> tuple[int, float | None]:
    """Count imag modes from the LAST VIBRATIONAL FREQUENCIES block."""
    if not orca_out.exists():
        return 0, None
    text = orca_out.read_text(errors="ignore")
    # Find last "VIBRATIONAL FREQUENCIES" header
    headers = [m.start() for m in re.finditer(r"VIBRATIONAL FREQUENCIES", text)]
    if not headers:
        return 0, None
    tail = text[headers[-1] :]
    # Cut at "NORMAL MODES" if present
    end = tail.find("NORMAL MODES")
    block = tail if end < 0 else tail[:end]
    imag_lines = re.findall(r"\s+\d+:\s+(-\d+\.\d+)\s+cm\*\*-1\s+\*\*\*imaginary mode\*\*\*", block)
    n_imag = len(imag_lines)
    first_freq = float(imag_lines[0]) if imag_lines else None
    return n_imag, first_freq


def assemble_one(row: dict) -> dict:
    rxn_id = row["id"]
    rec = {
        "id": rxn_id,
        "R": row["R"],
        "smiles": row["smiles"],
        "F": float(row["F"]),
        "R_res": float(row["R_res"]),
        "sigma_p": float(row["sigma_p"]),
        "geom_level": GEOM_LEVEL,
        "eda_level": EDA_LEVEL,
    }

    # ORCA outputs
    orca_dir = RUNS / rxn_id / "orca"
    e_r = grep_final_E(orca_dir / "reactant.out")
    e_t = grep_final_E(orca_dir / "ts.out")
    n_imag, imag_freq = parse_imag(orca_dir / "ts.out")
    rec["E_reactant_Eh"] = e_r
    rec["E_TS_Eh"] = e_t
    rec["dE_barrier_wb97x3c"] = (e_t - e_r) * HARTREE_TO_KCAL if (e_r is not None and e_t is not None) else None
    rec["n_imag_ts"] = n_imag
    rec["imag_freq_ts_cm1"] = imag_freq

    # ADF EDA
    asr_path = RUNS / rxn_id / "eda" / "asr_vector.json"
    if asr_path.exists():
        asr = json.loads(asr_path.read_text())
        for col in ("E_strain", "E_Pauli", "E_elstat", "E_oi", "E_disp",
                    "E_int", "dE_frag_barrier_eda"):
            rec[col] = asr.get(col)
        # Aliases per spec column names
        rec["dE_strain"] = rec["E_strain"]
        rec["dV_elst"] = rec["E_elstat"]
        rec["dE_Pauli"] = rec["E_Pauli"]
        rec["dE_oi"] = rec["E_oi"]
        rec["dE_disp"] = rec["E_disp"]
        rec["dE_int"] = rec["E_int"]
    else:
        for col in ("E_strain", "E_Pauli", "E_elstat", "E_oi", "E_disp",
                    "E_int", "dE_frag_barrier_eda",
                    "dE_strain", "dV_elst", "dE_Pauli", "dE_oi", "dE_disp", "dE_int"):
            rec[col] = None

    # Status decision
    if e_r is None or e_t is None:
        rec["status"] = "FAILED:orca"
    elif n_imag != 1:
        rec["status"] = f"FAILED:n_imag={n_imag}"
    elif imag_freq is not None and imag_freq > -150:
        rec["status"] = f"REVIEW:weak_imag={imag_freq:.0f}"
    elif rec["E_strain"] is None:
        rec["status"] = "FAILED:eda_missing"
    else:
        diff = abs(rec["dE_frag_barrier_eda"] -
                   (rec["E_strain"] + rec["E_int"])) if rec["E_strain"] is not None else 1e9
        rec["status"] = "OK" if diff < 0.5 else f"REVIEW:consist={diff:.2f}"
    return rec


def main() -> int:
    with SUBSTRATES_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    records = [assemble_one(r) for r in rows]
    df = pd.DataFrame(records)

    # Order columns per spec
    cols = [
        "id", "R", "smiles", "F", "R_res", "sigma_p",
        "dE_strain", "dV_elst", "dE_Pauli", "dE_oi", "dE_disp",
        "dE_int", "dE_frag_barrier_eda", "dE_barrier_wb97x3c",
        "n_imag_ts", "imag_freq_ts_cm1",
        "E_reactant_Eh", "E_TS_Eh",
        "eda_level", "geom_level", "status",
    ]
    df = df[[c for c in cols if c in df.columns]]

    OUTPUTS.mkdir(exist_ok=True)
    parquet_path = OUTPUTS / "v1_claisen_asr.parquet"
    csv_path = OUTPUTS / "v1_claisen_asr.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"wrote {parquet_path} and {csv_path}")
    print(df[["id", "sigma_p", "dE_barrier_wb97x3c", "dE_strain",
              "dV_elst", "dE_Pauli", "dE_oi", "dE_disp",
              "imag_freq_ts_cm1", "status"]].to_string(index=False))

    # Quick LFER-style summary in report.md (avoid tabulate dependency)
    rep = OUTPUTS / "report.md"
    with rep.open("a") as fh:
        fh.write(f"\n\n## Phase 4 assembly — {datetime.now(timezone.utc).isoformat()}\n\n")
        fh.write(f"Rows: {len(df)}.  Status breakdown:\n\n")
        for status_val, count in df["status"].value_counts().items():
            fh.write(f"- `{status_val}`: {count}\n")
        fh.write("\n### ASR table (kcal/mol)\n\n```\n")
        fh.write(df[["id", "sigma_p", "dE_barrier_wb97x3c",
                     "dE_strain", "dV_elst", "dE_Pauli",
                     "dE_oi", "dE_disp", "imag_freq_ts_cm1", "status"]]
                 .sort_values("sigma_p").to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        fh.write("\n```\n\n")
        # Spec §8 sanity checks
        consist = (df["dE_frag_barrier_eda"]
                   - (df["dE_strain"] + df["dE_int"])).abs()
        fh.write("### Spec §8 sanity checks\n\n")
        fh.write(f"1. TS qualification: all n_imag_ts == 1 → "
                 f"{(df['n_imag_ts'] == 1).all()}\n")
        fh.write(f"2. Consistency |strain+int - frag_barrier_eda| < 0.5: "
                 f"max diff = {consist.max():.4f}\n")
        fh.write(f"3. Imaginary freq range: {df['imag_freq_ts_cm1'].min():.0f} to "
                 f"{df['imag_freq_ts_cm1'].max():.0f} cm⁻¹\n")

    n_ok = (df["status"] == "OK").sum()
    print(f"\nstatus=OK: {n_ok}/{len(df)}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
