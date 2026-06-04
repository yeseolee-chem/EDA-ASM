"""Per-reaction inventory of ADF_250/adf_outputs/.

Classifies every reaction directory under batch_*/ into one of:

  ok                       parses to a valid label, signs OK
  physics_inconsistent     parses, but a 5-channel sign is wrong (NOT recoverable)
  scf_not_converged        any calc_status step == 'not_converged'
  failed_step              any calc_status step == 'failed'
  eda_incomplete           c3_eda.out present, EDA channels unparseable
  fragment_energy_missing  c1/c2/c4/c5 .out present, Total Bonding Energy not found
  no_output_file           status.json says converged but a referenced .out is missing
                           (covers the May-29 PBE0 relics whose .out was wiped)
  no_status_file           status.json missing — never produced
  invalid_status           status.json present but unparseable JSON

Emits:
  ADF_250/adf_outputs/parsed/statistics.json   — by_quality_flag + by_failure_reason
                                                  + by_family + ok-recoverable summary
  ADF_250/adf_outputs/parsed/failures.csv      — one row per non-ok reaction

Read-only. No ADF compute touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_VAL = re.compile(
    r"^\s*(?P<label>.+?):\s+(?P<ha>-?\d+\.\d+)\s+(?P<ev>-?\d+\.\d+)\s+"
    r"(?P<kcal>-?\d+\.\d+)\s+(?P<kj>-?\d+\.\d+)\s*$"
)

RECOVERABLE = {
    "scf_not_converged",
    "failed_step",
    "eda_incomplete",
    "fragment_energy_missing",
    "no_output_file",
    "no_status_file",
    "invalid_status",
}
NOT_RECOVERABLE = {"physics_inconsistent"}

# Sign convention from CLAUDE.md §"Decomposition Components" and asr_v1.models:
#   E_strain > 0, Pauli > 0, V_elst < 0, E_orb < 0, E_disp ≤ 0
SIGN = {"E_strain_kcal": +1, "Pauli_kcal": +1, "V_elst_kcal": -1,
        "E_orb_kcal": -1, "E_disp_kcal": -1}


def _first_value(text: str, label_regex: str) -> Optional[float]:
    rex = re.compile(label_regex, re.IGNORECASE)
    for line in text.splitlines():
        m = _VAL.match(line)
        if m and rex.search(m["label"].strip()):
            return float(m["kcal"])
    return None


def _parse_fragment_bond(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    return _first_value(path.read_text(errors="replace"), r"^Total\s+Bonding\s+Energy$")


def _parse_eda(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    pauli = _first_value(text, r"^Total\s+Pauli\s+Repulsion$")
    elst = _first_value(text, r"^Electrostatic\s+Interaction$")
    orb = _first_value(text, r"^Total\s+Orbital\s+Interactions$")
    total = _first_value(text, r"^Total\s+Bonding\s+Energy$")
    disp = (
        _first_value(text, r"^Dispersion\s+Energy$")
        or _first_value(text, r"^Total\s+Dispersion\s+Energy$")
        or _first_value(text, r"^Dispersion$")
        or 0.0
    )
    if pauli is None or elst is None or orb is None or total is None:
        return None
    return {"Pauli_kcal": pauli, "V_elst_kcal": elst,
            "E_orb_kcal": orb, "E_disp_kcal": disp,
            "E_int_total_kcal": total}


def _family_of(rxn_id: str) -> str:
    if rxn_id.startswith("qmrxn20_"):
        return "qmrxn20_" + rxn_id.split("_")[1]
    return rxn_id.split("_")[0]


def classify(rxn_dir: Path) -> dict:
    """Return {reaction_id, family, quality_flag, failure_reason, detail}."""
    rxn_id = rxn_dir.name
    rec = {"reaction_id": rxn_id, "family": _family_of(rxn_id),
           "quality_flag": None, "failure_reason": None, "detail": ""}

    status_p = rxn_dir / "status.json"
    if not status_p.is_file():
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "no_status_file"
        return rec
    try:
        s = json.loads(status_p.read_text())
    except Exception as exc:                                  # noqa: BLE001
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "invalid_status"
        rec["detail"] = f"json: {exc}"
        return rec

    # Step-level state from status.json
    calc = s.get("calc_status", {})
    step_states = list(calc.values())
    if "not_converged" in step_states:
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "scf_not_converged"
        rec["detail"] = ";".join(f"{k}={v}" for k, v in calc.items()
                                 if v == "not_converged")
        return rec
    if "failed" in step_states:
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "failed_step"
        rec["detail"] = ";".join(f"{k}={v}" for k, v in calc.items()
                                 if v == "failed")
        return rec

    if s.get("exit_code") != 0:
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "scf_not_converged"   # treat any non-zero exit as recoverable
        rec["detail"] = f"exit_code={s.get('exit_code')}"
        return rec

    # Attempt full parse path matching extract_asr_labels.extract_one
    out_files = {k: rxn_dir / v for k, v in s.get("output_files", {}).items()}
    missing = [k for k, p in out_files.items() if not p.is_file()]
    if missing:
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "no_output_file"
        rec["detail"] = "missing=" + ",".join(missing)
        return rec

    eda = _parse_eda(rxn_dir / "c3_eda.out")
    if eda is None:
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "eda_incomplete"
        return rec

    e_fA_TS = _parse_fragment_bond(rxn_dir / "c1_fragA_ts.out")
    e_fB_TS = _parse_fragment_bond(rxn_dir / "c2_fragB_ts.out")
    if e_fA_TS is None or e_fB_TS is None:
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "fragment_energy_missing"
        rec["detail"] = f"c1={e_fA_TS}, c2={e_fB_TS}"
        return rec

    single_a = calc.get("c4_fragA_opt") == "n/a_single_atom"
    single_b = calc.get("c5_fragB_opt") == "n/a_single_atom"
    e_fA_rel = e_fA_TS if single_a else _parse_fragment_bond(rxn_dir / "c4_fragA_opt.out")
    e_fB_rel = e_fB_TS if single_b else _parse_fragment_bond(rxn_dir / "c5_fragB_opt.out")
    if e_fA_rel is None or e_fB_rel is None:
        rec["quality_flag"] = "failed"
        rec["failure_reason"] = "fragment_energy_missing"
        rec["detail"] = f"c4={e_fA_rel}, c5={e_fB_rel}"
        return rec

    strain_A = 0.0 if single_a else (e_fA_TS - e_fA_rel)
    strain_B = 0.0 if single_b else (e_fB_TS - e_fB_rel)
    strain = strain_A + strain_B
    channels = {"E_strain_kcal": strain, **{k: eda[k] for k in
                ("Pauli_kcal", "V_elst_kcal", "E_orb_kcal", "E_disp_kcal")}}

    # Physical-sign check
    bad = [k for k, expected in SIGN.items()
           if expected * channels[k] < 0]
    if bad:
        rec["quality_flag"] = "warning"
        rec["failure_reason"] = "physics_inconsistent"
        rec["detail"] = ";".join(f"{k}={channels[k]:.2f}" for k in bad)
        return rec

    rec["quality_flag"] = "ok"
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adf-root", type=Path,
                   default=Path("ADF_250/adf_outputs"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("ADF_250/adf_outputs/parsed"))
    args = p.parse_args()

    rxn_dirs = sorted([d for d in args.adf_root.glob("batch_*/*") if d.is_dir()])
    print(f"[inventory] walking {len(rxn_dirs)} reaction dirs under {args.adf_root}")

    by_flag = Counter()
    by_reason = Counter()
    by_family_flag: dict[str, Counter] = defaultdict(Counter)
    failures = []
    for d in rxn_dirs:
        rec = classify(d)
        by_flag[rec["quality_flag"]] += 1
        by_family_flag[rec["family"]][rec["quality_flag"]] += 1
        if rec["quality_flag"] != "ok":
            by_reason[rec["failure_reason"]] += 1
            failures.append(rec)

    n_recoverable = sum(by_reason[r] for r in RECOVERABLE)
    n_not_recoverable = sum(by_reason[r] for r in NOT_RECOVERABLE)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "n_attempted": len(rxn_dirs),
        "by_quality_flag": dict(by_flag),
        "by_failure_reason": dict(by_reason),
        "by_family": {f: dict(c) for f, c in by_family_flag.items()},
        "recoverable_failure_reasons": sorted(RECOVERABLE),
        "not_recoverable_failure_reasons": sorted(NOT_RECOVERABLE),
        "n_recoverable_failures": n_recoverable,
        "n_not_recoverable_failures": n_not_recoverable,
        "adf_root": str(args.adf_root),
    }
    (args.out_dir / "statistics.json").write_text(json.dumps(stats, indent=2))

    csv_path = args.out_dir / "failures.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["reaction_id", "family", "quality_flag",
                           "failure_reason", "detail"],
        )
        w.writeheader()
        for r in failures:
            w.writerow(r)

    # Console summary
    print(f"\n=== Inventory: ADF_800 (n_attempted = {len(rxn_dirs)}) ===")
    print("by_quality_flag:")
    for k in ("ok", "warning", "failed"):
        print(f"  {k:8s} : {by_flag.get(k, 0)}")
    print("\nby_failure_reason:")
    for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        tag = "[RECOV]" if k in RECOVERABLE else "[NOT-RECOV]" if k in NOT_RECOVERABLE else "[?]"
        print(f"  {k:30s} {v:4d}   {tag}")
    print(f"\nRecoverable failures   : {n_recoverable}")
    print(f"Not-recoverable        : {n_not_recoverable}")
    print(f"\nby_family:")
    for fam, c in sorted(by_family_flag.items()):
        print(f"  {fam:18s} ok={c.get('ok',0):4d}  warn={c.get('warning',0):3d}  fail={c.get('failed',0):4d}")
    print(f"\nWrote {args.out_dir/'statistics.json'} and {csv_path}")


if __name__ == "__main__":
    main()
