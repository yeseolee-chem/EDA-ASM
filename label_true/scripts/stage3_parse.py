#!/usr/bin/env python3
"""Stage 3 — parse ORCA outputs → EDA channels + strain + d1/d2 → labels.json.

Replaces the regex-based parser (which returned None for every channel on real
ORCA 6.1.1 output). This one parses the EDA table verbatim, exactly as
analysis/b3lyp_full/03_parse.py in the EDA-ASM repo (verified in spec14 to
reproduce the phase5 labels to 1e-6):

    Energy Term                      Hartree         Kcal/mol
    ---------------------------------------------------------
    Bond Energy                   -0.0031727089        -1.99
    Orbital Energy                -0.0436904641       -27.42
    Electrostatic Energy          -0.0347532936       -21.81
    Pauli Energy                   0.1664243672       104.43
    Delta E^0(XC)                 -0.0704873030       -44.23
    Delta Dispersion              -0.0114391053        -7.18
    Delta CPCM Dielectric         -0.0036545500        -2.29
    Delta SMD CDS correction      -0.0055738517        -3.50

Channel convention (identical to the repo / phase5 labels):
    elst_dft  = Electrostatic Energy
    pauli_dft = Pauli Energy + Delta E^0(XC)
    oi_dft    = Orbital Energy
    disp_dft  = Delta Dispersion
    cpcm_dft  = Delta CPCM Dielectric
    cds_dft   = Delta SMD CDS correction
    e_bond_kcal = Bond Energy  (= total interaction; equals the sum of the 6 channels)
All values are taken from the Hartree column and converted with 627.5094740631.
The raw 8 table entries are also stored (eda_raw_eh) for auditing.

Fragment reference convention (BSSE-consistent, 2026-09-15 fix):
    e_frag1_dist_eh, e_frag2_dist_eh are taken from **eda_frag1.out** and
    **eda_frag2.out** — ORCA's internal fragment SPEs with ghost basis
    (counterpoise) from the other fragment. This matches the reference
    state that ORCA EDA-NOCV uses when producing the 6-channel decomposition
    and Bond Energy, so:
        eint_spe = (e_ab - e_frag1_dist - e_frag2_dist) * EH_TO_KCAL == e_bond_kcal
        d1 + d2 + eint_spe == barrier (ASM identity, exact)
        d1 + d2 + Σ(6 channels) == barrier (channel sum closure, exact)
    The old convention used the standalone frag*_dist.out SCFs (no ghost)
    which produced a systematic 3–8 kcal/mol gap between eint_spe and e_bond
    (BSSE). Standalone values are retained in the record for provenance under
    e_frag{1,2}_dist_standalone_eh.

A reaction is labelled "ok" only if all 5 SPEs terminated normally, each has
exactly one FINAL SINGLE POINT ENERGY, the EDA table has all 8 rows, and
|sum(6 channels) - Bond Energy| < 0.02 kcal/mol.
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BUNDLE_ROOT, EH_TO_KCAL, load_config, log, orca_terminated_normally  # noqa: E402

FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
EDA_TABLE_RE = re.compile(r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.DOTALL)
ROW_RE = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.MULTILINE)
LABEL_TO_KEY = {
    "Bond Energy": "bond", "Orbital Energy": "orb",
    "Electrostatic Energy": "elst", "Pauli Energy": "pauli",
    "Delta E^0(XC)": "xc", "Delta Dispersion": "disp",
    "Delta CPCM Dielectric": "cpcm", "Delta SMD CDS correction": "smd_cds",
}
NEED = ("elst", "pauli", "xc", "orb", "disp", "cpcm", "smd_cds", "bond")
# Distorted-fragment SPE now comes from eda_frag{1,2}.out (ghost basis,
# BSSE-corrected). Standalone frag{1,2}_dist.out are still parsed for
# provenance but no longer drive d1/d2/eint_spe.
SPES = ("eda", "eda_frag1", "eda_frag2", "frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel")
SUM_TOL_KCAL = 0.02
IDENTITY_TOL_KCAL = 0.02      # gap = eint_spe − e_bond must be near zero under new BSSE-consistent scheme


def read_fspe(path, expected_count=1):
    """Last FINAL SINGLE POINT ENERGY, with a count check (single points must have exactly 1)."""
    vals = FSPE_RE.findall(Path(path).read_text(errors="replace"))
    if not vals:
        raise ValueError(f"{Path(path).name}: no FSPE")
    if expected_count is not None and len(vals) != expected_count:
        raise ValueError(f"{Path(path).name}: FSPE count={len(vals)} (expected {expected_count})")
    return float(vals[-1])


def parse_eda_table(eda_path):
    """Return dict key -> Hartree for the 8 table rows, or raise."""
    text = Path(eda_path).read_text(errors="replace")
    m = EDA_TABLE_RE.search(text)
    if not m:
        raise ValueError("EDA table not found")
    out = {}
    for row in ROW_RE.finditer(m.group(1)):
        label = row.group(1).strip()
        if label in LABEL_TO_KEY:
            out[LABEL_TO_KEY[label]] = float(row.group(2))
    missing = [k for k in NEED if k not in out]
    if missing:
        raise ValueError(f"EDA table missing rows: {missing}")
    return out


def channels_from_table(raw_eh):
    """Repo convention: Pauli + XC folded into pauli_dft; all from the Hartree column."""
    k = {key: v * EH_TO_KCAL for key, v in raw_eh.items()}
    ch = dict(
        elst_dft=k["elst"], pauli_dft=k["pauli"] + k["xc"], oi_dft=k["orb"],
        disp_dft=k["disp"], cpcm_dft=k["cpcm"], cds_dft=k["smd_cds"], e_bond_kcal=k["bond"],
    )
    ch["eda_sum_minus_bond_kcal"] = (ch["elst_dft"] + ch["pauli_dft"] + ch["oi_dft"]
                                     + ch["disp_dft"] + ch["cpcm_dft"] + ch["cds_dft"]) - ch["e_bond_kcal"]
    return ch


def parse_one_rxn(rdir, meta_row):
    rid = int(meta_row["rxn_id"])
    d = {
        "rxn_id": rid,
        "ts_file": meta_row.get("ts_file"),
        "n_atoms": int(meta_row["n_atoms"]), "n_f1": int(meta_row["n_f1"]), "n_f2": int(meta_row["n_f2"]),
        "rel1_file": meta_row.get("rel1_file"), "rel2_file": meta_row.get("rel2_file"),
        "role1": meta_row.get("role1", "dipole"), "role2": meta_row.get("role2", "dipolarophile"),
        "charge1": int(meta_row.get("charge1", 0) or 0), "charge2": int(meta_row.get("charge2", 0) or 0),
        "alt_used": int(meta_row.get("alt_used", 0) or 0),
        "flag_foreign_bond": bool(meta_row.get("flag_foreign_bond", False)),
        "flag_async": bool(meta_row.get("flag_async", False)),
        "formed_d1": float(meta_row["formed_d1"]), "formed_d2": float(meta_row["formed_d2"]),
        "formed_pairs_ts": meta_row.get("formed_pairs_ts"),
        "recovery_method": meta_row.get("recovery_method", "graph"),
    }
    outs = {s: rdir / f"{s}.out" for s in SPES}
    not_ok = [s for s, p in outs.items() if not orca_terminated_normally(p)]
    if not_ok:
        d["status"] = "incomplete"; d["_missing"] = not_ok
        return d
    try:
        e = {s: read_fspe(p, 1) for s, p in outs.items()}
        raw = parse_eda_table(outs["eda"])
    except ValueError as err:
        d["status"] = "parse_fail"; d["_error"] = str(err)
        return d
    d["e_ab_eh"] = e["eda"]
    # NEW (2026-09-15): distorted fragment reference uses ghost-basis SPE from eda_frag{1,2}.out
    d["e_frag1_dist_eh"] = e["eda_frag1"]; d["e_frag2_dist_eh"] = e["eda_frag2"]
    d["e_frag1_dist_standalone_eh"] = e["frag1_dist"]           # kept for provenance / BSSE audit
    d["e_frag2_dist_standalone_eh"] = e["frag2_dist"]
    d["e_frag1_rel_eh"] = e["frag1_rel"];   d["e_frag2_rel_eh"] = e["frag2_rel"]
    d["d1_kcal"] = (e["eda_frag1"] - e["frag1_rel"]) * EH_TO_KCAL      # strain of frag1 (BSSE-consistent)
    d["d2_kcal"] = (e["eda_frag2"] - e["frag2_rel"]) * EH_TO_KCAL      # strain of frag2 (BSSE-consistent)
    d["eint_spe_kcal"] = (e["eda"] - e["eda_frag1"] - e["eda_frag2"]) * EH_TO_KCAL
    d["barrier_kcal"] = (e["eda"] - e["frag1_rel"] - e["frag2_rel"]) * EH_TO_KCAL
    d["bsse_shift_kcal"] = ((e["eda_frag1"] - e["frag1_dist"]) + (e["eda_frag2"] - e["frag2_dist"])) * EH_TO_KCAL
    d["eda_raw_eh"] = raw
    d.update(channels_from_table(raw))
    d["eint_spe_minus_bond_kcal"] = d["eint_spe_kcal"] - d["e_bond_kcal"]
    d["identity_residual_kcal"] = d["d1_kcal"] + d["d2_kcal"] + d["eint_spe_kcal"] - d["barrier_kcal"]
    if abs(d["eda_sum_minus_bond_kcal"]) > SUM_TOL_KCAL:
        d["status"] = "eda_sum_mismatch"
        return d
    if abs(d["eint_spe_minus_bond_kcal"]) > IDENTITY_TOL_KCAL:
        d["status"] = "bsse_reference_mismatch"          # eda_frag*.out & Bond Energy inconsistent — investigate
        return d
    d["status"] = "ok"
    return d


def load_permanent_ids(work):
    ids = {}
    log_path = work / "progress.jsonl"
    if not log_path.is_file():
        return ids
    for line in log_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if not isinstance(e, dict):
            continue
        try:
            rid = int(e["rxn"].split("_")[1])
        except Exception:
            continue
        if e.get("status") == "permanent":
            ids[rid] = e.get("classes")
        elif e.get("status") == "done":
            # Records are chronological. A reaction re-run via
            # nurion/sweep.sh INCLUDE_PERMANENT=1 (the 2026-09-10 quota
            # casualties) ends with "done" and must not stay excluded.
            ids.pop(rid, None)
    return ids


def main():
    cfg = load_config()
    work = BUNDLE_ROOT / cfg["work_dir"]
    inputs_root = work / "inputs"
    meta_csv = work / "input_meta.csv"
    assert meta_csv.is_file(), "run stage 1 first"
    meta = pd.read_csv(meta_csv)
    log(f"stage3: parsing {len(meta)} rxns")

    labels, failures, stats = [], [], Counter()
    all_parsed = []
    permanent_map = load_permanent_ids(work)
    for _, row in meta.iterrows():
        rid = int(row["rxn_id"])
        rdir = inputs_root / f"rxn_{rid:04d}"
        if rid in permanent_map:
            failures.append({"rxn_id": rid, "reason": "permanent_stage2", "classes": permanent_map[rid]})
            stats["permanent_stage2"] += 1
            continue
        if not rdir.is_dir():
            failures.append({"rxn_id": rid, "reason": "missing_dir"}); stats["missing_dir"] += 1
            continue
        d = parse_one_rxn(rdir, row)
        stats[d["status"]] += 1
        if d["status"] in ("ok", "eda_sum_mismatch"):
            # labels_all.json keeps every fully parsed reaction together with
            # its status and eda_sum_minus_bond_kcal, so the channel-sum
            # threshold can be re-decided downstream without re-parsing.
            all_parsed.append(d)
        if d["status"] == "ok":
            labels.append(d)
        else:
            failures.append({"rxn_id": rid, "reason": d["status"],
                             "detail": d.get("_error") or d.get("_missing") or d.get("eda_sum_minus_bond_kcal")})

    labels.sort(key=lambda x: x["rxn_id"]); failures.sort(key=lambda x: x["rxn_id"])
    for k, v in stats.most_common():
        log(f"  {k}: {v}")
    json.dump(labels, open(work / "labels.json", "w"), indent=2)
    all_parsed.sort(key=lambda x: x["rxn_id"])
    json.dump(all_parsed, open(work / "labels_all.json", "w"), indent=2)
    log(f"  labels_all.json: {len(all_parsed)} parsed rxns (ok + eda_sum_mismatch), status field per rxn")
    json.dump(failures, open(work / "failures.json", "w"), indent=2)
    json.dump({
        "generated_utc": datetime.now(timezone.utc).isoformat(), "bundle_version": "2.1",
        "n_target": len(meta), "n_labeled_ok": len(labels), "n_failures": len(failures),
        "status_counts": dict(stats),
        "channel_convention": "pauli_dft = Pauli Energy + Delta E^0(XC); all from Hartree column x 627.5094740631",
        "fragment_reference_convention": "e_frag{1,2}_dist_eh from eda_frag{1,2}.out (ghost basis, BSSE-consistent); standalone frag*_dist.out retained as e_frag{1,2}_dist_standalone_eh. Under this scheme eint_spe == e_bond (tol 0.02 kcal/mol) and d1+d2+eint_spe == barrier exactly.",
        "config": cfg, "spec_reference": "SPEC.md",
    }, open(work / "metadata.json", "w"), indent=2)
    (work / "GATE3_STATUS.txt").write_text(f"labeled_ok={len(labels)} failures={len(failures)}\n")
    log("stage3 complete")


if __name__ == "__main__":
    main()
