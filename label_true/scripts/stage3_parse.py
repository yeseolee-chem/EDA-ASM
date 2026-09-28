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

Fragment reference convention (2026-09-28, option ①):
    d1/d2 and eint_spe use the standalone distorted-fragment SPEs
    **frag{1,2}_dist.out** (own basis, own cavity), so d1/d2 are pure
    deformation energies at one level with frag{1,2}_rel.out:
        d1 = E(frag1_dist) - E(frag1_rel)          (d2 likewise)
        eint_spe = E(AB) - E(frag1_dist) - E(frag2_dist)
        d1 + d2 + eint_spe == barrier               (ASM identity, exact)
    ORCA's Bond Energy (and the 6 channels) are defined against ORCA's own
    fragment SCFs eda_frag{1,2}.out (ghost basis of the other fragment, AB
    cavity). The difference is kept as its own column:
        bsse_shift_kcal = eint_spe - e_bond         (bsse_shift{1,2}_kcal per fragment)
        d1 + d2 + e_bond + bsse_shift == barrier    (exact)
        d1 + d2 + Σ(6 channels) + bsse_shift == barrier + eda_sum_minus_bond
                                                    (ORCA table closure only)
    Rejected: ② counterpoise-correcting only the interaction (d1 + d2 + bond !=
    barrier); ③ taking d1/d2 from eda_frag*.out (the 2026-09-15 scheme) —
    the 8-channel sum closes, but d1/d2 then carry ghost-basis and AB-cavity
    terms and are no longer deformation energies. The eda_frag values stay in
    the record as e_frag{1,2}_dist_ghost_eh.

A reaction is labelled "ok" only if all 7 SPEs terminated normally, each has
exactly one FINAL SINGLE POINT ENERGY, the EDA table has all 8 rows,
|sum(6 channels) - Bond Energy| < 0.02 kcal/mol, and Bond Energy equals
E(AB) - E(eda_frag1) - E(eda_frag2) within 0.02 kcal/mol.
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
# d1/d2/eint_spe come from the standalone frag{1,2}_dist.out; eda_frag{1,2}.out
# (ghost basis) only define ORCA's Bond Energy and the bsse_shift column.
SPES = ("eda", "eda_frag1", "eda_frag2", "frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel")
SUM_TOL_KCAL = 0.02
IDENTITY_TOL_KCAL = 0.02      # Bond Energy vs E(AB) - E(eda_frag1) - E(eda_frag2)


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
    # option ① (2026-09-28): distorted fragment = standalone frag{1,2}_dist.out
    d["e_frag1_dist_eh"] = e["frag1_dist"]; d["e_frag2_dist_eh"] = e["frag2_dist"]
    d["e_frag1_dist_ghost_eh"] = e["eda_frag1"]                  # ORCA EDA fragment SCF (ghost basis, AB cavity)
    d["e_frag2_dist_ghost_eh"] = e["eda_frag2"]
    d["e_frag1_rel_eh"] = e["frag1_rel"];   d["e_frag2_rel_eh"] = e["frag2_rel"]
    d["d1_kcal"] = (e["frag1_dist"] - e["frag1_rel"]) * EH_TO_KCAL     # deformation energy of frag1
    d["d2_kcal"] = (e["frag2_dist"] - e["frag2_rel"]) * EH_TO_KCAL     # deformation energy of frag2
    d["eint_spe_kcal"] = (e["eda"] - e["frag1_dist"] - e["frag2_dist"]) * EH_TO_KCAL
    d["barrier_kcal"] = (e["eda"] - e["frag1_rel"] - e["frag2_rel"]) * EH_TO_KCAL
    d["eda_raw_eh"] = raw
    d.update(channels_from_table(raw))
    d["bsse_shift1_kcal"] = (e["eda_frag1"] - e["frag1_dist"]) * EH_TO_KCAL
    d["bsse_shift2_kcal"] = (e["eda_frag2"] - e["frag2_dist"]) * EH_TO_KCAL
    d["bsse_shift_kcal"] = d["eint_spe_kcal"] - d["e_bond_kcal"]
    d["identity_residual_kcal"] = d["d1_kcal"] + d["d2_kcal"] + d["eint_spe_kcal"] - d["barrier_kcal"]
    # Bond Energy must be E(AB) - E(eda_frag1) - E(eda_frag2): otherwise the
    # 6 channels and bsse_shift refer to fragment SCFs other than these files
    d["bond_reference_residual_kcal"] = ((e["eda"] - e["eda_frag1"] - e["eda_frag2"]) * EH_TO_KCAL
                                         - d["e_bond_kcal"])
    d["channel_closure_residual_kcal"] = (d["d1_kcal"] + d["d2_kcal"] + d["elst_dft"] + d["pauli_dft"]
                                          + d["oi_dft"] + d["disp_dft"] + d["cpcm_dft"] + d["cds_dft"]
                                          + d["bsse_shift_kcal"] - d["barrier_kcal"])
    # reference check first: eda_sum_mismatch records are promoted to ok downstream,
    # so they must not hide a Bond Energy built from other fragment SCFs
    if abs(d["bond_reference_residual_kcal"]) > IDENTITY_TOL_KCAL:
        d["status"] = "bsse_reference_mismatch"          # eda_frag*.out & Bond Energy inconsistent — investigate
        return d
    if abs(d["eda_sum_minus_bond_kcal"]) > SUM_TOL_KCAL:
        d["status"] = "eda_sum_mismatch"
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
        "fragment_reference_convention": "option ① (2026-09-28): e_frag{1,2}_dist_eh = standalone frag{1,2}_dist.out (own basis, own cavity); d1/d2 = deformation energies vs frag{1,2}_rel.out; eint_spe = E(AB) - E(frag1_dist) - E(frag2_dist), so d1+d2+eint_spe == barrier exactly. ORCA Bond Energy / 6 channels refer to eda_frag{1,2}.out (ghost basis, AB cavity; kept as e_frag{1,2}_dist_ghost_eh); bsse_shift_kcal = eint_spe - e_bond, so d1+d2+e_bond+bsse_shift == barrier exactly and d1+d2+sum(6ch)+bsse_shift == barrier + eda_sum_minus_bond_kcal.",
        "config": cfg, "spec_reference": "SPEC.md",
    }, open(work / "metadata.json", "w"), indent=2)
    (work / "GATE3_STATUS.txt").write_text(f"labeled_ok={len(labels)} failures={len(failures)}\n")
    log("stage3 complete")


if __name__ == "__main__":
    main()
