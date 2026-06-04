#!/usr/bin/env python3
"""Generate spin/multiplicity variants of the current winner fragmentation.

For reactions whose Check-4 residual is still > 0.5 kcal/mol after the
fragmentation-candidate sweep, the residual usually traces to a poorly-chosen
spin state (BS-singlet vs closed-shell vs triplet) rather than a bad fragment
partition. This script enumerates 1–3 spin variants per still-FAIL reaction
without changing the winner fragmentation, and writes them as new synthetic
candidates so the existing runner/selector picks them up.

Variant strategies (only those compatible with electron parity per fragment
are emitted):
  s0_closed_shell       all fragments forced mult=1 (only if all e-counts even)
  s1_bs_singlet         lowest-multiplicity open-shells, antiferromagnetic
  s2_ferromagnetic      lowest-multiplicity open-shells, parallel spins
  s3_high_spin          promote largest open-shell fragment to next mult level

Priority: reactions are ordered by:
  1) no Check-3 ts_not_max (excluded — geometric issue, not spin)
  2) no Check-1 schema (excluded — data corruption)
  3) residual size (smaller first → more likely to close with spin correction)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
WINNER_S5_DIR = ROOT / "Validate" / "refrag" / "stage5a" / "per_reaction"
CAND_S5_DIR = ROOT / "Validate" / "refrag" / "candidates_stage5a" / "per_reaction"
CANDIDATE_SUMMARY = ROOT / "Validate" / "refrag" / "candidate_summary.json"
SELECTION_REPORT = ROOT / "Validate" / "refrag" / "selection_report.json"
DB_IDX_PATH = ROOT / "outputs" / "asr_spec" / "db_idx_map.json"

Z_OF: dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
SYM_OF = {v: k for k, v in Z_OF.items()}


def _atom_symbols_for(rid: str, cache):
    """Symbols for one reaction; None if not in cache."""
    e = cache.get(rid)
    if e is None:
        return None
    try:
        return [SYM_OF.get(int(z), "?") for z in e.numbers]
    except Exception:
        return None


def _e_count(atoms, symbols) -> int:
    """Sum of atomic numbers as electron-count proxy."""
    if symbols is None:
        return len(atoms)
    return sum(Z_OF.get(symbols[a], 0) for a in atoms)


def _spin_signs(mults: list[int]) -> list[int]:
    """Antiferromagnetic alternation for open-shell fragments."""
    out: list[int] = []
    open_i = 0
    for m in mults:
        if m > 1:
            out.append(1 if open_i % 2 == 0 else -1)
            open_i += 1
        else:
            out.append(1)
    return out


def _coupling_label(mults: list[int], signs: list[int]) -> str:
    """Describe the spin state qualitatively."""
    n_open = sum(1 for m in mults if m > 1)
    if n_open == 0:
        return "closed_shell_singlet"
    if n_open == 1:
        return "doublet" if 2 in mults else f"multiplet_{max(mults)}"
    # multi open-shell
    same_sign = all(s > 0 for s in signs if any(m > 1 for m in mults))
    if all(signs[i] > 0 for i, m in enumerate(mults) if m > 1):
        return "ferromagnetic" if n_open >= 2 else "doublet"
    return "broken_symmetry_singlet"


def _variants_for(frags: list[dict], symbols) -> list[dict]:
    """Enumerate (mults, signs, label) variants keeping the partition fixed."""
    e_counts = [_e_count(set(f["atom_indices"]), symbols) for f in frags]
    cur_mults = tuple(int(f["multiplicity"]) for f in frags)
    cur_signs = tuple(_spin_signs(list(cur_mults)))

    seen: set = {(cur_mults, cur_signs)}
    out: list[dict] = []

    def _try(mults, signs, label, parity_check=True):
        m_t, s_t = tuple(mults), tuple(signs)
        if parity_check:
            # Each fragment's multiplicity-1 must match its electron parity
            for i, (m, e) in enumerate(zip(mults, e_counts)):
                if (m - 1) % 2 != e % 2:
                    return
        if (m_t, s_t) in seen:
            return
        seen.add((m_t, s_t))
        out.append({"mults": list(mults), "signs": list(signs), "label": label})

    # 1) Closed shell — only if every fragment has even electrons
    if all(e % 2 == 0 for e in e_counts):
        _try([1] * len(frags), [1] * len(frags), "s0_closed_shell")

    # 2) Lowest open-shells matching parity
    base_mults = [1 if e % 2 == 0 else 2 for e in e_counts]
    open_idx = [i for i, m in enumerate(base_mults) if m > 1]
    if len(open_idx) >= 2:
        # BS-singlet
        signs = [1] * len(frags)
        for j, oi in enumerate(open_idx):
            signs[oi] = 1 if j % 2 == 0 else -1
        _try(base_mults, signs, "s1_bs_singlet")
        # Ferromagnetic
        _try(base_mults, [1] * len(frags), "s2_ferromagnetic")

    # 3) High-spin: promote the LARGEST fragment to mult+2
    if open_idx:
        sizes = [(i, len(frags[i]["atom_indices"])) for i in open_idx]
        biggest = max(sizes, key=lambda x: x[1])[0]
        mults_hs = list(base_mults)
        mults_hs[biggest] = base_mults[biggest] + 2
        if mults_hs[biggest] >= 5:
            mults_hs[biggest] = base_mults[biggest]  # don't go to mult=5
        else:
            _try(mults_hs, [1] * len(frags), "s3_high_spin")

    return out


def _classify_skip(d, raw):
    """Return reason to skip (or None to keep)."""
    if d.max_abs_res_cons is None:
        return "no_residual"
    # Schema corruption: e.g. None in asr vector
    if not raw.get("asr_vector_kcal", {}).get("TS"):
        return "schema_no_asr"
    if any(v is None for v in raw["asr_vector_kcal"].get("TS", {}).values()):
        return "schema_none_value"
    # E_TS topology
    irc = raw.get("irc_points", {})
    if all(k in irc for k in ("R", "TS", "P")):
        e_r = irc["R"].get("energy_kcal_adf")
        e_ts = irc["TS"].get("energy_kcal_adf")
        e_p = irc["P"].get("energy_kcal_adf")
        if e_r is not None and e_ts is not None and e_p is not None:
            if e_ts <= e_r or e_ts <= e_p:
                return "ts_not_max"
    return None


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "Validate"))
    from validate_asr import derive  # type: ignore

    import pickle
    with open(ROOT / "ADF_500/stage5a/frames_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)

    sel = json.loads(SELECTION_REPORT.read_text())
    print(f"Reactions with winners: {len(sel['winners'])}")

    targets: list[dict] = []
    skipped_low_res: list[str] = []
    skipped_ts: list[str] = []
    skipped_schema: list[str] = []

    for w in sel["winners"]:
        rid = w["rid"]
        verdict = w["winner"]["verdict"]
        res = w["winner"]["max_abs_res_cons"]
        if verdict in ("PASS", "WARN"):
            skipped_low_res.append(rid)
            continue
        # Load winner result + winner stage5a
        rp = ROOT / "Validate" / "refrag" / "results" / f"{rid}.json"
        sp = WINNER_S5_DIR / rid / "result.json"
        if not rp.exists() or not sp.exists():
            continue
        raw = json.loads(rp.read_text())
        stage5a = json.loads(sp.read_text())
        d = derive("w", raw)
        skip = _classify_skip(d, raw)
        if skip == "ts_not_max":
            skipped_ts.append(rid); continue
        if skip in ("schema_no_asr", "schema_none_value", "no_residual"):
            skipped_schema.append(f"{rid} ({skip})"); continue
        targets.append({
            "rid": rid,
            "residual": res,
            "stage5a": stage5a,
            "verdict_now": verdict,
        })

    # Sort by residual ascending (small residuals = closer to being fixable)
    targets.sort(key=lambda t: t["residual"])

    print(f"  Skipped (already passing):  {len(skipped_low_res)}")
    print(f"  Skipped (ts_not_max):       {len(skipped_ts)}")
    print(f"  Skipped (schema/asr None):  {len(skipped_schema)}")
    print(f"  Targets for spin variants:  {len(targets)}")
    print()

    db_idx_map = json.loads(DB_IDX_PATH.read_text())
    cand_summary = json.loads(CANDIDATE_SUMMARY.read_text())

    n_written = 0
    n_no_variants = 0
    rows: list[dict] = []

    for tgt in targets:
        rid = tgt["rid"]
        stage5a = tgt["stage5a"]
        frags = stage5a["result"]["fragments"]
        symbols = _atom_symbols_for(rid, cache)
        variants = _variants_for(frags, symbols)
        if not variants:
            n_no_variants += 1
            continue

        # Ensure candidate_summary has an entry for this rid (it might already)
        existing = cand_summary["rids"].get(rid, {"n_candidates": 0, "candidates": []})
        # Find current max __c<N> / __s<N> index
        existing_labels = {c["label"] for c in existing["candidates"]}
        # Use __s<N> indexing for spin variants
        next_idx = max(
            [int(c["synth_rid"].rsplit("__s", 1)[-1])
             for c in existing["candidates"]
             if "__s" in c["synth_rid"] and c["synth_rid"].rsplit("__s", 1)[-1].isdigit()],
            default=-1,
        ) + 1

        for v in variants:
            if v["label"] in existing_labels:
                continue
            synth_rid = f"{rid}__s{next_idx}"
            next_idx += 1
            db_idx_map[synth_rid] = db_idx_map[rid]

            # Build new fragments list with same atom partition but new mults
            new_frags = []
            for i, f in enumerate(frags):
                new_frags.append({
                    "atom_indices": sorted(f["atom_indices"]),
                    "role": f"comp_{i}",
                    "multiplicity": v["mults"][i],
                    "cap_attachment": None,
                })
            total_spin = sum(s * (m - 1) for s, m in zip(v["signs"], v["mults"]))
            new_stage5a = dict(stage5a)
            new_stage5a["reaction_id"] = synth_rid
            new_stage5a["result"] = {
                "pattern": v["label"],
                "fragments": new_frags,
                "spin_signs": v["signs"],
                "total_spin_polarization": int(total_spin),
                "coupling": _coupling_label(v["mults"], v["signs"]),
                "n_fragments": len(new_frags),
                "cap_h_positions": None,
                "confidence": 0.8,
                "notes": f"Spin variant of winner fragmentation ({v['label']})",
                "debug": {"source": "derive_spin_variants.py", "label": v["label"]},
            }
            new_stage5a["fragmentation_revision"] = 5
            out_path = CAND_S5_DIR / synth_rid / "result.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(new_stage5a, indent=2))

            existing["candidates"].append({
                "synth_rid": synth_rid,
                "label": v["label"],
                "n_fragments": len(new_frags),
                "fragment_sizes": sorted([len(f["atom_indices"]) for f in new_frags],
                                          reverse=True),
                "multiplicities": v["mults"],
                "coupling": new_stage5a["result"]["coupling"],
            })
            n_written += 1
            rows.append({
                "rid": rid,
                "synth_rid": synth_rid,
                "label": v["label"],
                "mults": v["mults"],
                "signs": v["signs"],
                "winner_residual": tgt["residual"],
            })

        existing["n_candidates"] = len(existing["candidates"])
        cand_summary["rids"][rid] = existing

    DB_IDX_PATH.write_text(json.dumps(db_idx_map, indent=2))
    CANDIDATE_SUMMARY.write_text(json.dumps(cand_summary, indent=2))

    # Save a focused report of new spin candidates
    new_report = ROOT / "Validate" / "refrag" / "spin_variant_summary.json"
    new_report.write_text(json.dumps({
        "n_targets": len(targets),
        "n_no_variants_possible": n_no_variants,
        "n_spin_variants_written": n_written,
        "skipped_ts_not_max": skipped_ts,
        "skipped_schema": skipped_schema,
        "skipped_already_passing": skipped_low_res,
        "rows": rows,
    }, indent=2))

    print(f"Spin variants written: {n_written}")
    print(f"No variants possible:  {n_no_variants}")
    print(f"summary: {new_report}")
    print(f"db_idx_map size: {len(db_idx_map)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
