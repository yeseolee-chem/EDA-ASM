#!/usr/bin/env python3
"""Paper-inspired fragmentation candidates for still-FAIL reactions.

Built on the prescriptions in Fernández & Bickelhaupt, Chem. Soc. Rev. 2014,
43, 4953:
  - Bimolecular ASM: fragments = the separate reactants 'from infinity'.
    For our P → R reading: if P is dissociated into two species, those are
    the natural ASM fragments. Implementation: connected components of
    bonds_P.
  - Unimolecular (§4): "careful and chemically meaningful fragmentation".
    The paper's examples cleave a single defining bond *homolytically*:
      • Type-I dyotropic (§4.1): rotate [X..X] vs [H2C=CH2] — fragments
        obtained by cleaving the two C–X bonds homolytically.
      • Hopf cyclization (§4.2): cleave the newly forming σ bond → two
        doublet radicals.

Strategies emitted per reaction:

  P_product       Connected components of bonds_P (only if nC_P ≥ 2 in P).
                  Each component → fragment; multiplicity from electron parity;
                  BS-singlet coupling if ≥2 open shells.
  H_break_<n>     Homolytic cleavage of the n-th broken bond (R-only bond).
                  R-graph minus that one bond → components.
  H_form_<n>      Homolytic cleavage of the n-th formed bond (P-only bond).
                  P-graph minus that one bond → components.
  DA_path_min     Minimum-cut between donor/acceptor sets in (bonds_R ∪
                  bonds_P) — already covered by c4_da_single, included for
                  multiplicity-sweep coverage.

For all paper-inspired candidates we assign multiplicity = 2 to any fragment
left with an odd electron count (homolytic cleavage always produces unpaired
electrons) and pair them as BS-singlet by default.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
WINNER_S5 = ROOT / "Validate" / "refrag" / "stage5a" / "per_reaction"
CAND_S5 = ROOT / "Validate" / "refrag" / "candidates_stage5a" / "per_reaction"
CAND_SUM = ROOT / "Validate" / "refrag" / "candidate_summary.json"
DB_IDX = ROOT / "outputs" / "asr_spec" / "db_idx_map.json"
DIAG = ROOT / "Validate" / "refrag" / "still_fail_diagnosis.json"

Z_OF = {"H":1,"B":5,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"Cl":17,"Br":35,"I":53}
SYM_OF = {v: k for k, v in Z_OF.items()}


def connected_components(n_atoms: int, edges) -> list[set[int]]:
    g: dict[int, set[int]] = collections.defaultdict(set)
    for a, b in edges:
        g[a].add(b); g[b].add(a)
    seen: set[int] = set()
    comps: list[set[int]] = []
    for v in range(n_atoms):
        if v in seen:
            continue
        comp: set[int] = set()
        stack = [v]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x)
            for y in g[x]:
                if y not in seen:
                    stack.append(y)
        comps.append(comp)
    return comps


def _e_count(atoms, symbols) -> int:
    if symbols is None:
        return len(atoms)
    return sum(Z_OF.get(symbols[a], 0) for a in atoms)


def _coupling_label(mults: list[int], signs: list[int]) -> str:
    n_open = sum(1 for m in mults if m > 1)
    if n_open == 0: return "closed_shell_singlet"
    if n_open == 1: return f"multiplet_{max(mults)}"
    if all(signs[i] > 0 for i, m in enumerate(mults) if m > 1):
        return "ferromagnetic_high_spin"
    return "broken_symmetry_singlet"


def _build_stage5a(orig: dict, comps: list[set[int]], symbols, label: str) -> dict:
    """Wrap a list of atom-sets as a stage5a result.json, with multiplicity
    assigned from electron parity. Homolytic cleavage → doublet/quartet etc."""
    comps_sorted = sorted(comps, key=lambda c: -len(c))
    fragments: list[dict] = []
    for i, comp in enumerate(comps_sorted):
        atoms = sorted(comp)
        ne = _e_count(comp, symbols)
        # Homolytic-cleavage convention: any odd-electron count → doublet
        mult = 1 if (ne % 2 == 0) else 2
        fragments.append({
            "atom_indices": atoms,
            "role": f"comp_{i}",
            "multiplicity": mult,
            "cap_attachment": None,
        })
    # Antiferromagnetic pairing for open-shell fragments
    spin_signs: list[int] = []
    open_i = 0
    for f in fragments:
        if f["multiplicity"] > 1:
            spin_signs.append(1 if open_i % 2 == 0 else -1)
            open_i += 1
        else:
            spin_signs.append(1)
    total_spin = sum(s * (m - 1) for s, m in zip(spin_signs,
                                                   [f["multiplicity"] for f in fragments]))
    new = dict(orig)
    new["result"] = {
        "pattern": label,
        "fragments": fragments,
        "spin_signs": spin_signs,
        "total_spin_polarization": int(total_spin),
        "coupling": _coupling_label([f["multiplicity"] for f in fragments], spin_signs),
        "n_fragments": len(fragments),
        "cap_h_positions": None,
        "confidence": 0.85,
        "notes": (f"Paper-inspired ({label}): homolytic cleavage convention; "
                  "doublet multiplicities from electron parity; BS-singlet pairing."),
        "debug": {"source": "derive_paper_inspired.py", "label": label},
    }
    new["fragmentation_revision"] = 7
    return new


def _atom_symbols_for(rid: str, cache):
    e = cache.get(rid)
    if e is None: return None
    try:
        return [SYM_OF.get(int(z), "?") for z in e.numbers]
    except Exception:
        return None


def _comps_key(comps):
    return tuple(sorted(tuple(sorted(c)) for c in comps))


def main() -> int:
    sys.path.insert(0, str(ROOT / "Validate"))
    import pickle
    with open(ROOT / "ADF_500/stage5a/frames_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)

    diag = json.loads(DIAG.read_text())
    cand_sum = json.loads(CAND_SUM.read_text())
    db_idx_map = json.loads(DB_IDX.read_text())

    n_p = n_h = 0
    rows: list[dict] = []
    skipped_dup: list[str] = []
    skipped_schema: list[str] = []

    for rec in diag["rows"]:
        rid = rec["rid"]
        # Schema-corrupt sources can still try paper-inspired; the run itself
        # may surface clean data. Don't skip just because the *previous* run
        # had non_finite values.
        src = ROOT / "ADF_500/stage5a/per_reaction" / rid / "result.json"
        if not src.exists():
            skipped_schema.append(f"{rid}: no source stage5a")
            continue
        s5 = json.loads(src.read_text())
        symbols = _atom_symbols_for(rid, cache)
        dbg = s5["debug"]
        n = s5["n_atoms"]
        bonds_R = [tuple(sorted(e)) for e in dbg["bonds_R"]]
        bonds_P = [tuple(sorted(e)) for e in dbg["bonds_P"]]
        bonds_broken = [tuple(sorted(e)) for e in dbg["bonds_broken"]]
        bonds_formed = [tuple(sorted(e)) for e in dbg["bonds_formed"]]

        existing = cand_sum["rids"].get(rid, {"n_candidates": 0, "candidates": []})
        existing_keys = set()
        for c in existing["candidates"]:
            # Match by atom partition signature
            comps_listed = []
            # Don't have access to the actual partition; use synth_rid alone
            existing_keys.add(c.get("synth_rid"))
        # Synthetic-rid counter for paper-inspired candidates
        n_existing_p = sum(1 for c in existing["candidates"]
                            if c["synth_rid"].rsplit("__p", 1)[-1].isdigit()
                              and "__p" in c["synth_rid"])
        n_existing_h = sum(1 for c in existing["candidates"]
                            if c["synth_rid"].rsplit("__h", 1)[-1].isdigit()
                              and "__h" in c["synth_rid"])

        # Track unique partitions tried for this rid (best effort: by atom set)
        tried_partitions: set = set()
        for c in existing["candidates"]:
            sp = CAND_S5 / c["synth_rid"] / "result.json"
            if sp.exists():
                try:
                    ss = json.loads(sp.read_text())
                    parts = [set(f["atom_indices"]) for f in ss["result"]["fragments"]]
                    tried_partitions.add(_comps_key(parts))
                except Exception:
                    pass

        def _emit(comps, label_suffix, kind: str) -> bool:
            nonlocal n_p, n_h
            key = _comps_key(comps)
            if key in tried_partitions:
                skipped_dup.append(f"{rid}__{label_suffix} (dup partition)")
                return False
            # Need at least 2 components and no empty
            if len(comps) < 2 or any(len(c) == 0 for c in comps):
                return False
            tried_partitions.add(key)
            counter = "__p" if kind == "P" else "__h"
            next_idx = (n_existing_p if kind == "P" else n_existing_h)
            if kind == "P":
                idx = n_existing_p
            else:
                idx = n_existing_h
            synth = f"{rid}{counter}{idx}"
            if kind == "P":
                n_existing_p_local = idx + 1
            else:
                n_existing_h_local = idx + 1
            db_idx_map[synth] = db_idx_map.get(rid)
            if db_idx_map[synth] is None:
                return False
            new_s = _build_stage5a(s5, comps, symbols, f"paper_{label_suffix}")
            new_s["reaction_id"] = synth
            out_p = CAND_S5 / synth / "result.json"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(new_s, indent=2))
            frags = new_s["result"]["fragments"]
            existing["candidates"].append({
                "synth_rid": synth,
                "label": f"paper_{label_suffix}",
                "n_fragments": len(frags),
                "fragment_sizes": sorted([len(f["atom_indices"]) for f in frags], reverse=True),
                "multiplicities": [f["multiplicity"] for f in frags],
                "coupling": new_s["result"]["coupling"],
            })
            rows.append({"rid": rid, "synth_rid": synth, "label": label_suffix,
                          "fragment_sizes": sorted([len(f["atom_indices"]) for f in frags],
                                                    reverse=True),
                          "mults": [f["multiplicity"] for f in frags]})
            if kind == "P": n_p += 1
            else: n_h += 1
            return True

        # ── Strategy P_product: bonds_P connected components (dissociative case) ─
        comps_P = connected_components(n, bonds_P)
        if len(comps_P) >= 2:
            # Re-index counters using lengths of existing candidates lists
            n_existing_p = sum(1 for c in existing["candidates"]
                                if c["synth_rid"].count("__p") and
                                c["synth_rid"].rsplit("__p", 1)[-1].isdigit())
            _emit(comps_P, "P_product", "P")

        # ── Strategy H_break_<n>: homolytic cleavage of each broken bond ─
        for i, bond in enumerate(bonds_broken[:3]):  # cap at 3 per rxn
            sub = [e for e in bonds_R if e != bond]
            comps_h = connected_components(n, sub)
            if len(comps_h) < 2:
                continue
            # Re-index counter
            n_existing_h = sum(1 for c in existing["candidates"]
                                if c["synth_rid"].count("__h") and
                                c["synth_rid"].rsplit("__h", 1)[-1].isdigit())
            _emit(comps_h, f"H_break_{bond[0]}_{bond[1]}", "H")

        # ── Strategy H_form_<n>: homolytic cleavage of each formed bond ─
        for i, bond in enumerate(bonds_formed[:3]):
            sub = [e for e in bonds_P if e != bond]
            comps_h = connected_components(n, sub)
            if len(comps_h) < 2:
                continue
            n_existing_h = sum(1 for c in existing["candidates"]
                                if c["synth_rid"].count("__h") and
                                c["synth_rid"].rsplit("__h", 1)[-1].isdigit())
            _emit(comps_h, f"H_form_{bond[0]}_{bond[1]}", "H")

        existing["n_candidates"] = len(existing["candidates"])
        cand_sum["rids"][rid] = existing

    DB_IDX.write_text(json.dumps(db_idx_map, indent=2))
    CAND_SUM.write_text(json.dumps(cand_sum, indent=2))

    out_p = ROOT / "Validate" / "refrag" / "paper_inspired_summary.json"
    out_p.write_text(json.dumps({
        "n_P_product": n_p,
        "n_H_cleavage": n_h,
        "total": n_p + n_h,
        "skipped_dup": skipped_dup[:50],
        "skipped_schema": skipped_schema,
        "rows": rows,
    }, indent=2))

    print(f"Paper-inspired candidates written: {n_p + n_h}")
    print(f"  P_product (bonds_P comps):       {n_p}")
    print(f"  H_break + H_form homolytic:      {n_h}")
    print(f"  Skipped (duplicate partition):   {len(skipped_dup)}")
    print(f"  Skipped (no source):             {len(skipped_schema)}")
    print(f"  Summary: {out_p}")
    print(f"  db_idx size: {len(db_idx_map)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
