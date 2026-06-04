#!/usr/bin/env python3
"""Derive alternative fragmentations for the 62 check3/check4 FAIL reactions.

Strategy: kept-bonds connected components. A "kept bond" is one present in
both R and P (the inert skeleton). Removing all reactive bonds (broken +
formed) and isolating reactive atoms gives connected components in the
kept-bond graph; these are physically defensible fragment groups because
they share only non-reactive bonds with the rest. Isolated reactive atoms
(e.g., a migrating H) are folded into the component they bond to in R.
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SRC_STAGE5A = ROOT / "ADF_500" / "stage5a" / "per_reaction"
OUT_STAGE5A = ROOT / "Validate" / "refrag" / "stage5a" / "per_reaction"
MANIFEST = ROOT / "Validate" / "manifest.csv"

# Atomic numbers needed to count electrons for multiplicity assignment.
Z_OF: dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def connected_components(n_atoms: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    """Undirected connected components over atom indices 0..n_atoms-1."""
    g: dict[int, set[int]] = collections.defaultdict(set)
    for a, b in edges:
        g[a].add(b)
        g[b].add(a)
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
            seen.add(x)
            comp.add(x)
            for y in g[x]:
                if y not in seen:
                    stack.append(y)
        comps.append(comp)
    return comps


def _load_failing_rids() -> list[str]:
    """Pull reaction_ids that FAILed Check 3 or Check 4 from manifest.csv."""
    rids: list[str] = []
    for row in csv.DictReader(open(MANIFEST)):
        if row["verdict"] != "FAIL":
            continue
        fc = set(row["failed_checks"].split(";"))
        if "3" in fc or "4" in fc:
            rids.append(row["reaction_id"])
    return rids


def _atom_symbols(rid: str) -> list[str] | None:
    """Recover element symbols by reading frames_cache.pkl if available, else None."""
    try:
        import pickle
        cache_path = ROOT / "ADF_500" / "stage5a" / "frames_cache.pkl"
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        entry = cache.get(rid) if isinstance(cache, dict) else None
        if entry and "R" in entry:
            return list(entry["R"].get("symbols", []))
    except Exception:
        pass
    return None


def _multiplicity(n_electrons: int) -> int:
    """1 for closed shell (even), 2 for open shell (odd)."""
    return 1 if (n_electrons % 2 == 0) else 2


def _merge_singletons_into_neighbors(
    comps: list[set[int]],
    g_R: dict[int, set[int]],
    g_P: dict[int, set[int]],
) -> list[set[int]]:
    """Fold isolated reactive atoms into the component they bond to in R, then P."""
    while True:
        target_idx = None
        for i in range(len(comps)):
            if len(comps[i]) == 1 and len(comps) > 1:
                target_idx = i
                break
        if target_idx is None:
            break
        atom = next(iter(comps[target_idx]))
        merge_into = None
        for graph in (g_R, g_P):
            neigh = graph.get(atom, set())
            for j, c in enumerate(comps):
                if j == target_idx:
                    continue
                if c & neigh:
                    merge_into = j
                    break
            if merge_into is not None:
                break
        if merge_into is None:
            break
        comps[merge_into] = comps[merge_into] | comps[target_idx]
        comps.pop(target_idx)
    return sorted(comps, key=lambda c: -len(c))


def _cap_to_three(comps: list[set[int]], g_R: dict[int, set[int]]) -> list[set[int]]:
    """If more than 3 fragments, merge the smallest into its strongest R-neighbor."""
    while len(comps) > 3:
        comps = sorted(comps, key=lambda c: -len(c))
        smallest = comps.pop()
        best = 0
        best_overlap = -1
        for i, c in enumerate(comps):
            overlap = sum(1 for a in smallest if g_R.get(a, set()) & c)
            if overlap > best_overlap:
                best_overlap = overlap
                best = i
        comps[best] = comps[best] | smallest
    return sorted(comps, key=lambda c: -len(c))


def derive_alt(stage5a: dict, symbols: list[str] | None) -> dict | None:
    """Return a new fragments dict (or None if alt is not meaningfully different)."""
    n_atoms = int(stage5a["n_atoms"])
    dbg = stage5a["debug"]
    bonds_R = [tuple(sorted(e)) for e in dbg["bonds_R"]]
    bonds_P = [tuple(sorted(e)) for e in dbg["bonds_P"]]
    kept = list(set(bonds_R) & set(bonds_P))

    g_R: dict[int, set[int]] = collections.defaultdict(set)
    for a, b in bonds_R:
        g_R[a].add(b); g_R[b].add(a)
    g_P: dict[int, set[int]] = collections.defaultdict(set)
    for a, b in bonds_P:
        g_P[a].add(b); g_P[b].add(a)

    comps = connected_components(n_atoms, kept)
    if len(comps) < 2:
        return None
    comps = sorted(comps, key=lambda c: -len(c))
    comps = _merge_singletons_into_neighbors(comps, g_R, g_P)
    comps = _cap_to_three(comps, g_R)
    if len(comps) < 2:
        return None

    # Check that this alt is actually different from original.
    orig = [set(f["atom_indices"]) for f in stage5a["result"]["fragments"]]
    orig_norm = sorted(tuple(sorted(s)) for s in orig)
    alt_norm = sorted(tuple(sorted(s)) for s in comps)
    if orig_norm == alt_norm:
        return None

    fragments: list[dict] = []
    for i, comp in enumerate(sorted(comps, key=lambda c: -len(c))):
        atom_idx = sorted(comp)
        if symbols is not None:
            n_e = sum(Z_OF.get(symbols[k], 0) for k in atom_idx)
        else:
            n_e = len(atom_idx)
        mult = _multiplicity(n_e)
        fragments.append({
            "atom_indices": atom_idx,
            "role": f"comp_{i}",
            "multiplicity": mult,
            "cap_attachment": None,
        })

    open_shells = [f for f in fragments if f["multiplicity"] > 1]
    if len(open_shells) >= 2:
        coupling = "broken_symmetry_singlet"
    elif len(open_shells) == 1:
        coupling = "doublet"
    else:
        coupling = "closed_shell_singlet"

    spin_signs = []
    open_i = 0
    for f in fragments:
        if f["multiplicity"] > 1:
            spin_signs.append(1 if open_i % 2 == 0 else -1)
            open_i += 1
        else:
            spin_signs.append(1)
    total_spin = sum(s * (f["multiplicity"] - 1) for f, s in zip(fragments, spin_signs))

    return {
        "pattern": stage5a["result"]["pattern"] + "_ALT_KEPTBOND",
        "fragments": fragments,
        "spin_signs": spin_signs,
        "total_spin_polarization": int(total_spin),
        "coupling": coupling,
        "n_fragments": len(fragments),
        "cap_h_positions": None,
        "confidence": 0.7,
        "notes": "Kept-bond connected-component cut; isolated reactive atoms folded "
                 "into R-neighbor component; multiplicities from electron parity.",
        "debug": {"source": "Validate/refrag/derive_alt_frag.py"},
    }


def main() -> int:
    """Iterate FAILed rids and emit alt stage5a result.json files."""
    rids = _load_failing_rids()
    print(f"Failing reactions: {len(rids)}")

    OUT_STAGE5A.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped_same = 0
    n_skipped_single = 0
    skipped_rids: list[str] = []
    summary_rows: list[dict] = []

    for rid in rids:
        src = SRC_STAGE5A / rid / "result.json"
        if not src.exists():
            print(f"  [WARN] no source stage5a for {rid}, skip")
            continue
        stage5a = json.loads(src.read_text())
        symbols = _atom_symbols(rid)
        alt = derive_alt(stage5a, symbols)
        if alt is None:
            comps_n = len(connected_components(
                stage5a["n_atoms"],
                [tuple(sorted(e)) for e in (set(tuple(sorted(x)) for x in stage5a["debug"]["bonds_R"]) &
                                              set(tuple(sorted(x)) for x in stage5a["debug"]["bonds_P"]))]
            ))
            if comps_n < 2:
                n_skipped_single += 1
                skipped_rids.append(f"{rid} (kept-comps<2)")
            else:
                n_skipped_same += 1
                skipped_rids.append(f"{rid} (alt==orig)")
            continue

        new_stage5a = dict(stage5a)
        new_stage5a["result"] = alt
        new_stage5a["fragmentation_revision"] = 1
        out_path = OUT_STAGE5A / rid / "result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(new_stage5a, indent=2))
        n_written += 1

        summary_rows.append({
            "rid": rid,
            "pattern": stage5a["result"]["pattern"],
            "n_frag_orig": stage5a["result"]["n_fragments"],
            "n_frag_alt": alt["n_fragments"],
            "alt_coupling": alt["coupling"],
            "alt_multiplicities": [f["multiplicity"] for f in alt["fragments"]],
            "alt_fragment_sizes": [len(f["atom_indices"]) for f in alt["fragments"]],
        })

    summary_path = Path(__file__).parent / "alt_frag_summary.json"
    summary_path.write_text(json.dumps({
        "n_total": len(rids),
        "n_alt_written": n_written,
        "n_skipped_same_as_original": n_skipped_same,
        "n_skipped_single_component": n_skipped_single,
        "skipped_rids": skipped_rids,
        "rows": summary_rows,
    }, indent=2))

    print(f"Wrote alt fragmentations: {n_written}")
    print(f"Skipped (alt == original):   {n_skipped_same}")
    print(f"Skipped (single component):  {n_skipped_single}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
