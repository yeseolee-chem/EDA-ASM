#!/usr/bin/env python3
"""Generate up to 5 fragmentation candidates per failing reaction.

Candidate strategies:
  c0_v1_keptbond   — kept-bond connected components, floaters folded into R-neighbor
  c1_v2_3frag      — D/A min-cut on kept-bond skeleton, prefers 3 fragments
  c2_v2_2frag      — D/A min-cut, prefers 2 fragments (D-side + A-side, floater folded)
  c3_isolated_mig  — D/A min-cut, migrating atoms kept as their own (separate) fragment
  c4_da_single     — best single-bond cut on the donor→acceptor shortest path

Each candidate is written as a stage5a result.json with a synthetic rid of the
form '<original_rid>__c<N>' so the existing runner can execute it without
collisions. db_idx_map is expanded so the runner can find the Halo8 source DB
for every synthetic rid.

Writes:
  Validate/refrag/candidates_stage5a/<rid>__c<N>/result.json   (per candidate)
  outputs/asr_spec/db_idx_map.json                            (expanded inline)
  Validate/refrag/candidate_summary.json                      (manifest)
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SRC_STAGE5A = ROOT / "ADF_500" / "stage5a" / "per_reaction"
OUT_DIR = ROOT / "Validate" / "refrag" / "candidates_stage5a"
MANIFEST = ROOT / "Validate" / "manifest.csv"
DB_IDX_PATH = ROOT / "outputs" / "asr_spec" / "db_idx_map.json"

Z_OF: dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
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


def shortest_path(adj: dict, src: int, dst: int) -> list[int] | None:
    """BFS shortest path from src to dst; return [src, ..., dst] or None."""
    if src == dst:
        return [src]
    prev: dict[int, int] = {src: -1}
    q = collections.deque([src])
    while q:
        v = q.popleft()
        for u in adj.get(v, set()):
            if u in prev:
                continue
            prev[u] = v
            if u == dst:
                path = [u]
                while v != -1:
                    path.append(v)
                    v = prev[v]
                return list(reversed(path))
            q.append(u)
    return None


def _failing_rids() -> list[str]:
    rids: list[str] = []
    for row in csv.DictReader(open(MANIFEST)):
        if row["verdict"] != "FAIL":
            continue
        fc = set(row["failed_checks"].split(";"))
        if "3" in fc or "4" in fc:
            rids.append(row["reaction_id"])
    return rids


def _atom_symbols(rid: str, cache=None) -> list[str] | None:
    if cache is None:
        return None
    e = cache.get(rid)
    if e is None:
        return None
    try:
        return [SYM_OF.get(int(z), f"Z{int(z)}") for z in e.numbers]
    except Exception:
        return None


def _e_count(atoms, symbols):
    if symbols is None:
        return len(atoms)
    return sum(Z_OF.get(symbols[a], 0) for a in atoms)


def _adj(edges):
    g: dict[int, set[int]] = collections.defaultdict(set)
    for a, b in edges:
        g[a].add(b); g[b].add(a)
    return g


def _fold_floaters(comps: list[set[int]], floaters: set[int],
                    g_R: dict, g_P: dict) -> list[set[int]]:
    """Move each floater out of its singleton component into its R/P-neighbor's."""
    atom_to_ci = {a: i for i, c in enumerate(comps) for a in c}
    for fl in floaters:
        ci = atom_to_ci.get(fl)
        if ci is None or len(comps[ci]) != 1:
            continue
        target = None
        for g in (g_R, g_P):
            for nb in g.get(fl, set()):
                tci = atom_to_ci.get(nb)
                if tci is not None and tci != ci:
                    target = tci; break
            if target is not None:
                break
        if target is None:
            continue
        comps[target].add(fl)
        comps[ci].discard(fl)
        atom_to_ci[fl] = target
    return [c for c in comps if c]


def _comps_key(comps: list[set[int]]) -> tuple:
    return tuple(sorted(tuple(sorted(c)) for c in comps))


def _build_partition_dict(comps: list[set[int]], symbols, label: str):
    """Wrap a list of atom-sets as a stage5a 'result' dict."""
    comps_sorted = sorted(comps, key=lambda c: -len(c))
    fragments: list[dict] = []
    for i, comp in enumerate(comps_sorted):
        atoms = sorted(comp)
        ne = _e_count(comp, symbols)
        mult = 1 if (ne % 2 == 0) else 2
        fragments.append({
            "atom_indices": atoms,
            "role": f"comp_{i}",
            "multiplicity": mult,
            "cap_attachment": None,
        })
    open_shell = sum(1 for f in fragments if f["multiplicity"] > 1)
    if open_shell >= 2:
        coupling = "broken_symmetry_singlet"
    elif open_shell == 1:
        coupling = "doublet"
    else:
        coupling = "closed_shell_singlet"
    spin_signs: list[int] = []
    open_i = 0
    for f in fragments:
        if f["multiplicity"] > 1:
            spin_signs.append(1 if open_i % 2 == 0 else -1)
            open_i += 1
        else:
            spin_signs.append(1)
    total_spin = sum(s * (f["multiplicity"] - 1) for f, s in zip(fragments, spin_signs))
    return {
        "pattern": label,
        "fragments": fragments,
        "spin_signs": spin_signs,
        "total_spin_polarization": int(total_spin),
        "coupling": coupling,
        "n_fragments": len(fragments),
        "cap_h_positions": None,
        "confidence": 0.8,
        "notes": f"Generated by derive_candidates.py ({label})",
        "debug": {"source": "derive_candidates.py", "label": label},
    }


# ─── Strategies ─────────────────────────────────────────────────────────

def strat_c0_keptbond(stage5a, g_R, g_P, floaters):
    """v1: kept-bond components, floaters folded."""
    n = stage5a["n_atoms"]
    dbg = stage5a["debug"]
    kept = list(set(tuple(sorted(e)) for e in dbg["bonds_R"]) &
                 set(tuple(sorted(e)) for e in dbg["bonds_P"]))
    comps = connected_components(n, kept)
    comps = sorted(comps, key=lambda c: -len(c))
    comps = _fold_floaters(comps, floaters, g_R, g_P)
    # Merge any extra singletons into largest
    while len(comps) > 3:
        smallest = comps[-1]
        # merge into a neighbor by R-edges
        comps.pop()
        target = 0
        for i, c in enumerate(comps):
            if any(g_R.get(a, set()) & c for a in smallest):
                target = i; break
        comps[target] = comps[target] | smallest
        comps = sorted(comps, key=lambda c: -len(c))
    return comps if len(comps) >= 2 else None


def _enumerate_cuts(stage5a, g_R, g_P, floaters,
                     min_comps: int, max_comps: int,
                     keep_migrators_isolated: bool = False):
    """Enumerate 1/2/3-bond cuts; return list of unique component-lists."""
    n = stage5a["n_atoms"]
    dbg = stage5a["debug"]
    bonds_R = set(tuple(sorted(e)) for e in dbg["bonds_R"])
    bonds_P = set(tuple(sorted(e)) for e in dbg["bonds_P"])
    kept = list(bonds_R & bonds_P)

    donors: set[int] = set()
    acceptors: set[int] = set()
    for a, b in dbg["bonds_broken"]:
        donors.add(a); donors.add(b)
    for a, b in dbg["bonds_formed"]:
        acceptors.add(a); acceptors.add(b)
    migrating = donors & acceptors
    pure_donors = donors - migrating
    pure_acceptors = acceptors - migrating

    seen: set = set()
    candidates: list[tuple[float, list[set[int]]]] = []

    max_k = min(3, len(kept))
    for k in range(0, max_k + 1):
        for cut in itertools.combinations(kept, k):
            remaining = [e for e in kept if e not in cut]
            comps = connected_components(n, remaining)
            if keep_migrators_isolated:
                # Leave migrating singletons alone; fold non-migrating floaters
                non_mig_floaters = floaters - migrating
                comps = _fold_floaters(comps, non_mig_floaters, g_R, g_P)
            else:
                comps = _fold_floaters(comps, floaters, g_R, g_P)
            if len(comps) < min_comps or len(comps) > max_comps:
                continue
            # Skip more than 1 trivial (1-atom) component unless we explicitly want migrators isolated
            n_tiny = sum(1 for c in comps if len(c) == 1)
            allowed_tiny = len(migrating) if keep_migrators_isolated else 1
            if n_tiny > allowed_tiny:
                continue
            # Verify D/A separated when possible
            separated = True
            if pure_donors and pure_acceptors:
                d_ci = {i for i, c in enumerate(comps) for d in pure_donors if d in c}
                a_ci = {i for i, c in enumerate(comps) for a in pure_acceptors if a in c}
                if not d_ci.isdisjoint(a_ci):
                    separated = False
            key = _comps_key(comps)
            if key in seen:
                continue
            seen.add(key)
            sizes = sorted([len(c) for c in comps], reverse=True)
            mean_s = sum(sizes) / len(sizes)
            variance = sum((s - mean_s) ** 2 for s in sizes) / len(sizes)
            score = 0.0
            if len(comps) == 3: score += 200
            elif len(comps) == 2: score += 140
            elif len(comps) == 4: score += 60
            score -= variance * 2
            score -= k * 5
            if separated: score += 100
            candidates.append((score, comps))
    candidates.sort(key=lambda x: -x[0])
    return [c for _, c in candidates]


def strat_c1_v2_3frag(stage5a, g_R, g_P, floaters):
    """Top scored cut preferring 3 fragments."""
    cands = _enumerate_cuts(stage5a, g_R, g_P, floaters, min_comps=3, max_comps=4)
    return cands[0] if cands else None


def strat_c2_v2_2frag(stage5a, g_R, g_P, floaters):
    """Top scored cut limited to 2 fragments."""
    cands = _enumerate_cuts(stage5a, g_R, g_P, floaters, min_comps=2, max_comps=2)
    return cands[0] if cands else None


def strat_c3_isolated_mig(stage5a, g_R, g_P, floaters):
    """Same as v2_3frag but migrating atoms stay isolated as their own fragment."""
    cands = _enumerate_cuts(stage5a, g_R, g_P, floaters, min_comps=3, max_comps=4,
                              keep_migrators_isolated=True)
    return cands[0] if cands else None


def strat_c4_da_single(stage5a, g_R, g_P, floaters):
    """Best single-bond cut on the donor→acceptor shortest path."""
    n = stage5a["n_atoms"]
    dbg = stage5a["debug"]
    bonds_R = set(tuple(sorted(e)) for e in dbg["bonds_R"])
    bonds_P = set(tuple(sorted(e)) for e in dbg["bonds_P"])
    kept = bonds_R & bonds_P

    donors: set[int] = set()
    acceptors: set[int] = set()
    for a, b in dbg["bonds_broken"]:
        donors.add(a); donors.add(b)
    for a, b in dbg["bonds_formed"]:
        acceptors.add(a); acceptors.add(b)
    migrating = donors & acceptors
    pure_donors = donors - migrating
    pure_acceptors = acceptors - migrating
    if not pure_donors or not pure_acceptors:
        return None

    kept_adj = _adj(kept)
    best: list[set[int]] | None = None
    best_score = -float("inf")
    for d in pure_donors:
        for a in pure_acceptors:
            path = shortest_path(kept_adj, d, a)
            if not path or len(path) < 2:
                continue
            # Try cutting each kept edge along the path
            for i in range(len(path) - 1):
                edge = tuple(sorted((path[i], path[i + 1])))
                if edge not in kept:
                    continue
                remaining = [e for e in kept if e != edge]
                comps = connected_components(n, remaining)
                comps = _fold_floaters(comps, floaters, g_R, g_P)
                if len(comps) < 2:
                    continue
                if any(len(c) == 0 for c in comps):
                    continue
                sizes = sorted([len(c) for c in comps], reverse=True)
                mean_s = sum(sizes) / len(sizes)
                variance = sum((s - mean_s) ** 2 for s in sizes) / len(sizes)
                score = 100 - variance * 2
                if score > best_score:
                    best_score = score
                    best = comps
    return best


STRATEGIES = [
    ("c0_v1_keptbond", strat_c0_keptbond),
    ("c1_v2_3frag",    strat_c1_v2_3frag),
    ("c2_v2_2frag",    strat_c2_v2_2frag),
    ("c3_isolated_mig", strat_c3_isolated_mig),
    ("c4_da_single",   strat_c4_da_single),
]


def main() -> int:
    rids = _failing_rids()
    print(f"Failing rids: {len(rids)}")

    cache = None
    try:
        import pickle
        with open(ROOT / "ADF_500/stage5a/frames_cache.pkl", "rb") as fh:
            cache = pickle.load(fh)
    except Exception as exc:
        print(f"WARN: no frames_cache: {exc}")

    db_idx_map = json.loads(DB_IDX_PATH.read_text())
    if not isinstance(db_idx_map, dict):
        print(f"FAIL: db_idx_map.json is not a dict")
        return 1

    summary: dict = {"rids": {}}
    n_cand_total = 0
    n_unique_total = 0
    for rid in rids:
        src = SRC_STAGE5A / rid / "result.json"
        if not src.exists():
            continue
        stage5a = json.loads(src.read_text())
        symbols = _atom_symbols(rid, cache)
        dbg = stage5a["debug"]
        bonds_R = set(tuple(sorted(e)) for e in dbg["bonds_R"])
        bonds_P = set(tuple(sorted(e)) for e in dbg["bonds_P"])
        kept = bonds_R & bonds_P
        n = stage5a["n_atoms"]
        # Floaters: atoms not touched by any kept bond
        in_kept = set()
        for a, b in kept:
            in_kept.add(a); in_kept.add(b)
        floaters = set(range(n)) - in_kept
        g_R = _adj(bonds_R)
        g_P = _adj(bonds_P)

        seen_keys: set = set()
        # Add the original fragmentation as candidate c_orig (baseline)
        orig_comps = [set(f["atom_indices"]) for f in stage5a["result"]["fragments"]]
        cands: list[tuple[str, list[set[int]]]] = [("c_orig_2frag", orig_comps)]
        seen_keys.add(_comps_key(orig_comps))

        for label, fn in STRATEGIES:
            try:
                comps = fn(stage5a, g_R, g_P, floaters)
            except Exception as e:
                print(f"  {rid} {label}: error {e}")
                continue
            if comps is None or len(comps) < 2:
                continue
            key = _comps_key(comps)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cands.append((label, comps))

        rid_summary = {"n_candidates": len(cands), "candidates": []}
        for ci, (label, comps) in enumerate(cands):
            synth_rid = f"{rid}__c{ci}"
            db_idx_map[synth_rid] = db_idx_map[rid]
            part = _build_partition_dict(comps, symbols, label)
            new_stage5a = dict(stage5a)
            new_stage5a["reaction_id"] = synth_rid
            new_stage5a["result"] = part
            new_stage5a["fragmentation_revision"] = 3
            out_path = OUT_DIR / "per_reaction" / synth_rid / "result.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(new_stage5a, indent=2))
            rid_summary["candidates"].append({
                "synth_rid": synth_rid,
                "label": label,
                "n_fragments": part["n_fragments"],
                "fragment_sizes": sorted([len(f["atom_indices"])
                                          for f in part["fragments"]], reverse=True),
                "multiplicities": [f["multiplicity"] for f in part["fragments"]],
                "coupling": part["coupling"],
            })
            n_cand_total += 1
        n_unique_total += len(cands)
        summary["rids"][rid] = rid_summary

    DB_IDX_PATH.write_text(json.dumps(db_idx_map, indent=2))
    summary_path = ROOT / "Validate" / "refrag" / "candidate_summary.json"
    summary["n_candidates_total"] = n_cand_total
    summary["n_rids"] = len(rids)
    summary_path.write_text(json.dumps(summary, indent=2))

    n_per_rid = sorted([s["n_candidates"] for s in summary["rids"].values()],
                       reverse=True)
    print()
    print(f"Total candidates written:      {n_cand_total}")
    print(f"Candidates per reaction:       min={min(n_per_rid)}  max={max(n_per_rid)}  mean={sum(n_per_rid)/len(n_per_rid):.1f}")
    print(f"db_idx_map expanded to:        {len(db_idx_map)} entries")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
