#!/usr/bin/env python3
"""Smart alt fragmentation v2 — donor/acceptor min-cut on the kept-bond graph.

v1 (derive_alt_frag.py) used kept-bond connected components as-is. For most
P5_HSHIFT / P2_CLOSED cyclic systems this matched the original [scaffold,
migrating-atom] cut and produced no improvement.

v2 enumerates small (1–3 bond) cuts of the kept-bond skeleton and picks the
cut that:
  (a) separates 'donor' atoms (broken-bond endpoints minus migrating) from
      'acceptor' atoms (formed-bond endpoints minus migrating) into different
      components, and
  (b) produces a balanced 2- or 3-fragment partition.

Floater atoms (atoms with no kept bonds — typically migrating H/halogen) are
assigned to the component containing their R-side neighbor.
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
OUT_STAGE5A = ROOT / "Validate" / "refrag" / "stage5a" / "per_reaction"
MANIFEST = ROOT / "Validate" / "manifest.csv"

Z_OF: dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def connected_components(n_atoms: int, edges) -> list[set[int]]:
    """Undirected connected components on indices 0..n_atoms-1."""
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


def _failing_rids() -> list[str]:
    """All rids whose verdict==FAIL and Check 3 or Check 4 failed."""
    rids: list[str] = []
    for row in csv.DictReader(open(MANIFEST)):
        if row["verdict"] != "FAIL":
            continue
        fc = set(row["failed_checks"].split(";"))
        if "3" in fc or "4" in fc:
            rids.append(row["reaction_id"])
    return rids


def _atom_symbols(rid: str) -> list[str] | None:
    """Lookup atom symbols from frames_cache.pkl."""
    try:
        import pickle
        with open(ROOT / "ADF_500/stage5a/frames_cache.pkl", "rb") as fh:
            cache = pickle.load(fh)
        entry = cache.get(rid)
        if entry is None:
            return None
        nums = getattr(entry, "numbers", None)
        if nums is None:
            return None
        sym_table = {v: k for k, v in Z_OF.items()}
        return [sym_table.get(int(z), f"Z{int(z)}") for z in nums]
    except Exception:
        return None


def _build_adj(edges) -> dict[int, set[int]]:
    """Build adjacency dict from edge list."""
    g: dict[int, set[int]] = collections.defaultdict(set)
    for a, b in edges:
        g[a].add(b); g[b].add(a)
    return g


def find_smart_partition(stage5a: dict) -> dict | None:
    """Return best fragment partition as a dict, or None if nothing better than original."""
    n_atoms = int(stage5a["n_atoms"])
    dbg = stage5a["debug"]
    bonds_R = set(tuple(sorted(e)) for e in dbg["bonds_R"])
    bonds_P = set(tuple(sorted(e)) for e in dbg["bonds_P"])
    bonds_broken = set(tuple(sorted(e)) for e in dbg["bonds_broken"])
    bonds_formed = set(tuple(sorted(e)) for e in dbg["bonds_formed"])
    kept = bonds_R & bonds_P
    union = bonds_R | bonds_P
    g_R = _build_adj(bonds_R)
    g_P = _build_adj(bonds_P)
    g_union = _build_adj(union)

    donors_all: set[int] = set()
    acceptors_all: set[int] = set()
    for a, b in bonds_broken:
        donors_all.add(a); donors_all.add(b)
    for a, b in bonds_formed:
        acceptors_all.add(a); acceptors_all.add(b)
    migrating = donors_all & acceptors_all
    donors = donors_all - migrating
    acceptors = acceptors_all - migrating

    floaters: set[int] = set()
    for atom in range(n_atoms):
        if not g_R.get(atom) or not any((a, b) in kept or (b, a) in kept
                                          for a, b in (e for e in bonds_R if atom in e)):
            # Atom has no kept bonds — pure migrating / isolated
            has_kept = any(atom in (a, b) for (a, b) in kept)
            if not has_kept:
                floaters.add(atom)

    kept_list = list(kept)
    orig = [set(f["atom_indices"]) for f in stage5a["result"]["fragments"]]
    orig_norm = tuple(sorted(tuple(sorted(s)) for s in orig))

    best = None
    best_score = -float("inf")

    max_cuts = min(3, len(kept_list))
    for k in range(0, max_cuts + 1):
        for cut in itertools.combinations(kept_list, k):
            cut_set = set(cut)
            remaining = [e for e in kept if e not in cut_set]
            comps = connected_components(n_atoms, remaining)
            # Assign floaters: pull them out of their singleton component and
            # attach to the component containing their R-neighbor.
            # First find each floater's component index.
            atom_to_comp = {a: ci for ci, c in enumerate(comps) for a in c}
            for fl in floaters:
                fl_ci = atom_to_comp[fl]
                # Floater is alone in its component (true if it has no kept bond)
                if len(comps[fl_ci]) != 1:
                    continue
                # Find R-neighbor's component
                neigh_R = g_R.get(fl, set())
                target_ci = None
                for nb in neigh_R:
                    nb_ci = atom_to_comp.get(nb)
                    if nb_ci is not None and nb_ci != fl_ci:
                        target_ci = nb_ci
                        break
                if target_ci is None:
                    for nb in g_P.get(fl, set()):
                        nb_ci = atom_to_comp.get(nb)
                        if nb_ci is not None and nb_ci != fl_ci:
                            target_ci = nb_ci
                            break
                if target_ci is None:
                    continue
                comps[target_ci].add(fl)
                comps[fl_ci].discard(fl)
            # Drop now-empty components
            comps = [c for c in comps if c]

            if len(comps) < 2 or len(comps) > 4:
                continue

            # Verify donor/acceptor separation (when both groups exist)
            separated = True
            if donors and acceptors:
                d_comps = {ci for ci, c in enumerate(comps) for d in donors if d in c}
                a_comps = {ci for ci, c in enumerate(comps) for a in acceptors if a in c}
                if not d_comps.isdisjoint(a_comps):
                    separated = False

            # Disallow tiny (1-atom) components unless it's a floater fragment
            # (which would be a meaningful 3-fragment with migrating atom isolated)
            n_tiny = sum(1 for c in comps if len(c) == 1)
            if n_tiny > 1:
                continue

            sizes = sorted((len(c) for c in comps), reverse=True)
            mean_s = sum(sizes) / len(sizes)
            variance = sum((s - mean_s) ** 2 for s in sizes) / len(sizes)

            score = 0.0
            if len(comps) == 3:
                score += 220
            elif len(comps) == 2:
                score += 140
            elif len(comps) == 4:
                score += 60
            score -= variance * 2
            score -= k * 6
            if separated:
                score += 120
            if n_tiny == 0:
                score += 25

            comps_norm = tuple(sorted(tuple(sorted(c)) for c in comps))
            if comps_norm == orig_norm:
                continue  # identical to original — useless

            if score > best_score:
                best_score = score
                best = {
                    "comps": comps,
                    "cut": cut_set,
                    "separated": separated,
                    "score": score,
                }
    return best


def _atom_e_count(atoms: set[int], symbols: list[str] | None) -> int:
    """Sum of atomic numbers (proxy for electrons) over an atom set."""
    if symbols is None:
        return len(atoms)  # fallback parity guess
    return sum(Z_OF.get(symbols[a], 0) for a in atoms)


def build_alt_stage5a(stage5a: dict, partition: dict, symbols: list[str] | None) -> dict:
    """Wrap a partition into the stage5a result.json schema."""
    comps = sorted(partition["comps"], key=lambda c: -len(c))
    fragments: list[dict] = []
    for i, comp in enumerate(comps):
        atoms = sorted(comp)
        ne = _atom_e_count(comp, symbols)
        mult = 1 if (ne % 2 == 0) else 2
        fragments.append({
            "atom_indices": atoms,
            "role": f"comp_{i}",
            "multiplicity": mult,
            "cap_attachment": None,
        })

    open_shell = [f for f in fragments if f["multiplicity"] > 1]
    if len(open_shell) >= 2:
        coupling = "broken_symmetry_singlet"
    elif len(open_shell) == 1:
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

    new_stage5a = dict(stage5a)
    new_stage5a["result"] = {
        "pattern": stage5a["result"]["pattern"] + "_ALT_V2_DA_MINCUT",
        "fragments": fragments,
        "spin_signs": spin_signs,
        "total_spin_polarization": int(total_spin),
        "coupling": coupling,
        "n_fragments": len(fragments),
        "cap_h_positions": None,
        "confidence": 0.85,
        "notes": ("Smart cut: donor/acceptor min-cut on kept-bond skeleton. "
                  f"score={partition['score']:.1f}, separated={partition['separated']}, "
                  f"#cuts={len(partition['cut'])}"),
        "debug": {
            "source": "Validate/refrag/derive_alt_frag_v2.py",
            "cut_edges": sorted(list(partition["cut"])),
        },
    }
    new_stage5a["fragmentation_revision"] = 2
    return new_stage5a


def main() -> int:
    """Target the still-FAIL reactions that have no alt stage5a or a bad one;
    write new alt stage5a using v2 algorithm."""
    rids = _failing_rids()
    # Skip ones that have a currently-in-flight rerun (active run_one.sh)
    active = set()
    try:
        import subprocess
        out = subprocess.check_output(["ps", "-ef"], text=True)
        for line in out.splitlines():
            if "run_one.sh" in line and "grep" not in line:
                toks = line.split()
                if toks and toks[-1].startswith(("Halogen_", "T1x_")):
                    active.add(toks[-1])
    except Exception:
        pass

    # Skip ones that already have a non-FAILED result (auto-fix succeeded)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "Validate"))
    from validate_asr import (
        Config, derive, check1_schema, check3_topology,
        check4_conservation, check5_signs, aggregate,
    )
    cfg = Config()
    fixed_already: set[str] = set()
    for rid in rids:
        rp = ROOT / "Validate" / "refrag" / "results" / f"{rid}.json"
        if not rp.exists():
            continue
        try:
            j = json.loads(rp.read_text())
            d = derive("o", j)
            v = aggregate(check1_schema(d) + check3_topology(d, cfg)
                           + check4_conservation(d, cfg) + check5_signs(d))
            if v != "FAIL":
                fixed_already.add(rid)
        except Exception:
            pass

    skipped_active: list[str] = []
    skipped_fixed: list[str] = []
    skipped_no_alt: list[str] = []
    skipped_same: list[str] = []
    written: list[dict] = []

    for rid in rids:
        if rid in active:
            skipped_active.append(rid); continue
        if rid in fixed_already:
            skipped_fixed.append(rid); continue
        src = SRC_STAGE5A / rid / "result.json"
        if not src.exists():
            skipped_no_alt.append(f"{rid} (no source)"); continue
        stage5a = json.loads(src.read_text())
        partition = find_smart_partition(stage5a)
        if partition is None:
            skipped_same.append(rid); continue

        symbols = _atom_symbols(rid)
        new_stage5a = build_alt_stage5a(stage5a, partition, symbols)
        out_path = OUT_STAGE5A / rid / "result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(new_stage5a, indent=2))
        written.append({
            "rid": rid,
            "pattern_orig": stage5a["result"]["pattern"],
            "orig_n_frag": stage5a["result"]["n_fragments"],
            "alt_n_frag": new_stage5a["result"]["n_fragments"],
            "alt_sizes": sorted([len(f["atom_indices"])
                                 for f in new_stage5a["result"]["fragments"]],
                                 reverse=True),
            "alt_mults": [f["multiplicity"]
                          for f in new_stage5a["result"]["fragments"]],
            "coupling": new_stage5a["result"]["coupling"],
            "cuts": new_stage5a["result"]["debug"]["cut_edges"],
        })

    summary = {
        "total_targets": len(rids),
        "skipped_active_rerun": len(skipped_active),
        "skipped_already_fixed": len(skipped_fixed),
        "skipped_no_source": len(skipped_no_alt),
        "skipped_no_alt_found": len(skipped_same),
        "n_new_alt_written": len(written),
        "active_rids": sorted(active & set(rids)),
        "fixed_already": sorted(fixed_already),
        "rows": written,
    }
    sp = Path(__file__).parent / "alt_frag_v2_summary.json"
    sp.write_text(json.dumps(summary, indent=2))

    print(f"Failing rids:           {len(rids)}")
    print(f"  in-flight reruns:     {len(skipped_active)}  {sorted(active & set(rids))}")
    print(f"  already passing now:  {len(skipped_fixed)}")
    print(f"  no source stage5a:    {len(skipped_no_alt)}")
    print(f"  no smart alt found:   {len(skipped_same)}")
    print(f"  ★ new alt written:    {len(written)}")
    if written:
        print()
        print(f"{'rid':45s}{'pat':12s}{'n_frag':>8}{'sizes':>15}{'mults':>10}{'cuts':>10}")
        for w in written[:30]:
            print(f"{w['rid']:45s}{w['pattern_orig']:12s}"
                  f"{w['alt_n_frag']:>8}{str(w['alt_sizes']):>15}"
                  f"{str(w['alt_mults']):>10}{len(w['cuts']):>10}")
        if len(written) > 30:
            print(f"... and {len(written) - 30} more")
    print()
    print(f"summary: {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
