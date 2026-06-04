"""Aggregate v3 EDA-ASM benchmark results — final summary.

Produces:
    outputs/stage5b/benchmark_500_summary.json   — overview statistics
    outputs/stage5b/adf_summary.csv              — flat tabular dataset for ML

CSV columns (one row per reaction):
    rxn_id, pattern, p2_subtype, n_fragments, mult_tuple,
    eda_available, status, wall_time_min,
    halo8_Ea_eV, ts_frame_idx, n_frames_total,
    # 5-point energetics Δ(from R), eV
    total_R, total_pre_TS, total_TS, total_post_TS, total_P,
    strain_R, strain_pre_TS, strain_TS, strain_post_TS, strain_P,
    int_R, int_pre_TS, int_TS, int_post_TS, int_P,
    Pauli_R, Pauli_pre_TS, Pauli_TS, Pauli_post_TS, Pauli_P,
    elstat_R, elstat_pre_TS, elstat_TS, elstat_post_TS, elstat_P,
    orb_R, orb_pre_TS, orb_TS, orb_post_TS, orb_P,
    disp_R, disp_pre_TS, disp_TS, disp_post_TS, disp_P,
    # Per-fragment strain (frag1, frag2 only — others nan)
    frag1_strain_TS, frag2_strain_TS,
    # Diagnostics
    s2_max,
    asm_closure_TS_eV, eda_sum_TS_eV,
    quality_flags
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
PER_RXN_DIR = REPO / "outputs" / "stage5b" / "per_reaction"
SUMMARY_JSON = REPO / "outputs" / "stage5b" / "benchmark_500_summary.json"
SUMMARY_CSV = REPO / "outputs" / "stage5b" / "adf_summary.csv"

ZETA_LABELS = ["R", "pre_TS", "TS", "post_TS", "P"]
CHANNELS = ["total", "strain", "int", "Pauli", "elstat", "orb", "disp"]


def _safe(v, default=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return v


def main():
    records = []
    stats = {
        "status_count": Counter(),
        "pattern_count": Counter(),
        "eda_available_count": Counter(),
        "n_fragments_count": Counter(),
        "open_shell_count": Counter(),
        "wall_time_min_sum": 0.0,
        "wall_time_min_max": 0.0,
        "wall_time_min_min": float("inf"),
        "asm_closure_TS_distribution": [],
        "eda_sum_TS_distribution": [],
        "flag_counts": Counter(),
    }

    files = sorted(PER_RXN_DIR.glob("*/eda_result.json"))
    print(f"Found {len(files)} eda_result.json files")

    for fp in files:
        try:
            r = json.load(open(fp))
        except Exception as e:
            print(f"  [WARN] cannot parse {fp}: {e}")
            continue

        rxn_id = r["rxn_id"]
        meta = r.get("metadata", {})
        pattern = meta.get("stage5a_pattern", "?")
        status = meta.get("status", "?")
        eda_avail = r.get("eda_available", False)
        wall_min = meta.get("wall_time_seconds", 0) / 60
        n_frag = len(r.get("fragments", []))
        mults = sorted(int(f.get("multiplicity", 1)) for f in r.get("fragments", []))
        mult_tuple = "(" + ",".join(str(m) for m in mults) + ")"
        is_open = any(m > 1 for m in mults)

        # Halo8 reference
        halo = r.get("halo8_reference", {})

        # ζ point energetics
        en = r.get("energetics_delta_from_R_eV", {})
        row = {
            "rxn_id": rxn_id,
            "pattern": pattern,
            "p2_subtype": meta.get("p2_subtype", ""),
            "n_fragments": n_frag,
            "mult_tuple": mult_tuple,
            "open_shell": is_open,
            "eda_available": eda_avail,
            "status": status,
            "wall_time_min": round(wall_min, 2),
            "halo8_Ea_eV": halo.get("Ea_eV"),
            "ts_frame_idx": halo.get("ts_frame_idx"),
        }

        # Flatten 5×channels
        for ch in CHANNELS:
            vals = en.get(ch, [None] * 5)
            for zi, lab in enumerate(ZETA_LABELS):
                row[f"{ch}_{lab}"] = vals[zi] if zi < len(vals) else None

        # Per-fragment strain at TS only (most informative)
        frag_strain = r.get("fragment_strain_eV", {})
        roles = list(frag_strain.keys())
        for i, role in enumerate(roles[:2]):
            row[f"frag{i+1}_role"] = role
            row[f"frag{i+1}_strain_TS"] = (frag_strain[role][2]
                                            if len(frag_strain[role]) > 2 else None)

        # ⟨S²⟩ max across all fragments/ζ
        s2_dict = r.get("spin_diagnostics_s2", {})
        s2_vals = [v for v in s2_dict.values() if v is not None]
        row["s2_max"] = max(s2_vals) if s2_vals else None

        # ASM closure at TS: total - (strain + int)
        t, s, i = en.get("total", [None]*5)[2], en.get("strain", [None]*5)[2], en.get("int", [None]*5)[2]
        if t is not None and s is not None and i is not None:
            row["asm_closure_TS_eV"] = t - (s + i)
        else:
            row["asm_closure_TS_eV"] = None

        # EDA sum check at TS: Pauli + elstat + orb + disp - int
        comps = [en.get(ch, [None]*5)[2] for ch in ["Pauli","elstat","orb","disp","int"]]
        if all(v is not None for v in comps):
            row["eda_sum_TS_eV"] = sum(comps[:4]) - comps[4]
        else:
            row["eda_sum_TS_eV"] = None

        row["quality_flags"] = ";".join(r.get("quality_flags", []))

        records.append(row)

        # Stats
        stats["status_count"][status] += 1
        stats["pattern_count"][pattern] += 1
        stats["eda_available_count"][eda_avail] += 1
        stats["n_fragments_count"][n_frag] += 1
        stats["open_shell_count"][is_open] += 1
        stats["wall_time_min_sum"] += wall_min
        stats["wall_time_min_max"] = max(stats["wall_time_min_max"], wall_min)
        if wall_min > 0:
            stats["wall_time_min_min"] = min(stats["wall_time_min_min"], wall_min)
        if row["asm_closure_TS_eV"] is not None:
            stats["asm_closure_TS_distribution"].append(row["asm_closure_TS_eV"])
        if row["eda_sum_TS_eV"] is not None:
            stats["eda_sum_TS_distribution"].append(row["eda_sum_TS_eV"])
        for fl in r.get("quality_flags", []):
            stats["flag_counts"][fl] += 1

    # Write CSV
    if records:
        # union of all keys (preserve first-seen order)
        seen = []
        seen_set = set()
        for r in records:
            for k in r:
                if k not in seen_set:
                    seen.append(k); seen_set.add(k)
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=seen)
            w.writeheader()
            w.writerows(records)
        print(f"[OK] CSV → {SUMMARY_CSV}  ({len(records)} rows × {len(seen)} cols)")

    # Convert stats for JSON serialisation
    n = len(records)

    def _stats_summary(vals):
        if not vals:
            return None
        vals = sorted(vals)
        return {
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
            "abs_max": max(abs(v) for v in vals),
            "median": vals[len(vals) // 2],
            "mean": sum(vals) / len(vals),
        }

    summary = {
        "total_reactions": n,
        "status_count": dict(stats["status_count"]),
        "pattern_count": dict(stats["pattern_count"]),
        "eda_available_count": {str(k): v for k, v in stats["eda_available_count"].items()},
        "n_fragments_count": {str(k): v for k, v in stats["n_fragments_count"].items()},
        "open_shell_count": {str(k): v for k, v in stats["open_shell_count"].items()},
        "wall_time_min": {
            "sum": stats["wall_time_min_sum"],
            "mean": stats["wall_time_min_sum"] / n if n else 0,
            "min": stats["wall_time_min_min"] if stats["wall_time_min_min"] < float("inf") else None,
            "max": stats["wall_time_min_max"],
        },
        "asm_closure_TS_eV": _stats_summary(stats["asm_closure_TS_distribution"]),
        "eda_sum_TS_eV": _stats_summary(stats["eda_sum_TS_distribution"]),
        "flag_counts": dict(stats["flag_counts"].most_common()),
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] JSON → {SUMMARY_JSON}")

    # Print headline summary
    print()
    print("=" * 60)
    print(f"Total: {n} reactions")
    print(f"Status: {dict(stats['status_count'])}")
    print(f"Pattern: {dict(stats['pattern_count'])}")
    print(f"EDA available: {dict(stats['eda_available_count'])}")
    print(f"Mean wall: {summary['wall_time_min']['mean']:.1f} min")
    if summary["asm_closure_TS_eV"]:
        c = summary["asm_closure_TS_eV"]
        print(f"ASM closure @ TS: n={c['n']}, |max|={c['abs_max']:.2e}, median={c['median']:.2e}")
    print(f"Top flags: {dict(list(stats['flag_counts'].most_common(5)))}")


if __name__ == "__main__":
    main()
