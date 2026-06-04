"""ASR_Reselect_250_FixAndFill_v1.1 — Phase C + D implementation.

Re-selects a 250-reaction stratified seed using the SAME criteria as the
ADF_800 v1.1 selection (3-tier balanced + delta_Ea quartiles + Morgan
maximin), reusing already-labeled reactions and computing the gap.

This script is Phase C ("select") + Phase D ("new-compute gate") fused.
Phase A (scancel) and Phase B (extract+inventory) are done separately in
shell; this script consumes their outputs.

Outputs into --out-dir:
    selected_250.csv          full final selection (reuse + new), stratum labels
    reuse_set.csv             reused reactions (0 new ADF)
    new_reactions.csv         to_compute = to_finish + to_run_fresh
    stratum_table.csv         (family, quartile) → n_target/n_have/n_deficit/final
    compute_savings_report.txt
    reselect_manifest.json
    quartile_edges.json       per-family quartile bin edges from full pool

Exit codes:
    0  selection produced N_new > epsilon (proceed to Phase E)
    1  preflight error (missing inputs)
    2  data mismatch (reused reactions not in pool)
    3  N_new ≤ epsilon — informational gate, await user confirmation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---- ADF_800 v1.1 selection parameters (HARDCODED — must match ADF_800) ----

FAMILY_TARGETS_800 = {"dipolar": 300, "e2": 250, "sn2": 250}
TOTAL_800 = 800
N_QUARTILES = 4
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
TANIMOTO_MIN = 0.30
RANDOM_SEED = 42
STRATIFY_COLUMN = "delta_Ea"
# Recoverable failure_reasons (from inventory_labels.py)
RECOVERABLE_REASONS = {
    "scf_not_converged", "failed_step",
    "eda_incomplete", "fragment_energy_missing",
    "no_output_file", "no_status_file", "invalid_status",
}


# ---- helpers --------------------------------------------------------------


def _scaled_family_targets(total: int) -> dict[str, int]:
    """Tier 1 — scale ADF_800 family ratios to a new total, adjusting
    rounding so the sum equals `total` exactly."""
    raw = {fam: total * cnt / TOTAL_800 for fam, cnt in FAMILY_TARGETS_800.items()}
    rounded = {fam: int(round(v)) for fam, v in raw.items()}
    diff = total - sum(rounded.values())
    if diff != 0:
        order = sorted(raw, key=lambda f: (raw[f] - rounded[f]), reverse=(diff > 0))
        for i in range(abs(diff)):
            rounded[order[i % len(order)]] += 1 if diff > 0 else -1
    assert sum(rounded.values()) == total, rounded
    return rounded


def _quartile_targets(family_target: int) -> dict[int, int]:
    """Tier 2 — quartile_remainder_policy: front_load."""
    base, rem = divmod(family_target, N_QUARTILES)
    return {q: base + (1 if q < rem else 0) for q in range(N_QUARTILES)}


def _per_family_quartile_edges(pool: pd.DataFrame) -> dict[str, list[float]]:
    """Compute Tier-2 quartile bin edges per family on the FULL pool —
    these edges are then applied to BOTH reused and new selections.
    """
    edges = {}
    for fam, sub in pool.groupby("family"):
        qs = np.quantile(sub[STRATIFY_COLUMN].to_numpy(), [0.0, 0.25, 0.5, 0.75, 1.0])
        edges[fam] = qs.tolist()
    return edges


def _assign_quartile(value: float, edges: list[float]) -> int:
    """Return quartile index 0..3 for a delta_Ea value given bin edges
    [min, q1, q2, q3, max]. Boundary policy: right-inclusive on first 3
    bins, inclusive on top end too."""
    if value <= edges[1]:
        return 0
    if value <= edges[2]:
        return 1
    if value <= edges[3]:
        return 2
    return 3


def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity for bit fingerprints."""
    inter = int(np.bitwise_and(a, b).sum())
    union = int(np.bitwise_or(a, b).sum())
    return inter / union if union else 0.0


def _maximin_select(
    candidates_idx: np.ndarray,
    fingerprints: np.ndarray,
    locked_idx: list[int],
    n_pick: int,
    tanimoto_min: float,
    rng: np.random.Generator,
) -> list[int]:
    """Kennard-Stone maximin selection from `candidates_idx`, given a set
    of already-locked indices that must be respected for distance
    computation. Returns the `n_pick` new indices."""
    if n_pick <= 0:
        return []
    picked: list[int] = []
    cand = list(candidates_idx)
    # If nothing locked, seed with the candidate whose FP is most "average"
    # (equivalently: pick deterministically by index sorted order).
    locked = list(locked_idx)
    if not locked:
        # Seed = first candidate by sorted reaction order (deterministic
        # under fixed pool ordering and seed)
        seed_pos = int(rng.integers(0, len(cand)))
        first = cand.pop(seed_pos)
        picked.append(first)
        locked.append(first)
    while len(picked) < n_pick and cand:
        # for each candidate, distance to nearest locked
        best_idx = None
        best_min_dist = -1.0
        for j, c in enumerate(cand):
            min_dist = 1.0
            ok_pair = True
            for L in locked:
                t = _tanimoto(fingerprints[c], fingerprints[L])
                d = 1.0 - t
                if d < min_dist:
                    min_dist = d
                if (1.0 - t) < tanimoto_min:
                    ok_pair = False
                    break
            if not ok_pair:
                continue
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = j
        if best_idx is None:
            break
        chosen = cand.pop(best_idx)
        picked.append(chosen)
        locked.append(chosen)
    return picked


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---- main -----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adf-root", default="ADF_250/adf_outputs", type=Path)
    ap.add_argument("--pool", default="ADF_250/seed_selection/initial_seed_v1/"
                    "pool_after_conformer_collapse.parquet", type=Path)
    ap.add_argument("--fingerprints", default="ADF_250/seed_selection/initial_seed_v1/"
                    "morgan_fingerprints.npy", type=Path)
    ap.add_argument("--labels", default="output/reselect_250/inventory/asr_labels.parquet",
                    type=Path,
                    help="extracted asr labels (R_done) from Phase B")
    ap.add_argument("--failures", default="output/reselect_250/inventory/failures.csv",
                    type=Path, help="failures.csv from inventory_labels.py")
    ap.add_argument("--target-total", type=int, default=250)
    ap.add_argument("--overfill-policy", choices=["keep", "trim"], default="keep")
    ap.add_argument("--new-compute-epsilon", type=int, default=10)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--out", default="output/reselect_250", type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- preflight
    for p in (args.pool, args.fingerprints, args.labels):
        if not p.is_file():
            print(f"[preflight err] missing input: {p}", file=sys.stderr)
            return 1

    pool = pd.read_parquet(args.pool)
    print(f"[loaded pool] {len(pool)} reactions, families={dict(pool['family'].value_counts())}")
    fps = np.load(args.fingerprints)
    if fps.shape[0] != len(pool):
        print(f"[err] pool/fp shape mismatch: pool={len(pool)} fps={fps.shape}", file=sys.stderr)
        return 2

    labels = pd.read_parquet(args.labels)
    print(f"[loaded labels] {len(labels)} reactions with valid 5-vector "
          f"(families={dict(labels['family'].value_counts())})")

    # Pool index lookup (used by both TRIM and fix-and-fill code paths)
    pool_ids_list = pool["reaction_id"].tolist()
    pool_id_to_idx = {rid: i for i, rid in enumerate(pool_ids_list)}
    pool_lookup = pool.set_index("reaction_id")

    # ---- C.1 stratification grid
    family_targets = _scaled_family_targets(args.target_total)
    quartile_targets_per_fam = {fam: _quartile_targets(n) for fam, n in family_targets.items()}
    edges = _per_family_quartile_edges(pool)
    (args.out / "quartile_edges.json").write_text(json.dumps(edges, indent=2))
    print(f"[grid] family_targets={family_targets}")
    for fam, q in quartile_targets_per_fam.items():
        print(f"  {fam}: quartile_targets={q}")

    # ---- C.2 assign reused reactions to strata using SAME edges
    # Normalize family names: labels uses 'qmrxn20_e2'/'qmrxn20_sn2', pool uses 'e2'/'sn2'.
    fam_map = {"qmrxn20_e2": "e2", "qmrxn20_sn2": "sn2", "dipolar": "dipolar"}
    labels = labels.copy()
    labels["family_norm"] = labels["family"].map(fam_map).fillna(labels["family"])
    # Reused reactions must exist in the pool (by reaction_id):
    pool_ids = set(pool["reaction_id"])
    reuse_known = labels[labels["reaction_id"].isin(pool_ids)].copy()
    reuse_orphan = labels[~labels["reaction_id"].isin(pool_ids)]
    if len(reuse_orphan):
        print(f"[warn] {len(reuse_orphan)} labelled reactions not in pool "
              f"(e.g. {list(reuse_orphan['reaction_id'].head(3))})")
    # Find their delta_Ea from the pool (pool_lookup defined above)
    reuse_known["delta_Ea"] = pool_lookup.loc[reuse_known["reaction_id"], STRATIFY_COLUMN].values
    reuse_known["family"] = pool_lookup.loc[reuse_known["reaction_id"], "family"].values
    reuse_known["quartile"] = [
        _assign_quartile(r.delta_Ea, edges[r.family]) for r in reuse_known.itertuples()
    ]
    n_have = reuse_known.groupby(["family", "quartile"]).size().unstack(fill_value=0)
    print(f"[reuse] {len(reuse_known)} labels assigned to strata (orphans dropped: {len(reuse_orphan)})")

    # ---- C.3 deficit per stratum
    rows = []
    deficit_total = 0
    overfill_total = 0
    for fam in family_targets:
        for q in range(N_QUARTILES):
            target = quartile_targets_per_fam[fam][q]
            have = int(n_have.at[fam, q]) if (fam in n_have.index and q in n_have.columns) else 0
            deficit = max(0, target - have)
            overfill = max(0, have - target)
            deficit_total += deficit
            overfill_total += overfill
            rows.append({"family": fam, "quartile": q,
                         "n_target": target, "n_have": have,
                         "n_deficit": deficit, "n_overfill": overfill})
    strat = pd.DataFrame(rows)
    print(f"[deficit] total_deficit={deficit_total}  total_overfill={overfill_total}")
    print(strat.to_string(index=False))

    # ---- C.5 partition new reactions into to_finish vs to_run_fresh
    # Read failures.csv to find which currently-failed reactions are RECOVERABLE
    partial_ids: set[str] = set()
    if args.failures.is_file():
        fdf = pd.read_csv(args.failures)
        partial_ids = set(
            fdf[fdf["failure_reason"].isin(RECOVERABLE_REASONS)]["reaction_id"]
        )

    # ---- TRIM policy: when overfill_policy='trim', maximin-select exactly
    # n_target from each labeled stratum (drop the overfill).
    trimmed_reuse_ids: set[str] | None = None
    if args.overfill_policy == "trim":
        trimmed_reuse_ids = set()
        rng_trim = np.random.default_rng(args.seed)
        pool["quartile"] = [
            _assign_quartile(r.delta_Ea, edges[r.family]) for r in pool.itertuples()
        ]
        for fam in family_targets:
            for q in range(N_QUARTILES):
                target = quartile_targets_per_fam[fam][q]
                cell_ids = reuse_known[
                    (reuse_known["family"] == fam) & (reuse_known["quartile"] == q)
                ]["reaction_id"].tolist()
                if len(cell_ids) <= target:
                    trimmed_reuse_ids.update(cell_ids)
                    continue
                # maximin among labeled candidates in this cell
                cand_idx = np.array([pool_id_to_idx[rid] for rid in cell_ids])
                picks = _maximin_select(
                    cand_idx, fps, [], target, TANIMOTO_MIN, rng_trim,
                )
                trimmed_reuse_ids.update(pool_ids_list[i] for i in picks)
        # Replace reuse_known with the trimmed subset
        reuse_known = reuse_known[
            reuse_known["reaction_id"].isin(trimmed_reuse_ids)
        ].reset_index(drop=True)
        # Recompute n_have for downstream reports
        n_have = reuse_known.groupby(["family", "quartile"]).size().unstack(fill_value=0)
        # Recompute reuse_idx mapping
        reuse_idx_by_rid = {rid: pool_id_to_idx[rid] for rid in reuse_known["reaction_id"]}
        # Recompute deficit table
        new_rows = []
        for fam in family_targets:
            for q in range(N_QUARTILES):
                target = quartile_targets_per_fam[fam][q]
                have = int(n_have.at[fam, q]) if (fam in n_have.index and q in n_have.columns) else 0
                deficit = max(0, target - have)
                overfill = max(0, have - target)
                new_rows.append({"family": fam, "quartile": q,
                                 "n_target": target, "n_have": have,
                                 "n_deficit": deficit, "n_overfill": overfill})
        strat = pd.DataFrame(new_rows)
        print(f"[trim] kept exactly {len(reuse_known)} reactions across all strata")
        print(strat.to_string(index=False))

    # ---- C.4 fix-and-fill maximin selection
    # (pool_id_to_idx and pool_ids_list defined above; reuse_idx_by_rid may
    # have been re-computed by the TRIM block — recompute to be safe)
    reuse_idx_by_rid = {rid: pool_id_to_idx[rid] for rid in reuse_known["reaction_id"]}
    if "quartile" not in pool.columns:
        pool["quartile"] = [
            _assign_quartile(r.delta_Ea, edges[r.family]) for r in pool.itertuples()
        ]
    rng = np.random.default_rng(args.seed)
    new_picks: list[dict] = []
    for fam in family_targets:
        for q in range(N_QUARTILES):
            row = strat[(strat["family"] == fam) & (strat["quartile"] == q)].iloc[0]
            deficit = int(row["n_deficit"])
            if deficit == 0:
                continue
            cell = pool[(pool["family"] == fam) & (pool["quartile"] == q)]
            already_in_reuse_idx = [
                reuse_idx_by_rid[rid] for rid in reuse_known[
                    (reuse_known["family"] == fam) & (reuse_known["quartile"] == q)
                ]["reaction_id"]
            ]
            available_idx = np.array([
                pool_id_to_idx[rid] for rid in cell["reaction_id"]
                if rid not in reuse_idx_by_rid
            ])
            if len(available_idx) == 0:
                print(f"  [{fam} q{q}] deficit={deficit} but no candidates left")
                continue
            picks = _maximin_select(
                available_idx, fps, already_in_reuse_idx, deficit,
                TANIMOTO_MIN, rng,
            )
            for idx in picks:
                new_picks.append({
                    "reaction_id": pool_ids_list[idx],
                    "family": fam,
                    "quartile": q,
                    "delta_Ea": float(pool_lookup.loc[pool_ids_list[idx], STRATIFY_COLUMN]),
                    "kind": "to_finish" if pool_ids_list[idx] in partial_ids else "to_run_fresh",
                })
            print(f"  [{fam} q{q}] picked {len(picks)} new (deficit was {deficit})")

    # ---- assemble outputs
    reuse_df = reuse_known[["reaction_id", "family", "quartile", "delta_Ea"]].copy()
    reuse_df["source"] = "reuse"
    new_df = pd.DataFrame(new_picks)
    if len(new_df):
        new_df["source"] = new_df["kind"]
    selected = pd.concat([reuse_df.assign(kind="reuse"), new_df], ignore_index=True)
    selected.to_csv(args.out / "selected_250.csv", index=False)
    reuse_df.to_csv(args.out / "reuse_set.csv", index=False)
    if len(new_df):
        new_df.to_csv(args.out / "new_reactions.csv", index=False)
    else:
        (args.out / "new_reactions.csv").write_text("reaction_id,family,quartile,delta_Ea,kind,source\n")
    strat["n_final"] = strat["n_have"] + strat.apply(
        lambda r: int((new_df["family"] == r["family"]).sum() if len(new_df) else 0)
        if r["n_deficit"] > 0 else 0, axis=1,
    )
    strat.to_csv(args.out / "stratum_table.csv", index=False)

    n_reused = len(reuse_df)
    n_new = len(new_df)
    n_finish = int((new_df["kind"] == "to_finish").sum()) if len(new_df) else 0
    n_fresh = int((new_df["kind"] == "to_run_fresh").sum()) if len(new_df) else 0
    final_n = n_reused + n_new
    savings_pct = (1.0 - (n_new / max(args.target_total, 1))) * 100

    report = (
        f"=== Reselect 250 — Fix-and-Fill report ===\n\n"
        f"reused (0 new ADF): {n_reused}\n"
        f"new compute       : {n_new}  (to_finish={n_finish}, to_run_fresh={n_fresh})\n"
        f"final total       : {final_n}  (target was {args.target_total}, "
        f"overfill_policy={args.overfill_policy})\n"
        f"estimated ADF savings vs computing 250 from scratch: ~{savings_pct:.0f}%\n"
    )
    (args.out / "compute_savings_report.txt").write_text(report)
    print("\n" + report)

    # ---- manifest
    manifest = {
        "spec_version": "1.1",
        "target_total": args.target_total,
        "family_targets": family_targets,
        "overfill_policy": args.overfill_policy,
        "criteria_source": "ADF_800 v1.1",
        "quartile_bin_edges": edges,
        "morgan_params": {"radius": MORGAN_RADIUS, "nbits": MORGAN_NBITS,
                          "tanimoto_min": TANIMOTO_MIN},
        "random_seed": args.seed,
        "n_reused": n_reused,
        "n_new_to_finish": n_finish,
        "n_new_fresh": n_fresh,
        "final_total": final_n,
        "new_compute_epsilon": args.new_compute_epsilon,
        "input_hashes": {
            "pool": _sha(args.pool),
            "fingerprints": _sha(args.fingerprints),
            "labels": _sha(args.labels),
        },
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (args.out / "reselect_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- Phase D gate
    if n_new <= args.new_compute_epsilon:
        print(f"\n[Phase D] new_compute = {n_new} ≤ epsilon={args.new_compute_epsilon}")
        print("[Phase D] ADF submission GATED — exit 3.")
        print("[Phase D] reuse-only ⇒ proceed to modeling, or run "
              "scripts/reselect_250.py phase=compute-gap after user approval.")
        return 3
    print(f"\n[Phase D] new_compute = {n_new} > epsilon={args.new_compute_epsilon}")
    print("[Phase D] await user confirmation before running Phase E.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
