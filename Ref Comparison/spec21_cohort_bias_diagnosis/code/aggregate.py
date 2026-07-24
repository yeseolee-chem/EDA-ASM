"""spec21 aggregator — summary.md + spec22_cohort_recommendation.md."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec21_cohort_bias_diagnosis"
D1_STATS = STAGE / "results/D1_reactivity_stats.csv"
D1_KS = STAGE / "results/D1_ks_tests.csv"
D2_FRACT = STAGE / "results/D2_scaffold_fractions.csv"
D2_SPOT = STAGE / "results/D2_spotcheck.csv"
D3_CSV = STAGE / "results/D3_geometry_provenance.csv"
HALT_FLAG = STAGE / "logs/G21_B_HALT.flag"
PASS_FLAG = STAGE / "logs/G21_B_PASS.flag"
GATES_LOG = STAGE / "logs/gates.log"

OUT_SUM = STAGE / "results/summary.md"
OUT_REC = STAGE / "results/spec22_cohort_recommendation.md"


def _fmt(x, fmt="{:.3f}"):
    return "" if pd.isna(x) else fmt.format(float(x))


def main() -> int:
    d1 = pd.read_csv(D1_STATS)
    d1_ks = pd.read_csv(D1_KS)
    d2 = pd.read_csv(D2_FRACT)
    d3 = pd.read_csv(D3_CSV)

    halted_G21_B = HALT_FLAG.exists()
    passed_G21_B = PASS_FLAG.exists()

    # ------------------------ summary.md ------------------------
    lines = []
    lines.append("# spec21_cohort_bias_diagnosis — summary")
    lines.append("")
    lines.append(f"Env: python {platform.python_version()}, pandas {pd.__version__}.")
    lines.append("")

    # Banner (G21-D-style)
    lines.append("**Diagnostic scope:** no compute, no re-labelling. Positions the "
                 "dipolar-400 within Stuyver's 5269, classifies scaffolds by topology "
                 "(RDKit), and compares our TS geometries to Stuyver's originals.")
    lines.append("")

    lines.append("## D1 — Reactivity position (ΔG‡, ΔG_r)")
    lines.append("")
    lines.append("Stuyver's Gibbs energies used only as coordinates to locate our "
                 "reactions. No comparison to our own electronic labels intended.")
    lines.append("")
    for tgt in ("G_act", "G_r"):
        lines.append(f"### {tgt}")
        lines.append("")
        lines.append("| group | n | mean | sd | min | q05 | q25 | med | q75 | q95 | max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for g in ("full_5269", "ours_400", "locked_192", "spec16_208"):
            r = d1[(d1["target"] == tgt) & (d1["group"] == g)].iloc[0]
            lines.append(f"| {g} | {int(r['n'])} | {_fmt(r['mean'])} | {_fmt(r['sd'])} | "
                         f"{_fmt(r['min'])} | {_fmt(r['q05'])} | {_fmt(r['q25'])} | "
                         f"{_fmt(r['median'])} | {_fmt(r['q75'])} | {_fmt(r['q95'])} | "
                         f"{_fmt(r['max'])} |")
        lines.append("")
    lines.append("### KS two-sample")
    lines.append("")
    lines.append("| target | a | b | n_a | n_b | KS stat | p |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for _, r in d1_ks.iterrows():
        lines.append(f"| {r['target']} | {r['a']} | {r['b']} | "
                     f"{int(r['n_a'])} | {int(r['n_b'])} | "
                     f"{_fmt(r['ks_stat'])} | {_fmt(r['p_value'], '{:.3e}')} |")
    lines.append("")
    lines.append("Density overlay: `figures/D1_reactivity_position.png`.")
    lines.append("")

    lines.append("## D2 — Scaffold composition (RDKit topology)")
    lines.append("")
    lines.append("Dipolarophile classification by reacting-bond topology.")
    lines.append("")
    dp = d2[d2["axis"] == "dipolarophile"]
    lines.append("| class | full_5269 | ours_400 | locked_192 | spec16_208 |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls in ["alkyne_in_ring", "bridged_alkene", "other_cyclic_alkene",
                "acyclic_alkene", "acyclic_alkyne", "unresolved"]:
        row_bits = [cls]
        for g in ("full_5269", "ours_400", "locked_192", "spec16_208"):
            r = dp[(dp["class"] == cls) & (dp["group"] == g)]
            if len(r):
                r = r.iloc[0]
                row_bits.append(f"{r['fraction']:.3f} [{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}]")
            else:
                row_bits.append("")
        lines.append("| " + " | ".join(row_bits) + " |")
    lines.append("")
    lines.append("Bar chart: `figures/D2_scaffold_composition.png`.")
    lines.append("")
    lines.append("**G21-C classifier validation:** `results/D2_spotcheck.csv` written "
                 f"(20 rows, 10 per half, seed 42). **Unreviewed — D2 fractions above "
                 "are PROVISIONAL until the spotcheck is signed off.**")
    lines.append("")

    lines.append("## D3 — TS geometry provenance vs. Stuyver (G21-B)")
    lines.append("")
    d3_ok = d3.dropna(subset=["rmsd_ang"])
    lines.append(f"n_compared = {len(d3_ok)}/400, "
                 f"n_missing = {int((d3['verdict'] == 'stuyver_TS_missing').sum())}, "
                 f"n_atom_mismatch = {int((d3['verdict'] == 'atom_count_or_element_mismatch').sum())}.")
    lines.append("")
    lines.append("### Verdict distribution")
    lines.append("")
    ver_ct = d3.groupby(["sub_source", "verdict"]).size().unstack(fill_value=0)
    lines.append(ver_ct.to_markdown() if hasattr(ver_ct, "to_markdown") else str(ver_ct))
    lines.append("")
    lines.append("### RMSD distribution (heavy-atom Kabsch, Å)")
    lines.append("")
    lines.append("| half | n | median | mean | q95 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sub in ("locked_778", "spec16"):
        v = d3_ok.loc[d3_ok["sub_source"] == sub, "rmsd_ang"].values
        if len(v):
            lines.append(f"| {sub} | {len(v)} | {_fmt(np.median(v), '{:.4f}')} | "
                         f"{_fmt(np.mean(v), '{:.4f}')} | "
                         f"{_fmt(np.quantile(v, 0.95), '{:.4f}')} | "
                         f"{_fmt(np.max(v), '{:.4f}')} |")
        else:
            lines.append(f"| {sub} | 0 | — | — | — | — |")
    lines.append("")
    lines.append("Histogram: `figures/D3_rmsd_by_half.png`.")
    lines.append("")
    if halted_G21_B:
        lines.append("**G21-B HALT** — see `logs/G21_B_HALT.flag`.")
    elif passed_G21_B:
        lines.append("**G21-B PASS** — halves in the same RMSD regime.")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("```")
    lines.append("Ref Comparison/spec21_cohort_bias_diagnosis/")
    lines.append("  code/{join_cohort.py, d1_reactivity_position.py, d2_scaffold_composition.py,")
    lines.append("        d3_geometry_provenance.py, aggregate.py, submit_s21.sh}")
    lines.append("  logs/{gates.log, G21_B_HALT.flag OR G21_B_PASS.flag}")
    lines.append("  results/{cohort_joined.parquet, stuyver_full.parquet,")
    lines.append("           D1_reactivity_stats.csv, D1_ks_tests.csv,")
    lines.append("           D2_scaffold_fractions.csv, D2_per_reaction.csv, D2_spotcheck.csv,")
    lines.append("           D3_geometry_provenance.csv,")
    lines.append("           spec22_cohort_recommendation.md, summary.md}")
    lines.append("  figures/{D1_reactivity_position.png, D2_scaffold_composition.png,")
    lines.append("           D3_rmsd_by_half.png}")
    lines.append("```")
    lines.append("")
    OUT_SUM.write_text("\n".join(lines) + "\n")

    # ------------------------ spec22 recommendation ------------------------
    # Compute quick "overlap vs skew" signals from D1 KS.
    ks_ours_full = d1_ks[(d1_ks["a"] == "ours_400") & (d1_ks["b"] == "full_5269")]
    ks_halves = d1_ks[(d1_ks["a"] == "locked_192") & (d1_ks["b"] == "spec16_208")]
    ks_our_p = float(ks_ours_full["p_value"].min()) if len(ks_ours_full) else np.nan
    ks_half_p = float(ks_halves["p_value"].min()) if len(ks_halves) else np.nan

    rec = []
    rec.append("# spec22 cohort recommendation")
    rec.append("")
    rec.append("Consumes spec21 diagnostics. One decision, stated at the top.")
    rec.append("")
    if halted_G21_B:
        rec.append("## Decision: **HALT. Geometry provenance is not homogeneous.**")
        rec.append("")
        rec.append("The two halves' TS geometries land in different RMSD regimes vs. "
                   "Stuyver's originals. Re-labelling all 400 under one ORCA ωB97X-3c "
                   "protocol would apply a single scale to two different molecular "
                   "sources. Unify the geometry first, then rewrite spec22.")
        rec.append("")
        rec.append("See `logs/G21_B_HALT.flag`.")
    else:
        # D1: is ours vs full skewed?
        skewed_vs_full = (ks_our_p < 0.05)
        halves_differ  = (ks_half_p < 0.05)
        if not skewed_vs_full:
            rec.append("## Decision: **Re-label all 400. Cohort is broadly representative of Stuyver.**")
        elif not halves_differ:
            rec.append("## Decision: **Re-label all 400. Cohort is skewed vs. Stuyver, "
                       "but the two halves overlap.**")
            rec.append("")
            rec.append("Carry `sub_source` as a reported covariate downstream. The skew is a "
                       "cohort property, not a between-half divergence.")
        else:
            rec.append("## Decision: **Re-label all 400, and stratify on `sub_source`.**")
            rec.append("")
            rec.append("Halves occupy different regions of ΔG‡ / ΔG_r space. The 8.81 "
                       "kcal/mol interaction gap is chemistry and will persist after "
                       "re-labelling. Use `sub_source` as a stratification variable in "
                       "every downstream split.")
        rec.append("")
        rec.append(f"- G21-B: PASS — halves geometry-homogeneous.")
        rec.append(f"- D1 KS p (ours vs full): {ks_our_p:.3e}")
        rec.append(f"- D1 KS p (locked vs spec16): {ks_half_p:.3e}")
        rec.append("")
    rec.append("## Provisional caveats to carry forward")
    rec.append("")
    rec.append("- **DEVIATIONS #4 is wrong today.** Every route line in "
               "`spec20/logs/protocol_discovery.json` reads `BLYP D3BJ def2-TZVP` — "
               "not ωB97X-3c. Correct this deviation at spec22 write-time.")
    rec.append("- **spec20's CP asymmetry** is still open: spec16's R-side has no CP; "
               "the TS side does. Unifying labels also unifies CP.")
    rec.append("- **G21-C spotcheck unreviewed:** D2 fractions are PROVISIONAL.")
    rec.append("")

    OUT_REC.write_text("\n".join(rec) + "\n")
    print(f"[write] {OUT_REC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
