"""Visualize a single eda_result.json: 5 EDA channels + Ea profile + sanity.

Usage:
    python scripts/visualize_eda.py --rxn_id <ID>
    # writes outputs/stage5b/per_reaction/<ID>/eda_visualization.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE5B_DIR = REPO / "outputs" / "stage5b" / "per_reaction"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn_id", required=True)
    args = ap.parse_args()
    rid = args.rxn_id

    p = STAGE5B_DIR / rid / "eda_result.json"
    with open(p) as f:
        r = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- (a) R / TS / P energy profile (ADF) ----
    ax = axes[0, 0]
    w = r["whole_eV"]
    states = ["R", "TS", "P"]
    e_adf = [w.get(f"E_{s}") for s in states]
    e_halo = [w.get(f"E_{s}_halo8_reference") for s in states]
    # Shift both to E_R = 0 for visualization
    if all(v is not None for v in e_adf):
        e_adf_shift = [v - e_adf[0] for v in e_adf]
        ax.plot(states, e_adf_shift, "o-", lw=2, ms=10, label="ADF (this run)")
        for s, y in zip(states, e_adf_shift):
            ax.annotate(f"{y:+.3f}", (s, y), textcoords="offset points",
                        xytext=(8, 8), fontsize=10)
    if all(v is not None for v in e_halo):
        e_halo_shift = [v - e_halo[0] for v in e_halo]
        ax.plot(states, e_halo_shift, "s--", lw=1.5, ms=8, alpha=0.6,
                color="gray", label="Halo8 ωB97X-3c (ref)")
    ax.set_xlabel("State")
    ax.set_ylabel("E − E(R)  [eV]")
    ax.set_title("(a) Reaction energy profile")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- (b) EDA channels at R vs TS (absolute values) ----
    ax = axes[0, 1]
    channels = ["Pauli", "elstat", "orb", "disp"]
    eda_R = r["eda_at_R_eV"]
    eda_TS = r["eda_at_TS_eV"]
    x = list(range(len(channels)))
    if eda_R and eda_TS:
        vals_R = [eda_R.get(c, 0.0) for c in channels]
        vals_TS = [eda_TS.get(c, 0.0) for c in channels]
        ax.bar([i - 0.2 for i in x], vals_R, width=0.4, label="@ R-geom", color="#4a90d9")
        ax.bar([i + 0.2 for i in x], vals_TS, width=0.4, label="@ TS-geom", color="#d94a4a")
        for i, (vr, vts) in enumerate(zip(vals_R, vals_TS)):
            ax.text(i - 0.2, vr, f"{vr:+.2f}", ha="center",
                    va="bottom" if vr > 0 else "top", fontsize=8)
            ax.text(i + 0.2, vts, f"{vts:+.2f}", ha="center",
                    va="bottom" if vts > 0 else "top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("Energy  [eV]")
    ax.set_title("(b) EDA components at R and TS")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # ---- (c) ΔX‡ = X(TS) − X(R) — the 5 GPR labels ----
    ax = axes[1, 0]
    lbl = r["labels_for_gpr_eV"]
    label_keys = ["dE_strain_ddag", "dE_Pauli_ddag", "dE_elstat_ddag",
                  "dE_orb_ddag", "dE_disp_ddag"]
    label_names = ["ΔE_strain‡", "ΔE_Pauli‡", "ΔE_elstat‡",
                   "ΔE_orb‡", "ΔE_disp‡"]
    vals = [lbl.get(k) or 0.0 for k in label_keys]
    colors = ["#7eba7e", "#d94a4a", "#4a90d9", "#9a4ad9", "#d99a4a"]
    bars = ax.bar(label_names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:+.3f}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("ΔE  [eV]")
    ax.set_title("(c) GPR labels — Δ(TS−R) per channel")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=12)

    # ---- (d) Sanity / closure summary ----
    ax = axes[1, 1]
    ax.axis("off")
    v = r["validation"]
    sign = v.get("sign_checks", {})

    Ea_adf = w.get("Ea_adf")
    Ea_halo = w.get("Ea_halo8_reference")
    sum_ddag = (lbl.get("dE_strain_ddag", 0) + lbl.get("dE_int_ddag", 0)) \
               if lbl.get("dE_strain_ddag") is not None and lbl.get("dE_int_ddag") is not None else None

    rows = [
        ("rxn_id", r["rxn_id"]),
        ("pattern", f"{r['pattern']}" + (f" / {r['p2_subtype']}" if r.get("p2_subtype") else "")),
        ("level of theory", f"{r['level_of_theory']['functional']}-{r['level_of_theory']['dispersion']}/"
                            f"{r['level_of_theory']['basis']}  ({r['level_of_theory']['relativity']})"),
        ("", ""),
        ("Ea (ADF this run)",            f"{Ea_adf:+.4f} eV" if Ea_adf is not None else "—"),
        ("Ea (Halo8 ωB97X-3c ref)",      f"{Ea_halo:+.4f} eV" if Ea_halo is not None else "—"),
        ("ΔE_strain‡ + ΔE_int‡",         f"{sum_ddag:+.4f} eV" if sum_ddag is not None else "—"),
        ("ASM closure  ((strain+int) − Ea)", f"{v['asm_closure_eV']:+.6f} eV"
                                              if v.get("asm_closure_eV") is not None else "—"),
        ("ASM closure < 0.05 eV?",       "✓" if v.get("asm_closure_within_0p05eV") else "✗"),
        ("EDA sum check at TS",          f"{v.get('eda_sum_check_at_TS_eV'):+.2e} eV"
                                          if v.get("eda_sum_check_at_TS_eV") is not None else "—"),
        ("", ""),
        ("strain > 0?",         "✓" if sign.get("strain_positive") else "✗"),
        ("Pauli destabilizes?", "✓" if sign.get("pauli_destabilizing") else "✗"),
        ("elstat stabilizes?",  "✓" if sign.get("elstat_stabilizing") else "✗"),
        ("orb stabilizes?",     "✓" if sign.get("orb_stabilizing") else "✗"),
        ("", ""),
        ("wall time",           f"{r['metadata']['wall_time_seconds']:.1f} s"),
        ("AMS version",         r["metadata"]["ams_version"]),
        ("status",              r["metadata"]["status"]),
    ]
    y = 1.0
    for label, val in rows:
        if label == "" and val == "":
            y -= 0.025
            continue
        ax.text(0.02, y, label, fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.55, y, str(val), fontsize=10, family="monospace", transform=ax.transAxes)
        y -= 0.052
    ax.set_title("(d) Sanity checks")

    fig.suptitle(f"EDA-ASM result — {r['rxn_id']}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = STAGE5B_DIR / rid / "eda_visualization.png"
    fig.savefig(out_png, dpi=140)
    print(f"[OK] {out_png}")


if __name__ == "__main__":
    main()
