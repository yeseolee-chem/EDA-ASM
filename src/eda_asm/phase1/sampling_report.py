"""Sampling-report HTML builder used after Stage 3.3 (and refreshed at end of Phase 1)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import SAMPLING_REPORT_HTML, ensure_dirs
from .stage_3_3_sampling import (
    BOND_CHANGE_RATIOS,
    EA_TERTILE_RATIOS,
    HEAVY_BINS,
    SOURCE_TARGETS,
)


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img_tag(b64: str, alt: str) -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width: 720px; margin: 8px 0;">'


def _section(title: str, body: str) -> str:
    return f'<section><h2>{title}</h2>{body}</section>'


def _expected_marginals(total: int) -> dict[str, dict[str, float]]:
    src = {k: v / total for k, v in SOURCE_TARGETS.items()}
    heavy = {h: 1 / len(HEAVY_BINS) for h in HEAVY_BINS}
    bond = {"2-3": BOND_CHANGE_RATIOS[0], "4-6": BOND_CHANGE_RATIOS[1]}
    ea = {"low": EA_TERTILE_RATIOS[0], "mid": EA_TERTILE_RATIOS[1], "high": EA_TERTILE_RATIOS[2]}
    return {"source": src, "heavy": heavy, "bond_bin": bond, "ea_tertile": ea}


def build(
    selected: pd.DataFrame,
    population: pd.DataFrame,
    quotas: dict[str, int],
    cell_log: dict[str, dict],
    output_html: Path | None = None,
    *,
    extra_sections: list[tuple[str, str]] | None = None,
) -> Path:
    ensure_dirs()
    if output_html is None:
        output_html = SAMPLING_REPORT_HTML

    figs_html: list[str] = []
    expected = _expected_marginals(len(selected))

    # 1) Source distribution (sample vs expected)
    fig, ax = plt.subplots(figsize=(7, 4))
    src_counts = selected["source"].value_counts().reindex(list(SOURCE_TARGETS), fill_value=0)
    x = np.arange(len(src_counts))
    ax.bar(x - 0.2, src_counts.values, width=0.4, label="sample")
    ax.bar(x + 0.2, [expected["source"][s] * len(selected) for s in src_counts.index], width=0.4, label="target")
    ax.set_xticks(x)
    ax.set_xticklabels(src_counts.index, rotation=20)
    ax.set_ylabel("count")
    ax.set_title("Source distribution (sample vs target)")
    ax.legend()
    figs_html.append(_img_tag(_png_b64(fig), "source"))

    # 2) Heavy atoms
    fig, ax = plt.subplots(figsize=(7, 4))
    heavy_counts = selected["n_heavy_atoms"].value_counts().reindex(HEAVY_BINS, fill_value=0)
    ax.bar(heavy_counts.index, heavy_counts.values, label="sample")
    ax.bar(
        heavy_counts.index,
        [expected["heavy"][h] * len(selected) for h in heavy_counts.index],
        alpha=0.4,
        label="target",
    )
    ax.set_xlabel("heavy atoms")
    ax.set_ylabel("count")
    ax.set_title("Heavy atom distribution")
    ax.legend()
    figs_html.append(_img_tag(_png_b64(fig), "heavy"))

    # 3) Bond change bin
    fig, ax = plt.subplots(figsize=(6, 4))
    bond_counts = selected["bond_change_bin"].value_counts().reindex(["2-3", "4-6"], fill_value=0)
    ax.bar(bond_counts.index, bond_counts.values, label="sample")
    ax.bar(
        bond_counts.index,
        [expected["bond_bin"][b] * len(selected) for b in bond_counts.index],
        alpha=0.4,
        label="target",
    )
    ax.set_title("Bond change bin")
    ax.set_ylabel("count")
    ax.legend()
    figs_html.append(_img_tag(_png_b64(fig), "bond_bin"))

    # 4) EA tertile
    fig, ax = plt.subplots(figsize=(6, 4))
    ea_counts = selected["ea_tertile"].value_counts().reindex(["low", "mid", "high"], fill_value=0)
    ax.bar(ea_counts.index, ea_counts.values, label="sample")
    ax.bar(
        ea_counts.index,
        [expected["ea_tertile"][b] * len(selected) for b in ea_counts.index],
        alpha=0.4,
        label="target",
    )
    ax.set_title("Activation-energy tertile")
    ax.set_ylabel("count")
    ax.legend()
    figs_html.append(_img_tag(_png_b64(fig), "ea_tertile"))

    # 5) Source × heavy heatmap
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = (
        selected.pivot_table(index="source", columns="n_heavy_atoms", values="reaction_id", aggfunc="count", fill_value=0)
        .reindex(index=list(SOURCE_TARGETS), columns=HEAVY_BINS, fill_value=0)
    )
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("heavy atoms")
    ax.set_ylabel("source")
    ax.set_title("Source × heavy atoms (sample counts)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, pivot.values[i, j], ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax)
    figs_html.append(_img_tag(_png_b64(fig), "src_heavy"))

    # 6) Activation energy histogram (population overlay)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(population["activation_energy"], bins=60, alpha=0.4, density=True, label=f"population n={len(population)}")
    ax.hist(selected["activation_energy"], bins=60, alpha=0.7, density=True, label=f"sample n={len(selected)}")
    ax.set_xlabel("activation energy (eV)")
    ax.set_ylabel("density")
    ax.set_title("Activation energy: sample vs population")
    ax.legend()
    figs_html.append(_img_tag(_png_b64(fig), "ea_hist"))

    # Sample table
    head_cols = [
        "reaction_id",
        "source",
        "n_heavy_atoms",
        "n_bond_changes",
        "activation_energy",
        "n_snapshots",
        "ts_frame_idx",
        "cell_label",
    ]
    table_html = selected[head_cols].head(20).to_html(index=False, float_format="%.4f")

    # Cell log table (only cells with non-zero quota for brevity)
    cells_with_data = [(k, v) for k, v in cell_log.items() if v.get("quota", 0) > 0 or v.get("borrowed_for")]
    cells_with_data.sort(key=lambda kv: kv[0])
    cells_rows = ""
    for label, info in cells_with_data:
        cells_rows += (
            f"<tr><td>{label}</td>"
            f"<td>{info.get('quota', 0)}</td>"
            f"<td>{info.get('available', 0)}</td>"
            f"<td>{info.get('taken_pass1', 0)}</td>"
            f"<td>{info.get('filled_from_neighbors', 0)}</td>"
            f"<td>{info.get('unfilled', 0)}</td></tr>"
        )
    cell_table = (
        "<table><thead><tr><th>cell</th><th>quota</th><th>available</th>"
        "<th>pass1</th><th>filled by neighbors</th><th>unfilled</th></tr></thead>"
        f"<tbody>{cells_rows}</tbody></table>"
    )

    parts = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8"><title>Phase 1 Sampling Report</title>',
        "<style>body{font-family:sans-serif;max-width:980px;margin:24px auto;padding:0 16px;}"
        "table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:4px 8px;font-size:12px;}"
        "section{margin:24px 0;}h1{font-size:20px;}h2{font-size:16px;border-bottom:1px solid #ddd;}</style>",
        "</head><body>",
        "<h1>Phase 1 — Sampling report</h1>",
        f"<p>Selected <b>{len(selected)}</b> reactions from a candidate population of <b>{len(population)}</b>.</p>",
        _section("1. Marginal distributions", "".join(figs_html[:4])),
        _section("2. Joint distribution (Source × Heavy atoms)", figs_html[4]),
        _section("3. Activation-energy comparison", figs_html[5]),
        _section("4. First 20 selected rows", table_html),
        _section("5. Per-cell sampling log", cell_table),
    ]
    if extra_sections:
        for title, body in extra_sections:
            parts.append(_section(title, body))
    parts.append("</body></html>")
    output_html.write_text("\n".join(parts))
    return output_html
