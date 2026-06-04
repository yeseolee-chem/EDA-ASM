#!/usr/bin/env python3
"""Generate per-reaction documentation with R/TS/P structure figures.

Outputs two HTML reports + a markdown index under ADF_500_edited/docs/:
  index.html         — summary + links to both reports
  pass_43.html       — 43 reactions whose final canonical winner now passes
                       Check-4 conservation (verdict WARN, not FAIL)
  fail_19.html       — 19 reactions still FAILing, grouped by failure category
  README.md          — plaintext index for terminal users

Each reaction section contains:
  - reaction_id, family, pattern, n_atoms, dE_act, residual
  - Final selected fragmentation (which strategy won, role / atom_indices /
    multiplicity / coupling)
  - Three inline SVG views of R, TS, P with atoms colored by element, fragment
    membership shown as a colored ring around each atom, and bond status
    (kept / broken in R / formed in P) color-coded
"""

from __future__ import annotations

import csv
import html
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
OUT = ROOT / "ADF_500_edited" / "docs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "Validate"))
from validate_asr import (  # type: ignore
    Config, derive, check1_schema, check3_topology,
    check4_conservation, check5_signs, aggregate,
)
cfg = Config()

ATOMIC_NUMBER = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
    "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
SYM_OF = {v: k for k, v in ATOMIC_NUMBER.items()}

# CPK-ish element colors + covalent radii (Å scaled later).
ELEM = {
    "H":  ("#e8e8e8", 0.31),
    "B":  ("#ffb5b5", 0.84),
    "C":  ("#3a3a3a", 0.76),
    "N":  ("#3050f8", 0.71),
    "O":  ("#ff2222", 0.66),
    "F":  ("#9bd35e", 0.57),
    "P":  ("#ff8000", 1.07),
    "S":  ("#ffdc00", 1.05),
    "Cl": ("#1ff01f", 1.02),
    "Br": ("#a62929", 1.20),
    "I":  ("#940094", 1.39),
}
FRAG_COLORS = ["#4ea1ff", "#f0b94c", "#a37bff", "#5ed27a", "#ef5b5b", "#7feaff"]


def project_xy(positions: np.ndarray) -> np.ndarray:
    """PCA project (n_atoms, 3) to (n_atoms, 2) on principal plane."""
    p = positions - positions.mean(axis=0)
    cov = p.T @ p / max(1, len(p))
    w, v = np.linalg.eigh(cov)
    # take two largest eigenvectors
    order = np.argsort(w)[::-1]
    pc = v[:, order[:2]]
    return p @ pc


def render_svg(symbols, positions_3d, bonds_R, bonds_P, fragments,
                width: int = 320, height: int = 260, title: str = "") -> str:
    """Build a self-contained SVG showing the molecule at one geometry.

    fragments: list of dicts each with 'atom_indices' + 'role' + 'multiplicity'.
    bonds_R, bonds_P: lists of [a, b] pairs.
    """
    if positions_3d is None or len(positions_3d) == 0:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20" font-size="12" fill="#888">no coords</text></svg>'
    pos = np.asarray(positions_3d, dtype=float)
    xy = project_xy(pos)
    pad = 28
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    sx = (width - 2 * pad) / max(xmax - xmin, 1e-9)
    sy = (height - 2 * pad) / max(ymax - ymin, 1e-9)
    s = min(sx, sy)
    cx0 = (width - (xmax - xmin) * s) / 2
    cy0 = (height - (ymax - ymin) * s) / 2
    px = [(x - xmin) * s + cx0 for x, _ in xy]
    py = [(y - ymin) * s + cy0 for _, y in xy]

    set_R = {tuple(sorted(e)) for e in bonds_R}
    set_P = {tuple(sorted(e)) for e in bonds_P}
    kept = set_R & set_P
    broken = set_R - set_P
    formed = set_P - set_R

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="background:#0a0c12; border-radius:6px">'
    ]
    if title:
        out.append(f'<text x="8" y="14" font-size="11" font-family="monospace" fill="#8a93a6">{html.escape(title)}</text>')

    # Bonds first
    def draw_bond(edge, color, dash=False):
        a, b = edge
        if a >= len(px) or b >= len(px):
            return
        dasharray = ' stroke-dasharray="5,3"' if dash else ''
        out.append(f'<line x1="{px[a]:.1f}" y1="{py[a]:.1f}" x2="{px[b]:.1f}" y2="{py[b]:.1f}" '
                   f'stroke="{color}" stroke-width="2.5"{dasharray}/>')
    for e in kept:
        draw_bond(e, "#9ba3b8")
    for e in broken:
        draw_bond(e, "#ef5b5b", dash=True)
    for e in formed:
        draw_bond(e, "#5ed27a", dash=True)

    # Atom → fragment idx
    atom_to_frag: dict[int, int] = {}
    for fi, frag in enumerate(fragments):
        for ai in frag["atom_indices"]:
            atom_to_frag[ai] = fi

    # Atoms
    for i, sym in enumerate(symbols):
        color, _ = ELEM.get(sym, ("#888", 0.7))
        r = 9
        # Fragment ring
        fi = atom_to_frag.get(i, -1)
        if fi >= 0:
            ring_color = FRAG_COLORS[fi % len(FRAG_COLORS)]
            out.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="{r+3}" '
                        f'fill="none" stroke="{ring_color}" stroke-width="2.5"/>')
        out.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="{r}" '
                    f'fill="{color}" stroke="#000" stroke-width="0.8"/>')
        # Atom label
        is_dark = sym in ("C", "N", "Br", "I", "P")
        text_color = "#fff" if is_dark else "#111"
        out.append(f'<text x="{px[i]:.1f}" y="{py[i]+3.5:.1f}" font-size="10" '
                    f'font-family="monospace" fill="{text_color}" text-anchor="middle">{sym}{i}</text>')

    out.append('</svg>')
    return "\n".join(out)


def render_legend_svg(width: int = 320) -> str:
    """Tiny legend SVG explaining bond colors + element colors."""
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="56" '
         f'style="background:#0a0c12; border-radius:6px">']
    items = [
        ("#9ba3b8", "kept bond"),
        ("#ef5b5b", "broken (R only) – dashed"),
        ("#5ed27a", "formed (P only) – dashed"),
    ]
    x = 14
    for color, label in items:
        s.append(f'<line x1="{x}" y1="14" x2="{x+22}" y2="14" stroke="{color}" stroke-width="2.5"/>')
        s.append(f'<text x="{x+28}" y="18" font-size="11" font-family="monospace" fill="#d6dae6">{label}</text>')
        x += 145
    # Fragment ring legend
    s.append(f'<text x="14" y="44" font-size="11" font-family="monospace" fill="#d6dae6">'
              f'Fragment ring colors: F0 F1 F2 …</text>')
    cx = 195
    for col in FRAG_COLORS[:5]:
        s.append(f'<circle cx="{cx}" cy="40" r="6" fill="none" stroke="{col}" stroke-width="2.5"/>')
        cx += 16
    s.append('</svg>')
    return "\n".join(s)


def _positions_for(cache, rid: str, label: str) -> np.ndarray | None:
    """Look up R/TS/P 3D coords from the frames cache."""
    e = cache.get(rid)
    if e is None:
        return None
    arr = {"R": e.positions_R, "TS": e.positions_TS, "P": e.positions_P}.get(label)
    if arr is None:
        return None
    return np.asarray(arr).reshape(-1, 3)


def _symbols_for(cache, rid: str) -> list[str]:
    e = cache.get(rid)
    if e is None:
        return []
    return [SYM_OF.get(int(z), "?") for z in e.numbers]


def _derived(raw: dict):
    """Run the validator's derive + verdict on a result JSON."""
    d = derive("o", raw)
    issues = (check1_schema(d) + check3_topology(d, cfg)
              + check4_conservation(d, cfg) + check5_signs(d))
    return d, aggregate(issues), issues


def _build_card(rid: str, raw_new: dict, raw_old: dict,
                 stage5a_new: dict, cache, fail_detail: str = "") -> str:
    """Build the HTML for one reaction card."""
    new_d, new_verdict, new_issues = _derived(raw_new)
    old_d, _, _ = _derived(raw_old)
    family = "T1x" if rid.startswith("T1x") else "Halogen"
    pattern = stage5a_new["result"].get("pattern", "?")
    fragments = stage5a_new["result"]["fragments"]
    coupling = stage5a_new["result"].get("coupling", "?")
    selected = raw_new.get("selected_candidate", stage5a_new.get("selected_candidate", "?"))

    bonds_R = stage5a_new["debug"]["bonds_R"]
    bonds_P = stage5a_new["debug"]["bonds_P"]
    symbols = _symbols_for(cache, rid)

    svgs = []
    for lbl in ("R", "TS", "P"):
        pos = _positions_for(cache, rid, lbl)
        svgs.append(render_svg(symbols, pos, bonds_R, bonds_P, fragments,
                                 title=f"{lbl} geometry"))
    svg_block = '<div class="three-up">' + ''.join(f'<div>{s}</div>' for s in svgs) + '</div>'

    rows_frag = ""
    for fi, f in enumerate(fragments):
        col = FRAG_COLORS[fi % len(FRAG_COLORS)]
        atoms_str = ",".join(str(a) for a in f["atom_indices"])
        ne = sum(ATOMIC_NUMBER.get(symbols[a], 0) for a in f["atom_indices"] if a < len(symbols))
        rows_frag += (
            f'<tr>'
            f'<td><span class="dot" style="border:2.5px solid {col}"></span> F{fi}</td>'
            f'<td>{html.escape(str(f.get("role","?")))}</td>'
            f'<td>{len(f["atom_indices"])}</td>'
            f'<td>{ne}</td>'
            f'<td>{f.get("multiplicity","?")}</td>'
            f'<td><code style="font-size:11px">[{atoms_str}]</code></td>'
            f'</tr>'
        )

    # Bond change summary
    set_R = {tuple(sorted(e)) for e in bonds_R}
    set_P = {tuple(sorted(e)) for e in bonds_P}
    broken_list = ", ".join(f"{a}-{b}" for a, b in sorted(set_R - set_P))
    formed_list = ", ".join(f"{a}-{b}" for a, b in sorted(set_P - set_R))

    badge_class = "ok" if new_verdict != "FAIL" else "bad"
    fail_badge = f'<span class="cat-badge">{html.escape(fail_detail)}</span>' if fail_detail else ""

    return f"""
<section class="card" id="{html.escape(rid)}">
  <h3>
    <a class="anchor" href="#{html.escape(rid)}">¶</a>
    {html.escape(rid)}
    <span class="badge {family.lower()}">{family}</span>
    <span class="badge pattern">{html.escape(pattern)}</span>
    <span class="badge verdict-{badge_class}">{new_verdict}</span>
    {fail_badge}
  </h3>
  <table class="meta">
    <tr><th>n_atoms</th><td>{stage5a_new.get('n_atoms','?')}</td>
        <th>ΔE‡ (kcal/mol)</th><td>{new_d.dE_act:.3f}</td>
        <th>ΔE_rxn (kcal/mol)</th><td>{new_d.dE_rxn:.3f}</td></tr>
    <tr><th>res_cons (old)</th><td>{old_d.max_abs_res_cons:.3f}</td>
        <th>res_cons (final)</th><td class="{'pos' if new_d.max_abs_res_cons<=0.5 else 'neg'}">{new_d.max_abs_res_cons:.4f}</td>
        <th>selected strategy</th><td><code>{html.escape(selected)}</code></td></tr>
    <tr><th>coupling</th><td colspan="5"><code>{html.escape(coupling)}</code></td></tr>
  </table>
  <h4>Selected fragmentation</h4>
  <table class="frag">
    <tr><th>frag</th><th>role</th><th>n_atoms</th><th>Σ Z</th><th>multiplicity</th><th>atom_indices</th></tr>
    {rows_frag}
  </table>
  <h4>Bond changes</h4>
  <div class="bondbox">
    <span class="bondlbl">broken (R-only):</span> <code>{html.escape(broken_list or '—')}</code><br>
    <span class="bondlbl">formed (P-only):</span> <code>{html.escape(formed_list or '—')}</code>
  </div>
  <h4>R / TS / P geometries</h4>
  {svg_block}
</section>
"""


HTML_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #0f1115; color: #d6dae6; margin: 0; }}
  header {{ position: sticky; top: 0; z-index: 10; background: #161922;
            border-bottom: 1px solid #2b3142; padding: 10px 20px; }}
  header h1 {{ margin: 0; font-size: 16px; }}
  header nav {{ font-size: 12px; color: #8a93a6; margin-top: 4px; }}
  header nav a {{ color: #4ea1ff; text-decoration: none; margin-right: 12px; }}
  main {{ padding: 18px 28px; max-width: 1100px; margin: 0 auto; }}
  .card {{ background: #161922; border: 1px solid #2b3142; border-radius: 8px;
           padding: 14px 18px; margin: 0 0 18px 0; }}
  .card h3 {{ margin: 0 0 10px 0; font-family: monospace; font-size: 14px;
              font-weight: 600; }}
  .card h4 {{ margin: 14px 0 6px 0; font-size: 11px; text-transform: uppercase;
              letter-spacing: 0.5px; color: #8a93a6; font-weight: 600; }}
  .anchor {{ text-decoration: none; color: #4ea1ff; margin-right: 6px; }}
  .badge {{ display: inline-block; font-size: 10px; padding: 2px 7px;
            border-radius: 3px; margin-left: 6px; text-transform: uppercase;
            letter-spacing: 0.3px; vertical-align: middle; }}
  .badge.t1x {{ background: rgba(43,205,176,0.18); color: #2bcdb0; }}
  .badge.halogen {{ background: rgba(163,123,255,0.18); color: #a37bff; }}
  .badge.pattern {{ background: rgba(78,161,255,0.15); color: #4ea1ff; }}
  .badge.verdict-ok {{ background: rgba(94,210,122,0.20); color: #5ed27a; }}
  .badge.verdict-bad {{ background: rgba(239,91,91,0.20); color: #ef5b5b; }}
  .cat-badge {{ display: inline-block; font-size: 10px; padding: 2px 7px;
                border-radius: 3px; margin-left: 6px; background: rgba(239,91,91,0.10);
                color: #ef5b5b; font-family: monospace; }}
  table {{ border-collapse: collapse; margin-top: 4px; font-size: 12px; }}
  table.meta th {{ font-weight: 600; color: #8a93a6; text-align: left;
                   padding: 3px 10px 3px 0; font-family: monospace; }}
  table.meta td {{ padding: 3px 18px 3px 0; font-family: monospace; }}
  table.frag {{ width: 100%; font-size: 11px; font-family: monospace; }}
  table.frag th {{ text-align: left; color: #8a93a6; font-weight: 600;
                   padding: 4px 8px; border-bottom: 1px solid #2b3142; }}
  table.frag td {{ padding: 4px 8px; border-bottom: 1px solid #2b3142; }}
  .dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%;
          margin-right: 4px; vertical-align: middle; }}
  .three-up {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
              margin-top: 6px; }}
  .three-up svg {{ display: block; width: 100%; height: auto; }}
  .bondbox {{ font-family: monospace; font-size: 11px; color: #d6dae6;
              background: #0a0c12; padding: 8px 12px; border-radius: 5px; }}
  .bondbox .bondlbl {{ color: #8a93a6; }}
  code {{ color: #c0e0ff; }}
  .pos {{ color: #5ed27a; }}
  .neg {{ color: #ef5b5b; }}
  .summary {{ background: #161922; border: 1px solid #2b3142; border-radius: 8px;
              padding: 14px 18px; margin-bottom: 18px; }}
  .legend-box {{ margin-top: 6px; }}
  .toc {{ font-size: 11px; columns: 3; margin-top: 8px; }}
  .toc a {{ color: #c0e0ff; text-decoration: none; font-family: monospace; }}
  .toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <nav>
    <a href="index.html">← back to index</a>
    <a href="pass_43.html">43 PASS</a>
    <a href="fail_19.html">19 FAIL</a>
  </nav>
</header>
<main>
"""


def main() -> int:
    with open(ROOT / "ADF_500/stage5a/frames_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)
    diag = json.loads((ROOT / "Validate/refrag/still_fail_diagnosis.json").read_text())
    fail_cat_by_rid = {r["rid"]: r["cat"] for r in diag["rows"]}
    fail_detail_by_rid = {r["rid"]: r["detail"] for r in diag["rows"]}

    # Read all original FAIL rids (we work on those)
    orig_fails: list[str] = []
    for row in csv.DictReader(open(ROOT / "Validate/manifest.csv")):
        if row["verdict"] != "FAIL":
            continue
        fc = set(row["failed_checks"].split(";"))
        if "3" in fc or "4" in fc:
            orig_fails.append(row["reaction_id"])

    pass_cards: list[tuple[str, str]] = []
    fail_cards: list[tuple[str, str, str]] = []
    n_skipped = 0

    for rid in orig_fails:
        new_path = ROOT / f"ADF_500_edited/results/{rid}.json"
        s5_path = ROOT / f"ADF_500_edited/stage5a/per_reaction/{rid}/result.json"
        old_path = ROOT / f"ADF_500/results/{rid}.json"
        if not new_path.exists() or not s5_path.exists() or not old_path.exists():
            n_skipped += 1
            continue
        raw_new = json.loads(new_path.read_text())
        raw_old = json.loads(old_path.read_text())
        s5_new = json.loads(s5_path.read_text())
        _, verdict, _ = _derived(raw_new)
        cat = fail_cat_by_rid.get(rid, "")
        if verdict == "FAIL":
            card = _build_card(rid, raw_new, raw_old, s5_new, cache, fail_detail=cat)
            fail_cards.append((rid, card, cat))
        else:
            card = _build_card(rid, raw_new, raw_old, s5_new, cache)
            pass_cards.append((rid, card))

    # Sort
    pass_cards.sort(key=lambda x: x[0])
    fail_cards.sort(key=lambda x: (x[2], x[0]))

    # --- PASS report ---
    pass_html = HTML_HEAD.format(title=f"ADF-EDA: {len(pass_cards)} reactions — Check-4 PASS (WARN)")
    pass_html += f"""
<section class="summary">
  <p>The {len(pass_cards)} reactions below satisfy Check-4 conservation
     (<code>max_abs_res_cons ≤ 0.5 kcal/mol</code>) after the multi-sweep
     pipeline finished. Their verdict is <code>WARN</code> only because of the
     metadata-only <code>recovered</code> flag from Check-1.</p>
  <p>Selected strategies per reaction can be inspected in each card; the
     "selected_candidate" field tells which sweep won (e.g.
     <code>s3_high_spin</code>, <code>c2_v2_2frag</code>, etc.).</p>
  <div class="legend-box">{render_legend_svg(380)}</div>
  <h4>Table of contents</h4>
  <div class="toc">
    {''.join(f'<a href="#{rid}">{rid}</a><br>' for rid, _ in pass_cards)}
  </div>
</section>
"""
    pass_html += "\n".join(card for _, card in pass_cards)
    pass_html += "</main></body></html>"
    (OUT / "pass_43.html").write_text(pass_html)

    # --- FAIL report ---
    fail_html = HTML_HEAD.format(title=f"ADF-EDA: {len(fail_cards)} reactions — still FAIL")
    fail_html += f"""
<section class="summary">
  <p>The {len(fail_cards)} reactions below still fail at least one
     non-recovered check after the full multi-sweep pipeline
     (fragmentation, spin variants, targeted retries, paper-inspired
     homolytic cuts). They are sorted by failure category, then by
     reaction_id.</p>
  <p>Detailed root-cause analysis is in
     <a href="../REMAINING_19_ANALYSIS.md" style="color:#c0e0ff">
     REMAINING_19_ANALYSIS.md</a>. The dominant cause is irreducible
     ETSNOCV decomposition truncation, not a wrong fragmentation choice
     (paper-inspired homolytic candidates produced Δres &lt; 0.05 kcal/mol
     versus the canonical winner).</p>
  <div class="legend-box">{render_legend_svg(380)}</div>
  <h4>Table of contents (grouped by failure category)</h4>
  <div class="toc">
    {''.join(f'<a href="#{rid}">[{cat}] {rid}</a><br>' for rid, _, cat in fail_cards)}
  </div>
</section>
"""
    fail_html += "\n".join(card for _, card, _ in fail_cards)
    fail_html += "</main></body></html>"
    (OUT / "fail_19.html").write_text(fail_html)

    # --- Index ---
    idx_html = HTML_HEAD.format(title="ADF-EDA documentation index")
    idx_html += f"""
<section class="summary">
  <h2 style="margin-top:0">EDA/ASM 62-reaction pipeline — final state</h2>
  <p>This is the documentation deliverable for the multi-sweep EDA/ASM
     fragmentation + spin-variant pipeline applied to the 62 reactions in
     <code>ADF_500/</code> that originally FAILed Check-3 or Check-4.</p>
  <table class="meta" style="margin-top:8px">
    <tr><th>Total originally FAILing</th><td>62</td></tr>
    <tr><th>Now PASS (Check-4 closed)</th><td><span class="pos">{len(pass_cards)}</span></td></tr>
    <tr><th>Still FAIL</th><td><span class="neg">{len(fail_cards)}</span></td></tr>
    <tr><th>Skipped (no canonical output)</th><td>{n_skipped}</td></tr>
  </table>
  <h4>Reports</h4>
  <ul style="line-height:1.8">
    <li><a href="pass_43.html"><b>pass_43.html</b></a> — {len(pass_cards)} reactions: R/TS/P geometries, selected fragmentation, residuals</li>
    <li><a href="fail_19.html"><b>fail_19.html</b></a> — {len(fail_cards)} reactions still failing, grouped by failure category</li>
    <li><a href="../manifest_edited.csv">manifest_edited.csv</a> — full validator output</li>
    <li><a href="../selection_report.json">selection_report.json</a> — per-reaction winner + all candidate scores</li>
    <li><a href="../REMAINING_19_ANALYSIS.md">REMAINING_19_ANALYSIS.md</a> — why those 19 cannot be reduced further</li>
  </ul>
  <h4>Pipeline timeline</h4>
  <ol style="line-height:1.8">
    <li>Sweep 1 — Fragmentation candidates (225 ADF runs, 5 strategies × 62 reactions)</li>
    <li>Sweep 2 — Spin variants (76 ADF runs: closed-shell / BS-singlet / FM / high-spin)</li>
    <li>Sweep 3 — Targeted retry (41 ADF runs: trajectory re-pick + multiplicity sweep)</li>
    <li>Sweep 4 — Paper-inspired homolytic cleavage (17 ADF runs per Fernández &amp; Bickelhaupt 2014, §4)</li>
  </ol>
  <p>Total: 359 ADF candidate runs feeding the canonical winner per reaction.</p>
</section>
"""
    idx_html += "</main></body></html>"
    (OUT / "index.html").write_text(idx_html)

    # Plaintext index for terminal users
    (OUT / "README.md").write_text(f"""# ADF-EDA documentation

| File | Contents |
|---|---|
| `index.html` | Overview + counts + links to reports |
| `pass_43.html` | {len(pass_cards)} reactions where Check-4 conservation now passes — R/TS/P geometries + chosen fragmentation per reaction |
| `fail_19.html` | {len(fail_cards)} reactions still FAIL, grouped by failure category |
| `README.md` | (this file) |

Each reaction card shows:
- Reaction id, family (T1x / Halogen), pattern, n_atoms, ΔE‡, residuals before / after pipeline
- Selected fragmentation strategy + per-fragment atom indices, electron count, multiplicity
- Bond changes (broken in R-only, formed in P-only)
- Three side-by-side SVG views (R, TS, P) with atoms colored by element and fragment membership shown as a colored ring

Open `index.html` in a browser. Or serve via the existing port-8889 server
(symlink it under `Validate/viz/` if you want to navigate from there).
""")

    print(f"Wrote {OUT}/index.html")
    print(f"      {OUT}/pass_43.html  ({len(pass_cards)} reactions)")
    print(f"      {OUT}/fail_19.html  ({len(fail_cards)} reactions)")
    print(f"      {OUT}/README.md")
    if n_skipped:
        print(f"  (skipped {n_skipped} reactions due to missing files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
