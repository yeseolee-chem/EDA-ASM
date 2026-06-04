"""Forming/breaking bond detection → cut → H-cap pipeline demo.

Picks a handful of multi-molecular reactions (P has ≥ 2 components at tight
cutoff) and generates standalone HTML viewers showing:

    * R with breaking bonds highlighted (red dashed)
    * P with each fragment coloured separately
    * H caps placed at each cut site (gray semi-transparent spheres)
    * Reactive-bond table and fragment SMILES (when RDKit succeeds)

Output goes under ``tools/phase1_5_review/static/demo/`` so the running
Flask app serves it directly at ``http://localhost:8888/static/demo/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.phase1.bonds import covalent_radius  # noqa: E402

OUT_DIR = ROOT / "tools" / "phase1_5_review" / "static" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIGHT_TOL = 1.10
H_CAP_LEN = 1.09
ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
ELEM_COLOR = {1: "#cccccc", 6: "#444444", 7: "#3050f0", 8: "#ff0d0d",
              9: "#90e050", 16: "#ffff30", 17: "#1ff01f", 35: "#a62929"}


def detect_tight(numbers, positions):
    n = len(numbers)
    radii = np.array([covalent_radius(int(z)) for z in numbers])
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    bonds = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            if d < 1e-3:
                continue
            if d < TIGHT_TOL * (radii[i] + radii[j]):
                bonds.add((i, j))
    return bonds


def components(n_atoms, bonds):
    g = nx.Graph()
    g.add_nodes_from(range(n_atoms))
    g.add_edges_from(bonds)
    return [set(c) for c in nx.connected_components(g)]


def h_cap_position(coords, anchor, partner):
    direction = coords[partner] - coords[anchor]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        unit = np.array([1.0, 0.0, 0.0])
    else:
        unit = direction / norm
    return (coords[anchor] + unit * H_CAP_LEN).tolist()


def xyz_block(numbers, positions, label="R"):
    lines = [str(len(numbers)), label]
    for z, p in zip(numbers, positions):
        sym = ELEM.get(int(z), "X")
        lines.append(f"{sym} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    return "\n".join(lines)


def build_html(rxn_id, source, numbers, coords_R, coords_P, bonds_R, bonds_P,
               P_comps, breaking_bonds, h_caps):
    """One self-contained HTML page with R and P viewers + metadata."""
    xyz_R = xyz_block(numbers, coords_R, "R")
    xyz_P = xyz_block(numbers, coords_P, "P")

    # Fragment colour for each atom (frag1 blue, frag2 orange, frag3 green)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    P_comps_sorted = sorted(P_comps, key=lambda c: (-len(c), min(c) if c else 0))
    frag_of = {}
    for idx, comp in enumerate(P_comps_sorted):
        for atom in comp:
            frag_of[atom] = idx

    # Build viewer JS
    def viewer_js(div_id, xyz, label, show_breaks, show_caps):
        cmds = [
            f'var v_{div_id} = $3Dmol.createViewer("{div_id}", {{backgroundColor: "white"}});',
            f'v_{div_id}.addModel(`{xyz}`, "xyz");',
            f'v_{div_id}.setStyle({{}}, {{sphere: {{scale: 0.25}}, stick: {{}}}});',
        ]
        # Colour atoms by fragment
        for atom, frag_idx in frag_of.items():
            colour = palette[frag_idx % len(palette)]
            cmds.append(
                f'v_{div_id}.setStyle({{"serial": {atom + 1}}}, '
                f'{{sphere: {{scale: 0.32, color: "{colour}"}}, '
                f'stick: {{color: "{colour}"}}}});'
            )
        # Atom labels: index + element
        for i, z in enumerate(numbers):
            sym = ELEM.get(int(z), "X")
            x, y, z_pos = coords_R[i] if label == "R" else coords_P[i]
            cmds.append(
                f'v_{div_id}.addLabel("{i} {sym}", '
                f'{{fontSize: 11, fontColor: "white", backgroundColor: "black", '
                f'backgroundOpacity: 0.6, inFront: true, '
                f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z_pos:.4f}}}}});'
            )
        # Breaking bonds — red dashed
        if show_breaks:
            for i, j in breaking_bonds:
                a, b = (coords_R[i], coords_R[j]) if label == "R" else (coords_P[i], coords_P[j])
                cmds.append(
                    f'v_{div_id}.addCylinder({{'
                    f'start: {{x: {a[0]:.4f}, y: {a[1]:.4f}, z: {a[2]:.4f}}}, '
                    f'end: {{x: {b[0]:.4f}, y: {b[1]:.4f}, z: {b[2]:.4f}}}, '
                    f'radius: 0.06, color: "#d62728", dashed: true}});'
                )
        # H caps — small gray spheres
        if show_caps:
            for cap in h_caps:
                p = cap["h_position"]
                cmds.append(
                    f'v_{div_id}.addSphere({{center: {{x: {p[0]:.4f}, '
                    f'y: {p[1]:.4f}, z: {p[2]:.4f}}}, radius: 0.32, '
                    f'color: "#888888", opacity: 0.65}});'
                )
                cmds.append(
                    f'v_{div_id}.addLabel("H*", {{'
                    f'fontSize: 10, fontColor: "white", backgroundColor: "#555", '
                    f'backgroundOpacity: 0.7, inFront: true, '
                    f'position: {{x: {p[0]:.4f}, y: {p[1]:.4f}, z: {p[2]:.4f}}}}});'
                )
        cmds.append(f'v_{div_id}.zoomTo();')
        cmds.append(f'v_{div_id}.render();')
        return "\n".join(cmds)

    # Reactive bonds table
    rows = []
    for i, j in sorted(breaking_bonds):
        si = ELEM.get(int(numbers[i]), "?")
        sj = ELEM.get(int(numbers[j]), "?")
        d_R = float(np.linalg.norm(coords_R[i] - coords_R[j]))
        d_P = float(np.linalg.norm(coords_P[i] - coords_P[j]))
        rows.append(
            f"<tr><td>{si}{i} — {sj}{j}</td><td>{d_R:.3f}</td><td>{d_P:.3f}</td>"
            f"<td>{frag_of.get(i, '?')}/{frag_of.get(j, '?')}</td></tr>"
        )
    bond_table = (
        "<table><thead><tr><th>bond</th><th>d(R) Å</th><th>d(P) Å</th>"
        "<th>frag(i)/frag(j)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )

    frag_rows = []
    for idx, comp in enumerate(P_comps_sorted):
        atoms = sorted(comp)
        formula_counts: dict[str, int] = {}
        for a in atoms:
            sym = ELEM.get(int(numbers[a]), "?")
            formula_counts[sym] = formula_counts.get(sym, 0) + 1
        formula = "".join(
            sym + (str(formula_counts[sym]) if formula_counts[sym] > 1 else "")
            for sym in sorted(formula_counts)
        )
        frag_rows.append(
            f"<tr><td>{idx}</td><td>{len(atoms)}</td>"
            f"<td>{formula}</td><td>{atoms}</td></tr>"
        )
    frag_table = (
        "<table><thead><tr><th>frag</th><th>n</th><th>formula</th>"
        "<th>atoms</th></tr></thead><tbody>"
        + "".join(frag_rows) + "</tbody></table>"
    )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{rxn_id}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 16px; }}
  h1, h2 {{ margin: 12px 0 4px; }}
  .viewers {{ display: flex; gap: 12px; }}
  .vbox {{ flex: 1; }}
  .v {{ height: 360px; border: 1px solid #ccc; }}
  table {{ border-collapse: collapse; font-size: 13px; margin: 6px 0; }}
  td, th {{ border: 1px solid #ddd; padding: 4px 8px; }}
  th {{ background: #eef; }}
  .legend {{ font-size: 12px; color: #555; margin-top: 4px; }}
</style></head><body>
<h1>{rxn_id}</h1>
<p><b>Source:</b> {source} · <b>n_atoms:</b> {len(numbers)} ·
   <b>P fragments:</b> {len(P_comps_sorted)} ·
   <b>breaking bonds (R→P):</b> {len(breaking_bonds)} ·
   <b>H caps placed:</b> {len(h_caps)}</p>

<div class="viewers">
  <div class="vbox">
    <h2>R (with breaking bonds dashed red)</h2>
    <div id="vR" class="v"></div>
    <div class="legend">Atoms coloured by their future P-fragment.</div>
  </div>
  <div class="vbox">
    <h2>P + H caps (gray spheres)</h2>
    <div id="vP" class="v"></div>
    <div class="legend">Fragments split; H atoms (semi-transparent) sit where
      the breaking bond used to attach.</div>
  </div>
</div>

<h2>Breaking bonds (cut sites)</h2>
{bond_table}

<h2>P-side fragments</h2>
{frag_table}

<script>
{viewer_js("vR", xyz_R, "R", show_breaks=True, show_caps=False)}
{viewer_js("vP", xyz_P, "P", show_breaks=False, show_caps=True)}
</script>
</body></html>
"""


def process_reaction(rxn_id, source):
    npz_r = ROOT / "outputs" / "phase1" / ".tmp" / f"{rxn_id}.npz"
    npz_p = ROOT / "outputs" / "phase1" / ".tmp_p" / f"{rxn_id}.npz"
    if not npz_r.exists() or not npz_p.exists():
        return None
    with np.load(npz_r, allow_pickle=True) as d:
        numbers = np.asarray(d["numbers"], dtype=int)
        coords_R = np.asarray(d["coords_5pts"])[0]
    with np.load(npz_p, allow_pickle=True) as d:
        coords_P = np.asarray(d["p_positions"])

    bonds_R = detect_tight(numbers, coords_R)
    bonds_P = detect_tight(numbers, coords_P)
    P_comps = components(len(numbers), bonds_P)

    # Map atom → fragment index (by P components, largest first)
    P_comps_sorted = sorted(P_comps, key=lambda c: (-len(c), min(c) if c else 0))
    frag_of = {}
    for idx, comp in enumerate(P_comps_sorted):
        for atom in comp:
            frag_of[atom] = idx

    # Breaking bonds = bonds in R that cross the fragment boundary
    breaking_bonds = sorted(
        (i, j) for (i, j) in bonds_R
        if frag_of.get(i) != frag_of.get(j)
    )

    # Place H caps at each side of each breaking bond
    h_caps = []
    for i, j in breaking_bonds:
        h_caps.append({
            "attached_to_atom": int(i),
            "h_position": h_cap_position(coords_P, i, j),
            "partner": int(j),
        })
        h_caps.append({
            "attached_to_atom": int(j),
            "h_position": h_cap_position(coords_P, j, i),
            "partner": int(i),
        })

    html = build_html(
        rxn_id, source, numbers, coords_R, coords_P,
        bonds_R, bonds_P, P_comps_sorted, breaking_bonds, h_caps,
    )
    out_path = OUT_DIR / f"{rxn_id}.html"
    out_path.write_text(html)
    return {
        "rxn_id": rxn_id,
        "source": source,
        "n_atoms": int(len(numbers)),
        "n_fragments": len(P_comps_sorted),
        "n_breaking_bonds": len(breaking_bonds),
        "n_h_caps": len(h_caps),
        "html_path": str(out_path.relative_to(ROOT)),
    }


def main():
    # Load 122 multi-molecular reactions among our 400 selected
    mm = json.loads((ROOT / "outputs" / "phase1" / "multimolecular_at_tight_110.json").read_text())
    df = pd.DataFrame(mm)
    df["total_atoms"] = df["sizes_R"].apply(lambda xs: sum(xs))
    # Pick a diverse sample: vary source AND molecular size.
    df_sorted = df.sort_values(["source", "total_atoms"], kind="mergesort")
    samples = []
    quotas = {"T1x": 4, "Halo_F": 2, "Halo_Cl": 2, "Halo_Br": 2}
    for src, group in df_sorted.groupby("source"):
        n_pick = quotas.get(src, 2)
        step = max(1, len(group) // n_pick)
        picks = group.iloc[::step].head(n_pick)
        samples.extend(picks["rxn_id"].tolist())

    print(f"Sampling {len(samples)} reactions:")
    summaries = []
    for rid in samples:
        src = df.loc[df["rxn_id"] == rid, "source"].iloc[0]
        summary = process_reaction(rid, src)
        if summary:
            print(f"  {rid:38s} ({src}) → n_atoms={summary['n_atoms']}, "
                  f"frags={summary['n_fragments']}, breaks={summary['n_breaking_bonds']}")
            summaries.append(summary)

    # Index page
    rows = "".join(
        f'<tr><td><a href="{s["rxn_id"]}.html">{s["rxn_id"]}</a></td>'
        f'<td>{s["source"]}</td><td>{s["n_atoms"]}</td>'
        f'<td>{s["n_fragments"]}</td><td>{s["n_breaking_bonds"]}</td>'
        f'<td>{s["n_h_caps"]}</td></tr>'
        for s in summaries
    )
    index_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Pipeline demo: forming/breaking → cut → H-cap</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 24px auto; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; font-size: 13px; }}
th {{ background: #eef; text-align: left; }}
a {{ color: #1a4d8c; }}
</style></head><body>
<h1>Pipeline demo</h1>
<p>Cordero × {TIGHT_TOL} bond cutoff. Multi-molecular reactions
(R or P with ≥ 2 components) were detected. For each shown below, the
breaking bonds were located and an H atom was placed at each side of the cut.</p>
<table>
<thead><tr><th>Reaction</th><th>source</th><th>atoms</th>
<th>P frags</th><th>breaking bonds</th><th>H caps</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body></html>
"""
    (OUT_DIR / "index.html").write_text(index_html)
    print(f"\nWrote {len(summaries)} HTML pages to {OUT_DIR}")
    print(f"Index: {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
