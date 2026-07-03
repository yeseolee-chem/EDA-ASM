"""Stage 3.8 â Build the manual review queue HTML pages.

For every reaction whose fragment definition was flagged (Case B with
auto_confidence < 0.8 or Case C if option 1 is chosen), render a static HTML
that the user can open in a browser. Page content:
- Reaction metadata (source, heavy atoms, bond changes, Ea).
- Two embedded 3Dmol.js viewers (R and TS) with atom-index labels.
- Bond-change table (broken / formed bond list with R and TS distances).
- Auto-suggested fragment cut and SMILES.
- A small JSON form so the reviewer's decision can be appended to
  manual_review_log.json (the page writes a textbox the user can paste, since
  static HTML cannot persist server-side state).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .halo8_io import _formula_from_numbers
from .logging_setup import get_logger, log_header
from .paths import (
    BOND_CHANGES_JSON,
    CASE_JSON,
    FRAGMENTS_AUTO_JSON,
    MANUAL_REVIEW_LOG,
    REVIEW_DIR,
    SELECTED_CSV,
    TMP_DIR,
    ensure_dirs,
)

ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def _xyz_block(numbers: np.ndarray, positions: np.ndarray) -> str:
    lines = [str(len(numbers)), "snapshot"]
    for z, p in zip(numbers, positions):
        lines.append(f"{ELEM.get(int(z), '?')} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    return "\n".join(lines)


def _viewer_div(div_id: str, xyz: str, label_atoms: bool = True, color_groups: dict | None = None) -> str:
    style = '{"sphere": {"scale": 0.25}, "stick": {}}'
    js_load = f'viewer.addModel(`{xyz}`, "xyz");'
    js_style = f'viewer.setStyle({{}}, {style});'
    js_extra = ""
    if color_groups:
        for color, atoms in color_groups.items():
            atoms_arg = ",".join(str(a) for a in atoms)
            js_extra += (
                f'viewer.setStyle({{"serial": [{atoms_arg}]}},'
                f'{{"sphere": {{"scale": 0.32, "color": "{color}"}}, "stick": {{"color": "{color}"}}}});'
            )
    if label_atoms:
        js_extra += "for (var i=0;i<viewer.getModel().selectedAtoms({}).length;i++){var a=viewer.getModel().selectedAtoms({})[i]; viewer.addLabel(i, {fontSize:11, backgroundOpacity:0.4}, a);}"
    return (
        f'<div id="{div_id}" style="height:300px;width:100%;position:relative;border:1px solid #ccc;"></div>'
        f'<script>$(function(){{var viewer=$3Dmol.createViewer("{div_id}",{{backgroundColor:"white"}});'
        f'{js_load}{js_style}{js_extra}viewer.zoomTo();viewer.render();}});</script>'
    )


def _bond_change_rows(numbers: np.ndarray, pos_R: np.ndarray, pos_TS: np.ndarray, bonds_broken, bonds_formed) -> str:
    rows = []
    for kind, lst in (("broken", bonds_broken), ("formed", bonds_formed)):
        for b in lst:
            i, j = int(b[0]), int(b[1])
            d_R = float(np.linalg.norm(pos_R[i] - pos_R[j]))
            d_TS = float(np.linalg.norm(pos_TS[i] - pos_TS[j]))
            sym_i = ELEM.get(int(numbers[i]), "?")
            sym_j = ELEM.get(int(numbers[j]), "?")
            rows.append(
                f"<tr><td>{kind}</td><td>{sym_i}{i}â{sym_j}{j}</td><td>{d_R:.3f}</td><td>{d_TS:.3f}</td><td>{abs(d_R-d_TS):.3f}</td></tr>"
            )
    if not rows:
        return ""
    return (
        "<table><thead><tr><th>kind</th><th>bond</th><th>d_R (Ã)</th><th>d_TS (Ã)</th><th>|Î|</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _meta_table(rid: str, info: dict, frag: dict | None) -> str:
    rows = [
        ("reaction_id", rid),
        ("case", info.get("case", "?")),
        ("rationale", info.get("rationale", "")),
        ("frag1 SMILES", frag.get("frag1_smiles") if frag else ""),
        ("frag2 SMILES", frag.get("frag2_smiles") if frag else ""),
        ("frag1 atoms", str(frag.get("frag1_atoms")) if frag else ""),
        ("frag2 atoms", str(frag.get("frag2_atoms")) if frag else ""),
        ("auto_confidence", str(frag.get("auto_confidence")) if frag else ""),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _decision_form(rid: str) -> str:
    return f"""
<h3>Reviewer decision</h3>
<p>Edit the JSON below to record your decision, then paste it into
<code>manual_review_log.json</code>.</p>
<textarea id="dec_{rid}" rows="9" style="width:100%;font-family:monospace;">
{{
  "reaction_id": "{rid}",
  "decision": "accept",
  "frag1_atoms": null,
  "frag2_atoms": null,
  "rationale": "",
  "reviewer": ""
}}
</textarea>
<p><i>Allowed values for "decision": "accept", "modify", "reject", "needs_help".</i></p>
"""


def _build_one(rid: str, info: dict, bond_data: dict, frag: dict | None) -> str | None:
    npz_path = TMP_DIR / f"{rid}.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as data:
        numbers = np.asarray(data["numbers"], dtype=int)
        coords = np.asarray(data["coords_5pts"])
    pos_R = coords[0]
    pos_TS = coords[4]
    xyz_R = _xyz_block(numbers, pos_R)
    xyz_TS = _xyz_block(numbers, pos_TS)

    color_groups = None
    if frag:
        color_groups = {
            "blue": frag.get("frag1_atoms", []),
            "orange": frag.get("frag2_atoms", []),
        }

    bond_html = _bond_change_rows(
        numbers,
        pos_R,
        pos_TS,
        bond_data.get("bonds_broken", []),
        bond_data.get("bonds_formed", []),
    )

    head = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Reaction {rid}</title>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>body{{font-family:sans-serif;max-width:920px;margin:24px auto;padding:0 16px;}}
h2{{font-size:16px;border-bottom:1px solid #ddd;}}
table{{border-collapse:collapse;margin:8px 0;}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:12px;}}</style>
</head><body>
<h1>Reaction {rid}</h1>
<p>{info.get('rationale','')}</p>"""
    body = []
    body.append("<h2>Metadata</h2>")
    body.append(_meta_table(rid, info, frag))
    body.append("<h2>R structure (atom indices labeled, fragment colors applied)</h2>")
    body.append(_viewer_div(f"v_R_{rid}", xyz_R, label_atoms=True, color_groups=color_groups))
    body.append("<h2>TS structure</h2>")
    body.append(_viewer_div(f"v_TS_{rid}", xyz_TS, label_atoms=True, color_groups=color_groups))
    body.append("<h2>Bond changes</h2>")
    body.append(bond_html or "<p>(no bond-change records)</p>")
    body.append(_decision_form(rid))
    body.append("</body></html>")
    return head + "\n".join(body)


def run(
    case_json: Path | None = None,
    bond_changes_json: Path | None = None,
    fragments_auto_json: Path | None = None,
    review_dir: Path | None = None,
    review_log: Path | None = None,
    *,
    case_c_strategy: str = "manual",
) -> dict:
    """case_c_strategy: 'manual' (build pages for every Case C) or 'exclude'."""
    ensure_dirs()
    log = get_logger("phase1.stage3_8")
    log_header(log, "3.8 manual review queue", case_c_strategy=case_c_strategy)
    if case_json is None:
        case_json = CASE_JSON
    if bond_changes_json is None:
        bond_changes_json = BOND_CHANGES_JSON
    if fragments_auto_json is None:
        fragments_auto_json = FRAGMENTS_AUTO_JSON
    if review_dir is None:
        review_dir = REVIEW_DIR
    if review_log is None:
        review_log = MANUAL_REVIEW_LOG

    cases = json.loads(case_json.read_text())
    bonds = json.loads(bond_changes_json.read_text())
    frags = json.loads(fragments_auto_json.read_text()) if fragments_auto_json.exists() else {}

    queue: list[str] = []
    for rid, info in cases.items():
        case = info["case"]
        if case == "C":
            if case_c_strategy == "manual":
                queue.append(rid)
            continue
        f = frags.get(rid)
        if not f:
            queue.append(rid)
            continue
        if f.get("review_status") in ("needs_review", "low_confidence"):
            queue.append(rid)
        if f.get("note"):
            queue.append(rid)
    queue = sorted(set(queue))
    log.info("Queueing %d reactions for manual review", len(queue))

    written = []
    for rid in queue:
        html = _build_one(rid, cases[rid], bonds.get(rid, {}), frags.get(rid))
        if html is None:
            log.warning("skip %s (no npz)", rid)
            continue
        out = review_dir / f"{rid}.html"
        out.write_text(html)
        written.append(rid)

    if not review_log.exists():
        review_log.write_text("[]")
    log.info("Wrote %d review pages to %s", len(written), review_dir)
    return {"queued": queue, "written": written}
