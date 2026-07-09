"""Build a single self-contained HTML file that visualises fragment assignments
for all 776 ORCA EDA input files. Parses each eda.inp for atoms + fragment IDs,
embeds everything as JSON in the page. 3Dmol.js loaded from CDN.

No server needed — user just opens the HTML in a local browser.
"""
from __future__ import annotations
import json, re
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
INPUT_ROOT = REPO / "outputs/orca_eda/inputs"
OUT_HTML = REPO / "outputs/orca_eda/fragmentation_viewer.html"

ATOM_RE = re.compile(r"^\s*([A-Z][a-z]?)\((\d+)\)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")


def parse_inp(path: Path):
    xyz = ""
    in_block = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("* xyz"):
            in_block = True; continue
        if in_block and s == "*":
            break
        if in_block:
            m = ATOM_RE.match(line)
            if m:
                elem, fid, x, y, z = m.groups()
                xyz += f"{elem} {fid} {x} {y} {z}\n"
    return xyz


def main():
    import pandas as pd
    labels = pd.read_parquet(REPO / "outputs/frag_review/cohort_v7.parquet")
    fam_of = dict(zip(labels.reaction_id, labels.family))

    reactions = []
    for d in sorted(INPUT_ROOT.iterdir()):
        if not d.is_dir(): continue
        inp = d / "eda.inp"
        if not inp.exists(): continue
        data = parse_inp(inp)
        if not data: continue
        reactions.append({
            "rid": d.name,
            "family": fam_of.get(d.name, "?"),
            "atoms": data.strip(),
        })
    print(f"packed {len(reactions)} reactions")

    js_payload = json.dumps(reactions)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", js_payload)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML}  ({OUT_HTML.stat().st_size/1e6:.1f} MB)")


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ORCA EDA fragmentation viewer</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, sans-serif; background:#111; color:#eee; }
  #app { display: grid; grid-template-columns: 280px 1fr 260px; height: 100vh; }
  #left { border-right: 1px solid #333; overflow-y: auto; padding: 8px; }
  #right { border-left: 1px solid #333; overflow-y: auto; padding: 8px; }
  #center { display: flex; flex-direction: column; }
  #viewer { flex: 1; background:#000; position: relative; }
  #status { padding: 6px; font-size: 12px; color:#aaa; border-top: 1px solid #333; }
  .rxn-row { padding: 5px 6px; cursor: pointer; border-bottom: 1px solid #222; font-size: 12px; display:flex; justify-content:space-between; }
  .rxn-row:hover { background:#222; }
  .rxn-row.active { background:#036; }
  .badge { font-size:10px; padding:0 4px; border-radius:3px; }
  .fam-dipolar { background:#456; }
  .fam-rgd1 { background:#546; }
  .fam-qmrxn20_e2 { background:#645; }
  .fam-qmrxn20_sn2 { background:#564; }
  h4 { margin: 6px 0 4px; font-size: 13px; color:#aaa; text-transform: uppercase; }
  select { background:#222; color:#eee; border:1px solid #444; padding:3px; font-size:12px; width:100%; margin-bottom:4px; }
  input[type="text"] { background:#222; color:#eee; border:1px solid #444; padding:3px 5px; font-size:12px; width:100%; margin-bottom:4px; }
  .atom-chip { display:inline-block; margin:2px; padding:2px 6px; border-radius:3px; font-size:12px; }
  .chip-A { background:#26a; color:#fff; }
  .chip-B { background:#a52; color:#fff; }
</style>
</head>
<body>
<div id="app">
  <div id="left">
    <input type="text" id="search" placeholder="search reaction_id...">
    <select id="fam-filter">
      <option value="">all families</option>
      <option value="dipolar">dipolar</option>
      <option value="qmrxn20_e2">qmrxn20_e2</option>
      <option value="qmrxn20_sn2">qmrxn20_sn2</option>
      <option value="rgd1">rgd1</option>
    </select>
    <div id="rxn-list"></div>
    <div id="status"></div>
  </div>
  <div id="center">
    <div id="viewer"></div>
  </div>
  <div id="right">
    <h4>Reaction</h4>
    <div id="rxn-info"></div>
    <h4>Fragment A (<span id="count-a">0</span>)</h4>
    <div id="frag-a-list"></div>
    <h4>Fragment B (<span id="count-b">0</span>)</h4>
    <div id="frag-b-list"></div>
  </div>
</div>

<script>
const REACTIONS = __PAYLOAD__;

const state = { current: null, atoms: [], filter: {fam:'', text:''} };
const COLOR_A = '0x3388ff', COLOR_B = '0xff7733';

function parseAtoms(text) {
  const atoms = [];
  for (const line of text.split('\n')) {
    const [elem, fid, x, y, z] = line.trim().split(/\s+/);
    if (!elem) continue;
    atoms.push({ elem, fid: parseInt(fid), x: parseFloat(x), y: parseFloat(y), z: parseFloat(z) });
  }
  return atoms;
}

function makeXYZ(atoms) {
  const n = atoms.length;
  return `${n}\n\n` + atoms.map(a => `${a.elem} ${a.x} ${a.y} ${a.z}`).join('\n');
}

function renderViewer(atoms) {
  const el = document.getElementById('viewer');
  el.innerHTML = '';
  const v = $3Dmol.createViewer(el, { backgroundColor: 'black' });
  v.addModel(makeXYZ(atoms), 'xyz');
  const model_atoms = v.getModel().selectedAtoms({});
  const serialMap = {};
  for (let i = 0; i < model_atoms.length; i++) serialMap[i] = model_atoms[i].serial;
  v.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.30 } });
  for (let i = 0; i < atoms.length; i++) {
    const color = atoms[i].fid === 1 ? COLOR_A : COLOR_B;
    v.setStyle({ serial: serialMap[i] }, {
      stick: { radius: 0.15, color }, sphere: { scale: 0.30, color }
    });
    v.addLabel(String(i), {
      position: atoms[i], fontSize: 16, fontColor: 'white',
      backgroundColor: 'black', backgroundOpacity: 0.5,
      borderThickness: 0, inFront: true
    });
  }
  v.zoomTo(); v.render();
}

function getVisible() {
  const {fam, text} = state.filter;
  const t = text.toLowerCase();
  return REACTIONS.filter(r =>
    (!fam || r.family === fam) && (!t || r.rid.toLowerCase().includes(t))
  );
}

function renderList() {
  const el = document.getElementById('rxn-list');
  el.innerHTML = '';
  const vis = getVisible();
  for (const r of vis) {
    const row = document.createElement('div');
    row.className = 'rxn-row' + (r.rid === state.current ? ' active' : '');
    row.innerHTML = `<span>${r.rid}</span><span class="badge fam-${r.family}">${r.family.replace('qmrxn20_','')}</span>`;
    row.onclick = () => selectReaction(r.rid);
    el.appendChild(row);
  }
  document.getElementById('status').textContent = `${vis.length} shown / ${REACTIONS.length} total`;
}

function renderInfo(r, atoms) {
  document.getElementById('rxn-info').innerHTML =
    `<b>${r.rid}</b><br>family: ${r.family}<br>n_atoms: ${atoms.length}`;
  const A = atoms.filter((_,i) => atoms[i].fid === 1);
  const B = atoms.filter((_,i) => atoms[i].fid === 2);
  document.getElementById('count-a').textContent = A.length;
  document.getElementById('count-b').textContent = B.length;
  const chipsA = atoms.map((a,i) => a.fid === 1 ? `<span class="atom-chip chip-A">${i}${a.elem}</span>` : '').join('');
  const chipsB = atoms.map((a,i) => a.fid === 2 ? `<span class="atom-chip chip-B">${i}${a.elem}</span>` : '').join('');
  document.getElementById('frag-a-list').innerHTML = chipsA || '<i>(none)</i>';
  document.getElementById('frag-b-list').innerHTML = chipsB || '<i>(none)</i>';
}

function selectReaction(rid) {
  const r = REACTIONS.find(x => x.rid === rid);
  if (!r) return;
  state.current = rid;
  state.atoms = parseAtoms(r.atoms);
  renderViewer(state.atoms);
  renderInfo(r, state.atoms);
  renderList();
}

document.getElementById('fam-filter').onchange = e => { state.filter.fam = e.target.value; renderList(); };
document.getElementById('search').oninput = e => { state.filter.text = e.target.value; renderList(); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const vis = getVisible();
  const idx = vis.findIndex(r => r.rid === state.current);
  if (e.key === 'n' && idx + 1 < vis.length) selectReaction(vis[idx+1].rid);
  else if (e.key === 'p' && idx > 0) selectReaction(vis[idx-1].rid);
});

// Initial
renderList();
if (REACTIONS.length) selectReaction(REACTIONS[0].rid);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
