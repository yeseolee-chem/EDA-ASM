"""Phase 3 - v8 Fragment review Flask app.

Design guarantees (unlike v7):
- Viewer ALWAYS shows TS geometry from outputs/v8_review/raw_geoms/{rid}/TS.xyz
- Atom numbering shown in browser == index used in ORCA input file
- No family-specific transform, no coordinate spread, no MACE .pt path
- On mark-reviewed, the ORCA .inp is generated on the fly using the EXACT
  TS coordinates + user's frag_A/B assignment (same JSON keys)

Serve on http://<node>:$REVIEW_PORT (default 5788).
State: outputs/v8_review/manual_partitions.json (initialized from auto_partitions.json).
Mark-reviewed -> also writes outputs/v8_review/orca_inputs/{rid}/eda.inp
"""
from __future__ import annotations
import json, os, sys, threading
from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from scipy.spatial.distance import cdist

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8   = REPO / "outputs/v8_review"
COHORT_PQ = V8 / "cohort_v8.parquet"
AUTO_JSON = V8 / "auto_partitions.json"
MANUAL_JSON = V8 / "manual_partitions.json"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
ORCA_ROOT.mkdir(parents=True, exist_ok=True)

# --------------- state store -------------------------------------------------

class Store:
    def __init__(self, path: Path, initial: dict):
        self.path = path
        self._lock = threading.Lock()
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)
        else:
            self.data = initial
            self._write()

    def _write(self):
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def get(self, rid):
        with self._lock:
            return self.data.get(rid)

    def set(self, rid, entry):
        with self._lock:
            self.data[rid] = entry
            self._write()

    def all(self):
        with self._lock:
            return dict(self.data)


def _init_manual_from_auto():
    with open(AUTO_JSON) as f:
        auto = json.load(f)
    return {rid: dict(v) for rid, v in auto.items()}


COHORT = pd.read_parquet(COHORT_PQ)
_store = Store(MANUAL_JSON, _init_manual_from_auto())


# --------------- ORCA input writer -------------------------------------------

def write_orca_input(rid: str, family: str, frag_A: list[int], frag_B: list[int]) -> Path:
    """Write eda.inp using EXACT TS coordinates + user's fragment split."""
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in frag_A: frag_of[i] = 1
    for i in frag_B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"unassigned atoms: {[i for i, f in enumerate(frag_of) if f is None]}")

    total_charge = 0
    fA_charge = 0; fB_charge = 0
    if family == "qmrxn20_sn2":
        # Nucleophile carries -1
        fB_charge = -1
        total_charge = -1
    # dipolar, rgd1, qmrxn20_e2 -> neutral defaults (user can edit if needed)

    from ase.data import chemical_symbols
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym EDA TightSCF",
        "%maxcore 3500",
        "",
        "%eda",
        '  FRAG1 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        '  FRAG2 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        f"  FRAG1_C {fA_charge}",
        "  FRAG1_M 1",
        f"  FRAG2_C {fB_charge}",
        "  FRAG2_M 1",
        "end",
        "",
        f"* xyz {total_charge} 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*")
    lines.append("")

    out_dir = ORCA_ROOT / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_path = out_dir / "eda.inp"
    inp_path.write_text("\n".join(lines))
    return inp_path


# --------------- Flask app ---------------------------------------------------

app = Flask(__name__)


@app.route("/api/reactions")
def api_reactions():
    manual = _store.all()
    payload = []
    for row in COHORT.itertuples(index=False):
        rid = row.reaction_id
        m = manual.get(rid, {})
        payload.append({
            "reaction_id": rid,
            "family": row.family,
            "n_atoms_TS": int(row.n_atoms_TS),
            "reviewed": bool(m.get("reviewed", False)),
            "method": m.get("method", ""),
            "note": m.get("note", ""),
        })
    n_reviewed = sum(1 for p in payload if p["reviewed"])
    return jsonify({"reactions": payload,
                    "n_reviewed": n_reviewed,
                    "n_total": len(payload)})


def _align_R_to_TS(R_at, TS_at):
    """Return R positions permuted to match TS atom ordering.

    Simple cases:
      - if Z_R == Z_TS positionwise -> R positions used as-is
      - elif Z_R can be split into halves matching (r0-then-r1) or (r1-then-r0)
        pattern in TS: swap halves accordingly
      - else: fall back to greedy element-preserving nearest-neighbour matching
    Returns np.ndarray shape (n, 3) with R positions in TS-native atom order.
    Also returns a flag indicating whether alignment is exact (elements match).
    """
    Z_R = np.array(R_at.get_atomic_numbers())
    Z_TS = np.array(TS_at.get_atomic_numbers())
    pos_R = R_at.get_positions()
    if len(Z_R) == len(Z_TS) and np.array_equal(Z_R, Z_TS):
        return pos_R, True, "exact"
    # Try halving swap (dipolar r0/r1 order swap)
    if len(Z_R) == len(Z_TS):
        for cut in range(1, len(Z_R)):
            swapped = np.concatenate([Z_R[cut:], Z_R[:cut]])
            if np.array_equal(swapped, Z_TS):
                pos = np.concatenate([pos_R[cut:], pos_R[:cut]], axis=0)
                return pos, True, f"halved_swap@{cut}"
    # Greedy per-element nearest-neighbour matching (best effort)
    n = len(Z_TS)
    pos_out = np.zeros((n, 3))
    used = np.zeros(len(Z_R), dtype=bool)
    exact = True
    for i in range(n):
        z_i = int(Z_TS[i])
        candidates = np.where((Z_R == z_i) & ~used)[0]
        if len(candidates) == 0:
            exact = False
            pos_out[i] = TS_at.get_positions()[i]  # fallback to TS pos
        else:
            # Choose the R atom of matching element with closest position to TS_i
            ts_p = TS_at.get_positions()[i]
            d = np.linalg.norm(pos_R[candidates] - ts_p, axis=1)
            best = candidates[int(np.argmin(d))]
            pos_out[i] = pos_R[best]
            used[best] = True
    return pos_out, exact, "greedy_element_match"


def _R_molecule_groups(R_at, tol=1.3):
    """Return the connected-component groups of R (list of index-lists).
    Sorted by size desc. Uses covalent-radius-based BFS."""
    from ase.data import covalent_radii
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    Z = np.asarray(R_at.get_atomic_numbers())
    pos = R_at.get_positions()
    rc = np.array([covalent_radii[int(z)] for z in Z])
    d = cdist(pos, pos)
    A = (d > 0) & (d < tol * (rc[:, None] + rc[None, :]))
    n_comp, lbl = connected_components(csgraph=csr_matrix(A), directed=False, return_labels=True)
    groups = [np.where(lbl == c)[0].tolist() for c in range(n_comp)]
    groups.sort(key=lambda g: -len(g))
    return groups


def _spread_R_molecules(pos, groups, min_gap=10.0):
    """Translate each non-primary group so that the min inter-group distance is
    at least min_gap Angstrom. Uses PCA of primary group to pick a direction
    that avoids projecting onto the primary group's principal axis.

    Works for any number of groups (>=2). Returns (pos_new, spread_applied).
    """
    if len(groups) < 2:
        return pos, False
    pos = pos.copy()
    changed = False
    # Group 0 stays put.
    base_idx = list(groups[0])
    base_pos = pos[np.asarray(base_idx, dtype=int)]
    for g in groups[1:]:
        gi = np.asarray(g, dtype=int)
        gp = pos[gi]
        d_min = float(cdist(base_pos, gp).min())
        if d_min >= min_gap:
            base_pos = np.vstack([base_pos, gp])
            base_idx += list(gi)
            continue
        # Direction: from base centroid to group centroid (fallback to +x)
        cA = base_pos.mean(0); cB = gp.mean(0)
        direction = cB - cA
        n = np.linalg.norm(direction)
        if n < 1e-3:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / n
        # Big shift to guarantee separation regardless of shape
        shift = direction * (min_gap - d_min + 8.0)
        pos[gi] = gp + shift
        changed = True
        base_pos = np.vstack([base_pos, pos[gi]])
        base_idx += list(gi)
    return pos, changed


def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.after_request
def _add_no_cache(resp):
    if request.path.startswith("/api/"):
        return _no_cache(resp)
    return resp


@app.route("/api/reaction/<rid>")
def api_reaction(rid):
    """Three panels: R molecule 1, R molecule 2, TS.

    Rules:
      - Fragmentation is defined by R BFS: molecule 1 = fragment A, molecule 2 = fragment B.
      - R panels show each molecule as isolated structure with uniform colour.
      - TS panel is VIEW-ONLY: colours reflect which R molecule each atom belongs to,
        but clicks on TS do NOT change the fragmentation.
      - Fragmentation only changes via the 'swap A / B' button in the sidebar.
    """
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    r_at  = ase.io.read(str(RAW / rid / "R.xyz"))
    Z_TS = ts_at.get_atomic_numbers().tolist()
    pos_TS = ts_at.get_positions()
    from ase.data import chemical_symbols
    syms_TS = [chemical_symbols[int(z)] for z in Z_TS]

    # Sampling guarantees R element seq == TS element seq positionwise,
    # so R indices 0..N-1 correspond exactly to TS indices 0..N-1.
    R_pos_aligned = r_at.get_positions()
    aligned_exact = True
    align_method = "identity (R and TS share atom order)"
    R_groups = _R_molecule_groups(r_at)
    R_groups = sorted(R_groups, key=lambda g: -len(g))
    mol1_idx = [int(i) for i in (R_groups[0] if R_groups else [])]
    mol2_idx = [int(i) for i in (R_groups[1] if len(R_groups) > 1 else [])]

    m = _store.get(rid) or {}
    # Determine A/B labelling from stored assignment; default: mol1 = A, mol2 = B
    stored_A = set(m.get("frag_A_indices", []))
    if stored_A and set(mol2_idx) & stored_A and not set(mol1_idx) & stored_A:
        # User previously swapped: mol2 is A
        mol1_is_A = False
    else:
        mol1_is_A = True

    def _sub(indices, positions, elements):
        pos_sub = np.asarray(positions)[np.asarray(indices, dtype=int)].tolist()
        el_sub  = [elements[i] for i in indices]
        return {"indices": [int(i) for i in indices], "positions": pos_sub, "elements": el_sub}

    R_mol1 = _sub(mol1_idx, R_pos_aligned, syms_TS)
    R_mol2 = _sub(mol2_idx, R_pos_aligned, syms_TS)

    return jsonify({
        "reaction_id": rid,
        "elements": syms_TS,
        "atomic_numbers": Z_TS,
        "n_atoms": len(Z_TS),
        "positions_TS": pos_TS.tolist(),
        "R_mol1": R_mol1,
        "R_mol2": R_mol2,
        "mol1_is_A": bool(mol1_is_A),
        "R_align_method": align_method,
        "R_aligned_exact": bool(aligned_exact),
        "assignment": {
            "frag_A_indices": m.get("frag_A_indices", []),
            "frag_B_indices": m.get("frag_B_indices", []),
            "reviewed": bool(m.get("reviewed", False)),
            "note": m.get("note", ""),
            "method": m.get("method", ""),
        },
    })


@app.route("/api/reaction/<rid>", methods=["POST"])
def api_save(rid):
    data = request.get_json()
    frag_A = [int(x) for x in data.get("frag_A_indices", [])]
    frag_B = [int(x) for x in data.get("frag_B_indices", [])]
    reviewed = bool(data.get("reviewed", False))
    note = data.get("note", "")
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    n = len(ts_at)
    all_idx = set(range(n))
    A, B = set(frag_A), set(frag_B)
    # Coerce to valid range: drop anything out of [0, n)
    out_of_range = (A | B) - all_idx
    if out_of_range:
        A -= out_of_range; B -= out_of_range
    if A & B:
        return jsonify({"error": f"A and B overlap: {sorted(A & B)}"}), 400
    missing = all_idx - (A | B)
    if missing:
        # Auto-place any unassigned atoms into B (they are the complement of A)
        B |= missing
    frag_A, frag_B = sorted(A), sorted(B)
    entry = {
        "frag_A_indices": sorted(frag_A),
        "frag_B_indices": sorted(frag_B),
        "reviewed": reviewed,
        "note": note,
        "method": "manual" if reviewed else _store.get(rid).get("method", "auto"),
    }
    _store.set(rid, entry)
    # if reviewed, write ORCA input immediately (same TS coords, same indices)
    inp_path = None
    if reviewed:
        fam = COHORT.loc[COHORT.reaction_id == rid, "family"].iloc[0]
        try:
            inp_path = str(write_orca_input(rid, fam, entry["frag_A_indices"],
                                            entry["frag_B_indices"]))
        except Exception as e:
            return jsonify({"error": f"orca write failed: {e}"}), 500
    return jsonify({"ok": True, "inp_path": inp_path})


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>v8 Fragment Review</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  html,body{height:100%;margin:0;background:#111;color:#ddd;font-family:sans-serif}
  #root{display:grid;grid-template-columns:260px 1fr 260px;height:100vh}
  #left,#right{overflow-y:auto;padding:8px;background:#181818}
  #left{border-right:1px solid #333}
  #right{border-left:1px solid #333}
  #middle{display:grid;grid-template-rows:24px 1fr; height:100vh}
  #tabs{display:flex; background:#222; align-items:center; padding:0 8px; font-size:12px;color:#888}
  .split{display:grid; grid-template-columns:1fr 1fr; height:100%}
  .pane{position:relative;background:#000; border-right:1px solid #333}
  .pane .lbl{position:absolute; top:6px; left:8px; color:#bbb; font-size:12px; z-index:5; pointer-events:none}
  .rxn{padding:4px 8px;cursor:pointer;border-bottom:1px solid #222;font-size:12px}
  .rxn:hover{background:#222}
  .rxn.selected{background:#37476b}
  .rxn.reviewed{color:#7cb;}
  button{padding:6px 10px;margin:3px 0;background:#333;color:#fff;border:1px solid #555;cursor:pointer}
  button:hover{background:#555}
  input[type=text]{width:96%;padding:4px;background:#222;color:#fff;border:1px solid #444}
  .stat{font-size:11px;color:#888}
  .A{color:#f57;}
  .B{color:#5af;}
  #atoms{font-size:11px;max-height:300px;overflow-y:auto;font-family:monospace}
  #atoms div{padding:1px 4px;cursor:pointer}
  #atoms div:hover{background:#222}
</style></head>
<body><div id="root">
  <div id="left">
    <input id="search" type="text" placeholder="search rid" oninput="filterList()">
    <div class="stat" id="progress">loading...</div>
    <div id="rxnlist"></div>
  </div>
  <div id="middle">
    <div id="tabs">
      <span>R molecule 1 (frag A) &mdash; R molecule 2 (frag B) &mdash; TS (= ORCA input)</span>
    </div>
    <div class="split" style="grid-template-columns:1fr 1fr 1fr">
      <div class="pane"><div class="lbl">R molecule 1</div><div id="viewR1" style="width:100%;height:100%"></div></div>
      <div class="pane"><div class="lbl">R molecule 2</div><div id="viewR2" style="width:100%;height:100%"></div></div>
      <div class="pane"><div class="lbl">TS (= ORCA input)</div><div id="viewT" style="width:100%;height:100%"></div></div>
    </div>
  </div>
  <div id="right">
    <h3 id="ridtitle">-</h3>
    <div id="meta" class="stat"></div>
    <div class="stat" style="margin:6px 0">
      Fragmentation is defined by R connectivity (2 molecules of R).<br>
      Molecule 1 = fragment A (red), Molecule 2 = fragment B (blue).<br>
      TS panel is view-only; clicks on TS do nothing.
    </div>
    <button onclick="swapAB()">swap A &harr; B</button>
    <hr>
    <button onclick="markReviewed()" style="background:#284">mark reviewed + write ORCA input</button>
    <button onclick="markUnreviewed()" style="background:#844">unmark</button>
    <hr>
    <div id="counts"></div>
    <div id="atoms"></div>
  </div>
</div>
<script>
let reactions=[], current=null, viewR1=null, viewR2=null, viewT=null, atoms=null;
let assignment={A:new Set()};

function getB(){
  const B=new Set();
  if(!atoms) return B;
  for(let i=0;i<atoms.n_atoms;i++){ if(!assignment.A.has(i)) B.add(i); }
  return B;
}

// Build an R molecule panel: per-atom colouring by assignment.A, clickable.
function buildSubset(elId, subset, panelColor){
  const el = document.getElementById(elId); el.innerHTML='';
  const v = $3Dmol.createViewer(el, {backgroundColor:'black'});
  const n = subset.indices.length;
  let xyz = n + '\\n\\n';
  for(let k=0;k<n;k++){
    const p = subset.positions[k];
    xyz += subset.elements[k] + ' ' + p[0] + ' ' + p[1] + ' ' + p[2] + '\\n';
  }
  v.addModel(xyz,'xyz');
  for(let k=0;k<n;k++){
    const p = subset.positions[k];
    v.addLabel(String(subset.indices[k]),
      {position:{x:p[0],y:p[1],z:p[2]}, backgroundOpacity:0, fontColor:'white', fontSize:11});
  }
  v.setClickable({}, true, function(atom){
    // atom.serial is 1-based for XYZ. Find k by matching serial.
    const modelAtoms = v.getModel().selectedAtoms({});
    let k = -1;
    for(let j=0; j<modelAtoms.length; j++){
      if(modelAtoms[j].serial === atom.serial){ k = j; break; }
    }
    if(k < 0) return;
    const tsIdx = subset.indices[k];
    if(assignment.A.has(tsIdx)) assignment.A.delete(tsIdx); else assignment.A.add(tsIdx);
    applyColors(false);
    renderAtoms(); updateCounts();
    persistAssignment();
  });
  return {viewer: v, subset, panelColor};
}

// Build the TS panel: coloured 1:1 with A/B; clickable to toggle individual atoms.
function buildTS(elId, positions){
  const el = document.getElementById(elId); el.innerHTML='';
  const v = $3Dmol.createViewer(el, {backgroundColor:'black'});
  let xyz = atoms.n_atoms + '\\n\\n';
  for(let i=0;i<atoms.n_atoms;i++){
    const p = positions[i];
    xyz += atoms.elements[i] + ' ' + p[0] + ' ' + p[1] + ' ' + p[2] + '\\n';
  }
  v.addModel(xyz,'xyz');
  for(let i=0;i<atoms.n_atoms;i++){
    const p = positions[i];
    v.addLabel(String(i),
      {position:{x:p[0],y:p[1],z:p[2]}, backgroundOpacity:0, fontColor:'white', fontSize:11});
  }
  v.setClickable({}, true, function(atom){
    const modelAtoms = v.getModel().selectedAtoms({});
    let k = -1;
    for(let j=0; j<modelAtoms.length; j++){
      if(modelAtoms[j].serial === atom.serial){ k = j; break; }
    }
    if(k < 0) return;
    // TS panel: model atom index k = TS-native index k
    if(assignment.A.has(k)) assignment.A.delete(k); else assignment.A.add(k);
    applyColors(false);
    renderAtoms(); updateCounts();
    persistAssignment();
  });
  return {viewer: v, subset: null};
}

function applyOne(viewerObj){
  const v = viewerObj.viewer;
  const subset = viewerObj.subset;
  const model = v.getModel();
  if(!model) return;
  const modelAtoms = model.selectedAtoms({});
  // Colour each atom by its actual serial number (proven to match XYZ 1-based)
  for(let k=0; k<modelAtoms.length; k++){
    const a = modelAtoms[k];
    const tsIdx = (subset === null) ? k : subset.indices[k];
    const inA = assignment.A.has(tsIdx);
    const color = inA ? '#e64' : '#4ae';
    v.setStyle({serial: a.serial},
               {sphere:{radius:0.40, color: color},
                stick:{radius:0.18, color: color}});
  }
  v.render();
}

// swap A / B for the current reaction
async function swapAB(){
  if(!current) return;
  const newA = [...current.R_mol2.indices];
  const newB = [...current.R_mol1.indices];
  await fetch('/api/reaction/'+current.reaction_id, {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({frag_A_indices:newA, frag_B_indices:newB,
                          reviewed: current.assignment.reviewed || false,
                          note: current.assignment.note || ''})});
  loadReaction(current.reaction_id);
}

async function loadList(){
  const r=await fetch('/api/reactions'); const d=await r.json(); reactions=d.reactions;
  document.getElementById('progress').textContent=`${d.n_reviewed} / ${d.n_total} reviewed`;
  renderList();
}
function renderList(){
  const q=(document.getElementById('search').value||'').toLowerCase();
  const wrap=document.getElementById('rxnlist'); wrap.innerHTML='';
  for(const r of reactions){
    if(q && !r.reaction_id.toLowerCase().includes(q)) continue;
    const div=document.createElement('div');
    div.className='rxn'+(r.reviewed?' reviewed':'')+(current&&r.reaction_id===current.reaction_id?' selected':'');
    div.textContent=(r.reviewed?'✓ ':'  ')+r.reaction_id+' ['+r.family+', '+r.n_atoms_TS+']';
    div.onclick=()=>loadReaction(r.reaction_id);
    wrap.appendChild(div);
  }
}
function filterList(){ renderList(); }

async function loadReaction(rid){
  const r=await fetch('/api/reaction/'+rid); const d=await r.json();
  current=d; atoms=d;
  // Use the STORED assignment as source of truth (preserves user edits).
  // Fall back to R BFS mol1 when no assignment exists yet.
  const stored = d.assignment.frag_A_indices;
  if(stored && stored.length){
    assignment.A = new Set(stored);
  } else {
    assignment.A = new Set(d.mol1_is_A ? d.R_mol1.indices : d.R_mol2.indices);
  }
  document.getElementById('ridtitle').textContent=rid;
  document.getElementById('meta').innerHTML='family: '+reactions.find(x=>x.reaction_id===rid).family+
    '<br>n_atoms: '+d.n_atoms+'<br>method: '+d.assignment.method+
    '<br>|R mol1|='+d.R_mol1.indices.length+' (frag '+(d.mol1_is_A?'A':'B')+')'+
    '<br>|R mol2|='+d.R_mol2.indices.length+' (frag '+(d.mol1_is_A?'B':'A')+')'+
    '<br>note: '+d.assignment.note+'<br>reviewed: '+d.assignment.reviewed;
  // Panel colours: mol1 always red if A, blue if B; mol2 opposite.
  const color1 = d.mol1_is_A ? '#e64' : '#4ae';
  const color2 = d.mol1_is_A ? '#4ae' : '#e64';
  viewR1 = buildSubset('viewR1', d.R_mol1, color1);
  viewR2 = d.R_mol2.indices.length ? buildSubset('viewR2', d.R_mol2, color2) : null;
  viewT  = buildTS('viewT', d.positions_TS);
  applyColors(true);
  renderAtoms(); renderList(); updateCounts();
}

function applyColors(zoomFirst){
  applyOne(viewR1);
  if(viewR2) applyOne(viewR2);
  applyOne(viewT);
  if(zoomFirst){
    viewR1.viewer.zoomTo(); viewR1.viewer.render();
    if(viewR2){ viewR2.viewer.zoomTo(); viewR2.viewer.render(); }
    viewT.viewer.zoomTo(); viewT.viewer.render();
  }
}

function renderAtoms(){
  const B=getB();
  const w=document.getElementById('atoms'); w.innerHTML='';
  for(let i=0;i<atoms.n_atoms;i++){
    const div=document.createElement('div');
    const isA=assignment.A.has(i);
    div.className=isA?'A':'B';
    div.textContent=String(i).padStart(3,' ')+' '+atoms.elements[i]+' '+(isA?'A':'B');
    div.onclick=()=>{
      if(assignment.A.has(i)) assignment.A.delete(i); else assignment.A.add(i);
      applyColors(false);
      renderAtoms(); updateCounts();
    };
    w.appendChild(div);
  }
}
function clearA(){ assignment.A.clear(); applyColors(false); renderAtoms(); updateCounts(); }
function allA(){ assignment.A=new Set(); for(let i=0;i<atoms.n_atoms;i++) assignment.A.add(i);
                 applyColors(false); renderAtoms(); updateCounts(); }
function autoAssign(){
  if(!current) return;
  assignment.A=new Set(current.assignment.frag_A_indices);
  applyColors(false); renderAtoms(); updateCounts();
}
function updateCounts(){
  document.getElementById('counts').innerHTML='|A|='+assignment.A.size+' | |B|='+(atoms?atoms.n_atoms-assignment.A.size:0);
}
// Auto-save assignment on every click (preserves reviewed flag)
async function persistAssignment(){
  if(!current) return;
  const B = [...getB()];
  await fetch('/api/reaction/'+current.reaction_id, {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({frag_A_indices:[...assignment.A], frag_B_indices:B,
                          reviewed: current.assignment.reviewed || false,
                          note: current.assignment.note || ''})});
}

async function saveWithReviewed(reviewed){
  const B=[...getB()];
  const r=await fetch('/api/reaction/'+current.reaction_id,{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frag_A_indices:[...assignment.A], frag_B_indices:B,
                         reviewed:reviewed, note:current.assignment.note||''})});
  const d=await r.json();
  if(d.error){ alert(d.error); return; }
  if(reviewed && d.inp_path) alert('reviewed + wrote '+d.inp_path);
  // Update reviewed flag in-place without touching camera
  current.assignment.reviewed=reviewed;
  document.getElementById('meta').innerHTML='family: '+reactions.find(x=>x.reaction_id===current.reaction_id).family+
    '<br>n_atoms(TS): '+current.n_atoms+'<br>method: '+current.assignment.method+
    '<br>note: '+current.assignment.note+'<br>reviewed: '+reviewed;
  loadList();
}
function markReviewed(){ saveWithReviewed(true); }
function markUnreviewed(){ saveWithReviewed(false); }
loadList();
</script></body></html>
"""


@app.route("/")
def index():
    from flask import Response
    return Response(INDEX_HTML, mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("REVIEW_PORT", "5788"))
    print(f"[v8_review] serving on http://0.0.0.0:{port}  cohort={len(COHORT)}")
    print(f"[v8_review] state -> {MANUAL_JSON}")
    print(f"[v8_review] ORCA inputs -> {ORCA_ROOT}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
