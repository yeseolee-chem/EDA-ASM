"""Flask app for manually reviewing fragA/fragB assignment across 787 reactions.

Launches a browser UI where each reaction's TS geometry is shown in 3D and the
user clicks atoms to assign them to fragment A or B. Auto-saves every change
to manual_partitions.json.

Layout:
  - Left panel  : reaction list w/ family filter + review-status flags
  - Center      : 3D viewer (3Dmol.js) — click atoms, keyboard shortcuts
  - Right       : atom index list + assignment summary + save/skip controls

Run via `scripts/frag_review_app.sh` on a compute node (see that script for
port-forwarding instructions).
"""
from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.data import chemical_symbols
from flask import Flask, jsonify, request, send_from_directory

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
FEAT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
AUTO_PART = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json")
# Prefer refined cohort_v7 (with replacements) if available, else v6.
_V7 = REPO / "outputs/frag_review/cohort_v7.parquet"
_V6 = REPO / "labels/adf/adf_labels_v6_multifamily.parquet"
LABELS_PQ = _V7 if _V7.exists() else _V6

STATE_DIR = REPO / "outputs/frag_review"
STATE_DIR.mkdir(parents=True, exist_ok=True)
# PART_FILE env var selects which partitions to display (default: manual).
_PART_NAME = os.environ.get("PART_FILE", "manual_partitions.json")
MANUAL_PART = STATE_DIR / _PART_NAME
STATIC_DIR = STATE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class Store:
    """Thread-safe atomic-write JSON store keyed by reaction_id."""

    def __init__(self, path: Path, initial: dict):
        self.path = path
        self._lock = threading.Lock()
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)
        else:
            self.data = initial
            self._write_locked()

    def _write_locked(self):
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def get(self, rid: str):
        with self._lock:
            return self.data.get(rid)

    def set(self, rid: str, entry: dict):
        with self._lock:
            self.data[rid] = entry
            self._write_locked()

    def all(self):
        with self._lock:
            return dict(self.data)


def _load_reactions() -> list[dict]:
    """Return list of {reaction_id, family, n_atoms} for all 787."""
    labels = pd.read_parquet(LABELS_PQ)
    with open(AUTO_PART) as f:
        auto = json.load(f)
    out = []
    for row in labels.itertuples(index=False):
        rid, fam = row.reaction_id, row.family
        n = auto.get(rid, {}).get("n_atoms", None)
        out.append({"reaction_id": rid, "family": fam, "n_atoms": n})
    return out


VIEW_GEOM = os.environ.get("VIEW_GEOM", "family")  # "R" | "TS" | "family"

# Per-family geom preference (matches refine_fragments.py FAM_GEOM):
FAM_GEOM = {"dipolar": "R", "rgd1": "R",
            "qmrxn20_e2": "R", "qmrxn20_sn2": "R"}


def _resolve_geom(family: str) -> str:
    if VIEW_GEOM == "family":
        return FAM_GEOM.get(family, "R")
    return VIEW_GEOM


VIEW_MIN_GAP = 5.0   # Å — artificially spread fragments apart if closer than this


def _maybe_spread_fragments(z: np.ndarray, pos: np.ndarray, rid: str) -> tuple[np.ndarray, bool]:
    """If frag A and B are physically close (bound complex), translate frag B
    along the vector from A's centroid to B's centroid so they clearly split
    apart in the viewer. Returns (new_pos, was_spread).
    """
    entry = _store.get(rid) or {}
    A = entry.get("frag_A_indices") or []
    B = entry.get("frag_B_indices") or []
    if not A or not B or (len(A) + len(B) != len(pos)):
        return pos, False
    pA = pos[A]; pB = pos[B]
    dmin = np.linalg.norm(pA[:, None, :] - pB[None, :, :], axis=-1).min()
    if dmin >= VIEW_MIN_GAP:
        return pos, False
    cA = pA.mean(axis=0); cB = pB.mean(axis=0)
    v = cB - cA
    n = np.linalg.norm(v)
    if n < 1e-6:
        v = np.array([1.0, 0.0, 0.0]); n = 1.0
    unit = v / n
    # Push B outward so that min A–B distance is ≈ VIEW_MIN_GAP.
    shift = unit * (VIEW_MIN_GAP - dmin + 2.0)
    new_pos = pos.copy()
    new_pos[B] = pos[B] + shift
    return new_pos, True


def _load_view_atoms(rid: str, family: str) -> dict:
    d = torch.load(str(FEAT_DIR / f"{rid}.pt"), map_location="cpu", weights_only=False)
    want = _resolve_geom(family)
    key = want if want in d else next(k for k in ("R", "TS", "P") if k in d)
    z = np.asarray(d[key]["z"], dtype=int)
    pos = np.asarray(d[key]["pos"], dtype=float)
    pos, spread = _maybe_spread_fragments(z, pos, rid)
    atoms = [
        {"i": i, "elem": chemical_symbols[int(zi)], "x": p[0], "y": p[1], "z": p[2]}
        for i, (zi, p) in enumerate(zip(z.tolist(), pos.tolist()))
    ]
    geom_shown = f"{key}{'+spread' if spread else ''}"
    return {"reaction_id": rid, "atoms": atoms, "geom_shown": geom_shown}


def _initial_manual_from_auto(auto_path: Path) -> dict:
    with open(auto_path) as f:
        auto = json.load(f)
    out = {}
    for rid, v in auto.items():
        if "frag_A_indices" not in v:
            continue
        out[rid] = {
            "frag_A_indices": list(v["frag_A_indices"]),
            "frag_B_indices": list(v["frag_B_indices"]),
            "reviewed": False,
            "note": "",
        }
    return out


app = Flask(__name__)
_reactions = _load_reactions()
_store = Store(MANUAL_PART, _initial_manual_from_auto(AUTO_PART))


@app.route("/")
def index():
    from flask import Response
    resp = Response(INDEX_HTML, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/reactions")
def api_reactions():
    manual = _store.all()
    payload = []
    for r in _reactions:
        rid = r["reaction_id"]
        m = manual.get(rid, {})
        payload.append({
            **r,
            "reviewed": bool(m.get("reviewed", False)),
            "discard": bool(m.get("discard", False)),
            "n_A": len(m.get("frag_A_indices", [])),
            "n_B": len(m.get("frag_B_indices", [])),
        })
    return jsonify(payload)


@app.route("/api/reaction/<rid>")
def api_reaction(rid):
    fam = next((r["family"] for r in _reactions if r["reaction_id"] == rid), "dipolar")
    try:
        atoms = _load_view_atoms(rid, fam)
    except FileNotFoundError:
        return jsonify({"error": f"no feature file for {rid}"}), 404
    manual = _store.get(rid) or {"frag_A_indices": [], "frag_B_indices": [], "reviewed": False, "note": ""}
    return jsonify({**atoms, "assignment": manual})


@app.route("/api/reaction/<rid>", methods=["POST"])
def api_reaction_save(rid):
    body = request.get_json(force=True)
    frag_A = sorted(set(int(i) for i in body.get("frag_A_indices", [])))
    frag_B = sorted(set(int(i) for i in body.get("frag_B_indices", [])))
    overlap = set(frag_A) & set(frag_B)
    if overlap:
        return jsonify({"error": f"atoms in both fragments: {sorted(overlap)}"}), 400
    entry = {
        "frag_A_indices": frag_A,
        "frag_B_indices": frag_B,
        "reviewed": bool(body.get("reviewed", True)),
        "note": str(body.get("note", "")),
        "discard": bool(body.get("discard", False)),
    }
    _store.set(rid, entry)
    return jsonify({"ok": True, "assignment": entry})


@app.route("/api/reaction/<rid>/unmark", methods=["POST"])
def api_reaction_unmark(rid):
    entry = _store.get(rid) or {}
    entry["reviewed"] = False
    _store.set(rid, entry)
    return jsonify({"ok": True})


@app.route("/api/reaction/<rid>/discard", methods=["POST"])
def api_reaction_discard(rid):
    entry = _store.get(rid) or {}
    entry["discard"] = True
    entry["reviewed"] = False
    _store.set(rid, entry)
    return jsonify({"ok": True})


@app.route("/api/reaction/<rid>/undiscard", methods=["POST"])
def api_reaction_undiscard(rid):
    entry = _store.get(rid) or {}
    entry["discard"] = False
    _store.set(rid, entry)
    return jsonify({"ok": True})


@app.route("/api/progress")
def api_progress():
    manual = _store.all()
    total = len(_reactions)
    n_reviewed = sum(1 for r in _reactions if manual.get(r["reaction_id"], {}).get("reviewed"))
    per_fam = {}
    for r in _reactions:
        fam = r["family"]
        per_fam.setdefault(fam, {"total": 0, "reviewed": 0})
        per_fam[fam]["total"] += 1
        if manual.get(r["reaction_id"], {}).get("reviewed"):
            per_fam[fam]["reviewed"] += 1
    return jsonify({"total": total, "reviewed": n_reviewed, "per_family": per_fam})


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Fragment Review — 787 reactions</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, sans-serif; background:#111; color:#eee; }
  #app { display: grid; grid-template-columns: 260px 1fr 300px; height: 100vh; }
  #left { border-right: 1px solid #333; overflow-y: auto; padding: 8px; }
  #right { border-left: 1px solid #333; overflow-y: auto; padding: 8px; }
  #center { display: flex; flex-direction: column; }
  #viewer { flex: 1; position: relative; background:#000; }
  #controls { padding: 8px; border-top: 1px solid #333; display: flex; gap: 6px; align-items:center; flex-wrap: wrap; }
  button { background: #2a2a2a; color: #eee; border: 1px solid #555; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 13px; }
  button:hover { background:#3a3a3a; }
  button.primary { background:#0a6; border-color:#080; }
  button.danger { background:#a20; border-color:#800; }
  .rxn-row { padding: 5px 6px; cursor: pointer; border-bottom: 1px solid #222; font-size: 12px; display:flex; justify-content:space-between; }
  .rxn-row:hover { background:#222; }
  .rxn-row.active { background:#036; }
  .rxn-row.reviewed { color:#8f8; }
  .rxn-row.discard { color:#f66; text-decoration: line-through; }
  .badge { font-size:10px; padding:0 4px; border-radius:3px; }
  .fam-dipolar { background:#456; }
  .fam-rgd1 { background:#546; }
  .fam-qmrxn20_e2 { background:#645; }
  .fam-qmrxn20_sn2 { background:#564; }
  #status { padding: 6px; font-size: 12px; color:#aaa; border-top: 1px solid #333; }
  .atom-row { display: grid; grid-template-columns: 40px 30px 1fr 40px 40px; gap: 4px; padding: 2px 4px; font-size: 12px; border-bottom: 1px solid #222; align-items: center; }
  .atom-row.frag-A { background: #205; }
  .atom-row.frag-B { background: #520; }
  .atom-row button { padding: 1px 4px; font-size: 11px; }
  h4 { margin: 6px 0 4px; font-size: 13px; color:#aaa; text-transform: uppercase; }
  input[type="text"] { background:#222; color:#eee; border:1px solid #444; border-radius:3px; padding: 3px 6px; font-size:12px; width: 100%; }
  select { background:#222; color:#eee; border:1px solid #444; padding:3px; font-size:12px; }
  .hint { font-size: 11px; color:#888; }
  .flexrow { display: flex; gap: 4px; align-items: center; }
</style>
</head>
<body>
<div id="app">
  <div id="left">
    <div class="flexrow" style="margin-bottom:8px">
      <select id="fam-filter">
        <option value="">all families</option>
        <option value="dipolar">dipolar</option>
        <option value="qmrxn20_e2">qmrxn20_e2</option>
        <option value="qmrxn20_sn2">qmrxn20_sn2</option>
        <option value="rgd1">rgd1</option>
      </select>
      <select id="review-filter">
        <option value="">all</option>
        <option value="pending">pending</option>
        <option value="done">reviewed</option>
      </select>
    </div>
    <div id="rxn-list"></div>
    <div id="status"></div>
  </div>
  <div id="center">
    <div id="viewer"></div>
    <div id="controls">
      <button id="prev-btn">← Prev</button>
      <button id="next-btn">Next →</button>
      <button id="assign-a" class="primary">Assign A (a)</button>
      <button id="assign-b" class="primary">Assign B (b)</button>
      <button id="clear-sel">Clear (c)</button>
      <button id="sel-all">Sel all</button>
      <button id="sel-invert">Invert</button>
      <button id="sel-unassigned">Sel unassigned</button>
      <button id="sel-h">Sel H</button>
      <button id="swap-ab">Swap A↔B (s)</button>
      <button id="mark-reviewed" class="primary">Mark Reviewed (r)</button>
      <button id="unmark-reviewed">Unmark (u)</button>
      <button id="discard" class="danger">Discard (d)</button>
      <button id="reset-auto" class="danger">Reset to auto</button>
      <input type="text" id="range-input" placeholder="e.g. 0-5, 12, 17-19" style="width:180px">
      <button id="sel-range">Sel range</button>
      <span class="hint">click atoms (3D or chip) · a/b=assign · r=review · n/p=next/prev · s=swap</span>
    </div>
  </div>
  <div id="right">
    <h4>Current reaction</h4>
    <div id="rxn-info"></div>
    <h4>All atoms (click to select)</h4>
    <div id="all-atoms-list"></div>
    <h4>Fragment A (<span id="count-a">0</span>)</h4>
    <div id="frag-a-list" style="color:#8bf"></div>
    <h4>Fragment B (<span id="count-b">0</span>)</h4>
    <div id="frag-b-list" style="color:#fb8"></div>
    <h4>Unassigned (<span id="count-u">0</span>)</h4>
    <div id="unassigned-list" style="color:#888"></div>
    <h4>Note</h4>
    <input type="text" id="note-input" placeholder="e.g. LG=Cl">
  </div>
</div>

<script>
const state = {
  reactions: [],
  current: null,          // reaction_id
  atoms: [],              // [{i,elem,x,y,z}]
  A: new Set(),
  B: new Set(),
  selected: new Set(),    // for user click-selection before assign
  viewer: null,
  clickHandlers: [],
  filter: { fam: '', review: '' },
};

const COLOR_A = '0x3388ff';
const COLOR_B = '0xff7733';
const COLOR_U = '0xaaaaaa';
const COLOR_S = '0xffff33';   // selected before assign

async function loadReactions() {
  const r = await fetch('/api/reactions').then(r => r.json());
  state.reactions = r;
  renderList();
  const first = r.find(x => !x.reviewed) || r[0];
  if (first) selectReaction(first.reaction_id);
}

function renderList() {
  const el = document.getElementById('rxn-list');
  el.innerHTML = '';
  const filt = state.filter;
  let shown = 0;
  for (const r of state.reactions) {
    if (filt.fam && r.family !== filt.fam) continue;
    if (filt.review === 'pending' && r.reviewed) continue;
    if (filt.review === 'done' && !r.reviewed) continue;
    const row = document.createElement('div');
    row.className = 'rxn-row' + (r.reviewed ? ' reviewed' : '') + (r.discard ? ' discard' : '') + (r.reaction_id === state.current ? ' active' : '');
    const prefix = r.discard ? '✗ ' : (r.reviewed ? '✓ ' : '');
    row.innerHTML = `<span>${prefix}${r.reaction_id}</span>
                     <span class="badge fam-${r.family}">${r.family.replace('qmrxn20_','')}</span>`;
    row.onclick = () => selectReaction(r.reaction_id);
    el.appendChild(row);
    shown++;
  }
  updateStatus(shown);
}

async function updateStatus(shown) {
  const p = await fetch('/api/progress').then(r => r.json());
  document.getElementById('status').innerHTML = `
    <b>${p.reviewed}/${p.total}</b> reviewed &nbsp; (${shown} shown)<br>
    ` + Object.entries(p.per_family).map(([k,v]) =>
      `<span class="badge fam-${k}">${k.replace('qmrxn20_','')}</span> ${v.reviewed}/${v.total}`
    ).join(' ');
}

async function selectReaction(rid) {
  const data = await fetch('/api/reaction/' + rid).then(r => r.json());
  if (data.error) { alert(data.error); return; }
  state.current = rid;
  state.atoms = data.atoms;
  state.geom_shown = data.geom_shown || 'R';
  state.A = new Set(data.assignment.frag_A_indices);
  state.B = new Set(data.assignment.frag_B_indices);
  state.selected = new Set();
  document.getElementById('note-input').value = data.assignment.note || '';
  renderViewer();
  renderRightPanel();
  renderList();
}

function renderViewer() {
  const el = document.getElementById('viewer');
  el.innerHTML = '';
  const v = $3Dmol.createViewer(el, { backgroundColor: 'black' });
  const xyz = state.atoms.length + '\n\n' +
    state.atoms.map(a => `${a.elem} ${a.x} ${a.y} ${a.z}`).join('\n');
  v.addModel(xyz, 'xyz');
  // 3Dmol.js's XYZ parser may serial-index atoms as 0-based or 1-based
  // depending on version. Build an explicit map by iterating the model in
  // insertion order and matching against the input XYZ order.
  const modelAtoms = v.getModel().selectedAtoms({});
  state.serialToIndex = {};
  state.indexToSerial = {};
  for (let idx = 0; idx < modelAtoms.length; idx++) {
    const s = modelAtoms[idx].serial;
    state.serialToIndex[s] = idx;
    state.indexToSerial[idx] = s;
  }
  v.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.30 } });
  colorAtoms(v);
  v.zoomTo();
  v.render();
  v.setClickable({}, true, (atom, viewer, event) => {
    const i = state.serialToIndex[atom.serial];
    if (i === undefined) { console.warn('unknown atom.serial', atom.serial); return; }
    toggleSelect(i);
  });
  state.viewer = v;
}

function colorAtoms(v) {
  v.removeAllLabels();
  for (let i = 0; i < state.atoms.length; i++) {
    let color;
    if (state.selected.has(i)) color = COLOR_S;
    else if (state.A.has(i)) color = COLOR_A;
    else if (state.B.has(i)) color = COLOR_B;
    else color = COLOR_U;
    const serial = state.indexToSerial[i];
    v.setStyle({ serial: serial }, { stick: { radius: 0.15, color }, sphere: { scale: 0.30, color } });
    v.addLabel(String(i), { position: state.atoms[i], fontSize: 18, fontColor: 'white',
                             backgroundColor: 'black', backgroundOpacity: 0.5,
                             borderThickness: 0, inFront: true });
  }
}

function toggleSelect(i) {
  if (state.selected.has(i)) state.selected.delete(i);
  else state.selected.add(i);
  colorAtoms(state.viewer); state.viewer.render();
  renderRightPanel();
}

function makeAtomChip(i, forceBg) {
  const sel = state.selected.has(i);
  let bg = forceBg;
  if (!bg) {
    if (sel) bg = '#ff3';
    else if (state.A.has(i)) bg = '#26a';
    else if (state.B.has(i)) bg = '#a52';
    else bg = '#333';
  }
  const fg = sel ? '#000' : '#fff';
  const style = `margin:2px; padding:3px 7px; border:1px solid #666; border-radius:3px;
                 cursor:pointer; display:inline-block; font-size:13px; user-select:none;
                 background:${bg}; color:${fg}; font-weight:${sel ? 'bold' : 'normal'};`;
  return `<span data-atom="${i}" class="atom-chip" style="${style}">${i}${state.atoms[i]?.elem||'?'}</span>`;
}

function renderRightPanel() {
  const info = document.getElementById('rxn-info');
  const r = state.reactions.find(x => x.reaction_id === state.current);
  info.innerHTML = `<b>${state.current}</b><br>family: ${r?.family} · n_atoms: ${state.atoms.length}
    · <b style="color:#8f8">view: ${state.geom_shown || 'R'}</b><br>
    <span class="hint">Click chips below to (de)select atoms — same as clicking in 3D view.</span>`;
  const listAtoms = (idx_set, elid) => {
    const el = document.getElementById(elid);
    const arr = [...idx_set].sort((a,b) => a - b);
    el.innerHTML = arr.length === 0 ? '<span class="hint">(none)</span>' :
      arr.map(i => makeAtomChip(i)).join('');
  };
  // All atoms (always visible, colored by A/B/unassigned)
  const allEl = document.getElementById('all-atoms-list');
  allEl.innerHTML = state.atoms.map(a => makeAtomChip(a.i)).join('');
  listAtoms(state.A, 'frag-a-list');
  listAtoms(state.B, 'frag-b-list');
  const assigned = new Set([...state.A, ...state.B]);
  const unassigned = state.atoms.map(a => a.i).filter(i => !assigned.has(i));
  document.getElementById('count-a').textContent = state.A.size;
  document.getElementById('count-b').textContent = state.B.size;
  document.getElementById('count-u').textContent = unassigned.length;
  const uel = document.getElementById('unassigned-list');
  uel.innerHTML = unassigned.length === 0 ? '<span class="hint">(all assigned)</span>' :
    unassigned.map(i => makeAtomChip(i)).join('');
  // wire chip clicks
  for (const chip of document.querySelectorAll('.atom-chip')) {
    chip.onclick = () => toggleSelect(parseInt(chip.dataset.atom));
  }
}

function selectAll() {
  state.selected = new Set(state.atoms.map(a => a.i));
  colorAtoms(state.viewer); state.viewer.render(); renderRightPanel();
}
function selectNone() {
  state.selected.clear();
  colorAtoms(state.viewer); state.viewer.render(); renderRightPanel();
}
function selectInvert() {
  const all = new Set(state.atoms.map(a => a.i));
  const inv = new Set([...all].filter(i => !state.selected.has(i)));
  state.selected = inv;
  colorAtoms(state.viewer); state.viewer.render(); renderRightPanel();
}
function selectUnassigned() {
  const assigned = new Set([...state.A, ...state.B]);
  state.selected = new Set(state.atoms.map(a => a.i).filter(i => !assigned.has(i)));
  colorAtoms(state.viewer); state.viewer.render(); renderRightPanel();
}
function selectByElement(elem) {
  state.selected = new Set(state.atoms.filter(a => a.elem === elem).map(a => a.i));
  colorAtoms(state.viewer); state.viewer.render(); renderRightPanel();
}
function selectByRange(str) {
  // "0-5, 12, 17-19" → set of indices
  const out = new Set();
  for (const part of str.split(',')) {
    const p = part.trim();
    if (!p) continue;
    if (p.includes('-')) {
      const [a,b] = p.split('-').map(x => parseInt(x.trim()));
      for (let i = a; i <= b; i++) out.add(i);
    } else {
      out.add(parseInt(p));
    }
  }
  state.selected = out;
  colorAtoms(state.viewer); state.viewer.render(); renderRightPanel();
}

async function saveCurrent(reviewed) {
  if (!state.current) return;
  const body = {
    frag_A_indices: [...state.A],
    frag_B_indices: [...state.B],
    reviewed: reviewed,
    note: document.getElementById('note-input').value,
  };
  const r = await fetch('/api/reaction/' + state.current, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).then(r => r.json());
  if (r.error) { alert(r.error); return false; }
  const row = state.reactions.find(x => x.reaction_id === state.current);
  if (row) { row.reviewed = reviewed; row.n_A = state.A.size; row.n_B = state.B.size; }
  renderList();
  return true;
}

function assignSelected(target) {
  if (state.selected.size === 0) return;
  const other = target === 'A' ? state.B : state.A;
  const targ = target === 'A' ? state.A : state.B;
  for (const i of state.selected) { other.delete(i); targ.add(i); }
  state.selected.clear();
  colorAtoms(state.viewer); state.viewer.render();
  renderRightPanel();
  saveCurrent(state.reactions.find(x => x.reaction_id === state.current)?.reviewed || false);
}

function swapAB() {
  const t = state.A; state.A = state.B; state.B = t;
  colorAtoms(state.viewer); state.viewer.render();
  renderRightPanel();
  saveCurrent(state.reactions.find(x => x.reaction_id === state.current)?.reviewed || false);
}

async function resetToAuto() {
  if (!confirm('Reset this reaction to the auto partition?')) return;
  // The auto partition is the fallback; fetch it from /api/reaction and reload
  const rid = state.current;
  // Force reload by re-fetching (but manual store has been modified). We reset via server.
  const r = await fetch('/api/reset_to_auto/' + rid, {method: 'POST'}).then(r => r.json()).catch(_ => null);
  if (r && r.ok) selectReaction(rid);
  else alert('reset endpoint not available; skip and re-load auto by clearing frag_review dir');
}

async function markReviewed() {
  if (!state.current) return;
  const assigned = state.A.size + state.B.size;
  if (assigned !== state.atoms.length) {
    if (!confirm(`Only ${assigned}/${state.atoms.length} atoms assigned. Mark reviewed anyway?`)) return;
  }
  await saveCurrent(true);
  nextReaction();
}

async function unmarkReviewed() {
  if (!state.current) return;
  await fetch('/api/reaction/' + state.current + '/unmark', {method: 'POST'}).then(r => r.json());
  const row = state.reactions.find(x => x.reaction_id === state.current);
  if (row) row.reviewed = false;
  renderList();
}

async function discardReaction() {
  if (!state.current) return;
  if (!confirm('Discard this reaction and replace it with a new one from the raw pool on next refinement run?')) return;
  await fetch('/api/reaction/' + state.current + '/discard', {method: 'POST'}).then(r => r.json());
  const row = state.reactions.find(x => x.reaction_id === state.current);
  if (row) { row.reviewed = false; row.discard = true; }
  renderList();
  nextReaction();
}

function nextReaction() {
  const visible = getVisibleList();
  const idx = visible.findIndex(r => r.reaction_id === state.current);
  if (idx < 0 || idx + 1 >= visible.length) return;
  selectReaction(visible[idx + 1].reaction_id);
}

function prevReaction() {
  const visible = getVisibleList();
  const idx = visible.findIndex(r => r.reaction_id === state.current);
  if (idx <= 0) return;
  selectReaction(visible[idx - 1].reaction_id);
}

function getVisibleList() {
  const filt = state.filter;
  return state.reactions.filter(r =>
    (!filt.fam || r.family === filt.fam) &&
    (!filt.review || (filt.review === 'pending' ? !r.reviewed : r.reviewed))
  );
}

document.getElementById('assign-a').onclick = () => assignSelected('A');
document.getElementById('assign-b').onclick = () => assignSelected('B');
document.getElementById('clear-sel').onclick = selectNone;
document.getElementById('sel-all').onclick = selectAll;
document.getElementById('sel-invert').onclick = selectInvert;
document.getElementById('sel-unassigned').onclick = selectUnassigned;
document.getElementById('sel-h').onclick = () => selectByElement('H');
document.getElementById('sel-range').onclick = () => selectByRange(document.getElementById('range-input').value);
document.getElementById('swap-ab').onclick = swapAB;
document.getElementById('mark-reviewed').onclick = markReviewed;
document.getElementById('unmark-reviewed').onclick = unmarkReviewed;
document.getElementById('discard').onclick = discardReaction;
document.getElementById('reset-auto').onclick = resetToAuto;
document.getElementById('prev-btn').onclick = prevReaction;
document.getElementById('next-btn').onclick = nextReaction;
document.getElementById('fam-filter').onchange = e => { state.filter.fam = e.target.value; renderList(); };
document.getElementById('review-filter').onchange = e => { state.filter.review = e.target.value; renderList(); };
document.getElementById('note-input').onblur = () => saveCurrent(state.reactions.find(x => x.reaction_id === state.current)?.reviewed || false);
document.getElementById('range-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') selectByRange(e.target.value);
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'a') assignSelected('A');
  else if (e.key === 'b') assignSelected('B');
  else if (e.key === 'r') markReviewed();
  else if (e.key === 'u') unmarkReviewed();
  else if (e.key === 'd') discardReaction();
  else if (e.key === 'n') nextReaction();
  else if (e.key === 'p') prevReaction();
  else if (e.key === 'c') selectNone();
  else if (e.key === 's') swapAB();
});

loadReactions();
</script>
</body>
</html>
"""


@app.route("/api/reset_to_auto/<rid>", methods=["POST"])
def api_reset_to_auto(rid):
    with open(AUTO_PART) as f:
        auto = json.load(f)
    v = auto.get(rid)
    if not v or "frag_A_indices" not in v:
        return jsonify({"error": "no auto entry"}), 404
    entry = {
        "frag_A_indices": list(v["frag_A_indices"]),
        "frag_B_indices": list(v["frag_B_indices"]),
        "reviewed": False,
        "note": "",
    }
    _store.set(rid, entry)
    return jsonify({"ok": True})


def main():
    port = int(os.environ.get("REVIEW_PORT", "5788"))
    host = os.environ.get("REVIEW_HOST", "0.0.0.0")
    node = socket.gethostname()
    print(f"[frag_review] geometry shown = {VIEW_GEOM}", flush=True)
    print(f"[frag_review] serving on http://{node}:{port}  (state → {MANUAL_PART})", flush=True)
    print(f"[frag_review] port-forward from local:  ssh -N -L {port}:{node}:{port} gate1.hpc", flush=True)
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
