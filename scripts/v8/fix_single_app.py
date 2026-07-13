"""Small Flask app for editing a single rxn's fragment A/B assignment.
Reuses the failed_viewer UI + adds click-to-toggle + save button.

Usage: RID=dipolar_000658 python fix_single_app.py

Endpoints:
  GET  /                     -> HTML editor
  GET  /api/rxn              -> {R_xyz, TS_xyz, frag_A, frag_B, err, sumZ_A, sumZ_B}
  POST /api/save             -> {frag_A, frag_B} -> saves + regens eda.inp
"""
from __future__ import annotations
import json, os, threading
from pathlib import Path

import ase.io
import numpy as np
from ase.data import chemical_symbols
from flask import Flask, jsonify, request, Response

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
MANUAL_JSON = V8 / "manual_partitions.json"

RID = os.environ.get("RID", "dipolar_000658")
FAMILY = RID.split("_")[0] + (("_" + RID.split("_")[1]) if RID.startswith("qmrxn20") else "")
PORT = int(os.environ.get("FIX_PORT", "5578"))

_lock = threading.Lock()

app = Flask(__name__)


def _read_manual():
    return json.loads(MANUAL_JSON.read_text())


def _write_manual(d):
    tmp = MANUAL_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, MANUAL_JSON)


def write_orca_input(rid: str, family: str, frag_A, frag_B) -> Path:
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in frag_A:
        frag_of[i] = 1
    for i in frag_B:
        frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"unassigned atoms: {[i for i, f in enumerate(frag_of) if f is None]}")

    total_charge = 0
    fA_c = 0; fB_c = 0
    if family == "qmrxn20_sn2":
        fB_c = -1; total_charge = -1
    elif family == "qmrxn20_e2":
        fB_c = -1; total_charge = -1

    lines = [
        "! BLYP D3BJ def2-TZVP NoSym EDA TightSCF",
        "%maxcore 3500",
        "",
        "%eda",
        '  FRAG1 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        '  FRAG2 "BLYP D3BJ def2-TZVP NoSym TightSCF"',
        f"  FRAG1_C {fA_c}",
        "  FRAG1_M 1",
        f"  FRAG2_C {fB_c}",
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
    inp = out_dir / "eda.inp"
    inp.write_text("\n".join(lines))
    # also drop previous eda.out so runner retries
    (out_dir / "eda.out").unlink(missing_ok=True)
    (out_dir / "eda.err").unlink(missing_ok=True)
    return inp


HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>fix __RID__</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
body { font-family: -apple-system, sans-serif; margin: 0; padding: 12px; background: #111; color: #ddd; }
h1 { font-size: 15px; margin: 0 0 8px 0; color: #fff; }
.controls { margin-bottom: 8px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
button { background: #222; color: #ddd; border: 1px solid #444; padding: 6px 12px; font-size: 13px; cursor: pointer; }
button:hover { background: #333; }
button.primary { background: #2a5; border-color: #4a8; color: white; font-weight: bold; }
button.primary:hover { background: #3b6; }
.info { font-size: 12px; color: #aaa; }
.err { color: #f77; font-size: 12px; margin-top: 4px; }
.ok { color: #7f7; font-size: 12px; margin-top: 4px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; height: calc(100vh - 160px); }
.panel { background: #000; border: 1px solid #333; position: relative; }
.panel-title { position: absolute; top: 4px; left: 8px; font-size: 12px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 2px 6px; }
.legend { position: absolute; bottom: 4px; left: 8px; font-size: 11px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 3px 6px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
#counter { font-family: monospace; font-size: 13px; }
</style>
</head>
<body>
<h1>fix single rxn: __RID__ (click an atom to toggle A ↔ B)</h1>
<div class="controls">
  <button class="primary" onclick="save()">💾 save + regen inp</button>
  <button onclick="swap()">swap all A ↔ B</button>
  <button onclick="reload()">reload</button>
  <label class="info"><input type="checkbox" id="labels" onchange="renderAll()" checked> atom labels</label>
  <span id="counter"></span>
</div>
<div id="msg" class="info"></div>
<div class="panels">
  <div class="panel">
    <div class="panel-title">R (reactants) — click any atom to toggle</div>
    <div class="legend"><span class="dot" style="background:#e64"></span>frag A &nbsp; <span class="dot" style="background:#4ae"></span>frag B</div>
    <div id="viewR" style="width:100%; height:100%"></div>
  </div>
  <div class="panel">
    <div class="panel-title">TS — click any atom to toggle (this geom goes into eda.inp)</div>
    <div class="legend"><span class="dot" style="background:#e64"></span>frag A &nbsp; <span class="dot" style="background:#4ae"></span>frag B</div>
    <div id="viewTS" style="width:100%; height:100%"></div>
  </div>
</div>

<script>
let DATA = null;
let vR = null, vTS = null;
let modelR = null, modelTS = null;

async function fetchData() {
  const r = await fetch("/api/rxn?ts=" + Date.now());
  DATA = await r.json();
  reloadUI();
}

function updateCounter() {
  const el = document.getElementById("counter");
  const nA = DATA.frag_A.length, nB = DATA.frag_B.length;
  let sumA = 0, sumB = 0;
  for (const i of DATA.frag_A) sumA += DATA.Z[i];
  for (const i of DATA.frag_B) sumB += DATA.Z[i];
  const parA = sumA % 2 === 0 ? "even" : "ODD";
  const parB = sumB % 2 === 0 ? "even" : "ODD";
  el.innerHTML = `|A|=${nA} (sumZ=${sumA}, ${parA})  |B|=${nB} (sumZ=${sumB}, ${parB})`;
  el.style.color = (parA === "even" && parB === "even") ? "#7f7" : "#f77";
}

function applyStyle(view, atoms, showLabels) {
  const A = new Set(DATA.frag_A), B = new Set(DATA.frag_B);
  atoms.forEach((a, i) => {
    const c = A.has(i) ? "#e64" : (B.has(i) ? "#4ae" : "#888");
    view.setStyle({serial: a.serial}, {
      stick: {radius: 0.13, color: c},
      sphere: {scale: 0.32, color: c},
    });
    if (showLabels) {
      view.addLabel(String(i), {
        position: {x: a.x, y: a.y, z: a.z},
        fontSize: 11, fontColor: "#fff", backgroundOpacity: 0,
        inFront: true, showBackground: false,
      });
    }
  });
}

function loadPanel(divId, xyz, isR) {
  const el = document.getElementById(divId);
  el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor: "black"});
  v.addModel(xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  const showLabels = document.getElementById("labels").checked;
  applyStyle(v, atoms, showLabels);
  // click handler
  atoms.forEach((a) => {
    v.setClickable({serial: a.serial}, true, (clicked) => {
      const idx = atoms.findIndex(x => x.serial === clicked.serial);
      if (idx < 0) return;
      toggle(idx);
    });
  });
  v.zoomTo();
  v.render();
  if (isR) { vR = v; modelR = atoms; } else { vTS = v; modelTS = atoms; }
}

function renderAll() {
  loadPanel("viewR", DATA.R_xyz, true);
  loadPanel("viewTS", DATA.TS_xyz, false);
  updateCounter();
}

function reloadUI() {
  document.getElementById("msg").textContent = DATA.err ? "orig err: " + DATA.err : "";
  document.getElementById("msg").className = DATA.err ? "err" : "info";
  renderAll();
}

function toggle(i) {
  const A = new Set(DATA.frag_A), B = new Set(DATA.frag_B);
  if (A.has(i)) { A.delete(i); B.add(i); }
  else if (B.has(i)) { B.delete(i); A.add(i); }
  else { A.add(i); }
  DATA.frag_A = Array.from(A).sort((a,b)=>a-b);
  DATA.frag_B = Array.from(B).sort((a,b)=>a-b);
  renderAll();
}

function swap() {
  [DATA.frag_A, DATA.frag_B] = [DATA.frag_B, DATA.frag_A];
  renderAll();
}

async function save() {
  const r = await fetch("/api/save", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({frag_A: DATA.frag_A, frag_B: DATA.frag_B}),
  });
  const j = await r.json();
  const msg = document.getElementById("msg");
  if (j.ok) {
    msg.textContent = "SAVED. eda.inp regenerated at " + j.inp_path;
    msg.className = "ok";
  } else {
    msg.textContent = "ERROR: " + (j.error || "unknown");
    msg.className = "err";
  }
}

function reload() { fetchData(); }
window.addEventListener("load", fetchData);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML.replace("__RID__", RID), mimetype="text/html")


@app.route("/api/rxn")
def api_rxn():
    manual = _read_manual()
    p = manual.get(RID, {})
    R_xyz = (RAW / RID / "R.xyz").read_text()
    TS_xyz = (RAW / RID / "TS.xyz").read_text()
    ts_at = ase.io.read(str(RAW / RID / "TS.xyz"))
    Z = [int(z) for z in ts_at.get_atomic_numbers()]
    A = p.get("frag_A_indices", [])
    B = p.get("frag_B_indices", [])
    err = ""
    out = ORCA_ROOT / RID / "eda.out"
    if out.exists():
        for line in out.read_text(errors="ignore").splitlines():
            if "Error" in line and ("multiplicity" in line or "electrons" in line):
                err = line.strip(); break
    resp = jsonify({
        "rid": RID, "family": FAMILY,
        "R_xyz": R_xyz, "TS_xyz": TS_xyz,
        "frag_A": A, "frag_B": B,
        "Z": Z,
        "err": err,
    })
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/save", methods=["POST"])
def api_save():
    body = request.get_json(force=True)
    A = list(body.get("frag_A") or [])
    B = list(body.get("frag_B") or [])
    with _lock:
        try:
            inp = write_orca_input(RID, FAMILY, A, B)
            manual = _read_manual()
            entry = manual.get(RID, {})
            entry["frag_A_indices"] = sorted(A)
            entry["frag_B_indices"] = sorted(B)
            entry["reviewed"] = True
            entry["method"] = "manual_fix_v8"
            manual[RID] = entry
            _write_manual(manual)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True, "inp_path": str(inp)})


if __name__ == "__main__":
    print(f"fix single-rxn app: rid={RID} family={FAMILY} port={PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
