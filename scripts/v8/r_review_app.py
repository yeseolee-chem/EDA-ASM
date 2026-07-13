"""R-geometry fragment review app.

Two-panel viewer for the R (reactant) geometry:
  Left  panel = R.xyz with fragment A prominently rendered (red), fragment B faded
  Right panel = R.xyz with fragment B prominently rendered (blue), fragment A faded

Click ANY atom in either panel to toggle A ↔ B (works on faded atoms too).
Save writes manual_partitions.json and regenerates eda.inp.

Serves on port FIX_PORT (default 5578).
"""
from __future__ import annotations
import json, os, threading
from pathlib import Path

import ase.io
import pandas as pd
from ase.data import chemical_symbols
from flask import Flask, jsonify, request, Response

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
MANUAL_JSON = V8 / "manual_partitions.json"
COHORT_PQ = V8 / "cohort_v8.parquet"

PORT = int(os.environ.get("FIX_PORT", "5578"))
_lock = threading.Lock()
app = Flask(__name__)


def _fam(rid: str) -> str:
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


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
    for i in frag_A: frag_of[i] = 1
    for i in frag_B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"unassigned atoms: {[i for i, f in enumerate(frag_of) if f is None]}")

    total_charge = 0; fA_c = 0; fB_c = 0
    if family in ("qmrxn20_sn2", "qmrxn20_e2"):
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
    lines.append("*"); lines.append("")

    out_dir = ORCA_ROOT / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    inp = out_dir / "eda.inp"
    inp.write_text("\n".join(lines))
    (out_dir / "eda.out").unlink(missing_ok=True)
    (out_dir / "eda.err").unlink(missing_ok=True)
    return inp


COHORT = pd.read_parquet(COHORT_PQ)
RIDS = COHORT["reaction_id"].tolist()


@app.route("/api/rids")
def api_rids():
    manual = _read_manual()
    out = []
    for rid in RIDS:
        m = manual.get(rid, {})
        out.append({
            "rid": rid,
            "family": _fam(rid),
            "reviewed_R": bool(m.get("R_reviewed", False)),
            "reviewed_TS": bool(m.get("reviewed", False)),
        })
    n_R = sum(1 for x in out if x["reviewed_R"])
    resp = jsonify({"rids": out, "n_total": len(out), "n_R_reviewed": n_R})
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/rxn/<rid>")
def api_rxn(rid):
    manual = _read_manual()
    p = manual.get(rid, {})
    R_xyz = (RAW / rid / "R.xyz").read_text()
    r_at = ase.io.read(str(RAW / rid / "R.xyz"))
    Z = [int(z) for z in r_at.get_atomic_numbers()]
    A = p.get("frag_A_indices", [])
    B = p.get("frag_B_indices", [])
    resp = jsonify({
        "rid": rid, "family": _fam(rid),
        "R_xyz": R_xyz, "Z": Z,
        "frag_A": A, "frag_B": B,
        "R_reviewed": bool(p.get("R_reviewed", False)),
        "TS_reviewed": bool(p.get("reviewed", False)),
    })
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/save/<rid>", methods=["POST"])
def api_save(rid):
    body = request.get_json(force=True)
    A = sorted(list(body.get("frag_A") or []))
    B = sorted(list(body.get("frag_B") or []))
    mark_reviewed = bool(body.get("R_reviewed", False))
    fam = _fam(rid)
    with _lock:
        try:
            inp = write_orca_input(rid, fam, A, B)
            manual = _read_manual()
            entry = manual.get(rid, {})
            entry["frag_A_indices"] = A
            entry["frag_B_indices"] = B
            entry["R_reviewed"] = mark_reviewed
            entry["reviewed"] = True
            if mark_reviewed:
                entry["method"] = "R_reviewed_v8"
            manual[rid] = entry
            _write_manual(manual)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True, "inp_path": str(inp)})


HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>R fragment review</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
body { font-family: -apple-system, sans-serif; margin: 0; padding: 12px; background: #111; color: #ddd; }
h1 { font-size: 15px; margin: 0 0 8px 0; color: #fff; }
.controls { margin-bottom: 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
button, select, input { background: #222; color: #ddd; border: 1px solid #444; padding: 5px 10px; font-size: 13px; }
button { cursor: pointer; }
button:hover { background: #333; }
button.primary { background: #2a5; border-color: #4a8; color: white; font-weight: bold; }
button.primary:hover { background: #3b6; }
.info { font-size: 12px; color: #aaa; }
.warn { color: #fc6; }
.ok { color: #7f7; }
.err { color: #f77; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; height: calc(100vh - 160px); }
.panel { background: #000; border: 1px solid #333; position: relative; }
.panel-title { position: absolute; top: 4px; left: 8px; font-size: 12px; z-index: 10; background: rgba(0,0,0,0.6); padding: 2px 6px; }
.pt-A { color: #f88; }
.pt-B { color: #8bf; }
.hint { position: absolute; bottom: 4px; left: 8px; font-size: 11px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 3px 6px; }
#msg { margin-top: 6px; font-size: 12px; }
</style>
</head>
<body>
<h1>R fragment review — click any atom (in either panel) to toggle A ↔ B</h1>
<div class="controls">
  <button onclick="prev()">« prev</button>
  <select id="picker" onchange="pick(this.value)"></select>
  <button onclick="next()">next »</button>
  <button class="primary" onclick="saveMark()">💾 save + mark R-reviewed</button>
  <button onclick="save()">save only</button>
  <button onclick="swap()">swap A↔B</button>
  <label class="info"><input type="checkbox" id="labels" onchange="renderAll()" checked> labels</label>
  <span id="counter" class="info"></span>
</div>
<div id="msg" class="info"></div>
<div class="panels">
  <div class="panel">
    <div class="panel-title"><span class="pt-A">■ Fragment A</span> — click atom to move to B (grey = B skeleton for context)</div>
    <div id="viewA" style="width:100%; height:100%"></div>
  </div>
  <div class="panel">
    <div class="panel-title"><span class="pt-B">■ Fragment B</span> — click atom to move to A (grey = A skeleton for context)</div>
    <div id="viewB" style="width:100%; height:100%"></div>
  </div>
</div>

<script>
let ALL = [];
let idx = 0;
let DATA = null;
const picker = document.getElementById("picker");
// preserved 3Dmol camera state per panel (survives style updates)
let viewA_ = null, viewB_ = null;
let atomsA_ = null, atomsB_ = null;
let savedViewA = null, savedViewB = null;
let currentRid = null;

async function fetchList() {
  const r = await fetch("/api/rids?ts=" + Date.now());
  const j = await r.json();
  ALL = j.rids;
  picker.innerHTML = "";
  ALL.forEach((it, i) => {
    const o = document.createElement("option");
    o.value = i;
    const rmark = it.reviewed_R ? "R✔" : "  ";
    o.textContent = `[${i+1}/${ALL.length}] ${rmark} ${it.rid}`;
    picker.appendChild(o);
  });
  document.getElementById("counter").innerHTML =
    `${j.n_R_reviewed}/${j.n_total} R-reviewed`;
  await fetchRxn();
}

async function fetchRxn() {
  const rid = ALL[idx].rid;
  const r = await fetch(`/api/rxn/${rid}?ts=${Date.now()}`);
  DATA = await r.json();
  picker.value = idx;
  renderAll();
  updateMsg();
}

function updateMsg() {
  const nA = DATA.frag_A.length, nB = DATA.frag_B.length;
  const nTot = DATA.Z.length;
  const unassigned = nTot - nA - nB;
  let sumA = 0, sumB = 0;
  for (const i of DATA.frag_A) sumA += DATA.Z[i];
  for (const i of DATA.frag_B) sumB += DATA.Z[i];
  const parA = sumA % 2 === 0 ? "even" : "ODD";
  const parB = sumB % 2 === 0 ? "even" : "ODD";
  let msg = `<b>${DATA.rid}</b> (${DATA.family})   |A|=${nA} sumZ=${sumA}(${parA})   |B|=${nB} sumZ=${sumB}(${parB})   n=${nTot}`;
  if (unassigned > 0) msg += `   <span class="warn">unassigned=${unassigned}</span>`;
  if (parA !== "even" || parB !== "even") msg += ` <span class="err">— odd-electron risk!</span>`;
  msg += DATA.R_reviewed ? ` <span class="ok">R-reviewed</span>` : "";
  document.getElementById("msg").innerHTML = msg;
}

function stylesFor(v, atoms, focusFrag) {
  const A = new Set(DATA.frag_A), B = new Set(DATA.frag_B);
  const showLabels = document.getElementById("labels").checked;
  const focusSet = focusFrag === "A" ? A : B;
  const otherSet = focusFrag === "A" ? B : A;
  const focusColor = focusFrag === "A" ? "#e64" : "#4ae";

  v.removeAllLabels();

  atoms.forEach((a, i) => {
    const inFocus = focusSet.has(i);
    const inOther = otherSet.has(i);
    const unassigned = !inFocus && !inOther;

    let color, stickR, sphereScale;
    if (inFocus) {
      color = focusColor; stickR = 0.14; sphereScale = 0.34;
    } else if (inOther) {
      color = "#666"; stickR = 0.06; sphereScale = 0.0;
    } else {
      color = "#fa0"; stickR = 0.12; sphereScale = 0.24;
    }
    const style = {stick: {radius: stickR, color: color}};
    if (sphereScale > 0) style.sphere = {scale: sphereScale, color: color};
    v.setStyle({serial: a.serial}, style);

    if (showLabels && (inFocus || unassigned)) {
      v.addLabel(String(i), {
        position: {x: a.x, y: a.y, z: a.z},
        fontSize: 11, fontColor: "#fff", backgroundOpacity: 0,
        inFront: true, showBackground: false,
      });
    }
  });
}

function buildViewer(divId, focusFrag) {
  const el = document.getElementById(divId);
  el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor: "black"});
  v.addModel(DATA.R_xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  atoms.forEach((a) => {
    v.setClickable({serial: a.serial}, true, (clicked) => {
      const idx2 = atoms.findIndex(x => x.serial === clicked.serial);
      if (idx2 < 0) return;
      toggle(idx2);
    });
  });
  stylesFor(v, atoms, focusFrag);
  return {v, atoms};
}

function renderAll() {
  // FULL rebuild only when the rxn changes; otherwise just update styles
  const ridChanged = currentRid !== DATA.rid;
  if (ridChanged || viewA_ === null) {
    const a = buildViewer("viewA", "A");
    viewA_ = a.v; atomsA_ = a.atoms;
    viewA_.zoomTo(); viewA_.render();
    savedViewA = null;
    const b = buildViewer("viewB", "B");
    viewB_ = b.v; atomsB_ = b.atoms;
    viewB_.zoomTo(); viewB_.render();
    savedViewB = null;
    currentRid = DATA.rid;
  } else {
    // preserve camera
    savedViewA = viewA_.getView();
    savedViewB = viewB_.getView();
    stylesFor(viewA_, atomsA_, "A");
    stylesFor(viewB_, atomsB_, "B");
    viewA_.setView(savedViewA);
    viewB_.setView(savedViewB);
    viewA_.render();
    viewB_.render();
  }
}

function toggle(i) {
  const A = new Set(DATA.frag_A), B = new Set(DATA.frag_B);
  if (A.has(i)) { A.delete(i); B.add(i); }
  else if (B.has(i)) { B.delete(i); A.add(i); }
  else { A.add(i); }
  DATA.frag_A = Array.from(A).sort((a,b)=>a-b);
  DATA.frag_B = Array.from(B).sort((a,b)=>a-b);
  renderAll();
  updateMsg();
}

function swap() {
  [DATA.frag_A, DATA.frag_B] = [DATA.frag_B, DATA.frag_A];
  renderAll(); updateMsg();
}

async function save() { await _save(false); }
async function saveMark() { await _save(true); }

async function _save(markReviewed) {
  const r = await fetch(`/api/save/${DATA.rid}`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({frag_A: DATA.frag_A, frag_B: DATA.frag_B, R_reviewed: markReviewed}),
  });
  const j = await r.json();
  const el = document.getElementById("msg");
  if (j.ok) {
    if (markReviewed) {
      // update local reviewed flag, refresh list
      ALL[idx].reviewed_R = true;
      picker.options[idx].textContent = `[${idx+1}/${ALL.length}] R✔ ${DATA.rid}`;
      // auto-advance
      if (idx + 1 < ALL.length) { idx++; await fetchRxn(); return; }
    }
    updateMsg();
    el.innerHTML += ` <span class="ok">SAVED</span>`;
  } else {
    el.innerHTML += ` <span class="err">SAVE ERR: ${j.error}</span>`;
  }
}

function pick(i) { idx = +i; fetchRxn(); }
function next() { idx = (idx + 1) % ALL.length; fetchRxn(); }
function prev() { idx = (idx - 1 + ALL.length) % ALL.length; fetchRxn(); }

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight" || e.key === "j") next();
  else if (e.key === "ArrowLeft" || e.key === "k") prev();
  else if (e.key === "s") saveMark();
});
fetchList();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    print(f"R review app, port={PORT}, cohort={len(RIDS)}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
