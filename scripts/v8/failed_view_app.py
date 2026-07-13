"""Flask app: lists ORCA EDA failed rxns and shows them in a 3-panel viewer
with per-partition editing.

Panels (independent — only rid shared, coordinates are NOT linked):
  [R fragment A]  [R fragment B]  [TS (full)]
- Click in R panel  -> toggles R partition (frag_A_indices_R, frag_B_indices_R)
- Click in TS panel -> toggles TS partition (frag_A_indices, frag_B_indices)

Save button:
  - Overwrites eda.inp using the (new) TS partition
  - Overwrites manual_partitions.json with both TS and R partitions (independent fields)
  - Deletes the failed eda.out so ORCA re-runs on next submit
"""
from __future__ import annotations
import json, os, threading
from pathlib import Path
import ase.io
from ase.data import chemical_symbols
from flask import Flask, jsonify, request, Response

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
MP = V8 / "manual_partitions.json"
PORT = int(os.environ.get("FIX_PORT", "5578"))
_lock = threading.Lock()
app = Flask(__name__)


def _fam(rid):
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def _read():
    return json.loads(MP.read_text())


def _write(d):
    tmp = MP.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, MP)


def write_orca_input(rid, family, A_TS, B_TS):
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A_TS: frag_of[i] = 1
    for i in B_TS: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"TS: unassigned atoms: {[i for i, f in enumerate(frag_of) if f is None]}")
    tc = 0; fA_c = 0; fB_c = 0
    if family in ("qmrxn20_sn2", "qmrxn20_e2"):
        fB_c = -1; tc = -1
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
        f"* xyz {tc} 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines.append("*"); lines.append("")
    d = ORCA_ROOT / rid
    d.mkdir(parents=True, exist_ok=True)
    inp = d / "eda.inp"
    inp.write_text("\n".join(lines))
    (d / "eda.out").unlink(missing_ok=True)
    (d / "eda.err").unlink(missing_ok=True)
    return inp


def find_failed():
    fails = []
    for d in sorted(ORCA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        out = d / "eda.out"
        if not out.exists():
            continue
        try:
            txt = out.read_text(errors="ignore")
        except Exception:
            continue
        if "ORCA TERMINATED NORMALLY" in txt:
            continue
        err = ""
        for line in txt.splitlines():
            if "Error" in line or "ERROR" in line:
                err = line.strip()[:400]; break
        if not err:
            for line in reversed(txt.splitlines()):
                if line.strip():
                    err = "last: " + line.strip()[:200]; break
        fails.append((d.name, err))
    return fails


@app.route("/api/rids")
def api_rids():
    fails = find_failed()
    resp = jsonify({"rids": [{"rid": r, "err": e, "family": _fam(r)} for r, e in fails],
                    "n_total": len(fails)})
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/rxn/<rid>")
def api_rxn(rid):
    m = _read()
    e = m.get(rid, {})
    R_xyz = (RAW / rid / "R.xyz").read_text()
    TS_xyz = (RAW / rid / "TS.xyz").read_text()
    r_at = ase.io.read(str(RAW / rid / "R.xyz"))
    Z_R = [int(z) for z in r_at.get_atomic_numbers()]
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z_TS = [int(z) for z in ts_at.get_atomic_numbers()]
    err = ""
    out = ORCA_ROOT / rid / "eda.out"
    if out.exists():
        for line in out.read_text(errors="ignore").splitlines():
            if "Error" in line or "ERROR" in line:
                err = line.strip()[:400]; break
    resp = jsonify({
        "rid": rid, "family": _fam(rid),
        "R_xyz": R_xyz, "TS_xyz": TS_xyz,
        "Z_R": Z_R, "Z_TS": Z_TS,
        "frag_A_TS": sorted(e.get("frag_A_indices", [])),
        "frag_B_TS": sorted(e.get("frag_B_indices", [])),
        "frag_A_R":  sorted(e.get("frag_A_indices_R", [])),
        "frag_B_R":  sorted(e.get("frag_B_indices_R", [])),
        "err": err,
    })
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/save/<rid>", methods=["POST"])
def api_save(rid):
    """Only TS partition is edited by this app (EDA failures).
    R partition remains untouched (strain SP hasn't started yet)."""
    body = request.get_json(force=True)
    A_TS = sorted(list(body.get("frag_A_TS") or []))
    B_TS = sorted(list(body.get("frag_B_TS") or []))
    fam = _fam(rid)
    with _lock:
        try:
            m = _read()
            e = m.get(rid, {})
            e["frag_A_indices"] = A_TS
            e["frag_B_indices"] = B_TS
            e["needs_TS_review"] = False
            m[rid] = e
            _write(m)
            inp = write_orca_input(rid, fam, A_TS, B_TS)
        except Exception as ex:
            return jsonify({"ok": False, "error": str(ex)})
    return jsonify({"ok": True, "inp_path": str(inp)})


HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>failed EDA rxns — edit</title>
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
.err  { color: #f77; font-family: monospace; }
.ok   { color: #7f7; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; height: calc(100vh - 190px); }
.panel { background: #000; border: 1px solid #333; position: relative; }
.pt { position: absolute; top: 4px; left: 8px; font-size: 12px; z-index: 10; background: rgba(0,0,0,0.6); padding: 2px 6px; }
.pt-RA { color: #f88; } .pt-RB { color: #8bf; } .pt-TS { color: #fd0; }
.hint { position: absolute; bottom: 4px; left: 8px; font-size: 11px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 3px 6px; }
#msg { margin-top: 4px; font-size: 13px; }
#errmsg { margin-top: 4px; font-size: 11px; color: #f77; font-family: monospace; }
</style>
</head>
<body>
<h1>Failed ORCA EDA — edit TS only (R = view only, strain SP not started yet)</h1>
<div class="controls">
  <button onclick="prev()">« prev</button>
  <select id="picker" onchange="pick(this.value)"></select>
  <button onclick="next()">next »</button>
  <button class="primary" onclick="save()">💾 save + regen eda.inp</button>
  <button onclick="refresh()">🔄 refresh failed list</button>
  <label class="info"><input type="checkbox" id="labels" onchange="renderAll()" checked> labels</label>
  <span id="counter" class="info"></span>
</div>
<div id="msg" class="info"></div>
<div id="errmsg"></div>
<div class="panels">
  <div class="panel">
    <div class="pt">R (reactants) — view only, colored by TS partition for reference</div>
    <div id="viewR" style="width:100%; height:100%"></div>
  </div>
  <div class="panel">
    <div class="pt"><span class="pt-TS">■ TS (full)</span> — click atom to toggle TS partition (goes into eda.inp)</div>
    <div id="viewTS" style="width:100%; height:100%"></div>
  </div>
</div>

<script>
let ALL = [];
let idx = 0;
let DATA = null;
const picker = document.getElementById("picker");
let vR=null, vTS=null, atomsR=null, atomsTS=null, currentRid=null;

async function refresh() {
  const r = await fetch("/api/rids?ts=" + Date.now());
  const j = await r.json();
  ALL = j.rids;
  picker.innerHTML = "";
  if (ALL.length === 0) {
    document.getElementById("counter").textContent = "no failures";
    document.getElementById("msg").textContent = "";
    document.getElementById("errmsg").textContent = "";
    return;
  }
  ALL.forEach((it, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `[${i+1}/${ALL.length}] ${it.rid}`;
    picker.appendChild(o);
  });
  document.getElementById("counter").textContent = `${ALL.length} failed`;
  if (idx >= ALL.length) idx = 0;
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

function sumZ(indices, Z) { let s = 0; for (const i of indices) s += Z[i]; return s; }

function updateMsg() {
  const nAT = DATA.frag_A_TS.length, nBT = DATA.frag_B_TS.length;
  const sAT = sumZ(DATA.frag_A_TS, DATA.Z_TS), sBT = sumZ(DATA.frag_B_TS, DATA.Z_TS);
  const parTA = sAT%2===0?"even":"ODD", parTB = sBT%2===0?"even":"ODD";
  document.getElementById("msg").innerHTML =
    `<b>${DATA.rid}</b> (${DATA.family})   ` +
    `TS: |A|=${nAT} sumZ=${sAT}(${parTA})  |B|=${nBT} sumZ=${sBT}(${parTB})` +
    ((parTA!=="even"||parTB!=="even") ? ` <span class="err">TS odd-electron!</span>` : "");
  document.getElementById("errmsg").textContent = DATA.err || "";
}

function styleR(v, atoms) {
  // R panel is view-only, colored by TS partition just for reference
  const A = new Set(DATA.frag_A_TS), B = new Set(DATA.frag_B_TS);
  const showLabels = document.getElementById("labels").checked;
  v.removeAllLabels();
  atoms.forEach((a, i) => {
    const color = A.has(i) ? "#e64" : (B.has(i) ? "#4ae" : "#888");
    v.setStyle({serial: a.serial}, {
      stick: {radius: 0.13, color: color},
      sphere: {scale: 0.30, color: color},
    });
    if (showLabels) {
      v.addLabel(String(i), {position:{x:a.x,y:a.y,z:a.z}, fontSize:11, fontColor:"#fff", backgroundOpacity:0, inFront:true, showBackground:false});
    }
  });
}

function styleTS(v, atoms) {
  const A = new Set(DATA.frag_A_TS), B = new Set(DATA.frag_B_TS);
  const showLabels = document.getElementById("labels").checked;
  v.removeAllLabels();
  atoms.forEach((a, i) => {
    const inA = A.has(i), inB = B.has(i);
    let color, stickR, sphS;
    if (inA) { color = "#e64"; stickR = 0.13; sphS = 0.30; }
    else if (inB) { color = "#4ae"; stickR = 0.13; sphS = 0.30; }
    else { color = "#fa0"; stickR = 0.12; sphS = 0.24; }
    v.setStyle({serial: a.serial}, {stick:{radius:stickR, color:color}, sphere:{scale:sphS, color:color}});
    if (showLabels) {
      v.addLabel(String(i), {position:{x:a.x,y:a.y,z:a.z}, fontSize:11, fontColor:"#fff", backgroundOpacity:0, inFront:true, showBackground:false});
    }
  });
}

function buildR(divId) {
  const el = document.getElementById(divId); el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor:"black"});
  v.addModel(DATA.R_xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  // no click handler - R panel is view only
  styleR(v, atoms);
  return {v, atoms};
}

function buildTS(divId) {
  const el = document.getElementById(divId); el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor:"black"});
  v.addModel(DATA.TS_xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  atoms.forEach((a) => {
    v.setClickable({serial: a.serial}, true, (clk) => {
      const i = atoms.findIndex(x => x.serial === clk.serial);
      if (i < 0) return;
      toggleTS(i);
    });
  });
  styleTS(v, atoms);
  return {v, atoms};
}

function renderAll() {
  const ridChanged = currentRid !== DATA.rid;
  if (ridChanged || vR === null) {
    const r = buildR("viewR"); vR = r.v; atomsR = r.atoms;
    vR.zoomTo(); vR.render();
    const t = buildTS("viewTS"); vTS = t.v; atomsTS = t.atoms;
    vTS.zoomTo(); vTS.render();
    currentRid = DATA.rid;
  } else {
    const vwR = vR.getView(), vwT = vTS.getView();
    styleR(vR, atomsR);
    styleTS(vTS, atomsTS);
    vR.setView(vwR); vTS.setView(vwT);
    vR.render(); vTS.render();
  }
}

function toggleTS(i) {
  const A = new Set(DATA.frag_A_TS), B = new Set(DATA.frag_B_TS);
  if (A.has(i)) { A.delete(i); B.add(i); }
  else if (B.has(i)) { B.delete(i); A.add(i); }
  else { A.add(i); }
  DATA.frag_A_TS = Array.from(A).sort((a,b)=>a-b);
  DATA.frag_B_TS = Array.from(B).sort((a,b)=>a-b);
  renderAll(); updateMsg();
}

async function save() {
  const r = await fetch(`/api/save/${DATA.rid}`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({frag_A_TS: DATA.frag_A_TS, frag_B_TS: DATA.frag_B_TS}),
  });
  const j = await r.json();
  const el = document.getElementById("msg");
  if (j.ok) {
    el.innerHTML += ` <span class="ok">SAVED (eda.inp regen, eda.out deleted → next run will retry)</span>`;
  } else {
    el.innerHTML += ` <span class="err">SAVE ERR: ${j.error}</span>`;
  }
}

function pick(i) { idx = +i; fetchRxn(); }
function next() { if (ALL.length) { idx = (idx + 1) % ALL.length; fetchRxn(); } }
function prev() { if (ALL.length) { idx = (idx - 1 + ALL.length) % ALL.length; fetchRxn(); } }
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight" || e.key === "j") next();
  else if (e.key === "ArrowLeft" || e.key === "k") prev();
  else if (e.key === "s") save();
});
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    print(f"Failed-viewer + edit app on port {PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
