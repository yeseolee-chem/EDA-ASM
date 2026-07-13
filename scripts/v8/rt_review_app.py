"""3-panel R+TS review app for the 258 rxns whose TS and R partitions
diverged (or whose TS partition is unrecoverable).

Panels (independent — only rid is shared):
  [Fragment A (R)] [Fragment B (R)] [TS (full, colored by TS partition)]

Data model in manual_partitions.json:
  frag_A_indices,   frag_B_indices   -> TS partition (used for eda.inp)
  frag_A_indices_R, frag_B_indices_R -> R partition  (used for strain SP)

Click on R panel -> toggles R partition (independent of TS).
Click on TS panel -> toggles TS partition (independent of R).
Save button persists both.

Filter: only rxns listed in outputs/v8_review/needs_review.txt (258).
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
MANUAL_JSON = V8 / "manual_partitions.json"
NEEDS = V8 / "needs_review.txt"
PORT = int(os.environ.get("FIX_PORT", "5578"))
_lock = threading.Lock()
app = Flask(__name__)


def _fam(rid: str) -> str:
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def _read():
    return json.loads(MANUAL_JSON.read_text())


def _write(d):
    tmp = MANUAL_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, MANUAL_JSON)


def write_orca_input(rid: str, family: str, frag_A_TS, frag_B_TS) -> Path:
    """Rewrite eda.inp using TS partition. Deletes eda.out."""
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in frag_A_TS: frag_of[i] = 1
    for i in frag_B_TS: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"TS: unassigned atoms: {[i for i, f in enumerate(frag_of) if f is None]}")
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
    d = ORCA_ROOT / rid
    d.mkdir(parents=True, exist_ok=True)
    inp = d / "eda.inp"
    inp.write_text("\n".join(lines))
    (d / "eda.out").unlink(missing_ok=True)
    (d / "eda.err").unlink(missing_ok=True)
    return inp


RIDS = [l.strip() for l in NEEDS.read_text().splitlines() if l.strip()]


@app.route("/api/rids")
def api_rids():
    m = _read()
    out = []
    for rid in RIDS:
        e = m.get(rid, {})
        out.append({
            "rid": rid,
            "family": _fam(rid),
            "needs_TS_review": bool(e.get("needs_TS_review", True)),
            "TS_recoverable": bool(e.get("TS_recoverable", False)),
            "TS_confirmed": bool(e.get("TS_confirmed", False)),
        })
    n_conf = sum(1 for x in out if x["TS_confirmed"])
    resp = jsonify({"rids": out, "n_total": len(out), "n_confirmed": n_conf})
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
    resp = jsonify({
        "rid": rid, "family": _fam(rid),
        "R_xyz": R_xyz, "TS_xyz": TS_xyz,
        "Z_R": Z_R, "Z_TS": Z_TS,
        "frag_A_TS": sorted(e.get("frag_A_indices", [])),
        "frag_B_TS": sorted(e.get("frag_B_indices", [])),
        "frag_A_R":  sorted(e.get("frag_A_indices_R", [])),
        "frag_B_R":  sorted(e.get("frag_B_indices_R", [])),
        "TS_recoverable": bool(e.get("TS_recoverable", False)),
        "TS_confirmed":   bool(e.get("TS_confirmed", False)),
        "needs_TS_review": bool(e.get("needs_TS_review", True)),
    })
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/save/<rid>", methods=["POST"])
def api_save(rid):
    body = request.get_json(force=True)
    A_TS = sorted(list(body.get("frag_A_TS") or []))
    B_TS = sorted(list(body.get("frag_B_TS") or []))
    A_R  = sorted(list(body.get("frag_A_R")  or []))
    B_R  = sorted(list(body.get("frag_B_R")  or []))
    confirm = bool(body.get("confirm", False))
    fam = _fam(rid)
    with _lock:
        try:
            m = _read()
            e = m.get(rid, {})
            e["frag_A_indices"] = A_TS
            e["frag_B_indices"] = B_TS
            e["frag_A_indices_R"] = A_R
            e["frag_B_indices_R"] = B_R
            if confirm:
                e["TS_confirmed"] = True
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
<title>R+TS review (258 needs decision)</title>
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
.ok   { color: #7f7; }
.err  { color: #f77; }
.panels { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; height: calc(100vh - 170px); }
.panel { background: #000; border: 1px solid #333; position: relative; }
.pt { position: absolute; top: 4px; left: 8px; font-size: 12px; z-index: 10; background: rgba(0,0,0,0.6); padding: 2px 6px; }
.pt-RA { color: #f88; } .pt-RB { color: #8bf; } .pt-TS { color: #fd0; }
.hint { position: absolute; bottom: 4px; left: 8px; font-size: 11px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 3px 6px; }
#msg { margin-top: 6px; font-size: 12px; }
.badge { padding: 1px 5px; border-radius: 3px; font-size: 11px; margin-left: 4px; }
.b-yellow { background: #665; color: #fd8; }
.b-red    { background: #522; color: #f77; }
</style>
</head>
<body>
<h1>R + TS partition review — 258 rxns needing decision (independent per panel, only rid is shared)</h1>
<div class="controls">
  <button onclick="prev()">« prev</button>
  <select id="picker" onchange="pick(this.value)"></select>
  <button onclick="next()">next »</button>
  <button class="primary" onclick="saveConfirm()">💾 save + confirm TS</button>
  <button onclick="save()">save only</button>
  <label class="info"><input type="checkbox" id="labels" onchange="renderAll()" checked> labels</label>
  <span id="counter" class="info"></span>
</div>
<div id="msg" class="info"></div>
<div class="panels">
  <div class="panel">
    <div class="pt"><span class="pt-RA">■ R fragment A</span> — click atom to toggle R partition</div>
    <div class="hint">R coord, A prominent, B grey skeleton</div>
    <div id="viewRA" style="width:100%; height:100%"></div>
  </div>
  <div class="panel">
    <div class="pt"><span class="pt-RB">■ R fragment B</span> — click atom to toggle R partition</div>
    <div class="hint">R coord, B prominent, A grey skeleton</div>
    <div id="viewRB" style="width:100%; height:100%"></div>
  </div>
  <div class="panel">
    <div class="pt"><span class="pt-TS">■ TS (full)</span> — click atom to toggle TS partition (used for EDA)</div>
    <div class="hint">TS coord, A=red, B=blue</div>
    <div id="viewTS" style="width:100%; height:100%"></div>
  </div>
</div>

<script>
let ALL = [];
let idx = 0;
let DATA = null;
const picker = document.getElementById("picker");
let vRA=null, vRB=null, vTS=null, atomsR=null, atomsTS=null, currentRid=null;

async function fetchList() {
  const r = await fetch("/api/rids?ts=" + Date.now());
  const j = await r.json();
  ALL = j.rids;
  picker.innerHTML = "";
  ALL.forEach((it, i) => {
    const o = document.createElement("option");
    o.value = i;
    const badge = it.TS_confirmed ? "✔" : (it.TS_recoverable ? " " : "?");
    o.textContent = `[${i+1}/${ALL.length}] ${badge} ${it.rid}`;
    picker.appendChild(o);
  });
  document.getElementById("counter").innerHTML = `${j.n_confirmed}/${j.n_total} confirmed`;
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
  const nAR = DATA.frag_A_R.length, nBR = DATA.frag_B_R.length;
  const nAT = DATA.frag_A_TS.length, nBT = DATA.frag_B_TS.length;
  const sAR = sumZ(DATA.frag_A_R, DATA.Z_R),  sBR = sumZ(DATA.frag_B_R, DATA.Z_R);
  const sAT = sumZ(DATA.frag_A_TS, DATA.Z_TS), sBT = sumZ(DATA.frag_B_TS, DATA.Z_TS);
  const parRA = sAR%2===0 ? "even" : "ODD", parRB = sBR%2===0 ? "even" : "ODD";
  const parTA = sAT%2===0 ? "even" : "ODD", parTB = sBT%2===0 ? "even" : "ODD";
  let msg = `<b>${DATA.rid}</b> (${DATA.family})`;
  if (!DATA.TS_recoverable) msg += ` <span class="badge b-red">no TS history</span>`;
  else if (DATA.needs_TS_review) msg += ` <span class="badge b-yellow">TS≠R</span>`;
  msg += ` <br>`;
  msg += `TS: |A|=${nAT} sumZ=${sAT}(${parTA})  |B|=${nBT} sumZ=${sBT}(${parTB}) &nbsp;&nbsp;`;
  msg += `R: |A|=${nAR} sumZ=${sAR}(${parRA})  |B|=${nBR} sumZ=${sBR}(${parRB})`;
  if (parRA!=="even"||parRB!=="even"||parTA!=="even"||parTB!=="even") msg += ` <span class="err">odd-electron risk!</span>`;
  document.getElementById("msg").innerHTML = msg;
}

function styleR(v, atoms, focusFrag) {
  const A = new Set(DATA.frag_A_R), B = new Set(DATA.frag_B_R);
  const showLabels = document.getElementById("labels").checked;
  const focusSet = focusFrag === "A" ? A : B;
  const otherSet = focusFrag === "A" ? B : A;
  const focusColor = focusFrag === "A" ? "#e64" : "#4ae";
  v.removeAllLabels();
  atoms.forEach((a, i) => {
    const inFocus = focusSet.has(i);
    const inOther = otherSet.has(i);
    const unass = !inFocus && !inOther;
    let color, stickR, sphS;
    if (inFocus) { color = focusColor; stickR = 0.14; sphS = 0.34; }
    else if (inOther) { color = "#666"; stickR = 0.06; sphS = 0.0; }
    else { color = "#fa0"; stickR = 0.12; sphS = 0.24; }
    const st = {stick: {radius: stickR, color: color}};
    if (sphS > 0) st.sphere = {scale: sphS, color: color};
    v.setStyle({serial: a.serial}, st);
    if (showLabels && (inFocus || unass)) {
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

function buildR(divId, focusFrag) {
  const el = document.getElementById(divId); el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor:"black"});
  v.addModel(DATA.R_xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  atoms.forEach((a) => {
    v.setClickable({serial: a.serial}, true, (clk) => {
      const i = atoms.findIndex(x => x.serial === clk.serial);
      if (i < 0) return;
      toggleR(i);
    });
  });
  styleR(v, atoms, focusFrag);
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
  if (ridChanged || vRA === null) {
    const a = buildR("viewRA", "A"); vRA = a.v; atomsR = a.atoms;
    vRA.zoomTo(); vRA.render();
    const b = buildR("viewRB", "B"); vRB = b.v;
    vRB.zoomTo(); vRB.render();
    const t = buildTS("viewTS"); vTS = t.v; atomsTS = t.atoms;
    vTS.zoomTo(); vTS.render();
    currentRid = DATA.rid;
  } else {
    const vwRA = vRA.getView(), vwRB = vRB.getView(), vwTS = vTS.getView();
    styleR(vRA, atomsR, "A");
    styleR(vRB, atomsR, "B");
    styleTS(vTS, atomsTS);
    vRA.setView(vwRA); vRB.setView(vwRB); vTS.setView(vwTS);
    vRA.render(); vRB.render(); vTS.render();
  }
}

function toggleR(i) {
  const A = new Set(DATA.frag_A_R), B = new Set(DATA.frag_B_R);
  if (A.has(i)) { A.delete(i); B.add(i); }
  else if (B.has(i)) { B.delete(i); A.add(i); }
  else { A.add(i); }
  DATA.frag_A_R = Array.from(A).sort((a,b)=>a-b);
  DATA.frag_B_R = Array.from(B).sort((a,b)=>a-b);
  renderAll(); updateMsg();
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
async function save() { await _save(false); }
async function saveConfirm() { await _save(true); }
async function _save(confirm) {
  const r = await fetch(`/api/save/${DATA.rid}`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      frag_A_TS: DATA.frag_A_TS, frag_B_TS: DATA.frag_B_TS,
      frag_A_R:  DATA.frag_A_R,  frag_B_R:  DATA.frag_B_R,
      confirm: confirm,
    }),
  });
  const j = await r.json();
  const el = document.getElementById("msg");
  if (j.ok) {
    if (confirm) {
      ALL[idx].TS_confirmed = true;
      picker.options[idx].textContent = `[${idx+1}/${ALL.length}] ✔ ${DATA.rid}`;
      if (idx + 1 < ALL.length) { idx++; await fetchRxn(); return; }
    }
    updateMsg();
    el.innerHTML += ` <span class="ok">SAVED</span>`;
  } else {
    el.innerHTML += ` <span class="err">ERR: ${j.error}</span>`;
  }
}
function pick(i) { idx = +i; fetchRxn(); }
function next() { idx = (idx + 1) % ALL.length; fetchRxn(); }
function prev() { idx = (idx - 1 + ALL.length) % ALL.length; fetchRxn(); }
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight" || e.key === "j") next();
  else if (e.key === "ArrowLeft" || e.key === "k") prev();
  else if (e.key === "s") saveConfirm();
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
    print(f"RT review app on port={PORT}  |  filter={len(RIDS)} rxns", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
