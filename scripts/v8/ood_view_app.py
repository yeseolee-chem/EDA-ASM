"""OOD viewer + editor.

Two independent panels (rule: only reaction_id shared):
  R panel  = R.xyz colored by R partition (frag_A_indices_R / frag_B_indices_R)
             Click atom -> toggle R partition -> regen fragA_R.inp / fragB_R.inp,
             delete .out (next strain SP retry recomputes)
  TS panel = TS.xyz colored by TS partition (frag_A_indices / frag_B_indices)
             Click atom -> toggle TS partition -> regen eda.inp, delete eda.out

Right column: 7-channel table + z-score + EDA output snippet.
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
SP_ROOT = V8 / "strain_sp"
MP = V8 / "manual_partitions.json"
LABELS = V8 / "labels/labels_v8_5channel.parquet"
OOD_CSV = V8 / "labels/ood_report_v8.csv"
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


def write_eda_inp(rid, family, A_TS, B_TS):
    """TS partition -> eda.inp. Deletes eda.out."""
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A_TS: frag_of[i] = 1
    for i in B_TS: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"TS unassigned")
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
    (d / "eda.inp").write_text("\n".join(lines))
    (d / "eda.out").unlink(missing_ok=True)
    (d / "eda.err").unlink(missing_ok=True)
    return d / "eda.inp"


def write_strain_sp_inps(rid, family, A_R, B_R):
    """R partition -> fragA_R.inp / fragB_R.inp. Deletes .out for both."""
    r_at = ase.io.read(str(RAW / rid / "R.xyz"))
    Z = r_at.get_atomic_numbers()
    pos = r_at.get_positions()
    fA_c = 0; fB_c = 0
    if family in ("qmrxn20_sn2", "qmrxn20_e2"):
        fB_c = -1
    d = SP_ROOT / rid
    d.mkdir(parents=True, exist_ok=True)
    for frag_idx, charge, name in [(A_R, fA_c, "fragA"), (B_R, fB_c, "fragB")]:
        lines = ["! BLYP D3BJ def2-TZVP NoSym TightSCF",
                 "%maxcore 3500", "", f"* xyz {charge} 1"]
        for i in frag_idx:
            sym = chemical_symbols[int(Z[i])]
            lines.append(f"{sym}   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
        lines += ["*", ""]
        (d / f"{name}_R.inp").write_text("\n".join(lines))
        (d / f"{name}_R.out").unlink(missing_ok=True)
        (d / f"{name}_R.err").unlink(missing_ok=True)
    return d


LABELS_DF = pd.read_parquet(LABELS)
OOD_DF = pd.read_csv(OOD_CSV)
OOD_SET = set(OOD_DF["reaction_id"].tolist())
CHANNELS = ["pauli_kcal","elst_kcal","orb_kcal","disp_kcal","strain_kcal","int_eda_kcal","act_kcal"]
FAM_STATS = {}
for fam, grp in LABELS_DF.groupby("family"):
    FAM_STATS[fam] = {ch: {"mean": grp[ch].mean(), "std": grp[ch].std()} for ch in CHANNELS}


def _max_abs_z(rid):
    row = LABELS_DF[LABELS_DF.reaction_id == rid]
    if len(row) == 0:
        return 0.0
    row = row.iloc[0]
    fam = row["family"]
    stats = FAM_STATS.get(fam)
    if not stats:
        return 0.0
    max_z = 0.0
    for ch in CHANNELS:
        m, s = stats[ch]["mean"], stats[ch]["std"]
        if s < 1e-6: continue
        z = abs((row[ch] - m) / s)
        if z > max_z: max_z = z
    return max_z


FILTER_FILE = os.environ.get("FILTER_FILE", "")
FILTER_SET = None
if FILTER_FILE and Path(FILTER_FILE).exists():
    FILTER_SET = set(l.strip() for l in Path(FILTER_FILE).read_text().splitlines() if l.strip())
    print(f"filter active: {len(FILTER_SET)} rids from {FILTER_FILE}")


@app.route("/api/rids")
def api_rids():
    """rxns from cohort, sorted by max |z| descending. Filter file limits set.
    If FILTER_SET contains rids not in LABELS_DF, they're still shown (z=0)."""
    out = []
    labels_rids = set(LABELS_DF["reaction_id"].tolist())
    universe = FILTER_SET if FILTER_SET is not None else labels_rids
    for rid in universe:
        fam = _fam(rid)
        z = _max_abs_z(rid) if rid in labels_rids else 0.0
        out.append({"reaction_id": rid, "family": fam,
                    "max_abs_z": z, "is_ood": rid in OOD_SET})
    out.sort(key=lambda x: -x["max_abs_z"])
    resp = jsonify({"rids": out, "n_total": len(out),
                    "n_ood": sum(1 for x in out if x["is_ood"])})
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/rxn/<rid>")
def api_rxn(rid):
    e = _read().get(rid, {})
    R_xyz = (RAW / rid / "R.xyz").read_text()
    TS_xyz = (RAW / rid / "TS.xyz").read_text()
    row_df = LABELS_DF[LABELS_DF.reaction_id == rid]
    channels = {}
    if len(row_df) > 0:
        row = row_df.iloc[0].to_dict()
        fam = row["family"]
        stats = FAM_STATS[fam]
        for ch in CHANNELS:
            v = row[ch]; m = stats[ch]["mean"]; s = stats[ch]["std"]
            z = (v - m) / s if s > 1e-6 else 0.0
            channels[ch] = {"value": v, "z": z, "mean": m, "std": s}
    else:
        fam = _fam(rid)
    eda_out = ORCA_ROOT / rid / "eda.out"
    eda_snippet = ""
    if eda_out.exists():
        txt = eda_out.read_text(errors="ignore")
        # if not TERMINATED NORMALLY, show error section instead
        if "ORCA TERMINATED NORMALLY" not in txt:
            for line in txt.splitlines():
                if "Error" in line or "ERROR" in line:
                    eda_snippet = "FAILED: " + line.strip()
                    break
            if not eda_snippet:
                eda_snippet = "FAILED (no error line found)\n" + txt[-1500:]
        else:
            i = txt.find("Energy Decomposition Analysis")
            if i > 0: eda_snippet = txt[i:i+2000]
    resp = jsonify({
        "rid": rid, "family": fam,
        "R_xyz": R_xyz, "TS_xyz": TS_xyz,
        "frag_A_TS": sorted(e.get("frag_A_indices", [])),
        "frag_B_TS": sorted(e.get("frag_B_indices", [])),
        "frag_A_R":  sorted(e.get("frag_A_indices_R", [])),
        "frag_B_R":  sorted(e.get("frag_B_indices_R", [])),
        "channels": channels,
        "eda_snippet": eda_snippet,
    })
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/api/save_ts/<rid>", methods=["POST"])
def api_save_ts(rid):
    """Save TS partition only. Regen eda.inp, delete eda.out."""
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
            m[rid] = e
            _write(m)
            write_eda_inp(rid, fam, A_TS, B_TS)
        except Exception as ex:
            return jsonify({"ok": False, "error": str(ex)})
    return jsonify({"ok": True, "target": "eda.inp"})


@app.route("/api/save_r/<rid>", methods=["POST"])
def api_save_r(rid):
    """Save R partition only. Regen fragA_R.inp/fragB_R.inp, delete .out."""
    body = request.get_json(force=True)
    A_R = sorted(list(body.get("frag_A_R") or []))
    B_R = sorted(list(body.get("frag_B_R") or []))
    fam = _fam(rid)
    with _lock:
        try:
            m = _read()
            e = m.get(rid, {})
            e["frag_A_indices_R"] = A_R
            e["frag_B_indices_R"] = B_R
            m[rid] = e
            _write(m)
            write_strain_sp_inps(rid, fam, A_R, B_R)
        except Exception as ex:
            return jsonify({"ok": False, "error": str(ex)})
    return jsonify({"ok": True, "target": "fragA_R.inp + fragB_R.inp"})


HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>OOD investigation + edit</title>
<script src="/assets/3Dmol-min.js"></script>
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
.ok { color: #7f7; } .err { color: #f77; }
.container { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; height: calc(100vh - 160px); }
.panels-col { display: grid; grid-template-rows: 1fr 1fr; gap: 8px; }
.panel { background: #000; border: 1px solid #333; position: relative; }
.pt { position: absolute; top: 4px; left: 8px; font-size: 12px; z-index: 10; background: rgba(0,0,0,0.6); padding: 2px 6px; }
.pt-R { color: #6cf; } .pt-TS { color: #fd0; }
.hint { position: absolute; bottom: 4px; left: 8px; font-size: 11px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 3px 6px; }
.right-col { overflow-y: auto; padding: 8px; background: #1a1a1a; border: 1px solid #333; }
.tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
.tbl th, .tbl td { padding: 4px 6px; border-bottom: 1px solid #333; text-align: right; }
.tbl th { background: #222; color: #ccc; text-align: center; }
.tbl td.name { text-align: left; font-family: monospace; color: #ccc; }
.tbl tr.ood { background: #401; }
.tbl tr.ood td { color: #ff8; }
pre { background: #000; color: #9a9; padding: 6px; font-size: 10px; overflow-x: auto; max-height: 260px; overflow-y: auto; }
.save-row { display: flex; gap: 6px; margin: 6px 0; }
.save-row .primary { flex: 1; }
</style>
</head>
<body>
<h1>OOD investigation — R panel edits strain input (fragA_R/fragB_R.inp), TS panel edits EDA input (eda.inp). Coords independent.</h1>
<div class="controls">
  <button onclick="prev()">« prev</button>
  <select id="picker" onchange="pick(this.value)"></select>
  <button onclick="next()">next »</button>
  <label class="info"><input type="checkbox" id="labels" onchange="renderAll()" checked> labels</label>
  <span id="counter" class="info"></span>
</div>
<div id="msg" class="info"></div>
<div class="container">
  <div class="panels-col">
    <div class="panel">
      <div class="pt"><span class="pt-R">■ R (strain SP input)</span> — click atom to toggle R partition</div>
      <div class="hint">R coord, A=red B=blue — click regen fragA_R.inp + fragB_R.inp</div>
      <div id="viewR" style="width:100%; height:100%"></div>
    </div>
    <div class="panel">
      <div class="pt"><span class="pt-TS">■ TS (EDA input)</span> — click atom to toggle TS partition</div>
      <div class="hint">TS coord, A=red B=blue — click regen eda.inp</div>
      <div id="viewTS" style="width:100%; height:100%"></div>
    </div>
  </div>
  <div class="right-col">
    <div class="save-row">
      <button class="primary" onclick="saveR()">💾 save R (strain inp)</button>
      <button class="primary" onclick="saveTS()">💾 save TS (eda.inp)</button>
    </div>
    <div id="channel_table"></div>
    <h3 style="color:#ccc; font-size:13px; margin-top: 20px;">EDA output snippet</h3>
    <pre id="eda_snip"></pre>
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
  ALL.forEach((it, i) => {
    const o = document.createElement("option");
    o.value = i;
    const flag = it.is_ood ? "⚠" : " ";
    o.textContent = `[${i+1}/${ALL.length}] ${flag} |z|=${it.max_abs_z.toFixed(2)}  ${it.reaction_id}`;
    picker.appendChild(o);
  });
  document.getElementById("counter").textContent = `${j.n_ood} OOD / ${j.n_total} total`;
  await fetchRxn();
}

async function fetchRxn() {
  const rid = ALL[idx].reaction_id || ALL[idx].rid;
  const r = await fetch(`/api/rxn/${rid}?ts=${Date.now()}`);
  DATA = await r.json();
  picker.value = idx;
  document.getElementById("msg").innerHTML =
    `<b>${DATA.rid}</b> (${DATA.family})  ` +
    `TS: |A|=${DATA.frag_A_TS.length} |B|=${DATA.frag_B_TS.length} &nbsp;&nbsp;` +
    `R: |A|=${DATA.frag_A_R.length} |B|=${DATA.frag_B_R.length}`;
  buildTable();
  document.getElementById("eda_snip").textContent = DATA.eda_snippet || "(no EDA snippet)";
  renderAll();
}

function buildTable() {
  const order = ["pauli_kcal","elst_kcal","orb_kcal","disp_kcal","strain_kcal","int_eda_kcal","act_kcal"];
  if (!DATA.channels || Object.keys(DATA.channels).length === 0) {
    document.getElementById("channel_table").innerHTML =
      '<div style="color:#f77; padding:8px;">no channel values yet (EDA not converged)</div>';
    return;
  }
  let body = `<table class="tbl"><tr><th>channel</th><th>value</th><th>fam mean</th><th>fam std</th><th>z-score</th></tr>`;
  for (const ch of order) {
    const c = DATA.channels[ch];
    if (!c) continue;
    const isOOD = Math.abs(c.z) > 4;
    const cls = isOOD ? ' class="ood"' : "";
    body += `<tr${cls}><td class="name">${ch}</td>`
         + `<td>${c.value.toFixed(2)}</td>`
         + `<td>${c.mean.toFixed(2)}</td>`
         + `<td>${c.std.toFixed(2)}</td>`
         + `<td>${c.z.toFixed(2)}</td></tr>`;
  }
  body += "</table>";
  document.getElementById("channel_table").innerHTML = body;
}

function styleR(v, atoms) {
  const A = new Set(DATA.frag_A_R), B = new Set(DATA.frag_B_R);
  const showLabels = document.getElementById("labels").checked;
  v.removeAllLabels();
  atoms.forEach((a, i) => {
    const color = A.has(i) ? "#e64" : (B.has(i) ? "#4ae" : "#fa0");
    v.setStyle({serial: a.serial}, {stick: {radius: 0.13, color}, sphere: {scale: 0.30, color}});
    if (showLabels) v.addLabel(String(i), {position:{x:a.x,y:a.y,z:a.z}, fontSize:11, fontColor:"#fff", backgroundOpacity:0, inFront:true, showBackground:false});
  });
}

function styleTS(v, atoms) {
  const A = new Set(DATA.frag_A_TS), B = new Set(DATA.frag_B_TS);
  const showLabels = document.getElementById("labels").checked;
  v.removeAllLabels();
  atoms.forEach((a, i) => {
    const color = A.has(i) ? "#e64" : (B.has(i) ? "#4ae" : "#fa0");
    v.setStyle({serial: a.serial}, {stick: {radius: 0.13, color}, sphere: {scale: 0.30, color}});
    if (showLabels) v.addLabel(String(i), {position:{x:a.x,y:a.y,z:a.z}, fontSize:11, fontColor:"#fff", backgroundOpacity:0, inFront:true, showBackground:false});
  });
}

function buildR(divId) {
  const el = document.getElementById(divId); el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor:"black"});
  v.addModel(DATA.R_xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  atoms.forEach((a) => {
    v.setClickable({serial: a.serial}, true, (clk) => {
      const i = atoms.findIndex(x => x.serial === clk.serial);
      if (i < 0) return; toggleR(i);
    });
  });
  styleR(v, atoms); return {v, atoms};
}

function buildTS(divId) {
  const el = document.getElementById(divId); el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor:"black"});
  v.addModel(DATA.TS_xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  atoms.forEach((a) => {
    v.setClickable({serial: a.serial}, true, (clk) => {
      const i = atoms.findIndex(x => x.serial === clk.serial);
      if (i < 0) return; toggleTS(i);
    });
  });
  styleTS(v, atoms); return {v, atoms};
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
    styleR(vR, atomsR); styleTS(vTS, atomsTS);
    vR.setView(vwR); vTS.setView(vwT);
    vR.render(); vTS.render();
  }
}

function toggleR(i) {
  const A = new Set(DATA.frag_A_R), B = new Set(DATA.frag_B_R);
  if (A.has(i)) { A.delete(i); B.add(i); }
  else if (B.has(i)) { B.delete(i); A.add(i); }
  else { A.add(i); }
  DATA.frag_A_R = Array.from(A).sort((a,b)=>a-b);
  DATA.frag_B_R = Array.from(B).sort((a,b)=>a-b);
  renderAll();
}

function toggleTS(i) {
  const A = new Set(DATA.frag_A_TS), B = new Set(DATA.frag_B_TS);
  if (A.has(i)) { A.delete(i); B.add(i); }
  else if (B.has(i)) { B.delete(i); A.add(i); }
  else { A.add(i); }
  DATA.frag_A_TS = Array.from(A).sort((a,b)=>a-b);
  DATA.frag_B_TS = Array.from(B).sort((a,b)=>a-b);
  renderAll();
}

async function saveR() {
  const r = await fetch(`/api/save_r/${DATA.rid}`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({frag_A_R: DATA.frag_A_R, frag_B_R: DATA.frag_B_R}),
  });
  const j = await r.json();
  const el = document.getElementById("msg");
  if (j.ok) el.innerHTML += ` <span class="ok">R SAVED (${j.target})</span>`;
  else el.innerHTML += ` <span class="err">R SAVE ERR: ${j.error}</span>`;
}

async function saveTS() {
  const r = await fetch(`/api/save_ts/${DATA.rid}`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({frag_A_TS: DATA.frag_A_TS, frag_B_TS: DATA.frag_B_TS}),
  });
  const j = await r.json();
  const el = document.getElementById("msg");
  if (j.ok) el.innerHTML += ` <span class="ok">TS SAVED (${j.target})</span>`;
  else el.innerHTML += ` <span class="err">TS SAVE ERR: ${j.error}</span>`;
}

function pick(i) { idx = +i; fetchRxn(); }
function next() { idx = (idx + 1) % ALL.length; fetchRxn(); }
function prev() { idx = (idx - 1 + ALL.length) % ALL.length; fetchRxn(); }
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight" || e.key === "j") next();
  else if (e.key === "ArrowLeft" || e.key === "k") prev();
});
refresh();
</script>
</body>
</html>
"""


STATIC_DIR = V8 / "static"


@app.route("/assets/<path:name>")
def static_file(name):
    from flask import send_from_directory
    return send_from_directory(str(STATIC_DIR), name)


@app.route("/")
def index():
    resp = Response(HTML, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


if __name__ == "__main__":
    print(f"OOD edit app on port {PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
