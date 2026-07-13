"""Build a self-contained static HTML viewer for the failed EDA rxns.

Embeds R.xyz + TS.xyz + fragment A/B partition for every failed reaction
into one HTML file. Uses 3Dmol.js from CDN for interactive 3D rendering.

No server required beyond a static file host.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
MANUAL_JSON = V8 / "manual_partitions.json"
OUT_HTML = V8 / "failed_viewer.html"


def find_failed():
    failed = []
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
            if "Error" in line and ("multiplicity" in line or "electrons" in line):
                err = line.strip()
                break
        failed.append((d.name, err))
    return failed


def load_all():
    parts = json.loads(MANUAL_JSON.read_text())
    failed = find_failed()
    data = []
    for rid, err in failed:
        r_path = RAW / rid / "R.xyz"
        ts_path = RAW / rid / "TS.xyz"
        if not (r_path.exists() and ts_path.exists()):
            continue
        p = parts.get(rid, {})
        fam = rid.split("_")[0]
        if "qmrxn20" in rid:
            fam = "qmrxn20_" + rid.split("_")[1]
        data.append({
            "rid": rid,
            "family": fam,
            "err": err,
            "R_xyz": r_path.read_text(),
            "TS_xyz": ts_path.read_text(),
            "frag_A": p.get("frag_A_indices", []),
            "frag_B": p.get("frag_B_indices", []),
        })
    return data


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v8 failed EDA rxns viewer (__N__ rxns)</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
body { font-family: -apple-system, sans-serif; margin: 0; padding: 12px; background: #111; color: #ddd; }
h1 { font-size: 15px; margin: 0 0 8px 0; color: #fff; }
.controls { margin-bottom: 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
select, button { background: #222; color: #ddd; border: 1px solid #444; padding: 4px 8px; font-size: 13px; }
button:hover { background: #333; cursor: pointer; }
.info { font-size: 12px; color: #aaa; }
.err { color: #f77; font-size: 12px; margin-top: 4px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; height: calc(100vh - 130px); }
.panel { background: #000; border: 1px solid #333; position: relative; }
.panel-title { position: absolute; top: 4px; left: 8px; font-size: 12px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 2px 6px; }
.legend { position: absolute; bottom: 4px; left: 8px; font-size: 11px; color: #ccc; z-index: 10; background: rgba(0,0,0,0.6); padding: 3px 6px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
</style>
</head>
<body>
<h1>v8 failed EDA rxns (odd-electron / bad charge spec) — __N__ reactions</h1>
<div class="controls">
  <button onclick="prev()">&laquo; prev</button>
  <select id="picker" onchange="pick(this.value)"></select>
  <button onclick="next()">next &raquo;</button>
  <span class="info" id="counter"></span>
  <label class="info"><input type="checkbox" id="labels" onchange="reload()" checked> atom labels</label>
</div>
<div class="err" id="err"></div>
<div class="panels">
  <div class="panel">
    <div class="panel-title">R (reactants)</div>
    <div class="legend"><span class="dot" style="background:#e64"></span>fragment A &nbsp; <span class="dot" style="background:#4ae"></span>fragment B</div>
    <div id="viewR" style="width:100%; height:100%"></div>
  </div>
  <div class="panel">
    <div class="panel-title">TS (transition state)</div>
    <div class="legend"><span class="dot" style="background:#e64"></span>fragment A &nbsp; <span class="dot" style="background:#4ae"></span>fragment B</div>
    <div id="viewTS" style="width:100%; height:100%"></div>
  </div>
</div>

<script>
const DATA = __DATA__;
let idx = 0;
const picker = document.getElementById("picker");
DATA.forEach((d, i) => {
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = "[" + (i+1) + "/" + DATA.length + "] " + d.rid;
  picker.appendChild(opt);
});

function applyStyle(view, atoms, fragA, fragB, showLabels) {
  const A = new Set(fragA), B = new Set(fragB);
  atoms.forEach((a, i) => {
    const inA = A.has(i);
    const c = inA ? "#e64" : (B.has(i) ? "#4ae" : "#888");
    view.setStyle({serial: a.serial}, {
      stick: {radius: 0.12, color: c},
      sphere: {scale: 0.30, color: c},
    });
    if (showLabels) {
      view.addLabel(String(i), {
        position: {x: a.x, y: a.y, z: a.z},
        fontSize: 10, fontColor: "#fff", backgroundOpacity: 0,
        inFront: true, showBackground: false,
      });
    }
  });
}

function loadOne(divId, xyz, fragA, fragB) {
  const el = document.getElementById(divId);
  el.innerHTML = "";
  const v = $3Dmol.createViewer(el, {backgroundColor: "black"});
  v.addModel(xyz, "xyz");
  const atoms = v.getModel().selectedAtoms({});
  const showLabels = document.getElementById("labels").checked;
  applyStyle(v, atoms, fragA, fragB, showLabels);
  v.zoomTo();
  v.render();
}

function reload() {
  const d = DATA[idx];
  document.getElementById("counter").textContent =
    d.rid + "   |   |A|=" + d.frag_A.length + "   |B|=" + d.frag_B.length + "   |   family=" + d.family;
  document.getElementById("err").textContent = d.err || "";
  picker.value = idx;
  loadOne("viewR", d.R_xyz, d.frag_A, d.frag_B);
  loadOne("viewTS", d.TS_xyz, d.frag_A, d.frag_B);
}

function pick(i) { idx = +i; reload(); }
function next() { idx = (idx + 1) % DATA.length; reload(); }
function prev() { idx = (idx - 1 + DATA.length) % DATA.length; reload(); }
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "j") next();
  else if (e.key === "ArrowLeft" || e.key === "k") prev();
});

if (typeof $3Dmol === "undefined") {
  document.body.innerHTML = "<h1 style='color:#f77'>ERROR: 3Dmol.js failed to load from CDN. Check internet access.</h1>";
} else {
  reload();
}
</script>
</body>
</html>
"""


def main():
    data = load_all()
    print(f"embedding {len(data)} failed rxns")
    html = HTML_TEMPLATE.replace("__N__", str(len(data)))
    html = html.replace("__DATA__", json.dumps(data))
    OUT_HTML.write_text(html)
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"wrote {OUT_HTML}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
