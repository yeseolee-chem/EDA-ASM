"""Generate self-contained 3D viewer HTML.

Avoids f-string brace-escape pitfalls by keeping the JS in a separate template
file and only substituting two named placeholders: __SUBSTRATES__ and __XYZ__.
"""
from __future__ import annotations
import json
from pathlib import Path

V1 = Path(__file__).resolve().parent.parent
TEMPLATE = V1 / "viz" / "template.html"
OUT = V1 / "viz" / "index.html"

SUBSTRATES = [
    # (id, R-label, sigma_p, Ea, imag_freq, Estrain, Eelst, EPauli, Eoi, Edisp, Eint)
    ("nme2", "N(CH3)2", -0.83, 29.28, -603, 18.15, -78.67, 168.40, -114.64, -9.19, -34.10),
    ("nh2",  "NH2",     -0.66, 30.79, -599, 16.72, -75.25, 162.22, -112.54, -7.70, -33.27),
    ("oh",   "OH",      -0.37, 25.17, -569, 15.64, -68.46, 148.40, -104.35, -7.43, -31.84),
    ("ome",  "OCH3",    -0.27, 30.50, -585, 15.56, -67.75, 145.71, -104.82, -8.10, -34.96),
    ("me",   "CH3",     -0.17, 30.49, -631, 14.08, -77.76, 167.67, -118.86, -8.16, -37.11),
    ("ph",   "C6H5",    -0.01, 29.71, -631, 15.31, -82.00, 176.03, -123.44, -9.49, -38.90),
    ("h",    "H",        0.00, 40.54, -610, 12.35, -60.60, 132.74,  -97.98, -7.05, -32.89),
    ("f",    "F",        0.06, 24.05, -566, 14.06, -68.03, 147.91, -108.10, -7.08, -35.30),
    ("i",    "I",        0.18, 24.63, -562, 13.21, -67.78, 148.77, -108.22, -9.05, -36.28),
    ("br",   "Br",       0.23, 24.34, -564, 13.81, -68.16, 148.91, -109.03, -8.63, -36.91),
    ("cl",   "Cl",       0.23, 24.68, -572, 13.94, -69.38, 150.89, -110.40, -8.29, -37.18),
    ("ac",   "COCH3",    0.50, 37.75, -606, 13.85, -63.19, 137.29, -101.64, -8.72, -36.26),
    ("cf3",  "CF3",      0.54, 36.35, -570, 13.33, -56.82, 123.12,  -95.71, -8.33, -37.74),
    ("cn",   "CN",       0.66, 35.97, -561, 11.84, -55.42, 120.04,  -94.79, -8.14, -38.31),
    ("no2",  "NO2",      0.78, 25.59, -556, 15.14, -67.02, 144.87, -112.11, -8.53, -42.79),
]


def main() -> None:
    data = {}
    for s in SUBSTRATES:
        rid = s[0]
        data[rid] = {
            "R":     (V1 / "runs" / rid / "build" / "mol.xyz").read_text() if (V1 / "runs" / rid / "build" / "mol.xyz").exists() else "",
            "TS":    (V1 / "runs" / rid / "orca" / "ts.xyz").read_text() if (V1 / "runs" / rid / "orca" / "ts.xyz").exists() else "",
            "fragA": (V1 / "runs" / rid / "eda"  / "frag_A.xyz").read_text() if (V1 / "runs" / rid / "eda" / "frag_A.xyz").exists() else "",
            "fragB": (V1 / "runs" / rid / "eda"  / "frag_B.xyz").read_text() if (V1 / "runs" / rid / "eda" / "frag_B.xyz").exists() else "",
        }
    subs_records = [
        {"id": s[0], "R": s[1], "sigp": s[2], "Ea": s[3], "imag": s[4],
         "Estrain": s[5], "Eelst": s[6], "EPauli": s[7], "Eoi": s[8],
         "Edisp": s[9], "Eint": s[10]}
        for s in SUBSTRATES
    ]
    subs_js = json.dumps(subs_records)
    xyz_js = json.dumps(data)

    tmpl = TEMPLATE.read_text()
    html = tmpl.replace("__SUBSTRATES__", subs_js).replace("__XYZ__", xyz_js)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
