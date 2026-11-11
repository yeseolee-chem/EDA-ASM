"""Start conditions 1-1 .. 1-4 of docs/specs/GNN3_CLEAN.md -> gnn3/results/S0_start.json.

Run on an RTX 3090 node in env_d1fu2 (jobs/s0.slurm); exits 0 when every condition holds, 3 otherwise (STOP).
Also records which developer tools exist in the environment (pytest, ruff) for the later steps.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]          # gnn3/
REPO = PACKAGE_ROOT.parent
RESULTS = PACKAGE_ROOT / "results"

CH3M_UNITS = Path("/gpfs/tmp_cpu2/yeseo1ee/ch3m/units")
CH3M_RESULTS = REPO / "analysis" / "ch3m" / "results"
LF_INPUTS = Path("/gpfs/tmp_cpu2/yeseo1ee/ch3/lf")
TOKEN_MANIFEST = REPO / "analysis" / "ch3" / "results" / "lf_inputs" / "fu2_lf" / "T_manifest.csv"
PAIR_LABELS = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv")
SPLIT_FILES = [Path(f"/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f{k}.json") for k in range(5)]

ABS3_HEAD = "04642c1c5f9dc100bb4d3bc599f8c315ab506659"
WANT = {"X182_lf.parquet": "cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199",
        "T_manifest.csv": "d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72",
        "F_pairs_cr2.csv": "f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def main() -> int:
    import pandas as pd
    checks = {}
    # 1-1: the abs3 branch the work starts from
    origin_abs3 = git("rev-parse", "origin/abs3")
    checks["1-1"] = dict(origin_abs3=origin_abs3, want=ABS3_HEAD, ok=origin_abs3 == ABS3_HEAD)
    # 1-2: the 5 GNN3 units (json, parquet, 5 seed .pt each); parquet sha256 = the unit json record, and the unit rows
    # are exactly the GNN lf3 rows of KM_predictions.parquet (whose sha256 KM_report.json records)
    km_report = json.loads((CH3M_RESULTS / "KM_report.json").read_text())
    km_path = CH3M_RESULTS / "KM_predictions.parquet"
    km_sha_ok = sha256(km_path) == km_report.get("predictions_sha256")
    km = pd.read_parquet(km_path)
    km = km[(km.rung == "lf3") & (km.model == "GNN")]
    units = {}
    for k in range(5):
        name = f"D__lf3__S__GNN__f{k}"
        js, pq = CH3M_UNITS / f"{name}.json", CH3M_UNITS / f"{name}.parquet"
        pts = [CH3M_UNITS / f"{name}__s{j}.pt" for j in range(5)]
        present = js.is_file() and pq.is_file() and all(p.is_file() for p in pts)
        rec = json.loads(js.read_text()) if js.is_file() else {}
        pq_ok = present and sha256(pq) == rec.get("parquet_sha256")
        same_rows = False
        if pq_ok:
            unit = pd.read_parquet(pq).sort_values(["pair_id", "target"]).reset_index(drop=True)
            ref = km[km.fold == k].sort_values(["pair_id", "target"]).reset_index(drop=True)
            same_rows = len(unit) == len(ref) and bool((unit[ref.columns].values == ref.values).all())
        units[name] = dict(files_present=present, parquet_sha256_eq_unit_json=pq_ok, rows_eq_KM_predictions=same_rows,
                           pt_sha256={p.name: sha256(p) for p in pts if p.is_file()})
    checks["1-2"] = dict(units=units, KM_predictions_sha256_eq_KM_report=km_sha_ok,
                         ok=km_sha_ok and all(u["files_present"] and u["parquet_sha256_eq_unit_json"]
                                              and u["rows_eq_KM_predictions"] for u in units.values()))
    # 1-3: inputs identical to CH3M
    prereg = json.loads((CH3M_RESULTS / "prereg_ch3m.json").read_text())["inputs_sha256"]
    want_split = {Path(p).name: s for p, s in prereg.items() if Path(p).name.startswith("T1__f")}
    got = {"X182_lf.parquet": sha256(LF_INPUTS / "X182_lf.parquet"), "T_manifest.csv": sha256(TOKEN_MANIFEST),
           "F_pairs_cr2.csv": sha256(PAIR_LABELS)}
    splits = {p.name: dict(sha256=sha256(p), want=want_split.get(p.name)) for p in SPLIT_FILES}
    manifest = pd.read_csv(TOKEN_MANIFEST)
    token_dir = LF_INPUTS / "fu2_lf" / "tokens"
    bad_tokens = [int(r) for r, s in zip(manifest.rxn_id, manifest.sha256)
                  if not (token_dir / f"{int(r)}.npz").is_file() or sha256(token_dir / f"{int(r)}.npz") != s]
    checks["1-3"] = dict(sha256=got, want=WANT, splits=splits, n_tokens=int(len(manifest)), n_token_files_bad=len(bad_tokens),
                         ok=got == WANT and all(v["sha256"] == v["want"] for v in splits.values()) and not bad_tokens
                         and len(manifest) == 8502)
    # 1-4: env_d1fu2 on an RTX 3090
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    checks["1-4"] = dict(python=sys.version.split()[0], prefix=sys.prefix, torch=torch.__version__, cuda=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), gpu=gpu,
                         ok=bool(gpu and "3090" in gpu and torch.__version__ == "2.2.1+cu118"
                                 and Path(sys.prefix).name == "env_d1fu2"))
    tools = {m: importlib.util.find_spec(m) is not None for m in ("pytest", "ruff", "scipy", "sklearn", "pyarrow")}
    ruff_bin = Path(sys.prefix) / "bin" / "ruff"
    tools["ruff_binary"] = ruff_bin.is_file()
    failed = [k for k, v in checks.items() if not v["ok"]]
    out = dict(spec="docs/specs/GNN3_CLEAN.md 1", written=time.strftime("%Y-%m-%d %H:%M:%S"), all_pass=not failed,
               failed=failed, checks=checks, tools_in_env_d1fu2=tools, git_head=git("rev-parse", "HEAD"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS / ".S0_start.json.tmp"
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.replace(RESULTS / "S0_start.json")
    print(json.dumps(dict(all_pass=not failed, failed=failed, tools=tools, **{k: v["ok"] for k, v in checks.items()}),
                     indent=1))
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
