"""Group A — re-optimize R/P endpoints, verify bond graph, decide PASS / EXCLUDED.

For each of 4 reactions:
  1. Re-optimize R coords from Halo8 with ADF GeometryOptimization
  2. Re-optimize P coords from Halo8 with ADF GeometryOptimization
  3. Compare drift |E_reopt - E_halo8| against endpoint_match_tol_kcal
  4. Compare bond graph (1.6 × Σ covalent-radii cutoff) between original and reopt
  5. PASS_PENDING (re-validator) iff drift OK on both endpoints AND graph unchanged
     EXCLUDED otherwise (with reason)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from .adf_runner import run_geo_opt
from .config import Config


# Covalent radii (Å), Cordero 2008 — same set the spec assumes is reasonable.
COV_R = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}
SYM_OF = {1:"H", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 15:"P", 16:"S",
           17:"Cl", 35:"Br", 53:"I"}


def bond_graph(symbols: list[str], coords: list[list[float]],
                factor: float) -> set[tuple[int, int]]:
    """Return undirected edges (a,b) with distance ≤ factor × Σ covalent radii."""
    edges: set[tuple[int, int]] = set()
    pts = np.asarray(coords, dtype=float)
    n = len(symbols)
    for i in range(n):
        ri = COV_R.get(symbols[i], 0.8)
        for j in range(i + 1, n):
            rj = COV_R.get(symbols[j], 0.8)
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d <= factor * (ri + rj):
                edges.add((i, j))
    return edges


def _load_halo8_frames(rid: str, halo8_dir: Path) -> dict:
    """Pull R / TS / P symbols + coords for one reaction.

    Prefers the project-local simplified cache (no eda_asm import needed under
    amspython's Python 3.8) at work_fix_fail_19/frames_cache_simple.pkl.
    Falls back to the original eda_asm-pickled cache, then per-reaction files.
    Returns dict with keys: symbols, R_coords, TS_coords, P_coords (or {}).
    """
    simple = Path("work_fix_fail_19/frames_cache_simple.pkl")
    if simple.exists():
        with open(simple, "rb") as fh:
            cache = pickle.load(fh)
        e = cache.get(rid)
        if e is None:
            return {}
        numbers = e["numbers"]
        symbols = [SYM_OF.get(int(z), "?") for z in numbers]
        return {
            "symbols": symbols,
            "R_coords": [list(row) for row in e["positions_R"]],
            "TS_coords": [list(row) for row in e["positions_TS"]],
            "P_coords": [list(row) for row in e["positions_P"]],
        }
    cache_path = halo8_dir / "frames_cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        e = cache.get(rid)
        if e is None:
            return {}
        try:
            numbers = [int(z) for z in e.numbers]
        except TypeError:
            numbers = list(e.numbers)
        symbols = [SYM_OF.get(int(z), "?") for z in numbers]
        try:
            R = np.asarray(e.positions_R).reshape(-1, 3).tolist()
            P = np.asarray(e.positions_P).reshape(-1, 3).tolist()
            TS = np.asarray(e.positions_TS).reshape(-1, 3).tolist()
        except Exception:
            return {}
        return {"symbols": symbols, "R_coords": R, "TS_coords": TS, "P_coords": P}
    per_rxn = halo8_dir / f"{rid}.frames"
    if per_rxn.exists():
        with open(per_rxn, "rb") as fh:
            d = pickle.load(fh)
        return d
    return {}


def _load_original_json(json_path: Path) -> dict:
    """Read the ASR JSON to recover E_R/E_P + bond graphs if present."""
    return json.loads(json_path.read_text())


def process_one(entry: dict, halo8_dir: Path, out_dir: Path,
                 cfg: Config, runner=run_geo_opt) -> dict:
    """Run the Group A workflow on a single reaction; never raises."""
    rid = entry["reaction_id"]
    # Idempotency: skip if a non-error result already exists for this rid
    cached = out_dir / "A" / f"{rid}_result.json"
    if cached.exists():
        try:
            prev = json.loads(cached.read_text())
            if prev.get("new_verdict") in ("EXCLUDED", "PASS", "PENDING_REVALIDATE"):
                return prev
        except Exception:
            pass
    work = out_dir / "A" / f"{rid}_workdir"
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "reaction_id": rid,
        "drift_R": None, "drift_P": None,
        "graph_match_R": None, "graph_match_P": None,
        "new_verdict": "EXCLUDED",
        "exclude_reason": "",
        "action_taken": "reendpoint",
    }
    try:
        frames = _load_halo8_frames(rid, halo8_dir)
        if not frames or "R_coords" not in frames or "P_coords" not in frames:
            result["exclude_reason"] = "halo8 frames missing"
            _write(result, out_dir / "A" / f"{rid}_result.json")
            return result

        orig = _load_original_json(Path(entry["json_path"]))
        E_R_halo8 = float(orig["irc_points"]["R"]["energy_kcal_adf"])
        E_P_halo8 = float(orig["irc_points"]["P"]["energy_kcal_adf"])

        symbols = frames["symbols"]
        R0 = frames["R_coords"]
        P0 = frames["P_coords"]

        # Re-opt R
        opt_R = runner(symbols, R0, charge=int(orig.get("halo8_meta", {}).get("charge", 0) or 0),
                       spin_polarization=0, workdir=str(work / "opt_R"),
                       jobname="opt_R", cfg=cfg)
        if not opt_R.ok or opt_R.final_coords is None:
            result["exclude_reason"] = f"opt_R failed: {opt_R.error or 'no coords'}"
            _write(result, out_dir / "A" / f"{rid}_result.json")
            return result
        result["E_R_reopt"] = opt_R.energy_kcal

        # Re-opt P
        opt_P = runner(symbols, P0, charge=int(orig.get("halo8_meta", {}).get("charge", 0) or 0),
                       spin_polarization=0, workdir=str(work / "opt_P"),
                       jobname="opt_P", cfg=cfg)
        if not opt_P.ok or opt_P.final_coords is None:
            result["exclude_reason"] = f"opt_P failed: {opt_P.error or 'no coords'}"
            _write(result, out_dir / "A" / f"{rid}_result.json")
            return result
        result["E_P_reopt"] = opt_P.energy_kcal

        # Drifts
        result["drift_R"] = E_R_halo8 - opt_R.energy_kcal
        result["drift_P"] = E_P_halo8 - opt_P.energy_kcal

        # Bond graphs before / after for both endpoints
        g_R_before = bond_graph(symbols, R0, cfg.bond_cutoff_factor)
        g_R_after = bond_graph(symbols, opt_R.final_coords, cfg.bond_cutoff_factor)
        g_P_before = bond_graph(symbols, P0, cfg.bond_cutoff_factor)
        g_P_after = bond_graph(symbols, opt_P.final_coords, cfg.bond_cutoff_factor)
        result["graph_match_R"] = (g_R_before == g_R_after)
        result["graph_match_P"] = (g_P_before == g_P_after)

        within_R = abs(result["drift_R"]) <= cfg.endpoint_match_tol_kcal
        within_P = abs(result["drift_P"]) <= cfg.endpoint_match_tol_kcal

        if not result["graph_match_R"] or not result["graph_match_P"]:
            diffs = []
            if not result["graph_match_R"]:
                diffs.append(f"R changed: added={sorted(g_R_after - g_R_before)}, removed={sorted(g_R_before - g_R_after)}")
            if not result["graph_match_P"]:
                diffs.append(f"P changed: added={sorted(g_P_after - g_P_before)}, removed={sorted(g_P_before - g_P_after)}")
            result["exclude_reason"] = "; ".join(diffs)
            result["new_verdict"] = "EXCLUDED"
        elif not (within_R and within_P):
            result["exclude_reason"] = (f"drift exceeded: |dR|={abs(result['drift_R']):.3f}, "
                                          f"|dP|={abs(result['drift_P']):.3f} > {cfg.endpoint_match_tol_kcal}")
            result["new_verdict"] = "EXCLUDED"
        else:
            result["new_verdict"] = "PENDING_REVALIDATE"
            # Write a re-endpointed stage5a + result.json shell for the validator
            asr_v2 = out_dir / "A" / "asr_v2"
            asr_v2.mkdir(parents=True, exist_ok=True)
            new_json = dict(orig)
            new_json["irc_points"] = dict(new_json["irc_points"])
            new_json["irc_points"]["R"] = dict(new_json["irc_points"]["R"],
                                                  energy_kcal_adf=opt_R.energy_kcal)
            new_json["irc_points"]["P"] = dict(new_json["irc_points"]["P"],
                                                  energy_kcal_adf=opt_P.energy_kcal)
            new_json["reendpointed"] = True
            (asr_v2 / f"{rid}.json").write_text(json.dumps(new_json, indent=2))

    except Exception as exc:
        result["exclude_reason"] = f"unexpected exception: {exc}\n{traceback.format_exc()}"
    _write(result, out_dir / "A" / f"{rid}_result.json")
    return result


def _write(obj: dict, path: Path) -> None:
    """Write JSON; create parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def main() -> int:
    """CLI entry per fix_fail_19_spec §3."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--halo8-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    cfg = Config()
    queue = json.loads(args.queue.read_text())
    n_excluded = n_pending = 0
    for entry in queue:
        r = process_one(entry, args.halo8_dir, args.out_dir, cfg)
        if r["new_verdict"] == "EXCLUDED":
            n_excluded += 1
        elif r["new_verdict"] == "PENDING_REVALIDATE":
            n_pending += 1
    print(f"group_a: pending_revalidate={n_pending}, excluded={n_excluded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
