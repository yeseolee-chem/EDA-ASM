"""Group B — <S²> diagnosis + 4-coupling EDA sweep with Yamaguchi projection.

For each of 7 reactions:
  1. Run unrestricted supermolecule SP at TS to read <S²>
  2. Pick coupling candidates per <S²> band
  3. Run EDA at each candidate, extract σ residual via validator.derive
  4. For BS-singlet candidate: also run triplet at same fragmentation; apply
     Yamaguchi projection ΔE_int_singlet ≈ (2·ΔE_BS - <S²>_BS·ΔE_T)/(2 - <S²>_BS)
  5. Pick the winning coupling (smallest residual); verdict PASS / WARN /
     demoted_to_C per the spec.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adf_runner import run_sp, run_eda
from .config import Config


def _candidates_for_s2(s2: float, cfg: Config) -> list[str]:
    """Pick coupling candidates from the <S²> band (spec §4 Step 1)."""
    if s2 is None or s2 < cfg.s2_pure_singlet_max:
        return ["closed_shell_singlet"]
    if s2 < cfg.s2_triplet_min:
        return ["BS_singlet_yamaguchi", "triplet_ferro"]
    if s2 < cfg.s2_quintet_min:
        return ["triplet_ferro", "quintet_ferro"]
    return ["quintet_ferro"]


def yamaguchi_project(de_int_bs: float, de_int_triplet: float, s2_bs: float) -> float:
    """Spin-projected singlet ΔE_int from BS singlet + triplet (Yamaguchi)."""
    denom = 2.0 - s2_bs
    if abs(denom) < 1e-12:
        return de_int_bs
    return (2.0 * de_int_bs - s2_bs * de_int_triplet) / denom


def _frag_spec_for_coupling(cpl: str, fragments: list[dict],
                              symbols: list[str]) -> tuple[list[dict], int]:
    """Build per-fragment (multiplicity, spin_sign) per spec §4 Step 2 table.

    Returns (frag_states, supermol_total_spin_polarization).
    """
    # Atomic-number proxy for electron parity
    Z = {"H":1,"B":5,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"Cl":17,"Br":35,"I":53}
    e_counts = [sum(Z.get(symbols[a], 0) for a in f["atom_indices"]) for f in fragments]
    states: list[dict] = []
    if cpl == "closed_shell_singlet":
        for f in fragments:
            states.append({"role": f["role"], "multiplicity": 1, "spin_sign": 1})
        return states, 0
    if cpl == "BS_singlet_yamaguchi":
        # Anti-parallel doublets — must have ≥ 2 open-shell-capable fragments
        if len(fragments) < 2:
            return [], 0
        states.append({"role": fragments[0]["role"], "multiplicity": 2, "spin_sign": 1})
        states.append({"role": fragments[1]["role"], "multiplicity": 2, "spin_sign": -1})
        for f in fragments[2:]:
            states.append({"role": f["role"], "multiplicity": 1, "spin_sign": 1})
        return states, 0
    if cpl == "triplet_ferro":
        if len(fragments) < 2:
            return [], 0
        states.append({"role": fragments[0]["role"], "multiplicity": 2, "spin_sign": 1})
        states.append({"role": fragments[1]["role"], "multiplicity": 2, "spin_sign": 1})
        for f in fragments[2:]:
            states.append({"role": f["role"], "multiplicity": 1, "spin_sign": 1})
        return states, 2
    if cpl == "quintet_ferro":
        if len(fragments) < 2:
            return [], 0
        states.append({"role": fragments[0]["role"], "multiplicity": 3, "spin_sign": 1})
        states.append({"role": fragments[1]["role"], "multiplicity": 3, "spin_sign": 1})
        for f in fragments[2:]:
            states.append({"role": f["role"], "multiplicity": 1, "spin_sign": 1})
        return states, 4
    return [], 0


def _validate_residual(asr_vector_R: dict, asr_vector_TS: dict,
                        asr_vector_P: dict, irc_R: float, irc_TS: float,
                        irc_P: float, fragment_opt_sum: float) -> dict:
    """Re-derive residuals using the same logic as validate_asr.derive,
    avoiding a circular import + a re-implementation of the math."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "Validate"))
    from validate_asr import derive  # type: ignore
    raw = {
        "reaction_id": "stub",
        "schema_version": "spinsweep",
        "adf_settings": {"functional": "x", "basis": "x", "dispersion": "x",
                          "relativity": "x", "frozen_core": "x", "integration": "x"},
        "irc_points": {
            "R":  {"energy_kcal_adf": irc_R},
            "TS": {"energy_kcal_adf": irc_TS},
            "P":  {"energy_kcal_adf": irc_P},
        },
        "fragment_opt_energy_kcal": {"sum_placeholder": fragment_opt_sum},
        "asr_vector_kcal": {"R": asr_vector_R, "TS": asr_vector_TS,
                              "P": asr_vector_P},
    }
    d = derive("stub", raw)
    return {
        "max_abs_res_cons": d.max_abs_res_cons,
        "max_abs_res_ref": d.max_abs_res_ref,
        "offset_spread": d.offset_spread,
        "sigma_R": d.sigma.get("R"),
        "sigma_TS": d.sigma.get("TS"),
        "sigma_P": d.sigma.get("P"),
    }


def process_one(entry: dict, halo8_dir: Path, out_dir: Path, cfg: Config,
                 sp_runner=run_sp, eda_runner=run_eda) -> dict:
    """Run Group B spin sweep on a single reaction."""
    rid = entry["reaction_id"]
    # Idempotency: re-use cached result so the dispatcher can read it without
    # needing scm.plams again.
    cached = out_dir / "B" / f"{rid}_result.json"
    if cached.exists():
        try:
            prev = json.loads(cached.read_text())
            if prev.get("new_verdict") in ("PASS", "WARN", "FAIL",
                                            "demoted_to_C"):
                return prev
        except Exception:
            pass
    work = out_dir / "B" / f"{rid}_workdir"
    work.mkdir(parents=True, exist_ok=True)
    res: dict = {"reaction_id": rid, "action_taken": "spinsweep",
                 "s2_supermol_TS": None, "candidates_tried": [],
                 "residuals": {}, "winning_coupling": None,
                 "new_max_abs_res_cons_kcal": None,
                 "new_verdict": "FAIL"}

    # Load source data
    orig = json.loads(Path(entry["json_path"]).read_text())
    symbols = orig.get("symbols") or []
    if not symbols and "fragmentation" in orig:
        # fallback: build from frames_cache
        from .group_a_reendpoint import _load_halo8_frames
        f = _load_halo8_frames(rid, halo8_dir)
        symbols = f.get("symbols") or []
    # TS coords come from frames cache as well (orig JSON often lacks positions)
    from .group_a_reendpoint import _load_halo8_frames
    f = _load_halo8_frames(rid, halo8_dir)
    if not f:
        res["error"] = "no halo8 frames"
        _write(res, out_dir / "B" / f"{rid}_result.json")
        return res
    symbols = f.get("symbols") or symbols
    ts_coords = f.get("TS_coords") or f.get("positions_TS")
    if ts_coords is None:
        # Some pickles use positions_TS; try variations
        try:
            import pickle
            with open(halo8_dir / "frames_cache.pkl", "rb") as fh:
                cache = pickle.load(fh)
            entry_obj = cache.get(rid)
            ts_coords = (getattr(entry_obj, "positions_TS", None) or
                         getattr(entry_obj, "TS_coords", None))
            ts_coords = ts_coords.reshape(-1, 3).tolist() if ts_coords is not None else None
        except Exception:
            ts_coords = None
    if ts_coords is None:
        res["error"] = "no TS coords"
        _write(res, out_dir / "B" / f"{rid}_result.json")
        return res

    # 1) supermolecule TS SP — try unrestricted with sp=0 first (closed-shell)
    sp = sp_runner(symbols, ts_coords, charge=0, spin_polarization=2,
                    workdir=str(work / "s2_probe"), jobname="supermol_TS_probe",
                    cfg=cfg)
    if not sp.ok:
        res["error"] = f"supermol probe failed: {sp.error}"
        res["new_verdict"] = "FAIL"
        _write(res, out_dir / "B" / f"{rid}_result.json")
        return res
    res["s2_supermol_TS"] = sp.s2 if sp.s2 is not None else 0.0

    # 2) Candidate set by <S²> band
    cands = _candidates_for_s2(res["s2_supermol_TS"], cfg)
    res["candidates_tried"] = list(cands)

    fragments_in = orig.get("fragmentation", {}).get("fragments", [])
    if not fragments_in:
        # fall back to stage5a winner if main JSON missing fragmentation
        s5 = Path("ADF_500_edited/stage5a/per_reaction") / rid / "result.json"
        if s5.exists():
            fragments_in = json.loads(s5.read_text())["result"]["fragments"]
    if not fragments_in:
        res["error"] = "no fragmentation available"
        _write(res, out_dir / "B" / f"{rid}_result.json")
        return res

    # 3) For each candidate, would normally rebuild fragment .t21 references and
    #    run a coupled EDA on R/TS/P. The infrastructure for that depends on the
    #    full run_asr_spec workflow (fragment SP at each ζ + fragment opt). We
    #    delegate to run_asr_spec when possible; otherwise we surface an error.
    try:
        import sys
        ROOT = Path(__file__).resolve().parent.parent
        for p in (ROOT, ROOT / "ADF_500" / "scripts"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import run_asr_spec  # type: ignore  (only available under amspython)
        has_runner = True
    except Exception as exc:
        has_runner = False

    if not has_runner:
        res["error"] = ("run_asr_spec import failed — Group B requires the "
                         "ADF runner. Re-run under amspython.")
        _write(res, out_dir / "B" / f"{rid}_result.json")
        return res

    # Reuse run_asr_spec.run_one with a tweaked stage5a where the multiplicities
    # match the candidate coupling. We monkey-patch STAGE5A_DIR / OUT_DIR + the
    # frame loader (synthetic rid as in earlier candidate sweeps).
    run_asr_spec.STAGE5A_DIR = work / "stage5a"
    run_asr_spec.OUT_DIR = work / "results"
    (run_asr_spec.STAGE5A_DIR / "per_reaction").mkdir(parents=True, exist_ok=True)
    run_asr_spec.OUT_DIR.mkdir(parents=True, exist_ok=True)
    orig_load_3 = run_asr_spec.load_3_frames
    def _patched_load_3_frames(rid_x, stage5a):
        base = rid_x.split("__", 1)[0]
        return orig_load_3(base, stage5a)
    run_asr_spec.load_3_frames = _patched_load_3_frames

    # Make sure db_idx_map includes the synthetic rids for this rxn
    db_map_path = ROOT / "outputs" / "asr_spec" / "db_idx_map.json"
    if db_map_path.exists():
        db_map = json.loads(db_map_path.read_text())
        base_idx = db_map.get(rid)
        if base_idx is not None:
            for cpl in cands:
                synth = f"{rid}__bcpl_{cpl}"
                db_map[synth] = base_idx
            db_map_path.write_text(json.dumps(db_map, indent=2))

    residuals: dict[str, float] = {}
    extras: dict[str, dict] = {}
    s5_src = json.loads(
        (Path("ADF_500_edited/stage5a/per_reaction") / rid / "result.json").read_text()
    )
    for cpl in cands:
        synth = f"{rid}__bcpl_{cpl}"
        states, total_sp = _frag_spec_for_coupling(cpl, fragments_in, symbols)
        if not states:
            continue
        # Write candidate stage5a with new multiplicities
        new_s5 = dict(s5_src)
        new_s5["reaction_id"] = synth
        new_s5_result = dict(s5_src["result"])
        new_s5_result["fragments"] = []
        new_s5_result["pattern"] = f"spinsweep_{cpl}"
        spin_signs = [st["spin_sign"] for st in states]
        for orig_f, st in zip(fragments_in, states):
            new_s5_result["fragments"].append({
                "atom_indices": list(orig_f["atom_indices"]),
                "role": orig_f["role"],
                "multiplicity": st["multiplicity"],
                "cap_attachment": None,
            })
        new_s5_result["spin_signs"] = spin_signs
        new_s5_result["total_spin_polarization"] = int(total_sp)
        new_s5_result["coupling"] = cpl
        new_s5["result"] = new_s5_result
        (run_asr_spec.STAGE5A_DIR / "per_reaction" / synth).mkdir(parents=True, exist_ok=True)
        (run_asr_spec.STAGE5A_DIR / "per_reaction" / synth / "result.json").write_text(
            json.dumps(new_s5, indent=2))
        try:
            r = run_asr_spec.run_one(synth)
            (run_asr_spec.OUT_DIR / f"{synth}.json").write_text(json.dumps(r, indent=2, default=str))
            # Compute residual via validator.derive
            sys.path.insert(0, str(ROOT / "Validate"))
            from validate_asr import derive  # type: ignore
            d = derive(synth, r)
            residuals[cpl] = d.max_abs_res_cons if d.max_abs_res_cons is not None else float("inf")
            extras[cpl] = {
                "asr_vector": r.get("asr_vector_kcal", {}),
                "status": r.get("status_at_queue"),
                "sigma_TS": d.sigma.get("TS"), "sigma_R": d.sigma.get("R"),
            }
        except Exception as exc:
            residuals[cpl] = float("inf")
            extras[cpl] = {"error": str(exc)}

    # 4) Yamaguchi projection — only if BS_singlet candidate was tried and we
    #    additionally have a triplet ΔE_int from the same fragmentation
    if ("BS_singlet_yamaguchi" in residuals and "triplet_ferro" in residuals
        and extras["BS_singlet_yamaguchi"].get("asr_vector")
        and extras["triplet_ferro"].get("asr_vector")):
        try:
            sum_TS_bs = sum(extras["BS_singlet_yamaguchi"]["asr_vector"]["TS"].values())
            sum_TS_tr = sum(extras["triplet_ferro"]["asr_vector"]["TS"].values())
            s2_bs = res["s2_supermol_TS"] or 0.0
            projected_TS = yamaguchi_project(sum_TS_bs, sum_TS_tr, s2_bs)
            extras["BS_singlet_yamaguchi"]["yamaguchi_sigma_TS_projected"] = projected_TS
            # We can't fully re-derive the residual from a single projected σ(TS)
            # without σ(R) projection too, so we keep the BS residual as-is and
            # surface the projected value for the manifest. The spec calls this
            # out — the projection feeds the downstream component vector, not
            # the verdict gate.
        except Exception as exc:
            extras["BS_singlet_yamaguchi"]["yamaguchi_error"] = str(exc)

    res["residuals"] = residuals
    res["extras"] = extras
    if residuals:
        winning = min(residuals, key=residuals.get)
        res["winning_coupling"] = winning
        win_res = residuals[winning]
        res["new_max_abs_res_cons_kcal"] = win_res
        if win_res <= cfg.tau_warn:
            res["new_verdict"] = "PASS"
        elif win_res <= cfg.spinsweep_winner_tol:  # ≤ 0.5 also; spec uses tau_warn
            res["new_verdict"] = "WARN"
        else:
            res["new_verdict"] = "demoted_to_C"
            res["action_taken"] = "spinsweep_then_relax"
    else:
        res["new_verdict"] = "FAIL"
        res["action_taken"] = "scf_failed"
    _write(res, out_dir / "B" / f"{rid}_result.json")
    return res


def _write(obj: dict, path: Path) -> None:
    """JSON dump with parent dir mkdir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def main() -> int:
    """CLI entry per fix_fail_19_spec §4."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--halo8-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    cfg = Config()
    queue = json.loads(args.queue.read_text())
    summary = {"PASS": 0, "WARN": 0, "demoted_to_C": 0, "FAIL": 0}
    for entry in queue:
        r = process_one(entry, args.halo8_dir, args.out_dir, cfg)
        summary[r["new_verdict"]] = summary.get(r["new_verdict"], 0) + 1
    print(f"group_b: PASS={summary.get('PASS',0)}  WARN={summary.get('WARN',0)}  "
          f"demoted={summary.get('demoted_to_C',0)}  FAIL={summary.get('FAIL',0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
