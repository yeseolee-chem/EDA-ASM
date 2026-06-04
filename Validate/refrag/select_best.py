#!/usr/bin/env python3
"""Pick the winning candidate per failing reaction and write canonical outputs.

Scoring (lower is better):
  primary    : max_abs_res_cons + max_abs_res_ref + offset_spread (Check-4 residuals)
  + sign_pen : 5 kcal each for strain≤0, ΔPauli_TS≤0, Δoi_TS≥0
  + spin_pen : 1 kcal for any open-shell fragment (prefer closed shell)
  + nfrag_pen: 0.5 kcal per fragment beyond 2 (prefer 2-fragment if tied)
  hard veto  : if E_TS not max (Check 3 ts_not_max), big penalty (+1000)
              if barrier ≤ 0, hard penalty (+1000)

The canonical winner is copied to:
  Validate/refrag/results/<rid>.json
  Validate/refrag/stage5a/per_reaction/<rid>/result.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
CAND_S5 = ROOT / "Validate" / "refrag" / "candidates_stage5a"
CAND_RES = ROOT / "Validate" / "refrag" / "candidate_results"
CANON_S5 = ROOT / "Validate" / "refrag" / "stage5a" / "per_reaction"
CANON_RES = ROOT / "Validate" / "refrag" / "results"
SUMMARY = ROOT / "Validate" / "refrag" / "candidate_summary.json"


def score(d, raw, label: str) -> tuple[float, dict]:
    """Lower is better. Returns (score, breakdown_dict)."""
    bd: dict = {"label": label}
    base = 0.0

    if d.max_abs_res_cons is None or d.max_abs_res_ref is None or d.offset_spread is None:
        bd["fatal"] = "missing_residuals"; return (1e9, bd)

    base += d.max_abs_res_cons + d.max_abs_res_ref + d.offset_spread
    bd["res_sum"] = base

    # Sign penalties
    sign_pen = 0.0
    if "TS" in d.comp and "R" in d.comp:
        ds = d.comp["TS"]["strain"] - d.comp["R"]["strain"]
        dp = d.comp["TS"]["Pauli"]  - d.comp["R"]["Pauli"]
        do = d.comp["TS"]["oi"]     - d.comp["R"]["oi"]
        if ds <= 0: sign_pen += 5
        if dp <= 0: sign_pen += 5
        if do >= 0: sign_pen += 5
    bd["sign_pen"] = sign_pen
    base += sign_pen

    # Spin penalty: prefer closed shell
    fragmentation = (raw.get("fragmentation") or {})
    frags = fragmentation.get("fragments") or []
    n_open = sum(1 for f in frags if int(f.get("multiplicity", 1)) > 1)
    bd["n_open_shell"] = n_open
    bd["n_frag"] = len(frags)
    base += n_open * 1.0
    base += max(0, len(frags) - 2) * 0.5

    # Topology vetoes
    veto = 0.0
    if "TS" in d.E and "R" in d.E and "P" in d.E:
        if d.E["TS"] <= d.E["R"] or d.E["TS"] <= d.E["P"]:
            veto += 1000
    if d.dE_act is not None and d.dE_act <= 0:
        veto += 1000
    bd["veto"] = veto
    base += veto

    return (base, bd)


def main() -> int:
    sys.path.insert(0, str(ROOT / "Validate"))
    from validate_asr import (  # type: ignore
        Config, derive, check1_schema, check3_topology,
        check4_conservation, check5_signs, aggregate,
    )
    cfg = Config()

    cand_summary = json.loads(SUMMARY.read_text())
    rids = list(cand_summary["rids"].keys())
    print(f"Selecting best for {len(rids)} reactions")

    winners: list[dict] = []
    losers_by_rid: dict[str, list[dict]] = {}
    no_results: list[str] = []

    for rid in rids:
        cands = cand_summary["rids"][rid]["candidates"]
        scored = []
        for c in cands:
            srid = c["synth_rid"]
            rpath = CAND_RES / f"{srid}.json"
            spath = CAND_S5 / "per_reaction" / srid / "result.json"
            if not rpath.exists() or not spath.exists():
                continue
            try:
                raw = json.loads(rpath.read_text())
                stage5a = json.loads(spath.read_text())
            except Exception:
                continue
            if raw.get("status_at_queue") == "FAILED":
                continue
            d = derive("c", raw)
            sc, bd = score(d, raw, c["label"])
            scored.append({
                "label": c["label"],
                "synth_rid": srid,
                "score": sc,
                "breakdown": bd,
                "max_abs_res_cons": d.max_abs_res_cons,
                "max_abs_res_ref": d.max_abs_res_ref,
                "offset_spread": d.offset_spread,
                "verdict": aggregate(check1_schema(d) + check3_topology(d, cfg)
                                       + check4_conservation(d, cfg) + check5_signs(d)),
                "stage5a_path": str(spath),
                "raw_path": str(rpath),
            })
        if not scored:
            no_results.append(rid)
            continue
        scored.sort(key=lambda x: x["score"])
        win = scored[0]
        winners.append({
            "rid": rid,
            "winner": win,
            "n_candidates_tried": len(cands),
            "n_candidates_with_result": len(scored),
            "all_scores": [{"label": s["label"], "score": round(s["score"], 4),
                            "res_cons": round(s["max_abs_res_cons"] or 0, 4),
                            "verdict": s["verdict"]} for s in scored],
        })
        losers_by_rid[rid] = scored[1:]

        # Copy winner to canonical paths
        target_res = CANON_RES / f"{rid}.json"
        # Rewrite reaction_id back from synthetic to canonical
        raw = json.loads(Path(win["raw_path"]).read_text())
        raw["reaction_id"] = rid
        raw["selected_candidate"] = win["label"]
        target_res.write_text(json.dumps(raw, indent=2, default=str))
        target_s5 = CANON_S5 / rid / "result.json"
        target_s5.parent.mkdir(parents=True, exist_ok=True)
        stage5a = json.loads(Path(win["stage5a_path"]).read_text())
        stage5a["reaction_id"] = rid
        stage5a["fragmentation_revision"] = 4
        stage5a["selected_candidate"] = win["label"]
        target_s5.write_text(json.dumps(stage5a, indent=2))

    out_path = ROOT / "Validate" / "refrag" / "selection_report.json"
    out_path.write_text(json.dumps({
        "n_total": len(rids),
        "n_with_winner": len(winners),
        "n_no_results": len(no_results),
        "no_results": no_results,
        "winners": winners,
    }, indent=2, default=str))

    # Console summary
    by_label: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for w in winners:
        by_label[w["winner"]["label"]] = by_label.get(w["winner"]["label"], 0) + 1
        by_verdict[w["winner"]["verdict"]] = by_verdict.get(w["winner"]["verdict"], 0) + 1
    print()
    print(f"Reactions with winner picked: {len(winners)}/{len(rids)}")
    print(f"No candidate result yet:      {len(no_results)}")
    print()
    print("Winners by label:")
    for k, v in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"  {k:24s}: {v}")
    print()
    print("Winners by verdict:")
    for k, v in sorted(by_verdict.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()
    print(f"Full report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
