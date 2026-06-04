"""Phase 1.5 — Comprehensive Fragment Review Tool (Flask app).

Usage:
    cd tools/phase1_5_review
    flask run --port 8888 --host 0.0.0.0
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from flask_cors import CORS

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from data_loader import (  # noqa: E402
    PHASE15_DIR,
    PROGRESS_JSON,
    SNAPSHOT_DIR,
    REVIEW_LOG,
    all_reactions,
    all_reviews,
    append_audit,
    ensure_loaded,
    get_review,
    get_static,
    progress_summary,
    save_review_log,
    update_review,
)
from validation import validate_fragment  # noqa: E402

import stage5a_loader  # noqa: E402

# Configure logging once.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(THIS_DIR.parent.parent / "logs" / "phase1_5.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("phase1_5")

app = Flask(__name__, template_folder=str(THIS_DIR / "templates"), static_folder=str(THIS_DIR / "static"))
CORS(app)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reviewer_name() -> str:
    return os.environ.get("REVIEWER", os.environ.get("USER", "anonymous"))


@app.before_request
def _ensure():
    ensure_loaded()


# --------------------------------------------------------------------------- #
# Static routing for vendored libraries
# --------------------------------------------------------------------------- #


@app.route("/lib/<path:fname>")
def lib(fname: str):
    return send_from_directory(THIS_DIR / "static" / "lib", fname)


# --------------------------------------------------------------------------- #
# HTML routes
# --------------------------------------------------------------------------- #


@app.route("/")
def root():
    return redirect("/stage5a", code=302)


@app.route("/phase1_5")
def phase1_5_dashboard():
    return render_template("dashboard.html")


@app.route("/reaction/<rxn_id>")
def review_page(rxn_id: str):
    static = get_static(rxn_id)
    if static is None:
        return jsonify({"error": "unknown reaction id"}), 404
    case = get_review(rxn_id)["case"]
    template = "scratch.html" if case == "C" and not get_review(rxn_id)["current_definition"] else "review.html"
    return render_template(template, rxn_id=rxn_id)


# --------------------------------------------------------------------------- #
# Stage 5-A review (new P0/P1/P2/P3 fragmentation pass)
# --------------------------------------------------------------------------- #


@app.route("/stage5a")
def stage5a_dashboard():
    stage5a_loader.ensure_loaded()
    return render_template("stage5a_dashboard.html")


@app.route("/stage5a/<rxn_id>")
def stage5a_review_page(rxn_id: str):
    stage5a_loader.ensure_loaded()
    payload = stage5a_loader.get_reaction_payload(rxn_id)
    if payload is None:
        return jsonify({"error": "unknown reaction id"}), 404
    return render_template("stage5a_review.html", rxn_id=rxn_id)


@app.route("/api/stage5a/state")
def api_stage5a_state():
    stage5a_loader.ensure_loaded()
    return jsonify({
        "reactions": stage5a_loader.list_reactions(),
        "progress": stage5a_loader.progress(),
    })


@app.route("/api/stage5a/reaction/<rxn_id>")
def api_stage5a_reaction(rxn_id: str):
    stage5a_loader.ensure_loaded()
    payload = stage5a_loader.get_reaction_payload(rxn_id)
    if payload is None:
        return jsonify({"error": "unknown reaction id"}), 404
    return jsonify(payload)


@app.route("/api/stage5a/decision/<rxn_id>", methods=["POST"])
def api_stage5a_decision(rxn_id: str):
    stage5a_loader.ensure_loaded()
    payload = stage5a_loader.get_reaction_payload(rxn_id)
    if payload is None:
        return jsonify({"error": "unknown reaction id"}), 404
    body = request.get_json(force=True)
    status = body.get("status")
    if status not in {"accepted", "modified", "rejected", "bookmarked"}:
        return jsonify({"error": f"invalid status: {status}"}), 400

    patch: dict = {
        "review_status": status if status != "bookmarked"
                          else payload["review"].get("review_status", "not_reviewed"),
        "reviewer": _reviewer_name(),
        "review_completed_at": _now(),
    }
    if status == "bookmarked":
        patch["bookmarked"] = True
    rec = stage5a_loader.update_review(rxn_id, patch)
    stage5a_loader.append_audit({
        "ts": _now(),
        "rxn": rxn_id,
        "action": "stage5a_decision",
        "reviewer": _reviewer_name(),
        "status": status,
    })
    return jsonify({"ok": True, "review": rec})


# --------------------------------------------------------------------------- #
# JSON API
# --------------------------------------------------------------------------- #


@app.route("/api/state")
def api_state():
    """Return the per-reaction summary used by the dashboard."""
    rows = []
    for rxn_id in sorted(all_reactions()):
        static = get_static(rxn_id)
        rec = get_review(rxn_id)
        rows.append({
            "rxn_id": rxn_id,
            "source": static.source,
            "case": rec["case"],
            "n_heavy_atoms": static.n_heavy_atoms,
            "activation_energy": static.activation_energy,
            "auto_confidence": (rec.get("auto_suggestion") or {}).get("auto_confidence"),
            "review_status": rec["review_status"],
            "bookmarked": bool(rec.get("bookmarked")),
            "reviewer": rec["review_metadata"]["reviewer"],
            "review_completed_at": rec["review_metadata"]["review_completed_at"],
            "rationale_present": bool((rec["review_metadata"]["rationale"] or "").strip()),
        })
    return jsonify({"reactions": rows, "progress": progress_summary()})


@app.route("/api/structure/<rxn_id>")
def api_structure(rxn_id: str):
    """Send everything the front-end 3D viewers need for a reaction."""
    static = get_static(rxn_id)
    if static is None:
        return jsonify({"error": "unknown reaction"}), 404
    rec = get_review(rxn_id)
    return jsonify({
        "rxn_id": rxn_id,
        "source": static.source,
        "case": rec["case"],
        "n_atoms": static.n_atoms,
        "n_heavy_atoms": static.n_heavy_atoms,
        "formula": static.formula,
        "activation_energy": static.activation_energy,
        "energies_5pts": static.energies_5pts,
        "energy_R": static.energy_R,
        "energy_TS": static.energy_TS,
        "energy_P": static.energy_P,
        "numbers": static.numbers,
        "coords_5pts": static.coords_5pts,  # shape (5, N, 3)
        "coords_P": static.coords_P,  # (N, 3) if extracted, else null
        "bonds_broken": static.bonds_broken,
        "bonds_formed": static.bonds_formed,
        "auto_suggestion": rec["auto_suggestion"],
        "current_definition": rec["current_definition"],
        "review_status": rec["review_status"],
        "review_metadata": rec["review_metadata"],
        "modification_history": rec["modification_history"],
        "bookmarked": bool(rec.get("bookmarked")),
    })


@app.route("/api/validate/<rxn_id>", methods=["POST"])
def api_validate(rxn_id: str):
    static = get_static(rxn_id)
    if static is None:
        return jsonify({"error": "unknown reaction"}), 404
    body = request.get_json(force=True)
    f1 = list(map(int, body.get("frag1_atoms", [])))
    f2 = list(map(int, body.get("frag2_atoms", [])))
    rep = validate_fragment(static.numbers, static.coords_5pts[0], f1, f2)
    return jsonify(rep.as_dict())


@app.route("/api/recompute_caps/<rxn_id>", methods=["POST"])
def api_recompute_caps(rxn_id: str):
    """Same as /api/validate but only returns h_caps + smiles (lightweight)."""
    static = get_static(rxn_id)
    if static is None:
        return jsonify({"error": "unknown reaction"}), 404
    body = request.get_json(force=True)
    f1 = list(map(int, body.get("frag1_atoms", [])))
    f2 = list(map(int, body.get("frag2_atoms", [])))
    rep = validate_fragment(static.numbers, static.coords_5pts[0], f1, f2)
    return jsonify({
        "h_caps": rep.h_caps,
        "frag1_smiles": rep.frag1_smiles,
        "frag2_smiles": rep.frag2_smiles,
        "frag1_formula": rep.frag1_formula,
        "frag2_formula": rep.frag2_formula,
    })


@app.route("/api/decision/<rxn_id>", methods=["POST"])
def api_decision(rxn_id: str):
    static = get_static(rxn_id)
    if static is None:
        return jsonify({"error": "unknown reaction"}), 404
    body = request.get_json(force=True)

    status = body.get("status")  # accepted / modified / rejected / bookmarked
    if status not in {"accepted", "modified", "rejected", "bookmarked"}:
        return jsonify({"error": f"invalid status: {status}"}), 400

    rec = get_review(rxn_id)
    rec.setdefault("modification_history", []).append({
        "timestamp": _now(),
        "action": "save_decision",
        "status_before": rec["review_status"],
        "status_after": status,
    })

    rationale = (body.get("rationale") or "").strip()
    confidence = body.get("confidence")
    override = bool(body.get("override", False))
    is_scratch = body.get("is_scratch", False)

    min_rationale = 50 if (status == "rejected" or is_scratch) else 10
    if status == "bookmarked":
        min_rationale = 0
    if len(rationale) < min_rationale:
        return jsonify({
            "error": f"rationale too short ({len(rationale)} chars; required >= {min_rationale})"
        }), 400

    f1 = list(map(int, body.get("frag1_atoms", [])))
    f2 = list(map(int, body.get("frag2_atoms", [])))

    if status in ("accepted", "modified"):
        rep = validate_fragment(static.numbers, static.coords_5pts[0], f1, f2)
        if not rep.ok and not override:
            return jsonify({
                "error": "validation failed",
                "validation": rep.as_dict(),
                "hint": "set override=true with a longer rationale to force-save",
            }), 400
        rec["current_definition"] = {
            "frag1_atoms": f1,
            "frag2_atoms": f2,
            "h_caps": rep.h_caps,
            "frag1_smiles": rep.frag1_smiles,
            "frag2_smiles": rep.frag2_smiles,
            "frag1_formula": rep.frag1_formula,
            "frag2_formula": rep.frag2_formula,
            "frag1_charge": int(body.get("frag1_charge", 0)),
            "frag2_charge": int(body.get("frag2_charge", 0)),
            "frag1_multiplicity": int(body.get("frag1_multiplicity", 1)),
            "frag2_multiplicity": int(body.get("frag2_multiplicity", 1)),
        }
        rec["review_metadata"]["validation_warnings"] = rep.warnings
        rec["review_metadata"]["validated"] = rep.ok
        rec["review_metadata"]["override_used"] = override and not rep.ok
        validation_payload = rep.as_dict()
    elif status == "rejected":
        rec["current_definition"] = None
        rec["review_metadata"]["validation_warnings"] = []
        rec["review_metadata"]["validated"] = True
        rec["review_metadata"]["override_used"] = False
        validation_payload = {"ok": True, "errors": [], "warnings": []}
    else:  # bookmarked
        rec["bookmarked"] = True
        validation_payload = {"ok": True, "errors": [], "warnings": []}

    started = rec["review_metadata"].get("review_started_at") or _now()
    rec["review_metadata"]["reviewer"] = _reviewer_name()
    rec["review_metadata"]["review_started_at"] = started
    rec["review_metadata"]["review_completed_at"] = _now()
    rec["review_metadata"]["rationale"] = rationale
    rec["review_metadata"]["confidence"] = confidence
    rec["review_metadata"]["is_scratch"] = is_scratch
    rec["review_status"] = status if status != "bookmarked" else rec["review_status"]
    if status == "bookmarked":
        rec["bookmarked"] = True

    update_review(rxn_id, rec)

    append_audit({
        "ts": _now(),
        "rxn": rxn_id,
        "action": "save_decision",
        "reviewer": _reviewer_name(),
        "status": status,
        "details": {
            "frag1_atoms": f1,
            "frag2_atoms": f2,
            "rationale_len": len(rationale),
            "validation_ok": validation_payload.get("ok"),
            "override": override,
        },
    })

    _maybe_snapshot()
    return jsonify({"ok": True, "validation": validation_payload, "review": rec})


@app.route("/api/bookmark/<rxn_id>", methods=["POST"])
def api_bookmark(rxn_id: str):
    rec = get_review(rxn_id)
    if rec is None:
        return jsonify({"error": "unknown reaction"}), 404
    rec["bookmarked"] = bool(request.get_json(force=True).get("bookmarked", True))
    update_review(rxn_id, rec)
    append_audit({"ts": _now(), "rxn": rxn_id, "action": "bookmark",
                  "reviewer": _reviewer_name(), "details": {"bookmarked": rec["bookmarked"]}})
    return jsonify({"ok": True, "bookmarked": rec["bookmarked"]})


@app.route("/api/open/<rxn_id>", methods=["POST"])
def api_open(rxn_id: str):
    """Reviewer opened the reaction page — record the start time once."""
    rec = get_review(rxn_id)
    if rec is None:
        return jsonify({"error": "unknown reaction"}), 404
    if not rec["review_metadata"].get("review_started_at"):
        rec["review_metadata"]["review_started_at"] = _now()
        update_review(rxn_id, rec)
    append_audit({"ts": _now(), "rxn": rxn_id, "action": "open",
                  "reviewer": _reviewer_name()})
    return jsonify({"ok": True})


@app.route("/api/export", methods=["POST"])
def api_export():
    """Refresh the progress snapshot + return paths."""
    PROGRESS_JSON.write_text(json.dumps(progress_summary(), indent=2))
    return jsonify({
        "review_log": str(REVIEW_LOG),
        "progress": str(PROGRESS_JSON),
        "snapshot_dir": str(SNAPSHOT_DIR),
    })


# --------------------------------------------------------------------------- #
# Snapshot helper (every ~5 minutes worth of activity)
# --------------------------------------------------------------------------- #


_LAST_SNAPSHOT_TS: float = 0.0


def _maybe_snapshot():
    """Write a timestamped backup of review_log_complete.json every ~5 min."""
    import time
    global _LAST_SNAPSHOT_TS
    now = time.time()
    if now - _LAST_SNAPSHOT_TS < 300:
        return
    _LAST_SNAPSHOT_TS = now
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target = SNAPSHOT_DIR / f"review_log_{stamp}.json"
    target.write_text(json.dumps(all_reviews(), indent=2))
    log.info("snapshot written: %s", target)


if __name__ == "__main__":
    ensure_loaded()
    app.run(host="0.0.0.0", port=8888, debug=False)
