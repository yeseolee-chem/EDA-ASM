#!/usr/bin/env python3
"""Validate ASR JSON outputs against the ASR validation spec."""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any


_COMPONENT_KEYS: tuple = ("strain", "elst", "Pauli", "oi", "disp")
_POINT_KEYS: tuple = ("R", "TS", "P")


@dataclass
class Config:
    """Validator thresholds and constants (kcal/mol unless noted)."""
    tau_pass: float = 0.1
    tau_fail: float = 0.5
    de_act_min: float = 0.0
    de_act_warn_max: float = 300.0
    robust_z_warn: float = 5.0
    energy_z_warn: float = 6.0
    required_settings: tuple = ("functional", "basis", "dispersion",
                                "relativity", "frozen_core", "integration")


@dataclass
class Issue:
    """One validation finding."""
    check_no: int
    level: str
    code: str
    detail: str


@dataclass
class Derived:
    """Per-file derived quantities."""
    rid: str
    raw: dict
    path: str
    E: dict
    comp: dict
    sigma: dict
    E_frag: float | None
    dE_act: float | None
    dE_rxn: float | None
    dX_TS: dict
    dX_P: dict
    offset: dict
    max_abs_res_cons: float | None
    max_abs_res_ref: float | None
    offset_spread: float | None
    n_points: int
    n_components_ok: int
    schema_errors: list = field(default_factory=list)


def _is_num(x: Any) -> bool:
    """True iff x is a real, finite (non-bool) number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def robust_z(x: float, values: list[float]) -> float:
    """Robust z-score using median + MAD; returns 0 when MAD is 0."""
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    if mad == 0:
        return 0.0
    return 0.6745 * (x - med) / mad


def load_records(input_dir: str) -> list[tuple[str, dict]]:
    """Load all *.json files in input_dir; returns (path, raw) list."""
    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    out: list[tuple[str, dict]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                raw = {"__load_error__": "JSON root is not an object"}
        except Exception as exc:
            raw = {"__load_error__": f"{type(exc).__name__}: {exc}"}
        out.append((p, raw))
    return out


def derive(path: str, raw: dict) -> Derived:
    """Compute per-file derived quantities; accumulates schema errors instead of raising."""
    errors: list[Issue] = []
    rid_val = raw.get("reaction_id") if isinstance(raw, dict) else None
    rid = rid_val if isinstance(rid_val, str) and rid_val else ""
    if not rid:
        errors.append(Issue(1, "FAIL", "missing_id",
                            f"reaction_id missing or empty in {os.path.basename(path)}"))

    if "__load_error__" in raw:
        errors.append(Issue(1, "FAIL", "non_finite", raw["__load_error__"]))
        return Derived(rid=rid or os.path.basename(path),
                       raw=raw, path=path,
                       E={}, comp={}, sigma={}, E_frag=None,
                       dE_act=None, dE_rxn=None, dX_TS={}, dX_P={},
                       offset={}, max_abs_res_cons=None,
                       max_abs_res_ref=None, offset_spread=None,
                       n_points=0, n_components_ok=0,
                       schema_errors=errors)

    irc_raw = raw.get("irc_points")
    irc: dict = irc_raw if isinstance(irc_raw, dict) else {}
    asr_raw = raw.get("asr_vector_kcal")
    asr: dict = asr_raw if isinstance(asr_raw, dict) else {}

    for pt in _POINT_KEYS:
        if pt not in irc or not isinstance(irc.get(pt), dict):
            errors.append(Issue(1, "FAIL", "missing_point",
                                f"irc_points.{pt} missing or non-dict"))
        if pt not in asr or not isinstance(asr.get(pt), dict):
            errors.append(Issue(1, "FAIL", "missing_vector",
                                f"asr_vector_kcal.{pt} missing or non-dict"))

    E: dict = {}
    for pt in _POINT_KEYS:
        if not isinstance(irc.get(pt), dict):
            continue
        v = irc[pt].get("energy_kcal_adf")
        if v is None:
            errors.append(Issue(1, "FAIL", "missing_point",
                                f"irc_points.{pt}.energy_kcal_adf missing"))
        elif not _is_num(v):
            errors.append(Issue(1, "FAIL", "non_finite",
                                f"irc_points.{pt}.energy_kcal_adf not finite ({v!r})"))
        else:
            E[pt] = float(v)

    comp: dict = {}
    sigma: dict = {}
    for pt in _POINT_KEYS:
        if not isinstance(asr.get(pt), dict):
            continue
        cdict: dict = {}
        ok_all = True
        for k in _COMPONENT_KEYS:
            if k not in asr[pt]:
                errors.append(Issue(1, "FAIL", "missing_component",
                                    f"asr_vector_kcal.{pt}.{k} missing"))
                ok_all = False
                continue
            v = asr[pt][k]
            if not _is_num(v):
                errors.append(Issue(1, "FAIL", "non_finite",
                                    f"asr_vector_kcal.{pt}.{k} not finite ({v!r})"))
                ok_all = False
                continue
            cdict[k] = float(v)
        if ok_all:
            comp[pt] = cdict
            sigma[pt] = sum(cdict.values())

    fdict = raw.get("fragment_opt_energy_kcal")
    E_frag: float | None = None
    if not isinstance(fdict, dict) or len(fdict) == 0:
        errors.append(Issue(1, "FAIL", "missing_fragment",
                            "fragment_opt_energy_kcal missing or empty"))
    else:
        total = 0.0
        ok = True
        for k, v in fdict.items():
            if not _is_num(v):
                errors.append(Issue(1, "FAIL", "non_finite",
                                    f"fragment_opt_energy_kcal.{k} not finite ({v!r})"))
                ok = False
                continue
            total += float(v)
        if ok:
            E_frag = total

    adf = raw.get("adf_settings")
    if not isinstance(adf, dict):
        for key in Config.required_settings:
            errors.append(Issue(1, "FAIL", "missing_setting",
                                f"adf_settings.{key} missing (adf_settings absent)"))
    else:
        for key in Config.required_settings:
            if key not in adf or adf[key] in (None, ""):
                errors.append(Issue(1, "FAIL", "missing_setting",
                                    f"adf_settings.{key} missing"))

    dE_act = E["TS"] - E["R"] if ("TS" in E and "R" in E) else None
    dE_rxn = E["P"] - E["R"] if ("P" in E and "R" in E) else None

    dX_TS: dict = {}
    dX_P: dict = {}
    if "TS" in comp and "R" in comp:
        for k in _COMPONENT_KEYS:
            dX_TS[k] = comp["TS"][k] - comp["R"][k]
    if "P" in comp and "R" in comp:
        for k in _COMPONENT_KEYS:
            dX_P[k] = comp["P"][k] - comp["R"][k]

    offset: dict = {}
    for pt in _POINT_KEYS:
        if pt in E and pt in sigma:
            offset[pt] = E[pt] - sigma[pt]

    res_cons_terms: list[float] = []
    if "TS" in sigma and "R" in sigma and dE_act is not None:
        res_cons_terms.append(abs((sigma["TS"] - sigma["R"]) - dE_act))
    if "P" in sigma and "R" in sigma and dE_rxn is not None:
        res_cons_terms.append(abs((sigma["P"] - sigma["R"]) - dE_rxn))
    max_abs_res_cons = max(res_cons_terms) if res_cons_terms else None

    if offset and E_frag is not None:
        diffs = [abs(offset[pt] - E_frag) for pt in offset]
        max_abs_res_ref = max(diffs)
    else:
        max_abs_res_ref = None

    offset_spread = (max(offset.values()) - min(offset.values())) if offset else None

    n_points = sum(1 for pt in _POINT_KEYS if pt in E)
    n_components_ok = sum(1 for pt in _POINT_KEYS if pt in comp)

    return Derived(rid=rid or os.path.basename(path),
                   raw=raw, path=path,
                   E=E, comp=comp, sigma=sigma, E_frag=E_frag,
                   dE_act=dE_act, dE_rxn=dE_rxn,
                   dX_TS=dX_TS, dX_P=dX_P, offset=offset,
                   max_abs_res_cons=max_abs_res_cons,
                   max_abs_res_ref=max_abs_res_ref,
                   offset_spread=offset_spread,
                   n_points=n_points,
                   n_components_ok=n_components_ok,
                   schema_errors=errors)


def check1_schema(d: Derived) -> list[Issue]:
    """Per-file schema and integrity (excludes batch-level duplicate_id)."""
    issues: list[Issue] = list(d.schema_errors)
    raw = d.raw
    sq = raw.get("status_at_queue") if isinstance(raw, dict) else None
    rn = raw.get("recovery_note") if isinstance(raw, dict) else None
    has_rn = isinstance(rn, str) and rn != ""
    if (sq is not None and sq != "OK") or has_rn:
        detail = f"status_at_queue={sq!r}, recovery_note_present={has_rn}"
        issues.append(Issue(1, "WARN", "recovered", detail))
    return issues


def check1_duplicates(ds: list[Derived]) -> dict[str, list[Issue]]:
    """Batch: detect duplicate reaction_ids; returns rid -> issues to attach to every matching file."""
    out: dict[str, list[Issue]] = {}
    counts = collections.Counter(d.rid for d in ds if d.rid)
    for rid, n in counts.items():
        if n > 1:
            out[rid] = [Issue(1, "FAIL", "duplicate_id",
                              f"reaction_id appears {n} times in batch")]
    return out


def check2_homogeneity(ds: list[Derived], cfg: Config) -> dict[str, list[Issue]]:
    """Batch: every file's adf_settings must match the per-key mode of the batch."""
    out: dict[str, list[Issue]] = collections.defaultdict(list)
    mode: dict[str, Any] = {}
    for key in cfg.required_settings:
        vals: list = []
        for d in ds:
            adf = d.raw.get("adf_settings") if isinstance(d.raw, dict) else None
            if isinstance(adf, dict) and key in adf:
                vals.append(adf[key])
        if not vals:
            continue
        counter = collections.Counter(repr(v) for v in vals)
        top_repr, _ = counter.most_common(1)[0]
        for v in vals:
            if repr(v) == top_repr:
                mode[key] = v
                break
    for d in ds:
        adf = d.raw.get("adf_settings") if isinstance(d.raw, dict) else None
        if not isinstance(adf, dict):
            continue
        for key, mode_val in mode.items():
            if key in adf and adf[key] != mode_val:
                out[d.path].append(Issue(2, "FAIL", f"level_mismatch:{key}",
                                          f"{adf[key]!r} != mode {mode_val!r}"))
    return dict(out)


def check3_topology(d: Derived, cfg: Config) -> list[Issue]:
    """Per-file barrier topology and barrier sanity (energy outlier handled in batch)."""
    issues: list[Issue] = []
    if "TS" in d.E and "R" in d.E and "P" in d.E:
        if d.E["TS"] <= d.E["R"] or d.E["TS"] <= d.E["P"]:
            issues.append(Issue(3, "FAIL", "ts_not_max",
                                f"E_TS={d.E['TS']:.6f}, E_R={d.E['R']:.6f}, E_P={d.E['P']:.6f}"))
    if d.dE_act is not None:
        if d.dE_act <= cfg.de_act_min:
            issues.append(Issue(3, "FAIL", "barrier_nonpositive",
                                f"dE_act={d.dE_act:.6f} <= {cfg.de_act_min}"))
        elif d.dE_act > cfg.de_act_warn_max:
            issues.append(Issue(3, "WARN", "barrier_high",
                                f"dE_act={d.dE_act:.6f} > {cfg.de_act_warn_max}"))
    return issues


def check3_magnitude(ds: list[Derived], cfg: Config) -> dict[str, list[Issue]]:
    """Batch: flag files whose E[R] is a robust-z outlier."""
    out: dict[str, list[Issue]] = collections.defaultdict(list)
    er_vals = [d.E["R"] for d in ds if "R" in d.E]
    if not er_vals:
        return dict(out)
    for d in ds:
        if "R" not in d.E:
            continue
        z = robust_z(d.E["R"], er_vals)
        if abs(z) > cfg.energy_z_warn:
            out[d.path].append(Issue(3, "WARN", "energy_outlier",
                                      f"E_R={d.E['R']:.6f}, robust_z={z:.3f}"))
    return dict(out)


def check4_conservation(d: Derived, cfg: Config) -> list[Issue]:
    """Per-file conservation + fragment-reference residual band."""
    issues: list[Issue] = []
    if d.max_abs_res_cons is None or d.max_abs_res_ref is None or d.offset_spread is None:
        return issues
    m = max(d.max_abs_res_cons, d.max_abs_res_ref, d.offset_spread)
    detail = (f"max_abs_res_cons={d.max_abs_res_cons:.6f}, "
              f"max_abs_res_ref={d.max_abs_res_ref:.6f}, "
              f"offset_spread={d.offset_spread:.6f}")
    if m > cfg.tau_fail:
        issues.append(Issue(4, "FAIL", "conservation_fail", detail))
    elif m > cfg.tau_pass:
        issues.append(Issue(4, "WARN", "conservation_soft", detail))
    return issues


def check5_signs(d: Derived) -> list[Issue]:
    """Per-file Delta(TS-R) sign sanity for strain, Pauli, oi."""
    issues: list[Issue] = []
    if not d.dX_TS:
        return issues
    if "strain" in d.dX_TS and d.dX_TS["strain"] <= 0:
        issues.append(Issue(5, "WARN", "sign_strain",
                            f"dX_TS[strain]={d.dX_TS['strain']:.6f} <= 0"))
    if "Pauli" in d.dX_TS and d.dX_TS["Pauli"] <= 0:
        issues.append(Issue(5, "WARN", "sign_pauli",
                            f"dX_TS[Pauli]={d.dX_TS['Pauli']:.6f} <= 0"))
    if "oi" in d.dX_TS and d.dX_TS["oi"] >= 0:
        issues.append(Issue(5, "WARN", "sign_oi",
                            f"dX_TS[oi]={d.dX_TS['oi']:.6f} >= 0"))
    return issues


def check5_distribution(ds: list[Derived], cfg: Config) -> dict[str, list[Issue]]:
    """Batch: per-component robust-z outliers on dX_TS[k]."""
    out: dict[str, list[Issue]] = collections.defaultdict(list)
    for k in _COMPONENT_KEYS:
        vals = [d.dX_TS[k] for d in ds if k in d.dX_TS]
        if not vals:
            continue
        for d in ds:
            if k not in d.dX_TS:
                continue
            z = robust_z(d.dX_TS[k], vals)
            if abs(z) > cfg.robust_z_warn:
                out[d.path].append(Issue(5, "WARN", f"dist_outlier:{k}",
                                          f"dX_TS[{k}]={d.dX_TS[k]:.6f}, robust_z={z:.3f}"))
    return dict(out)


def aggregate(issues: list[Issue]) -> str:
    """Reduce per-file issues to PASS/WARN/FAIL."""
    if any(i.level == "FAIL" for i in issues):
        return "FAIL"
    if any(i.level == "WARN" for i in issues):
        return "WARN"
    return "PASS"


def write_manifest(rows: list[dict], out_path: str) -> None:
    """Write manifest CSV with the exact column set defined by the spec."""
    cols = ["reaction_id", "verdict", "failed_checks", "warn_flags",
            "dE_act_kcal", "dE_rxn_kcal",
            "max_abs_res_cons_kcal", "max_abs_res_ref_kcal", "offset_spread_kcal",
            "recovered", "level_ok", "n_points", "n_components_ok",
            "dStrain_TS", "dElst_TS", "dPauli_TS", "dOi_TS", "dDisp_TS"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


def _fmt_float(x: Any) -> Any:
    """Format a float to 6 decimal places, or empty string if not computable."""
    if x is None:
        return ""
    try:
        return round(float(x), 6)
    except (TypeError, ValueError):
        return ""


def _evaluate(derived: list[Derived], cfg: Config) -> tuple[list[dict], dict]:
    """Run all per-file and batch checks; return rows for manifest and summary stats."""
    per_file: dict[str, list[Issue]] = {d.path: [] for d in derived}
    for d in derived:
        per_file[d.path].extend(check1_schema(d))
        per_file[d.path].extend(check3_topology(d, cfg))
        per_file[d.path].extend(check4_conservation(d, cfg))
        per_file[d.path].extend(check5_signs(d))

    dup_by_rid = check1_duplicates(derived)
    hom_by_path = check2_homogeneity(derived, cfg)
    mag_by_path = check3_magnitude(derived, cfg)
    dist_by_path = check5_distribution(derived, cfg)

    for d in derived:
        per_file[d.path].extend(dup_by_rid.get(d.rid, []))
        per_file[d.path].extend(hom_by_path.get(d.path, []))
        per_file[d.path].extend(mag_by_path.get(d.path, []))
        per_file[d.path].extend(dist_by_path.get(d.path, []))

    rows: list[dict] = []
    pass_n = warn_n = fail_n = 0
    recovered_warn_n = 0
    by_check: collections.Counter = collections.Counter()

    for d in derived:
        issues = per_file[d.path]
        verdict = aggregate(issues)
        if verdict == "PASS":
            pass_n += 1
        elif verdict == "WARN":
            warn_n += 1
        else:
            fail_n += 1

        failed_check_nums = sorted({i.check_no for i in issues if i.level == "FAIL"})
        failed_str = ";".join(str(c) for c in failed_check_nums)

        warn_codes_seen: list[str] = []
        seen: set = set()
        for i in issues:
            if i.level == "WARN" and i.code not in seen:
                warn_codes_seen.append(i.code)
                seen.add(i.code)
        warn_str = ";".join(warn_codes_seen)

        for cn in failed_check_nums:
            by_check[cn] += 1

        recovered = any(i.level == "WARN" and i.code == "recovered" for i in issues)
        if recovered:
            recovered_warn_n += 1
        level_ok = not any(i.check_no == 2 and i.level == "FAIL" for i in issues)

        rows.append({
            "reaction_id": d.rid,
            "verdict": verdict,
            "failed_checks": failed_str,
            "warn_flags": warn_str,
            "dE_act_kcal": _fmt_float(d.dE_act),
            "dE_rxn_kcal": _fmt_float(d.dE_rxn),
            "max_abs_res_cons_kcal": _fmt_float(d.max_abs_res_cons),
            "max_abs_res_ref_kcal": _fmt_float(d.max_abs_res_ref),
            "offset_spread_kcal": _fmt_float(d.offset_spread),
            "recovered": recovered,
            "level_ok": level_ok,
            "n_points": d.n_points,
            "n_components_ok": d.n_components_ok,
            "dStrain_TS": _fmt_float(d.dX_TS.get("strain")),
            "dElst_TS": _fmt_float(d.dX_TS.get("elst")),
            "dPauli_TS": _fmt_float(d.dX_TS.get("Pauli")),
            "dOi_TS": _fmt_float(d.dX_TS.get("oi")),
            "dDisp_TS": _fmt_float(d.dX_TS.get("disp")),
        })

    summary = {
        "n": len(derived),
        "pass": pass_n,
        "warn": warn_n,
        "fail": fail_n,
        "recovered": recovered_warn_n,
        "by_check": by_check,
    }
    return rows, summary


def _selftest() -> bool:
    """Inline tests over the bundled sample reaction and four synthetic mutants."""
    cfg = Config()
    sample = {
        "reaction_id": "Halogen_BrC6H4N_rxn11968",
        "schema_version": "asr_spec_v1_recovered",
        "adf_settings": {
            "functional": "BP86", "dispersion": "D3BJ", "basis": "TZ2P",
            "frozen_core": "None", "relativity": "ZORA_scalar",
            "integration": "Becke_Good",
        },
        "irc_points": {
            "R":  {"energy_kcal_adf": -1741.3906252869203},
            "TS": {"energy_kcal_adf": -1621.805591960825},
            "P":  {"energy_kcal_adf": -1703.229606096811},
        },
        "fragment_opt_energy_kcal": {
            "migrating_H": -21.878873479933613,
            "scaffold": -1594.8840811408504,
        },
        "asr_vector_kcal": {
            "R":  {"strain": 3.0294095779105277, "elst": -60.446426150285625,
                   "Pauli": 67.24785813501082, "oi": -133.73571525541396,
                   "disp": -0.7211295409716827},
            "TS": {"strain": 48.84048021723714, "elst": -82.70567188640374,
                   "Pauli": 223.3681472119086, "oi": -193.62655881575523,
                   "disp": -0.9238113075387213},
            "P":  {"strain": 25.330251817190707, "elst": -65.84252810441214,
                   "Pauli": 99.97484063606382, "oi": -144.94897039665477,
                   "disp": -0.9775775262516391},
        },
        "status_at_queue": "RECOVERED_FROM_RKF",
        "recovery_note": "Result JSON re-generated from /tmp workdir rkfs (orig JSON was deleted).",
    }

    failures: list[str] = []

    def _ck(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    d = derive("sample.json", sample)
    _ck(abs(d.dE_act - 119.585033) < 1e-3, f"dE_act={d.dE_act}")
    _ck(abs(d.dE_rxn - 38.161019) < 1e-3, f"dE_rxn={d.dE_rxn}")
    _ck(abs(d.sigma["TS"] - (-5.047415)) < 1e-3, f"sigma_TS={d.sigma['TS']}")
    _ck(abs(d.sigma["R"] - (-124.626003)) < 1e-3, f"sigma_R={d.sigma['R']}")
    _ck(abs(d.E_frag - (-1616.762955)) < 1e-3, f"E_frag={d.E_frag}")
    _ck(abs(d.max_abs_res_cons - 0.006445) < 1e-3, f"res_cons={d.max_abs_res_cons}")
    _ck(abs(d.max_abs_res_ref - 0.004777) < 1e-3, f"res_ref={d.max_abs_res_ref}")
    _ck(abs(d.offset_spread - 0.007445) < 1e-3, f"offset_spread={d.offset_spread}")
    _ck(d.max_abs_res_cons <= cfg.tau_pass, "max_abs_res_cons not within tau_pass")
    _ck(d.dX_TS["strain"] > 0, f"dStrain_TS={d.dX_TS['strain']}")
    _ck(d.dX_TS["Pauli"] > 0, f"dPauli_TS={d.dX_TS['Pauli']}")
    _ck(d.dX_TS["oi"] < 0, f"dOi_TS={d.dX_TS['oi']}")
    _ck(d.E["TS"] > d.E["R"] and d.E["TS"] > d.E["P"], "TS not max")

    issues = (check1_schema(d) + check3_topology(d, cfg)
              + check4_conservation(d, cfg) + check5_signs(d))
    verdict = aggregate(issues)
    failed_checks = ";".join(str(c) for c in sorted({i.check_no for i in issues if i.level == "FAIL"}))
    warn_flags = ";".join(i.code for i in issues if i.level == "WARN")
    _ck(verdict == "WARN", f"verdict={verdict}")
    _ck("recovered" in warn_flags, f"warn_flags={warn_flags}")
    _ck(failed_checks == "", f"failed_checks={failed_checks}")

    s2 = json.loads(json.dumps(sample))
    s2["irc_points"]["TS"]["energy_kcal_adf"] = s2["irc_points"]["R"]["energy_kcal_adf"] - 1.0
    d2 = derive("s2.json", s2)
    i2 = (check1_schema(d2) + check3_topology(d2, cfg)
          + check4_conservation(d2, cfg) + check5_signs(d2))
    v2 = aggregate(i2)
    fc2 = sorted({i.check_no for i in i2 if i.level == "FAIL"})
    _ck(v2 == "FAIL", f"case1 verdict={v2}")
    _ck(3 in fc2, f"case1 failed_checks={fc2}")
    _ck(any(i.code == "ts_not_max" for i in i2), "case1 missing ts_not_max code")

    s3 = json.loads(json.dumps(sample))
    s3["asr_vector_kcal"]["TS"]["Pauli"] += 200.0
    d3 = derive("s3.json", s3)
    i3 = (check1_schema(d3) + check3_topology(d3, cfg)
          + check4_conservation(d3, cfg) + check5_signs(d3))
    v3 = aggregate(i3)
    fc3 = sorted({i.check_no for i in i3 if i.level == "FAIL"})
    _ck(d3.max_abs_res_cons > cfg.tau_fail, f"case2 res_cons={d3.max_abs_res_cons}")
    _ck(v3 == "FAIL", f"case2 verdict={v3}")
    _ck(4 in fc3, f"case2 failed_checks={fc3}")

    s4 = json.loads(json.dumps(sample))
    del s4["asr_vector_kcal"]["TS"]["disp"]
    d4 = derive("s4.json", s4)
    i4 = (check1_schema(d4) + check3_topology(d4, cfg)
          + check4_conservation(d4, cfg) + check5_signs(d4))
    v4 = aggregate(i4)
    _ck(v4 == "FAIL", f"case3 verdict={v4}")
    _ck(any(i.code == "missing_component" for i in i4), "case3 missing_component absent")

    s5a = json.loads(json.dumps(sample))
    s5b = json.loads(json.dumps(sample))
    s5b["reaction_id"] = "Halogen_other"
    s5b["adf_settings"]["functional"] = "PBE"
    s5c = json.loads(json.dumps(sample))
    s5c["reaction_id"] = "Halogen_third"
    da = derive("a.json", s5a)
    db = derive("b.json", s5b)
    dc = derive("c.json", s5c)
    hom = check2_homogeneity([da, db, dc], cfg)
    b_issues = hom.get("b.json", [])
    _ck(any(i.code == "level_mismatch:functional" for i in b_issues),
        f"case4 b_issues={[i.code for i in b_issues]}")

    if failures:
        for f in failures:
            print("FAIL:", f, file=sys.stderr)
        return False
    return True


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Validate ASR JSON outputs.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="manifest.csv")
    p.add_argument("--tau-pass", type=float, default=0.1)
    p.add_argument("--tau-fail", type=float, default=0.5)
    p.add_argument("--de-act-warn-max", type=float, default=300.0)
    p.add_argument("--robust-z-warn", type=float, default=5.0)
    p.add_argument("--energy-z-warn", type=float, default=6.0)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        ok = _selftest()
        if ok:
            print("selftest: OK")
            sys.exit(0)
        print("selftest: FAIL", file=sys.stderr)
        sys.exit(1)

    if not args.input:
        print("--input is required", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.input):
        print(f"--input directory not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    files = sorted(glob.glob(os.path.join(args.input, "*.json")))
    if not files:
        print(f"no *.json files in {args.input}", file=sys.stderr)
        sys.exit(2)

    cfg = Config(tau_pass=args.tau_pass, tau_fail=args.tau_fail,
                 de_act_warn_max=args.de_act_warn_max,
                 robust_z_warn=args.robust_z_warn,
                 energy_z_warn=args.energy_z_warn)

    records = load_records(args.input)
    derived = [derive(path, raw) for path, raw in records]
    rows, summary = _evaluate(derived, cfg)
    write_manifest(rows, args.output)

    by_check = summary["by_check"]
    print(f"files scanned   : {summary['n']}")
    print(f"PASS / WARN / FAIL : {summary['pass']} / {summary['warn']} / {summary['fail']}")
    print(f"recovered (WARN) : {summary['recovered']}")
    print("failed by check :")
    print(f"  check 1 (schema)        : {by_check.get(1, 0)}")
    print(f"  check 2 (level)         : {by_check.get(2, 0)}")
    print(f"  check 3 (topology)      : {by_check.get(3, 0)}")
    print(f"  check 4 (conservation)  : {by_check.get(4, 0)}")
    print(f"  check 5 (sign/dist)     : {by_check.get(5, 0)}")
    print(f"manifest written: {args.output}")


if __name__ == "__main__":
    main()
