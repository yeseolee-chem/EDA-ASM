#!/usr/bin/env python3
"""d1c_merge.py — SPEC_D1_ALL Phase C, last step: the two SPEC feature parquets and gate C.

  python d1c_merge.py OUT_DIR       (rev 5 venv python, inside phase_c_step.slurm merge; OUT_DIR = results_final)

Inputs (D1 roots of rev5_common, set by phase_c_env.sh): $ESPLEY_OUT/xtb_features_g1.parquet (d1c_aggregate.py),
$R5_SCRATCH/ext/slices/slice_*.parquet (ext_features.py slice), $G1_ROOT/<rid>/ markers, labels_g1in.json,
OUT_DIR/gate_c_control.json (d1c_control.py, gate C iii).
Writes:
  rev5_common.D1_FEAT       base columns (xtb_slice.py) + job_id first; every D1 label row
  rev5_common.D1_FEAT_EXT   ext_features.py columns + job_id first; the G1-ok rows (as rev 5)
  OUT_DIR/FEATURES_SHA256.txt, gate_c.json, GATE_C.md, gate_c_fail_reasons.csv (one row per D1 label row)
Gate C (SPEC):
  (i)   EXT_SEL = train_ml_single.resolve_arm("EXT_SEL") (ESPLEY73 + prereg_rev5b.json blocks B1, B4; 133 columns) is
        identical, names and order, to the columns of the rev 5 EXT_SEL fit record
        ($REV5_UNITS/EXT_SEL__KRR_rbf__dft_barrier_kcal.json), and every column is present here;
  (ii)  rows without a full EXT_SEL vector (G1 failed, xtb_status != ok, status_B1 / status_B4 != ok, or a
        non-finite value) <= D1_MAX_MISSING of the D1 label rows; the failure reasons are tabulated. Blocks B2, B3,
        B5, B6 are computed (same command as rev 5) but are not EXT_SEL: their failures are reported, not gated;
  (iii) gate_c_control.json passed (D0 controls, max |difference| <= D1_CONTROL_TOL).
Exit 0 = gate C passed, 1 = gate C failed (STOP), 2 = inputs missing.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "espley_xtb_repro"))
import rev5_common as R5  # noqa: E402
import train_ml_single as T  # noqa: E402
import xtb_slice as xs  # noqa: E402

REV5_UNITS = Path(os.environ.get("REV5_UNITS", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/lockbox/units"))


def die(code, msg):
    print(msg, flush=True)
    sys.exit(code)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    if Path(os.environ.get("ESPLEY_OUT", "")) != R5.D1_XTB_ROOT or R5.SCRATCH != R5.D1_R5_SCRATCH:
        die(2, "STOP: not the D1 environment (source phase_c_env.sh)")
    labels = json.loads(R5.D1_LABELS_G1.read_text())
    jid = {int(d["rxn_id"]): d["job_id"] for d in labels}
    base_p = R5.D1_XTB_ROOT / "xtb_features_g1.parquet"
    slices = sorted((R5.D1_R5_SCRATCH / "ext" / "slices").glob("slice_*.parquet"))
    if not base_p.is_file() or len(slices) != R5.D1_N_SLICES:
        die(2, f"inputs missing: {base_p} ({base_p.is_file()}), ext slices {len(slices)}/{R5.D1_N_SLICES}")
    base = pd.read_parquet(base_p)
    ext = pd.concat([pd.read_parquet(p) for p in slices], ignore_index=True).sort_values("rxn_id").reset_index(drop=True)
    done = sorted(r for r in jid if xs.geom_marker("g1", r) == ".done")
    if ext.rxn_id.astype(int).tolist() != done:
        die(2, f"ext rows ({len(ext)}) are not exactly the G1 .done rxns ({len(done)})")

    # ---- (i) EXT_SEL columns
    sel = list(T.resolve_arm("EXT_SEL"))
    ref_f = REV5_UNITS / "EXT_SEL__KRR_rbf__dft_barrier_kcal.json"
    ref = json.loads(ref_f.read_text())["fingerprint"]["columns"] if ref_f.is_file() else None
    have = set(base.columns) | set(ext.columns)
    g_i = dict(n_ext_sel=len(sel), equals_rev5_fit_columns=ref == sel, rev5_fit_record=str(ref_f),
               absent=[c for c in sel if c not in have])
    g_i["passed"] = bool(g_i["equals_rev5_fit_columns"] and not g_i["absent"] and len(sel) == 133)

    # ---- (ii) missing rows
    m = base[["rxn_id", "xtb_status"]].merge(ext[["rxn_id", "ext_status"] + [f"status_{b}" for b in R5.BLOCK_ORDER]],
                                             on="rxn_id", how="left")
    full = base.merge(ext, on="rxn_id", how="left", suffixes=("", "_ext"))
    fin = np.isfinite(full[sel].to_numpy(dtype=float)).all(axis=1) if not g_i["absent"] else np.zeros(len(full), bool)
    reasons = []
    for k, r in enumerate(m.itertuples()):
        mk = xs.geom_marker("g1", r.rxn_id)
        if mk != ".done":
            why = f"g1:{(mk or 'missing').removeprefix('.fail_')}"
        elif r.xtb_status != "ok":
            why = f"xtb:{r.xtb_status}"
        elif r.status_B1 != "ok":
            why = f"B1:{r.status_B1}"
        elif r.status_B4 != "ok":
            why = f"B4:{r.status_B4}"
        elif not fin[k]:
            why = "nonfinite"
        else:
            why = "ok"
        reasons.append(why)
    m["job_id"] = m.rxn_id.astype(int).map(jid)
    m["ext_sel_status"] = reasons
    R5.write_atomic(out / "gate_c_fail_reasons.csv", lambda t: m.to_csv(t, index=False))
    n_bad = int((m.ext_sel_status != "ok").sum())
    g1ok = np.array([xs.geom_marker("g1", r) == ".done" for r in m.rxn_id])
    g_ii = dict(n_rows=len(m), n_missing=n_bad, frac_missing=n_bad / len(m), max_frac=R5.D1_MAX_MISSING,
                denominator="all D1 label rows (SPEC wording); the G1-ok-only rate below is information",
                n_g1_fail=int((~g1ok).sum()), frac_missing_among_g1_ok=float((m.ext_sel_status[g1ok] != "ok").mean()) if g1ok.any() else None,
                reasons=m.loc[m.ext_sel_status != "ok", "ext_sel_status"].value_counts().to_dict(),
                other_blocks_not_ok={b: int(((m.ext_sel_status == "ok") & (m[f"status_{b}"] != "ok")).sum())
                                     for b in ("B2", "B3", "B5", "B6")},
                n_B5_scan_gate_fail=int((m.status_B5 == "scan_gate_fail").sum()))
    g_ii["passed"] = g_ii["frac_missing"] <= R5.D1_MAX_MISSING

    # ---- (iii) D0 controls
    cf = out / "gate_c_control.json"
    ctl = json.loads(cf.read_text()) if cf.is_file() else None
    g_iii = dict(file=str(cf), present=ctl is not None, passed=bool(ctl and ctl.get("passed")),
                 max_abs_diff=(ctl or {}).get("max_abs_diff"), tol=R5.D1_CONTROL_TOL)

    # ---- SPEC parquets (job_id key first)
    b = base.copy(); b.insert(0, "job_id", b.rxn_id.astype(int).map(jid))
    e = ext.copy(); e.insert(0, "job_id", e.rxn_id.astype(int).map(jid))
    R5.write_atomic(R5.D1_FEAT, lambda t: b.to_parquet(t, index=False))
    R5.write_atomic(R5.D1_FEAT_EXT, lambda t: e.to_parquet(t, index=False))
    sha = {str(p): R5.sha256(p) for p in (R5.D1_FEAT, R5.D1_FEAT_EXT)}
    R5.write_atomic(out / "FEATURES_SHA256.txt", "".join(f"{v}  {k}\n" for k, v in sha.items()))

    rep = dict(gate_i=g_i, gate_ii=g_ii, gate_iii=g_iii, passed=bool(g_i["passed"] and g_ii["passed"] and g_iii["passed"]),
               sha256=sha, n_label_rows=len(labels), n_g1_done=len(done))
    R5.write_json(out / "gate_c.json", rep)
    md = ["# 게이트 C (SPEC_D1_ALL Phase C)", "",
          f"- 전체 판정: **{'통과' if rep['passed'] else '실패 — STOP'}**",
          f"- (i) EXT_SEL {len(sel)}열, rev5 fit 기록과 이름·순서 같음: {g_i['equals_rev5_fit_columns']}, 없는 열 {len(g_i['absent'])} → {'통과' if g_i['passed'] else '실패'}",
          f"- (ii) feature 결측 {n_bad}/{len(m)} = {g_ii['frac_missing']:.2%} (기준 ≤ {R5.D1_MAX_MISSING:.0%}, 분모 = D1 라벨 전체) → {'통과' if g_ii['passed'] else '실패'}",
          f"  - 참고: G1 실패 {g_ii['n_g1_fail']}행; G1 성공 행만 보면 결측 {g_ii['frac_missing_among_g1_ok']}",
          f"- (iii) D0 대조 (C0020 · C0105): {'있음' if ctl else '없음 (d1c_control.py 를 먼저)'}, 최대 차이 {g_iii['max_abs_diff']} (기준 ≤ {R5.D1_CONTROL_TOL:g}) → {'통과' if g_iii['passed'] else '실패'}",
          "  - C0020 (D0 rxn 20) 은 D0 에서 G1 실패 (optts_not_converged) 라 상태 대조만 된다; 수치 대조는 C0105 한 건 (+ 참고용 rxn 2434, 판정에 넣지 않음)",
          "", "## 결측 사유 (EXT_SEL)", "", "| 사유 | 행 |", "|---|---|"]
    md += [f"| `{k}` | {v} |" for k, v in g_ii["reasons"].items()]
    md += ["", f"EXT_SEL 이 아닌 블록의 실패 (EXT_SEL ok 행 중, 판정에 쓰지 않음): {g_ii['other_blocks_not_ok']}; "
           f"B5 scan_gate_fail {g_ii['n_B5_scan_gate_fail']}", "", "## sha256", ""]
    md += [f"- `{v}`  {k}" for k, v in sha.items()]
    R5.write_atomic(out / "GATE_C.md", "\n".join(md) + "\n")
    print("\n".join(md))
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
