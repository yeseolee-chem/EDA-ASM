#!/usr/bin/env python3
"""tests_feat.py — checks of the GEO2b FEAT slice (feat_geo.py, d0d1.py, g3c.py, analysis/d1_ladder/inputs.py geo
additions). sbatch / srun on a compute node only; reactot (python 3.10) unless noted.

  python tests_feat.py lint      # undefined-name check (symtable) of the four files; any python env
  python tests_feat.py unit      # unit tests: synthetic data + read-only real inputs; writes only under
                                 #   $GEO2B/_selftest (removed at the end) and results/U_feat_tests.json
Exit 0 = all pass, 3 = a check failed.
"""
from __future__ import annotations

import builtins
import json
import os
import shutil
import symtable
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FILES = [HERE / "feat_geo.py", HERE / "d0d1.py", HERE / "g3c.py", REPO / "analysis" / "d1_ladder" / "inputs.py"]
SELF = Path("/gpfs/tmp_cpu2/yeseo1ee/geo2b/_selftest")


def undefined(path):
    src = Path(path).read_text()
    top = symtable.symtable(src, str(path), "exec")
    mod = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    bad = []

    def walk(t):
        for s in t.get_symbols():
            n = s.get_name()
            if t.get_type() != "module" and s.is_referenced() and s.is_global() and not s.is_declared_global() \
                    and n not in mod and not hasattr(builtins, n) and not (n.startswith("__") and n.endswith("__")):
                bad.append(f"{t.get_name()}:{t.get_lineno()} {n}")
        for c in t.get_children():
            walk(c)
    walk(top)
    for s in top.get_symbols():
        n = s.get_name()
        if s.is_referenced() and not (s.is_assigned() or s.is_imported() or s.is_namespace()) \
                and not hasattr(builtins, n) and not (n.startswith("__") and n.endswith("__")):
            bad.append(f"<module> {n}")
    return bad


def cmd_lint():
    out = {str(f.relative_to(REPO)): undefined(f) for f in FILES}
    print(json.dumps(out, indent=1))
    return 3 if any(out.values()) else 0


# ---------------------------------------------------------------- unit tests
RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        info = fn()
        RESULTS.append(dict(name=name, ok=True, info=info, s=round(time.time() - t0, 2)))
        print(f"PASS {name}: {json.dumps(info, default=str)[:600]}", flush=True)
    except Exception as e:                                         # noqa: BLE001
        RESULTS.append(dict(name=name, ok=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-3000:],
                            s=round(time.time() - t0, 2)))
        print(f"FAIL {name}: {type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}", flush=True)


def cmd_unit():
    scr = SELF / f"scr_{os.getpid()}"
    os.environ["GEO2_SCRATCH"] = str(scr)                          # before any import: C.G3 / inputs.GEO2 = scr
    scr.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import numpy as np
    import pandas as pd
    import feat_geo as FG
    import d0d1 as XD
    C = FG.C
    sys.path.insert(0, str(C.LAD_DIR))
    import inputs as IN
    assert C.G3 == scr and IN.GEO2 == scr, (C.G3, IN.GEO2)

    def t_rate_gate():
        base = list(range(1000001, 1000101)) + list(range(0, 100))
        info, why = FG.rate_gate("x", [1000001, 1000002, 5], base, 0.01)
        assert info["D1"]["n"] == 100 and info["D1"]["n_fail"] == 2 and info["D0"]["n_fail"] == 1, info
        assert len(why) == 1 and why[0].startswith("D1:"), why
        assert info["combined_info"]["n_fail"] == 3
        info2, why2 = FG.rate_gate("x", [5, 6], base, 0.01)
        assert len(why2) == 1 and why2[0].startswith("D0:")
        return dict(why=why, why2=why2)

    def t_qa_max():
        assert FG.qa_max(8337) == 42 and FG.qa_max(14000) == 70 and FG.qa_max(20) == 1, \
            (FG.qa_max(8337), FG.qa_max(14000), FG.qa_max(20))
        return dict(q8337=FG.qa_max(8337), q14000=FG.qa_max(14000))

    rng0 = np.random.default_rng(1)
    jobs, lab = [], {}
    for i in range(1000):
        T = f"T{i + 1:05d}"
        jobs.append(T)
        lab[T] = dict(rxn_id=1000001 + i, n_atoms=int(rng0.integers(10, 64)), charge1=0, charge2=0)
    for i in range(800):
        T = f"D0G2_{i:04d}"
        jobs.append(T)
        q = -2 if i % 37 == 0 else (1 if i % 53 == 0 else 0)
        lab[T] = dict(rxn_id=i, n_atoms=int(rng0.integers(12, 95)), charge1=0, charge2=q)

    def t_pilot_select():
        p1, s1 = FG.pilot_select(jobs, lab)
        p2, s2 = FG.pilot_select(jobs, lab)
        assert p1 == p2 and s1 == s2, "not deterministic"
        from collections import Counter
        assert Counter(d for _, d, _ in p1) == {"D1": 25, "D0": 25}, Counter(d for _, d, _ in p1)
        for d in FG.DATASETS:
            cq = Counter(q for _, dd, q in p1 if dd == d)
            assert [cq[q] for q in range(4)] == list(FG.PILOT_QUOTA), (d, cq)
            for q in range(4):
                members = set(s1[(d, q)])
                picks = [T for T, dd, qq in p1 if dd == d and qq == q]
                assert set(picks) <= members and len(set(picks)) == len(picks)
                if d == "D0":
                    nchg_cell = sum(1 for T in members if lab[T]["charge2"])
                    nchg = sum(1 for T in picks if lab[T]["charge2"])
                    assert nchg == (1 if nchg_cell else 0), (q, nchg, nchg_cell)
        # quartiles by rank: every n_atoms of quartile q <= every n_atoms of quartile q+1
        for d in FG.DATASETS:
            mx = [max(lab[T]["n_atoms"] for T in s1[(d, q)]) for q in range(4)]
            mn = [min(lab[T]["n_atoms"] for T in s1[(d, q)]) for q in range(4)]
            assert all(mx[q] <= mn[q + 1] for q in range(3)), (d, mx, mn)
        return dict(n=len(p1), cells={f"{d}|{q}": len(v) for (d, q), v in s1.items()})

    def t_projection():
        p, s = FG.pilot_select(jobs, lab)
        plan = dict(strata={f"{d}|{q}": v for (d, q), v in s.items()})
        per = {T: 0.1 * (q + 1) * (2 if d == "D0" else 1) for T, d, q in p}
        qof = {T: (d, q) for T, d, q in p}
        tot, by_d, cells = FG.pilot_projection(plan, per, qof)
        want = sum(0.1 * (q + 1) * (2 if d == "D0" else 1) * len(s[(d, q)]) for (d, q) in s)
        assert abs(tot - want) < 1e-9 and abs(by_d["D1"] + by_d["D0"] - tot) < 1e-9, (tot, want)
        per.pop(next(T for T, d, q in p if (d, q) == ("D0", 3)))
        tot2, _, _ = FG.pilot_projection(plan, per, qof)
        assert np.isfinite(tot2)
        per2 = {T: v for T, v in per.items() if qof[T] != ("D1", 0)}
        tot3, _, _ = FG.pilot_projection(plan, per2, qof)
        assert not np.isfinite(tot3), "an empty cell must make the projection non-finite (STOP)"
        return dict(total=tot, by_dataset=by_d)

    def t_td0_quota():
        rows = [(str(lab[T]["rxn_id"]), lab[T]["n_atoms"], lab[T]["charge2"]) for T in jobs if T.startswith("D0G2_")]
        p, qs = FG.strat_pick(rows, (5, 5, 5, 5), np.random.default_rng(FG.C.SEED_S), charged_one=True)
        assert len(p) == 20 and len({x for x, _ in p}) == 20
        chg = {x for x, _ in p if dict((r[0], r[2]) for r in rows)[x]}
        assert 1 <= len(chg) <= 4, chg
        return dict(n=len(p), n_charged=len(chg))

    def t_rung_cfg():
        a, b = FG.cfg_str(FG.rung_cfg("geo")), FG.cfg_str(IN.RUNGS["geo"])
        assert a == b, {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert "geo" not in IN.RUNG_ORDER and list(IN.RUNG_ORDER) == ["dft", "g1", "g2"]
        return dict(keys=sorted(a))

    def t_defaults_unchanged():
        L = IN.L
        assert IN.targets_of("dft") == L.targets() and IN.targets_of("g2") == L.targets()
        K0, K1 = IN._Q().load_keys(), IN._load_keys("g2")
        assert K0.equals(K1), "_load_keys('g2') != refs_q.load_keys()"
        roots = dict((str(p), lab_) for p, lab_ in IN.rung_roots())
        for r in ("dft", "g1", "g2"):
            assert IN.check_paths(r, [str(IN.table_path(r))])["ok"], r
        g = IN.check_paths("geo", [str(scr / "inputs" / "geo.parquet"), str(IN.GEO1B_D0 / "D0_pairs.csv")])
        assert g["ok"], g
        f = IN.check_paths("g2", [str(scr / "inputs" / "geo.parquet")])
        assert not f["ok"] and f["foreign"], f
        return dict(n_targets=len(L.targets()), n_keys=len(K0), n_roots=len(roots))

    def t_keys_union():
        K = IN._load_keys("geo")
        n1, n0 = len(pd.read_parquet(IN.Q_KEYS)), len(pd.read_parquet(IN.Q_KEYS_D0))
        assert len(K) == n1 + n0, (len(K), n1, n0)
        assert set(K.dataset) == {"D1", "D0"}, set(K.dataset)
        k1, k0 = set(K.key[K.dataset == "D1"]), set(K.key[K.dataset == "D0"])
        assert not (k1 & k0)
        assert K.constrained.dtype == bool
        return dict(n_d1=n1, n_d0=n0, n_constrained=dict(K.groupby("dataset").constrained.sum().astype(int)))

    def t_targets_geo():
        L = IN.L
        tg = L.targets()
        d0 = [dict(rxn_id=i, job_id=f"D0G2_{i:04d}", kind="D0") for i in (0, 1, 7)]
        p = scr / "comb" / "labels_d0d1.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(tg + d0))
        rows = IN.targets_of("geo")
        assert [d["job_id"] for d in rows] == sorted(d["job_id"] for d in tg + d0)
        p.write_text(json.dumps(tg + [dict(rxn_id=3, job_id="D0G2_3")]))
        try:
            IN.targets_of("geo")
            raise AssertionError("bad D0 job_id accepted")
        except IN._Stop:
            pass
        p.write_text(json.dumps(tg[:-1] + d0))
        try:
            IN.targets_of("geo")
            raise AssertionError("missing D1 row accepted")
        except IN._Stop:
            pass
        p.unlink()
        return dict(n=len(rows))

    def t_cons_own():
        K = IN._load_keys("geo")
        cons = K[K.constrained]
        free = K[~K.constrained]
        pick = pd.concat([cons[cons.dataset == "D1"].head(2), cons[cons.dataset == "D0"].head(2),
                          free[free.dataset == "D1"].head(1), free[free.dataset == "D0"].head(1)])
        prov, ov = {}, {}
        for j, k, key in zip(pick.job_id, pick.frag, pick.key):
            prov[(j, int(k))] = dict(src="min", src_job="X", src_frag=1, key=key, e_orig=-1.0, e_new=-1.001)
            ov.setdefault(j, {})[int(k)] = -1.001
        ov2, prov2, rep = FG.cons_own(IN, "geo", ov, prov)
        kept = {(j, k) for (j, k), p_ in prov2.items() if p_["src"] == "own"}
        want = {(j, int(k)) for j, k, c in zip(pick.job_id, pick.frag, pick.constrained) if c}
        assert kept == want, (kept, want)
        assert all(ov2.get(j, {}).get(k) is None for j, k in want)
        n_cons = {d: int((cons.dataset == d).sum()) for d in ("D1", "D0")}
        return dict(n_kept=len(kept), by_dataset=rep["n_rows_constrained_by_dataset"], n_constrained_all=n_cons)

    def t_phase_x_real():
        H, PR = XD.hf_modules()
        allrows = json.loads(XD.D0_LABELS.read_text())
        lab0 = {int(d["rxn_id"]): d for d in allrows}
        meta = XD.read_meta_csv(XD.D0_META)
        out = {}
        for rid in (0, 1, 2, 5, 3312, 3323):
            rec, files = XD.check_d0(rid, lab0[rid], meta[rid], H, PR)
            assert not rec["reasons"] and not rec["stop"], (rid, rec["reasons"], rec["stop"])
            hf = files["hf/eda.inp"].decode()
            assert "B3LYP" not in hf and "def2-TZVP" not in hf and "6-31G*" in hf and "nprocs 1" in hf
            out[rid] = dict(n_atoms=rec["n_atoms"], charge=rec["charge"], x4=rec["x4_provenance"])
        return out

    def t_hf_text_equals_d1():
        H, PR = XD.hf_modules()
        res = {}
        for T in ("T00001", "T00003", "T00614"):
            for c in H.CALCS:
                o = (C.HF_SCR / "inputs" / T / "orig" / f"{c}.inp").read_text()
                h = (C.HF_SCR / "inputs" / T / "hf" / f"{c}.inp").read_bytes()
                new, bad = XD.hf_diff_bad(H, PR, o, "eda" if c == "eda" else "frag", H.parse_inp(o))
                assert not bad and new.encode() == h, (T, c, bad)
            res[T] = "5 templates byte-identical to d1_hf/inputs/<T>/hf"
        return res

    def t_s1_select():
        man = [dict(job_id=f"D0G2_{i:04d}", rxn_id=str(i), dataset="D0", n_atoms=str(n), charge=str(q), x_fail=xf)
               for i, n, q, xf in ((0, 19, 0, "False"), (1, 12, 0, "False"), (2, 11, 0, "True"), (3, 14, -2, "False"),
                                   (4, 13, 0, "False"), (5, 30, 1, "False"))]
        man.append(dict(job_id="T00001", rxn_id="1000001", dataset="D1", n_atoms="5", charge="0", x_fail="False"))
        sel = [r["job_id"] for r in XD.s1_select(man)]
        assert sel == ["D0G2_0001", "D0G2_0004", "D0G2_0003"], sel
        return dict(sel=sel)

    def t_comb_build():
        """build_comb on the real D1 + D0 sources with synthetic Phase X records (templates not built here: only the
        D0 hf_templates links may dangle)."""
        rec = {}
        d0, folds = XD.load_d0_rows(rec)
        d0x = {r: dict(reasons=[], n_atoms=d0[r]["n_atoms"], n_f1=d0[r]["n_f1"], n_f2=d0[r]["n_f2"],
                       charge=int(d0[r]["charge1"]) + int(d0[r]["charge2"]), elements="") for r in d0}
        bad1 = sorted(d0)[:3]
        for r in bad1:
            d0x[r]["reasons"] = ["x2:test"]
        outs, dig, probs = XD.build_comb(d0, d0x, rec)
        assert len(probs) == 1 and probs[0].startswith(f"{len(d0) - 3} links point to nothing"), probs[:3]
        rows = json.loads(XD.COMB_LABELS.read_text())
        d1 = json.loads(XD.D1_LABELS.read_text())
        assert rows[:len(d1)] == d1 and len(rows) == len(d1) + len(d0)
        r0 = rows[len(d1)]
        want = dict(d0[sorted(d0)[0]], job_id=XD.job_id(sorted(d0)[0]), kind="D0")
        assert r0 == want
        meta = XD.read_comb_meta()
        assert len(meta) == len(rows) and meta[0] == sorted(int(x) for x in "2 3 4 5 6 7 13 14 15 16".split())
        man = XD.read_comb_manifest()
        assert sum(1 for r in man if r["x_fail"] == "True") == 3 and len(man) == len(rows)
        assert dig["prof"]["n"] == len(rows) and dig["hf_templates"]["n"] == len(d1) + len(d0) - 3
        assert os.readlink(XD.COMB_PROF / "1000001") == "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs/T00001"
        assert os.readlink(XD.COMB_JOBS / "D0G2_0000") == str(XD.D0_LABJOBS / "D0G2_0000")
        # rebuild with the same inputs: no problem other than the dangling links; a changed file is refused
        outs2, dig2, probs2 = XD.build_comb(d0, d0x, rec)
        assert outs2 == outs and dig2 == dig and probs2 == probs
        d0x[sorted(d0)[5]]["reasons"] = ["x2:test2"]
        _, _, probs3 = XD.build_comb(d0, d0x, rec)
        assert any("exists with other content" in p for p in probs3), probs3[:3]
        return dict(n_rows=len(rows), n_d0=len(d0), digests={k: v["n"] for k, v in dig.items()},
                    overlap=rec["d0_sources"]["overlap"])

    def t_comb_state_and_labels():
        # comb_state on the files of t_comb_build (made complete by hand) and the FG.inputs_check D1 rule
        outs = {str(p): C.sha256(p) for p in (XD.COMB_LABELS, XD.COMB_META, XD.COMB_MANIFEST)}
        dig = {nm: dict(digest=XD.listing_digest(XD.COMB / nm)[0]) for nm in XD.LINK_DIRS}
        XD.X_REPORT_SAVED = XD.X_REPORT
        XD.X_REPORT = scr / "X_report_test.json"
        try:
            XD.X_REPORT.write_text(json.dumps(dict(stop=False, complete=True, outputs=outs, link_digests=dig)))
            st, pr = XD.comb_state()
            assert not pr, pr
            FG._INP.clear()
            FG._LAB.clear()
            rec = {}
            FG.inputs_check(rec)
            assert rec["comb_inputs"]["d1_rows_eq_targets"] and rec["comb_inputs"]["n_d0"] > 5000
            os.symlink("/nonexistent", XD.COMB_PROF / "999999")
            st, pr = XD.comb_state()
            assert pr and pr[0][0] == 3, pr
            (XD.COMB_PROF / "999999").unlink()
        finally:
            XD.X_REPORT = XD.X_REPORT_SAVED
            FG._INP.clear()
        return dict(n_d0=rec["comb_inputs"]["n_d0"])

    def t_geo_state():
        FG._LAB.clear()
        lab = {1000001: dict(rxn_id=1000001, job_id="T00001"), 1000002: dict(rxn_id=1000002, job_id="T00002"),
               3: dict(rxn_id=3, job_id="D0G2_0003"), 4: dict(rxn_id=4, job_id="D0G2_0004")}
        FG._LAB.update(lab)
        z, zs = scr / "Z_report.json", scr / "Z_status.csv"
        save = FG.Z_REPORT, FG.Z_STATUS
        FG.Z_REPORT, FG.Z_STATUS = z, zs
        try:
            fin = FG.paths(FG.GEO).final
            for r in (1000001, 3, 4):
                (fin / str(r)).mkdir(parents=True, exist_ok=True)
                (fin / str(r) / ".done").write_text("x\n")
            zs.write_text("rxn_id,status\n1000001,ok\n1000002,no_g2\n3,kept\n4,ok\n")
            z.write_text(json.dumps(dict(stop=False, n_folders=3)))
            base, info = FG.geo_state()
            assert not info["problems"] and [r for r, _ in base] == [3, 4, 1000001], (info["problems"], base)
            z.write_text(json.dumps(dict(stop=False, n_folders=dict(D1=1, D0=2))))
            assert not FG.geo_state()[1]["problems"]
            z.write_text(json.dumps(dict(stop=False, n_folders=2)))
            assert FG.geo_state()[1]["problems"][0][0] == 3
            z.write_text(json.dumps(dict(stop=True, n_folders=3)))
            assert FG.geo_state()[1]["problems"][0][0] == 3
            return dict(base=base, by_dataset=info["n_base_by_dataset"])
        finally:
            FG.Z_REPORT, FG.Z_STATUS = save
            FG._LAB.clear()

    def t_g_scan():
        FG._LAB.clear()
        P = FG.paths(FG.GEO)
        base = []
        cases = {1000011: "ok", 1000012: "stale", 1000013: ".fail_order_x", 21: ".fail_freq_not_terminated", 22: "open",
                 23: "ok"}
        for r, kind in cases.items():
            src, d = P.final / str(r), P.feat / str(r)
            src.mkdir(parents=True, exist_ok=True)
            for f in FG.XYZ_FILES:
                (src / f).write_text(f"1\n\nH 0 0 {r}\n")
            if kind == "open":
                base.append((r, f"J{r}"))
                continue
            d.mkdir(parents=True, exist_ok=True)
            res = dict(sources={f: dict(sha256=C.sha256(src / f)) for f in FG.XYZ_FILES}, n_imag=1,
                       imag_freqs_cm=[-300.0], fail_reasons=[kind[6:]] if kind.startswith(".fail") else [])
            (d / "result.json").write_text(json.dumps(res))
            (d / (kind if kind.startswith(".fail") else ".done")).write_text("x\n")
            if kind == "stale":
                (src / "ts.xyz").write_text("changed\n")
            base.append((r, f"J{r}"))
        s, rows = FG.g_scan(FG.GEO, base)
        cat = {x["rxn_id"]: x["category"] for x in rows}
        assert cat == {1000011: "ok", 1000012: "stale", 1000013: "order", 21: "freq", 22: "open", 23: "ok"}, cat
        assert s["rates"]["order"]["D1"]["n_fail"] == 1 and s["rates"]["freq"]["D0"]["n_fail"] == 1
        assert s["stop"] and not s["complete"]
        return dict(categories=cat, stop_reasons=s["stop_reasons"])

    def t_rel_gate():
        import pandas as pd_
        FG._LAB.clear()
        G = pd_.read_parquet(FG.G2_HF_LABELS)
        G = G[(G.geom == "g2") & (G.status == "ok")].sort_values("job_id").head(4)
        T4 = list(G.job_id)
        L_ = {int(rid): d for rid, d in ((r["rxn_id"], r) for r in IN.L.targets()) if d["job_id"] in set(T4)}
        rid_of = {d["job_id"]: int(r) for r, d in L_.items()}
        FG._LAB.update(L_)
        sel = scr / "C_select.csv"
        chosen = {T4[0]: "orig", T4[1]: "orig", T4[2]: "orig", T4[3]: "r1"}
        sel.write_text("src,rxn_id,chosen\n" + "".join(f"d1,{rid_of[T]},{c}\n" for T, c in chosen.items()))
        save = FG.G2C_SELECT
        FG.G2C_SELECT = sel
        try:
            fin = FG.paths(FG.GEO).final
            for T in T4:
                d = fin / str(rid_of[T])
                d.mkdir(parents=True, exist_ok=True)
                for f in ("rel1.xyz", "rel2.xyz"):
                    shutil.copyfile(FG.D1_G2_FINAL / str(rid_of[T]) / f, d / f)
            ref = {j: (a, b) for j, a, b in zip(G.job_id, G.e_frag1_rel_eh, G.e_frag2_rel_eh)}
            E = {T: ref[T] for T in T4}
            rg = FG.rel_gate(E, T4)
            assert rg["n_comparable"] == 3 and rg["n_mismatch"] == 0 and not rg["stop"], rg
            E2 = dict(E)
            E2[T4[1]] = (ref[T4[1]][0] + 2e-6, ref[T4[1]][1])
            with open(fin / str(rid_of[T4[2]]) / "rel1.xyz", "a") as fh:
                fh.write(" ")
            rg2 = FG.rel_gate(E2, T4)
            assert rg2["n_mismatch"] == 2 and rg2["n_rel_bytes_differ"] == 1 and rg2["stop"], rg2
            return dict(rg=dict(n_comparable=rg["n_comparable"], n_d1_not_orig=rg["n_d1_not_orig"]),
                        rg2=dict(n_mismatch=rg2["n_mismatch"], stop=rg2["stop_reasons"]))
        finally:
            FG.G2C_SELECT = save
            FG._LAB.clear()

    def t_refs_min_geo():
        """inputs.refs_min('geo') on synthetic HF reference files (frag{k}_rel.inp = the label reference coordinates):
        P6b passes for D1 and D0 rows, the per-key minimum stays inside a key (D0 rxn 0 / 1 share their references)."""
        import pandas as pd_
        rows = [("T00001", None), ("T00002", None), ("D0G2_0000", 0), ("D0G2_0001", 1), ("D0G2_3312", 3312)]
        runs = Path(IN.RUNGS["geo"]["hf_runs"]) / "g2"
        recs = []
        for i, (T, rid) in enumerate(rows):
            for k in (1, 2):
                src = (C.HF_SCR / "inputs" / T / "hf" / f"frag{k}_rel.inp") if rid is None else \
                    (XD.rxn_dir(XD.D0_LABEL_INPUTS, rid) / f"frag{k}_rel.inp")
                d = runs / T
                d.mkdir(parents=True, exist_ok=True)
                (d / f"frag{k}_rel.inp").write_text(src.read_text())
            recs.append(dict(job_id=T, e_frag1_rel_eh=-100.0 - 0.001 * i, e_frag2_rel_eh=-50.0 - 0.002 * i))
        Hq = pd_.DataFrame(recs)
        rep = {}
        ov, prov, rr = IN.refs_min("geo", Hq, rep)
        assert rep["reference_graphs"]["ok"], rep["reference_graphs"]
        K = IN._load_keys("geo").set_index(["job_id", "frag"])
        for (j, k), p in prov.items():
            if p["src"] == "min":
                assert K.at[(j, k), "key"] == K.at[(p["src_job"], p["src_frag"]), "key"]
                assert (str(j).startswith("D0G2_")) == (str(p["src_job"]).startswith("D0G2_"))
        k00, k01 = K.at[("D0G2_0000", 1), "key"], K.at[("D0G2_0001", 1), "key"]
        return dict(n_replaced=rr["n_fragments_replaced"], keys_rxn0_rxn1_frag1_equal=k00 == k01,
                    replaced={f"{j}/{k}": p["src_job"] for (j, k), p in prov.items() if p["src"] == "min"})

    def t_listing_digest():
        d = scr / "lnk"
        d.mkdir(parents=True, exist_ok=True)
        os.symlink("/a", d / "x")
        s1 = XD.listing_digest(d)
        os.symlink("/b", d / "y")
        s2 = XD.listing_digest(d)
        assert s1[1] == 1 and s2[1] == 2 and s1[0] != s2[0]
        return dict(s1=s1, s2=s2)

    try:
        for name, fn in [("rate_gate", t_rate_gate), ("qa_max", t_qa_max), ("pilot_select", t_pilot_select),
                         ("pilot_projection", t_projection), ("td0_quota", t_td0_quota), ("rung_cfg", t_rung_cfg),
                         ("inputs_defaults_unchanged", t_defaults_unchanged), ("keys_union", t_keys_union),
                         ("targets_geo", t_targets_geo), ("cons_own", t_cons_own), ("phase_x_real", t_phase_x_real),
                         ("hf_text_equals_d1", t_hf_text_equals_d1), ("s1_select", t_s1_select),
                         ("comb_build", t_comb_build), ("comb_state_inputs_check", t_comb_state_and_labels),
                         ("geo_state", t_geo_state), ("g_scan", t_g_scan), ("rel_gate", t_rel_gate),
                         ("refs_min_geo", t_refs_min_geo), ("listing_digest", t_listing_digest)]:
            check(name, fn)
    finally:
        shutil.rmtree(scr, ignore_errors=True)
    rep = dict(written=time.strftime("%F %T"), host=os.uname().nodename, python=sys.executable,
               n=len(RESULTS), n_pass=sum(r["ok"] for r in RESULTS), all_pass=all(r["ok"] for r in RESULTS),
               results=RESULTS)
    out = HERE / "results" / "U_feat_tests.json"
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, indent=1, default=str))
    os.replace(tmp, out)
    print(f"[unit] {rep['n_pass']}/{rep['n']} pass -> {out}", flush=True)
    return 0 if rep["all_pass"] else 3


def cmd_wrap():
    """end-to-end without ORCA: d0d1.cmd_x into $GEO2B/_selftest (report / manifest paths redirected there), then the
    _geominputs / _allgeom wrappers (children, D1HF_* overrides) for the 3 T-3 D1 jobs on $LAD/dft_final and 6 D0
    reactions on their Coley TS / references: statuses ok, generated inputs = the templates (elements, header,
    coordinates <= 1e-6 A), _allgeom rows carry the combined rxn_id (D0 included)."""
    import subprocess
    scr = SELF / f"wrap_{os.getpid()}"
    os.environ["GEO2_SCRATCH"] = str(scr)
    scr.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import pandas as pd
    import feat_geo as FG
    import d0d1 as XD
    out = {}
    try:
        XD.X_REPORT, XD.X_MANIFEST_D0 = scr / "X_d0d1_report.json", scr / "X_d0_manifest.csv"
        t0 = time.time()
        rc = XD.cmd_x(1)
        rep = json.loads(XD.X_REPORT.read_text())
        out["x"] = dict(rc=rc, wall_s=round(time.time() - t0, 1), n_d0=rep.get("n_d0"), n_x_fail=rep.get("n_x_fail"),
                        built=rep.get("x_built"), stop=rep.get("stop"), problems=rep.get("problems"),
                        links={k: v.get("n") for k, v in (rep.get("link_digests") or {}).items()})
        assert rc == 0 and rep["complete"] and not rep["stop"], out["x"]
        st, pr = XD.comb_state()
        assert not pr, pr
        rc2 = XD.cmd_x(1)                                          # idempotent rerun: kept, same outputs
        rep2 = json.loads(XD.X_REPORT.read_text())
        assert rc2 == 0 and rep2["outputs"] == rep["outputs"] and rep2["x_built"] == {"kept": rep["n_d0"]}, rep2["x_built"]
        out["x_rerun"] = dict(rc=rc2, built=rep2["x_built"])
        man = {r["job_id"]: r for r in XD.read_comb_manifest()}
        lab = {int(d["rxn_id"]): d for d in XD.load_comb_labels()}

        def run_child(args, env):
            r = subprocess.run([sys.executable, str(HERE / "feat_geo.py")] + [str(a) for a in args], cwd=str(FG.C.HF_DIR),
                               env=env, capture_output=True, text=True)
            return r.returncode, (r.stdout + r.stderr)[-3000:]

        def compare(root, jobs):
            vt = FG.inputs_vs_templates(root, jobs)
            bad = {T: {c: v for c, v in r.items() if not (v["same_elements"] and v["same_header"]
                                                           and v["max_coord_diff_A"] is not None
                                                           and v["max_coord_diff_A"] <= 1e-6)}
                   for T, r in vt.items()}
            return {T: v for T, v in bad.items() if v}, {T: all(v["identical"] for v in r.values()) for T, r in vt.items()}
        # D1: the T-3 jobs on $LAD/dft_final
        h1 = scr / "h_d1"
        h1.mkdir(parents=True, exist_ok=True)
        lst1 = h1 / "list.csv"
        lst1.write_text("job_id\n" + "".join(f"{T}\n" for T in FG.T3_JOBS))
        env1 = dict(FG.env_h_roots(h1, h1 / "results", FG.G0F.DFT_FINAL), GEO2_SCRATCH=str(scr))
        rc, log = run_child(["_geominputs", lst1], env1)
        st1 = pd.read_csv(h1 / "results" / "g2_inputs.csv", dtype=str, keep_default_na=False)
        assert rc == 0 and (st1.status == "ok").all() and len(st1) == 3, (rc, st1.to_dict("records"), log)
        bad1, ident1 = compare(h1 / "inputs", FG.T3_JOBS)
        assert not bad1, bad1
        rc, log = run_child(["_allgeom", lst1], env1)
        S1 = pd.read_csv(h1 / "results" / "H_status.csv")
        assert rc == 0 and sorted(S1.job_id) == FG.T3_JOBS and set(S1.status) == {"not_started"}, (rc, S1, log)
        want1 = {d["job_id"]: r for r, d in lab.items() if d["job_id"] in FG.T3_JOBS}
        assert all(int(r) == want1[j] for j, r in zip(S1.job_id, S1.rxn_id)), S1
        out["d1"] = dict(inputs_identical=ident1, allgeom=S1[["job_id", "rxn_id", "status"]].to_dict("records"))
        # D0: 6 reactions (two charged) on their Coley TS / references (td0_src layout)
        d0 = [0, 1, 5, 3312, 3323, max(r for r in lab if r < 1_000_000)]
        jobs0 = sorted(lab[r]["job_id"] for r in d0)
        src = scr / "src0"
        rec = {}
        FG._LAB.clear()
        FG.td0_src(d0, rec, src)
        h0 = scr / "h_d0"
        h0.mkdir(parents=True, exist_ok=True)
        lst0 = h0 / "list.csv"
        lst0.write_text("job_id\n" + "".join(f"{T}\n" for T in jobs0 + ["T00001"]))
        env0 = dict(FG.env_h_roots(h0, h0 / "results", src), GEO2_SCRATCH=str(scr))
        rc, log = run_child(["_geominputs", lst0], env0)
        st0 = pd.read_csv(h0 / "results" / "g2_inputs.csv", dtype=str, keep_default_na=False)
        s0 = dict(zip(st0.job_id, st0.status))
        assert rc == 0 and all(s0[T] == "ok" for T in jobs0) and s0["T00001"] == "no_g2", (rc, s0, log)
        bad0, ident0 = compare(h0 / "inputs", jobs0)
        assert not bad0, bad0
        rc, log = run_child(["_allgeom", lst0], env0)
        S0 = pd.read_csv(h0 / "results" / "H_status.csv")
        rid0 = dict(zip(S0.job_id, S0.rxn_id))
        assert rc == 0 and all(int(rid0[lab[r]["job_id"]]) == r for r in d0), (rc, rid0, log)
        out["d0"] = dict(jobs=jobs0, charges={T: man[T]["charge"] for T in jobs0}, inputs_identical=ident0,
                         statuses=s0, allgeom_rxn_id={k: int(v) for k, v in rid0.items()})
        ok = True
    except Exception as e:                                         # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-4000:]
        ok = False
    finally:
        shutil.rmtree(scr, ignore_errors=True)
    rep = dict(written=time.strftime("%F %T"), host=os.uname().nodename, all_pass=ok, results=out)
    p = HERE / "results" / "U_feat_wrap.json"
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, indent=1, default=str))
    os.replace(tmp, p)
    print(json.dumps(rep, indent=1, default=str)[:6000], flush=True)
    return 0 if ok else 3


def cmd_gorder():
    """D1 run env (python 3.9, $D1_PY): feat_geo imports, g0f0.g0_modules (no ORCA run) and the G atom-order gate
    order_g on 1 D1 + 4 D0 reactions (Coley TS / references, two charged), plus two negative cases."""
    scr = SELF / f"gorder_{os.getpid()}"
    os.environ["GEO2_SCRATCH"] = str(scr)
    scr.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import numpy as np
    import feat_geo as FG
    import d0d1 as XD
    out = {}
    try:
        rec = {}
        M = FG.G0F.g0_modules(rec)
        XD.COMB_JOBS = scr / "jobs"
        XD.COMB_JOBS.mkdir(parents=True, exist_ok=True)
        d1 = {int(d["rxn_id"]): d for d in json.loads(XD.D1_LABELS.read_text())}
        d0 = {int(d["rxn_id"]): dict(d, job_id=XD.job_id(d["rxn_id"])) for d in json.loads(XD.D0_LABELS.read_text())}
        meta = {r: sorted(int(x) for x in s.split()) for r, s in XD.read_meta_csv(XD.D1_META).items()}
        meta.update({r: sorted(int(x) for x in s.split()) for r, s in XD.read_meta_csv(XD.D0_META).items()})
        cases = [(1000003, d1[1000003], XD.D1_JOBS / "T00003")] + \
            [(r, d0[r], XD.D0_LABJOBS / XD.job_id(r)) for r in (0, 5, 3312, 3323)]
        for rid, lab, jd in cases:
            T = lab["job_id"]
            os.symlink(jd, XD.COMB_JOBS / T)
            d = scr / "f" / str(rid)
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(jd / "ref_dipole.xyz", d / "rel1.xyz")
            shutil.copyfile(jd / "ref_dipolarophile.xyz", d / "rel2.xyz")
            syms, X = M.pc.read_xyz(jd / "ts.xyz")
            chk = FG.order_g(lab, T, rid, list(syms), np.asarray(X, float), d, meta, M)
            out[str(rid)] = dict(ok=True, charge=int(lab["charge1"]) + int(lab["charge2"]), n_f1=chk["n_f1"])
            if rid == 3312:
                s2 = list(syms)
                i = next(k for k in range(1, len(s2)) if s2[k] != s2[0])
                s2[0], s2[i] = s2[i], s2[0]
                try:
                    FG.order_g(lab, T, rid, s2, np.asarray(X, float), d, meta, M)
                    raise AssertionError("swapped TS elements accepted")
                except FG.Fail as e:
                    out["neg_ts_swap"] = str(e)
                shutil.copyfile(jd / "ref_dipolarophile.xyz", d / "rel1.xyz")
                try:
                    FG.order_g(lab, T, rid, list(syms), np.asarray(X, float), d, meta, M)
                    raise AssertionError("rel1 = dipolarophile accepted")
                except FG.Fail as e:
                    out["neg_rel_swap"] = str(e)
        ok = out.get("neg_ts_swap") == "order_ts_elements_vs_label" and out.get("neg_rel_swap", "").startswith("order_rel1")
    except Exception as e:                                         # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-4000:]
        ok = False
    finally:
        shutil.rmtree(scr, ignore_errors=True)
    rep = dict(written=time.strftime("%F %T"), host=os.uname().nodename, python=sys.executable, all_pass=ok, results=out,
               g0_modules=rec.get("code"))
    p = HERE / "results" / "U_feat_gorder.json"
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rep, indent=1, default=str))
    os.replace(tmp, p)
    print(json.dumps(rep, indent=1, default=str)[:4000], flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["lint"]:
        sys.exit(cmd_lint())
    if a == ["gorder"]:
        sys.exit(cmd_gorder())
    if a == ["unit"]:
        sys.exit(cmd_unit())
    if a == ["wrap"]:
        sys.exit(cmd_wrap())
    raise SystemExit(__doc__)
