"""Unit tests for fix_fail_19 (per spec §7)."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from fix_fail_19.config import (Config, GROUP_A_IDS, GROUP_B_IDS, GROUP_C_IDS)
from fix_fail_19.group_b_spinsweep import _candidates_for_s2, yamaguchi_project
from fix_fail_19.group_c_relax import classify
from fix_fail_19.triage import triage


def _make_manifest(tmp: Path, rids: list[str]) -> Path:
    """Write a minimal manifest.csv with FAIL rows for the given rids."""
    p = tmp / "manifest.csv"
    cols = ["reaction_id", "verdict", "max_abs_res_cons_kcal", "failed_checks"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rid in rids:
            w.writerow({"reaction_id": rid, "verdict": "FAIL",
                         "max_abs_res_cons_kcal": "0.6",
                         "failed_checks": "4"})
    return p


from dataclasses import dataclass
@dataclass
class _FramesStub:
    """Module-level stub so pickle can locate the class on load."""
    numbers: tuple
    positions_R: object
    positions_TS: object
    positions_P: object


def _touch_paths(tmp: Path, rids: list[str]) -> tuple[Path, Path, Path]:
    """Create stub json/halo8/rkf trees so triage's existence check passes."""
    j = tmp / "json"; j.mkdir()
    h = tmp / "halo8"; h.mkdir()
    r = tmp / "rkf"; r.mkdir()
    for rid in rids:
        (j / f"{rid}.json").write_text("{}")
        (h / f"{rid}.frames").write_text("")
        (r / rid).mkdir()
    return j, h, r


class TestTriage(unittest.TestCase):
    """§7.1"""

    def test_19_split_4_7_8(self):
        all_19 = list(GROUP_A_IDS) + list(GROUP_B_IDS) + list(GROUP_C_IDS)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            m = _make_manifest(tmp, all_19)
            j, h, r = _touch_paths(tmp, all_19)
            out = tmp / "out"
            summary = triage(m, j, h, r, out)
        self.assertEqual(summary, {"A": 4, "B": 7, "C": 8})

    def test_unknown_id_exits(self):
        rids = list(GROUP_A_IDS) + list(GROUP_B_IDS) + list(GROUP_C_IDS[:-1]) + ["bogus_id"]
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            m = _make_manifest(tmp, rids)
            j, h, r = _touch_paths(tmp, rids)
            out = tmp / "out"
            with self.assertRaises(SystemExit):
                triage(m, j, h, r, out)

    def test_non_19_count_exits(self):
        rids = list(GROUP_A_IDS) + list(GROUP_B_IDS)  # only 11
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            m = _make_manifest(tmp, rids)
            j, h, r = _touch_paths(tmp, rids)
            out = tmp / "out"
            with self.assertRaises(SystemExit):
                triage(m, j, h, r, out)


class TestGroupBCandidates(unittest.TestCase):
    """§7.3 — candidate selection from <S²>"""

    def setUp(self):
        self.cfg = Config()

    def test_closed_shell(self):
        self.assertEqual(_candidates_for_s2(0.05, self.cfg),
                          ["closed_shell_singlet"])

    def test_bs_or_triplet(self):
        self.assertEqual(_candidates_for_s2(0.83, self.cfg),
                          ["BS_singlet_yamaguchi", "triplet_ferro"])

    def test_triplet_or_quintet(self):
        self.assertEqual(_candidates_for_s2(2.01, self.cfg),
                          ["triplet_ferro", "quintet_ferro"])

    def test_pure_quintet(self):
        self.assertEqual(_candidates_for_s2(6.5, self.cfg), ["quintet_ferro"])


class TestYamaguchi(unittest.TestCase):
    """§7.4 — Yamaguchi spin projection"""

    def test_projected_singlet(self):
        # Inputs from spec
        bs = -50.0
        tr = -30.0
        s2_bs = 1.0
        got = yamaguchi_project(bs, tr, s2_bs)
        self.assertAlmostEqual(got, -70.0, places=6)


class TestGroupCClassify(unittest.TestCase):
    """§7.5 — Group C thresholds"""

    def setUp(self):
        self.cfg = Config()

    def test_pass(self):
        self.assertEqual(classify(0.08, self.cfg), ("PASS", False))

    def test_warn_clean(self):
        self.assertEqual(classify(0.35, self.cfg), ("WARN", False))

    def test_warn_marginal(self):
        self.assertEqual(classify(2.4, self.cfg), ("WARN", True))

    def test_fail(self):
        self.assertEqual(classify(6.0, self.cfg), ("FAIL", True))


class TestGroupAReendpointMock(unittest.TestCase):
    """§7.2 — Group A with mocked ADF runner."""

    def _entry(self, tmp: Path, rid: str) -> dict:
        j = tmp / "json"; j.mkdir(exist_ok=True)
        # Minimal ASR JSON
        orig = {
            "reaction_id": rid,
            "irc_points": {
                "R":  {"energy_kcal_adf": -100.0},
                "TS": {"energy_kcal_adf": -50.0},
                "P":  {"energy_kcal_adf": -90.0},
            },
            "halo8_meta": {"charge": 0},
        }
        (j / f"{rid}.json").write_text(json.dumps(orig))
        return {"reaction_id": rid, "json_path": str(j / f"{rid}.json"),
                "halo8_path": "", "rkf_path": ""}

    def _halo8_cache(self, tmp: Path, rid: str, R, P):
        """Build a pickled frames_cache.pkl with one rxn."""
        import pickle
        import numpy as np
        stub = _FramesStub(numbers=(6, 1),
                            positions_R=np.asarray(R, dtype=float),
                            positions_TS=np.asarray(R, dtype=float),
                            positions_P=np.asarray(P, dtype=float))
        h = tmp / "halo8"; h.mkdir(exist_ok=True)
        with open(h / "frames_cache.pkl", "wb") as fh:
            pickle.dump({rid: stub}, fh)
        return h

    def _opt_factory(self, drift_kcal: float, final_coords):
        """Return a runner that pretends to optimise but reports a fixed drift."""
        def fake(symbols, coords, charge, spin_polarization, workdir, jobname, cfg):
            from fix_fail_19.adf_runner import OptResult
            # The runner is called with R or P coords; we return the same energy
            # shifted by drift relative to the JSON's stored value
            e_R_halo8 = -100.0
            e_P_halo8 = -90.0
            base = e_R_halo8 if jobname == "opt_R" else e_P_halo8
            return OptResult(ok=True, energy_kcal=base - drift_kcal,
                              final_coords=final_coords, rkf_path="/tmp/fake.rkf",
                              converged=True)
        return fake

    def test_pending_revalidate(self):
        # Mock 0.5 kcal drift, same geometry → graph identical, drift < 1.0 tol
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            rid = "Halogen_C4ClH5N2_rxn12962"
            entry = self._entry(tmp, rid)
            R = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            P = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
            self._halo8_cache(tmp, rid, R, P)
            from fix_fail_19.group_a_reendpoint import process_one
            cfg = Config()
            r = process_one(entry, tmp / "halo8", tmp / "out", cfg,
                             runner=self._opt_factory(0.5, R))
            # The mock returns same coords for opt_R and opt_P; graph matches
            self.assertEqual(r["new_verdict"], "PENDING_REVALIDATE")

    def test_excluded_drift(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            rid = "Halogen_C4ClH5N2_rxn12962"
            entry = self._entry(tmp, rid)
            R = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            P = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
            self._halo8_cache(tmp, rid, R, P)
            from fix_fail_19.group_a_reendpoint import process_one
            cfg = Config()
            r = process_one(entry, tmp / "halo8", tmp / "out", cfg,
                             runner=self._opt_factory(50.0, R))
            self.assertEqual(r["new_verdict"], "EXCLUDED")


if __name__ == "__main__":
    unittest.main()
