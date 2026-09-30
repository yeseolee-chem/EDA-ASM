#!/usr/bin/env python3
"""ext_features.py — rev 5 Phase B: the extended feature blocks B1..B6 on the G1 structures.

Spec: docs/specs/REV5_FEATURES_FIGURES.md §B. Column names (112), block membership and every constant come from
rev5_common (BLOCKS, EXT_COLS, DE_FLOOR_EV, FRONT, BM_RHO, VDW_S, RDF_EDGES, BONDI, PEN_ALPHA, GFN2_VALENCE,
SCAN_DELTAS, SCAN_GATE_EH, GATE_*, SMOKE_RXNS); nothing here redefines them.

  python ext_features.py smoke [--workers N] [--out results_rev5/B0_smoke.json]                  B-0 (exit 3 = STOP)
  python ext_features.py slice <slice_id> <n_slices> <out.parquet> [--workers N] [--block B1,..] [--retry-failed]
  python ext_features.py rxn <rxn_id> [--block B1,..] [--cache DIR] [--retry-failed]    one rxn in-process -> JSON row

smoke = SMOKE_RXNS with G1 structures (all neutral) + one extra rxn per nonzero (charge1, charge2) pattern of rows_rev4
(the smallest rxn_id; charged_smoke_rxns), all under the same B-0 stop rules, so gate 1 is also tested on ions.

Inputs (never modified). Only G1-ok rxns ($G1_ROOT/<rid>/.done; accepted = labels_all.json minus xtb_slice.EXCLUDE):
ts.xyz, rel1.xyz, rel2.xyz, result.json, work/ts.hess; the label (charge1 / charge2 = q1 / q2, formed_pairs_ts);
A_idx ($ESPLEY_META); the rev 4 G1 feature row (rev5_common.FEAT_G1); the Coley product p*.xyz (B6, $ESPLEY_PROF).
Geometry = xtb_slice.ts_fragments(label, "g1"), the process_one code path: partition == label A_idx; fA = the TS
atoms A_idx (frag1 = dipole, charge q1), fB = the rest (q2); reacting atoms dip_a, dip_mid, dip_b (in A), dph_a,
dph_b (in B); bond ad = (dip_a, dph_a) = the shorter forming bond at the G1 TS, be = (dip_b, dph_b).
GATE per rxn: the G1 parquet row must be xtb_status ok and its 11 distances must equal xtb_slice.ts_distances()
(1e-9 A), else ext_status geom:g1_parquet:<reason>.

Engines. xtb 6.7.1 ($XTB_BIN, GFN2, --alpb water, 1 thread) for every value that binary prints; tblite 0.7.0
(GFN2-xTB, ALPB water, save-integrals) only for the AO overlap / MO coefficients of B1. A -> bohr: rev5_common.BOHR.

B1 tblite: fA (q1), fB (q2) and the TS (q1+q2), uhf 0. AO count per element = size of the orbital map of a
   single-atom tblite calculator (cached per process); the per-atom AO ranges must equal the calculator's
   orbital -> shell -> atom map and sum to nao. S_AB = S_TS[AO(A), AO(B)] (atom order); MO overlap
   𝒮 = C_A^T S_AB C_B. Occupied = tblite occupation > 1 (must be orbitals 0..nel/2-1 and sum to nel); HOMO = nel/2-1.
   ε in eV; Δε floored at DE_FLOOR_EV in every 𝒮²/Δε term (non-feature columns: n_de_floored = floored pairs of the
   two occ->virt sums, n_de_floored_win = of the two frontier windows).
     mo_pauli_S2_occ   Σ_{i occ A, j occ B} 𝒮_ij²              mo_pauli_S2_front  same over HOMO..HOMO-2 of both
     mo_oi_AB          Σ_{i occ A, a virt B} 𝒮_ia² / Δε_ia, Δε_ia = ε_a(B) - ε_i(A); mo_oi_BA the reverse;
                       mo_oi_tot = mo_oi_AB + mo_oi_BA [1/eV]
     mo_S_HA_LB        |𝒮(HOMO_A, LUMO_B)|;  mo_S_HB_LA = |𝒮(HOMO_B, LUMO_A)|
     mo_dE_HA_LB       ε(LUMO_B) - ε(HOMO_A), unfloored [eV];  mo_dE_HB_LA = ε(LUMO_A) - ε(HOMO_B)
     mo_S2dE_*         𝒮² / Δε of those pairs [1/eV]
     mo_S2dE_win_AB    max 𝒮²/Δε over HOMO_A..HOMO_A-2 x LUMO_B..LUMO_B+2 (FRONT = 3); _BA the reverse
     eps_{H,L}_{dA,dB,rA,rB}  HOMO / LUMO [eV] of fA, fB, rel1, rel2 from the xtb --json orbital energies
   Per rxn (else B1 fails), the B-0 gates on the structures B1 uses: gate 1 |E_tblite - E_xtb| < GATE_E_EH for fA,
   fB, TS (qc_b1_dE_*: vs the G1 parquet xtb_e_d1_eh / xtb_e_d2_eh / xtb_e_ts_eh, same structure and charge); gate 3
   max|C^T S C - I| < GATE_ORTHO for fA, fB, TS; gate 2 fragment overlap == TS block within GATE_S_BLOCK.
   Recorded also: the xtb json vs tblite HOMO / LUMO differences.
B2 geometry only; all inter-fragment pairs (i in A, j in B) of the G1 TS; pair types hh / hH / HH = heavy-heavy /
   heavy-H (either side) / H-H. bm_* = Σ exp(-r/ρ); vdw_pen_sum = Σ max(0, R_i + R_j - r) [A] (Bondi);
   vdw_n_* = #(r < s (R_i + R_j)); dmin_nf_1 / _2 = shortest / 2nd shortest pair other than the two label forming
   pairs [A]; rdf_hh_* = heavy-heavy pair counts in the RDF_EDGES bins (numpy.histogram, last bin closed), named by
   the left edge.
B3 xtb --sp --json of fA and fB (ALPB, q1 / q2): frozen densities. q = the Mulliken `charges` file (as b_elst),
   atomic dipoles μ [e bohr] and quadrupoles Θ [e bohr²] from xtbout.json. R = r_j - r_i (i in A, j in B) [bohr],
   undamped Cartesian interaction tensors, Hartree -> kcal/mol:
     mp_E_qmu    = Σ [q_j (μ_i·R) - q_i (μ_j·R)] / R³
     mp_E_mumu   = Σ [(μ_i·μ_j) R² - 3 (μ_i·R)(μ_j·R)] / R⁵
     mp_E_qTheta = Σ [q_i (R·Θ_j·R) + q_j (R·Θ_i·R)] / R⁵
   QUADRUPOLE CONVENTION. xtb stores traceless atomic quadrupoles in the Buckingham form Θ = ½ Σ q (3 r r - r² 1)
   (GFN2 anisotropic ES, Bannwarth et al., JCTC 2019); the potential of such a Θ is Σ_ab Θ_ab R_a R_b / R⁵, hence
   the q-Θ term above (the plain traceless Q - tr(Q)/3 convention would only rescale this one feature by 3/2).
   The 6 json components are read in xtb's packed order (xx, xy, yy, xz, yz, zz). The order is VERIFIED per
   structure by the trace test: Θ_xx + Θ_yy + Θ_zz = 0 must hold in exactly one of the two usual orders (packed or
   xx, yy, zz, xy, xz, yz), else parse failure; the order found is qc_b3_quad_order.
   Other per-structure self-checks (json parse): json partial charges == `charges` file (5e-4 e); for neutral
   structures Σ q r + Σ μ == the json molecular dipole (1e-3 a.u.; checks sign, frame and unit of μ).
   pen_damp_{10,20} = COULOMB Σ q_i q_j (1 - e^{-α r}) / r [kcal/mol]; pen_ovl_{10,20} = Σ N_i N_j e^{-α r} with
   N = GFN2_VALENCE - q; r in A, α = PEN_ALPHA [1/A].
B4 xtb (ALPB): --vipea of fA, fB, rel1, rel2 -> IP, EA [eV] ("delta SCC IP / EA (eV)"); μ = -(IP+EA)/2,
   η = IP - EA, ω = μ²/2η [eV]; dN_rel = (μ_rB - μ_rA)/(η_rA + η_rB), dN_dist = (μ_dB - μ_dA)/(η_dA + η_dB)
   (A = dipole, B = dipolarophile). --vfukui of fA, fB: f(+), f(-), f(0) of the 5 reacting atoms (fk_p / fk_m /
   fk_0). D4 α(0) [a.u.] of the 5 reacting atoms from the fA / fB atom table, and the molecular α(0) of fA, fB,
   rel1, rel2 ("Mol. α(0) /au"), both parsed from the --json single-point output. gedt = TS Mulliken charges summed
   over A minus q1 = the G1 parquet b_ct (reused; recomputed from a fresh TS single point and asserted to 1e-8).
B5 u = centroid(dph_a, dph_b) - centroid(dip_a, dip_b) at the G1 TS (unit vector); fragment B translated by δ u,
   δ in SCAN_DELTAS [A], fragment geometries frozen; complex single points with xtb_slice.xtb_sp (ALPB, q1+q2);
   fragment terms from fresh fA / fB single points (xtb_slice.xtb_sp, energies unchanged by the translation).
   X(δ) [kcal/mol]: Eint = ΔE_total, pauli = Δrep, oi = Δeht (xtb_slice B_CH8 definitions), elst = the frozen-charge
   Coulomb sum at the new distances. scan_slope_X = [X(+h) - X(-h)] / 2h, scan_curv_X = [X(+h) - 2X(0) + X(-h)] / h²
   (h = 0.05 A), scan_m10_X / scan_p10_X = X(-0.10) / X(+0.10).
   GATE: the δ = 0 recomputation vs the G1 parquet (xtb_e_ts_eh, xtb_e_d1_eh, xtb_e_d2_eh within SCAN_GATE_EH;
   xtb_interaction_kcal, b_pauli, b_oi, b_elst within SCAN_GATE_EH x EH2KCAL), else status_B5 = scan_gate_fail
   (r5_ext_merge.py turns any into a STOP).
B6 ts.hess (ORCA, parsed like D1 orca_direct.parse_hess): nu_imag = the most negative frequency [cm-1, negative as
   ORCA prints it], asserted == result.json imag_freqs_cm[0]; mode_share = result.json imag_mode_forming_share,
   asserted == the D1 run_ts.forming_share formula on the hess mode; mode_async = |Δr_ad| / (|Δr_ad| + |Δr_be|),
   Δr_ij = the imaginary-mode displacement projected on the bond (the forming_share projection).
   Product: the Coley file $ESPLEY_PROF/<rid>/p*.xyz (sorted, first; as D1 run_ts.load_replay), xtb --opt tight
   (ALPB, q1+q2). Gates: heavy graph (fragmenter.heavy_graph) of the optimised product isomorphic to the Coley
   product's; atom map product -> TS = isomorphism of the product heavy graph with (G1 TS heavy graph +
   formed_pairs_ts edges): the identity when it is one (the Coley product keeps the TS atom order), else the lowest
   heavy-atom Kabsch RMSD over at most MAX_ISO enumerated isomorphisms (a non-identity map whose enumeration hit
   MAX_ISO fails B6: iso_enumeration_truncated). Assert: the mapped forming pairs are bonds of the Coley START
   product graph (same atom order as xtbopt.xyz; not implied by the map, which is onto the optimised product graph).
   xtb_dErxn = E_P - e_rel1_eh - e_rel2_eh (G1 result.json, same xtb --opt tight / ALPB level) [kcal/mol];
   prog_ad / prog_be = r_TS / r_P of the two forming bonds.
   Information level: the product start structure is the DFT (Coley) product, the same level as the G1 references
   (started from the DFT references). G2 (autodE from SMILES) must use the autodE product instead.

A block value that is not finite fails its block ("nonfinite:<cols>"), so an ok row never carries NaN; r5_ext_merge.py
turns any such failure among the rows_rev4 rxns into a STOP (spec B-7 NaN gate), not a dropped row.
Row: rxn_id, geom ('g1'), ext_status ('ok', '<block>:<reason>' of the first failing block in B1..B6 order, or
'geom:<reason>'), status_B1..status_B6, the 112 EXT_COLS, n_de_floored(_win), qc_* self-check columns, t_B1..t_B6
and t_total_s (wall s of one single-threaded worker = core-s).

Cache / idempotency: $R5_SCRATCH/ext/<rid>/{geom,B1..B6}.json (written atomically, keyed by CODE_VERSION and a
fingerprint of the G1 inputs), so a rerun recomputes only missing blocks (--retry-failed: also failed ones).
Failures are cached (deterministic); OSErrors (missing binary, disk) are not. A code change that alters any value
must bump CODE_VERSION (every record of the old version is then recomputed). The smoke always starts from scratch
($R5_SCRATCH/ext_smoke/<rid>/ is removed first). slice: an existing <out.parquet> is skipped; every rxn runs in a
persistent spawn worker process (1 thread; xtb / tblite scratch in a temp dir per call); a worker that dies (e.g. a
native crash) marks only its current rxn (crash.json) and is replaced.
Exit codes: 0 ok, 1 G1 incomplete (an accepted rxn without .done / .fail_*), 2 bad inputs, 3 B-0 gate or check
failed (STOP), 4 slice not written: > SLICE_MAX_FAIL of its rows failed or workers crash systematically.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # before numpy / tblite start their thread pools

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing.connection import wait as mp_wait  # noqa: E402
from pathlib import Path  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from networkx.algorithms.isomorphism import GraphMatcher  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev5_common as R5  # noqa: E402
import xtb_slice as xs  # noqa: E402

fr = xs.fr

# ---------------------------------------------------------------- run configuration (not features)
CODE_VERSION = "r5-ext-2"                    # stored in every cache record; a record of another version is recomputed
#                                              (r5-ext-2: B1 per-rxn gate 1, B6 Coley-product bond assert + MAX_ISO guard)
CACHE_ROOT = R5.SCRATCH / "ext"
SMOKE_ROOT = R5.SCRATCH / "ext_smoke"
SLICE_DIR = CACHE_ROOT / "slices"
N_SLICES = 18
SMOKE_OUT = R5.RES5 / "B0_smoke.json"
HOST = socket.gethostname()
SOLVENT = xs.SOLVENT                         # "water"
COULOMB = xs.COULOMB                         # kcal A / (mol e²), as b_elst
FRAG_STRUCT = {"dA": "fA", "dB": "fB", "rA": "rel1", "rB": "rel2"}      # rev5_common.FRAGS -> structure
assert tuple(FRAG_STRUCT) == tuple(R5.FRAGS)
COMPUTE_ORDER = ("B2", "B3", "B4", "B5", "B6", "B1")   # tblite (B1) last: an in-process crash loses only B1
XTB_TIMEOUT_S = 1800
XTB_OPT_TIMEOUT_S = 6 * 3600
SLICE_MAX_FAIL = 0.20                        # operational guard of one slice (the pre-registered gate is MAX_EXT_FAIL)
CRASH_ABORT = 0.5                            # stop a pool when > half of >= 20 finished rxns killed their worker
# tolerances of parser / engine self-checks (consistency checks, not features)
TOL_DIST_A = 1e-9                            # 11 distances vs the G1 parquet
TOL_GEDT = 1e-8                              # spec B4
TOL_TRACE = 2e-3                             # atomic quadrupole trace test (component order)
TOL_DIP_AU = 1e-3                            # Σ q r + Σ μ vs the json dipole (neutral structures)
TOL_Q_JSON = 5e-4                            # json partial charges vs the charges file
TOL_EPS_EV = 5e-3                            # xtb json vs tblite HOMO / LUMO (B-0 check)
TOL_FUKUI = 2e-3                             # f(0) vs (f(+) + f(-))/2, values printed with 3 decimals
TOL_HESS = 1e-6                              # nu_imag [cm-1] and mode_share recomputed vs result.json
TOL_HESS_GEOM_A = 1e-4                       # .hess $atoms == ts.xyz up to a translation (g1_geom.py gate)
MAX_ISO = 10000                              # product <-> TS isomorphisms enumerated at most

DIST_COLS = ["dist_R_dip_ab", "dist_R_dip_bc", "dist_R_dip_ac", "dist_R_dph_ab", "dist_TS_dip_ab", "dist_TS_dip_bc",
             "dist_TS_dip_ac", "dist_TS_dph_ab", "dist_TS_form_ad", "dist_TS_form_be", "dist_TS_diag_ae"]
G1_COLS = ["rxn_id", "geom", "xtb_status", "xtb_e_ts_eh", "xtb_e_d1_eh", "xtb_e_d2_eh", "xtb_interaction_kcal",
           "b_ct", "b_pauli", "b_oi", "b_elst"] + DIST_COLS
EXTRA = {                                    # non-feature columns written by each block
    "B1": ["n_de_floored", "n_de_floored_win", "qc_b1_nao_ts", "qc_b1_ortho_max", "qc_b1_sblock_max",
           "qc_b1_dE_fA_eh", "qc_b1_dE_fB_eh", "qc_b1_dE_ts_eh", "qc_b1_dHOMO_dA_ev", "qc_b1_dHOMO_dB_ev",
           "qc_b1_dLUMO_dA_ev", "qc_b1_dLUMO_dB_ev"],
    "B2": [],
    "B3": ["qc_b3_dip_check_au", "qc_b3_q_json_maxdiff", "qc_b3_quad_order"],
    "B4": ["qc_b4_gedt_diff"],
    "B5": ["qc_b5_dE_ts_eh", "qc_b5_dE_fA_eh", "qc_b5_dE_fB_eh", "qc_b5_dch_max_kcal"],
    "B6": ["qc_b6_nu_diff_cm", "qc_b6_mode_share_diff", "qc_b6_map_identity", "qc_b6_n_iso", "qc_b6_map_rmsd_A",
           "qc_b6_e_product_eh", "qc_b6_product_file"],
}
STR_EXTRA = {"qc_b3_quad_order", "qc_b6_product_file"}
EXTRA_COLS = [c for b in R5.BLOCK_ORDER for c in EXTRA[b]]
ROW_COLS = (["rxn_id", "geom", "ext_status"] + [f"status_{b}" for b in R5.BLOCK_ORDER] + list(R5.EXT_COLS)
            + EXTRA_COLS + [f"t_{b}" for b in R5.BLOCK_ORDER] + ["t_total_s"])
assert not set(EXTRA_COLS) & set(R5.EXT_COLS) and len(set(ROW_COLS)) == len(ROW_COLS)

QUAD_ORDERS = {"xx,xy,yy,xz,yz,zz": (0, 2, 5), "xx,yy,zz,xy,xz,yz": (0, 1, 2)}   # diagonal positions
NUM = r"(-?\d*\.\d+(?:[eEdD][-+]?\d+)?)"
TOTAL_E_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
IP_RE = re.compile(rf"delta\s+SCC\s+IP\s*\(eV\)\s*:?\s*{NUM}")
EA_RE = re.compile(rf"delta\s+SCC\s+EA\s*\(eV\)\s*:?\s*{NUM}")
FUKUI_ROW = re.compile(r"^\s*(\d+)\s*([A-Za-z]{1,2})\s+(\S.*?)\s*$")          # idx sym | value tail
D4_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,2})\s+(\S.*?)\s*$")        # idx Z sym | value tail
MOL_ALPHA_RE = re.compile(rf"Mol\.\s+\S*\(0\)\s*/\s*au\s*:\s*{NUM}")


class BlockFail(Exception):
    """An expected per-rxn failure: the block status becomes the (short) reason; cached like a result."""

    def __init__(self, reason, qc=None):
        self.detail = str(reason)
        super().__init__(self.detail[:200])
        self.qc = dict(qc or {})


class ParseError(BlockFail):
    """An xtb / ORCA output that does not have the expected format (the message names the expected pattern)."""

    def __init__(self, msg, qc=None):
        super().__init__(f"parse:{msg}", qc)


def die(code, msg):
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


def _f(s):
    return float(str(s).replace("D", "E").replace("d", "e"))


def n_valence(syms):
    return int(sum(R5.GFN2_VALENCE[s] for s in syms))


# ================================================================ xtb 6.7.1 binary
def _write_xyz(path, syms, xyz):
    """Same format as xtb_slice.xtb_sp (8 decimals)."""
    with open(path, "w") as f:
        f.write(f"{len(syms)}\n\n")
        for s_, x in zip(syms, np.asarray(xyz, dtype=float)):
            f.write(f"{s_} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}\n")


def xtb_run(syms, xyz, charge, args, keep=(), timeout=XTB_TIMEOUT_S):
    """xtb GFN2 / ALPB(water) at `charge` with extra `args`, in a fresh temp dir -> (stdout, {kept file: text})."""
    with tempfile.TemporaryDirectory(prefix="r5ext_") as d:
        _write_xyz(os.path.join(d, "m.xyz"), syms, xyz)
        cmd = [xs.XTB_BIN, "m.xyz", "--gfn", "2", "--chrg", str(int(charge)), "--alpb", SOLVENT, *args]
        try:
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("XTB_THREADS", "1")))
        except subprocess.TimeoutExpired:
            raise BlockFail(f"xtb_timeout:{' '.join(args)}")
        files = {k: (Path(d) / k).read_text(errors="replace") for k in keep if (Path(d) / k).is_file()}
    if r.returncode != 0 or "abnormal termination" in (r.stdout + r.stderr):
        tail = " ".join((r.stderr.strip() or r.stdout.strip()).splitlines()[-3:])
        raise BlockFail(f"xtb_fail:{' '.join(args)}:exit{r.returncode}:{tail}")
    return r.stdout, files


def _jkey(d, alts, what, required=True):
    """The xtbout.json key of `what`: the first (need, avoid) substring alternative that matches exactly one key."""
    for need, avoid in alts:
        ks = [k for k in d if all(s in k.lower() for s in need) and not any(s in k.lower() for s in avoid)]
        if len(ks) == 1:
            return ks[0]
        if len(ks) > 1:
            raise ParseError(f"xtbout.json: several keys for {what}: {ks}")
    if required:
        raise ParseError(f"xtbout.json: no key for {what} (substrings {alts}); keys present: {sorted(d)}")
    return None


def quad_matrices(th, tag):
    """(n, 6) atomic quadrupoles -> ((n, 3, 3), component order). The order is the one in which every atom's
    quadrupole is traceless (trace test); a structure where neither order is traceless is a parse failure."""
    tr = {o: float(np.abs(th[:, list(ix)].sum(axis=1)).max()) for o, ix in QUAD_ORDERS.items()}
    ok = [o for o, t in tr.items() if t <= TOL_TRACE]
    if not ok:
        raise ParseError(f"{tag}: atomic quadrupoles are traceless in neither component order (max |trace| {tr})")
    order = ok[0] if len(ok) == 1 else "xx,xy,yy,xz,yz,zz(ambiguous)"
    if order.startswith("xx,xy"):
        xx, xy, yy, xz, yz, zz = th.T
    else:
        xx, yy, zz, xy, xz, yz = th.T
    T = np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1), np.stack([xz, yz, zz], -1)], -2)
    return T, order


def _floats(s):
    try:
        return [_f(t) for t in s.split()]
    except ValueError:
        return None


def _table_rows(lines, start, n, row_re, need):
    """Up to n consecutive rows after line `start` (blank / '----' lines before the first row are skipped):
    (match, floats) of the lines matching row_re whose last group splits into > need floats; stops at any other line."""
    rows, k = [], start + 1
    while k < len(lines) and len(rows) < n:
        m = row_re.match(lines[k])
        vals = _floats(m.group(m.lastindex)) if m else None
        if vals is not None and len(vals) > need:
            rows.append((m, vals))
        elif rows or not set(lines[k].strip()) <= set("-=*_ "):
            break
        k += 1
    return rows


def parse_d4(out, syms, tag):
    """(atomic α(0) (n,), molecular α(0)) [a.u.] from the D4 block of a GFN2 single-point output: header
    '#  Z  covCN  q  C6AA  α(0)', rows 'idx Z sym <values>' (the α(0) column is located from the right by the header,
    so an extra column does not shift it), then 'Mol. α(0) /au : value'."""
    n, lines = len(syms), out.splitlines()
    hdr = [k for k, l in enumerate(lines) if "covCN" in l and "C6AA" in l and "(0)" in l]
    if not hdr:
        raise ParseError(f"{tag}: D4 table header (a line with 'covCN', 'C6AA' and '(0)') not in the xtb output")
    htok = lines[hdr[-1]].split()
    ia = [i for i, t in enumerate(htok) if "(0)" in t]
    if len(ia) != 1:
        raise ParseError(f"{tag}: D4 header {htok} has {len(ia)} '(0)' columns, expected 1")
    from_end = len(htok) - 1 - ia[0]
    rows = _table_rows(lines, hdr[-1], n, D4_ROW, from_end)
    if len(rows) != n:
        raise ParseError(f"{tag}: D4 table has {len(rows)} rows 'idx Z sym values', expected {n}")
    for i, ((m, _), s) in enumerate(zip(rows, syms)):
        if int(m.group(1)) != i + 1 or int(m.group(2)) != fr.Z[s] or m.group(3).lower() != s.lower():
            raise ParseError(f"{tag}: D4 row {i + 1} starts {m.groups()[:3]}, expected ({i + 1}, {fr.Z[s]}, {s})")
    alpha = np.array([vals[len(vals) - 1 - from_end] for _, vals in rows])
    if not np.all(alpha > 0):
        raise ParseError(f"{tag}: D4 α(0) column not positive: {alpha[:5]}")
    ms = MOL_ALPHA_RE.findall(out)
    if not ms:
        raise ParseError(f"{tag}: 'Mol. α(0) /au : <value>' line not in the xtb output")
    return alpha, _f(ms[-1])


def parse_json_sp(out, files, syms, xyz, charge, tag):
    """xtb --sp --json of one structure -> charges (`charges` file), μ (n,3), Θ (n,3,3), HOMO / LUMO [eV],
    D4 α(0) (atomic, molecular) and the self-check values (see the module docstring, B3)."""
    n = len(syms)
    if "charges" not in files:
        raise ParseError(f"{tag}: no `charges` file after xtb --sp --json")
    q = np.array([_f(x) for x in files["charges"].split()])
    if q.size != n:
        raise ParseError(f"{tag}: `charges` has {q.size} values for {n} atoms")
    if "xtbout.json" not in files:
        raise ParseError(f"{tag}: xtbout.json not written (xtb --json)")
    try:
        d = json.loads(files["xtbout.json"])
    except json.JSONDecodeError as e:
        raise ParseError(f"{tag}: xtbout.json is not valid JSON ({e})")
    # ---- orbital energies / occupations -> HOMO, LUMO [eV]
    ke = _jkey(d, [(("orbital energ",), ())], "orbital energies")
    kl = ke.lower()
    if "ev" in kl:
        scale = 1.0
    elif any(u in kl for u in ("eh", "hartree", "a.u")):
        scale = R5.EH2EV
    else:
        raise ParseError(f"{tag}: unit of xtbout.json '{ke}' unknown (expected 'eV' or 'Eh' in the key)")
    eps = np.asarray(d[ke], dtype=float).ravel() * scale
    nel = n_valence(syms) - int(charge)
    if nel % 2:
        raise BlockFail(f"{tag}_odd_electron_count")
    nocc = nel // 2
    ko = _jkey(d, [(("occupation",), ())], "orbital occupations", required=False)
    if ko is not None:
        occ = np.asarray(d[ko], dtype=float).ravel()
        if occ.size != eps.size:
            raise ParseError(f"{tag}: {occ.size} occupations for {eps.size} orbital energies")
        if not np.array_equal(np.flatnonzero(occ > 1.0), np.arange(nocc)):
            raise BlockFail(f"{tag}_xtb_occupations_not_aufbau")
    if eps.size <= nocc or np.any(np.diff(eps) < -1e-6):
        raise ParseError(f"{tag}: {eps.size} orbital energies (ascending?) for {nocc} occupied orbitals")
    # ---- atomic multipoles
    km = _jkey(d, [(("atomic", "dipole"), ())], "atomic dipole moments")
    kt = _jkey(d, [(("atomic", "quadrupole"), ())], "atomic quadrupole moments")
    mu, th = np.asarray(d[km], dtype=float), np.asarray(d[kt], dtype=float)
    if mu.size != 3 * n or th.size != 6 * n:
        raise ParseError(f"{tag}: '{km}' has {mu.size} values (expected {3 * n}), '{kt}' {th.size} (expected {6 * n})")
    mu, th = mu.reshape(n, 3), th.reshape(n, 6)
    theta, order = quad_matrices(th, tag)
    # ---- self-checks: json charges == charges file; Σ q r + Σ μ == molecular dipole (neutral only)
    kq = _jkey(d, [(("partial charge",), ()), (("charges",), ("atomic",))], "partial charges", required=False)
    q_diff = np.nan
    if kq is not None:
        qj = np.asarray(d[kq], dtype=float).ravel()
        if qj.size != n:
            raise ParseError(f"{tag}: '{kq}' has {qj.size} values for {n} atoms")
        q_diff = float(np.abs(qj - q).max())
        if q_diff > TOL_Q_JSON:
            raise ParseError(f"{tag}: '{kq}' differs from the charges file by {q_diff:.1e} e")
    dip_check = np.nan
    if int(charge) == 0:
        kd = [k for k in d if "dipole" in k.lower() and "atomic" not in k.lower()]
        if len(kd) > 1:
            kd = [k for k in kd if k.strip().lower() == "dipole"]
        if len(kd) == 1:
            dv = np.asarray(d[kd[0]], dtype=float).ravel()
            if dv.size >= 3:
                dv = dv[:3] / (xs.AU2DEBYE if "debye" in kd[0].lower() else 1.0)
                calc = (q[:, None] * np.asarray(xyz, dtype=float) * R5.BOHR).sum(axis=0) + mu.sum(axis=0)
                dip_check = float(np.abs(calc - dv).max())
                if dip_check > TOL_DIP_AU:
                    raise ParseError(f"{tag}: Σ q r + Σ μ differs from '{kd[0]}' by {dip_check:.1e} a.u. "
                                     f"(atomic dipole sign / frame / unit)")
    alpha_atoms, alpha_mol = parse_d4(out, syms, tag)
    return dict(q=q, mu=mu, theta=theta, quad_order=order, q_json_maxdiff=q_diff, dip_check=dip_check,
                eps_homo=float(eps[nocc - 1]), eps_lumo=float(eps[nocc]), alpha_atoms=alpha_atoms, alpha_mol=alpha_mol)


def parse_vipea(out, tag):
    ip, ea = IP_RE.findall(out), EA_RE.findall(out)
    if not ip or not ea:
        raise ParseError(f"{tag}: 'delta SCC IP (eV): <v>' / 'delta SCC EA (eV): <v>' not in the xtb --vipea output")
    return _f(ip[-1]), _f(ea[-1])


def parse_fukui(out, syms, tag):
    """(n, 3) = f(+), f(-), f(0) per atom from the xtb --vfukui table: header naming 'f(+)', 'f(-)', 'f(0)', rows
    'idx sym <values>' (idx and symbol may be glued); each column is located from the right by the header.
    Self-check: f(0) = (f(+) + f(-)) / 2 within the 3-decimal print precision (else the columns are misread)."""
    n, lines = len(syms), out.splitlines()
    hdr = [k for k, l in enumerate(lines) if "f(+)" in l and "f(-)" in l and "f(0)" in l]
    if not hdr:
        raise ParseError(f"{tag}: Fukui header (a line with 'f(+)', 'f(-)', 'f(0)') not in the xtb --vfukui output")
    htok = lines[hdr[-1]].split()
    from_end = [len(htok) - 1 - next(i for i, t in enumerate(htok) if key in t) for key in ("f(+)", "f(-)", "f(0)")]
    rows = _table_rows(lines, hdr[-1], n, FUKUI_ROW, max(from_end))
    if len(rows) != n:
        raise ParseError(f"{tag}: Fukui table has {len(rows)} rows 'idx sym f(+) f(-) f(0)', expected {n}")
    for i, ((m, _), s) in enumerate(zip(rows, syms)):
        if int(m.group(1)) != i + 1 or m.group(2).lower() != s.lower():
            raise ParseError(f"{tag}: Fukui row {i + 1} starts {m.groups()[:2]}, expected ({i + 1}, {s})")
    f = np.array([[vals[len(vals) - 1 - e] for e in from_end] for _, vals in rows])
    dev = float(np.abs(f[:, 2] - 0.5 * (f[:, 0] + f[:, 1])).max())
    if dev > TOL_FUKUI:
        raise ParseError(f"{tag}: Fukui f(0) != (f(+) + f(-))/2 (max dev {dev:.1e}): columns misread")
    return f


def read_xyz_text(text):
    lines = text.splitlines()
    n = int(lines[0].split()[0])
    rows = [lines[2 + k].split() for k in range(n)]
    return [r[0] for r in rows], np.array([[_f(v) for v in r[1:4]] for r in rows])


# ================================================================ tblite 0.7.0 (B1 only)
_AO_COUNT: dict = {}


def _silent(_msg):
    return None


def tb_calc(syms, xyz, charge, uhf=0, solvent=True, integrals=True):
    from tblite.interface import Calculator
    nums = np.array([fr.Z[s] for s in syms], dtype=np.int32)
    calc = Calculator("GFN2-xTB", nums, np.asarray(xyz, dtype=float) * R5.BOHR, charge=float(charge), uhf=int(uhf),
                      color=False, logger=_silent)
    calc.set("verbosity", 0)
    if integrals:
        calc.set("save-integrals", 1)
    if solvent:
        calc.add("alpb-solvation", SOLVENT)
    return calc


def ao_count(sym):
    """GFN2 AOs of one element = size of the orbital map of a single-atom tblite calculator (cached per process)."""
    if sym not in _AO_COUNT:
        calc = tb_calc([sym], np.zeros((1, 3)), 0, uhf=R5.GFN2_VALENCE[sym] % 2, solvent=False, integrals=False)
        _AO_COUNT[sym] = int(np.asarray(calc.get("orbital-map")).size)
    return _AO_COUNT[sym]


def ao_atoms(syms):
    return np.repeat(np.arange(len(syms)), [ao_count(s) for s in syms])


def tb_sp(syms, xyz, charge, tag):
    """tblite GFN2 / ALPB(water) single point with integrals -> energy [Eh], ε [Eh], occupations, C [ao, mo], S,
    AO -> atom map (checked against the calculator's own map), nel and max|C^T S C - I|."""
    from tblite.exceptions import TBLiteRuntimeError, TBLiteValueError
    nel = n_valence(syms) - int(charge)
    if nel % 2:
        raise BlockFail(f"{tag}_odd_electron_count")
    try:
        calc = tb_calc(syms, xyz, charge)
        res = calc.singlepoint()
        orb = np.asarray(calc.get("orbital-map")).astype(int)
        sh = np.asarray(calc.get("shell-map")).astype(int)
        nao = int(res.get("norbitals"))
        energy = float(res.get("energy"))
        eps = np.asarray(res.get("orbital-energies"), dtype=float)
        occ = np.asarray(res.get("orbital-occupations"), dtype=float)
        C = np.asarray(res.get("orbital-coefficients"), dtype=float)
        S = np.asarray(res.get("overlap-matrix"), dtype=float)
    except (TBLiteRuntimeError, TBLiteValueError) as e:
        raise BlockFail(f"tblite_fail:{tag}:{type(e).__name__}:{e}")
    if eps.shape != (nao,) or occ.shape != (nao,) or C.shape != (nao, nao) or S.shape != (nao, nao):
        raise BlockFail(f"tblite_shapes:{tag}:eps{eps.shape},occ{occ.shape},C{C.shape},S{S.shape},nao{nao}")
    ao = ao_atoms(syms)
    if ao.size != nao:
        raise BlockFail(f"ao_count_sum_{ao.size}_ne_nao_{nao}:{tag}")
    if orb.size != nao or not np.array_equal(ao, (sh - sh.min())[orb - orb.min()]):
        raise BlockFail(f"ao_map_mismatch:{tag}")
    ortho = float(np.abs(C.T @ S @ C - np.eye(nao)).max())
    return dict(energy=energy, eps=eps, occ=occ, C=C, S=S, ao_atom=ao, nao=nao, nel=nel, ortho=ortho)


def n_occupied(t, tag):
    nel, occ, eps = t["nel"], t["occ"], t["eps"]
    nocc = nel // 2
    if abs(float(occ.sum()) - nel) > 1e-6:
        raise BlockFail(f"{tag}_occupation_sum_{float(occ.sum()):.6f}_nel_{nel}")
    if not np.array_equal(np.flatnonzero(occ > 1.0), np.arange(nocc)):
        raise BlockFail(f"{tag}_occupations_not_aufbau")
    if np.any(np.diff(eps) < -1e-10):
        raise BlockFail(f"{tag}_orbital_energies_not_ascending")
    if not 0 < nocc < eps.size:
        raise BlockFail(f"{tag}_no_homo_or_lumo")
    return nocc


def mo_features(tA, tB, S_AB, noA, noB):
    """The 13 mo_* features from the fragment MOs and the inter-fragment AO overlap block (docstring, B1)."""
    SS = tA["C"].T @ S_AB @ tB["C"]                               # (MO of A, MO of B)
    S2 = SS ** 2
    eA, eB = tA["eps"] * R5.EH2EV, tB["eps"] * R5.EH2EV
    fl = R5.DE_FLOOR_EV
    hA, lA, hB, lB = noA - 1, noA, noB - 1, noB
    frA, frB = np.arange(max(0, noA - R5.FRONT), noA), np.arange(max(0, noB - R5.FRONT), noB)
    vwA = np.arange(noA, min(eA.size, noA + R5.FRONT)) - noA        # LUMO..LUMO+2 as virtual offsets
    vwB = np.arange(noB, min(eB.size, noB + R5.FRONT)) - noB
    dAB = eB[None, noB:] - eA[:noA, None]                            # rows occ A, cols virt B
    dBA = eA[noA:, None] - eB[None, :noB]                            # rows virt A, cols occ B
    tAB = S2[:noA, noB:] / np.maximum(dAB, fl)
    tBA = S2[noA:, :noB] / np.maximum(dBA, fl)
    v = {"mo_pauli_S2_occ": float(S2[:noA, :noB].sum()),
         "mo_pauli_S2_front": float(S2[np.ix_(frA, frB)].sum()),
         "mo_oi_AB": float(tAB.sum()), "mo_oi_BA": float(tBA.sum()),
         "mo_S_HA_LB": float(abs(SS[hA, lB])), "mo_S_HB_LA": float(abs(SS[lA, hB])),
         "mo_dE_HA_LB": float(eB[lB] - eA[hA]), "mo_dE_HB_LA": float(eA[lA] - eB[hB]),
         "mo_S2dE_HA_LB": float(tAB[hA, 0]), "mo_S2dE_HB_LA": float(tBA[0, hB]),
         "mo_S2dE_win_AB": float(tAB[np.ix_(frA, vwB)].max()), "mo_S2dE_win_BA": float(tBA[np.ix_(vwA, frB)].max())}
    v["mo_oi_tot"] = v["mo_oi_AB"] + v["mo_oi_BA"]
    n_fl = int((dAB < fl).sum() + (dBA < fl).sum())
    n_fl_win = int((dAB[np.ix_(frA, vwB)] < fl).sum() + (dBA[np.ix_(vwA, frB)] < fl).sum())
    return v, n_fl, n_fl_win


# ================================================================ one reaction
class Ctx:
    """One rxn: structures + charges, the G1 parquet row, and memoised engine calls (each structure runs once)."""

    def __init__(self, label, g, g1row, cache_dir=None):
        self.label, self.g, self.g1, self.cache_dir = label, g, g1row, cache_dir
        self.rid = int(label["rxn_id"])
        self.q1, self.q2 = int(label.get("charge1", 0) or 0), int(label.get("charge2", 0) or 0)
        self.st = {"fA": (list(g.fA[0]), np.asarray(g.fA[1], dtype=float), self.q1),
                   "fB": (list(g.fB[0]), np.asarray(g.fB[1], dtype=float), self.q2),
                   "rel1": (list(g.rel1[0]), np.asarray(g.rel1[1], dtype=float), self.q1),
                   "rel2": (list(g.rel2[0]), np.asarray(g.rel2[1], dtype=float), self.q2),
                   "ts": (list(g.ts_syms), np.asarray(g.ts_xyz, dtype=float), self.q1 + self.q2)}
        self._memo = {}

    def _get(self, key, fn):
        if key not in self._memo:
            try:
                self._memo[key] = (True, fn())
            except BlockFail as e:
                self._memo[key] = (False, e)
        ok, v = self._memo[key]
        if not ok:
            raise v
        return v

    def json_sp(self, name):
        syms, xyz, q = self.st[name]

        def run():
            out, files = xtb_run(syms, xyz, q, ["--sp", "--json"], keep=("xtbout.json", "charges"))
            return parse_json_sp(out, files, syms, xyz, q, f"{name}:json")
        return self._get(("json", name), run)

    def vipea(self, name):
        syms, xyz, q = self.st[name]
        return self._get(("vipea", name), lambda: parse_vipea(xtb_run(syms, xyz, q, ["--vipea"])[0], f"{name}:vipea"))

    def vfukui(self, name):
        syms, xyz, q = self.st[name]
        return self._get(("vfukui", name),
                         lambda: parse_fukui(xtb_run(syms, xyz, q, ["--vfukui"])[0], syms, f"{name}:vfukui"))

    def sp(self, name):
        """xtb_slice.xtb_sp (the rev 4 feature engine) of one structure."""
        syms, xyz, q = self.st[name]

        def run():
            try:
                return xs.xtb_sp(syms, xyz, q)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                raise BlockFail(f"xtb_sp_fail:{name}:{type(e).__name__}:{e}")
        return self._get(("sp", name), run)

    def tb(self, name):
        syms, xyz, q = self.st[name]
        return self._get(("tb", name), lambda: tb_sp(syms, xyz, q, name))


def react_pos(g):
    """Reacting atom -> (fragment structure, index inside that fragment)."""
    return {"dip_a": ("fA", g.posA[g.dip_a]), "dip_mid": ("fA", g.posA[g.dip_mid]), "dip_b": ("fA", g.posA[g.dip_b]),
            "dph_a": ("fB", g.posB[g.dph_a]), "dph_b": ("fB", g.posB[g.dph_b])}


def block_b1(c):
    g, g1 = c.g, c.g1
    tA, tB, tT = c.tb("fA"), c.tb("fB"), c.tb("ts")
    aoA = np.concatenate([np.flatnonzero(tT["ao_atom"] == i) for i in g.A_idx])
    aoB = np.concatenate([np.flatnonzero(tT["ao_atom"] == i) for i in g.B_idx])
    if aoA.size != tA["nao"] or aoB.size != tB["nao"]:
        raise BlockFail(f"ao_partition:{aoA.size}/{tA['nao']},{aoB.size}/{tB['nao']}")
    sblock = max(float(np.abs(tA["S"] - tT["S"][np.ix_(aoA, aoA)]).max()),
                 float(np.abs(tB["S"] - tT["S"][np.ix_(aoB, aoB)]).max()))
    ortho = max(tA["ortho"], tB["ortho"], tT["ortho"])
    noA, noB = n_occupied(tA, "fA"), n_occupied(tB, "fB")
    jA, jB = c.json_sp("fA"), c.json_sp("fB")
    qc = {"qc_b1_nao_ts": tT["nao"], "qc_b1_ortho_max": ortho, "qc_b1_sblock_max": sblock,
          "qc_b1_dE_fA_eh": tA["energy"] - float(g1["xtb_e_d1_eh"]),
          "qc_b1_dE_fB_eh": tB["energy"] - float(g1["xtb_e_d2_eh"]),
          "qc_b1_dE_ts_eh": tT["energy"] - float(g1["xtb_e_ts_eh"]),
          "qc_b1_dHOMO_dA_ev": tA["eps"][noA - 1] * R5.EH2EV - jA["eps_homo"],
          "qc_b1_dHOMO_dB_ev": tB["eps"][noB - 1] * R5.EH2EV - jB["eps_homo"],
          "qc_b1_dLUMO_dA_ev": tA["eps"][noA] * R5.EH2EV - jA["eps_lumo"],
          "qc_b1_dLUMO_dB_ev": tB["eps"][noB] * R5.EH2EV - jB["eps_lumo"]}
    dEmax = float(np.max(np.abs([qc[k] for k in ("qc_b1_dE_fA_eh", "qc_b1_dE_fB_eh", "qc_b1_dE_ts_eh")])))
    if not dEmax < R5.GATE_E_EH:                                     # gate 1 per rxn (np.max keeps NaN: fails too)
        raise BlockFail(f"engine_gate1:{dEmax:.1e}", qc)
    if not ortho < R5.GATE_ORTHO:
        raise BlockFail(f"ortho_gate:{ortho:.1e}", qc)
    if not sblock < R5.GATE_S_BLOCK:
        raise BlockFail(f"overlap_block_gate:{sblock:.1e}", qc)
    v, n_fl, n_fl_win = mo_features(tA, tB, tT["S"][np.ix_(aoA, aoB)], noA, noB)
    for f, st in FRAG_STRUCT.items():
        j = c.json_sp(st)
        v[f"eps_H_{f}"], v[f"eps_L_{f}"] = j["eps_homo"], j["eps_lumo"]
    qc.update(n_de_floored=n_fl, n_de_floored_win=n_fl_win)
    return v, qc


def block_b2(c):
    g = c.g
    syms, A, B = g.ts_syms, g.A_idx, g.B_idx
    D = g.D_ts[np.ix_(A, B)]
    hA = np.array([syms[i] != "H" for i in A])
    hB = np.array([syms[j] != "H" for j in B])
    hh = hA[:, None] & hB[None, :]
    HH = ~hA[:, None] & ~hB[None, :]
    hH = ~(hh | HH)
    v = {}
    for p, m in (("hh", hh), ("hH", hH), ("HH", HH)):
        for k, rho in R5.BM_RHO.items():
            v[f"bm_{p}_{k}"] = float(np.exp(-D[m] / rho).sum())
    Rs = np.array([R5.BONDI[syms[i]] for i in A])[:, None] + np.array([R5.BONDI[syms[j]] for j in B])[None, :]
    v["vdw_pen_sum"] = float(np.maximum(0.0, Rs - D).sum())
    for k, s in R5.VDW_S.items():
        v[f"vdw_n_{k}"] = float((D < s * Rs).sum())
    formed = {frozenset(p) for p in g.pairs}
    nf = sorted(float(D[a, b]) for a, i in enumerate(A) for b, j in enumerate(B) if frozenset((i, j)) not in formed)
    if len(nf) < 2:
        raise BlockFail("fewer_than_two_non_forming_contacts")
    v["dmin_nf_1"], v["dmin_nf_2"] = nf[0], nf[1]
    counts, edges = np.histogram(D[hh], bins=R5.RDF_EDGES)
    names = [f"rdf_hh_{int(round(e * 100)):03d}" for e in edges[:-1]]
    if names != [x for x in R5.BLOCKS["B2"] if x.startswith("rdf_hh_")]:
        raise RuntimeError(f"RDF bin names {names} differ from rev5_common.BLOCKS['B2']")
    v.update({nm: float(cn) for nm, cn in zip(names, counts)})
    return v, {}


def _max_finite(*xs_):
    v = [float(x) for x in xs_ if x is not None and np.isfinite(x)]
    return max(v) if v else np.nan


def block_b3(c):
    g = c.g
    jA, jB = c.json_sp("fA"), c.json_sp("fB")
    XA = np.asarray(g.ts_xyz[g.A_idx], dtype=float) * R5.BOHR
    XB = np.asarray(g.ts_xyz[g.B_idx], dtype=float) * R5.BOHR
    R = XB[None, :, :] - XA[:, None, :]                              # r_j - r_i [bohr], (nA, nB, 3)
    r = np.linalg.norm(R, axis=2)
    qA, qB, mA, mB, tA, tB = jA["q"], jB["q"], jA["mu"], jB["mu"], jA["theta"], jB["theta"]
    mAR = np.einsum("ik,ijk->ij", mA, R)                             # μ_i · R_ij
    mBR = np.einsum("jk,ijk->ij", mB, R)                             # μ_j · R_ij
    e_qmu = np.sum((qB[None, :] * mAR - qA[:, None] * mBR) / r ** 3)
    e_mumu = np.sum(((mA @ mB.T) * r ** 2 - 3.0 * mAR * mBR) / r ** 5)
    RtBR = np.einsum("ijk,jkl,ijl->ij", R, tB, R)                    # R · Θ_j · R
    RtAR = np.einsum("ijk,ikl,ijl->ij", R, tA, R)                    # R · Θ_i · R
    e_qth = np.sum((qA[:, None] * RtBR + qB[None, :] * RtAR) / r ** 5)
    v = {"mp_E_qmu": float(e_qmu) * R5.EH2KCAL, "mp_E_mumu": float(e_mumu) * R5.EH2KCAL,
         "mp_E_qTheta": float(e_qth) * R5.EH2KCAL}
    rA = r / R5.BOHR                                                 # Angstrom
    qq = qA[:, None] * qB[None, :]
    NA = np.array([R5.GFN2_VALENCE[s] for s in g.fA[0]], dtype=float) - qA
    NB = np.array([R5.GFN2_VALENCE[s] for s in g.fB[0]], dtype=float) - qB
    for k, al in R5.PEN_ALPHA.items():
        v[f"pen_damp_{k}"] = float(COULOMB * np.sum(qq * (1.0 - np.exp(-al * rA)) / rA))
        v[f"pen_ovl_{k}"] = float(np.sum(NA[:, None] * NB[None, :] * np.exp(-al * rA)))
    qc = {"qc_b3_dip_check_au": _max_finite(jA["dip_check"], jB["dip_check"]),
          "qc_b3_q_json_maxdiff": _max_finite(jA["q_json_maxdiff"], jB["q_json_maxdiff"]),
          "qc_b3_quad_order": "|".join(sorted({jA["quad_order"], jB["quad_order"]}))}
    return v, qc


def block_b4(c):
    g, v = c.g, {}
    for f, st in FRAG_STRUCT.items():
        ip, ea = c.vipea(st)
        mu, eta = -(ip + ea) / 2.0, ip - ea
        if not eta > 0:
            raise BlockFail(f"nonpositive_hardness_{f}:IP{ip:.4f},EA{ea:.4f}")
        v[f"mu_{f}"], v[f"eta_{f}"], v[f"omega_{f}"] = mu, eta, mu ** 2 / (2.0 * eta)
    v["dN_rel"] = (v["mu_rB"] - v["mu_rA"]) / (v["eta_rA"] + v["eta_rB"])
    v["dN_dist"] = (v["mu_dB"] - v["mu_dA"]) / (v["eta_dA"] + v["eta_dB"])
    fk = {"fA": c.vfukui("fA"), "fB": c.vfukui("fB")}
    js = {st: c.json_sp(st) for st in FRAG_STRUCT.values()}
    for a, (st, k) in react_pos(g).items():
        for col, key in enumerate(("p", "m", "0")):
            v[f"fk_{key}_{a}"] = float(fk[st][k, col])
        v[f"alpha_{a}"] = float(js[st]["alpha_atoms"][k])
    for f, st in FRAG_STRUCT.items():
        v[f"alpha_mol_{f}"] = js[st]["alpha_mol"]
    p0 = c.sp("ts")
    gedt = float(np.asarray(p0["charges"], dtype=float)[g.A_idx].sum() - c.q1)
    qc = {"qc_b4_gedt_diff": gedt - float(c.g1["b_ct"])}
    if not abs(qc["qc_b4_gedt_diff"]) <= TOL_GEDT:
        raise BlockFail(f"gedt_vs_g1_b_ct:{qc['qc_b4_gedt_diff']:.1e}", qc)
    v["gedt"] = float(c.g1["b_ct"])
    return v, qc


def block_b5(c):
    g, g1 = c.g, c.g1
    pA, pB, p0 = c.sp("fA"), c.sp("fB"), c.sp("ts")
    qA, qB = np.asarray(pA["charges"], dtype=float), np.asarray(pB["charges"], dtype=float)
    XA = np.asarray(g.ts_xyz[g.A_idx], dtype=float)

    def chans(p, XB):
        D = np.linalg.norm(XA[:, None, :] - XB[None, :, :], axis=2)
        return {"Eint": (p["total"] - pA["total"] - pB["total"]) * R5.EH2KCAL,
                "pauli": (p["rep"] - pA["rep"] - pB["rep"]) * R5.EH2KCAL,
                "oi": (p["eht"] - pA["eht"] - pB["eht"]) * R5.EH2KCAL,
                "elst": float(COULOMB * np.sum(qA[:, None] * qB[None, :] / D))}

    X0 = chans(p0, np.asarray(g.ts_xyz[g.B_idx], dtype=float))
    ref = {"Eint": "xtb_interaction_kcal", "pauli": "b_pauli", "oi": "b_oi", "elst": "b_elst"}
    qc = {"qc_b5_dE_ts_eh": p0["total"] - float(g1["xtb_e_ts_eh"]),
          "qc_b5_dE_fA_eh": pA["total"] - float(g1["xtb_e_d1_eh"]),
          "qc_b5_dE_fB_eh": pB["total"] - float(g1["xtb_e_d2_eh"]),
          "qc_b5_dch_max_kcal": float(np.max(np.abs([X0[x] - float(g1[k]) for x, k in ref.items()])))}
    gate = (all(abs(qc[k]) <= R5.SCAN_GATE_EH for k in ("qc_b5_dE_ts_eh", "qc_b5_dE_fA_eh", "qc_b5_dE_fB_eh"))
            and qc["qc_b5_dch_max_kcal"] <= R5.SCAN_GATE_EH * R5.EH2KCAL)
    if not gate:
        raise BlockFail("scan_gate_fail", qc)
    dm2, dm1, dp1, dp2 = R5.SCAN_DELTAS
    if not (abs(dm1 + dp1) < 1e-12 and abs(dm2 + dp2) < 1e-12 and abs(dp2 - 2 * dp1) < 1e-12 and dp1 > 0):
        raise RuntimeError(f"SCAN_DELTAS {R5.SCAN_DELTAS} is not (-2h, -h, +h, +2h)")
    u = (g.ts_xyz[g.dph_a] + g.ts_xyz[g.dph_b]) / 2.0 - (g.ts_xyz[g.dip_a] + g.ts_xyz[g.dip_b]) / 2.0
    nu = float(np.linalg.norm(u))
    if not nu > 1e-6:
        raise BlockFail("scan_direction_degenerate", qc)
    u = u / nu
    X = {}
    for dl in R5.SCAN_DELTAS:
        xyz = np.array(g.ts_xyz, dtype=float, copy=True)
        xyz[g.B_idx] += dl * u
        try:
            p = xs.xtb_sp(list(g.ts_syms), xyz, c.q1 + c.q2)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            raise BlockFail(f"scan_sp_fail:{dl:+.2f}:{type(e).__name__}:{e}", qc)
        X[dl] = chans(p, xyz[g.B_idx])
    v = {}
    for x in ("Eint", "pauli", "oi", "elst"):
        v[f"scan_slope_{x}"] = (X[dp1][x] - X[dm1][x]) / (dp1 - dm1)
        v[f"scan_curv_{x}"] = (X[dp1][x] - 2.0 * X0[x] + X[dm1][x]) / dp1 ** 2
        v[f"scan_m10_{x}"] = X[dm2][x]
        v[f"scan_p10_{x}"] = X[dp2][x]
    return v, qc


def parse_hess(path):
    """(frequencies [cm-1], normal modes 3N x 3N (columns = modes), $atoms symbols, $atoms xyz [A]) of an ORCA
    .hess file; the same parse as D1 orca_direct.parse_hess (+ the $atoms block as g1_geom.hess_atoms)."""
    lines = Path(path).read_text().splitlines()
    sec = {l.strip(): i for i, l in enumerate(lines) if l.startswith("$")}
    for key in ("$vibrational_frequencies", "$normal_modes", "$atoms"):
        if key not in sec:
            raise ParseError(f"{Path(path).name}: section {key} missing")
    i = sec["$vibrational_frequencies"]
    nf = int(lines[i + 1].split()[0])
    freqs = np.array([_f(lines[i + 2 + k].split()[1]) for k in range(nf)])
    j = sec["$normal_modes"]
    nr, nc = (int(x) for x in lines[j + 1].split()[:2])
    modes = np.zeros((nr, nc))
    k, filled = j + 2, 0
    while filled < nc:
        cols = [int(x) for x in lines[k].split()]
        k += 1
        for r in range(nr):
            for cc, val in zip(cols, lines[k + r].split()[1:]):
                modes[r, cc] = _f(val)
        k += nr
        filled += len(cols)
    a = sec["$atoms"]
    na = int(lines[a + 1].split()[0])
    rows = [lines[a + 2 + m].split() for m in range(na)]
    return freqs, modes, [row[0] for row in rows], np.array([[_f(x) for x in row[2:5]] for row in rows]) / R5.BOHR


def forming_share(X, mode, pairs, heavy):
    """D1 run_ts.forming_share (d1-build@d573111e), as used by g1_geom.py for result.json imag_mode_forming_share."""
    def ddot(i, j):
        r = X[i] - X[j]
        return float(np.dot(r, mode[i] - mode[j]) / np.linalg.norm(r))
    allp = [(i, j) for k, i in enumerate(heavy) for j in heavy[k + 1:] if np.linalg.norm(X[i] - X[j]) < 3.5]
    vals = sorted((abs(ddot(i, j)) for i, j in allp), reverse=True)
    return sum(abs(ddot(i, j)) for i, j in pairs) / (sum(vals[:2]) or 1e-12)


def kabsch_rmsd(P, Q):
    P = np.asarray(P, dtype=float) - np.mean(P, axis=0)
    Q = np.asarray(Q, dtype=float) - np.mean(Q, axis=0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ Rm.T - Q) ** 2).sum(axis=1).mean()))


def product_map(G_ts, G_p, X_ts, X_p):
    """TS heavy index -> product heavy index: an isomorphism of (TS heavy graph + forming edges) with the product
    heavy graph; the identity when it is one, else the lowest heavy-atom Kabsch RMSD (ties: smallest image tuple)."""
    heavy = sorted(G_ts.nodes)
    ident = (sorted(G_p.nodes) == heavy and all(G_ts.nodes[i]["lab"] == G_p.nodes[i]["lab"] for i in heavy)
             and {frozenset(e) for e in G_ts.edges} == {frozenset(e) for e in G_p.edges})
    maps = []
    for m in GraphMatcher(G_ts, G_p, node_match=fr._nm).isomorphisms_iter():     # m: TS node -> product node
        maps.append(dict(m))
        if len(maps) >= MAX_ISO:
            break
    if not maps:
        raise BlockFail("product_vs_ts_plus_forming_graph_not_isomorphic")
    if ident:
        best = {i: i for i in heavy}
    else:
        best = min(maps, key=lambda m: (kabsch_rmsd(X_ts[heavy], X_p[[m[i] for i in heavy]]),
                                        tuple(m[i] for i in heavy)))
    rmsd = kabsch_rmsd(X_ts[heavy], X_p[[best[i] for i in heavy]])
    return best, dict(identity=ident, n_iso=len(maps), rmsd=rmsd)


def block_b6(c):
    g, rid = c.g, c.rid
    gd = R5.G1_ROOT / str(rid)
    for f in ("result.json", "work/ts.hess"):
        if not (gd / f).is_file():
            raise BlockFail(f"missing_g1_{f.replace('/', '_')}")
    res = json.loads((gd / "result.json").read_text())
    freqs, modes, hs, hx = parse_hess(gd / "work" / "ts.hess")
    n = len(g.ts_syms)
    if hs != list(g.ts_syms) or freqs.size != 3 * n or modes.shape != (3 * n, 3 * n):
        raise BlockFail("hess_does_not_match_ts_xyz")
    shift = hx - g.ts_xyz
    if not float(np.abs(shift - shift.mean(axis=0)).max()) <= TOL_HESS_GEOM_A:
        raise BlockFail("hess_geometry_is_not_ts_xyz")
    k = int(np.argmin(freqs))
    nu = float(freqs[k])
    if not nu < 0:
        raise BlockFail("no_imaginary_frequency")
    imag, share_json = res.get("imag_freqs_cm") or [], res.get("imag_mode_forming_share")
    if not imag or share_json is None:
        raise BlockFail("result_json_lacks_imag_freqs_or_share")
    X, mode = np.asarray(g.ts_xyz, dtype=float), modes[:, k].reshape(n, 3)
    heavy = [i for i, s in enumerate(g.ts_syms) if s != "H"]
    qc = {"qc_b6_nu_diff_cm": nu - float(imag[0]),
          "qc_b6_mode_share_diff": forming_share(X, mode, g.pairs, heavy) - float(share_json)}
    if not (abs(qc["qc_b6_nu_diff_cm"]) <= TOL_HESS and abs(qc["qc_b6_mode_share_diff"]) <= TOL_HESS):
        raise BlockFail("hess_mode_vs_result_json", qc)

    def proj(i, j):
        r = X[i] - X[j]
        return abs(float(np.dot(r, mode[i] - mode[j]) / np.linalg.norm(r)))
    p_ad, p_be = proj(g.dip_a, g.dph_a), proj(g.dip_b, g.dph_b)
    if not p_ad + p_be > 0:
        raise BlockFail("mode_async_undefined", qc)
    v = {"nu_imag": nu, "mode_share": float(share_json), "mode_async": p_ad / (p_ad + p_be)}

    # ---- product: Coley p*.xyz (sorted, first) -> xtb --opt tight (ALPB, q1+q2)
    ps = sorted((R5.PROF / str(rid)).glob("p*.xyz"))
    if not ps:
        raise BlockFail("no_coley_product_file", qc)
    qc["qc_b6_product_file"] = ps[0].name
    p_syms, p_xyz = fr.read_xyz(ps[0])
    if collections.Counter(p_syms) != collections.Counter(g.ts_syms):
        raise BlockFail("product_formula_differs_from_ts", qc)
    out, files = xtb_run(p_syms, p_xyz, c.q1 + c.q2, ["--opt", "tight"], keep=("xtbopt.xyz",),
                         timeout=XTB_OPT_TIMEOUT_S)
    if "GEOMETRY OPTIMIZATION CONVERGED" not in out or "xtbopt.xyz" not in files:
        raise BlockFail("product_opt_not_converged", qc)
    es = TOTAL_E_RE.findall(out)
    if not es:
        raise ParseError("product opt: no 'TOTAL ENERGY <v> Eh' line", qc)
    e_p = _f(es[-1])
    qc["qc_b6_e_product_eh"] = e_p
    o_syms, o_xyz = read_xyz_text(files["xtbopt.xyz"])
    if [s.lower() for s in o_syms] != [s.lower() for s in p_syms]:
        raise ParseError("xtbopt.xyz atoms differ from the input product", qc)
    if c.cache_dir is not None:
        R5.write_atomic(Path(c.cache_dir) / "product_xtbopt.xyz", files["xtbopt.xyz"])
    G_p = fr.heavy_graph(p_syms, o_xyz)[0]                               # optimised product
    G_p0 = fr.heavy_graph(p_syms, p_xyz)[0]                              # Coley start product (same atom order)
    if not nx.is_isomorphic(G_p, G_p0, node_match=fr._nm):
        raise BlockFail("product_heavy_graph_changed_on_opt", qc)
    G_ts = fr.heavy_graph(list(g.ts_syms), X)[0].copy()
    for i, j in g.pairs:
        if i not in G_ts or j not in G_ts:
            raise BlockFail("forming_pair_atom_not_heavy", qc)
        G_ts.add_edge(i, j)
    try:
        pm, info = product_map(G_ts, G_p, X, o_xyz)
    except BlockFail as e:                                                # keep the qc values in the cache record
        raise BlockFail(e.detail, qc)
    qc.update(qc_b6_map_identity=float(info["identity"]), qc_b6_n_iso=float(info["n_iso"]),
              qc_b6_map_rmsd_A=info["rmsd"])
    if not info["identity"] and info["n_iso"] >= MAX_ISO:                 # the lowest-RMSD map may be missing
        raise BlockFail(f"iso_enumeration_truncated_{MAX_ISO}", qc)
    # assert: the mapped forming pairs are bonds of the Coley start product. pm maps onto G_p (optimised), so a check
    # on G_p would hold by construction; G_p0 shares the node indices (atom order kept by xtb --opt, checked above).
    for i, j in g.pairs:
        if not G_p0.has_edge(pm[i], pm[j]):
            raise BlockFail("mapped_forming_pair_not_bonded_in_coley_product", qc)
    e1, e2 = res.get("e_rel1_eh"), res.get("e_rel2_eh")
    if e1 is None or e2 is None:
        raise BlockFail("result_json_lacks_e_rel", qc)
    rP_ad = float(np.linalg.norm(o_xyz[pm[g.dip_a]] - o_xyz[pm[g.dph_a]]))
    rP_be = float(np.linalg.norm(o_xyz[pm[g.dip_b]] - o_xyz[pm[g.dph_b]]))
    v.update(xtb_dErxn=(e_p - float(e1) - float(e2)) * R5.EH2KCAL,
             prog_ad=float(g.D_ts[g.dip_a, g.dph_a]) / rP_ad, prog_be=float(g.D_ts[g.dip_b, g.dph_b]) / rP_be)
    return v, qc


BLOCK_FUNCS = {"B1": block_b1, "B2": block_b2, "B3": block_b3, "B4": block_b4, "B5": block_b5, "B6": block_b6}


# ================================================================ geometry gate, cache, row
def geometry(label, g1row, meta=None):
    """('ok', ts_fragments namespace) or (reason, None): G1 geometry step + the G1 parquet row gate."""
    try:
        g = xs.ts_fragments(label, "g1", meta)
    except Exception as e:                                               # noqa: BLE001
        return f"g1:error_{type(e).__name__}", None
    if isinstance(g, str):
        return f"g1:{g}", None
    if g1row is None:
        return "g1_parquet:no_row", None
    st = str(g1row.get("xtb_status"))
    if st != "ok":
        return f"g1_parquet:{st}", None
    dist = xs.ts_distances(g)
    if sorted(dist) != sorted(DIST_COLS):
        raise RuntimeError(f"xtb_slice.ts_distances keys {sorted(dist)} != DIST_COLS")
    worst = max(abs(dist[k] - float(g1row[k])) for k in DIST_COLS)
    if not worst <= TOL_DIST_A:
        return f"g1_parquet:distances_differ_{worst:.1e}", None
    return "ok", g


def inputs_fp(label):
    """Fingerprint of the G1 inputs + label fields a cache record was computed from."""
    rid = int(label["rxn_id"])
    h = hashlib.sha256()
    for f in ("ts.xyz", "rel1.xyz", "rel2.xyz", "result.json"):
        p = R5.G1_ROOT / str(rid) / f
        h.update(f.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    h.update(json.dumps({k: label.get(k) for k in ("formed_pairs_ts", "charge1", "charge2")}, sort_keys=True).encode())
    return h.hexdigest()[:20]


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def load_rec(path, fp, retry_failed=False):
    rec = _read_json(path)
    if not isinstance(rec, dict) or rec.get("code_version") != CODE_VERSION or rec.get("inputs") != fp:
        return None
    if retry_failed and rec.get("status") != "ok":
        return None
    return rec


def _clean_extra(block, extra):
    out = {}
    for k, v in extra.items():
        if k not in EXTRA[block]:
            continue
        if k in STR_EXTRA:
            out[k] = None if v is None else str(v)
        else:
            out[k] = None if v is None else float(v)
    return out


def run_rxn(label, g1row, cache_root, blocks, retry_failed=False, meta=None):
    """Compute the missing blocks of one rxn and write $cache_root/<rid>/{geom,Bk}.json (atomic)."""
    rid = int(label["rxn_id"])
    d = Path(cache_root) / str(rid)
    fp = inputs_fp(label)
    todo = [b for b in COMPUTE_ORDER if b in blocks and load_rec(d / f"{b}.json", fp, retry_failed) is None]
    if not todo and load_rec(d / "geom.json", fp) is not None:
        return
    t0 = time.time()
    status, g = geometry(label, g1row, meta)
    R5.write_json(d / "geom.json", dict(status=status, t_s=time.time() - t0, host=HOST, code_version=CODE_VERSION,
                                        inputs=fp))
    if status != "ok":
        return
    c = Ctx(label, g, g1row, d)
    for b in todo:
        t1, detail, tb_txt = time.time(), None, None
        try:
            vals, extra = BLOCK_FUNCS[b](c)
            missing = [k for k in R5.BLOCKS[b] if k not in vals]
            unknown = [k for k in vals if k not in R5.BLOCKS[b]] + [k for k in extra if k not in EXTRA[b]]
            if missing or unknown:
                raise RuntimeError(f"block {b}: columns missing {missing}, unknown {unknown}")
            vals = {k: float(vals[k]) for k in R5.BLOCKS[b]}
            nonfin = [k for k, x in vals.items() if not np.isfinite(x)]
            if nonfin:
                raise BlockFail("nonfinite:" + ",".join(nonfin[:6]), extra)
            status = "ok"
        except BlockFail as e:
            status, vals, extra, detail = str(e), {}, e.qc, e.detail
        except OSError:
            raise                                   # not cached: missing binary / disk problems are not results
        except Exception as e:                      # noqa: BLE001
            status, vals, extra = f"error:{type(e).__name__}:{str(e)[:120]}", {}, {}
            tb_txt = traceback.format_exc()
        R5.write_json(d / f"{b}.json", dict(block=b, status=status, values=vals, extra=_clean_extra(b, extra),
                                            detail=detail, traceback=tb_txt, t_s=time.time() - t1, host=HOST,
                                            code_version=CODE_VERSION, inputs=fp))


def rxn_complete(label, cache_root, blocks, retry_failed=False):
    d, fp = Path(cache_root) / str(int(label["rxn_id"])), inputs_fp(label)
    geo = load_rec(d / "geom.json", fp)
    if geo is None:
        return False
    if geo.get("status") != "ok":
        return not retry_failed
    return all(load_rec(d / f"{b}.json", fp, retry_failed) is not None for b in blocks)


def assemble_row(label, cache_root, blocks=R5.BLOCK_ORDER):
    """One output row from the cache records (see the module docstring)."""
    rid = int(label["rxn_id"])
    d, fp = Path(cache_root) / str(rid), inputs_fp(label)
    row = {c: np.nan for c in ROW_COLS}
    row.update({c: None for c in STR_EXTRA}, rxn_id=rid, geom="g1")
    crash = _read_json(d / "crash.json")
    lost = f"error:worker_crash_exit{crash.get('exitcode')}" if isinstance(crash, dict) else "error:no_result"
    geo = load_rec(d / "geom.json", fp)
    gstat = geo["status"] if geo else lost
    if gstat != "ok":
        row.update({f"status_{b}": "skipped" for b in R5.BLOCK_ORDER})
        row["ext_status"] = f"geom:{gstat}"
        return row
    first = None
    for b in R5.BLOCK_ORDER:
        if b not in blocks:
            st = "not_requested"
        else:
            rec = load_rec(d / f"{b}.json", fp)
            if rec is None:
                st = lost
            else:
                st = str(rec["status"])
                row[f"t_{b}"] = float(rec.get("t_s", np.nan))
                for k, x in (rec.get("extra") or {}).items():
                    if k in EXTRA[b]:
                        row[k] = x if k in STR_EXTRA else (np.nan if x is None else float(x))
                if st == "ok":
                    for k in R5.BLOCKS[b]:
                        row[k] = float(rec["values"][k])
        row[f"status_{b}"] = st
        if st != "ok" and first is None:
            first = f"{b}:{st}"
    row["ext_status"] = first or "ok"
    if row["status_B5"] == "scan_gate_fail":                            # a STOP condition: always visible
        row["ext_status"] = "B5:scan_gate_fail"
    t = [row[f"t_{b}"] for b in R5.BLOCK_ORDER if np.isfinite(row[f"t_{b}"])]
    row["t_total_s"] = float(sum(t)) if t else np.nan
    return row


# ================================================================ worker pool (crash-isolating)
def _worker(conn, cache_root, blocks, retry_failed):
    """Persistent spawn worker: receives (label, g1row), answers with the rxn_id once its cache records exist."""
    meta = xs.load_meta()
    while True:
        try:
            task = conn.recv()
        except EOFError:
            return
        if task is None:
            return
        label, g1row = task
        run_rxn(label, g1row, cache_root, blocks, retry_failed, meta)
        conn.send(int(label["rxn_id"]))


def run_pool(tasks, cache_root, workers, blocks, retry_failed, tag):
    """Run run_rxn for every (label, g1row) task in <= workers spawn processes. A worker that dies marks its
    current rxn with crash.json and is replaced. Returns (finished, crashed); exits 4 on systematic crashes."""
    if not tasks:
        return 0, 0
    ctx = mp.get_context("spawn")
    queue = collections.deque(tasks)
    total, done, crashed, t0 = len(tasks), 0, 0, time.time()

    def start():
        parent, child = ctx.Pipe()
        # daemonic: never outlives the parent (xtb runs through subprocess, which a daemonic process may use)
        p = ctx.Process(target=_worker, args=(child, str(cache_root), list(blocks), bool(retry_failed)), daemon=True)
        p.start()
        child.close()
        return {"p": p, "conn": parent, "rid": None}

    def feed(w):
        if queue:
            label, g1row = queue.popleft()
            w["rid"] = int(label["rxn_id"])
            w["conn"].send((label, g1row))
        else:
            w["rid"] = None
            try:
                w["conn"].send(None)
            except (OSError, ValueError):
                pass

    pool = [start() for _ in range(max(1, min(int(workers), total)))]
    for w in pool:
        feed(w)
    while any(w["rid"] is not None for w in pool):
        busy = [w for w in pool if w["rid"] is not None]
        mp_wait([w["conn"] for w in busy] + [w["p"].sentinel for w in busy], timeout=300)
        for k, w in enumerate(pool):
            if w["rid"] is None:
                continue
            got, dead = None, False
            try:
                if w["conn"].poll():
                    got = w["conn"].recv()
            except (EOFError, OSError):
                dead = True
            if got is None and not dead and not w["p"].is_alive():
                dead = True
            if got is not None:
                done += 1
                feed(w)
            elif dead:
                rid = w["rid"]
                w["p"].join(timeout=10)
                code = w["p"].exitcode
                R5.write_json(Path(cache_root) / str(rid) / "crash.json",
                              dict(exitcode=code, host=HOST, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   code_version=CODE_VERSION))
                print(f"[{tag}] worker died on rxn {rid} (exit code {code}); replaced", flush=True)
                crashed += 1
                done += 1
                try:
                    w["conn"].close()
                except OSError:
                    pass
                w["rid"] = None
                if queue:
                    pool[k] = start()
                    feed(pool[k])
            else:
                continue
            if done % 25 == 0 or done == total:
                print(f"[{tag}] {done}/{total} rxns, {crashed} worker crashes, {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        if done >= 20 and crashed > CRASH_ABORT * done:
            for w in pool:
                if w["p"].is_alive():
                    w["p"].terminate()
            die(4, f"[{tag}] STOP: {crashed}/{done} rxns killed their worker process (systematic failure; see the "
                   f"tracebacks above); nothing written")
    for w in pool:
        w["p"].join(timeout=60)
        if w["p"].is_alive():
            w["p"].terminate()
    return done, crashed


# ================================================================ inputs
def load_labels():
    """Accepted labels (labels_all.json minus xtb_slice.EXCLUDE), rxn_id -> record, ascending."""
    path = Path(R5.LABELS)
    if not path.is_file():
        die(2, f"labels {path} missing")
    acc = {int(d["rxn_id"]): d for d in json.loads(path.read_text()) if int(d["rxn_id"]) not in xs.EXCLUDE}
    bad = [r for r, d in acc.items() if d.get("status") != "ok"]
    if bad:
        die(2, f"labels {path}: {len(bad)} accepted rxns are not status ok: {bad[:10]}")
    return dict(sorted(acc.items()))


def load_g1_rows():
    """rxn_id -> the G1 parquet values the gates need (G1_COLS)."""
    p = Path(R5.FEAT_G1)
    if not p.is_file():
        die(2, f"G1 feature parquet {p} missing")
    try:
        df = pd.read_parquet(p, columns=G1_COLS)
    except Exception as e:                                               # noqa: BLE001
        die(2, f"{p}: cannot read the columns {G1_COLS}: {type(e).__name__}: {e}")
    tags = sorted(set(df["geom"].astype(str)))
    if tags != ["g1"]:
        die(2, f"{p}: geom tags {tags}, expected ['g1']")
    if df["rxn_id"].duplicated().any():
        die(2, f"{p}: duplicate rxn_id")
    return {int(r["rxn_id"]): r for r in df.to_dict("records")}


def g1_done_ids(labels):
    """rxn_ids with G1 .done; exits 1 if an accepted rxn has neither .done nor .fail_* (G1 incomplete)."""
    marks = {rid: xs.geom_marker("g1", rid) for rid in labels}
    unmarked = [r for r, m in marks.items() if m is None]
    if unmarked:
        die(1, f"GATE: {len(unmarked)} accepted rxns have no .done / .fail_* under {xs.GEOM_ROOT['g1']} "
               f"(G1 incomplete): {unmarked[:10]}")
    return [r for r, m in marks.items() if m == ".done"]


def parse_blocks(vals):
    if not vals:
        return list(R5.BLOCK_ORDER)
    bs = {b.strip().upper() for v in vals for b in str(v).split(",") if b.strip()}
    bad = sorted(bs - set(R5.BLOCKS))
    if bad:
        die(2, f"--block: unknown {bad}; choose from {list(R5.BLOCK_ORDER)}")
    return [b for b in R5.BLOCK_ORDER if b in bs]


def _ncpu():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)


def versions():
    v = {"code_version": CODE_VERSION, "python": sys.version.split()[0], "numpy": np.__version__, "xtb_bin": xs.XTB_BIN}
    try:
        r = subprocess.run([xs.XTB_BIN, "--version"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120)
        v["xtb"] = [l.strip() for l in (r.stdout + r.stderr).splitlines() if "version" in l.lower()][:2]
    except Exception as e:                                               # noqa: BLE001
        v["xtb"] = [f"error {type(e).__name__}: {e}"]
    try:
        import importlib.metadata as md
        v["tblite"] = md.version("tblite")
    except Exception as e:                                               # noqa: BLE001
        v["tblite"] = f"error {type(e).__name__}: {e}"
    return v


def _status_counts(df):
    return {b: {str(k): int(x) for k, x in df[f"status_{b}"].value_counts().head(8).items()} for b in R5.BLOCK_ORDER}


# ================================================================ commands
def cmd_slice(a):
    out = Path(a.out)
    if out.exists():
        print(f"[slice {a.slice_id}] {out} exists — skip", flush=True)
        return 0
    if not 0 <= a.slice_id < a.n_slices:
        die(2, f"slice_id {a.slice_id} not in [0, {a.n_slices})")
    blocks = parse_blocks(a.block)
    labels = load_labels()
    mine = g1_done_ids(labels)[a.slice_id::a.n_slices]
    g1 = load_g1_rows()
    print(f"[slice {a.slice_id}/{a.n_slices}] {len(mine)} G1-ok rxns (interleaved), blocks {blocks}, workers "
          f"{a.workers}; xtb {xs.XTB_BIN}; G1 {R5.G1_ROOT}; G1 features {R5.FEAT_G1}; cache {CACHE_ROOT}", flush=True)
    tasks = [(labels[r], g1.get(r)) for r in mine if not rxn_complete(labels[r], CACHE_ROOT, blocks, a.retry_failed)]
    print(f"[slice {a.slice_id}] {len(mine) - len(tasks)} rxns fully cached, {len(tasks)} to compute", flush=True)
    run_pool(tasks, CACHE_ROOT, a.workers, blocks, a.retry_failed, tag=f"slice {a.slice_id}")
    df = pd.DataFrame([assemble_row(labels[r], CACHE_ROOT, blocks) for r in mine], columns=ROW_COLS)
    n_bad = int((df["ext_status"] != "ok").sum())
    print(f"[slice {a.slice_id}] ext_status ok {len(df) - n_bad}/{len(df)}; per block: {_status_counts(df)}", flush=True)
    if len(df) and n_bad > SLICE_MAX_FAIL * len(df):
        die(4, f"[slice {a.slice_id}] STOP: {n_bad}/{len(df)} rows failed (> {SLICE_MAX_FAIL:.0%}) — systematic "
               f"problem, slice not written; cache records in {CACHE_ROOT}/<rid>/ (rerun with --retry-failed after "
               f"the fix): {df.loc[df.ext_status != 'ok', 'ext_status'].value_counts().head(10).to_dict()}")
    R5.write_atomic(out, lambda tmp: df.to_parquet(tmp, index=False))
    print(f"[slice {a.slice_id}] wrote {len(df)} rows -> {out}", flush=True)
    return 0


def cmd_rxn(a):
    blocks = parse_blocks(a.block)
    labels = load_labels()
    if a.rxn_id not in labels:
        die(2, f"rxn {a.rxn_id} is not an accepted label")
    if xs.geom_marker("g1", a.rxn_id) != ".done":
        die(2, f"rxn {a.rxn_id} has no G1 .done (marker {xs.geom_marker('g1', a.rxn_id)})")
    root = Path(a.cache) if a.cache else CACHE_ROOT
    g1 = load_g1_rows()
    run_rxn(labels[a.rxn_id], g1.get(a.rxn_id), root, blocks, a.retry_failed, xs.load_meta())
    row = assemble_row(labels[a.rxn_id], root, blocks)
    print(json.dumps(row, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    return 0 if row["ext_status"] == "ok" else 1


def smoke_gates(label, g1row, meta):
    """B-0 gates for one rxn (tblite vs xtb energies; overlap block; orthonormality) + parser self-checks."""
    status, g = geometry(label, g1row, meta)
    if status != "ok":
        return {"geometry": status}
    c = Ctx(label, g, g1row)
    o = {"geometry": "ok", "gate1_dE_eh": {}, "gate2_sblock": {}, "gate3_ortho": {}, "eps_dHOMO_ev": {},
         "eps_dLUMO_ev": {}, "quad_order": {}, "dip_check_au": {}, "q_json_maxdiff": {}, "nao": {}, "errors": []}
    for name in ("rel1", "rel2", "fA", "fB", "ts"):
        try:
            t = c.tb(name)
            o["gate1_dE_eh"][name] = t["energy"] - float(c.sp(name)["total"])
            o["gate3_ortho"][name] = t["ortho"]
            o["nao"][name] = t["nao"]
        except BlockFail as e:
            o["errors"].append(f"{name}: {e.detail}")
    try:
        tT = c.tb("ts")
        for name, idx in (("fA", g.A_idx), ("fB", g.B_idx)):
            ao = np.concatenate([np.flatnonzero(tT["ao_atom"] == i) for i in idx])
            tf = c.tb(name)
            o["gate2_sblock"][name] = (float(np.abs(tf["S"] - tT["S"][np.ix_(ao, ao)]).max())
                                       if ao.size == tf["nao"] else float("inf"))
    except BlockFail as e:
        o["errors"].append(f"gate2: {e.detail}")
    for name in ("rel1", "rel2", "fA", "fB"):
        try:
            j, t = c.json_sp(name), c.tb(name)
            no = n_occupied(t, name)
            o["eps_dHOMO_ev"][name] = float(t["eps"][no - 1] * R5.EH2EV - j["eps_homo"])
            o["eps_dLUMO_ev"][name] = float(t["eps"][no] * R5.EH2EV - j["eps_lumo"])
            o["quad_order"][name] = j["quad_order"]
            o["dip_check_au"][name] = j["dip_check"]
            o["q_json_maxdiff"][name] = j["q_json_maxdiff"]
        except BlockFail as e:
            o["errors"].append(f"{name} json/eps: {e.detail}")
    o["ao_per_element"] = dict(_AO_COUNT)
    return o


def charged_smoke_rxns(labels):
    """Extra B-0 rxns (SMOKE_RXNS are all neutral, so gate 1 would never see an ion): for every nonzero
    (charge1, charge2) pattern among the rows_rev4 rxns, the smallest rxn_id with G1 .done. {"q1,q2": rxn_id}."""
    p = Path(R5.ROWS4)
    if not p.is_file():
        die(2, f"{p} missing (needed to pick the charged B-0 rxns)")
    rows4 = set(pd.read_csv(p)["rxn_id"].astype(int).tolist())
    pick = {}
    for rid, lab in labels.items():                                      # ascending rxn_id
        q = f"{int(lab.get('charge1', 0) or 0)},{int(lab.get('charge2', 0) or 0)}"
        if q == "0,0" or q in pick or rid not in rows4 or rid in R5.SMOKE_RXNS:
            continue
        if xs.geom_marker("g1", rid) == ".done":
            pick[q] = rid
    return pick


def cmd_smoke(a):
    labels, g1, meta = load_labels(), load_g1_rows(), xs.load_meta()
    ver = versions()
    ev, skipped = [], {}
    for rid in R5.SMOKE_RXNS:
        if rid not in labels:
            skipped[str(rid)] = "not an accepted label"
        elif xs.geom_marker("g1", rid) != ".done":
            skipped[str(rid)] = f"no G1 structures (G1 marker {xs.geom_marker('g1', rid)})"
        else:
            ev.append(rid)
    charged = charged_smoke_rxns(labels)
    ev_all = ev + [r for r in sorted(charged.values()) if r not in ev]
    print(f"[smoke] rxns {list(R5.SMOKE_RXNS)}: evaluated {ev}, skipped {skipped}; extra charged (q1,q2 -> rxn) "
          f"{charged}; versions {ver}", flush=True)
    for r in ev_all:                        # always from scratch: a rerun after a fix must not reuse old records
        shutil.rmtree(SMOKE_ROOT / str(r), ignore_errors=True)
    tasks = [(labels[r], g1.get(r)) for r in ev_all]
    run_pool(tasks, SMOKE_ROOT, max(1, a.workers), R5.BLOCK_ORDER, False, tag="smoke")
    stop, per, t_rxn = [], {}, []
    q_of = {rid: q for q, rid in charged.items()}
    for r in ev_all:                        # the same stop rules for the SMOKE_RXNS and the charged extras
        nm = f"rxn {r}" + (f" (extra charged q1,q2 = {q_of[r]})" if r in q_of else "")
        row = assemble_row(labels[r], SMOKE_ROOT)
        gt = smoke_gates(labels[r], g1.get(r), meta)
        blocks = {}
        d, fp = SMOKE_ROOT / str(r), inputs_fp(labels[r])
        for b in R5.BLOCK_ORDER:
            rec = load_rec(d / f"{b}.json", fp) or {}
            blocks[b] = {"status": row[f"status_{b}"], "t_s": row[f"t_{b}"]}
            if row[f"status_{b}"] != "ok":
                blocks[b].update(detail=rec.get("detail"), traceback=(rec.get("traceback") or "")[-2000:] or None)
        per[str(r)] = dict(ext_status=row["ext_status"], blocks=blocks, core_s=row["t_total_s"], gates=gt,
                           features={k: row[k] for k in R5.EXT_COLS},
                           extra={k: row[k] for k in EXTRA_COLS})
        if np.isfinite(row["t_total_s"]):
            t_rxn.append(row["t_total_s"])
        # ---- per-rxn verdicts
        if gt.get("geometry") != "ok":
            stop.append(f"{nm}: geometry {gt.get('geometry')}")
            continue
        if gt["errors"]:
            stop.append(f"{nm}: {gt['errors']}")
        g1v, g2v, g3v = gt["gate1_dE_eh"], gt["gate2_sblock"], gt["gate3_ortho"]
        if len(g1v) != 5 or not all(abs(x) < R5.GATE_E_EH for x in g1v.values()):
            stop.append(f"{nm}: gate 1 |E_tblite - E_xtb| >= {R5.GATE_E_EH} Eh (or missing): {g1v}")
        if len(g2v) != 2 or not all(x < R5.GATE_S_BLOCK for x in g2v.values()):
            stop.append(f"{nm}: gate 2 fragment overlap vs TS block >= {R5.GATE_S_BLOCK}: {g2v}")
        if len(g3v) != 5 or not all(x < R5.GATE_ORTHO for x in g3v.values()):
            stop.append(f"{nm}: gate 3 max|C^T S C - I| >= {R5.GATE_ORTHO}: {g3v}")
        eps = list(gt["eps_dHOMO_ev"].values()) + list(gt["eps_dLUMO_ev"].values())
        if len(eps) != 8 or not all(abs(x) < TOL_EPS_EV for x in eps):
            stop.append(f"{nm}: xtb json vs tblite HOMO/LUMO >= {TOL_EPS_EV} eV (or missing): "
                        f"{gt['eps_dHOMO_ev']} {gt['eps_dLUMO_ev']}")
        if row["ext_status"] != "ok":
            stop.append(f"{nm}: block failure {row['ext_status']}")
    orders = sorted({o for r in per.values() for o in (r["gates"].get("quad_order") or {}).values()})
    determined = sorted({o for o in orders if "ambiguous" not in o})
    if len(determined) != 1:
        stop.append(f"atomic quadrupole component order not uniquely determined by the trace test: {orders}")
    dips = [x for r in per.values() for x in (r["gates"].get("dip_check_au") or {}).values() if np.isfinite(x)]
    if not dips:
        stop.append("atomic dipole convention unverified: no neutral structure with a json molecular dipole")
    xtb_ok = any("6.7.1" in l and "edcfbbe" in l for l in ver.get("xtb", []))
    if not xtb_ok:
        stop.append(f"xtb binary is not {R5.XTB_VERSION}: {ver.get('xtb')}")
    if ver.get("tblite") != R5.TBLITE_VERSION:
        stop.append(f"tblite is not {R5.TBLITE_VERSION}: {ver.get('tblite')}")
    if not ev:
        stop.append("no smoke rxn has G1 structures")
    n_done = len(g1_done_ids(labels))
    mean_s = float(np.mean(t_rxn)) if t_rxn else float("nan")
    rep = dict(spec="REV5 B-0", smoke_rxns=list(R5.SMOKE_RXNS), evaluated=ev, skipped=skipped,
               extra_charged=charged, evaluated_all=ev_all, versions=ver,
               tolerances=dict(GATE_E_EH=R5.GATE_E_EH, GATE_S_BLOCK=R5.GATE_S_BLOCK, GATE_ORTHO=R5.GATE_ORTHO,
                               TOL_EPS_EV=TOL_EPS_EV, TOL_TRACE=TOL_TRACE, TOL_DIP_AU=TOL_DIP_AU,
                               TOL_Q_JSON=TOL_Q_JSON, TOL_GEDT=TOL_GEDT, TOL_HESS=TOL_HESS, SCAN_GATE_EH=R5.SCAN_GATE_EH),
               quad_orders_found=orders, dip_checks_au=dips,
               core_s_per_rxn={str(r): per[str(r)]["core_s"] for r in ev_all},
               core_h_per_rxn_mean=mean_s / 3600.0, n_g1_ok=n_done, projected_core_h_all=mean_s * n_done / 3600.0,
               per_rxn=per, stop_reasons=stop, passed=not stop)
    R5.write_json(a.out, rep)
    print(json.dumps({k: rep[k] for k in ("evaluated", "skipped", "extra_charged", "quad_orders_found",
                                          "core_h_per_rxn_mean", "projected_core_h_all", "stop_reasons", "passed")},
                     indent=1, default=str))
    print(f"[smoke] wrote {a.out}", flush=True)
    if stop:
        die(3, f"[smoke] STOP (B-0): {len(stop)} gate / check failures — see {a.out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="rev 5 feature blocks B1..B6 (REV5 spec §B)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="B-0 engine smoke on rev5_common.SMOKE_RXNS (exit 3 = STOP)")
    s.add_argument("--workers", type=int, default=_ncpu())
    s.add_argument("--out", type=Path, default=SMOKE_OUT)
    s = sub.add_parser("slice", help="one interleaved slice of the G1-ok rxns -> parquet")
    s.add_argument("slice_id", type=int)
    s.add_argument("n_slices", type=int)
    s.add_argument("out", type=Path)
    s.add_argument("--workers", type=int, default=_ncpu())
    s.add_argument("--block", action="append", help="B1..B6, comma-separated or repeated (default: all)")
    s.add_argument("--retry-failed", action="store_true", help="recompute cached failed blocks")
    s = sub.add_parser("rxn", help="one rxn in this process; prints the row as JSON")
    s.add_argument("rxn_id", type=int)
    s.add_argument("--block", action="append", help="B1..B6, comma-separated or repeated (default: all)")
    s.add_argument("--cache", type=Path, help=f"cache root (default {CACHE_ROOT})")
    s.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    return {"smoke": cmd_smoke, "slice": cmd_slice, "rxn": cmd_rxn}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
