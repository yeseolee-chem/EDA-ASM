"""SPEC_11 Gate-A - unit-check the d29 (anisotropic elst) formula.

Synthetic +/- point-charge "dimer" with programmed dipoles in the far-field
limit (R >> l). The multipole formula (§1.1) is compared to an explicit
Coulomb sum computed at the level of the underlying primitives:

  Configuration:
    Fragment A: two nuclear "atoms" at r = C_A +/- (l/2) * z_hat with equal
      Z_a. Nuclear-charge centre C_A. Programmed dipole p_A = (0,0,mu_A) via
      charge injection (a +q, -q dipole along z, with q, l chosen so that
      p = q*l).
    Fragment B: analogous at C_B, dipole p_B along a rotated axis to exercise
      both T_qd and T_dd terms.

  We bypass tblite: dipoles p_A_raw = p_A + Q_A * C_A are constructed manually
  (matching what tblite would return about the origin).

  Formula: d29_formula = kappa_H * ( T_qd + T_dd )                 (kcal/mol)
  Explicit: pair-sum the Coulomb energy of the point charges directly
             (Bohr, Hartree units) -> kcal/mol.

  At R >> l, the two must agree to <1% and share sign. This is a pure
  algebra check - no xTB involvement.

Exit code 0 => Gate-A passes.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "spec/spec11_electronic_33d/code"))
from compute_d29_d33 import compute_d29, HARTREE_TO_KCAL


def explicit_coulomb(points_A, charges_A, points_B, charges_B):
    """Direct Coulomb energy [Hartree] between two clouds of point charges,
    positions in Bohr."""
    e = 0.0
    for rA, qA in zip(points_A, charges_A):
        for rB, qB in zip(points_B, charges_B):
            r = np.linalg.norm(np.asarray(rA) - np.asarray(rB))
            e += qA * qB / max(r, 1e-12)
    return e


def build_case(seed, R_len=30.0, l=0.5, mu_A=1.0, mu_B=0.7,
               Q_A=0, Q_B=0, tilt_B=(1.0, 0.0, 0.5)):
    """Build a synthetic 2-atom-per-fragment case. Positions in Bohr.

    A is at origin; B is displaced by R along +x.
    Each fragment has 2 nuclear centres separated by l along a chosen axis;
    the electronic dipole is faked by placing +q, -q at those centres so
    p = q * l pointing along the axis.
    """
    rng = np.random.default_rng(seed)
    z_hat = np.array([0.0, 0.0, 1.0])
    tilt = np.asarray(tilt_B, float); tilt /= np.linalg.norm(tilt)
    C_A = np.zeros(3)
    C_B = np.array([R_len, 0.0, 0.0])

    # Nuclear positions (Bohr).
    posA = np.stack([C_A + 0.5 * l * z_hat, C_A - 0.5 * l * z_hat])
    posB = np.stack([C_B + 0.5 * l * tilt, C_B - 0.5 * l * tilt])
    # Atomic numbers ("dummy" Z=1 so nuclear-charge centre is midpoint).
    Z = np.array([1, 1, 1, 1])   # 2 in A, 2 in B
    positions_bohr = np.vstack([posA, posB])
    idx_A = np.array([0, 1]); idx_B = np.array([2, 3])

    # Programmed fragment dipoles pointing along the +l direction.
    p_A = mu_A * z_hat
    p_B = mu_B * tilt

    # "Raw" dipole about origin (tblite returns dipole about coord origin):
    # p_raw = sum_i q_i r_i.
    # For a *neutral* fragment (Q=0), any charge distribution with dipole p
    # about its own centre also has p_raw = p + Q * C = p (Q=0).
    # For a charged fragment (Q!=0), p_raw = p + Q * C.
    p_A_raw = p_A + Q_A * C_A
    p_B_raw = p_B + Q_B * C_B

    # Explicit multi-charge distribution matching (Q, p):
    # Place two charges (+q_e, -q_e) at C +/- l/2 * ax, with q_e * l = |p|;
    # then add uniform monopole charge Q/2 at each nucleus so total = Q.
    q_e_A = mu_A / max(l, 1e-9)
    q_e_B = mu_B / max(l, 1e-9)
    # Half of Q on each nuclear centre.
    charges_A = np.array([q_e_A + Q_A / 2, -q_e_A + Q_A / 2])
    charges_B = np.array([q_e_B + Q_B / 2, -q_e_B + Q_B / 2])
    return dict(
        Z=Z, positions_bohr=positions_bohr,
        idx_A=idx_A, idx_B=idx_B,
        p_A_raw=p_A_raw, p_B_raw=p_B_raw,
        Q_A=Q_A, Q_B=Q_B,
        posA=posA, posB=posB,
        charges_A=charges_A, charges_B=charges_B,
    )


def run_case(case, label):
    d29 = compute_d29(case["Z"], case["positions_bohr"],
                      case["idx_A"], case["idx_B"],
                      case["p_A_raw"], case["p_B_raw"],
                      case["Q_A"], case["Q_B"])
    e_h = explicit_coulomb(case["posA"], case["charges_A"],
                           case["posB"], case["charges_B"])
    e_explicit = e_h * HARTREE_TO_KCAL
    # d29 omits the monopole-monopole term Q_A*Q_B/R, so subtract it out of
    # the explicit reference. Also, our "explicit" cloud is an ARBITRARY
    # representation of (Q, p) so it contains higher multipoles - the
    # comparison is meaningful only at R >> l where those decay away.
    C_A = case["positions_bohr"][case["idx_A"]].mean(0)  # atoms all Z=1
    C_B = case["positions_bohr"][case["idx_B"]].mean(0)
    R = np.linalg.norm(C_B - C_A)
    mono_mono = case["Q_A"] * case["Q_B"] / R * HARTREE_TO_KCAL
    e_ref_anis = e_explicit - mono_mono

    rel = abs(d29 - e_ref_anis) / max(abs(e_ref_anis), 1e-12)
    same_sign = np.sign(d29) == np.sign(e_ref_anis) or abs(e_ref_anis) < 1e-9
    print(f"  [{label}] d29_formula={d29:+.6e}  ref_anisotropic={e_ref_anis:+.6e}"
          f"  rel_err={rel:.3%}  same_sign={same_sign}")
    return rel, same_sign


def main():
    print("Gate-A: multipole unit test (synthetic +/- charge dimer)")
    ok = True

    # (i) neutral-neutral, dipole-dipole (T_qd = 0)
    c = build_case(0, R_len=40.0, l=0.4, mu_A=1.0, mu_B=0.8, Q_A=0, Q_B=0)
    rel, sign_ok = run_case(c, "neutral-neutral")
    ok = ok and rel < 0.01 and sign_ok

    # (ii) charged A neutral B - monopole-dipole enters
    c = build_case(1, R_len=50.0, l=0.4, mu_A=0.5, mu_B=1.2, Q_A=1, Q_B=0)
    rel, sign_ok = run_case(c, "cation-neutral")
    ok = ok and rel < 0.01 and sign_ok

    # (iii) opposite-charge - all terms active
    c = build_case(2, R_len=60.0, l=0.35, mu_A=0.9, mu_B=0.6, Q_A=1, Q_B=-1)
    rel, sign_ok = run_case(c, "cation-anion")
    ok = ok and rel < 0.01 and sign_ok

    if not ok:
        print("Gate-A FAIL")
        sys.exit(1)
    print("Gate-A PASS")


if __name__ == "__main__":
    main()
